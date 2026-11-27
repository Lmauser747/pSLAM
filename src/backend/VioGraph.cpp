#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <memory>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <gtsam/geometry/Cal3_S2.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Factor.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/GaussianFactorGraph.h>
#include <gtsam/linear/JacobianFactor.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/linear/linearExceptions.h>
#include <gtsam/navigation/ImuBias.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/NavState.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/ISAM2Result.h>
#include <gtsam/nonlinear/ISAM2UpdateParams.h>
#include <gtsam/nonlinear/LinearContainerFactor.h>
#include <gtsam/nonlinear/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/SmartProjectionPoseFactor.h>

#include "backend/Backend.hpp"
#include "CoreTypes.hpp"
#include "frontend/Frontend.hpp"
#include "frontend/LfTracker.hpp"
#include "pool_slam/SystemTypes.hpp"

namespace PoolSLAM{

using gtsam::symbol_shorthand::B;
using gtsam::symbol_shorthand::V;
using gtsam::symbol_shorthand::X;

namespace{

std::shared_ptr<SmartFactor> buildSmartFactor(
    const gtsam::SharedNoiseModel &noise,
    const std::shared_ptr<gtsam::Cal3_S2> &kPtr,
    const gtsam::Pose3 &bodyToCam,
    const gtsam::SmartProjectionParams &params,
    const std::vector<std::pair<gtsam::Point2, gtsam::Key>> &track){
    auto factor = std::make_shared<SmartFactor>(noise, kPtr, bodyToCam, params);
    for(const auto &obs : track){
        factor->add(obs.first, obs.second);
    }
    return factor;
}

} // namespace

void Backend::processFrame(const BackendHfPacket &pkt){
    if(window.empty() || pkt.imuDelta == nullptr){
        return;
    }

    // Snapshot maps for rollback if iSAM2 solve fails
    std::unordered_map<uint32_t, std::optional<std::vector<std::pair<gtsam::Point2, gtsam::Key>>>> savedTracks;
    std::unordered_map<uint32_t, std::optional<std::shared_ptr<SmartFactor>>> savedFactors;
    std::unordered_map<uint32_t, std::optional<gtsam::FactorIndex>> savedSlots;
    auto snapshotFeature = [&](uint32_t featId){
        if(savedTracks.find(featId) != savedTracks.end()){
            return;
        }
        auto tIt = activeFeatureTracks.find(featId);
        savedTracks[featId] = (tIt != activeFeatureTracks.end()) ? std::optional(tIt->second) : std::nullopt;
        auto fIt = smartFactors.find(featId);
        savedFactors[featId] = (fIt != smartFactors.end()) ? std::optional(fIt->second) : std::nullopt;
        auto sIt = smartFactorSlots.find(featId);
        savedSlots[featId] = (sIt != smartFactorSlots.end()) ? std::optional(sIt->second) : std::nullopt;
    };

    gtsam::FactorIndices removeIndices;
    std::optional<WindowFrame> droppedFrame;
    std::unique_ptr<gtsam::PreintegratedImuMeasurements> mergedPim;
    std::vector<Imu> frameRawImu = pkt.rawImu;
    const gtsam::PreintegratedImuMeasurements *imuToUse = pkt.imuDelta.get();
    std::unordered_set<uint32_t> featuresNeedingReinsert;

    // Collapse the previous frame if it is not a keyframe and raw IMU is available
    if(window.size() >= 2 && !window.back().isKeyframe && !window.back().rawImu.empty() && !pkt.rawImu.empty()){
        droppedFrame = std::move(window.back());
        window.pop_back();

        if(droppedFrame->imuFactorSlot.has_value()){
            removeIndices.push_back(*droppedFrame->imuFactorSlot);
        }
        if(droppedFrame->biasFactorSlot.has_value()){
            removeIndices.push_back(*droppedFrame->biasFactorSlot);
        }
        if(droppedFrame->depthFactorSlot.has_value()){
            removeIndices.push_back(*droppedFrame->depthFactorSlot);
        }
        if(droppedFrame->zuptVelFactorSlot.has_value()){
            removeIndices.push_back(*droppedFrame->zuptVelFactorSlot);
        }
        if(droppedFrame->zuptPoseFactorSlot.has_value()){
            removeIndices.push_back(*droppedFrame->zuptPoseFactorSlot);
        }

        const gtsam::Key dropPoseKey = X(droppedFrame->frameId);
        for(uint32_t featId : droppedFrame->observedFeatureIds){
            snapshotFeature(featId);

            auto trackIt = activeFeatureTracks.find(featId);
            if(trackIt != activeFeatureTracks.end()){
                auto &track = trackIt->second;
                track.erase(std::remove_if(track.begin(), track.end(), [dropPoseKey](const auto &obs){
                    return obs.second == dropPoseKey;
                }), track.end());
            }

            auto slotIt = smartFactorSlots.find(featId);
            if(slotIt != smartFactorSlots.end()){
                removeIndices.push_back(slotIt->second);
                smartFactors.erase(featId);
                smartFactorSlots.erase(slotIt);
                if(trackIt != activeFeatureTracks.end() && trackIt->second.size() >= 2){
                    featuresNeedingReinsert.insert(featId);
                }
            }
        }

        // Merge raw IMU across the collapsed frame
        std::vector<Imu> mergedRawImu;
        mergedRawImu.reserve(droppedFrame->rawImu.size() + pkt.rawImu.size());
        mergedRawImu.insert(mergedRawImu.end(), droppedFrame->rawImu.begin(), droppedFrame->rawImu.end());
        mergedRawImu.insert(mergedRawImu.end(), pkt.rawImu.begin(), pkt.rawImu.end());

        const auto &anchorFrame = window.back();
        const auto anchorBias = smootherEstimate.at<gtsam::imuBias::ConstantBias>(B(anchorFrame.frameId));
        mergedPim = std::make_unique<gtsam::PreintegratedImuMeasurements>(pkt.imuDelta->params(), anchorBias);

        // Skip shared boundary IMU samples
        double tPrev = anchorFrame.timestamp;
        for(const auto &s : mergedRawImu){
            const double sampleDt = s.timestamp - tPrev;
            if(sampleDt > 0.0){
                mergedPim->integrateMeasurement(s.a, s.w, sampleDt);
                tPrev = s.timestamp;
            }
        }

        imuToUse = mergedPim.get();
        frameRawImu = std::move(mergedRawImu);
    }

    const auto &prev = window.back();
    const double dt = std::max(1e-4, pkt.timestamp - prev.timestamp);

    const gtsam::NavState prevState(
        smootherEstimate.at<gtsam::Pose3>(X(prev.frameId)),
        smootherEstimate.at<gtsam::Vector3>(V(prev.frameId))
    );
    const auto prevWindowBias = smootherEstimate.at<gtsam::imuBias::ConstantBias>(B(prev.frameId));
    const gtsam::NavState predictedState = imuToUse->predict(prevState, prevWindowBias);

    stagingValues.insert(X(pkt.frameId), predictedState.pose());
    stagingValues.insert(V(pkt.frameId), predictedState.velocity());
    stagingValues.insert(B(pkt.frameId), prevWindowBias);

    // Track stagingGraph order so factor slots can be saved from newFactorsIndices
    const size_t imuGraphIdx = stagingGraph.size();
    stagingGraph.emplace_shared<gtsam::ImuFactor>(
        X(prev.frameId), V(prev.frameId), X(pkt.frameId), V(pkt.frameId), B(prev.frameId), *imuToUse
    );

    const double sqrtDt = std::sqrt(dt);
    gtsam::Vector6 biasRwSigmas;
    biasRwSigmas << config.accelBiasRwSigma * sqrtDt,
                    config.accelBiasRwSigma * sqrtDt,
                    config.accelBiasRwSigma * sqrtDt,
                    config.gyroBiasRwSigma * sqrtDt,
                    config.gyroBiasRwSigma * sqrtDt,
                    config.gyroBiasRwSigma * sqrtDt;
    auto biasRwNoise = gtsam::noiseModel::Diagonal::Sigmas(biasRwSigmas);

    const size_t biasGraphIdx = stagingGraph.size();
    stagingGraph.emplace_shared<gtsam::BetweenFactor<gtsam::imuBias::ConstantBias>>(
        B(prev.frameId), B(pkt.frameId), gtsam::imuBias::ConstantBias(), biasRwNoise
    );

    std::optional<size_t> depthGraphIdx;
    if(pkt.depth.has_value()){
        depthGraphIdx = stagingGraph.size();
        stagingGraph.emplace_shared<DepthFactor>(
            X(pkt.frameId), pkt.depth.value(), depthNoise, config.bodyToDepthSensor
        );
    }

    std::optional<size_t> zuptVelGraphIdx;
    std::optional<size_t> zuptPoseGraphIdx;
    if(pkt.isStationary){
        const gtsam::Vector3 zeroVel = gtsam::Vector3::Zero();
        stagingValues.update(X(pkt.frameId), prevState.pose());
        stagingValues.update(V(pkt.frameId), zeroVel);
        zuptVelGraphIdx = stagingGraph.size();
        stagingGraph.emplace_shared<gtsam::PriorFactor<gtsam::Vector3>>(
            V(pkt.frameId), zeroVel, zuptVelNoise
        );
        zuptPoseGraphIdx = stagingGraph.size();
        stagingGraph.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(
            X(prev.frameId), X(pkt.frameId), gtsam::Pose3::Identity(), zuptPoseNoise
        );
    }

    // Rebuild smart factors whose tracks changed or lost a collapsed frame
    std::vector<std::pair<uint32_t, size_t>> stagedVisualEntries;
    stagedVisualEntries.reserve(pkt.visualFeatures.size() + featuresNeedingReinsert.size());
    stagedFeatureIds.clear();

    auto stageSmartFactor = [&](uint32_t featId, const std::vector<std::pair<gtsam::Point2, gtsam::Key>> &track){
        auto factor = buildSmartFactor(visualNoise, Kptr, config.bodyToCameraFrame, smartParams, track);
        smartFactors[featId] = factor;

        const size_t graphIdx = stagingGraph.size();
        stagingGraph.push_back(factor);
        stagedVisualEntries.emplace_back(featId, graphIdx);
        stagedFeatureIds.push_back(featId);
    };

    for(const auto &feat : pkt.visualFeatures){
        snapshotFeature(feat.featureId);
        featuresNeedingReinsert.erase(feat.featureId);

        if(feat.isNewTrack){
            // Drop any old smart factor slot if this feature ID was reused
            activeFeatureTracks[feat.featureId].clear();
            auto slotIt = smartFactorSlots.find(feat.featureId);
            if(slotIt != smartFactorSlots.end()){
                removeIndices.push_back(slotIt->second);
                smartFactorSlots.erase(slotIt);
                smartFactors.erase(feat.featureId);
            }
        }

        auto &track = activeFeatureTracks[feat.featureId];
        track.emplace_back(feat.pt, X(pkt.frameId));

        if(track.size() >= 2){
            auto slotIt = smartFactorSlots.find(feat.featureId);
            if(slotIt != smartFactorSlots.end()){
                removeIndices.push_back(slotIt->second);
            }
            stageSmartFactor(feat.featureId, track);
        }
    }

    for(uint32_t featId : featuresNeedingReinsert){
        auto trackIt = activeFeatureTracks.find(featId);
        if(trackIt == activeFeatureTracks.end() || trackIt->second.size() < 2){
            continue;
        }
        stageSmartFactor(featId, trackIt->second);
    }

    std::sort(removeIndices.begin(), removeIndices.end());
    removeIndices.erase(std::unique(removeIndices.begin(), removeIndices.end()), removeIndices.end());

    auto result = solveSmoother(pkt.frameId, removeIndices);
    if(!result.has_value()){
        // Roll back track and window state if smoother update fails
        for(const auto &[featId, optTrack] : savedTracks){
            if(optTrack.has_value()){
                activeFeatureTracks[featId] = *optTrack;
            }else{
                activeFeatureTracks.erase(featId);
            }
        }
        for(const auto &[featId, optFactor] : savedFactors){
            if(optFactor.has_value()){
                smartFactors[featId] = *optFactor;
            }else{
                smartFactors.erase(featId);
            }
        }
        for(const auto &[featId, optSlot] : savedSlots){
            if(optSlot.has_value()){
                smartFactorSlots[featId] = *optSlot;
            }else{
                smartFactorSlots.erase(featId);
            }
        }
        if(droppedFrame.has_value()){
            window.push_back(std::move(*droppedFrame));
        }
        return;
    }

    WindowFrame wf;
    wf.frameId = pkt.frameId;
    wf.timestamp = pkt.timestamp;
    wf.isKeyframe = pkt.isKeyframe;
    wf.rawImu = std::move(frameRawImu);
    wf.observedFeatureIds.reserve(pkt.visualFeatures.size());
    for(const auto &feat : pkt.visualFeatures){
        wf.observedFeatureIds.push_back(feat.featureId);
    }

    const auto &newFactors = result->newFactorsIndices;
    if(imuGraphIdx < newFactors.size()){
        wf.imuFactorSlot = newFactors[imuGraphIdx];
    }
    if(biasGraphIdx < newFactors.size()){
        wf.biasFactorSlot = newFactors[biasGraphIdx];
    }
    if(depthGraphIdx.has_value() && *depthGraphIdx < newFactors.size()){
        wf.depthFactorSlot = newFactors[*depthGraphIdx];
    }
    if(zuptVelGraphIdx.has_value() && *zuptVelGraphIdx < newFactors.size()){
        wf.zuptVelFactorSlot = newFactors[*zuptVelGraphIdx];
    }
    if(zuptPoseGraphIdx.has_value() && *zuptPoseGraphIdx < newFactors.size()){
        wf.zuptPoseFactorSlot = newFactors[*zuptPoseGraphIdx];
    }
    for(const auto &[featId, graphIdx] : stagedVisualEntries){
        if(graphIdx < newFactors.size()){
            smartFactorSlots[featId] = newFactors[graphIdx];
        }
    }

    window.push_back(std::move(wf));

    // Refresh active global odometry edges using refined window poses before marginalization
    for(const auto &[kfA, kfB, factorIdx] : activeGlobalOdomEdges){
        if(factorIdx < globalPoseGraph.size() &&
        smootherEstimate.exists(X(kfA)) &&
        smootherEstimate.exists(X(kfB))){
            const auto refinedRel = smootherEstimate.at<gtsam::Pose3>(X(kfA)).between(
                smootherEstimate.at<gtsam::Pose3>(X(kfB))
            );
            globalPoseGraph.replace(
                factorIdx,
                std::make_shared<gtsam::BetweenFactor<gtsam::Pose3>>(X(kfA), X(kfB), refinedRel, odomNoise)
            );
        }
    }
    
    while(window.size() > config.slidingWindowSize){
        WindowFrame oldFrame = std::move(window.front());
        window.pop_front();
        marginalizeOldestFrame(oldFrame);
    }

    while(!activeGlobalOdomEdges.empty()){
        const uint32_t kfA = std::get<0>(activeGlobalOdomEdges.front());
        if(smootherEstimate.exists(X(kfA))){
            break;
        }
        activeGlobalOdomEdges.pop_front();
    }

    bool loopClosed = false;
    // Deal with global graph if this is a keyframe
    if(pkt.isKeyframe){
        const gtsam::Pose3 prevKfPose = smootherEstimate.exists(X(lastCommittedKeyframeId))
            ? smootherEstimate.at<gtsam::Pose3>(X(lastCommittedKeyframeId))
            : lastCommittedKeyframeNavState.pose();
        const auto relPose = prevKfPose.between(prevNavState.pose());
        const size_t odomFactorIdx = globalPoseGraph.size();
        globalPoseGraph.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(
            X(lastCommittedKeyframeId), X(pkt.frameId), relPose, odomNoise
        );
        activeGlobalOdomEdges.emplace_back(lastCommittedKeyframeId, pkt.frameId, odomFactorIdx);

        const gtsam::Pose3 initGlobalPose = globalPoseValues.exists(X(lastCommittedKeyframeId))
            ? (globalPoseValues.at<gtsam::Pose3>(X(lastCommittedKeyframeId)) * relPose)
            : prevNavState.pose();
        globalPoseValues.insert(X(pkt.frameId), initGlobalPose);
        if(pkt.depth.has_value()) globalPoseGraph.emplace_shared<DepthFactor>(X(pkt.frameId), *pkt.depth, depthNoise, config.bodyToDepthSensor);

        // Drain lf queue
        BackendLfPacket lfPkt;
        while(lf.getMatch(lfPkt)){
            if(processLfPacket(lfPkt)){
                loopClosed = true;
            }
        }

        solveGlobal(pkt.frameId);

        TrackedKeyframe kf;
        kf.frameId = pkt.frameId;
        kf.featureIds.reserve(pkt.visualFeatures.size());
        for(const auto &feat : pkt.visualFeatures){
            kf.featureIds.push_back(feat.featureId);
        }

        if(trackedKeyframes.size() >= config.keyframeWindowSize){
            trackedKeyframes.pop_front();
        }
        trackedKeyframes.push_back(std::move(kf));

        lastCommittedKeyframeId = pkt.frameId;
        lastCommittedKeyframeNavState = prevNavState;
    }

    lastCommittedFrameId = pkt.frameId;
    lastCommittedTimestamp = pkt.timestamp;
    lastCommittedNavState = prevNavState;

    publishPose(prevNavState.pose(), prevNavState.velocity(), pkt.timestamp);
    syncLfTracker(loopClosed);
}

void Backend::marginalizeOldestFrame(const WindowFrame &oldFrame){
    const gtsam::Key oldPoseKey = X(oldFrame.frameId);
    const gtsam::Key oldVelKey = V(oldFrame.frameId);
    const gtsam::Key oldBiasKey = B(oldFrame.frameId);

    gtsam::FactorIndices preRem;
    gtsam::NonlinearFactorGraph preGraph;
    std::vector<uint32_t> reStagedIds;
    std::vector<uint32_t> tracksToClear;

    for(uint32_t featId : oldFrame.observedFeatureIds){
        auto trackIt = activeFeatureTracks.find(featId);
        if(trackIt == activeFeatureTracks.end()){
            continue;
        }

        auto &track = trackIt->second;
        const size_t beforeSize = track.size();
        track.erase(std::remove_if(track.begin(), track.end(), [oldPoseKey](const auto &obs){
            return obs.second == oldPoseKey;
        }), track.end());

        if(track.empty()){
            tracksToClear.push_back(featId);
            continue;
        }

        auto slotIt = smartFactorSlots.find(featId);
        if(slotIt != smartFactorSlots.end() && track.size() < beforeSize){
            if(track.size() >= 2){
                // Remove oldest pose before marginalization so remaining views stay nonlinear
                preRem.push_back(slotIt->second);
                auto factor = buildSmartFactor(visualNoise, Kptr, config.bodyToCameraFrame, smartParams, track);
                smartFactors[featId] = factor;
                preGraph.push_back(factor);
                reStagedIds.push_back(featId);
            }else{
                // Fold single remaining view into the prior and drop track so it is not reused
                tracksToClear.push_back(featId);
            }
        }
    }

    gtsam::KeyVector leafKeyVec;
    if(smoother.valueExists(oldPoseKey)){
        leafKeyVec.push_back(oldPoseKey);
    }
    if(smoother.valueExists(oldVelKey)){
        leafKeyVec.push_back(oldVelKey);
    }
    if(smoother.valueExists(oldBiasKey)){
        leafKeyVec.push_back(oldBiasKey);
    }

    if(leafKeyVec.empty()){
        for(uint32_t featId : tracksToClear){
            activeFeatureTracks.erase(featId);
        }
        return;
    }

    // Collect remaining factors on the oldest frame while skipping updated smart factors
    const std::unordered_set<gtsam::FactorIndex> preRemSet(preRem.begin(), preRem.end());
    const auto &varIdx = smoother.getVariableIndex();
    const auto &graph = smoother.getFactorsUnsafe();

    std::unordered_set<gtsam::FactorIndex> leafFactorIndexSet;
    for(gtsam::Key k : leafKeyVec){
        auto it = varIdx.find(k);
        if(it != varIdx.end()){
            for(gtsam::FactorIndex idx : it->second){
                if(preRemSet.find(idx) == preRemSet.end()){
                    leafFactorIndexSet.insert(idx);
                }
            }
        }
    }

    gtsam::NonlinearFactorGraph leafSubGraph;
    gtsam::FactorIndices allRem = preRem;
    allRem.reserve(preRem.size() + leafFactorIndexSet.size());
    for(gtsam::FactorIndex idx : leafFactorIndexSet){
        allRem.push_back(idx);
        if(idx < graph.size() && graph.at(idx) != nullptr){
            leafSubGraph.push_back(graph.at(idx));
        }
    }
    std::sort(allRem.begin(), allRem.end());
    allRem.erase(std::unique(allRem.begin(), allRem.end()), allRem.end());

    try{
        // Eliminate oldest frame via QR Schur complement to avoid ISAM2 clique merges
        //const auto &linPoint = smootherEstimate;//smoother.calculateEstimate();
        const gtsam::KeySet subKeys = leafSubGraph.keys();
        gtsam::Values subValues;
        for(gtsam::Key k : subKeys){
            if(smootherEstimate.exists(k)){
                subValues.insert(k, smootherEstimate.at(k));
            }
        }

        if(!leafSubGraph.empty()){
            auto linSubGraph = leafSubGraph.linearize(subValues);
            const gtsam::KeySet linKeys = linSubGraph->keys();
            gtsam::KeyVector activeLeafKeys;
            for(gtsam::Key k : leafKeyVec){
                if(linKeys.exists(k)){
                    activeLeafKeys.push_back(k);
                }
            }

            if(!activeLeafKeys.empty()){
                auto [margBN, remGraph] = linSubGraph->eliminatePartialSequential(activeLeafKeys, gtsam::EliminateQR);
                (void)margBN;

                if(remGraph){
                    for(const auto &gf : *remGraph){
                        if(!gf || gf->keys().empty()){
                            continue;
                        }
                        gtsam::Values sepValues;
                        for(gtsam::Key k : gf->keys()){
                            if(subValues.exists(k)){
                                sepValues.insert(k, subValues.at(k));
                            }
                        }
                        preGraph.emplace_shared<gtsam::LinearContainerFactor>(gf, sepValues);
                    }
                }
            }
        }

        gtsam::ISAM2UpdateParams margParams;
        margParams.removeFactorIndices = allRem;
        auto res = smoother.update(preGraph, gtsam::Values(), margParams);

        for(size_t i = 0; i < reStagedIds.size() && i < res.newFactorsIndices.size(); i++){
            smartFactorSlots[reStagedIds[i]] = res.newFactorsIndices[i];
        }

        // Clear consumed IMU bias and stationary pose between slots on the new window front
        if(!window.empty()){
            window.front().imuFactorSlot.reset();
            window.front().biasFactorSlot.reset();
            window.front().zuptPoseFactorSlot.reset();
        }

        // Drop smart factors folded into the marginal prior
        for(gtsam::FactorIndex idx : leafFactorIndexSet){
            for(auto factorIt = smartFactorSlots.begin(); factorIt != smartFactorSlots.end(); ){
                if(factorIt->second == idx){
                    smartFactors.erase(factorIt->first);
                    factorIt = smartFactorSlots.erase(factorIt);
                }else{
                    factorIt++;
                }
            }
        }

        for(auto factorIt = smartFactorSlots.begin(); factorIt != smartFactorSlots.end(); ){
            if(leafFactorIndexSet.count(factorIt->second) > 0){
                smartFactors.erase(factorIt->first);
                factorIt = smartFactorSlots.erase(factorIt);
            }else{
                factorIt++;
            }
        }

        for(uint32_t featId : tracksToClear){
            activeFeatureTracks.erase(featId);
        }

        smootherEstimate = smoother.calculateEstimate();
    }catch(const std::exception &e){
        std::cerr << "WARNING: Exception in marginalizeOldestFrame at frame: " << oldFrame.frameId
                  << ": " << e.what() << std::endl;
    }
}

std::optional<gtsam::ISAM2Result> Backend::solveSmoother(uint32_t currentFrameId, const gtsam::FactorIndices &removeIndices){
    gtsam::ISAM2UpdateParams updateParams;
    updateParams.removeFactorIndices = removeIndices;

    // Solve the smoother graph
    gtsam::ISAM2Result result;
    try{
        result = smoother.update(stagingGraph, stagingValues, updateParams);
    }

    catch(const gtsam::IndeterminantLinearSystemException &e){
        std::cerr << "WARNING: GTSAM IndeterminantLinearSystemException at frame: " << currentFrameId
                  << ", nearby variable: " << gtsam::DefaultKeyFormatter(e.nearbyVariable()) << std::endl;
        stagingGraph.resize(0);
        stagingValues.clear();
        stagedFeatureIds.clear();
        return std::nullopt;
    }

    catch(const std::exception &e){
        std::cerr << "WARNING: Exception in solveSmoother at frame: " << currentFrameId
                  << ": " << e.what() << std::endl;
        stagingGraph.resize(0);
        stagingValues.clear();
        stagedFeatureIds.clear();
        return std::nullopt;
    }

    gtsam::Values estimate;
    try{
        estimate = smoother.calculateEstimate();
    }catch(const std::exception &e){
        std::cerr << "WARNING: Exception in calculateEstimate at frame: " << currentFrameId
                  << ": " << e.what() << std::endl;
        stagingGraph.resize(0);
        stagingValues.clear();
        stagedFeatureIds.clear();
        return std::nullopt;
    }

    // Make sure the smoother estimate contains the current frame
    if(estimate.find(X(currentFrameId)) == estimate.end()){
        stagingGraph.resize(0);
        stagingValues.clear();
        stagedFeatureIds.clear();
        return std::nullopt;
    }

    smootherEstimate = std::move(estimate);

    prevNavState = gtsam::NavState(
        smootherEstimate.at<gtsam::Pose3>(X(currentFrameId)),
        smootherEstimate.at<gtsam::Vector3>(V(currentFrameId))
    );
    prevBias = smootherEstimate.at<gtsam::imuBias::ConstantBias>(B(currentFrameId));

    fe.setBias(prevBias);

    stagingGraph.resize(0);
    stagingValues.clear();
    stagedFeatureIds.clear();

    return result;
}

}
