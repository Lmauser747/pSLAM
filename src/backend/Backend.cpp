#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <gtsam/nonlinear/FixedLagSmoother.h>
#include <gtsam/nonlinear/IncrementalFixedLagSmoother.h>
#include <iostream>
#include <memory>
#include <thread>
#include <unordered_set>
#include <utility>

#include <gtsam/base/Vector.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/LossFunctions.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/navigation/ImuBias.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/ISAM2Params.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/geometry/Cal3_S2.h>
#include <gtsam/slam/SmartProjectionPoseFactor.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Quaternion.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/inference/Factor.h>
#include <gtsam/linear/linearExceptions.h>
#include <gtsam/navigation/NavState.h>
#include <gtsam/nonlinear/ISAM2Result.h>
#include <gtsam/nonlinear/PriorFactor.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>

#include "backend/Backend.hpp"
#include "CoreTypes.hpp"
#include "frontend/Frontend.hpp"
#include "frontend/LfTracker.hpp"
#include "pool_slam/SystemTypes.hpp"

namespace PoolSLAM{

using gtsam::symbol_shorthand::B;
using gtsam::symbol_shorthand::V;
using gtsam::symbol_shorthand::X;

Backend::Backend(Frontend &frontend, LfTracker &lfTracker, const BackendConfig &cfg)
    : fe(frontend),
      lf(lfTracker),
      config(cfg),
      Kptr(std::make_shared<gtsam::Cal3_S2>(1.0, 1.0, 0.0, 0.0, 0.0)){
    gtsam::ISAM2Params smootherParams;
    smootherParams.findUnusedFactorSlots = true;
    smootherParams.relinearizeThreshold = config.relinearizeThreshold;
    smootherParams.relinearizeSkip = config.relinearizeSkip;
    smootherParams.factorization = gtsam::ISAM2Params::CHOLESKY;
    smoother = gtsam::IncrementalFixedLagSmoother(config.smootherLag, smootherParams);
    smartParams.degeneracyMode = gtsam::ZERO_ON_DEGENERACY;
    
    // Initial noise models
    gtsam::Vector6 initBiasSigmas;
    initBiasSigmas << config.initAccelBiasSigma, config.initAccelBiasSigma, config.initAccelBiasSigma,
        config.initGyroBiasSigma, config.initGyroBiasSigma, config.initGyroBiasSigma;
    initBiasNoise = gtsam::noiseModel::Diagonal::Sigmas(initBiasSigmas);

    initPoseNoise = gtsam::noiseModel::Diagonal::Sigmas(config.initPoseSigma);
    
    gtsam::Vector3 initVelSigmas;
    initVelSigmas << config.initVelSigma, config.initVelSigma, config.initVelSigma;
    initVelNoise = gtsam::noiseModel::Diagonal::Sigmas(initVelSigmas);

    // Odometry noise model for global pose graph
    gtsam::Vector6 odomSigmas;
    odomSigmas << 0.05, 0.05, 0.05, 0.01, 0.01, 0.01;
    odomNoise = gtsam::noiseModel::Diagonal::Sigmas(odomSigmas);

    auto baseDepthNoise = gtsam::noiseModel::Isotropic::Sigma(1, config.depthSigma);
    depthNoise = gtsam::noiseModel::Robust::Create(
        gtsam::noiseModel::mEstimator::Huber::Create(config.depthHuberK),
        baseDepthNoise
    );

    for(size_t i = 0; i < 2; i++){
        poseBuffer[i].timestamp = 0.0;
        poseBuffer[i].t = Eigen::Vector3d::Zero();
        poseBuffer[i].r = Eigen::Quaterniond::Identity();
        poseBuffer[i].v = Eigen::Vector3d::Zero();
    }
    prevBias = gtsam::imuBias::ConstantBias();

    visualNoise = gtsam::noiseModel::Isotropic::Sigma(2, 1.0 / config.K(0, 0));
}

Backend::~Backend(){
    end();
}

Pose Backend::getCurrentPose() const{
    const uint8_t idx = activePose.load(std::memory_order_acquire);
    return poseBuffer[idx];
}

void Backend::start(){
    if(runBackend.load(std::memory_order_acquire)){
        return;
    }
    if(backendThread.joinable()){
        backendThread.join();
    }
    runBackend.store(true, std::memory_order_release);
    backendThread = std::thread(&Backend::backendTask, this);
}

void Backend::end(){
    if(!runBackend.load(std::memory_order_acquire)){
        return;
    }
    runBackend.store(false, std::memory_order_release);
    // Ensure fe.getHfPacket() unblocks if backendThread is waiting on c_hfQueue
    fe.end();
    if(backendThread.joinable()){
        backendThread.join();
    }
}

void Backend::backendTask(){
    while(runBackend.load(std::memory_order_acquire)){
        auto hfPkt = fe.getHfPacket();
        if(!hfPkt){
            break;
        }

        if(!isInitialized){
            initializeSystem(*hfPkt);
            publishPose(prevNavState.pose(), prevNavState.velocity(), hfPkt->timestamp);
            continue;
        }

        if(hfPkt->frameId <= prevFrameId){
            continue;
        }

        // Propogate state with Imu data for high frequency navigation tasks
        if(hfPkt->imuDelta != nullptr && hfPkt->imuDelta->deltaTij() > 0.0){
            prevNavState = hfPkt->imuDelta->predict(lastCommittedNavState, prevBias);
        }
        publishPose(prevNavState.pose(), prevNavState.velocity(), hfPkt->timestamp);

        // Solve smoother graph
        processFrame(*hfPkt);
        
        prevFrameId = hfPkt->frameId;
        prevTimestamp = hfPkt->timestamp;
    }
}

void Backend::processFrame(const BackendHfPacket &pkt){
    double dt = pkt.timestamp - lastCommittedTimestamp;
    dt = std::max(1e-4, dt);
    double imuDt = pkt.imuDelta ? pkt.imuDelta->deltaTij() : 0.0;
    if(std::abs(dt - imuDt) > 0.05){
        std::cerr << "[Backend DIAG] Keyframe " << pkt.frameId
                  << " dt mismatch: dataset dt=" << dt
                  << "s, imuDelta dt=" << imuDt << "s" << std::endl;
    }
    
    // Insert values
    stagingValues.insert(X(pkt.frameId), prevNavState.pose());
    stagingValues.insert(V(pkt.frameId), prevNavState.velocity());
    stagingValues.insert(B(pkt.frameId), prevBias);

    // Insert non-visual factors
    stagingGraph.emplace_shared<gtsam::ImuFactor>(X(lastCommittedFrameId), V(lastCommittedFrameId), X(pkt.frameId), V(pkt.frameId), B(lastCommittedFrameId), *pkt.imuDelta);

    const double sqrtDt = std::sqrt(dt);
    gtsam::Vector6 biasRwSigmas;
    biasRwSigmas << config.accelBiasRwSigma * sqrtDt,
                    config.accelBiasRwSigma * sqrtDt,
                    config.accelBiasRwSigma * sqrtDt,
                    config.gyroBiasRwSigma * sqrtDt,
                    config.gyroBiasRwSigma * sqrtDt,
                    config.gyroBiasRwSigma * sqrtDt;
    auto biasRwNoise = gtsam::noiseModel::Diagonal::Sigmas(biasRwSigmas);

    stagingGraph.emplace_shared<gtsam::BetweenFactor<gtsam::imuBias::ConstantBias>>(B(lastCommittedFrameId), B(pkt.frameId), gtsam::imuBias::ConstantBias(), biasRwNoise);
    if(pkt.depth.has_value()){
         stagingGraph.emplace_shared<DepthFactor>(X(pkt.frameId), pkt.depth.value(), depthNoise, config.bodyToDepthSensor);
    }

    // Update features found from the hfTracker and modify their smartProjectionFactor
    gtsam::FactorIndices removeIndices;
    stagedFeatureIds.clear();

    const auto &smootherTimestamps = smoother.timestamps();
    const double minAllowedTime = pkt.timestamp - config.smootherLag + 1e-4;

    std::unordered_set<uint32_t> currentVisualFeatures;
    for(const auto &feat : pkt.visualFeatures){
        currentVisualFeatures.insert(feat.featureId);
        if(feat.isNewTrack){
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

        // Prune observations outside the smoother lag window
        track.erase(std::remove_if(track.begin(), track.end(), [&](const auto &obs){
            if(obs.second == X(pkt.frameId)){
                return false;
            }
            auto it = smootherTimestamps.find(obs.second);
            if(it == smootherTimestamps.end()){
                return true;
            }
            return it->second < minAllowedTime;
        }), track.end());

        // If track has at least 2 views in the sliding window, create/replace factor
        if(track.size() >= 2){
            auto factor = std::make_shared<SmartFactor>(visualNoise, Kptr, config.bodyToCameraFrame, smartParams);
            for(const auto &obs : track){
                factor->add(obs.first, obs.second);
            }
            smartFactors[feat.featureId] = factor;

            // If a factor for this feature already exists in the smoother, mark it for removal
            auto slotIt = smartFactorSlots.find(feat.featureId);
            if(slotIt != smartFactorSlots.end()){
                removeIndices.push_back(slotIt->second);
            }

            stagingGraph.push_back(factor);
            stagedFeatureIds.push_back(feat.featureId);
        }else{
            // Track no longer has >= 2 views in the window; remove existing factor if any
            auto slotIt = smartFactorSlots.find(feat.featureId);
            if(slotIt != smartFactorSlots.end()){
                removeIndices.push_back(slotIt->second);
                smartFactorSlots.erase(slotIt);
                smartFactors.erase(feat.featureId);
            }
        }
    }

    // Sweep factors not detected in the current packet whose observations crossed the horizon
    for(auto it = smartFactorSlots.begin(); it != smartFactorSlots.end(); ){
        uint32_t featId = it->first;
        if(currentVisualFeatures.count(featId)){
            it++;
            continue;
        }

        auto trackIt = activeFeatureTracks.find(featId);
        if(trackIt != activeFeatureTracks.end()){
            auto &track = trackIt->second;
            size_t beforeSize = track.size();
            track.erase(std::remove_if(track.begin(), track.end(), [&](const auto &obs){
                auto tsIt = smootherTimestamps.find(obs.second);
                if(tsIt == smootherTimestamps.end()){
                    return true;
                }
                return tsIt->second < minAllowedTime;
            }), track.end());

            if(track.size() < 2){
                removeIndices.push_back(it->second);
                smartFactors.erase(featId);
                it = smartFactorSlots.erase(it);
            } else if(track.size() < beforeSize){
                auto factor = std::make_shared<SmartFactor>(visualNoise, Kptr, config.bodyToCameraFrame, smartParams);
                for(const auto &obs : track){
                    factor->add(obs.first, obs.second);
                }
                smartFactors[featId] = factor;
                removeIndices.push_back(it->second);
                stagingGraph.push_back(factor);
                stagedFeatureIds.push_back(featId);
                it++;
            } else {
                it++;
            }
        } else {
            removeIndices.push_back(it->second);
            smartFactors.erase(featId);
            it = smartFactorSlots.erase(it);
        }
    }

    std::sort(removeIndices.begin(), removeIndices.end());
    removeIndices.erase(std::unique(removeIndices.begin(), removeIndices.end()), removeIndices.end());

    if(solveSmoother(pkt.frameId, pkt.timestamp, removeIndices)){

        bool loopClosed = false;
        // Deal with global graph if this is a keyframe
        if(pkt.isKeyframe){
            // Add odometry factor to global graph
            auto relPose = lastCommittedKeyframeNavState.pose().between(prevNavState.pose());
            globalPoseGraph.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(X(lastCommittedKeyframeId), X(pkt.frameId), relPose, odomNoise);
            globalPoseValues.insert(X(pkt.frameId), prevNavState.pose());

            // Drain lfPacket queue
            BackendLfPacket lfPkt;
            while(lf.getMatch(lfPkt)){
                processLfPacket(lfPkt);
                loopClosed = true;
            }

            solveGlobal(pkt.frameId);

            // Add the keyframe to the tracked keyframes
            TrackedKeyframe kf;
            kf.frameId = pkt.frameId;
            kf.featureIds.reserve(pkt.visualFeatures.size());
            for(auto &feat : pkt.visualFeatures){
                kf.featureIds.push_back(feat.featureId);
            }

            if(trackedKeyframes.size() >= config.keyframeWindowSize){
                trackedKeyframes.pop_front();
            }
            trackedKeyframes.push_back(kf);
            
            lastCommittedKeyframeId = pkt.frameId;
            lastCommittedKeyframeNavState = prevNavState;
        }
    
        lastCommittedFrameId = pkt.frameId;
        lastCommittedTimestamp = pkt.timestamp;
        lastCommittedNavState = prevNavState;

        publishPose(prevNavState.pose(), prevNavState.velocity(), pkt.timestamp);
        syncLfTracker(loopClosed);
    }else{
        for(const auto &feat : pkt.visualFeatures){
            activeFeatureTracks[feat.featureId].pop_back();
        }
    }
}

void Backend::processLfPacket(const BackendLfPacket &lfPkt){
    // make sure all of the keys exist
    bool curKeyValid = globalPoseValues.exists(X(lfPkt.currentFrameId)); 
    bool matchKeyValid = globalPoseValues.exists(X(lfPkt.matchedFrameId));

    if(curKeyValid && matchKeyValid){
        auto betweenPose = config.bodyToCameraFrame * lfPkt.relativePose.inverse() * config.bodyToCameraFrame.inverse();
        gtsam::SharedNoiseModel loopNoise;
        if(lfPkt.noiseModel != nullptr){
            loopNoise = lfPkt.noiseModel;
            auto robust = std::dynamic_pointer_cast<gtsam::noiseModel::Robust>(lfPkt.noiseModel);
            auto baseModel = robust ? robust->noise() : lfPkt.noiseModel;
            auto gaussian = std::dynamic_pointer_cast<gtsam::noiseModel::Gaussian>(baseModel);
            if(gaussian && gaussian->dim() == 6){
                const gtsam::Matrix6 Ad = config.bodyToCameraFrame.AdjointMap();
                gtsam::Matrix6 covBody = Ad * gaussian->covariance() * Ad.transpose();
                covBody = 0.5 * (covBody + covBody.transpose());
                auto bodyGaussian = gtsam::noiseModel::Gaussian::Covariance(covBody);
                loopNoise = robust
                    ? gtsam::noiseModel::Robust::Create(robust->robust(), bodyGaussian)
                    : gtsam::SharedNoiseModel(bodyGaussian);
            }
        }else{
            // Fallback if noiseModel is null for some reason
            gtsam::Vector6 sigmas;
            sigmas << 0.1, 0.1, 0.1, 0.1, 0.1, 0.1;
            auto baseGaussian = gtsam::noiseModel::Diagonal::Sigmas(sigmas);
            loopNoise = gtsam::noiseModel::Robust::Create(
                gtsam::noiseModel::mEstimator::Huber::Create(1.345), baseGaussian
            );
        }
        globalPoseGraph.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(X(lfPkt.matchedFrameId), X(lfPkt.currentFrameId), betweenPose, loopNoise);
    }else{
        return;
    }
}

bool Backend::solveSmoother(uint32_t currentFrameId, double timestamp, const gtsam::FactorIndices &removeIndices){
    auto keyToTimestamp = gtsam::IncrementalFixedLagSmoother::KeyTimestampMap();
    for(const auto& value : stagingValues){
        keyToTimestamp[value.key] = timestamp;
    }

    // Solve the smoother graph
    gtsam::FixedLagSmootherResult result;    
    try{
        result = smoother.update(stagingGraph, stagingValues, keyToTimestamp, removeIndices);
    }

    catch(const gtsam::IndeterminantLinearSystemException &e){
        std::cerr << "WARNING: GTSAM IndeterminantLinearSystemException at frame: " << currentFrameId
                  << ", nearby variable: " << gtsam::DefaultKeyFormatter(e.nearbyVariable()) << std::endl;
        stagingGraph.resize(0);
        stagingValues.clear();
        stagedFeatureIds.clear();
        return false;
    }

    catch(const std::exception &e){
        std::cerr << "WARNING: Exception in solveSmoother at frame: " << currentFrameId
                  << ": " << e.what() << std::endl;
        stagingGraph.resize(0);
        stagingValues.clear();
        stagedFeatureIds.clear();
        return false;
    }


    smootherEstimate = smoother.calculateEstimate();

    // Makse sure the smoother estimate contains the current frame
    if(smootherEstimate.find(X(currentFrameId)) == smootherEstimate.end()){
        stagingGraph.resize(0);
        stagingValues.clear();
        stagedFeatureIds.clear();
        return false;
    }

    // Remove marginalized tracks
    auto deletedKeys = result.getKeysOfDeletedNodes();

    if(!deletedKeys.empty()){
        for(auto trackIt = activeFeatureTracks.begin(); trackIt != activeFeatureTracks.end(); ){
            auto &track = trackIt->second;
            
            track.erase(std::remove_if(track.begin(), track.end(), [&deletedKeys](const auto &obs){
                return deletedKeys.find(obs.second) != deletedKeys.end();
            }), track.end());

            if(track.empty()){
                trackIt = activeFeatureTracks.erase(trackIt);
            }else{
                trackIt++;
            }
        }
    }

    // Remove marginalized factors
    auto deletedFactors = result.getDeletedFactorIndices();
    if(!deletedFactors.empty()){
        for(auto factorIt = smartFactorSlots.begin(); factorIt != smartFactorSlots.end(); ){
            if(std::find(deletedFactors.begin(), deletedFactors.end(), factorIt->second) != deletedFactors.end()){
                smartFactors.erase(factorIt->first);
                factorIt = smartFactorSlots.erase(factorIt);
            }else{
                factorIt++;
            }
        }
    }

    // Get new factors
    auto newFactors = smoother.getISAM2Result().newFactorsIndices;
    if(newFactors.size() >= stagedFeatureIds.size()){
        auto newFactorOffset = newFactors.size() - stagedFeatureIds.size();
        for(size_t i = 0; i < stagedFeatureIds.size(); i++){
            smartFactorSlots[stagedFeatureIds[i]] = newFactors[i + newFactorOffset];
        }
    }

    prevNavState = gtsam::NavState(smootherEstimate.at<gtsam::Pose3>(X(currentFrameId)), smootherEstimate.at<gtsam::Vector3>(V(currentFrameId)));
    prevBias = smootherEstimate.at<gtsam::imuBias::ConstantBias>(B(currentFrameId));

    fe.setBias(prevBias);

    stagingGraph.resize(0);
    stagingValues.clear();
    stagedFeatureIds.clear();

    return true;
}

bool Backend::solveGlobal(uint32_t currentFrameId){
    auto opt = gtsam::LevenbergMarquardtOptimizer(globalPoseGraph, globalPoseValues);
    gtsam::Values newValues;
    try{
        newValues = opt.optimize();
    }

    catch(const gtsam::IndeterminantLinearSystemException &e){
        std::cerr << "WARNING: GTSAM IndeterminantLinearSystemException at frame: " << currentFrameId
                  << ", nearby variable: " << gtsam::DefaultKeyFormatter(e.nearbyVariable()) << std::endl;
        return false;
    }

    catch(const std::exception &e){
        std::cerr << "WARNING: Exception in solveGlobal at frame: " << currentFrameId
                  << ": " << e.what() << std::endl;
        return false;
    }

    globalPoseValues = newValues;

    return true;
}

void Backend::initializeSystem(const BackendHfPacket &pkt){
    gtsam::SharedNoiseModel priorPoseNoise = initPoseNoise;

    // If the config pose is given, use that
    if(config.initialPose.has_value()){
        prevNavState = gtsam::NavState(*config.initialPose, gtsam::Vector3(0, 0, 0));
    }else{
        // Initialize position
        gtsam::Point3 position;
        position = gtsam::Point3::Zero();
        if(pkt.depth.has_value()){
            position.z() = *pkt.depth - config.bodyToDepthSensor.z();
        }else{
            // Set the Z axis noise model to be very loose if there is not a good initial Z estimate
            auto modifiedSigma = config.initPoseSigma;
            modifiedSigma[5] = 100.0;
            priorPoseNoise = gtsam::noiseModel::Diagonal::Sigmas(modifiedSigma);
        }

        // Initialize pose
        gtsam::Rot3 attitude;
        if(pkt.initialAccel.has_value() && pkt.initialAccel->norm() > 1e-3){
            attitude = gtsam::Rot3(gtsam::Quaternion::FromTwoVectors(*pkt.initialAccel, gtsam::Vector3::UnitZ()));
        }

        prevNavState = gtsam::NavState(gtsam::Pose3(attitude, position), gtsam::Vector3(0, 0, 0));
    }

    publishPose(prevNavState.pose(), prevNavState.velocity(), pkt.timestamp);

    // Intialize biases
    if(pkt.initialGyro.has_value()){
        prevBias = gtsam::imuBias::ConstantBias(gtsam::Vector3(0, 0, 0), *pkt.initialGyro);
    }else{
        prevBias = gtsam::imuBias::ConstantBias();
    }

    // Emplace initial factors for smoother
    stagingGraph.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(X(pkt.frameId), prevNavState.pose(), priorPoseNoise);
    stagingGraph.emplace_shared<gtsam::PriorFactor<gtsam::Vector3>>(V(pkt.frameId), prevNavState.velocity(), initVelNoise);
    stagingGraph.emplace_shared<gtsam::PriorFactor<gtsam::imuBias::ConstantBias>>(B(pkt.frameId), prevBias, initBiasNoise);

    // Add initial values to smoother
    stagingValues.insert(X(pkt.frameId), prevNavState.pose());
    stagingValues.insert(V(pkt.frameId), prevNavState.velocity());
    stagingValues.insert(B(pkt.frameId), prevBias);

    // Seed activeFeatureTracks 
    for(const auto &feat : pkt.visualFeatures){
        activeFeatureTracks[feat.featureId].emplace_back(feat.pt, X(pkt.frameId));
    }

    // Do initial solve of smoother
    if(!solveSmoother(pkt.frameId, pkt.timestamp)){
        activeFeatureTracks.clear();
        return;
    }

    // Add prior and initial pose to the global graph
    globalPoseGraph.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(X(pkt.frameId), prevNavState.pose(), priorPoseNoise);
    globalPoseValues.insert(X(pkt.frameId), prevNavState.pose());

    TrackedKeyframe kf;
    kf.frameId = pkt.frameId;
    kf.lastLandmarkCount = 0;
    for(const auto &feat : pkt.visualFeatures){
        kf.featureIds.push_back(feat.featureId);
    }
    trackedKeyframes.push_back(kf);
    
    isInitialized = true;

    lastCommittedFrameId = pkt.frameId;
    lastCommittedKeyframeId = pkt.frameId; 
    lastCommittedTimestamp = pkt.timestamp;
    lastCommittedNavState = prevNavState;
    lastCommittedKeyframeNavState = prevNavState;
    prevFrameId = pkt.frameId;
    prevTimestamp = pkt.timestamp;
}

void Backend::syncLfTracker(bool forceAll){
    for(auto &kf : trackedKeyframes){
        BackendKeyframeUpdate update;
        update.id = kf.frameId;
        if(!globalPoseValues.exists(X(kf.frameId))){
            continue;
        }
        update.pose = globalPoseValues.at<gtsam::Pose3>(X(kf.frameId)) * config.bodyToCameraFrame;
        update.landmarks.reserve(kf.lastLandmarkCount);
        size_t count = 0;
        for(auto featId : kf.featureIds){
            auto sfIt = smartFactors.find(featId);
            if(sfIt == smartFactors.end() || sfIt->second == nullptr){
                continue;
            }

            bool allKeysExist = true;
            for(gtsam::Key k : sfIt->second->keys()){
                if(!smootherEstimate.exists(k)){
                    allKeysExist = false;
                    break;
                }
            }
            if(!allKeysExist){
                continue;
            }

            try{
                auto tri = sfIt->second->point(smootherEstimate);
                if(tri.has_value()){
                    auto ptCam = update.pose.transformTo(tri.value());
                    if(ptCam.z() <= 0.0){
                        continue;
                    }
                    update.landmarks.push_back({ptCam, featId});
                    count++;
                }
            }catch(const std::exception &){
                continue;
            }
        }
        if(count > kf.lastLandmarkCount || forceAll){
            kf.lastLandmarkCount = count;
            lf.updateKeyframe(update);
        }
    }
}

void Backend::publishPose(const gtsam::Pose3 &pose, const gtsam::Vector3 &velocity, double timestamp){
    uint8_t inactivePose = activePose.load(std::memory_order_relaxed) == 0 ? 1 : 0;
    poseBuffer[inactivePose].timestamp = timestamp;
    poseBuffer[inactivePose].t = pose.translation();
    poseBuffer[inactivePose].r = pose.rotation().toQuaternion();
    poseBuffer[inactivePose].v = velocity;

    activePose.store(inactivePose, std::memory_order_release);
}

}