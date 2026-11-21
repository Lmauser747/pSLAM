#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <gtsam/geometry/Cal3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Quaternion.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/inference/Factor.h>
#include <gtsam/navigation/NavState.h>
#include <gtsam/nonlinear/ISAM2Result.h>
#include <gtsam/nonlinear/PriorFactor.h>
#include <memory>
#include <thread>
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

#include "backend/Backend.hpp"
#include "CoreTypes.hpp"
#include "frontend/Frontend.hpp"
#include "frontend/LfTracker.hpp"
#include "pool_slam/SystemTypes.hpp"


namespace PoolSLAM{

using gtsam::symbol_shorthand::B;
using gtsam::symbol_shorthand::L;
using gtsam::symbol_shorthand::V;
using gtsam::symbol_shorthand::X;

Backend::Backend(Frontend &frontend, LfTracker &lfTracker, const BackendConfig &cfg)
    : fe(frontend),
      lf(lfTracker),
      config(cfg),
      Kptr(std::make_shared<gtsam::Cal3_S2>(1.0, 1.0, 0.0, 0.0, 0.0)){
    gtsam::ISAM2Params isamParams;
    isamParams.relinearizeThreshold = config.relinearizeThreshold;
    isamParams.relinearizeSkip = config.relinearizeSkip;
    isamParams.factorization = gtsam::ISAM2Params::QR;
    isam = gtsam::ISAM2(isamParams);
    
    // Initial noise models
    gtsam::Vector6 initBiasSigmas;
    initBiasSigmas << config.initAccelBiasSigma, config.initAccelBiasSigma, config.initAccelBiasSigma,
        config.initGyroBiasSigma, config.initGyroBiasSigma, config.initGyroBiasSigma;
    initBiasNoise = gtsam::noiseModel::Diagonal::Sigmas(initBiasSigmas);

    initPoseNoise = gtsam::noiseModel::Diagonal::Sigmas(config.initPoseSigma);
    
    gtsam::Vector3 initVelSigmas;
    initVelSigmas << config.initVelSigma, config.initVelSigma, config.initVelSigma;
    initVelNoise = gtsam::noiseModel::Diagonal::Sigmas(initVelSigmas);

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
            prevNavState = hfPkt->imuDelta->predict(prevNavState, prevBias);
        }
        publishPose(prevNavState.pose(), prevNavState.velocity(), hfPkt->timestamp);

        // Do the factor graph solve if the packet is a keyframe
        if(hfPkt->isKeyframe){
           processKeyframe(*hfPkt);
        }
        
        prevFrameId = hfPkt->frameId;
        prevTimestamp = hfPkt->timestamp;
    }
}

void Backend::processKeyframe(const BackendHfPacket &pkt){
    double dt = pkt.timestamp - lastCommittedTimestamp;
    dt = std::max(1e-4, dt);
    
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

    // Drain lfPacket queue
    BackendLfPacket lfPkt;
    bool loopClosed = false;
    while(lf.getMatch(lfPkt)){
        processLfPacket(lfPkt);
        loopClosed = true;
    }

    // Update features found from the hfTracker and modify their smartProjectionFactor
    gtsam::FactorIndices removeIndices;
    stagedFeatureIds.clear();
    
    for(auto &feat : pkt.visualFeatures){
        auto it = smartFactors.find(feat.featureId);

        if(feat.isNewTrack || it == smartFactors.end()){
            auto factor = std::make_shared<SmartFactor>(visualNoise, Kptr, config.bodyToCameraFrame);
            factor->add(feat.pt, X(pkt.frameId));
            smartFactors[feat.featureId] = factor;
        }else{
            auto factor = it->second;
            factor->add(feat.pt, X(pkt.frameId));

            auto slotIt = smartFactorSlots.find(feat.featureId);

            // If this is the third track, add it to the list of indices to be removed
            if(slotIt != smartFactorSlots.end()){
                removeIndices.push_back(slotIt->second);
            }

            // Ad to staging graph
            stagingGraph.push_back(factor);
            stagedFeatureIds.push_back(feat.featureId);
        }
    }

    if(solveGraph(pkt.frameId, removeIndices)){
        lastCommittedFrameId = pkt.frameId;
        lastCommittedTimestamp = pkt.timestamp;

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

        publishPose(prevNavState.pose(), prevNavState.velocity(), pkt.timestamp);
        syncLfTracker(pkt.frameId, loopClosed);
    }
}

void Backend::processLfPacket(const BackendLfPacket &lfPkt){
    // make sure all of the keys exist
    bool curKeyValid = stagingValues.exists(X(lfPkt.currentFrameId)) || isam.valueExists(X(lfPkt.currentFrameId));
    bool matchKeyValid = stagingValues.exists(X(lfPkt.matchedFrameId)) || isam.valueExists(X(lfPkt.matchedFrameId));

    if(curKeyValid && matchKeyValid){
        auto betweenPose = config.bodyToCameraFrame * lfPkt.relativePose.inverse() * config.bodyToCameraFrame.inverse();
        if(lfPkt.noiseModel != nullptr){
            stagingGraph.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(X(lfPkt.matchedFrameId), X(lfPkt.currentFrameId), betweenPose, lfPkt.noiseModel);
        }else{ 
            // Fallback if noiseModel is null for soem reason
            gtsam::Vector6 sigmas;
            sigmas << 0.1, 0.1, 0.1, 0.1, 0.1, 0.1;
            auto baseGaussian = gtsam::noiseModel::Diagonal::Sigmas(sigmas);
            auto fallbackNoise = gtsam::noiseModel::Robust::Create(
                gtsam::noiseModel::mEstimator::Huber::Create(1.345), baseGaussian
            );
            stagingGraph.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(X(lfPkt.matchedFrameId), X(lfPkt.currentFrameId), betweenPose, fallbackNoise);
        }
    }else{
        return;
    }
}

bool Backend::solveGraph(uint32_t currentFrameId, const gtsam::FactorIndices &removeIndices){
    gtsam::ISAM2Result result;
    try{
        // Solve the ISAM2 graph
        result = isam.update(stagingGraph, stagingValues, removeIndices);
    }

    // Check for execptions
    catch(const gtsam::IndeterminantLinearSystemException &e){
        std::cerr << "WARNING: GTSAM IndeterminantLinearSystemException at frame: " << currentFrameId;
        stagingGraph.resize(0);
        stagingValues.clear();
        return false;
    }

    catch(const std::exception &e){
        std::cerr << "WARNING: Exception in PoolSLAM::solveGraph at frame: " << currentFrameId;
        stagingGraph.resize(0);
        stagingValues.clear();
        return false;
    }

    isamEstimate = isam.calculateEstimate();
    
    // Make sure the desried psoe exists
    if(!isamEstimate.exists(X(currentFrameId))){
        stagingValues.clear();
        stagingGraph.resize(0);
        return false;
    }

    auto newFactors = result.getNewFactorsIndices();
    
    // Update slots of remvoed factors
    auto startingOffset = newFactors.size() - stagedFeatureIds.size();
    for(size_t i = 0; i < stagedFeatureIds.size(); i++){
        smartFactorSlots[stagedFeatureIds[i]] = newFactors[i + startingOffset];
    }

    // Update state
    prevNavState = gtsam::NavState(isamEstimate.at<gtsam::Pose3>(X(currentFrameId)), isamEstimate.at<gtsam::Vector3>(V(currentFrameId)));
    prevBias = isamEstimate.at<gtsam::imuBias::ConstantBias>(B(currentFrameId));

    // Pass IMU bias to Frontend
    fe.setBias(prevBias);

    stagingValues.clear();
    stagingGraph.resize(0);

    return true;
}

void Backend::initializeSystem(const BackendHfPacket &pkt){
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
            initPoseNoise =  gtsam::noiseModel::Diagonal::Sigmas(modifiedSigma);
        }

        // Initialize pose
        gtsam::Rot3 attitude;
        if(pkt.initialAccel.has_value() && pkt.initialAccel->norm() > 1e-3){
            attitude = gtsam::Rot3(gtsam::Quaternion::FromTwoVectors(*pkt.initialAccel, gtsam::Vector3::UnitZ()));
        }

        prevNavState = gtsam::NavState(gtsam::Pose3(attitude, position), gtsam::Vector3(0, 0, 0));
    }

    // Intialize biases
    if(pkt.initialGyro.has_value()){
        prevBias = gtsam::imuBias::ConstantBias(gtsam::Vector3(0, 0, 0), *pkt.initialGyro);
    }else{
        prevBias = gtsam::imuBias::ConstantBias();
    }

    // Emplace initial factors
    stagingGraph.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(X(pkt.frameId), prevNavState.pose(), initPoseNoise);
    stagingGraph.emplace_shared<gtsam::PriorFactor<gtsam::Vector3>>(V(pkt.frameId), prevNavState.velocity(), initVelNoise);
    stagingGraph.emplace_shared<gtsam::PriorFactor<gtsam::imuBias::ConstantBias>>(B(pkt.frameId), prevBias, initBiasNoise);

    // Add initial values
    stagingValues.insert(X(pkt.frameId), prevNavState.pose());
    stagingValues.insert(V(pkt.frameId), prevNavState.velocity());
    stagingValues.insert(B(pkt.frameId), prevBias);

    // Do initial solve
    if(!solveGraph(pkt.frameId)){
        return;
    }

    isInitialized = true;

    lastCommittedFrameId = pkt.frameId;
    lastCommittedTimestamp = pkt.timestamp;
    prevFrameId = pkt.frameId;
    prevTimestamp = pkt.timestamp;
}

void Backend::syncLfTracker(uint32_t currentFrameId, bool forceAll){
    for(auto &kf : trackedKeyframes){
        BackendKeyframeUpdate update;
        update.id = kf.frameId;
        if(!isamEstimate.exists(X(kf.frameId)));
        update.pose = isamEstimate.at<gtsam::Pose3>(X(kf.frameId)) * config.bodyToCameraFrame;
        update.landmarks.reserve(kf.lastLandmarkCount);
        size_t count = 0;
        for(auto featId : kf.featureIds){
            auto sfIt = smartFactors.find(featId);
            if(sfIt == smartFactors.end() || sfIt->second == nullptr){
                continue;
            }
            auto tri = sfIt->second->point(isamEstimate);
            if(tri.has_value()){
                auto ptCam = update.pose.transformTo(tri.value());
                if(ptCam.z() <= 0.0){
                    continue;
                }
                update.landmarks.emplace_back(ptCam, featId);
                count++;
            } 
        }
        if(count > kf.lastLandmarkCount || forceAll){
            kf.lastLandmarkCount = count;
            lf.updateKeyframe(update);
        }
    }
}

}