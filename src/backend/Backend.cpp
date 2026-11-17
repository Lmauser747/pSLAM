#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <thread>
#include <utility>

#include <gtsam/base/Matrix.h>
#include <gtsam/base/Vector.h>
#include <gtsam/base/types.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/LossFunctions.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/linear/linearExceptions.h>
#include <gtsam/navigation/ImuBias.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/NavState.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/ISAM2Params.h>
#include <gtsam/nonlinear/NoiseModelFactorN.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/nonlinear/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>

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

// ============================================================================
// DepthFactor
// ============================================================================

DepthFactor::DepthFactor(
    gtsam::Key poseKey,
    double depthIn,
    const gtsam::SharedNoiseModel &model,
    const gtsam::Point3 &leverArm)
    : gtsam::NoiseModelFactorN<gtsam::Pose3>(model, poseKey),
      measuredDepth(depthIn),
      bodyToSensor(leverArm){}

gtsam::Vector DepthFactor::evaluateError(
    const gtsam::Pose3 &pose,
    gtsam::OptionalMatrixType H) const{
    gtsam::Matrix36 dSensor_dPose;
    const gtsam::Point3 sensorWorld =
        pose.transformFrom(bodyToSensor, H ? &dSensor_dPose : nullptr);

    if(H){
        *H = dSensor_dPose.row(2);
    }

    return (gtsam::Vector(1) << (sensorWorld.z() - measuredDepth)).finished();
}

// ============================================================================
// Backend Lifecycle & Public API
// ============================================================================

Backend::Backend(Frontend &frontend, LfTracker &lfTracker, const BackendConfig &cfg)
    : fe(frontend),
      lf(lfTracker),
      config(cfg),
      bk(stagingGraph, stagingValues, isamEstimate, cfg.bodyToCameraFrame, cfg.bkConfig){
    gtsam::ISAM2Params isamParams;
    isamParams.relinearizeThreshold = config.relinearizeThreshold;
    isamParams.relinearizeSkip = config.relinearizeSkip;
    isam = gtsam::ISAM2(isamParams);

    initPoseNoise = gtsam::noiseModel::Diagonal::Sigmas(config.initPoseSigma);
    initVelNoise = gtsam::noiseModel::Isotropic::Sigma(3, config.initVelSigma);

    gtsam::Vector6 initBiasSigmas;
    initBiasSigmas << config.initAccelBiasSigma, config.initAccelBiasSigma, config.initAccelBiasSigma,
                      config.initGyroBiasSigma, config.initGyroBiasSigma, config.initGyroBiasSigma;
    initBiasNoise = gtsam::noiseModel::Diagonal::Sigmas(initBiasSigmas);

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

// ============================================================================
// Worker Loop & Factor Graph Assembly
// ============================================================================

void Backend::backendTask(){
    while(runBackend.load(std::memory_order_acquire)){
        std::unique_ptr<BackendHfPacket> hfPkt = fe.getHfPacket();
        if(!hfPkt){
            break;
        }

        if(!isInitialized){
            initializeSystem(*hfPkt);
        }else{
            if(hfPkt->frameId <= prevFrameId){
                continue;
            }
            processHfPacket(*hfPkt);
        }

        if(hfPkt->isKeyframe){
            recordKeyframeFeatures(*hfPkt);
        }

        bool loopClosed = false;
        BackendLfPacket lfPkt;
        while(lf.getMatch(lfPkt)){
            processLfPacket(lfPkt);
            loopClosed = true;
        }

        optimizeAndPublish(hfPkt->frameId, hfPkt->timestamp, loopClosed);
    }
}

void Backend::initializeSystem(const BackendHfPacket &pkt){
    const uint32_t frameId = pkt.frameId;

    const double initZ = pkt.depth.has_value()
                             ? (*pkt.depth - config.bodyToDepthSensor.z())
                             : 0.0;

    const gtsam::Pose3 initPose(gtsam::Rot3(), gtsam::Point3(0.0, 0.0, initZ));
    const gtsam::Vector3 initVel = gtsam::Vector3::Zero();
    const gtsam::imuBias::ConstantBias initBias;

    stagingValues.insert(X(frameId), initPose);
    stagingValues.insert(V(frameId), initVel);
    stagingValues.insert(B(frameId), initBias);

    stagingGraph.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(
        X(frameId), initPose, initPoseNoise);
    stagingGraph.emplace_shared<gtsam::PriorFactor<gtsam::Vector3>>(
        V(frameId), initVel, initVelNoise);
    stagingGraph.emplace_shared<gtsam::PriorFactor<gtsam::imuBias::ConstantBias>>(
        B(frameId), initBias, initBiasNoise);

    bk.startFrame(frameId);
    for(size_t i = 0; i < pkt.visualMeasurements.size(); i++){
        const auto &meas = pkt.visualMeasurements[i];
        bk.updateBookeeper(meas.featureId, meas.pt);
    }
    bk.pruneDeadTracks();

    prevFrameId = frameId;
    prevTimestamp = pkt.timestamp;
    prevNavState = gtsam::NavState(initPose, initVel);
    prevBias = initBias;
    isInitialized = true;
}

void Backend::processHfPacket(const BackendHfPacket &pkt){
    const uint32_t frameId = pkt.frameId;
    const bool isContiguousFrame = (frameId == prevFrameId + 1);

    double dt = pkt.timestamp - prevTimestamp;
    if(dt <= 0.0 && pkt.imuDelta != nullptr){
        dt = pkt.imuDelta->deltaTij();
    }
    if(dt <= 0.0){
        dt = 1e-3;
    }

    // 1. Predict initial state guess and attach motion constraint
    gtsam::Pose3 predPose;
    gtsam::Vector3 predVel;

    if(isContiguousFrame && pkt.imuDelta != nullptr && pkt.imuDelta->deltaTij() > 0.0){
        const gtsam::NavState predState = pkt.imuDelta->predict(prevNavState, prevBias);
        predPose = predState.pose();
        predVel = predState.velocity();

        stagingGraph.emplace_shared<gtsam::ImuFactor>(
            X(prevFrameId), V(prevFrameId),
            X(frameId), V(frameId),
            B(prevFrameId),
            *pkt.imuDelta
        );
    }else{
        // Fallback if an intermediate packet was evicted from hfQueue or IMU delta is absent
        predPose = gtsam::Pose3(
            prevNavState.attitude(),
            prevNavState.position() + prevNavState.velocity() * dt
        );
        predVel = prevNavState.velocity();

        const gtsam::Pose3 relPose = prevNavState.pose().between(predPose);
        auto fallbackPoseNoise = gtsam::noiseModel::Diagonal::Sigmas(
            (gtsam::Vector(6) << 0.1 * dt, 0.1 * dt, 0.1 * dt,
                                 0.5 * dt, 0.5 * dt, 0.5 * dt).finished()
        );
        auto fallbackVelNoise = gtsam::noiseModel::Isotropic::Sigma(3, 0.5 * dt);

        stagingGraph.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(
            X(prevFrameId), X(frameId), relPose, fallbackPoseNoise);
        stagingGraph.emplace_shared<gtsam::BetweenFactor<gtsam::Vector3>>(
            V(prevFrameId), V(frameId), gtsam::Vector3::Zero(), fallbackVelNoise);
    }

    stagingValues.insert(X(frameId), predPose);
    stagingValues.insert(V(frameId), predVel);
    stagingValues.insert(B(frameId), prevBias);

    // 2. IMU bias random-walk constraint: sigma_discrete = sigma_continuous * sqrt(dt)
    const double sqrtDt = std::sqrt(dt);
    gtsam::Vector6 biasRwSigmas;
    biasRwSigmas << config.accelBiasRwSigma * sqrtDt,
                    config.accelBiasRwSigma * sqrtDt,
                    config.accelBiasRwSigma * sqrtDt,
                    config.gyroBiasRwSigma * sqrtDt,
                    config.gyroBiasRwSigma * sqrtDt,
                    config.gyroBiasRwSigma * sqrtDt;

    auto biasRwNoise = gtsam::noiseModel::Diagonal::Sigmas(biasRwSigmas);
    stagingGraph.emplace_shared<gtsam::BetweenFactor<gtsam::imuBias::ConstantBias>>(
        B(prevFrameId),
        B(frameId),
        gtsam::imuBias::ConstantBias(),
        biasRwNoise
    );

    // 3. Optional 1D pressure-transducer depth factor
    if(pkt.depth.has_value()){
        stagingGraph.emplace_shared<DepthFactor>(
            X(frameId), *pkt.depth, depthNoise, config.bodyToDepthSensor);
    }

    // 4. Visual observation buffering, triangulation, and projection factors
    bk.startFrame(frameId);
    for(size_t i = 0; i < pkt.visualMeasurements.size(); i++){
        const auto &meas = pkt.visualMeasurements[i];
        bk.updateBookeeper(meas.featureId, meas.pt);
    }
    bk.pruneDeadTracks();

    prevFrameId = frameId;
    prevTimestamp = pkt.timestamp;
}

void Backend::processLfPacket(const BackendLfPacket &pkt){
    const gtsam::Key matchedKey = X(pkt.matchedFrameId);
    const gtsam::Key currentKey = X(pkt.currentFrameId);

    const bool hasMatched = isamEstimate.exists(matchedKey) || stagingValues.exists(matchedKey);
    const bool hasCurrent = isamEstimate.exists(currentKey) || stagingValues.exists(currentKey);
    if(!hasMatched || !hasCurrent){
        return;
    }

    gtsam::SharedNoiseModel loopNoise = pkt.noiseModel;
    if(!loopNoise){
        auto baseLoop = gtsam::noiseModel::Diagonal::Sigmas(
            (gtsam::Vector(6) << 0.02, 0.02, 0.02, 0.05, 0.05, 0.05).finished());
        loopNoise = gtsam::noiseModel::Robust::Create(
            gtsam::noiseModel::mEstimator::Huber::Create(1.345),
            baseLoop
        );
    }

    // LfTracker solves P_cam_curr = R * P_cam_matched + t (i.e., T_cam_curr_from_matched).
    // BetweenFactor<Pose3>(X_matched, X_curr, T_between) requires T_body_matched_from_curr:
    // T_body_matched_from_curr = T_BC * (pkt.relativePose)^{-1} * T_BC^{-1}
    const gtsam::Pose3 camMatchedFromCurr = pkt.relativePose.inverse();
    const gtsam::Pose3 bodyMatchedFromCurr =
        config.bodyToCameraFrame.compose(camMatchedFromCurr)
            .compose(config.bodyToCameraFrame.inverse());

    stagingGraph.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(
        matchedKey,
        currentKey,
        bodyMatchedFromCurr,
        loopNoise
    );
}

void Backend::recordKeyframeFeatures(const BackendHfPacket &pkt){
    PendingKfRecord rec;
    rec.frameId = pkt.frameId;
    rec.featureIds.reserve(pkt.visualMeasurements.size());
    for(size_t i = 0; i < pkt.visualMeasurements.size(); i++){
        rec.featureIds.push_back(pkt.visualMeasurements[i].featureId);
    }
    rec.lastLandmarkCount = 0;
    trackedKeyframes.push_back(std::move(rec));
}

void Backend::syncKeyframesToLfTracker(bool forceAll){
    for(size_t i = 0; i < trackedKeyframes.size(); i++){
        auto &rec = trackedKeyframes[i];
        const gtsam::Key poseKey = X(rec.frameId);
        if(!isamEstimate.exists(poseKey)){
            continue;
        }

        // Count how many of this keyframe's features are now triangulated in isamEstimate
        size_t validCount = 0;
        for(size_t j = 0; j < rec.featureIds.size(); j++){
            if(isamEstimate.exists(L(rec.featureIds[j]))){
                validCount++;
            }
        }

        if(validCount == 0 || (!forceAll && validCount == rec.lastLandmarkCount)){
            continue;
        }

        const gtsam::Pose3 bodyPose = isamEstimate.at<gtsam::Pose3>(poseKey);
        const gtsam::Pose3 camPose = bodyPose.compose(config.bodyToCameraFrame);

        BackendKeyframeUpdate update;
        update.id = rec.frameId;
        update.pose = camPose;
        update.landmarks.reserve(validCount);

        for(size_t j = 0; j < rec.featureIds.size(); j++){
            const uint32_t featId = rec.featureIds[j];
            const gtsam::Key lmKey = L(featId);
            if(isamEstimate.exists(lmKey)){
                const gtsam::Point3 worldPt = isamEstimate.at<gtsam::Point3>(lmKey);
                // Transform 3D landmark into the keyframe's local camera optical frame
                const gtsam::Point3 camPt = camPose.transformTo(worldPt);
                if(camPt.z() > 0.0){
                    Landmark3d lm;
                    lm.pt3d = camPt;
                    lm.featureId = featId;
                    update.landmarks.push_back(lm);
                }
            }
        }

        if(!update.landmarks.empty()){
            lf.updateKeyframe(update);
            rec.lastLandmarkCount = validCount;
        }
    }
}

void Backend::optimizeAndPublish(uint32_t frameId, double timestamp, bool loopClosed){
    try{
        isam.update(stagingGraph, stagingValues);
        isamEstimate = isam.calculateEstimate();
    }catch(const gtsam::IndeterminantLinearSystemException &){
        stagingGraph.resize(0);
        stagingValues.clear();
        return;
    }

    stagingGraph.resize(0);
    stagingValues.clear();

    if(!isamEstimate.exists(X(frameId))){
        return;
    }

    const gtsam::Pose3 optPose = isamEstimate.at<gtsam::Pose3>(X(frameId));
    const gtsam::Vector3 optVel = isamEstimate.at<gtsam::Vector3>(V(frameId));
    const gtsam::imuBias::ConstantBias optBias =
        isamEstimate.at<gtsam::imuBias::ConstantBias>(B(frameId));

    prevNavState = gtsam::NavState(optPose, optVel);
    prevBias = optBias;

    // Feed updated IMU bias back to the frontend pre-integrator
    fe.setBias(prevBias);

    // Push newly triangulated or loop-refined 3D landmarks to LfTracker keyframes
    syncKeyframesToLfTracker(loopClosed);

    // Write new state into the inactive double-buffer slot and publish atomically
    const uint8_t nextIdx = static_cast<uint8_t>(1u - activePose.load(std::memory_order_relaxed));
    Pose &outPose = poseBuffer[nextIdx];
    outPose.timestamp = timestamp;
    outPose.t = optPose.translation();
    outPose.r = optPose.rotation().toQuaternion();
    outPose.v = optVel;

    activePose.store(nextIdx, std::memory_order_release);
}

} // namespace PoolSLAM