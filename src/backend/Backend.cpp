#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <thread>
#include <utility>

#include <gtsam/base/Vector.h>
#include <gtsam/geometry/Cal3_S2.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Quaternion.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/LossFunctions.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/navigation/ImuBias.h>
#include <gtsam/navigation/NavState.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/ISAM2Params.h>
#include <gtsam/nonlinear/PriorFactor.h>

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
      Kptr(std::make_shared<gtsam::Cal3_S2>(1.0, 1.0, 0.0, 0.0, 0.0)){ // Visual observations arrive normalized so Cal3_S2 uses identity intrinsics
    gtsam::ISAM2Params smootherParams;
    smootherParams.factorization = gtsam::ISAM2Params::QR;
    smootherParams.findUnusedFactorSlots = true;
    smootherParams.relinearizeThreshold = config.relinearizeThreshold;
    smootherParams.relinearizeSkip = config.relinearizeSkip;
    smoother = gtsam::ISAM2(smootherParams);

    smartParams.linearizationMode = gtsam::JACOBIAN_Q;
    // Keep degenerate tracks zeroed instead of throwing during stationary motion
    smartParams.degeneracyMode = gtsam::ZERO_ON_DEGENERACY;
    smartParams.setLandmarkDistanceThreshold(100.0);
    smartParams.setEnableEPI(true);
    smartParams.setDynamicOutlierRejectionThreshold(5.0 / config.K(0, 0));
    smartParams.setRankTolerance(config.smartRankTolerance);
    
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
    odomSigmas << config.odomRotSigma, config.odomRotSigma, config.odomRotSigma,
                  config.odomTransSigma, config.odomTransSigma, config.odomTransSigma;
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
    zuptVelNoise = gtsam::noiseModel::Isotropic::Sigma(3, config.zuptVelSigma);

    gtsam::Vector6 zuptPoseSigmas;
    zuptPoseSigmas << config.zuptPoseRotSigma, config.zuptPoseRotSigma, config.zuptPoseRotSigma,
                      config.zuptPoseTransSigma, config.zuptPoseTransSigma, config.zuptPoseTransSigma;
    zuptPoseNoise = gtsam::noiseModel::Diagonal::Sigmas(zuptPoseSigmas);
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
    gtsam::Vector3 initGyroBias = pkt.initialGyro.value_or(gtsam::Vector3::Zero());
    gtsam::Vector3 initAccelBias = gtsam::Vector3::Zero();
    if(config.initialPose.has_value() && pkt.initialAccel.has_value()){
        // Recover static accel bias from known attitude when startup motion is negligible
        const gtsam::Vector3 expectedSpecificForce =
            prevNavState.pose().rotation().unrotate(gtsam::Vector3(0.0, 0.0, config.gravity));
        const gtsam::Vector3 candidateBias = *pkt.initialAccel - expectedSpecificForce;
        if(std::abs(pkt.initialAccel->norm() - config.gravity) < config.initMaxAccelBiasNorm &&
           candidateBias.norm() < config.initMaxAccelBiasNorm){
            initAccelBias = candidateBias;
        }
    }
    prevBias = gtsam::imuBias::ConstantBias(initAccelBias, initGyroBias);

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
    auto result = solveSmoother(pkt.frameId);
    if(!result.has_value()){
        activeFeatureTracks.clear();
        return;
    }

    WindowFrame wf;
    wf.frameId = pkt.frameId;
    wf.timestamp = pkt.timestamp;
    wf.isKeyframe = true;
    for (const auto &f : pkt.visualFeatures) wf.observedFeatureIds.push_back(f.featureId);
    window.push_back(std::move(wf));

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
    
    hasClosedLoop = false;
    vioToGlobalDrift = gtsam::Pose3::Identity();

    isInitialized = true;

    lastCommittedFrameId = pkt.frameId;
    lastCommittedKeyframeId = pkt.frameId; 
    lastCommittedTimestamp = pkt.timestamp;
    lastCommittedNavState = prevNavState;
    lastCommittedKeyframeNavState = prevNavState;
    prevFrameId = pkt.frameId;
    prevTimestamp = pkt.timestamp;
}

void Backend::publishPose(const gtsam::Pose3 &pose, const gtsam::Vector3 &velocity, double timestamp){
    const gtsam::Pose3 correctedPose = vioToGlobalDrift * pose;
    const gtsam::Vector3 correctedVelocity = vioToGlobalDrift.rotation().rotate(velocity);

    uint8_t inactivePose = activePose.load(std::memory_order_relaxed) == 0 ? 1 : 0;
    poseBuffer[inactivePose].timestamp = timestamp;
    poseBuffer[inactivePose].t = correctedPose.translation();
    poseBuffer[inactivePose].r = correctedPose.rotation().toQuaternion();
    poseBuffer[inactivePose].v = correctedVelocity;

    activePose.store(inactivePose, std::memory_order_release);
}

}
