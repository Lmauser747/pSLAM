#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <gtsam/geometry/Cal3.h>
#include <gtsam/inference/Factor.h>
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
    stagingValues.clear();
    stagingGraph.resize(0);
    
    double dt = pkt.timestamp - lastCommittedTimestamp;
    
    stagingValues.insert(X(pkt.frameId), prevNavState.pose());
    stagingValues.insert(V(pkt.frameId), prevNavState.velocity());
    stagingValues.insert(B(pkt.frameId), prevBias);

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

    stagingGraph.emplace_shared<gtsam::BetweenFactor<gtsam::imuBias::ConstantBias>>(B(prevFrameId), B(pkt.frameId), gtsam::imuBias::ConstantBias(), biasRwNoise);
    if(pkt.depth.has_value()){
         stagingGraph.emplace_shared<DepthFactor>(X(pkt.frameId), pkt.depth.value(), depthNoise, config.bodyToDepthSensor);
    }

    BackendLfPacket lfPkt;
    bool loopClosed = false;
    while(lf.getMatch(lfPkt)){
        processLfPacket(lfPkt);
        loopClosed = true;
    }

    gtsam::FactorIndices removeIndices;
    std::vector<uint32_t> stagedFeatureIds;
    
    for(auto &feat : pkt.visualFeatures){
        auto it = smartFactors.find(feat.featureId);

        if(feat.isNewTrack){
            auto factor = std::make_shared<SmartFactor>(visualNoise, Kptr, config.bodyToCameraFrame);
            factor->add(feat.pt, X(pkt.frameId));
            smartFactors[feat.featureId] = factor;
        }else{
            auto factor = it->second;
            factor->add(feat.pt, X(pkt.frameId));

            auto slotIt = smartFactorSlots.find(feat.featureId);

            if(slotIt != smartFactorSlots.end()){
                removeIndices.push_back(slotIt->second);
            }

            stagingGraph.push_back(factor);
            stagedFeatureIds.push_back(feat.featureId);
        }
    }

    if(solveGraph(pkt.frameId, removeIndices)){
        lastCommittedFrameId = pkt.frameId;
        lastCommittedTimestamp = pkt.timestamp;
        publishPose(prevNavState.pose(), prevNavState.velocity(), pkt.timestamp);
        syncLfTracker(pkt.frameId, loopClosed);
    }
}

} // namespace PoolSLAM