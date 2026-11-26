#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <memory>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/LossFunctions.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/linear/linearExceptions.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/slam/BetweenFactor.h>

#include "backend/Backend.hpp"
#include "frontend/LfTracker.hpp"
#include "pool_slam/SystemTypes.hpp"

namespace PoolSLAM{

using gtsam::symbol_shorthand::X;

bool Backend::processLfPacket(const BackendLfPacket &lfPkt){
    // make sure all of the keys exist
    bool curKeyValid = globalPoseValues.exists(X(lfPkt.currentFrameId)); 
    bool matchKeyValid = globalPoseValues.exists(X(lfPkt.matchedFrameId));

    if(!curKeyValid || !matchKeyValid){
        return false;
    }

    // Convert camera loop delta into a body frame factor from match to current
    auto betweenPose = config.bodyToCameraFrame * lfPkt.relativePose.inverse() * config.bodyToCameraFrame.inverse();

    const gtsam::Pose3 poseMatch = globalPoseValues.at<gtsam::Pose3>(X(lfPkt.matchedFrameId));
    const gtsam::Pose3 poseCur = globalPoseValues.at<gtsam::Pose3>(X(lfPkt.currentFrameId));
    const gtsam::Pose3 residual = poseMatch.between(poseCur).between(betweenPose);
    const double transErr = residual.translation().norm();
    const double rotErr = gtsam::Rot3::Logmap(residual.rotation()).norm();
    if(transErr > config.loopMaxTransResidual || rotErr > config.loopMaxRotResidual){
        return false;
    }

    const double minRotVar = config.loopMinRotSigma * config.loopMinRotSigma;
    const double minTransVar = config.loopMinTransSigma * config.loopMinTransSigma;
    auto cauchyKernel = gtsam::noiseModel::mEstimator::Cauchy::Create(config.loopCauchyK);

    gtsam::SharedNoiseModel loopNoise;
    if(lfPkt.noiseModel != nullptr){
        loopNoise = lfPkt.noiseModel;
        auto robust = std::dynamic_pointer_cast<gtsam::noiseModel::Robust>(lfPkt.noiseModel);
        auto baseModel = robust ? robust->noise() : lfPkt.noiseModel;
        auto gaussian = std::dynamic_pointer_cast<gtsam::noiseModel::Gaussian>(baseModel);
        if(gaussian && gaussian->dim() == 6){
            // Map covariance from camera frame into body frame and inflate for landmark depth error
            const gtsam::Matrix6 Ad = config.bodyToCameraFrame.AdjointMap();
            gtsam::Matrix6 covBody = (Ad * gaussian->covariance() * Ad.transpose()) * config.loopCovScale;
            covBody = 0.5 * (covBody + covBody.transpose());
            for(int i = 0; i < 3; i++){
                covBody(i, i) = std::max(covBody(i, i), minRotVar);
                covBody(i + 3, i + 3) = std::max(covBody(i + 3, i + 3), minTransVar);
            }
            auto bodyGaussian = gtsam::noiseModel::Gaussian::Covariance(covBody);
            loopNoise = gtsam::noiseModel::Robust::Create(cauchyKernel, bodyGaussian);
        }
    }else{
        // Fallback if noiseModel is null for some reason
        gtsam::Vector6 sigmas;
        sigmas << std::max(0.1, config.loopMinRotSigma),
                  std::max(0.1, config.loopMinRotSigma),
                  std::max(0.1, config.loopMinRotSigma),
                  std::max(0.1, config.loopMinTransSigma),
                  std::max(0.1, config.loopMinTransSigma),
                  std::max(0.1, config.loopMinTransSigma);
        auto baseGaussian = gtsam::noiseModel::Diagonal::Sigmas(sigmas);
        loopNoise = gtsam::noiseModel::Robust::Create(cauchyKernel, baseGaussian);
    }
    globalPoseGraph.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(X(lfPkt.matchedFrameId), X(lfPkt.currentFrameId), betweenPose, loopNoise);
    hasClosedLoop = true;
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

    if(hasClosedLoop &&
       globalPoseValues.exists(X(currentFrameId)) &&
       smootherEstimate.exists(X(currentFrameId))){
        const gtsam::Pose3 globalPose = globalPoseValues.at<gtsam::Pose3>(X(currentFrameId));
        const gtsam::Pose3 vioPose = smootherEstimate.at<gtsam::Pose3>(X(currentFrameId));
        vioToGlobalDrift = globalPose * vioPose.inverse();
    }

    return true;
}

void Backend::syncLfTracker(bool forceAll){
    for(auto &kf : trackedKeyframes){
        BackendKeyframeUpdate update;
        update.id = kf.frameId;
        if(!smootherEstimate.exists(X(kf.frameId))){
            continue;
        }
        update.pose = smootherEstimate.at<gtsam::Pose3>(X(kf.frameId)) * config.bodyToCameraFrame;
        update.landmarks.reserve(kf.lastLandmarkCount);
        size_t count = 0;
        for(auto featId : kf.featureIds){
            auto sfIt = smartFactors.find(featId);
            if(sfIt == smartFactors.end() || sfIt->second == nullptr){
                continue;
            }

            // Skip landmarks whose older poses were already marginalized out of smootherEstimate
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

}
