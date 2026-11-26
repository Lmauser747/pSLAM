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

void Backend::processLfPacket(const BackendLfPacket &lfPkt){
    // make sure all of the keys exist
    bool curKeyValid = globalPoseValues.exists(X(lfPkt.currentFrameId)); 
    bool matchKeyValid = globalPoseValues.exists(X(lfPkt.matchedFrameId));

    if(curKeyValid && matchKeyValid){
        // Convert camera loop delta into a body frame factor from match to current
        auto betweenPose = config.bodyToCameraFrame * lfPkt.relativePose.inverse() * config.bodyToCameraFrame.inverse();
        gtsam::SharedNoiseModel loopNoise;
        if(lfPkt.noiseModel != nullptr){
            loopNoise = lfPkt.noiseModel;
            auto robust = std::dynamic_pointer_cast<gtsam::noiseModel::Robust>(lfPkt.noiseModel);
            auto baseModel = robust ? robust->noise() : lfPkt.noiseModel;
            auto gaussian = std::dynamic_pointer_cast<gtsam::noiseModel::Gaussian>(baseModel);
            if(gaussian && gaussian->dim() == 6){
                // Map covariance from camera frame into body frame
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
