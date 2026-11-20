#include <atomic>
#include <cassert>
#include <cstddef>
#include <mutex>
#include <optional>
#include <memory>
#include <utility>
#include <thread>

#include <gtsam/navigation/ImuBias.h>
#include <gtsam/navigation/ImuFactor.h>

#include "frontend/Frontend.hpp"
#include "CoreTypes.hpp"
#include "frontend/LfTracker.hpp"
#include "hal/Synchronizer.hpp"
#include "pool_slam/SystemTypes.hpp"

namespace PoolSLAM{

void Frontend::processPacket(){
    size_t lastKfFeatures = 0;
    double lastKfTime = -1.0;

    while(runFrontend.load(std::memory_order_relaxed)){
        // Get packet from sync
        SyncedPacket packet;
        if(!sync.getNextPacket(packet)){
            if(sync.getStatus() != SyncStatus::RUNNING){
                {
                    std::lock_guard<std::mutex> lock(m_hfQueue);
                    runFrontend.store(false, std::memory_order_release);
                }
                c_hfQueue.notify_all();
                break;
            }
            continue;
        }

        std::unique_ptr<gtsam::PreintegratedImuMeasurements> integrator;

        // Track existing features and integrate IMU on frames after T0
        if(!packet.imu.empty()){
            {
                std::lock_guard<std::mutex> lock(m_imuBias);
                auto params = config.preintParams;
                if(!params){
                    params = gtsam::PreintegrationParams::MakeSharedU(9.81);
                    params->accelerometerCovariance = Eigen::Matrix3d::Identity() * 1e-3;
                    params->gyroscopeCovariance = Eigen::Matrix3d::Identity() * 1e-4;
                    params->integrationCovariance = Eigen::Matrix3d::Identity() * 1e-5;
                }
                integrator = std::make_unique<gtsam::PreintegratedImuMeasurements>(params, imuBias);
            }

            auto lastImu = packet.imu[0];
            for(size_t i = 1; i < packet.imu.size(); i++){
                auto &curImu = packet.imu[i];
                if(curImu.timestamp <= lastImu.timestamp){
                    continue;
                }
                integrator->integrateMeasurement(lastImu.a, lastImu.w, curImu.timestamp - lastImu.timestamp);
                lastImu = curImu;
            }

            const Eigen::Matrix3f R_body = integrator->deltaRij().matrix().cast<float>();
            const Eigen::Matrix3f R_cam = config.bodyToCameraRot.transpose() * R_body * config.bodyToCameraRot;
            hf.trackExistingFeatures(packet.image->arr.data(), R_cam);
        }

        auto &features = hf.getActiveFeatures();

        // On Frame 0, replenish first so X_0 has initial landmarks to anchor the trajectory
        if(frameId == 0){
            hf.replenishBuckets(packet.image->arr.data());
        }

        // Evaluate keyframe criteria on surviving tracks (or initial features on Frame 0)
        bool isKeyframe = false;
        if(frameId == 0 ||
           (packet.timestamp - lastKfTime) > config.maxTimeBetweenKf ||
           static_cast<float>(features.size()) < config.droppedFeatureThreshold * static_cast<float>(lastKfFeatures)){
            isKeyframe = true;
            lastKfTime = packet.timestamp;
            lastKfFeatures = features.size();
        }

        // Assemble BackendHfPacket BEFORE replenishing on frames > 0 so only surviving
        // RANSAC inliers are sent to the Backend
        std::unique_ptr<BackendHfPacket> hfPacket = std::make_unique<BackendHfPacket>();
        hfPacket->timestamp = packet.timestamp;
        hfPacket->frameId = frameId;
        hfPacket->isKeyframe = isKeyframe;
        hfPacket->imuDelta = std::move(integrator);

        if(packet.depth.has_value()){
            hfPacket->depth = packet.depth->z;
        }else{
            hfPacket->depth = std::nullopt;
        }

        if(packet.initialAccel.has_value()){
            hfPacket->initialAccel = *packet.initialAccel;
        }else if(!packet.imu.empty()){
            Eigen::Vector3d sumA = Eigen::Vector3d::Zero();
            for(const auto &s : packet.imu){
                sumA += s.a;
            }
            hfPacket->initialAccel = sumA / static_cast<double>(packet.imu.size());
        }

        if(packet.initialGyro.has_value()){
            hfPacket->initialGyro = *packet.initialGyro;
        }else if(!packet.imu.empty()){
            Eigen::Vector3d sumW = Eigen::Vector3d::Zero();
            for(const auto &s : packet.imu){
                sumW += s.w;
            }
            hfPacket->initialGyro = sumW / static_cast<double>(packet.imu.size());
        }

        hfPacket->visualMeasurements.reserve(features.size());
        for(const auto &feat : features){
            hfPacket->visualMeasurements.emplace_back(feat.id, feat.ptUndis.cast<double>());
        }

        // Pass keyframe to LfTracker using the surviving features that match BackendHfPacket
        if(isKeyframe){
            lf.passKeyframe(frameId, packet.timestamp, packet.image->arr.data(), features);
        }

        // Replenish empty buckets for subsequent frames after packing BackendHfPacket and passing keyframe
        if(frameId > 0){
            hf.replenishBuckets(packet.image->arr.data());
        }

        // Recycle the raw image buffer now that HfTracker and LfTracker are done with it
        sync.recycleFrame(packet.image);

        // Push the packet onto the queue
        {
            std::lock_guard<std::mutex> lock(m_hfQueue);
            if(config.hfQueueSize > 0 && hfQueue.size() >= config.hfQueueSize){
                hfQueue.pop();
            }
            hfQueue.push(std::move(hfPacket));
        }
        c_hfQueue.notify_all();

        frameId++;
    }
}
    
std::unique_ptr<BackendHfPacket> Frontend::getHfPacket(){
    {
        std::unique_lock<std::mutex> lock(m_hfQueue);

        c_hfQueue.wait(lock, [&](){
            return hfQueue.size() > 0 || !runFrontend.load(std::memory_order_relaxed);
        });

        if(hfQueue.empty()){
            return nullptr;
        }

        auto retPtr = std::move(hfQueue.front());
        hfQueue.pop();
        return retPtr;
    }
}

LfTracker &Frontend::getLfTracker(){
    return lf;
}

void Frontend::start(){
    if(runFrontend.load(std::memory_order_acquire)){
        return;
    }
    if(frontendThread.joinable()){
        frontendThread.join();
    }
    {
        std::lock_guard<std::mutex> lock(m_hfQueue);
        while(!hfQueue.empty()){
            hfQueue.pop();
        }
        frameId = 0;
        runFrontend.store(true, std::memory_order_release);
    }
    lf.start();
    frontendThread = std::thread(&Frontend::processPacket, this);
}

bool Frontend::setBias(const gtsam::imuBias::ConstantBias &bias){
    {
        std::lock_guard<std::mutex> lock(m_imuBias);
        imuBias = bias;
    }
    return true;
}

void Frontend::end(){
    {
        std::lock_guard<std::mutex> lock(m_hfQueue);
        runFrontend.store(false, std::memory_order_release);
    }
    c_hfQueue.notify_all();
    if(frontendThread.joinable()){
        frontendThread.join();
    }
    lf.end();
}

}