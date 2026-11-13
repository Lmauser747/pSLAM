#include <atomic>
#include <cstddef>
#include <cstdint>
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

namespace PoolSLAM{

void Frontend::processPacket(){
    size_t lastKfFeatures = UINT64_MAX;
    double lastKfTime = -1.0;
    while(runFrontend.load(std::memory_order_relaxed)){
        // Get packet from sync
        SyncedPacket packet;
        if(!sync.getNextPacket(packet)){
            continue;
        }

        std::unique_ptr<gtsam::PreintegratedImuMeasurements> integrator;

        {
            std::lock_guard<std::mutex> lock(m_imuBias);
            integrator = std::make_unique<gtsam::PreintegratedImuMeasurements>(config.preintParams, imuBias);
        }
            
        // Handle first frame
        if(!packet.imu.empty()){
            // Integrate IMU measurements
            auto lastImu = packet.imu.front();
            for(size_t i = 1; i < packet.imu.size(); i++){
                auto &curImu = packet.imu[i];
                integrator->integrateMeasurement(lastImu.a, lastImu.w, curImu.timestamp - lastImu.timestamp);
                lastImu = curImu;
            }

            auto R = integrator->deltaRij();

            hf.trackExistingFeatures(packet.image->arr.data(), R.matrix().cast<float>());
        }

        auto &features = hf.getActiveFeatures();
        
        std::unique_ptr<BackendHfPacket> hfPacket = std::make_unique<BackendHfPacket>();

        hfPacket->timestamp = packet.timestamp;
        hfPacket->frameId = frameId;

        hfPacket->imuDelta = std::move(integrator);
        
        if(packet.depth.has_value()){
            hfPacket->depth = packet.depth->z;
        }else{
            hfPacket->depth = std::nullopt;
        }
        
        hfPacket->visualMeasurements.reserve(features.size());

        for(auto &feat : features){
            hfPacket->visualMeasurements.emplace_back(feat.id, feat.ptUndis.cast<double>());
        }

        bool isKeyframe = false;
        if((packet.timestamp - lastKfTime) > config.maxTimeBetweenKf || static_cast<float>(features.size()) < config.droppedFeatureThreshold * static_cast<float>(lastKfFeatures)){
            isKeyframe = true;
            lastKfTime = packet.timestamp;
        }

        hfPacket->isKeyframe = isKeyframe;

        // Push the packet onto the queue
        {
            std::lock_guard<std::mutex> lock(m_hfQueue);
            if(hfQueue.size() >= config.hfQueueSize){
                hfQueue.pop();
            }
            hfQueue.push(std::move(hfPacket));
        }
        c_hfQueue.notify_all();

        hf.replenishBuckets(packet.image->arr.data());

        if(isKeyframe){
            lastKfFeatures = features.size();
            lf.passKeyframe(frameId, packet.timestamp, packet.image->arr.data(), features);
        }

        sync.recycleFrame(packet.image);

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
    if(runFrontend.load(std::memory_order_relaxed)){
        return;
    }
    frameId = 0;
    runFrontend.store(true, std::memory_order_release);
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
    if(!runFrontend.load(std::memory_order_relaxed)){
        return;
    }
    runFrontend.store(false, std::memory_order_release);
    c_hfQueue.notify_all();
    if(frontendThread.joinable()){
        frontendThread.join();
    }
    lf.end();
}
}