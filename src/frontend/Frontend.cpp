#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <memory>
#include <utility>

#include <gtsam/navigation/ImuFactor.h>

#include "frontend/Frontend.hpp"
#include "CoreTypes.hpp"
#include "hal/Synchronizer.hpp"

namespace PoolSLAM{

void Frontend::processPacket(){
    size_t lastNumFeatures = UINT64_MAX;
    double lastKfTime = -1.1 * config.maxTimeBetweenKf;
    while(runFrontend.load(std::memory_order_relaxed)){
        // Get packet from sync
        sync.getNextPacket(*syncPacket);
        auto packet = *syncPacket;

        auto integrator = std::make_unique<gtsam::PreintegratedImuMeasurements>(config.preintParams, imuBias);

        // Handle first frame
        if(lastKfTime >= 0.0){
            // Integrate IMU measurements
            auto &lastImu = packet.imu.front();
            for(size_t i = 1; i < packet.imu.size(); i++){
                auto &curImu = packet.imu[i];
                integrator->integrateMeasurement(lastImu.a, lastImu.w, curImu.timestamp - lastImu.timestamp);
                lastImu = curImu;
            }

            auto R = integrator->deltaRij();

            hf.trackExistingFeatures(packet.image->arr.data(), R.matrix());
        }

        auto &features = hf.getActiveFeatures();
        
        std::unique_ptr<BackendHfPacket> hfPacket = std::make_unique<BackendHfPacket>();

        hfPacket->timestamp = packet.timestamp;
        hfPacket->frameId = frameId;
        frameId++;

        hfPacket->imuDelta = std::move(integrator);
        
        if(packet.depth.has_value()){
            hfPacket->depth = packet.depth->z;
        }else{
            hfPacket->depth = std::nullopt;
        }
        
        hfPacket->visualMeasurements.reserve(features.size());

        for(auto &feat : features){
            hfPacket->visualMeasurements.emplace_back(feat.id, feat.ptUndis);
        }

        bool isKeyframe = false;
        if((packet.timestamp - lastKfTime) > config.maxTimeBetweenKf || static_cast<float>(features.size()) / static_cast<float>(lastNumFeatures) < config.droppedFeatureThreshold){
            isKeyframe = true;
            lastKfTime = packet.timestamp;
        }

        hfPacket->isKeyframe = isKeyframe;

        {
            std::lock_guard<std::mutex> lock(m_hfQueue);
            if(hfQueue.size() >= config.hfQueueSize){
                hfQueue.pop();
            }
            hfQueue.push(std::move(hfPacket));
        }
        c_hfQueue.notify_all();

        lastNumFeatures = features.size();

        if(isKeyframe){
            lf.passKeyframe(hfPacket->frameId, packet.timestamp, packet.image->arr.data(), features);
        }

        hf.replenishBuckets(packet.image->data());

        sync.recycleFrame(packet.image);

        syncPacket.reset();
    }
}
    
std::unique_ptr<BackendHfPacket> Frontend::getHfPacket(){
    {
        std::unique_lock<std::mutex> lock(m_hfQueue);

        c_hfQueue.wait(lock, [&](){
            return hfQueue.size() > 0;
        });

        auto retPtr = std::move(hfQueue.front());
        hfQueue.pop();
        return retPtr;
    }
}

}