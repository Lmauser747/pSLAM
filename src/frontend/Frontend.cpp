#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <utility>

#include <gtsam/navigation/ImuBias.h>
#include <gtsam/navigation/ImuFactor.h>

#include "frontend/Frontend.hpp"
#include "CoreTypes.hpp"
#include "frontend/LfTracker.hpp"
#include "hal/Synchronizer.hpp"
#include "pool_slam/SystemTypes.hpp"

namespace PoolSLAM{

void Frontend::processPacket(){
    double lastKfTimestamp = 0.0;
    size_t lastKfCount = UINT64_MAX;
    while(runFrontend.load(std::memory_order_relaxed)){
        SyncedPacket syncPkt;
        if(!sync.getNextPacket(syncPkt)){
            if(sync.getStatus() != SyncStatus::RUNNING){
                c_hfQueue.notify_all();
                break;
            }else{
                continue;
            }
        }

        std::unique_ptr<gtsam::PreintegratedImuMeasurements> imuDelta = std::make_unique<gtsam::PreintegratedImuMeasurements>(config.preintParams, imuBias);
        if(!syncPkt.imu.empty()){
            auto tPrev = syncPkt.imu.front().timestamp;
            for(size_t i = 0; i < syncPkt.imu.size(); i++){
                auto tCurr = syncPkt.imu[i].timestamp;
                auto dt = tCurr - tPrev;
                if(dt <= 0.0){
                    continue;
                }
                imuDelta->integrateMeasurement(syncPkt.imu[i].a, syncPkt.imu[i].w, dt);
                tPrev = tCurr;
            }
        }
            

        std::unique_ptr<BackendHfPacket> hfPkt = std::make_unique<BackendHfPacket>();

        // Get the IMU estimate of the rotation between frames
        auto Rcam = config.bodyToCameraRot.transpose() * imuDelta->deltaRij().matrix() * config.bodyToCameraRot; // Frame to frame relative camera rotation
        
        // Track the existing features using IC-LK
        auto meanDisplacement = hf.trackExistingFeatures(syncPkt.image->arr.data(), Rcam.cast<float>()); 
        
        auto featureCount = hf.getActiveFeatures().size();

        // Determine if keyframe
        auto percentRetainedFeatures = static_cast<float>(featureCount) / static_cast<float>(lastKfCount);
        if(frameId == 0 || 
            meanDisplacement > config.meanDisplacementThreshold || 
            percentRetainedFeatures < config.droppedFeatureThreshold ||
            (syncPkt.timestamp - lastKfTimestamp > config.maxTimeBetweenKf)
        ){
            hfPkt->isKeyframe = true;

            //Replenish buckets before adding features
            hf.replenishBuckets(syncPkt.image->arr.data());
            lf.passKeyframe(frameId, syncPkt.timestamp, syncPkt.image->arr.data(), hf.getActiveFeatures());

            lastKfCount = hf.getActiveFeatures().size();
            lastKfTimestamp = syncPkt.timestamp;
        }

        // Fill packet
        hfPkt->imuDelta = std::move(imuDelta);
        if(syncPkt.depth.has_value()){
            hfPkt->depth = syncPkt.depth->z;
        }
        hfPkt->timestamp = syncPkt.timestamp;
        hfPkt->frameId = frameId;

        auto &features = hf.getActiveFeatures();

        size_t count = 0;
        for(auto &feat : features){
            hfPkt->visualFeatures.emplace_back(feat.id, feat.ptUndis.cast<double>(), count >= featureCount);
            count++;
        }

        if(frameId == 0){
            hfPkt->initialAccel = syncPkt.initialAccel;
            hfPkt->initialGyro = syncPkt.initialGyro;
        }

        frameId++;
        
        // Recycle frame back to sync
        sync.recycleFrame(syncPkt.image);

        // Update queue
        {
            std::lock_guard<std::mutex> lock(m_hfQueue);
            hfQueue.push(std::move(hfPkt));
        }

        c_hfQueue.notify_all();
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