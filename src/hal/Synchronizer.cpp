#include <atomic>
#include <mutex>
#include <optional>
#include <thread>
#include <chrono>

#include "hal/Synchronizer.hpp"
#include "pool_slam/HalTypes.hpp"
#include "pool_slam/SystemTypes.hpp"


namespace PoolSLAM{

void Synchronizer::threadTask(){
    while(isRunning.load(std::memory_order_relaxed)){
        PixelArray *nextFrame = nullptr;
        SensorHealth camHealth = imageStream.waitForFrame(nextFrame);

        // Check if stopped mid loop
        if(!isRunning.load(std::memory_order_relaxed)){
            if(nextFrame){
                imageStream.recycleFrame(nextFrame);
            }
            break;
        }

        // Deal with degraded/disabled camera
        if(camHealth == SensorHealth::DISABLED){
            syncFatalShutdown(SyncStatus::FATALCAM, nextFrame);
            break;
        }else if(camHealth == SensorHealth::DEGRADED || nextFrame == nullptr){
            imageStream.recycleFrame(nextFrame);
            continue;
        }

        // Handle first frame
        if(previousFrameTime < 0.0){
            SyncedPacket initPacket;
            initPacket.timestamp = nextFrame->timestamp;
            initPacket.image = nextFrame;
            
            SensorHealth depthHealth = depthStream.waitUntil(initPacket.timestamp);
            if (depthHealth == SensorHealth::ACTIVE) {
                initPacket.depth = depthStream.getInterDepth(initPacket.timestamp);
            } else {
                initPacket.depth = std::nullopt;
            }

            SensorHealth imuHealth = imuStream.waitUntil(initPacket.timestamp);
            if (imuHealth != SensorHealth::DISABLED) {
                initPacket.initialAccel = imuStream.getAverageAcceleration(initPacket.timestamp);
                initPacket.initialGyro = imuStream.getAverageAngularVelocity(initPacket.timestamp);
            }

            {
                {
                    std::lock_guard<std::mutex> lock(m_outputQueue);

                    if(outputQueue.size() >= maxOutputQueueSize){
                        SyncedPacket &stale = outputQueue.front();
                        if(stale.image){
                            imageStream.recycleFrame(stale.image);
                        }
                        outputQueue.pop();
                    }

                    outputQueue.push(initPacket);
                    previousFrameTime = initPacket.timestamp;
                }
            }
            c_outputQueue.notify_one();
            continue;
        }

        SyncedPacket packet;
        packet.image = nextFrame;
        packet.timestamp = nextFrame->timestamp;

        SensorHealth imuHealth = imuStream.waitUntil(packet.timestamp);


        // Deal with degreaded or disabled IMU
        if(imuHealth == SensorHealth::DISABLED){
            syncFatalShutdown(SyncStatus::FATALIMU, nextFrame);
            break;
        }else if(imuHealth == SensorHealth::DEGRADED
                || !imuStream.extractRange(previousFrameTime, nextFrame->timestamp, packet.imu)){
            imageStream.recycleFrame(nextFrame);
            continue;
        }

        SensorHealth depthHealth = depthStream.waitUntil(packet.timestamp);


        // Deal with degraded or disabled depth sensor
        if(depthHealth == SensorHealth::ACTIVE){
            packet.depth = depthStream.getInterDepth(packet.timestamp);
        }else{
            packet.depth = std::nullopt;
        }

        {
            std::lock_guard<std::mutex> lock(m_outputQueue);
            if(outputQueue.size() >= maxOutputQueueSize){
                SyncedPacket &stale = outputQueue.front();
                if(stale.image){
                    imageStream.recycleFrame(stale.image);
                }
                outputQueue.pop();
            }
            outputQueue.push(packet);
            imuStream.pruneUntil(previousFrameTime);
            depthStream.pruneUntil(previousFrameTime);
            previousFrameTime = packet.timestamp;
        }

        c_outputQueue.notify_one();
    }
}

void Synchronizer::syncFatalShutdown(SyncStatus reason, PixelArray *checkedOutFrame){
    syncStatus.store(reason, std::memory_order_relaxed);
    isRunning.store(false);

    if(checkedOutFrame){
        imageStream.recycleFrame(checkedOutFrame);
    }

    // Drain the queue
    {
        std::lock_guard<std::mutex> lock(m_outputQueue);
        while(!outputQueue.empty()){
            SyncedPacket &packet = outputQueue.front();
            outputQueue.pop();
            imageStream.recycleFrame(packet.image);
        }
    }

    // Notify all of the consumers downstream 
    c_outputQueue.notify_all();
}

void Synchronizer::start(){
    SyncStatus status = syncStatus.load(std::memory_order_relaxed);
    if(syncThread.joinable() || (status != SyncStatus::IDLE && status != SyncStatus::STOPPED)){
        return;
    }

    // Reset previous frame time to allow for clean restarts
    previousFrameTime = -1.0;

    isRunning.store(true, std::memory_order_relaxed);
    syncThread = std::thread(&Synchronizer::threadTask, this);
    syncStatus.store(SyncStatus::RUNNING);
}

void Synchronizer::stop(){
    if(!syncThread.joinable()){
        return;
    }
    isRunning.store(false, std::memory_order_relaxed);
    c_outputQueue.notify_all();
    syncThread.join();
    syncStatus.store(SyncStatus::STOPPED);
}

bool Synchronizer::getNextPacket(SyncedPacket &target, std::chrono::milliseconds timeout){
    std::unique_lock<std::mutex> lock(m_outputQueue);

    bool valid = c_outputQueue.wait_for(lock, timeout,
    [&](){return !outputQueue.empty() || !isRunning.load(std::memory_order_relaxed);});
    
    if(!valid || outputQueue.empty()){
        return false;
    }

    target = outputQueue.front();
    outputQueue.pop();
    return true;
}

}