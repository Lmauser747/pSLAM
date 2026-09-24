#include <atomic>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <cstring>

#include "HAL.hpp"
#include "HALTypes.hpp"

namespace PoolSLAM{

bool ImageStream::pushFrame(uint8_t *rawPixels, double timestamp){
    PixelArray *copyTarget;
    {
        std::lock_guard<std::mutex> lock(m_imPool);
        if(poolFreeList.empty()){
            // If there are avaible frames evict them from readyQueue right now
            if(!readyQueue.empty()){
                copyTarget = readyQueue.front();
                readyQueue.pop();
            }else{
                return false;
            }
        }else{
            // Checkout frame buffer
            copyTarget = poolFreeList.front();
            poolFreeList.pop();
        }
    }

    // Copy to frame buffer
    std::memcpy(copyTarget->arr.data(), rawPixels, imW * imH);
    copyTarget->timestamp = timestamp;

    {
        std::lock_guard<std::mutex> lock(m_imPool);
        // Evict frames from readyQueue if it is too big
        if(readyQueue.size() >= 3){
            poolFreeList.push(readyQueue.front());
            readyQueue.pop();
        }
        // Add the copied data to the ready queue
        readyQueue.push(copyTarget);
    }

    c_imPool.notify_one();
    return true;
}

SensorHealth ImageStream::waitForFrame(PixelArray *&target, std::chrono::milliseconds timeout){
    if(health.load(std::memory_order_relaxed) == SensorHealth::DISABLED){
        return SensorHealth::DISABLED;
    }

    std::unique_lock<std::mutex> lock(m_imPool);
    bool valid = c_imPool.wait_for(lock, timeout, 
    [&](){return !readyQueue.empty();});

    // Update the sensor health if a frame is missed
    if(!valid){
        target = nullptr;
        if(health.load(std::memory_order_relaxed) == SensorHealth::ACTIVE){
            health.store(SensorHealth::DEGRADED, std::memory_order_relaxed);
        }
        missedFrames++;
        // If too many frames are missed, disable the sensor
        if(missedFrames > MAX_MISSED_FRAMES){
            health.store(SensorHealth::DISABLED, std::memory_order_relaxed);
            return SensorHealth::DISABLED;
        }
        return SensorHealth::DEGRADED;
    }

    target = readyQueue.front();
    readyQueue.pop();
    missedFrames = 0;
    health.store(SensorHealth::ACTIVE);
    return SensorHealth::ACTIVE;
}

void ImageStream::recycleFrame(PixelArray *frame){
    if(!frame){
        return;
    }

    std::lock_guard<std::mutex> lock(m_imPool);
    poolFreeList.push(frame);
}

SensorHealth ImageStream::getHealth(){
    return health.load(std::memory_order_relaxed);
}

}