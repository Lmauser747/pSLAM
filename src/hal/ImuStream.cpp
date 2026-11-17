#include <algorithm>
#include <atomic>
#include <iterator>
#include <mutex>
#include <vector>
#include <cstddef>
#include <cmath>

#include <eigen3/Eigen/Core>

#include "pool_slam/Hal.hpp"
#include "pool_slam/HalTypes.hpp"

namespace PoolSLAM {

bool ImuStream::fastPush(const Imu &imu){
    // If incoming data is in order, just add it to the buffer
    if(imuBuffer.empty() || imu.timestamp >= imuBuffer.back().timestamp){
        imuBuffer.push_back(imu);
        return true;
    }

    // Don't add data older than the oldest data in the buffer
    if(imu.timestamp < imuBuffer[head].timestamp){
        return false;
    }

    // If the data is out of order, insert it in the correct location
    auto insertPosition = std::upper_bound(
        imuBuffer.begin() + static_cast<ptrdiff_t>(head), imuBuffer.end(), imu.timestamp, 
        [](double t, const Imu &i){return t < i.timestamp;}
    );

    imuBuffer.insert(insertPosition, imu);
    return true;
}

bool ImuStream::pushImu(const Imu &imu){
    if(health.load(std::memory_order_relaxed) == SensorHealth::DISABLED){
        return false;
    }

    {
        std::lock_guard<std::mutex> lock(m_imuBuffer);
        if(!fastPush(imu)){
            return false;
        }
    }

    c_imuBuffer.notify_one();
    return true;
}

Imu ImuStream::linInter(const Imu &first, const Imu &second, double target){
    double diff = (second.timestamp - first.timestamp);

    // Check for small diffs
    if(diff < MIN_DIFF){
        Imu fallback = first;
        fallback.timestamp = target;
        return fallback;
    }

    double alpha = (target - first.timestamp) / diff;

    Imu imu;

    imu.timestamp = target;

    imu.a = first.a + alpha * (second.a - first.a);
    imu.w = first.w + alpha * (second.w - first.w);
    imu.t = first.t + static_cast<float>(alpha) * (second.t - first.t);

    return imu;
}

bool ImuStream::extractRange(double tStart, double tEnd, std::vector<Imu> &targetBuffer){
    std::lock_guard<std::mutex> lock(m_imuBuffer);
    targetBuffer.clear();

    // Check for a valid time range
    if(tEnd < tStart){
        return false;
    }

    // Check if the buffer is empty
    if(imuBuffer.size() < head + 2){
        return false;
    }

    // Make sure the imuBuffer contains suitable data
    if((imuBuffer[head].timestamp > tStart) || (imuBuffer.back().timestamp < tEnd)){
        return false;
    }

    // First timepoint before or after tStart
    auto startIt = std::lower_bound(
        imuBuffer.begin() + static_cast<ptrdiff_t>(head), imuBuffer.end(), tStart,
        [](const Imu &i, double t){return i.timestamp < t;}
    );

    // First timepoint after tEnd
    auto endIt = std::upper_bound(
        imuBuffer.begin() + static_cast<ptrdiff_t>(head), imuBuffer.end(), tEnd,
        [](double t, const Imu &i){return t < i.timestamp;}
    );

    // Check if there is a timestamp too close to tStart
    if(std::abs(startIt->timestamp - tStart) < DB_EPSILON){
        Imu startSample = *startIt;
        startSample.timestamp = tStart;
        targetBuffer.push_back(startSample);
        startIt++;
    }else if(startIt > (imuBuffer.begin() + static_cast<ptrdiff_t>(head)) && std::abs((startIt - 1)->timestamp - tStart) < DB_EPSILON){
        Imu startSample = *(startIt - 1);
        startSample.timestamp = tStart;
        targetBuffer.push_back(startSample);
    }else{
        targetBuffer.push_back(linInter(*(startIt - 1),*startIt , tStart));
    }

    bool appendInterEnd = false;
    Imu endInter;


    // Check if there is a timestamp too close to tEnd
    if(std::abs((endIt - 1)->timestamp - tEnd) < DB_EPSILON){
        appendInterEnd = false;
    }else{
        appendInterEnd = true;
        endInter = linInter(*(endIt - 1), *endIt, tEnd);
    }

    if(startIt < endIt){
        targetBuffer.insert(targetBuffer.end(), startIt, endIt);
    }
    
    if(appendInterEnd){
        targetBuffer.push_back(endInter);
    }

    return true;
}

SensorHealth ImuStream::waitUntil(double timestamp){
    if(health.load(std::memory_order_relaxed) == SensorHealth::DISABLED){
        return SensorHealth::DISABLED;
    }

    std::unique_lock<std::mutex> lock(m_imuBuffer);
    bool valid = c_imuBuffer.wait_for(
        lock, WAIT_TIMEOUT, 
        [&](){return (!imuBuffer.empty() && imuBuffer.back().timestamp > timestamp);}
    );

    // Update the sensor health if a frame is missed
    if(!valid){
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

    missedFrames = 0;
    health.store(SensorHealth::ACTIVE, std::memory_order_relaxed);
    return SensorHealth::ACTIVE;
}

void ImuStream::pruneUntil(double horizonTime){
    std::lock_guard<std::mutex> lock(m_imuBuffer);
    if(imuBuffer.empty() || head >= imuBuffer.size()){
        return;
    }


    // Find the first sample with timestamp greater than horizonTime
    auto horizonIt = std::upper_bound(
        imuBuffer.begin() + static_cast<ptrdiff_t>(head), imuBuffer.end(), horizonTime,
        [](double t, const Imu &i){return i.timestamp > t;}
    );

    // Make sure we leave one piece of data before the horizon
    if(horizonIt > imuBuffer.begin() + static_cast<ptrdiff_t>(head)){
        head = static_cast<size_t>(std::distance(imuBuffer.begin(), horizonIt - 1));
    }

    // Shift all the data so head = 0 if head is greater than some threshold
    compactThreshold = imuBuffer.capacity() / 2;
    if(head > compactThreshold && compactThreshold > 0){
        size_t newSize = static_cast<size_t>(std::distance(imuBuffer.begin() + static_cast<ptrdiff_t>(head), imuBuffer.end()));
        std::move(imuBuffer.begin() + static_cast<ptrdiff_t>(head), imuBuffer.end(), imuBuffer.begin());

        imuBuffer.resize(newSize);
        head = 0;
    }
}

void ImuStream::flush(){
    std::lock_guard<std::mutex> lock(m_imuBuffer);
    imuBuffer.clear();
    head = 0;
}

SensorHealth ImuStream::getHealth(){
    return health.load(std::memory_order_relaxed);
}

}