#include <algorithm>
#include <atomic>
#include <iterator>
#include <mutex>
#include <vector>
#include <cstddef>

#include <eigen3/Eigen/Core>

#include "HAL.hpp"
#include "HALTypes.hpp"

namespace PoolSLAM {

bool ImuStream::fastPush(const IMU &imu){
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
        imuBuffer.begin() + head, imuBuffer.end(), imu.timestamp, 
        [](const IMU &i, double t){return t < i.timestamp;}
    );

    imuBuffer.insert(insertPosition, imu);
    return true;
}

bool ImuStream::pushIMU(const IMU &imu){
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

IMU ImuStream::linInter(const IMU &first, const IMU &second, double target){
    double diff = (second.timestamp - first.timestamp);

    // Check for small diffs
    if(diff < MIN_DIFF){
        IMU fallback = first;
        fallback.timestamp = target;
        return fallback;
    }

    double alpha = (target - first.timestamp) / diff;

    IMU imu;

    imu.timestamp = target;

    imu.a = first.a + alpha * (second.a - first.a);
    imu.w = first.w + alpha * (second.w - first.w);
    imu.t = first.t + alpha * (second.t - first.t);

    return imu;
}

bool ImuStream::extractRange(double tStart, double tEnd, std::vector<IMU> &targetBuffer){
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
        imuBuffer.begin() + head, imuBuffer.end(), tStart,
        [](const IMU &i, double t){return i.timestamp < t;}
    );

    // First timepoint after tEnd
    auto endIt = std::upper_bound(
        imuBuffer.begin() + head, imuBuffer.end(), tEnd,
        [](const IMU &i, double t){return i.timestamp > t;}
    );

    // Check if there is a timestamp too close to tStart
    if((startIt->timestamp - tStart) < DB_EPSILON){
        IMU startSample = *startIt;
        startSample.timestamp = tStart;
        targetBuffer.push_back(startSample);
        startIt++;
    }else{
        targetBuffer.push_back(linInter(*(startIt - 1),*startIt , tStart));
    }

    bool appendInterEnd = false;
    IMU endInter;


    // Check if there is a timestamp too close to tEnd
    if(((endIt - 1)->timestamp - tEnd) < DB_EPSILON){
        appendInterEnd = false;
    }else{
        appendInterEnd = true;
        endInter = linInter(*(endIt - 1), *endIt, tEnd);
    }


    auto innerCount = std::distance(startIt, endIt);
    targetBuffer.insert(targetBuffer.end(), startIt, endIt);
    
    if(appendInterEnd){
        targetBuffer.push_back(endInter);
    }
    targetBuffer.push_back(endInter);

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
        imuBuffer.begin() + head, imuBuffer.end(), horizonTime,
        [](const IMU &i, double t){return i.timestamp > t;}
    );

    // Make sure we leave one piece of data before the horizon
    if(horizonIt > imuBuffer.begin() + head){
        head = std::distance(imuBuffer.begin(), horizonIt - 1);
    }

    // Shift all the data so head = 0 if head is greater than some threshold
    compactThreshold = imuBuffer.capacity() / 2;
    if(head > compactThreshold && compactThreshold > 0){
        size_t newSize = std::distance(imuBuffer.begin() + head, imuBuffer.end());
        std::move(imuBuffer.begin() + head, imuBuffer.end(), imuBuffer.begin());

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