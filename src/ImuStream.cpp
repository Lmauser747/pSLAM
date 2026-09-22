#include <algorithm>

#include <atomic>
#include <eigen3/Eigen/Core>
#include <iterator>
#include <linux/limits.h>
#include <mutex>

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
        [](double t, const IMU &i){return t < i.timestamp;}
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
    double alpha = (target - first.timestamp) / diff;

    IMU imu;

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
    if(imuBuffer.size() - head < 3){
        return false;
    }

    // Make sure the imuBuffer contains suitable data
    if((imuBuffer[head].timestamp > tStart) || (imuBuffer.back().timestamp < tEnd)){
        return false;
    }

    auto startIterator = std::lower_bound(
        imuBuffer.begin() + head, imuBuffer.end(), tStart,
        [](double t, const IMU &i){return i.timestamp < t;}
    );

    auto endIterator = std::upper_bound(
        imuBuffer.begin() + head, imuBuffer.end(), tEnd,
        [](double t, const IMU &i){return i.timestamp > t;}
    );

    IMU startInter = linInter(*(startIterator -1), *startIterator, tStart);
    IMU endInter = linInter(*endIterator, *(endIterator + 1), tEnd);

    auto innerCount = std::distance(startIterator, endIterator);

    targetBuffer.reserve(innerCount + 2);
    targetBuffer.push_back(startInter);
    targetBuffer.insert(targetBuffer.end(), startIterator, endIterator);
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

void ImuStream::flush(){
    std::lock_guard<std::mutex> lock(m_imuBuffer);
    imuBuffer.clear();
}

SensorHealth ImuStream::getHealth(){
    return health.load(std::memory_order_relaxed);
}

}