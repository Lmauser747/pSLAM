#pragma once
#include <deque>
#include <mutex>
#include <condition_variable>
#include <optional>

#include "HALTypes.hpp"



namespace PoolSLAM{

constexpr uint32_t DEFAULT_DEPTH_BUFFER_SIZE = 500;
constexpr uint32_t DEFAULT_IMU_BUFFER_SIZE = 1000;

class IMUStream{
private:
    std::vector<IMU> imuBuffer; // Flat buffer for IMU data
    uint32_t head = 0; // Index of front of buffer
    uint32_t maxHead; // Maximum value of head before shuffling is needed
    uint32_t initSize; // Initial size of the buffer

    std::mutex m_imuBuffer; // Mutex to control access to imuBuffer
    std::condition_variable c_imuBuffer; // Condition variable for new data

    std::atomic<SensorHealth> health; // Stores the health of the IMU
    uint32_t missedFrames; // Number of consectutive frames that can be missed

    bool fastPush(const IMU &imu); // Quickly moves IMU data into the flat buffer

    
public:
    IMUStream(uint32_t bufferSize = DEFAULT_IMU_BUFFER_SIZE) : initSize(bufferSize){
        maxHead = initSize / 2;
        imuBuffer.reserve(initSize);
    }

    bool pushIMU(const IMU &imu); // Pushes IMU data onto the stack
    bool extractRange(double tStart, double tEnd, std::vector<IMU> &targetBuffer); // Extracts a range of IMU data to target buffer
    SensorHealth waitUntil(double timestamp); // Blocking wait until imu data more recent then timestamp is ingested
    void pruneUntil(double horizonTime); // Remove all items older than horizonTime

};

class DepthStream{
private:
    std::deque<Depth> depthBuffer; // Ring buffer for depth data
    uint32_t maxSize; // Maximum size of the ring buffer before it begins overwriting

    std::mutex m_depthBuffer; // Mutex to control access to depthBuffer
    std::condition_variable c_depthBuffer; // Condition variable for new data

    std::atomic<SensorHealth> health{SensorHealth::ACTIVE}; // Stores the health of the depth sensor
    uint32_t missedFrames; // Number of consecutive frames the stream failed to provide data for
    
    bool fastPush(const Depth &depth); // Function to quickly move data into the ring buffer   
public:
    DepthStream(uint32_t bufferSize = DEFAULT_DEPTH_BUFFER_SIZE) : maxSize(bufferSize){}

    bool pushDepth(const Depth &depth); // Pushes depth into the ring buffer in a thread safe manner
    std::optional<Depth> getInterDepth(double targetTime); // Gets interpolated depth from the buffer at target time
    SensorHealth waitUntil(double timestamp); // Bloccking wait until depth data more recent then timestamp is ingested
    void pruneUntil(double horizonTime); // Remove all items in the buffer older than hroizonTime
    void flush(); // Flush the buffer
    SensorHealth getHealth(); // Get the health of the sensor
};


}