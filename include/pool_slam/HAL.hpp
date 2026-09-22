#pragma once
#include <chrono>
#include <deque>
#include <mutex>
#include <condition_variable>
#include <optional>

#include "HALTypes.hpp"



namespace PoolSLAM{

constexpr size_t DEFAULT_DEPTH_BUFFER_SIZE = 500;
constexpr size_t DEFAULT_IMU_BUFFER_SIZE = 1000;

class ImuStream{
private:
    static constexpr std::chrono::milliseconds WAIT_TIMEOUT{100}; // Maximum wait for valid data
    static constexpr double MIN_DIFF = 1e-6; // Minimum gap between interpolated samples
    static constexpr double DB_EPSILON = 1e-5; // Epsilon for comparing doubles
    static constexpr size_t MAX_MISSED_FRAMES = 10; // Max number of missed frames before sensor is disabled

    std::vector<IMU> imuBuffer; // Flat buffer for IMU data
    size_t head = 0; // Index of front of buffer
    size_t compactThreshold; // Maximum value of head before compacting is needed
    size_t initSize; // Initial size of the buffer

    std::mutex m_imuBuffer; // Mutex to control access to imuBuffer
    std::condition_variable c_imuBuffer; // Condition variable for new data

    std::atomic<SensorHealth> health; // Stores the health of the IMU
    size_t missedFrames; // Number of consectutive frames that can be missed

    bool fastPush(const IMU &imu); // Quickly moves IMU data into the flat buffer
    IMU linInter(const IMU &first, const IMU &second, double target); // Linearly interpolate between two IMU measurements
    
public:
    ImuStream(size_t bufferSize = DEFAULT_IMU_BUFFER_SIZE) : initSize(bufferSize){
        compactThreshold = initSize / 2;
        imuBuffer.reserve(initSize);
    }

    bool pushIMU(const IMU &imu); // Pushes IMU data onto the stack
    bool extractRange(double tStart, double tEnd, std::vector<IMU> &targetBuffer); // Extracts a range of IMU data to target buffer
    SensorHealth waitUntil(double timestamp); // Blocking wait until imu data more recent then timestamp is ingested
    void pruneUntil(double horizonTime); // Remove all items older than horizonTime
    void flush(); // Flush the buffer
    SensorHealth getHealth(); // Get the health of the sensor
};

class DepthStream{
private:
    static constexpr std::chrono::milliseconds WAIT_TIMEOUT{100}; // Maximum wait for valid data
    static constexpr double MAX_DIFF = 0.2; // Maximum gap between timestamps for interpolation
    static constexpr double MIN_DIFF = 1e-6; // Minimum gap between timestamps for interpolation
    static constexpr size_t MAX_MISSED_FRAMES = 10; // Max number of missed frames before sensor is disabled


    std::deque<Depth> depthBuffer; // Ring buffer for depth data
    size_t maxSize; // Maximum size of the ring buffer before it begins overwriting

    std::mutex m_depthBuffer; // Mutex to control access to depthBuffer
    std::condition_variable c_depthBuffer; // Condition variable for new data

    std::atomic<SensorHealth> health{SensorHealth::ACTIVE}; // Stores the health of the depth sensor
    size_t missedFrames; // Number of consecutive frames the stream failed to provide data for
    
    bool fastPush(const Depth &depth); // Function to quickly move data into the ring buffer   
public:
    DepthStream(size_t bufferSize = DEFAULT_DEPTH_BUFFER_SIZE) : maxSize(bufferSize){}

    bool pushDepth(const Depth &depth); // Pushes depth into the ring buffer in a thread safe manner
    std::optional<Depth> getInterDepth(double targetTime); // Gets interpolated depth from the buffer at target time
    SensorHealth waitUntil(double timestamp); // Bloccking wait until depth data more recent then timestamp is ingested
    void pruneUntil(double horizonTime); // Remove all items in the buffer older than hroizonTime
    void flush(); // Flush the buffer
    SensorHealth getHealth(); // Get the health of the sensor
};


}