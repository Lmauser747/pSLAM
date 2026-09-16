#include <queue>
#include <mutex>
#include <condition_variable>
#include <optional>
#include "HALTypes.hpp"



namespace PoolSLAM{


class IMUStream{
private:
    std::queue<IMU> imuQueue; // Queue to store IMU data
    std::mutex m_imuQueue; // Mutex to control access to imuqueue
    std::condition_variable c_imuQueue; // Condition variable for new data

public:
    void pushIMU(const IMU &imu); // Pushes IMU onto the queue
    IMU &popIMU(); // Pops IMU from the queue
    IMU &peekDepth(); // Peeks at the IMU at the front of the queue
};

class DepthStream{
private:
    std::deque<Depth> depthBuffer; // Ring buffer for depth data 
    std::mutex m_depthBuffer; // Mutex to control access to depthBuffer
    std::condition_variable c_depthBuffer; // Condition variable for new data
    uint32_t maxSize; // Maximum size of the ring buffer before it begins overwriting
    bool fastPush(const Depth &depth); // FUnction to quickly move data into the ring buffer
public:
    explicit DepthStream(uint32_t bufferSize) : maxSize(bufferSize){

    }

    bool pushDepth(const Depth &depth); // Pushes depth into the ring buffer in a thread safe manner
    std::optional<Depth> getInterDepth(double targetTime); // Gets interpolated depth from the buffer at target time
    double getLatestTime(); // Gets the timestamp of the latest depth in the buffer
    void pruneUntil(double time); // Remove all items in the buffer older than time
    void flush(); // Flush the buffer
};


}