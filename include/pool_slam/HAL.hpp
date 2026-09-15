#include <queue>
#include <mutex>
#include <condition_variable>

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
    std::queue<Depth> depthQueue; // Queue to store depths 
    std::mutex m_depthQueue; // Mutex to control access to depthQueue
    std::condition_variable c_depthQueue; // Condition variable for new data

public:
    void pushDepth(const Depth &depth); // Pushes depth onto the queue
    Depth popDepth(); // Pops depth from the queue
    Depth peekDepth(); // Peeks at the depth at the front of the queue
};


}