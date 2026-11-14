#pragma once

#include <chrono>
#include <condition_variable>
#include <vector>
#include <optional>
#include <queue>
#include <mutex>
#include <thread>
#include <atomic>
#include <cstddef>

#include "pool_slam/Hal.hpp"
#include "pool_slam/HalTypes.hpp"
#include "pool_slam/SystemTypes.hpp"

namespace PoolSLAM{

struct SyncedPacket{
    // Packet timestamp in seconds
    double timestamp = 0.0;

    // Vision data
    PixelArray *image;

    // IMU data
    std::vector<Imu> imu;

    // Depth data
    std::optional<Depth> depth;
};

class Synchronizer{
private:
    static constexpr size_t DEFAULT_MAX_OUTPUT_QUEUE_SIZE = 10;
    static constexpr std::chrono::milliseconds DEFAULT_PACKET_TIMEOUT{100};

    // Data streams
    ImuStream &imuStream;
    DepthStream &depthStream;
    ImageStream &imageStream;

    // Output queue and corresponding mutex/cv
    std::queue<SyncedPacket> outputQueue;
    std::mutex m_outputQueue;
    std::condition_variable c_outputQueue;

    // Worker thread stuff
    std::thread syncThread;
    std::atomic<bool> isRunning{false};

    double previousFrameTime = -1.0; // Timestamp of previous frame
    const size_t maxOutputQueueSize; // Maximum size of the output queue

    std::atomic<SyncStatus> syncStatus{SyncStatus::IDLE};

    void threadTask(); // Task to be run in syncThread

    // Trigger a synchronizer based fatal shutdown and nreturn checked out frame to the image stream queue
    void syncFatalShutdown(SyncStatus reason, PixelArray *checkedOutFrame); 
public:
    Synchronizer(ImuStream &imu, DepthStream &depth, ImageStream &image, size_t queueSize = DEFAULT_MAX_OUTPUT_QUEUE_SIZE) : 
    imuStream(imu), depthStream(depth), imageStream(image), maxOutputQueueSize(queueSize){}

    void start(); // Starts synchronizer
    void stop(); // Stops synchronizer

    // Get the next synced packet in the queue
    bool getNextPacket(
        SyncedPacket &target, std::chrono::milliseconds timeout = DEFAULT_PACKET_TIMEOUT
    ); 

    void recycleFrame(PixelArray *frame){
        imageStream.recycleFrame(frame);
    }

    SyncStatus getStatus(){
        return syncStatus.load(std::memory_order_relaxed);
    }

    ~Synchronizer(){
        stop();

        // Drain the queue
        std::lock_guard<std::mutex> lock(m_outputQueue);
        while(!outputQueue.empty()){
            SyncedPacket &packet = outputQueue.front();
            if(packet.image){
                imageStream.recycleFrame(packet.image);
            }
            outputQueue.pop();
        }
    }

};

}