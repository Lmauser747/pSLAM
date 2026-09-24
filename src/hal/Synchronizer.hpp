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

#include "HAL.hpp"
#include "HALTypes.hpp"

namespace PoolSLAM{

struct SyncedPacket{
    // Packet timestamp in seconds
    double timestamp = 0.0;

    // Vision data
    PixelArray *image;

    // IMU data
    std::vector<IMU> imu;

    // Depth data
    std::optional<Depth> depth;
};

class Synchronizer{
private:
    static constexpr size_t DEFAULT_MAX_OUTPUT_QUEUE_SIZE = 10;

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
    const size_t maxOutputQueueSize = DEFAULT_MAX_OUTPUT_QUEUE_SIZE; // Maximum size of the output queue

    void threadTask(); // Task to be run in syncThread
public:
    Synchronizer(ImuStream &imu, DepthStream &depth, ImageStream &image) : 
    imuStream(imu), depthStream(depth), imageStream(image){}

    void start(); // Starts synchronizer
    void stop(); // Stops synchronizer

    bool getNextPacket(SyncedPacket &target, std::chrono::milliseconds timeout); // Get the next synced packet in the queue
};

}