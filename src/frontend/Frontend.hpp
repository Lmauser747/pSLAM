#pragma once

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <gtsam/navigation/ImuBias.h>
#include <string>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <memory>
#include <atomic>
#include <optional>

#include <eigen3/Eigen/Dense>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/PreintegrationParams.h>

#include "CoreTypes.hpp"
#include "frontend/HfTracker.hpp"
#include "frontend/LfTracker.hpp"
#include "hal/Synchronizer.hpp"

namespace PoolSLAM{

struct FrontendConfig{
    LfTrackerConfig lfConfig; // Config for lf tracker

    size_t w, h; // Width and height of image
    Eigen::Matrix3f K; // Camera calibration matrix
    Distortion dist; // Distorion of the camera
    std::shared_ptr<gtsam::PreintegrationParams> preintParams; // Parameters for IMU preintegration

    float droppedFeatureThreshold = 0.5f; // Percentrage of dropped features frame-to-frame that cause keyframe
    double maxTimeBetweenKf = 1.0; // Maximum time between keyframes
    size_t hfQueueSize = 10; // Max size of hfQueue
};

class Frontend{
    const FrontendConfig config;

    Synchronizer &sync; // Synchronizer
    HfTracker hf; // High-frequency optical-flow tracker
    LfTracker lf; // Low-frequency global tracker

    std::queue<std::unique_ptr<BackendHfPacket>> hfQueue; // Queue for hf packets to be sent to backend
    gtsam::imuBias::ConstantBias imuBias; // IMU bias data from the backend

    uint32_t frameId = 0;

    std::optional<SyncedPacket> syncPacket;

    std::mutex m_hfQueue; // Mutex for hfQueue access
    std::condition_variable c_hfQueue; // Condition variable to notify backend of new data

    std::atomic<bool> runFrontend{false};
    
    std::thread frontendThread; // Thread to run frontend and HfTracker tasks

    // Function to run in frontendThread
    void processPacket(); 
public:
    Frontend(FrontendConfig configuration, Synchronizer &syncronizer, const std::string &vocabPath) : config(configuration), sync(syncronizer), hf(config.h, config.w, config.K, config.dist), lf(config.lfConfig, vocabPath){}

    // Pop the Hf feature from the queue
    std::unique_ptr<BackendHfPacket> getHfPacket();

    // Start running the threads
    void start();

    // Stop running the threads
    void end();

    ~Frontend(){
        if(runFrontend.load(std::memory_order_relaxed)){
            end();
        }
    }
};

}