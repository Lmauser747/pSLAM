#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <memory>
#include <atomic>

#include <eigen3/Eigen/Dense>
#include <gtsam/navigation/ImuBias.h>
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
    std::shared_ptr<gtsam::PreintegrationParams> preintParams = []() {
        auto p = gtsam::PreintegrationParams::MakeSharedU(9.81);
        p->accelerometerCovariance = Eigen::Matrix3d::Identity() * 1e-3;
        p->gyroscopeCovariance = Eigen::Matrix3d::Identity() * 1e-4;
        p->integrationCovariance = Eigen::Matrix3d::Identity() * 1e-5;
        return p;
    }(); // Parameters for IMU preintegration

    float droppedFeatureThreshold = 0.5f; // Percentrage of dropped features frame-to-frame that cause keyframe
    float meanDisplacementThreshold = 15.0f; // Mean displacement between keyframes
    double maxTimeBetweenKf = 1.0; // Maximum time between keyframes
    size_t hfQueueSize = 10; // Max size of hfQueue
    Eigen::Matrix3d bodyToCameraRot = Eigen::Matrix3d::Identity(); // Rotation from body/IMU frame to camera frame
};

class Frontend{
    const FrontendConfig config;

    Synchronizer &sync; // Synchronizer
    HfTracker hf; // High-frequency optical-flow tracker
    LfTracker lf; // Low-frequency global tracker

    std::queue<std::unique_ptr<BackendHfPacket>> hfQueue; // Queue for hf packets to be sent to backend
    gtsam::imuBias::ConstantBias imuBias; // IMU bias data from the backend
    std::unique_ptr<gtsam::PreintegratedImuMeasurements> keyframeImuDelta; // Accumulated across keyframes
    double lastImuTimestamp = -1.0;

    uint32_t frameId = 0;

    std::mutex m_hfQueue; // Mutex for hfQueue access
    std::mutex m_imuBias;  // Mutex for IMU bias access
    std::condition_variable c_hfQueue; // Condition variable to notify backend of new data

    std::atomic<bool> runFrontend{false};
    
    std::thread frontendThread; // Thread to run frontend and HfTracker tasks

    // Function to run in frontendThread
    void processPacket(); 

    friend class TestFrontend;
public:
    Frontend(const FrontendConfig &configuration, Synchronizer &syncronizer, const std::string &vocabPath) : config(configuration), sync(syncronizer), hf(config.w, config.h, config.K, config.dist), lf(config.lfConfig, vocabPath){}

    // Pop the Hf feature from the queue
    std::unique_ptr<BackendHfPacket> getHfPacket();

    // Start running the threads
    void start();

    // Stop running the threads
    void end();

    // Get a reference to the LfTracker
    LfTracker &getLfTracker();

    bool setBias(const gtsam::imuBias::ConstantBias &bias);

    ~Frontend(){
        end();
    }
};

}