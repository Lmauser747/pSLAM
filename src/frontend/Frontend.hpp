#pragma once

#include <cstddef>
#include <string>
#include <thread>

#include <eigen3/Eigen/Dense>

#include "CoreTypes.hpp"
#include "frontend/HfTracker.hpp"
#include "frontend/LfTracker.hpp"
#include "hal/Synchronizer.hpp"

namespace PoolSLAM{

struct FrontendConfig{
    LfTrackerConfig lfConfig;
    size_t w, h;
    Eigen::Matrix3f K;
    Distortion dist;
};

class Frontend{
    const FrontendConfig config;

    Synchronizer &sync; // Synchronizer
    HfTracker hf; // High-frequency optical-flow tracker
    LfTracker lf; // Low-frequency global tracker

    std::thread frontendThread; // Thread to run frontend and HfTracker tasks

    // Function to run in frontendThread
    void processPacket(); 
public:
    Frontend(FrontendConfig configuration, Synchronizer &syncronizer, const std::string &vocabPath) : config(configuration), sync(syncronizer), hf(config.h, config.w, config.K, config.dist), lf(config.lfConfig, vocabPath){}

    // Start running the threads
    void start();

    // Stop running the threads
    void end();

};

}