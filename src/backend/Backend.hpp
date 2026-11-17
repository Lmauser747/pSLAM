#pragma once

#include <atomic>
#include <thread>
#include <array>
#include <cstdint>

#include <gtsam/nonlinear/PriorFactor.h>
#include <gtsam/geometry/Point2.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h> 

#include "backend/Bookkeeper.hpp"
#include "pool_slam/SystemTypes.hpp"
#include "frontend/Frontend.hpp"
#include "frontend/LfTracker.hpp"

namespace PoolSLAM{

struct BackendConfig{
    BookkeeperConfig bkConfig;
    gtsam::Pose3 bodyToCameraFrame; // Body to camera translation
};


class Backend{
    Frontend &fe; // Reference to frontend
    LfTracker &lf; // Reference to backend

    gtsam::NonlinearFactorGraph stagingGraph;
    gtsam::Values stagingValues;
    gtsam::Values isamEstimate;

    LandmarkBookkeeper bk; // Bookeeper for triangulating landmarks

    std::atomic<uint8_t> activePose = 0; // Index of active pose
    std::array<Pose, 2> poseBuffer; // Buffer to hold active pose

    std::atomic<bool> runBackend; // Variable to run backend
    std::thread backendThread; // Thread to run backend

    // Backend packet processing task
    void backendTask();

    // Function for the first frame to bootstrap
    void bootstrap();

public:
    Backend(Frontend &frontend, LfTracker &lfTracker, BackendConfig &cfg) : fe(frontend), lf(lfTracker), 
    bk(stagingGraph, stagingValues, isamEstimate, cfg.bodyToCameraFrame, cfg.bkConfig) {}

    // Gets the current pose 
    Pose getCurrentPose();

    // Start function
    void start();

    // End function
    void end();

};

}