#pragma once

#include <atomic>
#include <gtsam/geometry/Point2.h>
#include <thread>
#include <array>
#include <cstdint>
#include <vector>
#include <cstddef>

#include <gtsam/nonlinear/PriorFactor.h>

#include "backend/Bookkeeper.hpp"
#include "pool_slam/SystemTypes.hpp"
#include "frontend/Frontend.hpp"
#include "frontend/LfTracker.hpp"

namespace PoolSLAM{

struct BackendConfig{
    BookkeeperConfig bkConfig;
};


class Backend{
    Frontend &fe; // Reference to frontend
    LfTracker &lf; // Reference to backend

    std::atomic<uint8_t> activePose = 0; // Index of active pose
    std::array<Pose, 2> poseBuffer; // Buffer to hold active pose

    std::atomic<bool> runBackend; // Variable to run backend
    std::thread backendThread; // Thread to run backend

    // Backend packet processing task
    void backendTask();

    // Function for the first frame to bootstrap
    void bootstrap();

    // Function to keep track of landmarks
    void updateBookkeeper(uint32_t landmarkId, gtsam::Point2 &pt);

public:
    Backend(Frontend &frontend, LfTracker &lfTracker) : fe(frontend), lf(lfTracker){}

    // Gets the current pose 
    Pose getCurrentPose();

    // Start function
    void start();

    // End function
    void end();

};

}