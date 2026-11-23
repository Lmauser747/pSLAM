#pragma once

#include <cstddef>
#include <mutex>
#include <vector>
#include <queue>
#include <atomic>
#include <cstdint>

#include <gtest/gtest.h>
#include <eigen3/Eigen/Core>

#include "pool_slam/Hal.hpp"
#include "pool_slam/HalTypes.hpp"
#include "pool_slam/SystemTypes.hpp"
#include "hal/Synchronizer.hpp"

namespace PoolSLAM {

class TestSynchronizer : public ::testing::Test {
protected:
    static constexpr int IMG_W = 64;
    static constexpr int IMG_H = 64;
    static constexpr size_t POOL_CAP = 15;
    static constexpr size_t READY_QUEUE_CAP = 5;

    ImuStream imuStream{1000};
    DepthStream depthStream{500};
    ImageStream imageStream{IMG_W, IMG_H, READY_QUEUE_CAP, POOL_CAP};
    std::vector<uint8_t> dummyPixels;

    void SetUp() override {
        dummyPixels.assign(static_cast<size_t>(IMG_W * IMG_H), 128);
    }

    // Helper to feed monotonic IMU samples covering [t0, t1]
    void pushImuRange(double t0, double t1, double dt = 0.005) {
        for (double t = t0; t <= t1 + 1e-9; t += dt) {
            imuStream.pushImu(Imu{
                t,
                Eigen::Vector3d(0.0, 0.0, 9.81),
                Eigen::Vector3d(0.01, 0.02, 0.03),
                25.0f
            });
        }
    }

    // Helper to feed monotonic Depth samples covering [t0, t1]
    void pushDepthRange(double t0, double t1, double dt = 0.05, double z = 2.5) {
        for (double t = t0; t <= t1 + 1e-9; t += dt) {
            depthStream.pushDepth(Depth{t, z});
        }
    }

    // Helper to push a camera frame
    bool pushFrame(double t) {
        return imageStream.pushFrame(dummyPixels.data(), t);
    }

    // Accessors for Synchronizer private members
    static size_t getOutputQueueSize(Synchronizer& s) {
        std::lock_guard<std::mutex> lock(s.m_outputQueue);
        return s.outputQueue.size();
    }

    static bool getIsRunning(const Synchronizer& s) {
        return s.isRunning.load(std::memory_order_relaxed);
    }

    static double getPreviousFrameTime(const Synchronizer& s) {
        return s.previousFrameTime;
    }

    static void callSyncFatalShutdown(Synchronizer& s, SyncStatus reason, PixelArray* frame) {
        s.syncFatalShutdown(reason, frame);
    }
};

} // namespace PoolSLAM
