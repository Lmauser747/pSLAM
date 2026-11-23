#pragma once

#include <cstddef>
#include <vector>
#include <deque>
#include <queue>
#include <atomic>
#include <cstdint>

#include <gtest/gtest.h>
#include <eigen3/Eigen/Core>

#include "pool_slam/Hal.hpp"
#include "pool_slam/HalTypes.hpp"

namespace PoolSLAM {

class TestDepthStream : public ::testing::Test {
protected:
    static bool callFastPush(DepthStream& s, const Depth& d) {
        return s.fastPush(d);
    }

    static size_t getBufferSize(const DepthStream& s) {
        return s.depthBuffer.size();
    }

    static const std::deque<Depth>& getBuffer(const DepthStream& s) {
        return s.depthBuffer;
    }

    static int getMissedFrames(const DepthStream& s) {
        return s.missedFrames;
    }

    static void setMissedFrames(DepthStream& s, int count) {
        s.missedFrames = count;
    }

    static void setHealth(DepthStream& s, SensorHealth h) {
        s.health.store(h, std::memory_order_relaxed);
    }

    static Depth makeDepth(double t, double z) {
        Depth d;
        d.timestamp = t;
        d.z = z;
        return d;
    }
};

class TestImuStream : public ::testing::Test {
protected:
    static bool callFastPush(ImuStream& s, const Imu& imu) {
        return s.fastPush(imu);
    }

    static Imu callLinInter(ImuStream& s, const Imu& first, const Imu& second, double target) {
        return s.linInter(first, second, target);
    }

    static size_t getHead(const ImuStream& s) {
        return s.head;
    }

    static size_t getBufferSize(const ImuStream& s) {
        return s.imuBuffer.size();
    }

    static const std::vector<Imu>& getBuffer(const ImuStream& s) {
        return s.imuBuffer;
    }

    static size_t getCompactThreshold(const ImuStream& s) {
        return s.compactThreshold;
    }

    static int getMissedFrames(const ImuStream& s) {
        return s.missedFrames;
    }

    static void setMissedFrames(ImuStream& s, int count) {
        s.missedFrames = count;
    }

    static void setHealth(ImuStream& s, SensorHealth h) {
        s.health.store(h, std::memory_order_relaxed);
    }

    static Imu makeImu(double t, const Eigen::Vector3d& a, const Eigen::Vector3d& w, float temp = 25.0f) {
        Imu packet;
        packet.timestamp = t;
        packet.a = a;
        packet.w = w;
        packet.t = temp;
        return packet;
    }

    static void populateMonotonic(ImuStream& s, size_t count, double t0 = 1.0, double dt = 0.01) {
        for (size_t i = 0; i < count; ++i) {
            double t = t0 + static_cast<double>(i) * dt;
            s.pushImu(makeImu(t, Eigen::Vector3d(0.0, 0.0, 9.81), Eigen::Vector3d::Zero()));
        }
    }
};

class TestImageStream : public ::testing::Test {
public:
    static size_t getFreeListSize(const ImageStream& s) {
        return s.poolFreeList.size();
    }

    static size_t getReadyQueueSize(const ImageStream& s) {
        return s.readyQueue.size();
    }

    static size_t getPoolCapacity(const ImageStream& s) {
        return s.poolCapacity;
    }

    static size_t getMaxQueueSize(const ImageStream& s) {
        return s.maxQueueSize;
    }

    static int getMissedFrames(const ImageStream& s) {
        return s.missedFrames;
    }

    static void setMissedFrames(ImageStream& s, int count) {
        s.missedFrames = count;
    }

    static void setHealth(ImageStream& s, SensorHealth h) {
        s.health.store(h, std::memory_order_relaxed);
    }

    static std::vector<uint8_t> makePattern(int w, int h, uint8_t seed = 0) {
        std::vector<uint8_t> pixels(static_cast<size_t>(w * h));
        for (size_t i = 0; i < pixels.size(); ++i) {
            pixels[i] = static_cast<uint8_t>((seed + i) % 256);
        }
        return pixels;
    }
};

} // namespace PoolSLAM
