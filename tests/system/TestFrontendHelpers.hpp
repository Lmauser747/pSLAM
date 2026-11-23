#pragma once

#include <gtest/gtest.h>
#include <vector>
#include <memory>
#include <chrono>
#include <future>
#include <string>
#include <fstream>
#include <cmath>

#include "CoreTypes.hpp"
#include "frontend/Frontend.hpp"
#include "hal/Synchronizer.hpp"
#include "../unit/HalTestHelpers.hpp"

namespace PoolSLAM {

class TestFrontend : public ::testing::Test {
protected:
    static constexpr int W = 640;
    static constexpr int H = 480;

    static Imu makeImu(double t, const Eigen::Vector3d& a, const Eigen::Vector3d& w, float temp = 0.0f) {
        Imu sample;
        sample.timestamp = t;
        sample.a = a;
        sample.w = w;
        sample.t = temp;
        return sample;
    }

    static Depth makeDepth(double t, double z) {
        Depth sample;
        sample.timestamp = t;
        sample.z = z;
        return sample;
    }

    static size_t getImageFreeListSize(const ImageStream& stream) {
        return TestImageStream::getFreeListSize(stream);
    }

    static void pushInertialAndDepthWindow(ImuStream& imuStream,
                                           DepthStream& depthStream,
                                           double tStart,
                                           double tEnd,
                                           double depthZ = 5.0,
                                           const Eigen::Vector3d& accel = Eigen::Vector3d(0.0, 0.0, -9.81),
                                           const Eigen::Vector3d& omega = Eigen::Vector3d::Zero()) {
        const double imuDt = 0.005;   // 200 Hz
        const double depthDt = 0.050; // 20 Hz
        const size_t imuSteps = static_cast<size_t>(std::ceil((tEnd - tStart) / imuDt)) + 2;
        for (size_t i = 0; i <= imuSteps; ++i) {
            double t = tStart + static_cast<double>(i) * imuDt;
            imuStream.pushImu(makeImu(t, accel, omega));
        }
        const size_t depthSteps = static_cast<size_t>(std::ceil((tEnd - tStart) / depthDt)) + 2;
        for (size_t i = 0; i <= depthSteps; ++i) {
            double t = tStart + static_cast<double>(i) * depthDt;
            depthStream.pushDepth(makeDepth(t, depthZ));
        }
    }

    static std::string findVocabPath() {
        const std::vector<std::string> candidatePaths = {
            "benchmarks/accuracy/vocab.bin",
            "../benchmarks/accuracy/vocab.bin",
            "../../benchmarks/accuracy/vocab.bin",
            "../../../benchmarks/accuracy/vocab.bin"
        };
        for (const auto& path : candidatePaths) {
            std::ifstream f(path);
            if (f.good()) {
                return path;
            }
        }
        return "";
    }

    static FrontendConfig makeConfig() {
        FrontendConfig cfg;
        cfg.w = W;
        cfg.h = H;
        cfg.K << 400.0f, 0.0f, 320.0f,
                 0.0f, 400.0f, 240.0f,
                 0.0f, 0.0f, 1.0f;
        cfg.dist = Distortion{0.0f, 0.0f, 0.0f, 0.0f};
        cfg.droppedFeatureThreshold = 0.5f;
        cfg.meanDisplacementThreshold = 15.0f;
        cfg.maxTimeBetweenKf = 1.0;
        cfg.hfQueueSize = 10;
        cfg.bodyToCameraRot = Eigen::Matrix3d::Identity();

        cfg.lfConfig.w = W;
        cfg.lfConfig.h = H;
        cfg.lfConfig.dist = cfg.dist;
        cfg.lfConfig.numBuckets = 50;
        cfg.lfConfig.K = cfg.K;
        cfg.lfConfig.minTfIdfScore = 0.05f;
        cfg.lfConfig.excludedFrames = 20;

        return cfg;
    }

    // Helper to generate a textured image with high-contrast corners for FAST
    static std::vector<uint8_t> makeTexturedImage(int offsetX = 0, int offsetY = 0) {
        std::vector<uint8_t> img(W * H, 50);
        // Draw grid of corners spaced by 40 pixels
        for (int y = 40; y < static_cast<int>(H) - 40; y += 40) {
            for (int x = 40; x < static_cast<int>(W) - 40; x += 40) {
                int cx = x + offsetX;
                int cy = y + offsetY;
                for (int dy = -5; dy <= 5; ++dy) {
                    for (int dx = -5; dx <= 5; ++dx) {
                        int px = cx + dx;
                        int py = cy + dy;
                        if (px >= 0 && px < static_cast<int>(W) && py >= 0 && py < static_cast<int>(H)) {
                            if (dx >= 0 && dy >= 0) {
                                img[static_cast<size_t>(py) * W + static_cast<size_t>(px)] = 220;
                            }
                        }
                    }
                }
            }
        }
        return img;
    }

    // Safe packet pop with timeout (using std::promise to prevent future destructor deadlock)
    static std::unique_ptr<BackendHfPacket> popHfPacketWithTimeout(Frontend& fe, std::chrono::milliseconds timeout = std::chrono::milliseconds(400)) {
        auto prom = std::make_shared<std::promise<std::unique_ptr<BackendHfPacket>>>();
        auto fut = prom->get_future();
        std::thread([&fe, prom]() {
            try {
                prom->set_value(fe.getHfPacket());
            } catch (...) {
                prom->set_value(nullptr);
            }
        }).detach();
        if (fut.wait_for(timeout) == std::future_status::ready) {
            return fut.get();
        }
        return nullptr;
    }

    // Accessors for private members
    static bool getRunFrontend(const Frontend& fe) {
        return fe.runFrontend.load();
    }

    static uint32_t getFrameId(const Frontend& fe) {
        return fe.frameId;
    }

    static size_t getHfQueueSize(Frontend& fe) {
        std::lock_guard<std::mutex> lock(fe.m_hfQueue);
        return fe.hfQueue.size();
    }

    static gtsam::imuBias::ConstantBias getImuBias(Frontend& fe) {
        std::lock_guard<std::mutex> lock(fe.m_imuBias);
        return fe.imuBias;
    }

    static HfTracker& getHfTracker(Frontend& fe) {
        return fe.hf;
    }

    static LfTracker& getLfTracker(Frontend& fe) {
        return fe.lf;
    }
};

} // namespace PoolSLAM
