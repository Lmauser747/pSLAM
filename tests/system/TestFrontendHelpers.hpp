#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <memory>
#include <string>
#include <vector>
#include <ios>

#include <eigen3/Eigen/Dense>

#include <gtsam/navigation/ImuBias.h>
#include <gtsam/navigation/PreintegrationParams.h>

#include "CoreTypes.hpp"
#include "descriptor/BinaryBow.hpp"
#include "descriptor/Qov.hpp"
#include "frontend/Frontend.hpp"
#include "pool_slam/Hal.hpp"
#include "pool_slam/HalTypes.hpp"

namespace PoolSLAM {
namespace test_helpers {

constexpr size_t kDefaultWidth = 640;
constexpr size_t kDefaultHeight = 480;
constexpr uint8_t kDefaultBg = 128;
constexpr uint8_t kDarkVal = 25;
constexpr uint8_t kBrightVal = 230;
constexpr int kCornerRadius = 14;

// Writes a valid binary vocabulary file matching BinaryBowTree::readVocabFile layout
inline std::string writeTempVocabFile(const std::string &filename = "test_frontend_vocab.bin") {
    std::ofstream out(filename, std::ios::binary | std::ios::trunc);
    descriptor::QovDescriptor node;
    for (size_t i = 0; i < descriptor::BinaryBowTree::TREE_SIZE; i++) {
        node.bits = {i * 0x01010101ULL, ~i, i ^ 0x55555555ULL, i + 17ULL};
        out.write(reinterpret_cast<const char *>(node.bits.data()), sizeof(node.bits));
    }

    const float defaultIdf = 1.0f;
    for (size_t w = 0; w < descriptor::BinaryBowTree::VOCAB_SIZE; w++) {
        node.bits = {w * 0xABCDEF01ULL, ~(w * 3ULL), w ^ 0xF0F0F0F0ULL, w + 99ULL};
        out.write(reinterpret_cast<const char *>(node.bits.data()), sizeof(node.bits));
        out.write(reinterpret_cast<const char *>(&defaultIdf), sizeof(defaultIdf));
    }
    out.close();
    return filename;
}

// Constructs a default camera calibration matrix
inline Eigen::Matrix3f makeDefaultK() {
    Eigen::Matrix3f K = Eigen::Matrix3f::Identity();
    K(0, 0) = 525.0f;
    K(1, 1) = 525.0f;
    K(0, 2) = 320.0f;
    K(1, 2) = 240.0f;
    return K;
}

// Constructs a default FrontendConfig with valid GTSAM PreintegrationParams
inline FrontendConfig makeFrontendConfig(size_t width = kDefaultWidth,
                                         size_t height = kDefaultHeight,
                                         double maxTimeBetweenKf = 1.0,
                                         float droppedFeatureThreshold = 0.5f,
                                         size_t hfQueueSize = 10) {
    FrontendConfig cfg{};
    cfg.w = width;
    cfg.h = height;
    cfg.K = makeDefaultK();
    cfg.dist = Distortion{};

    cfg.preintParams = gtsam::PreintegrationParams::MakeSharedU(9.81);
    cfg.preintParams->accelerometerCovariance = Eigen::Matrix3d::Identity() * 1e-3;
    cfg.preintParams->gyroscopeCovariance = Eigen::Matrix3d::Identity() * 1e-4;
    cfg.preintParams->integrationCovariance = Eigen::Matrix3d::Identity() * 1e-5;

    cfg.droppedFeatureThreshold = droppedFeatureThreshold;
    cfg.maxTimeBetweenKf = maxTimeBetweenKf;
    cfg.hfQueueSize = hfQueueSize;

    cfg.lfConfig.w = width;
    cfg.lfConfig.h = height;
    cfg.lfConfig.K = cfg.K;
    cfg.lfConfig.dist = cfg.dist;
    return cfg;
}

// Creates a timestamped HAL IMU sample
inline Imu createImuSample(double t,
                           const Eigen::Vector3d &accel = Eigen::Vector3d(0.0, 0.0, -9.81),
                           const Eigen::Vector3d &omega = Eigen::Vector3d::Zero()) {
    Imu sample;
    sample.timestamp = t;
    sample.a = accel;
    sample.w = omega;
    sample.t = 25.0f;
    return sample;
}

// Pushes dense 200 Hz IMU and 20 Hz Depth samples across [tStart, tEnd]
// Ensures samples strictly newer than target frames exist to satisfy HAL waitUntil() predicates
inline void pushInertialAndDepthWindow(ImuStream &imuStream,
                                       DepthStream &depthStream,
                                       double tStart,
                                       double tEnd,
                                       double depthZ = 5.0,
                                       const Eigen::Vector3d &accel = Eigen::Vector3d(0.0, 0.0, -9.81),
                                       const Eigen::Vector3d &omega = Eigen::Vector3d::Zero()) {
    const double imuDt = 0.005;   // 200 Hz
    const double depthDt = 0.050; // 20 Hz (< 0.2s MAX_GAP)

    const size_t imuSteps = static_cast<size_t>(std::ceil((tEnd - tStart) / imuDt)) + 2;
    for (size_t i = 0; i <= imuSteps; i++) {
        const double t = tStart + static_cast<double>(i) * imuDt;
        imuStream.pushImu(createImuSample(t, accel, omega));
    }

    const size_t depthSteps = static_cast<size_t>(std::ceil((tEnd - tStart) / depthDt)) + 2;
    for (size_t i = 0; i <= depthSteps; i++) {
        const double t = tStart + static_cast<double>(i) * depthDt;
        depthStream.pushDepth({t, depthZ});
    }
}

// Renders a smooth 2D corner with a bilinear ramp profile for FAST + Shi-Tomasi + IC-LK
inline void drawSmoothCorner(std::vector<uint8_t> &buffer,
                             size_t width,
                             size_t height,
                             float cx,
                             float cy,
                             int radius = kCornerRadius,
                             uint8_t darkVal = kDarkVal,
                             uint8_t brightVal = kBrightVal,
                             float rampWidth = 2.0f) {
    const float midVal = 0.5f * (static_cast<float>(darkVal) + static_cast<float>(brightVal));
    const float halfRange = 0.5f * (static_cast<float>(brightVal) - static_cast<float>(darkVal));

    const int minX = std::max(0, static_cast<int>(cx) - radius);
    const int maxX = std::min(static_cast<int>(width) - 1, static_cast<int>(cx) + radius);
    const int minY = std::max(0, static_cast<int>(cy) - radius);
    const int maxY = std::min(static_cast<int>(height) - 1, static_cast<int>(cy) + radius);

    for (int y = minY; y <= maxY; y++) {
        const float dy = (static_cast<float>(y) - cy) / rampWidth;
        const float wy = std::clamp(dy, -1.0f, 1.0f);
        for (int x = minX; x <= maxX; x++) {
            const float dx = (static_cast<float>(x) - cx) / rampWidth;
            const float wx = std::clamp(dx, -1.0f, 1.0f);
            const float intensity = midVal + halfRange * (wx * wy);
            buffer[static_cast<size_t>(y * static_cast<int>(width) + x)] =
                static_cast<uint8_t>(std::clamp(intensity, 0.0f, 255.0f));
        }
    }
}

// Generates a 3D landmark grid in front of the camera and renders projected corners into an image
inline std::vector<uint8_t> renderProjectedGridScene(size_t width,
                                                     size_t height,
                                                     const Eigen::Matrix3f &K,
                                                     const Eigen::Matrix3f &R = Eigen::Matrix3f::Identity(),
                                                     const Eigen::Vector3f &t = Eigen::Vector3f::Zero(),
                                                     size_t cols = 6,
                                                     size_t rows = 5,
                                                     float depthMeters = 3.0f) {
    std::vector<uint8_t> img(width * height, kDefaultBg);
    const Eigen::Matrix3f KInv = K.inverse();

    const float startX = 90.0f;
    const float endX = static_cast<float>(width) - 90.0f;
    const float startY = 80.0f;
    const float endY = static_cast<float>(height) - 80.0f;

    const float stepX = (cols > 1) ? (endX - startX) / static_cast<float>(cols - 1) : 0.0f;
    const float stepY = (rows > 1) ? (endY - startY) / static_cast<float>(rows - 1) : 0.0f;

    for (size_t r = 0; r < rows; r++) {
        for (size_t c = 0; c < cols; c++) {
            const float u0 = startX + static_cast<float>(c) * stepX;
            const float v0 = startY + static_cast<float>(r) * stepY;

            const Eigen::Vector3f ray0 = KInv * Eigen::Vector3f(u0, v0, 1.0f);
            const Eigen::Vector3f pt3d = ray0 * depthMeters;
            const Eigen::Vector3f ptCam = R * pt3d + t;

            if (ptCam.z() > 0.1f) {
                const Eigen::Vector2f uv = (K * ptCam).hnormalized();
                drawSmoothCorner(img, width, height, uv.x(), uv.y());
            }
        }
    }
    return img;
}

} // namespace test_helpers
} // namespace PoolSLAM