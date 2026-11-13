#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <ios>
#include <optional>
#include <string>
#include <vector>

#include <eigen3/Eigen/Dense>

#include "descriptor/BinaryBow.hpp"
#include "descriptor/Qov.hpp"
#include "pool_slam/Hal.hpp"
#include "pool_slam/HalTypes.hpp"

namespace PoolSLAM {
namespace test_helpers {

// Writes a deterministic binary vocabulary file matching BinaryBowTree::readVocabFile layout
inline bool writeSyntheticVocabFile(const std::string &vocabPath) {
    std::ofstream stream(vocabPath, std::ios::binary | std::ios::trunc);
    if (!stream) {
        return false;
    }

    // 1. Internal branch centroids (TREE_SIZE = 1110 nodes)
    for (size_t i = 0; i < descriptor::BinaryBowTree::TREE_SIZE; i++) {
        descriptor::QovDescriptor branchDes{};
        branchDes.bits[0] = static_cast<uint64_t>(i * 2654435761ULL + 1ULL);
        branchDes.bits[1] = static_cast<uint64_t>((i + 17ULL) * 11400714819323198485ULL);
        branchDes.bits[2] = static_cast<uint64_t>((i + 31ULL) * 6364136223846793005ULL);
        branchDes.bits[3] = static_cast<uint64_t>(~(i * 1442695040888963407ULL));
        stream.write(reinterpret_cast<const char *>(branchDes.bits.data()), sizeof(branchDes.bits));
    }

    // 2. Vocabulary leaf words (VOCAB_SIZE = 10000 words: 32-byte descriptor + 4-byte float idf)
    for (size_t wordId = 0; wordId < descriptor::BinaryBowTree::VOCAB_SIZE; wordId++) {
        descriptor::QovDescriptor leafDes{};
        leafDes.bits[0] = static_cast<uint64_t>((wordId + 101ULL) * 2862933555777941757ULL);
        leafDes.bits[1] = static_cast<uint64_t>((wordId + 211ULL) * 3202034522624059733ULL);
        leafDes.bits[2] = static_cast<uint64_t>((wordId + 307ULL) * 3935559000370003845ULL);
        leafDes.bits[3] = static_cast<uint64_t>((wordId + 401ULL) * 7046029254386353131ULL);
        const float idf = 1.0f + static_cast<float>(wordId % 100) * 0.02f;

        stream.write(reinterpret_cast<const char *>(leafDes.bits.data()), sizeof(leafDes.bits));
        stream.write(reinterpret_cast<const char *>(&idf), sizeof(idf));
    }

    stream.flush();
    return stream.good();
}

// Renders a smooth 2D hyperbolic-ramp corner with strong gradients for FAST, Shi-Tomasi, and IC-LK
inline void drawSmoothCorner(std::vector<uint8_t> &buffer, size_t width, size_t height,
                             float cx, float cy, int radius = 12,
                             uint8_t darkVal = 30, uint8_t brightVal = 220,
                             float rampWidth = 2.0f) {
    const float midVal = 0.5f * (static_cast<float>(darkVal) + static_cast<float>(brightVal));
    const float halfRange = 0.5f * (static_cast<float>(brightVal) - static_cast<float>(darkVal));

    const int minX = std::max(0, static_cast<int>(std::floor(cx)) - radius);
    const int maxX = std::min(static_cast<int>(width) - 1, static_cast<int>(std::ceil(cx)) + radius);
    const int minY = std::max(0, static_cast<int>(std::floor(cy)) - radius);
    const int maxY = std::min(static_cast<int>(height) - 1, static_cast<int>(std::ceil(cy)) + radius);

    for (int y = minY; y <= maxY; y++) {
        const float dy = (static_cast<float>(y) - cy) / rampWidth;
        const float wy = std::clamp(dy, -1.0f, 1.0f);
        for (int x = minX; x <= maxX; x++) {
            const float dx = (static_cast<float>(x) - cx) / rampWidth;
            const float wx = std::clamp(dx, -1.0f, 1.0f);
            const float intensity = midVal + halfRange * (wx * wy);
            buffer[static_cast<size_t>(y) * width + static_cast<size_t>(x)] =
                static_cast<uint8_t>(std::clamp(std::round(intensity), 0.0f, 255.0f));
        }
    }
}

// Generates a synthetic image containing a grid of well-separated 2D corners,
// optionally rendering only the first maxCorners sites to simulate partial occlusion
inline std::vector<uint8_t> createCornerGridImage(size_t width, size_t height,
                                                  size_t cols = 6, size_t rows = 5,
                                                  float shiftX = 0.0f, float shiftY = 0.0f,
                                                  size_t maxCorners = SIZE_MAX,
                                                  uint8_t bgVal = 128) {
    std::vector<uint8_t> img(width * height, bgVal);
    const float marginX = 64.0f;
    const float marginY = 64.0f;
    const float stepX = (static_cast<float>(width) - 2.0f * marginX) /
                        static_cast<float>(std::max<size_t>(1, cols - 1));
    const float stepY = (static_cast<float>(height) - 2.0f * marginY) /
                        static_cast<float>(std::max<size_t>(1, rows - 1));

    size_t drawn = 0;
    for (size_t r = 0; r < rows && drawn < maxCorners; r++) {
        for (size_t c = 0; c < cols && drawn < maxCorners; c++) {
            const float cx = marginX + static_cast<float>(c) * stepX + shiftX;
            const float cy = marginY + static_cast<float>(r) * stepY + shiftY;
            drawSmoothCorner(img, width, height, cx, cy);
            drawn++;
        }
    }
    return img;
}

// Generates a uniform featureless image to force 0 tracked features
inline std::vector<uint8_t> createBlankImage(size_t width, size_t height, uint8_t bgVal = 128) {
    return std::vector<uint8_t>(width * height, bgVal);
}

// Constructs a single PoolSLAM::Imu sample
inline Imu makeImuSample(double timestamp,
                         const Eigen::Vector3d &accel = Eigen::Vector3d(0.0, 0.0, -9.81),
                         const Eigen::Vector3d &omega = Eigen::Vector3d::Zero(),
                         float tempC = 25.0f) {
    Imu sample{};
    sample.timestamp = timestamp;
    sample.a = accel;
    sample.w = omega;
    sample.t = tempC;
    return sample;
}

// Pushes a chronological sequence of IMU samples spanning [tStart, tEnd] with step dt
inline void pushImuInterval(ImuStream &imuStream, double tStart, double tEnd, double dt = 0.005,
                            const Eigen::Vector3d &accel = Eigen::Vector3d(0.0, 0.0, -9.81),
                            const Eigen::Vector3d &omega = Eigen::Vector3d::Zero()) {
    const size_t steps = static_cast<size_t>(std::ceil((tEnd - tStart) / dt)) + 1;
    for (size_t i = 0; i <= steps; i++) {
        const double t = tStart + static_cast<double>(i) * dt;
        imuStream.pushImu(makeImuSample(t, accel, omega));
    }
}

// Pushes a chronological sequence of Depth samples spanning [tStart, tEnd] with step dt
inline void pushDepthInterval(DepthStream &depthStream, double tStart, double tEnd,
                              double depthZ = 5.0, double dt = 0.025) {
    const size_t steps = static_cast<size_t>(std::ceil((tEnd - tStart) / dt)) + 1;
    for (size_t i = 0; i <= steps; i++) {
        const double t = tStart + static_cast<double>(i) * dt;
        depthStream.pushDepth({t, depthZ});
    }
}

// Pushes bracketing IMU and optional Depth samples followed by a camera frame at frameTime.
// Accepts non-const frameBuffer to match ImageStream::pushFrame(uint8_t*, double).
inline bool pushSynchronizedFrame(ImuStream &imuStream,
                                  DepthStream &depthStream,
                                  ImageStream &imageStream,
                                  double prevTime,
                                  double frameTime,
                                  std::vector<uint8_t> &frameBuffer,
                                  std::optional<double> depthZ = 5.0,
                                  const Eigen::Vector3d &accel = Eigen::Vector3d(0.0, 0.0, -9.81),
                                  const Eigen::Vector3d &omega = Eigen::Vector3d::Zero()) {
    const double startBracket = (prevTime >= 0.0) ? prevTime : std::max(0.0, frameTime - 0.050);
    pushImuInterval(imuStream, startBracket, frameTime + 0.015, 0.005, accel, omega);
    if (depthZ.has_value()) {
        pushDepthInterval(depthStream, startBracket, frameTime + 0.030, *depthZ, 0.020);
    }
    return imageStream.pushFrame(frameBuffer.data(), frameTime);
}

} // namespace test_helpers
} // namespace PoolSLAM