#pragma once

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <eigen3/Eigen/Dense>

#include "pool_slam/HalTypes.hpp"
#include "pool_slam/System.hpp"
#include "pool_slam/SystemTypes.hpp"
#include "TestFrontendHelpers.hpp"

namespace PoolSLAM {
namespace test_helpers {

// Generates a valid temporary YAML configuration file for SystemConfig tests
inline std::string writeValidSystemConfigFile(
    const std::string &filename = "/tmp/test_system_config.yaml",
    const std::string &vocabPath = "/tmp/test_system_vocab.bin",
    int width = 640,
    int height = 480,
    float fx = 525.40f,
    float fy = 525.85f,
    float cx = 319.50f,
    float cy = 239.80f,
    const std::vector<float> &distortion = {-0.2814f, 0.0912f, 0.0004f, -0.0002f},
    const Eigen::Vector3d &depthLeverArm = Eigen::Vector3d(0.0, 0.0, 0.05)) {
    
    std::ofstream out(filename, std::ios::trunc);
    out << "pipeline:\n";
    out << "  bow_vocabulary_path: \"" << vocabPath << "\"\n\n";

    out << "camera:\n";
    out << "  width: " << width << "\n";
    out << "  height: " << height << "\n";
    out << "  intrinsics:\n";
    out << "    fx: " << fx << "\n";
    out << "    fy: " << fy << "\n";
    out << "    cx: " << cx << "\n";
    out << "    cy: " << cy << "\n";
    out << "  distortion_coeffs: ["
        << distortion[0] << ", " << distortion[1] << ", "
        << distortion[2] << ", " << distortion[3] << "]\n\n";

    out << "extrinsics:\n";
    out << "  T_body_camera:\n";
    out << "    - [ 0.0,  0.0,  1.0,  0.12]\n";
    out << "    - [-1.0,  0.0,  0.0,  0.00]\n";
    out << "    - [ 0.0, -1.0,  0.0, -0.04]\n";
    out << "    - [ 0.0,  0.0,  0.0,  1.00]\n";
    out << "  t_body_depth: ["
        << depthLeverArm.x() << ", "
        << depthLeverArm.y() << ", "
        << depthLeverArm.z() << "]\n";
    out.close();
    return filename;
}

// Writes arbitrary YAML content to a file for negative/boundary tests
inline std::string writeCustomConfigFile(const std::string &filename, const std::string &content) {
    std::ofstream out(filename, std::ios::trunc);
    out << content;
    out.close();
    return filename;
}

// Pushes dense 200 Hz IMU and 20 Hz Depth samples into a System instance across [tStart, tEnd]
inline void pushSystemInertialAndDepthWindow(
    System &sys,
    double tStart,
    double tEnd,
    double depthZ = 5.0,
    const Eigen::Vector3d &accel = Eigen::Vector3d(0.0, 0.0, 9.81),
    const Eigen::Vector3d &omega = Eigen::Vector3d::Zero()) {
    const double imuDt = 0.005;   // 200 Hz
    const double depthDt = 0.050; // 20 Hz

    const size_t imuSteps = static_cast<size_t>(std::ceil((tEnd - tStart) / imuDt)) + 2;
    for (size_t i = 0; i <= imuSteps; i++) {
        const double t = tStart + static_cast<double>(i) * imuDt;
        sys.pushImu(createImuSample(t, accel, omega));
    }

    const size_t depthSteps = static_cast<size_t>(std::ceil((tEnd - tStart) / depthDt)) + 2;
    for (size_t i = 0; i <= depthSteps; i++) {
        const double t = tStart + static_cast<double>(i) * depthDt;
        sys.pushDepth(Depth{t, depthZ});
    }
}

// Polls System::getPose() until the estimated pose reaches or exceeds targetTimestamp
inline bool waitForSystemPoseTimestamp(
    const System &sys,
    double targetTimestamp,
    std::chrono::milliseconds timeout = std::chrono::milliseconds(2500)) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (std::chrono::steady_clock::now() < deadline) {
        Pose p = sys.getPose();
        if (p.timestamp >= targetTimestamp - 1e-4 && p.timestamp > 0.0) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    return sys.getPose().timestamp >= targetTimestamp - 1e-4 && sys.getPose().timestamp > 0.0;
}

} // namespace test_helpers
} // namespace PoolSLAM
