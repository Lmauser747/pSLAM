#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <chrono>
#include <thread>
#include <memory>

#include <Eigen/Core>
#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/linear/LossFunctions.h>

#include "backend/Backend.hpp"
#include "CoreTypes.hpp"
#include "TestFrontendHelpers.hpp"

namespace PoolSLAM {
namespace test_helpers {

// Constructs a default BackendConfig aligned with the standard sensor parameters
inline BackendConfig makeDefaultBackendConfig(
    const gtsam::Pose3 &bodyToCam = gtsam::Pose3(),
    const gtsam::Point3 &depthLeverArm = gtsam::Point3(0.0, 0.0, 0.0)) {
    BackendConfig cfg{};
    cfg.bodyToCameraFrame = bodyToCam;
    cfg.bodyToDepthSensor = depthLeverArm;

    cfg.relinearizeThreshold = 0.01;
    cfg.relinearizeSkip = 1;

    cfg.bkConfig.minBaselineMeters = 0.03;
    cfg.bkConfig.maxCosParallax = 0.99939;
    cfg.bkConfig.minDepthMeters = 0.15;
    cfg.bkConfig.maxDepthMeters = 20.0;
    cfg.bkConfig.maxReprojErrorSq = 2.5e-5;
    cfg.bkConfig.minLdltDiag = 1e-4;
    cfg.bkConfig.visualSigmaNorm = 0.002;
    cfg.bkConfig.huberK = 1.345;
    cfg.bkConfig.maxBufferLength = 15;

    cfg.depthSigma = 0.05;
    cfg.depthHuberK = 1.345;
    return cfg;
}

// Pushes synchronized inertial and depth data with rest acceleration countering gravity
inline void pushBackendInertialAndDepthWindow(
    ImuStream &imuStream,
    DepthStream &depthStream,
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
        imuStream.pushImu(createImuSample(t, accel, omega));
    }

    const size_t depthSteps = static_cast<size_t>(std::ceil((tEnd - tStart) / depthDt)) + 2;
    for (size_t i = 0; i <= depthSteps; i++) {
        const double t = tStart + static_cast<double>(i) * depthDt;
        depthStream.pushDepth({t, depthZ});
    }
}

// Generates a synthetic loop closure packet between a matched keyframe and the current keyframe
inline BackendLfPacket createSyntheticLoopPacket(
    uint32_t currentFrameId,
    uint32_t matchedFrameId,
    double timestamp,
    const gtsam::Pose3 &relativePose,
    uint32_t inlierCount = 35) {
    BackendLfPacket pkt;
    pkt.currentFrameId = currentFrameId;
    pkt.matchedFrameId = matchedFrameId;
    pkt.timestamp = timestamp;
    pkt.relativePose = relativePose;
    pkt.inlierCount = inlierCount;
    pkt.inlierRatio = 0.85f;
    pkt.matchScore = 0.92f;

    auto baseNoise = gtsam::noiseModel::Diagonal::Sigmas(
        (gtsam::Vector(6) << 0.01, 0.01, 0.01, 0.02, 0.02, 0.02).finished());
    pkt.noiseModel = gtsam::noiseModel::Robust::Create(
        gtsam::noiseModel::mEstimator::Huber::Create(1.345), baseNoise);
    return pkt;
}

} // namespace test_helpers
} // namespace PoolSLAM
