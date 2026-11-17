#pragma once

#include <atomic>
#include <cstddef>
#include <thread>
#include <array>
#include <cstdint>
#include <deque>
#include <vector>

#include <Eigen/Core>

#include <gtsam/geometry/Point2.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/navigation/ImuBias.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/NavState.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/NoiseModelFactorN.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/PriorFactor.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/base/Vector.h>
#include <gtsam/base/types.h>

#include "CoreTypes.hpp"
#include "backend/Bookkeeper.hpp"
#include "pool_slam/SystemTypes.hpp"
#include "frontend/Frontend.hpp"
#include "frontend/LfTracker.hpp"

namespace PoolSLAM{

struct BackendConfig{
    BookkeeperConfig bkConfig;

    // Sensor extrinsics relative to the ROV body/IMU frame
    gtsam::Pose3 bodyToCameraFrame; // T_BC: transforms points from camera frame to body frame
    gtsam::Point3 bodyToDepthSensor = gtsam::Point3(0.0, 0.0, 0.0); // Depth sensor lever arm in body frame

    // iSAM2 incremental solver parameters
    double relinearizeThreshold = 0.01;
    int relinearizeSkip = 1;

    // Initial state prior standard deviations (Frame 0 bootstrapping)
    Eigen::Matrix<double, 6, 1> initPoseSigma =
        (Eigen::Matrix<double, 6, 1>() << 1e-2, 1e-2, 1e-2, 1e-3, 1e-3, 1e-3).finished(); // [rot (rad), pos (m)]
    double initVelSigma = 1e-3;       // m/s
    double initAccelBiasSigma = 1e-2; // m/s^2
    double initGyroBiasSigma = 1e-3;  // rad/s

    // Continuous-time IMU bias random-walk noise densities
    double accelBiasRwSigma = 1e-4;   // m / (s^2 * sqrt(s))
    double gyroBiasRwSigma = 1e-5;    // rad / (s * sqrt(s))

    // 1D Depth sensor noise and robust M-estimator threshold
    double depthSigma = 0.05;         // m
    double depthHuberK = 1.345;
};

class DepthFactor : public gtsam::NoiseModelFactorN<gtsam::Pose3>{
    double measuredDepth;
    gtsam::Point3 bodyToSensor;

public:
    DepthFactor(gtsam::Key poseKey, double depthIn, const gtsam::SharedNoiseModel &model,
                const gtsam::Point3 &leverArm = gtsam::Point3(0.0, 0.0, 0.0));

    ~DepthFactor() override = default;

    gtsam::Vector evaluateError(
        const gtsam::Pose3 &pose,
        gtsam::OptionalMatrixType H = nullptr) const override;
};

class Backend{
    friend class BackendTest;

    struct PendingKfRecord{
        uint32_t frameId = 0;
        std::vector<uint32_t> featureIds;
        size_t lastLandmarkCount = 0;
    };

    Frontend &fe; // Reference to frontend
    LfTracker &lf; // Reference to low-frequency tracker

    BackendConfig config;

    gtsam::ISAM2 isam;
    gtsam::NonlinearFactorGraph stagingGraph; 
    gtsam::Values stagingValues;
    gtsam::Values isamEstimate;

    LandmarkBookkeeper bk; // Bookkeeper for triangulating landmarks

    // Pre-allocated noise models for priors and depth factors
    gtsam::SharedNoiseModel initPoseNoise;
    gtsam::SharedNoiseModel initVelNoise;
    gtsam::SharedNoiseModel initBiasNoise;
    gtsam::SharedNoiseModel depthNoise;

    // Sequential state tracking across frames (k-1 -> k)
    bool isInitialized = false;
    uint32_t prevFrameId = 0;
    double prevTimestamp = 0.0;
    uint32_t lastCommittedFrameId = 0;
    double lastCommittedTimestamp = 0.0;
    gtsam::NavState prevNavState;
    gtsam::imuBias::ConstantBias prevBias;

    // Tracks recent keyframes so LfTracker receives 3D landmarks once triangulated
    std::deque<PendingKfRecord> trackedKeyframes;
    std::vector<uint32_t> allKeyframeIds;

    std::atomic<uint8_t> activePose{0}; // Index of active pose
    std::array<Pose, 2> poseBuffer; // Buffer to hold active pose

    std::atomic<bool> runBackend{false}; // Variable to run backend
    std::thread backendThread; // Thread to run backend

    // Backend packet processing task and helpers
    void backendTask();
    void initializeSystem(const BackendHfPacket &pkt);
    void processHfPacket(const BackendHfPacket &pkt);
    void processLfPacket(const BackendLfPacket &pkt);
    void recordKeyframeFeatures(const BackendHfPacket &pkt);
    void syncKeyframesToLfTracker(uint32_t currentFrameId, bool forceAll);
    void optimizeAndPublish(uint32_t frameId, double timestamp, bool loopClosed);

public:
    Backend(Frontend &frontend, LfTracker &lfTracker, const BackendConfig &cfg);
    ~Backend();

    // Gets the current pose estimate
    Pose getCurrentPose() const;

    // Start function
    void start();

    // End function
    void end();
};

} // namespace PoolSLAM