#pragma once

#include <atomic>
#include <cstddef>
#include <memory>
#include <optional>
#include <thread>
#include <array>
#include <cstdint>
#include <deque>
#include <unordered_map>
#include <vector>

#include <Eigen/Core>

#include <gtsam/base/Vector.h>
#include <gtsam/base/types.h>
#include <gtsam/geometry/Cal3_S2.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Factor.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/navigation/ImuBias.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/NavState.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/NoiseModelFactorN.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/SmartProjectionPoseFactor.h>

#include "CoreTypes.hpp"
#include "pool_slam/SystemTypes.hpp"
#include "frontend/Frontend.hpp"
#include "frontend/LfTracker.hpp"

namespace PoolSLAM{

using SmartFactor = gtsam::SmartProjectionPoseFactor<gtsam::Cal3_S2>;

struct BackendConfig{

    // Sensor extrinsics relative to the ROV body/IMU frame
    gtsam::Pose3 bodyToCameraFrame; // T_BC: transforms points from camera frame to body frame
    gtsam::Point3 bodyToDepthSensor = gtsam::Point3(0.0, 0.0, 0.0); // Depth sensor lever arm in body frame
    Eigen::Matrix3d K;
    std::optional<gtsam::Pose3> initialPose = std::nullopt; // Optional user/dataset initial pose anchor

    // iSAM2 incremental solver parameters
    double relinearizeThreshold = 0.01;
    int relinearizeSkip = 1;

    // Initial state prior standard deviations (Frame 0 bootstrapping)
    Eigen::Matrix<double, 6, 1> initPoseSigma =
        (Eigen::Matrix<double, 6, 1>() << 1e-2, 1e-2, 1e-2, 1e-3, 1e-3, 1e-3).finished(); // [rot (rad), pos (m)]
    double initVelSigma = 1e-3;
    double initAccelBiasSigma = 0.1;
    double initGyroBiasSigma = 0.05;

    // Continuous-time IMU bias random-walk noise densities
    double accelBiasRwSigma = 1e-4;
    double gyroBiasRwSigma = 1e-5;

    // 1D Depth sensor noise and robust M-estimator threshold
    double depthSigma = 0.05; 
    double depthHuberK = 1.345;

    size_t keyframeWindowSize = 20; // Number of keyframes to keep in trackedKeyframes
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

struct TrackedKeyframe{
    uint32_t frameId;
    std::vector<uint32_t> featureIds;
    size_t lastLandmarkCount = 0;
};

class Backend{
    friend class BackendTest;

    using PendingKfRecord = TrackedKeyframe;

    Frontend &fe; // Reference to frontend
    LfTracker &lf; // Reference to low-frequency tracker

    BackendConfig config;

    gtsam::ISAM2 isam;
    gtsam::NonlinearFactorGraph stagingGraph; 
    gtsam::Values stagingValues;
    std::vector<uint32_t> stagedFeatureIds;
    gtsam::Values isamEstimate;

    // Pre-allocated noise models for priors and depth factors
    gtsam::SharedNoiseModel initPoseNoise;
    gtsam::SharedNoiseModel initVelNoise;
    gtsam::SharedNoiseModel visualNoise;
    gtsam::SharedNoiseModel initBiasNoise;
    gtsam::SharedNoiseModel depthNoise;

    // Camera intrinsics
    const std::shared_ptr<gtsam::Cal3_S2> Kptr;

    // Sequential state tracking across frames (k-1 -> k)
    bool isInitialized = false;
    uint32_t prevFrameId = 0;
    double prevTimestamp = 0.0;
    uint32_t lastCommittedFrameId = 0;
    double lastCommittedTimestamp = 0.0;
    gtsam::NavState prevNavState;
    gtsam::imuBias::ConstantBias prevBias;

    // Smart factor storage
    std::unordered_map<uint32_t, std::shared_ptr<SmartFactor>> smartFactors;
    std::unordered_map<uint32_t, gtsam::FactorIndex> smartFactorSlots;

    // Queue of recent keyframes and their associated features
    std::deque<TrackedKeyframe> trackedKeyframes;

    std::atomic<uint8_t> activePose{0}; // Index of active pose
    std::array<Pose, 2> poseBuffer; // Buffer to hold active pose

    std::atomic<bool> runBackend{false}; // Variable to run backend
    std::thread backendThread; // Thread to run backend

    // Main backend processing task
    void backendTask();

    // Use packet to initialize the system
    void initializeSystem(const BackendHfPacket &pkt);

    // Process keyframe data coming from HfTracker
    void processKeyframe(const BackendHfPacket &pkt);

    // Process lf packets for loop closure
    void processLfPacket(const BackendLfPacket &pkt);

    // Sync 3d landmarks to the LfTracker
    void syncLfTracker(bool forceAll);

    // Published pose to the buffer
    void publishPose(const gtsam::Pose3 &pose, const gtsam::Vector3 &velocity, double timestamp);

    // Solve the factor graph
    bool solveGraph(uint32_t currentFrameId, const gtsam::FactorIndices &removeIndices = {});

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