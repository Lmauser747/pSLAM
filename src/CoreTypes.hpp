#pragma once

#include <cstdint>
#include <array>
#include <vector>
#include <optional>
#include <memory>

#include <eigen3/Eigen/Dense>
#include <gtsam/geometry/Point2.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/base/Vector.h>

#include "pool_slam/HalTypes.hpp"

namespace PoolSLAM{

struct Distortion{
    float k1, k2, p1, p2;
};

// Data associated with each feature for optical flow tracking
template<int Radius>
struct TrackedFeature{
    uint32_t id; // Feature id for tracking
    int age = 0; // Feature age in frames

    Eigen::Vector2f ptCurr = Eigen::Vector2f::Zero(); // Disorted position of the feature in pixel coordinates
    Eigen::Vector2f ptPrev = Eigen::Vector2f::Zero(); // Disorted position of the feature in pixel coordinates in previous frame
    Eigen::Vector2f ptUndis = Eigen::Vector2f::Zero(); // Undistorted position of the points in normalzied camera coordinates
    Eigen::Vector2f ptKfUndis = Eigen::Vector2f::Zero(); // Undistorted normalized position at the last keyframe


    static constexpr int PATCH_SIZE = 2 * Radius + 1;
    static constexpr int NUM_PIXELS = PATCH_SIZE * PATCH_SIZE;

    std::array<uint8_t, NUM_PIXELS> patch; // Patch of pixels for IC-LK
    std::array<int, NUM_PIXELS> xGrad, yGrad; // Gradient in x and y directions 

    Eigen::Matrix2f HInv{{0.0, 0.0}, {0.0, 0.0}};
};

struct VisualFeatureObs{
    uint32_t featureId;
    gtsam::Point2 pt = Eigen::Vector2d::Zero(); // Normalized camera coordinates <x/, x/z> 
    bool isNewTrack = false;

    VisualFeatureObs(uint32_t id, const gtsam::Point2 &p, bool isNew = false) : featureId(id), pt(p), isNewTrack(isNew){}
};

// Data passed from the Frontend to the backend
struct BackendHfPacket{
    uint32_t frameId = 0;
    double timestamp = 0.0;
    bool isKeyframe = false; // If false only IMU data is used, true triggers a full factor graph solve

    std::vector<VisualFeatureObs> visualFeatures; // Visual Features
    
    // Preintegrated IMU measurements
    std::unique_ptr<gtsam::PreintegratedImuMeasurements> imuDelta;

    // Raw IMU samples spanning this frame interval (for backend re-preintegration across merged non-keyframes)
    std::vector<Imu> rawImu;

    // Depth in meters
    std::optional<double> depth;

    // Initial acceleration reading for stationary gravity leveling
    std::optional<gtsam::Vector3> initialAccel;
    // Initial angular velocity reading for stationary gyro bias initialization
    std::optional<gtsam::Vector3> initialGyro;
};

// Data passed from the Lf frontend to the backend
struct BackendLfPacket{
    // Stores the frame index of the current and matched keyframe
    uint32_t currentFrameId = 0;
    uint32_t matchedFrameId = 0;

    double timestamp = 0.0;

    // Pose transformation between matched keyframe and current keyframe
    gtsam::Pose3 relativePose;

    // Noise model of keyframe matching
    // Uses a normal Gaussian wrapped in a Huber loss function to reduce the affect of outliers
    gtsam::SharedNoiseModel noiseModel;

    // Metrics and telemtry from the low freqneucy tracker
    uint32_t inlierCount = 0; // Number of inlier landmarks
    float inlierRatio = 0.0f; // Ratio of inlier landmarks to total landmarks
    float matchScore = 0.0f; // Similarlity score between frames
};

struct Landmark3d{
    Eigen::Vector3d pt3d = Eigen::Vector3d::Zero();
    uint32_t featureId = UINT32_MAX;
};

struct BackendKeyframeUpdate{
    uint32_t id = 0; // Id of keyframe
    gtsam::Pose3 pose; // Pose of keyframe
    std::vector<Landmark3d> landmarks; // Positions of the landmarks in 3d camera coordinates
};


}
