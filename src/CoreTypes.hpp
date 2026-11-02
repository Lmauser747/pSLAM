#pragma once

#include <cstdint>
#include <array>
#include <gtsam/linear/NoiseModel.h>
#include <vector>
#include <optional>
#include <memory>

#include <eigen3/Eigen/Dense>
#include <gtsam/geometry/Point2.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/linear/NoiseModel.h>


namespace PoolSLAM{

struct Distortion{
    float k1, k2, p1, p2;
};

// Data associated with each feature for optical flow tracking
template<int Radius>
struct TrackedFeature{
    uint32_t id; // Feature id for tracking
    int age = 0; // Feature age in frames

    // Using homogenous coordinates
    Eigen::Vector2f ptCurr; // Sub-pixel position of feature
    Eigen::Vector2f ptPrev; // Sub-pixel position of feature in previous frame


    static constexpr int PATCH_SIZE = 2 * Radius + 1;
    static constexpr int NUM_PIXELS = PATCH_SIZE * PATCH_SIZE;

    std::array<uint8_t, NUM_PIXELS> patch; // Patch of pixels for IC-LK
    std::array<int, NUM_PIXELS> xGrad, yGrad; // Gradient in x and y directions 

    uint32_t Mad = 0.0; // Mean absolute difference between last two patches

    Eigen::Matrix2f HInv{{0.0, 0.0}, {0.0, 0.0}};
};

struct VisualMeasurement{
    uint32_t landmarkId;
    gtsam::Point2 pt;  
};

// Data passed from the Hf frontend to the backend
struct BackendHfPacket{
    uint64_t frameIdx = 0;
    double timestamp = 0.0;
    bool isKeyframe = false;

    // Vector of visual measurements
    std::vector<VisualMeasurement> visualMeasurements;
    
    // Preintegrated IMU measurements
    std::unique_ptr<gtsam::PreintegratedImuMeasurements> imuDelta;

    // Depth in meters
    std::optional<double> depth;
};

// Data passed from the Lf frontend to the backend
struct BackendLfPacket{
    // Stores the frame index of the current and matched keyframe
    uint64_t currentFrameIdx = 0;
    uint64_t matchedFrameIdx = 0;

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


}