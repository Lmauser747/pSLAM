#pragma once

#include <cstdint>
#include <array>
#include <cstddef>
#include <vector>
#include <optional>

#include <eigen3/Eigen/Dense>

namespace PoolSLAM{

struct Distortion{
    float k1, k2, p1, p2;
};

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
    std::array<int16_t, NUM_PIXELS> xGrad, yGrad; // Gradient in x and y directions 

    uint32_t Mad; // Mean absolute difference between last two patches

    Eigen::Matrix2f HInv{{0.0, 0.0}, {0.0, 0.0}};
};

struct VisualMeasurement{
    uint32_t landmarkId;
    // gtsam::Point2 pt;  
};

struct BackendPacket{
    size_t frameIdx;
    double timestamp;
    bool isKeyframe = false;

    // Vector of visual measurements
    std::vector<VisualMeasurement> visualMreasurements;
    
    // Preintegrated IMU measurements
    // std::unique_ptr<gtsam::PreintegratedImuMeasurements> 

    // Depth in meters
    std::optional<double> depth;
};


}