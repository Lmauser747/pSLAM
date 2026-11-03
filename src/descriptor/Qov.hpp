#pragma once

#include "frontend/HfTracker.hpp"
#include <Eigen/src/Core/Matrix.h>
#include <Eigen/src/Geometry/Rotation2D.h>
#include <bitset>
#include <cstdint>
#include <vector>
#include <cstddef>
#include <array>
#include <cmath>

#include <eigen3/Eigen/Dense>

#include <simd/Utils.hpp>
#include <descriptor/QovBaseOffsets.hpp>

// QOV(Quick ORB Variant) descriptor implementations

namespace PoolSLAM{

namespace descriptor{

constexpr size_t QOV_BITS = 256;

struct QovDescriptor{
    std::bitset<QOV_BITS> bits = 0;
};

struct QovFeature{
    QovDescriptor des;
    Eigen::Vector2i pt;
};

// QOV will compare I(pt2) > I(pt1)
struct QovCompPair{
    ptrdiff_t pt1, pt2;
};

// Store Qov feature data accosicated with a given frame
class FrameData{
public:
    FrameData(size_t maxNumFeatures){
        features.reserve(maxNumFeatures);
    }

    std::vector<QovFeature> features;
    uint64_t id = 0;
    double timestamp = 0.0;

    FORCE_INLINE void addDes(const QovDescriptor &des, Eigen::Vector2i pt){
        features.emplace_back(des, pt);
    }

    FORCE_INLINE void addDes(const QovFeature &feat){
        features.emplace_back(feat);
    }
};

// Generated set of ORB pairs to compare between 
class RoatedQovPairs{
private:
    static constexpr double PI = 3.1415926535;
    static constexpr int ROTATIONS = 32; // Number of rotated offsets to store
    static constexpr double ANGLE_STEP_DEG = 360.0 / ROTATIONS; // Step between each offset
    static constexpr double ANGLE_STEP_RAD = 2.0 * PI / ROTATIONS;
    std::array<std::array<QovCompPair, QOV_BITS>, ROTATIONS> offsets;

    double toRadians(double degrees){
        return (degrees / 180.0) * PI;
    }

    void getRotatedOffset(double angle, ptrdiff_t stride){
        double normalizedAngle = std::fmod(angle, 360.0);
        auto idx = static_cast<size_t>(std::round(normalizedAngle / ANGLE_STEP_DEG));
        for(size_t i = 0; i < QOV_BITS; i++){
            Eigen::Vector2d pt1 = {QOV_BASE_OFFSETS[i + 0], QOV_BASE_OFFSETS[i + 1]};
            Eigen::Vector2d pt2 = {QOV_BASE_OFFSETS[i + 2], QOV_BASE_OFFSETS[i + 3]};

            Eigen::Rotation2Dd rotation(toRadians(normalizedAngle));
            Eigen::Matrix2d R = rotation.toRotationMatrix();

            pt1 = R * pt1;
            pt2 = R * pt2;

            offsets[idx][i].pt1 = static_cast<ptrdiff_t>(std::round(pt1.x())) + stride * static_cast<ptrdiff_t>(std::round(pt1.y()));
            offsets[idx][i].pt2 = static_cast<ptrdiff_t>(std::round(pt2.x())) + stride * static_cast<ptrdiff_t>(std::round(pt2.y()));
        }

    }

public:
    RoatedQovPairs(size_t w){
        for(int i = 0; i < ROTATIONS; i++){
            getRotatedOffset(static_cast<float>(i) * ANGLE_STEP_DEG, static_cast<ptrdiff_t>(w));
        }
    }


    // Get the offset array at a given angle in degrees
    FORCE_INLINE const std::array<QovCompPair, QOV_BITS> offsetAtAngleDeg(double angle) const{
        constexpr double INV = 1 / ANGLE_STEP_DEG;
        return offsets[static_cast<size_t>(std::round(angle * INV))];
    }

    // Get the offset array at a given angle in radius
    FORCE_INLINE const std::array<QovCompPair, QOV_BITS> offsetAtAngleRad(double angle) const{
        constexpr double INV = 1 / ANGLE_STEP_RAD;
        return offsets[static_cast<size_t>(std::round(angle * INV))];
    }

};

FORCE_INLINE QovDescriptor generateDescriptor(const uint8_t *pt, const std::array<QovCompPair, QOV_BITS> &offsets){
    QovDescriptor des;
    for(size_t i = 0; i < QOV_BITS; i++){
        des.bits[i] = *(pt + offsets[i].pt1) < *(pt + offsets[i].pt2);
    }
    return des;
}

}

}