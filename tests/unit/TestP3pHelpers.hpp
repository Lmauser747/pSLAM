#pragma once

#include <gtest/gtest.h>
#include <array>
#include <optional>
#include <cmath>
#include <eigen3/Eigen/Dense>

#include "frontend/LfTracker.hpp"

namespace PoolSLAM {
namespace geometry {

struct DiagonalDecomp {
    double sigma1;
    double sigma2;
    Eigen::Vector3d e1;
    Eigen::Vector3d e2;
};

// Internal helpers defined in P3p.cpp
double solveLambdaTwistCubic(double a, double b, double c, double d);
DiagonalDecomp diagonalizeD0(const Eigen::Matrix3d &D);

} // namespace geometry

class TestP3p : public ::testing::Test {
protected:
    static Eigen::Matrix3d makeRotation(double rollDeg, double pitchDeg, double yawDeg) {
        constexpr double DEG2RAD = 3.14159265358979323846 / 180.0;
        Eigen::AngleAxisd roll(rollDeg * DEG2RAD, Eigen::Vector3d::UnitX());
        Eigen::AngleAxisd pitch(pitchDeg * DEG2RAD, Eigen::Vector3d::UnitY());
        Eigen::AngleAxisd yaw(yawDeg * DEG2RAD, Eigen::Vector3d::UnitZ());
        return (yaw * pitch * roll).toRotationMatrix();
    }

    // Projects 3D camera points to calibrated bearing rays [u, v, 1]^T
    static Eigen::Vector3d projectBearing(const Eigen::Vector3d& pCam) {
        return Eigen::Vector3d(pCam.x() / pCam.z(), pCam.y() / pCam.z(), 1.0);
    }

    // Computes reprojection error of 3D point under [R | t] against bearing ray
    static double computeReprojError(const Eigen::Matrix<double, 3, 4>& T,
                                    const Eigen::Vector3d& xWorld,
                                    const Eigen::Vector3d& yBearing) {
        Eigen::Vector3d pCam = T.block<3, 3>(0, 0) * xWorld + T.col(3);
        Eigen::Vector2d proj(pCam.x() / pCam.z(), pCam.y() / pCam.z());
        Eigen::Vector2d obs(yBearing.x() / yBearing.z(), yBearing.y() / yBearing.z());
        return (proj - obs).norm();
    }
};

} // namespace PoolSLAM
