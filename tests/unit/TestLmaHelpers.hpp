#pragma once

#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <eigen3/Eigen/Dense>

#include "frontend/LfTracker.hpp"

namespace PoolSLAM {

class TestLma : public ::testing::Test {
protected:
    LfTrackerConfig config;
    Eigen::Matrix3d K_cam;

    void SetUp() override {
        config.w = 640;
        config.h = 480;
        config.numBuckets = 50;

        K_cam << 400.0, 0.0, 320.0,
                 0.0, 400.0, 240.0,
                 0.0, 0.0, 1.0;
    }

    LfTracker createTracker() {
        return LfTracker(config, "");
    }

    // Lie Algebra se(3) Exponential Map matching specification in tests/Lma.md Section 3.1
    static Eigen::Matrix4d expSE3(const Eigen::Matrix<double, 6, 1>& xi) {
        Eigen::Vector3d rho = xi.head<3>();
        Eigen::Vector3d phi = xi.tail<3>();
        double theta = phi.norm();

        Eigen::Matrix4d T = Eigen::Matrix4d::Identity();

        if (theta <= 1e-8) {
            T.block<3, 1>(0, 3) = rho;
            return T;
        }

        Eigen::Vector3d a = phi / theta;
        Eigen::Matrix3d K;
        K <<  0.0,  -a.z(),  a.y(),
             a.z(),   0.0,  -a.x(),
            -a.y(),  a.x(),   0.0;

        Eigen::Matrix3d R = Eigen::Matrix3d::Identity() +
                            std::sin(theta) * K +
                            (1.0 - std::cos(theta)) * K * K;

        Eigen::Matrix3d V = Eigen::Matrix3d::Identity() +
                            ((1.0 - std::cos(theta)) / theta) * K +
                            ((theta - std::sin(theta)) / theta) * K * K;

        T.block<3, 3>(0, 0) = R;
        T.block<3, 1>(0, 3) = V * rho;
        return T;
    }

    // Helper to populate synthetic landmarks and observed bearings in LfTracker
    static void setupCorrespondences(LfTracker& tracker,
                                    const std::vector<Eigen::Vector3d>& landmarks,
                                    const Eigen::Matrix<double, 3, 4>& T_cam_world,
                                    double noisePixels = 0.0,
                                    const Eigen::Matrix3d& K = Eigen::Matrix3d::Identity()) {
        size_t n = landmarks.size();
        if (tracker.matching.matchLandmarks.cols() < static_cast<int>(n)) {
            tracker.matching.matchLandmarks.resize(3, static_cast<int>(n));
            tracker.matching.curFeatures.resize(3, static_cast<int>(n));
        }

        Eigen::Matrix3d R = T_cam_world.block<3, 3>(0, 0);
        Eigen::Vector3d t = T_cam_world.col(3);

        for (size_t i = 0; i < n; ++i) {
            tracker.matching.matchLandmarks.col(static_cast<int>(i)) = landmarks[i];

            Eigen::Vector3d pCam = R * landmarks[i] + t;
            double normX = pCam.x() / pCam.z();
            double normY = pCam.y() / pCam.z();

            if (noisePixels > 0.0 && K(0, 0) > 0.0) {
                normX += (noisePixels / K(0, 0));
                normY += (noisePixels / K(1, 1));
            }

            tracker.matching.curFeatures.col(static_cast<int>(i)) = Eigen::Vector3d(normX, normY, 1.0);
        }
    }

    // Accessor for refinePoseLM on LfTracker
    static bool callRefinePoseLM(LfTracker& tracker,
                                size_t numInliers,
                                const Eigen::Matrix3d& K,
                                Eigen::Matrix<double, 3, 4>& Tcand,
                                Eigen::Matrix<double, 6, 6>& outCovariance) {
        return tracker.refinePoseLM(numInliers, K, Tcand, outCovariance);
    }

    // Accessor for private matching struct on LfTracker
    static LfMatching& getMatching(LfTracker& tracker) {
        return tracker.matching;
    }

    static const LfMatching& getMatching(const LfTracker& tracker) {
        return tracker.matching;
    }

    // Computes total Huber cost of a pose given landmarks and features
    static double computeHuberCost(const LfTracker& tracker,
                                  size_t numInliers,
                                  const Eigen::Matrix3d& K,
                                  const Eigen::Matrix<double, 3, 4>& T,
                                  double delta = 2.0) {
        double cost = 0.0;
        Eigen::Matrix3d R = T.block<3, 3>(0, 0);
        Eigen::Vector3d t = T.col(3);
        double fx = K(0, 0);
        double fy = K(1, 1);

        for (size_t i = 0; i < numInliers; ++i) {
            Eigen::Vector3d X = tracker.matching.matchLandmarks.col(static_cast<int>(i));
            Eigen::Vector3d pCam = R * X + t;
            if (pCam.z() <= 1e-4) {
                return 1e12;
            }

            double rx = fx * (pCam.x() / pCam.z() - tracker.matching.curFeatures(0, static_cast<int>(i)));
            double ry = fy * (pCam.y() / pCam.z() - tracker.matching.curFeatures(1, static_cast<int>(i)));
            double e = std::hypot(rx, ry);

            if (e <= delta) {
                cost += 0.5 * e * e;
            } else {
                cost += delta * (e - 0.5 * delta);
            }
        }
        return cost;
    }
};

} // namespace PoolSLAM
