#pragma once

#include <gtest/gtest.h>
#include <cmath>
#include <gtsam/base/numericalDerivative.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/linear/LossFunctions.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/PriorFactor.h>

#include "backend/Backend.hpp"

namespace PoolSLAM {

class TestDepthFactor : public ::testing::Test {
protected:
    static constexpr double DEG2RAD = M_PI / 180.0;

    static gtsam::Rot3 makeRot3(double rollDeg, double pitchDeg, double yawDeg) {
        return gtsam::Rot3::Ypr(yawDeg * DEG2RAD, pitchDeg * DEG2RAD, rollDeg * DEG2RAD);
    }

    static gtsam::SharedNoiseModel makeGaussianNoise(double sigma = 0.05) {
        return gtsam::noiseModel::Isotropic::Sigma(1, sigma);
    }

    static gtsam::SharedNoiseModel makeHuberNoise(double sigma = 0.05, double k = 1.345) {
        return gtsam::noiseModel::Robust::Create(
            gtsam::noiseModel::mEstimator::Huber::Create(k),
            gtsam::noiseModel::Isotropic::Sigma(1, sigma)
        );
    }

    static gtsam::Matrix computeNumericalJacobian(const DepthFactor& factor,
                                                 const gtsam::Pose3& pose,
                                                 double delta = 1e-5) {
        return gtsam::numericalDerivative11<gtsam::Vector, gtsam::Pose3>(
            [&](const gtsam::Pose3& p) { return factor.evaluateError(p); },
            pose,
            delta
        );
    }
};

} // namespace PoolSLAM
