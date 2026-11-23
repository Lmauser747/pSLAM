#include "TestDepthFactorHelpers.hpp"

using gtsam::Point3;
using gtsam::Pose3;
using gtsam::Rot3;
using gtsam::symbol_shorthand::X;

namespace PoolSLAM {

// =========================================================================
// 1. Residual Accuracy Under Nominal Geometry
// =========================================================================

TEST_F(TestDepthFactor, ResidualZeroLeverArmLevelPose) {
    const auto model = makeGaussianNoise(0.05);
    const Pose3 pose(Rot3::Identity(), Point3(1.0, 2.0, 5.0));

    // Case 1: Exact measurement z_meas = 5.0 -> residual = 0.0
    DepthFactor factorExact(X(1), 5.0, model, Point3(0.0, 0.0, 0.0));
    gtsam::Vector rExact = factorExact.evaluateError(pose);
    ASSERT_EQ(rExact.size(), 1);
    EXPECT_NEAR(rExact(0), 0.0, 1e-9);

    // Case 2: Shallow measurement z_meas = 4.8 -> residual = +0.2
    DepthFactor factorShallow(X(1), 4.8, model, Point3(0.0, 0.0, 0.0));
    gtsam::Vector rShallow = factorShallow.evaluateError(pose);
    ASSERT_EQ(rShallow.size(), 1);
    EXPECT_NEAR(rShallow(0), 0.2, 1e-9);

    // Case 3: Deep measurement z_meas = 5.3 -> residual = -0.3
    DepthFactor factorDeep(X(1), 5.3, model, Point3(0.0, 0.0, 0.0));
    gtsam::Vector rDeep = factorDeep.evaluateError(pose);
    ASSERT_EQ(rDeep.size(), 1);
    EXPECT_NEAR(rDeep(0), -0.3, 1e-9);
}

TEST_F(TestDepthFactor, ResidualNonZeroLeverArm) {
    const auto model = makeGaussianNoise(0.05);

    // Lever arm: [0.2, 0.0, 0.5]^T, level pose at t = [0.0, 0.0, 5.0]^T
    // Predicted transducer position: [0.2, 0.0, 5.5]^T -> z_pred = 5.5 m
    const Point3 leverArm(0.2, 0.0, 0.5);
    const Pose3 pose(Rot3::Identity(), Point3(0.0, 0.0, 5.0));

    DepthFactor factorMatch(X(1), 5.5, model, leverArm);
    gtsam::Vector rMatch = factorMatch.evaluateError(pose);
    ASSERT_EQ(rMatch.size(), 1);
    EXPECT_NEAR(rMatch(0), 0.0, 1e-9);

    DepthFactor factorMismatch(X(1), 6.0, model, leverArm);
    gtsam::Vector rMismatch = factorMismatch.evaluateError(pose);
    ASSERT_EQ(rMismatch.size(), 1);
    EXPECT_NEAR(rMismatch(0), -0.5, 1e-9);

    // Arbitrary 3D lever arm
    const Point3 leverArm3D(0.5, -0.3, 1.2);
    const Pose3 pose3D(Rot3::Identity(), Point3(10.0, -4.0, 2.0));
    // z_pred = 2.0 + 1.2 = 3.2 m
    DepthFactor factor3D(X(1), 3.2, model, leverArm3D);
    EXPECT_NEAR(factor3D.evaluateError(pose3D)(0), 0.0, 1e-9);
}

TEST_F(TestDepthFactor, ResidualPitchRollCoupling) {
    const auto model = makeGaussianNoise(0.05);
    const Point3 bodyT(0.0, 0.0, 5.0);

    // Pitch vehicle by -90 deg around Y so forward vector [1, 0, 0] rotates downward to [0, 0, 1]
    const Point3 leverArmForward(1.0, 0.0, 0.0);
    const Rot3 pitchDown = Rot3::Ry(-M_PI / 2.0);
    const Point3 rotatedForward = pitchDown.rotate(leverArmForward);
    ASSERT_NEAR(rotatedForward.x(), 0.0, 1e-9);
    ASSERT_NEAR(rotatedForward.y(), 0.0, 1e-9);
    ASSERT_NEAR(rotatedForward.z(), 1.0, 1e-9);

    const Pose3 pitchedPose(pitchDown, bodyT);
    // Transducer world depth = 5.0 + 1.0 = 6.0 m
    DepthFactor factorPitched(X(1), 6.0, model, leverArmForward);
    EXPECT_NEAR(factorPitched.evaluateError(pitchedPose)(0), 0.0, 1e-9);

    DepthFactor factorPitchedOffset(X(1), 5.8, model, leverArmForward);
    EXPECT_NEAR(factorPitchedOffset.evaluateError(pitchedPose)(0), 0.2, 1e-9);

    // Roll vehicle by +90 deg around X so starboard vector [0, 1, 0] rotates downward to [0, 0, 1]
    const Point3 leverArmStarboard(0.0, 1.0, 0.0);
    const Rot3 rollRight = Rot3::Rx(M_PI / 2.0);
    const Point3 rotatedStarboard = rollRight.rotate(leverArmStarboard);
    ASSERT_NEAR(rotatedStarboard.x(), 0.0, 1e-9);
    ASSERT_NEAR(rotatedStarboard.y(), 0.0, 1e-9);
    ASSERT_NEAR(rotatedStarboard.z(), 1.0, 1e-9);

    const Pose3 rolledPose(rollRight, bodyT);
    // Transducer world depth = 5.0 + 1.0 = 6.0 m
    DepthFactor factorRolled(X(1), 6.0, model, leverArmStarboard);
    EXPECT_NEAR(factorRolled.evaluateError(rolledPose)(0), 0.0, 1e-9);
}

// =========================================================================
// 2. Analytical vs Numerical Jacobian Exactness
// =========================================================================

TEST_F(TestDepthFactor, AnalyticalVsNumericalJacobianIdentityPose) {
    const auto model = makeGaussianNoise(0.05);
    const Pose3 pose = Pose3::Identity();
    const DepthFactor factor(X(1), 5.0, model, Point3(0.0, 0.0, 0.0));

    gtsam::Matrix H_ana;
    factor.evaluateError(pose, &H_ana);

    ASSERT_EQ(H_ana.rows(), 1);
    ASSERT_EQ(H_ana.cols(), 6);

    const gtsam::Matrix H_num = computeNumericalJacobian(factor, pose);
    EXPECT_LT((H_ana - H_num).cwiseAbs().maxCoeff(), 1e-7);

    // For identity pose and zero lever arm, H is exactly [0, 0, 0, 0, 0, 1]
    gtsam::Matrix expectedH = (gtsam::Matrix(1, 6) << 0.0, 0.0, 0.0, 0.0, 0.0, 1.0).finished();
    EXPECT_LT((H_ana - expectedH).cwiseAbs().maxCoeff(), 1e-9);
}

TEST_F(TestDepthFactor, AnalyticalVsNumericalJacobianArbitraryTranslation) {
    const auto model = makeGaussianNoise(0.05);
    const Pose3 pose(Rot3::Identity(), Point3(-3.5, 12.0, 8.4));
    const DepthFactor factor(X(1), 8.4, model, Point3(0.0, 0.0, 0.0));

    gtsam::Matrix H_ana;
    factor.evaluateError(pose, &H_ana);

    const gtsam::Matrix H_num = computeNumericalJacobian(factor, pose);
    EXPECT_LT((H_ana - H_num).cwiseAbs().maxCoeff(), 1e-7);

    // Translation block must remain [0, 0, 1]
    EXPECT_NEAR(H_ana(0, 3), 0.0, 1e-9);
    EXPECT_NEAR(H_ana(0, 4), 0.0, 1e-9);
    EXPECT_NEAR(H_ana(0, 5), 1.0, 1e-9);
}

TEST_F(TestDepthFactor, AnalyticalVsNumericalJacobianArbitraryPoseAndLeverArm) {
    const auto model = makeGaussianNoise(0.05);
    const Rot3 R = makeRot3(25.0, -35.0, 80.0);
    const Point3 t(-3.5, 12.0, 8.4);
    const Point3 leverArm(0.4, -0.2, 0.15);
    const Pose3 pose(R, t);

    const DepthFactor factor(X(1), 10.2, model, leverArm);

    gtsam::Matrix H_ana;
    factor.evaluateError(pose, &H_ana);

    ASSERT_EQ(H_ana.rows(), 1);
    ASSERT_EQ(H_ana.cols(), 6);

    const gtsam::Matrix H_num = computeNumericalJacobian(factor, pose);
    EXPECT_LT((H_ana - H_num).cwiseAbs().maxCoeff(), 1e-7);

    // Additional configuration with complex rotation and negative lever arm
    const Rot3 R2 = makeRot3(-60.0, 45.0, -120.0);
    const Point3 t2(7.2, -8.1, 15.3);
    const Point3 leverArm2(-0.8, 0.5, -0.3);
    const Pose3 pose2(R2, t2);

    const DepthFactor factor2(X(1), 14.0, model, leverArm2);
    gtsam::Matrix H_ana2;
    factor2.evaluateError(pose2, &H_ana2);
    const gtsam::Matrix H_num2 = computeNumericalJacobian(factor2, pose2);
    EXPECT_LT((H_ana2 - H_num2).cwiseAbs().maxCoeff(), 1e-7);
}

TEST_F(TestDepthFactor, NullJacobianPointerSafety) {
    const auto model = makeGaussianNoise(0.05);
    const Rot3 R = makeRot3(15.0, -20.0, 40.0);
    const Point3 t(2.0, -3.0, 6.0);
    const Point3 leverArm(0.3, 0.1, -0.2);
    const Pose3 pose(R, t);

    const DepthFactor factor(X(1), 6.5, model, leverArm);

    gtsam::Matrix H_explicit;
    const gtsam::Vector r_explicit = factor.evaluateError(pose, &H_explicit);
    const gtsam::Vector r_null = factor.evaluateError(pose, nullptr);
    const gtsam::Vector r_default = factor.evaluateError(pose);

    EXPECT_NEAR((r_explicit - r_null).norm(), 0.0, 1e-12);
    EXPECT_NEAR((r_explicit - r_default).norm(), 0.0, 1e-12);
}

// =========================================================================
// 3. Noise Model & Loss Function Compatibility
// =========================================================================

TEST_F(TestDepthFactor, StandardGaussianNoiseErrorEvaluation) {
    constexpr double sigma = 0.05;
    const auto model = makeGaussianNoise(sigma);
    const DepthFactor factor(X(1), 5.0, model, Point3(0.0, 0.0, 0.0));

    // Vehicle at depth 5.1 -> unwhitened residual r = 0.1 m = 2 * sigma
    gtsam::Values values;
    values.insert(X(1), Pose3(Rot3::Identity(), Point3(0.0, 0.0, 5.1)));

    // Whitened error should be r / sigma = 0.1 / 0.05 = 2.0
    const gtsam::Vector whitenedErr = factor.whitenedError(values);
    ASSERT_EQ(whitenedErr.size(), 1);
    EXPECT_NEAR(whitenedErr(0), 2.0, 1e-7);

    // Total factor error in GTSAM is 0.5 * whitenedErr^2 = 0.5 * 4.0 = 2.0
    const double error = factor.error(values);
    EXPECT_NEAR(error, 2.0, 1e-7);
}

TEST_F(TestDepthFactor, HuberLossRobustWeighting) {
    constexpr double sigma = 0.05;
    constexpr double delta = 1.345;
    const auto gaussianModel = makeGaussianNoise(sigma);
    const auto huberModel = makeHuberNoise(sigma, delta);

    const DepthFactor factorGaussian(X(1), 5.0, gaussianModel, Point3(0.0, 0.0, 0.0));
    const DepthFactor factorHuber(X(1), 5.0, huberModel, Point3(0.0, 0.0, 0.0));

    // Inlier: r = 0.025 m -> x = 0.025 / 0.05 = 0.5 (< delta)
    // Loss is purely quadratic: 0.5 * x^2 = 0.5 * 0.25 = 0.125
    gtsam::Values inlierValues;
    inlierValues.insert(X(1), Pose3(Rot3::Identity(), Point3(0.0, 0.0, 5.025)));
    EXPECT_NEAR(factorGaussian.error(inlierValues), 0.125, 1e-7);
    EXPECT_NEAR(factorHuber.error(inlierValues), 0.125, 1e-7);

    // Outlier 1: r = 0.5 m -> x = 0.5 / 0.05 = 10.0 (> 3 sigma)
    // Huber loss: 0.5 * delta^2 + delta * (x - delta) = delta * x - 0.5 * delta^2
    // = 1.345 * 10.0 - 0.5 * (1.345)^2 = 13.45 - 0.9045125 = 12.5454875
    gtsam::Values outlier1Values;
    outlier1Values.insert(X(1), Pose3(Rot3::Identity(), Point3(0.0, 0.0, 5.5)));

    const double huberErr1 = factorHuber.error(outlier1Values);
    const double gaussErr1 = factorGaussian.error(outlier1Values);
    const double expectedHuberLoss1 = delta * 10.0 - 0.5 * delta * delta;

    EXPECT_NEAR(gaussErr1, 0.5 * 10.0 * 10.0, 1e-7); // 50.0
    EXPECT_NEAR(huberErr1, expectedHuberLoss1, 1e-5);
    EXPECT_LT(huberErr1, gaussErr1);

    // Outlier 2: r = 1.0 m -> x = 1.0 / 0.05 = 20.0
    gtsam::Values outlier2Values;
    outlier2Values.insert(X(1), Pose3(Rot3::Identity(), Point3(0.0, 0.0, 6.0)));

    const double huberErr2 = factorHuber.error(outlier2Values);
    // Linear growth verification: delta * (x2 - x1) = 1.345 * 10.0 = 13.45
    EXPECT_NEAR(huberErr2 - huberErr1, delta * 10.0, 1e-5);
}

// =========================================================================
// 4. Factor Graph Optimization Integration
// =========================================================================

TEST_F(TestDepthFactor, FactorGraphOptimizationConvergence) {
    constexpr double depthSigma = 0.05;
    const auto depthNoise = makeGaussianNoise(depthSigma);

    constexpr double measuredDepth = 8.5;

    // Prior pose at (x=1.5, y=2.5) with yaw=0.3 rad (level vehicle: roll=0, pitch=0)
    const Rot3 priorRot = Rot3::Yaw(0.3);
    const Point3 priorPoint(1.5, 2.5, measuredDepth);
    const Pose3 priorPose(priorRot, priorPoint);

    // Prior noise: tightly constrain (roll, pitch, yaw, x, y) at 1e-4, unconstrain z at 100.0
    gtsam::Vector6 priorSigmas;
    priorSigmas << 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 100.0;
    const auto priorNoise = gtsam::noiseModel::Diagonal::Sigmas(priorSigmas);

    gtsam::NonlinearFactorGraph graph;
    graph.addPrior(X(1), priorPose, priorNoise);

    // Depth factor: measurement is 8.5 m
    graph.emplace_shared<DepthFactor>(X(1), measuredDepth, depthNoise, Point3(0.0, 0.0, 0.0));

    // Initial value has depth off by 7.5 m (z = 1.0 m)
    gtsam::Values initial;
    initial.insert(X(1), Pose3(priorRot, Point3(1.5, 2.5, 1.0)));

    gtsam::LevenbergMarquardtOptimizer optimizer(graph, initial);
    const gtsam::Values result = optimizer.optimize();

    const Pose3 estimatedPose = result.at<Pose3>(X(1));

    // Verify depth converged to measured depth
    EXPECT_NEAR(estimatedPose.z(), measuredDepth, 1e-3);

    // Verify horizontal position and orientation remained at prior values
    EXPECT_NEAR(estimatedPose.x(), 1.5, 1e-3);
    EXPECT_NEAR(estimatedPose.y(), 2.5, 1e-3);
    EXPECT_LT((estimatedPose.rotation().matrix() - priorRot.matrix()).cwiseAbs().maxCoeff(), 1e-3);

    // Covariance recovery verification
    const gtsam::Marginals marginals(graph, result);
    const gtsam::Matrix6 cov = marginals.marginalCovariance(X(1));

    // Tangent index 5 is Z translation: marginal variance ~= (1/100^2 + 1/0.05^2)^-1 ~= 0.05^2 = 0.0025
    constexpr double expectedZVariance = depthSigma * depthSigma;
    EXPECT_NEAR(cov(5, 5), expectedZVariance, 1e-5);

    // Tangent indices 3 and 4 (X and Y translations) remain at prior variance 1e-8
    EXPECT_NEAR(cov(3, 3), 1e-8, 1e-9);
    EXPECT_NEAR(cov(4, 4), 1e-8, 1e-9);
}

TEST_F(TestDepthFactor, FactorGraphOptimizationWithLeverArm) {
    constexpr double depthSigma = 0.05;
    const auto depthNoise = makeGaussianNoise(depthSigma);

    const Point3 leverArm(0.2, -0.1, 0.6);
    constexpr double measuredDepth = 7.6; // World depth of transducer

    // Expected body depth: t_z = measuredDepth - leverArm.z() = 7.6 - 0.6 = 7.0 m
    const Rot3 identityRot = Rot3::Identity();
    const Point3 priorPoint(3.0, -1.0, 0.0);
    const Pose3 priorPose(identityRot, priorPoint);

    gtsam::Vector6 priorSigmas;
    priorSigmas << 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 100.0;
    const auto priorNoise = gtsam::noiseModel::Diagonal::Sigmas(priorSigmas);

    gtsam::NonlinearFactorGraph graph;
    graph.addPrior(X(1), priorPose, priorNoise);
    graph.emplace_shared<DepthFactor>(X(1), measuredDepth, depthNoise, leverArm);

    gtsam::Values initial;
    initial.insert(X(1), Pose3(identityRot, Point3(3.0, -1.0, 2.0)));

    gtsam::LevenbergMarquardtOptimizer optimizer(graph, initial);
    const gtsam::Values result = optimizer.optimize();

    const Pose3 estimatedPose = result.at<Pose3>(X(1));
    EXPECT_NEAR(estimatedPose.z(), 7.0, 1e-3);
    EXPECT_NEAR(estimatedPose.x(), 3.0, 1e-3);
    EXPECT_NEAR(estimatedPose.y(), -1.0, 1e-3);
}

} // namespace PoolSLAM
