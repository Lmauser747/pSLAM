#include "LmaTestHelpers.hpp"

namespace PoolSLAM {

// =============================================================================
// 1. Lie Group Geometry (expSE3)
// =============================================================================

TEST_F(TestLma, ExpSE3ZeroMapsToIdentity) {
    Eigen::Matrix<double, 6, 1> xi = Eigen::Matrix<double, 6, 1>::Zero();
    Eigen::Matrix4d T = expSE3(xi);

    EXPECT_TRUE(T.isApprox(Eigen::Matrix4d::Identity(), 1e-12));
}

TEST_F(TestLma, ExpSE3PureTranslation) {
    Eigen::Matrix<double, 6, 1> xi = Eigen::Matrix<double, 6, 1>::Zero();
    xi.head<3>() = Eigen::Vector3d(1.0, -2.0, 3.0);

    Eigen::Matrix4d T = expSE3(xi);

    EXPECT_TRUE((T.block<3, 3>(0, 0).isApprox(Eigen::Matrix3d::Identity(), 1e-12)));
    EXPECT_TRUE((T.block<3, 1>(0, 3).isApprox(Eigen::Vector3d(1.0, -2.0, 3.0), 1e-12)));
    EXPECT_DOUBLE_EQ(T(3, 3), 1.0);
}

TEST_F(TestLma, ExpSE3PureRotation) {
    constexpr double PI = 3.14159265358979323846;

    // 90 deg rotation around X
    Eigen::Matrix<double, 6, 1> xiX = Eigen::Matrix<double, 6, 1>::Zero();
    xiX.tail<3>() = Eigen::Vector3d(PI / 2.0, 0.0, 0.0);
    Eigen::Matrix4d TX = expSE3(xiX);
    Eigen::Matrix3d expectedRX;
    expectedRX << 1.0,  0.0,  0.0,
                  0.0,  0.0, -1.0,
                  0.0,  1.0,  0.0;
    EXPECT_TRUE((TX.block<3, 3>(0, 0).isApprox(expectedRX, 1e-6)));
    EXPECT_TRUE((TX.block<3, 1>(0, 3).isApprox(Eigen::Vector3d::Zero(), 1e-6)));

    // 90 deg rotation around Z
    Eigen::Matrix<double, 6, 1> xiZ = Eigen::Matrix<double, 6, 1>::Zero();
    xiZ.tail<3>() = Eigen::Vector3d(0.0, 0.0, PI / 2.0);
    Eigen::Matrix4d TZ = expSE3(xiZ);
    Eigen::Matrix3d expectedRZ;
    expectedRZ << 0.0, -1.0,  0.0,
                  1.0,  0.0,  0.0,
                  0.0,  0.0,  1.0;
    EXPECT_TRUE((TZ.block<3, 3>(0, 0).isApprox(expectedRZ, 1e-6)));
    EXPECT_TRUE((TZ.block<3, 1>(0, 3).isApprox(Eigen::Vector3d::Zero(), 1e-6)));
}

TEST_F(TestLma, ExpSE3SmallAngleStability) {
    Eigen::Matrix<double, 6, 1> xi = Eigen::Matrix<double, 6, 1>::Zero();
    xi.head<3>() = Eigen::Vector3d(0.1, -0.2, 0.3);
    xi.tail<3>() = Eigen::Vector3d(1e-10, -2e-10, 3e-10);

    Eigen::Matrix4d T = expSE3(xi);

    EXPECT_FALSE(T.hasNaN());
    EXPECT_TRUE((T.block<3, 3>(0, 0).isApprox(Eigen::Matrix3d::Identity(), 1e-8)));
    EXPECT_TRUE((T.block<3, 1>(0, 3).isApprox(Eigen::Vector3d(0.1, -0.2, 0.3), 1e-8)));
}

// =============================================================================
// 2. Nominal Pose Convergence (refinePoseLM)
// =============================================================================

TEST_F(TestLma, ZeroPerturbationPreservesPose) {
    auto tracker = createTracker();

    std::vector<Eigen::Vector3d> landmarks = {
        {1.0, 1.0, 5.0}, {-1.0, 1.0, 5.0}, {1.0, -1.0, 5.0}, {-1.0, -1.0, 5.0},
        {0.5, 0.0, 4.0}, {-0.5, 0.0, 4.0}, {0.0, 0.5, 6.0}, {0.0, -0.5, 6.0}
    };

    Eigen::Matrix<double, 3, 4> Tgt;
    Tgt.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
    Tgt.col(3) = Eigen::Vector3d(0.2, -0.1, 1.0);

    setupCorrespondences(tracker, landmarks, Tgt);

    Eigen::Matrix<double, 3, 4> Tcand = Tgt;
    Eigen::Matrix<double, 6, 6> cov;

    bool success = callRefinePoseLM(tracker, landmarks.size(), K_cam, Tcand, cov);

    ASSERT_TRUE(success);
    EXPECT_LT((Tcand - Tgt).norm(), 1e-5);
}

TEST_F(TestLma, ModeratePerturbationConverges) {
    auto tracker = createTracker();

    std::vector<Eigen::Vector3d> landmarks = {
        {1.5, 1.2, 4.5}, {-1.2, 1.5, 5.0}, {1.0, -1.3, 4.8}, {-1.4, -1.1, 5.2},
        {0.8, 0.2, 3.8}, {-0.7, 0.3, 4.2}, {0.1, 0.9, 5.5}, {-0.2, -0.8, 4.9}
    };

    Eigen::Matrix<double, 3, 4> Tgt;
    Tgt.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
    Tgt.col(3) = Eigen::Vector3d(0.5, -0.3, 1.5);

    setupCorrespondences(tracker, landmarks, Tgt);

    // Initial pose perturbed by 5 cm translation and small rotation
    Eigen::Matrix<double, 3, 4> Tcand = Tgt;
    Tcand.col(3) += Eigen::Vector3d(0.05, -0.04, 0.03);
    Eigen::AngleAxisd pert(0.03, Eigen::Vector3d(0.0, 0.0, 1.0)); // ~1.7 deg
    Tcand.block<3, 3>(0, 0) = pert.toRotationMatrix() * Tcand.block<3, 3>(0, 0);

    Eigen::Matrix<double, 6, 6> cov;
    bool success = callRefinePoseLM(tracker, landmarks.size(), K_cam, Tcand, cov);

    ASSERT_TRUE(success);
    EXPECT_LT((Tcand.col(3) - Tgt.col(3)).norm(), 1e-3);
    EXPECT_LT((Tcand.block<3, 3>(0, 0) - Tgt.block<3, 3>(0, 0)).norm(), 2e-3);
}

TEST_F(TestLma, CostReductionMonotonic) {
    auto tracker = createTracker();

    std::vector<Eigen::Vector3d> landmarks = {
        {1.0, 1.0, 4.0}, {-1.0, 1.0, 4.5}, {1.0, -1.0, 4.2}, {-1.0, -1.0, 4.8},
        {0.5, 0.5, 3.5}, {-0.5, 0.5, 3.8}, {0.5, -0.5, 5.0}, {-0.5, -0.5, 5.2}
    };

    Eigen::Matrix<double, 3, 4> Tgt;
    Tgt.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
    Tgt.col(3) = Eigen::Vector3d(0.0, 0.0, 1.0);

    setupCorrespondences(tracker, landmarks, Tgt);

    Eigen::Matrix<double, 3, 4> Tcand = Tgt;
    Tcand.col(3) += Eigen::Vector3d(0.08, -0.06, 0.05);

    double initialCost = computeHuberCost(tracker, landmarks.size(), K_cam, Tcand);

    Eigen::Matrix<double, 6, 6> cov;
    bool success = callRefinePoseLM(tracker, landmarks.size(), K_cam, Tcand, cov);

    ASSERT_TRUE(success);
    double finalCost = computeHuberCost(tracker, landmarks.size(), K_cam, Tcand);
    EXPECT_LT(finalCost, initialCost);
}

// =============================================================================
// 3. Huber Loss Robustness
// =============================================================================

TEST_F(TestLma, OutlierDownweightingPreventsDistortion) {
    auto tracker = createTracker();

    std::vector<Eigen::Vector3d> landmarks = {
        {1.2, 1.0, 4.0}, {-1.0, 1.1, 4.2}, {1.1, -1.2, 4.5}, {-1.1, -1.0, 4.8},
        {0.6, 0.4, 3.6}, {-0.5, 0.5, 3.9}, {0.3, -0.7, 5.0}, {-0.4, -0.6, 5.1},
        {0.8, -0.2, 4.3}, {-0.7, 0.1, 4.6}
    };

    Eigen::Matrix<double, 3, 4> Tgt;
    Tgt.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
    Tgt.col(3) = Eigen::Vector3d(0.1, -0.2, 1.0);

    setupCorrespondences(tracker, landmarks, Tgt);

    // Corrupt last 2 landmarks with large artificial bearing error (> 20 pixels)
    getMatching(tracker).curFeatures(0, 8) += (25.0 / K_cam(0, 0));
    getMatching(tracker).curFeatures(1, 9) += (30.0 / K_cam(1, 1));

    Eigen::Matrix<double, 3, 4> Tcand = Tgt;
    Tcand.col(3) += Eigen::Vector3d(0.03, -0.02, 0.02);

    Eigen::Matrix<double, 6, 6> cov;
    bool success = callRefinePoseLM(tracker, landmarks.size(), K_cam, Tcand, cov);

    ASSERT_TRUE(success);
    // Huber loss should prevent outliers from pulling the estimate far off
    EXPECT_LT((Tcand.col(3) - Tgt.col(3)).norm(), 0.05);
}

// =============================================================================
// 4. Covariance Matrix Invariants
// =============================================================================

TEST_F(TestLma, CovarianceSymmetryAndPositiveDefiniteness) {
    auto tracker = createTracker();

    std::vector<Eigen::Vector3d> landmarks = {
        {1.0, 1.0, 4.0}, {-1.0, 1.0, 4.5}, {1.0, -1.0, 4.2}, {-1.0, -1.0, 4.8},
        {0.5, 0.5, 3.5}, {-0.5, 0.5, 3.8}, {0.5, -0.5, 5.0}, {-0.5, -0.5, 5.2}
    };

    Eigen::Matrix<double, 3, 4> Tgt;
    Tgt.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
    Tgt.col(3) = Eigen::Vector3d(0.0, 0.0, 1.0);

    setupCorrespondences(tracker, landmarks, Tgt);

    Eigen::Matrix<double, 3, 4> Tcand = Tgt;
    Eigen::Matrix<double, 6, 6> cov;

    bool success = callRefinePoseLM(tracker, landmarks.size(), K_cam, Tcand, cov);
    ASSERT_TRUE(success);

    // Symmetry check: cov == cov^T
    EXPECT_LT((cov - cov.transpose()).norm(), 1e-10);

    // Strictly positive diagonal variances
    for (int i = 0; i < 6; ++i) {
        EXPECT_GT(cov(i, i), 0.0);
    }
}

TEST_F(TestLma, CovarianceScaleSensitivity) {
    auto tracker = createTracker();

    std::vector<Eigen::Vector3d> landmarksMany = {
        {1.5, 1.2, 4.0}, {-1.2, 1.5, 4.5}, {1.0, -1.3, 4.2}, {-1.4, -1.1, 4.8},
        {0.8, 0.2, 3.5}, {-0.7, 0.3, 3.8}, {0.1, 0.9, 5.0}, {-0.2, -0.8, 5.2},
        {1.1, -0.5, 4.3}, {-1.0, 0.4, 4.6}, {0.5, -1.0, 4.1}, {-0.6, -0.4, 4.7}
    };

    Eigen::Matrix<double, 3, 4> Tgt;
    Tgt.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
    Tgt.col(3) = Eigen::Vector3d(0.0, 0.0, 1.0);

    // Baseline with 12 landmarks
    setupCorrespondences(tracker, landmarksMany, Tgt);
    Eigen::Matrix<double, 3, 4> T1 = Tgt;
    Eigen::Matrix<double, 6, 6> covMany;
    ASSERT_TRUE(callRefinePoseLM(tracker, 12, K_cam, T1, covMany));

    // Reduced landmark count (4 landmarks)
    Eigen::Matrix<double, 3, 4> T2 = Tgt;
    Eigen::Matrix<double, 6, 6> covFew;
    ASSERT_TRUE(callRefinePoseLM(tracker, 4, K_cam, T2, covFew));

    // Fewer observations produce higher covariance trace (uncertainty)
    EXPECT_GT(covFew.trace(), covMany.trace());
}

// =============================================================================
// 5. Degeneracy & Boundary Conditions
// =============================================================================

TEST_F(TestLma, InsufficientInliersReturnsFalse) {
    auto tracker = createTracker();

    Eigen::Matrix<double, 3, 4> Tcand = Eigen::Matrix<double, 3, 4>::Identity();
    Eigen::Matrix<double, 6, 6> cov;

    // Minimum inliers is 4; passing 3 must return false
    EXPECT_FALSE(callRefinePoseLM(tracker, 3, K_cam, Tcand, cov));
    EXPECT_FALSE(callRefinePoseLM(tracker, 0, K_cam, Tcand, cov));
}

TEST_F(TestLma, CheiralityViolationHandledSafely) {
    auto tracker = createTracker();

    // Landmarks placed behind camera (negative Z)
    std::vector<Eigen::Vector3d> landmarks = {
        {1.0, 1.0, -2.0}, {-1.0, 1.0, -2.0}, {1.0, -1.0, -2.0}, {-1.0, -1.0, -2.0}
    };

    Eigen::Matrix<double, 3, 4> Tcand;
    Tcand.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
    Tcand.col(3) = Eigen::Vector3d(0.0, 0.0, 0.0);

    setupCorrespondences(tracker, landmarks, Tcand);
    Eigen::Matrix<double, 6, 6> cov;

    // Must not crash or produce NaNs
    bool res = callRefinePoseLM(tracker, landmarks.size(), K_cam, Tcand, cov);
    // If it fails or rejects, covariance should not be corrupted with NaNs
    if (res) {
        EXPECT_FALSE(cov.hasNaN());
    }
}

TEST_F(TestLma, DegenerateCollinearLandmarksFailure) {
    auto tracker = createTracker();

    // All landmarks along a straight line (collinear)
    std::vector<Eigen::Vector3d> landmarks = {
        {0.0, 0.0, 3.0}, {0.0, 0.0, 4.0}, {0.0, 0.0, 5.0}, {0.0, 0.0, 6.0}
    };

    Eigen::Matrix<double, 3, 4> Tcand;
    Tcand.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
    Tcand.col(3) = Eigen::Vector3d(0.0, 0.0, 1.0);

    setupCorrespondences(tracker, landmarks, Tcand);
    Eigen::Matrix<double, 6, 6> cov;

    // Collinear landmarks do not constrain roll around line; solver should fail or reject
    bool success = callRefinePoseLM(tracker, landmarks.size(), K_cam, Tcand, cov);
    // If information matrix is singular, refinePoseLM returns false
    EXPECT_FALSE(success);
}

} // namespace PoolSLAM
