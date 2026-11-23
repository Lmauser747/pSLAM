#include "TestP3pHelpers.hpp"

namespace PoolSLAM {

// =============================================================================
// 1. Polynomial & Linear Algebra Helpers
// =============================================================================

TEST_F(TestP3p, CubicSingleRealRoot) {
    // x^3 + 2x - 3 = 0, sole real root is x = 1
    double root = geometry::solveLambdaTwistCubic(1.0, 0.0, 2.0, -3.0);
    EXPECT_NEAR(root, 1.0, 1e-6);

    double residual = std::abs(root * root * root + 2.0 * root - 3.0);
    EXPECT_LT(residual, 1e-10);
}

TEST_F(TestP3p, CubicThreeRealRoots) {
    // (x - 1)(x - 2)(x - 3) = x^3 - 6x^2 + 11x - 6 = 0, roots: 1, 2, 3
    double root = geometry::solveLambdaTwistCubic(1.0, -6.0, 11.0, -6.0);

    double residual = std::abs(root * root * root - 6.0 * root * root + 11.0 * root - 6.0);
    EXPECT_LT(residual, 1e-10);

    bool matchesKnownRoot = (std::abs(root - 1.0) < 1e-5) ||
                            (std::abs(root - 2.0) < 1e-5) ||
                            (std::abs(root - 3.0) < 1e-5);
    EXPECT_TRUE(matchesKnownRoot);
}

TEST_F(TestP3p, DiagonalizeD0Rank2SymmetricMatrix) {
    // Construct a symmetric rank-2 matrix D = Q * diag(3.0, -2.0, 0.0) * Q^T
    Eigen::Matrix3d Q = makeRotation(25.0, -35.0, 45.0);
    Eigen::Vector3d lambdas(3.0, -2.0, 0.0);
    Eigen::Matrix3d D = Q * lambdas.asDiagonal() * Q.transpose();

    auto decomp = geometry::diagonalizeD0(D);

    // Opposite signs and descending absolute value
    EXPECT_LE(decomp.sigma1 * decomp.sigma2, 0.0);
    EXPECT_GE(std::abs(decomp.sigma1), std::abs(decomp.sigma2));

    // Unit norm eigenvectors
    EXPECT_NEAR(decomp.e1.norm(), 1.0, 1e-6);
    EXPECT_NEAR(decomp.e2.norm(), 1.0, 1e-6);

    // Orthogonality
    EXPECT_NEAR(decomp.e1.dot(decomp.e2), 0.0, 1e-6);

    // Eigenvector equation: D * e = sigma * e
    Eigen::Vector3d diff1 = D * decomp.e1 - decomp.sigma1 * decomp.e1;
    Eigen::Vector3d diff2 = D * decomp.e2 - decomp.sigma2 * decomp.e2;
    EXPECT_LT(diff1.norm(), 1e-5);
    EXPECT_LT(diff2.norm(), 1e-5);
}

// =============================================================================
// 2. Nominal 6-DoF Pose Recovery
// =============================================================================

TEST_F(TestP3p, SolveP4PIdentityPose) {
    std::array<Eigen::Vector3d, 4> x = {
        Eigen::Vector3d(1.2, 0.8, 4.5),
        Eigen::Vector3d(-1.0, 1.5, 5.2),
        Eigen::Vector3d(0.2, -1.1, 6.0),
        Eigen::Vector3d(0.7, 0.4, 4.0)
    };

    std::array<Eigen::Vector3d, 4> y;
    for (size_t i = 0; i < 4; ++i) {
        y[i] = projectBearing(x[i]);
    }

    auto optT = geometry::solveP4PLambdaTwist(x, y);
    ASSERT_TRUE(optT.has_value());

    Eigen::Matrix<double, 3, 4> T = optT.value();
    Eigen::Matrix3d R = T.block<3, 3>(0, 0);
    Eigen::Vector3d t = T.col(3);

    EXPECT_LT((R - Eigen::Matrix3d::Identity()).norm(), 1e-6);
    EXPECT_LT(t.norm(), 1e-6);

    for (size_t i = 0; i < 4; ++i) {
        EXPECT_LT(computeReprojError(T, x[i], y[i]), 1e-6);
    }
}

TEST_F(TestP3p, SolveP4PPureTranslation) {
    std::array<Eigen::Vector3d, 4> x = {
        Eigen::Vector3d(1.0, 1.0, 5.0),
        Eigen::Vector3d(-1.0, 1.0, 5.0),
        Eigen::Vector3d(0.0, -1.0, 6.0),
        Eigen::Vector3d(0.5, 0.5, 4.0)
    };

    Eigen::Vector3d tGt(0.5, -1.2, 3.0);
    std::array<Eigen::Vector3d, 4> y;
    for (size_t i = 0; i < 4; ++i) {
        y[i] = projectBearing(x[i] + tGt);
    }

    auto optT = geometry::solveP4PLambdaTwist(x, y);
    ASSERT_TRUE(optT.has_value());

    Eigen::Matrix<double, 3, 4> T = optT.value();
    Eigen::Matrix3d R = T.block<3, 3>(0, 0);
    Eigen::Vector3d t = T.col(3);

    EXPECT_LT((R - Eigen::Matrix3d::Identity()).norm(), 1e-6);
    EXPECT_LT((t - tGt).norm(), 1e-6);

    for (size_t i = 0; i < 4; ++i) {
        EXPECT_LT(computeReprojError(T, x[i], y[i]), 1e-6);
    }
}

TEST_F(TestP3p, SolveP4PPureRotation) {
    Eigen::Matrix3d RGt = makeRotation(30.0, -45.0, 60.0);
    Eigen::Vector3d tGt = Eigen::Vector3d::Zero();

    // World points placed so that camera coordinates RGt * x have positive depth
    std::array<Eigen::Vector3d, 4> xCam = {
        Eigen::Vector3d(1.2, 0.8, 4.5),
        Eigen::Vector3d(-1.0, 1.5, 5.2),
        Eigen::Vector3d(0.2, -1.1, 6.0),
        Eigen::Vector3d(0.7, 0.4, 4.0)
    };

    std::array<Eigen::Vector3d, 4> x;
    std::array<Eigen::Vector3d, 4> y;
    for (size_t i = 0; i < 4; ++i) {
        x[i] = RGt.transpose() * xCam[i];
        y[i] = projectBearing(xCam[i]);
    }

    auto optT = geometry::solveP4PLambdaTwist(x, y);
    ASSERT_TRUE(optT.has_value());

    Eigen::Matrix<double, 3, 4> T = optT.value();
    Eigen::Matrix3d R = T.block<3, 3>(0, 0);
    Eigen::Vector3d t = T.col(3);

    EXPECT_LT((R - RGt).norm(), 1e-6);
    EXPECT_LT(t.norm(), 1e-6);

    for (size_t i = 0; i < 4; ++i) {
        EXPECT_LT(computeReprojError(T, x[i], y[i]), 1e-6);
    }
}

TEST_F(TestP3p, SolveP4PGeneral6DoFMotion) {
    Eigen::Matrix3d RGt = makeRotation(15.0, 20.0, -25.0);
    Eigen::Vector3d tGt(0.3, -0.4, 1.5);

    std::array<Eigen::Vector3d, 4> x = {
        Eigen::Vector3d(1.2, 0.8, 4.5),
        Eigen::Vector3d(-1.0, 1.5, 5.2),
        Eigen::Vector3d(0.2, -1.1, 6.0),
        Eigen::Vector3d(0.7, 0.4, 4.0)
    };

    std::array<Eigen::Vector3d, 4> y;
    for (size_t i = 0; i < 4; ++i) {
        Eigen::Vector3d pCam = RGt * x[i] + tGt;
        y[i] = projectBearing(pCam);
    }

    auto optT = geometry::solveP4PLambdaTwist(x, y);
    ASSERT_TRUE(optT.has_value());

    Eigen::Matrix<double, 3, 4> T = optT.value();
    Eigen::Matrix3d R = T.block<3, 3>(0, 0);
    Eigen::Vector3d t = T.col(3);

    EXPECT_LT((R - RGt).norm(), 1e-6);
    EXPECT_LT((t - tGt).norm(), 1e-6);

    for (size_t i = 0; i < 4; ++i) {
        EXPECT_LT(computeReprojError(T, x[i], y[i]), 1e-6);
    }
}

// =============================================================================
// 3. Disambiguation & Cheirality
// =============================================================================

TEST_F(TestP3p, DisambiguationSelectsBestCandidate) {
    Eigen::Matrix3d RGt = makeRotation(10.0, -15.0, 20.0);
    Eigen::Vector3d tGt(0.2, 0.1, 2.0);

    std::array<Eigen::Vector3d, 4> x = {
        Eigen::Vector3d(1.0, 0.5, 4.0),
        Eigen::Vector3d(-0.8, 0.6, 4.2),
        Eigen::Vector3d(0.1, -0.9, 4.5),
        Eigen::Vector3d(0.4, 0.3, 3.8)
    };

    std::array<Eigen::Vector3d, 4> y;
    for (size_t i = 0; i < 4; ++i) {
        y[i] = projectBearing(RGt * x[i] + tGt);
    }

    auto optT = geometry::solveP4PLambdaTwist(x, y);
    ASSERT_TRUE(optT.has_value());

    // Selected candidate must have all points in front of camera
    for (size_t i = 0; i < 4; ++i) {
        Eigen::Vector3d pCam = optT.value().block<3, 3>(0, 0) * x[i] + optT.value().col(3);
        EXPECT_GT(pCam.z(), 1e-4);
    }
}

// =============================================================================
// 4. Degeneracy & Rejection
// =============================================================================

TEST_F(TestP3p, DegenerateCoincidentWorldPoints) {
    std::array<Eigen::Vector3d, 4> x = {
        Eigen::Vector3d(1.0, 1.0, 5.0),
        Eigen::Vector3d(1.0, 1.0, 5.0), // Coincident with x[0]
        Eigen::Vector3d(0.0, -1.0, 6.0),
        Eigen::Vector3d(0.5, 0.5, 4.0)
    };

    std::array<Eigen::Vector3d, 4> y;
    for (size_t i = 0; i < 4; ++i) {
        y[i] = projectBearing(x[i]);
    }

    EXPECT_FALSE(geometry::solveP4PLambdaTwist(x, y).has_value());
}

TEST_F(TestP3p, DegenerateCollinearWorldPoints) {
    std::array<Eigen::Vector3d, 4> x = {
        Eigen::Vector3d(0.0, 0.0, 5.0),
        Eigen::Vector3d(1.0, 1.0, 5.0),
        Eigen::Vector3d(2.0, 2.0, 5.0), // Collinear with x[0] and x[1]
        Eigen::Vector3d(0.5, 0.5, 4.0)
    };

    std::array<Eigen::Vector3d, 4> y;
    for (size_t i = 0; i < 4; ++i) {
        y[i] = projectBearing(x[i]);
    }

    EXPECT_FALSE(geometry::solveP4PLambdaTwist(x, y).has_value());
}

TEST_F(TestP3p, CoplanarNonCollinearPointsSucceed) {
    // Points sharing the plane Z = 5, forming a non-collinear triangle
    std::array<Eigen::Vector3d, 4> x = {
        Eigen::Vector3d(0.0, 0.0, 5.0),
        Eigen::Vector3d(2.0, 0.0, 5.0),
        Eigen::Vector3d(0.0, 2.0, 5.0),
        Eigen::Vector3d(1.0, 1.0, 5.0)
    };

    std::array<Eigen::Vector3d, 4> y;
    for (size_t i = 0; i < 4; ++i) {
        y[i] = projectBearing(x[i]);
    }

    EXPECT_TRUE(geometry::solveP4PLambdaTwist(x, y).has_value());
}

TEST_F(TestP3p, IncompatibleFourthPointReturnsBestFit) {
    std::array<Eigen::Vector3d, 4> x = {
        Eigen::Vector3d(1.2, 0.8, 4.5),
        Eigen::Vector3d(-1.0, 1.5, 5.2),
        Eigen::Vector3d(0.2, -1.1, 6.0),
        Eigen::Vector3d(0.7, 0.4, 4.0)
    };

    std::array<Eigen::Vector3d, 4> y;
    for (size_t i = 0; i < 3; ++i) {
        y[i] = projectBearing(x[i]);
    }
    // Perturb 4th ray slightly
    y[3] = projectBearing(x[3]) + Eigen::Vector3d(0.02, 0.02, 0.0);

    auto optT = geometry::solveP4PLambdaTwist(x, y);
    EXPECT_TRUE(optT.has_value());
}

} // namespace PoolSLAM
