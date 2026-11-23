#include <cstdint>
#include <gtest/gtest.h>
#include <math.h>
#include <vector>

#include "CoreTypes.hpp"
#include "HfTrackerTestHelpers.hpp"
#include "frontend/HfTracker.hpp"

namespace PoolSLAM {

// =============================================================================
// 1. Subpixel Bilinear Interpolation
// =============================================================================

// Exact integer coordinates match raw pixel intensities exactly
TEST_F(TestTrackHfFeatures, BilinearIntegerCoordinates) {
    auto tracker = createTracker();
    std::vector<uint8_t> img(W * H, 0);

    img[100 * W + 100] = 77;
    img[100 * W + 101] = 88;
    img[101 * W + 100] = 99;
    img[101 * W + 101] = 111;

    int16_t val = callBilinear(tracker, img.data(), Eigen::Vector2f(100.0f, 100.0f));
    EXPECT_EQ(val, 77);
}

// Half-pixel centroid returns exact arithmetic average of 4 neighboring pixels
TEST_F(TestTrackHfFeatures, BilinearHalfPixelCentroid) {
    auto tracker = createTracker();
    std::vector<uint8_t> img(W * H, 0);

    img[50 * W + 50] = 10;
    img[50 * W + 51] = 20;
    img[51 * W + 50] = 30;
    img[51 * W + 51] = 40;

    // Expected: (10 + 20 + 30 + 40) / 4 = 25
    int16_t val = callBilinear(tracker, img.data(), Eigen::Vector2f(50.5f, 50.5f));
    EXPECT_EQ(val, 25);
}

// Linear intensity ramp verifies continuity and interpolation weighting
TEST_F(TestTrackHfFeatures, BilinearLinearRampContinuity) {
    auto tracker = createTracker();
    std::vector<uint8_t> img(W * H, 0);

    // 1D horizontal ramp across row 100: pixel 10 -> 100, pixel 11 -> 200
    img[100 * W + 10] = 100;
    img[100 * W + 11] = 200;
    img[101 * W + 10] = 100;
    img[101 * W + 11] = 200;

    int16_t val25 = callBilinear(tracker, img.data(), Eigen::Vector2f(10.25f, 100.0f));
    EXPECT_NEAR(val25, 125, 1);

    int16_t val75 = callBilinear(tracker, img.data(), Eigen::Vector2f(10.75f, 100.0f));
    EXPECT_NEAR(val75, 175, 1);
}

// =============================================================================
// 2. Inverse Compositional Lucas-Kanade Alignment
// =============================================================================

// Subpixel displacement converges to ground truth location within subpixel precision
TEST_F(TestTrackHfFeatures, ICLKNominalConvergence) {
    auto tracker = createTracker();
    auto img0 = makeSyntheticImage(W, H, 50);
    drawCorner(img0, W, 100, 100);

    // Create a well-conditioned feature using addKp
    Eigen::Vector2i kp(100, 100);
    ASSERT_TRUE(callAddKp(tracker, img0.data(), kp));
    auto feature = getActiveFeatures(tracker).back();

    // In current frame, feature is displaced by (+1, 0) pixels
    auto img1 = makeSyntheticImage(W, H, 50);
    drawCorner(img1, W, 101, 100);

    // Initial estimate starting near the feature
    Eigen::Vector2f ptEst(100.0f, 100.0f);
    bool converged = callAlignFeatureICLK(tracker, feature, img1.data(), ptEst);

    EXPECT_TRUE(converged);
    EXPECT_EQ(feature.age, 1);
    EXPECT_DOUBLE_EQ(feature.ptPrev.x(), 100.0f);
    EXPECT_DOUBLE_EQ(feature.ptPrev.y(), 100.0f);
    EXPECT_NEAR(feature.ptCurr.x(), 101.0f, 0.5f);
    EXPECT_NEAR(feature.ptCurr.y(), 100.0f, 0.5f);
}

// Search coordinates crossing image margins are rejected
TEST_F(TestTrackHfFeatures, ICLKBoundaryMarginRejection) {
    auto tracker = createTracker();
    auto img = makeSyntheticImage(W, H, 50);
    auto feature = makeFeature(1, Eigen::Vector2f(20.0f, 20.0f));

    // Test position too close to border (<= RADIUS + 1 = 6)
    Eigen::Vector2f ptEstNearBorder(5.0f, 20.0f);
    EXPECT_FALSE(callAlignFeatureICLK(tracker, feature, img.data(), ptEstNearBorder));

    // Test position too close to max border (>= W - RADIUS - 2)
    Eigen::Vector2f ptEstFarBorder(static_cast<float>(W - 5), 20.0f);
    EXPECT_FALSE(callAlignFeatureICLK(tracker, feature, img.data(), ptEstFarBorder));
}

// Appearance degradation (MAD > 605) causes tracking rejection
TEST_F(TestTrackHfFeatures, ICLKAppearanceDegradationRejection) {
    auto tracker = createTracker();
    auto feature = makeFeature(1, Eigen::Vector2f(100.0f, 100.0f));
    feature.patch.fill(0); // Template is all-black

    // Current image is all-white (delta = 255 per pixel, MAD >> 605)
    auto img = makeSyntheticImage(W, H, 255);
    Eigen::Vector2f ptEst(100.0f, 100.0f);

    EXPECT_FALSE(callAlignFeatureICLK(tracker, feature, img.data(), ptEst));
}

// =============================================================================
// 3. Lens Undistortion
// =============================================================================

// Identity lens undistortion with zero distortion coefficients matches pinhole projection
TEST_F(TestTrackHfFeatures, UndistortPinholeIdentity) {
    auto tracker = createTracker();

    std::vector<float> xDist = {320.0f, 400.0f, 120.0f};
    std::vector<float> yDist = {240.0f, 300.0f, 100.0f};
    Eigen::Matrix3Xf ptUndist(3, 3);

    callUndistort(tracker, xDist, yDist, ptUndist, 3);

    // Principal point: (320 - 320)/400 = 0, (240 - 240)/400 = 0, z = 1
    EXPECT_NEAR(ptUndist(0, 0), 0.0f, 1e-5);
    EXPECT_NEAR(ptUndist(1, 0), 0.0f, 1e-5);
    EXPECT_NEAR(ptUndist(2, 0), 1.0f, 1e-5);

    // Point 1: (400 - 320)/400 = 0.20, (300 - 240)/400 = 0.15
    EXPECT_NEAR(ptUndist(0, 1), 0.20f, 1e-5);
    EXPECT_NEAR(ptUndist(1, 1), 0.15f, 1e-5);
    EXPECT_NEAR(ptUndist(2, 1), 1.0f, 1e-5);
}

// Undistortion inverts radial distortion iteratively
TEST_F(TestTrackHfFeatures, UndistortRadialDistortion) {
    Distortion radDist{-0.1f, 0.0f, 0.0f, 0.0f};
    HfTracker tracker(W, H, K, radDist, 50);

    // True normalized ray: (0.2, 0.2) -> r^2 = 0.08, radial = 1 - 0.1 * 0.08 = 0.992
    // Distorted normalized: x_d = 0.2 * 0.992 = 0.1984
    // Distorted pixel: u_d = 320 + 400 * 0.1984 = 399.36
    float u_d = 320.0f + 400.0f * (0.2f * (1.0f - 0.1f * 0.08f));
    float v_d = 240.0f + 400.0f * (0.2f * (1.0f - 0.1f * 0.08f));

    std::vector<float> xDist = {u_d};
    std::vector<float> yDist = {v_d};
    Eigen::Matrix3Xf ptUndist(3, 1);

    callUndistort(tracker, xDist, yDist, ptUndist, 1);

    // Should invert back to approximately (0.2, 0.2)
    EXPECT_NEAR(ptUndist(0, 0), 0.2f, 1e-3);
    EXPECT_NEAR(ptUndist(1, 0), 0.2f, 1e-3);
    EXPECT_NEAR(ptUndist(2, 0), 1.0f, 1e-5);
}

// =============================================================================
// 4. 2-Point RANSAC Outlier Rejection
// =============================================================================

// Pure rotation matching gyro prior identifies all features as inliers
TEST_F(TestTrackHfFeatures, RansacPureRotationConsensus) {
    auto tracker = createTracker();
    auto& features = getActiveFeatures(tracker);

    // Rotation around Z by 2 degrees
    float angle = 2.0f * static_cast<float>(M_PI) / 180.0f;
    Eigen::Matrix3f R;
    R = Eigen::AngleAxisf(angle, Eigen::Vector3f::UnitZ()).toRotationMatrix();

    // Create 30 features undergoing pure rotation
    for (int i = 0; i < 30; ++i) {
        float x = 100.0f + static_cast<float>(i % 6) * 70.0f;
        float y = 100.0f + static_cast<float>(static_cast<float>(i) / 6) * 60.0f;

        // Normalized ray rotated into current frame via R^T
        Eigen::Vector3f ray((x - 320.0f) / 400.0f, (y - 240.0f) / 400.0f, 1.0f);
        Eigen::Vector3f rotatedRay = R.transpose() * ray;
        float uCurr = 320.0f + 400.0f * (rotatedRay.x() / rotatedRay.z());
        float vCurr = 240.0f + 400.0f * (rotatedRay.y() / rotatedRay.z());

        auto f = makeFeature(static_cast<uint32_t>(i), Eigen::Vector2f(uCurr, vCurr));
        f.ptPrev = Eigen::Vector2f(x, y);
        features.push_back(f);
    }

    int inliers = callRansac(tracker, R);
    EXPECT_GE(inliers, 20);
    EXPECT_EQ(features.size(), 30);
}

// 2-point epipolar RANSAC separates translation inliers from injected outliers
TEST_F(TestTrackHfFeatures, RansacEpipolarSeparatesOutliers) {
    auto tracker = createTracker();
    auto& features = getActiveFeatures(tracker);

    Eigen::Matrix3f R = Eigen::Matrix3f::Identity();
    Eigen::Vector3f t(0.05f, 0.0f, 0.0f); // Pure horizontal baseline

    // 25 consistent inliers
    for (int i = 0; i < 25; ++i) {
        float xPrev = 150.0f + static_cast<float>(i % 5) * 60.0f;
        float yPrev = 150.0f + static_cast<float>(static_cast<float>(i) / 5) * 50.0f;
        float depth = 2.0f;

        // Epipolar displacement from translation: uCurr = uPrev - fx * tx / depth
        float xCurr = xPrev - 400.0f * (t.x() / depth);
        float yCurr = yPrev;

        auto f = makeFeature(static_cast<uint32_t>(i), Eigen::Vector2f(xCurr, yCurr));
        f.ptPrev = Eigen::Vector2f(xPrev, yPrev);
        features.push_back(f);
    }

    // 5 random gross outliers with vertical displacements
    for (int i = 25; i < 30; ++i) {
        auto f = makeFeature(static_cast<uint32_t>(i), Eigen::Vector2f(300.0f, 400.0f));
        f.ptPrev = Eigen::Vector2f(100.0f, 100.0f);
        features.push_back(f);
    }

    EXPECT_EQ(features.size(), 30);
    int inliers = callRansac(tracker, R);

    EXPECT_GE(inliers, 20);
    // Outliers should have been pruned
    EXPECT_LE(features.size(), 26);
}

// Consensus failure (< 20 inliers) preserves all features and populates ptUndis
TEST_F(TestTrackHfFeatures, RansacConsensusFailurePreservesFeatures) {
    auto tracker = createTracker();
    auto& features = getActiveFeatures(tracker);

    // Only 10 random features (below RANSAC_MIN_INLIERS = 20)
    for (int i = 0; i < 10; ++i) {
        float fi = static_cast<float>(i);
        auto f = makeFeature(static_cast<uint32_t>(i), Eigen::Vector2f(100.0f + fi * 20.0f, 100.0f));
        f.ptPrev = Eigen::Vector2f(200.0f, 200.0f + fi * 15.0f);
        features.push_back(f);
    }

    int inliers = callRansac(tracker, Eigen::Matrix3f::Identity());
    EXPECT_LT(inliers, 20);
    // Features must not be dropped when consensus fails
    EXPECT_EQ(features.size(), 10);
}

// =============================================================================
// 5. Bucket Density Regularization
// =============================================================================

// Features in distinct buckets are never evicted
TEST_F(TestTrackHfFeatures, BucketDensityIsolation) {
    auto tracker = createTracker();
    auto& features = getActiveFeatures(tracker);

    // Place features far apart in distinct buckets
    features.push_back(makeFeature(1, Eigen::Vector2f(50.0f, 50.0f), 10));
    features.push_back(makeFeature(2, Eigen::Vector2f(300.0f, 200.0f), 5));
    features.push_back(makeFeature(3, Eigen::Vector2f(500.0f, 400.0f), 1));

    callEnforceBucketDensity(tracker);
    EXPECT_EQ(features.size(), 3);
}

// Younger feature is evicted when two features are within 6.0 pixels in same bucket
TEST_F(TestTrackHfFeatures, BucketDensityIntraBucketCrowdingEvictsYounger) {
    auto tracker = createTracker();
    auto& features = getActiveFeatures(tracker);

    // Two features at (100, 100) and (102, 101) (distance = sqrt(5) = 2.23 < 6.0)
    features.push_back(makeFeature(1, Eigen::Vector2f(100.0f, 100.0f), 10)); // Older (age 10)
    features.push_back(makeFeature(2, Eigen::Vector2f(102.0f, 101.0f), 2));  // Younger (age 2)

    callEnforceBucketDensity(tracker);
    ASSERT_EQ(features.size(), 1);
    EXPECT_EQ(features[0].id, 1); // Older feature retained
}

// Third feature falling into a bucket evicts the youngest feature
TEST_F(TestTrackHfFeatures, BucketDensityCappingAtTwoFeatures) {
    auto tracker = createTracker();
    auto& features = getActiveFeatures(tracker);

    // 3 features in same bucket, separated by >= 6px
    features.push_back(makeFeature(1, Eigen::Vector2f(100.0f, 100.0f), 15)); // Oldest
    features.push_back(makeFeature(2, Eigen::Vector2f(110.0f, 100.0f), 8));  // Intermediate
    features.push_back(makeFeature(3, Eigen::Vector2f(100.0f, 110.0f), 1));  // Youngest

    callEnforceBucketDensity(tracker);
    ASSERT_EQ(features.size(), 2);
    EXPECT_EQ(features[0].id, 1);
    EXPECT_EQ(features[1].id, 2);
}

// =============================================================================
// 6. End-to-End Tracking Pipeline
// =============================================================================

// Passing nullptr buffer invokes handleTrackingError and clears active features
TEST_F(TestTrackHfFeatures, TrackExistingFeaturesNullptrErrorHandling) {
    auto tracker = createTracker();
    auto& features = getActiveFeatures(tracker);
    features.push_back(makeFeature(1, Eigen::Vector2f(100.0f, 100.0f)));

    float displacement = tracker.trackExistingFeatures(nullptr, Eigen::Matrix3f::Identity());
    EXPECT_FLOAT_EQ(displacement, 0.0f);
    EXPECT_TRUE(features.empty());
}

// Tracking on an empty active features set returns 0.0f
TEST_F(TestTrackHfFeatures, TrackExistingFeaturesEmptySetReturnsZero) {
    auto tracker = createTracker();
    auto img = makeSyntheticImage(W, H, 50);

    float displacement = tracker.trackExistingFeatures(img.data(), Eigen::Matrix3f::Identity());
    EXPECT_FLOAT_EQ(displacement, 0.0f);
}

} // namespace PoolSLAM
