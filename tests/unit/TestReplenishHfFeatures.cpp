#include <cstddef>
#include <cstdint>
#include <vector>
#include <cmath>
#include <optional>

#include <gtest/gtest.h>

#include "HfTrackerTestHelpers.hpp"

namespace PoolSLAM {

// =============================================================================
// 1. FAST-9 Corner Detection
// =============================================================================

// Flat uniform image generates zero FAST keypoints
TEST_F(TestReplenishHfFeatures, FASTDetectionFlatTextureReturnsZero) {
    auto tracker = createTracker();
    std::vector<uint8_t> flatImg(W * H, 128);

    size_t count = callGetFast(tracker, flatImg.data(), 0);
    EXPECT_EQ(count, 0);
    EXPECT_TRUE(getFastPoints(tracker).empty());
}

// 1D straight step edge is rejected by FAST-9 arc continuity criteria
TEST_F(TestReplenishHfFeatures, FASTDetectionStraightEdgeRejected) {
    auto tracker = createTracker();
    std::vector<uint8_t> edgeImg(W * H, 50);

    // Get bucket 0 coordinates
    Eigen::Vector2i tl = getTopLeftBucket(tracker, 0);

    // Draw a sharp straight horizontal edge through bucket 0
    for (size_t y = static_cast<size_t>(tl.y() + 15); y < H; ++y) {
        for (size_t x = 0; x < W; ++x) {
            edgeImg[y * W + x] = 200;
        }
    }

    size_t count = callGetFast(tracker, edgeImg.data(), 0);
    EXPECT_EQ(count, 0);
    EXPECT_TRUE(getFastPoints(tracker).empty());
}

// High-contrast corner satisfies the 9-contiguous-arc criterion and is detected
TEST_F(TestReplenishHfFeatures, FASTDetectionHighContrastCorner) {
    auto tracker = createTracker();
    std::vector<uint8_t> img(W * H, 50);

    // Place a sharp 90-degree corner well inside bucket 0
    Eigen::Vector2i tl = getTopLeftBucket(tracker, 0);
    int cx = tl.x() + 25;
    int cy = tl.y() + 25;
    drawCorner(img, W, cx, cy, 50, 220);

    size_t count = callGetFast(tracker, img.data(), 0);
    EXPECT_GE(count, 1);

    const auto& pts = getFastPoints(tracker);
    ASSERT_FALSE(pts.empty());

    // Verified detected coordinate is near the true corner location
    bool nearCorner = false;
    for (const auto& pt : pts) {
        if (std::abs(pt.x() - cx) <= 3 && std::abs(pt.y() - cy) <= 3) {
            nearCorner = true;
            break;
        }
    }
    EXPECT_TRUE(nearCorner);
}

// Corners outside bucket bounding box are never ingested into fastPoints for that bucket
TEST_F(TestReplenishHfFeatures, FASTBucketBoundingIsolation) {
    auto tracker = createTracker();
    std::vector<uint8_t> img(W * H, 50);

    // Place corner in bucket 1
    Eigen::Vector2i tl1 = getTopLeftBucket(tracker, 1);
    drawCorner(img, W, tl1.x() + 20, tl1.y() + 20, 50, 220);

    // Scanning bucket 0 should find 0 points
    size_t count0 = callGetFast(tracker, img.data(), 0);
    EXPECT_EQ(count0, 0);
    EXPECT_TRUE(getFastPoints(tracker).empty());

    // Scanning bucket 1 should find the corner
    size_t count1 = callGetFast(tracker, img.data(), 1);
    EXPECT_GE(count1, 1);
}

// Contrast thresholding: delta < FAST_THRESHOLD (20) rejected; delta > 20 accepted
TEST_F(TestReplenishHfFeatures, FASTThresholdDiscrimination) {
    auto tracker = createTracker();
    Eigen::Vector2i tl = getTopLeftBucket(tracker, 0);
    int cx = tl.x() + 25;
    int cy = tl.y() + 25;

    // Sub-threshold corner: delta = 15 (< FAST_THRESHOLD = 20)
    std::vector<uint8_t> subImg(W * H, 100);
    drawCorner(subImg, W, cx, cy, 100, 115);
    size_t countSub = callGetFast(tracker, subImg.data(), 0);
    EXPECT_EQ(countSub, 0);

    // Above-threshold corner: delta = 35 (> 20)
    std::vector<uint8_t> superImg(W * H, 100);
    drawCorner(superImg, W, cx, cy, 100, 135);
    size_t countSuper = callGetFast(tracker, superImg.data(), 0);
    EXPECT_GE(countSuper, 1);
}

// =============================================================================
// 2. Shi-Tomasi Scoring & Salience
// =============================================================================

// bestKp chooses candidate with maximal minimum eigenvalue lambda_min
TEST_F(TestReplenishHfFeatures, BestKpSalienceSelection) {
    auto tracker = createTracker();
    std::vector<uint8_t> img(W * H, 50);

    // Corner A: contrast delta = 80
    int ax = 100, ay = 100;
    drawCorner(img, W, ax, ay, 50, 130);

    // Corner B: contrast delta = 170 (sharper, higher lambda_min)
    int bx = 200, by = 200;
    drawCorner(img, W, bx, by, 50, 220);

    auto& fastPts = getFastPoints(tracker);
    fastPts.clear();
    fastPts.push_back(Eigen::Vector2i(ax, ay));
    fastPts.push_back(Eigen::Vector2i(bx, by));

    auto best = callBestKp(tracker, img.data());
    ASSERT_TRUE(best.has_value());
    EXPECT_EQ(best->x(), bx);
    EXPECT_EQ(best->y(), by);
}

// Candidates with lambda_min below ST_THRESHOLD (500.0) return std::nullopt
TEST_F(TestReplenishHfFeatures, BestKpLowTextureRejection) {
    auto tracker = createTracker();
    // Uniform low-contrast patch produces near-zero structure tensor
    std::vector<uint8_t> img(W * H, 100);

    auto& fastPts = getFastPoints(tracker);
    fastPts.clear();
    fastPts.push_back(Eigen::Vector2i(100, 100));

    auto best = callBestKp(tracker, img.data());
    EXPECT_FALSE(best.has_value());
}

// Empty fastPoints buffer returns std::nullopt
TEST_F(TestReplenishHfFeatures, BestKpEmptyCandidatesReturnsNullopt) {
    auto tracker = createTracker();
    std::vector<uint8_t> img(W * H, 100);

    getFastPoints(tracker).clear();
    auto best = callBestKp(tracker, img.data());
    EXPECT_FALSE(best.has_value());
}

// =============================================================================
// 3. Feature Construction & Conditioning
// =============================================================================

// addKp successfully constructs a well-conditioned feature and appends to activeFeatures
TEST_F(TestReplenishHfFeatures, AddKpConstructsWellConditionedFeature) {
    auto tracker = createTracker();
    std::vector<uint8_t> img(W * H, 50);

    int cx = 150, cy = 150;
    drawCorner(img, W, cx, cy, 50, 220);

    Eigen::Vector2i kp(cx, cy);
    bool added = callAddKp(tracker, img.data(), kp);

    EXPECT_TRUE(added);
    const auto& features = getActiveFeatures(tracker);
    ASSERT_EQ(features.size(), 1);

    const auto& f = features[0];
    EXPECT_EQ(f.id, 0);
    EXPECT_EQ(f.age, 0);
    EXPECT_EQ(f.ptCurr, Eigen::Vector2f(cx, cy));
    EXPECT_EQ(f.ptPrev, Eigen::Vector2f(cx, cy));

    // Pinhole ray under zero distortion: (150 - 320)/400 = -0.425, (150 - 240)/400 = -0.225
    EXPECT_NEAR(f.ptUndis.x(), (150.0f - 320.0f) / 400.0f, 1e-4);
    EXPECT_NEAR(f.ptUndis.y(), (150.0f - 240.0f) / 400.0f, 1e-4);

    // Inverted Hessian must be finite and invertible
    EXPECT_FALSE(std::isnan(f.HInv(0, 0)));
    EXPECT_FALSE(std::isnan(f.HInv(1, 1)));
}

// Aperture problem: 1D straight edge produces singular/degenerate Hessian and is rejected
TEST_F(TestReplenishHfFeatures, AddKpRejectsApertureProblemSingularHessian) {
    auto tracker = createTracker();
    std::vector<uint8_t> img(W * H, 50);

    // Draw straight 1D horizontal edge: zero horizontal gradient -> det(H) = 0
    for (size_t y = 150; y < H; ++y) {
        for (size_t x = 0; x < W; ++x) {
            img[y * W + x] = 220;
        }
    }

    Eigen::Vector2i kp(150, 150);
    bool added = callAddKp(tracker, img.data(), kp);

    EXPECT_FALSE(added);
    EXPECT_TRUE(getActiveFeatures(tracker).empty());
}

// =============================================================================
// 4. Bucket Replenishment Orchestration
// =============================================================================

// Passing nullptr image buffer returns 0 safely
TEST_F(TestReplenishHfFeatures, ReplenishBucketsNullptrHandling) {
    auto tracker = createTracker();
    EXPECT_EQ(tracker.replenishBuckets(nullptr), 0);
}

// Replenishes empty buckets containing corners and updates grid occupancy counts
TEST_F(TestReplenishHfFeatures, ReplenishBucketsPopulatesEmptyBuckets) {
    auto tracker = createTracker();
    std::vector<uint8_t> img(W * H, 50);

    // Draw strong corners in buckets 0 and 1
    Eigen::Vector2i tl0 = getTopLeftBucket(tracker, 0);
    Eigen::Vector2i tl1 = getTopLeftBucket(tracker, 1);
    drawCorner(img, W, tl0.x() + 25, tl0.y() + 25, 50, 220);
    drawCorner(img, W, tl1.x() + 25, tl1.y() + 25, 50, 220);

    size_t occupied = tracker.replenishBuckets(img.data());
    EXPECT_GE(occupied, 2);

    const auto& grid = getGridOccupancy(tracker);
    EXPECT_GE(grid[0].count, 1);
    EXPECT_GE(grid[1].count, 1);
    EXPECT_NE(grid[0].olderIdx, UINT32_MAX);
}

// Buckets that already contain active features are not redundantly replenished
TEST_F(TestReplenishHfFeatures, ReplenishBucketsPreservesOccupiedBuckets) {
    auto tracker = createTracker();
    std::vector<uint8_t> img(W * H, 50);

    Eigen::Vector2i tl0 = getTopLeftBucket(tracker, 0);
    drawCorner(img, W, tl0.x() + 25, tl0.y() + 25, 50, 220);

    // Pre-mark bucket 0 as already occupied
    auto& grid = getGridOccupancy(tracker);
    grid[0].count = 1;
    grid[0].olderIdx = 0;

    size_t initialFeatureCount = getActiveFeatures(tracker).size();
    tracker.replenishBuckets(img.data());

    // No new feature should be added to bucket 0
    EXPECT_EQ(getActiveFeatures(tracker).size(), initialFeatureCount);
}

} // namespace PoolSLAM
