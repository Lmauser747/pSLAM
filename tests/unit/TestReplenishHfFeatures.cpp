#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include <gtest/gtest.h>
#include <eigen3/Eigen/Dense>

#include "CoreTypes.hpp"
#include "frontend/HfTracker.hpp"

namespace PoolSLAM {
namespace {

// Renders a smooth continuous 2D corner with bilinear ramp profile
void drawSmoothCorner(std::vector<uint8_t> &buffer, size_t width, size_t height,
                      float cx, float cy, int radius, uint8_t darkVal, uint8_t brightVal,
                      float rampWidth = 2.0f) {
    const float midVal = 0.5f * (static_cast<float>(darkVal) + static_cast<float>(brightVal));
    const float halfRange = 0.5f * (static_cast<float>(brightVal) - static_cast<float>(darkVal));

    const int minX = std::max(0, static_cast<int>(cx) - radius);
    const int maxX = std::min(static_cast<int>(width) - 1, static_cast<int>(cx) + radius);
    const int minY = std::max(0, static_cast<int>(cy) - radius);
    const int maxY = std::min(static_cast<int>(height) - 1, static_cast<int>(cy) + radius);

    for (int y = minY; y <= maxY; y++) {
        const float dy = (static_cast<float>(y) - cy) / rampWidth;
        const float wy = std::clamp(dy, -1.0f, 1.0f);
        for (int x = minX; x <= maxX; x++) {
            const float dx = (static_cast<float>(x) - cx) / rampWidth;
            const float wx = std::clamp(dx, -1.0f, 1.0f);
            const float intensity = midVal + halfRange * (wx * wy);
            buffer[static_cast<size_t>(y * static_cast<int>(width) + x)] =
                static_cast<uint8_t>(std::clamp(intensity, 0.0f, 255.0f));
        }
    }
}

// Renders a 1D horizontal step edge across the image
void drawHorizontalEdge(std::vector<uint8_t> &buffer, size_t width, size_t height, int edgeY,
                        uint8_t topVal, uint8_t botVal) {
    for (size_t y = 0; y < height; y++) {
        const uint8_t val = (static_cast<int>(y) < edgeY) ? topVal : botVal;
        std::fill_n(&buffer[y * width], width, val);
    }
}

} // namespace

class HfTrackerReplenishTest : public ::testing::Test {
protected:
    static constexpr size_t kImageWidth = 640;
    static constexpr size_t kImageHeight = 480;
    static constexpr size_t kTargetFeatures = 200;

    static constexpr uint8_t kDefaultBackground = 128;
    static constexpr uint8_t kDarkIntensity = 30;
    static constexpr uint8_t kBrightIntensity = 220;
    static constexpr int kCornerRadius = 12;

    static constexpr int RADIUS = HfTracker::RADIUS;
    using Feature = TrackedFeature<RADIUS>;

    Eigen::Matrix3f K;
    Distortion dist;
    std::unique_ptr<HfTracker> tracker;
    std::vector<uint8_t> frameBuffer;

    void SetUp() override {
        K = Eigen::Matrix3f::Identity();
        K(0, 0) = 525.0f;
        K(1, 1) = 525.0f;
        K(0, 2) = 320.0f;
        K(1, 2) = 240.0f;

        dist = Distortion{};
        tracker = std::make_unique<HfTracker>(kImageWidth, kImageHeight, K, dist, kTargetFeatures);
        frameBuffer.assign(kImageWidth * kImageHeight, kDefaultBackground);
    }

    // Grid dimension and geometry accessors
    size_t getWBucket() const { return tracker->wBucket; }
    size_t getHBucket() const { return tracker->hBucket; }
    size_t getBRows() const { return tracker->bRows; }
    size_t getBCols() const { return tracker->bCols; }
    size_t getTotalBuckets() const { return tracker->totalBuckets; }

    // Internal buffer accessors
    std::vector<Eigen::Vector2i> &getFastPoints() { return tracker->fastPoints; }
    const std::vector<Eigen::Vector2i> &getFastPoints() const { return tracker->fastPoints; }

    std::vector<GridBucket> &getGridOccupancy() { return tracker->gridOccupancy; }
    const std::vector<GridBucket> &getGridOccupancy() const { return tracker->gridOccupancy; }

    std::vector<Feature> &getActiveFeatures() { return tracker->activeFeatures; }
    const std::vector<Feature> &getActiveFeatures() const { return tracker->activeFeatures; }

    // Private method wrappers
    Eigen::Vector2i callTopLeftBucket(size_t bucket) const {
        return tracker->topLeftBucket(bucket);
    }

    size_t callOffsetFromBucket(size_t bucket) const {
        return tracker->offsetFromBucket(bucket);
    }

    size_t callGetFast(const uint8_t *curBuffer, size_t bucket) {
        return tracker->getFast(curBuffer, bucket);
    }

    std::optional<Eigen::Vector2i> callBestKp(const uint8_t *curBuffer) {
        return tracker->bestKp(curBuffer);
    }

    bool callAddKp(const uint8_t *curBuffer, Eigen::Vector2i &kp) {
        return tracker->addKp(curBuffer, kp);
    }

    size_t callReplenishBuckets(const uint8_t *curBuffer) {
        return tracker->replenishBuckets(curBuffer);
    }
};

// ============================================================================
// 1. FAST Detection Tests (getFast)
// ============================================================================

// Verifies that getFast identifies FAST-9 keypoints on a 2D corner using bucket geometry helpers.
TEST_F(HfTrackerReplenishTest, getFastIdentifiesValidFast9Corners) {
    const size_t targetBucket = 10;
    const Eigen::Vector2i topLeft = callTopLeftBucket(targetBucket);

    const float cx = static_cast<float>(topLeft.x() + static_cast<int>(getWBucket() / 2));
    const float cy = static_cast<float>(topLeft.y() + static_cast<int>(getHBucket() / 2));

    drawSmoothCorner(frameBuffer, kImageWidth, kImageHeight, cx, cy,
                     kCornerRadius, kDarkIntensity, kBrightIntensity);

    const size_t nPts = callGetFast(frameBuffer.data(), targetBucket);

    EXPECT_GT(nPts, 0u);
    EXPECT_EQ(getFastPoints().size(), nPts);

    bool foundNearCenter = false;
    for (const auto &pt : getFastPoints()) {
        if (std::abs(static_cast<float>(pt.x()) - cx) <= 3.0f && std::abs(static_cast<float>(pt.y()) - cy) <= 3.0f) {
            foundNearCenter = true;
            break;
        }
    }
    EXPECT_TRUE(foundNearCenter);
}

// Verifies that getFast rejects flat regions and 1D horizontal step edges.
TEST_F(HfTrackerReplenishTest, getFastRejectsEdgesAndFlatRegions) {
    const size_t targetBucket = 12;
    const Eigen::Vector2i topLeft = callTopLeftBucket(targetBucket);
    const int edgeY = topLeft.y() + static_cast<int>(getHBucket() / 2);

    drawHorizontalEdge(frameBuffer, kImageWidth, kImageHeight, edgeY, kDarkIntensity, kBrightIntensity);

    const size_t nPts = callGetFast(frameBuffer.data(), targetBucket);

    EXPECT_EQ(nPts, 0u);
    EXPECT_TRUE(getFastPoints().empty());
}

// ============================================================================
// 2. Sub-Pixel Shi-Tomasi Selection Tests (bestKp)
// ============================================================================

// Verifies that bestKp selects the candidate point with the highest Shi-Tomasi minimum eigenvalue.
TEST_F(HfTrackerReplenishTest, bestKpSelectsStrongestCorner) {
    const Eigen::Vector2i weakPt(100, 100);
    drawSmoothCorner(frameBuffer, kImageWidth, kImageHeight, 100.0f, 100.0f,
                     kCornerRadius, 100, 150, 4.0f);

    const Eigen::Vector2i strongPt(200, 200);
    drawSmoothCorner(frameBuffer, kImageWidth, kImageHeight, 200.0f, 200.0f,
                     kCornerRadius, 10, 240, 1.0f);

    getFastPoints().push_back(weakPt);
    getFastPoints().push_back(strongPt);

    auto best = callBestKp(frameBuffer.data());

    ASSERT_TRUE(best.has_value());
    EXPECT_EQ(best->x(), strongPt.x());
    EXPECT_EQ(best->y(), strongPt.y());
}

// Verifies that candidate keypoints with eigenvalues below ST_THRESHOLD are discarded.
TEST_F(HfTrackerReplenishTest, bestKpRejectsBelowThreshold) {
    const Eigen::Vector2i edgePt(150, 150);
    drawHorizontalEdge(frameBuffer, kImageWidth, kImageHeight, edgePt.y(),
                     kDarkIntensity, kBrightIntensity);

    getFastPoints().push_back(edgePt);

    auto best = callBestKp(frameBuffer.data());
    EXPECT_FALSE(best.has_value());
}

// ============================================================================
// 3. Feature Registration Tests (addKp)
// ============================================================================

// Verifies that addKp extracts patches, precomputes gradients, inverts the Hessian, and updates activeFeatures.
TEST_F(HfTrackerReplenishTest, addKpPopulatesFeatureDataAndHessian) {
    Eigen::Vector2i cornerPt(300, 240);
    drawSmoothCorner(frameBuffer, kImageWidth, kImageHeight, 300.0f, 240.0f,
                     kCornerRadius, kDarkIntensity, kBrightIntensity);

    const size_t initialCount = getActiveFeatures().size();

    bool added = callAddKp(frameBuffer.data(), cornerPt);

    ASSERT_TRUE(added);
    ASSERT_EQ(getActiveFeatures().size(), initialCount + 1u);

    const auto &newFeat = getActiveFeatures().back();
    EXPECT_NEAR(newFeat.ptCurr.x(), 300.0f, 1e-4f);
    EXPECT_NEAR(newFeat.ptCurr.y(), 240.0f, 1e-4f);
    EXPECT_FALSE(newFeat.HInv.isZero(0.0f));

    bool hasNonZeroPatch = false;
    constexpr size_t kPatchElements = Feature::PATCH_SIZE * Feature::PATCH_SIZE;
    for (size_t i = 0; i < kPatchElements; i++) {
        if (newFeat.patch[i] > 0) {
            hasNonZeroPatch = true;
            break;
        }
    }
    EXPECT_TRUE(hasNonZeroPatch);
}

// ============================================================================
// 4. End-to-End Grid Replenishment Tests (replenishBuckets)
// ============================================================================

// Verifies that passing a null buffer returns 0 without side effects.
TEST_F(HfTrackerReplenishTest, replenishBucketsNullBufferReturnsZero) {
    EXPECT_EQ(callReplenishBuckets(nullptr), 0u);
}

// Verifies that replenishBuckets detects and populates features strictly into empty buckets.
TEST_F(HfTrackerReplenishTest, replenishBucketsFillsEmptyBucketsOnly) {
    auto &grid = getGridOccupancy();

    for (auto &bucket : grid) {
        bucket.count = 1;
    }

    grid[5].count = 0;
    grid[15].count = 0;

    const Eigen::Vector2i tl5 = callTopLeftBucket(5);
    drawSmoothCorner(frameBuffer, kImageWidth, kImageHeight,
                     static_cast<float>(tl5.x() + static_cast<int>(getWBucket() / 2)),
                     static_cast<float>(tl5.y() + static_cast<int>(getHBucket() / 2)),
                     kCornerRadius, kDarkIntensity, kBrightIntensity);

    const Eigen::Vector2i tl15 = callTopLeftBucket(15);
    drawSmoothCorner(frameBuffer, kImageWidth, kImageHeight,
                     static_cast<float>(tl15.x() + static_cast<int>(getWBucket() / 2)),
                     static_cast<float>(tl15.y() + static_cast<int>(getHBucket() / 2)),
                     kCornerRadius, kDarkIntensity, kBrightIntensity);

    const size_t initialActive = getActiveFeatures().size();

    const size_t occupied = callReplenishBuckets(frameBuffer.data());

    EXPECT_EQ(occupied, grid.size());
    EXPECT_EQ(getActiveFeatures().size(), initialActive + 2u);
}

} // namespace PoolSLAM