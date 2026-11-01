#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>
#include <algorithm>

#include <gtest/gtest.h>
#include <eigen3/Eigen/Dense>

#include "CoreTypes.hpp"
#include "frontend/HfTracker.hpp"

namespace PoolSLAM {
namespace {

// Smooth continuous corner to eliminate discrete aliasing
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

void drawHorizontalEdge(std::vector<uint8_t> &buffer, size_t width, size_t height, int edgeY,
                        uint8_t topVal, uint8_t botVal) {
    for (size_t y = 0; y < height; y++) {
        uint8_t val = (static_cast<int>(y) < edgeY) ? topVal : botVal;
        for (size_t x = 0; x < width; x++) {
            buffer[y * width + x] = val;
        }
    }
}

} // namespace

class HfTrackerTest : public ::testing::Test {
protected:
    // Core camera and image geometry
    static constexpr size_t kImageWidth = 640;
    static constexpr size_t kImageHeight = 480;
    static constexpr size_t kTargetFeatures = 200;

    static constexpr float kFx = 525.0f;
    static constexpr float kFy = 525.0f;
    static constexpr float kCx = static_cast<float>(kImageWidth) / 2.0f;
    static constexpr float kCy = static_cast<float>(kImageHeight) / 2.0f;

    // Numerical tolerances
    static constexpr float kFloatEps = 1e-4f;
    static constexpr float kSubPixelAlignmentTol = 0.05f;

    // Synthetic rendering parameters
    static constexpr uint8_t kDefaultBackground = 128;
    static constexpr uint8_t kDarkIntensity = 30;
    static constexpr uint8_t kBrightIntensity = 220;
    static constexpr int kCornerRadius = 12;

    // Inherited private constants from HfTracker via friendship
    static constexpr int RADIUS = HfTracker::RADIUS;
    static constexpr size_t MIN_MARGIN = HfTracker::MIN_MARGIN;
    static constexpr int RANSAC_MIN_INLIERS = HfTracker::RANSAC_MIN_INLIERS;
    static constexpr float RANSAC_EARLY_CONSENSUS = HfTracker::RANSAC_EARLY_CONSENSUS;
    static constexpr float MIN_DIST_IN_BUCKET = HfTracker::MIN_DIST_IN_BUCKET;
    static constexpr float HESSIAN_STABILITY_THRESHOLD = HfTracker::HESSIAN_STABILITY_THRESHOLD;

    using Feature = TrackedFeature<RADIUS>;

    Eigen::Matrix3f K;
    Distortion dist;
    std::unique_ptr<HfTracker> tracker;
    std::vector<uint8_t> frameBuffer;

    void SetUp() override {
        K = Eigen::Matrix3f::Identity();
        K(0, 0) = kFx;
        K(1, 1) = kFy;
        K(0, 2) = kCx;
        K(1, 2) = kCy;

        dist = Distortion{};

        tracker = std::make_unique<HfTracker>(kImageWidth, kImageHeight, K, dist, kTargetFeatures);
        frameBuffer.assign(kImageWidth * kImageHeight, kDefaultBackground);
    }

    std::unique_ptr<HfTracker> createTracker() const {
        return std::make_unique<HfTracker>(kImageWidth, kImageHeight, K, dist, kTargetFeatures);
    }

    // Grid geometry inspection accessors
    size_t getWBucket(const HfTracker &t) const { return t.wBucket; }
    size_t getWBucket() const { return getWBucket(*tracker); }

    size_t getHBucket(const HfTracker &t) const { return t.hBucket; }
    size_t getHBucket() const { return getHBucket(*tracker); }

    size_t getBRows(const HfTracker &t) const { return t.bRows; }
    size_t getBRows() const { return getBRows(*tracker); }

    size_t getBCols(const HfTracker &t) const { return t.bCols; }
    size_t getBCols() const { return getBCols(*tracker); }

    size_t getXMargin(const HfTracker &t) const { return t.xMargin; }
    size_t getXMargin() const { return getXMargin(*tracker); }

    size_t getYMargin(const HfTracker &t) const { return t.yMargin; }
    size_t getYMargin() const { return getYMargin(*tracker); }

    size_t getTotalBuckets(const HfTracker &t) const { return t.totalBuckets; }
    size_t getTotalBuckets() const { return getTotalBuckets(*tracker); }

    // Internal data structures accessors
    std::vector<Feature> &getActiveFeatures(HfTracker &t) { return t.activeFeatures; }
    std::vector<Feature> &getActiveFeatures() { return getActiveFeatures(*tracker); }

    const std::vector<Feature> &getActiveFeatures(const HfTracker &t) const { return t.activeFeatures; }
    const std::vector<Feature> &getActiveFeatures() const { return getActiveFeatures(*tracker); }

    const std::vector<GridBucket> &getGridOccupancy(const HfTracker &t) const { return t.gridOccupancy; }
    const std::vector<GridBucket> &getGridOccupancy() const { return getGridOccupancy(*tracker); }

    const std::vector<uint8_t> &getKeep(const HfTracker &t) const { return t.keep; }
    const std::vector<uint8_t> &getKeep() const { return getKeep(*tracker); }

    // Protected wrappers for invoking private methods
    size_t callGetBucketIdx(const HfTracker &t, double x, double y) const {
        return t.getBucketIdx(x, y);
    }
    size_t callGetBucketIdx(double x, double y) const {
        return callGetBucketIdx(*tracker, x, y);
    }

    size_t callOffsetFromBucket(const HfTracker &t, size_t bucket) const {
        return t.offsetFromBucket(bucket);
    }
    size_t callOffsetFromBucket(size_t bucket) const {
        return callOffsetFromBucket(*tracker, bucket);
    }

    Eigen::Vector2i callTopLeftBucket(const HfTracker &t, size_t bucket) const {
        return t.topLeftBucket(bucket);
    }
    Eigen::Vector2i callTopLeftBucket(size_t bucket) const {
        return callTopLeftBucket(*tracker, bucket);
    }

    int16_t callBilinear(const HfTracker &t, const uint8_t *curBuffer, const Eigen::Vector2f &pt) const {
        return t.bilinear(curBuffer, pt);
    }
    int16_t callBilinear(const uint8_t *curBuffer, const Eigen::Vector2f &pt) const {
        return callBilinear(*tracker, curBuffer, pt);
    }

    bool callAlignFeatureICLK(HfTracker &t, Feature &feature, const uint8_t *curBuffer, Eigen::Vector2f &ptEst) {
        return t.alignFeatureICLK(feature, curBuffer, ptEst);
    }
    bool callAlignFeatureICLK(Feature &feature, const uint8_t *curBuffer, Eigen::Vector2f &ptEst) {
        return callAlignFeatureICLK(*tracker, feature, curBuffer, ptEst);
    }

    void callUndistort(HfTracker &t, std::vector<float> &xDist, std::vector<float> &yDist, Eigen::Matrix3Xf &ptUndist, size_t count) {
        t.undistort(xDist, yDist, ptUndist, count);
    }
    void callUndistort(std::vector<float> &xDist, std::vector<float> &yDist, Eigen::Matrix3Xf &ptUndist, size_t count) {
        callUndistort(*tracker, xDist, yDist, ptUndist, count);
    }

    int callRansac(HfTracker &t, const Eigen::Matrix3f &R) {
        return t.ransac(R);
    }
    int callRansac(const Eigen::Matrix3f &R) {
        return callRansac(*tracker, R);
    }

    void callEnforceBucketDensity(HfTracker &t) {
        t.enforceBucketDensity();
    }
    void callEnforceBucketDensity() {
        callEnforceBucketDensity(*tracker);
    }

    bool callAddKp(HfTracker &t, const uint8_t *curBuffer, Eigen::Vector2i &kp) {
        return t.addKp(curBuffer, kp);
    }
    bool callAddKp(const uint8_t *curBuffer, Eigen::Vector2i &kp) {
        return callAddKp(*tracker, curBuffer, kp);
    }
};

// ============================================================================
// 1. Spatial Grid Bucketing Geometry & Indexing Tests (Private Methods)
// ============================================================================

TEST_F(HfTrackerTest, bucketGeometrySnappingAndInitialization) {
    constexpr size_t kNeonVectorAlignmentBytes = 16;

    EXPECT_GT(getWBucket(), 0);
    EXPECT_GT(getHBucket(), 0);
    EXPECT_EQ(getWBucket() % kNeonVectorAlignmentBytes, 0);
    EXPECT_GE(getXMargin(), MIN_MARGIN);
    EXPECT_GE(getYMargin(), MIN_MARGIN);
    EXPECT_EQ(getTotalBuckets(), getBCols() * getBRows());
    EXPECT_EQ(getGridOccupancy().size(), getTotalBuckets());
}

TEST_F(HfTrackerTest, getBucketIdxNominalAndCorners) {
    constexpr double kSubPixelCenterOffset = 0.5;

    const double xMarg = static_cast<double>(getXMargin());
    const double yMarg = static_cast<double>(getYMargin());
    const double wB = static_cast<double>(getWBucket());
    const double hB = static_cast<double>(getHBucket());

    size_t idxOrigin = callGetBucketIdx(xMarg + kSubPixelCenterOffset, yMarg + kSubPixelCenterOffset);
    EXPECT_EQ(idxOrigin, 0);

    size_t idxCol1 = callGetBucketIdx(xMarg + wB + kSubPixelCenterOffset, yMarg + kSubPixelCenterOffset);
    EXPECT_EQ(idxCol1, 1);

    size_t idxRow1 = callGetBucketIdx(xMarg + kSubPixelCenterOffset, yMarg + hB + kSubPixelCenterOffset);
    EXPECT_EQ(idxRow1, getBCols());

    const double xMax = xMarg + (static_cast<double>(getBCols()) * wB) - kSubPixelCenterOffset;
    const double yMax = yMarg + (static_cast<double>(getBRows()) * hB) - kSubPixelCenterOffset;
    size_t idxLast = callGetBucketIdx(xMax, yMax);
    EXPECT_EQ(idxLast, getTotalBuckets() - 1);
}

TEST_F(HfTrackerTest, offsetFromBucketAndTopLeftConsistency) {
    constexpr double kSubPixelInteriorOffset = 1.0;

    for (size_t b = 0; b < getTotalBuckets(); b++) {
        Eigen::Vector2i topLeft = callTopLeftBucket(b);
        size_t expectedOffset = static_cast<size_t>(topLeft.y()) * kImageWidth + static_cast<size_t>(topLeft.x());
        size_t actualOffset = callOffsetFromBucket(b);
        EXPECT_EQ(actualOffset, expectedOffset);

        size_t recoveredIdx = callGetBucketIdx(static_cast<double>(topLeft.x()) + kSubPixelInteriorOffset,
                                               static_cast<double>(topLeft.y()) + kSubPixelInteriorOffset);
        EXPECT_EQ(recoveredIdx, b);
    }
}

// ============================================================================
// 2. Sub-Pixel Bilinear Interpolation Kernel Tests (Private Method)
// ============================================================================

TEST_F(HfTrackerTest, bilinearExactIntegerCoordinate) {
    constexpr size_t kSampleX = 120;
    constexpr size_t kSampleY = 80;
    constexpr uint8_t kExpectedPixelVal = 195;

    frameBuffer[kSampleY * kImageWidth + kSampleX] = kExpectedPixelVal;

    int16_t sample = callBilinear(frameBuffer.data(), Eigen::Vector2f(static_cast<float>(kSampleX), static_cast<float>(kSampleY)));
    EXPECT_EQ(sample, static_cast<int16_t>(kExpectedPixelVal));
}

TEST_F(HfTrackerTest, bilinearHorizontalAndVerticalMidpoints) {
    constexpr size_t kSampleX = 200;
    constexpr size_t kSampleY = 150;
    constexpr size_t kSampleXNext = kSampleX + 1;
    constexpr size_t kSampleYNext = kSampleY + 1;

    constexpr uint8_t kValLeft = 100;
    constexpr uint8_t kValRight = 200;
    constexpr int16_t kExpectedMidValH = static_cast<int16_t>((static_cast<int>(kValLeft) + static_cast<int>(kValRight)) / 2);

    constexpr uint8_t kValTop = 100;
    constexpr uint8_t kValBottom = 160;
    constexpr int16_t kExpectedMidValV = static_cast<int16_t>((static_cast<int>(kValTop) + static_cast<int>(kValBottom)) / 2);

    constexpr float kHalfStep = 0.5f;

    frameBuffer[kSampleY * kImageWidth + kSampleX] = kValLeft;
    frameBuffer[kSampleY * kImageWidth + kSampleXNext] = kValRight;
    frameBuffer[kSampleYNext * kImageWidth + kSampleX] = kValLeft;
    frameBuffer[kSampleYNext * kImageWidth + kSampleXNext] = kValRight;

    int16_t sampleH = callBilinear(frameBuffer.data(), Eigen::Vector2f(static_cast<float>(kSampleX) + kHalfStep, static_cast<float>(kSampleY)));
    EXPECT_NEAR(sampleH, kExpectedMidValH, 1);

    frameBuffer[kSampleY * kImageWidth + kSampleX] = kValTop;
    frameBuffer[kSampleYNext * kImageWidth + kSampleX] = kValBottom;

    int16_t sampleV = callBilinear(frameBuffer.data(), Eigen::Vector2f(static_cast<float>(kSampleX), static_cast<float>(kSampleY) + kHalfStep));
    EXPECT_NEAR(sampleV, kExpectedMidValV, 1);
}

// ============================================================================
// 3. Feature Registration & Hessian Stability Tests (Private Method: addKp)
// ============================================================================

TEST_F(HfTrackerTest, addKpDegenerateFlatPatchRejected) {
    constexpr int kCandidateX = 250;
    constexpr int kCandidateY = 180;
    Eigen::Vector2i candidate(kCandidateX, kCandidateY);

    bool added = callAddKp(frameBuffer.data(), candidate);

    EXPECT_FALSE(added);
    EXPECT_TRUE(getActiveFeatures().empty());
}

TEST_F(HfTrackerTest, addKpApertureProblem1DEdgeRejected) {
    constexpr int kEdgeRowY = 200;
    constexpr int kCandidateX = 160;
    Eigen::Vector2i edgeKp(kCandidateX, kEdgeRowY);

    drawHorizontalEdge(frameBuffer, kImageWidth, kImageHeight, kEdgeRowY, kDarkIntensity, kBrightIntensity);

    bool added = callAddKp(frameBuffer.data(), edgeKp);

    EXPECT_FALSE(added);
    EXPECT_TRUE(getActiveFeatures().empty());
}

TEST_F(HfTrackerTest, addKpHighContrastCornerAccepted) {
    constexpr int kCornerX = 300;
    constexpr int kCornerY = 240;
    Eigen::Vector2i cornerKp(kCornerX, kCornerY);

    drawSmoothCorner(frameBuffer, kImageWidth, kImageHeight, static_cast<float>(cornerKp.x()),
                     static_cast<float>(cornerKp.y()), kCornerRadius, kDarkIntensity, kBrightIntensity);

    bool added = callAddKp(frameBuffer.data(), cornerKp);

    ASSERT_TRUE(added);
    ASSERT_EQ(getActiveFeatures().size(), 1);

    const auto &feature = getActiveFeatures().front();
    EXPECT_EQ(feature.id, 0);
    EXPECT_EQ(feature.age, 0);
    EXPECT_NEAR(feature.ptCurr.x(), static_cast<float>(cornerKp.x()), kFloatEps);
    EXPECT_NEAR(feature.ptCurr.y(), static_cast<float>(cornerKp.y()), kFloatEps);
}

// ============================================================================
// 4. Inverse Compositional LK Alignment Tests (Private Method: alignFeatureICLK)
// ============================================================================

TEST_F(HfTrackerTest, alignFeatureICLKZeroDisplacementExactConvergence) {
    constexpr int kCornerX = 220;
    constexpr int kCornerY = 180;
    Eigen::Vector2i cornerKp(kCornerX, kCornerY);

    drawSmoothCorner(frameBuffer, kImageWidth, kImageHeight, static_cast<float>(cornerKp.x()),
                     static_cast<float>(cornerKp.y()), kCornerRadius, kDarkIntensity, kBrightIntensity);

    ASSERT_TRUE(callAddKp(frameBuffer.data(), cornerKp));
    auto &feature = getActiveFeatures().front();

    Eigen::Vector2f ptEstimate = feature.ptCurr;
    bool aligned = callAlignFeatureICLK(feature, frameBuffer.data(), ptEstimate);

    EXPECT_TRUE(aligned);
    EXPECT_NEAR(ptEstimate.x(), feature.ptCurr.x(), kSubPixelAlignmentTol);
    EXPECT_NEAR(ptEstimate.y(), feature.ptCurr.y(), kSubPixelAlignmentTol);
}

TEST_F(HfTrackerTest, alignFeatureICLKSubPixelShiftConvergence) {
    constexpr int kCornerX = 310;
    constexpr int kCornerY = 210;
    constexpr float kSubPixelDisplacementX = 0.35f;
    constexpr float kSubPixelDisplacementY = -0.25f;

    Eigen::Vector2i initialKp(kCornerX, kCornerY);
    drawSmoothCorner(frameBuffer, kImageWidth, kImageHeight, static_cast<float>(initialKp.x()),
                     static_cast<float>(initialKp.y()), kCornerRadius, kDarkIntensity, kBrightIntensity);

    ASSERT_TRUE(callAddKp(frameBuffer.data(), initialKp));
    auto &feature = getActiveFeatures().front();

    // Perturb initial search guess by realistic sub-pixel parallax displacement
    Eigen::Vector2f ptEstimate = feature.ptCurr + Eigen::Vector2f(kSubPixelDisplacementX, kSubPixelDisplacementY);

    bool aligned = callAlignFeatureICLK(feature, frameBuffer.data(), ptEstimate);

    EXPECT_TRUE(aligned);
    EXPECT_NEAR(ptEstimate.x(), feature.ptCurr.x(), kSubPixelAlignmentTol);
    EXPECT_NEAR(ptEstimate.y(), feature.ptCurr.y(), kSubPixelAlignmentTol);
}

TEST_F(HfTrackerTest, alignFeatureICLKMADThresholdRejectionOnSevereOcclusion) {
    constexpr int kCornerX = 180;
    constexpr int kCornerY = 140;
    constexpr uint8_t kOccludedIntensity = 0;

    Eigen::Vector2i cornerKp(kCornerX, kCornerY);
    drawSmoothCorner(frameBuffer, kImageWidth, kImageHeight, static_cast<float>(cornerKp.x()),
                     static_cast<float>(cornerKp.y()), kCornerRadius, kDarkIntensity, kBrightIntensity);

    ASSERT_TRUE(callAddKp(frameBuffer.data(), cornerKp));
    auto &feature = getActiveFeatures().front();

    std::vector<uint8_t> corruptedFrame(kImageWidth * kImageHeight, kOccludedIntensity);

    Eigen::Vector2f ptEstimate = feature.ptCurr;
    bool aligned = callAlignFeatureICLK(feature, corruptedFrame.data(), ptEstimate);

    EXPECT_FALSE(aligned);
}

// ============================================================================
// 5. Coordinate Undistortion Tests (Private Method: undistort)
// ============================================================================

TEST_F(HfTrackerTest, undistortZeroDistortionPinholeNormalizedRays) {
    std::vector<float> xDist = {kCx, kCx + kFx, kCx};
    std::vector<float> yDist = {kCy, kCy, kCy + kFy};
    const size_t count = xDist.size();

    Eigen::Matrix3Xf ptUndist(3, count);
    callUndistort(xDist, yDist, ptUndist, count);

    EXPECT_NEAR(ptUndist(0, 0), 0.0f, kFloatEps);
    EXPECT_NEAR(ptUndist(1, 0), 0.0f, kFloatEps);
    EXPECT_NEAR(ptUndist(2, 0), 1.0f, kFloatEps);

    EXPECT_NEAR(ptUndist(0, 1), 1.0f, kFloatEps);
    EXPECT_NEAR(ptUndist(1, 1), 0.0f, kFloatEps);
    EXPECT_NEAR(ptUndist(2, 1), 1.0f, kFloatEps);

    EXPECT_NEAR(ptUndist(0, 2), 0.0f, kFloatEps);
    EXPECT_NEAR(ptUndist(1, 2), 1.0f, kFloatEps);
    EXPECT_NEAR(ptUndist(2, 2), 1.0f, kFloatEps);
}

// ============================================================================
// 6. 2-Point Epipolar RANSAC Tests (Private Method: ransac)
// ============================================================================

TEST_F(HfTrackerTest, ransacPrunesInjectedOutliers) {
    constexpr size_t kNumInliers = 30;
    constexpr size_t kNumOutliers = 10;
    constexpr size_t kTotalFeatures = kNumInliers + kNumOutliers;

    constexpr float kRotAngleRad = 0.015f;
    const Eigen::Matrix3f R = Eigen::AngleAxisf(kRotAngleRad, Eigen::Vector3f::UnitX()).toRotationMatrix();
    const Eigen::Vector3f tTrue = Eigen::Vector3f(0.08f, -0.04f, 0.20f).normalized();

    constexpr float kStartNormX = -0.3f;
    constexpr float kStartNormY = -0.2f;
    constexpr float kSpacingNorm = 0.08f;
    constexpr size_t kGridDim = 8;
    constexpr float kBaseDepthMeters = 2.0f;
    constexpr float kDepthStepMeters = 0.4f;

    std::mt19937 outlierRng(42);
    std::uniform_real_distribution<float> outlierDist(40.0f, 90.0f);
    std::uniform_real_distribution<float> signDist(-1.0f, 1.0f);

    for (size_t i = 0; i < kTotalFeatures; i++) {
        float xn = kStartNormX + (static_cast<float>(i % kGridDim) * kSpacingNorm);
        float yn = kStartNormY + (static_cast<float>(static_cast<float>(i) / kGridDim) * kSpacingNorm);
        float depth = kBaseDepthMeters + static_cast<float>(i % 5) * kDepthStepMeters;

        Eigen::Vector3f p3D_prev(xn * depth, yn * depth, depth);
        Eigen::Vector3f rPrev = p3D_prev.normalized();

        Eigen::Vector3f p3D_curr = R * p3D_prev + tTrue * 0.15f;
        Eigen::Vector3f rCurr = p3D_curr / p3D_curr.z();

        Feature feat{};
        feat.id = static_cast<uint32_t>(i);
        feat.age = 2;
        feat.ptPrev = Eigen::Vector2f(kFx * rPrev.x() / rPrev.z() + kCx,
                                     kFy * rPrev.y() / rPrev.z() + kCy);
        feat.ptCurr = Eigen::Vector2f(kFx * rCurr.x() + kCx,
                                     kFy * rCurr.y() + kCy);

        if (i >= kNumInliers) {
            float dx = outlierDist(outlierRng) * (signDist(outlierRng) > 0.0f ? 1.0f : -1.0f);
            float dy = outlierDist(outlierRng) * (signDist(outlierRng) > 0.0f ? 1.0f : -1.0f);
            feat.ptCurr.x() += dx;
            feat.ptCurr.y() += dy;
        }

        getActiveFeatures().push_back(feat);
    }

    int inlierCount = callRansac(R);

    constexpr int kConsensusMargin = 2;
    EXPECT_GE(inlierCount, static_cast<int>(kNumInliers) - kConsensusMargin);
    EXPECT_LE(inlierCount, static_cast<int>(kNumInliers) + 1);
}

// ============================================================================
// 7. Bucket Density Enforcement & Collision Pruning (Private Method)
// ============================================================================

TEST_F(HfTrackerTest, densitySingleFeatureRetained) {
    constexpr float kOffsetInside = 8.0f;
    constexpr uint32_t kFeatureAge = 3;

    Feature feat{};
    feat.id = 1;
    feat.age = kFeatureAge;
    feat.ptCurr = Eigen::Vector2f(static_cast<float>(getXMargin()) + kOffsetInside,
                                 static_cast<float>(getYMargin()) + kOffsetInside);

    getActiveFeatures().push_back(feat);

    callEnforceBucketDensity();

    const auto &grid = getGridOccupancy();
    EXPECT_EQ(grid[0].count, 1);
    EXPECT_EQ(grid[0].olderIdx, 0);
    EXPECT_EQ(grid[0].youngerIdx, UINT32_MAX);
    EXPECT_EQ(getKeep()[0], 1);
}

TEST_F(HfTrackerTest, densityTwoFeaturesSeparatedBothRetainedAndSorted) {
    const float xBase = static_cast<float>(getXMargin());
    const float yBase = static_cast<float>(getYMargin());

    constexpr float kOffsetNear = 4.0f;
    constexpr float kSeparationDistance = MIN_DIST_IN_BUCKET + 6.0f;
    constexpr float kOffsetFar = kOffsetNear + kSeparationDistance;

    constexpr uint32_t kAgeOlder = 10;
    constexpr uint32_t kAgeYounger = 2;

    Feature featA{};
    featA.id = 101;
    featA.age = kAgeOlder;
    featA.ptCurr = Eigen::Vector2f(xBase + kOffsetNear, yBase + kOffsetNear);

    Feature featB{};
    featB.id = 102;
    featB.age = kAgeYounger;
    featB.ptCurr = Eigen::Vector2f(xBase + kOffsetFar, yBase + kOffsetFar);

    getActiveFeatures().push_back(featA);
    getActiveFeatures().push_back(featB);

    callEnforceBucketDensity();

    const auto &bucket0 = getGridOccupancy()[0];
    EXPECT_EQ(bucket0.count, 2);
    EXPECT_EQ(bucket0.olderIdx, 0);
    EXPECT_EQ(bucket0.youngerIdx, 1);
    EXPECT_EQ(getKeep()[0], 1);
    EXPECT_EQ(getKeep()[1], 1);
}

// ============================================================================
// 8. Public Interface End-to-End Tracking: Rotational & Translational Suites
// ============================================================================

TEST_F(HfTrackerTest, trackExistingEmptyActiveFeaturesReturnsZero) {
    const Eigen::Matrix3f R = Eigen::Matrix3f::Identity();
    size_t tracked = tracker->trackExistingFeatures(frameBuffer.data(), R);

    EXPECT_EQ(tracked, 0);
    EXPECT_TRUE(tracker->getActiveFeatures().empty());
}

TEST_F(HfTrackerTest, trackExistingFeaturesNominalRotationalLifecycle) {
    constexpr size_t kGridSeedCols = 5;
    constexpr size_t kGridSeedRows = 4;
    constexpr size_t kTotalSeedFeatures = kGridSeedCols * kGridSeedRows;
    constexpr size_t kOffsetInsideBucket = 10;

    std::vector<Eigen::Vector2i> seedCorners;
    seedCorners.reserve(kTotalSeedFeatures);

    for (size_t col = 1; col <= kGridSeedCols; col++) {
        for (size_t row = 1; row <= kGridSeedRows; row++) {
            int cx = static_cast<int>(getXMargin() + col * getWBucket() + kOffsetInsideBucket);
            int cy = static_cast<int>(getYMargin() + row * getHBucket() + kOffsetInsideBucket);
            drawSmoothCorner(frameBuffer, kImageWidth, kImageHeight,
                             static_cast<float>(cx), static_cast<float>(cy),
                             kCornerRadius, kDarkIntensity, kBrightIntensity);
            seedCorners.emplace_back(cx, cy);
        }
    }

    for (size_t i = 0; i < seedCorners.size(); i++) {
        ASSERT_TRUE(callAddKp(frameBuffer.data(), seedCorners[i]));
    }

    const size_t initialFeatureCount = tracker->getActiveFeatures().size();
    ASSERT_EQ(initialFeatureCount, kTotalSeedFeatures);

    // Couple a 0.001 rad pitch rotation with small 2 mm lateral translation
    // so epipolar normals are well-conditioned and RANSAC does not degenerate
    constexpr float kPitchRad = 0.001f;
    const Eigen::Matrix3f R = Eigen::AngleAxisf(kPitchRad, Eigen::Vector3f::UnitX()).toRotationMatrix();
    constexpr float kSceneDepthMeters = 3.0f;
    const Eigen::Vector3f t(0.002f, 0.0f, 0.0f); // 2 mm lateral shift

    std::vector<uint8_t> nextFrame(kImageWidth * kImageHeight, kDefaultBackground);
    for (const auto &corner : seedCorners) {
        Eigen::Vector3f P0((static_cast<float>(corner.x()) - kCx) * kSceneDepthMeters / kFx,
                           (static_cast<float>(corner.y()) - kCy) * kSceneDepthMeters / kFy,
                           kSceneDepthMeters);
        Eigen::Vector3f P1 = R * P0 + t;
        float x1 = (kFx * P1.x() / P1.z()) + kCx;
        float y1 = (kFy * P1.y() / P1.z()) + kCy;

        drawSmoothCorner(nextFrame, kImageWidth, kImageHeight, x1, y1,
                         kCornerRadius, kDarkIntensity, kBrightIntensity);
    }

    size_t survived = tracker->trackExistingFeatures(nextFrame.data(), R);

    EXPECT_EQ(survived, initialFeatureCount);
    EXPECT_EQ(tracker->getActiveFeatures().size(), initialFeatureCount);

    for (const auto &feat : tracker->getActiveFeatures()) {
        EXPECT_EQ(feat.age, 1);
        // Positive pitch tilts camera down -> landmarks move UP in pixel coords (y decreases)
        EXPECT_LT(feat.ptCurr.y(), feat.ptPrev.y());
    }
}

TEST_F(HfTrackerTest, trackExistingFeaturesPureLateralTranslation) {
    constexpr size_t kGridCols = 5;
    constexpr size_t kGridRows = 4;
    constexpr size_t kTotalFeatures = kGridCols * kGridRows;
    constexpr size_t kOffsetInsideBucket = 10;

    // 3 mm physical lateral translation at 2.5 m depth produces 0.63 px parallax:
    // delta_u = (525 * 0.003) / 2.5 = 0.63 px (ideal for single-scale IC-LK convergence)
    constexpr float kTxMeters = 0.003f;
    const Eigen::Vector3f t(kTxMeters, 0.0f, 0.0f);
    const Eigen::Matrix3f R = Eigen::Matrix3f::Identity();

    constexpr float kSceneDepthMeters = 2.5f;
    constexpr float kExpectedParallaxPx = (kFx * kTxMeters) / kSceneDepthMeters;

    std::vector<Eigen::Vector2f> displacedPoints;
    displacedPoints.reserve(kTotalFeatures);

    for (size_t col = 1; col <= kGridCols; col++) {
        for (size_t row = 1; row <= kGridRows; row++) {
            float x0 = static_cast<float>(getXMargin() + col * getWBucket() + kOffsetInsideBucket);
            float y0 = static_cast<float>(getYMargin() + row * getHBucket() + kOffsetInsideBucket);

            drawSmoothCorner(frameBuffer, kImageWidth, kImageHeight, x0, y0,
                             kCornerRadius, kDarkIntensity, kBrightIntensity);

            Eigen::Vector2i kp(static_cast<int>(std::round(x0)), static_cast<int>(std::round(y0)));
            ASSERT_TRUE(callAddKp(frameBuffer.data(), kp));

            Eigen::Vector3f P0((x0 - kCx) * kSceneDepthMeters / kFx,
                               (y0 - kCy) * kSceneDepthMeters / kFy,
                               kSceneDepthMeters);
            Eigen::Vector3f P1 = P0 + t;
            float x1 = (kFx * P1.x() / P1.z()) + kCx;
            float y1 = (kFy * P1.y() / P1.z()) + kCy;
            displacedPoints.emplace_back(x1, y1);
        }
    }

    ASSERT_EQ(tracker->getActiveFeatures().size(), kTotalFeatures);

    std::vector<uint8_t> nextFrame(kImageWidth * kImageHeight, kDefaultBackground);
    for (size_t i = 0; i < kTotalFeatures; i++) {
        drawSmoothCorner(nextFrame, kImageWidth, kImageHeight,
                         displacedPoints[i].x(), displacedPoints[i].y(),
                         kCornerRadius, kDarkIntensity, kBrightIntensity);
    }

    size_t survived = tracker->trackExistingFeatures(nextFrame.data(), R);

    EXPECT_EQ(survived, kTotalFeatures);
    EXPECT_EQ(tracker->getActiveFeatures().size(), kTotalFeatures);

    for (size_t i = 0; i < tracker->getActiveFeatures().size(); i++) {
        const auto &feat = tracker->getActiveFeatures()[i];
        EXPECT_EQ(feat.age, 1);
        EXPECT_NEAR(feat.ptCurr.x() - feat.ptPrev.x(), kExpectedParallaxPx, kSubPixelAlignmentTol);
        EXPECT_NEAR(feat.ptCurr.y() - feat.ptPrev.y(), 0.0f, kSubPixelAlignmentTol);
    }
}

TEST_F(HfTrackerTest, trackExistingFeaturesForwardTranslationLooming) {
    constexpr size_t kGridCols = 5;
    constexpr size_t kGridRows = 4;
    constexpr size_t kTotalFeatures = kGridCols * kGridRows;
    constexpr size_t kOffsetInsideBucket = 12;

    // Camera moves forward along +Z by 8 mm
    constexpr float kTzMeters = 0.008f;
    const Eigen::Vector3f t(0.0f, 0.0f, kTzMeters);
    const Eigen::Matrix3f R = Eigen::Matrix3f::Identity();

    constexpr float kInitialDepthMeters = 2.0f;
    constexpr float kRemainingDepthMeters = kInitialDepthMeters - kTzMeters; // 1.992 m
    constexpr float kExpansionScale = kInitialDepthMeters / kRemainingDepthMeters; // ~1.004016

    std::vector<Eigen::Vector2f> projectedCoords;
    projectedCoords.reserve(kTotalFeatures);

    for (size_t col = 1; col <= kGridCols; col++) {
        for (size_t row = 1; row <= kGridRows; row++) {
            float x0 = static_cast<float>(getXMargin() + col * getWBucket() + kOffsetInsideBucket);
            float y0 = static_cast<float>(getYMargin() + row * getHBucket() + kOffsetInsideBucket);

            drawSmoothCorner(frameBuffer, kImageWidth, kImageHeight, x0, y0,
                             kCornerRadius, kDarkIntensity, kBrightIntensity);

            Eigen::Vector2i kp(static_cast<int>(std::round(x0)), static_cast<int>(std::round(y0)));
            ASSERT_TRUE(callAddKp(frameBuffer.data(), kp));

            Eigen::Vector3f P0((x0 - kCx) * kInitialDepthMeters / kFx,
                               (y0 - kCy) * kInitialDepthMeters / kFy,
                               kInitialDepthMeters);

            // Forward camera motion moves scene points closer to camera: P1 = P0 - t
            Eigen::Vector3f P1 = P0 - t;
            float x1 = (kFx * P1.x() / P1.z()) + kCx;
            float y1 = (kFy * P1.y() / P1.z()) + kCy;
            projectedCoords.emplace_back(x1, y1);
        }
    }

    std::vector<uint8_t> nextFrame(kImageWidth * kImageHeight, kDefaultBackground);
    for (size_t i = 0; i < kTotalFeatures; i++) {
        drawSmoothCorner(nextFrame, kImageWidth, kImageHeight,
                         projectedCoords[i].x(), projectedCoords[i].y(),
                         kCornerRadius, kDarkIntensity, kBrightIntensity);
    }

    size_t survived = tracker->trackExistingFeatures(nextFrame.data(), R);

    EXPECT_EQ(survived, kTotalFeatures);
    EXPECT_EQ(tracker->getActiveFeatures().size(), kTotalFeatures);

    for (size_t i = 0; i < tracker->getActiveFeatures().size(); i++) {
        const auto &feat = tracker->getActiveFeatures()[i];
        float prevDist = (feat.ptPrev - Eigen::Vector2f(kCx, kCy)).norm();
        float currDist = (feat.ptCurr - Eigen::Vector2f(kCx, kCy)).norm();

        EXPECT_GT(currDist, prevDist);
        EXPECT_NEAR(currDist / prevDist, kExpansionScale, 0.005f);
    }
}

TEST_F(HfTrackerTest, trackExistingFeaturesCoupled6DoFMotion) {
    constexpr size_t kGridCols = 5;
    constexpr size_t kGridRows = 4;
    constexpr size_t kTotalFeatures = kGridCols * kGridRows;
    constexpr size_t kOffsetInsideBucket = 10;

    // Relative motion: 0.0006 rad yaw + 1 mm lateral + 1 mm forward translation
    constexpr float kYawRad = 0.0006f;
    const Eigen::Matrix3f R = Eigen::AngleAxisf(kYawRad, Eigen::Vector3f::UnitY()).toRotationMatrix();
    const Eigen::Vector3f t(0.001f, 0.0f, 0.001f);

    constexpr float kSceneDepthMeters = 2.5f;
    constexpr float kTrackingTol = 0.10f;

    std::vector<Eigen::Vector2f> projectedCoords;
    projectedCoords.reserve(kTotalFeatures);

    for (size_t col = 1; col <= kGridCols; col++) {
        for (size_t row = 1; row <= kGridRows; row++) {
            float x0 = static_cast<float>(getXMargin() + col * getWBucket() + kOffsetInsideBucket);
            float y0 = static_cast<float>(getYMargin() + row * getHBucket() + kOffsetInsideBucket);

            drawSmoothCorner(frameBuffer, kImageWidth, kImageHeight, x0, y0,
                             kCornerRadius, kDarkIntensity, kBrightIntensity);

            Eigen::Vector2i kp(static_cast<int>(std::round(x0)), static_cast<int>(std::round(y0)));
            ASSERT_TRUE(callAddKp(frameBuffer.data(), kp));

            Eigen::Vector3f P0((x0 - kCx) * kSceneDepthMeters / kFx,
                               (y0 - kCy) * kSceneDepthMeters / kFy,
                               kSceneDepthMeters);

            // Forward camera translation moves scene closer: P1 = R * (P0 - t)
            Eigen::Vector3f P1 = R * (P0 - t);
            float x1 = (kFx * P1.x() / P1.z()) + kCx;
            float y1 = (kFy * P1.y() / P1.z()) + kCy;
            projectedCoords.emplace_back(x1, y1);
        }
    }

    std::vector<uint8_t> nextFrame(kImageWidth * kImageHeight, kDefaultBackground);
    for (size_t i = 0; i < kTotalFeatures; i++) {
        drawSmoothCorner(nextFrame, kImageWidth, kImageHeight,
                         projectedCoords[i].x(), projectedCoords[i].y(),
                         kCornerRadius, kDarkIntensity, kBrightIntensity);
    }

    size_t survived = tracker->trackExistingFeatures(nextFrame.data(), R);

    EXPECT_EQ(survived, kTotalFeatures);
    EXPECT_EQ(tracker->getActiveFeatures().size(), kTotalFeatures);

    for (size_t i = 0; i < tracker->getActiveFeatures().size(); i++) {
        const auto &feat = tracker->getActiveFeatures()[i];
        EXPECT_EQ(feat.age, 1);
        EXPECT_NEAR(feat.ptCurr.x(), projectedCoords[i].x(), kTrackingTol);
        EXPECT_NEAR(feat.ptCurr.y(), projectedCoords[i].y(), kTrackingTol);
    }
}

TEST_F(HfTrackerTest, trackExistingFeaturesTranslationPrunesDynamicOutliers) {
    constexpr size_t kNumInliers = 20;
    constexpr size_t kNumOutliers = 5;
    constexpr size_t kTotalFeatures = kNumInliers + kNumOutliers;
    constexpr size_t kOffsetInsideBucket = 8;

    // 1.8 px inlier parallax along X -> Delta x = 0.00343 > ransacTau (0.002857)
    constexpr float kInlierParallaxPx = 1.8f;
    const Eigen::Matrix3f R = Eigen::Matrix3f::Identity();

    // Outlier vertical drifts exceeding tau_pixel (1.5 px) with alternating signs
    const std::array<float, kNumOutliers> kOutlierDy = {2.2f, -2.2f, 2.0f, -2.0f, 2.4f};
    constexpr float kTrackingTol = 0.10f;

    std::vector<Eigen::Vector2f> targetCoords;
    targetCoords.reserve(kTotalFeatures);

    for (size_t i = 0; i < kTotalFeatures; i++) {
        size_t col = (i % 5) + 1;
        size_t row = (i / 5) + 1;
        float x0 = static_cast<float>(getXMargin() + col * getWBucket() + kOffsetInsideBucket);
        float y0 = static_cast<float>(getYMargin() + row * getHBucket() + kOffsetInsideBucket);

        drawSmoothCorner(frameBuffer, kImageWidth, kImageHeight, x0, y0,
                         kCornerRadius, kDarkIntensity, kBrightIntensity);

        Eigen::Vector2i kp(static_cast<int>(std::round(x0)), static_cast<int>(std::round(y0)));
        ASSERT_TRUE(callAddKp(frameBuffer.data(), kp));

        float x1 = x0 + kInlierParallaxPx;
        float y1 = y0;

        if (i >= kNumInliers) {
            y1 += kOutlierDy[i - kNumInliers];
        }
        targetCoords.emplace_back(x1, y1);
    }

    std::vector<uint8_t> nextFrame(kImageWidth * kImageHeight, kDefaultBackground);
    for (size_t i = 0; i < kTotalFeatures; i++) {
        drawSmoothCorner(nextFrame, kImageWidth, kImageHeight,
                         targetCoords[i].x(), targetCoords[i].y(),
                         kCornerRadius, kDarkIntensity, kBrightIntensity);
    }

    size_t survived = tracker->trackExistingFeatures(nextFrame.data(), R);

    // 2-point RANSAC identifies t = [1, 0, 0]^T and prunes the 5 vertical outliers
    EXPECT_EQ(survived, kNumInliers);
    EXPECT_EQ(tracker->getActiveFeatures().size(), kNumInliers);

    for (const auto &feat : tracker->getActiveFeatures()) {
        EXPECT_EQ(feat.age, 1);
        EXPECT_NEAR(feat.ptCurr.y(), feat.ptPrev.y(), kTrackingTol);
    }
}

TEST_F(HfTrackerTest, trackExistingFeaturesLateralTranslationMigratesBuckets) {
    size_t originBucket = 0;
    Eigen::Vector2i bucket0TopLeft = callTopLeftBucket(originBucket);

    // Position feature 0.3 px to the left of the bucket 0 right boundary
    float boundaryX = static_cast<float>(bucket0TopLeft.x() + static_cast<int>(getWBucket()));
    float startX = boundaryX - 0.3f;
    float startY = static_cast<float>(bucket0TopLeft.y() + 10);

    drawSmoothCorner(frameBuffer, kImageWidth, kImageHeight, startX, startY,
                     kCornerRadius, kDarkIntensity, kBrightIntensity);

    Eigen::Vector2i kp(static_cast<int>(std::round(startX)), static_cast<int>(std::round(startY)));
    ASSERT_TRUE(callAddKp(frameBuffer.data(), kp));

    // Pad with 20 anchor features spaced across other buckets to meet RANSAC_MIN_INLIERS
    for (size_t i = 1; i <= 21; i++) {
        Eigen::Vector2i tl = callTopLeftBucket(i);
        float x = static_cast<float>(tl.x() + 8);
        float y = static_cast<float>(tl.y() + 8);
        drawSmoothCorner(frameBuffer, kImageWidth, kImageHeight, x, y,
                         kCornerRadius, kDarkIntensity, kBrightIntensity);
        Eigen::Vector2i extraKp(static_cast<int>(x), static_cast<int>(y));
        ASSERT_TRUE(callAddKp(frameBuffer.data(), extraKp));
    }

    // Translate rightward by 0.6 px:
    // startX (boundaryX - 0.3) + 0.6 -> boundaryX + 0.3 (migrates into bucket 1)
    constexpr float kMigrationDisplacementPx = 0.6f;
    std::vector<uint8_t> nextFrame(kImageWidth * kImageHeight, kDefaultBackground);

    for (size_t i = 0; i < tracker->getActiveFeatures().size(); i++) {
        float x = tracker->getActiveFeatures()[i].ptCurr.x() + kMigrationDisplacementPx;
        float y = tracker->getActiveFeatures()[i].ptCurr.y();
        drawSmoothCorner(nextFrame, kImageWidth, kImageHeight, x, y,
                         kCornerRadius, kDarkIntensity, kBrightIntensity);
    }

    const Eigen::Matrix3f R = Eigen::Matrix3f::Identity();
    size_t survived = tracker->trackExistingFeatures(nextFrame.data(), R);

    EXPECT_GT(survived, 0);

    // Verify feature successfully crossed into bucket 1
    const auto &migratedFeature = tracker->getActiveFeatures()[0];
    size_t newBucketIdx = callGetBucketIdx(migratedFeature.ptCurr.x(), migratedFeature.ptCurr.y());
    EXPECT_EQ(newBucketIdx, 1);
}

} // namespace PoolSLAM