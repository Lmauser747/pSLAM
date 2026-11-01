#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>
#include <algorithm>
#include <random>

#include <eigen3/Eigen/Dense>
#include <gtest/gtest.h>

#include "CoreTypes.hpp"
#include "frontend/HfTracker.hpp"

namespace PoolSLAM {
namespace {

// Visual rendering helper for artificial corner generation (checkerboard)
void drawCorner(std::vector<uint8_t> &buffer, size_t width, size_t height, int cx, int cy, int radius,
                uint8_t darkVal, uint8_t brightVal) {
    constexpr float kRampWidth = 1.5f; // Transition width in pixels
    const float midVal = 0.5f * (static_cast<float>(darkVal) + static_cast<float>(brightVal));
    const float halfRange = 0.5f * (static_cast<float>(brightVal) - static_cast<float>(darkVal));

    const int minX = std::max(0, static_cast<int>(cx) - radius);
    const int maxX = std::min(static_cast<int>(width) - 1, static_cast<int>(cx) + radius);
    const int minY = std::max(0, static_cast<int>(cy) - radius);
    const int maxY = std::min(static_cast<int>(height) - 1, static_cast<int>(cy) + radius);

    for (int y = minY; y <= maxY; y++) {
        const float dy = (static_cast<float>(y) - static_cast<float>(cy)) / kRampWidth;
        const float wy = std::clamp(dy, -1.0f, 1.0f);
        for (int x = minX; x <= maxX; x++) {
            const float dx = (static_cast<float>(x) - static_cast<float>(cx)) / kRampWidth;
            const float wx = std::clamp(dx, -1.0f, 1.0f);
            
            // Continuous 2D quadrant modulation: -1 in quadrant II/IV, +1 in I/III
            const float intensity = midVal + halfRange * (wx * wy);
            buffer[static_cast<size_t>(y * static_cast<int>(width) + x)] = 
                static_cast<uint8_t>(std::clamp(intensity, 0.0f, 255.0f));
        }
    }
}

// Visual rendering helper for step edge generation
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
    // Core image dimensions
    static constexpr size_t kImageWidth = 640;
    static constexpr size_t kImageHeight = 480;
    static constexpr size_t kTargetFeatures = 200;

    // Derived camera calibration intrinsics (Pinhole model)
    static constexpr float kFx = 525.0f;
    static constexpr float kFy = 525.0f;
    static constexpr float kCx = static_cast<float>(kImageWidth) / 2.0f;
    static constexpr float kCy = static_cast<float>(kImageHeight) / 2.0f;

    // Tolerances and numerical thresholds
    static constexpr float kFloatEps = 1e-4f;
    static constexpr float kZeroDisplacementTol = 0.05f;
    static constexpr float kSubPixelAlignmentTol = 0.15f;

    // Synthetic rendering constants
    static constexpr uint8_t kDefaultBackground = 128;
    static constexpr uint8_t kDarkIntensity = 30;
    static constexpr uint8_t kBrightIntensity = 220;
    static constexpr int kCornerRadius = 12;

    // Aliases to access private HfTracker constants within the friend fixture
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

    // Bucket 0 (top-left)
    size_t idxOrigin = callGetBucketIdx(xMarg + kSubPixelCenterOffset, yMarg + kSubPixelCenterOffset);
    EXPECT_EQ(idxOrigin, 0);

    // Column 1, Row 0
    size_t idxCol1 = callGetBucketIdx(xMarg + wB + kSubPixelCenterOffset, yMarg + kSubPixelCenterOffset);
    EXPECT_EQ(idxCol1, 1);

    // Column 0, Row 1
    size_t idxRow1 = callGetBucketIdx(xMarg + kSubPixelCenterOffset, yMarg + hB + kSubPixelCenterOffset);
    EXPECT_EQ(idxRow1, getBCols());

    // Last Bucket (bottom-right)
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

    // Horizontal test configuration
    frameBuffer[kSampleY * kImageWidth + kSampleX] = kValLeft;
    frameBuffer[kSampleY * kImageWidth + kSampleXNext] = kValRight;
    frameBuffer[kSampleYNext * kImageWidth + kSampleX] = kValLeft;
    frameBuffer[kSampleYNext * kImageWidth + kSampleXNext] = kValRight;

    int16_t sampleH = callBilinear(frameBuffer.data(), Eigen::Vector2f(static_cast<float>(kSampleX) + kHalfStep, static_cast<float>(kSampleY)));
    EXPECT_NEAR(sampleH, kExpectedMidValH, 1);

    // Vertical test configuration
    frameBuffer[kSampleY * kImageWidth + kSampleX] = kValTop;
    frameBuffer[kSampleYNext * kImageWidth + kSampleX] = kValBottom;

    int16_t sampleV = callBilinear(frameBuffer.data(), Eigen::Vector2f(static_cast<float>(kSampleX), static_cast<float>(kSampleY) + kHalfStep));
    EXPECT_NEAR(sampleV, kExpectedMidValV, 1);
}

TEST_F(HfTrackerTest, bilinearSubPixelQuadraticWeighting) {
    constexpr size_t kSampleX = 50;
    constexpr size_t kSampleY = 50;
    constexpr size_t kSampleXNext = kSampleX + 1;
    constexpr size_t kSampleYNext = kSampleY + 1;

    constexpr uint8_t kVal00 = 0;
    constexpr uint8_t kVal10 = 100;
    constexpr uint8_t kVal01 = 200;
    constexpr uint8_t kVal11 = 255;

    constexpr float kSubX = 0.25f;
    constexpr float kSubY = 0.75f;
    constexpr float kCompX = 1.0f - kSubX;
    constexpr float kCompY = 1.0f - kSubY;

    constexpr float kExpectedInterpFloat =
        (kCompX * kCompY) * static_cast<float>(kVal00) +
        (kSubX  * kCompY) * static_cast<float>(kVal10) +
        (kCompX * kSubY)  * static_cast<float>(kVal01) +
        (kSubX  * kSubY)  * static_cast<float>(kVal11);
    constexpr int16_t kExpectedInterpVal = static_cast<int16_t>(kExpectedInterpFloat + 0.5f);

    frameBuffer[kSampleY * kImageWidth + kSampleX] = kVal00;
    frameBuffer[kSampleY * kImageWidth + kSampleXNext] = kVal10;
    frameBuffer[kSampleYNext * kImageWidth + kSampleX] = kVal01;
    frameBuffer[kSampleYNext * kImageWidth + kSampleXNext] = kVal11;

    int16_t sample = callBilinear(frameBuffer.data(), Eigen::Vector2f(static_cast<float>(kSampleX) + kSubX, static_cast<float>(kSampleY) + kSubY));
    EXPECT_NEAR(sample, kExpectedInterpVal, 2);
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

    drawCorner(frameBuffer, kImageWidth, kImageHeight, cornerKp.x(), cornerKp.y(), kCornerRadius, kDarkIntensity, kBrightIntensity);

    bool added = callAddKp(frameBuffer.data(), cornerKp);

    ASSERT_TRUE(added);
    ASSERT_EQ(getActiveFeatures().size(), 1);

    const auto &feature = getActiveFeatures().front();
    EXPECT_EQ(feature.id, 0);
    EXPECT_EQ(feature.age, 0);
    EXPECT_NEAR(feature.ptCurr.x(), static_cast<float>(cornerKp.x()), kFloatEps);
    EXPECT_NEAR(feature.ptCurr.y(), static_cast<float>(cornerKp.y()), kFloatEps);
    EXPECT_NEAR(feature.ptPrev.x(), static_cast<float>(cornerKp.x()), kFloatEps);
    EXPECT_NEAR(feature.ptPrev.y(), static_cast<float>(cornerKp.y()), kFloatEps);
}

// ============================================================================
// 4. Inverse Compositional LK Alignment Tests (Private Method: alignFeatureICLK)
// ============================================================================

TEST_F(HfTrackerTest, alignFeatureICLKZeroDisplacementExactConvergence) {
    constexpr int kCornerX = 220;
    constexpr int kCornerY = 180;
    Eigen::Vector2i cornerKp(kCornerX, kCornerY);

    drawCorner(frameBuffer, kImageWidth, kImageHeight, cornerKp.x(), cornerKp.y(), kCornerRadius, kDarkIntensity, kBrightIntensity);

    ASSERT_TRUE(callAddKp(frameBuffer.data(), cornerKp));
    auto &feature = getActiveFeatures().front();

    Eigen::Vector2f ptEstimate = feature.ptCurr;
    bool aligned = callAlignFeatureICLK(feature, frameBuffer.data(), ptEstimate);

    EXPECT_TRUE(aligned);
    EXPECT_NEAR(ptEstimate.x(), feature.ptCurr.x(), kZeroDisplacementTol);
    EXPECT_NEAR(ptEstimate.y(), feature.ptCurr.y(), kZeroDisplacementTol);
}

TEST_F(HfTrackerTest, alignFeatureICLKSubPixelShiftConvergence) {
    constexpr int kInitialCornerX = 310;
    constexpr int kInitialCornerY = 210;
    constexpr int kDisplacementPixelX = 1;
    constexpr int kDisplacedCornerX = kInitialCornerX + kDisplacementPixelX;
    constexpr int kDisplacedCornerY = kInitialCornerY;

    Eigen::Vector2i initialKp(kInitialCornerX, kInitialCornerY);
    drawCorner(frameBuffer, kImageWidth, kImageHeight, initialKp.x(), initialKp.y(), kCornerRadius, kDarkIntensity, kBrightIntensity);

    ASSERT_TRUE(callAddKp(frameBuffer.data(), initialKp));
    auto &feature = getActiveFeatures().front();

    std::vector<uint8_t> nextFrame(kImageWidth * kImageHeight, kDefaultBackground);
    drawCorner(nextFrame, kImageWidth, kImageHeight, kDisplacedCornerX, kDisplacedCornerY, kCornerRadius, kDarkIntensity, kBrightIntensity);

    Eigen::Vector2f ptEstimate = feature.ptCurr;
    bool aligned = callAlignFeatureICLK(feature, nextFrame.data(), ptEstimate);

    EXPECT_TRUE(aligned);
    EXPECT_NEAR(ptEstimate.x(), static_cast<float>(kDisplacedCornerX), kSubPixelAlignmentTol);
    EXPECT_NEAR(ptEstimate.y(), static_cast<float>(kDisplacedCornerY), kSubPixelAlignmentTol);
}

TEST_F(HfTrackerTest, alignFeatureICLKMADThresholdRejectionOnSevereOcclusion) {
    constexpr int kCornerX = 180;
    constexpr int kCornerY = 140;
    constexpr uint8_t kOccludedIntensity = 0;

    Eigen::Vector2i cornerKp(kCornerX, kCornerY);
    drawCorner(frameBuffer, kImageWidth, kImageHeight, cornerKp.x(), cornerKp.y(), kCornerRadius, kDarkIntensity, kBrightIntensity);

    ASSERT_TRUE(callAddKp(frameBuffer.data(), cornerKp));
    auto &feature = getActiveFeatures().front();

    std::vector<uint8_t> corruptedFrame(kImageWidth * kImageHeight, kOccludedIntensity);

    Eigen::Vector2f ptEstimate = feature.ptCurr;
    bool aligned = callAlignFeatureICLK(feature, corruptedFrame.data(), ptEstimate);

    EXPECT_FALSE(aligned);
}

TEST_F(HfTrackerTest, alignFeatureICLKOutOfBoundsGuardRejection) {
    constexpr int kCornerX = 200;
    constexpr int kCornerY = 200;
    constexpr float kOutOfBoundsMarginOffset = 2.0f;
    constexpr float kOutOfBoundsX = static_cast<float>(kImageWidth) - kOutOfBoundsMarginOffset;

    Eigen::Vector2i cornerKp(kCornerX, kCornerY);
    drawCorner(frameBuffer, kImageWidth, kImageHeight, cornerKp.x(), cornerKp.y(), kCornerRadius, kDarkIntensity, kBrightIntensity);

    ASSERT_TRUE(callAddKp(frameBuffer.data(), cornerKp));
    auto &feature = getActiveFeatures().front();

    Eigen::Vector2f outEstimate(kOutOfBoundsX, static_cast<float>(kCornerY));
    bool aligned = callAlignFeatureICLK(feature, frameBuffer.data(), outEstimate);

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

    // Principal point snaps to optical axis [0, 0, 1]^T
    EXPECT_NEAR(ptUndist(0, 0), 0.0f, kFloatEps);
    EXPECT_NEAR(ptUndist(1, 0), 0.0f, kFloatEps);
    EXPECT_NEAR(ptUndist(2, 0), 1.0f, kFloatEps);

    // +Fx displacement maps to X unit normalized ray [1, 0, 1]^T
    EXPECT_NEAR(ptUndist(0, 1), 1.0f, kFloatEps);
    EXPECT_NEAR(ptUndist(1, 1), 0.0f, kFloatEps);
    EXPECT_NEAR(ptUndist(2, 1), 1.0f, kFloatEps);

    // +Fy displacement maps to Y unit normalized ray [0, 1, 1]^T
    EXPECT_NEAR(ptUndist(0, 2), 0.0f, kFloatEps);
    EXPECT_NEAR(ptUndist(1, 2), 1.0f, kFloatEps);
    EXPECT_NEAR(ptUndist(2, 2), 1.0f, kFloatEps);
}

TEST_F(HfTrackerTest, undistortHomogeneousRayZInvariant) {
    constexpr size_t kNumPoints = 25;
    constexpr float kBaseX = 100.0f;
    constexpr float kBaseY = 80.0f;
    constexpr float kStepX = 15.0f;
    constexpr float kStepY = 12.0f;

    std::vector<float> xDist(kNumPoints);
    std::vector<float> yDist(kNumPoints);

    for (size_t i = 0; i < kNumPoints; i++) {
        xDist[i] = kBaseX + static_cast<float>(i) * kStepX;
        yDist[i] = kBaseY + static_cast<float>(i) * kStepY;
    }

    Eigen::Matrix3Xf ptUndist(3, kNumPoints);
    callUndistort(xDist, yDist, ptUndist, kNumPoints);

    for (size_t i = 0; i < kNumPoints; i++) {
        EXPECT_NEAR(ptUndist(2, static_cast<Eigen::Index>(i)), 1.0f, kFloatEps);
    }
}

// ============================================================================
// 6. 2-Point Epipolar RANSAC Tests (Private Method: ransac)
// ============================================================================

TEST_F(HfTrackerTest, ransacPureRotationExactConsensus) {
    constexpr size_t kNumFeatures = 35;
    constexpr float kRotAngleDeg = 1.5f;
    constexpr float kDegToRad = 3.14159265358979323846f / 180.0f;
    constexpr float kRotAngleRad = kRotAngleDeg * kDegToRad;

    constexpr float kStartNormX = -0.4f;
    constexpr float kStartNormY = -0.3f;
    constexpr float kSpacingNorm = 0.12f;
    constexpr size_t kGridDim = 7;

    const Eigen::Matrix3f R = Eigen::AngleAxisf(kRotAngleRad, Eigen::Vector3f::UnitY()).toRotationMatrix();

    for (size_t i = 0; i < kNumFeatures; i++) {
        float xn = kStartNormX + (static_cast<float>(i % kGridDim) * kSpacingNorm);
        float yn = kStartNormY + (static_cast<float>(static_cast<float>(i) / kGridDim) * kSpacingNorm);
        Eigen::Vector3f rPrev(xn, yn, 1.0f);

        // Project ray under pure rotation into current frame
        Eigen::Vector3f rCurr = R * rPrev;
        rCurr /= rCurr.z();

        Feature feat{};
        feat.id = static_cast<uint32_t>(i);
        feat.age = 1;
        // Project normalized pinhole rays to distorted pixel coordinates
        feat.ptPrev = rPrev.hnormalized();
        feat.ptCurr = rCurr.hnormalized();

        getActiveFeatures().push_back(feat);
    }

    int inlierCount = callRansac(R);

    EXPECT_GE(inlierCount, RANSAC_MIN_INLIERS);
    EXPECT_GE(static_cast<float>(inlierCount) / static_cast<float>(kNumFeatures), RANSAC_EARLY_CONSENSUS);
}

TEST_F(HfTrackerTest, ransacPrunesInjectedOutliers) {
    constexpr size_t kNumInliers = 30;
    constexpr size_t kNumOutliers = 10;
    constexpr size_t kTotalFeatures = kNumInliers + kNumOutliers;

    // Physical motion: rotation + non-zero translation
    constexpr float kRotAngleRad = 0.015f;
    const Eigen::Matrix3f R = Eigen::AngleAxisf(kRotAngleRad, Eigen::Vector3f::UnitX()).toRotationMatrix();
    const Eigen::Vector3f tTrue = Eigen::Vector3f(0.08f, -0.04f, 0.20f).normalized();

    constexpr float kStartNormX = -0.3f;
    constexpr float kStartNormY = -0.2f;
    constexpr float kSpacingNorm = 0.08f;
    constexpr size_t kGridDim = 8;
    constexpr float kBaseDepthMeters = 2.0f;

    std::mt19937 outlierRng(42);
    std::uniform_real_distribution<float> outlierDist(40.0f, 90.0f);
    std::uniform_real_distribution<float> signDist(-1.0f, 1.0f);

    for (size_t i = 0; i < kTotalFeatures; i++) {
        const float xn = kStartNormX + (static_cast<float>(i % kGridDim) * kSpacingNorm);
        const float yn = kStartNormY + (static_cast<float>(static_cast<float>(i) / kGridDim) * kSpacingNorm);
        const float depth = kBaseDepthMeters + static_cast<float>(i % 5) * 0.4f;

        const Eigen::Vector3f p3D_prev(xn * depth, yn * depth, depth);
        const Eigen::Vector3f rPrev = p3D_prev.normalized();

        // Rigid transform: P_curr = R * P_prev + t
        const Eigen::Vector3f p3D_curr = R * p3D_prev + tTrue * 0.15f;
        const Eigen::Vector3f rCurr = p3D_curr / p3D_curr.z();

        Feature feat{};
        feat.id = static_cast<uint32_t>(i);
        feat.age = 2;
        feat.ptPrev = Eigen::Vector2f(kFx * rPrev.x() / rPrev.z() + kCx, 
                                     kFy * rPrev.y() / rPrev.z() + kCy);
        feat.ptCurr = Eigen::Vector2f(kFx * rCurr.x() + kCx, 
                                     kFy * rCurr.y() + kCy);

        // Perturb outliers in independent, random 2D directions
        if (i >= kNumInliers) {
            const float dx = outlierDist(outlierRng) * (signDist(outlierRng) > 0.0f ? 1.0f : -1.0f);
            const float dy = outlierDist(outlierRng) * (signDist(outlierRng) > 0.0f ? 1.0f : -1.0f);
            feat.ptCurr.x() += dx;
            feat.ptCurr.y() += dy;
        }

        getActiveFeatures().push_back(feat);
    }

    const int inlierCount = callRansac(R);

    constexpr int kConsensusMargin = 2;
    EXPECT_GE(inlierCount, static_cast<int>(kNumInliers) - kConsensusMargin);
    EXPECT_LE(inlierCount, static_cast<int>(kNumInliers) + 1);
}

TEST_F(HfTrackerTest, ransacDeterministicBehaviorWithZeroSeed) {
    constexpr size_t kNumFeatures = 25;
    constexpr float kRotAngleRad = 0.02f;
    constexpr float kStartNormX = -0.2f;
    constexpr float kSpacingNormX = 0.015f;
    constexpr float kFixedNormY = 0.1f;

    const Eigen::Matrix3f R = Eigen::AngleAxisf(kRotAngleRad, Eigen::Vector3f::UnitZ()).toRotationMatrix();

    for (size_t i = 0; i < kNumFeatures; i++) {
        float xn = kStartNormX + (static_cast<float>(i) * kSpacingNormX);
        float yn = kFixedNormY;
        Eigen::Vector3f rPrev(xn, yn, 1.0f);

        Eigen::Vector3f rCurr = R * rPrev;
        rCurr /= rCurr.z();

        Feature feat{};
        feat.id = static_cast<uint32_t>(i);
        feat.age = 1;
        feat.ptPrev = Eigen::Vector2f(kFx * rPrev.x() + kCx, kFy * rPrev.y() + kCy);
        feat.ptCurr = Eigen::Vector2f(kFx * rCurr.x() + kCx, kFy * rCurr.y() + kCy);

        getActiveFeatures().push_back(feat);
    }

    auto trackerB = createTracker();
    for (size_t i = 0; i < kNumFeatures; i++) {
        getActiveFeatures(*trackerB).push_back(getActiveFeatures()[i]);
    }

    int runA = callRansac(R);
    int runB = callRansac(*trackerB, R);

    EXPECT_EQ(runA, runB);
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

TEST_F(HfTrackerTest, densityTwoFeaturesTooCloseEvictsYounger) {
    const float xBase = static_cast<float>(getXMargin());
    const float yBase = static_cast<float>(getYMargin());

    constexpr float kOffsetBase = 10.0f;
    constexpr float kTooCloseDistance = MIN_DIST_IN_BUCKET / 3.0f;
    constexpr float kOffsetClose = kOffsetBase + kTooCloseDistance;

    constexpr uint32_t kAgeOlder = 8;
    constexpr uint32_t kAgeYounger = 1;

    Feature olderFeat{};
    olderFeat.id = 201;
    olderFeat.age = kAgeOlder;
    olderFeat.ptCurr = Eigen::Vector2f(xBase + kOffsetBase, yBase + kOffsetBase);

    Feature youngerFeat{};
    youngerFeat.id = 202;
    youngerFeat.age = kAgeYounger;
    youngerFeat.ptCurr = Eigen::Vector2f(xBase + kOffsetClose, yBase + kOffsetBase);

    getActiveFeatures().push_back(olderFeat);
    getActiveFeatures().push_back(youngerFeat);

    callEnforceBucketDensity();

    const auto &bucket0 = getGridOccupancy()[0];
    EXPECT_EQ(bucket0.count, 1);
    EXPECT_EQ(bucket0.olderIdx, 0);
    EXPECT_EQ(getKeep()[0], 1);
    EXPECT_EQ(getKeep()[1], 0);
}

TEST_F(HfTrackerTest, densityThreeFeaturesCollisionEvictsYoungest) {
    const float xBase = static_cast<float>(getXMargin());
    const float yBase = static_cast<float>(getYMargin());

    constexpr float kOffsetA = 4.0f;
    constexpr float kOffsetB = 20.0f;
    constexpr float kOffsetC = 12.0f;

    constexpr uint32_t kAgeOldest = 12;
    constexpr uint32_t kAgeYoungest = 3;
    constexpr uint32_t kAgeMiddle = 7;

    Feature fA{};
    fA.id = 1; fA.age = kAgeOldest; fA.ptCurr = Eigen::Vector2f(xBase + kOffsetA, yBase + kOffsetA);

    Feature fB{};
    fB.id = 2; fB.age = kAgeYoungest; fB.ptCurr = Eigen::Vector2f(xBase + kOffsetB, yBase + kOffsetA);

    Feature fC{};
    fC.id = 3; fC.age = kAgeMiddle; fB.ptCurr = Eigen::Vector2f(xBase + kOffsetC, yBase + kOffsetB);

    getActiveFeatures().push_back(fA);
    getActiveFeatures().push_back(fB);
    getActiveFeatures().push_back(fC);

    callEnforceBucketDensity();

    const auto &bucket0 = getGridOccupancy()[0];
    EXPECT_EQ(bucket0.count, 2);
    EXPECT_EQ(bucket0.olderIdx, 0);
    EXPECT_EQ(bucket0.youngerIdx, 2);
    EXPECT_EQ(getKeep()[0], 1);
    EXPECT_EQ(getKeep()[1], 0);
    EXPECT_EQ(getKeep()[2], 1);
}

// ============================================================================
// 8. Public Interface End-to-End Tracking (trackExistingFeatures)
// ============================================================================

TEST_F(HfTrackerTest, trackExistingEmptyActiveFeaturesReturnsZero) {
    const Eigen::Matrix3f R = Eigen::Matrix3f::Identity();
    size_t tracked = tracker->trackExistingFeatures(frameBuffer.data(), R);

    EXPECT_EQ(tracked, 0);
    EXPECT_TRUE(tracker->getActiveFeatures().empty());
}

TEST_F(HfTrackerTest, trackExistingFeaturesNominalTrackingLifecycle) {
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
            drawCorner(frameBuffer, kImageWidth, kImageHeight, cx, cy, kCornerRadius, kDarkIntensity, kBrightIntensity);
            seedCorners.emplace_back(cx, cy);
        }
    }

    for (size_t i = 0; i < seedCorners.size(); i++) {
        ASSERT_TRUE(callAddKp(frameBuffer.data(), seedCorners[i]));
    }

    const size_t initialFeatureCount = tracker->getActiveFeatures().size();
    ASSERT_EQ(initialFeatureCount, kTotalSeedFeatures);
    ASSERT_GE(initialFeatureCount, static_cast<size_t>(RANSAC_MIN_INLIERS));

    const Eigen::Matrix3f R = Eigen::Matrix3f::Identity();
    size_t survived = tracker->trackExistingFeatures(frameBuffer.data(), R);

    EXPECT_EQ(survived, initialFeatureCount);
    EXPECT_EQ(tracker->getActiveFeatures().size(), initialFeatureCount);

    for (size_t i = 0; i < tracker->getActiveFeatures().size(); i++) {
        const auto &feat = tracker->getActiveFeatures()[i];
        EXPECT_EQ(feat.age, 1);
        EXPECT_NEAR(feat.ptCurr.x(), feat.ptPrev.x(), kZeroDisplacementTol);
        EXPECT_NEAR(feat.ptCurr.y(), feat.ptPrev.y(), kZeroDisplacementTol);
    }
}

TEST_F(HfTrackerTest, trackExistingFeaturesPrunesBoundaryExits) {
    constexpr int kBoundaryMarginOffset = 2;
    int cx = static_cast<int>(kImageWidth - getXMargin() - kBoundaryMarginOffset);
    int cy = static_cast<int>(kCy);

    drawCorner(frameBuffer, kImageWidth, kImageHeight, cx, cy, kCornerRadius, kDarkIntensity, kBrightIntensity);

    Eigen::Vector2i kp(cx, cy);
    ASSERT_TRUE(callAddKp(frameBuffer.data(), kp));

    constexpr float kYawRotationRad = -0.035f;
    const Eigen::Matrix3f R = Eigen::AngleAxisf(kYawRotationRad, Eigen::Vector3f::UnitY()).toRotationMatrix();

    std::vector<uint8_t> nextFrame(kImageWidth * kImageHeight, kDefaultBackground);
    size_t survived = tracker->trackExistingFeatures(nextFrame.data(), R);

    EXPECT_EQ(survived, 0);
    EXPECT_TRUE(tracker->getActiveFeatures().empty());
}

TEST_F(HfTrackerTest, trackExistingFeaturesPrunesSeverePhotometricOutliers) {
    constexpr size_t kTotalFeatures = 25;
    constexpr size_t kOccludedCount = 5;
    constexpr size_t kExpectedSurvivors = kTotalFeatures - kOccludedCount;
    constexpr size_t kOffsetInsideBucket = 6;
    constexpr int kOcclusionPatchRadius = 8;
    constexpr uint8_t kOcclusionVal = 0;

    for (size_t i = 0; i < kTotalFeatures; i++) {
        int cx = static_cast<int>(getXMargin() + (i % 5 + 1) * getWBucket() + kOffsetInsideBucket);
        int cy = static_cast<int>(getYMargin() + (i / 5 + 1) * getHBucket() + kOffsetInsideBucket);
        drawCorner(frameBuffer, kImageWidth, kImageHeight, cx, cy, kCornerRadius, kDarkIntensity, kBrightIntensity);
        Eigen::Vector2i kp(cx, cy);
        ASSERT_TRUE(callAddKp(frameBuffer.data(), kp));
    }

    ASSERT_EQ(tracker->getActiveFeatures().size(), kTotalFeatures);

    std::vector<uint8_t> nextFrame = frameBuffer;
    for (size_t i = 0; i < kOccludedCount; i++) {
        int cx = static_cast<int>(tracker->getActiveFeatures()[i].ptCurr.x());
        int cy = static_cast<int>(tracker->getActiveFeatures()[i].ptCurr.y());
        for (int y = cy - kOcclusionPatchRadius; y <= cy + kOcclusionPatchRadius; y++) {
            for (int x = cx - kOcclusionPatchRadius; x <= cx + kOcclusionPatchRadius; x++) {
                nextFrame[static_cast<size_t>(y * static_cast<int>(kImageWidth) + x)] = kOcclusionVal;
            }
        }
    }

    const Eigen::Matrix3f R = Eigen::Matrix3f::Identity();
    size_t survived = tracker->trackExistingFeatures(nextFrame.data(), R);

    EXPECT_EQ(survived, kExpectedSurvivors);
    EXPECT_EQ(tracker->getActiveFeatures().size(), kExpectedSurvivors);
}

} // namespace PoolSLAM