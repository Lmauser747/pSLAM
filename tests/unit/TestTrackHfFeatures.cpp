#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include <eigen3/Eigen/Dense>
#include <gtest/gtest.h>

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
        for (size_t x = 0; x < width; x++) {
            buffer[y * width + x] = val;
        }
    }
}

// Applies forward Brown-Conrady radial and tangential distortion to normalized coordinates
Eigen::Vector2f applyForwardDistortion(float xu, float yu, const Distortion &d) {
    const float r2 = xu * xu + yu * yu;
    const float r4 = r2 * r2;
    const float xy = xu * yu;
    const float radial = 1.0f + d.k1 * r2 + d.k2 * r4;

    const float xd = xu * radial + 2.0f * d.p1 * xy + d.p2 * (r2 + 2.0f * xu * xu);
    const float yd = yu * radial + 2.0f * d.p2 * xy + d.p1 * (r2 + 2.0f * yu * yu);
    return Eigen::Vector2f(xd, yd);
}

} // namespace

class HfTrackerTest : public ::testing::Test {
protected:
    static constexpr size_t kImageWidth = 640;
    static constexpr size_t kImageHeight = 480;
    static constexpr size_t kTargetFeatures = 200;

    static constexpr float kFx = 525.0f;
    static constexpr float kFy = 525.0f;
    static constexpr float kCx = static_cast<float>(kImageWidth) / 2.0f;
    static constexpr float kCy = static_cast<float>(kImageHeight) / 2.0f;

    static constexpr float kFloatEps = 1e-4f;
    static constexpr float kSubPixelAlignmentTol = 0.08f;

    static constexpr uint8_t kDefaultBackground = 128;
    static constexpr uint8_t kDarkIntensity = 30;
    static constexpr uint8_t kBrightIntensity = 220;
    static constexpr int kCornerRadius = 12;

    static constexpr int RADIUS = HfTracker::RADIUS;
    static constexpr size_t MIN_MARGIN = HfTracker::MIN_MARGIN;
    static constexpr int RANSAC_MIN_INLIERS = HfTracker::RANSAC_MIN_INLIERS;
    static constexpr float MIN_DIST_IN_BUCKET = HfTracker::MIN_DIST_IN_BUCKET;

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

    // Internal state accessors
    size_t getWBucket() const { return tracker->wBucket; }
    size_t getHBucket() const { return tracker->hBucket; }
    size_t getBRows() const { return tracker->bRows; }
    size_t getBCols() const { return tracker->bCols; }
    size_t getXMargin() const { return tracker->xMargin; }
    size_t getYMargin() const { return tracker->yMargin; }
    size_t getTotalBuckets() const { return tracker->totalBuckets; }

    std::vector<Feature> &getActiveFeatures() { return tracker->activeFeatures; }
    const std::vector<Feature> &getActiveFeatures() const { return tracker->activeFeatures; }
    const std::vector<GridBucket> &getGridOccupancy() const { return tracker->gridOccupancy; }
    const std::vector<uint8_t> &getKeep() const { return tracker->keep; }

    // Private method wrappers
    size_t callGetBucketIdx(double x, double y) const { return tracker->getBucketIdx(x, y); }
    size_t callOffsetFromBucket(size_t bucket) const { return tracker->offsetFromBucket(bucket); }
    Eigen::Vector2i callTopLeftBucket(size_t bucket) const { return tracker->topLeftBucket(bucket); }

    int16_t callBilinear(const uint8_t *curBuffer, const Eigen::Vector2f &pt) const {
        return tracker->bilinear(curBuffer, pt);
    }

    bool callAlignFeatureICLK(Feature &feature, const uint8_t *curBuffer, Eigen::Vector2f &ptEst) {
        return tracker->alignFeatureICLK(feature, curBuffer, ptEst);
    }

    void callUndistort(std::vector<float> &xDist, std::vector<float> &yDist,
                       Eigen::Matrix3Xf &ptUndist, size_t count) {
        tracker->undistort(xDist, yDist, ptUndist, count);
    }

    int callRansac(const Eigen::Matrix3f &R) { return tracker->ransac(R); }
    void callEnforceBucketDensity() { tracker->enforceBucketDensity(); }

    bool callAddKp(const uint8_t *curBuffer, Eigen::Vector2i &kp) {
        return tracker->addKp(curBuffer, kp);
    }

    // Helper: Creates a synthetic feature with specified ID, age, and coordinates
    static Feature makeFeature(uint32_t id, uint32_t age,
                               const Eigen::Vector2f &ptPrev,
                               const Eigen::Vector2f &ptCurr) {
        Feature feat{};
        feat.id = id;
        feat.age = static_cast<int>(age);
        feat.ptPrev = ptPrev;
        feat.ptCurr = ptCurr;
        return feat;
    }

    // Helper: Projects 3D points under P_curr = R * P_prev + t and populates activeFeatures
    void populateSyntheticEpipolarFeatures(size_t numInliers,
                                           const Eigen::Matrix3f &R,
                                           const Eigen::Vector3f &t,
                                           const std::vector<Eigen::Vector2f> &outlierOffsets) {
        const size_t total = numInliers + outlierOffsets.size();
        constexpr float kStartNormX = -0.35f;
        constexpr float kStartNormY = -0.25f;
        constexpr float kSpacingNorm = 0.09f;
        constexpr size_t kGridCols = 7;

        for (size_t i = 0; i < total; i++) {
            const float xn = kStartNormX + static_cast<float>(i % kGridCols) * kSpacingNorm;
            const float yn = kStartNormY + static_cast<float>(static_cast<float>(i) / kGridCols) * kSpacingNorm;
            const float depth = 2.0f + static_cast<float>(i % 5) * 0.45f;

            const Eigen::Vector3f pPrev(xn * depth, yn * depth, depth);
            const Eigen::Vector3f pCurr = R * pPrev + t;

            const Eigen::Vector2f uvPrev(kFx * (pPrev.x() / pPrev.z()) + kCx,
                                         kFy * (pPrev.y() / pPrev.z()) + kCy);
            Eigen::Vector2f uvCurr(kFx * (pCurr.x() / pCurr.z()) + kCx,
                                   kFy * (pCurr.y() / pCurr.z()) + kCy);

            if (i >= numInliers) {
                uvCurr += outlierOffsets[i - numInliers];
            }

            getActiveFeatures().push_back(makeFeature(static_cast<uint32_t>(i), 2, uvPrev, uvCurr));
        }
    }

    // Helper: Renders a grid of corners across distinct buckets, registers them via addKp,
    // and renders a second frame transformed by P_curr = R * P_prev + t
    std::vector<uint8_t> seedAndRenderTransformedScene(
        size_t gridCols, size_t gridRows, size_t offsetInBucket,
        const Eigen::Matrix3f &R, const Eigen::Vector3f &t, float depthMeters,
        std::vector<Eigen::Vector2f> &outExpectedCurrPts,
        const std::vector<std::pair<size_t, Eigen::Vector2f>> &outlierPerturbations = {}) {

        outExpectedCurrPts.clear();
        std::vector<uint8_t> nextFrame(kImageWidth * kImageHeight, kDefaultBackground);

        size_t featIdx = 0;
        for (size_t col = 1; col <= gridCols; col++) {
            for (size_t row = 1; row <= gridRows; row++) {
                const float x0 = static_cast<float>(getXMargin() + col * getWBucket() + offsetInBucket);
                const float y0 = static_cast<float>(getYMargin() + row * getHBucket() + offsetInBucket);

                drawSmoothCorner(frameBuffer, kImageWidth, kImageHeight, x0, y0,
                                 kCornerRadius, kDarkIntensity, kBrightIntensity);

                Eigen::Vector2i kp(static_cast<int>(std::round(x0)), static_cast<int>(std::round(y0)));
                EXPECT_TRUE(callAddKp(frameBuffer.data(), kp));

                const Eigen::Vector3f p0((x0 - kCx) * depthMeters / kFx,
                                         (y0 - kCy) * depthMeters / kFy,
                                         depthMeters);
                const Eigen::Vector3f p1 = R * p0 + t;

                Eigen::Vector2f pt1((kFx * p1.x() / p1.z()) + kCx,
                                    (kFy * p1.y() / p1.z()) + kCy);

                for (const auto &pert : outlierPerturbations) {
                    if (pert.first == featIdx) {
                        pt1 += pert.second;
                    }
                }

                drawSmoothCorner(nextFrame, kImageWidth, kImageHeight, pt1.x(), pt1.y(),
                                 kCornerRadius, kDarkIntensity, kBrightIntensity);
                outExpectedCurrPts.push_back(pt1);
                featIdx++;
            }
        }
        return nextFrame;
    }
};

// ============================================================================
// 1. Spatial Grid Geometry & Indexing
// ============================================================================

// Verifies bucket dimensions, 16-byte SIMD width alignment, margins, and occupancy allocation.
TEST_F(HfTrackerTest, bucketGeometryInitializationAndAlignment) {
    constexpr size_t kSimdAlignmentBytes = 16;

    EXPECT_GT(getWBucket(), 0u);
    EXPECT_GT(getHBucket(), 0u);
    EXPECT_EQ(getWBucket() % kSimdAlignmentBytes, 0u);
    EXPECT_GE(getXMargin(), MIN_MARGIN);
    EXPECT_GE(getYMargin(), MIN_MARGIN);
    EXPECT_EQ(getTotalBuckets(), getBCols() * getBRows());
    EXPECT_EQ(getGridOccupancy().size(), getTotalBuckets());
}

// Verifies bucket index lookup at corners and round-trip consistency with topLeftBucket and offsetFromBucket.
TEST_F(HfTrackerTest, bucketIndexingAndCoordinateRoundTrip) {
    const double xMarg = static_cast<double>(getXMargin());
    const double yMarg = static_cast<double>(getYMargin());
    const double wB = static_cast<double>(getWBucket());
    const double hB = static_cast<double>(getHBucket());

    EXPECT_EQ(callGetBucketIdx(xMarg + 0.5, yMarg + 0.5), 0u);
    EXPECT_EQ(callGetBucketIdx(xMarg + wB + 0.5, yMarg + 0.5), 1u);
    EXPECT_EQ(callGetBucketIdx(xMarg + 0.5, yMarg + hB + 0.5), getBCols());

    const double xMax = xMarg + static_cast<double>(getBCols()) * wB - 0.5;
    const double yMax = yMarg + static_cast<double>(getBRows()) * hB - 0.5;
    EXPECT_EQ(callGetBucketIdx(xMax, yMax), getTotalBuckets() - 1);

    for (size_t b = 0; b < getTotalBuckets(); b++) {
        const Eigen::Vector2i topLeft = callTopLeftBucket(b);
        const size_t expectedOffset =
            static_cast<size_t>(topLeft.y()) * kImageWidth + static_cast<size_t>(topLeft.x());
        EXPECT_EQ(callOffsetFromBucket(b), expectedOffset);
        EXPECT_EQ(callGetBucketIdx(topLeft.x() + 1.0, topLeft.y() + 1.0), b);
    }
}

// ============================================================================
// 2. Sub-Pixel Bilinear Interpolation (bilinear)
// ============================================================================

// Verifies that querying exact integer pixel coordinates returns the exact underlying pixel intensity.
TEST_F(HfTrackerTest, bilinearExactIntegerCoordinates) {
    constexpr size_t kX = 120;
    constexpr size_t kY = 80;
    constexpr uint8_t kCenterVal = 210;

    frameBuffer[kY * kImageWidth + kX] = kCenterVal;
    frameBuffer[kY * kImageWidth + kX + 1] = 10;
    frameBuffer[(kY + 1) * kImageWidth + kX] = 20;
    frameBuffer[(kY + 1) * kImageWidth + kX + 1] = 30;

    const int16_t sample = callBilinear(frameBuffer.data(),
                                        Eigen::Vector2f(static_cast<float>(kX), static_cast<float>(kY)));
    EXPECT_EQ(sample, static_cast<int16_t>(kCenterVal));
}

// Verifies 1D linear interpolation at horizontal (x + 0.5, y) and vertical (x, y + 0.5) midpoints.
TEST_F(HfTrackerTest, bilinearAxisAlignedMidpoints) {
    constexpr size_t kX = 200;
    constexpr size_t kY = 150;

    frameBuffer[kY * kImageWidth + kX] = 100;
    frameBuffer[kY * kImageWidth + kX + 1] = 200;
    frameBuffer[(kY + 1) * kImageWidth + kX] = 160;
    frameBuffer[(kY + 1) * kImageWidth + kX + 1] = 240;

    const int16_t sampleH = callBilinear(frameBuffer.data(),
                                         Eigen::Vector2f(static_cast<float>(kX) + 0.5f, static_cast<float>(kY)));
    EXPECT_NEAR(sampleH, 150, 1);

    const int16_t sampleV = callBilinear(frameBuffer.data(),
                                         Eigen::Vector2f(static_cast<float>(kX), static_cast<float>(kY) + 0.5f));
    EXPECT_NEAR(sampleV, 130, 1);
}

// Verifies 2D sub-pixel interpolation with asymmetric fractional weights across four distinct corner intensities.
TEST_F(HfTrackerTest, bilinear2DSubPixelAsymmetricWeights) {
    constexpr size_t kX = 100;
    constexpr size_t kY = 100;

    // 2x2 neighborhood with distinct values to catch transposed weight assignments (w10 vs w01)
    constexpr uint8_t kI00 = 40;  // (x, y)
    constexpr uint8_t kI10 = 120; // (x + 1, y)
    constexpr uint8_t kI01 = 200; // (x, y + 1)
    constexpr uint8_t kI11 = 240; // (x + 1, y + 1)

    frameBuffer[kY * kImageWidth + kX] = kI00;
    frameBuffer[kY * kImageWidth + kX + 1] = kI10;
    frameBuffer[(kY + 1) * kImageWidth + kX] = kI01;
    frameBuffer[(kY + 1) * kImageWidth + kX + 1] = kI11;

    // At dx = 0.25, dy = 0.75:
    // I = 0.75 * 0.25 * 40 + 0.25 * 0.25 * 120 + 0.75 * 0.75 * 200 + 0.25 * 0.75 * 240
    //   = 7.5 + 7.5 + 112.5 + 45.0 = 172.5
    const int16_t sample = callBilinear(
        frameBuffer.data(),
        Eigen::Vector2f(static_cast<float>(kX) + 0.25f, static_cast<float>(kY) + 0.75f));
    EXPECT_NEAR(sample, 172, 1);
}

// Verifies that minimum (0) and maximum (255) pixel intensities interpolate without integer overflow or underflow.
TEST_F(HfTrackerTest, bilinearExtremeIntensityBounds) {
    constexpr size_t kX = 50;
    constexpr size_t kY = 50;

    std::fill(frameBuffer.begin(), frameBuffer.end(), 0);
    EXPECT_EQ(callBilinear(frameBuffer.data(), Eigen::Vector2f(kX + 0.65f, kY + 0.35f)), 0);

    std::fill(frameBuffer.begin(), frameBuffer.end(), 255);
    EXPECT_NEAR(callBilinear(frameBuffer.data(), Eigen::Vector2f(kX + 0.65f, kY + 0.35f)), 255, 1);
}

// ============================================================================
// 3. Feature Registration & Hessian Stability (addKp)
// ============================================================================

// Verifies that flat patches and 1D edges (aperture problem) fail Hessian invertibility and are rejected.
TEST_F(HfTrackerTest, addKpRejectsFlatAnd1DEdgePatches) {
    Eigen::Vector2i flatCandidate(250, 180);
    EXPECT_FALSE(callAddKp(frameBuffer.data(), flatCandidate));
    EXPECT_TRUE(getActiveFeatures().empty());

    constexpr int kEdgeY = 200;
    drawHorizontalEdge(frameBuffer, kImageWidth, kImageHeight, kEdgeY, kDarkIntensity, kBrightIntensity);
    Eigen::Vector2i edgeCandidate(160, kEdgeY);
    EXPECT_FALSE(callAddKp(frameBuffer.data(), edgeCandidate));
    EXPECT_TRUE(getActiveFeatures().empty());
}

// Verifies that a well-conditioned 2D corner is accepted and initialized with age 0 and valid coordinates.
TEST_F(HfTrackerTest, addKpAcceptsWellConditioned2DCorner) {
    Eigen::Vector2i cornerKp(300, 240);
    drawSmoothCorner(frameBuffer, kImageWidth, kImageHeight,
                     static_cast<float>(cornerKp.x()), static_cast<float>(cornerKp.y()),
                     kCornerRadius, kDarkIntensity, kBrightIntensity);

    ASSERT_TRUE(callAddKp(frameBuffer.data(), cornerKp));
    ASSERT_EQ(getActiveFeatures().size(), 1u);

    const auto &feat = getActiveFeatures().front();
    EXPECT_EQ(feat.id, 0u);
    EXPECT_EQ(feat.age, 0u);
    EXPECT_NEAR(feat.ptCurr.x(), static_cast<float>(cornerKp.x()), kFloatEps);
    EXPECT_NEAR(feat.ptCurr.y(), static_cast<float>(cornerKp.y()), kFloatEps);
}

// ============================================================================
// 4. Iterative Coordinate Undistortion (undistort)
// ============================================================================

// Verifies that undistort maps pixel coordinates to pinhole normalized rays when distortion is zero.
TEST_F(HfTrackerTest, undistortZeroDistortionPinholeMapping) {
    std::vector<float> xDist = {kCx, kCx + kFx, kCx - 0.5f * kFx};
    std::vector<float> yDist = {kCy, kCy, kCy + 0.5f * kFy};
    const size_t count = xDist.size();

    Eigen::Matrix3Xf ptUndist(3, count);
    callUndistort(xDist, yDist, ptUndist, count);

    EXPECT_NEAR(ptUndist(0, 0), 0.0f, kFloatEps);
    EXPECT_NEAR(ptUndist(1, 0), 0.0f, kFloatEps);
    EXPECT_NEAR(ptUndist(2, 0), 1.0f, kFloatEps);

    EXPECT_NEAR(ptUndist(0, 1), 1.0f, kFloatEps);
    EXPECT_NEAR(ptUndist(1, 1), 0.0f, kFloatEps);
    EXPECT_NEAR(ptUndist(2, 1), 1.0f, kFloatEps);

    EXPECT_NEAR(ptUndist(0, 2), -0.5f, kFloatEps);
    EXPECT_NEAR(ptUndist(1, 2), 0.5f, kFloatEps);
    EXPECT_NEAR(ptUndist(2, 2), 1.0f, kFloatEps);
}

// Verifies that undistort accurately inverts non-zero radial (k1, k2) and tangential (p1, p2) distortion.
TEST_F(HfTrackerTest, undistortRadialAndTangentialInversion) {
    Distortion nonZeroDist{};
    nonZeroDist.k1 = -0.12f;
    nonZeroDist.k2 = 0.03f;
    nonZeroDist.p1 = 0.002f;
    nonZeroDist.p2 = -0.0015f;

    tracker = std::make_unique<HfTracker>(kImageWidth, kImageHeight, K, nonZeroDist, kTargetFeatures);

    const std::vector<Eigen::Vector2f> trueNormalizedRays = {
        {0.0f, 0.0f},
        {0.25f, -0.18f},
        {-0.30f, 0.22f},
        {0.35f, 0.30f}
    };

    std::vector<float> xDist;
    std::vector<float> yDist;
    for (const auto &ray : trueNormalizedRays) {
        const Eigen::Vector2f distortedNorm = applyForwardDistortion(ray.x(), ray.y(), nonZeroDist);
        xDist.push_back(kFx * distortedNorm.x() + kCx);
        yDist.push_back(kFy * distortedNorm.y() + kCy);
    }

    Eigen::Matrix3Xf ptUndist(3, trueNormalizedRays.size());
    callUndistort(xDist, yDist, ptUndist, trueNormalizedRays.size());

    for (size_t i = 0; i < trueNormalizedRays.size(); i++) {
        const auto idx = static_cast<Eigen::Index>(i);
        EXPECT_NEAR(ptUndist(0, idx), trueNormalizedRays[i].x(), 5e-4f);
        EXPECT_NEAR(ptUndist(1, idx), trueNormalizedRays[i].y(), 5e-4f);
        EXPECT_NEAR(ptUndist(2, idx), 1.0f, kFloatEps);
    }
}

// Verifies that undistort only processes the first 'count' elements and does not overwrite trailing columns.
TEST_F(HfTrackerTest, undistortRespectsCountLimit) {
    std::vector<float> xDist = {kCx + kFx, kCx - kFx, kCx};
    std::vector<float> yDist = {kCy + kFy, kCy - kFy, kCy};

    Eigen::Matrix3Xf ptUndist = Eigen::Matrix3Xf::Constant(3, 3, -99.0f);
    callUndistort(xDist, yDist, ptUndist, 1);

    EXPECT_NEAR(ptUndist(0, 0), 1.0f, kFloatEps);
    EXPECT_NEAR(ptUndist(1, 0), 1.0f, kFloatEps);
    EXPECT_NEAR(ptUndist(2, 0), 1.0f, kFloatEps);

    // Column 1 and 2 must remain untouched
    EXPECT_FLOAT_EQ(ptUndist(0, 1), -99.0f);
    EXPECT_FLOAT_EQ(ptUndist(0, 2), -99.0f);
}

// ============================================================================
// 5. Inverse Compositional Lucas-Kanade Alignment (alignFeatureICLK)
// ============================================================================

// Verifies zero-displacement alignment maintains the true corner position, updates ptPrev, and increments age.
TEST_F(HfTrackerTest, alignFeatureICLKZeroDisplacementPreservesPosition) {
    constexpr float kTrueX = 220.0f;
    constexpr float kTrueY = 180.0f;
    Eigen::Vector2i cornerKp(static_cast<int>(kTrueX), static_cast<int>(kTrueY));

    drawSmoothCorner(frameBuffer, kImageWidth, kImageHeight, kTrueX, kTrueY,
                     kCornerRadius, kDarkIntensity, kBrightIntensity);
    ASSERT_TRUE(callAddKp(frameBuffer.data(), cornerKp));

    auto &feature = getActiveFeatures().front();
    Eigen::Vector2f ptEstimate(kTrueX, kTrueY);

    ASSERT_TRUE(callAlignFeatureICLK(feature, frameBuffer.data(), ptEstimate));
    EXPECT_NEAR(ptEstimate.x(), kTrueX, kSubPixelAlignmentTol);
    EXPECT_NEAR(ptEstimate.y(), kTrueY, kSubPixelAlignmentTol);
    EXPECT_NEAR(feature.ptCurr.x(), kTrueX, kSubPixelAlignmentTol);
    EXPECT_NEAR(feature.ptCurr.y(), kTrueY, kSubPixelAlignmentTol);
    EXPECT_NEAR(feature.ptPrev.x(), kTrueX, kFloatEps);
    EXPECT_NEAR(feature.ptPrev.y(), kTrueY, kFloatEps);
    EXPECT_EQ(feature.age, 1u);
}

// Verifies that a perturbed initial estimate converges back to the true corner coordinate (not just self-consistency).
TEST_F(HfTrackerTest, alignFeatureICLKConvergesFromSubPixelPerturbation) {
    constexpr float kTrueX = 310.0f;
    constexpr float kTrueY = 210.0f;
    Eigen::Vector2i initialKp(static_cast<int>(kTrueX), static_cast<int>(kTrueY));

    drawSmoothCorner(frameBuffer, kImageWidth, kImageHeight, kTrueX, kTrueY,
                     kCornerRadius, kDarkIntensity, kBrightIntensity);
    ASSERT_TRUE(callAddKp(frameBuffer.data(), initialKp));

    auto &feature = getActiveFeatures().front();
    const Eigen::Vector2f trueCornerPos = feature.ptCurr;

    // Perturb initial estimate away from true corner
    Eigen::Vector2f ptEstimate = trueCornerPos + Eigen::Vector2f(0.45f, -0.35f);

    ASSERT_TRUE(callAlignFeatureICLK(feature, frameBuffer.data(), ptEstimate));
    EXPECT_NEAR(ptEstimate.x(), trueCornerPos.x(), kSubPixelAlignmentTol);
    EXPECT_NEAR(ptEstimate.y(), trueCornerPos.y(), kSubPixelAlignmentTol);
    EXPECT_NEAR(feature.ptCurr.x(), trueCornerPos.x(), kSubPixelAlignmentTol);
    EXPECT_NEAR(feature.ptCurr.y(), trueCornerPos.y(), kSubPixelAlignmentTol);
    EXPECT_NEAR(feature.ptPrev.x(), trueCornerPos.x(), kFloatEps);
    EXPECT_NEAR(feature.ptPrev.y(), trueCornerPos.y(), kFloatEps);
    EXPECT_EQ(feature.age, 1u);
}

// Verifies that IC-LK accurately tracks a feature that moved by a sub-pixel shift in a new frame.
TEST_F(HfTrackerTest, alignFeatureICLKTracksTranslatedCorner) {
    constexpr float kStartX = 260.0f;
    constexpr float kStartY = 190.0f;
    constexpr float kShiftX = 0.60f;
    constexpr float kShiftY = -0.40f;

    drawSmoothCorner(frameBuffer, kImageWidth, kImageHeight, kStartX, kStartY,
                     kCornerRadius, kDarkIntensity, kBrightIntensity);
    Eigen::Vector2i initialKp(static_cast<int>(kStartX), static_cast<int>(kStartY));
    ASSERT_TRUE(callAddKp(frameBuffer.data(), initialKp));

    std::vector<uint8_t> nextFrame(kImageWidth * kImageHeight, kDefaultBackground);
    drawSmoothCorner(nextFrame, kImageWidth, kImageHeight, kStartX + kShiftX, kStartY + kShiftY,
                     kCornerRadius, kDarkIntensity, kBrightIntensity);

    auto &feature = getActiveFeatures().front();
    Eigen::Vector2f ptEstimate = feature.ptCurr;

    ASSERT_TRUE(callAlignFeatureICLK(feature, nextFrame.data(), ptEstimate));
    EXPECT_NEAR(feature.ptPrev.x(), kStartX, kFloatEps);
    EXPECT_NEAR(feature.ptPrev.y(), kStartY, kFloatEps);
    EXPECT_NEAR(feature.ptCurr.x(), kStartX + kShiftX, kSubPixelAlignmentTol);
    EXPECT_NEAR(feature.ptCurr.y(), kStartY + kShiftY, kSubPixelAlignmentTol);
    EXPECT_EQ(feature.age, 1u);
}

// Verifies that initial estimates violating any of the four image border margins are rejected without mutating feature state.
TEST_F(HfTrackerTest, alignFeatureICLKRejectsOutOfBoundsInitialEstimate) {
    Eigen::Vector2i cornerKp(200, 200);
    drawSmoothCorner(frameBuffer, kImageWidth, kImageHeight, 200.0f, 200.0f,
                     kCornerRadius, kDarkIntensity, kBrightIntensity);
    ASSERT_TRUE(callAddKp(frameBuffer.data(), cornerKp));

    auto &feature = getActiveFeatures().front();
    const float minBound = static_cast<float>(RADIUS + 1);
    const float maxBoundX = static_cast<float>(kImageWidth - RADIUS - 2);
    const float maxBoundY = static_cast<float>(kImageHeight - RADIUS - 2);

    const std::array<Eigen::Vector2f, 4> invalidEstimates = {{
        {minBound - 0.1f, 200.0f},
        {200.0f, minBound - 0.1f},
        {maxBoundX + 0.1f, 200.0f},
        {200.0f, maxBoundY + 0.1f}
    }};

    for (const auto &badInit : invalidEstimates) {
        Eigen::Vector2f ptEst = badInit;
        EXPECT_FALSE(callAlignFeatureICLK(feature, frameBuffer.data(), ptEst));
        EXPECT_EQ(feature.age, 0u);
        EXPECT_NEAR(feature.ptCurr.x(), 200.0f, kFloatEps);
        EXPECT_NEAR(feature.ptCurr.y(), 200.0f, kFloatEps);
    }
}

// Verifies that severe photometric corruption exceeding ICLK_MAD_THRESHOLD is rejected without mutating feature state.
TEST_F(HfTrackerTest, alignFeatureICLKRejectsHighMADPhotometricMismatch) {
    Eigen::Vector2i cornerKp(180, 140);
    drawSmoothCorner(frameBuffer, kImageWidth, kImageHeight, 180.0f, 140.0f,
                     kCornerRadius, kDarkIntensity, kBrightIntensity);
    ASSERT_TRUE(callAddKp(frameBuffer.data(), cornerKp));

    auto &feature = getActiveFeatures().front();
    std::vector<uint8_t> occludedFrame(kImageWidth * kImageHeight, 0);

    Eigen::Vector2f ptEstimate = feature.ptCurr;
    EXPECT_FALSE(callAlignFeatureICLK(feature, occludedFrame.data(), ptEstimate));
    EXPECT_EQ(feature.age, 0u);
    EXPECT_NEAR(feature.ptCurr.x(), 180.0f, kFloatEps);
    EXPECT_NEAR(feature.ptCurr.y(), 140.0f, kFloatEps);
}

// ============================================================================
// 6. 2-Point Epipolar RANSAC (ransac)
// ============================================================================

// Verifies that ransac immediately returns 0 when fewer than 2 features exist.
TEST_F(HfTrackerTest, ransacReturnsZeroWhenFewerThanTwoFeatures) {
    const Eigen::Matrix3f R = Eigen::Matrix3f::Identity();
    EXPECT_EQ(callRansac(R), 0);

    getActiveFeatures().push_back(makeFeature(0, 1, {200.0f, 200.0f}, {202.0f, 200.0f}));
    EXPECT_EQ(callRansac(R), 0);
}

// Verifies that ransac handles pure 3D camera rotation (R != I, t = 0) and prunes rotational outliers.
TEST_F(HfTrackerTest, ransacPureRotationRetainsInliersAndPrunesOutliers) {
    constexpr size_t kNumInliers = 24;
    const Eigen::Matrix3f R =
        Eigen::AngleAxisf(0.025f, Eigen::Vector3f(0.6f, -0.8f, 0.1f).normalized()).toRotationMatrix();
    const Eigen::Vector3f tZero = Eigen::Vector3f::Zero();

    const std::vector<Eigen::Vector2f> outlierOffsets = {
        {35.0f, -40.0f}, {-50.0f, 30.0f}, {45.0f, 45.0f}, {-30.0f, -60.0f}
    };

    populateSyntheticEpipolarFeatures(kNumInliers, R, tZero, outlierOffsets);

    const int inliers = callRansac(R);
    EXPECT_EQ(inliers, static_cast<int>(kNumInliers));
    EXPECT_EQ(getActiveFeatures().size(), kNumInliers);

    for (const auto &feat : getActiveFeatures()) {
        EXPECT_LT(feat.id, static_cast<uint32_t>(kNumInliers));
    }
}

// Verifies that ransac does not falsely classify pure translation (R = I, t != 0) as pure rotation,
// and prunes outliers with both positive and negative algebraic epipolar residuals.
TEST_F(HfTrackerTest, ransacPureTranslationPrunesPositiveAndNegativeOutliers) {
    constexpr size_t kNumInliers = 25;
    const Eigen::Matrix3f R = Eigen::Matrix3f::Identity();
    const Eigen::Vector3f t(0.15f, 0.0f, 0.0f);

    // Outliers displaced vertically in both +Y and -Y directions produce opposite-sign residuals
    const std::vector<Eigen::Vector2f> outlierOffsets = {
        {0.0f, 45.0f},
        {0.0f, -45.0f},
        {0.0f, 60.0f},
        {0.0f, -60.0f},
        {0.0f, 50.0f},
        {0.0f, -50.0f}
    };

    populateSyntheticEpipolarFeatures(kNumInliers, R, t, outlierOffsets);

    const int inliers = callRansac(R);
    EXPECT_EQ(inliers, static_cast<int>(kNumInliers));
    EXPECT_EQ(getActiveFeatures().size(), kNumInliers);

    for (const auto &feat : getActiveFeatures()) {
        EXPECT_LT(feat.id, static_cast<uint32_t>(kNumInliers));
    }
}

// Verifies 6-DoF epipolar consensus (R != I, t != 0) accurately counts inliers and compacts activeFeatures.
TEST_F(HfTrackerTest, ransac6DoFPrunesOutliersAndCompactsActiveFeatures) {
    constexpr size_t kNumInliers = 28;
    const Eigen::Matrix3f R =
        Eigen::AngleAxisf(0.018f, Eigen::Vector3f::UnitX()).toRotationMatrix();
    const Eigen::Vector3f t(0.12f, -0.06f, 0.25f);

    const std::vector<Eigen::Vector2f> outlierOffsets = {
        {55.0f, -60.0f}, {-70.0f, 45.0f}, {65.0f, 70.0f},
        {-55.0f, -50.0f}, {80.0f, -40.0f}, {-60.0f, 80.0f}
    };

    populateSyntheticEpipolarFeatures(kNumInliers, R, t, outlierOffsets);

    const int inliers = callRansac(R);
    EXPECT_GE(inliers, static_cast<int>(kNumInliers) - 1);
    EXPECT_LE(inliers, static_cast<int>(kNumInliers));
    EXPECT_EQ(getActiveFeatures().size(), static_cast<size_t>(inliers));

    for (const auto &feat : getActiveFeatures()) {
        EXPECT_LT(feat.id, static_cast<uint32_t>(kNumInliers));
    }
}

// Verifies that when the inlier count equals RANSAC_MIN_INLIERS exactly, activeFeatures is still compacted.
TEST_F(HfTrackerTest, ransacExactMinimumInlierBoundaryCompactsOutliers) {
    const size_t kNumInliers = static_cast<size_t>(RANSAC_MIN_INLIERS);
    const Eigen::Matrix3f R =
        Eigen::AngleAxisf(0.012f, Eigen::Vector3f::UnitY()).toRotationMatrix();
    const Eigen::Vector3f t(0.15f, 0.0f, 0.0f);

    const std::vector<Eigen::Vector2f> outlierOffsets = {
        {0.0f, 65.0f}, {0.0f, -65.0f}, {0.0f, 80.0f}, {0.0f, -80.0f}
    };

    populateSyntheticEpipolarFeatures(kNumInliers, R, t, outlierOffsets);

    const int inliers = callRansac(R);
    EXPECT_EQ(inliers, RANSAC_MIN_INLIERS);
    EXPECT_EQ(getActiveFeatures().size(), kNumInliers);
}

/// ============================================================================
// 7. Spatial Bucket Density & Proximity Enforcement (enforceBucketDensity)
// ============================================================================

// Verifies that up to 2 well-separated features in a single bucket are retained and ordered by age[cite: 1, 3].
TEST_F(HfTrackerTest, densityRetainsUpToTwoWellSeparatedFeaturesSortedByAge) {
    const float xBase = static_cast<float>(getXMargin()) + 4.0f;
    const float yBase = static_cast<float>(getYMargin()) + 4.0f;
    const float sep = MIN_DIST_IN_BUCKET + 5.0f;

    // Push younger first, then older to verify age sorting in gridOccupancy
    getActiveFeatures().push_back(makeFeature(10, 2, {xBase, yBase}, {xBase, yBase}));
    getActiveFeatures().push_back(makeFeature(20, 9, {xBase + sep, yBase}, {xBase + sep, yBase}));

    callEnforceBucketDensity();

    ASSERT_EQ(getActiveFeatures().size(), 2u);
    const auto &bucket0 = getGridOccupancy()[0];
    EXPECT_EQ(bucket0.count, 2u);
    EXPECT_EQ(bucket0.olderIdx, 1u);
    EXPECT_EQ(bucket0.youngerIdx, 0u);
}

// Verifies that when 2 features in a bucket are closer than the 6-pixel proximity limit, only the older track survives[cite: 1, 3].
TEST_F(HfTrackerTest, densityProximityCollisionRetainsOlderFeature) {
    const float xBase = static_cast<float>(getXMargin()) + 6.0f;
    const float yBase = static_cast<float>(getYMargin()) + 6.0f;
    const float closeStep = MIN_DIST_IN_BUCKET * 0.4f;

    // Case 1: Older feature appears first, younger second
    getActiveFeatures().push_back(makeFeature(1, 8, {xBase, yBase}, {xBase, yBase}));
    getActiveFeatures().push_back(makeFeature(2, 3, {xBase + closeStep, yBase}, {xBase + closeStep, yBase}));

    callEnforceBucketDensity();
    ASSERT_EQ(getActiveFeatures().size(), 1u);
    EXPECT_EQ(getActiveFeatures()[0].id, 1u);

    // Case 2: Younger feature appears first, older second
    getActiveFeatures().clear();
    getActiveFeatures().push_back(makeFeature(3, 2, {xBase, yBase}, {xBase, yBase}));
    getActiveFeatures().push_back(makeFeature(4, 11, {xBase + closeStep, yBase}, {xBase + closeStep, yBase}));

    callEnforceBucketDensity();
    ASSERT_EQ(getActiveFeatures().size(), 1u);
    EXPECT_EQ(getActiveFeatures()[0].id, 4u);
}

// Verifies that when a 3rd feature arrives in a full bucket, a 3-way age comparison discards the youngest track and keeps the two oldest[cite: 3].
TEST_F(HfTrackerTest, densityCapacityOverflowRetainsTwoOldestFeatures) {
    const float xBase = static_cast<float>(getXMargin()) + 2.0f;
    const float yBase = static_cast<float>(getYMargin()) + 2.0f;
    const float sep = MIN_DIST_IN_BUCKET + 2.0f;

    getActiveFeatures().push_back(makeFeature(101, 4, {xBase, yBase}, {xBase, yBase}));
    getActiveFeatures().push_back(makeFeature(102, 12, {xBase + sep, yBase}, {xBase + sep, yBase}));
    getActiveFeatures().push_back(makeFeature(103, 1, {xBase, yBase + sep}, {xBase, yBase + sep}));
    getActiveFeatures().push_back(makeFeature(104, 9, {xBase + sep, yBase + sep}, {xBase + sep, yBase + sep}));

    callEnforceBucketDensity();

    ASSERT_EQ(getActiveFeatures().size(), 2u);
    
    // IDs 102 (age 12) and 104 (age 9) should survive over age 4 and 1[cite: 3].
    EXPECT_EQ(getActiveFeatures()[0].id, 102u);
    EXPECT_EQ(getActiveFeatures()[1].id, 104u);
}

// Verifies that a 3rd feature colliding with an existing feature in a full bucket does not evict a well-separated feature.
TEST_F(HfTrackerTest, densityThirdFeatureProximityCollisionDoesNotEvictSeparatedFeature) {
    const float xBase = static_cast<float>(getXMargin()) + 3.0f;
    const float yBase = static_cast<float>(getYMargin()) + 3.0f;
    const float sep = MIN_DIST_IN_BUCKET + 4.0f;
    const float collisionOffset = MIN_DIST_IN_BUCKET * 0.2f;

    // Feature A (age 10) at (xBase, yBase) and Feature B (age 5) well-separated at (xBase + sep, yBase)
    getActiveFeatures().push_back(makeFeature(1, 10, {xBase, yBase}, {xBase, yBase}));
    getActiveFeatures().push_back(makeFeature(2, 5, {xBase + sep, yBase}, {xBase + sep, yBase}));

    // Feature C (age 8) lands directly on top of Feature A (closer than MIN_DIST_IN_BUCKET)
    getActiveFeatures().push_back(
        makeFeature(3, 8, {xBase + collisionOffset, yBase}, {xBase + collisionOffset, yBase}));

    callEnforceBucketDensity();

    // Resulting bucket must still contain 2 features separated by >= MIN_DIST_IN_BUCKET (IDs 1 and 2)
    ASSERT_EQ(getActiveFeatures().size(), 2u);
    const float finalDist = (getActiveFeatures()[0].ptCurr - getActiveFeatures()[1].ptCurr).norm();
    EXPECT_GE(finalDist, MIN_DIST_IN_BUCKET);
    EXPECT_EQ(getActiveFeatures()[0].id, 1u);
    EXPECT_EQ(getActiveFeatures()[1].id, 2u);
}

// Verifies that density enforcement operates independently across distinct spatial buckets.
TEST_F(HfTrackerTest, densityMultiBucketIndependence) {
    const Eigen::Vector2i b0 = callTopLeftBucket(0);
    const Eigen::Vector2i b1 = callTopLeftBucket(1);
    const float sep = MIN_DIST_IN_BUCKET + 2.0f;

    // 3 features in bucket 0 (1 should be pruned)
    getActiveFeatures().push_back(makeFeature(1, 10, {static_cast<float>(b0.x()) + 2.0f, static_cast<float>(b0.y()) + 2.0f}, {static_cast<float>(b0.x()) + 2.0f, static_cast<float>(b0.y()) + 2.0f}));
    getActiveFeatures().push_back(makeFeature(2, 8, {static_cast<float>(b0.x()) + 2.0f + sep, static_cast<float>(b0.y()) + 2.0f}, {static_cast<float>(b0.x()) + 2.0f + sep, static_cast<float>(b0.y()) + 2.0f}));
    getActiveFeatures().push_back(makeFeature(3, 2, {static_cast<float>(b0.x()) + 2.0f, static_cast<float>(b0.y()) + 2.0f + sep}, {static_cast<float>(b0.x()) + 2.0f, static_cast<float>(b0.y()) + 2.0f + sep}));

    // 1 feature in bucket 1 (must be untouched)
    getActiveFeatures().push_back(makeFeature(4, 1, {static_cast<float>(b1.x()) + 5.0f, static_cast<float>(b1.y()) + 5.0f}, {static_cast<float>(b1.x()) + 5.0f, static_cast<float>(b1.y()) + 5.0f}));

    callEnforceBucketDensity();

    ASSERT_EQ(getActiveFeatures().size(), 3u);
    EXPECT_EQ(getActiveFeatures()[0].id, 1u);
    EXPECT_EQ(getActiveFeatures()[1].id, 2u);
    EXPECT_EQ(getActiveFeatures()[2].id, 4u);
}

// ============================================================================
// 8. End-to-End Feature Tracking Pipeline (trackExistingFeatures)
// ============================================================================

// Verifies that passing a null image buffer triggers error handling and returns 0.
TEST_F(HfTrackerTest, trackExistingNullBufferTriggersErrorAndReturnsZero) {
    Eigen::Vector2i kp(200, 200);
    drawSmoothCorner(frameBuffer, kImageWidth, kImageHeight, 200.0f, 200.0f,
                     kCornerRadius, kDarkIntensity, kBrightIntensity);
    ASSERT_TRUE(callAddKp(frameBuffer.data(), kp));
    ASSERT_EQ(tracker->getActiveFeatures().size(), 1u);

    const size_t tracked = tracker->trackExistingFeatures(nullptr, Eigen::Matrix3f::Identity());
    EXPECT_EQ(tracked, 0u);
}

// Verifies that calling trackExistingFeatures with an empty active feature list returns 0 safely.
TEST_F(HfTrackerTest, trackExistingEmptyFeaturesReturnsZero) {
    const size_t tracked = tracker->trackExistingFeatures(frameBuffer.data(), Eigen::Matrix3f::Identity());
    EXPECT_EQ(tracked, 0u);
    EXPECT_TRUE(tracker->getActiveFeatures().empty());
}

// Verifies that when fewer than RANSAC_MIN_INLIERS survive alignment, trackExistingFeatures handles the error and returns 0.
TEST_F(HfTrackerTest, trackExistingInsufficientInliersTriggersErrorAndReturnsZero) {
    std::vector<Eigen::Vector2f> expectedPts;
    (void)seedAndRenderTransformedScene(4, 3, 10, Eigen::Matrix3f::Identity(),
                                        Eigen::Vector3f::Zero(), 2.5f, expectedPts);
    ASSERT_EQ(tracker->getActiveFeatures().size(), 12u);

    // Feed a completely blank/occluded frame so IC-LK rejects all features
    std::vector<uint8_t> blankFrame(kImageWidth * kImageHeight, 0);
    const size_t survived = tracker->trackExistingFeatures(blankFrame.data(), Eigen::Matrix3f::Identity());
    EXPECT_EQ(survived, 0u);
}

// Verifies end-to-end tracking under pure camera rotation (R != I, t = 0) guided by the K * R * K^-1 prior.
TEST_F(HfTrackerTest, trackExistingPureRotationGuidedByPrior) {
    constexpr size_t kCols = 5;
    constexpr size_t kRows = 4;
    constexpr size_t kTotal = kCols * kRows;

    // 0.005 rad (~2.6 px shift) pure pitch rotation with zero translation
    const Eigen::Matrix3f R = Eigen::AngleAxisf(0.005f, Eigen::Vector3f::UnitX()).toRotationMatrix();
    const Eigen::Vector3f tZero = Eigen::Vector3f::Zero();

    std::vector<Eigen::Vector2f> expectedPts;
    std::vector<uint8_t> nextFrame =
        seedAndRenderTransformedScene(kCols, kRows, 10, R, tZero, 3.0f, expectedPts);

    const size_t survived = tracker->trackExistingFeatures(nextFrame.data(), R);
    ASSERT_EQ(survived, kTotal);

    for (size_t i = 0; i < survived; i++) {
        const auto &feat = tracker->getActiveFeatures()[i];
        EXPECT_EQ(feat.age, 1u);
        EXPECT_NEAR(feat.ptCurr.x(), expectedPts[i].x(), kSubPixelAlignmentTol);
        EXPECT_NEAR(feat.ptCurr.y(), expectedPts[i].y(), kSubPixelAlignmentTol);
    }
}

// Verifies end-to-end tracking under pure translation (lateral parallax and forward looming expansion).
TEST_F(HfTrackerTest, trackExistingPureTranslationLateralAndLooming) {
    constexpr size_t kCols = 5;
    constexpr size_t kRows = 4;
    constexpr size_t kTotal = kCols * kRows;

    const Eigen::Matrix3f R = Eigen::Matrix3f::Identity();
    // Coupled lateral (+3 mm X) and forward (-6 mm Z) camera motion
    const Eigen::Vector3f t(0.003f, 0.0f, -0.006f);

    std::vector<Eigen::Vector2f> expectedPts;
    std::vector<uint8_t> nextFrame =
        seedAndRenderTransformedScene(kCols, kRows, 10, R, t, 2.5f, expectedPts);

    const size_t survived = tracker->trackExistingFeatures(nextFrame.data(), R);
    ASSERT_EQ(survived, kTotal);

    for (size_t i = 0; i < survived; i++) {
        const auto &feat = tracker->getActiveFeatures()[i];
        EXPECT_EQ(feat.age, 1u);
        EXPECT_NEAR(feat.ptCurr.x(), expectedPts[i].x(), kSubPixelAlignmentTol);
        EXPECT_NEAR(feat.ptCurr.y(), expectedPts[i].y(), kSubPixelAlignmentTol);
    }
}

// Verifies end-to-end 6-DoF tracking with simultaneous dynamic outlier rejection and cross-bucket migration.
TEST_F(HfTrackerTest, trackExistingCoupled6DoFWithOutlierPruningAndBucketMigration) {
    constexpr size_t kCols = 5;
    constexpr size_t kRows = 5;
    constexpr size_t kTotal = kCols * kRows;

    const Eigen::Matrix3f R = Eigen::AngleAxisf(0.001f, Eigen::Vector3f::UnitY()).toRotationMatrix();
    const Eigen::Vector3f t(0.003f, 0.0f, 0.0f);

    // Inject 4 outliers that drift perpendicular to the horizontal epipolar lines (both +Y and -Y)
    // but remain small enough (~1.9 px) for IC-LK to converge on the rogue corners
    const std::vector<std::pair<size_t, Eigen::Vector2f>> outliers = {
        {21, {0.0f, 2.1f}},
        {22, {0.0f, -2.1f}},
        {23, {0.0f, 2.2f}},
        {24, {0.0f, -2.2f}}
    };

    std::vector<Eigen::Vector2f> expectedPts;
    std::vector<uint8_t> nextFrame =
        seedAndRenderTransformedScene(kCols, kRows, 10, R, t, 2.0f, expectedPts, outliers);

    const size_t expectedInliers = kTotal - outliers.size();
    const size_t survived = tracker->trackExistingFeatures(nextFrame.data(), R);

    ASSERT_EQ(survived, expectedInliers);
    for (const auto &feat : tracker->getActiveFeatures()) {
        EXPECT_LT(feat.id, static_cast<uint32_t>(expectedInliers));
        EXPECT_EQ(feat.age, 1u);
    }
}

} // namespace PoolSLAM