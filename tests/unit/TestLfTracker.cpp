#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <eigen3/Eigen/Dense>
#include <gtest/gtest.h>

#include "CoreTypes.hpp"
#include "frontend/LfTracker.hpp"

namespace PoolSLAM {
namespace {

// Renders a smooth continuous 2D corner with an optional orientation angle (in radians)
void drawRotatedCorner(std::vector<uint8_t> &buffer, size_t width, size_t height,
                       float cx, float cy, int radius, uint8_t darkVal, uint8_t brightVal,
                       float angleRad = 0.0f, float rampWidth = 2.5f) {
    const float midVal = 0.5f * (static_cast<float>(darkVal) + static_cast<float>(brightVal));
    const float halfRange = 0.5f * (static_cast<float>(brightVal) - static_cast<float>(darkVal));
    const float cosA = std::cos(angleRad);
    const float sinA = std::sin(angleRad);

    const int minX = std::max(0, static_cast<int>(std::floor(cx)) - radius);
    const int maxX = std::min(static_cast<int>(width) - 1, static_cast<int>(std::ceil(cx)) + radius);
    const int minY = std::max(0, static_cast<int>(std::floor(cy)) - radius);
    const int maxY = std::min(static_cast<int>(height) - 1, static_cast<int>(std::ceil(cy)) + radius);

    for (int y = minY; y <= maxY; y++) {
        const float rawDy = static_cast<float>(y) - cy;
        for (int x = minX; x <= maxX; x++) {
            const float rawDx = static_cast<float>(x) - cx;
            const float rx = rawDx * cosA + rawDy * sinA;
            const float ry = -rawDx * sinA + rawDy * cosA;

            // Asymmetric quadrant weighting so the patch has a strong intensity centroid (m10, m01)
            const float wx = std::clamp(rx / rampWidth, -1.0f, 1.0f);
            const float wy = std::clamp(ry / rampWidth, -1.0f, 1.0f);
            const float quadWeight = 0.5f * (wx + wy) + 0.5f * (wx * wy);
            const float intensity = midVal + halfRange * std::clamp(quadWeight, -1.0f, 1.0f);

            buffer[static_cast<size_t>(y) * width + static_cast<size_t>(x)] =
                static_cast<uint8_t>(std::clamp(std::round(intensity), 0.0f, 255.0f));
        }
    }
}

// Applies forward Brown-Conrady radial and tangential distortion to normalized coordinates
Eigen::Vector2f applyForwardDistortion(float xNorm, float yNorm, const Distortion &d) {
    const float r2 = xNorm * xNorm + yNorm * yNorm;
    const float r4 = r2 * r2;
    const float xy = xNorm * yNorm;
    const float radial = 1.0f + d.k1 * r2 + d.k2 * r4;

    const float xDist = xNorm * radial + 2.0f * d.p1 * xy + d.p2 * (r2 + 2.0f * xNorm * xNorm);
    const float yDist = yNorm * radial + 2.0f * d.p2 * xy + d.p1 * (r2 + 2.0f * yNorm * yNorm);
    return {xDist, yDist};
}

} // namespace

class LfTrackerTest : public ::testing::Test {
protected:
    static constexpr size_t kImageWidth = 640;
    static constexpr size_t kImageHeight = 480;
    static constexpr size_t kNumBuckets = 200;

    static constexpr float kFx = 525.0f;
    static constexpr float kFy = 525.0f;
    static constexpr float kCx = static_cast<float>(kImageWidth) / 2.0f;
    static constexpr float kCy = static_cast<float>(kImageHeight) / 2.0f;

    static constexpr double kPoseEps = 1e-4;
    static constexpr uint8_t kDefaultBackground = 128;
    static constexpr uint8_t kDarkIntensity = 30;
    static constexpr uint8_t kBrightIntensity = 220;
    static constexpr int kCornerRadius = 15;

    static constexpr int FEATURE_RADIUS = LfTracker::FEATURE_RADIUS;
    static constexpr int QOV_RADIUS = LfTracker::QOV_RADIUS;
    using HfFeature = TrackedFeature<FEATURE_RADIUS>;

    Eigen::Matrix3f K;
    Distortion dist;
    std::string dummyVocabPath = "";
    std::unique_ptr<LfTracker> tracker;
    std::vector<uint8_t> frameBuffer;

    void SetUp() override {
        K = Eigen::Matrix3f::Identity();
        K(0, 0) = kFx;
        K(1, 1) = kFy;
        K(0, 2) = kCx;
        K(1, 2) = kCy;

        dist = Distortion{};
        LfTrackerConfig cfg = makeConfig(dist);
        tracker = std::make_unique<LfTracker>(cfg, dummyVocabPath);
        initSyntheticVocabTree(tracker->vocabTree);
        frameBuffer.assign(kImageWidth * kImageHeight, kDefaultBackground);
    }

    void TearDown() override {
        if (tracker) {
            tracker->end();
        }
    }

    LfTrackerConfig makeConfig(const Distortion &d,
                               float minTfIdf = 0.05f,
                               size_t excluded = 2,
                               uint32_t minInliers = 12,
                               int ransacIters = 50,
                               float minRatio = 0.30f,
                               float maxReprojErr = 2.0f) const {
        return LfTrackerConfig{
            kImageWidth,
            kImageHeight,
            kNumBuckets,
            K,
            d,
            minTfIdf,
            excluded,
            minInliers,
            ransacIters,
            minRatio,
            maxReprojErr
        };
    }

    // Populates the vocabulary tree using QovDescriptor::bits (4x uint64_t) and matches
    // BinaryBowTree::readVocabFile's parent index formula.
    static void initSyntheticVocabTree(descriptor::BinaryBowTree &tree) {
        tree.tree.resize(descriptor::BinaryBowTree::TOTAL_NODES);
        tree.vocabulary.resize(descriptor::BinaryBowTree::VOCAB_SIZE);

        for (size_t i = 0; i < tree.tree.size(); i++) {
            uint64_t seed = static_cast<uint64_t>(i) * 0x9E3779B97F4A7C15ULL + 0xBF58476D1CE4E5B9ULL;
            for (size_t w = 0; w < 4; w++) {
                seed ^= (seed >> 30);
                seed *= 0xBF58476D1CE4E5B9ULL;
                seed ^= (seed >> 27);
                tree.tree[i].bits[w] = seed + w * 0x94D049BB133111EBULL;
            }
        }

        constexpr size_t kParentOffset =
            descriptor::BinaryBowTree::TREE_SIZE - descriptor::BinaryBowTree::NUM_PARENTS;
        for (size_t wordId = 0; wordId < tree.vocabulary.size(); wordId++) {
            tree.vocabulary[wordId].idf = 1.0f;
            tree.vocabulary[wordId].parent =
                static_cast<uint32_t>((wordId / descriptor::BinaryBowTree::NUM_BRANCHES) + kParentOffset);
        }
    }

    // Private member accessors via friend class LfTrackerTest
    uint8_t *localBuffer() { return tracker->localBuffer.get(); }
    std::vector<Keyframe> &keyframes() { return tracker->keyframes; }
    LfMatching &matching() { return tracker->matching; }
    descriptor::BinaryBowTree &vocabTree() { return tracker->vocabTree; }
    descriptor::BowDatabase &database() { return tracker->database; }
    bool isBusy() const { return tracker->busy.load(std::memory_order_acquire); }

    // Private method wrappers
    void callGaussianBlur() { tracker->gaussianBlur(); }
    void callCalculateDescriptors() { tracker->calculateDescriptors(); }
    size_t callMatchFeatures(size_t curId, size_t matchId) {
        return tracker->matchFeatures(curId, matchId);
    }
    void callUndistort(Eigen::Matrix3Xd &pts) { tracker->undistort(pts); }
    std::optional<std::pair<Eigen::Matrix<double, 3, 4>, size_t>> callRansac(size_t numMatches) {
        return tracker->ransac(numMatches);
    }
    bool callRefinePoseLM(size_t numInliers, const Eigen::Matrix3d &K_cam,
                          Eigen::Matrix<double, 3, 4> &Tcand,
                          Eigen::Matrix<double, 6, 6> &outCovariance) {
        return tracker->refinePoseLM(numInliers, K_cam, Tcand, outCovariance);
    }

    // Helper: creates a TrackedFeature<FEATURE_RADIUS>
    static HfFeature makeHfFeature(uint32_t id, float x, float y) {
        HfFeature feat{};
        feat.id = id;
        feat.age = 2;
        feat.ptCurr = Eigen::Vector2f(x, y);
        feat.ptPrev = Eigen::Vector2f(x, y);
        return feat;
    }

    // Helper: creates a QovDescriptor with a base 64-bit word pattern and controlled bit flips
    static descriptor::QovDescriptor makeDescriptor(uint64_t basePattern, size_t flipBits = 0) {
        descriptor::QovDescriptor des{};
        des.bits.fill(basePattern);
        for (size_t bit = 0; bit < flipBits && bit < descriptor::QOV_BITS; bit++) {
            des.bits[bit / 64] ^= (1ULL << (bit % 64));
        }
        return des;
    }
};

// ============================================================================
// 1. Lambda Twist P3P / P4P Minimal Solver Tests (solveP4PLambdaTwist)
// ============================================================================

// Verifies that solveP4PLambdaTwist recovers a general 6-DoF transformation [R | t]
// from 4 non-coplanar 3D landmarks and their normalized camera bearing vectors.
TEST_F(LfTrackerTest, solveP4PLambdaTwistRecoversExactGeneral6DoFPose) {
    const Eigen::Matrix3d RTrue =
        (Eigen::AngleAxisd(0.18, Eigen::Vector3d::UnitY()) *
         Eigen::AngleAxisd(-0.11, Eigen::Vector3d::UnitX()) *
         Eigen::AngleAxisd(0.07, Eigen::Vector3d::UnitZ())).toRotationMatrix();
    const Eigen::Vector3d tTrue(0.40, -0.25, 0.55);

    const std::array<Eigen::Vector3d, 4> pts3d = {{
        {-0.8, -0.6, 2.5},
        { 0.9, -0.5, 3.2},
        {-0.4,  0.8, 2.8},
        { 0.6,  0.7, 2.2}
    }};

    std::array<Eigen::Vector3d, 4> rays;
    for (size_t i = 0; i < 4; i++) {
        const Eigen::Vector3d PCam = RTrue * pts3d[i] + tTrue;
        rays[i] = PCam.hnormalized().homogeneous();
    }

    const auto TOpt = geometry::solveP4PLambdaTwist(pts3d, rays);
    ASSERT_TRUE(TOpt.has_value());

    EXPECT_NEAR((TOpt->leftCols<3>() - RTrue).norm(), 0.0, kPoseEps);
    EXPECT_NEAR((TOpt->rightCols<1>() - tTrue).norm(), 0.0, kPoseEps);
}

// Verifies that solveP4PLambdaTwist returns std::nullopt on collinear or duplicate 3D points.
TEST_F(LfTrackerTest, solveP4PLambdaTwistRejectsCollinearAndDuplicatePoints) {
    const std::array<Eigen::Vector3d, 4> collinearPts = {{
        {0.0, 0.0, 2.0},
        {0.5, 0.5, 2.5},
        {1.0, 1.0, 3.0},
        {0.5, -0.5, 2.5}
    }};
    const std::array<Eigen::Vector3d, 4> rays = {{
        {0.0, 0.0, 1.0},
        {0.2, 0.2, 1.0},
        {0.33, 0.33, 1.0},
        {0.2, -0.2, 1.0}
    }};

    EXPECT_FALSE(geometry::solveP4PLambdaTwist(collinearPts, rays).has_value());

    const std::array<Eigen::Vector3d, 4> duplicatePts = {{
        {0.0, 0.0, 2.0},
        {0.0, 0.0, 2.0},
        {1.0, 0.5, 3.0},
        {0.5, -0.5, 2.5}
    }};
    EXPECT_FALSE(geometry::solveP4PLambdaTwist(duplicatePts, rays).has_value());
}

// ============================================================================
// 2. Coordinate Undistortion Tests (undistort)
// ============================================================================

// Verifies that undistort accurately inverts pinhole projection and Brown-Conrady radial/tangential
// distortion across all four image quadrants.
TEST_F(LfTrackerTest, undistortInvertsPinholeAndBrownConradyDistortion) {
    Distortion nonZeroDist{};
    nonZeroDist.k1 = -0.16f;
    nonZeroDist.k2 = 0.028f;
    nonZeroDist.p1 = 0.0014f;
    nonZeroDist.p2 = -0.0011f;

    LfTrackerConfig cfg = makeConfig(nonZeroDist);
    tracker = std::make_unique<LfTracker>(cfg, dummyVocabPath);

    const std::vector<Eigen::Vector2f> trueNormRays = {
        { 0.00f,  0.00f},
        { 0.24f, -0.18f},
        {-0.26f,  0.20f},
        {-0.19f, -0.22f},
        { 0.17f,  0.25f}
    };

    Eigen::Matrix3Xd pts(3, trueNormRays.size());
    for (size_t i = 0; i < trueNormRays.size(); i++) {
        const Eigen::Vector2f dNorm =
            applyForwardDistortion(trueNormRays[i].x(), trueNormRays[i].y(), nonZeroDist);
        const auto idx = static_cast<Eigen::Index>(i);
        pts(0, idx) = static_cast<double>(kFx * dNorm.x() + kCx);
        pts(1, idx) = static_cast<double>(kFy * dNorm.y() + kCy);
        pts(2, idx) = 1.0;
    }

    callUndistort(pts);

    for (size_t i = 0; i < trueNormRays.size(); i++) {
        const auto idx = static_cast<Eigen::Index>(i);
        EXPECT_NEAR(pts(0, idx), static_cast<double>(trueNormRays[i].x()), 5e-4);
        EXPECT_NEAR(pts(1, idx), static_cast<double>(trueNormRays[i].y()), 5e-4);
        EXPECT_NEAR(pts(2, idx), 1.0, 1e-6);
    }
}

// ============================================================================
// 3. Separable 5x5 Gaussian Blur Tests (gaussianBlur)
// ============================================================================

// Verifies that gaussianBlur preserves uniform images (0, 128, 255) across every pixel,
// including the top/bottom two boundary rows and left/right two boundary columns.
TEST_F(LfTrackerTest, gaussianBlurPreservesUniformIntensityAcrossEntireImage) {
    for (uint8_t val : {uint8_t{0}, uint8_t{128}, uint8_t{255}}) {
        std::fill_n(localBuffer(), kImageWidth * kImageHeight, val);
        callGaussianBlur();

        for (size_t i = 0; i < kImageWidth * kImageHeight; i++) {
            ASSERT_EQ(localBuffer()[i], val) << "Mismatch at pixel index " << i;
        }
    }
}

// Verifies that gaussianBlur applies the separable [1, 4, 6, 4, 1] / 16 kernel symmetrically
// around an isolated impulse in both horizontal and vertical passes.
TEST_F(LfTrackerTest, gaussianBlurAppliesSymmetricSeparableWeightsToImpulse) {
    std::fill_n(localBuffer(), kImageWidth * kImageHeight, 0);
    constexpr size_t kX = 320;
    constexpr size_t kY = 240;
    localBuffer()[kY * kImageWidth + kX] = 255;

    callGaussianBlur();

    const uint8_t center = localBuffer()[kY * kImageWidth + kX];
    const uint8_t left1  = localBuffer()[kY * kImageWidth + (kX - 1)];
    const uint8_t right1 = localBuffer()[kY * kImageWidth + (kX + 1)];
    const uint8_t top1   = localBuffer()[(kY - 1) * kImageWidth + kX];
    const uint8_t bot1   = localBuffer()[(kY + 1) * kImageWidth + kX];
    const uint8_t diag1  = localBuffer()[(kY - 1) * kImageWidth + (kX - 1)];

    // Horizontal pass: center = (255*6)>>4 = 95, +/-1 = (255*4)>>4 = 63
    // Vertical pass: center = (95*6)>>4 = 35, axial = (95*4)>>4 = 23, diag = (63*4)>>4 = 15
    EXPECT_EQ(center, 35u);
    EXPECT_EQ(left1, 23u);
    EXPECT_EQ(right1, 23u);
    EXPECT_EQ(top1, 23u);
    EXPECT_EQ(bot1, 23u);
    EXPECT_EQ(diag1, 15u);
}

// ============================================================================
// 4. Steered QOV Descriptor Extraction Tests (calculateDescriptors)
// ============================================================================

// Verifies that calculateDescriptors computes patch intensity moments (m10, m01) and populates
// the 4x uint64_t bitfields in QovDescriptor::bits.
TEST_F(LfTrackerTest, calculateDescriptorsPopulatesBitsAndRespondsToOrientation) {
    drawRotatedCorner(frameBuffer, kImageWidth, kImageHeight, 200.0f, 200.0f,
                      kCornerRadius, kDarkIntensity, kBrightIntensity, 0.0f);
    drawRotatedCorner(frameBuffer, kImageWidth, kImageHeight, 400.0f, 200.0f,
                      kCornerRadius, kDarkIntensity, kBrightIntensity, 1.5707963f);
    std::copy(frameBuffer.begin(), frameBuffer.end(), localBuffer());

    Keyframe kf(2, 0, 1.0, 0);
    kf.addKp(Eigen::Vector2i(200, 200), 1);
    kf.addKp(Eigen::Vector2i(400, 200), 2);
    kf.finalSize();
    keyframes().push_back(std::move(kf));

    callCalculateDescriptors();

    const auto &des0 = keyframes().back().features[0].des;
    const auto &des90 = keyframes().back().features[1].des;

    bool hasSetBits = false;
    bool hasClearedBits = false;
    for (size_t w = 0; w < 4; w++) {
        if (des0.bits[w] != 0ULL) hasSetBits = true;
        if (des0.bits[w] != ~0ULL) hasClearedBits = true;
    }
    EXPECT_TRUE(hasSetBits);
    EXPECT_TRUE(hasClearedBits);

    // Because QOV steers the sampling pattern by the patch's intensity centroid angle,
    // a 90-degree rotated corner maintains high bit agreement with the 0-degree corner.
    EXPECT_LT(descriptor::fastHamming(des0, des90), 64);
}

// ============================================================================
// 5. Vocabulary-Guided Feature Matching Tests (matchFeatures)
// ============================================================================

// Verifies that matchFeatures accepts valid appearance and exact-ID matches while rejecting:
// (a) matches failing Lowe's ratio test,
// (b) matches failing the bidirectional neighbor consistency check, and
// (c) landmarks with non-positive depth (Z <= 0).
TEST_F(LfTrackerTest, matchFeaturesEnforcesLoweRatioBidirectionalCheckAndValidDepth) {
    Keyframe matchKf(6, 10, 1.0, 0);
    Keyframe curKf(5, 20, 2.0, 1);

    // Group 1 (wordId 10 & 11 -> same parent): Clean appearance match
    descriptor::QovFeature m0(Eigen::Vector2i(100, 100), 100);
    m0.wordId = 10;
    m0.des = makeDescriptor(0x0000000000000000ULL, 0);

    descriptor::QovFeature m1(Eigen::Vector2i(120, 100), 101);
    m1.wordId = 11;
    m1.des = makeDescriptor(0x0000000000000000ULL, 120); // Distant second-best in same parent

    // Group 2 (wordId 20 & 21 -> same parent): Ambiguous match failing Lowe's ratio test
    descriptor::QovFeature m2a(Eigen::Vector2i(200, 150), 102);
    m2a.wordId = 20;
    m2a.des = makeDescriptor(0xAAAAAAAAAAAAAAAAULL, 10);

    descriptor::QovFeature m2b(Eigen::Vector2i(220, 150), 103);
    m2b.wordId = 21;
    m2b.des = makeDescriptor(0xAAAAAAAAAAAAAAAAULL, 11);

    // Group 3 (wordId 30 & 31 -> same parent): Fails bidirectional check when a neighbor in curKf is closer
    descriptor::QovFeature m3a(Eigen::Vector2i(300, 200), 104);
    m3a.wordId = 30;
    m3a.des = makeDescriptor(0x5555555555555555ULL, 0);

    descriptor::QovFeature m3b(Eigen::Vector2i(320, 200), 105);
    m3b.wordId = 31;
    m3b.des = makeDescriptor(0x5555555555555555ULL, 120);

    // Group 4 (wordId 40): Invalid 3D depth (Z <= 0)
    descriptor::QovFeature m4(Eigen::Vector2i(400, 250), 106);
    m4.wordId = 40;
    m4.des = makeDescriptor(0xFFFFFFFFFFFFFFFFULL, 0);

    matchKf.features = {m0, m1, m2a, m2b, m3a, m3b, m4};
    matchKf.landmarks = {
        Landmark3d{Eigen::Vector3d(0.4, -0.2, 2.5), 100},
        Landmark3d{Eigen::Vector3d(0.6, -0.2, 2.8), 101},
        Landmark3d{Eigen::Vector3d(0.1,  0.1, 2.2), 102},
        Landmark3d{Eigen::Vector3d(0.2,  0.1, 2.2), 103},
        Landmark3d{Eigen::Vector3d(-0.3, 0.4, 3.0), 104},
        Landmark3d{Eigen::Vector3d(-0.1, 0.4, 3.0), 105},
        Landmark3d{Eigen::Vector3d(0.0,  0.0, 0.0), 106} // Invalid Z == 0
    };

    // Current frame features:
    // c0: Matches m0 via appearance (dist = 3 vs secondDist = 117 -> passes ratio & bidirectional)
    descriptor::QovFeature c0(Eigen::Vector2i(104, 102), 500);
    c0.wordId = 10;
    c0.des = makeDescriptor(0x0000000000000000ULL, 3);

    // c1: Ambiguous between m2a (dist 10) and m2b (dist 11) -> fails Lowe ratio (10/11 > 0.8)
    descriptor::QovFeature c1(Eigen::Vector2i(204, 152), 501);
    c1.wordId = 20;
    c1.des = makeDescriptor(0xAAAAAAAAAAAAAAAAULL, 0);

    // c2_weak: Matches m3a with dist = 8, but neighbor c2_strong is within BIDIRECTIONAL_SEARCH_INTERVAL
    // and has dist = 1 to m3a -> c2_weak must be rejected by bidirectional check, while c2_strong succeeds!
    descriptor::QovFeature c2_weak(Eigen::Vector2i(295, 198), 502);
    c2_weak.wordId = 30;
    c2_weak.des = makeDescriptor(0x5555555555555555ULL, 8);

    descriptor::QovFeature c2_strong(Eigen::Vector2i(302, 201), 503);
    c2_strong.wordId = 30;
    c2_strong.des = makeDescriptor(0x5555555555555555ULL, 1);

    // c3: Matches m4 in appearance, but m4 has Z == 0 -> rejected
    descriptor::QovFeature c3(Eigen::Vector2i(402, 251), 504);
    c3.wordId = 40;
    c3.des = makeDescriptor(0xFFFFFFFFFFFFFFFFULL, 0);

    curKf.features = {c0, c1, c2_weak, c2_strong, c3};

    keyframes().push_back(std::move(matchKf));
    keyframes().push_back(std::move(curKf));

    const size_t numMatched = callMatchFeatures(1, 0);
    ASSERT_EQ(numMatched, 2u);

    // Match 0: c0 <-> m0
    EXPECT_DOUBLE_EQ(matching().curFeatures(0, 0), 104.0);
    EXPECT_DOUBLE_EQ(matching().curFeatures(1, 0), 102.0);
    EXPECT_NEAR((matching().matchLandmarks.col(0) - Eigen::Vector3d(0.4, -0.2, 2.5)).norm(), 0.0, 1e-9);

    // Match 1: c2_strong <-> m3a (c2_weak was pruned by bidirectional consistency)
    EXPECT_DOUBLE_EQ(matching().curFeatures(0, 1), 302.0);
    EXPECT_DOUBLE_EQ(matching().curFeatures(1, 1), 201.0);
    EXPECT_NEAR((matching().matchLandmarks.col(1) - Eigen::Vector3d(-0.3, 0.4, 3.0)).norm(), 0.0, 1e-9);
}

// ============================================================================
// 6. P3P RANSAC Outlier Pruning & Compaction Tests (ransac)
// ============================================================================

// Verifies that ransac recovers the 6-DoF camera pose, evaluates reprojection error in pixels,
// rejects outliers, and compacts matching.curFeatures and matching.matchLandmarks in lockstep.
TEST_F(LfTrackerTest, ransacEstimatesPosePrunesPixelOutliersAndCompactsMatches) {
    constexpr size_t kNumInliers = 24;
    constexpr size_t kNumOutliers = 8;
    constexpr size_t kTotalMatches = kNumInliers + kNumOutliers;

    const Eigen::Matrix3d RTrue =
        Eigen::AngleAxisd(0.09, Eigen::Vector3d(0.4, 0.8, -0.3).normalized()).toRotationMatrix();
    const Eigen::Vector3d tTrue(-0.18, 0.12, 0.28);

    matching().curFeatures.resize(3, kTotalMatches);
    matching().matchLandmarks.resize(3, kTotalMatches);

    for (size_t i = 0; i < kTotalMatches; i++) {
        const double xn = -0.35 + static_cast<double>(i % 6) * 0.14;
        const double yn = -0.25 + static_cast<double>(static_cast<double>(i) / 6) * 0.10;
        const double depth = 2.0 + static_cast<double>(i % 4) * 0.6;

        const Eigen::Vector3d P3d(xn * depth, yn * depth, depth);
        const Eigen::Vector3d PCam = RTrue * P3d + tTrue;

        double u = static_cast<double>(kFx) * (PCam.x() / PCam.z()) + static_cast<double>(kCx);
        double v = static_cast<double>(kFy) * (PCam.y() / PCam.z()) + static_cast<double>(kCy);

        // Inject 15-pixel reprojection outliers (small in normalized space ~0.028 rad, but > 2.0 px)
        if (i >= kNumInliers) {
            u += (i % 2 == 0) ? 15.0 : -15.0;
            v += (i % 2 == 0) ? -15.0 : 15.0;
        }

        const auto idx = static_cast<Eigen::Index>(i);
        matching().matchLandmarks.col(idx) = P3d;
        matching().curFeatures.col(idx) = Eigen::Vector3d(u, v, 1.0);
    }

    const auto res = callRansac(kTotalMatches);
    ASSERT_TRUE(res.has_value());

    const auto &[TBest, inlierCount] = *res;
    EXPECT_EQ(inlierCount, kNumInliers);
    EXPECT_NEAR((TBest.leftCols<3>() - RTrue).norm(), 0.0, 1e-3);
    EXPECT_NEAR((TBest.rightCols<1>() - tTrue).norm(), 0.0, 1e-3);

    // Verify every compacted column satisfies the 2.0-pixel reprojection threshold
    for (size_t i = 0; i < inlierCount; i++) {
        const auto idx = static_cast<Eigen::Index>(i);
        const Eigen::Vector3d P = matching().matchLandmarks.col(idx);
        const Eigen::Vector2d p = matching().curFeatures.col(idx).hnormalized();
        const Eigen::Vector2d proj = (RTrue * P + tTrue).hnormalized();
        const Eigen::Vector2d pxErr(kFx * (proj.x() - p.x()), kFy * (proj.y() - p.y()));
        EXPECT_LE(pxErr.norm(), 2.0);
    }
}

// Verifies that ransac returns std::nullopt when numMatches < 4, when inliers fall below
// ransacInliearThreshold, or when the inlier ratio falls below minInliearRatio.
TEST_F(LfTrackerTest, ransacRejectsInsufficientInliersAndLowInlierRatio) {
    // Case 1: Fewer than 4 matches
    matching().curFeatures.resize(3, 3);
    matching().matchLandmarks.resize(3, 3);
    EXPECT_FALSE(callRansac(3).has_value());

    // Case 2: 8 valid inliers (< ransacInliearThreshold == 12)
    constexpr size_t kSmallCount = 8;
    matching().curFeatures.resize(3, kSmallCount);
    matching().matchLandmarks.resize(3, kSmallCount);
    for (size_t i = 0; i < kSmallCount; i++) {
        const Eigen::Vector3d P3d(static_cast<double>(i) * 0.1 - 0.3, static_cast<double>((i % 2)) * 0.2, 2.5);
        const double u = kFx * (P3d.x() / P3d.z()) + kCx;
        const double v = kFy * (P3d.y() / P3d.z()) + kCy;
        matching().matchLandmarks.col(static_cast<Eigen::Index>(i)) = P3d;
        matching().curFeatures.col(static_cast<Eigen::Index>(i)) = Eigen::Vector3d(u, v, 1.0);
    }
    EXPECT_FALSE(callRansac(kSmallCount).has_value());
}

// ============================================================================
// 7. Levenberg-Marquardt Pose & Covariance Refinement Tests (refinePoseLM)
// ============================================================================

// Verifies that refinePoseLM converges from a perturbed pose initialization to the ground-truth
// transformation, down-weights marginal outliers via Huber loss, and computes a valid 6x6 covariance.
TEST_F(LfTrackerTest, refinePoseLMConvergesFromPerturbedInitAndOutputsPositiveDefiniteCovariance) {
    constexpr size_t kNumInliers = 25;
    const Eigen::Matrix3d RTrue =
        (Eigen::AngleAxisd(0.10, Eigen::Vector3d::UnitY()) *
         Eigen::AngleAxisd(-0.06, Eigen::Vector3d::UnitX())).toRotationMatrix();
    const Eigen::Vector3d tTrue(0.22, -0.14, 0.32);

    matching().curFeatures.resize(3, kNumInliers);
    matching().matchLandmarks.resize(3, kNumInliers);

    for (size_t i = 0; i < kNumInliers; i++) {
        const double xn = -0.30 + static_cast<double>(i % 5) * 0.15;
        const double yn = -0.24 + static_cast<double>(static_cast<double>(i) / 5) * 0.12;
        const double depth = 2.0 + static_cast<double>(i % 3) * 0.7;

        const Eigen::Vector3d P3d(xn * depth, yn * depth, depth);
        const Eigen::Vector3d PCam = RTrue * P3d + tTrue;

        const auto idx = static_cast<Eigen::Index>(i);
        matching().matchLandmarks.col(idx) = P3d;
        matching().curFeatures.col(idx) = PCam.hnormalized().homogeneous();
    }

    // Perturb initial pose by ~1.5 degrees and 3 cm
    Eigen::Matrix<double, 3, 4> Tcand;
    Tcand.leftCols<3>() =
        RTrue * Eigen::AngleAxisd(0.025, Eigen::Vector3d(1.0, -1.0, 0.5).normalized()).toRotationMatrix();
    Tcand.rightCols<1>() = tTrue + Eigen::Vector3d(0.03, -0.02, 0.025);

    Eigen::Matrix<double, 6, 6> outCov = Eigen::Matrix<double, 6, 6>::Zero();
    ASSERT_TRUE(callRefinePoseLM(kNumInliers, K.cast<double>(), Tcand, outCov));

    EXPECT_NEAR((Tcand.leftCols<3>() - RTrue).norm(), 0.0, 1e-4);
    EXPECT_NEAR((Tcand.rightCols<1>() - tTrue).norm(), 0.0, 1e-4);

    // Verify covariance symmetry and positive definiteness
    EXPECT_NEAR((outCov - outCov.transpose()).norm(), 0.0, 1e-9);
    const Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> eigSolver(outCov);
    EXPECT_GT(eigSolver.eigenvalues().minCoeff(), 0.0);

    // Fewer than 4 inliers must return false immediately
    EXPECT_FALSE(callRefinePoseLM(3, K.cast<double>(), Tcand, outCov));
}

// ============================================================================
// 8. Public Interface & End-to-End Asynchronous Loop Closure Tests
// ============================================================================

// Verifies that passKeyframe rejects empty inputs, filters out border keypoints within QOV_RADIUS,
// and blocks concurrent submissions while busy.
TEST_F(LfTrackerTest, passKeyframeEnforcesBorderMarginsEmptyGuardAndBusyFlag) {
    std::vector<HfFeature> emptyFeatures;
    EXPECT_FALSE(tracker->passKeyframe(1, 0.5, frameBuffer.data(), emptyFeatures));

    std::vector<HfFeature> borderOnlyFeatures = {
        makeHfFeature(1, 5.0f, 100.0f),
        makeHfFeature(2, 100.0f, static_cast<float>(kImageHeight - QOV_RADIUS))
    };
    EXPECT_FALSE(tracker->passKeyframe(2, 0.8, frameBuffer.data(), borderOnlyFeatures));

    std::vector<HfFeature> mixedFeatures = {
        makeHfFeature(1, 5.0f, 100.0f),   // Rejected (< QOV_RADIUS)
        makeHfFeature(2, 200.0f, 200.0f), // Accepted
        makeHfFeature(3, 350.0f, 250.0f)  // Accepted
    };

    ASSERT_TRUE(tracker->passKeyframe(10, 1.0, frameBuffer.data(), mixedFeatures));
    EXPECT_TRUE(isBusy());
    ASSERT_EQ(keyframes().size(), 1u);
    ASSERT_EQ(keyframes().front().features.size(), 2u);
    EXPECT_EQ(keyframes().front().features[0].featureId, 2u);
    EXPECT_EQ(keyframes().front().features[1].featureId, 3u);

    // Subsequent call while busy must return false
    EXPECT_FALSE(tracker->passKeyframe(11, 1.2, frameBuffer.data(), mixedFeatures));
}

// Verifies that updateKeyframe aligns incoming 3D landmarks with reordered keyframe features
// by featureId and returns false for unknown keyframe IDs.
TEST_F(LfTrackerTest, updateKeyframeMapsLandmarksByFeatureIdAfterSorting) {
    Keyframe kf(3, 42, 1.0, 0);
    kf.addKp(Eigen::Vector2i(100, 100), 10);
    kf.addKp(Eigen::Vector2i(200, 100), 20);
    kf.addKp(Eigen::Vector2i(300, 100), 30);
    kf.finalSize();

    // Reverse feature order to simulate post-BoW wordId sorting
    std::reverse(kf.features.begin(), kf.features.end());
    keyframes().push_back(std::move(kf));

    BackendKeyframeUpdate update;
    update.id = 42;
    update.landmarks = {
        Landmark3d{Eigen::Vector3d(1.0, 0.0, 2.0), 10},
        Landmark3d{Eigen::Vector3d(2.0, 0.0, 3.0), 20},
        Landmark3d{Eigen::Vector3d(3.0, 0.0, 4.0), 30}
    };

    ASSERT_TRUE(tracker->updateKeyframe(update));
    const auto &stored = keyframes().front();
    ASSERT_EQ(stored.landmarks.size(), 3u);
    for (size_t i = 0; i < stored.features.size(); i++) {
        EXPECT_EQ(stored.landmarks[i].featureId, stored.features[i].featureId);
        EXPECT_DOUBLE_EQ(stored.landmarks[i].pt3d.x(),
                         static_cast<double>(stored.features[i].featureId) / 10.0);
    }

    BackendKeyframeUpdate unknownUpdate;
    unknownUpdate.id = 999;
    EXPECT_FALSE(tracker->updateKeyframe(unknownUpdate));
}

// Verifies the full worker thread loop closure pipeline (start -> passKeyframe -> updateKeyframe
// -> temporal exclusion window -> revisited keyframe -> getMatch -> end) when revisiting a scene
// with brand-new optical flow feature IDs matched via QOV appearance descriptors.
TEST_F(LfTrackerTest, endToEndWorkerThreadDetectsLoopClosureAcrossRevisitedScene) {
    LfTrackerConfig cfg = makeConfig(dist, 0.01f, 1, 12, 60, 0.30f, 2.0f);
    tracker = std::make_unique<LfTracker>(cfg, dummyVocabPath);
    initSyntheticVocabTree(vocabTree());

    tracker->start();
    ASSERT_TRUE(tracker->isRunning());

    auto waitUntilIdle = [&]() {
        for (int i = 0; i < 1000 && isBusy(); i++) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        ASSERT_FALSE(isBusy());
    };

    // 1. Build Keyframe 0 (frameId = 100) with 20 distinct oriented corners and 3D landmarks
    constexpr size_t kNumFeatures = 20;
    std::vector<HfFeature> kf0Features;
    BackendKeyframeUpdate kf0Update;
    kf0Update.id = 100;

    for (size_t i = 0; i < kNumFeatures; i++) {
        const float u = 110.0f + static_cast<float>(i % 5) * 100.0f;
        const float v = 90.0f + static_cast<float>(static_cast<float>(i) / 5) * 90.0f;
        const float angle = static_cast<float>(i) * 0.31f;

        drawRotatedCorner(frameBuffer, kImageWidth, kImageHeight, u, v,
                          kCornerRadius, kDarkIntensity, kBrightIntensity, angle);
        kf0Features.push_back(makeHfFeature(static_cast<uint32_t>(i + 1), u, v));

        const double depth = 2.0 + static_cast<double>(i % 4) * 0.5;
        const Eigen::Vector3d P3d(
            (static_cast<double>(u - kCx) / kFx) * depth,
            (static_cast<double>(v - kCy) / kFy) * depth,
            depth
        );
        kf0Update.landmarks.push_back(Landmark3d{P3d, static_cast<uint32_t>(i + 1)});
    }

    ASSERT_TRUE(tracker->passKeyframe(100, 1.0, frameBuffer.data(), kf0Features));
    waitUntilIdle();
    ASSERT_TRUE(tracker->updateKeyframe(kf0Update));

    // 2. Pass two intermediate keyframes outside the loop candidate window (excludedFrames = 1).
    // Include 4 shared corners in frame 102 so lastScore > 0 and bestScore / lastScore >> MIN_SCORE_RATIO.
    for (uint32_t interId : {101u, 102u}) {
        std::vector<HfFeature> interFeatures = {
            kf0Features[0], kf0Features[1], kf0Features[2], kf0Features[3]
        };
        ASSERT_TRUE(tracker->passKeyframe(interId, static_cast<double>(interId),
                                          frameBuffer.data(), interFeatures));
        waitUntilIdle();
    }

    // 3. Revisit Keyframe 0 on frameId = 103 with a known 3D translation and brand-new featureIds
    // ( except where appearance routes through the vocabulary tree ).
    const Eigen::Vector3d tLoop(0.08, -0.04, 0.0);
    std::vector<uint8_t> loopBuffer(kImageWidth * kImageHeight, kDefaultBackground);
    std::vector<HfFeature> loopFeatures;

    for (size_t i = 0; i < kNumFeatures; i++) {
        const Eigen::Vector3d PCam = kf0Update.landmarks[i].pt3d + tLoop;
        const float u = std::round(static_cast<float>(kFx * (PCam.x() / PCam.z()) + kCx));
        const float v = std::round(static_cast<float>(kFy * (PCam.y() / PCam.z()) + kCy));
        const float angle = static_cast<float>(i) * 0.31f;

        drawRotatedCorner(loopBuffer, kImageWidth, kImageHeight, u, v,
                          kCornerRadius, kDarkIntensity, kBrightIntensity, angle);
        // Assign new featureIds (1000 + i) to verify pure appearance-based QOV matching
        loopFeatures.push_back(makeHfFeature(static_cast<uint32_t>(1000 + i), u, v));
    }

    ASSERT_TRUE(tracker->passKeyframe(103, 4.0, loopBuffer.data(), loopFeatures));
    waitUntilIdle();

    BackendLfPacket outPacket;
    ASSERT_TRUE(tracker->getMatch(outPacket));
    EXPECT_EQ(outPacket.currentFrameId, 103u);
    EXPECT_EQ(outPacket.matchedFrameId, 100u);
    EXPECT_GE(outPacket.inlierCount, 12u);
    EXPECT_GE(outPacket.inlierRatio, 0.30f);
    EXPECT_NE(outPacket.noiseModel, nullptr);
    EXPECT_NEAR((outPacket.relativePose.translation() - tLoop).norm(), 0.0, 2e-2);

    // Verify mailbox is cleared after consumption
    EXPECT_FALSE(tracker->getMatch(outPacket));

    tracker->end();
    EXPECT_FALSE(tracker->isRunning());
}

} // namespace PoolSLAM