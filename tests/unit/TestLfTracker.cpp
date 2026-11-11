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
#include <atomic>

#include <eigen3/Eigen/Dense>
#include <gtsam/geometry/Pose3.h>
#include <gtest/gtest.h>

#include "CoreTypes.hpp"
#include "frontend/LfTracker.hpp"
#include "descriptor/BinaryBow.hpp"
#include "descriptor/Qov.hpp"

namespace PoolSLAM {
namespace {

// Renders a smooth continuous 2D corner with bilinear ramp profile
void drawSmoothCorner(std::vector<uint8_t> &buffer, size_t width, size_t height,
                      float cx, float cy, int radius, uint8_t darkVal, uint8_t brightVal,
                      float rampWidth = 2.0f) {
    const float midVal = 0.5f * (static_cast<float>(darkVal) + static_cast<float>(brightVal));
    const float halfRange = 0.5f * (static_cast<float>(brightVal) - static_cast<float>(darkVal));

    const int minX = std::max(0, static_cast<int>(std::floor(cx)) - radius);
    const int maxX = std::min(static_cast<int>(width) - 1, static_cast<int>(std::ceil(cx)) + radius);
    const int minY = std::max(0, static_cast<int>(std::floor(cy)) - radius);
    const int maxY = std::min(static_cast<int>(height) - 1, static_cast<int>(std::ceil(cy)) + radius);

    for (int y = minY; y <= maxY; y++) {
        const float dy = (static_cast<float>(y) - cy) / rampWidth;
        const float wy = std::clamp(dy, -1.0f, 1.0f);
        for (int x = minX; x <= maxX; x++) {
            const float dx = (static_cast<float>(x) - cx) / rampWidth;
            const float wx = std::clamp(dx, -1.0f, 1.0f);
            const float intensity = midVal + halfRange * (wx * wy);
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
                               int ransacIters = 40,
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

    // Populates the vocabulary tree with deterministic nodes so matchWord and parent lookups
    // function without requiring an external vocabulary binary file on disk.
    static void initSyntheticVocabTree(descriptor::BinaryBowTree &tree) {
        tree.tree.resize(descriptor::BinaryBowTree::TOTAL_NODES);
        tree.vocabulary.resize(descriptor::BinaryBowTree::VOCAB_SIZE);

        for (size_t i = 0; i < tree.tree.size(); i++) {
            for (size_t word = 0; word < 4; word++) {
                uint64_t val = 0;
                for (size_t byte = 0; byte < 8; byte++) {
                    size_t b = word * 8 + byte;
                    uint64_t byteVal = static_cast<uint64_t>((i * 31 + b * 17) & 0xFF);
                    val |= (byteVal << (byte * 8));
                }
                tree.tree[i].bits[word] = val;
            }
        }
        for (size_t w = 0; w < tree.vocabulary.size(); w++) {
            tree.vocabulary[w].idf = 1.0f;
            tree.vocabulary[w].parent =
                static_cast<uint32_t>(w / descriptor::BinaryBowTree::NUM_BRANCHES);
        }
    }

    // Private state accessors
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

    // Helper: creates a QovDescriptor filled with a repeating byte value or controlled bit errors
    static descriptor::QovDescriptor makeDescriptor(uint8_t fillByte, size_t flipBits = 0) {
        descriptor::QovDescriptor des{};
        for (size_t i = 0; i < 32; i++) {
            des.bits[i / 8] |= static_cast<uint64_t>(fillByte) << (8 * (i % 8));
        }

        for (size_t bit = 0; bit < flipBits; bit++) {
            des.bits[bit / 64] ^= (1uL << (bit % 64));
        }
        return des;
    }
};

// ============================================================================
// 1. Lambda Twist P3P / P4P Minimal Solver Tests (solveP4PLambdaTwist)
// ============================================================================

// Verifies that solveP4PLambdaTwist accurately recovers a known 6-DoF rigid transformation
// [R | t] from 4 non-coplanar 3D points and their normalized 2D camera bearing rays.
TEST_F(LfTrackerTest, solveP4PLambdaTwistRecoversExact6DoFPose) {
    const Eigen::Matrix3d RTrue =
        (Eigen::AngleAxisd(0.15, Eigen::Vector3d::UnitY()) *
         Eigen::AngleAxisd(-0.08, Eigen::Vector3d::UnitX()) *
         Eigen::AngleAxisd(0.05, Eigen::Vector3d::UnitZ())).toRotationMatrix();
    const Eigen::Vector3d tTrue(0.35, -0.20, 0.45);

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

    const Eigen::Matrix3d RRec = TOpt->leftCols<3>();
    const Eigen::Vector3d tRec = TOpt->rightCols<1>();

    EXPECT_NEAR((RRec - RTrue).norm(), 0.0, kPoseEps);
    EXPECT_NEAR((tRec - tTrue).norm(), 0.0, kPoseEps);
}

// Verifies that solveP4PLambdaTwist returns std::nullopt when the first 3 3D points are collinear.
TEST_F(LfTrackerTest, solveP4PLambdaTwistRejectsCollinearPoints) {
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
}

// ============================================================================
// 2. Coordinate Undistortion Tests (undistort)
// ============================================================================

// Verifies that undistort maps pixel coordinates to normalized homogeneous camera rays
// under both zero distortion and non-zero radial/tangential Brown-Conrady distortion.
TEST_F(LfTrackerTest, undistortInvertsPinholeAndBrownConradyDistortion) {
    Distortion nonZeroDist{};
    nonZeroDist.k1 = -0.15f;
    nonZeroDist.k2 = 0.025f;
    nonZeroDist.p1 = 0.0015f;
    nonZeroDist.p2 = -0.0012f;

    LfTrackerConfig cfg = makeConfig(nonZeroDist);
    tracker = std::make_unique<LfTracker>(cfg, dummyVocabPath);

    const std::vector<Eigen::Vector2f> trueNormRays = {
        {0.0f, 0.0f},
        {0.22f, -0.16f},
        {-0.28f, 0.19f},
        {0.15f, 0.25f}
    };

    Eigen::Matrix3Xd pts(3, trueNormRays.size());
    for (size_t i = 0; i < trueNormRays.size(); i++) {
        const Eigen::Vector2f dNorm =
            applyForwardDistortion(trueNormRays[i].x(), trueNormRays[i].y(), nonZeroDist);
        pts(0, static_cast<Eigen::Index>(i)) = static_cast<double>(kFx * dNorm.x() + kCx);
        pts(1, static_cast<Eigen::Index>(i)) = static_cast<double>(kFy * dNorm.y() + kCy);
        pts(2, static_cast<Eigen::Index>(i)) = 1.0;
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

// Verifies that gaussianBlur preserves flat constant-intensity images without uint8_t overflow
// or boundary corruption across the entire image including the last two rows and columns.
TEST_F(LfTrackerTest, gaussianBlurPreservesUniformIntensityAcrossAllBorders) {
    for (uint8_t val : {uint8_t{0}, uint8_t{128}, uint8_t{255}}) {
        std::fill_n(localBuffer(), kImageWidth * kImageHeight, val);
        callGaussianBlur();

        for (size_t i = 0; i < kImageWidth * kImageHeight; i++) {
            ASSERT_EQ(localBuffer()[i], val) << "Mismatch at pixel index " << i;
        }
    }
}

// Verifies that gaussianBlur applies the separable [1, 4, 6, 4, 1] / 16 kernel symmetrically
// to an isolated impulse pixel in the interior of the image.
TEST_F(LfTrackerTest, gaussianBlurSmoothsImpulseWithSymmetricKernelWeights) {
    std::fill_n(localBuffer(), kImageWidth * kImageHeight, 0);
    constexpr size_t kCenterX = 320;
    constexpr size_t kCenterY = 240;
    localBuffer()[kCenterY * kImageWidth + kCenterX] = 255;

    callGaussianBlur();

    const uint8_t centerVal = localBuffer()[kCenterY * kImageWidth + kCenterX];
    const uint8_t adjLeft = localBuffer()[kCenterY * kImageWidth + (kCenterX - 1)];
    const uint8_t adjRight = localBuffer()[kCenterY * kImageWidth + (kCenterX + 1)];
    const uint8_t adjTop = localBuffer()[(kCenterY - 1) * kImageWidth + kCenterX];
    const uint8_t adjBot = localBuffer()[(kCenterY + 1) * kImageWidth + kCenterX];

    EXPECT_GT(centerVal, adjLeft);
    EXPECT_EQ(adjLeft, adjRight);
    EXPECT_EQ(adjTop, adjBot);
    EXPECT_EQ(adjLeft, adjTop);
    EXPECT_GT(adjLeft, 0u);
}

// ============================================================================
// 4. Steered QOV Descriptor Extraction Tests (calculateDescriptors)
// ============================================================================

// Verifies that calculateDescriptors computes patch moments and extracts non-zero 256-bit
// binary descriptors for keypoints in the current keyframe.
TEST_F(LfTrackerTest, calculateDescriptorsExtractsValidBinaryDescriptors) {
    drawSmoothCorner(frameBuffer, kImageWidth, kImageHeight, 200.0f, 180.0f,
                     kCornerRadius, kDarkIntensity, kBrightIntensity);
    std::copy(frameBuffer.begin(), frameBuffer.end(), localBuffer());

    Keyframe kf(1, 0, 1.0, 0);
    kf.addKp(Eigen::Vector2i(200, 180), 42);
    kf.finalSize();
    keyframes().push_back(std::move(kf));

    callCalculateDescriptors();

    const auto &des = keyframes().back().features.front().des;
    bool hasNonZeroBits = false;
    bool hasNonOneBits = false;
    for (size_t i = 0; i < 4; i++) {
        if (des.bits[i] != 0x00) hasNonZeroBits = true;
        if (des.bits[i] != ~0uL) hasNonOneBits = true;
    }
    EXPECT_TRUE(hasNonZeroBits);
    EXPECT_TRUE(hasNonOneBits);
}

// ============================================================================
// 5. Vocabulary-Guided Feature Matching Tests (matchFeatures)
// ============================================================================

// Verifies that matchFeatures pairs descriptors within the same vocabulary parent node,
// enforces Lowe's ratio test and bidirectional consistency, skips invalid 3D landmarks (Z <= 0),
// and populates matching.curFeatures and matching.matchLandmarks.
TEST_F(LfTrackerTest, matchFeaturesEnforcesParentNodeLoweRatioAndValid3DDepth) {
    Keyframe matchKf(4, 10, 1.0, 0);
    Keyframe curKf(3, 20, 2.0, 1);

    // Feature 0: Valid appearance match in word 10 (parent 1), distinct from Feature 1
    descriptor::QovFeature m0(Eigen::Vector2i(100, 100), 100);
    m0.wordId = 10;
    m0.des = makeDescriptor(0x00, 0);

    // Feature 1: Competing descriptor in the same parent (word 11, parent 1), far Hamming distance (128 bits)
    descriptor::QovFeature m1(Eigen::Vector2i(150, 100), 101);
    m1.wordId = 11;
    m1.des = makeDescriptor(0x00, 128);

    // Feature 2: Ambiguous descriptor pair in word 20 & 21 (parent 2) that should fail Lowe's ratio test
    descriptor::QovFeature m2a(Eigen::Vector2i(200, 200), 102);
    m2a.wordId = 20;
    m2a.des = makeDescriptor(0xAA, 10);

    descriptor::QovFeature m2b(Eigen::Vector2i(210, 200), 103);
    m2b.wordId = 21;
    m2b.des = makeDescriptor(0xAA, 11);

    // Feature 3: Feature with invalid depth (Z <= 0) in word 30 (parent 3), must be skipped
    descriptor::QovFeature m3(Eigen::Vector2i(300, 300), 104);
    m3.wordId = 30;
    m3.des = makeDescriptor(0x55, 0);

    matchKf.features = {m0, m1, m2a, m2b, m3};
    matchKf.landmarks = {
        Landmark3d{Eigen::Vector3d(0.5, -0.2, 2.5), 100},
        Landmark3d{Eigen::Vector3d(0.8, -0.2, 3.0), 101},
        Landmark3d{Eigen::Vector3d(0.1,  0.1, 2.0), 102},
        Landmark3d{Eigen::Vector3d(0.2,  0.1, 2.0), 103},
        Landmark3d{Eigen::Vector3d(0.0,  0.0, 0.0), 104} // Uninitialized / Z == 0
    };

    // Current frame features:
    // c0 matches m0 cleanly (dist = 2 vs secondDist = 126 -> ratio = 2/126 < 0.8)
    descriptor::QovFeature c0(Eigen::Vector2i(105, 102), 500);
    c0.wordId = 10;
    c0.des = makeDescriptor(0x00, 2);

    // c1 is ambiguous between m2a (dist = 10) and m2b (dist = 11) -> ratio = 10/11 > 0.8 -> rejected
    descriptor::QovFeature c1(Eigen::Vector2i(205, 202), 501);
    c1.wordId = 20;
    c1.des = makeDescriptor(0xAA, 0);

    // c2 matches m3 in appearance, but m3 has Z <= 0 -> rejected
    descriptor::QovFeature c2(Eigen::Vector2i(305, 302), 502);
    c2.wordId = 30;
    c2.des = makeDescriptor(0x55, 0);

    curKf.features = {c0, c1, c2};

    keyframes().push_back(std::move(matchKf));
    keyframes().push_back(std::move(curKf));

    const size_t numMatched = callMatchFeatures(1, 0);
    ASSERT_EQ(numMatched, 1u);

    EXPECT_DOUBLE_EQ(matching().curFeatures(0, 0), 105.0);
    EXPECT_DOUBLE_EQ(matching().curFeatures(1, 0), 102.0);
    EXPECT_DOUBLE_EQ(matching().curFeatures(2, 0), 1.0);
    EXPECT_NEAR((matching().matchLandmarks.col(0) - Eigen::Vector3d(0.5, -0.2, 2.5)).norm(), 0.0, 1e-9);
}

// ============================================================================
// 6. P3P RANSAC Outlier Pruning & Compaction Tests (ransac)
// ============================================================================

// Verifies that ransac estimates the 6-DoF camera transformation across 3D-2D correspondences,
// rejects pixel reprojection outliers, and compacts matching.curFeatures and matching.matchLandmarks.
TEST_F(LfTrackerTest, ransacEstimatesPosePrunesOutliersAndCompactsMatches) {
    constexpr size_t kNumInliers = 24;
    constexpr size_t kNumOutliers = 6;
    constexpr size_t kTotalMatches = kNumInliers + kNumOutliers;

    const Eigen::Matrix3d RTrue =
        Eigen::AngleAxisd(0.08, Eigen::Vector3d(0.3, 0.9, -0.2).normalized()).toRotationMatrix();
    const Eigen::Vector3d tTrue(-0.15, 0.10, 0.25);

    matching().curFeatures.resize(3, kTotalMatches);
    matching().matchLandmarks.resize(3, kTotalMatches);

    for (size_t i = 0; i < kTotalMatches; i++) {
        const double xn = -0.35 + static_cast<double>(i % 6) * 0.14;
        const double yn = -0.25 + static_cast<double>(static_cast<double>(i) / 6) * 0.12;
        const double depth = 2.0 + static_cast<double>(i % 4) * 0.6;

        const Eigen::Vector3d P3d(xn * depth, yn * depth, depth);
        const Eigen::Vector3d PCam = RTrue * P3d + tTrue;

        double u = static_cast<double>(kFx) * (PCam.x() / PCam.z()) + static_cast<double>(kCx);
        double v = static_cast<double>(kFy) * (PCam.y() / PCam.z()) + static_cast<double>(kCy);

        if (i >= kNumInliers) {
            u += (i % 2 == 0) ? 45.0 : -45.0;
            v += (i % 2 == 0) ? -35.0 : 35.0;
        }

        matching().matchLandmarks.col(static_cast<Eigen::Index>(i)) = P3d;
        matching().curFeatures.col(static_cast<Eigen::Index>(i)) = Eigen::Vector3d(u, v, 1.0);
    }

    const auto res = callRansac(kTotalMatches);
    ASSERT_TRUE(res.has_value());

    const auto &[TBest, inlierCount] = *res;
    EXPECT_EQ(inlierCount, kNumInliers);
    EXPECT_NEAR((TBest.leftCols<3>() - RTrue).norm(), 0.0, 1e-3);
    EXPECT_NEAR((TBest.rightCols<1>() - tTrue).norm(), 0.0, 1e-3);

    // Verify in-place compaction retained only the true inliers
    for (size_t i = 0; i < inlierCount; i++) {
        const Eigen::Vector3d P = matching().matchLandmarks.col(static_cast<Eigen::Index>(i));
        const Eigen::Vector2d p = matching().curFeatures.col(static_cast<Eigen::Index>(i)).hnormalized();
        const Eigen::Vector2d proj = (RTrue * P + tTrue).hnormalized();
        const Eigen::Vector2d pxErr(kFx * (proj.x() - p.x()), kFy * (proj.y() - p.y()));
        EXPECT_LE(pxErr.norm(), 2.0);
    }
}

// Verifies that ransac returns std::nullopt when inliers fall below config.ransacInliearThreshold.
TEST_F(LfTrackerTest, ransacReturnsNulloptWhenInliersBelowThreshold) {
    constexpr size_t kInsufficientMatches = 8; // < ransacInliearThreshold (12)
    matching().curFeatures.resize(3, kInsufficientMatches);
    matching().matchLandmarks.resize(3, kInsufficientMatches);

    for (size_t i = 0; i < kInsufficientMatches; i++) {
        const Eigen::Vector3d P3d(static_cast<double>(i) * 0.1, 0.2, 2.5);
        const double u = kFx * (P3d.x() / P3d.z()) + kCx;
        const double v = kFy * (P3d.y() / P3d.z()) + kCy;
        matching().matchLandmarks.col(static_cast<Eigen::Index>(i)) = P3d;
        matching().curFeatures.col(static_cast<Eigen::Index>(i)) = Eigen::Vector3d(u, v, 1.0);
    }

    EXPECT_FALSE(callRansac(kInsufficientMatches).has_value());
}

// ============================================================================
// 7. Levenberg-Marquardt Pose & Covariance Refinement Tests (refinePoseLM)
// ============================================================================

// Verifies that refinePoseLM converges from a perturbed pose initialization back to the
// ground-truth [R | t] and outputs a positive-definite 6x6 covariance matrix.
TEST_F(LfTrackerTest, refinePoseLMConvergesFromPerturbedInitAndComputesCovariance) {
    constexpr size_t kNumInliers = 25;
    const Eigen::Matrix3d RTrue =
        (Eigen::AngleAxisd(0.10, Eigen::Vector3d::UnitY()) *
         Eigen::AngleAxisd(-0.05, Eigen::Vector3d::UnitX())).toRotationMatrix();
    const Eigen::Vector3d tTrue(0.20, -0.12, 0.30);

    matching().curFeatures.resize(3, kNumInliers);
    matching().matchLandmarks.resize(3, kNumInliers);

    for (size_t i = 0; i < kNumInliers; i++) {
        const double xn = -0.30 + static_cast<double>(i % 5) * 0.15;
        const double yn = -0.24 + static_cast<double>(static_cast<double>(i) / 5) * 0.12;
        const double depth = 2.0 + static_cast<double>(i % 3) * 0.7;

        const Eigen::Vector3d P3d(xn * depth, yn * depth, depth);
        const Eigen::Vector3d PCam = RTrue * P3d + tTrue;

        matching().matchLandmarks.col(static_cast<Eigen::Index>(i)) = P3d;
        matching().curFeatures.col(static_cast<Eigen::Index>(i)) = PCam.hnormalized().homogeneous();
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

    for (int i = 0; i < 6; i++) {
        EXPECT_GT(outCov(i, i), 0.0);
        EXPECT_TRUE(std::isfinite(outCov(i, i)));
    }
}

// ============================================================================
// 8. Public Interface & Asynchronous Worker Lifecycle Tests
// ============================================================================

// Verifies that passKeyframe filters out keypoints within QOV_RADIUS of the image border,
// stages valid keypoints, sets busy = true, and rejects concurrent calls while busy.
TEST_F(LfTrackerTest, passKeyframeFiltersBorderKeypointsAndEnforcesBusyGuard) {
    std::vector<HfFeature> features = {
        makeHfFeature(1, 5.0f, 100.0f),                                   // Too close to left border (< 15)
        makeHfFeature(2, 100.0f, static_cast<float>(kImageHeight - 10)),  // Too close to bottom border
        makeHfFeature(3, 200.0f, 200.0f),                                 // Valid interior keypoint
        makeHfFeature(4, 350.0f, 250.0f)                                  // Valid interior keypoint
    };

    ASSERT_TRUE(tracker->passKeyframe(10, 1.25, frameBuffer.data(), features));
    EXPECT_TRUE(isBusy());
    ASSERT_EQ(keyframes().size(), 1u);
    EXPECT_EQ(keyframes().front().features.size(), 2u);
    EXPECT_EQ(keyframes().front().features[0].featureId, 3u);
    EXPECT_EQ(keyframes().front().features[1].featureId, 4u);

    // Calling passKeyframe again while busy must return false without adding a frame
    EXPECT_FALSE(tracker->passKeyframe(11, 1.50, frameBuffer.data(), features));
    EXPECT_EQ(keyframes().size(), 1u);
}

// Verifies that updateKeyframe matches incoming 3D landmarks to a keyframe's features by featureId
// even after the keyframe's features vector has been reordered by wordId.
TEST_F(LfTrackerTest, updateKeyframeAlignsLandmarksWithReorderedFeaturesById) {
    Keyframe kf(3, 77, 2.0, 0);
    kf.addKp(Eigen::Vector2i(100, 100), 10);
    kf.addKp(Eigen::Vector2i(200, 100), 20);
    kf.addKp(Eigen::Vector2i(300, 100), 30);
    kf.finalSize();

    // Simulate post-BoW sorting where feature order is reversed
    std::reverse(kf.features.begin(), kf.features.end());
    keyframes().push_back(std::move(kf));

    BackendKeyframeUpdate update;
    update.id = 77;
    update.pose = gtsam::Pose3();
    update.landmarks = {
        Landmark3d{Eigen::Vector3d(1.0, 0.0, 2.0), 10},
        Landmark3d{Eigen::Vector3d(2.0, 0.0, 3.0), 20},
        Landmark3d{Eigen::Vector3d(3.0, 0.0, 4.0), 30}
    };

    ASSERT_TRUE(tracker->updateKeyframe(update));
    const auto &storedKf = keyframes().front();
    ASSERT_EQ(storedKf.landmarks.size(), storedKf.features.size());

    for (size_t i = 0; i < storedKf.features.size(); i++) {
        EXPECT_EQ(storedKf.landmarks[i].featureId, storedKf.features[i].featureId);
    }

    // Non-existent keyframe ID must return false
    BackendKeyframeUpdate badUpdate;
    badUpdate.id = 999;
    EXPECT_FALSE(tracker->updateKeyframe(badUpdate));
}

// Verifies the complete asynchronous worker thread lifecycle (start, passKeyframe, updateKeyframe,
// getMatch, end) across a historical keyframe, exclusion window frames, and a revisited loop frame.
TEST_F(LfTrackerTest, endToEndWorkerThreadProducesLoopClosurePacket) {
    LfTrackerConfig cfg = makeConfig(dist, 0.01f, 1, 12, 50, 0.30f, 2.0f);
    tracker = std::make_unique<LfTracker>(cfg, dummyVocabPath);
    initSyntheticVocabTree(vocabTree());

    tracker->start();
    ASSERT_TRUE(tracker->isRunning());

    auto waitUntilIdle = [&]() {
        for (int i = 0; i < 500 && isBusy(); i++) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        ASSERT_FALSE(isBusy());
    };

    // 1. Create 20 landmarks and render Keyframe 0 (historical frame)
    constexpr size_t kNumFeatures = 20;
    std::vector<HfFeature> kf0Features;
    BackendKeyframeUpdate kf0Update;
    kf0Update.id = 100;

    for (size_t i = 0; i < kNumFeatures; i++) {
        const float u = 120.0f + static_cast<float>(i % 5) * 90.0f;
        const float v = 100.0f + static_cast<float>(static_cast<double>(i) / 5) * 80.0f;
        drawSmoothCorner(frameBuffer, kImageWidth, kImageHeight, u, v,
                         kCornerRadius, kDarkIntensity, kBrightIntensity);
        kf0Features.push_back(makeHfFeature(static_cast<uint32_t>(i + 1), u, v));

        const double depth = 2.0 + static_cast<double>(i % 3) * 0.5;
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

    // 2. Pass two intermediate keyframes to clear the temporal exclusion window (excludedFrames = 1)
    for (uint32_t interId : {101u, 102u}) {
        std::vector<HfFeature> interFeatures = {makeHfFeature(900 + interId, 320.0f, 240.0f)};
        ASSERT_TRUE(tracker->passKeyframe(interId, static_cast<double>(interId),
                                          frameBuffer.data(), interFeatures));
        waitUntilIdle();
    }

    // 3. Pass Keyframe 3 revisiting Keyframe 0 with a known relative translation
    const Eigen::Vector3d tLoop(0.10, -0.05, 0.0);
    std::vector<uint8_t> loopBuffer(kImageWidth * kImageHeight, kDefaultBackground);
    std::vector<HfFeature> loopFeatures;

    for (size_t i = 0; i < kNumFeatures; i++) {
        const Eigen::Vector3d PCam = kf0Update.landmarks[i].pt3d + tLoop;
        const float u = static_cast<float>(kFx * (PCam.x() / PCam.z()) + kCx);
        const float v = static_cast<float>(kFy * (PCam.y() / PCam.z()) + kCy);
        drawSmoothCorner(loopBuffer, kImageWidth, kImageHeight, u, v,
                         kCornerRadius, kDarkIntensity, kBrightIntensity);
        // Use matching featureIds to guarantee deterministic correspondence in the synthetic scene
        loopFeatures.push_back(makeHfFeature(static_cast<uint32_t>(i + 1), u, v));
    }

    ASSERT_TRUE(tracker->passKeyframe(103, 4.0, loopBuffer.data(), loopFeatures));
    waitUntilIdle();

    BackendLfPacket outPacket;
    ASSERT_TRUE(tracker->getMatch(outPacket));
    EXPECT_EQ(outPacket.currentFrameId, 103u);
    EXPECT_EQ(outPacket.matchedFrameId, 100u);
    EXPECT_EQ(outPacket.inlierCount, static_cast<uint32_t>(kNumFeatures));
    EXPECT_NEAR(outPacket.inlierRatio, 1.0f, 1e-4f);
    EXPECT_NE(outPacket.noiseModel, nullptr);
    EXPECT_NEAR((outPacket.relativePose.translation() - tLoop).norm(), 0.0, 2e-2);

    // Subsequent call to getMatch must return false once the mailbox is drained
    EXPECT_FALSE(tracker->getMatch(outPacket));

    tracker->end();
    EXPECT_FALSE(tracker->isRunning());
}

} // namespace PoolSLAM