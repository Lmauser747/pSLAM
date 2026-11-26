#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <array>

#include <eigen3/Eigen/Dense>

#include "CoreTypes.hpp"


// idgaf about MSVC
#ifndef FORCE_INLINE
#if defined(__GNUC__) || defined(__clang__)
  #define FORCE_INLINE inline __attribute__((always_inline))
#else
  #define FORCE_INLINE inline
#endif
#endif

namespace PoolSLAM{

struct GridBucket{
    uint8_t count = 0;
    uint32_t olderIdx = UINT32_MAX;
    uint32_t youngerIdx = UINT32_MAX; 
};

// Describes the data used for Epipolar geometry
struct EpipolarData{
    // Structure of arrays for vectorized normalization
    std::vector<float> xCurr, xPrev;
    std::vector<float> yCurr, yPrev;

    Eigen::Matrix3Xf currRays; // Rays from the current frame
    Eigen::Matrix3Xf prevRays; // Rays from the previous frame

    Eigen::Matrix3Xf scratch; // Scratch space to hold intermediate values
};


class HfTracker{
private:
    static constexpr int RADIUS = 5; // Radius of feature patch
    static constexpr size_t PYR_LEVELS = 4; // Number of pyramid levels
    static constexpr size_t MIN_MARGIN = 16; // Minimum bucket margin in pixels
    static constexpr size_t NUM_FEATURES_TARGET_DEFAULT = 200; // Target number of features
    static constexpr int ICLK_ITERS = 20; // Number of ICLK iterations
    static constexpr int ICLK_COARSE_ITERS = 10; // Coarse pyramid level iterations
    static constexpr float COARSE_MAX_SHIFT = 2.5f * static_cast<float>(RADIUS); // Guard against divergent coarse steps
    static constexpr uint32_t ICLK_MAD_THRESHOLD = 5 * (TrackedFeature<RADIUS>::PATCH_SIZE) * (TrackedFeature<RADIUS>::PATCH_SIZE);
    static constexpr int UNDISTORT_ITERS = 10; // Number of times to undistort each point
    static constexpr int RANSAC_ITERS = 20; // Number of random samples to inspect in RANSAC
    static constexpr float RANSAC_TRANS_EP = 1e-6f; // Epislon to exclude translations which are too close to zero
    static constexpr float RANSAC_EARLY_CONSENSUS = 0.9f; // If consenus exceeds this threshold, exit RANSAC early
    static constexpr int RANSAC_MIN_INLIERS = 20; // Minimum inliers for RANSAC to succeed 
    static constexpr float MIN_DIST_IN_BUCKET = 6.0; // Minimum squared distance between two points in a bucket
    static constexpr uint8_t FAST_THRESHOLD = 20; // Threshold for FAST keypoint detection
    static constexpr float ST_THRESHOLD = 500.0f; // Minimum Shi-Tomasi score of canidate feature
    static constexpr int ST_PATCH_SIZE = 7; // Width and height of patch for Shi-Tomasi gradient computation
    static constexpr float HESSIAN_STABILITY_THRESHOLD = 409.6f; // Minimum absolute value of the determinant of the Hessian (scaled by 64^2)

    struct ImagePyramid{
        std::array<std::vector<uint8_t>, PYR_LEVELS> levels;
        std::array<size_t, PYR_LEVELS> widths{};
        std::array<size_t, PYR_LEVELS> heights{};
    };

    const size_t wImage, hImage; // Image pixel array height and width

    size_t wBucket, hBucket; // Bucket width and height
    size_t bRows, bCols; // Rows and columns of bucket grid
    size_t xMargin, yMargin; // Bucket start point
    size_t totalBuckets; // xBuckets * yBuckets

    Eigen::Matrix3f K, KInv; 

    float ransacTau; // Inliear-outlier threshold for RANSAC

    Distortion dist; // Disortion coefficients

    EpipolarData epi; // Normalized rays to be used in RANSAC and Epipolar geometry calcs

    std::vector<TrackedFeature<RADIUS>> activeFeatures; // Stores all active features
    std::vector<GridBucket> gridOccupancy; // Stores the age of the oldest frame in the grid
    std::vector<uint8_t> keep; // Stores all the features to keep during bucket size enforcing
    // Note: Using uint8_t for keep is better than bool. std::vector<bool> requires bitmasks for eac access
    // which can be slow

    std::vector<Eigen::Vector2i> fastPoints; // Scratch buffer to store all the fast points inside a given bucket
    const ptrdiff_t w_;
    const std::array<ptrdiff_t, 16> fastOffsets; // Offsets for FAST circle

    uint32_t nextId = 0; // id to assign to next keypoint

    std::minstd_rand ransacRng; // Random number generation for RANSAC

    std::array<ImagePyramid, 2> pyramids;
    std::vector<uint16_t> pyrBlurScratch;
    uint8_t curPyrIdx = 0;
    bool hasPrevPyramid = false;

    FORCE_INLINE uint8_t prevPyrIdx() const{return curPyrIdx ^ 1u;}

    // Handle errors in the tracking
    void handleTrackingError(){
        activeFeatures.clear();
        std::fill(gridOccupancy.begin(), gridOccupancy.end(), GridBucket{0, UINT32_MAX, UINT32_MAX});
        hasPrevPyramid = false;
    }

    // Gets the index of a bucket given sub-pixel coordinates
    FORCE_INLINE size_t getBucketIdx(double x, double y) const;

    // Gets the offset within the image array for a given bucket
    FORCE_INLINE size_t offsetFromBucket(size_t bucket) const;

    // Gets the point at the top-left of the bucket
    FORCE_INLINE Eigen::Vector2i topLeftBucket(size_t bucket) const;

    // Build all Gaussian pyramid levels into dst
    void buildPyramid(const uint8_t *curBuffer, ImagePyramid &dst);
    void downsampleLevel(const uint8_t *src, size_t srcW, size_t srcH, uint8_t *dst, size_t dstW, size_t dstH);

    // Extract patch, Sobel gradients, and inverse Hessian at subpixel pt
    bool extractTemplate(
        const uint8_t *buffer,
        size_t bufW,
        size_t bufH,
        const Eigen::Vector2f &pt,
        std::array<uint8_t, TrackedFeature<RADIUS>::NUM_PIXELS> &patch,
        std::array<int16_t, TrackedFeature<RADIUS>::NUM_PIXELS> &xGrad,
        std::array<int16_t, TrackedFeature<RADIUS>::NUM_PIXELS> &yGrad,
        Eigen::Matrix2f &HInv) const;

    // Core IC-LK alignment loop on a buffer of arbitrary size
    bool alignPatchICLK(
        const std::array<uint8_t, TrackedFeature<RADIUS>::NUM_PIXELS> &patch,
        const std::array<int16_t, TrackedFeature<RADIUS>::NUM_PIXELS> &xGrad,
        const std::array<int16_t, TrackedFeature<RADIUS>::NUM_PIXELS> &yGrad,
        const Eigen::Matrix2f &HInv,
        const uint8_t *curBuffer,
        size_t bufW,
        size_t bufH,
        int maxIters,
        Eigen::Vector2f &ptEst,
        float &madError) const;

    // Matches the new feature to a feature in the previous frame using IC-LK
    bool alignFeatureICLK(TrackedFeature<RADIUS> &feature, const uint8_t *curBuffer, Eigen::Vector2f &ptEst);

    // Coarse to fine pyramidal IC-LK alignment followed by level zero refinement
    bool alignFeaturePyramidICLK(TrackedFeature<RADIUS> &feature, const uint8_t *curBuffer, Eigen::Vector2f &ptEst);

    // Biliner interpolation helper
    int16_t bilinear(const uint8_t *curBuffer, const Eigen::Vector2f &pt) const;
    float bilinearFloat(const uint8_t *curBuffer, const Eigen::Vector2f &pt) const;
    float bilinearFloatStride(const uint8_t *curBuffer, size_t stride, const Eigen::Vector2f &pt) const;

    // Does RANSAC on the active features using R as the IMU prior
    int ransac(const Eigen::Matrix3f &R);

    // Undistort a single pixel into normalized camera coordinates
    Eigen::Vector2f undistortPoint(float xDist, float yDist) const;

    // Undistort into ptUndist
    void undistort(std::vector<float> &xDist, std::vector<float> &yDist, Eigen::Matrix3Xf &ptUndist, size_t count);

    // Remove excess features from buckets
    void enforceBucketDensity();

    // Find all the FAST keypoints in a given bucket and store them in fastPoints
    size_t getFast(const uint8_t *curBuffer, size_t bucket);

    // Get the best keypoint out of the FAST points using Shi-Tomasi
    std::optional<Eigen::Vector2i> bestKp(const uint8_t *curBuffer);

    // Calculates gradient and Hessian and then adds to activeFeatures array. If Hessian is degenerate, do not add feature
    bool addKp(const uint8_t *curBuffer, Eigen::Vector2i &kp);

    // Make the unit test class a friend for testing
    friend class TestTrackHfFeatures;
    friend class TestReplenishHfFeatures;

public:
    HfTracker(size_t w, size_t h, const Eigen::Matrix3f &K_, const Distortion &dist_, size_t nFeaturesTarget = NUM_FEATURES_TARGET_DEFAULT) : 
    wImage(w), hImage(h), K(K_), dist(dist_), w_(static_cast<ptrdiff_t>(wImage)), fastOffsets{
            -3 + 0 * w_, -3 + 1 * w_, -2 + 2 * w_, -1 + 3 * w_, 
            0 + 3 * w_, 1 + 3 * w_, 2 + 2 * w_, 3+ 1 * w_,
            3 + 0 * w_, 3 + (-1) * w_, 2 + (-2) * w_, 1 + (-3) * w_,
            0 + -3 * w_, -1 + (-3) * w_, -2 + (-2) * w_, -3 + (-1) * w_
        }{
        const size_t wUsable = wImage - 2 * MIN_MARGIN;
        const size_t hUsable = hImage - 2 * MIN_MARGIN;
        const size_t targetArea = (hUsable * wUsable) / nFeaturesTarget;
        const size_t idealSideSize = static_cast<size_t>((std::sqrt(static_cast<float>(targetArea))));

        wBucket = std::max(static_cast<size_t>(16), ((idealSideSize + 8) / 16) * 16);
        bCols = wUsable / wBucket;
        bRows = std::max(1u, static_cast<uint32_t>(std::round(static_cast<float>(nFeaturesTarget) / static_cast<float>(bCols))));
        hBucket = hUsable / bRows;
        xMargin = (wImage - (bCols * wBucket)) / 2;
        yMargin = (hImage - (bRows * hBucket)) / 2;
        totalBuckets = bCols * bRows;

        gridOccupancy.resize(totalBuckets);
        activeFeatures.reserve(totalBuckets * 2);
        
        KInv = K.inverse();

        // Always seed the RNG with 0
        ransacRng.seed(0); 

        // 1.5 / (average focal length)
        ransacTau = 3.0f / (K(0, 0) + K(1, 1));

        size_t maxFeatures = totalBuckets * 2;

        epi.currRays.resize(3, static_cast<Eigen::Index>(maxFeatures));
        epi.prevRays.resize(3, static_cast<Eigen::Index>(maxFeatures));
        epi.scratch.resize(3, static_cast<Eigen::Index>(maxFeatures));

        epi.xCurr.reserve(maxFeatures);
        epi.yCurr.reserve(maxFeatures);
        epi.xPrev.reserve(maxFeatures);
        epi.yPrev.reserve(maxFeatures);

        fastPoints.reserve(wBucket);

        for(size_t lvl = 0; lvl < PYR_LEVELS; lvl++){
            const size_t lvlW = wImage >> lvl;
            const size_t lvlH = hImage >> lvl;
            for(auto &pyr : pyramids){
                pyr.widths[lvl] = lvlW;
                pyr.heights[lvl] = lvlH;
                pyr.levels[lvl].resize(lvlW * lvlH);
            }
        }
        pyrBlurScratch.resize((wImage / 2) * hImage);
    };

    // Tracks existing features with IC-LK, removes outliers with RANSAC, and retunrs the number of active features
    float trackExistingFeatures(const uint8_t *curBuffer, const Eigen::Matrix3f &R);

    // Replenishs drained buckets with new keypoints obtained via FAST + Shi-Thomasi
    size_t replenishBuckets(const uint8_t *curBuffer);

    // Computes median rotation-compensated parallax (in pixels) since the last keyframe
    float computeKeyframeParallax(const Eigen::Matrix3f &R_kf_curr) const;

    // Anchors active features' ptKfUndis to their current undistorted coordinates
    void markKeyframe();

    FORCE_INLINE const std::vector<TrackedFeature<RADIUS>> &getActiveFeatures() const{return activeFeatures;}

    FORCE_INLINE size_t getNumBuckets() const{return totalBuckets;}
};

FORCE_INLINE size_t HfTracker::getBucketIdx(double x, double y) const{
    if(std::isnan(x) || std::isnan(y)){
        return 0;
    }
    const double colD = (x - static_cast<double>(xMargin)) / static_cast<double>(wBucket);
    const double rowD = (y - static_cast<double>(yMargin)) / static_cast<double>(hBucket);
    const size_t col = (colD <= 0.0) ? 0 : ((colD >= static_cast<double>(bCols)) ? (bCols - 1) : static_cast<size_t>(colD));
    const size_t row = (rowD <= 0.0) ? 0 : ((rowD >= static_cast<double>(bRows)) ? (bRows - 1) : static_cast<size_t>(rowD));
    return row * bCols + col;
}

FORCE_INLINE size_t HfTracker::offsetFromBucket(size_t bucket) const{
    size_t col = bucket % bCols;
    size_t row = (bucket - col) / bCols;
    return ((yMargin + row * hBucket) * wImage + (xMargin + col * wBucket));
}

FORCE_INLINE Eigen::Vector2i HfTracker::topLeftBucket(size_t bucket) const{
    size_t col = bucket % bCols;
    size_t row = (bucket - col) / bCols;
    return {xMargin + col * wBucket, yMargin + row * hBucket};
}

}