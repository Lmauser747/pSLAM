#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>

#include <eigen3/Eigen/Dense>

#include <CoreTypes.hpp>

#if defined(__GNUC__) || defined(__clang__)
  #define FORCE_INLINE inline __attribute__((always_inline))
#else
  #define FORCE_INLINE inline
#endif

namespace PoolSLAM{

struct GridBucket{
    uint8_t count = 0;
    int olderIdx = -1;
    int youngerIdx = -1; 
};

// Describes the data used for Epipolar geometry
struct EpipolarData{
    // Structure of arrays for vectorized normalization
    std::vector<float> xCurr, xPrev;
    std::vector<float> yCurr, yPrev;

    Eigen::Matrix3Xf currRays; // Rays from the current frame
    Eigen::Matrix3Xf prevRays; // Rays from the previous frame

    Eigen::Matrix3Xf scratch; // Scratch space to hold intermediate values

    Eigen::Vector3f residual; // Residual for inliear/outlier checking
};

class HfTracker{
private:
    static constexpr int RADIUS = 5; // Radius of feature patch
    static constexpr size_t MIN_MARGIN = 16; // Minimum bucket margin in pixels
    static constexpr size_t NUM_FEATURES_TARGET_DEFAULT = 200; // Target number of features
    static constexpr int ICLK_ITERS = 10; // Number of ICLK iterations
    static constexpr float ICLK_CONVERGENCE_THRESHOLD = 0.0009f; // ICLK convergence threashold
    static constexpr uint32_t ICLK_MAD_THRESHOLD = 20 * (TrackedFeature<RADIUS>::PATCH_SIZE) * (TrackedFeature<RADIUS>::PATCH_SIZE);
    static constexpr int UNDISTORT_ITERS = 5; // Number of times to undistort each point
    static constexpr int RANSAC_ITERS = 20; // Number of random samples to inspect in RANSAC
    static constexpr float RANSAC_TRANS_EP = 1e-6f; // Epislon to exclude translations which are too close to zero
    static constexpr float RANSAC_EARLY_CONSENSUS = 0.9f; // If consenus exceeds this threshold, exit RANSAC early
    static constexpr int RANSAC_MIN_INLIERS = 20; // Minimum inliers for RANSAC to exceed 

    const size_t wImage, hImage; // Image pixel array height and width

    size_t wBucket, hBucket; // Bucket width and height
    size_t bRows, bCols; // Rows and columns of bucket grid
    size_t xMargin, yMargin; // Bucket start point
    size_t totalBuckets; // xBuckets * yBuckets

    Eigen::Matrix3f K, KInv; 

    float ransacTau; // Inliear-outlier threshold for RANSAC

    Distortion dist; // Disortion coefficients

    EpipolarData epi; // Normalized rays to be used in RANSAC and Epipolar geometry calcs

    std::vector<TrackedFeature<RADIUS>> activeFeatures;
    std::vector<GridBucket> gridOccupancy; // Stores the age of the oldest frame in the grid

    std::minstd_rand ransacRng; // Random number generation for RANSAC

    // Handle errors in the tracking
    void handleTrackingError();

    // Gets the index of a bucket given sub-pixel coordinates
    FORCE_INLINE size_t getBucketIdx(double x, double y) const;

    // Gets the offset within the image array for a given bucket
    FORCE_INLINE size_t offsetFromBucket(size_t row, size_t column) const;

    // Matches the new feature to a feature in the previous frame using IC-LK
    FORCE_INLINE bool alignFeatureICLK(TrackedFeature<5> &feature, const uint8_t *curBuffer, Eigen::Vector2f &ptEst);

    // Biliner interpolation helper
    FORCE_INLINE int16_t bilinear(const uint8_t *curBuffer, Eigen::Vector2f &pt);

    // Does RANSAC on the active features using R as the IMU prior
    int ransac(const Eigen::Matrix3f &R);

    // Undistort into ptUndist
    void undistort(std::vector<float> &xDist, std::vector<float> &yDist, Eigen::Matrix3Xf &ptUndist, size_t count);

public:
    HfTracker(size_t w, size_t h, Eigen::Matrix3d &K_, Distortion &dist_, size_t nFeaturesTarget = NUM_FEATURES_TARGET_DEFAULT) : 
    wImage(w), hImage(h), K(K_), dist(dist_){
        const size_t wUsable = wImage - 2 * MIN_MARGIN;
        const size_t hUsable = hImage - 2 * MIN_MARGIN;
        const size_t targetArea = (hUsable * wUsable) / nFeaturesTarget;
        const size_t idealSideSize = static_cast<size_t>((std::sqrt(static_cast<float>(targetArea))));

        wBucket = std::max(static_cast<size_t>(16), ((idealSideSize + 8) / 16) / 16);
        bCols = wUsable / wBucket;
        bRows = std::max(1u, static_cast<uint32_t>(std::round(nFeaturesTarget / bCols)));
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
    };

    size_t trackAndExtract(const uint8_t *curBuffer, double timestamp, const std::optional<Eigen::Matrix3f> &R);

};

FORCE_INLINE size_t HfTracker::getBucketIdx(double x, double y) const{
    const size_t col = static_cast<size_t>((x - static_cast<double>(xMargin) / static_cast<double>(wBucket)));
    const size_t row = static_cast<size_t>((y - static_cast<double>(yMargin)) / static_cast<double>(hBucket));
    return row * bCols + col;
}

FORCE_INLINE size_t HfTracker::offsetFromBucket(size_t row, size_t col) const{
    return ((yMargin + row * hBucket) * wImage + (xMargin + col * wBucket));
}

}