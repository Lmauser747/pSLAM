#pragma once

#include <cstddef>
#include <vector>
#include <cstdint>
#include <optional>
#include <eigen3/Eigen/Dense>

#include <gtest/gtest.h>

#include "CoreTypes.hpp"
#include "frontend/HfTracker.hpp"

namespace PoolSLAM {

class TestTrackHfFeatures : public ::testing::Test {
protected:
    static constexpr size_t W = 640;
    static constexpr size_t H = 480;

    Eigen::Matrix3f K;
    Distortion dist{0.0f, 0.0f, 0.0f, 0.0f};

    void SetUp() override {
        K << 400.0f, 0.0f, 320.0f,
             0.0f, 400.0f, 240.0f,
             0.0f, 0.0f, 1.0f;
    }

    HfTracker createTracker(size_t nFeatures = 50) {
        return HfTracker(W, H, K, dist, nFeatures);
    }

    // Accessors for private members and methods
    static std::vector<TrackedFeature<5>>& getActiveFeatures(HfTracker& tracker) {
        return tracker.activeFeatures;
    }

    static const std::vector<TrackedFeature<5>>& getActiveFeatures(const HfTracker& tracker) {
        return tracker.activeFeatures;
    }

    static std::vector<GridBucket>& getGridOccupancy(HfTracker& tracker) {
        return tracker.gridOccupancy;
    }

    static bool callAlignFeatureICLK(HfTracker& tracker, TrackedFeature<5>& f, const uint8_t* curBuffer, Eigen::Vector2f& ptEst) {
        return tracker.alignFeatureICLK(f, curBuffer, ptEst);
    }

    static int16_t callBilinear(const HfTracker& tracker, const uint8_t* curBuffer, const Eigen::Vector2f& pt) {
        return tracker.bilinear(curBuffer, pt);
    }

    static int callRansac(HfTracker& tracker, const Eigen::Matrix3f& R) {
        return tracker.ransac(R);
    }

    static void callUndistort(HfTracker& tracker, std::vector<float>& xDist, std::vector<float>& yDist, Eigen::Matrix3Xf& ptUndist, size_t count) {
        tracker.undistort(xDist, yDist, ptUndist, count);
    }

    static void callEnforceBucketDensity(HfTracker& tracker) {
        tracker.enforceBucketDensity();
    }

    static void callHandleTrackingError(HfTracker& tracker) {
        tracker.handleTrackingError();
    }

    static size_t getBucketIdx(const HfTracker& tracker, double x, double y) {
        return tracker.getBucketIdx(x, y);
    }

    static bool callAddKp(HfTracker& tracker, const uint8_t* curBuffer, Eigen::Vector2i& kp) {
        return tracker.addKp(curBuffer, kp);
    }

    static Eigen::Vector2i getTopLeftBucket(const HfTracker& tracker, size_t bucket) {
        return tracker.topLeftBucket(bucket);
    }

    static bool getHasPrevPyramid(const HfTracker& tracker) {
        return tracker.hasPrevPyramid;
    }

    static uint8_t getCurPyrIdx(const HfTracker& tracker) {
        return tracker.curPyrIdx;
    }

    static size_t getPyrWidth(const HfTracker& tracker, size_t lvl) {
        return tracker.pyramids[tracker.prevPyrIdx()].widths[lvl];
    }

    static size_t getPyrHeight(const HfTracker& tracker, size_t lvl) {
        return tracker.pyramids[tracker.prevPyrIdx()].heights[lvl];
    }

    static const std::vector<uint8_t>& getPrevPyrLevel(const HfTracker& tracker, size_t lvl) {
        return tracker.pyramids[tracker.prevPyrIdx()].levels[lvl];
    }

    // Helper to generate a synthetic image with a textured circle / checker patch
    static std::vector<uint8_t> makeSyntheticImage(size_t w, size_t h, uint8_t bg = 50) {
        return std::vector<uint8_t>(w * h, bg);
    }

    // Draws a sharp high-contrast 90-degree corner at (cx, cy)
    static void drawCorner(std::vector<uint8_t>& img, size_t w, int cx, int cy, uint8_t bg = 50, uint8_t fg = 220) {
        for (int y = cy - 10; y <= cy + 10; ++y) {
            for (int x = cx - 10; x <= cx + 10; ++x) {
                if (x >= 0 && x < static_cast<int>(w) && y >= 0 && y < static_cast<int>(img.size() / w)) {
                    if (x >= cx && y >= cy) {
                        img[static_cast<size_t>(y) * w + static_cast<size_t>(x)] = fg;
                    } else {
                        img[static_cast<size_t>(y) * w + static_cast<size_t>(x)] = bg;
                    }
                }
            }
        }
    }

    // Draws a solid square patch in the image
    static void drawSquare(std::vector<uint8_t>& img, size_t w, int cx, int cy, int halfSize, uint8_t val = 220) {
        for (int y = cy - halfSize; y <= cy + halfSize; ++y) {
            for (int x = cx - halfSize; x <= cx + halfSize; ++x) {
                if (x >= 0 && x < static_cast<int>(w) && y >= 0 && y < static_cast<int>(img.size() / w)) {
                    img[static_cast<size_t>(y) * w + static_cast<size_t>(x)] = val;
                }
            }
        }
    }

    // Helper to construct a synthetic TrackedFeature
    static TrackedFeature<5> makeFeature(uint32_t id, const Eigen::Vector2f& pt, int age = 0) {
        TrackedFeature<5> f;
        f.id = id;
        f.age = age;
        f.ptCurr = pt;
        f.ptPrev = pt;
        f.ptUndis = Eigen::Vector2f(pt.x() / 400.0f, pt.y() / 400.0f);
        f.patch.fill(128);
        f.xGrad.fill(0);
        f.yGrad.fill(0);
        f.HInv = Eigen::Matrix2f::Identity();
        return f;
    }
};

class TestReplenishHfFeatures : public ::testing::Test {
protected:
    static constexpr size_t W = 640;
    static constexpr size_t H = 480;

    Eigen::Matrix3f K;
    Distortion dist{0.0f, 0.0f, 0.0f, 0.0f};

    void SetUp() override {
        K << 400.0f, 0.0f, 320.0f,
             0.0f, 400.0f, 240.0f,
             0.0f, 0.0f, 1.0f;
    }

    HfTracker createTracker(size_t nFeatures = 50) {
        return HfTracker(W, H, K, dist, nFeatures);
    }

    // Accessors for private members and methods
    static std::vector<TrackedFeature<5>>& getActiveFeatures(HfTracker& tracker) {
        return tracker.activeFeatures;
    }

    static const std::vector<TrackedFeature<5>>& getActiveFeatures(const HfTracker& tracker) {
        return tracker.activeFeatures;
    }

    static std::vector<GridBucket>& getGridOccupancy(HfTracker& tracker) {
        return tracker.gridOccupancy;
    }

    static std::vector<Eigen::Vector2i>& getFastPoints(HfTracker& tracker) {
        return tracker.fastPoints;
    }

    static size_t callGetFast(HfTracker& tracker, const uint8_t* curBuffer, size_t bucket) {
        return tracker.getFast(curBuffer, bucket);
    }

    static std::optional<Eigen::Vector2i> callBestKp(HfTracker& tracker, const uint8_t* curBuffer) {
        return tracker.bestKp(curBuffer);
    }

    static bool callAddKp(HfTracker& tracker, const uint8_t* curBuffer, Eigen::Vector2i& kp) {
        return tracker.addKp(curBuffer, kp);
    }

    static Eigen::Vector2i getTopLeftBucket(const HfTracker& tracker, size_t bucket) {
        return tracker.topLeftBucket(bucket);
    }

    static bool getHasPrevPyramid(const HfTracker& tracker) {
        return tracker.hasPrevPyramid;
    }

    static uint8_t getCurPyrIdx(const HfTracker& tracker) {
        return tracker.curPyrIdx;
    }

    static size_t getPyrWidth(const HfTracker& tracker, size_t lvl) {
        return tracker.pyramids[tracker.prevPyrIdx()].widths[lvl];
    }

    static size_t getPyrHeight(const HfTracker& tracker, size_t lvl) {
        return tracker.pyramids[tracker.prevPyrIdx()].heights[lvl];
    }

    static const std::vector<uint8_t>& getPrevPyrLevel(const HfTracker& tracker, size_t lvl) {
        return tracker.pyramids[tracker.prevPyrIdx()].levels[lvl];
    }

    static size_t getBucketIdx(const HfTracker& tracker, double x, double y) {
        return tracker.getBucketIdx(x, y);
    }

    // Draws a sharp high-contrast 90-degree corner at (cx, cy)
    static void drawCorner(std::vector<uint8_t>& img, size_t w, int cx, int cy, uint8_t bg = 50, uint8_t fg = 220) {
        for (int y = cy - 10; y <= cy + 10; ++y) {
            for (int x = cx - 10; x <= cx + 10; ++x) {
                if (x >= 0 && x < static_cast<int>(w) && y >= 0 && y < static_cast<int>(img.size() / w)) {
                    // Top-right quadrant is foreground; rest is background
                    if (x >= cx && y >= cy) {
                        img[static_cast<size_t>(y) * w + static_cast<size_t>(x)] = fg;
                    } else {
                        img[static_cast<size_t>(y) * w + static_cast<size_t>(x)] = bg;
                    }
                }
            }
        }
    }
};

} // namespace PoolSLAM
