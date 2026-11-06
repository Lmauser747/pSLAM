#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <optional>
#include <vector>
#include <memory>
#include <array>

#include <eigen3/Eigen/Dense>

#include "CoreTypes.hpp"
#include "descriptor/Qov.hpp"
#include "descriptor/BinaryBow.hpp"
#include "simd/Utils.hpp"

namespace PoolSLAM{

struct LfTrackerConfig{
    size_t wImage, hImage; // Width and height of image
    size_t numBuckets; // Number of buckets for features to go in

    const Eigen::Matrix3f K; // Camera matrix
    Distortion dist; // Distortion parameters
    
    // Disabled right now because we aren't generating new features in LfTracker right now
    // size_t targetFeatureCount; // Target number of QOV features per-frame
    // float fastThreshold = 20.0f; // Threshold for FAST in QOV

    float minTfIdfScore; // Minimum score for TF-IDF
    size_t excludedFrames; // Number of previous frames to exclude to ensure long-term matches

    uint32_t RansacInliearThreshold = 20; // Minimum number of RANSAC inliers
    float minInliearRatio = 0.30f; // Minimum ratio of inlier to total landmarks
    float maxReprojectionError = 2.00f; // Maximum reprojection error from mathec frame to current frame

};

// Struct to store data for Gaussian blur
struct BlurData{
    // Separated blur kernel
    static constexpr std::array<uint16_t, 5> KERNEL = {1, 4, 6, 4, 1};
    std::unique_ptr<uint8_t[]> rowBuffer; // Bufer to store rows being blured by row kerner
    std::array<std::unique_ptr<uint8_t[]>, 3> ringBuffer; // Buffer to store rows after being blured by column kernel
};

// Stores structs needed to caclulate the tranformation between frames
struct LfMatching{
    Eigen::Matrix3Xf curFeatures; // Homogenous points from the current frame
    Eigen::Matrix3Xf matchFeatures; // Homogenous points from the previous frame

};

class LfTracker{
private:
    static constexpr int FEATURE_RADIUS = 5; // Radius of the feature patch from HfTracker
    static constexpr int QOV_RADIUS = 15; // Radius of the patched used by QOV
    static constexpr int UNDISTORT_ITERS = 5; // Number of iterations in undistortion
    static constexpr float MIN_SCORE_RATIO = 0.3f; // Make sure the 

    const LfTrackerConfig config; // Class configuration

    std::unique_ptr<uint8_t[]> localBuffer; // Local buffer to store image frame
    BlurData blurData; // Data for blur functions
    std::vector<Keyframe> keyframes; // All keyframes
    const descriptor::RoatedQovPairs qovPairs; // Pairs of points for QOV compairisons
    descriptor::BinaryBowTree vocabTree; // Tree to store vocabulary and associated frames
    LfMatching matching; // Stores the data needed for Lf matching

    size_t nextFrameIdx = 0; // Next 

    std::optional<BackendLfPacket> stagedMatch; // Match staged for sending to backend

    std::atomic<bool> dataReady{false}; // Flag to tell the backend if the data is ready yet
    std::atomic<bool> runTracker{false}; // Controls if the tracker is running or not
    std::atomic<bool> busy{false}; // Flag to tell the frontend the tracker is busy

    std::mutex m_passKeyframe; // Mutex to control passing of keyframe
    std::mutex m_backendPacket; // Mutex to control output of backend packet

    std::condition_variable c_passKeyframe; // Condition varaiable to signal new keyframe

    // Process keyframe. Runs in thread
    void processKeyframe();

    // Blur the image with Gaussian blur kernel of radius 5
    void gaussianBlur();

    // Calculate the Qov descriptors
    void calculateDescriptors();

    // Match features
    size_t matchFeatures(size_t curIdx, size_t matchIdx);

    // Undistort the features
    void undistort(const std::vector<descriptor::QovFeature> &ptDist, Eigen::Matrix3Xf &ptUndist);

public:
    LfTracker(LfTrackerConfig c, std::string &vocabPath) : config(c), qovPairs(config.wImage),
    vocabTree(vocabPath){
        localBuffer = std::make_unique<uint8_t[]>(config.wImage * config.hImage);
        blurData.rowBuffer = std::make_unique<uint8_t[]>(config.wImage * config.hImage);
        for(auto &buf : blurData.ringBuffer){
            buf = std::make_unique<uint8_t[]>(config.wImage * config.hImage);
        }
    }

    // Function to pass keyframe from the frontend to the LfTracker
    bool passKeyframe(uint64_t id, double timestamp, const uint8_t *curBuffer, const std::vector<TrackedFeature<FEATURE_RADIUS>> &features);
    
    // Function to get the packet from the LfTracker to the backend through outPacket
    bool getMatch(BackendLfPacket &outPacket);

    // Start tracker
    void start();

    // End tracker
    void end();

    // Is tracker running
    void isRunning();

};
}
