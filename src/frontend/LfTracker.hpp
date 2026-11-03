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

#include <eigen3/Eigen/Dense>

#include <CoreTypes.hpp>
#include <descriptor/Qov.hpp>

namespace PoolSLAM{

struct LfTrackerConfig{
    size_t wImage, hImage; // Width and height of image
    size_t numBuckets; // Number of buckets for features to go in

    const Eigen::Matrix3f K; // Camera matrix
    Distortion dist; // Distortion parameters
    
    // Disabled right now because we aren't generating new features in LfTracker right now
    // size_t targetFeatureCount; // Target number of QOV features per-frame
    // float fastThreshold = 20.0f; // Threshold for FAST in QOV

    std::string vocabularyPath; // Path to binary file containing BoW vocabulary
    float minTfIdfScore; // Minimum score for TF-IDF
    size_t excludedFrames; // Number of previous frames to exclude to ensure long-term matches

    uint32_t RansacInliearThreshold = 20; // Minimum number of RANSAC inliers
    float minInliearRatio = 0.30f; // Minimum ratio of inlier to total landmarks
    float maxReprojectionError = 2.00f; // Maximum reprojection error from mathec frame to current frame

};


class LfTracker{
private:
    static constexpr int FEATURE_RADIUS = 5; // Radius of the feature patch from HfTracker
    static constexpr int QOV_RADIUS = 15; // Radius of the patched used by QOV

    const LfTrackerConfig config; // Class configuration

    std::unique_ptr<uint8_t[]> localBuffer; // Local buffer to store image frame
    descriptor::FrameData frame; // QOV Descriptors in the current image
    const descriptor::RoatedQovPairs qovPairs; // Pairs of points for QOV compairisons

    std::optional<BackendLfPacket> stagedMatch; // Match staged for sending to backend

    std::atomic<bool> dataReady{false}; // Flag to tell the backend if the data is ready yet
    std::atomic<bool> runTracker{false}; // Controls if the tracker is running or not
    std::atomic<bool> busy{false}; // Flag to tell the frontend the tracker is busy

    std::mutex m_passKeyframe; // Mutex to control passing of keyframe
    std::mutex m_backendPacket; // Mutex to control output of backend packet

    std::condition_variable c_passKeyframe; // Condition varaiable to signal new keyframe

    // Blur the image with Gaussian blur kernel of radius 5
    void blur();

public:
    LfTracker(LfTrackerConfig c) : config(c), frame(config.numBuckets * 2),
    qovPairs(config.wImage){

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
