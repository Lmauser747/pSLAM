#include <cstdint>
#include <cstddef>
#include <string>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <optional>
#include <vector>

#include <eigen3/Eigen/Dense>

#include <CoreTypes.hpp>
#include <descriptor/Orb.hpp>

namespace PoolSLAM{

struct LfTrackerConfig{
    size_t wImage, hImage; // Width and height of image
    size_t numBuckets; // Number of buckets for features to go in
    
    Eigen::Matrix3f K; // Camera matrix
    Distortion dist; // Distortion parameters
    
    size_t targetFeatureCount; // Target number of ORB features per-frame
    float fastThreshold = 20.0f; // Threshold for FAST in ORB

    std::string vocabularyPath; // Path to binary file containing BoW vocabulary
    float minTfIdfScore; // Minimum score for TF-IDF
    size_t excludedFrames; // Number of previous frames to exclude to ensure long-term matches

    uint32_t RansacInliearThreshold = 20; // Minimum number of RANSAC inliers
    float minInliearRatio = 0.30f; // Minimum ratio of inlier to total landmarks
    float maxReprojectionError = 2.00f; // Maximum reprojection error from mathec frame to current frame
};

class LfTracker{
private:
    static constexpr int ST_RADIUS = 5;
    static constexpr int ORB_RADIUS = 15;
    const LfTrackerConfig config; // Class configuration

    descriptor::FrameData frame; // Orb Descriptors in the current image
    const descriptor::RoatedOrbPairs orbPairs; // Pairs of points for ORB compairisons

    std::optional<BackendLfPacket> stagedMatch; // Match staged for sending to backend

    std::atomic<bool> dataReady{false}; // Flag to tell the backend if the data is ready yet
    std::atomic<bool> runTracker{false}; // Controls if the tracker is running or not
    std::atomic<bool> busy{false}; // Flag to tell the frontend the tracker is busy

    std::mutex m_passKeyframe; // Mutex to control passing of keyframe
    std::mutex m_backendPacket; // Mutex to control output of backend packet

    std::condition_variable c_passKeyframe; // Condition varaiable to signal new keyframe

public:
    LfTracker(LfTrackerConfig c) : config(c), frame(config.numBuckets * 2),
    orbPairs(config.wImage){}

    // Function to pass keyframe from the frontend to the LfTracker
    bool calculateDescriptors(uint64_t id, double timestamp, const uint8_t *curBuffer, const std::vector<TrackedFeature<ST_RADIUS>> &features);
    
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
