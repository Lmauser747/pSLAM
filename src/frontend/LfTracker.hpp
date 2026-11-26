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
#include <random>
#include <utility>
#include <thread>

#include <eigen3/Eigen/Dense>
#include <gtsam/geometry/Pose3.h>

#include "CoreTypes.hpp"
#include "descriptor/Qov.hpp"
#include "descriptor/BinaryBow.hpp"


namespace PoolSLAM{

struct LfTrackerConfig{
    size_t w, h; // Width and height of image
    size_t numBuckets = 400; // Number of buckets for features to go in

    Eigen::Matrix3f K; // Camera matrix
    Distortion dist; // Distortion parameters
    
    // Disabled right now because we aren't generating new features in LfTracker right now
    // size_t targetFeatureCount; // Target number of QOV features per-frame
    // float fastThreshold = 20.0f; // Threshold for FAST in QOV

    float minTfIdfScore = 0.03f; // Minimum score for TF-IDF
    size_t excludedFrames = 20; // Number of previous frames to exclude to ensure long-term matches

    uint32_t ransacInliearThreshold = 12; // Minimum number of RANSAC inliers
    int ransacIters = 400;
    float minInliearRatio = 0.30f; // Minimum ratio of inlier to total landmarks
    float maxReprojectionError = 2.50f; // Maximum reprojection error from mathec frame to current frame
    

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
    Eigen::Matrix3Xd curFeatures; // Homogenous points from the current frame
    Eigen::Matrix3Xd matchLandmarks; // 3d positions of landmarks from previous frame

};

// Data associated with each key frame
class Keyframe{
public: 
    std::vector<descriptor::QovFeature> features; // Features in the frame
    std::vector<Landmark3d> landmarks; // Position of each feature relative to the frame
    gtsam::Pose3 pose; // Global pose of the frame
    uint32_t id; // Frame id
    double timestamp; // Timestamp of the frame
    uint32_t keyframeId; // Keyframe id
    float score = 0; // Score of the frame compared to the current frame 

    Keyframe(size_t numFeatures, uint32_t frameId, double time, uint32_t kfId) : 
    id(frameId), timestamp(time), keyframeId(kfId){
        features.reserve(numFeatures);
    }

    void addKp(const Eigen::Vector2i &kp, uint32_t featureId){
        features.emplace_back(kp, featureId);
    }

    void addKp(const Eigen::Vector2f &kpSubpix, uint32_t featureId){
        features.emplace_back(kpSubpix, featureId);
    }

    // Resize the feature vector to final size once all keypoints have been added
    size_t finalSize(){
        features.shrink_to_fit();
        return features.size();
    }


};

namespace geometry{
std::optional<Eigen::Matrix<double, 3, 4>> solveP4PLambdaTwist(const std::array<Eigen::Vector3d, 4>& x, const std::array<Eigen::Vector3d, 4>& y);
}

class LfTracker{
private:
    static constexpr int FEATURE_RADIUS = 5; // Radius of the feature patch from HfTracker
    static constexpr int QOV_RADIUS = 15; // Radius of the patched used by QOV
    static constexpr int UNDISTORT_ITERS = 10; // Number of iterations in undistortion
    static constexpr float MIN_SCORE_RATIO = 0.25f; // Min ratio between score of matched keyframe and score of previous keyframe
    static constexpr float MAX_FEATURE_DIST_RATIO = 0.80f; // Max ratio between best feature match and second best match
    static constexpr int MAX_HAMMING_DIST = 64; // Maximum Hamming distance allowed for descriptor matching

    const LfTrackerConfig config; // Class configuration

    std::unique_ptr<uint8_t[]> localBuffer; // Local buffer to store image frame
    BlurData blurData; // Data for blur functions
    std::vector<Keyframe> keyframes; // All keyframes
    const descriptor::RotatedQovPairs qovPairs; // Pairs of points for QOV compairisons
    descriptor::BinaryBowTree vocabTree; // Tree to store vocabulary and associated frames
    descriptor::BowDatabase database; // Mantains lookup for features in the frame
    LfMatching matching; // Stores the data needed for Lf matching

    uint32_t nextFrameId = 0; // Id for next frame

    std::minstd_rand ransacRng; // Random number generator for RANSAC

    std::optional<BackendLfPacket> stagedMatch; // Match staged for sending to backend

    std::atomic<bool> dataReady{false}; // Flag to tell the backend if the data is ready yet
    std::atomic<bool> runTracker{false}; // Controls if the tracker is running or not
    std::atomic<bool> busy{false}; // Flag to tell the frontend the tracker is busy

    std::mutex m_passKeyframe; // Mutex to control passing of keyframe
    std::mutex m_backendPacket; // Mutex to control output of backend packet

    std::condition_variable c_passKeyframe; // Condition varaiable to signal new keyframe
    std::thread workerThread; // Worker thread for keyframe processing

    // Process keyframe. Runs in thread
    void processKeyframe();

    // Blur the image with Gaussian blur kernel of radius 5
    void gaussianBlur();

    // Calculate the Qov descriptors
    void calculateDescriptors();

    // Match features
    size_t matchFeatures(size_t curId, size_t matchId);

    // Undistort the features
    void undistort(Eigen::Matrix3Xd &pts, size_t count = 0);

    // Use RANSAC algorihtm to prune outliers based on P3P
    std::optional<std::pair<Eigen::Matrix<double, 3, 4>, size_t>> ransac(size_t numMatches);

    // Do LMA to find the optimal pose
    bool refinePoseLM(size_t numInliers, const Eigen::Matrix3d& K_cam, Eigen::Matrix<double, 3, 4>& Tcand, Eigen::Matrix<double, 6, 6>& outCovariance);

    friend class TestLfTracker;
    friend class TestLma;
    friend class TestP3p;
public:
    LfTracker(LfTrackerConfig c, const std::string &vocabPath) : config(c), qovPairs(config.w),
    vocabTree(vocabPath){
        localBuffer = std::make_unique<uint8_t[]>(config.w * config.h);
        blurData.rowBuffer = std::make_unique<uint8_t[]>(config.w);
        for(auto &buf : blurData.ringBuffer){
            buf = std::make_unique<uint8_t[]>(config.w);
        }

        matching.curFeatures.resize(3, 2 * static_cast<int>(config.numBuckets));
        matching.matchLandmarks.resize(3, 2 * static_cast<int>(config.numBuckets));
    }

    ~LfTracker(){
        end();
    }

    // Function to pass keyframe from the frontend to the LfTracker
    bool passKeyframe(uint32_t id, double timestamp, const uint8_t *curBuffer, const std::vector<TrackedFeature<FEATURE_RADIUS>> &features);
    
    // Function to get the packet from the LfTracker to the backend through outPacket
    bool getMatch(BackendLfPacket &outPacket);

    // Add 3d landmarks to the keyframe
    bool updateKeyframe(BackendKeyframeUpdate &update);

    // Start tracker
    void start();

    // End tracker
    void end();

    // Is tracker running
    bool isRunning() const;

};
}