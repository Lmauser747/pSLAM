#pragma once

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>
#include <limits>

#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <gtsam/geometry/Cal3_S2.h>
#include <gtsam/geometry/Point2.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/ProjectionFactor.h>

namespace PoolSLAM{

enum class LandmarkState : uint8_t{
    UNINITIALIZED = 0, // Buffering 2D rays until camera motion provides parallax
    INITIALIZED   = 1, // Triangulated and active inside GTSAM
    INVALID       = 2  // Failed geometry checks, ignored until HfTracker losses it
};

struct LandmarkObs{
    uint32_t frameId = 0;
    gtsam::Point2 ray; // Normalized 
};

struct LandmarkTrack{
    uint32_t id = 0;
    uint32_t lastSeenFrame = 0;
    LandmarkState state = LandmarkState::UNINITIALIZED;
    std::vector<LandmarkObs> observations;

    explicit LandmarkTrack(uint32_t landmarkId = 0) : id(landmarkId) {
        observations.reserve(8);
    }
};

struct BookkeeperConfig{
    double minBaselineMeters = 0.03;    // Mimumum baseline for triangulation
    double maxCosParallax = 0.99939;    // Mimimum cosine in parallax calculations
    double minDepthMeters = 0.15;       // Minimum valid cheirality depth in front of camera
    double maxDepthMeters = 20.0;       // Maximum valid depth of landmark in front of camera
    double maxReprojErrorSq = 1.0e-4;   // Maximum square of reprojection error
    double minLdltDiag = 1e-4;          // Minimum normalized eigenvalue for ray conditioning
    double visualSigmaNorm = 0.002;     // ~1.0 px / 500 fx in normalized coordinates
    double huberK = 1.345;              // Huber M-estimator tuning constant
    size_t maxBufferLength = 15;        // Max observations before flagging track INVALID
};

class LandmarkBookkeeper{
public:
    using ProjectionFactor = gtsam::GenericProjectionFactor<gtsam::Pose3, gtsam::Point3, gtsam::Cal3_S2>;

    LandmarkBookkeeper(
        gtsam::NonlinearFactorGraph &stagingGraphRef,
        gtsam::Values &stagingValuesRef,
        const gtsam::Values &isamEstimatesRef,
        const gtsam::Pose3 &bodyToCamExtrinsics = gtsam::Pose3(),
        const BookkeeperConfig &cfg = BookkeeperConfig()
    );

    // Called by Backend at the start of frame k after inserting the IMU-predicted X(k)
    void startFrame(uint32_t frameId);

    // Processes a single 2D normalized feature observation from BackendPacket
    void updateBookeeper(uint32_t landmarkId, const gtsam::Point2 &pt);

    // Called by Backend at the end of frame k before running isam2.update()
    void pruneDeadTracks();

    friend class LandmarkBookkeeperTest;

private:
    gtsam::Pose3 getCameraPose(uint32_t frameId) const;
    bool triangulateLandmark(const std::vector<LandmarkObs> &obs, gtsam::Point3 &outPt) const;

    BookkeeperConfig config;

    // References to the parent Backend's GTSAM state containers
    gtsam::NonlinearFactorGraph &stagingGraph;
    gtsam::Values &stagingValues;
    const gtsam::Values &isamEstimates;

    // Extrinsic calibration and unit pinhole calibration
    gtsam::Pose3 bodyPCam;
    gtsam::Cal3_S2::shared_ptr unitCalib; 
    gtsam::SharedNoiseModel visualNoise; 

    // Per-frame cached state to avoid repeated lookups into stagingValues
    uint32_t currentFrameId = std::numeric_limits<uint32_t>::max();
    gtsam::Pose3 currentCamPose;

    std::unordered_map<uint32_t, LandmarkTrack> activeLandmarks;
};

}