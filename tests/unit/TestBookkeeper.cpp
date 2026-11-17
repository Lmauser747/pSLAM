#include <cstddef>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

#include <Eigen/Core>
#include <gtest/gtest.h>
#include <gtsam/geometry/Point2.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>

#include "backend/Bookkeeper.hpp"

namespace PoolSLAM {

class LandmarkBookkeeperTest : public ::testing::Test {
protected:
    static constexpr double kEps = 1e-5;

    gtsam::NonlinearFactorGraph stagingGraph;
    gtsam::Values stagingValues;
    gtsam::Values isamEstimates;
    gtsam::Pose3 bodyPCam;
    BookkeeperConfig config;
    std::unique_ptr<LandmarkBookkeeper> bookkeeper;

    void SetUp() override {
        stagingGraph.resize(0);
        stagingValues.clear();
        isamEstimates.clear();
        bodyPCam = gtsam::Pose3();
        config = BookkeeperConfig();
        bookkeeper = std::make_unique<LandmarkBookkeeper>(
            stagingGraph, stagingValues, isamEstimates, bodyPCam, config
        );
    }

    // Rebuilds bookkeeper when a test customizes extrinsics or config thresholds
    void reinitBookkeeper(const gtsam::Pose3 &extrinsics, const BookkeeperConfig &cfg) {
        bodyPCam = extrinsics;
        config = cfg;
        bookkeeper = std::make_unique<LandmarkBookkeeper>(
            stagingGraph, stagingValues, isamEstimates, bodyPCam, config
        );
    }

    // Registers a body pose in either isamEstimates (past frame) or stagingValues (current frame)
    void setFramePose(uint32_t frameId, const gtsam::Pose3 &bodyPose, bool isCurrentStaging = true) {
        const gtsam::Symbol sym('x', frameId);
        if (isCurrentStaging) {
            stagingValues.insert_or_assign(sym, bodyPose);
            bookkeeper->startFrame(frameId);
        } else {
            isamEstimates.insert_or_assign(sym, bodyPose);
        }
    }

    // Projects a 3D world point into the normalized Z = 1 plane of camera at frameId
    gtsam::Point2 projectToNormalizedRay(uint32_t frameId, const gtsam::Point3 &worldPt) const {
        const gtsam::Pose3 camPose = callGetCameraPose(frameId);
        const Eigen::Vector3d pCam = camPose.transformTo(worldPt);
        return gtsam::Point2(pCam.x() / pCam.z(), pCam.y() / pCam.z());
    }

    // Private member accessors and wrappers
    uint32_t getCurrentFrameId() const { return bookkeeper->currentFrameId; }
    const gtsam::Pose3 &getCurrentCamPose() const { return bookkeeper->currentCamPose; }

    std::unordered_map<uint32_t, LandmarkTrack> &getActiveLandmarks() {
        return bookkeeper->activeLandmarks;
    }

    gtsam::Pose3 callGetCameraPose(uint32_t frameId) const {
        return bookkeeper->getCameraPose(frameId);
    }

    bool callTriangulateLandmark(const std::vector<LandmarkObs> &obs, gtsam::Point3 &outPt) const {
        return bookkeeper->triangulateLandmark(obs, outPt);
    }
};

// ============================================================================
// 1. Frame Setup & Camera Extrinsics Resolution (startFrame, getCameraPose)
// ============================================================================

// Verifies that getCameraPose resolves past poses from isamEstimates and current poses from stagingValues.
TEST_F(LandmarkBookkeeperTest, resolvesPosesFromIsamEstimatesAndStagingValues) {
    const gtsam::Pose3 pose0(gtsam::Rot3(), gtsam::Point3(0.0, 0.0, 0.0));
    const gtsam::Pose3 pose1(gtsam::Rot3(), gtsam::Point3(0.2, -0.1, 0.05));

    setFramePose(0, pose0, false);
    setFramePose(1, pose1, true);

    EXPECT_EQ(getCurrentFrameId(), 1u);
    EXPECT_TRUE(callGetCameraPose(0).equals(pose0, kEps));
    EXPECT_TRUE(callGetCameraPose(1).equals(pose1, kEps));
    EXPECT_TRUE(getCurrentCamPose().equals(pose1, kEps));
}

// Verifies that body-to-camera extrinsics (T_BC) are composed with the body pose in startFrame and getCameraPose.
TEST_F(LandmarkBookkeeperTest, composesBodyToCameraExtrinsics) {
    const gtsam::Pose3 extrinsics(
        gtsam::Rot3::RzRyRx(0.0, 0.0, 1.57079632679),
        gtsam::Point3(0.10, -0.05, 0.02)
    );
    reinitBookkeeper(extrinsics, config);

    const gtsam::Pose3 bodyPose0(gtsam::Rot3::Ypr(0.2, -0.1, 0.05), gtsam::Point3(1.0, 2.0, -0.5));
    const gtsam::Pose3 bodyPose1(gtsam::Rot3::Ypr(0.3, -0.1, 0.05), gtsam::Point3(1.2, 2.0, -0.5));

    setFramePose(0, bodyPose0, false);
    setFramePose(1, bodyPose1, true);

    EXPECT_TRUE(callGetCameraPose(0).equals(bodyPose0.compose(extrinsics), kEps));
    EXPECT_TRUE(callGetCameraPose(1).equals(bodyPose1.compose(extrinsics), kEps));
}

// ============================================================================
// 2. Multi-View 3D Triangulation & Sanity Gates (triangulateLandmark)
// ============================================================================

// Verifies accurate 3D triangulation across two views and four views with non-identity extrinsics.
TEST_F(LandmarkBookkeeperTest, triangulateLandmarkRecoversExact3DPointMultiView) {
    const gtsam::Pose3 extrinsics(gtsam::Rot3::Ypr(0.05, -0.02, 0.01), gtsam::Point3(0.08, 0.0, -0.03));
    reinitBookkeeper(extrinsics, config);

    const gtsam::Point3 trueWorldPt(0.45, -0.30, 3.20);
    std::vector<LandmarkObs> obs;

    for (size_t i = 0; i < 4; i++) {
        const uint32_t frameId = static_cast<uint32_t>(i);
        const gtsam::Pose3 bodyPose(
            gtsam::Rot3::Ypr(0.02 * static_cast<double>(i), 0.0, 0.0),
            gtsam::Point3(0.15 * static_cast<double>(i), -0.05 * static_cast<double>(i), 0.0)
        );
        setFramePose(frameId, bodyPose, i == 3);
        obs.push_back({frameId, projectToNormalizedRay(frameId, trueWorldPt)});
    }

    gtsam::Point3 estPt;
    ASSERT_TRUE(callTriangulateLandmark(obs, estPt));
    EXPECT_NEAR(estPt.x(), trueWorldPt.x(), kEps);
    EXPECT_NEAR(estPt.y(), trueWorldPt.y(), kEps);
    EXPECT_NEAR(estPt.z(), trueWorldPt.z(), kEps);
}

// Verifies that collinear rays (zero parallax angle) fail the LDLT conditioning gate.
TEST_F(LandmarkBookkeeperTest, triangulateLandmarkRejectsCollinearRays) {
    // Camera translates strictly along the optical Z axis; a point on the optical axis (0, 0) has collinear rays
    setFramePose(0, gtsam::Pose3(gtsam::Rot3(), gtsam::Point3(0.0, 0.0, 0.0)), false);
    setFramePose(1, gtsam::Pose3(gtsam::Rot3(), gtsam::Point3(0.0, 0.0, 0.5)), true);

    const std::vector<LandmarkObs> collinearObs = {
        {0, gtsam::Point2(0.0, 0.0)},
        {1, gtsam::Point2(0.0, 0.0)}
    };

    gtsam::Point3 estPt;
    EXPECT_FALSE(callTriangulateLandmark(collinearObs, estPt));
}

// Verifies that points triangulating behind the camera, closer than minDepth, or beyond maxDepth are rejected.
TEST_F(LandmarkBookkeeperTest, triangulateLandmarkRejectsDepthOutOfBoundsAndNegativeCheirality) {
    setFramePose(0, gtsam::Pose3(gtsam::Rot3(), gtsam::Point3(0.0, 0.0, 0.0)), false);
    setFramePose(1, gtsam::Pose3(gtsam::Rot3(), gtsam::Point3(0.4, 0.0, 0.0)), true);

    gtsam::Point3 estPt;

    // 1. Point too close (z = 0.08m < minDepthMeters = 0.15m)
    const gtsam::Point3 tooClosePt(0.2, 0.0, 0.08);
    const std::vector<LandmarkObs> closeObs = {
        {0, projectToNormalizedRay(0, tooClosePt)},
        {1, projectToNormalizedRay(1, tooClosePt)}
    };
    EXPECT_FALSE(callTriangulateLandmark(closeObs, estPt));

    // 2. Point too far (z = 25.0m > maxDepthMeters = 20.0m)
    const gtsam::Point3 tooFarPt(0.2, 0.0, 25.0);
    const std::vector<LandmarkObs> farObs = {
        {0, projectToNormalizedRay(0, tooFarPt)},
        {1, projectToNormalizedRay(1, tooFarPt)}
    };
    EXPECT_FALSE(callTriangulateLandmark(farObs, estPt));

    // 3. Diverging rays that intersect behind both cameras (z = -2.0m < 0)
    const gtsam::Point3 behindPt(0.2, 0.0, -2.0);
    const std::vector<LandmarkObs> behindObs = {
        {0, projectToNormalizedRay(0, behindPt)},
        {1, projectToNormalizedRay(1, behindPt)}
    };
    EXPECT_FALSE(callTriangulateLandmark(behindObs, estPt));
}

// Verifies that skew rays exceeding maxReprojErrorSq are rejected.
TEST_F(LandmarkBookkeeperTest, triangulateLandmarkRejectsHighReprojectionError) {
    setFramePose(0, gtsam::Pose3(gtsam::Rot3(), gtsam::Point3(0.0, 0.0, 0.0)), false);
    setFramePose(1, gtsam::Pose3(gtsam::Rot3(), gtsam::Point3(0.3, 0.0, 0.0)), true);

    const gtsam::Point3 truePt(0.15, 0.10, 2.5);
    const gtsam::Point2 ray0 = projectToNormalizedRay(0, truePt);
    gtsam::Point2 ray1 = projectToNormalizedRay(1, truePt);

    // Perturb vertical component of ray1 to create skew rays that cannot intersect cleanly
    ray1 = gtsam::Point2(ray1.x(), ray1.y() + 0.05);

    const std::vector<LandmarkObs> skewObs = {{0, ray0}, {1, ray1}};
    gtsam::Point3 estPt;
    EXPECT_FALSE(callTriangulateLandmark(skewObs, estPt));
}

// ============================================================================
// 3. Observation Buffering & Parallax Gatekeeper (updateBookeeper)
// ============================================================================

// Verifies that the first observation of a landmark buffers the ray without modifying GTSAM containers.
TEST_F(LandmarkBookkeeperTest, singleObservationBuffersWithoutGraphInsertion) {
    setFramePose(0, gtsam::Pose3(), true);
    bookkeeper->updateBookeeper(42, gtsam::Point2(0.1, -0.2));

    ASSERT_EQ(getActiveLandmarks().count(42u), 1u);
    const auto &track = getActiveLandmarks().at(42u);
    EXPECT_EQ(track.state, LandmarkState::UNINITIALIZED);
    EXPECT_EQ(track.lastSeenFrame, 0u);
    EXPECT_EQ(track.observations.size(), 1u);

    EXPECT_EQ(stagingGraph.size(), 0u);
    EXPECT_FALSE(stagingValues.exists(gtsam::Symbol('l', 42)));
}

// Verifies that sub-threshold camera baseline or sub-threshold ray parallax keeps the landmark buffered.
TEST_F(LandmarkBookkeeperTest, insufficientBaselineOrParallaxRemainsUninitialized) {
    const gtsam::Point3 worldPt(0.0, 0.0, 4.0);

    // Frame 0: Origin
    setFramePose(0, gtsam::Pose3(gtsam::Rot3(), gtsam::Point3(0.0, 0.0, 0.0)), false);
    bookkeeper->startFrame(0);
    bookkeeper->updateBookeeper(7, projectToNormalizedRay(0, worldPt));

    // Frame 1: Move only 1 cm (< minBaselineMeters = 3 cm)
    setFramePose(1, gtsam::Pose3(gtsam::Rot3(), gtsam::Point3(0.01, 0.0, 0.0)), false);
    bookkeeper->startFrame(1);
    bookkeeper->updateBookeeper(7, projectToNormalizedRay(1, worldPt));
    EXPECT_EQ(getActiveLandmarks().at(7).state, LandmarkState::UNINITIALIZED);
    EXPECT_EQ(getActiveLandmarks().at(7).observations.size(), 2u);

    // Frame 2: Move 5 cm (> minBaselineMeters), but parallax at Z = 4.0m is ~0.71 deg (< 2.0 deg threshold)
    setFramePose(2, gtsam::Pose3(gtsam::Rot3(), gtsam::Point3(0.05, 0.0, 0.0)), true);
    bookkeeper->updateBookeeper(7, projectToNormalizedRay(2, worldPt));
    EXPECT_EQ(getActiveLandmarks().at(7).state, LandmarkState::UNINITIALIZED);
    EXPECT_EQ(getActiveLandmarks().at(7).observations.size(), 3u);
    EXPECT_EQ(stagingGraph.size(), 0u);
}

// Verifies that a track failing to initialize within maxBufferLength is marked INVALID and ignored thereafter.
TEST_F(LandmarkBookkeeperTest, bufferOverflowMarksTrackInvalidAndIgnoresFutureObservations) {
    config.maxBufferLength = 4;
    reinitBookkeeper(bodyPCam, config);

    // Keep camera stationary across 4 frames so baseline gate repeatedly fails
    for (size_t k = 0; k < 4; k++) {
        const uint32_t frameId = static_cast<uint32_t>(k);
        setFramePose(frameId, gtsam::Pose3(), k == 3);
        bookkeeper->startFrame(frameId);
        bookkeeper->updateBookeeper(99, gtsam::Point2(0.05, 0.05));
    }

    const auto &track = getActiveLandmarks().at(99);
    EXPECT_EQ(track.state, LandmarkState::INVALID);
    EXPECT_TRUE(track.observations.empty());
    EXPECT_EQ(stagingGraph.size(), 0u);

    // Frame 4: Even if the camera moves to a valid baseline, the INVALID track must be ignored
    setFramePose(4, gtsam::Pose3(gtsam::Rot3(), gtsam::Point3(0.5, 0.0, 0.0)), true);
    bookkeeper->updateBookeeper(99, gtsam::Point2(-0.15, 0.05));
    EXPECT_EQ(track.state, LandmarkState::INVALID);
    EXPECT_TRUE(track.observations.empty());
    EXPECT_EQ(stagingGraph.size(), 0u);
}

// ============================================================================
// 4. Landmark Promotion & GTSAM Factor Insertion (updateBookeeper)
// ============================================================================

// Verifies that meeting parallax across 3 frames inserts Point3 into Values, flushes all 3 factors, and clears the buffer.
TEST_F(LandmarkBookkeeperTest, promotionInsertsLandmarkAndFlushesBufferedFactors) {
    const gtsam::Point3 truePt(0.2, -0.15, 2.0);
    const uint32_t lmId = 15;
    const gtsam::Symbol lmSym('l', lmId);

    // Frame 0: Origin (1st observation)
    setFramePose(0, gtsam::Pose3(gtsam::Rot3(), gtsam::Point3(0.0, 0.0, 0.0)), false);
    bookkeeper->startFrame(0);
    bookkeeper->updateBookeeper(lmId, projectToNormalizedRay(0, truePt));

    // Frame 1: Small step (0.04m -> ~1.1 deg parallax, stays buffered)
    setFramePose(1, gtsam::Pose3(gtsam::Rot3(), gtsam::Point3(0.04, 0.0, 0.0)), false);
    bookkeeper->startFrame(1);
    bookkeeper->updateBookeeper(lmId, projectToNormalizedRay(1, truePt));
    ASSERT_EQ(getActiveLandmarks().at(lmId).state, LandmarkState::UNINITIALIZED);

    // Frame 2: Larger step (0.12m -> ~3.4 deg parallax, triggers triangulation & promotion)
    setFramePose(2, gtsam::Pose3(gtsam::Rot3(), gtsam::Point3(0.12, 0.0, 0.0)), true);
    bookkeeper->updateBookeeper(lmId, projectToNormalizedRay(2, truePt));

    const auto &track = getActiveLandmarks().at(lmId);
    EXPECT_EQ(track.state, LandmarkState::INITIALIZED);
    EXPECT_TRUE(track.observations.empty());

    ASSERT_TRUE(stagingValues.exists(lmSym));
    const gtsam::Point3 estPt = stagingValues.at<gtsam::Point3>(lmSym);
    EXPECT_NEAR(estPt.x(), truePt.x(), kEps);
    EXPECT_NEAR(estPt.y(), truePt.y(), kEps);
    EXPECT_NEAR(estPt.z(), truePt.z(), kEps);

    // All 3 buffered observations (frames 0, 1, 2) must be flushed into stagingGraph
    EXPECT_EQ(stagingGraph.size(), 3u);
}

// Verifies that an already INITIALIZED landmark immediately appends a single factor per new frame
// and that all generated GenericProjectionFactors evaluate to zero error at the ground-truth state.
TEST_F(LandmarkBookkeeperTest, initializedLandmarkAppendsSingleFactorWithZeroGroundTruthError) {
    const gtsam::Pose3 extrinsics(gtsam::Rot3::Ypr(0.1, -0.05, 0.02), gtsam::Point3(0.05, 0.02, -0.01));
    reinitBookkeeper(extrinsics, config);

    const gtsam::Point3 truePt(-0.25, 0.30, 2.5);
    const uint32_t lmId = 21;

    const gtsam::Pose3 pose0(gtsam::Rot3(), gtsam::Point3(0.0, 0.0, 0.0));
    const gtsam::Pose3 pose1(gtsam::Rot3(), gtsam::Point3(0.20, 0.0, 0.0));
    const gtsam::Pose3 pose2(gtsam::Rot3(), gtsam::Point3(0.35, 0.05, -0.02));

    // Initialize across frames 0 and 1
    setFramePose(0, pose0, false);
    bookkeeper->startFrame(0);
    bookkeeper->updateBookeeper(lmId, projectToNormalizedRay(0, truePt));

    setFramePose(1, pose1, true);
    bookkeeper->updateBookeeper(lmId, projectToNormalizedRay(1, truePt));
    ASSERT_EQ(getActiveLandmarks().at(lmId).state, LandmarkState::INITIALIZED);
    ASSERT_EQ(stagingGraph.size(), 2u);

    // Simulate end-of-frame 1: move poses to isamEstimates and clear staging containers
    isamEstimates.insert(gtsam::Symbol('x', 1), pose1);
    isamEstimates.insert(gtsam::Symbol('l', lmId), stagingValues.at<gtsam::Point3>(gtsam::Symbol('l', lmId)));
    stagingGraph.resize(0);
    stagingValues.clear();

    // Frame 2: Subsequent observation on an INITIALIZED landmark
    setFramePose(2, pose2, true);
    bookkeeper->updateBookeeper(lmId, projectToNormalizedRay(2, truePt));

    // Must not re-insert L(21) into stagingValues, and must add exactly 1 factor to stagingGraph
    EXPECT_FALSE(stagingValues.exists(gtsam::Symbol('l', lmId)));
    ASSERT_EQ(stagingGraph.size(), 1u);

    auto projFactor = std::dynamic_pointer_cast<LandmarkBookkeeper::ProjectionFactor>(stagingGraph.at(0));
    ASSERT_NE(projFactor, nullptr);

    const Eigen::VectorXd err = projFactor->evaluateError(pose2, truePt);
    EXPECT_NEAR(err.norm(), 0.0, kEps);
}

// ============================================================================
// 5. Dead Track Garbage Collection (pruneDeadTracks)
// ============================================================================

// Verifies that pruneDeadTracks erases landmarks not seen in currentFrameId while retaining current tracks.
TEST_F(LandmarkBookkeeperTest, pruneDeadTracksRemovesOnlyStaleLandmarks) {
    setFramePose(0, gtsam::Pose3(), false);
    bookkeeper->startFrame(0);

    // Frame 0: Observe landmarks 1, 2, and 3
    bookkeeper->updateBookeeper(1, gtsam::Point2(-0.1, 0.1));
    bookkeeper->updateBookeeper(2, gtsam::Point2(0.0, 0.2));
    bookkeeper->updateBookeeper(3, gtsam::Point2(0.2, -0.1));
    EXPECT_EQ(getActiveLandmarks().size(), 3u);

    // Frame 1: Only landmark 2 survives; landmark 4 is newly detected
    setFramePose(1, gtsam::Pose3(gtsam::Rot3(), gtsam::Point3(0.05, 0.0, 0.0)), true);
    bookkeeper->updateBookeeper(2, gtsam::Point2(0.01, 0.2));
    bookkeeper->updateBookeeper(4, gtsam::Point2(-0.2, -0.2));

    bookkeeper->pruneDeadTracks();

    EXPECT_EQ(getActiveLandmarks().size(), 2u);
    EXPECT_EQ(getActiveLandmarks().count(1u), 0u);
    EXPECT_EQ(getActiveLandmarks().count(3u), 0u);
    EXPECT_EQ(getActiveLandmarks().count(2u), 1u);
    EXPECT_EQ(getActiveLandmarks().count(4u), 1u);
}

} // namespace PoolSLAM