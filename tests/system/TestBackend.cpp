#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>
#include <Eigen/Dense>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/navigation/ImuBias.h>
#include <gtsam/navigation/NavState.h>

#include "backend/Backend.hpp"
#include "frontend/Frontend.hpp"
#include "hal/Synchronizer.hpp"
#include "pool_slam/Hal.hpp"
#include "pool_slam/SystemTypes.hpp"
#include "CoreTypes.hpp"

#include "TestBackendHelpers.hpp"
#include "TestFrontendHelpers.hpp"

namespace PoolSLAM {

using gtsam::symbol_shorthand::B;
using gtsam::symbol_shorthand::L;
using gtsam::symbol_shorthand::V;
using gtsam::symbol_shorthand::X;

class BackendTest : public ::testing::Test {
protected:
    static constexpr size_t kImageWidth = test_helpers::kDefaultWidth;
    static constexpr size_t kImageHeight = test_helpers::kDefaultHeight;
    static constexpr size_t kQueueCapacity = 20;
    static constexpr size_t kPoolCapacity = 25;

    ImuStream imuStream;
    DepthStream depthStream;
    ImageStream imageStream{kImageWidth, kImageHeight, kQueueCapacity, kPoolCapacity};
    std::unique_ptr<Synchronizer> synchronizer;
    std::unique_ptr<Frontend> frontend;
    std::unique_ptr<Backend> backend;

    std::string vocabPath;
    FrontendConfig frontendConfig;
    BackendConfig backendConfig;

    std::vector<uint8_t> baseSceneBuffer;

    void SetUp() override {
        vocabPath = test_helpers::writeTempVocabFile("/tmp/test_backend_vocab.bin");
        frontendConfig = test_helpers::makeFrontendConfig(kImageWidth, kImageHeight);
        backendConfig = test_helpers::makeDefaultBackendConfig();

        baseSceneBuffer = test_helpers::renderProjectedGridScene(
            kImageWidth, kImageHeight, frontendConfig.K
        );

        synchronizer = std::make_unique<Synchronizer>(
            imuStream, depthStream, imageStream, kQueueCapacity
        );
        frontend = std::make_unique<Frontend>(frontendConfig, *synchronizer, vocabPath);
        backend = std::make_unique<Backend>(*frontend, frontend->getLfTracker(), backendConfig);
    }

    void TearDown() override {
        if (backend) {
            backend->end();
        }
        if (frontend) {
            frontend->end();
        }
        if (synchronizer) {
            synchronizer->stop();
        }
        if (!vocabPath.empty()) {
            std::remove(vocabPath.c_str());
        }
    }

    // Helper to wait until Backend processes and optimizes up to a target frame ID and timestamp
    bool waitUntilProcessedFrame(uint32_t targetFrameId,
                                 double expectedTimestamp,
                                 std::chrono::milliseconds timeout = std::chrono::milliseconds(2500)) {
        const auto deadline = std::chrono::steady_clock::now() + timeout;
        while (std::chrono::steady_clock::now() < deadline) {
            if (backend->isInitialized && backend->prevFrameId >= targetFrameId) {
                if (expectedTimestamp > 0.0) {
                    if (backend->getCurrentPose().timestamp >= expectedTimestamp - 1e-4 &&
                        backend->isamEstimate.exists(X(targetFrameId))) {
                        return true;
                    }
                } else if (backend->isamEstimate.exists(X(targetFrameId))) {
                    return true;
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        return backend->isInitialized &&
               backend->prevFrameId >= targetFrameId &&
               backend->isamEstimate.exists(X(targetFrameId));
    }

    // Helper overload without timestamp constraint
    bool waitUntilProcessedFrame(uint32_t targetFrameId,
                                 std::chrono::milliseconds timeout = std::chrono::milliseconds(2500)) {
        return waitUntilProcessedFrame(targetFrameId, -1.0, timeout);
    }

    // Accessors for private Backend state via friend class BackendTest
    bool isBackendInitialized(const Backend &b) const {
        return b.isInitialized;
    }

    uint32_t getPrevFrameId(const Backend &b) const {
        return b.prevFrameId;
    }

    const gtsam::Values &getIsamEstimate(const Backend &b) const {
        return b.isamEstimate;
    }

    const std::deque<Backend::PendingKfRecord> &getTrackedKeyframes(const Backend &b) const {
        return b.trackedKeyframes;
    }

    void injectLfPacket(Backend &b, const BackendLfPacket &pkt) {
        b.processLfPacket(pkt);
    }

    void optimizeAndPublish(Backend &b, uint32_t frameId, double timestamp, bool loopClosed) {
        b.optimizeAndPublish(frameId, timestamp, loopClosed);
    }

    void processSingleHfPacket(Backend &b, const BackendHfPacket &pkt) {
        if (!b.isInitialized) {
            b.initializeSystem(pkt);
        } else {
            b.processHfPacket(pkt);
        }
    }
};

// ============================================================================
// 1. Initial State Anchoring & Prior Insertion (Frame 0)
// ============================================================================

TEST_F(BackendTest, FirstFrameAnchorInitializesPoseWithDepthAndZeroVelocity) {
    synchronizer->start();
    frontend->start();
    backend->start();

    // Push initial depth bracketing t = 1.020s
    depthStream.pushDepth({1.000, 12.0});
    depthStream.pushDepth({1.050, 12.0});

    ASSERT_TRUE(imageStream.pushFrame(baseSceneBuffer.data(), 1.020));

    ASSERT_TRUE(waitUntilProcessedFrame(0u, 1.020));
    EXPECT_TRUE(isBackendInitialized(*backend));
    EXPECT_EQ(getPrevFrameId(*backend), 0u);

    // Verify Pose published through double-buffer API
    Pose p0 = backend->getCurrentPose();
    EXPECT_DOUBLE_EQ(p0.timestamp, 1.020);
    EXPECT_NEAR(p0.t.z(), 12.0, 0.1);
    EXPECT_NEAR(p0.v.norm(), 0.0, 1e-3);

    // Verify GTSAM factor graph initial estimates exist
    const gtsam::Values &est = getIsamEstimate(*backend);
    ASSERT_TRUE(est.exists(X(0)));
    ASSERT_TRUE(est.exists(V(0)));
    ASSERT_TRUE(est.exists(B(0)));

    const gtsam::Pose3 pose0 = est.at<gtsam::Pose3>(X(0));
    EXPECT_NEAR(pose0.z(), 12.0, 0.1);
    EXPECT_NEAR(pose0.x(), 0.0, 1e-2);
    EXPECT_NEAR(pose0.y(), 0.0, 1e-2);
}

TEST_F(BackendTest, FirstFrameStationaryImuGravityLevelingAlignsAttitude) {
    synchronizer->start();
    frontend->start();
    backend->start();

    // Push tilted stationary IMU measurement (accelerometer measures +9.81 along X axis)
    imuStream.pushImu(test_helpers::createImuSample(
        1.010, Eigen::Vector3d(9.81, 0.0, 0.0), Eigen::Vector3d::Zero()));
    imuStream.pushImu(test_helpers::createImuSample(
        1.020, Eigen::Vector3d(9.81, 0.0, 0.0), Eigen::Vector3d::Zero()));

    // Push initial depth
    depthStream.pushDepth({1.000, 5.0});
    depthStream.pushDepth({1.050, 5.0});

    ASSERT_TRUE(imageStream.pushFrame(baseSceneBuffer.data(), 1.020));

    ASSERT_TRUE(waitUntilProcessedFrame(0u, 1.020));
    EXPECT_TRUE(isBackendInitialized(*backend));

    Pose p0 = backend->getCurrentPose();
    EXPECT_DOUBLE_EQ(p0.timestamp, 1.020);
    EXPECT_NEAR(p0.t.z(), 5.0, 0.1);

    // Verify gravity leveling rotated body acceleration [9.81, 0, 0] to world [0, 0, 9.81]
    const Eigen::Vector3d a_body(9.81, 0.0, 0.0);
    const Eigen::Vector3d a_world = p0.r * a_body;
    EXPECT_NEAR(a_world.x(), 0.0, 1e-3);
    EXPECT_NEAR(a_world.y(), 0.0, 1e-3);
    EXPECT_NEAR(a_world.z(), 9.81, 1e-3);
}

// ============================================================================
// 2. Sequential Multi-Frame VIO Tracking
// ============================================================================

TEST_F(BackendTest, MultiFrameVioSequentialTrackingMaintainsFiniteEstimates) {
    synchronizer->start();
    frontend->start();
    backend->start();

    std::atomic<bool> runProducers{true};
    const auto startWall = std::chrono::steady_clock::now();

    auto sleepUntilSimTime = [&startWall](double simTime) {
        const auto offset = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
            std::chrono::duration<double>(simTime - 1.000));
        std::this_thread::sleep_until(startWall + offset);
    };

    // Producer 1: IMU @ 200 Hz (at rest: specific force is +9.81 m/s^2 opposing gravity)
    std::thread imuThread([&]() {
        double t = 1.000;
        while (runProducers.load(std::memory_order_relaxed)) {
            sleepUntilSimTime(t);
            imuStream.pushImu(test_helpers::createImuSample(
                t, Eigen::Vector3d(0.0, 0.0, 9.81), Eigen::Vector3d::Zero()));
            t += 0.005;
        }
    });

    // Producer 2: Depth @ 20 Hz
    std::thread depthThread([&]() {
        double t = 1.000;
        while (runProducers.load(std::memory_order_relaxed)) {
            sleepUntilSimTime(t);
            depthStream.pushDepth({t, 6.0});
            t += 0.050;
        }
    });

    // Producer 3: Camera @ 30 Hz
    constexpr size_t kNumFrames = 8;
    std::thread cameraThread([&]() {
        double t = 1.020;
        while (runProducers.load(std::memory_order_relaxed)) {
            sleepUntilSimTime(t);
            imageStream.pushFrame(baseSceneBuffer.data(), t);
            t += 0.033;
        }
    });

    ASSERT_TRUE(waitUntilProcessedFrame(static_cast<uint32_t>(kNumFrames - 1), std::chrono::milliseconds(4000)));

    runProducers.store(false, std::memory_order_relaxed);
    cameraThread.join();
    imuThread.join();
    depthThread.join();

    // Verify state estimates exist and are finite across all processed frames
    const gtsam::Values &est = getIsamEstimate(*backend);
    for (uint32_t f = 0; f < kNumFrames; f++) {
        ASSERT_TRUE(est.exists(X(f))) << "Missing pose for frame " << f;
        ASSERT_TRUE(est.exists(V(f))) << "Missing velocity for frame " << f;
        ASSERT_TRUE(est.exists(B(f))) << "Missing bias for frame " << f;

        const gtsam::Pose3 pose = est.at<gtsam::Pose3>(X(f));
        EXPECT_TRUE(std::isfinite(pose.x()));
        EXPECT_TRUE(std::isfinite(pose.y()));
        EXPECT_TRUE(std::isfinite(pose.z()));
        EXPECT_NEAR(pose.z(), 6.0, 0.2);

        const gtsam::Vector3 vel = est.at<gtsam::Vector3>(V(f));
        EXPECT_TRUE(std::isfinite(vel.x()));
        EXPECT_TRUE(std::isfinite(vel.y()));
        EXPECT_TRUE(std::isfinite(vel.z()));
    }

    Pose currentPose = backend->getCurrentPose();
    EXPECT_TRUE(std::isfinite(currentPose.t.x()));
    EXPECT_TRUE(std::isfinite(currentPose.t.y()));
    EXPECT_TRUE(std::isfinite(currentPose.t.z()));
}

// ============================================================================
// 3. Pressure Transducer Depth Factor Constraints
// ============================================================================

TEST_F(BackendTest, DepthFactorConstrainsVerticalPositionToTransducerReadings) {
    // 1. Verify DepthFactor error and analytical Jacobian formulation
    const gtsam::SharedNoiseModel baseNoise = gtsam::noiseModel::Isotropic::Sigma(1, 0.05);
    const gtsam::Point3 leverArm(0.0, 0.0, 0.20);
    DepthFactor factor(X(1), 5.20, baseNoise, leverArm);

    // Body pose at z = 5.00 -> sensor in world frame at z = 5.20 -> error = 0.0
    const gtsam::Pose3 testPose(gtsam::Rot3(), gtsam::Point3(0.0, 0.0, 5.00));
    gtsam::Matrix H;
    const gtsam::Vector err = factor.evaluateError(testPose, &H);
    EXPECT_NEAR(err(0), 0.0, 1e-6);
    EXPECT_EQ(H.rows(), 1);
    EXPECT_EQ(H.cols(), 6);
    EXPECT_NEAR(H(0, 5), 1.0, 1e-6);

    // 2. Verify streaming depth factor integration across multiple frames
    synchronizer->start();
    frontend->start();
    backend->start();

    // Frame 0: Initial depth 3.0m
    depthStream.pushDepth({1.000, 3.0});
    depthStream.pushDepth({1.050, 3.0});
    ASSERT_TRUE(imageStream.pushFrame(baseSceneBuffer.data(), 1.020));
    ASSERT_TRUE(waitUntilProcessedFrame(0u, 1.020));

    Pose p0 = backend->getCurrentPose();
    EXPECT_NEAR(p0.t.z(), 3.0, 0.1);

    // Frame 1: Consistent inertial and transducer readings at 3.0m
    test_helpers::pushBackendInertialAndDepthWindow(imuStream, depthStream, 1.020, 1.100, 3.0);
    ASSERT_TRUE(imageStream.pushFrame(baseSceneBuffer.data(), 1.060));
    ASSERT_TRUE(waitUntilProcessedFrame(1u, 1.060));

    const gtsam::Values &est = getIsamEstimate(*backend);
    ASSERT_TRUE(est.exists(X(1)));
    const gtsam::Pose3 pose1 = est.at<gtsam::Pose3>(X(1));
    EXPECT_NEAR(pose1.z(), 3.0, 0.2);
    EXPECT_NEAR(backend->getCurrentPose().t.z(), 3.0, 0.2);
}

// ============================================================================
// 4. Lever Arm Offset Applied to Depth Factor
// ============================================================================

TEST_F(BackendTest, DepthSensorLeverArmOffsetCompensatesInitialBodyPose) {
    // Reconfigure backend with a depth sensor lever arm offset of 0.40m along body Z
    const gtsam::Point3 leverArm(0.0, 0.0, 0.40);
    BackendConfig customCfg = test_helpers::makeDefaultBackendConfig(gtsam::Pose3(), leverArm);
    auto customBackend = std::make_unique<Backend>(*frontend, frontend->getLfTracker(), customCfg);

    synchronizer->start();
    frontend->start();
    customBackend->start();

    // Measured depth = 8.40m -> body pose Z should be 8.40 - 0.40 = 8.00m
    depthStream.pushDepth({1.000, 8.40});
    depthStream.pushDepth({1.050, 8.40});
    ASSERT_TRUE(imageStream.pushFrame(baseSceneBuffer.data(), 1.020));

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(2000);
    while (std::chrono::steady_clock::now() < deadline) {
        if (customBackend->getCurrentPose().timestamp > 0.0) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    Pose p0 = customBackend->getCurrentPose();
    EXPECT_DOUBLE_EQ(p0.timestamp, 1.020);
    EXPECT_NEAR(p0.t.z(), 8.00, 0.1);

    customBackend->end();
}

// ============================================================================
// 5. Visual Landmark Triangulation via Parallax Baseline
// ============================================================================

TEST_F(BackendTest, MultiViewVisualLandmarksTriangulateIntoIsamEstimate) {
    // Direct multi-view triangulation test through LandmarkBookkeeper into iSAM2
    // Landmark in world frame at P_world = (0.40, 0.20, 3.00)
    // Frame 0: Body at (0, 0, 1.00) -> Camera at (0, 0, 1.00)
    // Landmark in Cam 0 frame: (0.40, 0.20, 2.00) -> ray: (0.4/2, 0.2/2) = (0.20, 0.10)
    BackendHfPacket pkt0;
    pkt0.frameId = 0;
    pkt0.timestamp = 1.000;
    pkt0.isKeyframe = true;
    pkt0.depth = 1.0;
    pkt0.visualMeasurements.emplace_back(101u, gtsam::Point2(0.20, 0.10));

    processSingleHfPacket(*backend, pkt0);
    optimizeAndPublish(*backend, 0u, 1.000, false);

    EXPECT_TRUE(isBackendInitialized(*backend));

    // Frame 1: Translate body along X by 0.10m via preintegrated IMU delta
    // Baseline = 0.10m (> minBaselineMeters = 0.03m)
    // Body at (0.10, 0, 1.00) -> Camera at (0.10, 0, 1.00)
    // Landmark in Cam 1 frame: (0.40 - 0.10, 0.20, 2.00) = (0.30, 0.20, 2.00)
    // Ray in Cam 1: (0.30/2, 0.20/2) = (0.15, 0.10)
    BackendHfPacket pkt1;
    pkt1.frameId = 1;
    pkt1.timestamp = 1.100;
    pkt1.isKeyframe = true;
    pkt1.depth = 1.0;
    pkt1.visualMeasurements.emplace_back(101u, gtsam::Point2(0.15, 0.10));

    auto preint = std::make_unique<gtsam::PreintegratedImuMeasurements>(
        frontendConfig.preintParams, gtsam::imuBias::ConstantBias());
    // Integrate constant ax = 20.0 m/s^2 over dt = 0.1s -> dx = 0.5 * 20 * 0.01 = 0.10m
    // az = 9.81 m/s^2 counters ENU gravity
    preint->integrateMeasurement(Eigen::Vector3d(20.0, 0.0, 9.81), Eigen::Vector3d::Zero(), 0.10);
    pkt1.imuDelta = std::move(preint);

    processSingleHfPacket(*backend, pkt1);
    optimizeAndPublish(*backend, 1u, 1.100, false);

    // Inspect triangulated landmarks in iSAM2
    const gtsam::Values &est = getIsamEstimate(*backend);
    ASSERT_TRUE(est.exists(L(101))) << "Landmark 101 should be triangulated and present in iSAM2 estimate";

    const gtsam::Point3 triangulatedLm = est.at<gtsam::Point3>(L(101));
    EXPECT_TRUE(std::isfinite(triangulatedLm.x()));
    EXPECT_TRUE(std::isfinite(triangulatedLm.y()));
    EXPECT_TRUE(std::isfinite(triangulatedLm.z()));

    // World position: Camera Z was 1.0, depth was 2.0 -> Z_world = 3.0
    EXPECT_NEAR(triangulatedLm.x(), 0.40, 0.15);
    EXPECT_NEAR(triangulatedLm.y(), 0.20, 0.15);
    EXPECT_NEAR(triangulatedLm.z(), 3.00, 0.15);
}

// ============================================================================
// 6. Non-Contiguous Frame Drop Fallback Motion Model
// ============================================================================

TEST_F(BackendTest, DroppedIntermediateFramesEngageFallbackBetweenFactors) {
    // Test standalone Backend packet processing directly
    BackendHfPacket pkt0;
    pkt0.frameId = 0;
    pkt0.timestamp = 1.000;
    pkt0.isKeyframe = true;
    pkt0.depth = 5.0;
    processSingleHfPacket(*backend, pkt0);
    optimizeAndPublish(*backend, 0u, 1.000, false);

    EXPECT_TRUE(isBackendInitialized(*backend));
    EXPECT_EQ(getPrevFrameId(*backend), 0u);

    // Frame 5 arrives directly (frames 1..4 dropped/evicted, no imuDelta)
    BackendHfPacket pkt5;
    pkt5.frameId = 5;
    pkt5.timestamp = 1.250;
    pkt5.isKeyframe = true;
    pkt5.depth = 5.0;
    pkt5.imuDelta = nullptr; // Missing IMU preintegration forces fallback

    EXPECT_NO_THROW({
        processSingleHfPacket(*backend, pkt5);
        optimizeAndPublish(*backend, 5u, 1.250, false);
    });

    const gtsam::Values &est = getIsamEstimate(*backend);
    ASSERT_TRUE(est.exists(X(5)));
    ASSERT_TRUE(est.exists(V(5)));
    ASSERT_TRUE(est.exists(B(5)));

    const gtsam::Pose3 pose5 = est.at<gtsam::Pose3>(X(5));
    EXPECT_TRUE(std::isfinite(pose5.x()));
    EXPECT_TRUE(std::isfinite(pose5.y()));
    EXPECT_TRUE(std::isfinite(pose5.z()));
    EXPECT_NEAR(pose5.z(), 5.0, 0.2);
}

// ============================================================================
// 7. Loop Closure Factor Ingestion and State Optimization
// ============================================================================

TEST_F(BackendTest, LoopClosurePacketIntegratesBetweenFactorAndRefinesEstimate) {
    // Initialize Frame 0 and Frame 1
    BackendHfPacket pkt0;
    pkt0.frameId = 0;
    pkt0.timestamp = 1.000;
    pkt0.isKeyframe = true;
    pkt0.depth = 4.0;
    processSingleHfPacket(*backend, pkt0);
    optimizeAndPublish(*backend, 0u, 1.000, false);

    BackendHfPacket pkt1;
    pkt1.frameId = 1;
    pkt1.timestamp = 1.100;
    pkt1.isKeyframe = true;
    pkt1.depth = 4.0;
    processSingleHfPacket(*backend, pkt1);
    optimizeAndPublish(*backend, 1u, 1.100, false);

    // Inject loop closure between Frame 1 and Frame 0
    const gtsam::Pose3 relativePose(gtsam::Rot3(), gtsam::Point3(0.05, 0.0, 0.0));
    BackendLfPacket loopPkt = test_helpers::createSyntheticLoopPacket(
        1u, 0u, 1.100, relativePose, 30
    );

    EXPECT_NO_THROW({
        injectLfPacket(*backend, loopPkt);
        optimizeAndPublish(*backend, 1u, 1.100, true);
    });

    const gtsam::Values &est = getIsamEstimate(*backend);
    ASSERT_TRUE(est.exists(X(0)));
    ASSERT_TRUE(est.exists(X(1)));

    const gtsam::Pose3 pose0 = est.at<gtsam::Pose3>(X(0));
    const gtsam::Pose3 pose1 = est.at<gtsam::Pose3>(X(1));
    EXPECT_TRUE(std::isfinite(pose0.x()));
    EXPECT_TRUE(std::isfinite(pose1.x()));
}

// ============================================================================
// 8. Keyframe Sliding-Window Eviction (Item 6 Verification)
// ============================================================================

TEST_F(BackendTest, SlidingWindowEvictsExpiredKeyframesBeyondBufferHorizon) {
    // Initialize Frame 0
    BackendHfPacket pkt0;
    pkt0.frameId = 0;
    pkt0.timestamp = 1.000;
    pkt0.isKeyframe = true;
    pkt0.depth = 3.0;
    processSingleHfPacket(*backend, pkt0);
    optimizeAndPublish(*backend, 0u, 1.000, false);

    // Stream 25 keyframes sequentially past maxBufferLength (15)
    constexpr uint32_t kTotalKeyframes = 25;
    for (uint32_t f = 1; f <= kTotalKeyframes; f++) {
        BackendHfPacket kfPkt;
        kfPkt.frameId = f;
        kfPkt.timestamp = 1.000 + static_cast<double>(f) * 0.050;
        kfPkt.isKeyframe = true;
        kfPkt.depth = 3.0;
        // Visual feature observation that is not yet triangulated
        kfPkt.visualMeasurements.emplace_back(100 + f, gtsam::Point2(0.1, 0.1));

        processSingleHfPacket(*backend, kfPkt);
        optimizeAndPublish(*backend, f, kfPkt.timestamp, false);
    }

    // Verify trackedKeyframes does not grow unbounded and oldest entries have been evicted
    const auto &tracked = getTrackedKeyframes(*backend);
    EXPECT_LE(tracked.size(), backendConfig.bkConfig.maxBufferLength + 2);
    if (!tracked.empty()) {
        EXPECT_GT(tracked.front().frameId, 0u);
    }
}

// ============================================================================
// 9. Lifecycle Start, End, and Clean Thread Shutdown
// ============================================================================

TEST_F(BackendTest, LifecycleStartAndEndShutsDownWorkerThreadCleanly) {
    synchronizer->start();
    frontend->start();
    backend->start();

    depthStream.pushDepth({1.000, 5.0});
    depthStream.pushDepth({1.050, 5.0});
    ASSERT_TRUE(imageStream.pushFrame(baseSceneBuffer.data(), 1.020));

    ASSERT_TRUE(waitUntilProcessedFrame(0u, 1.020));

    // End backend and verify clean shutdown without deadlocks or hangs
    backend->end();
    frontend->end();
    synchronizer->stop();

    Pose finalPose = backend->getCurrentPose();
    EXPECT_DOUBLE_EQ(finalPose.timestamp, 1.020);
    EXPECT_NEAR(finalPose.t.z(), 5.0, 0.1);
}

} // namespace PoolSLAM
