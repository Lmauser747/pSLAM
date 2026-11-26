#include "TestBackendHelpers.hpp"

using gtsam::Point2;
using gtsam::Point3;
using gtsam::Pose3;
using gtsam::Rot3;
using gtsam::Vector3;
using gtsam::symbol_shorthand::B;
using gtsam::symbol_shorthand::V;
using gtsam::symbol_shorthand::X;

namespace PoolSLAM {

// =========================================================================
// 1. Lifecycle & Thread Concurrency
// =========================================================================

TEST_F(BackendTest, StartLaunchesThreadAndIsIdempotent) {
    TestHarness harness;
    EXPECT_FALSE(isRunning(harness.backend));

    harness.backend.start();
    EXPECT_TRUE(isRunning(harness.backend));

    // Repeated call is safe and idempotent
    harness.backend.start();
    EXPECT_TRUE(isRunning(harness.backend));

    harness.backend.end();
    EXPECT_FALSE(isRunning(harness.backend));
}

TEST_F(BackendTest, EndSafelyTerminatesAndDestructorClean) {
    TestHarness harness;
    harness.backend.start();
    EXPECT_TRUE(isRunning(harness.backend));

    harness.backend.end();
    EXPECT_FALSE(isRunning(harness.backend));

    // Calling end() again is a safe no-op
    harness.backend.end();
    EXPECT_FALSE(isRunning(harness.backend));

    // Test destructor joins running thread cleanly without hang
    {
        TestHarness harnessScoped;
        harnessScoped.backend.start();
        EXPECT_TRUE(isRunning(harnessScoped.backend));
    }
}

TEST_F(BackendTest, LockFreeStatePublicationConcurrency) {
    TestHarness harness;
    std::atomic<bool> stopReader{false};
    std::atomic<size_t> readCount{0};

    // Producer publishes 2,000 pose updates
    std::thread producer([&]() {
        for (int i = 0; i < 2000; ++i) {
            const double t = static_cast<double>(i) * 0.005;
            const Pose3 pose(Rot3::Yaw(0.01 * static_cast<double>(i)),
                             Point3(t, 2.0 * t, 3.0 * t));
            const Vector3 vel(1.0, 2.0, 3.0);
            callPublishPose(harness.backend, pose, vel, t);
            std::this_thread::yield();
        }
        stopReader.store(true);
    });

    // Consumer reads poses lock-free in parallel
    std::thread consumer([&]() {
        while (!stopReader.load()) {
            const Pose p = harness.backend.getCurrentPose();
            EXPECT_NEAR(p.r.norm(), 1.0, 1e-4);
            ++readCount;
            std::this_thread::yield();
        }
    });

    producer.join();
    consumer.join();

    EXPECT_GT(readCount.load(), 0u);
}

// =========================================================================
// 2. Bootstrapping & Gravity Alignment (Frame 0)
// =========================================================================

TEST_F(BackendTest, BootstrappingDepthLeverArmCompensation) {
    BackendConfig cfg = makeDefaultBackendConfig();
    cfg.bodyToDepthSensor = Point3(0.2, -0.3, 0.7);
    TestHarness harness(cfg);

    BackendHfPacket pkt0;
    pkt0.frameId = 0;
    pkt0.timestamp = 100.0;
    pkt0.depth = 12.7;
    pkt0.initialAccel = Vector3(0.0, 0.0, 9.81);

    callInitializeSystem(harness.backend, pkt0);

    EXPECT_TRUE(isInitialized(harness.backend));
    ASSERT_TRUE(getSmootherEstimate(harness.backend).exists(X(0)));

    const Pose3 pose0 = getSmootherEstimate(harness.backend).at<Pose3>(X(0));
    // Lever arm compensation: z_0 = z_meas - l_z = 12.7 - 0.7 = 12.0 m
    EXPECT_NEAR(pose0.z(), 12.0, 1e-2);
}

TEST_F(BackendTest, BootstrappingGravityAlignment) {
    TestHarness harness;

    // Stationary vehicle with accelerometer reading gravity
    BackendHfPacket pkt0;
    pkt0.frameId = 0;
    pkt0.timestamp = 0.0;
    pkt0.depth = 5.0;
    pkt0.initialAccel = Vector3(0.0, 0.0, 9.81);

    callInitializeSystem(harness.backend, pkt0);
    EXPECT_TRUE(isInitialized(harness.backend));

    const Pose3 pose0 = getSmootherEstimate(harness.backend).at<Pose3>(X(0));
    EXPECT_NEAR(pose0.z(), 5.0, 1e-2);

    // Initial velocity must be zero
    ASSERT_TRUE(getSmootherEstimate(harness.backend).exists(V(0)));
    const Vector3 v0 = getSmootherEstimate(harness.backend).at<Vector3>(V(0));
    EXPECT_NEAR(v0.norm(), 0.0, 1e-6);
}

TEST_F(BackendTest, BootstrappingSmallAccelDefaultsToIdentityAttitude) {
    TestHarness harness;

    // Acceleration norm < 1e-3 defaults to identity rotation
    BackendHfPacket pkt0;
    pkt0.frameId = 0;
    pkt0.timestamp = 0.0;
    pkt0.depth = 4.0;
    pkt0.initialAccel = Vector3(1e-5, 0.0, 0.0);

    callInitializeSystem(harness.backend, pkt0);
    EXPECT_TRUE(isInitialized(harness.backend));

    const Pose3 pose0 = getSmootherEstimate(harness.backend).at<Pose3>(X(0));
    EXPECT_LT((pose0.rotation().matrix() - Rot3::Identity().matrix()).cwiseAbs().maxCoeff(), 1e-4);
}

TEST_F(BackendTest, BootstrappingGyroBiasInitialization) {
    TestHarness harness;

    BackendHfPacket pkt0;
    pkt0.frameId = 0;
    pkt0.timestamp = 0.0;
    pkt0.depth = 3.0;
    pkt0.initialGyro = Vector3(0.02, -0.015, 0.04);

    callInitializeSystem(harness.backend, pkt0);
    EXPECT_TRUE(isInitialized(harness.backend));

    ASSERT_TRUE(getSmootherEstimate(harness.backend).exists(B(0)));
    const auto bias0 = getSmootherEstimate(harness.backend).at<gtsam::imuBias::ConstantBias>(B(0));

    EXPECT_NEAR(bias0.gyroscope().x(), 0.02, 1e-4);
    EXPECT_NEAR(bias0.gyroscope().y(), -0.015, 1e-4);
    EXPECT_NEAR(bias0.gyroscope().z(), 0.04, 1e-4);
    EXPECT_NEAR(bias0.accelerometer().norm(), 0.0, 1e-4);
}

TEST_F(BackendTest, BootstrappingMissingDepthFallback) {
    TestHarness harness;

    BackendHfPacket pkt0;
    pkt0.frameId = 0;
    pkt0.timestamp = 0.0;
    pkt0.depth = std::nullopt; // No depth measurement available

    callInitializeSystem(harness.backend, pkt0);
    EXPECT_TRUE(isInitialized(harness.backend));

    const Pose3 pose0 = getSmootherEstimate(harness.backend).at<Pose3>(X(0));
    EXPECT_NEAR(pose0.z(), 0.0, 1e-2);
}

// =========================================================================
// 3. High-Frequency Dead-Reckoning Navigation
// =========================================================================

TEST_F(BackendTest, DeadReckoningPropagatesNavState) {
    TestHarness harness;

    // Bootstrap at frame 0
    BackendHfPacket pkt0;
    pkt0.frameId = 0;
    pkt0.timestamp = 10.0;
    pkt0.depth = 5.0;
    callInitializeSystem(harness.backend, pkt0);

    // Non-keyframe packet with forward acceleration along X
    auto pkt1 = std::make_unique<BackendHfPacket>();
    pkt1->frameId = 1;
    pkt1->timestamp = 10.1;
    pkt1->isKeyframe = false;
    pkt1->imuDelta = makePim(Vector3(2.0, 0.0, 9.81), Vector3::Zero(), 0.1);

    TestFrontend::pushHfPacket(harness.frontend, std::move(pkt1));

    harness.backend.start();
    // Allow worker thread to consume non-keyframe packet
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    harness.backend.end();

    const Pose pose = harness.backend.getCurrentPose();
    EXPECT_GE(pose.timestamp, 10.0);
}

TEST_F(BackendTest, MonotonicityDropOutOfOrderPacket) {
    TestHarness harness;

    BackendHfPacket pkt0;
    pkt0.frameId = 5;
    pkt0.timestamp = 10.5;
    pkt0.depth = 5.0;
    callInitializeSystem(harness.backend, pkt0);
    setPrevFrameId(harness.backend, 5);

    // Out-of-order packet with frameId <= prevFrameId
    auto pktStale = std::make_unique<BackendHfPacket>();
    pktStale->frameId = 4;
    pktStale->timestamp = 10.4;
    pktStale->isKeyframe = false;

    TestFrontend::pushHfPacket(harness.frontend, std::move(pktStale));

    harness.backend.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    harness.backend.end();

    // Previous frame ID remains unchanged
    EXPECT_EQ(getPrevFrameId(harness.backend), 5u);
}

// =========================================================================
// 4. Keyframe Factor Graph Assembly & iSAM2 Solving
// =========================================================================

TEST_F(BackendTest, KeyframeFactorGraphAssembly) {
    TestHarness harness;

    // Frame 0 bootstrap
    BackendHfPacket pkt0;
    pkt0.frameId = 0;
    pkt0.timestamp = 0.0;
    pkt0.depth = 5.0;
    callInitializeSystem(harness.backend, pkt0);

    // Frame 1 keyframe
    BackendHfPacket pkt1;
    pkt1.frameId = 1;
    pkt1.timestamp = 0.5;
    pkt1.isKeyframe = true;
    pkt1.depth = 5.2;
    pkt1.imuDelta = makePim(Vector3(0.0, 0.0, 9.81), Vector3::Zero(), 0.5);

    callProcessFrame(harness.backend, pkt1);

    // Verify X(1), V(1), B(1) are added to estimate
    EXPECT_TRUE(getSmootherEstimate(harness.backend).exists(X(1)));
    EXPECT_TRUE(getSmootherEstimate(harness.backend).exists(V(1)));
    EXPECT_TRUE(getSmootherEstimate(harness.backend).exists(B(1)));

    // Verify depth converged near measured depth
    const Pose3 pose1 = getSmootherEstimate(harness.backend).at<Pose3>(X(1));
    EXPECT_NEAR(pose1.z(), 5.2, 0.2);
}

TEST_F(BackendTest, BiasFeedbackDeliveredToFrontend) {
    TestHarness harness;

    BackendHfPacket pkt0;
    pkt0.frameId = 0;
    pkt0.timestamp = 0.0;
    pkt0.depth = 5.0;
    pkt0.initialGyro = Vector3(0.01, -0.01, 0.02);
    callInitializeSystem(harness.backend, pkt0);

    // Verify bias was sent to Frontend
    const auto feBias = TestFrontend::getImuBias(harness.frontend);
    EXPECT_NEAR(feBias.gyroscope().x(), 0.01, 1e-4);
    EXPECT_NEAR(feBias.gyroscope().y(), -0.01, 1e-4);
    EXPECT_NEAR(feBias.gyroscope().z(), 0.02, 1e-4);
}

// =========================================================================
// 5. Smart Projection Factor Lifecycle
// =========================================================================

TEST_F(BackendTest, SmartProjectionFactorSingleViewDeferred) {
    TestHarness harness;

    BackendHfPacket pkt0;
    pkt0.frameId = 0;
    pkt0.timestamp = 0.0;
    pkt0.depth = 5.0;
    callInitializeSystem(harness.backend, pkt0);

    BackendHfPacket pkt1;
    pkt1.frameId = 1;
    pkt1.timestamp = 0.5;
    pkt1.isKeyframe = true;
    pkt1.depth = 5.0;
    pkt1.imuDelta = makePim(Vector3(0.0, 0.0, 9.81), Vector3::Zero(), 0.5);
    pkt1.visualFeatures.emplace_back(101, Point2(0.1, 0.2), true);

    callProcessFrame(harness.backend, pkt1);

    // Visual track registered in activeFeatureTracks but factor deferred (needs >= 2 views)
    EXPECT_EQ(getActiveFeatureTracks(harness.backend).count(101), 1u);
    EXPECT_EQ(getActiveFeatureTracks(harness.backend).at(101).size(), 1u);
    EXPECT_EQ(getSmartFactorSlots(harness.backend).count(101), 0u);
}

TEST_F(BackendTest, SmartProjectionFactorMultiViewUpdateAndSlotReplacement) {
    TestHarness harness;

    BackendHfPacket pkt0;
    pkt0.frameId = 0;
    pkt0.timestamp = 0.0;
    pkt0.depth = 5.0;
    callInitializeSystem(harness.backend, pkt0);

    // View 1 (Keyframe 1)
    BackendHfPacket pkt1;
    pkt1.frameId = 1;
    pkt1.timestamp = 0.5;
    pkt1.isKeyframe = true;
    pkt1.depth = 5.0;
    pkt1.imuDelta = makePim(Vector3(0.0, 0.0, 9.81), Vector3::Zero(), 0.5);
    pkt1.visualFeatures.emplace_back(101, Point2(0.1, 0.2), true);
    callProcessFrame(harness.backend, pkt1);
    EXPECT_EQ(getSmartFactorSlots(harness.backend).count(101), 0u);

    // View 2 (Keyframe 2) - triggers addition to smoother
    BackendHfPacket pkt2;
    pkt2.frameId = 2;
    pkt2.timestamp = 1.0;
    pkt2.isKeyframe = true;
    pkt2.depth = 5.0;
    pkt2.imuDelta = makePim(Vector3(0.0, 0.0, 9.81), Vector3::Zero(), 0.5);
    pkt2.visualFeatures.emplace_back(101, Point2(0.08, 0.2), false);
    callProcessFrame(harness.backend, pkt2);
    EXPECT_EQ(getSmartFactorSlots(harness.backend).count(101), 1u);

    // View 3 (Keyframe 3) - triggers slot replacement
    BackendHfPacket pkt3;
    pkt3.frameId = 3;
    pkt3.timestamp = 1.5;
    pkt3.isKeyframe = true;
    pkt3.depth = 5.0;
    pkt3.imuDelta = makePim(Vector3(0.0, 0.0, 9.81), Vector3::Zero(), 0.5);
    pkt3.visualFeatures.emplace_back(101, Point2(0.06, 0.2), false);
    callProcessFrame(harness.backend, pkt3);

    EXPECT_EQ(getSmartFactorSlots(harness.backend).count(101), 1u);
    EXPECT_TRUE(getSmootherEstimate(harness.backend).exists(X(3)));
}

// =========================================================================
// 6. Loop Closure BetweenFactor Integration
// =========================================================================

TEST_F(BackendTest, LoopClosureBetweenFactorTransformation) {
    BackendConfig cfg = makeDefaultBackendConfig();
    // Extrinsic transform: camera rotated 90 deg around Z relative to body
    cfg.bodyToCameraFrame = Pose3(Rot3::Yaw(M_PI / 2.0), Point3(0.1, 0.0, 0.0));
    TestHarness harness(cfg);

    BackendHfPacket pkt0;
    pkt0.frameId = 0;
    pkt0.timestamp = 0.0;
    pkt0.depth = 5.0;
    callInitializeSystem(harness.backend, pkt0);

    BackendHfPacket pkt1;
    pkt1.frameId = 1;
    pkt1.timestamp = 1.0;
    pkt1.isKeyframe = true;
    pkt1.depth = 5.0;
    pkt1.imuDelta = makePim(Vector3(0.0, 0.0, 9.81), Vector3::Zero(), 1.0);
    callProcessFrame(harness.backend, pkt1);

    BackendHfPacket pkt2;
    pkt2.frameId = 2;
    pkt2.timestamp = 2.0;
    pkt2.isKeyframe = true;
    pkt2.depth = 5.0;
    pkt2.imuDelta = makePim(Vector3(0.0, 0.0, 9.81), Vector3::Zero(), 1.0);
    callProcessFrame(harness.backend, pkt2);

    // Create loop closure match between frame 1 and frame 2
    BackendLfPacket lfPkt;
    lfPkt.matchedFrameId = 1;
    lfPkt.currentFrameId = 2;
    lfPkt.relativePose = Pose3(Rot3::Identity(), Point3(0.5, 0.0, 0.0));
    lfPkt.noiseModel = gtsam::noiseModel::Isotropic::Sigma(6, 0.05);

    callProcessLfPacket(harness.backend, lfPkt);

    // Verify BetweenFactor was inserted into global pose graph
    const auto& graph = getGlobalPoseGraph(harness.backend);
    bool foundLoopFactor = false;
    for (const auto& factor : graph) {
        if (auto bf = std::dynamic_pointer_cast<gtsam::BetweenFactor<Pose3>>(factor)) {
            if (bf->key1() == X(1) && bf->key2() == X(2)) {
                foundLoopFactor = true;
                break;
            }
        }
    }
    EXPECT_TRUE(foundLoopFactor);
}

// =========================================================================
// 7. Landmark Synchronization & Pruning
// =========================================================================

TEST_F(BackendTest, LandmarkSyncPushesToLfTracker) {
    TestHarness harness;

    BackendHfPacket pkt0;
    pkt0.frameId = 0;
    pkt0.timestamp = 0.0;
    pkt0.depth = 5.0;
    callInitializeSystem(harness.backend, pkt0);

    BackendHfPacket pkt1;
    pkt1.frameId = 1;
    pkt1.timestamp = 0.5;
    pkt1.isKeyframe = true;
    pkt1.depth = 5.0;
    pkt1.imuDelta = makePim(Vector3(0.0, 0.0, 9.81), Vector3::Zero(), 0.5);
    callProcessFrame(harness.backend, pkt1);

    // Call sync without throwing
    EXPECT_NO_THROW(callSyncLfTracker(harness.backend, true));
}

TEST_F(BackendTest, TrackedKeyframesSlidingWindowCapacity) {
    TestHarness harness;

    BackendHfPacket pkt0;
    pkt0.frameId = 0;
    pkt0.timestamp = 0.0;
    pkt0.depth = 5.0;
    callInitializeSystem(harness.backend, pkt0);

    // Ingest 25 keyframes sequentially
    for (uint32_t i = 1; i <= 25; ++i) {
        BackendHfPacket pkt;
        pkt.frameId = i;
        pkt.timestamp = static_cast<double>(i) * 0.5;
        pkt.isKeyframe = true;
        pkt.depth = 5.0;
        pkt.imuDelta = makePim(Vector3(0.0, 0.0, 9.81), Vector3::Zero(), 0.5);
        callProcessFrame(harness.backend, pkt);
    }

    // Capacity must not exceed keyframeWindowSize (20)
    EXPECT_LE(getTrackedKeyframes(harness.backend).size(), 20u);
}

// =========================================================================
// 8. Numerical Stability & Linear System Exception Safety
// =========================================================================

TEST_F(BackendTest, IndeterminantLinearSystemExceptionHandling) {
    TestHarness harness;

    BackendHfPacket pkt0;
    pkt0.frameId = 0;
    pkt0.timestamp = 0.0;
    pkt0.depth = 5.0;
    callInitializeSystem(harness.backend, pkt0);

    // Add disconnected, unanchored variables with gauge freedom (causes IndeterminantLinearSystemException)
    getStagingValues(harness.backend).insert(X(100), Pose3::Identity());
    getStagingValues(harness.backend).insert(X(101), Pose3(Rot3::Identity(), Point3(1.0, 0.0, 0.0)));
    getStagingGraph(harness.backend).emplace_shared<gtsam::BetweenFactor<Pose3>>(
        X(100), X(101), Pose3::Identity(), gtsam::noiseModel::Isotropic::Sigma(6, 1.0));

    // Solving smoother must catch IndeterminantLinearSystemException, clear staging, and return false
    const bool success = callSolveSmoother(harness.backend, 100, 1.0);
    EXPECT_FALSE(success);
    EXPECT_TRUE(getStagingGraph(harness.backend).empty());
    EXPECT_TRUE(getStagingValues(harness.backend).empty());
}

TEST_F(BackendTest, NonKeyframeMarginSecondNewCollapsesFrame) {
    TestHarness harness;

    BackendHfPacket pkt0;
    pkt0.frameId = 0;
    pkt0.timestamp = 0.0;
    pkt0.isKeyframe = true;
    pkt0.depth = 5.0;
    callInitializeSystem(harness.backend, pkt0);
    ASSERT_EQ(getWindow(harness.backend).size(), 1u);

    auto makeImuBatch = [](double tStart, double tEnd) {
        std::vector<Imu> samples;
        for (double t = tStart + 0.005; t <= tEnd + 1e-9; t += 0.005) {
            Imu s;
            s.timestamp = t;
            s.a = Vector3(0.0, 0.0, 9.81);
            s.w = Vector3::Zero();
            s.t = 20.0f;
            samples.push_back(s);
        }
        return samples;
    };

    // Frame 1: non-keyframe
    BackendHfPacket pkt1;
    pkt1.frameId = 1;
    pkt1.timestamp = 0.05;
    pkt1.isKeyframe = false;
    pkt1.depth = 5.0;
    pkt1.imuDelta = makePim(Vector3(0.0, 0.0, 9.81), Vector3::Zero(), 0.05);
    pkt1.rawImu = makeImuBatch(0.0, 0.05);
    pkt1.visualFeatures.emplace_back(201, Point2(0.0, 0.0), true);
    callProcessFrame(harness.backend, pkt1);

    EXPECT_EQ(getWindow(harness.backend).size(), 2u);
    EXPECT_TRUE(getSmoother(harness.backend).valueExists(X(1)));

    // Frame 2: non-keyframe collapses frame 1 via MARGIN_SECOND_NEW
    BackendHfPacket pkt2;
    pkt2.frameId = 2;
    pkt2.timestamp = 0.10;
    pkt2.isKeyframe = false;
    pkt2.depth = 5.0;
    pkt2.imuDelta = makePim(Vector3(0.0, 0.0, 9.81), Vector3::Zero(), 0.05);
    pkt2.rawImu = makeImuBatch(0.05, 0.10);
    pkt2.visualFeatures.emplace_back(201, Point2(0.0, 0.0), false);
    callProcessFrame(harness.backend, pkt2);

    const auto& win = getWindow(harness.backend);
    ASSERT_EQ(win.size(), 2u);
    EXPECT_EQ(win.front().frameId, 0u);
    EXPECT_EQ(win.back().frameId, 2u);
    EXPECT_FALSE(getSmoother(harness.backend).valueExists(X(1)));
    EXPECT_TRUE(getSmoother(harness.backend).valueExists(X(2)));
}

TEST_F(BackendTest, OldestKeyframeMarginOldCapsWindowSize) {
    BackendConfig cfg = makeDefaultBackendConfig();
    cfg.slidingWindowSize = 5;
    TestHarness harness(cfg);

    BackendHfPacket pkt0;
    pkt0.frameId = 0;
    pkt0.timestamp = 0.0;
    pkt0.isKeyframe = true;
    pkt0.depth = 5.0;
    pkt0.visualFeatures.emplace_back(301, Point2(-0.1, 0.0), true);
    callInitializeSystem(harness.backend, pkt0);

    for (uint32_t i = 1; i <= 8; ++i) {
        BackendHfPacket pkt;
        pkt.frameId = i;
        pkt.timestamp = static_cast<double>(i) * 0.2;
        pkt.isKeyframe = true;
        pkt.depth = 5.0;
        pkt.imuDelta = makePim(Vector3(0.0, 0.0, 9.81), Vector3::Zero(), 0.2);
        pkt.visualFeatures.emplace_back(301, Point2(-0.1 + 0.02 * static_cast<double>(i), 0.0), false);
        callProcessFrame(harness.backend, pkt);
    }

    const auto& win = getWindow(harness.backend);
    EXPECT_EQ(win.size(), cfg.slidingWindowSize);
    EXPECT_EQ(win.front().frameId, 4u);
    EXPECT_EQ(win.back().frameId, 8u);
    EXPECT_FALSE(getSmoother(harness.backend).valueExists(X(0)));
    EXPECT_FALSE(getSmoother(harness.backend).valueExists(X(3)));
    EXPECT_TRUE(getSmoother(harness.backend).valueExists(X(4)));
    EXPECT_TRUE(getSmoother(harness.backend).valueExists(X(8)));
    EXPECT_EQ(getSmartFactorSlots(harness.backend).count(301), 1u);
}

TEST_F(BackendTest, MarginOldPreservesSuccessorFactorsAcrossBatchRelinearization) {
    BackendConfig cfg = makeDefaultBackendConfig();
    cfg.slidingWindowSize = 5;
    cfg.relinearizeThreshold = 0.01;
    cfg.relinearizeSkip = 1;
    TestHarness harness(cfg);

    auto makeImuBatch = [](double tStart, double tEnd, const Vector3& accel) {
        std::vector<Imu> samples;
        for (double t = tStart + 0.005; t <= tEnd + 1e-9; t += 0.005) {
            Imu s;
            s.timestamp = t;
            s.a = accel;
            s.w = Vector3::Zero();
            s.t = 20.0f;
            samples.push_back(s);
        }
        return samples;
    };

    BackendHfPacket pkt0;
    pkt0.frameId = 0;
    pkt0.timestamp = 0.0;
    pkt0.isKeyframe = true;
    pkt0.depth = 5.0;
    pkt0.visualFeatures.emplace_back(401, Point2(0.0, 0.0), true);
    pkt0.visualFeatures.emplace_back(402, Point2(0.1, -0.05), true);
    callInitializeSystem(harness.backend, pkt0);

    // Interleave keyframes and non-keyframes with slight acceleration to force batch relinearization
    uint32_t fid = 1;
    double ts = 0.0;
    for (int step = 0; step < 12; ++step) {
        for (int sub = 0; sub < 2; ++sub) {
            const bool isKf = (sub == 1);
            const Vector3 accel(0.2 * std::sin(0.5 * step), 0.1 * std::cos(0.5 * step), 9.81);
            BackendHfPacket pkt;
            pkt.frameId = fid;
            ts += 0.05;
            pkt.timestamp = ts;
            pkt.isKeyframe = isKf;
            pkt.depth = 5.0;
            pkt.imuDelta = makePim(accel, Vector3::Zero(), 0.05);
            pkt.rawImu = makeImuBatch(ts - 0.05, ts, accel);
            const double shift = 0.01 * static_cast<double>(fid);
            pkt.visualFeatures.emplace_back(401, Point2(shift, 0.0), false);
            pkt.visualFeatures.emplace_back(402, Point2(0.1 + shift, -0.05), false);
            callProcessFrame(harness.backend, pkt);
            ++fid;
        }
    }

    const auto& win = getWindow(harness.backend);
    const auto& smoother = getSmoother(harness.backend);
    ASSERT_EQ(win.size(), cfg.slidingWindowSize);

    // Oldest frame in window must remain connected in VariableIndex (bias key not orphaned)
    EXPECT_FALSE(smoother.getVariableIndex().empty(B(win.front().frameId)));
    EXPECT_FALSE(win.front().imuFactorSlot.has_value());
    EXPECT_FALSE(win.front().biasFactorSlot.has_value());

    // Subsequent window frames must retain valid IMU/bias factors in ISAM2
    for (size_t i = 1; i < win.size(); ++i) {
        ASSERT_TRUE(win[i].imuFactorSlot.has_value());
        ASSERT_TRUE(win[i].biasFactorSlot.has_value());
        EXPECT_NE(smoother.getFactorsUnsafe().at(*win[i].imuFactorSlot), nullptr);
        EXPECT_NE(smoother.getFactorsUnsafe().at(*win[i].biasFactorSlot), nullptr);
    }

    EXPECT_TRUE(smoother.valueExists(X(win.back().frameId)));
}

TEST_F(BackendTest, BootstrappingStaticAccelBiasWithKnownPose) {
    BackendConfig cfg = makeDefaultBackendConfig();
    // 90 deg rotation around Y maps world +Z onto body -X
    cfg.initialPose = Pose3(Rot3::Ry(M_PI / 2.0), Point3(1.0, -2.0, 3.0));
    TestHarness harness(cfg);

    const Vector3 trueAccelBias(-0.025, 0.12, 0.075);
    const Vector3 expectedGravityInBody(-cfg.gravity, 0.0, 0.0);

    BackendHfPacket pkt0;
    pkt0.frameId = 0;
    pkt0.timestamp = 0.0;
    pkt0.isKeyframe = true;
    pkt0.depth = 3.0;
    pkt0.initialAccel = expectedGravityInBody + trueAccelBias;
    pkt0.initialGyro = Vector3(0.01, -0.02, 0.03);

    callInitializeSystem(harness.backend, pkt0);
    ASSERT_TRUE(isInitialized(harness.backend));

    const auto bias0 = getSmootherEstimate(harness.backend).at<gtsam::imuBias::ConstantBias>(B(0));
    EXPECT_NEAR(bias0.accelerometer().x(), trueAccelBias.x(), 1e-6);
    EXPECT_NEAR(bias0.accelerometer().y(), trueAccelBias.y(), 1e-6);
    EXPECT_NEAR(bias0.accelerometer().z(), trueAccelBias.z(), 1e-6);

    const auto feBias = TestFrontend::getImuBias(harness.frontend);
    EXPECT_NEAR(feBias.accelerometer().x(), trueAccelBias.x(), 1e-6);
    EXPECT_NEAR(feBias.accelerometer().y(), trueAccelBias.y(), 1e-6);
    EXPECT_NEAR(feBias.accelerometer().z(), trueAccelBias.z(), 1e-6);
}

TEST_F(BackendTest, BootstrappingStaticAccelBiasRejectsDynamicAccel) {
    BackendConfig cfg = makeDefaultBackendConfig();
    cfg.initialPose = Pose3(Rot3::Identity(), Point3::Zero());
    TestHarness harness(cfg);

    BackendHfPacket pkt0;
    pkt0.frameId = 0;
    pkt0.timestamp = 0.0;
    pkt0.depth = 3.0;
    pkt0.initialAccel = Vector3(0.0, 0.0, 15.0);

    callInitializeSystem(harness.backend, pkt0);
    ASSERT_TRUE(isInitialized(harness.backend));

    const auto bias0 = getSmootherEstimate(harness.backend).at<gtsam::imuBias::ConstantBias>(B(0));
    EXPECT_NEAR(bias0.accelerometer().norm(), 0.0, 1e-6);
}

TEST_F(BackendTest, ZuptClampsStandstillDriftAndTransitionsCleanlyToMotion) {
    BackendConfig cfg = makeDefaultBackendConfig();
    cfg.slidingWindowSize = 5;
    cfg.initialPose = Pose3(Rot3::Identity(), Point3(0.0, 0.0, 5.0));
    TestHarness harness(cfg);

    auto makeImuBatch = [](double tStart, double tEnd, const Vector3& accel) {
        std::vector<Imu> samples;
        for (double t = tStart + 0.005; t <= tEnd + 1e-9; t += 0.005) {
            Imu s;
            s.timestamp = t;
            s.a = accel;
            s.w = Vector3::Zero();
            s.t = 20.0f;
            samples.push_back(s);
        }
        return samples;
    };

    BackendHfPacket pkt0;
    pkt0.frameId = 0;
    pkt0.timestamp = 0.0;
    pkt0.isKeyframe = true;
    pkt0.depth = 5.0;
    callInitializeSystem(harness.backend, pkt0);

    // Feed stationary keyframes and non-keyframes carrying an uncompensated +0.2 m/s^2 accel bias
    const Vector3 biasedAccel(0.2, -0.15, 9.81);
    double ts = 0.0;
    uint32_t lastStationaryId = 0;
    for (uint32_t fid = 1; fid <= 12; ++fid) {
        BackendHfPacket pkt;
        pkt.frameId = fid;
        ts += 0.05;
        pkt.timestamp = ts;
        pkt.isKeyframe = (fid % 2 == 0) && (fid != 12);
        pkt.isStationary = true;
        pkt.depth = 5.0;
        pkt.imuDelta = makePim(biasedAccel, Vector3::Zero(), 0.05);
        pkt.rawImu = makeImuBatch(ts - 0.05, ts, biasedAccel);
        callProcessFrame(harness.backend, pkt);
        lastStationaryId = fid;
    }

    const auto& estStatic = getSmootherEstimate(harness.backend);
    ASSERT_TRUE(estStatic.exists(X(lastStationaryId)));
    ASSERT_TRUE(estStatic.exists(V(lastStationaryId)));
    const Pose3 staticPose = estStatic.at<Pose3>(X(lastStationaryId));
    const Vector3 staticVel = estStatic.at<Vector3>(V(lastStationaryId));

    // ZUPT should hold horizontal drift and velocity tight despite 0.6s of biased acceleration
    EXPECT_LT(staticPose.translation().head<2>().norm(), 0.02);
    EXPECT_LT(staticVel.norm(), 0.02);

    // Transition to moving keyframe, collapsing stationary non-keyframe 12 cleanly
    BackendHfPacket movePkt;
    movePkt.frameId = 13;
    ts += 0.05;
    movePkt.timestamp = ts;
    movePkt.isKeyframe = true;
    movePkt.isStationary = false;
    movePkt.depth = 5.0;
    movePkt.imuDelta = makePim(Vector3(1.5, 0.0, 9.81), Vector3::Zero(), 0.05);
    movePkt.rawImu = makeImuBatch(ts - 0.05, ts, Vector3(1.5, 0.0, 9.81));
    callProcessFrame(harness.backend, movePkt);

    const auto& win = getWindow(harness.backend);
    const auto& smoother = getSmoother(harness.backend);
    EXPECT_FALSE(smoother.valueExists(X(12)));
    EXPECT_FALSE(smoother.valueExists(V(12)));
    EXPECT_TRUE(smoother.valueExists(X(13)));
    EXPECT_TRUE(smoother.valueExists(V(13)));
    EXPECT_FALSE(win.front().zuptPoseFactorSlot.has_value());
}

} // namespace PoolSLAM
