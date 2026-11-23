#include "TestFrontendHelpers.hpp"

namespace PoolSLAM {

// =============================================================================
// 1. Thread Lifecycle & State Transitions
// =============================================================================

TEST_F(TestFrontend, StartLaunchesThreadAndResetsState) {
    ImuStream imuStream;
    DepthStream depthStream;
    ImageStream imageStream(W, H);
    Synchronizer sync(imuStream, depthStream, imageStream);

    Frontend fe(makeConfig(), sync, "");
    EXPECT_FALSE(getRunFrontend(fe));

    fe.start();
    EXPECT_TRUE(getRunFrontend(fe));
    EXPECT_EQ(getFrameId(fe), 0);
    EXPECT_TRUE(getLfTracker(fe).isRunning());

    fe.end();
    EXPECT_FALSE(getRunFrontend(fe));
    EXPECT_FALSE(getLfTracker(fe).isRunning());
}

TEST_F(TestFrontend, StartIsIdempotent) {
    ImuStream imuStream;
    DepthStream depthStream;
    ImageStream imageStream(W, H);
    Synchronizer sync(imuStream, depthStream, imageStream);

    Frontend fe(makeConfig(), sync, "");
    fe.start();
    EXPECT_TRUE(getRunFrontend(fe));

    // Calling start again while running should be a safe no-op
    fe.start();
    EXPECT_TRUE(getRunFrontend(fe));

    fe.end();
}

TEST_F(TestFrontend, EndHaltsThreadAndSetsState) {
    ImuStream imuStream;
    DepthStream depthStream;
    ImageStream imageStream(W, H);
    Synchronizer sync(imuStream, depthStream, imageStream);

    Frontend fe(makeConfig(), sync, "");
    fe.start();
    EXPECT_TRUE(getRunFrontend(fe));

    fe.end();
    EXPECT_FALSE(getRunFrontend(fe));

    // Calling end again is safe
    fe.end();
    EXPECT_FALSE(getRunFrontend(fe));
}

TEST_F(TestFrontend, CleanRestart) {
    ImuStream imuStream;
    DepthStream depthStream;
    ImageStream imageStream(W, H);
    Synchronizer sync(imuStream, depthStream, imageStream);

    Frontend fe(makeConfig(), sync, "");
    fe.start();
    EXPECT_TRUE(getRunFrontend(fe));
    fe.end();

    // Clean restart
    fe.start();
    EXPECT_TRUE(getRunFrontend(fe));
    EXPECT_EQ(getFrameId(fe), 0);
    fe.end();
}

TEST_F(TestFrontend, DestructorCleanShutdown) {
    ImuStream imuStream;
    DepthStream depthStream;
    ImageStream imageStream(W, H);
    Synchronizer sync(imuStream, depthStream, imageStream);

    {
        Frontend fe(makeConfig(), sync, "");
        fe.start();
        EXPECT_TRUE(getRunFrontend(fe));
    } // Destructor runs here and must cleanly exit
}

// =============================================================================
// 2. Synchronizer Ingestion & Shutdown Handshake
// =============================================================================

TEST_F(TestFrontend, SynchronizerShutdownTerminatesWorker) {
    ImuStream imuStream;
    DepthStream depthStream;
    ImageStream imageStream(W, H);
    Synchronizer sync(imuStream, depthStream, imageStream);

    Frontend fe(makeConfig(), sync, "");
    sync.start();
    fe.start();

    // Stopping synchronizer triggers shutdown handshake
    sync.stop();

    // getHfPacket should unblock and return nullptr once synchronizer is stopped
    auto pkt = popHfPacketWithTimeout(fe, std::chrono::milliseconds(500));
    EXPECT_EQ(pkt, nullptr);

    fe.end();
}

TEST_F(TestFrontend, BufferRecyclingInvariant) {
    ImuStream imuStream;
    DepthStream depthStream;
    ImageStream imageStream(W, H);
    Synchronizer sync(imuStream, depthStream, imageStream);

    Frontend fe(makeConfig(), sync, "");
    pushInertialAndDepthWindow(imuStream, depthStream, 1.000, 1.200, 5.0);

    sync.start();
    fe.start();

    auto img = makeTexturedImage(0, 0);

    size_t freeBefore = getImageFreeListSize(imageStream);
    ASSERT_TRUE(imageStream.pushFrame(img.data(), 1.020));

    auto pkt = popHfPacketWithTimeout(fe, std::chrono::milliseconds(500));
    ASSERT_NE(pkt, nullptr);

    // Frame buffer must have been recycled back to ImageStream
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    size_t freeAfter = getImageFreeListSize(imageStream);
    EXPECT_EQ(freeAfter, freeBefore);

    fe.end();
    sync.stop();
}

// =============================================================================
// 3. IMU Bias Thread Safety
// =============================================================================

TEST_F(TestFrontend, SetBiasThreadSafeMutation) {
    ImuStream imuStream;
    DepthStream depthStream;
    ImageStream imageStream(W, H);
    Synchronizer sync(imuStream, depthStream, imageStream);

    Frontend fe(makeConfig(), sync, "");

    gtsam::imuBias::ConstantBias newBias(
        gtsam::Vector3(0.01, -0.02, 0.03),
        gtsam::Vector3(0.001, -0.002, 0.003)
    );

    EXPECT_TRUE(fe.setBias(newBias));

    auto retrieved = getImuBias(fe);
    EXPECT_TRUE((retrieved.accelerometer() - newBias.accelerometer()).norm() < 1e-9);
    EXPECT_TRUE((retrieved.gyroscope() - newBias.gyroscope()).norm() < 1e-9);
}

// =============================================================================
// 4. Keyframe Trigger Logic
// =============================================================================

TEST_F(TestFrontend, Condition1InitialFrameIsKeyframe) {
    ImuStream imuStream;
    DepthStream depthStream;
    ImageStream imageStream(W, H);
    Synchronizer sync(imuStream, depthStream, imageStream);

    Frontend fe(makeConfig(), sync, "");
    pushInertialAndDepthWindow(imuStream, depthStream, 1.000, 1.200, 4.5,
                               Eigen::Vector3d(0.1, 0.2, -9.81), Eigen::Vector3d(0.01, 0.02, 0.03));

    sync.start();
    fe.start();

    auto img = makeTexturedImage(0, 0);
    ASSERT_TRUE(imageStream.pushFrame(img.data(), 1.020));

    auto pkt = popHfPacketWithTimeout(fe, std::chrono::milliseconds(500));
    ASSERT_NE(pkt, nullptr);

    EXPECT_EQ(pkt->frameId, 0);
    EXPECT_TRUE(pkt->isKeyframe);
    ASSERT_TRUE(pkt->depth.has_value());
    EXPECT_NEAR(pkt->depth.value(), 4.5, 0.01);

    // Initial stationary readings must be populated
    ASSERT_TRUE(pkt->initialAccel.has_value());
    ASSERT_TRUE(pkt->initialGyro.has_value());

    fe.end();
    sync.stop();
}

TEST_F(TestFrontend, NegativeCaseNonKeyframe) {
    ImuStream imuStream;
    DepthStream depthStream;
    ImageStream imageStream(W, H);
    Synchronizer sync(imuStream, depthStream, imageStream);

    Frontend fe(makeConfig(), sync, "");
    pushInertialAndDepthWindow(imuStream, depthStream, 1.000, 1.200, 5.0);

    sync.start();
    fe.start();

    auto img = makeTexturedImage(0, 0);

    // Frame 0: Keyframe at t = 1.020
    ASSERT_TRUE(imageStream.pushFrame(img.data(), 1.020));
    auto pkt0 = popHfPacketWithTimeout(fe, std::chrono::milliseconds(500));
    ASSERT_NE(pkt0, nullptr);
    EXPECT_TRUE(pkt0->isKeyframe);

    // Frame 1: Identical image, 40 ms later at t = 1.060 (0 displacement, 100% feature retention)
    ASSERT_TRUE(imageStream.pushFrame(img.data(), 1.060));
    auto pkt1 = popHfPacketWithTimeout(fe, std::chrono::milliseconds(500));
    ASSERT_NE(pkt1, nullptr);

    EXPECT_EQ(pkt1->frameId, 1);
    EXPECT_FALSE(pkt1->isKeyframe); // Non-keyframe
    EXPECT_FALSE(pkt1->initialAccel.has_value());

    fe.end();
    sync.stop();
}

TEST_F(TestFrontend, Condition4TimeHorizonTrigger) {
    ImuStream imuStream;
    DepthStream depthStream;
    ImageStream imageStream(W, H);
    Synchronizer sync(imuStream, depthStream, imageStream);

    Frontend fe(makeConfig(), sync, "");
    pushInertialAndDepthWindow(imuStream, depthStream, 1.000, 1.150, 5.0);
    pushInertialAndDepthWindow(imuStream, depthStream, 2.050, 2.200, 5.0);

    sync.start();
    fe.start();

    auto img = makeTexturedImage(0, 0);

    // Frame 0: Keyframe at t = 1.020
    ASSERT_TRUE(imageStream.pushFrame(img.data(), 1.020));
    auto pkt0 = popHfPacketWithTimeout(fe, std::chrono::milliseconds(500));
    ASSERT_NE(pkt0, nullptr);

    // Frame 1: Arrives at t = 2.100 (> 1.0s elapsed)
    ASSERT_TRUE(imageStream.pushFrame(img.data(), 2.100));
    auto pkt1 = popHfPacketWithTimeout(fe, std::chrono::milliseconds(500));
    ASSERT_NE(pkt1, nullptr);

    EXPECT_EQ(pkt1->frameId, 1);
    EXPECT_TRUE(pkt1->isKeyframe); // Time-based keyframe trigger

    fe.end();
    sync.stop();
}

// =============================================================================
// 5. Packet Construction & Queue Semantics
// =============================================================================

TEST_F(TestFrontend, MonotonicFrameIdAndQueueOrder) {
    ImuStream imuStream;
    DepthStream depthStream;
    ImageStream imageStream(W, H);
    Synchronizer sync(imuStream, depthStream, imageStream);

    Frontend fe(makeConfig(), sync, "");
    pushInertialAndDepthWindow(imuStream, depthStream, 1.000, 1.300, 5.0);

    sync.start();
    fe.start();

    auto img = makeTexturedImage(0, 0);

    // Stream 3 consecutive frames
    for (size_t i = 0; i < 3; ++i) {
        double t = 1.020 + static_cast<double>(i) * 0.050;
        ASSERT_TRUE(imageStream.pushFrame(img.data(), t));

        auto pkt = popHfPacketWithTimeout(fe, std::chrono::milliseconds(500));
        ASSERT_NE(pkt, nullptr);
        EXPECT_EQ(pkt->frameId, static_cast<uint32_t>(i));
        EXPECT_NEAR(pkt->timestamp, t, 0.001);
    }

    fe.end();
    sync.stop();
}

TEST_F(TestFrontend, VisualFeatureObservationsPopulated) {
    ImuStream imuStream;
    DepthStream depthStream;
    ImageStream imageStream(W, H);
    Synchronizer sync(imuStream, depthStream, imageStream);

    Frontend fe(makeConfig(), sync, "");
    pushInertialAndDepthWindow(imuStream, depthStream, 1.000, 1.200, 5.0);

    sync.start();
    fe.start();

    auto img = makeTexturedImage(0, 0);
    ASSERT_TRUE(imageStream.pushFrame(img.data(), 1.020));

    auto pkt = popHfPacketWithTimeout(fe, std::chrono::milliseconds(500));
    ASSERT_NE(pkt, nullptr);

    // Initial frame must have replenished features
    EXPECT_FALSE(pkt->visualFeatures.empty());
    for (const auto& obs : pkt->visualFeatures) {
        // Initial frame features are new tracks
        EXPECT_TRUE(obs.isNewTrack);
        // Normalized pinhole coordinates must be finite
        EXPECT_TRUE(std::isfinite(obs.pt.x()));
        EXPECT_TRUE(std::isfinite(obs.pt.y()));
    }

    fe.end();
    sync.stop();
}

} // namespace PoolSLAM
