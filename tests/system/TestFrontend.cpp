#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include <eigen3/Eigen/Dense>
#include <gtsam/geometry/Point2.h>
#include <gtsam/navigation/ImuBias.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/PreintegrationParams.h>
#include <gtest/gtest.h>

#include "CoreTypes.hpp"
#include "TestFrontendHelpers.hpp"
#include "frontend/Frontend.hpp"
#include "frontend/HfTracker.hpp"
#include "frontend/LfTracker.hpp"
#include "hal/Synchronizer.hpp"
#include "pool_slam/Hal.hpp"

namespace PoolSLAM {

class FrontendTest : public ::testing::Test {
protected:
    static constexpr size_t kImageWidth = 640;
    static constexpr size_t kImageHeight = 480;
    static constexpr size_t kReadyQueueCap = 8;
    static constexpr size_t kImagePoolCap = 16;

    static constexpr float kFx = 525.0f;
    static constexpr float kFy = 525.0f;
    static constexpr float kCx = 320.0f;
    static constexpr float kCy = 240.0f;
    static constexpr double kGravity = 9.81;

    std::string vocabFilePath = "test_frontend_vocab.bin";

    ImuStream imuStream;
    DepthStream depthStream;
    ImageStream imageStream{kImageWidth, kImageHeight, kReadyQueueCap, kImagePoolCap};
    std::unique_ptr<Synchronizer> sync;

    FrontendConfig config{};
    std::unique_ptr<Frontend> frontend;

    std::vector<uint8_t> cornerGridImage;
    std::vector<uint8_t> blankImage;

    void SetUp() override {
        ASSERT_TRUE(test_helpers::writeSyntheticVocabFile(vocabFilePath));

        Eigen::Matrix3f K = Eigen::Matrix3f::Identity();
        K(0, 0) = kFx;
        K(1, 1) = kFy;
        K(0, 2) = kCx;
        K(1, 2) = kCy;

        Distortion dist{};

        config.w = kImageWidth;
        config.h = kImageHeight;
        config.K = K;
        config.dist = dist;

        auto preintParams = gtsam::PreintegrationParams::MakeSharedU(kGravity);
        preintParams->accelerometerCovariance = Eigen::Matrix3d::Identity() * 1e-3;
        preintParams->gyroscopeCovariance = Eigen::Matrix3d::Identity() * 1e-4;
        preintParams->integrationCovariance = Eigen::Matrix3d::Identity() * 1e-4;
        config.preintParams = preintParams;

        config.droppedFeatureThreshold = 0.5f;
        config.maxTimeBetweenKf = 1.0;
        config.hfQueueSize = 5;

        config.lfConfig.wImage = kImageWidth;
        config.lfConfig.hImage = kImageHeight;
        config.lfConfig.K = K;
        config.lfConfig.dist = dist;

        cornerGridImage = test_helpers::createCornerGridImage(kImageWidth, kImageHeight, 6, 5);
        blankImage = test_helpers::createBlankImage(kImageWidth, kImageHeight);

        sync = std::make_unique<Synchronizer>(imuStream, depthStream, imageStream, 10);
        frontend = std::make_unique<Frontend>(config, *sync, vocabFilePath);
    }

    void TearDown() override {
        if (frontend) {
            frontend->end();
        }
        if (sync) {
            sync->stop();
        }
        std::remove(vocabFilePath.c_str());
    }

    // Accessor helpers for private Frontend state via friend class FrontendTest
    bool isFrontendRunning() const {
        return frontend->runFrontend.load(std::memory_order_acquire);
    }

    uint32_t getInternalFrameId() const {
        return frontend->frameId;
    }

    size_t getQueuedPacketCount() {
        std::lock_guard<std::mutex> lock(frontend->m_hfQueue);
        return frontend->hfQueue.size();
    }

    gtsam::imuBias::ConstantBias getInternalBias() {
        std::lock_guard<std::mutex> lock(frontend->m_imuBias);
        return frontend->imuBias;
    }

    HfTracker &getInternalHfTracker() {
        return frontend->hf;
    }
};

// ============================================================================
// 1. Thread Lifecycle & Consumer Unblocking
// ============================================================================

// Verifies that start() and end() transition the worker thread state cleanly and behave
// idempotently when invoked multiple times in a row.
TEST_F(FrontendTest, startAndEndManageLifecycleIdempotently) {
    EXPECT_FALSE(isFrontendRunning());

    sync->start();
    frontend->start();
    EXPECT_TRUE(isFrontendRunning());
    EXPECT_EQ(getInternalFrameId(), 0u);

    // Calling start() while already running must be a no-op
    frontend->start();
    EXPECT_TRUE(isFrontendRunning());

    frontend->end();
    EXPECT_FALSE(isFrontendRunning());

    // Calling end() while already stopped must be a no-op
    frontend->end();
    EXPECT_FALSE(isFrontendRunning());
}

// Verifies that calling end() immediately wakes a backend consumer thread blocked on getHfPacket()
// and returns nullptr when hfQueue is empty.
TEST_F(FrontendTest, endUnblocksWaitingConsumerWithNullptr) {
    sync->start();
    frontend->start();

    std::atomic<bool> consumerEntered{false};
    std::atomic<bool> consumerFinished{false};
    std::unique_ptr<BackendHfPacket> poppedPacket;

    std::thread consumerThread([&]() {
        consumerEntered.store(true, std::memory_order_release);
        poppedPacket = frontend->getHfPacket();
        consumerFinished.store(true, std::memory_order_release);
    });

    while (!consumerEntered.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(25));
    EXPECT_FALSE(consumerFinished.load(std::memory_order_acquire));

    const auto tStart = std::chrono::steady_clock::now();
    frontend->end();
    consumerThread.join();
    const auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - tStart).count();

    EXPECT_TRUE(consumerFinished.load(std::memory_order_acquire));
    EXPECT_EQ(poppedPacket, nullptr);
    EXPECT_LT(elapsedMs, 100);
}

// Verifies that stopping and restarting Frontend resets frameId back to 0 and reinitializes
// the keyframe tracking state for a new session.
TEST_F(FrontendTest, restartResetsFrameIdAndKeyframeState) {
    sync->start();
    frontend->start();

    ASSERT_TRUE(test_helpers::pushSynchronizedFrame(
        imuStream, depthStream, imageStream, -1.0, 1.000, cornerGridImage, 4.0));
    auto pkt0 = frontend->getHfPacket();
    ASSERT_NE(pkt0, nullptr);
    EXPECT_EQ(pkt0->frameId, 0u);
    EXPECT_TRUE(pkt0->isKeyframe);

    ASSERT_TRUE(test_helpers::pushSynchronizedFrame(
        imuStream, depthStream, imageStream, 1.000, 1.050, cornerGridImage, 4.0));
    auto pkt1 = frontend->getHfPacket();
    ASSERT_NE(pkt1, nullptr);
    EXPECT_EQ(pkt1->frameId, 1u);

    frontend->end();
    sync->stop();

    // Restart Synchronizer and Frontend on a new timeline
    sync->start();
    frontend->start();

    ASSERT_TRUE(test_helpers::pushSynchronizedFrame(
        imuStream, depthStream, imageStream, -1.0, 10.000, cornerGridImage, 6.0));
    auto pktRestart = frontend->getHfPacket();
    ASSERT_NE(pktRestart, nullptr);
    EXPECT_EQ(pktRestart->frameId, 0u);
    EXPECT_DOUBLE_EQ(pktRestart->timestamp, 10.000);
    EXPECT_TRUE(pktRestart->isKeyframe);
}

// ============================================================================
// 2. First-Frame (T0) Initialization & Multi-Frame Visual Tracking
// ============================================================================

// Verifies that the T0 boot frame (with an empty IMU batch) skips trackExistingFeatures,
// marks isKeyframe = true, produces a zero-dt IMU delta, and replenishes empty grid buckets.
TEST_F(FrontendTest, firstFrameInitializesKeyframeAndSeedsFeatures) {
    sync->start();
    frontend->start();

    constexpr double kT0 = 1.000;
    constexpr double kDepth0 = 3.25;
    ASSERT_TRUE(test_helpers::pushSynchronizedFrame(
        imuStream, depthStream, imageStream, -1.0, kT0, cornerGridImage, kDepth0));

    auto pkt0 = frontend->getHfPacket();
    ASSERT_NE(pkt0, nullptr);
    EXPECT_EQ(pkt0->frameId, 0u);
    EXPECT_DOUBLE_EQ(pkt0->timestamp, kT0);
    EXPECT_TRUE(pkt0->isKeyframe);
    EXPECT_TRUE(pkt0->visualMeasurements.empty());

    ASSERT_NE(pkt0->imuDelta, nullptr);
    EXPECT_DOUBLE_EQ(pkt0->imuDelta->deltaTij(), 0.0);

    ASSERT_TRUE(pkt0->depth.has_value());
    EXPECT_NEAR(*pkt0->depth, kDepth0, 1e-4);

    // Wait briefly for replenishBuckets to complete on frame 0
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    EXPECT_GT(getInternalHfTracker().getActiveFeatures().size(), 15u);
}

// Verifies that frame T1 tracks features seeded on T0 and packages unique featureIds and
// calibrated normalized coordinates (VisualMeasurement::featureId and VisualMeasurement::pt).
TEST_F(FrontendTest, subsequentFrameTracksFeaturesAndPackagesUndistortedMeasurements) {
    sync->start();
    frontend->start();

    ASSERT_TRUE(test_helpers::pushSynchronizedFrame(
        imuStream, depthStream, imageStream, -1.0, 1.000, cornerGridImage, 5.0));
    auto pkt0 = frontend->getHfPacket();
    ASSERT_NE(pkt0, nullptr);

    // Push frame T1 with a sub-pixel horizontal translation (+0.5 px) and zero rotation
    auto shiftedImage = test_helpers::createCornerGridImage(
        kImageWidth, kImageHeight, 6, 5, 0.5f, 0.0f);
    ASSERT_TRUE(test_helpers::pushSynchronizedFrame(
        imuStream, depthStream, imageStream, 1.000, 1.040, shiftedImage, 5.0));

    auto pkt1 = frontend->getHfPacket();
    ASSERT_NE(pkt1, nullptr);
    EXPECT_EQ(pkt1->frameId, 1u);
    EXPECT_DOUBLE_EQ(pkt1->timestamp, 1.040);
    EXPECT_FALSE(pkt1->isKeyframe);
    EXPECT_GT(pkt1->visualMeasurements.size(), 15u);

    // Verify all VisualMeasurement entries have unique featureIds and valid normalized rays
    const double maxNormX = (static_cast<double>(kImageWidth) - static_cast<double>(kCx)) / static_cast<double>(kFx);
    const double maxNormY = (static_cast<double>(kImageHeight) - static_cast<double>(kCy)) / static_cast<double>(kFy);
    std::unordered_set<uint32_t> seenIds;

    for (size_t i = 0; i < pkt1->visualMeasurements.size(); i++) {
        const auto &vm = pkt1->visualMeasurements[i];
        EXPECT_TRUE(seenIds.insert(vm.featureId).second);
        EXPECT_TRUE(std::isfinite(vm.pt.x()));
        EXPECT_TRUE(std::isfinite(vm.pt.y()));
        EXPECT_LE(std::abs(vm.pt.x()), maxNormX + 1e-3);
        EXPECT_LE(std::abs(vm.pt.y()), maxNormY + 1e-3);
    }
}

// ============================================================================
// 3. IMU Pre-Integration & Thread-Safe Bias Updates
// ============================================================================

// Verifies that processPacket() integrates the inter-frame IMU batch over (T0, T1] and attaches
// the resulting PreintegratedImuMeasurements delta to BackendHfPacket.
TEST_F(FrontendTest, imuPreintegrationAccumulatesAngularVelocityAndAcceleration) {
    sync->start();
    frontend->start();

    constexpr double kT0 = 1.000;
    constexpr double kT1 = 1.050;
    constexpr double kDt = kT1 - kT0;
    const Eigen::Vector3d hoverAccel(0.0, 0.0, -kGravity);
    const Eigen::Vector3d zeroOmega = Eigen::Vector3d::Zero();

    ASSERT_TRUE(test_helpers::pushSynchronizedFrame(
        imuStream, depthStream, imageStream, -1.0, kT0, cornerGridImage, 5.0, hoverAccel, zeroOmega));
    ASSERT_NE(frontend->getHfPacket(), nullptr);

    // Apply a small roll rate (0.02 rad/s about Z) and +0.5 m/s^2 X acceleration over 50 ms
    const Eigen::Vector3d activeAccel(0.5, 0.0, -kGravity);
    const Eigen::Vector3d activeOmega(0.0, 0.0, 0.02);
    ASSERT_TRUE(test_helpers::pushSynchronizedFrame(
        imuStream, depthStream, imageStream, kT0, kT1, cornerGridImage, 5.0, activeAccel, activeOmega));

    auto pkt1 = frontend->getHfPacket();
    ASSERT_NE(pkt1, nullptr);
    ASSERT_NE(pkt1->imuDelta, nullptr);

    EXPECT_NEAR(pkt1->imuDelta->deltaTij(), kDt, 1e-4);
    EXPECT_NEAR(pkt1->imuDelta->deltaVij().x(), 0.5 * kDt, 5e-3);

    const Eigen::AngleAxisd deltaAngleAxis(pkt1->imuDelta->deltaRij().matrix());
    EXPECT_NEAR(deltaAngleAxis.angle(), 0.02 * kDt, 1e-3);
}

// Verifies that setBias() updates imuBias under mutex lock and that subsequent frames construct
// their PreintegratedImuMeasurements with the updated bias linearization point.
TEST_F(FrontendTest, setBiasUpdatesPreintegratorBiasLinearizationPoint) {
    sync->start();
    frontend->start();

    const Eigen::Vector3d injectedAccelBias(0.25, -0.15, 0.10);
    const Eigen::Vector3d injectedGyroBias(0.01, -0.02, 0.015);
    const gtsam::imuBias::ConstantBias targetBias(injectedAccelBias, injectedGyroBias);

    frontend->setBias(targetBias);

    const auto storedBias = getInternalBias();
    EXPECT_NEAR((storedBias.accelerometer() - injectedAccelBias).norm(), 0.0, 1e-9);
    EXPECT_NEAR((storedBias.gyroscope() - injectedGyroBias).norm(), 0.0, 1e-9);

    constexpr double kT0 = 2.000;
    constexpr double kT1 = 2.050;
    const Eigen::Vector3d biasedHoverAccel = Eigen::Vector3d(0.0, 0.0, -kGravity) + injectedAccelBias;

    ASSERT_TRUE(test_helpers::pushSynchronizedFrame(
        imuStream, depthStream, imageStream, -1.0, kT0, cornerGridImage, 4.0,
        biasedHoverAccel, injectedGyroBias));
    ASSERT_NE(frontend->getHfPacket(), nullptr);

    // Sensor readings equal (hover + targetBias), so bias-compensated integration must yield ~0 delta
    ASSERT_TRUE(test_helpers::pushSynchronizedFrame(
        imuStream, depthStream, imageStream, kT0, kT1, cornerGridImage, 4.0,
        biasedHoverAccel, injectedGyroBias));

    auto pkt1 = frontend->getHfPacket();
    ASSERT_NE(pkt1, nullptr);
    ASSERT_NE(pkt1->imuDelta, nullptr);

    EXPECT_NEAR(pkt1->imuDelta->deltaVij().norm(), 0.0, 1e-3);
    const Eigen::AngleAxisd netRot(pkt1->imuDelta->deltaRij().matrix());
    EXPECT_NEAR(netRot.angle(), 0.0, 1e-4);
}

// ============================================================================
// 4. Depth Propagation & Graceful Degradation
// ============================================================================

// Verifies that BackendHfPacket::depth contains interpolated depth when active and degrades
// to std::nullopt when depth measurements stop arriving.
TEST_F(FrontendTest, depthPopulatedWhenActiveAndNulloptWhenDepthDegrades) {
    sync->start();
    frontend->start();

    // Frame 0 with valid depth = 7.5 m
    ASSERT_TRUE(test_helpers::pushSynchronizedFrame(
        imuStream, depthStream, imageStream, -1.0, 1.000, cornerGridImage, 7.5));
    auto pkt0 = frontend->getHfPacket();
    ASSERT_NE(pkt0, nullptr);
    ASSERT_TRUE(pkt0->depth.has_value());
    EXPECT_NEAR(*pkt0->depth, 7.5, 1e-4);

    // Frame 1 with no new depth pushed past t = 1.030s (depth times out at t = 1.250s)
    ASSERT_TRUE(test_helpers::pushSynchronizedFrame(
        imuStream, depthStream, imageStream, 1.000, 1.250, cornerGridImage, std::nullopt));
    auto pkt1 = frontend->getHfPacket();
    ASSERT_NE(pkt1, nullptr);
    EXPECT_EQ(pkt1->frameId, 1u);
    EXPECT_FALSE(pkt1->depth.has_value());
}

// ============================================================================
// 5. Keyframe Decision Logic (Temporal Timeout & Feature Drop)
// ============================================================================

// Verifies that a new keyframe is triggered when elapsed time since the previous keyframe
// exceeds config.maxTimeBetweenKf, even when all features survive tracking.
TEST_F(FrontendTest, keyframeTriggeredWhenMaxTimeBetweenKfExceeded) {
    sync->start();
    frontend->start();

    // T0 = 1.00s -> Keyframe
    ASSERT_TRUE(test_helpers::pushSynchronizedFrame(
        imuStream, depthStream, imageStream, -1.0, 1.000, cornerGridImage, 5.0));
    auto pkt0 = frontend->getHfPacket();
    ASSERT_NE(pkt0, nullptr);
    EXPECT_TRUE(pkt0->isKeyframe);

    // T1 = 1.40s (elapsed 0.40s <= maxTimeBetweenKf = 1.0s) -> Non-keyframe
    ASSERT_TRUE(test_helpers::pushSynchronizedFrame(
        imuStream, depthStream, imageStream, 1.000, 1.400, cornerGridImage, 5.0));
    auto pkt1 = frontend->getHfPacket();
    ASSERT_NE(pkt1, nullptr);
    EXPECT_FALSE(pkt1->isKeyframe);

    // T2 = 1.85s (elapsed 0.85s <= maxTimeBetweenKf = 1.0s) -> Non-keyframe
    ASSERT_TRUE(test_helpers::pushSynchronizedFrame(
        imuStream, depthStream, imageStream, 1.400, 1.850, cornerGridImage, 5.0));
    auto pkt2 = frontend->getHfPacket();
    ASSERT_NE(pkt2, nullptr);
    EXPECT_FALSE(pkt2->isKeyframe);

    // T3 = 2.05s (elapsed 1.05s > maxTimeBetweenKf = 1.0s) -> Keyframe triggered by timeout
    ASSERT_TRUE(test_helpers::pushSynchronizedFrame(
        imuStream, depthStream, imageStream, 1.850, 2.050, cornerGridImage, 5.0));
    auto pkt3 = frontend->getHfPacket();
    ASSERT_NE(pkt3, nullptr);
    EXPECT_TRUE(pkt3->isKeyframe);
}

// Verifies that a new keyframe is triggered immediately when the ratio of surviving features
// relative to lastKfFeatures drops below config.droppedFeatureThreshold, and that subsequent
// frames recover after bucket replenishment updates lastKfFeatures.
TEST_F(FrontendTest, keyframeTriggeredWhenSurvivingFeatureRatioDropsBelowThreshold) {
    sync->start();
    frontend->start();

    // T0 = 1.00s: Initial keyframe seeds ~30 features in replenishBuckets
    ASSERT_TRUE(test_helpers::pushSynchronizedFrame(
        imuStream, depthStream, imageStream, -1.0, 1.000, cornerGridImage, 5.0));
    auto pkt0 = frontend->getHfPacket();
    ASSERT_NE(pkt0, nullptr);
    EXPECT_TRUE(pkt0->isKeyframe);

    // T1 = 1.04s: All ~30 features survive -> Non-keyframe
    ASSERT_TRUE(test_helpers::pushSynchronizedFrame(
        imuStream, depthStream, imageStream, 1.000, 1.040, cornerGridImage, 5.0));
    auto pkt1 = frontend->getHfPacket();
    ASSERT_NE(pkt1, nullptr);
    EXPECT_FALSE(pkt1->isKeyframe);
    EXPECT_GT(pkt1->visualMeasurements.size(), 15u);

    // T2 = 1.08s: Push partially occluded image (only 6 of 30 corners visible)
    // Surviving ratio (~6 / 30 = 0.20 < 0.50) triggers a keyframe well before maxTimeBetweenKf (1.0s)
    auto occludedImage = test_helpers::createCornerGridImage(
        kImageWidth, kImageHeight, 6, 5, 0.0f, 0.0f, 6);
    ASSERT_TRUE(test_helpers::pushSynchronizedFrame(
        imuStream, depthStream, imageStream, 1.040, 1.080, occludedImage, 5.0));
    auto pkt2 = frontend->getHfPacket();
    ASSERT_NE(pkt2, nullptr);
    EXPECT_TRUE(pkt2->isKeyframe);

    // T3 = 1.12s: Pushing the same 6-corner scene again keeps all 6 features (ratio 6/6 = 1.0 >= 0.5)
    ASSERT_TRUE(test_helpers::pushSynchronizedFrame(
        imuStream, depthStream, imageStream, 1.080, 1.120, occludedImage, 5.0));
    auto pkt3 = frontend->getHfPacket();
    ASSERT_NE(pkt3, nullptr);
    EXPECT_FALSE(pkt3->isKeyframe);
}

// ============================================================================
// 6. Queue Backpressure Eviction & Image Pool Recycling
// ============================================================================

// Verifies that pushing more frames than config.hfQueueSize evicts the oldest BackendHfPackets
// while still recycling every raw ImageData* buffer back to ImageStream without pool starvation.
TEST_F(FrontendTest, queueBackpressureEvictsOldestPacketsAndRecyclesImages) {
    sync->start();
    frontend->start();

    constexpr size_t kTotalFramesToPush = 9; // Exceeds config.hfQueueSize (5)
    double prevTime = -1.0;

    for (size_t i = 0; i < kTotalFramesToPush; i++) {
        const double curTime = 1.000 + static_cast<double>(i) * 0.040;
        ASSERT_TRUE(test_helpers::pushSynchronizedFrame(
            imuStream, depthStream, imageStream, prevTime, curTime, cornerGridImage, 5.0));
        prevTime = curTime;
        std::this_thread::sleep_for(std::chrono::milliseconds(15));
    }

    // Wait for Frontend worker thread to process all 9 frames
    const auto waitStart = std::chrono::steady_clock::now();
    while (getInternalFrameId() < kTotalFramesToPush) {
        if (std::chrono::steady_clock::now() - waitStart > std::chrono::seconds(2)) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    EXPECT_EQ(getInternalFrameId(), kTotalFramesToPush);
    EXPECT_EQ(getQueuedPacketCount(), config.hfQueueSize);

    // Pop remaining packets: frames 0..3 must have been evicted, leaving frames 4..8 in FIFO order
    const size_t firstExpectedId = kTotalFramesToPush - config.hfQueueSize;
    for (size_t i = 0; i < config.hfQueueSize; i++) {
        auto pkt = frontend->getHfPacket();
        ASSERT_NE(pkt, nullptr);
        EXPECT_EQ(pkt->frameId, static_cast<uint32_t>(firstExpectedId + i));
    }

    // Verify ImageStream pool was not starved by pushing another frame
    EXPECT_TRUE(imageStream.pushFrame(cornerGridImage.data(), 2.000));
}

// ============================================================================
// 7. End-to-End Multi-Threaded Concurrency & LfTracker Access
// ============================================================================

// Verifies concurrent operation of sensor producer threads, Frontend worker thread, LfTracker
// worker thread, and a backend consumer thread polling getHfPacket(), setBias(), and getLfTracker().
TEST_F(FrontendTest, concurrentMultiThreadedStreamingAndBiasUpdates) {
    sync->start();
    frontend->start();

    constexpr size_t kNumFrames = 15;
    std::atomic<bool> producersDone{false};

    std::thread producerThread([&]() {
        double prevTime = -1.0;
        for (size_t i = 0; i < kNumFrames; i++) {
            const double curTime = 1.000 + static_cast<double>(i) * 0.035;
            const float shift = static_cast<float>(i % 3) * 0.25f;
            auto frame = test_helpers::createCornerGridImage(
                kImageWidth, kImageHeight, 6, 5, shift, 0.0f);

            EXPECT_TRUE(test_helpers::pushSynchronizedFrame(
                imuStream, depthStream, imageStream, prevTime, curTime, frame,
                4.5 + 0.1 * static_cast<double>(i)));
            prevTime = curTime;
            std::this_thread::sleep_for(std::chrono::milliseconds(12));
        }
        producersDone.store(true, std::memory_order_release);
    });

    size_t receivedCount = 0;
    double lastTimestamp = -1.0;

    while (receivedCount < kNumFrames) {
        auto pkt = frontend->getHfPacket();
        ASSERT_NE(pkt, nullptr);

        EXPECT_EQ(pkt->frameId, static_cast<uint32_t>(receivedCount));
        EXPECT_GT(pkt->timestamp, lastTimestamp);
        lastTimestamp = pkt->timestamp;

        // Periodically update IMU bias and poll LfTracker from the backend consumer thread
        if (receivedCount % 3 == 0) {
            const gtsam::imuBias::ConstantBias updatedBias(
                Eigen::Vector3d(0.01 * static_cast<double>(receivedCount), 0.0, 0.0),
                Eigen::Vector3d(0.0, 0.0, 0.001 * static_cast<double>(receivedCount)));
            frontend->setBias(updatedBias);
        }

        BackendLfPacket lfMatch{};
        (void)frontend->getLfTracker().getMatch(lfMatch);

        receivedCount++;
    }

    producerThread.join();
    EXPECT_TRUE(producersDone.load(std::memory_order_acquire));
    EXPECT_EQ(receivedCount, kNumFrames);
}

} // namespace PoolSLAM