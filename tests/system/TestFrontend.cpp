#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <future>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <vector>
#include <mutex>
#include <utility>

#include <gtest/gtest.h>

#include <eigen3/Eigen/Dense>
#include <gtsam/navigation/ImuBias.h>
#include <gtsam/navigation/ImuFactor.h>

#include "CoreTypes.hpp"
#include "frontend/Frontend.hpp"
#include "hal/Synchronizer.hpp"
#include "pool_slam/Hal.hpp"

#include "TestFrontendHelpers.hpp"

namespace PoolSLAM {

class FrontendTest : public ::testing::Test {
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

    std::string vocabPath;
    FrontendConfig config;
    std::vector<uint8_t> baseSceneBuffer;
    std::vector<uint8_t> blankSceneBuffer;

    void SetUp() override {
        vocabPath = test_helpers::writeTempVocabFile();
        config = test_helpers::makeFrontendConfig(kImageWidth, kImageHeight);

        baseSceneBuffer = test_helpers::renderProjectedGridScene(
            kImageWidth, kImageHeight, config.K
        );
        blankSceneBuffer.assign(kImageWidth * kImageHeight, test_helpers::kDefaultBg);

        synchronizer = std::make_unique<Synchronizer>(
            imuStream, depthStream, imageStream, kQueueCapacity
        );
        frontend = std::make_unique<Frontend>(config, *synchronizer, vocabPath);
    }

    void TearDown() override {
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

    // Helper to pop a BackendHfPacket with a timeout guard so tests never hang on failure
    std::unique_ptr<BackendHfPacket> popHfPacketWithTimeout(
        Frontend &fe,
        std::chrono::milliseconds timeout = std::chrono::milliseconds(400)) {
        auto fut = std::async(std::launch::async, [&fe]() {
            return fe.getHfPacket();
        });
        if (fut.wait_for(timeout) != std::future_status::ready) {
            return nullptr;
        }
        return fut.get();
    }

    // Accessors for private Frontend state via friend class FrontendTest
    size_t getHfQueueSize(Frontend &fe) {
        std::lock_guard<std::mutex> lock(fe.m_hfQueue);
        return fe.hfQueue.size();
    }

    bool hasProcessedFrameId(Frontend &fe, uint32_t targetFrameId) {
        std::lock_guard<std::mutex> lock(fe.m_hfQueue);
        return !fe.hfQueue.empty() && fe.hfQueue.back()->frameId >= targetFrameId;
    }

    gtsam::imuBias::ConstantBias getImuBias(Frontend &fe) {
        std::lock_guard<std::mutex> lock(fe.m_imuBias);
        return fe.imuBias;
    }

    uint32_t getCurrentFrameId(const Frontend &fe) const {
        return fe.frameId;
    }
};

// ============================================================================
// 1. First Frame (T0) Anchor & Feature Replenishment Test
// ============================================================================

// Verifies that T0 initializes frameId=0, is flagged as a keyframe, populates replenished
// visual measurements with undistorted coordinates, interpolates depth, and has no IMU factor.
TEST_F(FrontendTest, FirstFrameAnchorT0ReplenishesFeaturesWithoutImuDelta) {
    synchronizer->start();
    frontend->start();

    // Provide depth bracketing T0 = 1.020s (with strictly newer sample at 1.050s)
    depthStream.pushDepth({1.000, 10.0});
    depthStream.pushDepth({1.050, 15.0});

    ASSERT_TRUE(imageStream.pushFrame(baseSceneBuffer.data(), 1.020));

    auto pkt0 = popHfPacketWithTimeout(*frontend);
    ASSERT_NE(pkt0, nullptr);

    EXPECT_EQ(pkt0->frameId, 0u);
    EXPECT_DOUBLE_EQ(pkt0->timestamp, 1.020);
    EXPECT_TRUE(pkt0->isKeyframe);

    // T0 has no preceding frame; imuDelta must be nullptr
    EXPECT_EQ(pkt0->imuDelta, nullptr);

    // Depth interpolated at t = 1.020 between (1.000, 10.0) and (1.050, 15.0) -> 12.0m
    ASSERT_TRUE(pkt0->depth.has_value());
    EXPECT_NEAR(*pkt0->depth, 12.0, 1e-4);

    // ReplenishBuckets must have populated initial landmarks on T0
    EXPECT_GT(pkt0->visualMeasurements.size(), 0u);
    for (size_t i = 0; i < pkt0->visualMeasurements.size(); i++) {
        EXPECT_TRUE(std::isfinite(pkt0->visualMeasurements[i].pt.x()));
        EXPECT_TRUE(std::isfinite(pkt0->visualMeasurements[i].pt.y()));
    }
}

// ============================================================================
// 2. Multi-Rate Nominal Streaming Test
// ============================================================================

// Verifies end-to-end streaming across Camera (30 Hz), IMU (200 Hz), and Depth (20 Hz)
// with continuous feature tracking, IMU preintegration, and image buffer recycling.
TEST_F(FrontendTest, NominalMultiRateStreamingAndPreintegration) {
    synchronizer->start();
    frontend->start();

    std::atomic<bool> runProducers{true};
    const auto startWall = std::chrono::steady_clock::now();

    auto sleepUntilSimTime = [&startWall](double simTime) {
        const auto offset = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
            std::chrono::duration<double>(simTime - 1.000));
        std::this_thread::sleep_until(startWall + offset);
    };

    // Producer 1: IMU @ 200 Hz (dt = 5 ms)
    std::thread imuThread([&]() {
        double t = 1.000;
        while (runProducers.load(std::memory_order_relaxed)) {
            sleepUntilSimTime(t);
            imuStream.pushImu(test_helpers::createImuSample(
                t, Eigen::Vector3d(std::sin(t), 0.0, -9.81), Eigen::Vector3d::Zero()));
            t += 0.005;
        }
    });

    // Producer 2: Depth @ 20 Hz (dt = 50 ms)
    std::thread depthThread([&]() {
        double t = 1.000;
        while (runProducers.load(std::memory_order_relaxed)) {
            sleepUntilSimTime(t);
            depthStream.pushDepth({t, 8.5});
            t += 0.050;
        }
    });

    // Producer 3: Camera @ 30 Hz (dt = 33 ms) streaming until consumer finishes
    constexpr size_t kTotalFrames = 15;
    std::thread cameraThread([&]() {
        double t = 1.020;
        while (runProducers.load(std::memory_order_relaxed)) {
            sleepUntilSimTime(t);
            if (!runProducers.load(std::memory_order_relaxed)) {
                break;
            }
            imageStream.pushFrame(baseSceneBuffer.data(), t);
            t += 0.033;
        }
    });

    std::vector<std::unique_ptr<BackendHfPacket>> packets;
    double lastTimestamp = -1.0;

    for (size_t i = 0; i < kTotalFrames; i++) {
        auto pkt = popHfPacketWithTimeout(*frontend, std::chrono::milliseconds(400));
        if (!pkt) {
            break;
        }
        EXPECT_EQ(pkt->frameId, static_cast<uint32_t>(i));
        EXPECT_GT(pkt->timestamp, lastTimestamp);
        lastTimestamp = pkt->timestamp;

        if (i == 0) {
            EXPECT_TRUE(pkt->isKeyframe);
            EXPECT_EQ(pkt->imuDelta, nullptr);
        } else {
            ASSERT_NE(pkt->imuDelta, nullptr);
            EXPECT_NEAR(pkt->imuDelta->deltaTij(), 0.033, 1e-3);
        }

        EXPECT_TRUE(pkt->depth.has_value());
        EXPECT_GT(pkt->visualMeasurements.size(), 10u);
        packets.push_back(std::move(pkt));
    }

    runProducers.store(false, std::memory_order_relaxed);
    cameraThread.join();
    imuThread.join();
    depthThread.join();

    EXPECT_EQ(packets.size(), kTotalFrames);
}

// ============================================================================
// 3. Asynchronous Bus Jitter and Transit Delay
// ============================================================================

// Verifies Frontend processes packets accurately when IMU and Depth streams experience random jitter
TEST_F(FrontendTest, AsynchronousBusJitter) {
    synchronizer->start();
    frontend->start();

    std::atomic<bool> runProducers{true};

    std::thread imuThread([&]() {
        std::mt19937 rng(42);
        std::uniform_int_distribution<int> jitterDist(2, 8);
        double t = 1.000;
        while (runProducers.load(std::memory_order_relaxed)) {
            imuStream.pushImu(test_helpers::createImuSample(t));
            t += 0.005;
            std::this_thread::sleep_for(std::chrono::milliseconds(jitterDist(rng)));
        }
    });

    std::thread depthThread([&]() {
        std::mt19937 rng(84);
        std::uniform_int_distribution<int> jitterDist(35, 65);
        double t = 1.000;
        while (runProducers.load(std::memory_order_relaxed)) {
            depthStream.pushDepth({t, 6.0});
            t += 0.050;
            std::this_thread::sleep_for(std::chrono::milliseconds(jitterDist(rng)));
        }
    });

    constexpr size_t kNumFrames = 8;
    std::thread cameraThread([&]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(60));
        double camTime = 1.040;
        while (runProducers.load(std::memory_order_relaxed)) {
            imageStream.pushFrame(baseSceneBuffer.data(), camTime);
            camTime += 0.033;
            std::this_thread::sleep_for(std::chrono::milliseconds(33));
        }
    });

    size_t receivedCount = 0;
    for (size_t i = 0; i < kNumFrames; i++) {
        auto pkt = popHfPacketWithTimeout(*frontend, std::chrono::milliseconds(400));
        if (pkt) {
            EXPECT_EQ(pkt->frameId, static_cast<uint32_t>(i));
            receivedCount++;
        }
    }

    runProducers.store(false, std::memory_order_relaxed);
    cameraThread.join();
    imuThread.join();
    depthThread.join();

    EXPECT_EQ(receivedCount, kNumFrames);
}

// ============================================================================
// 4. Keyframe Selection Policies (Time Threshold & Dropped Feature Threshold)
// ============================================================================

// Verifies keyframes trigger strictly when maxTimeBetweenKf elapses under steady tracking
TEST_F(FrontendTest, KeyframeTriggersOnMaxTimeThreshold) {
    // Set maxTimeBetweenKf = 0.10s
    FrontendConfig kfCfg = test_helpers::makeFrontendConfig(kImageWidth, kImageHeight, 0.10, 0.5f, 10);
    Frontend kfFrontend(kfCfg, *synchronizer, vocabPath);

    test_helpers::pushInertialAndDepthWindow(imuStream, depthStream, 1.000, 1.300, 4.0);

    synchronizer->start();
    kfFrontend.start();

    // Frame 0 at t = 1.020s -> Initial Keyframe (lastKfTime = 1.020)
    ASSERT_TRUE(imageStream.pushFrame(baseSceneBuffer.data(), 1.020));
    auto p0 = popHfPacketWithTimeout(kfFrontend);
    ASSERT_NE(p0, nullptr);
    EXPECT_TRUE(p0->isKeyframe);

    // Frame 1 at t = 1.060s (dt = 0.040s <= 0.10s, 0 features dropped) -> Non-keyframe
    ASSERT_TRUE(imageStream.pushFrame(baseSceneBuffer.data(), 1.060));
    auto p1 = popHfPacketWithTimeout(kfFrontend);
    ASSERT_NE(p1, nullptr);
    EXPECT_FALSE(p1->isKeyframe);

    // Frame 2 at t = 1.100s (dt = 0.080s <= 0.10s, 0 features dropped) -> Non-keyframe
    ASSERT_TRUE(imageStream.pushFrame(baseSceneBuffer.data(), 1.100));
    auto p2 = popHfPacketWithTimeout(kfFrontend);
    ASSERT_NE(p2, nullptr);
    EXPECT_FALSE(p2->isKeyframe);

    // Frame 3 at t = 1.140s (1.140 - 1.020 = 0.120s > 0.10s) -> Keyframe triggered by time!
    ASSERT_TRUE(imageStream.pushFrame(baseSceneBuffer.data(), 1.140));
    auto p3 = popHfPacketWithTimeout(kfFrontend);
    ASSERT_NE(p3, nullptr);
    EXPECT_TRUE(p3->isKeyframe);

    kfFrontend.end();
}

// Verifies keyframes trigger immediately when tracked features drop below droppedFeatureThreshold
TEST_F(FrontendTest, KeyframeTriggersOnDroppedFeatureThreshold) {
    // Set maxTimeBetweenKf high (10.0s) to isolate the dropped-feature trigger
    FrontendConfig kfCfg = test_helpers::makeFrontendConfig(kImageWidth, kImageHeight, 10.0, 0.5f, 10);
    Frontend kfFrontend(kfCfg, *synchronizer, vocabPath);

    test_helpers::pushInertialAndDepthWindow(imuStream, depthStream, 1.000, 1.250, 4.0);

    synchronizer->start();
    kfFrontend.start();

    // Frame 0 at t = 1.020s: Rich scene (30 corners) -> Keyframe
    ASSERT_TRUE(imageStream.pushFrame(baseSceneBuffer.data(), 1.020));
    auto p0 = popHfPacketWithTimeout(kfFrontend);
    ASSERT_NE(p0, nullptr);
    EXPECT_TRUE(p0->isKeyframe);
    EXPECT_GE(p0->visualMeasurements.size(), 20u);

    // Frame 1 at t = 1.050s: Same rich scene, all features survive -> Non-keyframe
    ASSERT_TRUE(imageStream.pushFrame(baseSceneBuffer.data(), 1.050));
    auto p1 = popHfPacketWithTimeout(kfFrontend);
    ASSERT_NE(p1, nullptr);
    EXPECT_FALSE(p1->isKeyframe);

    // Frame 2 at t = 1.080s: Blank/occluded frame drops all tracked features (< 0.5 * lastKfFeatures)
    ASSERT_TRUE(imageStream.pushFrame(blankSceneBuffer.data(), 1.080));
    auto p2 = popHfPacketWithTimeout(kfFrontend);
    ASSERT_NE(p2, nullptr);
    EXPECT_TRUE(p2->isKeyframe);
    EXPECT_EQ(p2->visualMeasurements.size(), 0u);

    kfFrontend.end();
}

// ============================================================================
// 5. Underwater VIDO to Monocular VIO Graceful Degradation
// ============================================================================

// Verifies Frontend transitions to std::nullopt depth when DepthStream times out, without dropping frames
TEST_F(FrontendTest, DepthDropGracefulDegradationToMonocularVIO) {
    synchronizer->start();
    frontend->start();

    std::atomic<bool> runImu{true};
    std::thread imuThread([&]() {
        double t = 1.000;
        while (runImu.load(std::memory_order_relaxed)) {
            imuStream.pushImu(test_helpers::createImuSample(t));
            t += 0.005;
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    });

    // Provide depth samples only up to t = 1.150s
    for (size_t i = 0; i < 4; i++) {
        depthStream.pushDepth({1.000 + static_cast<double>(i) * 0.050, 14.0});
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(30));

    // Frame 0 at t = 1.050s (within depth coverage)
    ASSERT_TRUE(imageStream.pushFrame(baseSceneBuffer.data(), 1.050));
    auto p0 = popHfPacketWithTimeout(*frontend);
    ASSERT_NE(p0, nullptr);
    ASSERT_TRUE(p0->depth.has_value());
    EXPECT_NEAR(*p0->depth, 14.0, 1e-4);

    // Wait for IMU thread to advance past 1.250s while pushing no new depth data
    std::this_thread::sleep_for(std::chrono::milliseconds(240));

    // Frame 1 at t = 1.220s (past depth coverage; depthStream.waitUntil(1.220) times out)
    ASSERT_TRUE(imageStream.pushFrame(baseSceneBuffer.data(), 1.220));
    auto p1 = popHfPacketWithTimeout(*frontend, std::chrono::milliseconds(400));
    ASSERT_NE(p1, nullptr);
    EXPECT_EQ(p1->frameId, 1u);
    EXPECT_FALSE(p1->depth.has_value());
    EXPECT_NE(p1->imuDelta, nullptr);
    EXPECT_GT(p1->visualMeasurements.size(), 0u);

    runImu.store(false, std::memory_order_relaxed);
    imuThread.join();
}

// ============================================================================
// 6. Dynamic IMU Bias Update (setBias)
// ============================================================================

// Verifies setBias thread-safely updates the bias embedded into subsequent IMU preintegrators
TEST_F(FrontendTest, SetBiasUpdatesSubsequentPreintegratedImuMeasurements) {
    test_helpers::pushInertialAndDepthWindow(imuStream, depthStream, 1.000, 1.250, 5.0);

    synchronizer->start();
    frontend->start();

    // Dispatch T0
    ASSERT_TRUE(imageStream.pushFrame(baseSceneBuffer.data(), 1.020));
    auto p0 = popHfPacketWithTimeout(*frontend);
    ASSERT_NE(p0, nullptr);

    // Update IMU bias from Backend before pushing T1
    const Eigen::Vector3d accelBias(0.15, -0.08, 0.22);
    const Eigen::Vector3d gyroBias(0.01, -0.02, 0.005);
    const gtsam::imuBias::ConstantBias newBias(accelBias, gyroBias);

    EXPECT_TRUE(frontend->setBias(newBias));
    EXPECT_TRUE(getImuBias(*frontend).accelerometer().isApprox(accelBias, 1e-9));
    EXPECT_TRUE(getImuBias(*frontend).gyroscope().isApprox(gyroBias, 1e-9));

    // Dispatch T1 and verify the preintegrator inside BackendHfPacket carries newBias
    ASSERT_TRUE(imageStream.pushFrame(baseSceneBuffer.data(), 1.060));
    auto p1 = popHfPacketWithTimeout(*frontend);
    ASSERT_NE(p1, nullptr);
    ASSERT_NE(p1->imuDelta, nullptr);
    EXPECT_TRUE(p1->imuDelta->biasHat().accelerometer().isApprox(accelBias, 1e-7));
    EXPECT_TRUE(p1->imuDelta->biasHat().gyroscope().isApprox(gyroBias, 1e-7));
}

// ============================================================================
// 7. Bounded hfQueue Backpressure Eviction & Image Pool Recycle Audit
// ============================================================================

// Verifies that when hfQueue overflows, oldest packets are evicted while raw images are still recycled
TEST_F(FrontendTest, HfQueueBackpressureEvictsOldestPacketsAndRecyclesImages) {
    constexpr size_t kBoundedHfQueueSize = 3;
    FrontendConfig boundedCfg = test_helpers::makeFrontendConfig(
        kImageWidth, kImageHeight, 1.0, 0.5f, kBoundedHfQueueSize);
    Frontend boundedFrontend(boundedCfg, *synchronizer, vocabPath);

    test_helpers::pushInertialAndDepthWindow(imuStream, depthStream, 1.000, 1.600, 5.0);

    synchronizer->start();
    boundedFrontend.start();

    // Pace 7 pushes at 75 ms (< 100 ms ImageStream timeout) so ImageStream remains ACTIVE
    // while boundedFrontend processes all 7 frames without calling getHfPacket().
    constexpr size_t kFramesPushed = 7;
    for (size_t i = 0; i < kFramesPushed; i++) {
        const double t = 1.020 + static_cast<double>(i) * 0.040;
        EXPECT_TRUE(imageStream.pushFrame(baseSceneBuffer.data(), t));
        if (i + 1 < kFramesPushed) {
            std::this_thread::sleep_for(std::chrono::milliseconds(75));
        }
    }

    // Wait until Frontend has processed all 7 frames (up to frameId == 6)
    const auto waitStart = std::chrono::steady_clock::now();
    while (!hasProcessedFrameId(boundedFrontend, static_cast<uint32_t>(kFramesPushed - 1)) &&
           std::chrono::steady_clock::now() - waitStart < std::chrono::milliseconds(1500)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    ASSERT_TRUE(hasProcessedFrameId(boundedFrontend, static_cast<uint32_t>(kFramesPushed - 1)));
    EXPECT_EQ(getHfQueueSize(boundedFrontend), kBoundedHfQueueSize);

    // Pop the 3 retained packets: they must be the newest frames (frameIds 4, 5, 6)
    for (size_t i = 0; i < kBoundedHfQueueSize; i++) {
        auto pkt = popHfPacketWithTimeout(boundedFrontend);
        ASSERT_NE(pkt, nullptr);
        EXPECT_EQ(pkt->frameId, static_cast<uint32_t>(kFramesPushed - kBoundedHfQueueSize + i));
    }

    // Verify ImageStream pool was not starved by unconsumed BackendHfPackets
    EXPECT_TRUE(imageStream.pushFrame(baseSceneBuffer.data(), 1.500));
    auto pktAfterDrain = popHfPacketWithTimeout(boundedFrontend);
    ASSERT_NE(pktAfterDrain, nullptr);
    EXPECT_EQ(pktAfterDrain->frameId, static_cast<uint32_t>(kFramesPushed));

    boundedFrontend.end();
}

// ============================================================================
// 8. Fail-Safe Pipeline Termination on Upstream Sensor Failure
// ============================================================================

// Verifies that when ImageStream fails (4 consecutive timeouts -> DISABLED) and Synchronizer
// halts, Frontend exits processPacket() cleanly and unblocks getHfPacket() with nullptr.
TEST_F(FrontendTest, UpstreamFatalCameraFailureUnblocksGetHfPacketCleanly) {
    synchronizer->start();
    frontend->start();

    std::atomic<bool> runImu{true};
    std::thread imuThread([&]() {
        double t = 1.000;
        while (runImu.load(std::memory_order_relaxed)) {
            imuStream.pushImu(test_helpers::createImuSample(t));
            t += 0.005;
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    });

    // Push zero camera frames: ImageStream latches DISABLED on the 4th timeout (~400 ms),
    // Synchronizer transitions to FATALCAM, and Frontend must unblock getHfPacket().
    auto pkt = popHfPacketWithTimeout(*frontend, std::chrono::milliseconds(700));
    EXPECT_EQ(pkt, nullptr);

    // Subsequent calls to getHfPacket() must return nullptr immediately without blocking
    auto start = std::chrono::steady_clock::now();
    auto immediateNull = popHfPacketWithTimeout(*frontend, std::chrono::milliseconds(200));
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start
    ).count();

    EXPECT_EQ(immediateNull, nullptr);
    EXPECT_LT(duration, 50);

    runImu.store(false, std::memory_order_relaxed);
    imuThread.join();
}

// ============================================================================
// 9. Lifecycle: Start, End, and Restart
// ============================================================================

// Verifies Frontend stops cleanly, unblocks waiting consumers, and resets frameId on restart
TEST_F(FrontendTest, LifecycleStartEndRestartResetsFrameIdAndState) {
    // Session 1
    test_helpers::pushInertialAndDepthWindow(imuStream, depthStream, 1.000, 1.150, 3.0);
    synchronizer->start();
    frontend->start();

    ASSERT_TRUE(imageStream.pushFrame(baseSceneBuffer.data(), 1.020));
    auto p0 = popHfPacketWithTimeout(*frontend);
    ASSERT_NE(p0, nullptr);
    EXPECT_EQ(p0->frameId, 0u);

    ASSERT_TRUE(imageStream.pushFrame(baseSceneBuffer.data(), 1.060));
    auto p1 = popHfPacketWithTimeout(*frontend);
    ASSERT_NE(p1, nullptr);
    EXPECT_EQ(p1->frameId, 1u);

    frontend->end();
    synchronizer->stop();

    // Session 2: Restart on a new mission timeline (t = 50.0s)
    test_helpers::pushInertialAndDepthWindow(imuStream, depthStream, 50.000, 50.150, 9.0);
    synchronizer->start();
    frontend->start();

    ASSERT_TRUE(imageStream.pushFrame(baseSceneBuffer.data(), 50.020));
    auto pRestarted = popHfPacketWithTimeout(*frontend);
    ASSERT_NE(pRestarted, nullptr);

    // frameId must reset to 0 and T0 must have no IMU delta
    EXPECT_EQ(pRestarted->frameId, 0u);
    EXPECT_DOUBLE_EQ(pRestarted->timestamp, 50.020);
    EXPECT_TRUE(pRestarted->isKeyframe);
    EXPECT_EQ(pRestarted->imuDelta, nullptr);
    ASSERT_TRUE(pRestarted->depth.has_value());
    EXPECT_NEAR(*pRestarted->depth, 9.0, 1e-4);
}

} // namespace PoolSLAM