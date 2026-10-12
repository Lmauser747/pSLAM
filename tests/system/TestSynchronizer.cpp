#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <thread>
#include <vector>
#include <utility>

#include <gtest/gtest.h>

#include <eigen3/Eigen/Core>

#include "hal/Synchronizer.hpp"
#include "pool_slam/Hal.hpp"
#include "pool_slam/HalTypes.hpp"

namespace {
constexpr size_t kImageWidth = 640;
constexpr size_t kImageHeight = 480;
constexpr size_t kQueueCapacity = 3;
constexpr size_t kPoolCapacity = 10;
}

class SynchronizerSystemTest : public ::testing::Test {
protected:
    PoolSLAM::ImuStream imuStream;
    PoolSLAM::DepthStream depthStream;
    PoolSLAM::ImageStream imageStream{kImageWidth, kImageHeight, kQueueCapacity, kPoolCapacity};
    std::unique_ptr<PoolSLAM::Synchronizer> synchronizer;

    std::vector<uint8_t> dummyFrameBuffer;

    void SetUp() override {
        dummyFrameBuffer.assign(kImageWidth * kImageHeight, 128);
        synchronizer = std::make_unique<PoolSLAM::Synchronizer>(imuStream, depthStream, imageStream);
    }

    void TearDown() override {
        if (synchronizer) {
            synchronizer->stop();
        }
    }

    PoolSLAM::Imu createImuSample(double t, double ax = 0.0, double gz = 0.0) {
        PoolSLAM::Imu sample;
        sample.timestamp = t;
        sample.a = Eigen::Vector3d(ax, 0.0, -9.81);
        sample.w = Eigen::Vector3d(0.0, 0.0, gz);
        sample.t = 25.0f;
        return sample;
    }
};

// ============================================================================
// 1. Multi-Rate Nominal Streaming Test
// ============================================================================

// Verifies end-to-end ingestion and alignment across Camera (30 Hz), IMU (200 Hz), and Depth (20 Hz)
TEST_F(SynchronizerSystemTest, NominalMultiRateStreaming) {
    synchronizer->start();

    std::atomic<bool> runProducers{true};
    std::atomic<size_t> framesPushed{0};

    // Producer 1: IMU @ 200 Hz (dt = 5 ms)
    std::thread imuThread([&]() {
        double t = 1.000;
        while (runProducers.load(std::memory_order_relaxed)) {
            imuStream.pushImu(createImuSample(t, std::sin(t), 0.05));
            t += 0.005;
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    });

    // Producer 2: Depth @ 20 Hz (dt = 50 ms)
    std::thread depthThread([&]() {
        double t = 1.000;
        while (runProducers.load(std::memory_order_relaxed)) {
            depthStream.pushDepth({t, 10.0 + std::cos(t)});
            t += 0.050;
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    });

    // Producer 3: Camera @ 30 Hz (dt ~ 33.3 ms)
    std::thread cameraThread([&]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(40));
        double t = 1.020;
        for (size_t i = 0; i < 30 && runProducers.load(std::memory_order_relaxed); i++) {
            imageStream.pushFrame(dummyFrameBuffer.data(), t);
            framesPushed.fetch_add(1, std::memory_order_relaxed);
            t += 0.0333;
            std::this_thread::sleep_for(std::chrono::milliseconds(33));
        }
    });

    // Consumer: Front-End tracking loop
    std::vector<PoolSLAM::SyncedPacket> receivedPackets;
    double lastTimestamp = -1.0;

    while (receivedPackets.size() < 30) {
        PoolSLAM::SyncedPacket packet;
        if (synchronizer->getNextPacket(packet, std::chrono::milliseconds(150))) {
            EXPECT_GT(packet.timestamp, lastTimestamp);
            lastTimestamp = packet.timestamp;

            if (receivedPackets.empty()) {
                // Invariant: T0 anchor visual frame has an empty IMU factor batch
                EXPECT_TRUE(packet.imu.empty());
                EXPECT_TRUE(packet.depth.has_value());
            } else {
                // Invariant: Inter-frame IMU batch spans (T_{k-1}, Tk]
                ASSERT_FALSE(packet.imu.empty());
                EXPECT_NEAR(packet.imu.front().timestamp, receivedPackets.back().timestamp, 1e-4);
                EXPECT_NEAR(packet.imu.back().timestamp, packet.timestamp, 1e-4);
                EXPECT_TRUE(packet.depth.has_value());
            }

            ASSERT_NE(packet.image, nullptr);
            EXPECT_DOUBLE_EQ(packet.timestamp, packet.image->timestamp);
            imageStream.recycleFrame(packet.image);
            receivedPackets.push_back(std::move(packet));
        } else if (!runProducers.load(std::memory_order_relaxed)) {
            break;
        }
    }

    runProducers.store(false, std::memory_order_relaxed);
    cameraThread.join();
    imuThread.join();
    depthThread.join();

    EXPECT_GE(receivedPackets.size(), 28);
}

// ============================================================================
// 2. Asynchronous Bus Jitter and Transit Delay
// ============================================================================

// Verifies condition variables block until right brackets arrive under random bus jitter (+-15 ms)
TEST_F(SynchronizerSystemTest, AsynchronousBusJitter) {
    synchronizer->start();

    std::atomic<bool> runProducers{true};

    std::thread imuThread([&]() {
        std::mt19937 rng(42);
        std::uniform_int_distribution<int> jitterDist(2, 8);
        double t = 1.000;
        while (runProducers.load(std::memory_order_relaxed)) {
            imuStream.pushImu(createImuSample(t));
            t += 0.005;
            std::this_thread::sleep_for(std::chrono::milliseconds(jitterDist(rng)));
        }
    });

    std::thread depthThread([&]() {
        std::mt19937 rng(84);
        std::uniform_int_distribution<int> jitterDist(35, 65);
        double t = 1.000;
        while (runProducers.load(std::memory_order_relaxed)) {
            depthStream.pushDepth({t, 5.0});
            t += 0.050;
            std::this_thread::sleep_for(std::chrono::milliseconds(jitterDist(rng)));
        }
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(60));

    // Push 10 visual frames
    double camTime = 1.040;
    for (size_t i = 0; i < 10; i++) {
        imageStream.pushFrame(dummyFrameBuffer.data(), camTime);
        camTime += 0.033;
        std::this_thread::sleep_for(std::chrono::milliseconds(33));
    }

    size_t receivedCount = 0;
    for (size_t i = 0; i < 10; i++) {
        PoolSLAM::SyncedPacket packet;
        if (synchronizer->getNextPacket(packet, std::chrono::milliseconds(250))) {
            receivedCount++;
            ASSERT_NE(packet.image, nullptr);
            imageStream.recycleFrame(packet.image);
        }
    }

    runProducers.store(false, std::memory_order_relaxed);
    imuThread.join();
    depthThread.join();

    EXPECT_EQ(receivedCount, 10);
}

// ============================================================================
// 3. Underwater VIDO to Monocular VIO Graceful Degradation
// ============================================================================

// Verifies pipeline transitions to nullopt depth factors when depth stops, without skipping visual frames
TEST_F(SynchronizerSystemTest, DepthDropGracefulDegradation) {
    synchronizer->start();

    std::atomic<bool> runImu{true};
    std::thread imuThread([&]() {
        double t = 1.000;
        while (runImu.load(std::memory_order_relaxed)) {
            imuStream.pushImu(createImuSample(t));
            t += 0.005;
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    });

    // Provide initial depth measurements up to t = 1.200s
    for (size_t i = 0; i < 4; i++) {
        depthStream.pushDepth({1.000 + (static_cast<double>(i) * 0.050), 12.0});
    }
    depthStream.pushDepth({1.200, 12.0});

    // Allow IMU thread to generate past 1.050s
    std::this_thread::sleep_for(std::chrono::milliseconds(40));

    // Frame 0: Initial T0 anchor with depth
    imageStream.pushFrame(dummyFrameBuffer.data(), 1.050);
    PoolSLAM::SyncedPacket p0;
    ASSERT_TRUE(synchronizer->getNextPacket(p0, std::chrono::milliseconds(150)));
    EXPECT_TRUE(p0.depth.has_value());
    EXPECT_NEAR(p0.depth->z, 12.0, 1e-4);
    imageStream.recycleFrame(p0.image);

    // Frame 1: Intermediate frame with depth (t = 1.100s)
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    imageStream.pushFrame(dummyFrameBuffer.data(), 1.100);
    PoolSLAM::SyncedPacket p1;
    ASSERT_TRUE(synchronizer->getNextPacket(p1, std::chrono::milliseconds(150)));
    EXPECT_TRUE(p1.depth.has_value());
    EXPECT_NEAR(p1.depth->z, 12.0, 1e-4);
    imageStream.recycleFrame(p1.image);

    // Depth halts at 1.200s. Let IMU catch up to 1.260s (30 steps * 5ms ~ 150ms)
    std::this_thread::sleep_for(std::chrono::milliseconds(160));

    // Push visual frame past depth coverage (t = 1.250s > 1.200s)
    imageStream.pushFrame(dummyFrameBuffer.data(), 1.250);
    PoolSLAM::SyncedPacket p2;
    // Timeout must exceed DepthStream's 100ms WAIT_TIMEOUT
    ASSERT_TRUE(synchronizer->getNextPacket(p2, std::chrono::milliseconds(300)));

    // Depth timed out, but visual-inertial frame is preserved
    EXPECT_FALSE(p2.depth.has_value());
    EXPECT_FALSE(p2.imu.empty());
    imageStream.recycleFrame(p2.image);

    runImu.store(false, std::memory_order_relaxed);
    imuThread.join();
}

// ============================================================================
// 4. Fail-Safe Pipeline Termination on IMU Failure
// ============================================================================

// Verifies abrupt IMU loss triggers fail-fast halt, recycles checked-out frames, and unblocks consumer
TEST_F(SynchronizerSystemTest, FatalIMUFailureShutdown) {
    synchronizer->start();

    // 1. Prime pipeline with initial anchor
    imuStream.pushImu(createImuSample(1.000));
    imuStream.pushImu(createImuSample(1.060));
    depthStream.pushDepth({1.000, 5.0});
    depthStream.pushDepth({1.060, 5.0});

    imageStream.pushFrame(dummyFrameBuffer.data(), 1.020);
    PoolSLAM::SyncedPacket p0;
    ASSERT_TRUE(synchronizer->getNextPacket(p0, std::chrono::milliseconds(150)));
    imageStream.recycleFrame(p0.image);

    // 2. Drive ImuStream into DISABLED (> 10 consecutive timeouts = 11 misses)
    for (size_t i = 0; i < 11; i++) {
        imuStream.waitUntil(10.0 + (static_cast<double>(i) * 0.05));
    }
    ASSERT_EQ(imuStream.getHealth(), PoolSLAM::SensorHealth::DISABLED);

    // 3. Push visual frame: Synchronizer calls imuStream.waitUntil(), gets DISABLED immediately,
    //    and executes syncFatalShutdown(SyncStatus::FATALIMU, nextFrame)
    imageStream.pushFrame(dummyFrameBuffer.data(), 1.500);

    // 4. getNextPacket should fail immediately
    PoolSLAM::SyncedPacket p_fail;
    bool success = synchronizer->getNextPacket(p_fail, std::chrono::milliseconds(150));
    EXPECT_FALSE(success);

    // 5. Verify the consumer unblocks immediately (< 50 ms) because isRunning is now false
    auto start = std::chrono::steady_clock::now();
    bool immediateUnblock = synchronizer->getNextPacket(p_fail, std::chrono::milliseconds(200));
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start
    ).count();

    EXPECT_FALSE(immediateUnblock);
    EXPECT_LT(duration, 50);
}

// ============================================================================
// 5. Fail-Safe Pipeline Termination on Camera Failure
// ============================================================================

// Verifies imageStream timeout/disconnection triggers fatal camera failure and stops worker loop
TEST_F(SynchronizerSystemTest, FatalCameraFailureShutdown) {
    synchronizer->start();

    std::atomic<bool> runImu{true};
    std::thread imuThread([&]() {
        double t = 1.000;
        while (runImu.load(std::memory_order_relaxed)) {
            imuStream.pushImu(createImuSample(t));
            t += 0.005;
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    });

    // Zero camera frames pushed; waitForFrame() times out past threshold
    PoolSLAM::SyncedPacket p_fail;
    bool success = synchronizer->getNextPacket(p_fail, std::chrono::milliseconds(350));
    EXPECT_FALSE(success);

    runImu.store(false, std::memory_order_relaxed);
    imuThread.join();
}

// ============================================================================
// 6. Backpressure Eviction and Image Pool Recycle Audit
// ============================================================================

// Verifies that when outputQueue hits max capacity, old frames are evicted and recycled to prevent pool starvation
TEST_F(SynchronizerSystemTest, BackpressureEvictionRecyclesToPool) {
    constexpr size_t kSmallQueueCap = 3;
    PoolSLAM::Synchronizer smallSync(imuStream, depthStream, imageStream, kSmallQueueCap);
    smallSync.start();

    // Pre-populate dense IMU stream
    for (size_t i = 0; i <= 200; i++) {
        imuStream.pushImu(createImuSample(1.000 + (static_cast<double>(i) * 0.005)));
    }
    // Pre-populate dense Depth stream
    for (size_t i = 0; i <= 20; i++) {
        depthStream.pushDepth({1.000 + (static_cast<double>(i) * 0.050), 3.0});
    }

    // Push 8 frames into the pipeline without calling getNextPacket()
    for (size_t i = 1; i <= 8; i++) {
        double t = 1.000 + (static_cast<double>(i) * 0.040);
        bool pushOk = imageStream.pushFrame(dummyFrameBuffer.data(), t);
        EXPECT_TRUE(pushOk);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(60));

    // Drain output queue: bounded capacity retains only the newest packets
    size_t poppedPackets = 0;
    PoolSLAM::SyncedPacket packet;
    while (smallSync.getNextPacket(packet, std::chrono::milliseconds(40))) {
        poppedPackets++;
        ASSERT_NE(packet.image, nullptr);
        imageStream.recycleFrame(packet.image);
    }

    EXPECT_EQ(poppedPackets, kSmallQueueCap);

    // Verify pool is not starved: evicted buffers were recycled, allowing subsequent pushes
    bool canPushAfterDrain = imageStream.pushFrame(dummyFrameBuffer.data(), 2.000);
    EXPECT_TRUE(canPushAfterDrain);

    smallSync.stop();
}

// ============================================================================
// 7. Pipeline Lifecycle: Start, Stop, and Restart
// ============================================================================

// Verifies pipeline restarts cleanly and resets previousFrameTime to reinitialize T0 anchor
TEST_F(SynchronizerSystemTest, LifecycleStartStopRestart) {
    // Session 1: Run and process packets
    synchronizer->start();

    imuStream.pushImu(createImuSample(1.000));
    imuStream.pushImu(createImuSample(1.050));
    depthStream.pushDepth({1.000, 2.0});
    depthStream.pushDepth({1.050, 2.0});

    imageStream.pushFrame(dummyFrameBuffer.data(), 1.020);
    PoolSLAM::SyncedPacket p1;
    ASSERT_TRUE(synchronizer->getNextPacket(p1, std::chrono::milliseconds(150)));
    imageStream.recycleFrame(p1.image);

    synchronizer->stop();

    // Session 2: Restart pipeline on fresh mission timeline
    synchronizer->start();

    // Push fresh frames starting at t = 100.0s
    imuStream.pushImu(createImuSample(100.000));
    imuStream.pushImu(createImuSample(100.050));
    depthStream.pushDepth({100.000, 8.0});
    depthStream.pushDepth({100.050, 8.0});

    imageStream.pushFrame(dummyFrameBuffer.data(), 100.020);

    PoolSLAM::SyncedPacket p_restarted;
    ASSERT_TRUE(synchronizer->getNextPacket(p_restarted, std::chrono::milliseconds(150)));

    // Must re-anchor at T0 without attempting to extract IMU range across [1.020, 100.020]
    EXPECT_DOUBLE_EQ(p_restarted.timestamp, 100.020);
    EXPECT_TRUE(p_restarted.imu.empty());
    imageStream.recycleFrame(p_restarted.image);

    synchronizer->stop();
}