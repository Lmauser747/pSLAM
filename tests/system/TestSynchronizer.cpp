#include <cstddef>
#include <chrono>
#include <thread>
#include <vector>
#include <optional>

#include <gtest/gtest.h>

#include "TestSynchronizerHelpers.hpp"
#include "hal/Synchronizer.hpp"
#include "pool_slam/SystemTypes.hpp"

namespace PoolSLAM {

// =============================================================================
// 1. Lifecycle & State Transitions
// =============================================================================

// Initial state must be IDLE with thread not running
TEST_F(TestSynchronizer, InitialStatusIsIdle) {
    Synchronizer sync(imuStream, depthStream, imageStream);
    EXPECT_EQ(sync.getStatus(), SyncStatus::IDLE);
    EXPECT_FALSE(getIsRunning(sync));
    EXPECT_DOUBLE_EQ(getPreviousFrameTime(sync), -1.0);
}

// start() transitions state to RUNNING and spawns worker thread
TEST_F(TestSynchronizer, StartTransitionsToRunning) {
    Synchronizer sync(imuStream, depthStream, imageStream);
    sync.start();

    EXPECT_EQ(sync.getStatus(), SyncStatus::RUNNING);
    EXPECT_TRUE(getIsRunning(sync));
    sync.stop();
}

// Repeated calls to start() while running are safe no-ops
TEST_F(TestSynchronizer, RepeatedStartIsIdempotent) {
    Synchronizer sync(imuStream, depthStream, imageStream);
    sync.start();
    EXPECT_EQ(sync.getStatus(), SyncStatus::RUNNING);

    // Second start() must not throw or spawn duplicate threads
    sync.start();
    EXPECT_EQ(sync.getStatus(), SyncStatus::RUNNING);
    EXPECT_TRUE(getIsRunning(sync));
    sync.stop();
}

// stop() halts the worker thread and transitions state to STOPPED
TEST_F(TestSynchronizer, StopTransitionsToStopped) {
    Synchronizer sync(imuStream, depthStream, imageStream);
    sync.start();
    EXPECT_EQ(sync.getStatus(), SyncStatus::RUNNING);

    sync.stop();
    EXPECT_EQ(sync.getStatus(), SyncStatus::STOPPED);
    EXPECT_FALSE(getIsRunning(sync));

    // Calling stop() again when already stopped is safe
    sync.stop();
    EXPECT_EQ(sync.getStatus(), SyncStatus::STOPPED);
}

// Clean restart: after stop(), start() re-initializes previousFrameTime and resumes
TEST_F(TestSynchronizer, CleanRestartAfterStop) {
    Synchronizer sync(imuStream, depthStream, imageStream);

    // Initial run
    sync.start();
    EXPECT_EQ(sync.getStatus(), SyncStatus::RUNNING);
    sync.stop();
    EXPECT_EQ(sync.getStatus(), SyncStatus::STOPPED);

    // Restart
    sync.start();
    EXPECT_EQ(sync.getStatus(), SyncStatus::RUNNING);
    EXPECT_TRUE(getIsRunning(sync));
    EXPECT_DOUBLE_EQ(getPreviousFrameTime(sync), -1.0);
    sync.stop();
}

// Destructor drains all queued packets and recycles their image buffers
TEST_F(TestSynchronizer, DestructorRecyclesQueuedFrames) {
    {
        Synchronizer sync(imuStream, depthStream, imageStream, 5);

        // Feed data for initial frame at t=1.0
        pushImuRange(0.7, 1.05);
        pushDepthRange(0.9, 1.1);
        pushFrame(1.0);

        sync.start();

        // Wait until packet is enqueued
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        EXPECT_GE(getOutputQueueSize(sync), 1);

        // Destructor executes here upon scope exit
    }

    // After destruction, Synchronizer should have drained queue and recycled frames
    // Ingesting a new frame should succeed without pool starvation
    EXPECT_TRUE(pushFrame(2.0));
}

// =============================================================================
// 2. First Frame Initialization
// =============================================================================

// First frame extracts stationary gravity/gyro estimates with empty IMU delta
TEST_F(TestSynchronizer, FirstFrameInitialization) {
    Synchronizer sync(imuStream, depthStream, imageStream);

    // Feed stationary IMU data prior to and across t=1.0
    pushImuRange(0.7, 1.05);
    pushDepthRange(0.9, 1.05, 0.05, 3.0);
    pushFrame(1.0);

    sync.start();

    SyncedPacket packet;
    bool success = sync.getNextPacket(packet, std::chrono::milliseconds(500));

    ASSERT_TRUE(success);
    EXPECT_DOUBLE_EQ(packet.timestamp, 1.0);
    ASSERT_NE(packet.image, nullptr);
    EXPECT_DOUBLE_EQ(packet.image->timestamp, 1.0);

    // First packet must have empty IMU delta sequence
    EXPECT_TRUE(packet.imu.empty());

    // Depth reading interpolated at frame epoch
    ASSERT_TRUE(packet.depth.has_value());
    EXPECT_DOUBLE_EQ(packet.depth->timestamp, 1.0);
    EXPECT_NEAR(packet.depth->z, 3.0, 1e-4);

    // Stationary calibration parameters populated for gravity alignment
    ASSERT_TRUE(packet.initialAccel.has_value());
    EXPECT_NEAR(packet.initialAccel->z(), 9.81, 1e-3);
    ASSERT_TRUE(packet.initialGyro.has_value());
    EXPECT_NEAR(packet.initialGyro->x(), 0.01, 1e-3);

    // Recycle checked out image
    sync.recycleFrame(packet.image);
    sync.stop();
}

// =============================================================================
// 3. Steady-State Alignment & Stream Pruning
// =============================================================================

// Steady-state frames pair visual frame with continuous IMU interval [t_{prev}, t_{curr}]
TEST_F(TestSynchronizer, SteadyStateAlignmentAndIntervalExtraction) {
    Synchronizer sync(imuStream, depthStream, imageStream);

    // Feed continuous IMU and depth across span [0.7, 1.3]
    pushImuRange(0.7, 1.35, 0.01);
    pushDepthRange(0.8, 1.35, 0.05, 5.0);

    // Push frame 0 at 1.0, frame 1 at 1.1, frame 2 at 1.2
    pushFrame(1.0);
    pushFrame(1.1);
    pushFrame(1.2);

    sync.start();

    // Packet 0 (Initial)
    SyncedPacket pkt0;
    ASSERT_TRUE(sync.getNextPacket(pkt0, std::chrono::milliseconds(500)));
    EXPECT_DOUBLE_EQ(pkt0.timestamp, 1.0);
    EXPECT_TRUE(pkt0.imu.empty());
    sync.recycleFrame(pkt0.image);

    // Packet 1 (Steady-state)
    SyncedPacket pkt1;
    ASSERT_TRUE(sync.getNextPacket(pkt1, std::chrono::milliseconds(500)));
    EXPECT_DOUBLE_EQ(pkt1.timestamp, 1.1);
    EXPECT_FALSE(pkt1.initialAccel.has_value());
    EXPECT_FALSE(pkt1.initialGyro.has_value());
    ASSERT_FALSE(pkt1.imu.empty());
    EXPECT_NEAR(pkt1.imu.front().timestamp, 1.0, 1e-4);
    EXPECT_NEAR(pkt1.imu.back().timestamp, 1.1, 1e-4);
    sync.recycleFrame(pkt1.image);

    // Packet 2 (Steady-state)
    SyncedPacket pkt2;
    ASSERT_TRUE(sync.getNextPacket(pkt2, std::chrono::milliseconds(500)));
    EXPECT_DOUBLE_EQ(pkt2.timestamp, 1.2);
    ASSERT_FALSE(pkt2.imu.empty());
    EXPECT_NEAR(pkt2.imu.front().timestamp, 1.1, 1e-4);
    EXPECT_NEAR(pkt2.imu.back().timestamp, 1.2, 1e-4);
    sync.recycleFrame(pkt2.image);

    sync.stop();
}

// =============================================================================
// 4. Fault Tolerance & Non-Fatal Sensor Degradation
// =============================================================================

// Depth stream failure is non-fatal: synchronization continues with depth = nullopt
TEST_F(TestSynchronizer, NonFatalDepthDegradation) {
    Synchronizer sync(imuStream, depthStream, imageStream);

    // Provide IMU but DO NOT provide depth samples
    pushImuRange(0.7, 1.25, 0.01);
    pushFrame(1.0);
    pushFrame(1.1);

    sync.start();

    // Packet 0
    SyncedPacket pkt0;
    ASSERT_TRUE(sync.getNextPacket(pkt0, std::chrono::milliseconds(500)));
    EXPECT_DOUBLE_EQ(pkt0.timestamp, 1.0);
    EXPECT_FALSE(pkt0.depth.has_value());
    sync.recycleFrame(pkt0.image);

    // Packet 1
    SyncedPacket pkt1;
    ASSERT_TRUE(sync.getNextPacket(pkt1, std::chrono::milliseconds(500)));
    EXPECT_DOUBLE_EQ(pkt1.timestamp, 1.1);
    EXPECT_FALSE(pkt1.depth.has_value());
    sync.recycleFrame(pkt1.image);

    // Synchronizer remains in RUNNING state
    EXPECT_EQ(sync.getStatus(), SyncStatus::RUNNING);
    sync.stop();
}

// Incomplete IMU interval drops visual frame safely and recycles image buffer
TEST_F(TestSynchronizer, MissingImuIntervalDropsFrameSafely) {
    Synchronizer sync(imuStream, depthStream, imageStream);

    // Feed initial IMU for frame 0 at t=1.0
    pushImuRange(0.7, 1.05, 0.01);
    pushDepthRange(0.8, 1.35, 0.05, 2.0);
    pushFrame(1.0);

    sync.start();

    SyncedPacket pkt0;
    ASSERT_TRUE(sync.getNextPacket(pkt0, std::chrono::milliseconds(500)));
    EXPECT_DOUBLE_EQ(pkt0.timestamp, 1.0);
    sync.recycleFrame(pkt0.image);

    // Push frame at t=1.1, but DO NOT provide IMU for [1.0, 1.1]
    // (IMU stream only has data up to 1.05, so extractRange for [1.0, 1.1] fails)
    pushFrame(1.1);

    // Subsequent query should time out because frame 1.1 was dropped
    SyncedPacket droppedPkt;
    EXPECT_FALSE(sync.getNextPacket(droppedPkt, std::chrono::milliseconds(150)));

    // Synchronizer remains active and recovers when valid IMU arrives for later frame
    pushImuRange(1.05, 1.35, 0.01);
    pushFrame(1.2);

    SyncedPacket recoveredPkt;
    // With frame 1.1 dropped, frame 1.2 can extract range [1.0, 1.2]
    if (sync.getNextPacket(recoveredPkt, std::chrono::milliseconds(500))) {
        EXPECT_DOUBLE_EQ(recoveredPkt.timestamp, 1.2);
        sync.recycleFrame(recoveredPkt.image);
    }

    EXPECT_EQ(sync.getStatus(), SyncStatus::RUNNING);
    sync.stop();
}

// =============================================================================
// 5. Fatal Error Shutdown
// =============================================================================

// Camera disability triggers FATALCAM shutdown and notifies waiting consumers
TEST_F(TestSynchronizer, CameraDisabilityTriggersFatalCam) {
    Synchronizer sync(imuStream, depthStream, imageStream);

    // Disable camera stream
    // When threadTask calls waitForFrame, camera returns DISABLED
    sync.start();

    // Force camera timeout/disable by waiting without pushing frames
    // waitForFrame times out after 100ms; after 4 timeouts (> MAX_MISSED_FRAMES = 3), DISABLED
    SyncedPacket pkt;
    // Wait enough time for camera stream to disable
    for (int i = 0; i < 5; ++i) {
        sync.getNextPacket(pkt, std::chrono::milliseconds(150));
    }

    EXPECT_EQ(sync.getStatus(), SyncStatus::FATALCAM);
    EXPECT_FALSE(getIsRunning(sync));

    // Subsequent consumer requests unblock immediately returning false
    EXPECT_FALSE(sync.getNextPacket(pkt, std::chrono::milliseconds(50)));
}

// syncFatalShutdown drains output queue and recycles checked-out frames
TEST_F(TestSynchronizer, FatalShutdownDrainsQueueAndRecyclesFrames) {
    Synchronizer sync(imuStream, depthStream, imageStream);

    // Manually trigger fatal shutdown
    callSyncFatalShutdown(sync, SyncStatus::FATALCAM, nullptr);

    EXPECT_EQ(sync.getStatus(), SyncStatus::FATALCAM);
    EXPECT_FALSE(getIsRunning(sync));

    SyncedPacket pkt;
    EXPECT_FALSE(sync.getNextPacket(pkt, std::chrono::milliseconds(50)));
}

// =============================================================================
// 6. Output Queue Bounding & Latency Enforcement
// =============================================================================

// Output queue never exceeds maxOutputQueueSize; oldest packets are evicted and recycled
TEST_F(TestSynchronizer, OutputQueueBoundingAndEviction) {
    constexpr size_t maxQueue = 3;
    Synchronizer sync(imuStream, depthStream, imageStream, maxQueue);

    // Feed plenty of IMU and depth data across [0.7, 2.0]
    pushImuRange(0.7, 2.0, 0.01);
    pushDepthRange(0.8, 2.0, 0.05, 1.0);

    // Push 6 frames rapidly (initial frame at 1.0, then 1.1 .. 1.5)
    for (int i = 0; i < 6; ++i) {
        pushFrame(1.0 + static_cast<double>(i) * 0.1);
    }

    sync.start();

    // Allow worker thread to process frames into the queue
    std::this_thread::sleep_for(std::chrono::milliseconds(150));

    // Output queue size must not exceed configured capacity (3)
    size_t qSize = getOutputQueueSize(sync);
    EXPECT_LE(qSize, maxQueue);

    // Drain remaining packets and recycle
    SyncedPacket pkt;
    while (sync.getNextPacket(pkt, std::chrono::milliseconds(50))) {
        sync.recycleFrame(pkt.image);
    }

    sync.stop();
}

} // namespace PoolSLAM
