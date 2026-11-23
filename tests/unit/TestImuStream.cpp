#include <chrono>
#include <cstddef>
#include <thread>
#include <atomic>
#include <vector>
#include <optional>

#include <gtest/gtest.h>
#include <eigen3/Eigen/Core>

#include "HalTestHelpers.hpp"
#include "pool_slam/Hal.hpp"
#include "pool_slam/HalTypes.hpp"

namespace PoolSLAM {

// =============================================================================
// 1. Ingestion & Buffer Mechanics
// =============================================================================

// Verifies monotonic sequential ingestion into the IMU buffer
TEST_F(TestImuStream, MonotonicSequentialIngestion) {
    ImuStream stream(100);

    for (int i = 1; i <= 5; ++i) {
        double t = static_cast<double>(i) * 0.01;
        EXPECT_TRUE(stream.pushImu(makeImu(t, Eigen::Vector3d(0, 0, 9.81), Eigen::Vector3d::Zero())));
    }

    EXPECT_EQ(getBufferSize(stream), 5);
    const auto& buf = getBuffer(stream);
    EXPECT_DOUBLE_EQ(buf.front().timestamp, 0.01);
    EXPECT_DOUBLE_EQ(buf.back().timestamp, 0.05);
}

// Verifies rejection of samples older than the active logical head
TEST_F(TestImuStream, RejectionOfStaleSamples) {
    ImuStream stream(100);
    stream.pushImu(makeImu(1.0, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()));
    stream.pushImu(makeImu(1.1, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()));

    // Stale sample with timestamp older than head (1.0)
    EXPECT_FALSE(stream.pushImu(makeImu(0.95, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero())));
    EXPECT_EQ(getBufferSize(stream), 2);
}

// Verifies out-of-order in-window sample insertion maintains sorted order
TEST_F(TestImuStream, SortedInsertionOutOfOrderWithinWindow) {
    ImuStream stream(100);
    stream.pushImu(makeImu(1.0, Eigen::Vector3d(1, 0, 0), Eigen::Vector3d::Zero()));
    stream.pushImu(makeImu(1.2, Eigen::Vector3d(3, 0, 0), Eigen::Vector3d::Zero()));

    // Out-of-order sample within window [1.0, 1.2)
    EXPECT_TRUE(stream.pushImu(makeImu(1.1, Eigen::Vector3d(2, 0, 0), Eigen::Vector3d::Zero())));
    EXPECT_EQ(getBufferSize(stream), 3);

    const auto& buf = getBuffer(stream);
    EXPECT_DOUBLE_EQ(buf[0].timestamp, 1.0);
    EXPECT_DOUBLE_EQ(buf[1].timestamp, 1.1);
    EXPECT_DOUBLE_EQ(buf[2].timestamp, 1.2);
    EXPECT_DOUBLE_EQ(buf[1].a.x(), 2.0);
}

// Verifies immediate rejection when sensor health is DISABLED
TEST_F(TestImuStream, RejectionWhenDisabled) {
    ImuStream stream(100);
    setHealth(stream, SensorHealth::DISABLED);

    EXPECT_FALSE(stream.pushImu(makeImu(1.0, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero())));
    EXPECT_EQ(getBufferSize(stream), 0);
}

// =============================================================================
// 2. Range Extraction & Interpolation
// =============================================================================

// Extraction fails if target end time is strictly before start time
TEST_F(TestImuStream, ExtractionFailsWhenEndBeforeStart) {
    ImuStream stream(100);
    populateMonotonic(stream, 10, 1.0, 0.05);

    std::vector<Imu> extracted;
    EXPECT_FALSE(stream.extractRange(1.3, 1.1, extracted));
    EXPECT_TRUE(extracted.empty());
}

// Extraction fails if buffer contains fewer than 2 active samples
TEST_F(TestImuStream, ExtractionFailsWhenInsufficientSamples) {
    ImuStream stream(100);
    std::vector<Imu> extracted;

    // 0 samples
    EXPECT_FALSE(stream.extractRange(1.0, 2.0, extracted));

    // 1 sample
    stream.pushImu(makeImu(1.0, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()));
    EXPECT_FALSE(stream.extractRange(1.0, 2.0, extracted));
}

// Extraction fails when requested range extends outside available buffer span
TEST_F(TestImuStream, ExtractionFailsOnUnderRunOrOverRun) {
    ImuStream stream(100);
    // Span: [1.0, 1.5]
    populateMonotonic(stream, 6, 1.0, 0.1);

    std::vector<Imu> extracted;
    // Under-run: tStart (0.8) < oldest (1.0)
    EXPECT_FALSE(stream.extractRange(0.8, 1.3, extracted));

    // Over-run: tEnd (1.8) > newest (1.5)
    EXPECT_FALSE(stream.extractRange(1.1, 1.8, extracted));
}

// Exact boundary match within DB_EPSILON (1e-5s) uses boundary samples directly
TEST_F(TestImuStream, ExactBoundaryMatchWithinEpsilon) {
    ImuStream stream(100);
    stream.pushImu(makeImu(1.000000, Eigen::Vector3d(1, 0, 0), Eigen::Vector3d::Zero()));
    stream.pushImu(makeImu(1.050000, Eigen::Vector3d(2, 0, 0), Eigen::Vector3d::Zero()));
    stream.pushImu(makeImu(1.100000, Eigen::Vector3d(3, 0, 0), Eigen::Vector3d::Zero()));

    std::vector<Imu> extracted;
    // Query within 1e-6 (< DB_EPSILON = 1e-5) of boundaries
    EXPECT_TRUE(stream.extractRange(1.000002, 1.099998, extracted));
    ASSERT_GE(extracted.size(), 2);

    EXPECT_DOUBLE_EQ(extracted.front().timestamp, 1.000002);
    EXPECT_NEAR(extracted.front().a.x(), 1.0, 1e-4);
    EXPECT_DOUBLE_EQ(extracted.back().timestamp, 1.099998);
    EXPECT_NEAR(extracted.back().a.x(), 3.0, 1e-4);
}

// Boundary interpolation creates synthetic endpoint samples between raw samples
TEST_F(TestImuStream, BoundaryInterpolationWhenBetweenSamples) {
    ImuStream stream(100);
    stream.pushImu(makeImu(1.0, Eigen::Vector3d(10, 0, 0), Eigen::Vector3d::Zero()));
    stream.pushImu(makeImu(1.1, Eigen::Vector3d(20, 0, 0), Eigen::Vector3d::Zero()));
    stream.pushImu(makeImu(1.2, Eigen::Vector3d(30, 0, 0), Eigen::Vector3d::Zero()));
    stream.pushImu(makeImu(1.3, Eigen::Vector3d(40, 0, 0), Eigen::Vector3d::Zero()));

    std::vector<Imu> extracted;
    // Extract [1.05, 1.25]
    EXPECT_TRUE(stream.extractRange(1.05, 1.25, extracted));
    ASSERT_EQ(extracted.size(), 4);

    // Index 0: interpolated at t=1.05 -> a.x = 15.0
    EXPECT_DOUBLE_EQ(extracted[0].timestamp, 1.05);
    EXPECT_NEAR(extracted[0].a.x(), 15.0, 1e-6);

    // Index 1: raw sample at t=1.1
    EXPECT_DOUBLE_EQ(extracted[1].timestamp, 1.1);
    EXPECT_DOUBLE_EQ(extracted[1].a.x(), 20.0);

    // Index 2: raw sample at t=1.2
    EXPECT_DOUBLE_EQ(extracted[2].timestamp, 1.2);
    EXPECT_DOUBLE_EQ(extracted[2].a.x(), 30.0);

    // Index 3: interpolated at t=1.25 -> a.x = 35.0
    EXPECT_DOUBLE_EQ(extracted[3].timestamp, 1.25);
    EXPECT_NEAR(extracted[3].a.x(), 35.0, 1e-6);
}

// Linear interpolation accuracy across acceleration, angular velocity, and temperature
TEST_F(TestImuStream, InterpolationAccuracyVectorAndScalar) {
    ImuStream stream(10);
    Imu first = makeImu(1.0, Eigen::Vector3d(2, 4, 6), Eigen::Vector3d(0.2, 0.4, 0.6), 20.0f);
    Imu second = makeImu(2.0, Eigen::Vector3d(6, 12, 18), Eigen::Vector3d(0.6, 1.2, 1.8), 30.0f);

    // Midpoint interpolation (alpha = 0.5)
    Imu mid = callLinInter(stream, first, second, 1.5);
    EXPECT_DOUBLE_EQ(mid.timestamp, 1.5);
    EXPECT_TRUE(mid.a.isApprox(Eigen::Vector3d(4, 8, 12)));
    EXPECT_TRUE(mid.w.isApprox(Eigen::Vector3d(0.4, 0.8, 1.2)));
    EXPECT_NEAR(mid.t, 25.0f, 1e-5);
}

// Fallback to first sample when temporal interval is below MIN_DIFF (1e-6s)
TEST_F(TestImuStream, InterpolationFallbackWhenDeltaBelowMinDiff) {
    ImuStream stream(10);
    Imu first = makeImu(1.0, Eigen::Vector3d(1, 2, 3), Eigen::Vector3d(0.1, 0.2, 0.3), 20.0f);
    Imu second = makeImu(1.0 + 5e-7, Eigen::Vector3d(5, 5, 5), Eigen::Vector3d(1, 1, 1), 40.0f);

    // dt < 1e-6 -> returns copy of first with target timestamp
    Imu result = callLinInter(stream, first, second, 1.0 + 2e-7);
    EXPECT_DOUBLE_EQ(result.timestamp, 1.0 + 2e-7);
    EXPECT_TRUE(result.a.isApprox(first.a));
    EXPECT_TRUE(result.w.isApprox(first.w));
    EXPECT_FLOAT_EQ(result.t, first.t);
}

// =============================================================================
// 3. Windowed Acceleration & Angular Velocity Averaging
// =============================================================================

// Averaging on an empty buffer returns std::nullopt
TEST_F(TestImuStream, AverageOnEmptyBufferReturnsNullopt) {
    ImuStream stream(100);
    EXPECT_FALSE(stream.getAverageAcceleration(1.0).has_value());
    EXPECT_FALSE(stream.getAverageAngularVelocity(1.0).has_value());
}

// Windowed averaging computes exact arithmetic mean over the requested horizon
TEST_F(TestImuStream, UniformWindowAveraging) {
    ImuStream stream(100);
    // Push 5 samples over span [0.8, 1.0] with constant acceleration & gyro
    for (int i = 0; i < 5; ++i) {
        double t = 0.8 + static_cast<double>(i) * 0.05;
        stream.pushImu(makeImu(t, Eigen::Vector3d(0, 0, 9.8), Eigen::Vector3d(0.1, 0.2, 0.3)));
    }

    auto avgAccel = stream.getAverageAcceleration(1.0, 0.2);
    ASSERT_TRUE(avgAccel.has_value());
    EXPECT_TRUE(avgAccel->isApprox(Eigen::Vector3d(0, 0, 9.8), 1e-5));

    auto avgGyro = stream.getAverageAngularVelocity(1.0, 0.2);
    ASSERT_TRUE(avgGyro.has_value());
    EXPECT_TRUE(avgGyro->isApprox(Eigen::Vector3d(0.1, 0.2, 0.3), 1e-5));
}

// Sparse fallback returns the single most recent historical sample
TEST_F(TestImuStream, SparseWindowFallback) {
    ImuStream stream(100);
    // Samples at t=0.1, 0.2
    stream.pushImu(makeImu(0.1, Eigen::Vector3d(1, 0, 0), Eigen::Vector3d::Zero()));
    stream.pushImu(makeImu(0.2, Eigen::Vector3d(5, 0, 0), Eigen::Vector3d(0, 1, 0)));

    // Window [0.8, 1.0] is empty -> fallback finds sample at t=0.2
    auto avgAccel = stream.getAverageAcceleration(1.0, 0.2);
    ASSERT_TRUE(avgAccel.has_value());
    EXPECT_DOUBLE_EQ(avgAccel->x(), 5.0);

    auto avgGyro = stream.getAverageAngularVelocity(1.0, 0.2);
    ASSERT_TRUE(avgGyro.has_value());
    EXPECT_DOUBLE_EQ(avgGyro->y(), 1.0);
}

// Averaging returns std::nullopt if all available samples lie in the future
TEST_F(TestImuStream, FailureWhenAllSamplesInFuture) {
    ImuStream stream(100);
    stream.pushImu(makeImu(2.0, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()));
    stream.pushImu(makeImu(2.1, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()));

    EXPECT_FALSE(stream.getAverageAcceleration(1.0, 0.2).has_value());
    EXPECT_FALSE(stream.getAverageAngularVelocity(1.0, 0.2).has_value());
}

// =============================================================================
// 4. Synchronization & Health Monitoring
// =============================================================================

// Producer wakes waiting consumer immediately upon arrival of a newer sample
TEST_F(TestImuStream, ProducerConsumerUnblocksOnPush) {
    ImuStream stream(100);
    std::atomic<bool> consumerReady{false};
    SensorHealth result = SensorHealth::DISABLED;

    std::thread consumer([&]() {
        consumerReady = true;
        result = stream.waitUntil(1.0);
    });

    while (!consumerReady) {
        std::this_thread::yield();
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    EXPECT_TRUE(stream.pushImu(makeImu(1.05, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero())));
    consumer.join();

    EXPECT_EQ(result, SensorHealth::ACTIVE);
    EXPECT_EQ(stream.getHealth(), SensorHealth::ACTIVE);
}

// Consumer times out after WAIT_TIMEOUT (100ms) on lack of data
TEST_F(TestImuStream, ConsumerTimesOutOnLackOfData) {
    ImuStream stream(100);
    auto start = std::chrono::steady_clock::now();
    SensorHealth status = stream.waitUntil(1.0);
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start
    );

    EXPECT_GE(elapsed.count(), 90);
    EXPECT_EQ(status, SensorHealth::DEGRADED);
    EXPECT_EQ(stream.getHealth(), SensorHealth::DEGRADED);
    EXPECT_EQ(getMissedFrames(stream), 1);
}

// Disability transition after 4 consecutive timeouts (> MAX_MISSED_FRAMES = 3)
TEST_F(TestImuStream, DisabilityTransitionAfterMaxMissedFrames) {
    ImuStream stream(100);

    // Fast-forward missed frames to 3 (MAX_MISSED_FRAMES)
    setMissedFrames(stream, 3);
    setHealth(stream, SensorHealth::DEGRADED);

    // 4th timeout: missedFrames becomes 4 (> 3) -> transitions to DISABLED
    SensorHealth status = stream.waitUntil(1.0);
    EXPECT_EQ(status, SensorHealth::DISABLED);
    EXPECT_EQ(stream.getHealth(), SensorHealth::DISABLED);

    // Subsequent calls return DISABLED immediately
    auto start = std::chrono::steady_clock::now();
    SensorHealth immediate = stream.waitUntil(2.0);
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start
    );
    EXPECT_LT(elapsed.count(), 20);
    EXPECT_EQ(immediate, SensorHealth::DISABLED);
}

// Sensor recovers to ACTIVE upon valid data ingestion following degradation
TEST_F(TestImuStream, SensorRecovery) {
    ImuStream stream(100);
    stream.waitUntil(1.0); // Degrades stream
    EXPECT_EQ(stream.getHealth(), SensorHealth::DEGRADED);

    stream.pushImu(makeImu(2.0, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()));
    SensorHealth status = stream.waitUntil(1.5);

    EXPECT_EQ(status, SensorHealth::ACTIVE);
    EXPECT_EQ(stream.getHealth(), SensorHealth::ACTIVE);
    EXPECT_EQ(getMissedFrames(stream), 0);
}

// =============================================================================
// 5. Pruning, Compaction, & Memory Management
// =============================================================================

// Pruning preserves at least one historical sample prior to the horizon
TEST_F(TestImuStream, PruningPreservesOneHistoricalSample) {
    ImuStream stream(100);
    for (int i = 0; i < 5; ++i) {
        stream.pushImu(makeImu(1.0 + static_cast<double>(i) * 0.1, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()));
    }
    // Samples: 1.0, 1.1, 1.2, 1.3, 1.4

    // Prune up to horizon 1.25: first sample > 1.25 is 1.3; head should point to 1.2
    stream.pruneUntil(1.25);
    size_t head = getHead(stream);
    const auto& buf = getBuffer(stream);

    EXPECT_DOUBLE_EQ(buf[head].timestamp, 1.2);
}

// In-place vector compaction triggers when head exceeds compactThreshold
TEST_F(TestImuStream, CompactionTriggerWhenHeadExceedsThreshold) {
    // initSize = 10 -> compactThreshold = 5
    ImuStream stream(10);
    EXPECT_EQ(getCompactThreshold(stream), 5);

    // Push 10 samples: t = 1.0 to 1.9
    for (int i = 0; i < 10; ++i) {
        stream.pushImu(makeImu(1.0 + static_cast<double>(i) * 0.1, Eigen::Vector3d(i, 0, 0), Eigen::Vector3d::Zero()));
    }
    EXPECT_EQ(getBufferSize(stream), 10);
    EXPECT_EQ(getHead(stream), 0);

    // Prune to 1.65: first sample > 1.65 is 1.7 (index 7).
    // Head would advance to index 6. Since 6 > compactThreshold (5), compaction triggers!
    stream.pruneUntil(1.65);

    // After compaction: elements [6, 9] (count 4) shifted to [0, 3], head reset to 0
    EXPECT_EQ(getHead(stream), 0);
    EXPECT_EQ(getBufferSize(stream), 4);

    const auto& buf = getBuffer(stream);
    EXPECT_DOUBLE_EQ(buf[0].timestamp, 1.6);
    EXPECT_DOUBLE_EQ(buf[0].a.x(), 6.0);
    EXPECT_DOUBLE_EQ(buf[3].timestamp, 1.9);
    EXPECT_DOUBLE_EQ(buf[3].a.x(), 9.0);
}

// Flush clears the buffer and resets head to zero
TEST_F(TestImuStream, FlushClearsBufferAndResetsHead) {
    ImuStream stream(100);
    populateMonotonic(stream, 5, 1.0, 0.05);
    EXPECT_EQ(getBufferSize(stream), 5);

    stream.flush();
    EXPECT_EQ(getBufferSize(stream), 0);
    EXPECT_EQ(getHead(stream), 0);
    EXPECT_FALSE(stream.getAverageAcceleration(1.0).has_value());
}

} // namespace PoolSLAM
