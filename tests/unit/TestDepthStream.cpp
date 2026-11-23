#include <cstddef>
#include <chrono>
#include <thread>
#include <atomic>
#include <optional>

#include <gtest/gtest.h>

#include "HalTestHelpers.hpp"
#include "pool_slam/Hal.hpp"
#include "pool_slam/HalTypes.hpp"

namespace PoolSLAM {

// =============================================================================
// 1. Ingestion & Buffer Mechanics
// =============================================================================

// Verifies monotonic in-order pushes up to capacity
TEST_F(TestDepthStream, MonotonicPushesUpToCapacity) {
    constexpr size_t capacity = 5;
    DepthStream stream(capacity);

    for (size_t i = 1; i <= capacity; ++i) {
        EXPECT_TRUE(stream.pushDepth(makeDepth(static_cast<double>(i), static_cast<double>(i * 10))));
    }

    EXPECT_EQ(getBufferSize(stream), capacity);
    const auto& buf = getBuffer(stream);
    EXPECT_DOUBLE_EQ(buf.front().timestamp, 1.0);
    EXPECT_DOUBLE_EQ(buf.back().timestamp, 5.0);
}

// Verifies ring buffer eviction of oldest sample when exceeding capacity
TEST_F(TestDepthStream, RingBufferEvictionWhenFull) {
    constexpr size_t capacity = 3;
    DepthStream stream(capacity);

    stream.pushDepth(makeDepth(1.0, 10.0));
    stream.pushDepth(makeDepth(2.0, 20.0));
    stream.pushDepth(makeDepth(3.0, 30.0));
    EXPECT_EQ(getBufferSize(stream), capacity);

    // 4th push should evict sample at t=1.0
    EXPECT_TRUE(stream.pushDepth(makeDepth(4.0, 40.0)));
    EXPECT_EQ(getBufferSize(stream), capacity);

    const auto& buf = getBuffer(stream);
    EXPECT_DOUBLE_EQ(buf.front().timestamp, 2.0);
    EXPECT_DOUBLE_EQ(buf.back().timestamp, 4.0);
}

// Verifies rejection of samples older than the oldest buffer entry
TEST_F(TestDepthStream, RejectionOfStaleSamples) {
    constexpr size_t capacity = 3;
    DepthStream stream(capacity);

    stream.pushDepth(makeDepth(2.0, 20.0));
    stream.pushDepth(makeDepth(3.0, 30.0));
    stream.pushDepth(makeDepth(4.0, 40.0));

    // Stale sample with timestamp older than front (2.0)
    EXPECT_FALSE(stream.pushDepth(makeDepth(1.5, 15.0)));
    EXPECT_EQ(getBufferSize(stream), capacity);
    EXPECT_DOUBLE_EQ(getBuffer(stream).front().timestamp, 2.0);
}

// Verifies out-of-order sample insertion preserves sorted timestamp order
TEST_F(TestDepthStream, SortedInsertionOutOfOrderWithinWindow) {
    constexpr size_t capacity = 5;
    DepthStream stream(capacity);

    stream.pushDepth(makeDepth(1.0, 10.0));
    stream.pushDepth(makeDepth(3.0, 30.0));
    stream.pushDepth(makeDepth(4.0, 40.0));

    // Delayed sample arriving out-of-order within the buffer window
    EXPECT_TRUE(stream.pushDepth(makeDepth(2.0, 20.0)));
    EXPECT_EQ(getBufferSize(stream), 4);

    const auto& buf = getBuffer(stream);
    EXPECT_DOUBLE_EQ(buf[0].timestamp, 1.0);
    EXPECT_DOUBLE_EQ(buf[1].timestamp, 2.0);
    EXPECT_DOUBLE_EQ(buf[2].timestamp, 3.0);
    EXPECT_DOUBLE_EQ(buf[3].timestamp, 4.0);
}

// Verifies out-of-order insertion evicts oldest element if buffer exceeds capacity
TEST_F(TestDepthStream, SortedInsertionEvictsOldestWhenOverCapacity) {
    constexpr size_t capacity = 3;
    DepthStream stream(capacity);

    stream.pushDepth(makeDepth(1.0, 10.0));
    stream.pushDepth(makeDepth(3.0, 30.0));
    stream.pushDepth(makeDepth(4.0, 40.0));

    // Insert out-of-order sample at t=2.0; capacity exceeded -> t=1.0 evicted
    EXPECT_TRUE(stream.pushDepth(makeDepth(2.0, 20.0)));
    EXPECT_EQ(getBufferSize(stream), capacity);

    const auto& buf = getBuffer(stream);
    EXPECT_DOUBLE_EQ(buf[0].timestamp, 2.0);
    EXPECT_DOUBLE_EQ(buf[1].timestamp, 3.0);
    EXPECT_DOUBLE_EQ(buf[2].timestamp, 4.0);
}

// Verifies rejection of pushes when sensor health is DISABLED
TEST_F(TestDepthStream, RejectionWhenDisabled) {
    DepthStream stream(5);
    setHealth(stream, SensorHealth::DISABLED);

    EXPECT_FALSE(stream.pushDepth(makeDepth(1.0, 10.0)));
    EXPECT_EQ(getBufferSize(stream), 0);
}

// =============================================================================
// 2. Interpolation Accuracy & Edge Cases
// =============================================================================

// Querying an empty buffer returns std::nullopt
TEST_F(TestDepthStream, QueryEmptyBufferReturnsNullopt) {
    DepthStream stream;
    EXPECT_FALSE(stream.getInterDepth(1.0).has_value());
}

// Extrapolation beyond newest sample returns std::nullopt
TEST_F(TestDepthStream, ExtrapolationPastNewestReturnsNullopt) {
    DepthStream stream;
    stream.pushDepth(makeDepth(1.0, 10.0));
    stream.pushDepth(makeDepth(2.0, 20.0));

    EXPECT_FALSE(stream.getInterDepth(2.5).has_value());
}

// Extrapolation prior to oldest sample returns std::nullopt
TEST_F(TestDepthStream, ExtrapolationPriorToOldestReturnsNullopt) {
    DepthStream stream;
    stream.pushDepth(makeDepth(2.0, 20.0));
    stream.pushDepth(makeDepth(3.0, 30.0));

    EXPECT_FALSE(stream.getInterDepth(1.5).has_value());
}

// Exact match at sample boundaries returns exact measurement
TEST_F(TestDepthStream, ExactBoundaryMatches) {
    DepthStream stream;
    stream.pushDepth(makeDepth(1.0, 10.0));
    stream.pushDepth(makeDepth(2.0, 20.0));

    auto oldest = stream.getInterDepth(1.0);
    ASSERT_TRUE(oldest.has_value());
    EXPECT_DOUBLE_EQ(oldest->timestamp, 1.0);
    EXPECT_DOUBLE_EQ(oldest->z, 10.0);

    auto newest = stream.getInterDepth(2.0);
    ASSERT_TRUE(newest.has_value());
    EXPECT_DOUBLE_EQ(newest->timestamp, 2.0);
    EXPECT_DOUBLE_EQ(newest->z, 20.0);
}

// Linear interpolation accuracy across multiple alpha fractions
TEST_F(TestDepthStream, LinearInterpolationAccuracy) {
    DepthStream stream;
    // Samples with dt = 0.1s (well within [1e-6, 0.2] range)
    stream.pushDepth(makeDepth(1.0, 10.0));
    stream.pushDepth(makeDepth(1.1, 20.0));

    // alpha = 0.25 -> z = 12.5
    auto d25 = stream.getInterDepth(1.025);
    ASSERT_TRUE(d25.has_value());
    EXPECT_DOUBLE_EQ(d25->timestamp, 1.025);
    EXPECT_NEAR(d25->z, 12.5, 1e-9);

    // alpha = 0.50 -> z = 15.0
    auto d50 = stream.getInterDepth(1.050);
    ASSERT_TRUE(d50.has_value());
    EXPECT_DOUBLE_EQ(d50->timestamp, 1.050);
    EXPECT_NEAR(d50->z, 15.0, 1e-9);

    // alpha = 0.75 -> z = 17.5
    auto d75 = stream.getInterDepth(1.075);
    ASSERT_TRUE(d75.has_value());
    EXPECT_DOUBLE_EQ(d75->timestamp, 1.075);
    EXPECT_NEAR(d75->z, 17.5, 1e-9);
}

// Gap exceeding MAX_DIFF (0.2s) cannot be interpolated
TEST_F(TestDepthStream, GapExceedingMaxDiffReturnsNullopt) {
    DepthStream stream;
    stream.pushDepth(makeDepth(1.0, 10.0));
    stream.pushDepth(makeDepth(1.25, 20.0)); // diff = 0.25s > 0.2s

    EXPECT_FALSE(stream.getInterDepth(1.10).has_value());
}

// Gap below MIN_DIFF (1e-6s) cannot be interpolated
TEST_F(TestDepthStream, GapBelowMinDiffReturnsNullopt) {
    DepthStream stream;
    callFastPush(stream, makeDepth(1.0, 10.0));
    callFastPush(stream, makeDepth(1.0 + 5e-7, 20.0)); // diff = 5e-7 < 1e-6

    EXPECT_FALSE(stream.getInterDepth(1.0 + 2e-7).has_value());
}

// =============================================================================
// 3. Concurrency & Blocking Synchronization
// =============================================================================

// Producer wakes waiting consumer immediately upon sample arrival
TEST_F(TestDepthStream, ProducerConsumerUnblocksOnPush) {
    DepthStream stream;
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

    EXPECT_TRUE(stream.pushDepth(makeDepth(1.0, 5.0)));
    consumer.join();

    EXPECT_EQ(result, SensorHealth::ACTIVE);
    EXPECT_EQ(stream.getHealth(), SensorHealth::ACTIVE);
}

// Consumer times out when no sample arrives within WAIT_TIMEOUT (100ms)
TEST_F(TestDepthStream, ConsumerTimesOutOnEmptyStream) {
    DepthStream stream;
    auto start = std::chrono::steady_clock::now();
    SensorHealth status = stream.waitUntil(1.0);
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start
    );

    EXPECT_GE(elapsed.count(), 90);
    EXPECT_EQ(status, SensorHealth::DEGRADED);
    EXPECT_EQ(stream.getHealth(), SensorHealth::DEGRADED);
}

// First timeout transitions ACTIVE health to DEGRADED
TEST_F(TestDepthStream, HealthTransitionOnTimeout) {
    DepthStream stream;
    EXPECT_EQ(stream.getHealth(), SensorHealth::ACTIVE);

    SensorHealth status = stream.waitUntil(1.0);
    EXPECT_EQ(status, SensorHealth::DEGRADED);
    EXPECT_EQ(stream.getHealth(), SensorHealth::DEGRADED);
    EXPECT_EQ(getMissedFrames(stream), 1);
}

// Transitions to DISABLED after exceeding MAX_MISSED_FRAMES (10 consecutive timeouts)
TEST_F(TestDepthStream, DisabilityTransitionAfterMaxMissedFrames) {
    DepthStream stream;

    // Fast-forward missed frames to 10
    setMissedFrames(stream, 10);
    setHealth(stream, SensorHealth::DEGRADED);

    // 11th timeout: missedFrames becomes 11 (> 10) -> DISABLED
    SensorHealth status = stream.waitUntil(1.0);
    EXPECT_EQ(status, SensorHealth::DISABLED);
    EXPECT_EQ(stream.getHealth(), SensorHealth::DISABLED);

    // Subsequent calls return DISABLED immediately without waiting
    auto start = std::chrono::steady_clock::now();
    SensorHealth immediate = stream.waitUntil(2.0);
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start
    );
    EXPECT_LT(elapsed.count(), 20);
    EXPECT_EQ(immediate, SensorHealth::DISABLED);
}

// Ingesting valid data after degradation restores ACTIVE status
TEST_F(TestDepthStream, RecoveryAfterDegradation) {
    DepthStream stream;

    // Force degradation
    stream.waitUntil(1.0);
    EXPECT_EQ(stream.getHealth(), SensorHealth::DEGRADED);
    EXPECT_GT(getMissedFrames(stream), 0);

    // Push valid data and verify recovery
    stream.pushDepth(makeDepth(2.0, 15.0));
    SensorHealth status = stream.waitUntil(2.0);

    EXPECT_EQ(status, SensorHealth::ACTIVE);
    EXPECT_EQ(stream.getHealth(), SensorHealth::ACTIVE);
    EXPECT_EQ(getMissedFrames(stream), 0);
}

// =============================================================================
// 4. Maintenance
// =============================================================================

// Pruning evicts all samples strictly older than the specified horizon
TEST_F(TestDepthStream, PruneUntilRemovesOlderSamples) {
    DepthStream stream;
    for (int i = 1; i <= 5; ++i) {
        stream.pushDepth(makeDepth(static_cast<double>(i), static_cast<double>(i * 10)));
    }
    EXPECT_EQ(getBufferSize(stream), 5);

    // Prune up to horizon 3.0: evicts t=1.0, 2.0; retains t=3.0, 4.0, 5.0
    stream.pruneUntil(3.0);
    EXPECT_EQ(getBufferSize(stream), 3);

    const auto& buf = getBuffer(stream);
    EXPECT_DOUBLE_EQ(buf.front().timestamp, 3.0);
    EXPECT_DOUBLE_EQ(buf.back().timestamp, 5.0);
}

// Flush clears all samples from the buffer
TEST_F(TestDepthStream, FlushClearsBuffer) {
    DepthStream stream;
    stream.pushDepth(makeDepth(1.0, 10.0));
    stream.pushDepth(makeDepth(2.0, 20.0));
    EXPECT_EQ(getBufferSize(stream), 2);

    stream.flush();
    EXPECT_EQ(getBufferSize(stream), 0);
    EXPECT_FALSE(stream.getInterDepth(1.0).has_value());
}

} // namespace PoolSLAM
