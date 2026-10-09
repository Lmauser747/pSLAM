#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "pool_slam/Hal.hpp"
#include "pool_slam/HalTypes.hpp"

namespace {
constexpr double kEps = 1e-6;
}

class ImuStreamTest : public ::testing::Test {
protected:
    PoolSLAM::ImuStream stream;

    static PoolSLAM::Imu createImu(double t, double val = 0.0) {
        PoolSLAM::Imu sample{};
        sample.timestamp = t;
        sample.a = {val, 0.0, 0.0};
        sample.w = {0.0, val, 0.0};
        sample.t = static_cast<float>(val);
        return sample;
    }
};

// ============================================================================
// 1. Ingestion and Buffer Mechanics (pushImu)
// ============================================================================

// Verifies nominal ingestion of a single IMU sample
TEST_F(ImuStreamTest, pushSingleSampleNominal) {
    EXPECT_NO_THROW(stream.pushImu(createImu(1.00, 2.5)));
    EXPECT_EQ(stream.getHealth(), PoolSLAM::SensorHealth::ACTIVE);
}

// Verifies fast-path chronological push under monotonic timestamps
TEST_F(ImuStreamTest, pushMonotonicSequenceFastPath) {
    for (size_t i = 0; i < 50; i++) {
        double t = 1.00 + (static_cast<double>(i) * 0.05);
        double val = 10.0 + static_cast<double>(i);
        EXPECT_NO_THROW(stream.pushImu(createImu(t, val)));
    }

    std::vector<PoolSLAM::Imu> batch;
    ASSERT_TRUE(stream.extractRange(1.50, 1.55, batch));
    ASSERT_GE(batch.size(), 2);
    EXPECT_NEAR(batch.front().timestamp, 1.50, kEps);
    EXPECT_NEAR(batch.front().a.x(), 20.0, kEps);
    EXPECT_NEAR(batch.back().timestamp, 1.55, kEps);
    EXPECT_NEAR(batch.back().a.x(), 21.0, kEps);
}

// Verifies out-of-order delayed packets are sorted chronologically
TEST_F(ImuStreamTest, pushOutOfOrderSortedInsertion) {
    stream.pushImu(createImu(1.00, 10.0));
    stream.pushImu(createImu(1.10, 30.0));
    stream.pushImu(createImu(1.20, 50.0));

    // Delayed packet arrivals strictly greater than head
    stream.pushImu(createImu(1.05, 20.0));
    stream.pushImu(createImu(1.15, 40.0));

    std::vector<PoolSLAM::Imu> batchA;
    ASSERT_TRUE(stream.extractRange(1.00, 1.10, batchA));
    ASSERT_EQ(batchA.size(), 3);
    EXPECT_NEAR(batchA[1].timestamp, 1.05, kEps);
    EXPECT_NEAR(batchA[1].a.x(), 20.0, kEps);

    std::vector<PoolSLAM::Imu> batchB;
    ASSERT_TRUE(stream.extractRange(1.10, 1.20, batchB));
    ASSERT_EQ(batchB.size(), 3);
    EXPECT_NEAR(batchB[1].timestamp, 1.15, kEps);
    EXPECT_NEAR(batchB[1].a.x(), 40.0, kEps);
}

// Verifies out-of-order insertion strictly into the middle of existing samples
TEST_F(ImuStreamTest, pushOutOfOrderMiddleInsertion) {
    stream.pushImu(createImu(1.00, 10.0));
    stream.pushImu(createImu(1.10, 20.0));
    stream.pushImu(createImu(1.20, 30.0));

    // Insert sample into the middle (strictly after the head sample)
    stream.pushImu(createImu(1.05, 15.0));

    std::vector<PoolSLAM::Imu> batch;
    ASSERT_TRUE(stream.extractRange(1.025, 1.075, batch));
    ASSERT_GE(batch.size(), 2);
    EXPECT_NEAR(batch.front().timestamp, 1.025, kEps);
    EXPECT_NEAR(batch.front().a.x(), 12.5, kEps);
    EXPECT_NEAR(batch.back().timestamp, 1.075, kEps);
    EXPECT_NEAR(batch.back().a.x(), 17.5, kEps);
}

// Verifies the internal buffer dynamically grows to accommodate new data without evicting history
TEST_F(ImuStreamTest, pushDynamicExpansionRetainsHistoricalData) {
    constexpr size_t numSamples = 500;
    for (size_t i = 0; i < numSamples; ++i) {
        double t = 1.00 + (static_cast<double>(i) * 0.01);
        stream.pushImu(createImu(t, static_cast<double>(i)));
    }

    // Verify earliest history is preserved and not evicted
    std::vector<PoolSLAM::Imu> earlyBatch;
    ASSERT_TRUE(stream.extractRange(1.01, 1.05, earlyBatch));
    ASSERT_GE(earlyBatch.size(), 2);
    EXPECT_NEAR(earlyBatch.front().timestamp, 1.01, kEps);
    EXPECT_NEAR(earlyBatch.front().a.x(), 1.0, kEps);

    // Verify latest data is preserved
    std::vector<PoolSLAM::Imu> lateBatch;
    ASSERT_TRUE(stream.extractRange(5.90, 5.95, lateBatch));
    ASSERT_GE(lateBatch.size(), 2);
    EXPECT_NEAR(lateBatch.front().timestamp, 5.90, kEps);
}

// Verifies out-of-order insertion dynamically accommodates elements without dropping early history
TEST_F(ImuStreamTest, pushOutOfOrderDynamicExpansionPreservesAll) {
    for (size_t i = 0; i < 50; ++i) {
        double t = 1.00 + (static_cast<double>(i) * 0.10);
        stream.pushImu(createImu(t, static_cast<double>(i) * 10.0));
    }

    // Insert multiple out-of-order samples into the sequence
    stream.pushImu(createImu(1.25, 25.0));
    stream.pushImu(createImu(2.35, 135.0));
    stream.pushImu(createImu(3.45, 245.0));

    // Earliest samples must remain intact
    std::vector<PoolSLAM::Imu> headBatch;
    ASSERT_TRUE(stream.extractRange(1.00, 1.10, headBatch));
    ASSERT_GE(headBatch.size(), 2);
    EXPECT_NEAR(headBatch.front().timestamp, 1.00, kEps);

    // Out-of-order sample at 1.25 must be queryable
    std::vector<PoolSLAM::Imu> midBatch;
    ASSERT_TRUE(stream.extractRange(1.20, 1.30, midBatch));
    ASSERT_EQ(midBatch.size(), 3);
    EXPECT_NEAR(midBatch[1].timestamp, 1.25, kEps);
    EXPECT_NEAR(midBatch[1].a.x(), 25.0, kEps);
}

// Verifies handling of non-positive timestamps without internal numeric corruption
TEST_F(ImuStreamTest, pushTimestampNegativeOrZero) {
    stream.pushImu(createImu(-0.10, 5.0));
    stream.pushImu(createImu(-0.05, 10.0));
    stream.pushImu(createImu(0.00, 15.0));
    stream.pushImu(createImu(0.05, 20.0));

    std::vector<PoolSLAM::Imu> batch;
    ASSERT_TRUE(stream.extractRange(-0.075, -0.025, batch));
    ASSERT_GE(batch.size(), 2);
    EXPECT_NEAR(batch.front().timestamp, -0.075, kEps);
    EXPECT_NEAR(batch.front().a.x(), 7.5, kEps);
    EXPECT_NEAR(batch.back().timestamp, -0.025, kEps);
    EXPECT_NEAR(batch.back().a.x(), 12.5, kEps);
}

// ============================================================================
// 2. Temporal Retrieval and Range Extraction (extractRange)
// ============================================================================

// Verifies query against an empty buffer returns false
TEST_F(ImuStreamTest, extractEmptyBufferReturnsFalse) {
    std::vector<PoolSLAM::Imu> batch;
    EXPECT_FALSE(stream.extractRange(1.00, 1.10, batch));
}

// Verifies query with a single sample returns false due to insufficient bounding bracket
TEST_F(ImuStreamTest, extractSingleSampleReturnsFalse) {
    stream.pushImu(createImu(1.00, 5.0));
    std::vector<PoolSLAM::Imu> batch;
    EXPECT_FALSE(stream.extractRange(1.00, 1.00, batch));
    EXPECT_FALSE(stream.extractRange(1.00, 1.10, batch));
}

// Verifies reverse temporal queries (tStart > tEnd) are rejected
TEST_F(ImuStreamTest, extractReverseTimeQueryRejection) {
    stream.pushImu(createImu(1.00, 10.0));
    stream.pushImu(createImu(1.10, 20.0));

    std::vector<PoolSLAM::Imu> batch;
    EXPECT_FALSE(stream.extractRange(1.10, 1.00, batch));
}

// Verifies exact match on left and right boundaries captures exact samples without duplication
TEST_F(ImuStreamTest, extractExactMatchBoundaries) {
    stream.pushImu(createImu(1.00, 2.0));
    stream.pushImu(createImu(1.10, 4.0));

    std::vector<PoolSLAM::Imu> batch;
    ASSERT_TRUE(stream.extractRange(1.00, 1.10, batch));
    ASSERT_EQ(batch.size(), 2);
    EXPECT_NEAR(batch.front().timestamp, 1.00, kEps);
    EXPECT_NEAR(batch.front().a.x(), 2.0, kEps);
    EXPECT_NEAR(batch.back().timestamp, 1.10, kEps);
    EXPECT_NEAR(batch.back().a.x(), 4.0, kEps);
}

// Verifies linear midpoint math on extracted boundaries
TEST_F(ImuStreamTest, extractMidpointLinearMath) {
    stream.pushImu(createImu(1.00, 10.0));
    stream.pushImu(createImu(1.10, 20.0));

    std::vector<PoolSLAM::Imu> batch;
    ASSERT_TRUE(stream.extractRange(1.025, 1.075, batch));
    ASSERT_EQ(batch.size(), 2);
    EXPECT_NEAR(batch.front().timestamp, 1.025, kEps);
    EXPECT_NEAR(batch.front().a.x(), 12.5, kEps);
    EXPECT_NEAR(batch.back().timestamp, 1.075, kEps);
    EXPECT_NEAR(batch.back().a.x(), 17.5, kEps);
}

// Verifies floating-point interpolation accuracy on non-trivial fractions
TEST_F(ImuStreamTest, extractArbitraryFractionPrecision) {
    stream.pushImu(createImu(1.00, 10.0));
    stream.pushImu(createImu(1.10, 20.0));

    std::vector<PoolSLAM::Imu> batch;
    ASSERT_TRUE(stream.extractRange(1.0333, 1.0667, batch));
    ASSERT_EQ(batch.size(), 2);
    EXPECT_NEAR(batch.front().timestamp, 1.0333, kEps);
    EXPECT_NEAR(batch.front().a.x(), 13.330, kEps);
    EXPECT_NEAR(batch.back().timestamp, 1.0667, kEps);
    EXPECT_NEAR(batch.back().a.x(), 16.670, kEps);
}

// Verifies bracket identification across multiple elements
TEST_F(ImuStreamTest, extractMultiElementCorrectBracketLookup) {
    stream.pushImu(createImu(1.00, 1.0));
    stream.pushImu(createImu(1.10, 2.0));
    stream.pushImu(createImu(1.20, 3.0));
    stream.pushImu(createImu(1.30, 4.0));

    std::vector<PoolSLAM::Imu> batch;
    ASSERT_TRUE(stream.extractRange(1.05, 1.25, batch));

    ASSERT_EQ(batch.size(), 4);
    EXPECT_NEAR(batch[0].timestamp, 1.05, kEps);
    EXPECT_NEAR(batch[0].a.x(), 1.5, kEps);
    EXPECT_NEAR(batch[1].timestamp, 1.10, kEps);
    EXPECT_NEAR(batch[1].a.x(), 2.0, kEps);
    EXPECT_NEAR(batch[2].timestamp, 1.20, kEps);
    EXPECT_NEAR(batch[2].a.x(), 3.0, kEps);
    EXPECT_NEAR(batch[3].timestamp, 1.25, kEps);
    EXPECT_NEAR(batch[3].a.x(), 3.5, kEps);
}

// Verifies query timestamps strictly prior to head return false
TEST_F(ImuStreamTest, extractPastHeadUnderflowProtection) {
    stream.pushImu(createImu(1.00, 5.0));
    stream.pushImu(createImu(1.10, 10.0));

    std::vector<PoolSLAM::Imu> batch;
    EXPECT_FALSE(stream.extractRange(0.999999, 1.05, batch));
}

// Verifies query timestamps strictly past tail return false
TEST_F(ImuStreamTest, extractFutureTailOverflowProtection) {
    stream.pushImu(createImu(1.00, 5.0));
    stream.pushImu(createImu(1.10, 10.0));

    std::vector<PoolSLAM::Imu> batch;
    EXPECT_FALSE(stream.extractRange(1.05, 1.100001, batch));
}

// Verifies identical duplicate timestamps protect against division by zero
TEST_F(ImuStreamTest, extractDuplicateTimestampsZeroDtSafety) {
    stream.pushImu(createImu(1.00, 5.0));
    stream.pushImu(createImu(1.00, 6.0));
    stream.pushImu(createImu(1.10, 10.0));

    std::vector<PoolSLAM::Imu> batch;
    if (stream.extractRange(1.00, 1.05, batch)) {
        for (const auto& sample : batch) {
            EXPECT_FALSE(std::isnan(sample.a.x()));
            EXPECT_FALSE(std::isinf(sample.a.x()));
        }
    }
}

// ============================================================================
// 3. Pruning and Memory Eviction (pruneUntil, flush)
// ============================================================================

// Verifies pruning an empty buffer is a safe no-op
TEST_F(ImuStreamTest, pruneEmptyBufferSafeNoOp) {
    EXPECT_NO_THROW(stream.pruneUntil(10.0));
}

// Verifies horizon older than head leaves all buffer elements intact
TEST_F(ImuStreamTest, pruneUntilHeadPreservesAll) {
    stream.pushImu(createImu(1.00, 10.0));
    stream.pushImu(createImu(1.10, 20.0));
    stream.pushImu(createImu(1.20, 30.0));

    stream.pruneUntil(0.90);

    std::vector<PoolSLAM::Imu> batch;
    ASSERT_TRUE(stream.extractRange(1.00, 1.15, batch));
    ASSERT_GE(batch.size(), 2);
    EXPECT_NEAR(batch.front().timestamp, 1.00, kEps);
    EXPECT_NEAR(batch.front().a.x(), 10.0, kEps);
}

// Verifies pruning at exact head boundary retains valid brackets
TEST_F(ImuStreamTest, pruneExactMatchHead) {
    stream.pushImu(createImu(1.00, 10.0));
    stream.pushImu(createImu(1.10, 20.0));
    stream.pushImu(createImu(1.20, 30.0));

    stream.pruneUntil(1.00);

    std::vector<PoolSLAM::Imu> batch;
    ASSERT_TRUE(stream.extractRange(1.05, 1.15, batch));
    ASSERT_GE(batch.size(), 2);
    EXPECT_NEAR(batch.front().timestamp, 1.05, kEps);
    EXPECT_NEAR(batch.front().a.x(), 15.0, kEps);
}

// Verifies pruning intermediate horizon drops stale samples only
TEST_F(ImuStreamTest, pruneIntermediateHorizonDropsStaleOnly) {
    stream.pushImu(createImu(1.00, 10.0));
    stream.pushImu(createImu(1.10, 20.0));
    stream.pushImu(createImu(1.20, 30.0));
    stream.pushImu(createImu(1.30, 40.0));
    stream.pushImu(createImu(1.40, 50.0));

    stream.pruneUntil(1.25);

    std::vector<PoolSLAM::Imu> staleBatch;
    EXPECT_FALSE(stream.extractRange(1.05, 1.15, staleBatch));

    std::vector<PoolSLAM::Imu> freshBatch;
    ASSERT_TRUE(stream.extractRange(1.30, 1.40, freshBatch));
    ASSERT_GE(freshBatch.size(), 2);
    EXPECT_NEAR(freshBatch.front().timestamp, 1.30, kEps);
    EXPECT_NEAR(freshBatch.front().a.x(), 40.0, kEps);
}

// Verifies pruning beyond tail clears active elements
TEST_F(ImuStreamTest, pruneBeyondTailEmptiesBuffer) {
    stream.pushImu(createImu(1.00, 10.0));
    stream.pushImu(createImu(1.10, 20.0));
    stream.pushImu(createImu(1.20, 30.0));

    stream.pruneUntil(2.00);

    std::vector<PoolSLAM::Imu> batch;
    EXPECT_FALSE(stream.extractRange(1.05, 1.15, batch));
}

// Verifies subsequent range queries succeed after selective prune
TEST_F(ImuStreamTest, pruneSubsequentQueryValidation) {
    stream.pushImu(createImu(1.00, 1.0));
    stream.pushImu(createImu(1.10, 2.0));
    stream.pushImu(createImu(1.20, 3.0));
    stream.pushImu(createImu(1.30, 4.0));

    stream.pruneUntil(1.15);

    std::vector<PoolSLAM::Imu> staleBatch;
    EXPECT_FALSE(stream.extractRange(1.05, 1.10, staleBatch));

    std::vector<PoolSLAM::Imu> validBatch;
    ASSERT_TRUE(stream.extractRange(1.20, 1.30, validBatch));
    ASSERT_EQ(validBatch.size(), 2);
    EXPECT_NEAR(validBatch.front().timestamp, 1.20, kEps);
    EXPECT_NEAR(validBatch.front().a.x(), 3.0, kEps);
    EXPECT_NEAR(validBatch.back().timestamp, 1.30, kEps);
    EXPECT_NEAR(validBatch.back().a.x(), 4.0, kEps);
}

// Verifies flush resets buffer and offsets
TEST_F(ImuStreamTest, flushResetsBuffer) {
    for (size_t i = 0; i < 100; i++) {
        stream.pushImu(createImu(1.00 + (static_cast<double>(i) * 0.05), static_cast<double>(i)));
    }

    stream.flush();

    std::vector<PoolSLAM::Imu> batch;
    EXPECT_FALSE(stream.extractRange(1.50, 2.00, batch));
}

// ============================================================================
// 4. Synchronization and Condition Variables (waitUntil)
// ============================================================================

// Verifies waitUntil returns immediately when data strictly newer than target already exists
TEST_F(ImuStreamTest, waitPredicateAlreadySatisfiedZeroDelay) {
    // Latest sample (1.10) is strictly newer than target (1.05)
    stream.pushImu(createImu(1.10, 10.0));

    auto start = std::chrono::steady_clock::now();
    PoolSLAM::SensorHealth health = stream.waitUntil(1.05);
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start
    ).count();

    EXPECT_EQ(health, PoolSLAM::SensorHealth::ACTIVE);
    EXPECT_LT(duration, 15);
}

// Verifies sample at exact target timestamp does not satisfy waitUntil; data strictly newer is required
TEST_F(ImuStreamTest, waitExactTimestampRequiresStrictlyNewerData) {
    // Pushing sample exactly at target timestamp (1.10)
    stream.pushImu(createImu(1.10, 5.0));

    std::atomic<bool> unblocked{false};
    PoolSLAM::SensorHealth resultHealth = PoolSLAM::SensorHealth::DEGRADED;

    std::thread waiter([&]() {
        resultHealth = stream.waitUntil(1.10);
        unblocked.store(true);
    });

    // Sample exactly at 1.10 must not unblock the waiter
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    EXPECT_FALSE(unblocked.load());

    // Pushing sample strictly newer than target (1.15 > 1.10) unblocks consumer
    stream.pushImu(createImu(1.15, 10.0));
    waiter.join();

    EXPECT_TRUE(unblocked.load());
    EXPECT_EQ(resultHealth, PoolSLAM::SensorHealth::ACTIVE);
}

// Verifies producer unblocks waiting consumer thread upon timely push of strictly newer data
TEST_F(ImuStreamTest, waitProducerWakeupNominal) {
    stream.pushImu(createImu(1.00, 5.0));

    std::thread producer([&]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        // Push sample strictly newer than target 1.10
        stream.pushImu(createImu(1.15, 10.0));
    });

    auto start = std::chrono::steady_clock::now();
    PoolSLAM::SensorHealth health = stream.waitUntil(1.10);
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start
    ).count();

    producer.join();

    EXPECT_EQ(health, PoolSLAM::SensorHealth::ACTIVE);
    EXPECT_GE(duration, 15);
    EXPECT_LT(duration, 90);
}

// Verifies waitUntil times out and returns DEGRADED when data does not arrive
TEST_F(ImuStreamTest, waitInsufficientDataTimesOut) {
    stream.pushImu(createImu(1.00, 5.0));

    auto start = std::chrono::steady_clock::now();
    PoolSLAM::SensorHealth health = stream.waitUntil(1.50);
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start
    ).count();

    EXPECT_EQ(health, PoolSLAM::SensorHealth::DEGRADED);
    EXPECT_GE(duration, 85);
}

// Verifies partial arrival notifies CV without prematurely unblocking wait predicate
TEST_F(ImuStreamTest, waitSpuriousWakeupProtection) {
    stream.pushImu(createImu(1.00, 10.0));
    std::atomic<bool> finishedWait{false};
    PoolSLAM::SensorHealth resultHealth = PoolSLAM::SensorHealth::ACTIVE;

    std::thread waiter([&]() {
        resultHealth = stream.waitUntil(1.15);
        finishedWait.store(true);
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(15));
    // Push sample that advances time but is not newer than target (1.08 < 1.15)
    stream.pushImu(createImu(1.08, 12.0));

    std::this_thread::sleep_for(std::chrono::milliseconds(15));
    EXPECT_FALSE(finishedWait.load());

    // Push sample strictly newer than target (1.16 > 1.15)
    stream.pushImu(createImu(1.16, 15.0));
    waiter.join();

    EXPECT_TRUE(finishedWait.load());
    EXPECT_EQ(resultHealth, PoolSLAM::SensorHealth::ACTIVE);
}

// ============================================================================
// 5. Sensor Health State Machine & Latching FSM
// ============================================================================

// Verifies initial health state is ACTIVE
TEST_F(ImuStreamTest, healthInitialStateActive) {
    EXPECT_EQ(stream.getHealth(), PoolSLAM::SensorHealth::ACTIVE);
}

// Verifies a single timeout transitions health from ACTIVE to DEGRADED
TEST_F(ImuStreamTest, healthSingleTimeoutTransitionsToDegraded) {
    PoolSLAM::SensorHealth health = stream.waitUntil(10.0);
    EXPECT_EQ(health, PoolSLAM::SensorHealth::DEGRADED);
    EXPECT_EQ(stream.getHealth(), PoolSLAM::SensorHealth::DEGRADED);
}

// Verifies up to 10 consecutive timeouts maintain DEGRADED (strictly greater threshold)
TEST_F(ImuStreamTest, healthConsecutiveMissesAccumulate) {
    for (size_t i = 0; i < 10; i++) {
        PoolSLAM::SensorHealth health = stream.waitUntil(10.0 + (static_cast<double>(i) * 0.05));
        EXPECT_EQ(health, PoolSLAM::SensorHealth::DEGRADED);
        EXPECT_EQ(stream.getHealth(), PoolSLAM::SensorHealth::DEGRADED);
    }
}

// Verifies timely arrival of strictly newer data restores health from DEGRADED back to ACTIVE
TEST_F(ImuStreamTest, healthRecoveryDegradedToActive) {
    for (size_t i = 0; i < 5; i++) {
        EXPECT_EQ(stream.waitUntil(10.0 + (static_cast<double>(i) * 0.05)), PoolSLAM::SensorHealth::DEGRADED);
    }
    EXPECT_EQ(stream.getHealth(), PoolSLAM::SensorHealth::DEGRADED);

    // Timely packet arrives strictly newer than wait target 20.00 (20.05 > 20.00)
    stream.pushImu(createImu(20.05, 5.0));
    PoolSLAM::SensorHealth recovered = stream.waitUntil(20.00);

    EXPECT_EQ(recovered, PoolSLAM::SensorHealth::ACTIVE);
    EXPECT_EQ(stream.getHealth(), PoolSLAM::SensorHealth::ACTIVE);
}

// Verifies exceeding 10 misses latches into DISABLED strictly on the 11th miss
TEST_F(ImuStreamTest, healthMaxMissThresholdLatchesDisabled) {
    for (size_t i = 0; i < 10; i++) {
        EXPECT_EQ(stream.waitUntil(1.00 + (static_cast<double>(i) * 0.05)), PoolSLAM::SensorHealth::DEGRADED);
        EXPECT_EQ(stream.getHealth(), PoolSLAM::SensorHealth::DEGRADED);
    }

    // 11th miss crosses strictly greater threshold (> 10)
    PoolSLAM::SensorHealth latched = stream.waitUntil(10.00);
    EXPECT_EQ(latched, PoolSLAM::SensorHealth::DISABLED);
    EXPECT_EQ(stream.getHealth(), PoolSLAM::SensorHealth::DISABLED);
}

// Verifies DISABLED is a strict one-way latch that rejects recovery despite arrival of newer data
TEST_F(ImuStreamTest, healthDisabledOneWayLatch) {
    for (size_t i = 0; i < 11; i++) {
        stream.waitUntil(1.00 + (static_cast<double>(i) * 0.05));
    }
    ASSERT_EQ(stream.getHealth(), PoolSLAM::SensorHealth::DISABLED);

    // Push valid, strictly newer data (50.05 > 50.00)
    stream.pushImu(createImu(50.05, 10.0));
    PoolSLAM::SensorHealth health = stream.waitUntil(50.00);

    EXPECT_EQ(health, PoolSLAM::SensorHealth::DISABLED);
    EXPECT_EQ(stream.getHealth(), PoolSLAM::SensorHealth::DISABLED);
}

// Verifies waitUntil bypasses condition variable delays once DISABLED
TEST_F(ImuStreamTest, healthDisabledFastBypass) {
    for (size_t i = 0; i < 11; i++) {
        stream.waitUntil(1.00 + (static_cast<double>(i) * 0.05));
    }
    ASSERT_EQ(stream.getHealth(), PoolSLAM::SensorHealth::DISABLED);

    auto start = std::chrono::steady_clock::now();
    PoolSLAM::SensorHealth health = stream.waitUntil(100.00);
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start
    ).count();

    EXPECT_EQ(health, PoolSLAM::SensorHealth::DISABLED);
    EXPECT_LT(duration, 5);
}

// ============================================================================
// 6. Concurrency and Multi-Threaded Stress
// ============================================================================

// Verifies thread safety during simultaneous ingestion and range extraction queries
TEST_F(ImuStreamTest, concurrentPushAndExtractRangeDataRace) {
    std::atomic<bool> keepRunning{true};

    // Producer thread: 200 Hz push (dt = 0.005s)
    std::thread producer([&]() {
        double t = 1.00;
        double aVal = 5.0;
        while (keepRunning.load()) {
            stream.pushImu(createImu(t, aVal));
            t += 0.005;
            aVal += 0.01;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    });

    // Consumer thread: extract trailing intervals
    std::thread consumer([&]() {
        double queryTime = 1.05;
        std::vector<PoolSLAM::Imu> batch;
        while (keepRunning.load()) {
            if (stream.extractRange(queryTime - 0.02, queryTime, batch)) {
                for (const auto& sample : batch) {
                    EXPECT_FALSE(std::isnan(sample.a.x()));
                }
                queryTime += 0.005;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    keepRunning.store(false);

    producer.join();
    consumer.join();
}

// Verifies buffer integrity under simultaneous pushing, pruning, and querying
TEST_F(ImuStreamTest, concurrentPushAndPruneHorizonIntegrity) {
    std::atomic<bool> keepRunning{true};

    // Producer thread: monotonic and out-of-order pushes (dt = 0.005s)
    std::thread producer([&]() {
        double t = 1.00;
        while (keepRunning.load()) {
            stream.pushImu(createImu(t, t * 2.0));
            if (std::fmod(t, 0.05) < 0.005) {
                // Delayed packet within middle boundary
                stream.pushImu(createImu(t - 0.002, (t - 0.002) * 2.0));
            }
            t += 0.005;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    });

    // Pruner thread: advancing horizon
    std::thread pruner([&]() {
        double horizon = 1.00;
        while (keepRunning.load()) {
            stream.pruneUntil(horizon);
            horizon += 0.01;
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
    });

    // Consumer thread: querying extracted ranges
    std::thread consumer([&]() {
        double queryTime = 1.05;
        std::vector<PoolSLAM::Imu> batch;
        while (keepRunning.load()) {
            if (stream.extractRange(queryTime - 0.01, queryTime, batch)) {
                for (const auto& sample : batch) {
                    EXPECT_NEAR(sample.a.x(), sample.timestamp * 2.0, 1e-4);
                }
                queryTime += 0.005;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    keepRunning.store(false);

    producer.join();
    pruner.join();
    consumer.join();
}

// Verifies multiple consumer threads safely wake up when strictly newer data arrives
TEST_F(ImuStreamTest, concurrentWaitUntilAndPushRaceCondition) {
    constexpr size_t numWaiters = 4;
    std::vector<std::thread> waiterThreads;
    std::vector<PoolSLAM::SensorHealth> results(numWaiters, PoolSLAM::SensorHealth::DEGRADED);

    stream.pushImu(createImu(1.00, 5.0));

    for (size_t i = 0; i < numWaiters; i++) {
        waiterThreads.emplace_back([&, i]() {
            results[i] = stream.waitUntil(1.10);
        });
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    // Push sample strictly newer than target 1.10 (1.15 > 1.10)
    stream.pushImu(createImu(1.15, 10.0));

    for (size_t i = 0; i < numWaiters; i++) {
        waiterThreads[i].join();
        EXPECT_EQ(results[i], PoolSLAM::SensorHealth::ACTIVE);
    }
}