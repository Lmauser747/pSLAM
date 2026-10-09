#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <optional>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "pool_slam/Hal.hpp"
#include "pool_slam/HalTypes.hpp"

namespace {
constexpr double kEps = 1e-6;
}

class DepthStreamTest : public ::testing::Test {
protected:
    PoolSLAM::DepthStream stream;
};

// ============================================================================
// 1. Ingestion and Buffer Mechanics (pushDepth)
// ============================================================================

// Verifies nominal ingestion of a single depth sample
TEST_F(DepthStreamTest, pushSingleSampleNominal) {
    EXPECT_NO_THROW(stream.pushDepth({1.00, 2.5}));
    EXPECT_EQ(stream.getHealth(), PoolSLAM::SensorHealth::ACTIVE);
}

// Verifies fast-path chronological push under monotonic timestamps
TEST_F(DepthStreamTest, pushMonotonicSequenceFastPath) {
    for (size_t i = 0; i < 50; i++) {
        double t = 1.00 + (static_cast<double>(i) * 0.05);
        double depthVal = 10.0 + static_cast<double>(i);
        EXPECT_NO_THROW(stream.pushDepth({t, depthVal}));
    }

    auto midSample = stream.getInterDepth(1.525);
    ASSERT_TRUE(midSample.has_value());
    EXPECT_NEAR(midSample->timestamp, 1.525, kEps);
    EXPECT_NEAR(midSample->z, 20.5, kEps);
}

// Verifies out-of-order delayed packets are sorted chronologically
TEST_F(DepthStreamTest, pushOutOfOrderSortedInsertion) {
    stream.pushDepth({1.00, 10.0});
    stream.pushDepth({1.10, 30.0});
    stream.pushDepth({1.20, 50.0});

    // Delayed packet arrivals strictly greater than head within 0.2s gap tolerance
    stream.pushDepth({1.05, 20.0});
    stream.pushDepth({1.15, 40.0});

    auto interSampleA = stream.getInterDepth(1.075);
    ASSERT_TRUE(interSampleA.has_value());
    EXPECT_NEAR(interSampleA->timestamp, 1.075, kEps);
    EXPECT_NEAR(interSampleA->z, 25.0, kEps);

    auto interSampleB = stream.getInterDepth(1.175);
    ASSERT_TRUE(interSampleB.has_value());
    EXPECT_NEAR(interSampleB->timestamp, 1.175, kEps);
    EXPECT_NEAR(interSampleB->z, 45.0, kEps);
}

// Verifies out-of-order insertion strictly into the middle of existing samples
TEST_F(DepthStreamTest, pushOutOfOrderMiddleInsertion) {
    stream.pushDepth({1.00, 10.0});
    stream.pushDepth({1.10, 20.0});
    stream.pushDepth({1.20, 30.0});

    // Insert sample into the middle (strictly after the head sample)
    stream.pushDepth({1.05, 15.0});

    auto sampleA = stream.getInterDepth(1.025);
    ASSERT_TRUE(sampleA.has_value());
    EXPECT_NEAR(sampleA->timestamp, 1.025, kEps);
    EXPECT_NEAR(sampleA->z, 12.5, kEps);

    auto sampleB = stream.getInterDepth(1.075);
    ASSERT_TRUE(sampleB.has_value());
    EXPECT_NEAR(sampleB->timestamp, 1.075, kEps);
    EXPECT_NEAR(sampleB->z, 17.5, kEps);
}

// Verifies bounded buffer eviction drops the oldest items on capacity overflow
TEST_F(DepthStreamTest, pushCapacityDropOldestOnOverflow) {
    PoolSLAM::DepthStream boundedStream(5);

    // Push 8 monotonic packets into a capacity-5 stream with dt = 0.05s
    for (size_t i = 1; i <= 8; i++) {
        boundedStream.pushDepth({1.00 + (static_cast<double>(i) * 0.05), static_cast<double>(i) * 10.0});
    }

    // Samples at t = 1.05, 1.10, 1.15 must be evicted
    EXPECT_FALSE(boundedStream.getInterDepth(1.10).has_value());
    EXPECT_FALSE(boundedStream.getInterDepth(1.175).has_value());

    // Samples at t = 1.20 through 1.40 must be retained
    auto retainedSample = boundedStream.getInterDepth(1.275);
    ASSERT_TRUE(retainedSample.has_value());
    EXPECT_NEAR(retainedSample->timestamp, 1.275, kEps);
    EXPECT_NEAR(retainedSample->z, 55.0, kEps);
}

// Verifies capacity enforcement when inserting an out-of-order packet with eviction
TEST_F(DepthStreamTest, pushCapacityOutOfOrderWithEviction) {
    PoolSLAM::DepthStream boundedStream(3);

    boundedStream.pushDepth({1.05, 20.0});
    boundedStream.pushDepth({1.10, 30.0});
    boundedStream.pushDepth({1.15, 40.0});

    // Insert delayed packet at t = 1.08 (strictly after head)
    boundedStream.pushDepth({1.08, 26.0});

    // Oldest sample (1.05) should be dropped; buffer now spans [1.08, 1.15]
    EXPECT_FALSE(boundedStream.getInterDepth(1.06).has_value());

    auto validSample = boundedStream.getInterDepth(1.09);
    ASSERT_TRUE(validSample.has_value());
    EXPECT_NEAR(validSample->timestamp, 1.09, kEps);
    EXPECT_NEAR(validSample->z, 28.0, kEps);
}

// Verifies handling of non-positive timestamps without internal numeric corruption
TEST_F(DepthStreamTest, pushTimestampNegativeOrZero) {
    stream.pushDepth({-0.10, 5.0});
    stream.pushDepth({-0.05, 10.0});
    stream.pushDepth({0.00, 15.0});
    stream.pushDepth({0.05, 20.0});

    auto negQuery = stream.getInterDepth(-0.075);
    ASSERT_TRUE(negQuery.has_value());
    EXPECT_NEAR(negQuery->timestamp, -0.075, kEps);
    EXPECT_NEAR(negQuery->z, 7.5, kEps);

    auto zeroCrossingQuery = stream.getInterDepth(-0.025);
    ASSERT_TRUE(zeroCrossingQuery.has_value());
    EXPECT_NEAR(zeroCrossingQuery->timestamp, -0.025, kEps);
    EXPECT_NEAR(zeroCrossingQuery->z, 12.5, kEps);
}

// ============================================================================
// 2. Temporal Retrieval and Interpolation (getInterDepth)
// ============================================================================

// Verifies query against an empty buffer returns std::nullopt
TEST_F(DepthStreamTest, interpEmptyBufferReturnsNullopt) {
    EXPECT_FALSE(stream.getInterDepth(1.00).has_value());
}

// Verifies exact match with a single sample succeeds without interpolation
TEST_F(DepthStreamTest, interpSingleSampleExactMatch) {
    stream.pushDepth({1.00, 5.0});

    auto sample = stream.getInterDepth(1.00);
    ASSERT_TRUE(sample.has_value());
    EXPECT_NEAR(sample->timestamp, 1.00, kEps);
    EXPECT_NEAR(sample->z, 5.0, kEps);
}

// Verifies query with a single sample returns std::nullopt when timestamp differs
TEST_F(DepthStreamTest, interpSingleSampleMismatchReturnsNullopt) {
    stream.pushDepth({1.00, 5.0});
    EXPECT_FALSE(stream.getInterDepth(1.05).has_value());
}

// Verifies exact match on the left boundary returns the sample without error
TEST_F(DepthStreamTest, interpExactMatchLeftBoundary) {
    stream.pushDepth({1.00, 2.0});
    stream.pushDepth({1.10, 4.0});

    auto sample = stream.getInterDepth(1.00);
    ASSERT_TRUE(sample.has_value());
    EXPECT_NEAR(sample->timestamp, 1.00, kEps);
    EXPECT_NEAR(sample->z, 2.0, kEps);
}

// Verifies exact match on the right boundary returns the sample without error
TEST_F(DepthStreamTest, interpExactMatchRightBoundary) {
    stream.pushDepth({1.00, 2.0});
    stream.pushDepth({1.10, 4.0});

    auto sample = stream.getInterDepth(1.10);
    ASSERT_TRUE(sample.has_value());
    EXPECT_NEAR(sample->timestamp, 1.10, kEps);
    EXPECT_NEAR(sample->z, 4.0, kEps);
}

// Verifies linear midpoint math
TEST_F(DepthStreamTest, interpMidpointLinearMath) {
    stream.pushDepth({1.00, 10.0});
    stream.pushDepth({1.10, 20.0});

    auto sample = stream.getInterDepth(1.05);
    ASSERT_TRUE(sample.has_value());
    EXPECT_NEAR(sample->timestamp, 1.05, kEps);
    EXPECT_NEAR(sample->z, 15.0, kEps);
}

// Verifies floating-point interpolation accuracy on non-trivial fractions
TEST_F(DepthStreamTest, interpArbitraryFractionPrecision) {
    stream.pushDepth({1.00, 10.0});
    stream.pushDepth({1.10, 20.0});

    auto sample = stream.getInterDepth(1.0333);
    ASSERT_TRUE(sample.has_value());
    EXPECT_NEAR(sample->timestamp, 1.0333, kEps);
    EXPECT_NEAR(sample->z, 13.330, kEps);
}

// Verifies lower_bound bracket identification across multiple elements
TEST_F(DepthStreamTest, interpMultiElementCorrectBracketLookup) {
    stream.pushDepth({1.00, 1.0});
    stream.pushDepth({1.10, 2.0});
    stream.pushDepth({1.20, 3.0});
    stream.pushDepth({1.30, 4.0});

    auto sample = stream.getInterDepth(1.17);
    ASSERT_TRUE(sample.has_value());
    EXPECT_NEAR(sample->timestamp, 1.17, kEps);
    EXPECT_NEAR(sample->z, 2.7, kEps);
}

// Verifies timestamps strictly prior to head return std::nullopt
TEST_F(DepthStreamTest, interpPastHeadUnderflowProtection) {
    stream.pushDepth({1.00, 5.0});
    stream.pushDepth({1.10, 10.0});

    EXPECT_FALSE(stream.getInterDepth(0.999999).has_value());
}

// Verifies timestamps strictly past tail return std::nullopt
TEST_F(DepthStreamTest, interpFutureTailOverflowProtection) {
    stream.pushDepth({1.00, 5.0});
    stream.pushDepth({1.10, 10.0});

    EXPECT_FALSE(stream.getInterDepth(1.100001).has_value());
}

// Verifies queries bridging gaps exceeding 0.2s return std::nullopt
TEST_F(DepthStreamTest, interpMaxGapExceeded) {
    // Gap delta t = 0.25s > 0.2s MAX_GAP
    stream.pushDepth({1.00, 5.0});
    stream.pushDepth({1.25, 10.0});

    EXPECT_FALSE(stream.getInterDepth(1.125).has_value());
}

// Verifies query succeeds when time delta stays within configured gap limits
TEST_F(DepthStreamTest, interpMaxGapBoundaryLimit) {
    stream.pushDepth({2.00, 5.0});
    stream.pushDepth({2.08, 10.0}); // Gap = 0.08s <= 0.2s

    auto sample = stream.getInterDepth(2.04);
    ASSERT_TRUE(sample.has_value());
    EXPECT_NEAR(sample->timestamp, 2.04, kEps);
    EXPECT_NEAR(sample->z, 7.5, kEps);
}

// Verifies identical duplicate timestamps protect against division by zero
TEST_F(DepthStreamTest, interpDuplicateTimestampsZeroDtSafety) {
    stream.pushDepth({1.00, 5.0});
    stream.pushDepth({1.00, 6.0});

    auto sample = stream.getInterDepth(1.00);
    if (sample.has_value()) {
        EXPECT_FALSE(std::isnan(sample->z));
        EXPECT_FALSE(std::isinf(sample->z));
    }
}

// ============================================================================
// 3. Pruning and Memory Eviction (pruneUntil, flush)
// ============================================================================

// Verifies pruning an empty buffer is a safe no-op
TEST_F(DepthStreamTest, pruneEmptyBufferSafeNoOp) {
    EXPECT_NO_THROW(stream.pruneUntil(10.0));
}

// Verifies horizon older than head leaves all buffer elements intact
TEST_F(DepthStreamTest, pruneUntilHeadPreservesAll) {
    stream.pushDepth({1.00, 10.0});
    stream.pushDepth({1.10, 20.0});
    stream.pushDepth({1.20, 30.0});

    stream.pruneUntil(0.90);

    auto sample = stream.getInterDepth(1.05);
    ASSERT_TRUE(sample.has_value());
    EXPECT_NEAR(sample->timestamp, 1.05, kEps);
    EXPECT_NEAR(sample->z, 15.0, kEps);
}

// Verifies pruning at exact head boundary retains valid brackets
TEST_F(DepthStreamTest, pruneExactMatchHead) {
    stream.pushDepth({1.00, 10.0});
    stream.pushDepth({1.10, 20.0});
    stream.pushDepth({1.20, 30.0});

    stream.pruneUntil(1.00);

    auto sample = stream.getInterDepth(1.15);
    ASSERT_TRUE(sample.has_value());
    EXPECT_NEAR(sample->timestamp, 1.15, kEps);
    EXPECT_NEAR(sample->z, 25.0, kEps);
}

// Verifies pruning intermediate horizon drops stale samples only
TEST_F(DepthStreamTest, pruneIntermediateHorizonDropsStaleOnly) {
    stream.pushDepth({1.00, 10.0});
    stream.pushDepth({1.10, 20.0});
    stream.pushDepth({1.20, 30.0});
    stream.pushDepth({1.30, 40.0});
    stream.pushDepth({1.40, 50.0});

    stream.pruneUntil(1.25);

    EXPECT_FALSE(stream.getInterDepth(1.15).has_value());

    auto sample = stream.getInterDepth(1.35);
    ASSERT_TRUE(sample.has_value());
    EXPECT_NEAR(sample->timestamp, 1.35, kEps);
    EXPECT_NEAR(sample->z, 45.0, kEps);
}

// Verifies pruning beyond tail clears the active elements
TEST_F(DepthStreamTest, pruneBeyondTailEmptiesBuffer) {
    stream.pushDepth({1.00, 10.0});
    stream.pushDepth({1.10, 20.0});
    stream.pushDepth({1.20, 30.0});

    stream.pruneUntil(2.00);

    EXPECT_FALSE(stream.getInterDepth(1.10).has_value());
}

// Verifies subsequent interpolation works correctly after selective prune
TEST_F(DepthStreamTest, pruneSubsequentQueryValidation) {
    stream.pushDepth({1.00, 1.0});
    stream.pushDepth({1.10, 2.0});
    stream.pushDepth({1.20, 3.0});
    stream.pushDepth({1.30, 4.0});

    stream.pruneUntil(1.15);

    EXPECT_FALSE(stream.getInterDepth(1.10).has_value());

    auto sample = stream.getInterDepth(1.25);
    ASSERT_TRUE(sample.has_value());
    EXPECT_NEAR(sample->timestamp, 1.25, kEps);
    EXPECT_NEAR(sample->z, 3.5, kEps);
}

// Verifies flush resets the internal buffer
TEST_F(DepthStreamTest, flushResetsBuffer) {
    for (size_t i = 0; i < 100; i++) {
        stream.pushDepth({1.00 + (static_cast<double>(i) * 0.05), static_cast<double>(i)});
    }

    stream.flush();
    EXPECT_FALSE(stream.getInterDepth(2.00).has_value());
}

// ============================================================================
// 4. Synchronization and Condition Variables (waitUntil)
// ============================================================================

// Verifies waitUntil returns immediately when predicate is already met
TEST_F(DepthStreamTest, waitPredicateAlreadySatisfiedZeroDelay) {
    stream.pushDepth({1.10, 10.0});

    auto start = std::chrono::steady_clock::now();
    PoolSLAM::SensorHealth health = stream.waitUntil(1.05);
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start
    ).count();

    EXPECT_EQ(health, PoolSLAM::SensorHealth::ACTIVE);
    EXPECT_LT(duration, 15);
}

// Verifies exact target timestamp match satisfies wait predicate immediately
TEST_F(DepthStreamTest, waitExactTimestampMatch) {
    stream.pushDepth({1.10, 5.0});

    auto start = std::chrono::steady_clock::now();
    PoolSLAM::SensorHealth health = stream.waitUntil(1.10);
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start
    ).count();

    EXPECT_EQ(health, PoolSLAM::SensorHealth::ACTIVE);
    EXPECT_LT(duration, 15);
}

// Verifies producer unblocks waiting consumer thread upon timely push
TEST_F(DepthStreamTest, waitProducerWakeupNominal) {
    stream.pushDepth({1.00, 5.0});

    std::thread producer([&]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        stream.pushDepth({1.10, 10.0});
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
TEST_F(DepthStreamTest, waitInsufficientDataTimesOut) {
    stream.pushDepth({1.00, 5.0});

    auto start = std::chrono::steady_clock::now();
    PoolSLAM::SensorHealth health = stream.waitUntil(1.50);
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start
    ).count();

    EXPECT_EQ(health, PoolSLAM::SensorHealth::DEGRADED);
    EXPECT_GE(duration, 85);
}

// Verifies partial arrival notifies CV without prematurely unblocking wait predicate
TEST_F(DepthStreamTest, waitSpuriousWakeupProtection) {
    stream.pushDepth({1.00, 10.0});
    std::atomic<bool> finishedWait{false};
    PoolSLAM::SensorHealth resultHealth = PoolSLAM::SensorHealth::ACTIVE;

    std::thread waiter([&]() {
        resultHealth = stream.waitUntil(1.15);
        finishedWait.store(true);
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(15));
    // Push sample that advances time but does not satisfy target 1.15
    stream.pushDepth({1.08, 12.0});

    std::this_thread::sleep_for(std::chrono::milliseconds(15));
    EXPECT_FALSE(finishedWait.load());

    // Satisfy the target predicate
    stream.pushDepth({1.16, 15.0});
    waiter.join();

    EXPECT_TRUE(finishedWait.load());
    EXPECT_EQ(resultHealth, PoolSLAM::SensorHealth::ACTIVE);
}

// ============================================================================
// 5. Sensor Health State Machine & Latching FSM
// ============================================================================

// Verifies initial health state is ACTIVE
TEST_F(DepthStreamTest, healthInitialStateActive) {
    EXPECT_EQ(stream.getHealth(), PoolSLAM::SensorHealth::ACTIVE);
}

// Verifies a single timeout transitions health from ACTIVE to DEGRADED
TEST_F(DepthStreamTest, healthSingleTimeoutTransitionsToDegraded) {
    PoolSLAM::SensorHealth health = stream.waitUntil(10.0);
    EXPECT_EQ(health, PoolSLAM::SensorHealth::DEGRADED);
    EXPECT_EQ(stream.getHealth(), PoolSLAM::SensorHealth::DEGRADED);
}

// Verifies up to 10 consecutive timeouts maintain DEGRADED (strictly greater threshold)
TEST_F(DepthStreamTest, healthConsecutiveMissesAccumulate) {
    for (size_t i = 0; i < 10; i++) {
        PoolSLAM::SensorHealth health = stream.waitUntil(10.0 + (static_cast<double>(i) * 0.05));
        EXPECT_EQ(health, PoolSLAM::SensorHealth::DEGRADED);
        EXPECT_EQ(stream.getHealth(), PoolSLAM::SensorHealth::DEGRADED);
    }
}

// Verifies timely arrival restores health from DEGRADED back to ACTIVE before threshold
TEST_F(DepthStreamTest, healthRecoveryDegradedToActive) {
    for (size_t i = 0; i < 5; i++) {
        EXPECT_EQ(stream.waitUntil(10.0 + (static_cast<double>(i) * 0.05)), PoolSLAM::SensorHealth::DEGRADED);
    }
    EXPECT_EQ(stream.getHealth(), PoolSLAM::SensorHealth::DEGRADED);

    // Timely packet arrives satisfying wait target
    stream.pushDepth({20.00, 5.0});
    PoolSLAM::SensorHealth recovered = stream.waitUntil(20.00);

    EXPECT_EQ(recovered, PoolSLAM::SensorHealth::ACTIVE);
    EXPECT_EQ(stream.getHealth(), PoolSLAM::SensorHealth::ACTIVE);
}

// Verifies exceeding 10 misses latches into DISABLED strictly on the 11th miss
TEST_F(DepthStreamTest, healthMaxMissThresholdLatchesDisabled) {
    for (size_t i = 0; i < 10; i++) {
        EXPECT_EQ(stream.waitUntil(1.00 + (static_cast<double>(i) * 0.05)), PoolSLAM::SensorHealth::DEGRADED);
        EXPECT_EQ(stream.getHealth(), PoolSLAM::SensorHealth::DEGRADED);
    }

    // 11th miss crosses strictly greater threshold (> 10)
    PoolSLAM::SensorHealth latched = stream.waitUntil(10.00);
    EXPECT_EQ(latched, PoolSLAM::SensorHealth::DISABLED);
    EXPECT_EQ(stream.getHealth(), PoolSLAM::SensorHealth::DISABLED);
}

// Verifies DISABLED is a strict one-way latch that rejects recovery
TEST_F(DepthStreamTest, healthDisabledOneWayLatch) {
    for (size_t i = 0; i < 11; i++) {
        stream.waitUntil(1.00 + (static_cast<double>(i) * 0.05));
    }
    ASSERT_EQ(stream.getHealth(), PoolSLAM::SensorHealth::DISABLED);

    // Push valid, fresh data
    stream.pushDepth({50.00, 10.0});
    PoolSLAM::SensorHealth health = stream.waitUntil(50.00);

    EXPECT_EQ(health, PoolSLAM::SensorHealth::DISABLED);
    EXPECT_EQ(stream.getHealth(), PoolSLAM::SensorHealth::DISABLED);
}

// Verifies waitUntil bypasses condition variable delays once DISABLED
TEST_F(DepthStreamTest, healthDisabledFastBypass) {
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

// Verifies thread safety during simultaneous ingestion and interpolation queries
TEST_F(DepthStreamTest, concurrentPushAndInterpolateDataRace) {
    std::atomic<bool> keepRunning{true};

    // Producer thread: 50 Hz push (dt = 0.02s <= 0.2s MAX_GAP)
    std::thread producer([&]() {
        double t = 1.00;
        double zVal = 5.0;
        while (keepRunning.load()) {
            stream.pushDepth({t, zVal});
            t += 0.02;
            zVal += 0.05;
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
    });

    // Consumer thread: interpolate trailing timestamps
    std::thread consumer([&]() {
        double queryTime = 1.01;
        while (keepRunning.load()) {
            auto sample = stream.getInterDepth(queryTime);
            if (sample.has_value()) {
                EXPECT_FALSE(std::isnan(sample->z));
                queryTime += 0.02;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    keepRunning.store(false);

    producer.join();
    consumer.join();
}

// Verifies buffer integrity under simultaneous pushing, pruning, and querying
TEST_F(DepthStreamTest, concurrentPushAndPruneHorizonIntegrity) {
    std::atomic<bool> keepRunning{true};

    // Producer thread: monotonic and out-of-order pushes (dt <= 0.02s)
    std::thread producer([&]() {
        double t = 1.00;
        while (keepRunning.load()) {
            stream.pushDepth({t, t * 2.0});
            if (std::fmod(t, 0.1) < 0.01) {
                // Delayed packet within valid gap limit
                stream.pushDepth({t - 0.005, (t - 0.005) * 2.0});
            }
            t += 0.01;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    });

    // Pruner thread: advancing horizon
    std::thread pruner([&]() {
        double horizon = 1.00;
        while (keepRunning.load()) {
            stream.pruneUntil(horizon);
            horizon += 0.02;
            std::this_thread::sleep_for(std::chrono::milliseconds(3));
        }
    });

    // Consumer thread: querying interpolated values
    std::thread consumer([&]() {
        double queryTime = 1.05;
        while (keepRunning.load()) {
            auto sample = stream.getInterDepth(queryTime);
            if (sample.has_value()) {
                EXPECT_NEAR(sample->z, queryTime * 2.0, 1e-4);
                queryTime += 0.01;
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

// Verifies multiple consumer threads safely wake up when target arrives
TEST_F(DepthStreamTest, concurrentWaitUntilAndPushRaceCondition) {
    constexpr size_t numWaiters = 4;
    std::vector<std::thread> waiterThreads;
    std::vector<PoolSLAM::SensorHealth> results(numWaiters, PoolSLAM::SensorHealth::DEGRADED);

    stream.pushDepth({1.00, 5.0});

    for (size_t i = 0; i < numWaiters; i++) {
        waiterThreads.emplace_back([&, i]() {
            results[i] = stream.waitUntil(1.10);
        });
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    stream.pushDepth({1.10, 10.0});

    for (size_t i = 0; i < numWaiters; i++) {
        waiterThreads[i].join();
        EXPECT_EQ(results[i], PoolSLAM::SensorHealth::ACTIVE);
    }
}