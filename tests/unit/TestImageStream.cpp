#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "pool_slam/Hal.hpp"
#include "pool_slam/HalTypes.hpp"

namespace {
constexpr size_t kWidth = 640;
constexpr size_t kHeight = 480;
constexpr size_t kFrameBytes = kWidth * kHeight;
constexpr size_t kQueueCapacity = 3;
constexpr size_t kPoolCapacity = 5;
}

class ImageStreamTest : public ::testing::Test {
protected:
    PoolSLAM::ImageStream stream{kWidth, kHeight, kQueueCapacity, kPoolCapacity};
    std::vector<uint8_t> dummyPixels{std::vector<uint8_t>(kFrameBytes, 128)};

    std::vector<uint8_t> createPattern(uint8_t val) {
        return std::vector<uint8_t>(kFrameBytes, val);
    }
};

// ============================================================================
// 1. Ingestion Mechanics (pushFrame)
// ============================================================================

// Verifies nominal push of a single valid frame into the stream
TEST_F(ImageStreamTest, pushSingleFrameNominal) {
    EXPECT_TRUE(stream.pushFrame(dummyPixels.data(), 1.00));
    EXPECT_EQ(stream.getHealth(), PoolSLAM::SensorHealth::ACTIVE);
}

// Verifies pushed pixel payloads and timestamps are copied completely without truncation
TEST_F(ImageStreamTest, pushPreservesPixelPayloadAndTimestamp) {
    auto pattern = createPattern(0xAB);
    pattern.front() = 0x12;
    pattern.back() = 0x34;

    ASSERT_TRUE(stream.pushFrame(pattern.data(), 1.25));

    PoolSLAM::PixelArray* frame = nullptr;
    PoolSLAM::SensorHealth health = stream.waitForFrame(frame, std::chrono::milliseconds(50));

    ASSERT_EQ(health, PoolSLAM::SensorHealth::ACTIVE);
    ASSERT_NE(frame, nullptr);
    EXPECT_DOUBLE_EQ(frame->timestamp, 1.25);
    EXPECT_EQ(frame->arr.front(), 0x12);
    EXPECT_EQ(frame->arr.back(), 0x34);
    EXPECT_EQ(frame->arr[100], 0xAB);

    stream.recycleFrame(frame);
}

// Verifies passing a null data pointer is rejected without corrupting pool buffers
TEST_F(ImageStreamTest, pushNullPointerRejection) {
    EXPECT_FALSE(stream.pushFrame(nullptr, 1.00));

    EXPECT_TRUE(stream.pushFrame(dummyPixels.data(), 1.00));

    PoolSLAM::PixelArray* frame = nullptr;
    EXPECT_EQ(stream.waitForFrame(frame, std::chrono::milliseconds(50)), PoolSLAM::SensorHealth::ACTIVE);
    ASSERT_NE(frame, nullptr);
    stream.recycleFrame(frame);
}

// ============================================================================
// 2. Buffer Checkout & Recycling Lifecycle (waitForFrame, recycleFrame)
// ============================================================================

// Verifies nominal check-out and pointer pass-by-reference assignment
TEST_F(ImageStreamTest, waitAndPopNominal) {
    ASSERT_TRUE(stream.pushFrame(dummyPixels.data(), 1.00));

    PoolSLAM::PixelArray* frame = nullptr;
    PoolSLAM::SensorHealth health = stream.waitForFrame(frame, std::chrono::milliseconds(50));

    EXPECT_EQ(health, PoolSLAM::SensorHealth::ACTIVE);
    ASSERT_NE(frame, nullptr);
    EXPECT_DOUBLE_EQ(frame->timestamp, 1.00);

    stream.recycleFrame(frame);
}

// Verifies returning checked-out frames restores the free pool for subsequent pushes
TEST_F(ImageStreamTest, recycleReturnsBufferToPool) {
    PoolSLAM::ImageStream smallStream(kWidth, kHeight, 2, 2);

    ASSERT_TRUE(smallStream.pushFrame(dummyPixels.data(), 1.00));
    ASSERT_TRUE(smallStream.pushFrame(dummyPixels.data(), 2.00));

    PoolSLAM::PixelArray* f1 = nullptr;
    PoolSLAM::PixelArray* f2 = nullptr;
    ASSERT_EQ(smallStream.waitForFrame(f1, std::chrono::milliseconds(20)), PoolSLAM::SensorHealth::ACTIVE);
    ASSERT_EQ(smallStream.waitForFrame(f2, std::chrono::milliseconds(20)), PoolSLAM::SensorHealth::ACTIVE);
    ASSERT_NE(f1, nullptr);
    ASSERT_NE(f2, nullptr);

    // Free list exhausted and ready queue empty; push must fail gracefully
    EXPECT_FALSE(smallStream.pushFrame(dummyPixels.data(), 3.00));

    smallStream.recycleFrame(f1);

    EXPECT_TRUE(smallStream.pushFrame(dummyPixels.data(), 4.00));

    PoolSLAM::PixelArray* f3 = nullptr;
    ASSERT_EQ(smallStream.waitForFrame(f3, std::chrono::milliseconds(20)), PoolSLAM::SensorHealth::ACTIVE);
    ASSERT_NE(f3, nullptr);
    EXPECT_DOUBLE_EQ(f3->timestamp, 4.00);

    smallStream.recycleFrame(f2);
    smallStream.recycleFrame(f3);
}

// Verifies recycling a nullptr is a safe no-op
TEST_F(ImageStreamTest, recycleNullPointerSafeNoOp) {
    EXPECT_NO_THROW(stream.recycleFrame(nullptr));
}

// ============================================================================
// 3. Queue Eviction & Backpressure Policy
// ============================================================================

// Verifies oldest queued frame is evicted when readyQueue hits capacity
TEST_F(ImageStreamTest, evictOldestFrameWhenReadyQueueFull) {
    PoolSLAM::ImageStream boundedStream(kWidth, kHeight, 2, 4);

    auto p1 = createPattern(10);
    auto p2 = createPattern(20);
    auto p3 = createPattern(30);

    ASSERT_TRUE(boundedStream.pushFrame(p1.data(), 1.00));
    ASSERT_TRUE(boundedStream.pushFrame(p2.data(), 2.00));
    // Overflows readyQueue capacity (2); frame at t=1.00 should be evicted and recycled
    ASSERT_TRUE(boundedStream.pushFrame(p3.data(), 3.00));

    PoolSLAM::PixelArray* frame = nullptr;

    // First popped frame must be the second frame (t=2.00)
    ASSERT_EQ(boundedStream.waitForFrame(frame, std::chrono::milliseconds(20)), PoolSLAM::SensorHealth::ACTIVE);
    ASSERT_NE(frame, nullptr);
    EXPECT_DOUBLE_EQ(frame->timestamp, 2.00);
    EXPECT_EQ(frame->arr[0], 20);
    boundedStream.recycleFrame(frame);

    // Second popped frame must be the third frame (t=3.00)
    ASSERT_EQ(boundedStream.waitForFrame(frame, std::chrono::milliseconds(20)), PoolSLAM::SensorHealth::ACTIVE);
    ASSERT_NE(frame, nullptr);
    EXPECT_DOUBLE_EQ(frame->timestamp, 3.00);
    EXPECT_EQ(frame->arr[0], 30);
    boundedStream.recycleFrame(frame);
}

// Verifies pushFrame reclaims the oldest unconsumed frame when pool is dry but queue has elements
TEST_F(ImageStreamTest, poolStarvationReclaimsOldestQueuedFrame) {
    PoolSLAM::ImageStream boundedStream(kWidth, kHeight, 2, 2);

    ASSERT_TRUE(boundedStream.pushFrame(dummyPixels.data(), 1.00));
    ASSERT_TRUE(boundedStream.pushFrame(dummyPixels.data(), 2.00));

    // Both buffers in ready queue; pool is empty. 3rd push reclaims oldest queued buffer (t=1.00)
    ASSERT_TRUE(boundedStream.pushFrame(dummyPixels.data(), 3.00));

    PoolSLAM::PixelArray* frame = nullptr;

    ASSERT_EQ(boundedStream.waitForFrame(frame, std::chrono::milliseconds(20)), PoolSLAM::SensorHealth::ACTIVE);
    ASSERT_NE(frame, nullptr);
    EXPECT_DOUBLE_EQ(frame->timestamp, 2.00);
    boundedStream.recycleFrame(frame);

    ASSERT_EQ(boundedStream.waitForFrame(frame, std::chrono::milliseconds(20)), PoolSLAM::SensorHealth::ACTIVE);
    ASSERT_NE(frame, nullptr);
    EXPECT_DOUBLE_EQ(frame->timestamp, 3.00);
    boundedStream.recycleFrame(frame);
}

// Verifies pushFrame drops incoming frames gracefully when consumer holds all pool buffers
TEST_F(ImageStreamTest, consumerHoldingAllBuffersDropsGracefully) {
    PoolSLAM::ImageStream boundedStream(kWidth, kHeight, 2, 2);

    ASSERT_TRUE(boundedStream.pushFrame(dummyPixels.data(), 1.00));
    ASSERT_TRUE(boundedStream.pushFrame(dummyPixels.data(), 2.00));

    PoolSLAM::PixelArray* held1 = nullptr;
    PoolSLAM::PixelArray* held2 = nullptr;

    ASSERT_EQ(boundedStream.waitForFrame(held1, std::chrono::milliseconds(20)), PoolSLAM::SensorHealth::ACTIVE);
    ASSERT_EQ(boundedStream.waitForFrame(held2, std::chrono::milliseconds(20)), PoolSLAM::SensorHealth::ACTIVE);

    // Consumer holds both buffers; readyQueue and pool are empty
    EXPECT_FALSE(boundedStream.pushFrame(dummyPixels.data(), 3.00));

    boundedStream.recycleFrame(held1);
    boundedStream.recycleFrame(held2);
}

// ============================================================================
// 4. Synchronization and Condition Variables (waitForFrame)
// ============================================================================

// Verifies waiting consumer is unblocked when producer pushes timely frame
TEST_F(ImageStreamTest, waitProducerWakeupNominal) {
    PoolSLAM::PixelArray* receivedFrame = nullptr;
    PoolSLAM::SensorHealth resultHealth = PoolSLAM::SensorHealth::DEGRADED;

    std::thread consumer([&]() {
        resultHealth = stream.waitForFrame(receivedFrame, std::chrono::milliseconds(100));
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    ASSERT_TRUE(stream.pushFrame(dummyPixels.data(), 1.10));

    consumer.join();

    EXPECT_EQ(resultHealth, PoolSLAM::SensorHealth::ACTIVE);
    ASSERT_NE(receivedFrame, nullptr);
    EXPECT_DOUBLE_EQ(receivedFrame->timestamp, 1.10);

    stream.recycleFrame(receivedFrame);
}

// Verifies waitForFrame times out when no frames arrive within timeout window
TEST_F(ImageStreamTest, waitTimesOutWhenNoDataPushed) {
    PoolSLAM::PixelArray* frame = reinterpret_cast<PoolSLAM::PixelArray*>(0xDEADBEEF);

    auto start = std::chrono::steady_clock::now();
    PoolSLAM::SensorHealth health = stream.waitForFrame(frame, std::chrono::milliseconds(80));
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start
    ).count();

    EXPECT_EQ(health, PoolSLAM::SensorHealth::DEGRADED);
    EXPECT_EQ(frame, nullptr);
    EXPECT_GE(duration, 75);
}

// ============================================================================
// 5. Sensor Health State Machine & Latching FSM
// ============================================================================

// Verifies initial stream health is ACTIVE
TEST_F(ImageStreamTest, healthInitialStateActive) {
    EXPECT_EQ(stream.getHealth(), PoolSLAM::SensorHealth::ACTIVE);
}

// Verifies a single timeout transitions health from ACTIVE to DEGRADED
TEST_F(ImageStreamTest, healthSingleTimeoutTransitionsToDegraded) {
    PoolSLAM::PixelArray* frame = nullptr;
    PoolSLAM::SensorHealth health = stream.waitForFrame(frame, std::chrono::milliseconds(10));

    EXPECT_EQ(health, PoolSLAM::SensorHealth::DEGRADED);
    EXPECT_EQ(stream.getHealth(), PoolSLAM::SensorHealth::DEGRADED);
    EXPECT_EQ(frame, nullptr);
}

// Verifies up to 3 consecutive timeouts maintain DEGRADED (strictly greater threshold: > 3)
TEST_F(ImageStreamTest, healthThreeConsecutiveMissesRemainDegraded) {
    PoolSLAM::PixelArray* frame = nullptr;
    for (size_t i = 0; i < 3; i++) {
        PoolSLAM::SensorHealth health = stream.waitForFrame(frame, std::chrono::milliseconds(5));
        EXPECT_EQ(health, PoolSLAM::SensorHealth::DEGRADED);
        EXPECT_EQ(stream.getHealth(), PoolSLAM::SensorHealth::DEGRADED);
    }
}

// Verifies timely arrival restores health from DEGRADED back to ACTIVE before threshold
TEST_F(ImageStreamTest, healthRecoveryDegradedToActive) {
    PoolSLAM::PixelArray* frame = nullptr;
    for (size_t i = 0; i < 2; i++) {
        EXPECT_EQ(stream.waitForFrame(frame, std::chrono::milliseconds(5)), PoolSLAM::SensorHealth::DEGRADED);
    }
    EXPECT_EQ(stream.getHealth(), PoolSLAM::SensorHealth::DEGRADED);

    // Frame arrives before reaching miss threshold
    ASSERT_TRUE(stream.pushFrame(dummyPixels.data(), 5.00));

    PoolSLAM::SensorHealth recovered = stream.waitForFrame(frame, std::chrono::milliseconds(20));
    EXPECT_EQ(recovered, PoolSLAM::SensorHealth::ACTIVE);
    EXPECT_EQ(stream.getHealth(), PoolSLAM::SensorHealth::ACTIVE);
    ASSERT_NE(frame, nullptr);

    stream.recycleFrame(frame);
}

// Verifies exceeding 3 misses latches into DISABLED strictly on the 4th miss
TEST_F(ImageStreamTest, healthFourthMissLatchesDisabled) {
    PoolSLAM::PixelArray* frame = nullptr;
    for (size_t i = 0; i < 3; i++) {
        EXPECT_EQ(stream.waitForFrame(frame, std::chrono::milliseconds(5)), PoolSLAM::SensorHealth::DEGRADED);
        EXPECT_EQ(stream.getHealth(), PoolSLAM::SensorHealth::DEGRADED);
    }

    // 4th consecutive miss crosses strictly greater threshold (> 3)
    PoolSLAM::SensorHealth latched = stream.waitForFrame(frame, std::chrono::milliseconds(5));
    EXPECT_EQ(latched, PoolSLAM::SensorHealth::DISABLED);
    EXPECT_EQ(stream.getHealth(), PoolSLAM::SensorHealth::DISABLED);
}

// Verifies DISABLED is a strict one-way latch rejecting recovery
TEST_F(ImageStreamTest, healthDisabledOneWayLatch) {
    PoolSLAM::PixelArray* frame = nullptr;
    for (size_t i = 0; i < 4; i++) {
        stream.waitForFrame(frame, std::chrono::milliseconds(5));
    }
    ASSERT_EQ(stream.getHealth(), PoolSLAM::SensorHealth::DISABLED);

    // Fresh frame pushed after sensor latch
    stream.pushFrame(dummyPixels.data(), 50.00);

    PoolSLAM::SensorHealth health = stream.waitForFrame(frame, std::chrono::milliseconds(5));
    EXPECT_EQ(health, PoolSLAM::SensorHealth::DISABLED);
    EXPECT_EQ(stream.getHealth(), PoolSLAM::SensorHealth::DISABLED);
}

// Verifies waitForFrame bypasses condition variable delays once DISABLED
TEST_F(ImageStreamTest, healthDisabledFastBypass) {
    PoolSLAM::PixelArray* frame = nullptr;
    for (size_t i = 0; i < 4; i++) {
        stream.waitForFrame(frame, std::chrono::milliseconds(5));
    }
    ASSERT_EQ(stream.getHealth(), PoolSLAM::SensorHealth::DISABLED);

    auto start = std::chrono::steady_clock::now();
    PoolSLAM::SensorHealth health = stream.waitForFrame(frame, std::chrono::milliseconds(100));
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start
    ).count();

    EXPECT_EQ(health, PoolSLAM::SensorHealth::DISABLED);
    EXPECT_LT(duration, 5);
}

// ============================================================================
// 6. Concurrency and Multi-Threaded Stress
// ============================================================================

// Verifies multi-threaded nominal streaming between producer and consumer
TEST_F(ImageStreamTest, concurrentProducerConsumerStreaming) {
    std::atomic<bool> keepRunning{true};
    std::atomic<size_t> framesPushed{0};
    std::atomic<size_t> framesConsumed{0};

    // Producer thread: pushing frames at ~30 Hz (dt = 33 ms)
    std::thread producer([&]() {
        double t = 1.00;
        while (keepRunning.load()) {
            if (stream.pushFrame(dummyPixels.data(), t)) {
                framesPushed.fetch_add(1);
                t += 0.033;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    });

    // Consumer thread: reading and recycling frames
    std::thread consumer([&]() {
        double lastTimestamp = 0.0;
        while (keepRunning.load()) {
            PoolSLAM::PixelArray* frame = nullptr;
            PoolSLAM::SensorHealth health = stream.waitForFrame(frame, std::chrono::milliseconds(50));
            if (health == PoolSLAM::SensorHealth::ACTIVE && frame != nullptr) {
                EXPECT_GT(frame->timestamp, lastTimestamp);
                lastTimestamp = frame->timestamp;
                framesConsumed.fetch_add(1);
                stream.recycleFrame(frame);
            }
        }
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(150));
    keepRunning.store(false);

    producer.join();
    consumer.join();

    EXPECT_GT(framesPushed.load(), 0);
    EXPECT_GT(framesConsumed.load(), 0);
}

// Verifies queue eviction operates safely without deadlock under high producer load
TEST_F(ImageStreamTest, concurrentFastProducerSlowConsumerEviction) {
    std::atomic<bool> keepRunning{true};

    // Fast producer: 100 Hz push
    std::thread producer([&]() {
        double t = 1.00;
        while (keepRunning.load()) {
            stream.pushFrame(dummyPixels.data(), t);
            t += 0.01;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    });

    // Slow consumer: pops every 25 ms, forcing frequent evictions
    std::thread consumer([&]() {
        while (keepRunning.load()) {
            PoolSLAM::PixelArray* frame = nullptr;
            PoolSLAM::SensorHealth health = stream.waitForFrame(frame, std::chrono::milliseconds(50));
            if (health == PoolSLAM::SensorHealth::ACTIVE && frame != nullptr) {
                stream.recycleFrame(frame);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(25));
        }
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(150));
    keepRunning.store(false);

    producer.join();
    consumer.join();

    EXPECT_EQ(stream.getHealth(), PoolSLAM::SensorHealth::ACTIVE);
}