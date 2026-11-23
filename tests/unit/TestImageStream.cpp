#include <chrono>
#include <thread>
#include <atomic>
#include <vector>
#include <cstring>

#include <gtest/gtest.h>

#include "HalTestHelpers.hpp"
#include "pool_slam/Hal.hpp"
#include "pool_slam/HalTypes.hpp"

namespace PoolSLAM {

// =============================================================================
// 1. Pool Allocation & Lifecycle
// =============================================================================

// Verifies initial state: all buffers in free list, ready queue empty, ACTIVE health
TEST_F(TestImageStream, InitialStateVerification) {
    constexpr int w = 320, h = 240;
    constexpr size_t maxReady = 3, poolCap = 10;
    ImageStream stream(w, h, maxReady, poolCap);

    EXPECT_EQ(getFreeListSize(stream), poolCap);
    EXPECT_EQ(getReadyQueueSize(stream), 0);
    EXPECT_EQ(stream.getHealth(), SensorHealth::ACTIVE);
    EXPECT_EQ(getMissedFrames(stream), 0);
}

// Nominal checkout-consumption-recycle cycle returns buffer to free list
TEST_F(TestImageStream, NominalCycle) {
    constexpr int w = 100, h = 100;
    ImageStream stream(w, h, 3, 5);

    auto pattern = makePattern(w, h, 42);
    EXPECT_TRUE(stream.pushFrame(pattern.data(), 1.0));

    EXPECT_EQ(getFreeListSize(stream), 4);
    EXPECT_EQ(getReadyQueueSize(stream), 1);

    PixelArray* frame = nullptr;
    SensorHealth status = stream.waitForFrame(frame, std::chrono::milliseconds(50));

    EXPECT_EQ(status, SensorHealth::ACTIVE);
    ASSERT_NE(frame, nullptr);
    EXPECT_DOUBLE_EQ(frame->timestamp, 1.0);
    EXPECT_EQ(std::memcmp(frame->data(), pattern.data(), pattern.size()), 0);

    // After checkout, ready queue is empty
    EXPECT_EQ(getReadyQueueSize(stream), 0);
    EXPECT_EQ(getFreeListSize(stream), 4);

    // Recycling frame restores free list capacity
    stream.recycleFrame(frame);
    EXPECT_EQ(getFreeListSize(stream), 5);
}

// Null pointer checks: pushFrame rejects nullptr; recycleFrame safely ignores nullptr
TEST_F(TestImageStream, NullPointerHandling) {
    ImageStream stream(100, 100, 3, 5);

    EXPECT_FALSE(stream.pushFrame(nullptr, 1.0));
    EXPECT_EQ(getFreeListSize(stream), 5);

    // Safe no-op
    stream.recycleFrame(nullptr);
    EXPECT_EQ(getFreeListSize(stream), 5);
}

// =============================================================================
// 2. Buffer Starvation & Eviction Mechanics
// =============================================================================

// Reclaims and overwrites oldest unconsumed frame when free list is exhausted
TEST_F(TestImageStream, FreeListExhaustionWithReadyQueueEviction) {
    constexpr int w = 10, h = 10;
    // Pool capacity 2, max queue size 2
    ImageStream stream(w, h, 2, 2);

    auto p1 = makePattern(w, h, 1);
    auto p2 = makePattern(w, h, 2);
    auto p3 = makePattern(w, h, 3);

    EXPECT_TRUE(stream.pushFrame(p1.data(), 1.0));
    EXPECT_TRUE(stream.pushFrame(p2.data(), 2.0));
    EXPECT_EQ(getFreeListSize(stream), 0);
    EXPECT_EQ(getReadyQueueSize(stream), 2);

    // 3rd push: free list is empty -> evicts and reuses oldest frame from ready queue (t=1.0)
    EXPECT_TRUE(stream.pushFrame(p3.data(), 3.0));
    EXPECT_EQ(getReadyQueueSize(stream), 2);

    // Consumer should receive frame 2 (t=2.0), then frame 3 (t=3.0)
    PixelArray* f1 = nullptr;
    EXPECT_EQ(stream.waitForFrame(f1), SensorHealth::ACTIVE);
    ASSERT_NE(f1, nullptr);
    EXPECT_DOUBLE_EQ(f1->timestamp, 2.0);

    PixelArray* f2 = nullptr;
    EXPECT_EQ(stream.waitForFrame(f2), SensorHealth::ACTIVE);
    ASSERT_NE(f2, nullptr);
    EXPECT_DOUBLE_EQ(f2->timestamp, 3.0);

    stream.recycleFrame(f1);
    stream.recycleFrame(f2);
}

// Total pool exhaustion: returns false when all buffers are checked out and queue is empty
TEST_F(TestImageStream, TotalPoolExhaustionReturnsFalse) {
    constexpr int w = 10, h = 10;
    // Pool capacity 2
    ImageStream stream(w, h, 2, 2);
    auto pattern = makePattern(w, h, 5);

    EXPECT_TRUE(stream.pushFrame(pattern.data(), 1.0));
    EXPECT_TRUE(stream.pushFrame(pattern.data(), 2.0));

    PixelArray* checkedOut1 = nullptr;
    PixelArray* checkedOut2 = nullptr;
    stream.waitForFrame(checkedOut1);
    stream.waitForFrame(checkedOut2);

    // All 2 pool buffers are checked out; readyQueue is empty
    EXPECT_EQ(getFreeListSize(stream), 0);
    EXPECT_EQ(getReadyQueueSize(stream), 0);

    // Acquisition fails gracefully without corrupting state
    EXPECT_FALSE(stream.pushFrame(pattern.data(), 3.0));

    // Recycling a checked-out frame restores ingestion capability
    stream.recycleFrame(checkedOut1);
    EXPECT_EQ(getFreeListSize(stream), 1);
    EXPECT_TRUE(stream.pushFrame(pattern.data(), 3.0));

    stream.recycleFrame(checkedOut2);
    PixelArray* f3 = nullptr;
    stream.waitForFrame(f3);
    stream.recycleFrame(f3);
}

// Ready queue size never exceeds maxQueueSize under continuous producer pushes
TEST_F(TestImageStream, MaximumReadyQueueBound) {
    constexpr int w = 10, h = 10;
    constexpr size_t maxReady = 3, poolCap = 10;
    ImageStream stream(w, h, maxReady, poolCap);
    auto pattern = makePattern(w, h, 10);

    for (int i = 0; i < 7; ++i) {
        EXPECT_TRUE(stream.pushFrame(pattern.data(), static_cast<double>(i)));
        EXPECT_LE(getReadyQueueSize(stream), maxReady);
    }

    EXPECT_EQ(getReadyQueueSize(stream), maxReady);

    // Drain and recycle
    for (size_t i = 0; i < maxReady; ++i) {
        PixelArray* f = nullptr;
        stream.waitForFrame(f);
        stream.recycleFrame(f);
    }
}

// =============================================================================
// 3. Consumer Synchronization & Health State Machine
// =============================================================================

// Consumer blocking in waitForFrame unblocks immediately upon pushFrame
TEST_F(TestImageStream, BlockingRetrieval) {
    ImageStream stream(50, 50, 3, 5);
    auto pattern = makePattern(50, 50, 77);

    std::atomic<bool> consumerWaiting{false};
    PixelArray* target = nullptr;
    SensorHealth status = SensorHealth::DISABLED;

    std::thread consumer([&]() {
        consumerWaiting = true;
        status = stream.waitForFrame(target, std::chrono::milliseconds(500));
    });

    while (!consumerWaiting) {
        std::this_thread::yield();
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    EXPECT_TRUE(stream.pushFrame(pattern.data(), 1.0));
    consumer.join();

    EXPECT_EQ(status, SensorHealth::ACTIVE);
    ASSERT_NE(target, nullptr);
    EXPECT_DOUBLE_EQ(target->timestamp, 1.0);
    stream.recycleFrame(target);
}

// Consumer times out when no frame is pushed within specified timeout
TEST_F(TestImageStream, TimeoutHandling) {
    ImageStream stream(50, 50, 3, 5);
    PixelArray* target = reinterpret_cast<PixelArray*>(0x1234);

    auto start = std::chrono::steady_clock::now();
    SensorHealth status = stream.waitForFrame(target, std::chrono::milliseconds(50));
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start
    );

    EXPECT_GE(elapsed.count(), 45);
    EXPECT_EQ(target, nullptr);
    EXPECT_EQ(status, SensorHealth::DEGRADED);
    EXPECT_EQ(stream.getHealth(), SensorHealth::DEGRADED);
    EXPECT_EQ(getMissedFrames(stream), 1);
}

// Sensor transitions to DISABLED after exceeding MAX_MISSED_FRAMES (3)
TEST_F(TestImageStream, SensorDisablingAfterMaxMissedFrames) {
    ImageStream stream(50, 50, 3, 5);
    PixelArray* target = nullptr;

    // Fast-forward missed frames to 3 (MAX_MISSED_FRAMES)
    setMissedFrames(stream, 3);
    setHealth(stream, SensorHealth::DEGRADED);

    // 4th timeout: missedFrames becomes 4 (> 3) -> transitions to DISABLED
    SensorHealth status = stream.waitForFrame(target, std::chrono::milliseconds(20));
    EXPECT_EQ(status, SensorHealth::DISABLED);
    EXPECT_EQ(stream.getHealth(), SensorHealth::DISABLED);
    EXPECT_EQ(target, nullptr);

    // Subsequent calls return DISABLED immediately without blocking
    auto start = std::chrono::steady_clock::now();
    SensorHealth immediate = stream.waitForFrame(target, std::chrono::milliseconds(100));
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start
    );
    EXPECT_LT(elapsed.count(), 20);
    EXPECT_EQ(immediate, SensorHealth::DISABLED);
}

// Successfully receiving a frame while DEGRADED resets missedFrames and restores ACTIVE
TEST_F(TestImageStream, HealthRecovery) {
    ImageStream stream(50, 50, 3, 5);
    PixelArray* target = nullptr;

    // Trigger degradation
    stream.waitForFrame(target, std::chrono::milliseconds(20));
    EXPECT_EQ(stream.getHealth(), SensorHealth::DEGRADED);
    EXPECT_GT(getMissedFrames(stream), 0);

    // Push valid frame and check out
    auto pattern = makePattern(50, 50, 11);
    stream.pushFrame(pattern.data(), 1.0);

    SensorHealth status = stream.waitForFrame(target, std::chrono::milliseconds(50));
    EXPECT_EQ(status, SensorHealth::ACTIVE);
    EXPECT_EQ(stream.getHealth(), SensorHealth::ACTIVE);
    EXPECT_EQ(getMissedFrames(stream), 0);
    ASSERT_NE(target, nullptr);

    stream.recycleFrame(target);
}

// =============================================================================
// 4. Data Integrity & Memory Correctness
// =============================================================================

// Byte-level exactness across entire image resolution
TEST_F(TestImageStream, ByteLevelDataMatch) {
    constexpr int w = 128, h = 64;
    ImageStream stream(w, h, 3, 5);

    auto pattern = makePattern(w, h, 99);
    stream.pushFrame(pattern.data(), 123.456);

    PixelArray* out = nullptr;
    stream.waitForFrame(out);
    ASSERT_NE(out, nullptr);

    EXPECT_EQ(out->arr.size(), static_cast<size_t>(w * h));
    EXPECT_EQ(std::memcmp(out->data(), pattern.data(), pattern.size()), 0);

    stream.recycleFrame(out);
}

// Timestamp preservation matches capture epoch
TEST_F(TestImageStream, TimestampPreservation) {
    ImageStream stream(50, 50, 3, 5);
    auto pattern = makePattern(50, 50, 0);

    constexpr double epoch = 1700000000.123456;
    stream.pushFrame(pattern.data(), epoch);

    PixelArray* out = nullptr;
    stream.waitForFrame(out);
    ASSERT_NE(out, nullptr);
    EXPECT_DOUBLE_EQ(out->timestamp, epoch);

    stream.recycleFrame(out);
}

} // namespace PoolSLAM
