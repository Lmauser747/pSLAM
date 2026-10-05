#include <chrono>

#include <gtest/gtest.h>

#include "pool_slam/Hal.hpp"
#include "pool_slam/HalTypes.hpp"

class TestDepthStream : public ::testing::Test{
protected:
    PoolSLAM::DepthStream stream;
};

// Verify that sking for exact and interpolated timestamps results in the correct time
TEST_F(TestDepthStream, InterpolationNominal){
    stream.pushDepth({1.0, 10.0});
    stream.pushDepth({1.1, 11.0});

    // Test exact values
    auto exact = stream.getInterDepth(1.0);
    ASSERT_TRUE(exact.has_value());
    EXPECT_DOUBLE_EQ(exact->z, 10.0);

    // Test interpolated values
    auto inter = stream.getInterDepth(1.05);
    ASSERT_TRUE(inter.has_value());
    EXPECT_DOUBLE_EQ(inter->z, 10.5);
}

// Make sure depth stream does not return values outside of availible data
TEST_F(TestDepthStream, BoundsEnforcement) {
    stream.pushDepth({1.0, 10.0});
    stream.pushDepth({1.1, 11.0});

    EXPECT_FALSE(stream.getInterDepth(0.99).has_value());
    EXPECT_FALSE(stream.getInterDepth(1.11).has_value());
}

// Verify out of order insertion of packets
TEST_F(TestDepthStream, OutOfOrderPackets) {
    stream.pushDepth({1.0, 10.0});
    stream.pushDepth({1.2, 12.0});
    stream.pushDepth({1.1, 11.0}); // Delayed packet[cite: 4]

    auto mid = stream.getInterDepth(1.15);
    ASSERT_TRUE(mid.has_value());
    EXPECT_DOUBLE_EQ(mid->z, 11.5);
}

// Verify sensor health state transitions
TEST_F(TestDepthStream, SensorHealthTransitions) {
    // Test sensor timing out
    PoolSLAM::SensorHealth h1 = stream.waitUntil(10.0);
    EXPECT_EQ(h1, PoolSLAM::SensorHealth::DEGRADED);

    // Simulate repeated timeouts up to the maximum number of allowed misses
    for (int i = 0; i < 10; i++) {
        stream.waitUntil(10.0 + i);
    }

    PoolSLAM::SensorHealth disabledHealth = stream.waitUntil(20.0);
    EXPECT_EQ(disabledHealth, PoolSLAM::SensorHealth::DISABLED);

    // Must bypass wait immediately when disabled
    auto start = std::chrono::steady_clock::now();
    stream.waitUntil(25.0);
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start
    ).count();
    EXPECT_LT(duration, 5); // Should return instantly without hitting 100ms timeout
}
