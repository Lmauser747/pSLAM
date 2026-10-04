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

