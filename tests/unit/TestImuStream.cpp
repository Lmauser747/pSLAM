#include <vector>

#include <gtest/gtest.h>
#include <eigen3/Eigen/Core>

#include "pool_slam/Hal.hpp"
#include "pool_slam/HalTypes.hpp"

class TestImuStream : public ::testing::Test{
protected:
    PoolSLAM::ImuStream stream;

    PoolSLAM::Imu createSample(double t, double val) {
        PoolSLAM::Imu sample;
        sample.timestamp = t;
        sample.a = Eigen::Vector3d(val, 0.0, 0.0);
        sample.w = Eigen::Vector3d(0.0, val, 0.0);
        return sample;
    }
};

// Verify interpolation on boundaries and intermediate sample inclusion
TEST_F(TestImuStream, BoundaryInterpolation){
    stream.pushImu(createSample(1.0, 10.0));
    stream.pushImu(createSample(1.1, 11.0));
    stream.pushImu(createSample(1.2, 12.0));

    std::vector<PoolSLAM::Imu> batch;
    bool success = stream.extractRange(1.05, 1.15, batch);

    // Ensure extraction was successful
    ASSERT_TRUE(success);
    ASSERT_EQ(batch.size(), 3);

    // Test that all three values in batch are correct
    EXPECT_NEAR(batch[0].timestamp, 1.05, 1e-6);
    EXPECT_NEAR(batch[0].a.x(), 10.5, 1e-6);

    EXPECT_NEAR(batch[1].timestamp, 1.1, 1e-6);
    EXPECT_NEAR(batch[1].a.x(),11.0 , 1e-6);

    EXPECT_NEAR(batch[2].timestamp, 1.15, 1e-6);
    EXPECT_NEAR(batch[2].a.x(), 11.5, 1e-6);
}

