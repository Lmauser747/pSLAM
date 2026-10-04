#include <thread>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "hal/Synchronizer.hpp"
#include "pool_slam/Hal.hpp"
#include "pool_slam/HalTypes.hpp"
#include "pool_slam/SystemTypes.hpp"

class TestSynchronizer : public ::testing::Test{
protected:
    const size_t width = 640;
    const size_t height = 480;
    const size_t poolSize = 4;
    const size_t readyQueueSize = 2;

    std::vector<uint8_t> dummyPixels{std::vector<uint8_t>(width * height, 128)};

    PoolSLAM::ImuStream imuStream;
    PoolSLAM::DepthStream depthStream;
    PoolSLAM::ImageStream imageStream{width, height, readyQueueSize, poolSize};

    PoolSLAM::Synchronizer sync{imuStream, depthStream, imageStream};

    void SetUp() override{
        sync.start();
    }

    void TearDown() override{
        sync.stop();
    }
};

// Test that the first frame comes in and is given an empty IMU batch
TEST_F(TestSynchronizer, FirstFrameAnchorT0) {
    depthStream.pushDepth({1.0, 5.0});
    imageStream.pushFrame(dummyPixels.data(), 1.0);

    PoolSLAM::SyncedPacket packet;
    bool success = sync.getNextPacket(packet);

    ASSERT_TRUE(success);
    EXPECT_DOUBLE_EQ(packet.timestamp, 1.0);
    EXPECT_TRUE(packet.imu.empty()); // T0 must not contain IMU factors
    ASSERT_TRUE(packet.depth.has_value());
    EXPECT_DOUBLE_EQ(packet.depth->z, 5.0);

    imageStream.recycleFrame(packet.image);
}