#include <chrono>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include <pool_slam/Hal.hpp>
#include <pool_slam/HalTypes.hpp>

class TestImageStream : public ::testing::Test{
protected:
    const size_t width = 640;
    const size_t height = 480;
    const size_t poolSize = 4;
    const size_t readyQueueSize = 2;

    PoolSLAM::ImageStream stream{width, height, readyQueueSize, poolSize};

    std::vector<uint8_t> dummmyPixels{std::vector<uint8_t>(width * height, 128)};
};

// Verify successful frame checkout and recycling
TEST_F(TestImageStream, CheckoutAndRecycle){
    bool pushed = stream.pushFrame(dummmyPixels.data(), 1.0);
    ASSERT_TRUE(pushed);

    PoolSLAM::PixelArray *frame = nullptr;
    PoolSLAM::SensorHealth health = stream.waitForFrame(frame, std::chrono::milliseconds(50));

    // Make sure the frame comes back ok
    ASSERT_EQ(health, PoolSLAM::SensorHealth::ACTIVE);
    ASSERT_NE(frame, nullptr);

    // Check that the frame is the right one
    EXPECT_DOUBLE_EQ(frame->timestamp, 1.0);
    EXPECT_EQ(frame->arr[0], 128);

    // Recycle the frame
    stream.recycleFrame(frame);
}