#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>
#include <eigen3/Eigen/Dense>

#include "pool_slam/HalTypes.hpp"
#include "pool_slam/System.hpp"
#include "pool_slam/SystemTypes.hpp"

#include "TestFrontendHelpers.hpp"
#include "TestSystemHelpers.hpp"

namespace PoolSLAM {

// ============================================================================
// SystemConfig Tests: Black-box YAML parsing, bounds checking, and failure modes
// ============================================================================

class SystemConfigTest : public ::testing::Test {
protected:
    std::string validConfigFile = "/tmp/test_system_cfg_valid.yaml";
    std::string invalidConfigFile = "/tmp/test_system_cfg_invalid.yaml";
    std::string vocabFile = "/tmp/test_system_cfg_vocab.bin";

    void SetUp() override {
        test_helpers::writeTempVocabFile(vocabFile);
        test_helpers::writeValidSystemConfigFile(
            validConfigFile, vocabFile,
            640, 480, 525.40f, 525.85f, 319.50f, 239.80f,
            {-0.2814f, 0.0912f, 0.0004f, -0.0002f},
            Eigen::Vector3d(0.0, 0.0, 0.05));
    }

    void TearDown() override {
        std::remove(validConfigFile.c_str());
        std::remove(invalidConfigFile.c_str());
        std::remove(vocabFile.c_str());
    }
};

TEST_F(SystemConfigTest, LoadsValidYamlConfigAndPopulatesFields) {
    SystemConfig cfg(validConfigFile);

    EXPECT_EQ(cfg.w, 640);
    EXPECT_EQ(cfg.h, 480);
    EXPECT_FLOAT_EQ(cfg.fx, 525.40f);
    EXPECT_FLOAT_EQ(cfg.fy, 525.85f);
    EXPECT_FLOAT_EQ(cfg.cx, 319.50f);
    EXPECT_FLOAT_EQ(cfg.cy, 239.80f);
    EXPECT_FLOAT_EQ(cfg.k1, -0.2814f);
    EXPECT_FLOAT_EQ(cfg.k2, 0.0912f);
    EXPECT_FLOAT_EQ(cfg.p1, 0.0004f);
    EXPECT_FLOAT_EQ(cfg.p2, -0.0002f);
    EXPECT_EQ(cfg.vocabPath, vocabFile);

    // Verify 3x4 bodyToCamera rotation and translation
    EXPECT_DOUBLE_EQ(cfg.bodyToCamera(0, 0), 0.0);
    EXPECT_DOUBLE_EQ(cfg.bodyToCamera(0, 2), 1.0);
    EXPECT_DOUBLE_EQ(cfg.bodyToCamera(0, 3), 0.12);
    EXPECT_DOUBLE_EQ(cfg.bodyToCamera(1, 0), -1.0);
    EXPECT_DOUBLE_EQ(cfg.bodyToCamera(2, 1), -1.0);
    EXPECT_DOUBLE_EQ(cfg.bodyToCamera(2, 3), -0.04);

    // Verify 3x1 bodyToDepth lever arm
    EXPECT_DOUBLE_EQ(cfg.bodyToDepth(0, 0), 0.0);
    EXPECT_DOUBLE_EQ(cfg.bodyToDepth(1, 0), 0.0);
    EXPECT_DOUBLE_EQ(cfg.bodyToDepth(2, 0), 0.05);
}

TEST_F(SystemConfigTest, ThrowsOnNonexistentFile) {
    EXPECT_THROW({ SystemConfig cfg("/tmp/nonexistent_system_config_12345.yaml"); }, std::runtime_error);
}

TEST_F(SystemConfigTest, ThrowsOnMissingPipelineSection) {
    const std::string content =
        "camera:\n"
        "  width: 640\n"
        "  height: 480\n"
        "  intrinsics: {fx: 500, fy: 500, cx: 320, cy: 240}\n"
        "  distortion_coeffs: [0, 0, 0, 0]\n"
        "extrinsics:\n"
        "  T_body_camera: [[1,0,0,0],[0,1,0,0],[0,0,1,0]]\n"
        "  t_body_depth: [0, 0, 0]\n";
    test_helpers::writeCustomConfigFile(invalidConfigFile, content);
    EXPECT_THROW({ SystemConfig cfg(invalidConfigFile); }, std::runtime_error);
}

TEST_F(SystemConfigTest, ThrowsOnMissingCameraIntrinsics) {
    const std::string content =
        "pipeline:\n"
        "  bow_vocabulary_path: \"/tmp/vocab.bin\"\n"
        "camera:\n"
        "  width: 640\n"
        "  height: 480\n"
        "  distortion_coeffs: [0, 0, 0, 0]\n"
        "extrinsics:\n"
        "  T_body_camera: [[1,0,0,0],[0,1,0,0],[0,0,1,0]]\n"
        "  t_body_depth: [0, 0, 0]\n";
    test_helpers::writeCustomConfigFile(invalidConfigFile, content);
    EXPECT_THROW({ SystemConfig cfg(invalidConfigFile); }, std::runtime_error);
}

TEST_F(SystemConfigTest, ThrowsOnInvalidDistortionCoeffsCount) {
    const std::string content =
        "pipeline:\n"
        "  bow_vocabulary_path: \"/tmp/vocab.bin\"\n"
        "camera:\n"
        "  width: 640\n"
        "  height: 480\n"
        "  intrinsics: {fx: 500, fy: 500, cx: 320, cy: 240}\n"
        "  distortion_coeffs: [0.1, -0.05, 0.001]\n" // 3 elements instead of 4
        "extrinsics:\n"
        "  T_body_camera: [[1,0,0,0],[0,1,0,0],[0,0,1,0]]\n"
        "  t_body_depth: [0, 0, 0]\n";
    test_helpers::writeCustomConfigFile(invalidConfigFile, content);
    EXPECT_THROW({ SystemConfig cfg(invalidConfigFile); }, std::runtime_error);
}

TEST_F(SystemConfigTest, ThrowsOnInvalidExtrinsicsMatrixDimensions) {
    const std::string content =
        "pipeline:\n"
        "  bow_vocabulary_path: \"/tmp/vocab.bin\"\n"
        "camera:\n"
        "  width: 640\n"
        "  height: 480\n"
        "  intrinsics: {fx: 500, fy: 500, cx: 320, cy: 240}\n"
        "  distortion_coeffs: [0, 0, 0, 0]\n"
        "extrinsics:\n"
        "  T_body_camera:\n"
        "    - [1.0, 0.0, 0.0]\n" // 3 columns instead of 4
        "    - [0.0, 1.0, 0.0]\n"
        "    - [0.0, 0.0, 1.0]\n"
        "  t_body_depth: [0, 0, 0]\n";
    test_helpers::writeCustomConfigFile(invalidConfigFile, content);
    EXPECT_THROW({ SystemConfig cfg(invalidConfigFile); }, std::runtime_error);
}

TEST_F(SystemConfigTest, ThrowsOnInvalidDepthLeverArmDimensions) {
    const std::string content =
        "pipeline:\n"
        "  bow_vocabulary_path: \"/tmp/vocab.bin\"\n"
        "camera:\n"
        "  width: 640\n"
        "  height: 480\n"
        "  intrinsics: {fx: 500, fy: 500, cx: 320, cy: 240}\n"
        "  distortion_coeffs: [0, 0, 0, 0]\n"
        "extrinsics:\n"
        "  T_body_camera: [[1,0,0,0],[0,1,0,0],[0,0,1,0]]\n"
        "  t_body_depth: [0.0, 0.0]\n"; // 2 elements instead of 3
    test_helpers::writeCustomConfigFile(invalidConfigFile, content);
    EXPECT_THROW({ SystemConfig cfg(invalidConfigFile); }, std::runtime_error);
}

// ============================================================================
// System Fixture: Tests Lifecycle, Data Ingestion, and Full Pipeline
// ============================================================================

class SystemTest : public ::testing::Test {
protected:
    std::string configFile = "/tmp/test_system_main_config.yaml";
    std::string vocabFile = "/tmp/test_system_main_vocab.bin";

    std::unique_ptr<SystemConfig> config;
    std::unique_ptr<System> system;

    std::vector<uint8_t> testScene;

    void SetUp() override {
        test_helpers::writeTempVocabFile(vocabFile);
        test_helpers::writeValidSystemConfigFile(
            configFile, vocabFile,
            640, 480, 525.0f, 525.0f, 320.0f, 240.0f,
            {0.0f, 0.0f, 0.0f, 0.0f},
            Eigen::Vector3d(0.0, 0.0, 0.10)); // 0.10m lever arm

        config = std::make_unique<SystemConfig>(configFile);
        system = std::make_unique<System>(*config);

        Eigen::Matrix3f K = test_helpers::makeDefaultK();
        testScene = test_helpers::renderProjectedGridScene(640, 480, K);
    }

    void TearDown() override {
        if (system) {
            system->stop();
            system.reset();
        }
        std::remove(configFile.c_str());
        std::remove(vocabFile.c_str());
    }
};

// ----------------------------------------------------------------------------
// Lifecycle Tests
// ----------------------------------------------------------------------------

TEST_F(SystemTest, ConfigGetterMatchesLoadedConfiguration) {
    const SystemConfig &cfg = system->getConfig();
    EXPECT_EQ(cfg.w, 640);
    EXPECT_EQ(cfg.h, 480);
    EXPECT_FLOAT_EQ(cfg.fx, 525.0f);
    EXPECT_FLOAT_EQ(cfg.fy, 525.0f);
    EXPECT_DOUBLE_EQ(cfg.bodyToDepth(2, 0), 0.10);
}

TEST_F(SystemTest, InitialPoseIsIdentityZeroBeforeProcessing) {
    const Pose initialPose = system->getPose();
    EXPECT_DOUBLE_EQ(initialPose.timestamp, 0.0);
    EXPECT_TRUE(initialPose.t.isZero());
    EXPECT_TRUE(initialPose.v.isZero());
    EXPECT_DOUBLE_EQ(initialPose.r.w(), 1.0);
    EXPECT_DOUBLE_EQ(initialPose.r.x(), 0.0);
    EXPECT_DOUBLE_EQ(initialPose.r.y(), 0.0);
    EXPECT_DOUBLE_EQ(initialPose.r.z(), 0.0);
}

TEST_F(SystemTest, StartAndStopLifecycleAreIdempotent) {
    // Repeated starts
    EXPECT_NO_THROW(system->start());
    EXPECT_NO_THROW(system->start());

    // Repeated stops
    EXPECT_NO_THROW(system->stop());
    EXPECT_NO_THROW(system->stop());

    // Restart and final stop
    EXPECT_NO_THROW(system->start());
    EXPECT_NO_THROW(system->stop());
}

TEST_F(SystemTest, DestructorShutsDownRunningThreadsCleanly) {
    EXPECT_NO_THROW({
        System localSys(*config);
        localSys.start();
        test_helpers::pushSystemInertialAndDepthWindow(localSys, 0.0, 0.1);
        // Exits scope while threads are running; RAII destructor must join cleanly
    });
}

TEST_F(SystemTest, DestructorWhenNeverStartedIsSafeNoOp) {
    EXPECT_NO_THROW({
        System localSys(*config);
        // Destructs without ever calling start()
    });
}

// ----------------------------------------------------------------------------
// Data Ingestion Tests
// ----------------------------------------------------------------------------

TEST_F(SystemTest, PushImuAndPushDepthSucceedWithValidSamples) {
    Imu imuSample = test_helpers::createImuSample(0.1);
    EXPECT_TRUE(system->pushImu(imuSample));

    Depth depthSample{0.1, 5.0};
    EXPECT_TRUE(system->pushDepth(depthSample));
}

TEST_F(SystemTest, PushFrameNullPointerFailsSafely) {
    EXPECT_FALSE(system->pushFrame(nullptr, 0.1));
}

TEST_F(SystemTest, PushFrameValidBufferSucceeds) {
    EXPECT_TRUE(system->pushFrame(testScene.data(), 0.1));
}

// ----------------------------------------------------------------------------
// Full End-to-End Pipeline Integration Tests
// ----------------------------------------------------------------------------

TEST_F(SystemTest, FirstFrameAnchorInitializesPoseWithDepthAndLeverArm) {
    system->start();

    const double t0 = 0.10;
    const double rawDepth = 6.00; // Measured distance
    const double leverArmZ = 0.10; // Calibrated lever arm in config
    const double expectedInitialZ = rawDepth - leverArmZ; // 5.90m

    // Push synchronized IMU and depth around Frame 0
    test_helpers::pushSystemInertialAndDepthWindow(
        *system, 0.0, 0.25, rawDepth, Eigen::Vector3d(0.0, 0.0, 9.81));

    // Push Frame 0 with visual features
    ASSERT_TRUE(system->pushFrame(testScene.data(), t0));

    // Wait for pipeline processing
    ASSERT_TRUE(test_helpers::waitForSystemPoseTimestamp(*system, t0));

    const Pose currentPose = system->getPose();
    EXPECT_NEAR(currentPose.timestamp, t0, 1e-3);
    EXPECT_NEAR(currentPose.t.z(), expectedInitialZ, 0.15);
    EXPECT_NEAR(currentPose.t.x(), 0.0, 0.15);
    EXPECT_NEAR(currentPose.t.y(), 0.0, 0.15);
    EXPECT_LT(currentPose.v.norm(), 0.10);
}

TEST_F(SystemTest, SequentialMultiFrameTrackingAdvancesPoseTimestamp) {
    system->start();

    const double rawDepth = 5.0;
    test_helpers::pushSystemInertialAndDepthWindow(
        *system, 0.0, 0.50, rawDepth, Eigen::Vector3d(0.0, 0.0, 9.81));

    // Ingest sequential frames at 10 Hz
    ASSERT_TRUE(system->pushFrame(testScene.data(), 0.10));
    ASSERT_TRUE(test_helpers::waitForSystemPoseTimestamp(*system, 0.10));

    ASSERT_TRUE(system->pushFrame(testScene.data(), 0.20));
    ASSERT_TRUE(test_helpers::waitForSystemPoseTimestamp(*system, 0.20));

    ASSERT_TRUE(system->pushFrame(testScene.data(), 0.30));
    ASSERT_TRUE(test_helpers::waitForSystemPoseTimestamp(*system, 0.30));

    const Pose finalPose = system->getPose();
    EXPECT_GE(finalPose.timestamp, 0.30 - 1e-4);
    EXPECT_TRUE(std::isfinite(finalPose.t.x()));
    EXPECT_TRUE(std::isfinite(finalPose.t.y()));
    EXPECT_TRUE(std::isfinite(finalPose.t.z()));
    EXPECT_TRUE(std::isfinite(finalPose.v.x()));
    EXPECT_TRUE(std::isfinite(finalPose.v.y()));
    EXPECT_TRUE(std::isfinite(finalPose.v.z()));
}

} // namespace PoolSLAM
