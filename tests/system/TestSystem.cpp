#include "TestSystemHelpers.hpp"

namespace PoolSLAM {

// =========================================================================
// 1. SystemConfig YAML Schema Parsing
// =========================================================================

TEST_F(TestSystem, NominalParsingAllFieldsMatch) {
    const std::string yaml = makeValidYamlContent(
        "benchmarks/accuracy/vocab.bin",
        640, 480,
        525.40f, 525.85f, 319.50f, 239.80f,
        "[-0.2814, 0.0912, 0.0004, -0.0002]",
        "[-0.15, 0.00, 0.05]"
    );

    TempConfigFile tempFile(yaml);
    SystemConfig cfg(tempFile.path);

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
    EXPECT_EQ(cfg.vocabPath, "benchmarks/accuracy/vocab.bin");

    // Body to camera extrinsics: top 3x4 block
    EXPECT_DOUBLE_EQ(cfg.bodyToCamera(0, 0), 0.0);
    EXPECT_DOUBLE_EQ(cfg.bodyToCamera(0, 2), 1.0);
    EXPECT_DOUBLE_EQ(cfg.bodyToCamera(0, 3), 0.12);
    EXPECT_DOUBLE_EQ(cfg.bodyToCamera(1, 0), -1.0);
    EXPECT_DOUBLE_EQ(cfg.bodyToCamera(2, 1), -1.0);
    EXPECT_DOUBLE_EQ(cfg.bodyToCamera(2, 3), -0.04);

    // Body to depth lever arm
    EXPECT_DOUBLE_EQ(cfg.bodyToDepth(0, 0), -0.15);
    EXPECT_DOUBLE_EQ(cfg.bodyToDepth(1, 0), 0.00);
    EXPECT_DOUBLE_EQ(cfg.bodyToDepth(2, 0), 0.05);

    // Without initial_pose block
    EXPECT_FALSE(cfg.hasInitialPose);
}

TEST_F(TestSystem, OptionalInitialPoseParsing) {
    // Case 1: With valid initial_pose matrix in YAML
    const std::string initialPoseYaml =
        "initial_pose:\n"
        "  - [ 1.0,  0.0,  0.0,  5.0]\n"
        "  - [ 0.0,  1.0,  0.0, -3.0]\n"
        "  - [ 0.0,  0.0,  1.0,  8.0]\n"
        "  - [ 0.0,  0.0,  0.0,  1.0]\n";

    TempConfigFile tempFileWithPose(makeValidYamlContent("", W, H, 525.4f, 525.85f, 319.5f, 239.8f,
                                                        "[-0.2814, 0.0912, 0.0004, -0.0002]",
                                                        "[-0.15, 0.00, 0.05]",
                                                        initialPoseYaml));
    SystemConfig cfgWithPose(tempFileWithPose.path);
    EXPECT_TRUE(cfgWithPose.hasInitialPose);
    EXPECT_DOUBLE_EQ(cfgWithPose.initialPose(0, 3), 5.0);
    EXPECT_DOUBLE_EQ(cfgWithPose.initialPose(1, 3), -3.0);
    EXPECT_DOUBLE_EQ(cfgWithPose.initialPose(2, 3), 8.0);

    // Case 2: Without initial_pose block
    TempConfigFile tempFileNoPose(makeValidYamlContent());
    SystemConfig cfgNoPose(tempFileNoPose.path);
    EXPECT_FALSE(cfgNoPose.hasInitialPose);
}

TEST_F(TestSystem, MissingRequiredKeyThrows) {
    // Missing camera.width
    const std::string missingWidthYaml =
        "pipeline:\n"
        "  bow_vocabulary_path: \"\"\n"
        "camera:\n"
        "  height: 480\n"
        "  intrinsics:\n"
        "    fx: 500\n"
        "    fy: 500\n"
        "    cx: 320\n"
        "    cy: 240\n"
        "  distortion_coeffs: [0.0, 0.0, 0.0, 0.0]\n"
        "extrinsics:\n"
        "  T_body_camera:\n"
        "    - [1, 0, 0, 0]\n"
        "    - [0, 1, 0, 0]\n"
        "    - [0, 0, 1, 0]\n"
        "  t_body_depth: [0, 0, 0]\n";

    TempConfigFile fileNoWidth(missingWidthYaml);
    EXPECT_THROW(SystemConfig cfg(fileNoWidth.path), std::runtime_error);

    // Missing pipeline.bow_vocabulary_path
    const std::string missingVocabKeyYaml =
        "camera:\n"
        "  width: 640\n"
        "  height: 480\n"
        "  intrinsics:\n"
        "    fx: 500\n"
        "    fy: 500\n"
        "    cx: 320\n"
        "    cy: 240\n"
        "  distortion_coeffs: [0.0, 0.0, 0.0, 0.0]\n"
        "extrinsics:\n"
        "  T_body_camera:\n"
        "    - [1, 0, 0, 0]\n"
        "    - [0, 1, 0, 0]\n"
        "    - [0, 0, 1, 0]\n"
        "  t_body_depth: [0, 0, 0]\n";

    TempConfigFile fileNoVocab(missingVocabKeyYaml);
    EXPECT_THROW(SystemConfig cfg(fileNoVocab.path), std::runtime_error);

    // Missing camera.intrinsics
    const std::string missingIntrinsicsYaml =
        "pipeline:\n"
        "  bow_vocabulary_path: \"\"\n"
        "camera:\n"
        "  width: 640\n"
        "  height: 480\n"
        "  distortion_coeffs: [0.0, 0.0, 0.0, 0.0]\n"
        "extrinsics:\n"
        "  T_body_camera:\n"
        "    - [1, 0, 0, 0]\n"
        "    - [0, 1, 0, 0]\n"
        "    - [0, 0, 1, 0]\n"
        "  t_body_depth: [0, 0, 0]\n";

    TempConfigFile fileNoIntrinsics(missingIntrinsicsYaml);
    EXPECT_THROW(SystemConfig cfg(fileNoIntrinsics.path), std::runtime_error);
}

TEST_F(TestSystem, InvalidSequenceSizesThrow) {
    // Distortion array with 3 elements (must be 4)
    TempConfigFile fileDist3(makeValidYamlContent("", W, H, 500, 500, 320, 240,
                                                 "[0.1, -0.2, 0.0]"));
    EXPECT_THROW(SystemConfig cfg(fileDist3.path), std::runtime_error);

    // Distortion array with 5 elements (must be 4)
    TempConfigFile fileDist5(makeValidYamlContent("", W, H, 500, 500, 320, 240,
                                                 "[0.1, -0.2, 0.0, 0.0, 0.5]"));
    EXPECT_THROW(SystemConfig cfg(fileDist5.path), std::runtime_error);

    // Depth vector with 2 elements (must be 3)
    TempConfigFile fileDepth2(makeValidYamlContent("", W, H, 500, 500, 320, 240,
                                                  "[0.0, 0.0, 0.0, 0.0]",
                                                  "[0.1, 0.2]"));
    EXPECT_THROW(SystemConfig cfg(fileDepth2.path), std::runtime_error);

    // Depth vector with 4 elements (must be 3)
    TempConfigFile fileDepth4(makeValidYamlContent("", W, H, 500, 500, 320, 240,
                                                  "[0.0, 0.0, 0.0, 0.0]",
                                                  "[0.1, 0.2, 0.3, 0.4]"));
    EXPECT_THROW(SystemConfig cfg(fileDepth4.path), std::runtime_error);
}

TEST_F(TestSystem, NonexistentFileThrows) {
    EXPECT_THROW(SystemConfig cfg("/definitely/nonexistent/config_path.yaml"), std::runtime_error);
}

// =========================================================================
// 2. System Initialization & Construction
// =========================================================================

TEST_F(TestSystem, ValidConfigurationConstructsSuccessfully) {
    SystemConfig cfg = makeDefaultConfig();
    EXPECT_NO_THROW({
        System system(cfg);
    });
}

TEST_F(TestSystem, NonexistentVocabFileThrows) {
    SystemConfig cfg = makeDefaultConfig();
    cfg.vocabPath = "/definitely/nonexistent/vocab.bin";

    EXPECT_THROW({
        System system(cfg);
    }, std::runtime_error);
}

TEST_F(TestSystem, ConfigInspectionMatchesInput) {
    SystemConfig cfg = makeDefaultConfig();
    System system(cfg);

    const SystemConfig& inspectCfg = system.getConfig();
    EXPECT_EQ(inspectCfg.w, cfg.w);
    EXPECT_EQ(inspectCfg.h, cfg.h);
    EXPECT_FLOAT_EQ(inspectCfg.fx, cfg.fx);
    EXPECT_FLOAT_EQ(inspectCfg.fy, cfg.fy);
    EXPECT_FLOAT_EQ(inspectCfg.cx, cfg.cx);
    EXPECT_FLOAT_EQ(inspectCfg.cy, cfg.cy);
}

// =========================================================================
// 3. Lifecycle Management & Subsystem Bringup
// =========================================================================

TEST_F(TestSystem, StartAndStopLifecycle) {
    SystemConfig cfg = makeDefaultConfig();
    System system(cfg);

    EXPECT_NO_THROW(system.start());
    EXPECT_NO_THROW(system.stop());
}

TEST_F(TestSystem, RepeatedLifecycleCycles) {
    SystemConfig cfg = makeDefaultConfig();
    System system(cfg);

    EXPECT_NO_THROW(system.start());
    EXPECT_NO_THROW(system.stop());
    EXPECT_NO_THROW(system.start());
    EXPECT_NO_THROW(system.stop());
}

TEST_F(TestSystem, RAIITeardownWithoutExplicitStop) {
    SystemConfig cfg = makeDefaultConfig();
    EXPECT_NO_THROW({
        System system(cfg);
        system.start();
        // Exits scope while running; destructor must join threads cleanly
    });
}

// =========================================================================
// 4. Sensor Ingestion Pass-Through
// =========================================================================

TEST_F(TestSystem, PushImuReturnsTrue) {
    SystemConfig cfg = makeDefaultConfig();
    System system(cfg);
    system.start();

    Imu imu;
    imu.timestamp = 1.0;
    imu.a = Eigen::Vector3d(0.0, 0.0, 9.81);
    imu.w = Eigen::Vector3d::Zero();
    imu.t = 20.0f;

    EXPECT_TRUE(system.pushImu(imu));
    system.stop();
}

TEST_F(TestSystem, PushDepthReturnsTrue) {
    SystemConfig cfg = makeDefaultConfig();
    System system(cfg);
    system.start();

    Depth depth;
    depth.timestamp = 1.0;
    depth.z = 5.0;

    EXPECT_TRUE(system.pushDepth(depth));
    system.stop();
}

TEST_F(TestSystem, PushFrameValidation) {
    SystemConfig cfg = makeDefaultConfig();
    System system(cfg);
    system.start();

    // Nullptr must be rejected
    EXPECT_FALSE(system.pushFrame(nullptr, 1.0));

    // Valid frame must be accepted
    std::vector<uint8_t> frame(static_cast<size_t>(cfg.w * cfg.h), 128);
    EXPECT_TRUE(system.pushFrame(frame.data(), 1.0));

    system.stop();
}

// =========================================================================
// 5. State Query Execution (getPose)
// =========================================================================

TEST_F(TestSystem, InitialGetPoseReturnsValidPose) {
    SystemConfig cfg = makeDefaultConfig();
    System system(cfg);

    Pose p = system.getPose();
    EXPECT_NEAR(p.r.norm(), 1.0, 1e-4);
    EXPECT_FALSE(std::isnan(p.t.x()));
    EXPECT_FALSE(std::isnan(p.t.y()));
    EXPECT_FALSE(std::isnan(p.t.z()));
}

TEST_F(TestSystem, ThreadSafeStateQueryConcurrency) {
    SystemConfig cfg = makeDefaultConfig();
    System system(cfg);
    system.start();

    std::atomic<bool> stopProducer{false};
    std::atomic<size_t> readCount{0};

    // Producer pushes continuous IMU, depth, and frames
    std::thread producer([&]() {
        double t = 1.0;
        std::vector<uint8_t> frame(static_cast<size_t>(cfg.w * cfg.h), 100);
        for (int i = 0; i < 200 && !stopProducer.load(); ++i) {
            Imu imu;
            imu.timestamp = t;
            imu.a = Eigen::Vector3d(0.0, 0.0, 9.81);
            imu.w = Eigen::Vector3d::Zero();
            system.pushImu(imu);

            Depth d;
            d.timestamp = t;
            d.z = 5.0;
            system.pushDepth(d);

            if (i % 10 == 0) {
                system.pushFrame(frame.data(), t);
            }

            t += 0.01;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    });

    // Consumer queries pose in parallel
    for (int i = 0; i < 500; ++i) {
        Pose p = system.getPose();
        EXPECT_NEAR(p.r.norm(), 1.0, 1e-4);
        ++readCount;
        std::this_thread::yield();
    }

    stopProducer.store(true);
    producer.join();
    system.stop();

    EXPECT_GT(readCount.load(), 0u);
}

} // namespace PoolSLAM
