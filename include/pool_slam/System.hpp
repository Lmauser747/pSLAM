#pragma once

#include <memory>
#include <string>

#include <eigen3/Eigen/Dense>

#include "HalTypes.hpp"
#include "SystemTypes.hpp"


namespace PoolSLAM{

class SystemConfig{
public:
    int w, h; // Image size
    float fx, fy, cx, cy; // Intrinsic matrix
    float k1, k2, p1, p2; // Distortion coefficients
    std::string vocabPath; // Path of BoW vocabulary
    Eigen::Matrix<double, 3, 4> bodyToCamera; // pose transform between body and camera
    Eigen::Matrix<double, 3, 1> bodyToDepth; // pose tranform between body and deoth sensor

    SystemConfig(const std::string& configFile);
};

class System{
private:
    const SystemConfig config;
    class Core;

    // Pimpl
    std::unique_ptr<Core> core;
public:
    explicit System(SystemConfig sysConfig);
    ~System();

    System(const System&) = delete;
    System& operator=(const System&) = delete;
    System(System&&) = delete;
    System& operator=(System&&) = delete;

    // Lifecycle
    void start();
    void stop();

    // Data ingestion
    bool pushImu(const Imu &imu);
    bool pushDepth(const Depth &depth);
    bool pushFrame(uint8_t *rawPixels, double timestamp);

    // Get the pose
    Pose getPose() const;

    // Get config
    const SystemConfig& getConfig() const;
};

}
