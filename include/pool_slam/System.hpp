#pragma once

#include <memory>
#include <string>

#include <eigen3/Eigen/Dense>

#include "Hal.hpp"
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
public:
    ImuStream imuStream;
    DepthStream depthStream;
    ImageStream imageStream;
private:
    const SystemConfig config;
    class Core;

    // Pimpl
    std::unique_ptr<Core> core;
public:
    System(SystemConfig sysConfig);
    ~System();

    // get the pose
    Pose getPose();

    System(System&&) noexcept;
    System& operator=(System&&) noexcept;
};

}