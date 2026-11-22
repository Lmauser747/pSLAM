#include <cstddef>
#include <cstdint>
#include <ios>
#include <memory>
#include <utility>
#include <fstream>
#include <stdexcept>

#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>

#include "pool_slam/System.hpp"
#include "backend/Backend.hpp"
#include "core/Core.hpp"
#include "frontend/Frontend.hpp"
#include "pool_slam/HalTypes.hpp"
#include "pool_slam/SystemTypes.hpp"

namespace PoolSLAM {

System::System(SystemConfig sysConfig)
    : config(std::move(sysConfig)) {
    FrontendConfig feConfig;
    BackendConfig beConfig;

    feConfig.w = static_cast<size_t>(config.w);
    feConfig.h = static_cast<size_t>(config.h);
    Eigen::Matrix3f K;
    K << config.fx, 0.0f, config.cx,
         0.0f, config.fy, config.cy,
         0.0f, 0.0f, 1.0f;
    feConfig.K = K;
    feConfig.dist = {config.k1, config.k2, config.p1, config.p2};
    feConfig.bodyToCameraRot = config.bodyToCamera.block<3, 3>(0, 0);

    feConfig.lfConfig.w = feConfig.w;
    feConfig.lfConfig.h = feConfig.h;
    feConfig.lfConfig.K = feConfig.K;
    feConfig.lfConfig.dist = feConfig.dist;

    beConfig.bodyToCameraFrame = gtsam::Pose3(
        gtsam::Rot3(config.bodyToCamera.block<3, 3>(0, 0)),
        config.bodyToCamera.block<3, 1>(0, 3)
    );
    beConfig.bodyToDepthSensor = gtsam::Point3(
        config.bodyToDepth(0), config.bodyToDepth(1), config.bodyToDepth(2)
    );
    if (config.hasInitialPose) {
        beConfig.initialPose = gtsam::Pose3(
            gtsam::Rot3(config.initialPose.block<3, 3>(0, 0)),
            config.initialPose.block<3, 1>(0, 3)
        );
    }
    beConfig.K = K.cast<double>();

    if (!config.vocabPath.empty()) {
        std::ifstream testStream(config.vocabPath, std::ios::binary);
        if (!testStream.good()) {
            throw std::runtime_error("System: Vocabulary file does not exist or cannot be opened: " + config.vocabPath);
        }
    }

    core = std::make_unique<Core>(feConfig, beConfig, config.vocabPath, config.w, config.h);
}

System::~System() = default;

void System::start() {
    if (core) {
        core->start();
    }
}

void System::stop() {
    if (core) {
        core->stop();
    }
}

bool System::pushImu(const Imu &imu) {
    return core ? core->pushImu(imu) : false;
}

bool System::pushDepth(const Depth &depth) {
    return core ? core->pushDepth(depth) : false;
}

bool System::pushFrame(uint8_t *rawPixels, double timestamp) {
    return core ? core->pushFrame(rawPixels, timestamp) : false;
}

Pose System::getPose() const {
    return core ? core->getPose() : Pose{};
}

const SystemConfig& System::getConfig() const {
    return config;
}

} // namespace PoolSLAM
