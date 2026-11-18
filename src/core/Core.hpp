#pragma once

#include <memory>
#include <string>
#include <cstddef>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>

#include "frontend/Frontend.hpp"
#include "backend/Backend.hpp"
#include "hal/Synchronizer.hpp"
#include "pool_slam/Hal.hpp"
#include "pool_slam/System.hpp"

namespace PoolSLAM {


class System::Core{
    Synchronizer sync;
    Frontend frontend;
    Backend backend;
public:
    Core(
        FrontendConfig feConfig, BackendConfig beConfig,
        ImuStream &imuStream, DepthStream &depthStream, 
        ImageStream &imageStream, const std::string &voacbPath
    ) : 
    sync(imuStream, depthStream, imageStream), 
    frontend(feConfig, sync, voacbPath), 
    backend(frontend, frontend.getLfTracker(), beConfig){}
};

System::System(SystemConfig sysConfig) : 
imageStream(sysConfig.w, sysConfig.h), config(sysConfig){
    FrontendConfig feConfig;
    BackendConfig beConfig;
    feConfig.w = static_cast<size_t>(sysConfig.w);
    feConfig.h = static_cast<size_t>(sysConfig.h);
    auto K = Eigen::Matrix3f{
        {sysConfig.fx, 0.0, sysConfig.cx},
        {0.0, sysConfig.fy, sysConfig.cy},
        {0.0, 0.0,1.0}
    };
    feConfig.K = K;
    feConfig.dist = {sysConfig.k1, sysConfig.k2, sysConfig.p1, sysConfig.p2};
    beConfig.bodyToCameraFrame = gtsam::Pose3(
        gtsam::Rot3(sysConfig.bodyToCamera.block<3, 3>(0, 0)), 
        sysConfig.bodyToCamera.block<3, 1>(3, 0));

    core = std::make_unique<Core>(feConfig, beConfig, imuStream, depthStream, imageStream, sysConfig.vocabPath);
}


}