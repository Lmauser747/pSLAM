#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "backend/Backend.hpp"
#include "frontend/Frontend.hpp"
#include "hal/Synchronizer.hpp"
#include "pool_slam/Hal.hpp"
#include "pool_slam/System.hpp"
#include "pool_slam/SystemTypes.hpp"

namespace PoolSLAM {

class System::Core {
private:
    ImuStream imuStream;
    DepthStream depthStream;
    ImageStream imageStream;
    Synchronizer sync;
    Frontend frontend;
    Backend backend;

public:
    Core(
        const FrontendConfig &feConfig,
        const BackendConfig &beConfig,
        const std::string &vocabPath,
        int imageWidth,
        int imageHeight
    ) : 
        imageStream(imageWidth, imageHeight),
        sync(imuStream, depthStream, imageStream),
        frontend(feConfig, sync, vocabPath),
        backend(frontend, frontend.getLfTracker(), beConfig) {}

    ~Core() {
        stop();
    }

    void start() {
        sync.start();
        frontend.start();
        backend.start();
    }

    void stop() {
        backend.end();
        frontend.end();
        sync.stop();
    }

    bool pushImu(const Imu &imu) {
        return imuStream.pushImu(imu);
    }

    bool pushDepth(const Depth &depth) {
        return depthStream.pushDepth(depth);
    }

    bool pushFrame(uint8_t *rawPixels, double timestamp) {
        return imageStream.pushFrame(rawPixels, timestamp);
    }

    Pose getPose() const {
        return backend.getCurrentPose();
    }
};

} // namespace PoolSLAM
