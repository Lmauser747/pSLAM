#pragma once

#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <deque>
#include <unordered_map>
#include <string>
#include <chrono>
#include <thread>
#include <atomic>
#include <cmath>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Cal3_S2.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/navigation/ImuBias.h>
#include <gtsam/navigation/NavState.h>
#include <gtsam/navigation/PreintegrationParams.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/ISAM2Result.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>

#include "CoreTypes.hpp"
#include "pool_slam/SystemTypes.hpp"
#include "pool_slam/Hal.hpp"
#include "hal/Synchronizer.hpp"
#include "frontend/Frontend.hpp"
#include "frontend/LfTracker.hpp"
#include "backend/Backend.hpp"

namespace PoolSLAM {

// Friend accessor for Frontend to push packets directly into hfQueue
class TestFrontend {
public:
    static void pushHfPacket(Frontend& fe, std::unique_ptr<BackendHfPacket> pkt) {
        std::lock_guard<std::mutex> lock(fe.m_hfQueue);
        fe.hfQueue.push(std::move(pkt));
        fe.c_hfQueue.notify_one();
    }

    static gtsam::imuBias::ConstantBias getImuBias(Frontend& fe) {
        std::lock_guard<std::mutex> lock(fe.m_imuBias);
        return fe.imuBias;
    }
};

// Friend accessor for LfTracker to push loop closure matches
class TestLfTracker {
public:
    static void setStagedMatch(LfTracker& tracker, const BackendLfPacket& packet) {
        tracker.stagedMatch = packet;
        tracker.dataReady.store(true);
    }

    static const std::vector<Keyframe>& getKeyframes(const LfTracker& tracker) {
        return tracker.keyframes;
    }
};

class BackendTest : public ::testing::Test {
protected:
    static constexpr size_t W = 640;
    static constexpr size_t H = 480;

    static FrontendConfig makeDefaultFrontendConfig() {
        FrontendConfig cfg;
        cfg.w = W;
        cfg.h = H;
        cfg.K << 400.0f, 0.0f, 320.0f,
                 0.0f, 400.0f, 240.0f,
                 0.0f, 0.0f, 1.0f;
        cfg.dist = Distortion{0.0f, 0.0f, 0.0f, 0.0f};
        cfg.lfConfig.w = W;
        cfg.lfConfig.h = H;
        cfg.lfConfig.K = cfg.K;
        cfg.lfConfig.dist = cfg.dist;
        cfg.lfConfig.numBuckets = 50;
        cfg.bodyToCameraRot = Eigen::Matrix3d::Identity();
        return cfg;
    }

    static BackendConfig makeDefaultBackendConfig() {
        BackendConfig cfg;
        cfg.bodyToCameraFrame = gtsam::Pose3::Identity();
        cfg.bodyToDepthSensor = gtsam::Point3(0.0, 0.0, 0.0);
        cfg.K << 400.0, 0.0, 320.0,
                 0.0, 400.0, 240.0,
                 0.0, 0.0, 1.0;
        cfg.depthSigma = 0.05;
        cfg.depthHuberK = 1.345;
        cfg.keyframeWindowSize = 20;
        return cfg;
    }

    struct TestHarness {
        ImuStream imuStream;
        DepthStream depthStream;
        ImageStream imageStream{W, H};
        Synchronizer sync{imuStream, depthStream, imageStream};
        Frontend frontend;
        Backend backend;

        explicit TestHarness(const BackendConfig& bCfg = makeDefaultBackendConfig(),
                             const FrontendConfig& feCfg = makeDefaultFrontendConfig())
            : frontend(feCfg, sync, ""),
              backend(frontend, frontend.getLfTracker(), bCfg) {}
    };

    static std::unique_ptr<gtsam::PreintegratedImuMeasurements> makePim(
        const gtsam::Vector3& acc = gtsam::Vector3(0.0, 0.0, 9.81),
        const gtsam::Vector3& gyro = gtsam::Vector3::Zero(),
        double dt = 0.1,
        const gtsam::imuBias::ConstantBias& bias = gtsam::imuBias::ConstantBias()) {
        auto params = gtsam::PreintegrationParams::MakeSharedU(9.81);
        params->accelerometerCovariance = Eigen::Matrix3d::Identity() * 1e-3;
        params->gyroscopeCovariance = Eigen::Matrix3d::Identity() * 1e-4;
        params->integrationCovariance = Eigen::Matrix3d::Identity() * 1e-5;
        auto pim = std::make_unique<gtsam::PreintegratedImuMeasurements>(params, bias);
        pim->integrateMeasurement(acc, gyro, dt);
        return pim;
    }

    // Accessors for Backend private members
    static bool isInitialized(const Backend& b) { return b.isInitialized; }
    static void setInitialized(Backend& b, bool val) { b.isInitialized = val; }

    static bool isRunning(const Backend& b) { return b.runBackend.load(); }

    static uint32_t getPrevFrameId(const Backend& b) { return b.prevFrameId; }
    static void setPrevFrameId(Backend& b, uint32_t id) { b.prevFrameId = id; }

    static double getPrevTimestamp(const Backend& b) { return b.prevTimestamp; }
    static void setPrevTimestamp(Backend& b, double t) { b.prevTimestamp = t; }

    static const gtsam::NavState& getPrevNavState(const Backend& b) { return b.prevNavState; }
    static void setPrevNavState(Backend& b, const gtsam::NavState& state) { b.prevNavState = state; }

    static const gtsam::imuBias::ConstantBias& getPrevBias(const Backend& b) { return b.prevBias; }
    static void setPrevBias(Backend& b, const gtsam::imuBias::ConstantBias& bias) { b.prevBias = bias; }

    static const gtsam::NonlinearFactorGraph& getStagingGraph(const Backend& b) { return b.stagingGraph; }
    static gtsam::NonlinearFactorGraph& getStagingGraph(Backend& b) { return b.stagingGraph; }

    static const gtsam::Values& getStagingValues(const Backend& b) { return b.stagingValues; }
    static gtsam::Values& getStagingValues(Backend& b) { return b.stagingValues; }

    static const gtsam::ISAM2& getSmoother(const Backend& b) { return b.smoother; }
    static gtsam::ISAM2& getSmoother(Backend& b) { return b.smoother; }

    static const std::deque<WindowFrame>& getWindow(const Backend& b) { return b.window; }

    static const gtsam::Values& getSmootherEstimate(const Backend& b) { return b.smootherEstimate; }

    static const gtsam::NonlinearFactorGraph& getGlobalPoseGraph(const Backend& b) { return b.globalPoseGraph; }
    static const gtsam::Values& getGlobalPoseValues(const Backend& b) { return b.globalPoseValues; }

    static const std::unordered_map<uint32_t, std::vector<std::pair<gtsam::Point2, gtsam::Key>>>&
    getActiveFeatureTracks(const Backend& b) {
        return b.activeFeatureTracks;
    }

    static const std::unordered_map<uint32_t, std::shared_ptr<SmartFactor>>& getSmartFactors(const Backend& b) {
        return b.smartFactors;
    }
    static std::unordered_map<uint32_t, std::shared_ptr<SmartFactor>>& getSmartFactors(Backend& b) {
        return b.smartFactors;
    }

    static const std::unordered_map<uint32_t, gtsam::FactorIndex>& getSmartFactorSlots(const Backend& b) {
        return b.smartFactorSlots;
    }

    static const std::deque<TrackedKeyframe>& getTrackedKeyframes(const Backend& b) {
        return b.trackedKeyframes;
    }
    static std::deque<TrackedKeyframe>& getTrackedKeyframes(Backend& b) {
        return b.trackedKeyframes;
    }

    static const std::deque<std::tuple<uint32_t, uint32_t, size_t>>&
    getActiveGlobalOdomEdges(const Backend& b) {
        return b.activeGlobalOdomEdges;
    }

    static bool getHasClosedLoop(const Backend& b) {
        return b.hasClosedLoop;
    }

    static const gtsam::Pose3& getVioToGlobalDrift(const Backend& b) {
        return b.vioToGlobalDrift;
    }

    // Direct invocation of Backend private methods
    static void callInitializeSystem(Backend& b, const BackendHfPacket& pkt) {
        b.initializeSystem(pkt);
    }

    static void callProcessFrame(Backend& b, const BackendHfPacket& pkt) {
        b.processFrame(pkt);
    }

    static bool callProcessLfPacket(Backend& b, const BackendLfPacket& pkt) {
        return b.processLfPacket(pkt);
    }

    static void callSyncLfTracker(Backend& b, bool forceAll) {
        b.syncLfTracker(forceAll);
    }

    static void callPublishPose(Backend& b, const gtsam::Pose3& pose, const gtsam::Vector3& velocity, double timestamp) {
        b.publishPose(pose, velocity, timestamp);
    }

    static bool callSolveSmoother(Backend& b, uint32_t currentFrameId, double = 0.0, const gtsam::FactorIndices& removeIndices = {}) {
        return b.solveSmoother(currentFrameId, removeIndices).has_value();
    }

    static bool callSolveGlobal(Backend& b, uint32_t currentFrameId) {
        return b.solveGlobal(currentFrameId);
    }
};

} // namespace PoolSLAM
