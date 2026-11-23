#pragma once

#include <gtest/gtest.h>
#include <vector>
#include <array>
#include <string>
#include <fstream>
#include <memory>
#include <cmath>
#include <cstdint>

#include "CoreTypes.hpp"
#include "frontend/LfTracker.hpp"

namespace PoolSLAM {

class TestLfTracker : public ::testing::Test {
protected:
    static constexpr size_t W = 640;
    static constexpr size_t H = 480;

    static std::string findVocabPath() {
        const std::vector<std::string> candidatePaths = {
            "benchmarks/accuracy/vocab.bin",
            "../benchmarks/accuracy/vocab.bin",
            "../../benchmarks/accuracy/vocab.bin",
            "../../../benchmarks/accuracy/vocab.bin"
        };
        for (const auto& path : candidatePaths) {
            std::ifstream f(path);
            if (f.good()) {
                return path;
            }
        }
        return "";
    }

    static LfTrackerConfig makeConfig() {
        LfTrackerConfig cfg;
        cfg.w = W;
        cfg.h = H;
        cfg.numBuckets = 50;
        cfg.K << 400.0f, 0.0f, 320.0f,
                 0.0f, 400.0f, 240.0f,
                 0.0f, 0.0f, 1.0f;
        cfg.minTfIdfScore = 0.05f;
        cfg.excludedFrames = 20;
        cfg.ransacInliearThreshold = 20;
        cfg.ransacIters = 20;
        cfg.minInliearRatio = 0.30f;
        cfg.maxReprojectionError = 2.0f;
        return cfg;
    }

    static LfTracker createTracker(bool loadVocab = false) {
        std::string vocab = loadVocab ? findVocabPath() : "";
        return LfTracker(makeConfig(), vocab);
    }

    static TrackedFeature<5> makeTrackedFeature(uint32_t id, float x, float y) {
        TrackedFeature<5> f;
        f.id = id;
        f.ptCurr = Eigen::Vector2f(x, y);
        f.ptPrev = Eigen::Vector2f(x, y);
        f.ptUndis = Eigen::Vector2f((x - 320.0f) / 400.0f, (y - 240.0f) / 400.0f);
        f.patch.fill(128);
        f.xGrad.fill(0);
        f.yGrad.fill(0);
        f.HInv = Eigen::Matrix2f::Identity();
        f.age = 5;
        return f;
    }

    // Accessors for private members and methods
    static std::vector<Keyframe>& getKeyframes(LfTracker& tracker) {
        return tracker.keyframes;
    }

    static const std::vector<Keyframe>& getKeyframes(const LfTracker& tracker) {
        return tracker.keyframes;
    }

    static bool getBusy(const LfTracker& tracker) {
        return tracker.busy.load();
    }

    static void setBusy(LfTracker& tracker, bool val) {
        tracker.busy.store(val);
    }

    static bool getDataReady(const LfTracker& tracker) {
        return tracker.dataReady.load();
    }

    static void setDataReady(LfTracker& tracker, bool val) {
        tracker.dataReady.store(val);
    }

    static void setStagedMatch(LfTracker& tracker, const BackendLfPacket& packet) {
        tracker.stagedMatch = packet;
        tracker.dataReady.store(true);
    }

    static const std::optional<BackendLfPacket>& getStagedMatch(const LfTracker& tracker) {
        return tracker.stagedMatch;
    }

    static uint8_t* getLocalBuffer(LfTracker& tracker) {
        return tracker.localBuffer.get();
    }

    static void callGaussianBlur(LfTracker& tracker) {
        tracker.gaussianBlur();
    }

    static void callCalculateDescriptors(LfTracker& tracker) {
        tracker.calculateDescriptors();
    }

    static size_t callMatchFeatures(LfTracker& tracker, size_t curId, size_t matchId) {
        return tracker.matchFeatures(curId, matchId);
    }

    static LfMatching& getMatching(LfTracker& tracker) {
        return tracker.matching;
    }

    static descriptor::BinaryBowTree& getVocabTree(LfTracker& tracker) {
        return tracker.vocabTree;
    }

    static descriptor::BowDatabase& getDatabase(LfTracker& tracker) {
        return tracker.database;
    }
};

} // namespace PoolSLAM
