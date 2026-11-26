#pragma once

#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>
#include <cmath>

#include "pool_slam/System.hpp"
#include "pool_slam/HalTypes.hpp"
#include "pool_slam/SystemTypes.hpp"

namespace PoolSLAM {

class TestSystem : public ::testing::Test {
protected:
    static constexpr int W = 640;
    static constexpr int H = 480;

    struct TempConfigFile {
        std::string path;

        explicit TempConfigFile(const std::string& content) {
            static std::atomic<uint64_t> counter{0};
            const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
            const std::string filename = "pool_slam_test_" + std::to_string(now) + "_" +
                                         std::to_string(counter.fetch_add(1)) + ".yaml";
            path = (std::filesystem::temp_directory_path() / filename).string();
            std::ofstream ofs(path);
            ofs << content;
            ofs.close();
        }

        ~TempConfigFile() {
            std::error_code ec;
            std::filesystem::remove(path, ec);
        }
    };

    static std::string makeValidYamlContent(
        const std::string& vocabPath = "",
        int width = W,
        int height = H,
        float fx = 525.40f,
        float fy = 525.85f,
        float cx = 319.50f,
        float cy = 239.80f,
        const std::string& distCoeffs = "[-0.2814, 0.0912, 0.0004, -0.0002]",
        const std::string& tBodyDepth = "[-0.15, 0.00, 0.05]",
        const std::string& initialPoseBlock = "",
        const std::string& imuBlock = "") {

        std::string yaml =
            "pipeline:\n"
            "  bow_vocabulary_path: \"" + vocabPath + "\"\n"
            "\n"
            "camera:\n"
            "  width: " + std::to_string(width) + "\n"
            "  height: " + std::to_string(height) + "\n"
            "  intrinsics:\n"
            "    fx: " + std::to_string(fx) + "\n"
            "    fy: " + std::to_string(fy) + "\n"
            "    cx: " + std::to_string(cx) + "\n"
            "    cy: " + std::to_string(cy) + "\n"
            "  distortion_coeffs: " + distCoeffs + "\n"
            "\n"
            "extrinsics:\n"
            "  T_body_camera:\n"
            "    - [ 0.0,  0.0,  1.0,  0.12]\n"
            "    - [-1.0,  0.0,  0.0,  0.00]\n"
            "    - [ 0.0, -1.0,  0.0, -0.04]\n"
            "    - [ 0.0,  0.0,  0.0,  1.00]\n"
            "  t_body_depth: " + tBodyDepth + "\n";

        if (!initialPoseBlock.empty()) {
            yaml += initialPoseBlock + "\n";
        }
        if (!imuBlock.empty()) {
            yaml += imuBlock + "\n";
        }

        return yaml;
    }

    static std::string findExistingVocabPath() {
        const std::vector<std::string> candidates = {
            "benchmarks/accuracy/vocab.bin",
            "../benchmarks/accuracy/vocab.bin",
            "../../benchmarks/accuracy/vocab.bin"
        };
        for (const auto& p : candidates) {
            if (std::filesystem::exists(p)) {
                return std::filesystem::canonical(p).string();
            }
        }
        return "";
    }

    static SystemConfig makeDefaultConfig(const std::string& vocabPath = "") {
        TempConfigFile tempFile(makeValidYamlContent(vocabPath));
        return SystemConfig(tempFile.path);
    }
};

} // namespace PoolSLAM
