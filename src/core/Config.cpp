#include "pool_slam/System.hpp"

#include <cstddef>
#include <stdexcept>
#include <string>

#include <yaml-cpp/exceptions.h>
#include <yaml-cpp/node/node.h>
#include <yaml-cpp/node/parse.h>
#include <yaml-cpp/yaml.h>

namespace PoolSLAM {
namespace {

YAML::Node requireNode(const YAML::Node& parent, const std::string& key, const std::string& path) {
    YAML::Node child = parent[key];
    if (!child || !child.IsDefined()) {
        throw std::runtime_error("SystemConfig: missing required key '" + path + "'");
    }
    return child;
}

} // namespace

SystemConfig::SystemConfig(const std::string& configFile) {
    try {
        const YAML::Node root = YAML::LoadFile(configFile);

        // Pipeline resources
        const YAML::Node pipelineNode = requireNode(root, "pipeline", "pipeline");
        vocabPath = requireNode(pipelineNode, "bow_vocabulary_path", "pipeline.bow_vocabulary_path").as<std::string>();

        // Camera resolution and intrinsics
        const YAML::Node cameraNode = requireNode(root, "camera", "camera");
        w = requireNode(cameraNode, "width", "camera.width").as<int>();
        h = requireNode(cameraNode, "height", "camera.height").as<int>();

        const YAML::Node intrinsicsNode = requireNode(cameraNode, "intrinsics", "camera.intrinsics");
        fx = requireNode(intrinsicsNode, "fx", "camera.intrinsics.fx").as<float>();
        fy = requireNode(intrinsicsNode, "fy", "camera.intrinsics.fy").as<float>();
        cx = requireNode(intrinsicsNode, "cx", "camera.intrinsics.cx").as<float>();
        cy = requireNode(intrinsicsNode, "cy", "camera.intrinsics.cy").as<float>();

        // 4-parameter distortion model [k1, k2, p1, p2]
        const YAML::Node distNode = requireNode(cameraNode, "distortion_coeffs", "camera.distortion_coeffs");
        if (!distNode.IsSequence() || distNode.size() != 4) {
            throw std::runtime_error("SystemConfig: 'camera.distortion_coeffs' must be a sequence of 4 floats");
        }
        k1 = distNode[0].as<float>();
        k2 = distNode[1].as<float>();
        p1 = distNode[2].as<float>();
        p2 = distNode[3].as<float>();

        // Sensor extrinsics
        const YAML::Node extrinsicsNode = requireNode(root, "extrinsics", "extrinsics");

        const YAML::Node tCamNode = requireNode(extrinsicsNode, "T_body_camera", "extrinsics.T_body_camera");
        if (!tCamNode.IsSequence() || (tCamNode.size() != 3 && tCamNode.size() != 4)) {
            throw std::runtime_error("SystemConfig: 'extrinsics.T_body_camera' must be a 3x4 or 4x4 matrix");
        }
        for (size_t r = 0; r < 3; r++) {
            const YAML::Node row = tCamNode[r];
            if (!row.IsSequence() || row.size() != 4) {
                throw std::runtime_error("SystemConfig: row " + std::to_string(r) +
                                         " of 'extrinsics.T_body_camera' must have 4 elements");
            }
            for (size_t c = 0; c < 4; c++) {
                bodyToCamera(static_cast<Eigen::Index>(r), static_cast<Eigen::Index>(c)) = row[c].as<double>();
            }
        }

        const YAML::Node tDepthNode = requireNode(extrinsicsNode, "t_body_depth", "extrinsics.t_body_depth");
        if (!tDepthNode.IsSequence() || tDepthNode.size() != 3) {
            throw std::runtime_error("SystemConfig: 'extrinsics.t_body_depth' must be a sequence of 3 doubles");
        }
        for (size_t i = 0; i < 3; i++) {
            bodyToDepth(static_cast<Eigen::Index>(i), 0) = tDepthNode[i].as<double>();
        }

        // Optional initial state pose [R | t]
        const YAML::Node initPoseNode = root["initial_pose"];
        if (initPoseNode && initPoseNode.IsDefined()) {
            if (!initPoseNode.IsSequence() || (initPoseNode.size() != 3 && initPoseNode.size() != 4)) {
                throw std::runtime_error("SystemConfig: 'initial_pose' must be a 3x4 or 4x4 matrix");
            }
            hasInitialPose = true;
            for (size_t r = 0; r < 3; r++) {
                const YAML::Node row = initPoseNode[r];
                if (!row.IsSequence() || row.size() < 4) {
                    throw std::runtime_error("SystemConfig: row " + std::to_string(r) +
                                             " of 'initial_pose' must have at least 4 elements");
                }
                for (size_t c = 0; c < 4; c++) {
                    initialPose(static_cast<Eigen::Index>(r), static_cast<Eigen::Index>(c)) = row[c].as<double>();
                }
            }
        }
    } catch (const YAML::Exception& e) {
        throw std::runtime_error("SystemConfig: YAML error while parsing '" + configFile + "': " + e.what());
    }
}

} // namespace PoolSLAM