#include <cstddef>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <gtsam/geometry/Cal3_S2.h>
#include <gtsam/geometry/Point2.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/ProjectionFactor.h>
#include <gtsam/linear/LossFunctions.h>


#include "backend/Bookkeeper.hpp"

namespace PoolSLAM {

LandmarkBookkeeper::LandmarkBookkeeper(
    gtsam::NonlinearFactorGraph &stagingGraphRef,
    gtsam::Values &stagingValuesRef,
    const gtsam::Values &isamEstimatesRef,
    const gtsam::Pose3 &bodyToCamExtrinsics,
    const BookkeeperConfig &cfg
)
    : config(cfg),
      stagingGraph(stagingGraphRef),
      stagingValues(stagingValuesRef),
      isamEstimates(isamEstimatesRef),
      bodyPCam(bodyToCamExtrinsics),
      unitCalib(std::make_shared<gtsam::Cal3_S2>(1.0, 1.0, 0.0, 0.0, 0.0)),
      currentCamPose(bodyToCamExtrinsics)
{
    auto baseGaussian = gtsam::noiseModel::Isotropic::Sigma(2, config.visualSigmaNorm);
    visualNoise = gtsam::noiseModel::Robust::Create(
        gtsam::noiseModel::mEstimator::Huber::Create(config.huberK),
        baseGaussian
    );
    activeLandmarks.reserve(256);
}

void LandmarkBookkeeper::startFrame(uint32_t frameId) {
    currentFrameId = frameId;
    const gtsam::Symbol currSym('x', frameId);
    const gtsam::Pose3 bodyPose = stagingValues.exists(currSym)
        ? stagingValues.at<gtsam::Pose3>(currSym)
        : isamEstimates.at<gtsam::Pose3>(currSym);
    currentCamPose = bodyPose.compose(bodyPCam);
}

void LandmarkBookkeeper::updateBookeeper(uint32_t landmarkId, const gtsam::Point2 &pt) {
    const gtsam::Symbol landmarkSym('l', landmarkId);
    auto [it, inserted] = activeLandmarks.try_emplace(landmarkId, landmarkId);
    LandmarkTrack &track = it->second;
    track.lastSeenFrame = currentFrameId;

    // If landmark was already committed to ISAM2 or staged, immediately re-attach as INITIALIZED
    if (inserted && (isamEstimates.exists(landmarkSym) || stagingValues.exists(landmarkSym))) {
        track.state = LandmarkState::INITIALIZED;
    }

    // Skip tracks that previously failed geometric verification
    if (track.state == LandmarkState::INVALID) {
        return;
    }

    // Fast path: landmark is already initialized in GTSAM
    if (track.state == LandmarkState::INITIALIZED) {
        stagingGraph.emplace_shared<ProjectionFactor>(
            pt,
            visualNoise,
            gtsam::Symbol('x', currentFrameId),
            landmarkSym,
            unitCalib,
            bodyPCam
        );
        return;
    }

    // Buffer new 2D ray observation
    track.observations.push_back({currentFrameId, pt});
    if (track.observations.size() < 2) {
        return;
    }

    // Parallax and translation baseline check (comparing first view vs. current view)
    const LandmarkObs &firstObs = track.observations.front();
    const gtsam::Symbol firstSym('x', firstObs.frameId);
    if (firstObs.frameId != currentFrameId && !stagingValues.exists(firstSym) && !isamEstimates.exists(firstSym)) {
        track.observations.erase(track.observations.begin());
        return;
    }
    const gtsam::Pose3 poseFirstCam = getCameraPose(firstObs.frameId);

    const double baseline = (currentCamPose.translation() - poseFirstCam.translation()).norm();
    if (baseline < config.minBaselineMeters) {
        if (track.observations.size() > config.maxBufferLength) {
            track.observations.erase(track.observations.begin() + 1);
        }
        return;
    }

    const Eigen::Vector3d rayFirstUnit = Eigen::Vector3d(firstObs.ray.x(), firstObs.ray.y(), 1.0).normalized();
    const Eigen::Vector3d rayCurrUnit = Eigen::Vector3d(pt.x(), pt.y(), 1.0).normalized();

    const Eigen::Vector3d dirFirstWorld = poseFirstCam.rotation().rotate(rayFirstUnit);
    const Eigen::Vector3d dirCurrWorld = currentCamPose.rotation().rotate(rayCurrUnit);

    const double cosParallax = dirFirstWorld.dot(dirCurrWorld);
    if (cosParallax > config.maxCosParallax) {
        if (track.observations.size() > config.maxBufferLength) {
            track.observations.erase(track.observations.begin() + 1);
        }
        return;
    }

    // Attempt multi-view 3D triangulation
    gtsam::Point3 triangulatedPt;
    if (triangulateLandmark(track.observations, triangulatedPt)) {
        if (!stagingValues.exists(landmarkSym) && !isamEstimates.exists(landmarkSym)) {
            stagingValues.insert(landmarkSym, triangulatedPt);
        }

        for (size_t i = 0; i < track.observations.size(); i++) {
            const LandmarkObs &obs = track.observations[i];
            stagingGraph.emplace_shared<ProjectionFactor>(
                obs.ray,
                visualNoise,
                gtsam::Symbol('x', obs.frameId),
                landmarkSym,
                unitCalib,
                bodyPCam
            );
        }

        track.state = LandmarkState::INITIALIZED;
        track.observations.clear();
    } else if (track.observations.size() >= config.maxBufferLength) {
        track.state = LandmarkState::INVALID;
        track.observations.clear();
    }
}

void LandmarkBookkeeper::pruneDeadTracks() {
    for (auto it = activeLandmarks.begin(); it != activeLandmarks.end(); ) {
        if (it->second.lastSeenFrame < currentFrameId) {
            it = activeLandmarks.erase(it);
        } else {
            it++;
        }
    }
}

gtsam::Pose3 LandmarkBookkeeper::getCameraPose(uint32_t frameId) const {
    if (frameId == currentFrameId) {
        return currentCamPose;
    }
    const gtsam::Symbol poseSym('x', frameId);
    if (stagingValues.exists(poseSym)) {
        return stagingValues.at<gtsam::Pose3>(poseSym).compose(bodyPCam);
    }
    if (isamEstimates.exists(poseSym)) {
        return isamEstimates.at<gtsam::Pose3>(poseSym).compose(bodyPCam);
    }
    return currentCamPose;
}

bool LandmarkBookkeeper::triangulateLandmark(
    const std::vector<LandmarkObs> &obs,
    gtsam::Point3 &outPt
) const {
    const size_t numObs = obs.size();
    Eigen::Matrix3d A = Eigen::Matrix3d::Zero();
    Eigen::Vector3d b = Eigen::Vector3d::Zero();

    // Cache camera poses once to avoid querying gtsam::Values twice per observation
    std::vector<gtsam::Pose3> camPoses;
    camPoses.reserve(numObs);

    for (size_t i = 0; i < numObs; i++) {
        const gtsam::Symbol sym('x', obs[i].frameId);
        if (obs[i].frameId != currentFrameId && !stagingValues.exists(sym) && !isamEstimates.exists(sym)) {
            return false;
        }
        const gtsam::Pose3 camPose = getCameraPose(obs[i].frameId);
        camPoses.push_back(camPose);

        const Eigen::Vector3d rayCamUnit = Eigen::Vector3d(obs[i].ray.x(), obs[i].ray.y(), 1.0).normalized();
        const Eigen::Vector3d d = camPose.rotation().rotate(rayCamUnit);

        const Eigen::Matrix3d proj = Eigen::Matrix3d::Identity() - (d * d.transpose());
        A += proj;
        b += proj * camPose.translation();
    }

    // Normalize by observation count so conditioning threshold is independent of buffer length
    const double invN = 1.0 / static_cast<double>(numObs);
    A *= invN;
    b *= invN;

    // Gate 1: LDLT conditioning check
    const Eigen::LDLT<Eigen::Matrix3d> ldlt(A);
    if (ldlt.info() != Eigen::Success || ldlt.vectorD().minCoeff() < config.minLdltDiag) {
        return false;
    }

    const Eigen::Vector3d pWorld = ldlt.solve(b);

    // Cheirality, max pool depth, and normalized reprojection error
    for (size_t i = 0; i < numObs; i++) {
        const Eigen::Vector3d pCam = camPoses[i].transformTo(pWorld);

        if (pCam.z() < config.minDepthMeters || pCam.z() > config.maxDepthMeters) {
            return false;
        }

        const double invZ = 1.0 / pCam.z();
        const double du = (pCam.x() * invZ) - obs[i].ray.x();
        const double dv = (pCam.y() * invZ) - obs[i].ray.y();

        if ((du * du + dv * dv) > config.maxReprojErrorSq) {
            return false;
        }
    }

    outPt = pWorld;
    return true;
}

}