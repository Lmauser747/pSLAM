
#include "backend/Backend.hpp"

#include <gtsam/base/Matrix.h>
#include <gtsam/base/Vector.h>
#include <gtsam/base/types.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/NoiseModelFactorN.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

namespace PoolSLAM{

DepthFactor::DepthFactor(
    gtsam::Key poseKey,
    double depthIn,
    const gtsam::SharedNoiseModel &model,
    const gtsam::Point3 &leverArm)
    : gtsam::NoiseModelFactorN<gtsam::Pose3>(model, poseKey),
      measuredDepth(depthIn),
      bodyToSensor(leverArm){}

gtsam::Vector DepthFactor::evaluateError(
    const gtsam::Pose3 &pose,
    gtsam::OptionalMatrixType H) const{
    gtsam::Matrix36 dSensor_dPose;
    const gtsam::Point3 sensorWorld =
        pose.transformFrom(bodyToSensor, H ? &dSensor_dPose : nullptr);

    if(H){
        *H = dSensor_dPose.row(2);
    }

    return (gtsam::Vector(1) << (sensorWorld.z() - measuredDepth)).finished();
}

}