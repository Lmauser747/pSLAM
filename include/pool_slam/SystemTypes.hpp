#pragma once

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>

namespace PoolSLAM {

struct Pose{
    // timestamp is a time measured in seconds relative to some standard starting point
    double timestamp;

    // t = (x, y, z) is the posiiton component of the ROV pose measured in meters
    Eigen::Vector3d t;

    // r = (w, x, y, z) is the quaternion of the rotation component of the ROV pose
    Eigen::Quaterniond r;

    // v = (v_x, v_y, v_z) is the velocity of the ROV in m/s
    Eigen::Vector3d v;
};


enum class SyncStatus{
    IDLE,
    RUNNING,
    STOPPED,
    FATALCAM,
    FATALIMU
};

}
