#pragma once
#include <eigen3/Eigen/Core>

namespace PoolSLAM{

struct Pose{
    // timestamp is a time measured in seconds relative to some standard starting point
    double timestamp;

    // t = (x, y, z) is the posiiton component of the ROV pose measured in meters
    Eigen::Vector3d t;

    // r = (r, p, y) is the rotation component of the ROV pose measured in radians
    Eigen::Vector3d r;
};

}