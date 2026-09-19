#pragma once
#include <cstdint>
#include <vector>

#include <eigen3/Eigen/Core>

namespace PoolSLAM{

struct IMU{
    // timestamp is a time measured in seconds relative to some standard starting point
    double timestamp;

    // a = (ax, ay, az) is acceleration in the x, y, z directions respectively measured in m^2/s
    Eigen::Vector3d a;

    // w = (wx, wy, wz) is angular velocity around the x, y, and z axis repectively measureing in deg/s
    // wx = roll, wy = pitch, wz = yaw
    Eigen::Vector3d w;

    // t is the temperature in deg C measured by the IMU on-die temp sensor. This field is optional
    float t;
};

struct Depth{
    // timestamp is a time measured in seconds relative to some standard starting point
    double timestamp;

    // z is the distance below the surface of the water measured by the depth sensor
    double z;
};

class PixelArray{
private:
    uint32_t x, y; // Current pixel coordinates
public:
    double timestamp; // timestamp is a time measured in seconds relative to some standard starting point
    std::vector<uint8_t> arr; // Pixel array
    uint32_t w, h; // Width and height of the image
    PixelArray(uint32_t width, uint32_t height) : w(width), h(height){
        arr.resize(w * h);
    }
};

enum SensorHealth{
    ACTIVE, DEGRADED, DISABLED
};

}