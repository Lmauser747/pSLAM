#include <cstdint>
#include <vector>

struct IMU{
    // (ax, ay, az) is acceleration in the x, y, z directions respectively measured in m^2/s
    float ax, ay, az;

    // (wx, wy, wz) is angular velocity around the x, y, and z axis repectively measureing in deg/s
    // wx = roll, wy = pitch, wz = yaw
    float wx, wy, wz;

    // t is the temperature in deg C measured by the IMU on-die temp sensor. This field is optional
    float t;
};

struct Depth{
    // z is the distance below the surface of the water measured by the depth sensor
    float z;
};

class PixelArray{
private:
    uint16_t x, y; // Current pixel coordinates
public:
    std::vector<uint8_t> arr; // Pixel array
    uint16_t w, h; // Width and height of the image
    PixelArray(uint16_t width, uint16_t height) : w(width), h(height){
        arr.resize(w * h);
    }

    uint8_t getPixel(uint16_t x, uint16_t y) const; // gets pixel at (x, y) in camera coordinates
    uint8_t getNextPixel(); // 
    void reset();
};

