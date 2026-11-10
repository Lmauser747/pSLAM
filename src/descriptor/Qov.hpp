#pragma once

#include <cstdint>
#include <cstddef>
#include <array>
#include <cmath>

#include <eigen3/Eigen/Core>

#include "descriptor/QovBaseOffsets.hpp"

#ifndef FORCE_INLINE
#if defined(__GNUC__) || defined(__clang__)
  #define FORCE_INLINE inline __attribute__((always_inline))
#else
  #define FORCE_INLINE inline
#endif
#endif

// QOV (Quick ORB Variant) descriptor implementations
namespace PoolSLAM{
namespace descriptor{

constexpr size_t QOV_BITS = 256;

// 32-byte 16-byte aligned binary descriptor (POD / Trivially Copyable)
struct alignas(16) QovDescriptor{
    std::array<uint64_t, 4> bits = {0, 0, 0, 0};

    bool operator==(const QovDescriptor &other) const{
        return bits == other.bits;
    }

    bool operator!=(const QovDescriptor &other) const{
        return bits != other.bits;
    }
};

struct QovFeature{
    QovDescriptor des;
    Eigen::Vector2i pt = Eigen::Vector2i::Zero();
    uint32_t wordId = 0;
    uint32_t featureId = 0;

    QovFeature() = default;
    QovFeature(const Eigen::Vector2i &kp, uint32_t id) : pt(kp), featureId(id){}
};

struct QovCompPair{
    ptrdiff_t pt1 = 0;
    ptrdiff_t pt2 = 0;
};

// Computes Hamming distance between two 256-bit descriptors
FORCE_INLINE int fastHamming(const QovDescriptor &w1, const QovDescriptor &w2){
    return __builtin_popcountll(w1.bits[0] ^ w2.bits[0]) +
           __builtin_popcountll(w1.bits[1] ^ w2.bits[1]) +
           __builtin_popcountll(w1.bits[2] ^ w2.bits[2]) +
           __builtin_popcountll(w1.bits[3] ^ w2.bits[3]);
}


class RotatedQovPairs{
private:
    static constexpr double PI = 3.14159265358979323846;
    static constexpr size_t ROTATIONS = 32; // Number of rotated offsets to store
    static constexpr double ANGLE_STEP_DEG = 360.0 / ROTATIONS; // Step between each offset
    static constexpr double ANGLE_STEP_RAD = 2.0 * PI / ROTATIONS;

    std::array<std::array<QovCompPair, QOV_BITS>, ROTATIONS> offsets;

    static double toRadians(double degrees){
        return (degrees / 180.0) * PI;
    }

    void computeRotatedOffsets(size_t idx, double angleDeg, ptrdiff_t stride){
        double rad = toRadians(angleDeg);
        double cosA = std::cos(rad);
        double sinA = std::sin(rad);

        for(size_t i = 0; i < QOV_BITS; i++){
            double x1 = static_cast<double>(QOV_BASE_OFFSETS[i * 4 + 0]);
            double y1 = static_cast<double>(QOV_BASE_OFFSETS[i * 4 + 1]);
            double x2 = static_cast<double>(QOV_BASE_OFFSETS[i * 4 + 2]);
            double y2 = static_cast<double>(QOV_BASE_OFFSETS[i * 4 + 3]);

            double rx1 = x1 * cosA - y1 * sinA;
            double ry1 = x1 * sinA + y1 * cosA;
            double rx2 = x2 * cosA - y2 * sinA;
            double ry2 = x2 * sinA + y2 * cosA;

            ptrdiff_t ix1 = static_cast<ptrdiff_t>(std::lround(rx1));
            ptrdiff_t iy1 = static_cast<ptrdiff_t>(std::lround(ry1));
            ptrdiff_t ix2 = static_cast<ptrdiff_t>(std::lround(rx2));
            ptrdiff_t iy2 = static_cast<ptrdiff_t>(std::lround(ry2));

            offsets[idx][i].pt1 = ix1 + stride * iy1;
            offsets[idx][i].pt2 = ix2 + stride * iy2;
        }
    }

public:
    explicit RotatedQovPairs(size_t w){
        for(size_t i = 0; i < ROTATIONS; i++){
            computeRotatedOffsets(i, static_cast<double>(i) * ANGLE_STEP_DEG, static_cast<ptrdiff_t>(w));
        }
    }

    // Get the offset array at a given angle in degrees (zero-copy reference)
    FORCE_INLINE const std::array<QovCompPair, QOV_BITS>& offsetAtAngleDeg(double angle) const{
        constexpr double INV = 1.0 / ANGLE_STEP_DEG;
        double normAngle = std::fmod(angle, 360.0);
        if(normAngle < 0.0){
            normAngle += 360.0;
        }
        size_t bin = static_cast<size_t>(std::lround(normAngle * INV)) % ROTATIONS;
        return offsets[bin];
    }

    // Get the offset array at a given angle in radians (zero-copy reference)
    FORCE_INLINE const std::array<QovCompPair, QOV_BITS>& offsetAtAngleRad(double angle) const{
        constexpr double TWO_PI = 2.0 * PI;
        constexpr double INV = 1.0 / ANGLE_STEP_RAD;
        double normAngle = std::fmod(angle, TWO_PI);
        if(normAngle < 0.0){
            normAngle += TWO_PI;
        }
        size_t bin = static_cast<size_t>(std::lround(normAngle * INV)) % ROTATIONS;
        return offsets[bin];
    }
};

FORCE_INLINE QovDescriptor generateDescriptor(const uint8_t *pt, const std::array<QovCompPair, QOV_BITS> &offsets){
    QovDescriptor des;
    // BEGIN VECTORIZE
    for(size_t w = 0; w < 4; w++){
        uint64_t word = 0;
        const size_t baseIdx = w * 64;
        for(size_t b = 0; b < 64; b++){
            const auto &pair = offsets[baseIdx + b];
            if(*(pt + pair.pt1) < *(pt + pair.pt2)){
                word |= (static_cast<uint64_t>(1) << b);
            }
        }
        des.bits[w] = word;
    }
    // END VECTORIZE
    return des;
}

} 
}