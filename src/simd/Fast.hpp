#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>
#include <array>

#include <eigen3/Eigen/Core>
#include <simde/arm/neon/ld1.h>
#include <simde/arm/neon/st1.h>
#include <simde/arm/neon/dup_n.h>
#include <simde/arm/neon/qadd.h>
#include <simde/arm/neon/qsub.h>
#include <simde/arm/neon/types.h>
#include <simde/arm/neon/cgt.h>
#include <simde/arm/neon/clt.h>
#include <simde/arm/neon/orr.h>
#include <simde/arm/neon/and.h>
#include <simde/arm/neon/maxv.h>
#include <simde/arm/neon/reinterpret.h>

#include "simd/Utils.hpp"

namespace PoolSLAM{

namespace simd{

// Find the FAST keypoints in 16 consecutive pixels
// Returns number of added entries
FORCE_INLINE size_t fastLine16(const uint8_t *start, const Eigen::Vector2i &startPt, const std::array<ptrdiff_t, 16> &ring, uint8_t threshold, std::vector<Eigen::Vector2i> &kps);

FORCE_INLINE static simde_uint8x16_t isConsecutive(const std::array<simde_uint8x16_t, 16>& m);

FORCE_INLINE size_t fastLine16(const uint8_t *start, const Eigen::Vector2i &startPt, const std::array<ptrdiff_t, 16> &ringOffsets, uint8_t threshold, std::vector<Eigen::Vector2i> &kps){
    auto center = simde_vld1q_u8(start);
    auto t = simde_vdupq_n_u8(threshold);
    auto upper = simde_vqaddq_u8(t, center);
    auto lower = simde_vqsubq_u8(center, t);

    const simde_uint8x16_t zero = simde_vdupq_n_u8(0);

    std::array<simde_uint8x16_t, 16> ring; // Vector of ring points
    std::array<simde_uint8x16_t, 16> brighter; // Brigher mask
    std::array<simde_uint8x16_t, 16> darker; // Darker mask

    // Do compass checks at 0, 4, 8, and 12 pixel positions
    static_for(std::index_sequence<0, 4, 8, 12>{}, [&](auto kConst){
        constexpr size_t k = decltype(kConst)::value;
        ring[k] = simde_vld1q_u8(start + ringOffsets[k]);
    });

    static_for(std::index_sequence<0, 4, 8, 12>{}, [&](auto kConst){
        constexpr size_t k = decltype(kConst)::value;
        brighter[k] = simde_vcgtq_u8(ring[k], upper);
        darker[k] = simde_vcltq_u8(ring[k], lower);
    });

    // Compass must have 1s in (0 or 8) and (4 or 12)
    auto brighterCompass = simde_vandq_u8(simde_vorrq_u8(brighter[0], brighter[8]), simde_vorrq_u8(brighter[4], brighter[12]));
    auto darkerCompass = simde_vandq_u8(simde_vorrq_u8(darker[0], darker[8]), simde_vorrq_u8(darker[4], darker[12]));

    // Find if any entries are 1
    const uint8_t maxBrighter = simde_vmaxvq_u8(brighterCompass);
    const uint8_t maxDarker = simde_vmaxvq_u8(darkerCompass);

    if(!maxBrighter && !maxDarker){
         return 0;
    }

    // If compass passes generate main mask
    // Do 6 at a time for rw port and fpu staturation
    static_for(std::index_sequence<1, 2, 3, 5, 6, 7>{}, [&](auto kConst){
        constexpr size_t k = decltype(kConst)::value;
        ring[k] = simde_vld1q_u8(start + ringOffsets[k]);
    });

    if(maxBrighter){
        static_for(std::index_sequence<1, 2, 3, 5, 6, 7>{}, [&](auto kConst){
            constexpr size_t k = decltype(kConst)::value;
            brighter[k] = simde_vcgtq_u8(ring[k], upper);
        });
    }

    if(maxDarker){
        static_for(std::index_sequence<1, 2, 3, 5, 6, 7>{}, [&](auto kConst){
            constexpr size_t k = decltype(kConst)::value;
            darker[k] = simde_vcltq_u8(ring[k], lower);
        });
    }
        
    static_for(std::index_sequence<9, 10, 11, 13, 14, 15>{}, [&](auto kConst){
        constexpr size_t k = decltype(kConst)::value;
        ring[k] = simde_vld1q_u8(start + ringOffsets[k]);
    });

    auto brightFeatures = zero;
    auto darkFeatures = zero;
    if(maxBrighter){
        static_for(std::index_sequence<9, 10, 11, 13, 14, 15>{}, [&](auto kConst){
            constexpr size_t k = decltype(kConst)::value;
            brighter[k] = simde_vcgtq_u8(ring[k], upper);
        });
        brightFeatures = isConsecutive(brighter);
    }

    if(maxDarker){
        static_for(std::index_sequence<9, 10, 11, 13, 14, 15>{}, [&](auto kConst){
            constexpr size_t k = decltype(kConst)::value;
            darker[k] = simde_vcltq_u8(ring[k], lower);
        });
        darkFeatures = isConsecutive(darker);
    }

    auto features = simde_vorrq_u8(darkFeatures, brightFeatures);
    
    alignas(16) std::array<uint8_t, 16> mask;
    simde_vst1q_u8(mask.data(), features);

    size_t featureCount = 0;
    for (int i = 0; i < 16; ++i) {
        if (mask[static_cast<size_t>(i)]) {
            kps.emplace_back(startPt.x() + i, startPt.y());
            featureCount++;
        }
    }

    return featureCount;
}

FORCE_INLINE static simde_uint8x16_t isConsecutive(const std::array<simde_uint8x16_t, 16>& m){
    std::array<simde_uint8x16_t, 8> e;
    std::array<simde_uint8x16_t, 8> o;
    std::array<simde_uint8x16_t, 8> f;
    std::array<simde_uint8x16_t, 8> q;
    std::array<simde_uint8x16_t, 8> r;

    static_for<8>([&](auto kConst){
        constexpr size_t k = decltype(kConst)::value;
        e[k] = simde_vandq_u8(m[2 * k], m[2 * k + 1]);
        o[k] = simde_vorrq_u8(m[(2 * k + 15) % 16], m[(2 * k + 8) % 16]);
    });

    static_for<8>([&](auto kConst){
        constexpr size_t k = decltype(kConst)::value;
        f[k] = simde_vandq_u8(e[k], e[(k + 1) % 8]);
    });

    static_for<8>([&](auto kConst) {
        constexpr size_t k = decltype(kConst)::value;
        q[k] = simde_vandq_u8(f[k], f[(k + 2) % 8]);
    });

    static_for<8>([&](auto kConst) {
        constexpr size_t k = decltype(kConst)::value;
        r[k] = simde_vandq_u8(q[k], o[k]);
    });

    std::array<simde_uint8x16_t, 4> red1;

    static_for<4>([&](auto kConst) {
        constexpr size_t k = decltype(kConst)::value;
        red1[k] = simde_vorrq_u8(r[2 * k], r[2 * k + 1]);
    });

    std::array<simde_uint8x16_t, 2> red2;
    static_for<2>([&](auto kConst) {
        constexpr size_t k = decltype(kConst)::value;
        red2[k] = simde_vorrq_u8(red1[2 * k], red1[2 * k + 1]);
    });

    return simde_vorrq_u8(red2[0], red2[1]);
}

}

}