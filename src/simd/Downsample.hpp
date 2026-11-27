#pragma once

#include <cstdint>
#include <cstddef>
#include <array>
#include <utility>

#include <simde/arm/neon.h>
#include <simde/arm/neon/add.h>
#include <simde/arm/neon/addl.h>
#include <simde/arm/neon/addl_high.h>
#include <simde/arm/neon/combine.h>
#include <simde/arm/neon/get_low.h>
#include <simde/arm/neon/ld1q_x2.h>
#include <simde/arm/neon/ld2.h>
#include <simde/arm/neon/movl.h>
#include <simde/arm/neon/movl_high.h>
#include <simde/arm/neon/mul_n.h>
#include <simde/arm/neon/qmovn.h>
#include <simde/arm/neon/rshr_n.h>
#include <simde/arm/neon/shl_n.h>
#include <simde/arm/neon/st1.h>
#include <simde/arm/neon/types.h>

#include "frontend/HfTracker.hpp"
#include "simd/Utils.hpp"

namespace PoolSLAM{

namespace simd{

// Blur 32 bytes centered at src into 16 horizontal sums in dst
FORCE_INLINE void downsampleRow16(const uint8_t *src, uint16_t *dst);

// Blur 16 columns across five rows centered at src into dst
FORCE_INLINE void downsampleCol16(const uint16_t *src, size_t stride, uint8_t *dst);

FORCE_INLINE void downsampleRow16(const uint8_t *src, uint16_t *dst){
    const auto pair0 = simde_vld2q_u8(src - 2);
    const auto pair1 = simde_vld2q_u8(src);
    const auto pair2 = simde_vld2q_u8(src + 2);

    const simde_uint8x16_t p0 = pair0.val[0];
    const simde_uint8x16_t p1 = pair0.val[1];
    const simde_uint8x16_t p2 = pair1.val[0];
    const simde_uint8x16_t p3 = pair1.val[1];
    const simde_uint8x16_t p4 = pair2.val[0];

    const simde_uint16x8_t p04Lo = simde_vaddl_u8(simde_vget_low_u8(p0), simde_vget_low_u8(p4));
    const simde_uint16x8_t p04Hi = simde_vaddl_high_u8(p0, p4);
    const simde_uint16x8_t p13Lo = simde_vaddl_u8(simde_vget_low_u8(p1), simde_vget_low_u8(p3));
    const simde_uint16x8_t p13Hi = simde_vaddl_high_u8(p1, p3);
    const simde_uint16x8_t p2Lo = simde_vmovl_u8(simde_vget_low_u8(p2));
    const simde_uint16x8_t p2Hi = simde_vmovl_high_u8(p2);

    simde_uint16x8_t accLo = simde_vaddq_u16(p04Lo, simde_vshlq_n_u16(p13Lo, 2));
    simde_uint16x8_t accHi = simde_vaddq_u16(p04Hi, simde_vshlq_n_u16(p13Hi, 2));
    accLo = simde_vaddq_u16(accLo, simde_vmulq_n_u16(p2Lo, 6));
    accHi = simde_vaddq_u16(accHi, simde_vmulq_n_u16(p2Hi, 6));

    simde_vst1q_u16(dst, accLo);
    simde_vst1q_u16(dst + 8, accHi);
}

FORCE_INLINE void downsampleCol16(const uint16_t *src, size_t stride, uint8_t *dst){
    const ptrdiff_t s = static_cast<ptrdiff_t>(stride);
    const auto r0 = simde_vld1q_u16_x2(src - 2 * s);
    const auto r1 = simde_vld1q_u16_x2(src - s);
    const auto r2 = simde_vld1q_u16_x2(src);
    const auto r3 = simde_vld1q_u16_x2(src + s);
    const auto r4 = simde_vld1q_u16_x2(src + 2 * s);

    simde_uint16x8_t accLo = simde_vaddq_u16(r0.val[0], r4.val[0]);
    simde_uint16x8_t accHi = simde_vaddq_u16(r0.val[1], r4.val[1]);
    accLo = simde_vaddq_u16(accLo, simde_vshlq_n_u16(simde_vaddq_u16(r1.val[0], r3.val[0]), 2));
    accHi = simde_vaddq_u16(accHi, simde_vshlq_n_u16(simde_vaddq_u16(r1.val[1], r3.val[1]), 2));
    accLo = simde_vaddq_u16(accLo, simde_vmulq_n_u16(r2.val[0], 6));
    accHi = simde_vaddq_u16(accHi, simde_vmulq_n_u16(r2.val[1], 6));

    accLo = simde_vrshrq_n_u16(accLo, 8);
    accHi = simde_vrshrq_n_u16(accHi, 8);

    const simde_uint8x16_t out = simde_vcombine_u8(simde_vqmovn_u16(accLo), simde_vqmovn_u16(accHi));
    simde_vst1q_u8(dst, out);
}

}

}