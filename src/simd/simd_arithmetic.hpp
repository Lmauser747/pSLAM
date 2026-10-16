#pragma once

#include <arm_neon.h>
#include <cstdint>
#include <limits>

#include "simd/simd_wrappers.hpp"

namespace psSimd{

// dest = a + b
template <typename T>
FORCE_INLINE typename VectorType<T>::V add(typename VectorType<T>::V a, typename VectorType<T>::V b) {
#if defined(NEON_WRAPPERS)
    if constexpr (std::is_same_v<uint8_t, T>) {
        return vaddq_u8(a, b);
    } else if constexpr (std::is_same_v<int8_t, T>) {
        return vaddq_s8(a, b);
    } else if constexpr (std::is_same_v<uint16_t, T>) {
        return vaddq_u16(a, b);
    } else if constexpr (std::is_same_v<int16_t, T>) {
        return vaddq_s16(a, b);
    } else if constexpr (std::is_same_v<uint32_t, T>) {
        return vaddq_u32(a, b);
    } else if constexpr (std::is_same_v<int32_t, T>) {
        return vaddq_s32(a, b);
    } else if constexpr (std::is_same_v<uint64_t, T>) {
        return vaddq_u64(a, b);
    } else if constexpr (std::is_same_v<int64_t, T>) {
        return vaddq_s64(a, b);
    } else if constexpr (std::is_same_v<float, T>) {
        return vaddq_f32(a, b);
    } else if constexpr (std::is_same_v<double, T>) {
        return vaddq_f64(a, b);
    } else {
        static_assert(dependent_false_v<T>, "ERROR - psSimd::add(NEON): Unsupported data type");
    }
#elif defined(AVX2_WRAPPERS)
    if constexpr (std::is_same_v<uint8_t, T> || std::is_same_v<int8_t, T>) {
        return _mm_add_epi8(a, b);
    } else if constexpr (std::is_same_v<uint16_t, T> || std::is_same_v<int16_t, T>) {
        return _mm_add_epi16(a, b);
    } else if constexpr (std::is_same_v<uint32_t, T> || std::is_same_v<int32_t, T>) {
        return _mm_add_epi32(a, b);
    } else if constexpr (std::is_same_v<uint64_t, T> || std::is_same_v<int64_t, T>) {
        return _mm_add_epi64(a, b);
    } else if constexpr (std::is_same_v<float, T>) {
        return _mm_add_ps(a, b);
    } else if constexpr (std::is_same_v<double, T>) {
        return _mm_add_pd(a, b);
    } else {
        static_assert(dependent_false_v<T>, "ERROR - psSimd::add(AVX2): Unsupported data type");
    }
#else
    return a + b;
#endif
}

// dest = a - b
template <typename T>
FORCE_INLINE typename VectorType<T>::V sub(typename VectorType<T>::V a, typename VectorType<T>::V b) {
#if defined(NEON_WRAPPERS)
    if constexpr (std::is_same_v<uint8_t, T>) {
        return vsubq_u8(a, b);
    } else if constexpr (std::is_same_v<int8_t, T>) {
        return vsubq_s8(a, b);
    } else if constexpr (std::is_same_v<uint16_t, T>) {
        return vsubq_u16(a, b);
    } else if constexpr (std::is_same_v<int16_t, T>) {
        return vsubq_s16(a, b);
    } else if constexpr (std::is_same_v<uint32_t, T>) {
        return vsubq_u32(a, b);
    } else if constexpr (std::is_same_v<int32_t, T>) {
        return vsubq_s32(a, b);
    } else if constexpr (std::is_same_v<uint64_t, T>) {
        return vsubq_u64(a, b);
    } else if constexpr (std::is_same_v<int64_t, T>) {
        return vsubq_s64(a, b);
    } else if constexpr (std::is_same_v<float, T>) {
        return vsubq_f32(a, b);
    } else if constexpr (std::is_same_v<double, T>) {
        return vsubq_f64(a, b);
    } else {
        static_assert(dependent_false_v<T>, "ERROR - psSimd::sub(NEON): Unsupported data type");
    }
#elif defined(AVX2_WRAPPERS)
    if constexpr (std::is_same_v<uint8_t, T> || std::is_same_v<int8_t, T>) {
        return _mm_sub_epi8(a, b);
    } else if constexpr (std::is_same_v<uint16_t, T> || std::is_same_v<int16_t, T>) {
        return _mm_sub_epi16(a, b);
    } else if constexpr (std::is_same_v<uint32_t, T> || std::is_same_v<int32_t, T>) {
        return _mm_sub_epi32(a, b);
    } else if constexpr (std::is_same_v<uint64_t, T> || std::is_same_v<int64_t, T>) {
        return _mm_sub_epi64(a, b);
    } else if constexpr (std::is_same_v<float, T>) {
        return _mm_sub_ps(a, b);
    } else if constexpr (std::is_same_v<double, T>) {
        return _mm_sub_pd(a, b);
    } else {
        static_assert(dependent_false_v<T>, "ERROR - psSimd::sub(AVX2): Unsupported data type");
    }
#else
    return a - b;
#endif
}

// dest = a * b
template <typename T>
FORCE_INLINE typename VectorType<T>::V mul(typename VectorType<T>::V a, typename VectorType<T>::V b) {
#if defined(NEON_WRAPPERS)
    if constexpr (std::is_same_v<uint8_t, T>) {
        return vmulq_u8(a, b);
    } else if constexpr (std::is_same_v<int8_t, T>) {
        return vmulq_s8(a, b);
    } else if constexpr (std::is_same_v<uint16_t, T>) {
        return vmulq_u16(a, b);
    } else if constexpr (std::is_same_v<int16_t, T>) {
        return vmulq_s16(a, b);
    } else if constexpr (std::is_same_v<uint32_t, T>) {
        return vmulq_u32(a, b);
    } else if constexpr (std::is_same_v<int32_t, T>) {
        return vmulq_s32(a, b);
    } else if constexpr (std::is_same_v<float, T>) {
        return vmulq_f32(a, b);
    } else if constexpr (std::is_same_v<double, T>) {
        return vmulq_f64(a, b);
    } else {
        static_assert(dependent_false_v<T>, "ERROR - psSimd::mul(NEON): No native hardware SIMD multiply for this type");
    }
#elif defined(AVX2_WRAPPERS)
    if constexpr (std::is_same_v<uint16_t, T> || std::is_same_v<int16_t, T>) {
        return _mm_mullo_epi16(a, b);
    } else if constexpr (std::is_same_v<uint32_t, T> || std::is_same_v<int32_t, T>) {
        return _mm_mullo_epi32(a, b);
    } else if constexpr (std::is_same_v<float, T>) {
        return _mm_mul_ps(a, b);
    } else if constexpr (std::is_same_v<double, T>) {
        return _mm_mul_pd(a, b);
    } else {
        static_assert(dependent_false_v<T>, "ERROR - psSimd::mul(AVX2): No native hardware SIMD multiply for this type");
    }
#else
    return a * b;
#endif
}


// dest =  a / b
template <typename T>
FORCE_INLINE typename VectorType<T>::V div(typename VectorType<T>::V a, typename VectorType<T>::V b) {
    static_assert(std::is_floating_point_v<T>, "ERROR - psSimd::div: Integer vector division is not supported by SIMD hardware");
#if defined(NEON_WRAPPERS)
    if constexpr (std::is_same_v<float, T>) {
        return vdivq_f32(a, b);
    } else if constexpr (std::is_same_v<double, T>) {
        return vdivq_f64(a, b);
    } else {
        static_assert(dependent_false_v<T>, "ERROR - psSimd::div(NEON): Unsupported data type");
    }
#elif defined(AVX2_WRAPPERS)
    if constexpr (std::is_same_v<float, T>) {
        return _mm_div_ps(a, b);
    } else if constexpr (std::is_same_v<double, T>) {
        return _mm_div_pd(a, b);
    } else {
        static_assert(dependent_false_v<T>, "ERROR - psSimd::div(AVX2): Unsupported data type");
    }
#else
    return a / b;
#endif
}

// dest = -a
template <typename T>
FORCE_INLINE typename VectorType<T>::V neg(typename VectorType<T>::V a) {
    static_assert(std::is_signed_v<T>, "ERROR - psSimd::neg: Unsigned types cannot be negated");
#if defined(NEON_WRAPPERS)
    if constexpr (std::is_same_v<int8_t, T>) {
        return vnegq_s8(a);
    } else if constexpr (std::is_same_v<int16_t, T>) {
        return vnegq_s16(a);
    } else if constexpr (std::is_same_v<int32_t, T>) {
        return vnegq_s32(a);
    } else if constexpr (std::is_same_v<int64_t, T>) {
        return vnegq_s64(a);
    } else if constexpr (std::is_same_v<float, T>) {
        return vnegq_f32(a);
    } else if constexpr (std::is_same_v<double, T>) {
        return vnegq_f64(a);
    } else {
        static_assert(dependent_false_v<T>, "ERROR - psSimd::neg(NEON): Unsupported data type");
    }
#elif defined(AVX2_WRAPPERS)
    if constexpr (std::is_same_v<int8_t, T>) {
        return _mm_sub_epi8(_mm_setzero_si128(), a);
    } else if constexpr (std::is_same_v<int16_t, T>) {
        return _mm_sub_epi16(_mm_setzero_si128(), a);
    } else if constexpr (std::is_same_v<int32_t, T>) {
        return _mm_sub_epi32(_mm_setzero_si128(), a);
    } else if constexpr (std::is_same_v<int64_t, T>) {
        return _mm_sub_epi64(_mm_setzero_si128(), a);
    } else if constexpr (std::is_same_v<float, T>) {
        return _mm_sub_ps(_mm_setzero_ps(), a);
    } else if constexpr (std::is_same_v<double, T>) {
        return _mm_sub_pd(_mm_setzero_pd(), a);
    } else {
        static_assert(dependent_false_v<T>, "ERROR - psSimd::neg(AVX2): Unsupported data type");
    }
#else
    return -a;
#endif
}

template <typename T>
FORCE_INLINE typename VectorType<T>::V abs(typename VectorType<T>::V a) {
    static_assert(std::is_signed_v<T>, "ERROR - psSimd::abs: Absolute value of unsigned types is identity");
#if defined(NEON_WRAPPERS)
    if constexpr (std::is_same_v<int8_t, T>) {
        return vabsq_s8(a);
    } else if constexpr (std::is_same_v<int16_t, T>) {
        return vabsq_s16(a);
    } else if constexpr (std::is_same_v<int32_t, T>) {
        return vabsq_s32(a);
    } else if constexpr (std::is_same_v<int64_t, T>) {
        return vabsq_s64(a);
    } else if constexpr (std::is_same_v<float, T>) {
        return vabsq_f32(a);
    } else if constexpr (std::is_same_v<double, T>) {
        return vabsq_f64(a);
    } else {
        static_assert(dependent_false_v<T>, "ERROR - psSimd::abs(NEON): Unsupported data type");
    }
#elif defined(AVX2_WRAPPERS)
    if constexpr (std::is_same_v<int8_t, T>) {
        return _mm_abs_epi8(a);
    } else if constexpr (std::is_same_v<int16_t, T>) {
        return _mm_abs_epi16(a);
    } else if constexpr (std::is_same_v<int32_t, T>) {
        return _mm_abs_epi32(a);
    } else if constexpr (std::is_same_v<int64_t, T>) {
        __m128i sign = _mm_cmpgt_epi64(_mm_setzero_si128(), a);
        return _mm_sub_epi64(_mm_xor_si128(a, sign), sign);
    } else if constexpr (std::is_same_v<float, T>) {
        return _mm_andnot_ps(_mm_set1_ps(-0.0f), a);
    } else if constexpr (std::is_same_v<double, T>) {
        return _mm_andnot_pd(_mm_set1_pd(-0.0), a);
    } else {
        static_assert(dependent_false_v<T>, "ERROR - psSimd::abs(AVX2): Unsupported data type");
    }
#else
    return (a < 0) ? -a : a;
#endif
}

template <typename T>
FORCE_INLINE T scalarAddSat(T a, T b) {
    if constexpr (std::is_unsigned_v<T>) {
        T res = a + b;
        return (res < a) ? std::numeric_limits<T>::max() : res;
    } else {
        if (b > 0 && a > std::numeric_limits<T>::max() - b) {
            return std::numeric_limits<T>::max();
        }
        if (b < 0 && a < std::numeric_limits<T>::min() - b) {
            return std::numeric_limits<T>::min();
        }
        return a + b;
    }
}

template <typename T>
FORCE_INLINE T scalarSubSat(T a, T b) {
    if constexpr (std::is_unsigned_v<T>) {
        return (a < b) ? 0 : a - b;
    } else {
        if (b < 0 && a > std::numeric_limits<T>::max() + b) {
            return std::numeric_limits<T>::max();
        }
        if (b > 0 && a < std::numeric_limits<T>::min() + b) {
            return std::numeric_limits<T>::min();
        }
        return a - b;
    }
}

template <typename T>
FORCE_INLINE typename VectorType<T>::V addSat(typename VectorType<T>::V a, typename VectorType<T>::V b) {
#if defined(NEON_WRAPPERS)
    if constexpr (std::is_same_v<uint8_t, T>) {
        return vqaddq_u8(a, b);
    } else if constexpr (std::is_same_v<int8_t, T>) {
        return vqaddq_s8(a, b);
    } else if constexpr (std::is_same_v<uint16_t, T>) {
        return vqaddq_u16(a, b);
    } else if constexpr (std::is_same_v<int16_t, T>) {
        return vqaddq_s16(a, b);
    } else if constexpr (std::is_same_v<uint32_t, T>) {
        return vqaddq_u32(a, b);
    } else if constexpr (std::is_same_v<int32_t, T>) {
        return vqaddq_s32(a, b);
    } else if constexpr (std::is_same_v<uint64_t, T>) {
        return vqaddq_u64(a, b);
    } else if constexpr (std::is_same_v<int64_t, T>) {
        return vqaddq_s64(a, b);
    } else {
        static_assert(dependent_false_v<T>, "ERROR - psSimd::addSat(NEON): No native hardware SIMD saturating add for this type");
    }
#elif defined(AVX2_WRAPPERS)
    if constexpr (std::is_same_v<uint8_t, T>) {
        return _mm_adds_epu8(a, b);
    } else if constexpr (std::is_same_v<int8_t, T>) {
        return _mm_adds_epi8(a, b);
    } else if constexpr (std::is_same_v<uint16_t, T>) {
        return _mm_adds_epu16(a, b);
    } else if constexpr (std::is_same_v<int16_t, T>) {
        return _mm_adds_epi16(a, b);
    } else {
        static_assert(dependent_false_v<T>, "ERROR - psSimd::addSat(AVX2): No native hardware SIMD saturating add for this type");
    }
#else
    static_assert(std::is_integral_v<T>, "ERROR - psSimd::addSat: Only integral types support saturating arithmetic");
    return scalarAddSat<T>(a, b);
#endif
}

template <typename T>
FORCE_INLINE typename VectorType<T>::V subSat(typename VectorType<T>::V a, typename VectorType<T>::V b) {
#if defined(NEON_WRAPPERS)
    if constexpr (std::is_same_v<uint8_t, T>) {
        return vqsubq_u8(a, b);
    } else if constexpr (std::is_same_v<int8_t, T>) {
        return vqsubq_s8(a, b);
    } else if constexpr (std::is_same_v<uint16_t, T>) {
        return vqsubq_u16(a, b);
    } else if constexpr (std::is_same_v<int16_t, T>) {
        return vqsubq_s16(a, b);
    } else if constexpr (std::is_same_v<uint32_t, T>) {
        return vqsubq_u32(a, b);
    } else if constexpr (std::is_same_v<int32_t, T>) {
        return vqsubq_s32(a, b);
    } else if constexpr (std::is_same_v<uint64_t, T>) {
        return vqsubq_u64(a, b);
    } else if constexpr (std::is_same_v<int64_t, T>) {
        return vqsubq_s64(a, b);
    } else {
        static_assert(dependent_false_v<T>, "ERROR - psSimd::subSat(NEON): No native hardware SIMD saturating subtract for this type");
    }
#elif defined(AVX2_WRAPPERS)
    if constexpr (std::is_same_v<uint8_t, T>) {
        return _mm_subs_epu8(a, b);
    } else if constexpr (std::is_same_v<int8_t, T>) {
        return _mm_subs_epi8(a, b);
    } else if constexpr (std::is_same_v<uint16_t, T>) {
        return _mm_subs_epu16(a, b);
    } else if constexpr (std::is_same_v<int16_t, T>) {
        return _mm_subs_epi16(a, b);
    } else {
        static_assert(dependent_false_v<T>, "ERROR - psSimd::subSat(AVX2): No native hardware SIMD saturating subtract for this type");
    }
#else
    static_assert(std::is_integral_v<T>, "ERROR - psSimd::subSat: Only integral types support saturating arithmetic");
    return scalarSubSat<T>(a, b);
#endif
}

template <typename T>
FORCE_INLINE typename VectorType<T>::V fma(typename VectorType<T>::V a, typename VectorType<T>::V b, typename VectorType<T>::V c) {
#if defined(NEON_WRAPPERS)
    if constexpr (std::is_same_v<uint8_t, T>) {
        return vmlaq_u8(c, a, b);
    } else if constexpr (std::is_same_v<int8_t, T>) {
        return vmlaq_s8(c, a, b);
    } else if constexpr (std::is_same_v<uint16_t, T>) {
        return vmlaq_u16(c, a, b);
    } else if constexpr (std::is_same_v<int16_t, T>) {
        return vmlaq_s16(c, a, b);
    } else if constexpr (std::is_same_v<uint32_t, T>) {
        return vmlaq_u32(c, a, b);
    } else if constexpr (std::is_same_v<int32_t, T>) {
        return vmlaq_s32(c, a, b);
    } else if constexpr (std::is_same_v<float, T>) {
        return vfmaq_f32(c, a, b);
    } else if constexpr (std::is_same_v<double, T>) {
        return vfmaq_f64(c, a, b);
    } else {
        static_assert(dependent_false_v<T>, "ERROR - psSimd::fma(NEON): No native hardware SIMD FMA for this type");
    }
#elif defined(AVX2_WRAPPERS)
    if constexpr (std::is_same_v<float, T>) {
        return _mm_fmadd_ps(a, b, c);
    } else if constexpr (std::is_same_v<double, T>) {
        return _mm_fmadd_pd(a, b, c);
    } else {
        static_assert(dependent_false_v<T>, "ERROR - psSimd::fma(AVX2): No native hardware SIMD FMA for this type");
    }
#else
    if constexpr (std::is_floating_point_v<T>) {
        return std::fma(a, b, c);
    } else {
        return (a * b) + c;
    }
#endif
}

template <typename T>
FORCE_INLINE typename VectorType<T>::V rcp(typename VectorType<T>::V a) {
#if defined(NEON_WRAPPERS)
    if constexpr (std::is_same_v<float, T>) {
        return vrecpeq_f32(a);
    } else if constexpr (std::is_same_v<double, T>) {
        return vrecpeq_f64(a);
    } else {
        static_assert(dependent_false_v<T>, "ERROR - psSimd::rcp(NEON): No native hardware SIMD reciprocal for this type");
    }
#elif defined(AVX2_WRAPPERS)
    if constexpr (std::is_same_v<float, T>) {
        return _mm_rcp_ps(a);
    } else {
        static_assert(dependent_false_v<T>, "ERROR - psSimd::rcp(AVX2): No native hardware SIMD reciprocal for this type");
    }
#else
    static_assert(std::is_floating_point_v<T>, "ERROR - psSimd::rcp: Reciprocal is only valid for floating-point types");
    return static_cast<T>(1) / a;
#endif
}

template <typename T>
FORCE_INLINE typename VectorType<T>::V rsqrt(typename VectorType<T>::V a) {
#if defined(NEON_WRAPPERS)
    if constexpr (std::is_same_v<float, T>) {
        return vrsqrteq_f32(a);
    } else if constexpr (std::is_same_v<double, T>) {
        return vrsqrteq_f64(a);
    } else {
        static_assert(dependent_false_v<T>, "ERROR - psSimd::rsqrt(NEON): No native hardware SIMD reciprocal square root for this type");
    }
#elif defined(AVX2_WRAPPERS)
    if constexpr (std::is_same_v<float, T>) {
        return _mm_rsqrt_ps(a);
    } else {
        static_assert(dependent_false_v<T>, "ERROR - psSimd::rsqrt(AVX2): No native hardware SIMD reciprocal square root for this type");
    }
#else
    static_assert(std::is_floating_point_v<T>, "ERROR - psSimd::rsqrt: Reciprocal square root is only valid for floating-point types");
    return static_cast<T>(1) / std::sqrt(a);
#endif
}

template <typename T>
FORCE_INLINE typename VectorType<T>::V sqrt(typename VectorType<T>::V a) {
    static_assert(std::is_floating_point_v<T>, "ERROR - psSimd::sqrt: Square root is only valid for floating-point types");
#if defined(NEON_WRAPPERS)
    if constexpr (std::is_same_v<float, T>) {
        return vsqrtq_f32(a);
    } else if constexpr (std::is_same_v<double, T>) {
        return vsqrtq_f64(a);
    } else {
        static_assert(dependent_false_v<T>, "ERROR - psSimd::sqrt(NEON): Unsupported data type");
    }
#elif defined(AVX2_WRAPPERS)
    if constexpr (std::is_same_v<float, T>) {
        return _mm_sqrt_ps(a);
    } else if constexpr (std::is_same_v<double, T>) {
        return _mm_sqrt_pd(a);
    } else {
        static_assert(dependent_false_v<T>, "ERROR - psSimd::sqrt(AVX2): Unsupported data type");
    }
#else
    return std::sqrt(a);
#endif
}

template <typename T>
FORCE_INLINE typename VectorType<T>::V min(typename VectorType<T>::V a, typename VectorType<T>::V b) {
#if defined(NEON_WRAPPERS)
    if constexpr (std::is_same_v<uint8_t, T>) {
        return vminq_u8(a, b);
    } else if constexpr (std::is_same_v<int8_t, T>) {
        return vminq_s8(a, b);
    } else if constexpr (std::is_same_v<uint16_t, T>) {
        return vminq_u16(a, b);
    } else if constexpr (std::is_same_v<int16_t, T>) {
        return vminq_s16(a, b);
    } else if constexpr (std::is_same_v<uint32_t, T>) {
        return vminq_u32(a, b);
    } else if constexpr (std::is_same_v<int32_t, T>) {
        return vminq_s32(a, b);
    } else if constexpr (std::is_same_v<float, T>) {
        return vminq_f32(a, b);
    } else if constexpr (std::is_same_v<double, T>) {
        return vminq_f64(a, b);
    } else {
        static_assert(dependent_false_v<T>, "ERROR - psSimd::min(NEON): No native hardware SIMD min for this type");
    }
#elif defined(AVX2_WRAPPERS)
    if constexpr (std::is_same_v<uint8_t, T>) {
        return _mm_min_epu8(a, b);
    } else if constexpr (std::is_same_v<int8_t, T>) {
        return _mm_min_epi8(a, b);
    } else if constexpr (std::is_same_v<uint16_t, T>) {
        return _mm_min_epu16(a, b);
    } else if constexpr (std::is_same_v<int16_t, T>) {
        return _mm_min_epi16(a, b);
    } else if constexpr (std::is_same_v<uint32_t, T>) {
        return _mm_min_epu32(a, b);
    } else if constexpr (std::is_same_v<int32_t, T>) {
        return _mm_min_epi32(a, b);
    } else if constexpr (std::is_same_v<float, T>) {
        return _mm_min_ps(a, b);
    } else if constexpr (std::is_same_v<double, T>) {
        return _mm_min_pd(a, b);
    } else {
        static_assert(dependent_false_v<T>, "ERROR - psSimd::min(AVX2): No native hardware SIMD min for this type");
    }
#else
    return std::min(a, b);
#endif
}

template <typename T>
FORCE_INLINE typename VectorType<T>::V max(typename VectorType<T>::V a, typename VectorType<T>::V b) {
#if defined(NEON_WRAPPERS)
    if constexpr (std::is_same_v<uint8_t, T>) {
        return vmaxq_u8(a, b);
    } else if constexpr (std::is_same_v<int8_t, T>) {
        return vmaxq_s8(a, b);
    } else if constexpr (std::is_same_v<uint16_t, T>) {
        return vmaxq_u16(a, b);
    } else if constexpr (std::is_same_v<int16_t, T>) {
        return vmaxq_s16(a, b);
    } else if constexpr (std::is_same_v<uint32_t, T>) {
        return vmaxq_u32(a, b);
    } else if constexpr (std::is_same_v<int32_t, T>) {
        return vmaxq_s32(a, b);
    } else if constexpr (std::is_same_v<float, T>) {
        return vmaxq_f32(a, b);
    } else if constexpr (std::is_same_v<double, T>) {
        return vmaxq_f64(a, b);
    } else {
        static_assert(dependent_false_v<T>, "ERROR - psSimd::max(NEON): No native hardware SIMD max for this type");
    }
#elif defined(AVX2_WRAPPERS)
    if constexpr (std::is_same_v<uint8_t, T>) {
        return _mm_max_epu8(a, b);
    } else if constexpr (std::is_same_v<int8_t, T>) {
        return _mm_max_epi8(a, b);
    } else if constexpr (std::is_same_v<uint16_t, T>) {
        return _mm_max_epu16(a, b);
    } else if constexpr (std::is_same_v<int16_t, T>) {
        return _mm_max_epi16(a, b);
    } else if constexpr (std::is_same_v<uint32_t, T>) {
        return _mm_max_epu32(a, b);
    } else if constexpr (std::is_same_v<int32_t, T>) {
        return _mm_max_epi32(a, b);
    } else if constexpr (std::is_same_v<float, T>) {
        return _mm_max_ps(a, b);
    } else if constexpr (std::is_same_v<double, T>) {
        return _mm_max_pd(a, b);
    } else {
        static_assert(dependent_false_v<T>, "ERROR - psSimd::max(AVX2): No native hardware SIMD max for this type");
    }
#else
    return std::max(a, b);
#endif
}

template <typename To, typename From>
FORCE_INLINE typename VectorType<To>::V toFloat(typename VectorType<From>::V a) {
    static_assert(std::is_floating_point_v<To>, "ERROR - psSimd::toFloat: Destination type must be floating-point");
    static_assert(std::is_integral_v<From>, "ERROR - psSimd::toFloat: Source type must be integral");

#if defined(NEON_WRAPPERS)
    if constexpr (std::is_same_v<float, To> && std::is_same_v<int32_t, From>) {
        return vcvtq_f32_s32(a);
    } else if constexpr (std::is_same_v<float, To> && std::is_same_v<uint32_t, From>) {
        return vcvtq_f32_u32(a);
    } else if constexpr (std::is_same_v<double, To> && std::is_same_v<int64_t, From>) {
        return vcvtq_f64_s64(a);
    } else if constexpr (std::is_same_v<double, To> && std::is_same_v<uint64_t, From>) {
        return vcvtq_f64_u64(a);
    } else {
        static_assert(dependent_false_v<From>, "ERROR - psSimd::toFloat(NEON): No native hardware SIMD conversion for this type pair");
    }
#elif defined(AVX2_WRAPPERS)
    if constexpr (std::is_same_v<float, To> && std::is_same_v<int32_t, From>) {
        return _mm_cvtepi32_ps(a);
    } else {
        static_assert(dependent_false_v<From>, "ERROR - psSimd::toFloat(AVX2): No native hardware SIMD conversion for this type pair");
    }
#else
    return static_cast<To>(a);
#endif
}

struct FloatFrom8 {
    typename VectorType<float>::V v[4];
};

template <typename To, typename From>
FORCE_INLINE typename VectorType<To>::V toInt(typename VectorType<From>::V a) {
    static_assert(std::is_integral_v<To>, "ERROR - psSimd::toInt: Destination type must be integral");
    static_assert(std::is_floating_point_v<From>, "ERROR - psSimd::toInt: Source type must be floating-point");

#if defined(NEON_WRAPPERS)
    if constexpr (std::is_same_v<int32_t, To> && std::is_same_v<float, From>) {
        return vcvtq_s32_f32(a);
    } else if constexpr (std::is_same_v<uint32_t, To> && std::is_same_v<float, From>) {
        return vcvtq_u32_f32(a);
    } else if constexpr (std::is_same_v<int64_t, To> && std::is_same_v<double, From>) {
        return vcvtq_s64_f64(a);
    } else if constexpr (std::is_same_v<uint64_t, To> && std::is_same_v<double, From>) {
        return vcvtq_u64_f64(a);
    } else {
        static_assert(dependent_false_v<From>, "ERROR - psSimd::toInt(NEON): No native hardware SIMD conversion for this type pair");
    }
#elif defined(AVX2_WRAPPERS)
    if constexpr (std::is_same_v<int32_t, To> && std::is_same_v<float, From>) {
        return _mm_cvttps_epi32(a);
    } else {
        static_assert(dependent_false_v<From>, "ERROR - psSimd::toInt(AVX2): No native hardware SIMD conversion for this type pair");
    }
#else
    return static_cast<To>(a);
#endif
}

template <typename To, typename From>
FORCE_INLINE typename VectorType<To>::V roundToInt(typename VectorType<From>::V a) {
    static_assert(std::is_integral_v<To>, "ERROR - psSimd::roundToInt: Destination type must be integral");
    static_assert(std::is_floating_point_v<From>, "ERROR - psSimd::roundToInt: Source type must be floating-point");

#if defined(NEON_WRAPPERS)
    if constexpr (std::is_same_v<int32_t, To> && std::is_same_v<float, From>) {
        return vcvtnq_s32_f32(a);
    } else if constexpr (std::is_same_v<uint32_t, To> && std::is_same_v<float, From>) {
        return vcvtnq_u32_f32(a);
    } else if constexpr (std::is_same_v<int64_t, To> && std::is_same_v<double, From>) {
        return vcvtnq_s64_f64(a);
    } else if constexpr (std::is_same_v<uint64_t, To> && std::is_same_v<double, From>) {
        return vcvtnq_u64_f64(a);
    } else {
        static_assert(dependent_false_v<From>, "ERROR - psSimd::roundToInt(NEON): No native hardware SIMD conversion for this type pair");
    }
#elif defined(AVX2_WRAPPERS)
    if constexpr (std::is_same_v<int32_t, To> && std::is_same_v<float, From>) {
        return _mm_cvtps_epi32(a);
    } else {
        static_assert(dependent_false_v<From>, "ERROR - psSimd::roundToInt(AVX2): No native hardware SIMD conversion for this type pair");
    }
#else
    return static_cast<To>(std::round(a));
#endif
}

template <int Quarter, typename From>
FORCE_INLINE typename VectorType<float>::V convertQuarter8ToFloat(typename VectorType<From>::V a) {
    static_assert(std::is_same_v<int8_t, From> || std::is_same_v<uint8_t, From>, "ERROR - psSimd::convertQuarter8ToFloat: Source must be int8_t or uint8_t");
    static_assert(Quarter >= 0 && Quarter < 4, "ERROR - psSimd::convertQuarter8ToFloat: Quarter must be between 0 and 3");

#if defined(NEON_WRAPPERS)
    if constexpr (std::is_same_v<uint8_t, From>) {
        if constexpr (Quarter == 0) {
            uint16x8_t w16 = vmovl_u8(vget_low_u8(a));
            return vcvtq_f32_u32(vmovl_u16(vget_low_u16(w16)));
        } else if constexpr (Quarter == 1) {
            uint16x8_t w16 = vmovl_u8(vget_low_u8(a));
            return vcvtq_f32_u32(vmovl_u16(vget_high_u16(w16)));
        } else if constexpr (Quarter == 2) {
            uint16x8_t w16 = vmovl_u8(vget_high_u8(a));
            return vcvtq_f32_u32(vmovl_u16(vget_low_u16(w16)));
        } else {
            uint16x8_t w16 = vmovl_u8(vget_high_u8(a));
            return vcvtq_f32_u32(vmovl_u16(vget_high_u16(w16)));
        }
    } else {
        if constexpr (Quarter == 0) {
            int16x8_t w16 = vmovl_s8(vget_low_s8(a));
            return vcvtq_f32_s32(vmovl_s16(vget_low_s16(w16)));
        } else if constexpr (Quarter == 1) {
            int16x8_t w16 = vmovl_s8(vget_low_s8(a));
            return vcvtq_f32_s32(vmovl_s16(vget_high_s16(w16)));
        } else if constexpr (Quarter == 2) {
            int16x8_t w16 = vmovl_s8(vget_high_s8(a));
            return vcvtq_f32_s32(vmovl_s16(vget_low_s16(w16)));
        } else {
            int16x8_t w16 = vmovl_s8(vget_high_s8(a));
            return vcvtq_f32_s32(vmovl_s16(vget_high_s16(w16)));
        }
    }
#elif defined(AVX2_WRAPPERS)
    __m128i shifted = a;
    if constexpr (Quarter == 1) {
        shifted = _mm_srli_si128(a, 4);
    } else if constexpr (Quarter == 2) {
        shifted = _mm_srli_si128(a, 8);
    } else if constexpr (Quarter == 3) {
        shifted = _mm_srli_si128(a, 12);
    }

    if constexpr (std::is_same_v<uint8_t, From>) {
        return _mm_cvtepi32_ps(_mm_cvtepu8_epi32(shifted));
    } else {
        return _mm_cvtepi32_ps(_mm_cvtepi8_epi32(shifted));
    }
#else
    if constexpr (Quarter == 0) {
        return static_cast<float>(a);
    } else {
        return 0.0f;
    }
#endif
}

template <typename From>
FORCE_INLINE FloatFrom8 convert8ToFloat(typename VectorType<From>::V a) {
    FloatFrom8 res;
    res.v[0] = convertQuarter8ToFloat<0, From>(a);
    res.v[1] = convertQuarter8ToFloat<1, From>(a);
    res.v[2] = convertQuarter8ToFloat<2, From>(a);
    res.v[3] = convertQuarter8ToFloat<3, From>(a);
    return res;
}

}
