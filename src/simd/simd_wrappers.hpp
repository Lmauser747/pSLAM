#pragma once

#include <cstdint>
#include <vector>
#include <type_traits>


#if defined(__GNUC__) || defined(__clang__)
  #define FORCE_INLINE inline __attribute__((always_inline))
#else
  #define FORCE_INLINE inline
#endif

#ifdef __aarch64__
    #ifdef __ARM_NEON
    #include <arm_neon.h>
    #define NEON_WRAPPERS
    #else 
    #define SCALAR_WRAPPERS
    #endif
#endif  
#ifndef __aarch64__
    #ifdef __AVX2__
    #include <immintrin.h>
    #define AVX2_WRAPPERS
    #else
    #define SCALAR_WRAPPERS
    #endif
#endif

namespace psSimd{

template<typename T>
struct VectorType;

template <>
struct VectorType<uint8_t> {
#if defined(NEON_WRAPPERS)
    using V = uint8x16_t;
#elif defined(AVX2_WRAPPERS)
    using V = __m128i;
#else
    using V = uint8_t;
#endif
};

template <>
struct VectorType<int8_t> {
#if defined(NEON_WRAPPERS)
    using V = int8x16_t;
#elif defined(AVX2_WRAPPERS)
    using V = __m128i;
#else
    using V = int8_t;
#endif
};

template <>
struct VectorType<uint16_t> {
#if defined(NEON_WRAPPERS)
    using V = uint16x8_t;
#elif defined(AVX2_WRAPPERS)
    using V = __m128i;
#else
    using V = uint16_t;
#endif
};

template <>
struct VectorType<int16_t> {
#if defined(NEON_WRAPPERS)
    using V = int16x8_t;
#elif defined(AVX2_WRAPPERS)
    using V = __m128i;
#else
    using V = int16_t;
#endif
};

template <>
struct VectorType<uint32_t> {
#if defined(NEON_WRAPPERS)
    using V = uint32x4_t;
#elif defined(AVX2_WRAPPERS)
    using V = __m128i;
#else
    using V = uint32_t;
#endif
};

template <>
struct VectorType<int32_t> {
#if defined(NEON_WRAPPERS)
    using V = int32x4_t;
#elif defined(AVX2_WRAPPERS)
    using V = __m128i;
#else
    using V = int32_t;
#endif
};

template <>
struct VectorType<uint64_t> {
#if defined(NEON_WRAPPERS)
    using V = uint64x2_t;
#elif defined(AVX2_WRAPPERS)
    using V = __m128i;
#else
    using V = uint64_t;
#endif
};

template <>
struct VectorType<int64_t> {
#if defined(NEON_WRAPPERS)
    using V = int64x2_t;
#elif defined(AVX2_WRAPPERS)
    using V = __m128i;
#else
    using V = int64_t;
#endif
};

template <>
struct VectorType<float> {
#if defined(NEON_WRAPPERS)
    using V = float32x4_t;
#elif defined(AVX2_WRAPPERS)
    using V = __m128;
#else
    using V = float;
#endif
};

template <>
struct VectorType<double> {
#if defined(NEON_WRAPPERS)
    using V = float64x2_t;
#elif defined(AVX2_WRAPPERS)
    using V = __m128d;
#else
    using V = double;
#endif
};

template <typename T>
struct dependent_false : std::false_type {};

template <typename T>
inline constexpr bool dependent_false_v = dependent_false<T>::value;

template <typename T>
struct WidenType;

template <> struct WidenType<int8_t> {
    using type = int16_t;
};
template <> struct WidenType<uint8_t> {
    using type = uint16_t;
};
template <> struct WidenType<int16_t> {
    using type = int32_t;
};
template <> struct WidenType<uint16_t> {
    using type = uint32_t;
};
template <> struct WidenType<int32_t> {
    using type = int64_t;
};
template <> struct WidenType<uint32_t> {
    using type = uint64_t;
};

template <typename T>
using WidenType_t = typename WidenType<T>::type;

template <typename T>
struct WidenedResult {
    typename VectorType<WidenType_t<T>>::V low;
    typename VectorType<WidenType_t<T>>::V high;
};

template <typename T>
FORCE_INLINE typename VectorType<WidenType_t<T>>::V widenLow(typename VectorType<T>::V a) {
    static_assert(std::is_integral_v<T> && sizeof(T) < 8, "ERROR - psSimd::widenLow: Only 8, 16, and 32-bit integers can be widened");

#if defined(NEON_WRAPPERS)
    if constexpr (std::is_same_v<int8_t, T>) {
        return vmovl_s8(vget_low_s8(a));
    } else if constexpr (std::is_same_v<uint8_t, T>) {
        return vmovl_u8(vget_low_u8(a));
    } else if constexpr (std::is_same_v<int16_t, T>) {
        return vmovl_s16(vget_low_s16(a));
    } else if constexpr (std::is_same_v<uint16_t, T>) {
        return vmovl_u16(vget_low_u16(a));
    } else if constexpr (std::is_same_v<int32_t, T>) {
        return vmovl_s32(vget_low_s32(a));
    } else if constexpr (std::is_same_v<uint32_t, T>) {
        return vmovl_u32(vget_low_u32(a));
    } else {
        static_assert(dependent_false_v<T>, "ERROR - psSimd::widenLow(NEON): Unsupported type");
    }
#elif defined(AVX2_WRAPPERS)
    if constexpr (std::is_same_v<int8_t, T>) {
        return _mm_cvtepi8_epi16(a);
    } else if constexpr (std::is_same_v<uint8_t, T>) {
        return _mm_cvtepu8_epi16(a);
    } else if constexpr (std::is_same_v<int16_t, T>) {
        return _mm_cvtepi16_epi32(a);
    } else if constexpr (std::is_same_v<uint16_t, T>) {
        return _mm_cvtepu16_epi32(a);
    } else if constexpr (std::is_same_v<int32_t, T>) {
        return _mm_cvtepi32_epi64(a);
    } else if constexpr (std::is_same_v<uint32_t, T>) {
        return _mm_cvtepu32_epi64(a);
    } else {
        static_assert(dependent_false_v<T>, "ERROR - psSimd::widenLow(AVX2): Unsupported type");
    }
#else
    return static_cast<WidenType_t<T>>(a);
#endif
}

template <typename T>
FORCE_INLINE typename VectorType<WidenType_t<T>>::V widenHigh(typename VectorType<T>::V a) {
    static_assert(std::is_integral_v<T> && sizeof(T) < 8, "ERROR - psSimd::widenHigh: Only 8, 16, and 32-bit integers can be widened");

#if defined(NEON_WRAPPERS)
    if constexpr (std::is_same_v<int8_t, T>) {
        return vmovl_s8(vget_high_s8(a));
    } else if constexpr (std::is_same_v<uint8_t, T>) {
        return vmovl_u8(vget_high_u8(a));
    } else if constexpr (std::is_same_v<int16_t, T>) {
        return vmovl_s16(vget_high_s16(a));
    } else if constexpr (std::is_same_v<uint16_t, T>) {
        return vmovl_u16(vget_high_u16(a));
    } else if constexpr (std::is_same_v<int32_t, T>) {
        return vmovl_s32(vget_high_s32(a));
    } else if constexpr (std::is_same_v<uint32_t, T>) {
        return vmovl_u32(vget_high_u32(a));
    } else {
        static_assert(dependent_false_v<T>, "ERROR - psSimd::widenHigh(NEON): Unsupported type");
    }
#elif defined(AVX2_WRAPPERS)
    __m128i hi = _mm_srli_si128(a, 8);
    if constexpr (std::is_same_v<int8_t, T>) {
        return _mm_cvtepi8_epi16(hi);
    } else if constexpr (std::is_same_v<uint8_t, T>) {
        return _mm_cvtepu8_epi16(hi);
    } else if constexpr (std::is_same_v<int16_t, T>) {
        return _mm_cvtepi16_epi32(hi);
    } else if constexpr (std::is_same_v<uint16_t, T>) {
        return _mm_cvtepu16_epi32(hi);
    } else if constexpr (std::is_same_v<int32_t, T>) {
        return _mm_cvtepi32_epi64(hi);
    } else if constexpr (std::is_same_v<uint32_t, T>) {
        return _mm_cvtepu32_epi64(hi);
    } else {
        static_assert(dependent_false_v<T>, "ERROR - psSimd::widenHigh(AVX2): Unsupported type");
    }
#else
    return static_cast<WidenType_t<T>>(a);
#endif
}

template <typename T>
FORCE_INLINE WidenedResult<T> widen(typename VectorType<T>::V a) {
    return { widenLow<T>(a), widenHigh<T>(a) };
}

template <typename T>
FORCE_INLINE T reduceAdd(typename VectorType<T>::V a) {
#if defined(NEON_WRAPPERS)
    if constexpr (std::is_same_v<uint8_t, T>) {
        return vaddvq_u8(a);
    } else if constexpr (std::is_same_v<int8_t, T>) {
        return vaddvq_s8(a);
    } else if constexpr (std::is_same_v<uint16_t, T>) {
        return vaddvq_u16(a);
    } else if constexpr (std::is_same_v<int16_t, T>) {
        return vaddvq_s16(a);
    } else if constexpr (std::is_same_v<uint32_t, T>) {
        return vaddvq_u32(a);
    } else if constexpr (std::is_same_v<int32_t, T>) {
        return vaddvq_s32(a);
    } else if constexpr (std::is_same_v<uint64_t, T>) {
        return vaddvq_u64(a);
    } else if constexpr (std::is_same_v<int64_t, T>) {
        return vaddvq_s64(a);
    } else if constexpr (std::is_same_v<float, T>) {
        return vaddvq_f32(a);
    } else if constexpr (std::is_same_v<double, T>) {
        return vaddvq_f64(a);
    } else {
        static_assert(dependent_false_v<T>, "ERROR - psSimd::reduceAdd(NEON): Unsupported type");
    }
#elif defined(AVX2_WRAPPERS)
    if constexpr (std::is_same_v<float, T>) {
        __m128 shuf = _mm_movehdup_ps(a);
        __m128 sum = _mm_add_ps(a, shuf);
        shuf = _mm_movehl_ps(shuf, sum);
        sum = _mm_add_ss(sum, shuf);
        return _mm_cvtss_f32(sum);
    } else if constexpr (std::is_same_v<double, T>) {
        __m128d hi = _mm_unpackhi_pd(a, a);
        return _mm_cvtsd_f64(_mm_add_sd(a, hi));
    } else if constexpr (std::is_same_v<int64_t, T> || std::is_same_v<uint64_t, T>) {
        __m128i hi = _mm_unpackhi_epi64(a, a);
        __m128i sum = _mm_add_epi64(a, hi);
        return static_cast<T>(_mm_cvtsi128_si64(sum));
    } else if constexpr (std::is_same_v<int32_t, T> || std::is_same_v<uint32_t, T>) {
        __m128i hi64 = _mm_shuffle_epi32(a, _MM_SHUFFLE(1, 0, 3, 2));
        __m128i sum64 = _mm_add_epi32(a, hi64);
        __m128i hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));
        __m128i sum32 = _mm_add_epi32(sum64, hi32);
        return static_cast<T>(_mm_cvtsi128_si32(sum32));
    } else if constexpr (std::is_same_v<int16_t, T> || std::is_same_v<uint16_t, T>) {
        __m128i hi64 = _mm_srli_si128(a, 8);
        __m128i sum64 = _mm_add_epi16(a, hi64);
        __m128i hi32 = _mm_srli_si128(sum64, 4);
        __m128i sum32 = _mm_add_epi16(sum64, hi32);
        __m128i hi16 = _mm_srli_si128(sum32, 2);
        __m128i sum16 = _mm_add_epi16(sum32, hi16);
        return static_cast<T>(_mm_extract_epi16(sum16, 0));
    } else if constexpr (std::is_same_v<uint8_t, T>) {
        __m128i zero = _mm_setzero_si128();
        __m128i sad = _mm_sad_epu8(a, zero);
        __m128i hi = _mm_srli_si128(sad, 8);
        return static_cast<uint8_t>(_mm_cvtsi128_si32(_mm_add_epi16(sad, hi)));
    } else if constexpr (std::is_same_v<int8_t, T>) {
        __m128i lo = _mm_cvtepi8_epi16(a);
        __m128i hi = _mm_cvtepi8_epi16(_mm_srli_si128(a, 8));
        __m128i sum16 = _mm_add_epi16(lo, hi);
        __m128i hi64 = _mm_srli_si128(sum16, 8);
        sum16 = _mm_add_epi16(sum16, hi64);
        __m128i hi32 = _mm_srli_si128(sum16, 4);
        sum16 = _mm_add_epi16(sum16, hi32);
        __m128i hi16 = _mm_srli_si128(sum16, 2);
        sum16 = _mm_add_epi16(sum16, hi16);
        return static_cast<int8_t>(_mm_extract_epi16(sum16, 0));
    } else {
        static_assert(dependent_false_v<T>, "ERROR - psSimd::reduceAdd(AVX2): Unsupported type");
    }
#else
    return a;
#endif
}

}

