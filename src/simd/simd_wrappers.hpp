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

}

