#pragma once

#include <cstdint>
#include <type_traits>

#if defined(NEON_WRAPPERS)
#include <arm_neon.h>
#elif defined(AVX2_WRAPPERS)
#include <immintrin.h>
#endif

#include "simd/simd_wrappers.hpp"

namespace psSimd{

template <typename T>
FORCE_INLINE typename VectorType<T>::V bitAnd(typename VectorType<T>::V a, typename VectorType<T>::V b) {
#if defined(NEON_WRAPPERS)
    if constexpr (std::is_same_v<uint8_t, T>) {
        return vandq_u8(a, b);
    } else if constexpr (std::is_same_v<int8_t, T>) {
        return vandq_s8(a, b);
    } else if constexpr (std::is_same_v<uint16_t, T>) {
        return vandq_u16(a, b);
    } else if constexpr (std::is_same_v<int16_t, T>) {
        return vandq_s16(a, b);
    } else if constexpr (std::is_same_v<uint32_t, T>) {
        return vandq_u32(a, b);
    } else if constexpr (std::is_same_v<int32_t, T>) {
        return vandq_s32(a, b);
    } else if constexpr (std::is_same_v<uint64_t, T>) {
        return vandq_u64(a, b);
    } else if constexpr (std::is_same_v<int64_t, T>) {
        return vandq_s64(a, b);
    } else {
        static_assert(dependent_false_v<T>, "ERROR - psSimd::bitAnd(NEON): No native hardware SIMD bitwise AND for this type");
    }
#elif defined(AVX2_WRAPPERS)
    if constexpr (std::is_integral_v<T>) {
        return _mm_and_si128(a, b);
    } else if constexpr (std::is_same_v<float, T>) {
        return _mm_and_ps(a, b);
    } else if constexpr (std::is_same_v<double, T>) {
        return _mm_and_pd(a, b);
    } else {
        static_assert(dependent_false_v<T>, "ERROR - psSimd::bitAnd(AVX2): No native hardware SIMD bitwise AND for this type");
    }
#else
    static_assert(std::is_integral_v<T>, "ERROR - psSimd::bitAnd: Bitwise AND requires integral scalar types");
    return a & b;
#endif
}

template <typename T>
FORCE_INLINE typename VectorType<T>::V bitOr(typename VectorType<T>::V a, typename VectorType<T>::V b) {
#if defined(NEON_WRAPPERS)
    if constexpr (std::is_same_v<uint8_t, T>) {
        return vorrq_u8(a, b);
    } else if constexpr (std::is_same_v<int8_t, T>) {
        return vorrq_s8(a, b);
    } else if constexpr (std::is_same_v<uint16_t, T>) {
        return vorrq_u16(a, b);
    } else if constexpr (std::is_same_v<int16_t, T>) {
        return vorrq_s16(a, b);
    } else if constexpr (std::is_same_v<uint32_t, T>) {
        return vorrq_u32(a, b);
    } else if constexpr (std::is_same_v<int32_t, T>) {
        return vorrq_s32(a, b);
    } else if constexpr (std::is_same_v<uint64_t, T>) {
        return vorrq_u64(a, b);
    } else if constexpr (std::is_same_v<int64_t, T>) {
        return vorrq_s64(a, b);
    } else {
        static_assert(dependent_false_v<T>, "ERROR - psSimd::bitOr(NEON): No native hardware SIMD bitwise OR for this type");
    }
#elif defined(AVX2_WRAPPERS)
    if constexpr (std::is_integral_v<T>) {
        return _mm_or_si128(a, b);
    } else if constexpr (std::is_same_v<float, T>) {
        return _mm_or_ps(a, b);
    } else if constexpr (std::is_same_v<double, T>) {
        return _mm_or_pd(a, b);
    } else {
        static_assert(dependent_false_v<T>, "ERROR - psSimd::bitOr(AVX2): No native hardware SIMD bitwise OR for this type");
    }
#else
    static_assert(std::is_integral_v<T>, "ERROR - psSimd::bitOr: Bitwise OR requires integral scalar types");
    return a | b;
#endif
}

template <typename T>
FORCE_INLINE typename VectorType<T>::V bitXor(typename VectorType<T>::V a, typename VectorType<T>::V b) {
#if defined(NEON_WRAPPERS)
    if constexpr (std::is_same_v<uint8_t, T>) {
        return veorq_u8(a, b);
    } else if constexpr (std::is_same_v<int8_t, T>) {
        return veorq_s8(a, b);
    } else if constexpr (std::is_same_v<uint16_t, T>) {
        return veorq_u16(a, b);
    } else if constexpr (std::is_same_v<int16_t, T>) {
        return veorq_s16(a, b);
    } else if constexpr (std::is_same_v<uint32_t, T>) {
        return veorq_u32(a, b);
    } else if constexpr (std::is_same_v<int32_t, T>) {
        return veorq_s32(a, b);
    } else if constexpr (std::is_same_v<uint64_t, T>) {
        return veorq_u64(a, b);
    } else if constexpr (std::is_same_v<int64_t, T>) {
        return veorq_s64(a, b);
    } else {
        static_assert(dependent_false_v<T>, "ERROR - psSimd::bitXor(NEON): No native hardware SIMD bitwise XOR for this type");
    }
#elif defined(AVX2_WRAPPERS)
    if constexpr (std::is_integral_v<T>) {
        return _mm_xor_si128(a, b);
    } else if constexpr (std::is_same_v<float, T>) {
        return _mm_xor_ps(a, b);
    } else if constexpr (std::is_same_v<double, T>) {
        return _mm_xor_pd(a, b);
    } else {
        static_assert(dependent_false_v<T>, "ERROR - psSimd::bitXor(AVX2): No native hardware SIMD bitwise XOR for this type");
    }
#else
    static_assert(std::is_integral_v<T>, "ERROR - psSimd::bitXor: Bitwise XOR requires integral scalar types");
    return a ^ b;
#endif
}

template <typename T>
FORCE_INLINE typename VectorType<T>::V bitNot(typename VectorType<T>::V a) {
#if defined(NEON_WRAPPERS)
    if constexpr (std::is_same_v<uint8_t, T>) {
        return vmvnq_u8(a);
    } else if constexpr (std::is_same_v<int8_t, T>) {
        return vmvnq_s8(a);
    } else if constexpr (std::is_same_v<uint16_t, T>) {
        return vmvnq_u16(a);
    } else if constexpr (std::is_same_v<int16_t, T>) {
        return vmvnq_s16(a);
    } else if constexpr (std::is_same_v<uint32_t, T>) {
        return vmvnq_u32(a);
    } else if constexpr (std::is_same_v<int32_t, T>) {
        return vmvnq_s32(a);
    } else {
        static_assert(dependent_false_v<T>, "ERROR - psSimd::bitNot(NEON): No native hardware SIMD bitwise NOT for this type");
    }
#elif defined(AVX2_WRAPPERS)
    static_assert(dependent_false_v<T>, "ERROR - psSimd::bitNot(AVX2): No native hardware SIMD bitwise NOT instruction exists in AVX2/SSE");
#else
    static_assert(std::is_integral_v<T>, "ERROR - psSimd::bitNot: Bitwise NOT requires integral scalar types");
    return ~a;
#endif
}

template <typename T>
FORCE_INLINE typename VectorType<T>::V bitAndNot(typename VectorType<T>::V a, typename VectorType<T>::V b) {
#if defined(NEON_WRAPPERS)
    if constexpr (std::is_same_v<uint8_t, T>) {
        return vbicq_u8(a, b);
    } else if constexpr (std::is_same_v<int8_t, T>) {
        return vbicq_s8(a, b);
    } else if constexpr (std::is_same_v<uint16_t, T>) {
        return vbicq_u16(a, b);
    } else if constexpr (std::is_same_v<int16_t, T>) {
        return vbicq_s16(a, b);
    } else if constexpr (std::is_same_v<uint32_t, T>) {
        return vbicq_u32(a, b);
    } else if constexpr (std::is_same_v<int32_t, T>) {
        return vbicq_s32(a, b);
    } else if constexpr (std::is_same_v<uint64_t, T>) {
        return vbicq_u64(a, b);
    } else if constexpr (std::is_same_v<int64_t, T>) {
        return vbicq_s64(a, b);
    } else {
        static_assert(dependent_false_v<T>, "ERROR - psSimd::bitAndNot(NEON): No native hardware SIMD bitwise BIC for this type");
    }
#elif defined(AVX2_WRAPPERS)
    if constexpr (std::is_integral_v<T>) {
        return _mm_andnot_si128(b, a);
    } else if constexpr (std::is_same_v<float, T>) {
        return _mm_andnot_ps(b, a);
    } else if constexpr (std::is_same_v<double, T>) {
        return _mm_andnot_pd(b, a);
    } else {
        static_assert(dependent_false_v<T>, "ERROR - psSimd::bitAndNot(AVX2): No native hardware SIMD bitwise ANDNOT for this type");
    }
#else
    static_assert(std::is_integral_v<T>, "ERROR - psSimd::bitAndNot: Bitwise operations require integral scalar types");
    return a & (~b);
#endif
}

template <int Shift, typename T>
FORCE_INLINE typename VectorType<T>::V shiftLeft(typename VectorType<T>::V a) {
    static_assert(std::is_integral_v<T>, "ERROR - psSimd::shiftLeft: Shifts require integral types");
    static_assert(Shift >= 0 && Shift < static_cast<int>(sizeof(T) * 8), "ERROR - psSimd::shiftLeft: Shift amount out of bounds");

    if constexpr (Shift == 0) {
        return a;
    }

#if defined(NEON_WRAPPERS)
    if constexpr (std::is_same_v<uint8_t, T>) {
        return vshlq_n_u8(a, Shift);
    } else if constexpr (std::is_same_v<int8_t, T>) {
        return vshlq_n_s8(a, Shift);
    } else if constexpr (std::is_same_v<uint16_t, T>) {
        return vshlq_n_u16(a, Shift);
    } else if constexpr (std::is_same_v<int16_t, T>) {
        return vshlq_n_s16(a, Shift);
    } else if constexpr (std::is_same_v<uint32_t, T>) {
        return vshlq_n_u32(a, Shift);
    } else if constexpr (std::is_same_v<int32_t, T>) {
        return vshlq_n_s32(a, Shift);
    } else if constexpr (std::is_same_v<uint64_t, T>) {
        return vshlq_n_u64(a, Shift);
    } else if constexpr (std::is_same_v<int64_t, T>) {
        return vshlq_n_s64(a, Shift);
    } else {
        static_assert(dependent_false_v<T>, "ERROR - psSimd::shiftLeft(NEON): Unsupported type");
    }
#elif defined(AVX2_WRAPPERS)
    if constexpr (std::is_same_v<uint16_t, T> || std::is_same_v<int16_t, T>) {
        return _mm_slli_epi16(a, Shift);
    } else if constexpr (std::is_same_v<uint32_t, T> || std::is_same_v<int32_t, T>) {
        return _mm_slli_epi32(a, Shift);
    } else if constexpr (std::is_same_v<uint64_t, T> || std::is_same_v<int64_t, T>) {
        return _mm_slli_epi64(a, Shift);
    } else {
        static_assert(dependent_false_v<T>, "ERROR - psSimd::shiftLeft(AVX2): No native hardware SIMD shift left for this type");
    }
#else
    return a << Shift;
#endif
}

template <int Shift, typename T>
FORCE_INLINE typename VectorType<T>::V shiftRightLogical(typename VectorType<T>::V a) {
    static_assert(std::is_integral_v<T> && std::is_unsigned_v<T>, "ERROR - psSimd::shiftRightLogical: Logical shift right requires unsigned integral types");
    static_assert(Shift >= 0 && Shift < static_cast<int>(sizeof(T) * 8), "ERROR - psSimd::shiftRightLogical: Shift amount out of bounds");

    if constexpr (Shift == 0) {
        return a;
    }

#if defined(NEON_WRAPPERS)
    if constexpr (std::is_same_v<uint8_t, T>) {
        return vshrq_n_u8(a, Shift);
    } else if constexpr (std::is_same_v<uint16_t, T>) {
        return vshrq_n_u16(a, Shift);
    } else if constexpr (std::is_same_v<uint32_t, T>) {
        return vshrq_n_u32(a, Shift);
    } else if constexpr (std::is_same_v<uint64_t, T>) {
        return vshrq_n_u64(a, Shift);
    } else {
        static_assert(dependent_false_v<T>, "ERROR - psSimd::shiftRightLogical(NEON): Unsupported type");
    }
#elif defined(AVX2_WRAPPERS)
    if constexpr (std::is_same_v<uint16_t, T>) {
        return _mm_srli_epi16(a, Shift);
    } else if constexpr (std::is_same_v<uint32_t, T>) {
        return _mm_srli_epi32(a, Shift);
    } else if constexpr (std::is_same_v<uint64_t, T>) {
        return _mm_srli_epi64(a, Shift);
    } else {
        static_assert(dependent_false_v<T>, "ERROR - psSimd::shiftRightLogical(AVX2): No native hardware SIMD logical shift right for this type");
    }
#else
    return a >> Shift;
#endif
}

template <int Shift, typename T>
FORCE_INLINE typename VectorType<T>::V shiftRightArithmetic(typename VectorType<T>::V a) {
    static_assert(std::is_integral_v<T> && std::is_signed_v<T>, "ERROR - psSimd::shiftRightArithmetic: Arithmetic shift right requires signed integral types");
    static_assert(Shift >= 0 && Shift < static_cast<int>(sizeof(T) * 8), "ERROR - psSimd::shiftRightArithmetic: Shift amount out of bounds");

    if constexpr (Shift == 0) {
        return a;
    }

#if defined(NEON_WRAPPERS)
    if constexpr (std::is_same_v<int8_t, T>) {
        return vshrq_n_s8(a, Shift);
    } else if constexpr (std::is_same_v<int16_t, T>) {
        return vshrq_n_s16(a, Shift);
    } else if constexpr (std::is_same_v<int32_t, T>) {
        return vshrq_n_s32(a, Shift);
    } else if constexpr (std::is_same_v<int64_t, T>) {
        return vshrq_n_s64(a, Shift);
    } else {
        static_assert(dependent_false_v<T>, "ERROR - psSimd::shiftRightArithmetic(NEON): Unsupported type");
    }
#elif defined(AVX2_WRAPPERS)
    if constexpr (std::is_same_v<int16_t, T>) {
        return _mm_srai_epi16(a, Shift);
    } else if constexpr (std::is_same_v<int32_t, T>) {
        return _mm_srai_epi32(a, Shift);
    } else {
        static_assert(dependent_false_v<T>, "ERROR - psSimd::shiftRightArithmetic(AVX2): No native hardware SIMD arithmetic shift right for this type");
    }
#else
    return a >> Shift;
#endif
}

}