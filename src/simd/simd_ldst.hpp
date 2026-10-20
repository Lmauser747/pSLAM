#pragma once

#include <cstdint>
#include <cstring>
#include <type_traits>

#include "simd/simd_wrappers.hpp"

#if defined(NEON_WRAPPERS)
#include <arm_neon.h>
#elif defined(AVX2_WRAPPERS)
#include <immintrin.h>
#endif

#ifndef FORCE_INLINE
#if defined(_MSC_VER)
#define FORCE_INLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
#define FORCE_INLINE inline __attribute__((always_inline))
#else
#define FORCE_INLINE inline
#endif
#endif

namespace psSimd{

template <typename T>
struct LaneCount {
    static constexpr int value = 16 / sizeof(T);
};

template <typename T>
inline constexpr int LaneCount_v = LaneCount<T>::value;

template <typename T>
FORCE_INLINE typename VectorType<T>::V load(const T* ptr) {
#if defined(NEON_WRAPPERS)
    if constexpr (std::is_same_v<uint8_t, T>) {
        return vld1q_u8(ptr);
    } else if constexpr (std::is_same_v<int8_t, T>) {
        return vld1q_s8(ptr);
    } else if constexpr (std::is_same_v<uint16_t, T>) {
        return vld1q_u16(ptr);
    } else if constexpr (std::is_same_v<int16_t, T>) {
        return vld1q_s16(ptr);
    } else if constexpr (std::is_same_v<uint32_t, T>) {
        return vld1q_u32(ptr);
    } else if constexpr (std::is_same_v<int32_t, T>) {
        return vld1q_s32(ptr);
    } else if constexpr (std::is_same_v<uint64_t, T>) {
        return vld1q_u64(ptr);
    } else if constexpr (std::is_same_v<int64_t, T>) {
        return vld1q_s64(ptr);
    } else if constexpr (std::is_same_v<float, T>) {
        return vld1q_f32(ptr);
    } else if constexpr (std::is_same_v<double, T>) {
        return vld1q_f64(ptr);
    } else {
        static_assert(dependent_false_v<T>, "ERROR - psSimd::load(NEON): Unsupported type");
    }
#elif defined(AVX2_WRAPPERS)
    if constexpr (std::is_integral_v<T>) {
        return _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr));
    } else if constexpr (std::is_same_v<float, T>) {
        return _mm_loadu_ps(ptr);
    } else if constexpr (std::is_same_v<double, T>) {
        return _mm_loadu_pd(ptr);
    } else {
        static_assert(dependent_false_v<T>, "ERROR - psSimd::load(AVX2): Unsupported type");
    }
#else
    return *ptr;
#endif
}

template <typename T>
FORCE_INLINE void store(T* ptr, typename VectorType<T>::V val) {
#if defined(NEON_WRAPPERS)
    if constexpr (std::is_same_v<uint8_t, T>) {
        vst1q_u8(ptr, val);
    } else if constexpr (std::is_same_v<int8_t, T>) {
        vst1q_s8(ptr, val);
    } else if constexpr (std::is_same_v<uint16_t, T>) {
        vst1q_u16(ptr, val);
    } else if constexpr (std::is_same_v<int16_t, T>) {
        vst1q_s16(ptr, val);
    } else if constexpr (std::is_same_v<uint32_t, T>) {
        vst1q_u32(ptr, val);
    } else if constexpr (std::is_same_v<int32_t, T>) {
        vst1q_s32(ptr, val);
    } else if constexpr (std::is_same_v<uint64_t, T>) {
        vst1q_u64(ptr, val);
    } else if constexpr (std::is_same_v<int64_t, T>) {
        vst1q_s64(ptr, val);
    } else if constexpr (std::is_same_v<float, T>) {
        vst1q_f32(ptr, val);
    } else if constexpr (std::is_same_v<double, T>) {
        vst1q_f64(ptr, val);
    } else {
        static_assert(dependent_false_v<T>, "ERROR - psSimd::store(NEON): Unsupported type");
    }
#elif defined(AVX2_WRAPPERS)
    if constexpr (std::is_integral_v<T>) {
        _mm_storeu_si128(reinterpret_cast<__m128i*>(ptr), val);
    } else if constexpr (std::is_same_v<float, T>) {
        _mm_storeu_ps(ptr, val);
    } else if constexpr (std::is_same_v<double, T>) {
        _mm_storeu_pd(ptr, val);
    } else {
        static_assert(dependent_false_v<T>, "ERROR - psSimd::store(AVX2): Unsupported type");
    }
#else
    *ptr = val;
#endif
}

template <int Count, typename T>
FORCE_INLINE typename VectorType<T>::V loadPartial(const T* ptr) {
    static_assert(Count >= 0 && Count <= LaneCount_v<T>, "ERROR - psSimd::loadPartial: Count out of range");

    if constexpr (Count == 0) {
#if defined(NEON_WRAPPERS)
        if constexpr (std::is_same_v<float, T>) return vdupq_n_f32(0.0f);
        else if constexpr (std::is_same_v<double, T>) return vdupq_n_f64(0.0);
        else if constexpr (std::is_same_v<uint8_t, T>) return vdupq_n_u8(0);
        else if constexpr (std::is_same_v<int8_t, T>) return vdupq_n_s8(0);
        else if constexpr (std::is_same_v<uint16_t, T>) return vdupq_n_u16(0);
        else if constexpr (std::is_same_v<int16_t, T>) return vdupq_n_s16(0);
        else if constexpr (std::is_same_v<uint32_t, T>) return vdupq_n_u32(0);
        else if constexpr (std::is_same_v<int32_t, T>) return vdupq_n_s32(0);
        else if constexpr (std::is_same_v<uint64_t, T>) return vdupq_n_u64(0);
        else if constexpr (std::is_same_v<int64_t, T>) return vdupq_n_s64(0);
#elif defined(AVX2_WRAPPERS)
        if constexpr (std::is_integral_v<T>) return _mm_setzero_si128();
        else if constexpr (std::is_same_v<float, T>) return _mm_setzero_ps();
        else if constexpr (std::is_same_v<double, T>) return _mm_setzero_pd();
#else
        return T(0);
#endif
    } else if constexpr (Count == LaneCount_v<T>) {
        return load<T>(ptr);
    } else {
#if defined(NEON_WRAPPERS)
        typename VectorType<T>::V res;
        if constexpr (std::is_same_v<float, T>) res = vdupq_n_f32(0.0f);
        else if constexpr (std::is_same_v<double, T>) res = vdupq_n_f64(0.0);
        else if constexpr (std::is_same_v<uint8_t, T>) res = vdupq_n_u8(0);
        else if constexpr (std::is_same_v<int8_t, T>) res = vdupq_n_s8(0);
        else if constexpr (std::is_same_v<uint16_t, T>) res = vdupq_n_u16(0);
        else if constexpr (std::is_same_v<int16_t, T>) res = vdupq_n_s16(0);
        else if constexpr (std::is_same_v<uint32_t, T>) res = vdupq_n_u32(0);
        else if constexpr (std::is_same_v<int32_t, T>) res = vdupq_n_s32(0);
        else if constexpr (std::is_same_v<uint64_t, T>) res = vdupq_n_u64(0);
        else if constexpr (std::is_same_v<int64_t, T>) res = vdupq_n_s64(0);

        if constexpr (std::is_same_v<float, T>) {
            if constexpr (Count >= 1) res = vsetq_lane_f32(ptr[0], res, 0);
            if constexpr (Count >= 2) res = vsetq_lane_f32(ptr[1], res, 1);
            if constexpr (Count >= 3) res = vsetq_lane_f32(ptr[2], res, 2);
        } else if constexpr (std::is_same_v<double, T>) {
            if constexpr (Count >= 1) res = vsetq_lane_f64(ptr[0], res, 0);
        } else if constexpr (sizeof(T) == 4) {
            if constexpr (Count >= 1) res = vsetq_lane_u32(static_cast<uint32_t>(ptr[0]), res, 0);
            if constexpr (Count >= 2) res = vsetq_lane_u32(static_cast<uint32_t>(ptr[1]), res, 1);
            if constexpr (Count >= 3) res = vsetq_lane_u32(static_cast<uint32_t>(ptr[2]), res, 2);
        } else if constexpr (sizeof(T) == 8) {
            if constexpr (Count >= 1) res = vsetq_lane_u64(static_cast<uint64_t>(ptr[0]), res, 0);
        } else {
            alignas(16) T buffer[LaneCount_v<T>] = {};
            std::memcpy(buffer, ptr, Count * sizeof(T));
            return load<T>(buffer);
        }
        return res;
#elif defined(AVX2_WRAPPERS)
        if constexpr (std::is_same_v<float, T>) {
            __m128i mask = _mm_set_epi32(
                (Count > 3) ? -1 : 0,
                (Count > 2) ? -1 : 0,
                (Count > 1) ? -1 : 0,
                (Count > 0) ? -1 : 0
            );
            return _mm_maskload_ps(ptr, mask);
        } else if constexpr (std::is_same_v<double, T>) {
            __m128i mask = _mm_set_epi64x(
                (Count > 1) ? -1LL : 0LL,
                (Count > 0) ? -1LL : 0LL
            );
            return _mm_maskload_pd(ptr, mask);
        } else if constexpr (sizeof(T) == 4) {
            __m128i mask = _mm_set_epi32(
                (Count > 3) ? -1 : 0,
                (Count > 2) ? -1 : 0,
                (Count > 1) ? -1 : 0,
                (Count > 0) ? -1 : 0
            );
            return _mm_maskload_epi32(reinterpret_cast<const int*>(ptr), mask);
        } else if constexpr (sizeof(T) == 8) {
            __m128i mask = _mm_set_epi64x(
                (Count > 1) ? -1LL : 0LL,
                (Count > 0) ? -1LL : 0LL
            );
            return _mm_maskload_epi64(reinterpret_cast<const long long*>(ptr), mask);
        } else {
            alignas(16) T buffer[LaneCount_v<T>] = {};
            std::memcpy(buffer, ptr, Count * sizeof(T));
            return load<T>(buffer);
        }
#else
        if constexpr (Count > 0) {
            return *ptr;
        } else {
            return T(0);
        }
#endif
    }
}

template <int Count, typename T>
FORCE_INLINE void storePartial(T* ptr, typename VectorType<T>::V val) {
    static_assert(Count >= 0 && Count <= LaneCount_v<T>, "ERROR - psSimd::storePartial: Count out of range");

    if constexpr (Count == 0) {
        return;
    } else if constexpr (Count == LaneCount_v<T>) {
        store<T>(ptr, val);
    } else {
#if defined(NEON_WRAPPERS)
        if constexpr (std::is_same_v<float, T>) {
            if constexpr (Count == 1) {
                ptr[0] = vgetq_lane_f32(val, 0);
            } else if constexpr (Count == 2) {
                vst1_f32(ptr, vget_low_f32(val));
            } else if constexpr (Count == 3) {
                vst1_f32(ptr, vget_low_f32(val));
                ptr[2] = vgetq_lane_f32(val, 2);
            }
        } else if constexpr (std::is_same_v<double, T>) {
            if constexpr (Count == 1) {
                ptr[0] = vgetq_lane_f64(val, 0);
            }
        } else if constexpr (sizeof(T) == 4) {
            if constexpr (Count == 1) {
                ptr[0] = static_cast<T>(vgetq_lane_u32(vreinterpretq_u32_s32(val), 0));
            } else if constexpr (Count == 2) {
                vst1_u32(reinterpret_cast<uint32_t*>(ptr), vget_low_u32(vreinterpretq_u32_s32(val)));
            } else if constexpr (Count == 3) {
                vst1_u32(reinterpret_cast<uint32_t*>(ptr), vget_low_u32(vreinterpretq_u32_s32(val)));
                ptr[2] = static_cast<T>(vgetq_lane_u32(vreinterpretq_u32_s32(val), 2));
            }
        } else if constexpr (sizeof(T) == 8) {
            if constexpr (Count == 1) {
                ptr[0] = static_cast<T>(vgetq_lane_u64(vreinterpretq_u64_s64(val), 0));
            }
        } else {
            alignas(16) T buffer[LaneCount_v<T>];
            store<T>(buffer, val);
            std::memcpy(ptr, buffer, Count * sizeof(T));
        }
#elif defined(AVX2_WRAPPERS)
        if constexpr (std::is_same_v<float, T>) {
            __m128i mask = _mm_set_epi32(
                (Count > 3) ? -1 : 0,
                (Count > 2) ? -1 : 0,
                (Count > 1) ? -1 : 0,
                (Count > 0) ? -1 : 0
            );
            _mm_maskstore_ps(ptr, mask, val);
        } else if constexpr (std::is_same_v<double, T>) {
            __m128i mask = _mm_set_epi64x(
                (Count > 1) ? -1LL : 0LL,
                (Count > 0) ? -1LL : 0LL
            );
            _mm_maskstore_pd(ptr, mask, val);
        } else if constexpr (sizeof(T) == 4) {
            __m128i mask = _mm_set_epi32(
                (Count > 3) ? -1 : 0,
                (Count > 2) ? -1 : 0,
                (Count > 1) ? -1 : 0,
                (Count > 0) ? -1 : 0
            );
            _mm_maskstore_epi32(reinterpret_cast<int*>(ptr), mask, val);
        } else if constexpr (sizeof(T) == 8) {
            __m128i mask = _mm_set_epi64x(
                (Count > 1) ? -1LL : 0LL,
                (Count > 0) ? -1LL : 0LL
            );
            _mm_maskstore_epi64(reinterpret_cast<long long*>(ptr), mask, val);
        } else {
            alignas(16) T buffer[LaneCount_v<T>];
            store<T>(buffer, val);
            std::memcpy(ptr, buffer, Count * sizeof(T));
        }
#else
        if constexpr (Count > 0) {
            *ptr = val;
        }
#endif
    }
}

}