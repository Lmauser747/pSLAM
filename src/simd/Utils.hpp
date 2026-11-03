#pragma once

#include <cstddef>
#include <utility>
#include <type_traits>


// idgaf about MSVC
#ifndef FORCE_INLINE
#if defined(__GNUC__) || defined(__clang__)
  #define FORCE_INLINE inline __attribute__((always_inline))
#else
  #define FORCE_INLINE inline
#endif
#endif

namespace PoolSLAM{

namespace simd{

template <size_t... Is, typename F>
inline constexpr void static_for(std::index_sequence<Is...>, F&& func) {
    (func(std::integral_constant<size_t, Is>{}), ...);
}

template <size_t N, typename F>
inline constexpr void static_for(F&& func) {
    static_for(std::make_index_sequence<N>{}, std::forward<F>(func));
}

}

}