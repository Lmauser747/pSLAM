enable_language(C)

# Common include directory for all internal src/ components
set(INTERNAL_INCLUDE_DIR $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)
set(PUBLIC_INCLUDE_DIR $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>)

# --- 1. SIMD Target (Header-only / Interface) ---
# add_library(poolslam_simd INTERFACE)
# target_include_directories(poolslam_simd INTERFACE
#     ${INTERNAL_INCLUDE_DIR}
#     ${PUBLIC_INCLUDE_DIR}
# )

# if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|armv8)")
#     target_compile_definitions(poolslam_simd INTERFACE POOLSLAM_USE_NEON)
#     if(NOT MSVC)
#         target_compile_options(poolslam_simd INTERFACE -march=armv8-a+simd)
#     endif()
# endif()

# --- 2. HAL Target ---[cite: 1, 8]
add_library(poolslam_hal STATIC
    hal/ImageStream.cpp
    hal/ImuStream.cpp
    hal/DepthStream.cpp
    hal/Synchronizer.cpp
)

target_include_directories(poolslam_hal
    PUBLIC
        ${PUBLIC_INCLUDE_DIR}
        ${INTERNAL_INCLUDE_DIR}
)
target_link_libraries(poolslam_hal
    PUBLIC
        Eigen3::Eigen
        Threads::Threads
    PRIVATE
        # poolslam_simd
        gtsam
        simde::simde
)



# --- 3. Frontend Target (Tracking / Odometry) ---[cite: 8]
# add_library(poolslam_frontend STATIC
#     frontend/feature_tracker.cpp
# )
# target_include_directories(poolslam_frontend
#     PUBLIC
#         ${PUBLIC_INCLUDE_DIR}
#         ${INTERNAL_INCLUDE_DIR}
# )
# target_link_libraries(poolslam_frontend
#     PUBLIC
#         poolslam_hal
#         poolslam_simd
#         Eigen3::Eigen
# )

# --- 4. Backend Target (Factor Graph / Optimization) ---[cite: 8]
# find_package(GTSAM REQUIRED)
# add_library(poolslam_backend STATIC
#     backend/factor_graph.cpp
# )
# target_include_directories(poolslam_backend
#     PUBLIC
#         ${PUBLIC_INCLUDE_DIR}
#         ${INTERNAL_INCLUDE_DIR}
# )
# target_link_libraries(poolslam_backend
#     PUBLIC
#         poolslam_frontend
#         Eigen3::Eigen
#         # GTSAM::gtsam
# )