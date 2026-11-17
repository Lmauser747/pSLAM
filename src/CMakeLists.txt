enable_language(C)

# Common include directory for all internal src/ components
set(INTERNAL_INCLUDE_DIR $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)
set(PUBLIC_INCLUDE_DIR $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>)

# HAL
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
)

# Frontend
add_library(poolslam_frontend STATIC
    frontend/ReplenishHfFeatures.cpp
    frontend/TrackHfFeatures.cpp
    frontend/LfTracker.cpp
    frontend/P3p.cpp
    frontend/Lma.cpp
    frontend/Frontend.cpp
)
target_compile_definitions(poolslam_frontend
    PRIVATE
        POOL_SLAM_SIMD
)
target_include_directories(poolslam_frontend
    PUBLIC
        ${PUBLIC_INCLUDE_DIR}
        ${INTERNAL_INCLUDE_DIR}
)
target_link_libraries(poolslam_frontend
    PUBLIC
        poolslam_hal
        Eigen3::Eigen
    PRIVATE
        gtsam
        simde::simde
)

# Backend
find_package(GTSAM REQUIRED)
add_library(poolslam_backend STATIC
    backend/Bookkeeper.cpp
)
target_include_directories(poolslam_backend
    PUBLIC
        ${PUBLIC_INCLUDE_DIR}
        ${INTERNAL_INCLUDE_DIR}
)
target_link_libraries(poolslam_backend
    PUBLIC
        poolslam_frontend
        Eigen3::Eigen
    PRIVATE
        gtsam
)


add_library(poolslam_descriptor STATIC
    descriptor/BinaryBow.cpp
)

target_compile_definitions(poolslam_descriptor
    PRIVATE
        POOL_SLAM_SIMD
)
target_include_directories(poolslam_descriptor
    PUBLIC
        ${PUBLIC_INCLUDE_DIR}
        ${INTERNAL_INCLUDE_DIR}
)
target_link_libraries(poolslam_descriptor
    PUBLIC
        Eigen3::Eigen
    PRIVATE
        simde::simde
)