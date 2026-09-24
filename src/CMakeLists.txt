add_library(pool_slam_core
    hal/DepthStream.cpp
    hal/ImuStream.cpp
)

target_include_directories(pool_slam_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include/pool_slam>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(pool_slam_core PUBLIC
    Eigen3::Eigen
    # gtsam
)