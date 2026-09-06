add_library(pool_slam_core
    hal.cpp
    synchronizer.cpp
    frontend.cpp
    backend.cpp
)

target_include_directories(pool_slam_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(pool_slam_core PUBLIC
    Eigen3::Eigen
    gtsam
)