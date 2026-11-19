find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        URL https://github.com/pybind/pybind11/archive/refs/tags/v2.13.6.zip
    )
    FetchContent_MakeAvailable(pybind11)
endif()

pybind11_add_module(pool_slam
    bindings.cpp
)

target_link_libraries(pool_slam PRIVATE
    poolslam_core
    poolslam_descriptor
    Eigen3::Eigen
)