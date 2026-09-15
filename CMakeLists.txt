cmake_minimum_required(VERSION 3.14)
project(pool_slam VERSION 1.0.0 LANGUAGES CXX)

if(NOT CMAKE_GENERATOR MATCHES "Ninja")
    message(FATAL_ERROR "The required build backend is Ninja. Generate using -G Ninja.")
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "RelWithDebInfo" "Release")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-mfpu=neon" HAS_ARM_NEON)
if(HAS_ARM_NEON)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfpu=neon")
endif()

find_package(Eigen3 REQUIRED)
#find_package(GTSAM REQUIRED)
find_package(pybind11 REQUIRED)

add_subdirectory(src)
# add_subdirectory(python)
# add_subdirectory(benchmarks)

# enable_testing()
# add_subdirectory(tests/unit)
# add_subdirectory(tests/system)