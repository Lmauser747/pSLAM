cmake_minimum_required(VERSION 3.24)
project(PoolSLAM VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(BUILD_TESTING "Build unit and system test suites" ON)
option(BUILD_BENCHMARKS "Build micro-benchmarks" OFF)
option(BUILD_PYTHON_BINDINGS "Build Pybind11 Python wrappers" OFF)

# Force colored diagnostics under Ninja build buffering
if(CMAKE_GENERATOR STREQUAL "Ninja")
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        add_compile_options(-fdiagnostics-color=always)
    endif()
endif()

if(MSVC)
    add_compile_options(/W4 /permissive-)
else()
    add_compile_options(-Wall -Wextra -Wpedantic -Wconversion -Wshadow)
endif()

find_package(Threads REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(GTSAM REQUIRED)
find_package(simde REQUIRED)
find_package(yaml-cpp REQUIRED)

add_subdirectory(src)

if(BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()

if(BUILD_BENCHMARKS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks")
    add_subdirectory(benchmarks)
endif()

if(BUILD_PYTHON_BINDINGS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/bindings/python")
    add_subdirectory(bindings/python)
endif()
