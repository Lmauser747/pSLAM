find_package(benchmark REQUIRED)

add_executable(slam_benchmarks benchmarks.cpp)

target_link_libraries(slam_benchmarks PRIVATE
    pool_slam_core
    benchmark::benchmark
)