bench_sources = files(
  'bench_tracking.cpp',
  'bench_mapping.cpp'
)

bench_exe = executable(
  'pool_slam_benchmarks',
  bench_sources,
  dependencies: [pool_slam_dep, benchmark_dep]
)

benchmark('slam_performance', bench_exe)