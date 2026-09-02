system_test_exe = executable(
  'test_full_pipeline',
  files('test_full_pipeline.cpp'),
  dependencies: [pool_slam_dep, gtest_dep]
)

test(
  'system_full_pipeline',
  system_test_exe,
  suite: 'system',
  timeout: 120
)