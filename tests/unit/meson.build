unit_tests = [
  'test_tracking',
  'test_types'
]

foreach t : unit_tests
  test_exe = executable(
    t,
    files(t + '.cpp'),
    dependencies: [pool_slam_dep, gtest_dep]
  )
  test(t, test_exe, suite: 'unit')
endforeach