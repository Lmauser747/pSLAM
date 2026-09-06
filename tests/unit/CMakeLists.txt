find_package(GTest REQUIRED)

add_executable(unit_tests
    test_eigen_spatial.cpp
    test_synchronizer_interpolation.cpp
)

target_link_libraries(unit_tests PRIVATE
    pool_slam_core
    GTest::gtest_main
)

add_test(NAME UnitTests COMMAND unit_tests)