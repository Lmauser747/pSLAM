add_executable(system_tests
    TestSynchronizer.cpp
    TestFrontend.cpp
)

target_link_libraries(system_tests PRIVATE
    # poolslam_backend
    poolslam_frontend
    poolslam_hal
    poolslam_descriptor
    GTest::gtest_main
    Eigen3::Eigen
    Threads::Threads
    gtsam
)

gtest_discover_tests(system_tests TEST_PREFIX "system_tests_.")