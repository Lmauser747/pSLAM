add_executable(system_tests
    test_synchronizer.cpp
    test_pipeline_e2e.cpp
)

target_link_libraries(system_tests PRIVATE
    poolslam_backend
    poolslam_frontend
    poolslam_hal
    GTest::gtest_main
    Eigen3::Eigen
    Threads::Threads
)

gtest_discover_tests(system_tests)