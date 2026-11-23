add_executable(system_tests_synchronizer
    TestSynchronizer.cpp
)

target_link_libraries(system_tests_synchronizer PRIVATE
    poolslam_hal
    GTest::gtest_main
    Eigen3::Eigen
    Threads::Threads
)

add_executable(system_tests_frontend
    TestFrontend.cpp
)

target_link_libraries(system_tests_frontend PRIVATE
    poolslam_frontend
    poolslam_hal
    poolslam_descriptor
    GTest::gtest_main
    Eigen3::Eigen
    gtsam
    Threads::Threads
)

gtest_discover_tests(system_tests_synchronizer TEST_PREFIX "system_tests_.")
gtest_discover_tests(system_tests_frontend TEST_PREFIX "system_tests_.")

add_executable(system_tests_backend
    TestBackend.cpp
)

target_link_libraries(system_tests_backend PRIVATE
    poolslam_backend
    poolslam_descriptor
    GTest::gtest_main
    Eigen3::Eigen
    gtsam
    Threads::Threads
)

gtest_discover_tests(system_tests_backend TEST_PREFIX "system_tests_.")
