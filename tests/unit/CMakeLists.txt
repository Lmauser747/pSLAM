# --- HAL Unit Tests ---
add_executable(unit_tests_hal
    TestDepthStream.cpp
    TestImuStream.cpp
    TestImageStream.cpp
)
target_link_libraries(unit_tests_hal PRIVATE
    poolslam_hal
    GTest::gtest_main
    Eigen3::Eigen
    Threads::Threads
)
gtest_discover_tests(unit_tests_hal)

# --- Frontend Unit Tests ---[cite: 8]
add_executable(unit_tests_frontend
    test_feature_tracker.cpp
)
target_link_libraries(unit_tests_frontend PRIVATE
    poolslam_frontend
    GTest::gtest_main
    Eigen3::Eigen
)
gtest_discover_tests(unit_tests_frontend)

# --- Backend Unit Tests ---[cite: 8]
add_executable(unit_tests_backend
    test_factor_graph.cpp
)
target_link_libraries(unit_tests_backend PRIVATE
    poolslam_backend
    GTest::gtest_main
    Eigen3::Eigen
)
gtest_discover_tests(unit_tests_backend)