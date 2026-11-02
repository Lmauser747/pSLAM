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



# --- Frontend Unit Tests ---[cite: 8]
add_executable(unit_tests_frontend
    TestTrackHfFeatures.cpp
    TestReplenishHfFeatures.cpp
)
target_link_libraries(unit_tests_frontend PRIVATE
    poolslam_frontend
    GTest::gtest_main
    Eigen3::Eigen
    gtsam
)
#[[
# --- Backend Unit Tests ---[cite: 8]
add_executable(unit_tests_backend
)
target_link_libraries(unit_tests_backend PRIVATE
    poolslam_backend
    GTest::gtest_main
    Eigen3::Eigen
)
gtest_discover_tests(unit_tests_backend)
]]

gtest_discover_tests(unit_tests_hal TEST_PREFIX "unit_tests_.")
gtest_discover_tests(unit_tests_frontend TEST_PREFIX "unit_tests_.")
# gtest_discover_tests(unit_tests_backend TEST_PREFIX "unit_tests_.")