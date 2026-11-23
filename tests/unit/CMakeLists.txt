# HAL Unit tests
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



# Frontend Unit Tests
add_executable(unit_tests_frontend
    TestTrackHfFeatures.cpp
    TestReplenishHfFeatures.cpp
    TestP3p.cpp
    TestLma.cpp
    TestLfTracker.cpp
)
target_link_libraries(unit_tests_frontend PRIVATE
    poolslam_frontend
    poolslam_descriptor
    GTest::gtest_main
    Eigen3::Eigen
    gtsam
)



# Backend Unit Tests
# add_executable(unit_tests_backend
#     TestBookkeeper.cpp
# )
# target_link_libraries(unit_tests_backend PRIVATE
#     poolslam_backend
#     GTest::gtest_main
#     Eigen3::Eigen
#     gtsam
# )


# Descriptor Unit Tests
add_executable(unit_tests_descriptor
    TestDescriptor.cpp
)
target_link_libraries(unit_tests_descriptor PRIVATE
    poolslam_descriptor
    GTest::gtest_main
    Eigen3::Eigen
)

gtest_discover_tests(unit_tests_hal TEST_PREFIX "unit_tests_.")
gtest_discover_tests(unit_tests_frontend TEST_PREFIX "unit_tests_.")
gtest_discover_tests(unit_tests_descriptor TEST_PREFIX "unit_tests_.")
# gtest_discover_tests(unit_tests_backend TEST_PREFIX "unit_tests_.")
