find_package(GTest REQUIRED)

add_executable(system_tests
    test_hal_concurrency.cpp
    test_dynamic_routing.cpp
)

target_link_libraries(system_tests PRIVATE
    pool_slam_core
    GTest::gtest_main
)

add_test(NAME SystemTests COMMAND system_tests)