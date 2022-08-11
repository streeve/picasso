add_executable(InterpolationPerformance Picasso_InterpolationPerformance.cpp)
target_link_libraries(InterpolationPerformance picasso ${gtest_target})

if(Picasso_ENABLE_TESTING)
  add_test(NAME Version_test COMMAND Version_test ${gtest_args})
endif()
