find_package(GTest)
include(GoogleTest)
enable_testing()
add_executable(
      kmmTest
      ${PROJECT_SOURCE_DIR}/test/test_integer_fun.cpp
      ${PROJECT_SOURCE_DIR}/test/test_checked_math.cpp
      ${PROJECT_SOURCE_DIR}/test/test_small_vector.cpp
      ${PROJECT_SOURCE_DIR}/test/test_geom.cpp
      ${PROJECT_SOURCE_DIR}/test/test_view.cpp
)

target_compile_features(kmmTest PRIVATE cxx_std_17)

target_include_directories(
      kmmTest
      PRIVATE
      ${GTEST_INCLUDE_DIRS}
      ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(
      kmmTest
      PRIVATE
      GTest::gtest_main
      kmm
)