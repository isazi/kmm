file(GLOB_RECURSE sources
  "${PROJECT_SOURCE_DIR}/test/*.cpp"
  "${PROJECT_SOURCE_DIR}/test/*.cu")

add_executable(kmmTest ${sources})

target_compile_features(kmmTest PRIVATE cxx_std_17)
target_link_libraries(kmmTest PRIVATE kmm)
target_link_libraries(kmmTest PRIVATE Catch2::Catch2WithMain)

include(Catch)
catch_discover_tests(kmmTest)