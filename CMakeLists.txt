cmake_minimum_required(VERSION 3.10)

set(PROJECT_NAME "kmm")
project(${PROJECT_NAME} LANGUAGES CXX VERSION 0.1)

# User options
option(KMM_USE_CUDA "Build the CUDA backend" OFF)
option(KMM_BUILD_TESTS "Build tests" OFF)
option(KMM_BUILD_EXAMPLES "Build examples" OFF)

if(KMM_USE_CUDA)
  enable_language(CUDA)
  add_definitions(-DKMM_USE_CUDA=1)
  set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
endif()

file(GLOB_RECURSE sources
  "${PROJECT_SOURCE_DIR}/src/*.cpp"
  "${PROJECT_SOURCE_DIR}/src/*/*.cpp"
  "${PROJECT_SOURCE_DIR}/include/kmm/*.h"
  "${PROJECT_SOURCE_DIR}/include/kmm/*/*.h")

file(GLOB_RECURSE includes "${PROJECT_SOURCE_DIR}/include/*.hpp")
add_library(${PROJECT_NAME} SHARED ${sources} ${includes})

target_include_directories(${PROJECT_NAME} PUBLIC "${PROJECT_SOURCE_DIR}/include")

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(SPDLOG_FMT_EXTERNAL ON)
set(SPDLOG_BUILD_PIC ON)

add_subdirectory(external/fmt)
add_subdirectory(external/spdlog)
target_link_libraries(${PROJECT_NAME} PUBLIC spdlog fmt)

# Install
set_target_properties(
  ${PROJECT_NAME}
  PROPERTIES
  VERSION ${PROJECT_VERSION}
  SOVERSION 1
  PUBLIC_HEADER "${includes}"
)

install(TARGETS ${PROJECT_NAME}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

target_compile_options(${PROJECT_NAME} PUBLIC ${CXXFLAGS})

# Enable C++17 support
set(CXX_STANDARD 17)
set(CXX_STANDARD_REQUIRED ON)
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)

# Enable PIC
set_property(TARGET ${PROJECT_NAME} PROPERTY POSITION_INDEPENDENT_CODE ON)

set(PROJECT_CLANG_TIDY clang-tidy)
set_target_properties(${PROJECT_NAME} PROPERTIES CXX_CLANG_TIDY "${PROJECT_CLANG_TIDY}")

if(DEFINED CMAKE_CUDA_COMPILER)
  find_package(CUDAToolkit REQUIRED)
  target_link_libraries(${PROJECT_NAME} PUBLIC CUDA::cudart_static)
  target_link_libraries(${PROJECT_NAME} PUBLIC CUDA::cuda_driver)
  target_link_libraries(${PROJECT_NAME} PUBLIC CUDA::cublas)
  target_link_libraries(${PROJECT_NAME} PUBLIC CUDA::nvrtc)
  set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES "80")
endif()

# Unit testing
if(KMM_BUILD_TESTS)
  enable_testing()
  find_package(GTest)

  add_executable(
    kmmTest
    test/test_memory_manager.cpp
    test/test_result.cpp
    test/test_memory_pool.cpp
  )
  target_include_directories(
      kmmTest
      PRIVATE
      ${GTEST_INCLUDE_DIRS}
      ${PROJECT_SOURCE_DIR}/include
  )
  target_link_libraries(
    kmmTest
    PRIVATE
    kmm
    GTest::gtest_main
  )
  target_compile_features(kmmTest PRIVATE cxx_std_17)

  include(GoogleTest)
  gtest_discover_tests(kmmTest)
endif()

# Compile examples
if(KMM_BUILD_EXAMPLES)
  if(DEFINED CMAKE_CUDA_COMPILER)
    add_executable(
      vector_add
      examples/vector_add.cu
    )
    target_include_directories(
        vector_add
        PRIVATE
        ${PROJECT_SOURCE_DIR}/include
    )
    target_link_libraries(
      vector_add
      PRIVATE
      kmm
    )
  endif()
endif()