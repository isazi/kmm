cmake_minimum_required(VERSION 3.10)

set(PROJECT_NAME "kmm")
project(${PROJECT_NAME} LANGUAGES CXX VERSION 0.1)

if(${USE_CUDA})
  enable_language(CUDA)
  add_definitions(-DUSE_CUDA=${USE_CUDA})
  set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
endif()

if(DEFINED CMAKE_CUDA_COMPILER)
  file(GLOB_RECURSE sources
      "${PROJECT_SOURCE_DIR}/src/*.cpp"
      "${PROJECT_SOURCE_DIR}/src/*/*.cpp"
      "${PROJECT_SOURCE_DIR}/include/kmm/*.h"
      "${PROJECT_SOURCE_DIR}/include/kmm/*/*.h"
      "${PROJECT_SOURCE_DIR}/src/*.cu"
      "${PROJECT_SOURCE_DIR}/src/*/*.cu")
#    set_source_files_properties(${sources} PROPERTIES LANGUAGE CXX)
else()
  file(GLOB_RECURSE sources
      "${PROJECT_SOURCE_DIR}/src/*.cpp"
      "${PROJECT_SOURCE_DIR}/include/kmm/*.h")
endif()

file(GLOB_RECURSE includes "${PROJECT_SOURCE_DIR}/include/*.hpp" "${PROJECT_SOURCE_DIR}/include/*.cuh")
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


# The CUDA compiler struggles with the "Werror" options, so we need to explicitly forward it to the host compiler
# using `-Xcompiler=-Werror` if we are compiling CUDA code.
set(CXXFLAGS
  #$<$<COMPILE_LANGUAGE:CUDA>:-forward-unknown-to-host-compiler>
  #$<$<COMPILE_LANGUAGE:CUDA>:--generate-line-info>
  $<$<COMPILE_LANGUAGE:CUDA>:--forward-unknown-to-host-compiler>
  -Wall -Wextra -Wconversion -Wno-unused-parameter
  #$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Werror>
  #$<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
  #$<$<COMPILE_LANGUAGE:CUDA>:-Xptxas="-v">
)
target_compile_options(${PROJECT_NAME} PRIVATE ${CXXFLAGS})
#target_compile_options(${PROJECT_NAME} PRIVATE "-fsanitize=address")

# Enable C++17 support
set(CMAKE_CXX_STANDARD 17)
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)

# Enable PIC
set_property(TARGET ${PROJECT_NAME} PROPERTY POSITION_INDEPENDENT_CODE ON)

set(PROJECT_CLANG_TIDY clang-tidy -checks=-*,readability-*,bugprone-*,-readability-magic-numbers,-readability-use-anyofallof,-readability-else-after-return)
set_target_properties(${PROJECT_NAME} PROPERTIES CXX_CLANG_TIDY "${PROJECT_CLANG_TIDY}")

if(DEFINED CMAKE_CUDA_COMPILER)
  find_package(CUDAToolkit REQUIRED)
  target_link_libraries(${PROJECT_NAME} PUBLIC CUDA::cudart_static)
  target_link_libraries(${PROJECT_NAME} PUBLIC CUDA::cuda_driver)
  target_link_libraries(${PROJECT_NAME} PUBLIC CUDA::nvrtc)
  set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES "50")
endif()

# Unit testing
enable_testing()
find_package(GTest)

if(DEFINED CMAKE_CUDA_COMPILER)
    #set_source_files_properties(test/kmmTest.cpp PROPERTIES LANGUAGE CUDA)
endif()
add_executable(
  kmmTestCxx
  test/test_memory_manager.cpp
  test/test_result.cpp
  test/test_memory_pool.cpp
)
target_include_directories(
    kmmTestCxx
    PRIVATE
    ${GTEST_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/include
)
target_link_libraries(
  kmmTestCxx
  PRIVATE
  kmm
  GTest::gtest
  GTest::gtest_main
)
if(DEFINED CMAKE_CUDA_COMPILER)
endif()

include(GoogleTest)
gtest_discover_tests(kmmTestCxx)

# Examples
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
