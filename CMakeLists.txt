cmake_minimum_required(VERSION 3.10)
include(CheckLanguage)

set(PROJECT_NAME "kmm")
project(${PROJECT_NAME} LANGUAGES CXX VERSION 0.1)

if(${USE_CUDA})
  check_language(CUDA)
  add_definitions(-DUSE_CUDA=${USE_CUDA})
endif()

if(DEFINED CMAKE_CUDA_COMPILER)
  file(GLOB_RECURSE sources "${PROJECT_SOURCE_DIR}/src/*.cpp" "${PROJECT_SOURCE_DIR}/src/*.cu")
else()
  file(GLOB_RECURSE sources "${PROJECT_SOURCE_DIR}/src/*.cpp")
endif()
file(GLOB_RECURSE includes "${PROJECT_SOURCE_DIR}/include/*.hpp")
add_library(${PROJECT_NAME} SHARED ${sources} ${includes})

target_include_directories(${PROJECT_NAME} PUBLIC "${PROJECT_SOURCE_DIR}/include")

# Install
if(DEFINED CMAKE_CUDA_COMPILER)
  set_target_properties(
    ${PROJECT_NAME}
    PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
    PUBLIC_HEADER ${includes}
    CUDA_SEPARABLE_COMPILATION ON
  )
else()
  set_target_properties(
    ${PROJECT_NAME}
    PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
    PUBLIC_HEADER ${includes}
  )
endif()

install(TARGETS ${PROJECT_NAME}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# The CUDA compiler struggles with the "Werror" options, so we need to explicitly forward it to the host compiler
# using `-Xcompiler=-Werror` if we are compiling CUDA code.
set(CXXFLAGS
        $<$<COMPILE_LANGUAGE:CUDA>:-forward-unknown-to-host-compiler>
        $<$<COMPILE_LANGUAGE:CUDA>:--generate-line-info>
        -Wall -Wextra -Wconversion -Wno-unused-parameter
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Werror>
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas="-v">
)
target_compile_options(${PROJECT_NAME} PRIVATE ${CXXFLAGS})

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
endif()

# Unit testing
enable_testing()
find_package(GTest)

add_executable(
  kmmTestCxx
  test/kmmTest.cpp
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
  add_executable(
    kmmTestCu
    test/kmmTest.cu
  )
  target_include_directories(
      kmmTestCu
      PRIVATE
      ${GTEST_INCLUDE_DIRS}
      ${PROJECT_SOURCE_DIR}/include
  )
  target_link_libraries(
    kmmTestCu
    PRIVATE
    kmm
    GTest::gtest
    GTest::gtest_main
  )
endif()

include(GoogleTest)
gtest_discover_tests(kmmTestCxx)
if(DEFINED CMAKE_CUDA_COMPILER)
  gtest_discover_tests(kmmTestCu)
endif()

# Examples
if(DEFINED CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
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