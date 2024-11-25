cmake_minimum_required(VERSION 3.10)

set(PROJECT_NAME "kmm")
project(${PROJECT_NAME} LANGUAGES CXX VERSION 0.1)

# User options (features)
option(KMM_USE_CUDA "Build the CUDA backend" OFF)
option(KMM_USE_HIP "Build the HIP backend" OFF)
if(KMM_USE_CUDA AND KMM_USE_HIP)
  message(FATAL_ERROR "CUDA and HIP backend are mutually exclusive.")
endif()
option(KMM_STATIC "Build a static library" OFF)

# User options (development)
option(KMM_ENABLE_LINTER "Enable clang-tidy linter" OFF)
option(KMM_BUILD_TESTS "Build tests" OFF)
option(KMM_BUILD_EXAMPLES "Build examples" OFF)

# Enable C++17 support
set(CXX_STANDARD 17)
set(CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(KMM_USE_CUDA)
  enable_language(CUDA)
  SET(CUDA_SEPARABLE_COMPILATION ON)
  set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
endif()

file(GLOB_RECURSE sources
  "${PROJECT_SOURCE_DIR}/src/*.cpp"
  "${PROJECT_SOURCE_DIR}/src/*.cu")

if(KMM_STATIC)
  add_library(${PROJECT_NAME} STATIC ${sources})
else()
  add_library(${PROJECT_NAME} SHARED ${sources})
endif()

target_include_directories(${PROJECT_NAME} PUBLIC "${PROJECT_SOURCE_DIR}/include")

# CXX flags
set(CXXFLAGS
        $<$<COMPILE_LANGUAGE:CUDA>:-forward-unknown-to-host-compiler>
        -Wall -Wextra -Wconversion -Wno-unused-parameter
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Werror>)
target_compile_options(${PROJECT_NAME} PUBLIC ${CXXFLAGS})

# Enable PIC
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set_property(TARGET ${PROJECT_NAME} PROPERTY POSITION_INDEPENDENT_CODE ON)

set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
add_subdirectory(external/fmt)
target_link_libraries(${PROJECT_NAME} PUBLIC fmt)

set(SPDLOG_FMT_EXTERNAL ON)
set(SPDLOG_BUILD_PIC ON)
add_subdirectory(external/spdlog)
target_link_libraries(${PROJECT_NAME} PUBLIC spdlog)

# Install
set_target_properties(
  ${PROJECT_NAME}
  PROPERTIES
  VERSION ${PROJECT_VERSION}
  SOVERSION 1
)

install(
  TARGETS ${PROJECT_NAME}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(
  DIRECTORY include/kmm
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  FILES_MATCHING PATTERN "*.hpp"
)

if(KMM_ENABLE_LINTER)
  set(PROJECT_CLANG_TIDY clang-tidy)
  set_target_properties(${PROJECT_NAME} PROPERTIES CXX_CLANG_TIDY "${PROJECT_CLANG_TIDY}")
endif()

if(DEFINED CMAKE_CUDA_COMPILER)
  find_package(CUDAToolkit REQUIRED)
  target_link_libraries(${PROJECT_NAME} PUBLIC CUDA::cudart_static)
  target_link_libraries(${PROJECT_NAME} PUBLIC CUDA::cuda_driver)
  target_link_libraries(${PROJECT_NAME} PUBLIC CUDA::cublas)
  target_link_libraries(${PROJECT_NAME} PUBLIC CUDA::nvrtc)
  set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES "80")
  set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

  # we must expose the USE_CUDA flag since it is used in the header files
  target_compile_definitions(${PROJECT_NAME} PUBLIC -DKMM_USE_CUDA=1)
endif()

# Unit testing
if(KMM_BUILD_TESTS)
  add_subdirectory(test)
endif()

# Compile examples
if(KMM_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()
