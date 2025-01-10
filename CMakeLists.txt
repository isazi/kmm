cmake_minimum_required(VERSION 3.10)

# Project setup
set(PROJECT_NAME "kmm")
project(${PROJECT_NAME} LANGUAGES CXX VERSION 0.1)

# User options (features)
option(KMM_STATIC   "Build a static library" OFF)
option(KMM_USE_CUDA "Build the CUDA backend" OFF)
option(KMM_USE_HIP  "Build the HIP backend" OFF)

# User options (development)
option(KMM_ENABLE_LINTER  "Enable clang-tidy linter" OFF)
option(KMM_BUILD_TESTS    "Build tests" OFF)
option(KMM_BUILD_EXAMPLES "Build examples" OFF)

# Check options
if(KMM_USE_CUDA AND KMM_USE_HIP)
  message(FATAL_ERROR "CUDA and HIP backend are mutually exclusive.")
endif()

if(KMM_USE_CUDA)
  enable_language(CUDA)
  SET(CUDA_SEPARABLE_COMPILATION ON)
  set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
endif()

# Enable C++17 support
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

file(GLOB_RECURSE sources
  "${PROJECT_SOURCE_DIR}/src/*.cpp"
  "${PROJECT_SOURCE_DIR}/src/*.cu"
)

# Create library
if(KMM_STATIC)
  add_library(${PROJECT_NAME} STATIC ${sources})
else()
  add_library(${PROJECT_NAME} SHARED ${sources})
endif()

target_include_directories(${PROJECT_NAME} PUBLIC "${PROJECT_SOURCE_DIR}/include")

# CXX flags
target_compile_options(${PROJECT_NAME}
        PUBLIC
        $<$<COMPILE_LANGUAGE:CUDA>:-forward-unknown-to-host-compiler>
        -Wall -Wextra -Wconversion -Wno-unused-parameter
        #$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Werror>
)
target_compile_options(${PROJECT_NAME} PUBLIC ${CXXFLAGS})

# Enable PIC
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set_property(TARGET ${PROJECT_NAME} PROPERTY POSITION_INDEPENDENT_CODE ON)

# Dependencies: fmt
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
add_subdirectory(external/fmt)
target_link_libraries(${PROJECT_NAME} PUBLIC fmt)

# Dependencies: spdlog
set(SPDLOG_FMT_EXTERNAL ON)
set(SPDLOG_BUILD_PIC ON)
add_subdirectory(external/spdlog)
target_link_libraries(${PROJECT_NAME} PUBLIC spdlog)

# Install
include(GNUInstallDirs)
set_target_properties(
  ${PROJECT_NAME}
  PROPERTIES
  VERSION ${PROJECT_VERSION}
  SOVERSION 1
)

install(
  TARGETS ${PROJECT_NAME}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(
  DIRECTORY "${PROJECT_SOURCE_DIR}/include/kmm"
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  FILES_MATCHING PATTERN "*.hpp"
)

if(KMM_ENABLE_LINTER)
  set(PROJECT_CLANG_TIDY clang-tidy)
  set_target_properties(${PROJECT_NAME} PROPERTIES CXX_CLANG_TIDY "${PROJECT_CLANG_TIDY}")
endif()

if(KMM_USE_CUDA)
  find_package(CUDAToolkit REQUIRED)
  target_link_libraries(${PROJECT_NAME}
          PUBLIC
          CUDA::cudart_static
          CUDA::cuda_driver
          CUDA::cublas
          CUDA::nvrtc
  )

  set_target_properties(
          ${PROJECT_NAME}
          PROPERTIES
          CUDA_ARCHITECTURES "80"
          CUDA_SEPARABLE_COMPILATION ON
          CUDA_RESOLVE_DEVICE_SYMBOLS ON
  )

  # Define `KMM_USE_CUDA` macro so that headers can detect CUDA usage
  target_compile_definitions(${PROJECT_NAME} PUBLIC KMM_USE_CUDA=1)
endif()

# Compile unit tests
if(KMM_BUILD_TESTS)
  include(CTest)
  add_subdirectory(external/Catch2)
  add_subdirectory(test)
endif()

# Compile examples
if(KMM_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()
