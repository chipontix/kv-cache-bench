cmake_minimum_required(VERSION 3.24)
project(kv_cache_bench LANGUAGES CXX OBJCXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OBJCXX_STANDARD 20)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Comparison baselines (make bench-compare), off by default: llama.cpp's ggml at a fixed release,
# built from source with its Metal backend. Only bench/vs_llamacpp links it.
option(KVB_LLAMACPP "build bench/vs_llamacpp against llama.cpp's ggml (downloads the source)" OFF)
if(KVB_LLAMACPP)
  include(FetchContent)
  set(GGML_METAL ON CACHE BOOL "" FORCE)
  set(GGML_BLAS OFF CACHE BOOL "" FORCE)
  set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(ggml
    URL https://github.com/ggml-org/llama.cpp/archive/refs/tags/b12000.tar.gz  # commit 68d20b8
    URL_HASH SHA256=cb6e28ed73c46156d8db0b17b51ba86ba5b86d9fe70063502b0c80f6e1bc33b3
    SOURCE_SUBDIR ggml
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
  FetchContent_MakeAvailable(ggml)
endif()

add_compile_options(-Wall -Wextra)

# The shader source is compiled at run time (newLibraryWithSource), so it is embedded into the
# binary as a string instead of being built with the offline Metal toolchain.
set(KERNELS_INC ${CMAKE_CURRENT_BINARY_DIR}/kernels_src.inc)
add_custom_command(
  OUTPUT ${KERNELS_INC}
  COMMAND ${CMAKE_COMMAND} -DIN=${CMAKE_CURRENT_SOURCE_DIR}/src/kernels.metal -DOUT=${KERNELS_INC}
          -DNAME=kKernelSource -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed.cmake
  DEPENDS src/kernels.metal cmake/embed.cmake)

add_library(kvref src/reference.cpp)
target_include_directories(kvref PUBLIC src)

add_library(kvmetal src/metal_attn.mm ${KERNELS_INC})
target_include_directories(kvmetal PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
target_compile_options(kvmetal PRIVATE -fobjc-arc)
target_link_libraries(kvmetal PUBLIC kvref "-framework Metal" "-framework Foundation")

add_executable(cpu_tests tests/cpu_tests.cpp)
target_link_libraries(cpu_tests PRIVATE kvref)

add_executable(gpu_tests tests/gpu_tests.cpp)
target_link_libraries(gpu_tests PRIVATE kvmetal)

add_executable(kvbench bench/bench.cpp)
target_link_libraries(kvbench PRIVATE kvmetal)

enable_testing()
add_test(NAME cpu_tests COMMAND cpu_tests)
add_test(NAME gpu_tests COMMAND gpu_tests)
set_tests_properties(gpu_tests PROPERTIES SKIP_RETURN_CODE 77)

if(KVB_LLAMACPP)
  add_executable(vs_llamacpp bench/vs_llamacpp.cpp)
  target_link_libraries(vs_llamacpp PRIVATE kvmetal ggml)
endif()
