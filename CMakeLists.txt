cmake_minimum_required(VERSION 3.24)
project(kv_cache_bench LANGUAGES CXX OBJCXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OBJCXX_STANDARD 20)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
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
