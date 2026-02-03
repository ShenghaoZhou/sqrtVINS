cmake_minimum_required(VERSION 3.22)
project(sqrtvins)

find_program(CCACHE_PROGRAM ccache)
message (STATUS "CCACHE_PROGRAM: ${CCACHE_PROGRAM}")
if(CCACHE_PROGRAM)
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    set(CMAKE_CUDA_COMPILER_LAUNCHER "${CCACHE_PROGRAM}") # CMake 3.9+
endif()

add_subdirectory(ov_core)
add_subdirectory(ov_srvins)
