cmake_minimum_required(VERSION 3.22)
project(sqrt_vins_native)

# Set common flags
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Enable optimizations
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fno-signed-zeros -fno-math-errno -funroll-loops")

# ROS support has been removed from the repository.

# Set path to the root of the project
set(SRCDIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../../)

# -----------------------------------------------------------------
# 1. OpenCV for Android
# -----------------------------------------------------------------
# Assumes OpenCV_DIR is set in the environment or passed via gradle
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV found at: ${OpenCV_DIR}")
include_directories(${OpenCV_INCLUDE_DIRS})

# -----------------------------------------------------------------
# 1.5 Yaml-cpp (assuming it's installed or provided)
# -----------------------------------------------------------------
find_package(yaml-cpp REQUIRED)
include_directories(${yaml-cpp_INCLUDE_DIRS})

# -----------------------------------------------------------------
# 2. Eigen for Android
# -----------------------------------------------------------------
# Usually header only, can be provided as a subdirectory or path
# Here we'll assume it's available in the system or passed as -DEIGEN3_INCLUDE_DIR
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# -----------------------------------------------------------------
# 3. Source files for sqrtVINS
# -----------------------------------------------------------------
# Include directories
include_directories(${SRCDIR}/ov_core/src)
include_directories(${SRCDIR}/ov_srvins/src)

# Collect ov_core sources (excluding tests and mains)
file(GLOB_RECURSE CORE_SOURCES "${SRCDIR}/ov_core/src/*.cpp")
list(FILTER CORE_SOURCES EXCLUDE REGEX ".*test_.*\\.cpp$")
list(FILTER CORE_SOURCES EXCLUDE REGEX ".*dummy\\.cpp$")

# Collect ov_srvins sources (excluding tests and mains)
file(GLOB_RECURSE SRVINS_SOURCES "${SRCDIR}/ov_srvins/src/*.cpp")
list(FILTER SRVINS_SOURCES EXCLUDE REGEX ".*run_simulation\\.cpp$")
list(FILTER SRVINS_SOURCES EXCLUDE REGEX ".*test_.*\\.cpp$")
list(FILTER SRVINS_SOURCES EXCLUDE REGEX ".*dummy\\.cpp$")

# -----------------------------------------------------------------
# 4. Build the JNI library
# -----------------------------------------------------------------
add_library(sqrt_vins_native SHARED
        sqrt_vins_native.cpp
        ${CORE_SOURCES}
        ${SRVINS_SOURCES}
)

# Link libraries
target_link_libraries(sqrt_vins_native
        ${OpenCV_LIBRARIES}
        yaml-cpp
        log # Android log library
        android
)
