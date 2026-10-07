cmake_minimum_required(VERSION 3.22)
project(pysqrtvins)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV 3 QUIET)
if (NOT OpenCV_FOUND)
    find_package(OpenCV 4 REQUIRED)
endif ()

pybind11_add_module(pysqrtvins
        src/PyBackend.cpp
        src/bindings.cpp
)
target_link_libraries(pysqrtvins PRIVATE ov_core_lib Eigen3::Eigen ${OpenCV_LIBRARIES})
target_include_directories(pysqrtvins PRIVATE ${CMAKE_SOURCE_DIR}/ov_core/src ${OpenCV_INCLUDE_DIRS})
target_compile_features(pysqrtvins PRIVATE cxx_std_17)

# Let the module find libov_core_lib.so next to the build tree without
# needing LD_LIBRARY_PATH
set_target_properties(pysqrtvins PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/pysqrtvins
        BUILD_RPATH "$ORIGIN/../ov_core"
)
