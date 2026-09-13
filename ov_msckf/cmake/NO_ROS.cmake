cmake_minimum_required(VERSION 3.5)

add_definitions(-DROS_AVAILABLE=0)

include(GNUInstallDirs)
set(CATKIN_PACKAGE_LIB_DESTINATION "${CMAKE_INSTALL_LIBDIR}")
set(CATKIN_PACKAGE_BIN_DESTINATION "${CMAKE_INSTALL_BINDIR}")
set(CATKIN_GLOBAL_INCLUDE_DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")

# Must match the ov_core build type (USE_FLOAT) so that shared types
# (Vec3, MatX, ...) have the same ABI across all OpenVINS libraries
option(USE_FLOAT "Use float version when built" ON)
if(USE_FLOAT)
    add_definitions(-DUSE_FLOAT=1)
endif()

# Include our header files
include_directories(
        src
        ${EIGEN3_INCLUDE_DIR}
        ${Boost_INCLUDE_DIRS}
        ${CMAKE_SOURCE_DIR}/ov_core/src
        ${CMAKE_SOURCE_DIR}/ov_init/src
)

# Set link libraries used by all binaries
list(APPEND thirdparty_libraries
        ${OpenCV_LIBRARIES}
        yaml-cpp
)

##################################################
# Make the full-covariance msckf library
##################################################

list(APPEND LIBRARY_SOURCES
        src/dummy.cpp
        src/core/VioManager.cpp
        src/core/VioManagerHelper.cpp
        src/state/State.cpp
        src/state/StateHelper.cpp
        src/state/Propagator.cpp
        src/update/UpdaterHelper.cpp
        src/update/UpdaterMSCKF.cpp
        src/update/UpdaterSLAM.cpp
        src/update/UpdaterZeroVelocity.cpp
)
file(GLOB_RECURSE LIBRARY_HEADERS "src/*.h")
add_library(ov_msckf_lib SHARED ${LIBRARY_SOURCES} ${LIBRARY_HEADERS})
target_link_libraries(ov_msckf_lib ov_core_lib Eigen3::Eigen ov_init_lib ${thirdparty_libraries})
target_include_directories(ov_msckf_lib PUBLIC src/)
install(TARGETS ov_msckf_lib
        ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
install(DIRECTORY src/
        DESTINATION ${CATKIN_GLOBAL_INCLUDE_DESTINATION}
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)

##################################################
# Runner and pybind translation units for the full-covariance formulation.
# These are compiled here (not in ov_srvins) so that their include dirs are
# pinned to this module - the sqrt and full-covariance modules expose
# colliding header paths.
##################################################

add_library(euroc_full_part OBJECT src/run_euroc_full.cpp)
target_link_libraries(euroc_full_part PRIVATE ov_msckf_lib)
target_include_directories(euroc_full_part PRIVATE ${CMAKE_SOURCE_DIR}/common)

find_package(pybind11 REQUIRED)
add_library(pybind_full_part OBJECT src/pybind_full.cpp)
target_link_libraries(pybind_full_part PRIVATE ov_msckf_lib pybind11::pybind11)
target_include_directories(pybind_full_part PRIVATE ${CMAKE_SOURCE_DIR}/common)
