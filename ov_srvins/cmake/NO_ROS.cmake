cmake_minimum_required(VERSION 3.5)

add_definitions(-DROS_AVAILABLE=0)
message(WARNING "BUILDING WITHOUT ROS!")

# Find Eigen3 if not already set (fallback for safety)
if(NOT EIGEN3_INCLUDE_DIR)
    if(TARGET Eigen3::Eigen)
        get_target_property(EIGEN3_INCLUDE_DIR Eigen3::Eigen INTERFACE_INCLUDE_DIRECTORIES)
    endif()
endif()

IF (NOT EIGEN3_INCLUDE_DIR)
     MESSAGE(WARNING "EIGEN3_INCLUDE_DIR not set, relying on standard include paths or manual setting.")
ENDIF ()

INCLUDE_DIRECTORIES("${EIGEN3_INCLUDE_DIR}")

include(GNUInstallDirs)
set(CATKIN_PACKAGE_LIB_DESTINATION "${CMAKE_INSTALL_LIBDIR}")
set(CATKIN_PACKAGE_BIN_DESTINATION "${CMAKE_INSTALL_BINDIR}")
set(CATKIN_GLOBAL_INCLUDE_DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")

option(USE_FLOAT "Use float version when built" ON)
if (USE_FLOAT)
    add_definitions(-DUSE_FLOAT=1)
endif ()

# Include our header files
include_directories(
        src
        ${EIGEN3_INCLUDE_DIR}
)

# Set link libraries used by all binaries
list(APPEND thirdparty_libraries
        ${OpenCV_LIBRARIES}
        yaml-cpp
)

# Optional Ocean framework backend for the vision abstraction layer
include(${CMAKE_SOURCE_DIR}/ov_core/cmake/OceanBackend.cmake)
list(APPEND thirdparty_libraries ${OCEAN_LIBRARIES})


# Link against the shared ov_core / ov_init / ov_msckf libraries
message(STATUS "LINKING TO OV_CORE / OV_INIT / OV_MSCKF LIBRARIES....")
list(APPEND thirdparty_libraries ov_core_lib ov_init_lib ov_msckf_lib)



# #################################################
# Make the shared library
# #################################################
list(APPEND LIBRARY_SOURCES
        src/dummy.cpp
        src/state/State.cpp
        src/state/StateHelper.cpp
        src/state/Propagator.cpp
        src/state/IMUHandler.cpp
        src/core/SqrtEstimator.cpp
        src/core/Frontend.cpp
        src/core/VioManager.cpp
        src/core/VioManagerOptions.cpp
        src/update/UpdaterHelper.cpp
        src/update/UpdaterMSCKF.cpp
        src/update/UpdaterSLAM.cpp
        src/update/UpdaterZeroVelocity.cpp

        src/initializer/InertialInitializer.cpp
        src/initializer/InertialInitializerOptions.cpp
        src/initializer/dynamic/Solver.cpp
        src/initializer/dynamic/OpengvHelper.cpp
        src/initializer/dynamic/DynamicInitializer.cpp
        src/initializer/static/StaticInitializer.cpp
        
        src/utils/Timer.cpp
        src/utils/Helper.cpp
        src/utils/NoiseManager.cpp
        src/utils/CameraPoseBuffer.cpp
        src/utils/EigenMatrixBuffer.cpp
        )


file(GLOB_RECURSE LIBRARY_HEADERS "src/*.h")
add_library(ov_srvins_lib SHARED ${LIBRARY_SOURCES} ${LIBRARY_HEADERS})

# C++ dataset runner: main + per-formulation translation units.
# The sqrt and full-covariance modules expose colliding header paths, so each
# formulation's TU is compiled as its own object library with include dirs
# pinned to its module (via the linked library).
add_library(euroc_sqrt_part OBJECT src/run_euroc_sqrt.cpp)
target_link_libraries(euroc_sqrt_part PRIVATE ov_srvins_lib)
target_include_directories(euroc_sqrt_part PRIVATE ${CMAKE_SOURCE_DIR}/common)
add_executable(run_euroc_vio src/run_euroc_vio.cpp)
target_link_libraries(run_euroc_vio ov_srvins_lib ${thirdparty_libraries} isoc23_shim)

add_executable(run_euroc src/run_euroc.cpp)
target_link_libraries(run_euroc euroc_sqrt_part euroc_full_part ${thirdparty_libraries})
# must come after the Ocean archives so it resolves their glibc-2.38+ refs
target_link_libraries(run_euroc isoc23_shim)
target_include_directories(run_euroc PRIVATE ${CMAKE_SOURCE_DIR}/common)

add_library(isoc23_shim STATIC ${CMAKE_SOURCE_DIR}/common/isoc23_shim.c)
target_link_libraries(ov_srvins_lib ${thirdparty_libraries})
target_include_directories(ov_srvins_lib PUBLIC src/)
install(TARGETS ov_srvins_lib
        ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
        )
install(DIRECTORY src/
        DESTINATION ${CATKIN_GLOBAL_INCLUDE_DESTINATION}
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
        )

# #################################################
# Launch files!
# #################################################
# Use CMAKE_INSTALL_DATADIR or similar if CMAKE_PACKAGE_SHARE_DESTINATION is strictly catkin
# But here we used CATKIN_PACKAGE_SHARE_DESTINATION in ROS1.cmake which was undefined in NO ROS block? 
# In ROS1.cmake else block, CATKIN_PACKAGE_SHARE_DESTINATION was NOT defined. 
# So install(DIRECTORY launch/ ...) likely failed or went to invalid path if this was hit?
# We'll just define it or skip it.
set(CATKIN_PACKAGE_SHARE_DESTINATION "${CMAKE_INSTALL_DATADIR}/geometry")
install(DIRECTORY launch/
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
        )

# #################################################
# Python Bindings
# #################################################
find_package(pybind11 REQUIRED)
message(STATUS "PYBIND11: " ${pybind11_VERSION})

pybind11_add_module(ov_srvins_py src/pybind.cpp)
target_link_libraries(ov_srvins_py PRIVATE ov_srvins_lib pybind_full_part)
target_include_directories(ov_srvins_py PRIVATE src/ ${CMAKE_SOURCE_DIR})

install(TARGETS ov_srvins_py
        ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
        )

