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
        ${Boost_INCLUDE_DIRS}
)

# Set link libraries used by all binaries
list(APPEND thirdparty_libraries
        ${Boost_LIBRARIES}
        ${OpenCV_LIBRARIES}
)

# Manually link ov_core/ov_init
message(STATUS "MANUALLY LINKING TO OV_CORE LIBRARY....")
include_directories(${CMAKE_SOURCE_DIR}/ov_core/src/)
file(GLOB_RECURSE OVCORE_LIBRARY_SOURCES "${CMAKE_SOURCE_DIR}/ov_core/src/*.cpp")
# Filter out mains
list(FILTER OVCORE_LIBRARY_SOURCES EXCLUDE REGEX ".*test_webcam\\.cpp$")
list(FILTER OVCORE_LIBRARY_SOURCES EXCLUDE REGEX ".*test_tracking\\.cpp$")
list(FILTER OVCORE_LIBRARY_SOURCES EXCLUDE REGEX ".*test_profile\\.cpp$")
list(FILTER OVCORE_LIBRARY_SOURCES EXCLUDE REGEX ".*dummy\\.cpp$")
list(APPEND LIBRARY_SOURCES ${OVCORE_LIBRARY_SOURCES})
file(GLOB_RECURSE OVCORE_LIBRARY_HEADERS "${CMAKE_SOURCE_DIR}/ov_core/src/*.h")
list(APPEND LIBRARY_HEADERS ${OVCORE_LIBRARY_HEADERS})

message(STATUS "MANUALLY LINKING TO OV_INIT LIBRARY....")
include_directories(${CMAKE_SOURCE_DIR}/ov_init/src/)
file(GLOB_RECURSE OVINIT_LIBRARY_SOURCES "${CMAKE_SOURCE_DIR}/ov_init/src/*.cpp")
# Filter out mains or unused files if ov_init existed (preserved from ROS1.cmake)
list(FILTER OVINIT_LIBRARY_SOURCES EXCLUDE REGEX ".*test_dynamic_init\\.cpp$")
list(FILTER OVINIT_LIBRARY_SOURCES EXCLUDE REGEX ".*test_dynamic_mle\\.cpp$")
list(FILTER OVINIT_LIBRARY_SOURCES EXCLUDE REGEX ".*test_simulation\\.cpp$")
list(FILTER OVINIT_LIBRARY_SOURCES EXCLUDE REGEX ".*Simulator\\.cpp$")
list(APPEND LIBRARY_SOURCES ${OVINIT_LIBRARY_SOURCES})
file(GLOB_RECURSE OVINIT_LIBRARY_HEADERS "${CMAKE_SOURCE_DIR}/ov_init/src/*.h")
list(FILTER OVINIT_LIBRARY_HEADERS EXCLUDE REGEX ".*Simulator\\.h$")
list(APPEND LIBRARY_HEADERS ${OVINIT_LIBRARY_HEADERS})


# #################################################
# Make the shared library
# #################################################
list(APPEND LIBRARY_SOURCES
        src/dummy.cpp
        src/sim/Simulator.cpp
        src/state/State.cpp
        src/state/StateHelper.cpp
        src/state/Propagator.cpp
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
# Make binary files!
# #################################################

add_executable(run_simulation src/run_simulation.cpp)
target_link_libraries(run_simulation ov_srvins_lib ${thirdparty_libraries})
install(TARGETS run_simulation
        ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
        )

add_executable(test_sim_meas src/test_sim_meas.cpp)
target_link_libraries(test_sim_meas ov_srvins_lib ${thirdparty_libraries})
install(TARGETS test_sim_meas
        ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
        )

add_executable(test_sim_repeat src/test_sim_repeat.cpp)
target_link_libraries(test_sim_repeat ov_srvins_lib ${thirdparty_libraries})
install(TARGETS test_sim_repeat
        ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
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
