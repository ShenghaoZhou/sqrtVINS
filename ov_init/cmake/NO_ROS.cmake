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
        ${CMAKE_SOURCE_DIR}/ov_core/src
        ${CERES_INCLUDE_DIRS}
)

# Set link libraries used by all binaries
list(APPEND thirdparty_libraries
        ${OpenCV_LIBRARIES}
        yaml-cpp
)

##################################################
# Make the initializer library
##################################################

list(APPEND LIBRARY_SOURCES
        src/dummy.cpp
        src/ceres/Factor_GenericPrior.cpp
        src/ceres/Factor_ImageReprojCalib.cpp
        src/ceres/Factor_ImuCPIv1.cpp
        src/ceres/State_JPLQuatLocal.cpp
        src/dynamic/DynamicInitializer.cpp
        src/static/StaticInitializer.cpp
        src/init/InertialInitializer.cpp
)
file(GLOB_RECURSE LIBRARY_HEADERS "src/*.h")
add_library(ov_init_lib SHARED ${LIBRARY_SOURCES} ${LIBRARY_HEADERS})
target_link_libraries(ov_init_lib ov_core_lib Eigen3::Eigen ${thirdparty_libraries} ${CERES_LIBRARIES})
target_include_directories(ov_init_lib PUBLIC src/)
install(TARGETS ov_init_lib
        ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
install(DIRECTORY src/
        DESTINATION ${CATKIN_GLOBAL_INCLUDE_DESTINATION}
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)
