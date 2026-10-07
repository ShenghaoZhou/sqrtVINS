cmake_minimum_required(VERSION 3.22)
project(sqrtvins)

find_program(CCACHE_PROGRAM ccache)
message (STATUS "CCACHE_PROGRAM: ${CCACHE_PROGRAM}")
if(CCACHE_PROGRAM)
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    set(CMAKE_CUDA_COMPILER_LAUNCHER "${CCACHE_PROGRAM}") # CMake 3.9+
endif()

# Target ISA for the dense linear algebra (Eigen). The pixi/conda toolchain
# defaults to -march=nocona (SSE2-era); overriding it to "native" (or the
# portable "x86-64-v3" for distributable AVX2+FMA builds) measurably speeds
# up the estimator backend (~25% on the sqrt update). Set to "" for a fully
# portable build.
set(SQRTVINS_TARGET_ARCH "native" CACHE STRING
    "Value for -march (e.g. native, x86-64-v3, or empty for a portable build)")
if(SQRTVINS_TARGET_ARCH)
    message(STATUS "Building with -march=${SQRTVINS_TARGET_ARCH}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=${SQRTVINS_TARGET_ARCH}")
endif()

# Bundle-adjustment backend built on a vendored minimal colmap + Ceres.
# Disable only if the Ceres/glog deps are unavailable.
option(SQRTVINS_ENABLE_BACKEND
       "Build the colmap/Ceres bundle-adjustment backend" ON)
if(SQRTVINS_ENABLE_BACKEND)
    add_subdirectory(third_party/colmap-lite)
endif()

add_subdirectory(ov_core)
add_subdirectory(ov_init)
add_subdirectory(ov_msckf)
add_subdirectory(ov_srvins)

# Python bindings for the visual frontend (extension point for custom
# Python-implemented detectors/matchers, e.g. XFeat)
option(SQRTVINS_ENABLE_PYTHON "Build the pysqrtvins python bindings" ON)
if(SQRTVINS_ENABLE_PYTHON)
    add_subdirectory(pysqrtvins)
endif()
