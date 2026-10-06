# Minimal vendored build of third_party/colmap for the SqrtVINS BA backend.
#
# The full colmap build pulls heavy optional deps (OpenImageIO, SQLite-backed
# database, PoseLib, FAISS, ONNX, Qt, CUDA) that the backend never uses. We
# therefore compile an explicit subset of translation units covering:
#   - scene representation (Reconstruction, Rig, Frame, Image, Camera, Track)
#   - geometry (rigid3, sim3, triangulation, pose priors)
#   - Ceres-based bundle adjustment (CreateDefaultCeresBundleAdjuster)
#   - marginal covariance estimation (used by the filter feedback bridge)
#
# Sources that transitively reference unavailable deps (sensor/bitmap.cc ->
# OpenImageIO, scene/database*.cc -> SQLite/version.cc) are only reachable
# from Reconstruction methods we never call, so they are compiled with
# -ffunction-sections and dropped by the linker via --gc-sections. Do NOT
# compile those .cc files into this target.

cmake_minimum_required(VERSION 3.22)
project(colmap_lite CXX)

find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)
find_package(glog REQUIRED)
find_package(Threads REQUIRED)

set(COLMAP_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../colmap/src)

set(COLMAP_LITE_SRCS
    # util (no oiio_utils.cc / opengl_utils.cc / onnx.cc / cuda.cc)
    util/logging.cc
    util/string.cc
    util/misc.cc
    util/file.cc
    util/threading.cc
    util/timer.cc
    util/endian.cc
    util/ply.cc
    # math
    math/math.cc
    math/random.cc
    math/polynomial.cc
    # geometry
    geometry/rigid3.cc
    geometry/sim3.cc
    geometry/pose.cc
    geometry/pose_prior.cc
    geometry/normalization.cc
    geometry/triangulation.cc
    geometry/gps.cc
    # sensor (no bitmap.cc: requires OpenImageIO)
    sensor/models.cc
    sensor/rig.cc
    sensor/specs.cc
    sensor/database.cc
    # feature types (keypoint/descriptor containers used by scene types)
    feature/types.cc
    # scene (no database.cc / database_sqlite.cc / database_cache.cc: require
    # SQLite + generated version.cc; reachable only from unused code paths)
    scene/camera.cc
    scene/frame.cc
    scene/image.cc
    scene/point2d.cc
    scene/point3d.cc
    scene/rig.cc
    scene/track.cc
    scene/two_view_geometry.cc
    scene/correspondence_graph.cc
    scene/projection.cc
    scene/reconstruction.cc
    scene/reconstruction_io.cc
    scene/reconstruction_io_text.cc
    scene/reconstruction_io_binary.cc
    scene/reconstruction_io_utils.cc
    scene/reconstruction_pruning.cc
    # estimators: Ceres BA + covariance (cost_functions are header-only)
    estimators/bundle_adjustment.cc
    estimators/bundle_adjustment_ceres.cc
    estimators/ceres_loss_function.cc
    estimators/covariance.cc
)

list(TRANSFORM COLMAP_LITE_SRCS PREPEND ${COLMAP_SRC_DIR}/colmap/)

add_library(colmap_lite STATIC ${COLMAP_LITE_SRCS})
add_library(colmap_lite::colmap_lite ALIAS colmap_lite)

# gc-sections strip Reconstruction methods that reference bitmap/database
# symbols (see header comment); this relies on per-function sections here.
# Hidden visibility keeps colmap symbols local to consumers (they are
# vendored implementation details), which is also what allows the linker to
# drop unreferenced functions when colmap_lite is linked into shared libs.
target_compile_options(colmap_lite PRIVATE -ffunction-sections -fdata-sections
                       -fvisibility=hidden -fvisibility-inlines-hidden)
# Consumers must link with --gc-sections for the stripping to happen
target_link_options(colmap_lite INTERFACE -Wl,--gc-sections)
# ov_srvins_lib is SHARED and links this static archive
set_target_properties(colmap_lite PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(colmap_lite PUBLIC ${COLMAP_SRC_DIR})

# conda-forge glog 0.7.x has the new (>=0.7) CheckOpString API but does not
# define GLOG_VERSION_*, which colmap's util/logging.h uses to select the
# matching LogMessageFatalThrow overload. Declare the version explicitly so
# colmap picks the correct branch. PUBLIC because our backend sources include
# colmap headers as well.
target_compile_definitions(colmap_lite PUBLIC GLOG_VERSION_MAJOR=0
                                              GLOG_VERSION_MINOR=7)

target_link_libraries(colmap_lite
    PUBLIC
        Eigen3::Eigen
        Ceres::ceres
        glog::glog
        Threads::Threads
)

# colmap requires C++17 minimum; match the parent project's standard
target_compile_features(colmap_lite PUBLIC cxx_std_17)

# colmap sources are warning-heavy under the parent project's stricter flags
target_compile_options(colmap_lite PRIVATE -w)
