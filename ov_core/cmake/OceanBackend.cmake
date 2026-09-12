# Support for the Ocean framework backend of the vision abstraction layer.
#
# When ENABLE_OCEAN_BACKEND is ON, the "ocean" CV backend (see
# ov_core/src/vision/OceanBackend.h) is compiled in and can be selected at
# runtime via the "cv_backend" option. This requires a built Ocean framework:
#
#   cmake -S third_party/ocean -B build_ocean -GNinja
#   cmake --build build_ocean
#
# and the OCEAN_ROOT cache variable pointing to the Ocean base directory
# (defaults to <source dir>/third_party/ocean), containing the impl/ headers
# and the built ocean_base/ocean_cv/ocean_cv_detector/ocean_geometry/
# ocean_math libraries.
#
# Variables defined by this file:
#   OCEAN_LIBRARIES     - libraries to link for the Ocean backend (empty if disabled)
#   OCEAN_BACKEND_SOURCES - source files that include Ocean headers (may need
#                           special compile flags, e.g. C++20)

option(ENABLE_OCEAN_BACKEND "Enable the Ocean framework CV backend (requires a built Ocean)" OFF)

if (ENABLE_OCEAN_BACKEND)
    if (NOT OCEAN_ROOT)
        set(OCEAN_ROOT "${CMAKE_SOURCE_DIR}/third_party/ocean" CACHE PATH "Base directory of the Ocean framework" FORCE)
    endif ()

    find_path(OCEAN_INCLUDE_DIR ocean/base/Frame.h
            HINTS "${OCEAN_ROOT}/impl"
            NO_DEFAULT_PATH)

    set(OCEAN_LIBRARIES "")
    set(OCEAN_MISSING "")
    foreach (_ocean_module base cv cv_detector geometry math)
        find_library(OCEAN_${_ocean_module}_LIBRARY
                NAMES ocean_${_ocean_module}
                HINTS "${OCEAN_ROOT}/build" "${OCEAN_ROOT}/build/lib" "${OCEAN_ROOT}/lib"
                NO_DEFAULT_PATH)
        if (OCEAN_${_ocean_module}_LIBRARY)
            list(APPEND OCEAN_LIBRARIES ${OCEAN_${_ocean_module}_LIBRARY})
        else ()
            list(APPEND OCEAN_MISSING ocean_${_ocean_module})
        endif ()
    endforeach ()

    if (OCEAN_MISSING OR NOT OCEAN_INCLUDE_DIR)
        message(FATAL_ERROR
                "ENABLE_OCEAN_BACKEND is ON but Ocean was not found "
                "(missing: ${OCEAN_MISSING}; headers: ${OCEAN_INCLUDE_DIR}). "
                "Build Ocean first (see third_party/ocean/doc/building_for_linux.md) "
                "and pass -DOCEAN_ROOT=<path to Ocean> with built libraries.")
    endif ()

    message(STATUS "OCEAN: enabling Ocean CV backend (root: ${OCEAN_ROOT})")

    add_definitions(-DOV_HAVE_OCEAN=1)
    include_directories(${OCEAN_INCLUDE_DIR})

    # Ocean requires C++20; the SIMD paths of its headers benefit from SSE4.1
    set(OCEAN_BACKEND_SOURCES
            "${CMAKE_SOURCE_DIR}/ov_core/src/vision/OceanBackend.cpp"
            "${CMAKE_SOURCE_DIR}/ov_core/src/vision/CVBackend.cpp")
    set_source_files_properties(${OCEAN_BACKEND_SOURCES}
            PROPERTIES COMPILE_OPTIONS "-std=c++20;-msse4.1")
else ()
    set(OCEAN_LIBRARIES "")
    set(OCEAN_BACKEND_SOURCES "")
endif ()
