cmake_minimum_required(VERSION 3.25)
project(metal_linalg VERSION 2.0.0 LANGUAGES CXX OBJCXX
        DESCRIPTION "QR, symmetric eigendecomposition and SVD on Apple GPUs, for MLX")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Standalone (top level) it is a shared library with install rules, which is
# what Homebrew ships. As a subproject (add_subdirectory / FetchContent) it is
# a static library folded into the parent's binary, with no install rules, so
# a parent that builds it from source gets the shaders and routing tables of
# that checkout and nothing else to ship.
option(METAL_LINALG_SHARED      "Build libmetal_linalg as a shared library" ${PROJECT_IS_TOP_LEVEL})
option(METAL_LINALG_INSTALL     "Generate install and CMake package rules" ${PROJECT_IS_TOP_LEVEL})
option(METAL_LINALG_BUILD_TESTS "Build the tests, benchmarks and tuning harnesses" ${PROJECT_IS_TOP_LEVEL})
option(METAL_LINALG_USE_PREBUILT_SHADERS
       "Use shaders/prebuilt/*.metallib instead of compiling the .metal sources" OFF)

find_package(MLX REQUIRED)
include(GNUInstallDirs)

# =============================================================================
# Metal shaders
# =============================================================================
# Each .metal is compiled to a metallib and the metallib's bytes are compiled
# into the library, so an installed libmetal_linalg is self-contained: nothing
# is looked up on disk at run time and consumers need no Metal toolchain.
#
# Building from source needs the Metal shader compiler. From Xcode 26 it is a
# separate download (xcodebuild -downloadComponent MetalToolchain); where it is
# missing, the metallibs committed under shaders/prebuilt/ are used instead.
# They are the same AIR whatever GPU compiled them. After editing a shader on
# a machine that has the compiler, `cmake --build build --target
# update_prebuilt_shaders` refreshes that directory.

execute_process(COMMAND xcrun -sdk macosx metal --version
                RESULT_VARIABLE _metal_rc OUTPUT_QUIET ERROR_QUIET)
if (_metal_rc EQUAL 0)
    set(METAL_LINALG_HAVE_METAL_COMPILER ON)
else ()
    set(METAL_LINALG_HAVE_METAL_COMPILER OFF)
endif ()
if (NOT METAL_LINALG_HAVE_METAL_COMPILER AND NOT METAL_LINALG_USE_PREBUILT_SHADERS)
    message(STATUS "metal-linalg: no Metal shader compiler (xcodebuild -downloadComponent "
                   "MetalToolchain installs it); using shaders/prebuilt")
    set(METAL_LINALG_USE_PREBUILT_SHADERS ON)
endif ()

set(_ml_shader_dir "${CMAKE_CURRENT_BINARY_DIR}/shaders")
file(MAKE_DIRECTORY "${_ml_shader_dir}")

# metal_linalg_add_shader(<name> [FLAGS ...] [DEPENDS <header> ...])
# Compiles shaders/<name>.metal (or copies the prebuilt metallib) and embeds
# it as the C symbols metal_linalg_<name>_metallib / _len; see src/shaders.h.
function(metal_linalg_add_shader name)
    cmake_parse_arguments(_arg "" "" "FLAGS;DEPENDS" ${ARGN})
    set(_src "${CMAKE_CURRENT_SOURCE_DIR}/shaders/${name}.metal")
    set(_air "${_ml_shader_dir}/${name}.air")
    set(_lib "${_ml_shader_dir}/${name}.metallib")
    set(_cpp "${_ml_shader_dir}/${name}_metallib.cpp")

    if (METAL_LINALG_USE_PREBUILT_SHADERS)
        set(_pre "${CMAKE_CURRENT_SOURCE_DIR}/shaders/prebuilt/${name}.metallib")
        if (NOT EXISTS "${_pre}")
            message(FATAL_ERROR "metal-linalg: no Metal shader compiler and no prebuilt ${name}.metallib")
        endif ()
        add_custom_command(
            OUTPUT "${_lib}"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_pre}" "${_lib}"
            DEPENDS "${_pre}"
            COMMENT "Using prebuilt ${name}.metallib"
            VERBATIM)
    else ()
        set(_deps "${_src}")
        foreach (_d IN LISTS _arg_DEPENDS)
            list(APPEND _deps "${CMAKE_CURRENT_SOURCE_DIR}/shaders/${_d}")
        endforeach ()
        # Re-runs whenever the source or a header it includes changes, so an
        # edited shader can never be masked by a stale metallib.
        add_custom_command(
            OUTPUT "${_lib}"
            COMMAND xcrun -sdk macosx metal ${_arg_FLAGS} -c "${_src}" -o "${_air}"
            COMMAND xcrun -sdk macosx metallib "${_air}" -o "${_lib}"
            DEPENDS ${_deps}
            BYPRODUCTS "${_air}"
            COMMENT "Compiling ${name}.metal -> ${name}.metallib"
            VERBATIM)
    endif ()

    add_custom_command(
        OUTPUT "${_cpp}"
        COMMAND ${CMAKE_COMMAND} -DINPUT=${_lib} -DOUTPUT=${_cpp}
                -DSYMBOL=metal_linalg_${name}_metallib
                -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/EmbedMetallib.cmake"
        DEPENDS "${_lib}" "${CMAKE_CURRENT_SOURCE_DIR}/cmake/EmbedMetallib.cmake"
        COMMENT "Embedding ${name}.metallib"
        VERBATIM)

    set_property(GLOBAL APPEND PROPERTY METAL_LINALG_EMBEDDED_SOURCES "${_cpp}")
    set_property(GLOBAL APPEND PROPERTY METAL_LINALG_METALLIBS "${_lib}")
endfunction()

metal_linalg_add_shader(QR_Unblocked)
metal_linalg_add_shader(QR_Streaming_AMX_Reduced)
metal_linalg_add_shader(QR_Streaming_AMX_Complete)

# The Jacobi rotations need c^2 + s^2 == 1 to roundoff and the non-finite
# check needs IEEE semantics, neither of which fast-math guarantees. The
# kernels are barrier- and memory-bound, so precise arithmetic costs nothing
# measurable.
metal_linalg_add_shader(Eigh_Jacobi      FLAGS -fno-fast-math DEPENDS eigh_jacobi_common.h)
metal_linalg_add_shader(Eigh_BlockJacobi FLAGS -fno-fast-math DEPENDS eigh_jacobi_common.h block_jacobi_common.h)
metal_linalg_add_shader(Svd_Jacobi       FLAGS -fno-fast-math DEPENDS eigh_jacobi_common.h)
metal_linalg_add_shader(Svd_BlockJacobi  FLAGS -fno-fast-math DEPENDS eigh_jacobi_common.h block_jacobi_common.h)

get_property(_ml_embedded GLOBAL PROPERTY METAL_LINALG_EMBEDDED_SOURCES)
get_property(_ml_metallibs GLOBAL PROPERTY METAL_LINALG_METALLIBS)

if (NOT METAL_LINALG_USE_PREBUILT_SHADERS)
    add_custom_target(update_prebuilt_shaders
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${_ml_metallibs}
                "${CMAKE_CURRENT_SOURCE_DIR}/shaders/prebuilt/"
        DEPENDS ${_ml_metallibs}
        COMMENT "Refreshing shaders/prebuilt/ from this build"
        VERBATIM)
endif ()

# =============================================================================
# The library
# =============================================================================

if (METAL_LINALG_SHARED)
    set(_ml_kind SHARED)
else ()
    set(_ml_kind STATIC)
endif ()
add_library(metal_linalg ${_ml_kind}
    src/device.mm
    src/metal_runtime.mm
    src/qr.mm
    src/qr_unblocked.mm
    src/qr_streaming_amx_reduced.mm
    src/qr_streaming_amx_complete.mm
    src/eigh.mm
    src/eigh_block_jacobi.mm
    src/svd.mm
    src/svd_block_jacobi.mm
    ${_ml_embedded}
)
add_library(metal_linalg::metal_linalg ALIAS metal_linalg)

# An absolute install name, so a consumer linked against the installed
# library finds it without an rpath; Homebrew passes its opt path.
set_target_properties(metal_linalg PROPERTIES
    VERSION   ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    EXPORT_NAME metal_linalg
    INSTALL_NAME_DIR "${CMAKE_INSTALL_FULL_LIBDIR}"
)
target_compile_features(metal_linalg PUBLIC cxx_std_20)
target_include_directories(metal_linalg
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
)
# The public headers only need MLX; the frameworks are the library's own, but
# a static build hands them to the consumer's link line.
target_link_libraries(metal_linalg
    PUBLIC  mlx
    PRIVATE "-framework Metal" "-framework Foundation" "-framework IOKit"
)
if (NOT METAL_LINALG_SHARED)
    target_link_libraries(metal_linalg INTERFACE "-framework Metal" "-framework Foundation" "-framework IOKit")
endif ()

# =============================================================================
# Install and export: find_package(MetalLinalg) -> metal_linalg::metal_linalg
# =============================================================================

if (METAL_LINALG_INSTALL)
install(TARGETS metal_linalg
    EXPORT MetalLinalgTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

set(_ml_cmake_dir "${CMAKE_INSTALL_LIBDIR}/cmake/MetalLinalg")
install(EXPORT MetalLinalgTargets
    FILE MetalLinalgTargets.cmake
    NAMESPACE metal_linalg::
    DESTINATION "${_ml_cmake_dir}"
)
include(CMakePackageConfigHelpers)
configure_package_config_file(cmake/MetalLinalgConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/MetalLinalgConfig.cmake"
    INSTALL_DESTINATION "${_ml_cmake_dir}"
)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/MetalLinalgConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/MetalLinalgConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/MetalLinalgConfigVersion.cmake"
    DESTINATION "${_ml_cmake_dir}"
)
endif ()

# =============================================================================
# Tests, benchmarks and tuning harnesses
# =============================================================================

if (METAL_LINALG_BUILD_TESTS)
    include(CTest)

    function(metal_linalg_add_tool name)
        add_executable(${name} ${ARGN})
        target_link_libraries(${name} PRIVATE metal_linalg mlx)
    endfunction()

    metal_linalg_add_tool(test_qr   tests/test_qr.cpp)
    metal_linalg_add_tool(test_eigh tests/test_eigh.cpp)
    metal_linalg_add_tool(test_svd  tests/test_svd.cpp)
    add_test(NAME qr_correctness   COMMAND test_qr)
    add_test(NAME eigh_correctness COMMAND test_eigh)
    add_test(NAME svd_correctness  COMMAND test_svd)

    metal_linalg_add_tool(benchmark_qr   benchmarks/benchmark_qr.cpp)
    metal_linalg_add_tool(benchmark_eigh benchmarks/benchmark_eigh.cpp)
    metal_linalg_add_tool(benchmark_svd  benchmarks/benchmark_svd.cpp)

    # Tuning harnesses: run these when porting to a different Apple GPU, since
    # core count and per-core throughput set every routing crossover; see
    # docs/tuning.md.
    metal_linalg_add_tool(sweep_qr   tuning/sweep_qr.cpp)
    metal_linalg_add_tool(sweep_eigh tuning/sweep_eigh.cpp)
    metal_linalg_add_tool(sweep_svd  tuning/sweep_svd.cpp)

    metal_linalg_add_tool(probe_occupancy tuning/probe_occupancy.mm)
    target_include_directories(probe_occupancy PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(probe_occupancy PRIVATE "-framework Metal" "-framework Foundation")
endif ()
