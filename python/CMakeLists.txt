# The Python extension, metal_linalg._core. Built by scikit-build-core from
# pyproject.toml at the repository root, with METAL_LINALG_BUILD_PYTHON on.
#
# It must share one MLX with the `mlx` Python package: the same libmlx (the
# root CMakeLists.txt points find_package(MLX) at the one that package uses)
# and the same nanobind ABI, so that mx.array objects pass between the two
# modules. Both are checked here, before anything is compiled.

find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)

execute_process(COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
                OUTPUT_VARIABLE nanobind_ROOT OUTPUT_STRIP_TRAILING_WHITESPACE
                RESULT_VARIABLE _nb_rc ERROR_QUIET)
if (NOT _nb_rc EQUAL 0)
    message(FATAL_ERROR "metal-linalg: nanobind is not installed for ${Python_EXECUTABLE}. "
                        "See python/README.md.")
endif ()
find_package(nanobind CONFIG REQUIRED)

execute_process(COMMAND "${Python_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/nanobind_abi.py"
                OUTPUT_VARIABLE _mlx_nb_abi OUTPUT_STRIP_TRAILING_WHITESPACE
                RESULT_VARIABLE _abi_rc ERROR_VARIABLE _abi_err)
if (NOT _abi_rc EQUAL 0)
    message(FATAL_ERROR "metal-linalg: could not import mlx with ${Python_EXECUTABLE}:\n${_abi_err}")
endif ()
# The tag is v<N>_... from nanobind 2.x and nanobind_..._v<N> from 3.x, whose
# internals never interoperate with 2.x's whatever the number.
if (_mlx_nb_abi MATCHES "^v([0-9]+)_")
    set(_mlx_nb_version "${CMAKE_MATCH_1}")
    set(_mlx_nb_major 2)
elseif (_mlx_nb_abi MATCHES "^nanobind_.*_v([0-9]+)$")
    set(_mlx_nb_version "${CMAKE_MATCH_1}")
    set(_mlx_nb_major 3)
else ()
    message(FATAL_ERROR "metal-linalg: could not read the nanobind ABI mlx.core uses "
                        "(its tag is '${_mlx_nb_abi}')")
endif ()
# nanobind defines its internals version in src/nb_internals.h or nb_abi.h, indented.
file(GLOB_RECURSE _nb_headers "${nanobind_ROOT}/../src/*.h" "${nanobind_ROOT}/../include/nanobind/*.h")
set(_our_nb_version "")
foreach (_h IN LISTS _nb_headers)
    file(STRINGS "${_h}" _line REGEX "#[ \t]*define[ \t]+NB_INTERNALS_VERSION[ \t]+[0-9]+")
    if (_line AND NOT _our_nb_version)
        string(REGEX MATCH "[0-9]+$" _our_nb_version "${_line}")
    endif ()
endforeach ()
string(REGEX MATCH "^[0-9]+" _nb_major "${nanobind_VERSION}")
if (NOT _our_nb_version STREQUAL _mlx_nb_version OR NOT _nb_major STREQUAL _mlx_nb_major)
    message(FATAL_ERROR "metal-linalg: mlx.core was built with nanobind ${_mlx_nb_major}.x, internals "
                        "v${_mlx_nb_version}, but the installed nanobind ${nanobind_VERSION} uses "
                        "v${_our_nb_version}, so mx.array could not pass between the two modules. "
                        "Install the nanobind release that matches (see python/README.md).")
endif ()
message(STATUS "metal-linalg: Python ${Python_VERSION}, mlx nanobind ABI ${_mlx_nb_abi}, "
               "nanobind ABI v${_our_nb_version}, MLX ${MLX_VERSION} at ${MLX_LIBRARY}")

nanobind_add_module(_core NB_STATIC NB_DOMAIN mlx src/bindings.cpp)
target_link_libraries(_core PRIVATE metal_linalg mlx)
target_compile_definitions(_core PRIVATE
    "METAL_LINALG_VERSION=\"${PROJECT_VERSION}\""
    "METAL_LINALG_MLX_VERSION=\"${MLX_VERSION}\"")

# libmlx is found where the mlx package keeps it: inside the package for a
# pip-installed mlx (site-packages/mlx/lib, beside this package), in the
# Homebrew prefix for Homebrew's. Only the second needs an absolute path; a
# wheel must not carry one, as it would name the machine that built it.
get_filename_component(_mlx_libdir "${MLX_LIBRARY}" DIRECTORY)
set(_core_rpath "@loader_path/../mlx/lib")
if (NOT _mlx_libdir MATCHES "/mlx/lib$")
    list(APPEND _core_rpath "${_mlx_libdir}")
endif ()
set_target_properties(_core PROPERTIES
    BUILD_WITH_INSTALL_RPATH ON
    INSTALL_RPATH "${_core_rpath}")

install(TARGETS _core LIBRARY DESTINATION metal_linalg)

# The MLX version, readable before _core is loaded (see metal_linalg/__init__.py).
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/_build.py"
     "# Written by the build.\nmlx_version = \"${MLX_VERSION}\"\n")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/_build.py" DESTINATION metal_linalg)
