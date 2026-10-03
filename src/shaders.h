// The compiled Metal libraries, embedded in libmetal_linalg by CMake
// (cmake/EmbedMetallib.cmake) so that nothing is looked up on disk at run
// time. One symbol pair per shader; see metal_linalg_add_shader in
// CMakeLists.txt.
#pragma once

#include <cstddef>

#define METAL_LINALG_DECLARE_SHADER(sym)                              \
    extern "C" const unsigned char metal_linalg_##sym##_metallib[]; \
    extern "C" const size_t        metal_linalg_##sym##_metallib_len;

METAL_LINALG_DECLARE_SHADER(QR_Unblocked)
METAL_LINALG_DECLARE_SHADER(QR_Streaming_AMX_Reduced)
METAL_LINALG_DECLARE_SHADER(QR_Streaming_AMX_Complete)
METAL_LINALG_DECLARE_SHADER(Eigh_Jacobi)
METAL_LINALG_DECLARE_SHADER(Eigh_BlockJacobi)
METAL_LINALG_DECLARE_SHADER(Eigh_Tridiag)
METAL_LINALG_DECLARE_SHADER(Svd_Jacobi)
METAL_LINALG_DECLARE_SHADER(Svd_BlockJacobi)
METAL_LINALG_DECLARE_SHADER(Svd_Bidiag)

#undef METAL_LINALG_DECLARE_SHADER

namespace metal_linalg::detail {

struct EmbeddedShader {
    const unsigned char* bytes;
    size_t               len;
    const char*          name;   // e.g. "QR_Unblocked"; keys the runtime cache
};

} // namespace metal_linalg::detail

// METAL_LINALG_SHADER(QR_Unblocked) is the EmbeddedShader for that metallib.
#define METAL_LINALG_SHADER(sym) \
    (::metal_linalg::detail::EmbeddedShader{metal_linalg_##sym##_metallib, metal_linalg_##sym##_metallib_len, #sym})
