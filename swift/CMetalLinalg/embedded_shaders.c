// The compiled shaders for the Swift package, which cannot run CMake's
// embedding step (cmake/EmbedMetallib.cmake): the metallibs committed under
// shaders/prebuilt/, pulled in with C23's #embed. After editing a shader,
// refresh those (`cmake --build build --target update_prebuilt_shaders`).
// Defines the symbols src/shaders.h declares.
#pragma clang diagnostic ignored "-Wc23-extensions"

#include <stddef.h>

const unsigned char metal_linalg_QR_Unblocked_metallib[] = {
#embed "../../shaders/prebuilt/QR_Unblocked.metallib"
};
const size_t metal_linalg_QR_Unblocked_metallib_len = sizeof(metal_linalg_QR_Unblocked_metallib);

const unsigned char metal_linalg_QR_Streaming_AMX_Reduced_metallib[] = {
#embed "../../shaders/prebuilt/QR_Streaming_AMX_Reduced.metallib"
};
const size_t metal_linalg_QR_Streaming_AMX_Reduced_metallib_len = sizeof(metal_linalg_QR_Streaming_AMX_Reduced_metallib);

const unsigned char metal_linalg_QR_Streaming_AMX_Complete_metallib[] = {
#embed "../../shaders/prebuilt/QR_Streaming_AMX_Complete.metallib"
};
const size_t metal_linalg_QR_Streaming_AMX_Complete_metallib_len = sizeof(metal_linalg_QR_Streaming_AMX_Complete_metallib);

const unsigned char metal_linalg_Eigh_Jacobi_metallib[] = {
#embed "../../shaders/prebuilt/Eigh_Jacobi.metallib"
};
const size_t metal_linalg_Eigh_Jacobi_metallib_len = sizeof(metal_linalg_Eigh_Jacobi_metallib);

const unsigned char metal_linalg_Eigh_BlockJacobi_metallib[] = {
#embed "../../shaders/prebuilt/Eigh_BlockJacobi.metallib"
};
const size_t metal_linalg_Eigh_BlockJacobi_metallib_len = sizeof(metal_linalg_Eigh_BlockJacobi_metallib);

const unsigned char metal_linalg_Svd_Jacobi_metallib[] = {
#embed "../../shaders/prebuilt/Svd_Jacobi.metallib"
};
const size_t metal_linalg_Svd_Jacobi_metallib_len = sizeof(metal_linalg_Svd_Jacobi_metallib);

const unsigned char metal_linalg_Svd_BlockJacobi_metallib[] = {
#embed "../../shaders/prebuilt/Svd_BlockJacobi.metallib"
};
const size_t metal_linalg_Svd_BlockJacobi_metallib_len = sizeof(metal_linalg_Svd_BlockJacobi_metallib);
