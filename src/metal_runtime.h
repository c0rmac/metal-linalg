#pragma once

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include <mlx/mlx.h>

#include <vector>

namespace metal_linalg::detail {

// Rounds `v` up to the next multiple of `multiple`.
uint pad_up(uint v, uint multiple);

// Returns `a` as an evaluated, row-contiguous float32 array whose data starts
// on a page boundary, i.e. exactly what newBufferWithBytesNoCopy can wrap.
//
// The evaluation has to come first. An unevaluated MLX array reports itself
// as row-contiguous (its flags default to true), so checking flags on a lazy
// transposed view and then evaluating hands back the *un-transposed* buffer.
mlx::core::array prepare_input(const mlx::core::array& a);

// As prepare_input, with every matrix of the batch scaled by a power of two
// so that its largest entry lies in [0.5, 1).
//
// The Householder kernels compare squared column norms with an absolute
// threshold, and a squared norm under- or overflows float32 well inside the
// range of a float. Unscaled, that made QR lose accuracy from entries around
// 1e-3 (relative error 4e-3 at 64 x 64), fail outright below 1e-5, and return
// NaN above 1e+18. A power of two is exact, so scaling costs no accuracy: Q is
// unchanged and R is recovered by multiplying with `unscale`.
//
// A matrix that is all zero or contains a non-finite entry is left unscaled.
struct ScaledInput {
    mlx::core::array  a;          // evaluated, row-contiguous, page-aligned, scaled
    mlx::core::array  unscale;    // [batch dims..., 1, 1]: multiply R by this
    bool              scaled;     // false if every factor is 1, and unscale can be skipped
    std::vector<char> nonfinite;  // [batch]: 1 where the matrix holds a NaN or an infinity
};
ScaledInput prepare_input_scaled(const mlx::core::array& a);

struct EmbeddedShader;

// Device, command queue and shader library for one embedded metallib
// (src/shaders.h). One instance per shader, shared by every backend that
// uses that library.
//
// This deliberately lives in a single translation unit. Each backend used to
// carry its own copy of this scaffolding under the same namespace; the inline
// members then collided at link time, one definition won for the whole binary,
// and `qr_streaming_amx_complete` silently ran the *reduced* metallib.
struct MetalRuntime {
    id<MTLDevice>       device;
    id<MTLCommandQueue> queue;
    id<MTLLibrary>      library;

    // `tag` only decorates error messages, e.g. "qr_unblocked".
    static MetalRuntime& shared(const EmbeddedShader& shader, const char* tag);
};

// Compiles `name` from `library` into a pipeline state. Pass `constants` for
// functions declared with function constants, or nil for plain kernels.
id<MTLComputePipelineState> make_pipeline(id<MTLDevice> device,
                                          id<MTLLibrary> library,
                                          NSString* name,
                                          MTLFunctionConstantValues* constants);

} // namespace metal_linalg::detail
