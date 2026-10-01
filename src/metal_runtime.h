#pragma once

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include <metal_linalg/core.h>

#include <dispatch/dispatch.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace metal_linalg::detail {

using core::Matrices;

// Rounds `v` up to the next multiple of `multiple`.
uint pad_up(uint v, uint multiple);

// Floats in a batch of matrices.
inline size_t element_count(const Matrices& a) { return (size_t)a.batch * a.rows * a.cols; }

// -----------------------------------------------------------------------------
// Inputs
// -----------------------------------------------------------------------------

// The matrices as a Metal buffer the kernels can read. Memory that starts on
// a page boundary (an MLX array's, or a large malloc) is wrapped in place;
// anything else is copied once. With `transpose`, always a copy, holding each
// matrix transposed: [batch, cols, rows].
id<MTLBuffer> input_buffer(id<MTLDevice> device, const Matrices& a, bool transpose = false);

// Which entries of each matrix scan() reads. The eigensolvers read one
// triangle, and junk in the other must not set the scale.
enum class Part { all, lower, upper };

// Largest magnitude and finiteness of each matrix, on the CPU.
void scan(const Matrices& a, Part part, float* amax, char* finite);

// As input_buffer, with every matrix scaled by a power of two so that its
// largest entry lies in [0.5, 1).
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
    id<MTLBuffer>      buffer;
    std::vector<float> unscale;    // [batch]: multiply R (or S) by this
    bool               scaled;     // false if every factor is 1, and unscale can be skipped
    std::vector<char>  nonfinite;  // [batch]: 1 where the matrix holds a NaN or an infinity
};
ScaledInput scaled_input(id<MTLDevice> device, const Matrices& a, bool transpose = false);

// -----------------------------------------------------------------------------
// Outputs and host work
// -----------------------------------------------------------------------------

// Page-aligned host memory, for intermediates the GPU reads in place.
class HostBuffer {
public:
    explicit HostBuffer(size_t floats);
    ~HostBuffer();
    HostBuffer(const HostBuffer&) = delete;
    HostBuffer& operator=(const HostBuffer&) = delete;
    float* data() { return data_; }
private:
    float* data_ = nullptr;
};

// `batch` matrices of `per` floats from src to dst, matrix b multiplied by
// factor[b]. With no factors, a plain copy.
void copy_out(const float* src, float* dst, uint32_t batch, size_t per,
              const std::vector<float>* factor = nullptr);

// Each rows x cols matrix of src, transposed into dst as cols x rows.
void transpose_out(const float* src, float* dst, uint32_t batch, uint32_t rows, uint32_t cols);

// Floats of host work worth a thread of its own.
constexpr size_t kGrain = size_t(1) << 16;

// Runs f(t) for t in [0, tasks), on the CPU's cores when there is more than one.
template <class F>
void parallel_for(size_t tasks, const F& f) {
    if (tasks <= 1) {
        if (tasks == 1) f(0);
        return;
    }
    dispatch_apply_f(tasks, DISPATCH_APPLY_AUTO, (void*)&f,
                     [](void* ctx, size_t t) { (*static_cast<const F*>(ctx))(t); });
}

// Runs f(b) for every matrix b, in parallel across groups of matrices large
// enough to be worth a thread.
template <class F>
void for_each_matrix(uint32_t batch, size_t per, const F& f) {
    const size_t group = std::max<size_t>(1, kGrain / std::max<size_t>(per, 1));
    parallel_for((batch + group - 1) / group, [&](size_t t) {
        const size_t end = std::min<size_t>(batch, (t + 1) * group);
        for (size_t b = t * group; b < end; ++b) f((uint32_t)b);
    });
}

// Runs f(b, r0, r1) over rows [r0, r1) of every matrix b, covering each
// matrix once: whole matrices, grouped, when they are small, and row blocks
// of a matrix in parallel when it is large, so a lone big matrix still uses
// every core.
template <class F>
void for_each_rows(uint32_t batch, uint32_t rows, uint32_t cols, const F& f) {
    const size_t per = (size_t)rows * cols;
    const size_t blocks = std::min<size_t>(rows, std::max<size_t>(1, per / kGrain));
    if (blocks <= 1) {
        for_each_matrix(batch, per, [&](uint32_t b) { f(b, 0u, rows); });
        return;
    }
    const uint32_t step = (uint32_t)((rows + blocks - 1) / blocks);
    const size_t per_matrix = (rows + step - 1) / step;
    parallel_for((size_t)batch * per_matrix, [&](size_t t) {
        const uint32_t b  = (uint32_t)(t / per_matrix);
        const uint32_t r0 = (uint32_t)(t % per_matrix) * step;
        f(b, r0, std::min(rows, r0 + step));
    });
}

// -----------------------------------------------------------------------------
// Metal
// -----------------------------------------------------------------------------

// The library is built with ARC; Metal still autoreleases command buffers and
// encoders. One of these at the top of every GPU entry point drains them when
// the call returns, also by an exception, so a caller without a pool of its
// own (a C++ or Python program) does not accumulate them. These two functions
// are what @autoreleasepool compiles to.
extern "C" void* objc_autoreleasePoolPush(void);
extern "C" void  objc_autoreleasePoolPop(void* token);

class AutoreleasePool {
public:
    AutoreleasePool() : token_(objc_autoreleasePoolPush()) {}
    ~AutoreleasePool() { objc_autoreleasePoolPop(token_); }
    AutoreleasePool(const AutoreleasePool&) = delete;
    AutoreleasePool& operator=(const AutoreleasePool&) = delete;
private:
    void* token_;
};

// `floats` floats of page-aligned host memory (a HostBuffer, or an
// allocation known to be aligned) as a Metal buffer, without copying.
id<MTLBuffer> wrap_host(id<MTLDevice> device, float* data, size_t floats);

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
