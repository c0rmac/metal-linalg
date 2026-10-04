// Eigenvalues of a symmetric tridiagonal, and singular values of a bidiagonal,
// by bisection on the GPU (sturm_bisect in shaders/Eigh_Tridiag.metal): the
// last step of the eigenvalue-only and singular-value-only paths of the
// tridiag, bidiag and band backends.
//
// Why: LAPACK's ssterf and sbdsqr (dqds) are sequential, O(n^2): on an M5 Pro
// 79 and 77 ms at n = 4096, 305 and 300 at 8192, about a quarter of what is
// left of eigvalsh and svdvals after their reductions moved to the GPU. A
// thread an eigenvalue, the n of them in parallel: 6 and 12 ms at 4096, 16 and
// 31 at 8192, to the same accuracy (absolute, a few float32 ulps of the
// matrix's norm). Below about 512 (tridiagonal) or 1024 (bidiagonal) LAPACK is
// the faster and is used.
//
// A queue of its own, so that a batch's CPU stage, which calls this, overlaps
// the GPU's reduction of the next matrix instead of queueing behind it.

#ifndef ACCELERATE_NEW_LAPACK
#define ACCELERATE_NEW_LAPACK
#endif
#include <Accelerate/Accelerate.h>

#include "metal_runtime.h"
#include "shaders.h"

#import <Metal/Metal.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace metal_linalg::detail {
namespace {

using L = __LAPACK_int;

constexpr uint32_t kTridiagonalMinN = 512;    // GPU bisection from this order
constexpr uint32_t kBidiagonalMinN  = 1024;
constexpr uint32_t kThreads         = 64;

// Must match SturmParams in Eigh_Tridiag.metal.
struct SturmParams {
    uint32_t n, size, offset, tgk;
    float    lo, hi, pivmin;
    uint32_t passes;
};

struct Bisector {
    MetalRuntime& rt = MetalRuntime::shared(METAL_LINALG_SHADER(Eigh_Tridiag), "eigh_tridiag");
    id<MTLCommandQueue>         queue = [rt.device newCommandQueue];
    id<MTLComputePipelineState> pso = make_pipeline(rt.device, rt.library, @"sturm_bisect", nil);
    std::mutex                  lock;   // the buffers; calls may come from a batch's worker thread

    static Bisector& shared() {
        static Bisector b;
        return b;
    }

    // The `count` eigenvalues from index `offset` (ascending) of the
    // tridiagonal of order `size` (d, or none: tgk) with squared
    // off-diagonal e2, all in [lo, hi].
    void run(uint32_t size, const float* d, const std::vector<float>& e2, bool tgk, uint32_t offset, uint32_t count,
             float lo, float hi, float* out) {
        std::lock_guard<std::mutex> guard(lock);
        AutoreleasePool pool;
        float emax = 0.0f;
        for (float x : e2) emax = std::max(emax, x);
        const float tol = 2.0f * FLT_EPSILON * std::max(std::fabs(lo), std::fabs(hi));
        uint32_t passes = 1;
        for (float r = hi - lo; r > tol && passes < 64; r *= 0.5f) ++passes;
        const SturmParams p{count, size, offset, tgk ? 1u : 0u, lo, hi, FLT_MIN * std::max(1.0f, emax), passes};
        id<MTLDevice> dev = rt.device;
        const MTLResourceOptions opt = MTLResourceStorageModeShared;
        id<MTLBuffer> bd = [dev newBufferWithBytes:(tgk ? e2.data() : d) length:std::max<size_t>(4, (tgk ? 1 : size) * 4)
                                           options:opt];
        id<MTLBuffer> be = [dev newBufferWithBytes:e2.data() length:std::max<size_t>(4, e2.size() * 4) options:opt];
        id<MTLBuffer> bo = [dev newBufferWithLength:std::max<size_t>(4, (size_t)count * 4) options:opt];
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:bd offset:0 atIndex:0];
        [enc setBuffer:be offset:0 atIndex:1];
        [enc setBuffer:bo offset:0 atIndex:2];
        [enc setBytes:&p length:sizeof p atIndex:3];
        [enc dispatchThreads:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(kThreads, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.error)
            throw std::runtime_error(std::string("[bisect] GPU error: ") + cb.error.localizedDescription.UTF8String);
        const float* r = static_cast<const float*>(bo.contents);
        std::copy(r, r + count, out);
    }
};

} // namespace

bool tridiagonal_eigenvalues(uint32_t n, const float* d, const float* e, float* w) {
    if (n < kTridiagonalMinN) return false;
    std::vector<float> e2(n - 1);
    float lo = INFINITY, hi = -INFINITY;
    for (uint32_t i = 0; i < n; ++i) {
        const float r = (i > 0 ? std::fabs(e[i - 1]) : 0.0f) + (i + 1 < n ? std::fabs(e[i]) : 0.0f);
        lo = std::min(lo, d[i] - r);
        hi = std::max(hi, d[i] + r);
        if (i + 1 < n) e2[i] = e[i] * e[i];
    }
    const float pad = 2.0f * FLT_EPSILON * n * std::max(std::fabs(lo), std::fabs(hi)) + FLT_MIN;
    Bisector::shared().run(n, d, e2, false, 0, n, lo - pad, hi + pad, w);
    std::sort(w, w + n);   // ascending; neighbours within the tolerance may come out swapped
    return true;
}

bool bidiagonal_singular_values(uint32_t n, const float* d, const float* e, float* s) {
    if (n < kBidiagonalMinN) return false;
    // The Golub-Kahan form: off-diagonal d0, e0, d1, e1, ..., d(n-1); its
    // eigenvalues from index n up are the singular values, ascending.
    std::vector<float> e2(2 * (size_t)n - 1);
    float hi = 0.0f;
    for (uint32_t i = 0; i < n; ++i) {
        e2[2 * i] = d[i] * d[i];
        if (i + 1 < n) e2[2 * i + 1] = e[i] * e[i];
        const float r = std::max(i > 0 ? std::fabs(e[i - 1]) : 0.0f, i + 1 < n ? std::fabs(e[i]) : 0.0f);
        hi = std::max(hi, std::fabs(d[i]) + r);
    }
    hi = hi * (1.0f + 4.0f * FLT_EPSILON * n) + FLT_MIN;
    Bisector::shared().run(2 * n, nullptr, e2, true, n, n, 0.0f, hi, s);
    std::sort(s, s + n, std::greater<float>());   // descending, as LAPACK's
    return true;
}

} // namespace metal_linalg::detail
