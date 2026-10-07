// The eigensolver's `band` backend, eigenvalues alone: the two-stage
// reduction, as LAPACK's ssyevd_2stage. A to a band of width b on the GPU
// (band_reduce.mm), the band to tridiagonal on the CPU's cores (band_chase.cpp),
// then its eigenvalues by bisection on the GPU (bisect.mm).
//
// Why: the `tridiag` backend's reduction is a symmetric matrix-vector product
// a column, bound by memory bandwidth, and the CPU's eigenvalue path already
// reduces in two stages; on an M5 Pro the one-stage GPU path led it by only
// 1.1-1.6x. The band reduction reads the matrix three times a block of b
// columns instead of once a column, and the chase runs on every core.
//
// A batch is pipelined over two slots: while the CPU chases and solves one
// matrix, the GPU reduces the next. Each matrix is scaled by a power of two
// first (exact), so magnitudes whose products over- or underflow float32 work.

#ifndef ACCELERATE_NEW_LAPACK
#define ACCELERATE_NEW_LAPACK
#endif
#include <Accelerate/Accelerate.h>

#include <metal_linalg/core.h>
#include <metal_linalg/device.h>
#include "band_chase.h"
#include "metal_runtime.h"

#import <Metal/Metal.h>

#include <algorithm>
#include <cmath>
#include <future>
#include <stdexcept>
#include <string>
#include <vector>

using metal_linalg::core::Matrices;
using metal_linalg::detail::AutoreleasePool;
using metal_linalg::detail::Part;
using metal_linalg::detail::scan;
using metal_linalg::detail::transpose_scaled;

namespace metal_linalg {
namespace {

using L = __LAPACK_int;

// Two slots of n x n shared storage, the latest n's kept.
struct Slots {
    uint32_t      n = 0, lda = 0;
    id<MTLBuffer> A[2];

    void ensure(uint32_t order) {
        if (n == order && A[0]) return;
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        n = order;
        lda = (order + 7) / 8 * 8;
        for (auto& a : A)
            a = [dev newBufferWithLength:std::max<size_t>((size_t)lda * n, 4) * 4 options:MTLResourceStorageModeShared];
    }
};

// A(c, r) = A(r, c) for r > c: the lower triangle mirrored above, in blocks
// on every core.
void mirror_lower(float* A, size_t lda, uint32_t n) {
    constexpr uint32_t kBlock = 32;
    const uint32_t blocks = (n + kBlock - 1) / kBlock;
    detail::parallel_for(blocks, [&](size_t jb) {
        const uint32_t c0 = (uint32_t)jb * kBlock, c1 = std::min(n, c0 + kBlock);
        for (uint32_t r0 = c0; r0 < n; r0 += kBlock) {
            const uint32_t r1 = std::min(n, r0 + kBlock);
            for (uint32_t c = c0; c < c1; ++c)
                for (uint32_t r = std::max(r0, c + 1); r < r1; ++r) A[(size_t)r * lda + c] = A[(size_t)c * lda + r];
        }
    });
}

} // namespace

namespace core::detail {

void eigh_band(const Matrices& a, bool lower, float* w_out, uint32_t* info_out, uint32_t width) {
    const uint32_t n = a.cols, batch = a.batch;
    if (a.rows != n) throw std::invalid_argument("[eigh] Input matrices must be square.");
    if (n == 0 || batch == 0) {
        if (info_out) std::fill(info_out, info_out + batch, 0u);
        return;
    }
    // The band's width: as asked, narrower if n is too large for the panel
    // kernels at that width, and if even the narrowest is, the tridiag backend.
    const uint32_t b = metal_linalg::detail::band_fit(n, metal_linalg::detail::band_width(width, "EIGH_BAND_WIDTH"));
    if (b == 0) {
        eigh_tridiag(a, lower, w_out, nullptr, info_out);
        return;
    }
    AutoreleasePool pool;
    static Slots slots;
    slots.ensure(n);
    const size_t per = (size_t)n * n;

    std::vector<float> amax(batch);
    std::vector<char>  finite(batch);
    scan(a, lower ? Part::lower : Part::upper, amax.data(), finite.data());
    std::vector<uint32_t> todo;
    for (uint32_t m = 0; m < batch; ++m) {
        if (finite[m]) { todo.push_back(m); continue; }
        std::fill(w_out + (size_t)m * n, w_out + (size_t)(m + 1) * n, NAN);
        if (info_out) info_out[m] = 1u << 17;
    }
    if (todo.empty()) return;

    // Per slot: the band, with room for the chase's bulges (ld = 2b + 1).
    struct Work {
        std::vector<float> band, d, e;
        float scale = 1.0f;
    };
    Work work[2];
    const size_t ld = 2 * (size_t)b + 1;
    for (Work& wk : work) { wk.d.resize(n); wk.e.resize(n); }

    // Stage 1 for matrix m in slot s: scale, copy in both triangles, reduce.
    auto reduce = [&](uint32_t m, int s) {
        Work& wk = work[s];
        int ex = 0;
        if (amax[m] > 0.0f) std::frexp(amax[m], &ex);
        wk.scale = std::ldexp(1.0f, -ex);
        const float* src = a.data + m * per;
        float* A = static_cast<float*>(slots.A[s].contents);
        const size_t lda = slots.lda;
        // The given triangle as the column-major lower one, then mirrored.
        if (lower) {
            transpose_scaled(src, n, A, lda, n, n, wk.scale);
        } else {
            metal_linalg::detail::parallel_for(n, [&](size_t i) {
                const float* row = src + i * n;
                float* col = A + i * lda;
                for (uint32_t j = (uint32_t)i; j < n; ++j) col[j] = row[j] * wk.scale;
            });
        }
        mirror_lower(A, lda, n);
        if (!metal_linalg::detail::band_reduce_symmetric(slots.A[s], n, (uint32_t)lda, b))
            throw std::logic_error("[eigh] band: band_fit and band_reduce_symmetric disagree.");
        wk.band.assign(ld * n, 0.0f);
        for (uint32_t c = 0; c < n; ++c)
            for (uint32_t r = c; r < n && r <= c + b; ++r) wk.band[(size_t)c * ld + r - c] = A[(size_t)c * lda + r];
    };

    // Stage 2 for matrix m in slot s, on the CPU: the chase (on the cores but
    // two, which the GPU's host work keeps), then ssterf.
    auto solve = [&](uint32_t m, int s) {
        Work& wk = work[s];
        metal_linalg::detail::band_to_tridiagonal(n, b, wk.band.data(), ld, wk.d.data(), wk.e.data(),
                                                  metal_linalg::detail::cpu_threads_beside_gpu());
        L N = n, info = 0;
        // By bisection on the GPU where it is the faster, else ssterf.
        std::vector<float> wb(n);
        if (metal_linalg::detail::tridiagonal_eigenvalues(n, wk.d.data(), wk.e.data(), wb.data()))
            wk.d = std::move(wb);
        else
            ssterf_(&N, wk.d.data(), wk.e.data(), &info);
        if (info != 0) {
            throw std::runtime_error("[eigh] band: LAPACK ssterf failed on matrix " + std::to_string(m) + " (N=" +
                                     std::to_string(n) + "), info " + std::to_string((long long)info) + ".");
        }
        float* w = w_out + (size_t)m * n;
        for (uint32_t i = 0; i < n; ++i) w[i] = wk.d[i] / wk.scale;   // ascending
        if (info_out) info_out[m] = 1u | (1u << 16);   // converged; one "sweep", as the CPU path
    };

    // The pipeline: matrix t in slot t % 2; the CPU solves t while the GPU
    // reduces t + 1.
    const size_t count = todo.size();
    reduce(todo[0], 0);
    for (size_t t = 0; t < count; ++t) {
        const int s = (int)(t % 2);
        std::future<void> solving = std::async(std::launch::async, solve, todo[t], s);
        if (t + 1 < count) {
            try {
                reduce(todo[t + 1], s ^ 1);
            } catch (...) {
                solving.wait();
                throw;
            }
        }
        solving.get();
    }
}

} // namespace core::detail
} // namespace metal_linalg
