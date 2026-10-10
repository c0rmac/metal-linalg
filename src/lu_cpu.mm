// LU, solve and inverse on the CPU: LAPACK's sgetrf, sgetrs and sgetri
// (Accelerate), the matrices of a batch spread over the cores
// (lapack_batches); one large matrix gets Accelerate's own threads. The route
// for what the GPU does not win; see LuPolicy.
//
// LAPACK is column-major and a row-major matrix read column-major is its
// transpose, whose LU is not A's, so each matrix goes into a column-major
// copy and its results come out transposed (transpose.h). The copy's leading
// dimension is padded off a power of two: sgetrf at 4096 took 169 ms at a
// leading dimension of 4096 and 62 at 4112 (macOS 27, M5 Pro).
#ifndef ACCELERATE_NEW_LAPACK
#define ACCELERATE_NEW_LAPACK   // LAPACK's current interface; before any Accelerate header
#endif
#include <metal_linalg/core.h>
#include "metal_runtime.h"
#include "transpose.h"

#include <Accelerate/Accelerate.h>

#include <algorithm>
#include <cstring>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using metal_linalg::core::Matrices;
using metal_linalg::detail::lapack_batches;
using metal_linalg::detail::padded_ld;
using metal_linalg::detail::transpose_scaled;

namespace metal_linalg::core::detail {
namespace {

// As cholesky_cpu.mm: above this the matrices go one at a time with
// Accelerate's own threads, rather than several LAPACK calls at once with its
// threading off (which corrupted spotrf's factors from n ~ 1536).
constexpr uint32_t kSerialMinN = 1024;

void require_square(const Matrices& a, const char* what) {
    if (a.rows != a.cols)
        throw std::invalid_argument(std::string("[") + what + "] cpu: the matrices must be square, got " +
                                    std::to_string(a.rows) + "x" + std::to_string(a.cols) + ".");
}

// Each thread's scratch, kept between calls (a fresh one's page faults cost
// as much as the copies).
struct Scratch {
    std::vector<float> w, b, work;
    std::vector<__LAPACK_int> ipiv;
};
Scratch& scratch(size_t w, size_t b, size_t work, size_t n) {
    thread_local Scratch s;
    if (s.w.size() < w) s.w.resize(w);
    if (s.b.size() < b) s.b.resize(b);
    if (s.work.size() < work) s.work.resize(work);
    if (s.ipiv.size() < n) s.ipiv.resize(n);
    return s;
}

// in (rows x cols, row stride ldi) transposed into out (row stride ldo): on
// every core for a matrix solved alone (above kSerialMinN, where the batch
// does not occupy them), else on this thread.
void transpose(const float* in, size_t ldi, float* out, size_t ldo, uint32_t rows, uint32_t cols) {
    if (rows > kSerialMinN || cols > kSerialMinN) transpose_scaled(in, ldi, out, ldo, rows, cols, 1.0f);
    else metal_linalg::detail::transpose(in, ldi, out, ldo, rows, cols);
}

// The matrix at src into s.w column-major (leading dimension ld) and
// factored: sgetrf's info (k > 0: U(k, k) is exactly zero).
__LAPACK_int factor(const float* src, uint32_t n, uint32_t ld, Scratch& s) {
    transpose(src, n, s.w.data(), ld, n, n);
    __LAPACK_int ln = (__LAPACK_int)n, lld = (__LAPACK_int)ld, err = 0;
    sgetrf_(&ln, &ln, s.w.data(), &lld, s.ipiv.data(), &err);
    if (err < 0) throw std::runtime_error("[lu] cpu: sgetrf rejected argument " + std::to_string(-err));
    return err;
}

void run(uint32_t n, uint32_t batch, const std::function<void(uint32_t, uint32_t)>& solve) {
    if (n > kSerialMinN) solve(0, batch);
    else lapack_batches(batch, (size_t)n * n, solve);
}

} // namespace

void lu_factor_cpu(const Matrices& a, float* lu, uint32_t* pivots, uint32_t* info) {
    require_square(a, "lu_factor");
    const uint32_t n = a.cols, batch = a.batch;
    if (n == 0 || batch == 0) return;
    const size_t per = (size_t)n * n;
    const uint32_t ld = padded_ld(n);
    run(n, batch, [&](uint32_t b0, uint32_t b1) {
        Scratch& s = scratch((size_t)n * ld, 0, 0, n);
        for (uint32_t b = b0; b < b1; ++b) {
            const __LAPACK_int err = factor(a.data + b * per, n, ld, s);
            transpose(s.w.data(), ld, lu + b * per, n, n, n);
            for (uint32_t k = 0; k < n; ++k) pivots[(size_t)b * n + k] = (uint32_t)(s.ipiv[k] - 1);
            if (info) info[b] = (uint32_t)err;
        }
    });
}

void solve_cpu(const Matrices& a, const float* bm, uint32_t nrhs, float* x, uint32_t* info) {
    require_square(a, "solve");
    const uint32_t n = a.cols, batch = a.batch;
    if (n == 0 || batch == 0 || nrhs == 0) {
        if (info) std::fill(info, info + batch, 0u);
        return;
    }
    const size_t per = (size_t)n * n, bper = (size_t)n * nrhs;
    const uint32_t ld = padded_ld(n);
    run(n, batch, [&](uint32_t b0, uint32_t b1) {
        Scratch& s = scratch((size_t)n * ld, (size_t)nrhs * ld, 0, n);
        for (uint32_t b = b0; b < b1; ++b) {
            const __LAPACK_int err = factor(a.data + b * per, n, ld, s);
            float* out = x + b * bper;
            if (info) info[b] = (uint32_t)err;
            if (err > 0) {   // singular: no solution, as sgesv
                std::fill(out, out + bper, std::numeric_limits<float>::quiet_NaN());
                continue;
            }
            // B column-major: one column is already, more are transposed
            if (nrhs == 1) std::memcpy(s.b.data(), bm + b * bper, n * sizeof(float));
            else transpose(bm + b * bper, nrhs, s.b.data(), ld, n, nrhs);
            char trans = 'N';
            __LAPACK_int ln = (__LAPACK_int)n, lr = (__LAPACK_int)nrhs, lld = (__LAPACK_int)ld, e2 = 0;
            sgetrs_(&trans, &ln, &lr, s.w.data(), &lld, s.ipiv.data(), s.b.data(), &lld, &e2);
            if (nrhs == 1) std::memcpy(out, s.b.data(), n * sizeof(float));
            else transpose(s.b.data(), ld, out, nrhs, nrhs, n);
        }
    });
}

void inv_cpu(const Matrices& a, float* x, uint32_t* info) {
    require_square(a, "inv");
    const uint32_t n = a.cols, batch = a.batch;
    if (n == 0 || batch == 0) return;
    const size_t per = (size_t)n * n;
    const uint32_t ld = padded_ld(n);
    const size_t lwork = (size_t)n * 64;   // sgetri's blocked form wants n * its block size
    run(n, batch, [&](uint32_t b0, uint32_t b1) {
        Scratch& s = scratch((size_t)n * ld, 0, lwork, n);
        for (uint32_t b = b0; b < b1; ++b) {
            const __LAPACK_int err = factor(a.data + b * per, n, ld, s);
            float* out = x + b * per;
            if (info) info[b] = (uint32_t)err;
            if (err > 0) {
                std::fill(out, out + per, std::numeric_limits<float>::quiet_NaN());
                continue;
            }
            __LAPACK_int ln = (__LAPACK_int)n, lld = (__LAPACK_int)ld, lw = (__LAPACK_int)lwork, e2 = 0;
            sgetri_(&ln, s.w.data(), &lld, s.ipiv.data(), s.work.data(), &lw, &e2);
            transpose(s.w.data(), ld, out, n, n, n);
        }
    });
}

} // namespace metal_linalg::core::detail
