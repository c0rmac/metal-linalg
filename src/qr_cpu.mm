// QR on the CPU: LAPACK (Accelerate) sgeqrf and sorgqr, one matrix at a time.
// The route for calls too small to pay for a GPU launch; see QrPolicy.
#ifndef ACCELERATE_NEW_LAPACK
#define ACCELERATE_NEW_LAPACK   // LAPACK's current interface; before any Accelerate header
#endif
#include <metal_linalg/core.h>
#include "metal_runtime.h"

#include <Accelerate/Accelerate.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

using metal_linalg::core::Matrices;
using metal_linalg::detail::Part;
using metal_linalg::detail::scan;

namespace metal_linalg::core::detail {

void qr_cpu(const Matrices& a, float* q_out, float* r_out) {
    const uint32_t M = a.rows, N = a.cols, K = std::min(M, N), batch = a.batch;
    if (K == 0 || batch == 0) return;

    // LAPACK is column-major, so each matrix is transposed into a column-major
    // copy of itself first; that is also the fast route in Accelerate, whose
    // QR routines run at about twice the speed of its LQ ones (see the SVD's
    // CPU path). sgeqrf leaves R in the upper triangle, read off before
    // sorgqr overwrites the copy with Q's K columns.
    __LAPACK_int lm = (__LAPACK_int)M, ln = (__LAPACK_int)N, lk = (__LAPACK_int)K, err = 0;
    const size_t per = (size_t)M * N;
    std::vector<float> work_a(per), tau(K);

    __LAPACK_int lwork = 1, query = -1;
    float q = 0.0f;
    sgeqrf_(&lm, &ln, work_a.data(), &lm, tau.data(), &q, &query, &err);
    lwork = std::max<__LAPACK_int>(lwork, (__LAPACK_int)std::ceil(q));
    sorgqr_(&lm, &lk, &lk, work_a.data(), &lm, tau.data(), &q, &query, &err);
    lwork = std::max<__LAPACK_int>(lwork, (__LAPACK_int)std::ceil(q));
    std::vector<float> work(lwork);

    // Non-finite input gives NaN for that matrix, as the eigensolver and the
    // SVD do, rather than whatever LAPACK makes of it.
    std::vector<float> amax(batch);
    std::vector<char>  finite(batch);
    scan(a, Part::all, amax.data(), finite.data());

    for (uint32_t b = 0; b < batch; ++b) {
        float* qb = q_out + (size_t)b * M * K;
        float* rb = r_out + (size_t)b * K * N;
        if (!finite[b]) {
            std::fill(qb, qb + (size_t)M * K, NAN);
            std::fill(rb, rb + (size_t)K * N, NAN);
            continue;
        }
        vDSP_mtrans(a.data + b * per, 1, work_a.data(), 1, N, M);   // column-major A, M x N
        sgeqrf_(&lm, &ln, work_a.data(), &lm, tau.data(), work.data(), &lwork, &err);
        if (err != 0) {
            throw std::runtime_error("[qr] LAPACK sgeqrf failed on matrix " + std::to_string(b) +
                                     " of " + std::to_string(batch) + " (" + std::to_string(M) +
                                     "x" + std::to_string(N) + "), info " +
                                     std::to_string((long long)err) + ".");
        }
        for (uint32_t i = 0; i < K; ++i)          // R, row-major K x N, upper trapezoidal
            for (uint32_t j = 0; j < N; ++j)
                rb[(size_t)i * N + j] = j >= i ? work_a[i + (size_t)j * M] : 0.0f;
        sorgqr_(&lm, &lk, &lk, work_a.data(), &lm, tau.data(), work.data(), &lwork, &err);
        if (err != 0) {
            throw std::runtime_error("[qr] LAPACK sorgqr failed on matrix " + std::to_string(b) +
                                     " of " + std::to_string(batch) + ", info " +
                                     std::to_string((long long)err) + ".");
        }
        vDSP_mtrans(work_a.data(), 1, qb, 1, M, K);                   // row-major Q, M x K
    }
}

} // namespace metal_linalg::core::detail
