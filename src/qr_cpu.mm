// QR on the CPU: LAPACK (Accelerate) sgeqrf and sorgqr, the matrices of a batch
// spread over the cores (lapack_batches). R alone skips sorgqr; Q square (a
// tall matrix's complete mode) has sorgqr form all M columns.
// The route for calls too small to pay for a GPU launch; see QrPolicy.
//
// A wide matrix (M < N) is factored by its leading M x M block, A1 = Q R1,
// and R2 = Q^T A2 by one matrix product: the same reflectors and the same R
// as sgeqrf on the whole matrix, which Accelerate ran 10-40x slower (on an
// M5 Pro one 64 x 2048 in 1.47 ms against 0.063 ms).
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
using metal_linalg::core::QrMode;
using metal_linalg::detail::Part;
using metal_linalg::detail::lapack_batches;
using metal_linalg::detail::scan;

namespace metal_linalg::core::detail {

void qr_cpu(const Matrices& a, float* q_out, float* r_out, QrMode mode) {
    const uint32_t M = a.rows, N = a.cols, K = std::min(M, N), batch = a.batch;
    if (K == 0 || batch == 0) return;
    // Q's columns (K, M, or none) and R's rows (K, or M with zeros below K)
    const uint32_t QC = core::qr_q_cols(mode, M, N), RR = core::qr_r_rows(mode, M, N);

    // LAPACK is column-major, so each matrix is transposed into a column-major
    // copy of itself first; that is also the fast route in Accelerate, whose
    // QR routines run at about twice the speed of its LQ ones (see the SVD's
    // CPU path). sgeqrf leaves R in the upper triangle, read off before
    // sorgqr overwrites the copy with Q's K columns. A wide matrix factors
    // only its leading K x K block that way (see the top of the file).
    const bool wide = M < N;
    const uint32_t lcols = wide ? K : N;   // the columns LAPACK factors
    const uint32_t wcols = std::max(lcols, QC);   // and the work array's: Q's M for a square Q
    __LAPACK_int lm = (__LAPACK_int)M, ln = (__LAPACK_int)lcols, lk = (__LAPACK_int)K, err = 0;
    __LAPACK_int lq = (__LAPACK_int)std::max(QC, K);   // Q's columns for sorgqr (K for R alone's wide case)
    const size_t per = (size_t)M * N;

    // Workspace size, by query (LAPACK reads no array during one).
    __LAPACK_int lwork = 1, query = -1;
    float q = 0.0f, scratch = 0.0f;
    sgeqrf_(&lm, &ln, &scratch, &lm, &scratch, &q, &query, &err);
    lwork = std::max<__LAPACK_int>(lwork, (__LAPACK_int)std::ceil(q));
    sorgqr_(&lm, &lq, &lk, &scratch, &lm, &scratch, &q, &query, &err);
    lwork = std::max<__LAPACK_int>(lwork, (__LAPACK_int)std::ceil(q));

    // Non-finite input gives NaN for that matrix, as the eigensolver and the
    // SVD do, rather than whatever LAPACK makes of it.
    std::vector<float> amax(batch);
    std::vector<char>  finite(batch);
    scan(a, Part::all, amax.data(), finite.data());

    // Each chunk of the batch, on its own thread with its own workspace.
    lapack_batches(batch, per, [&](uint32_t b0, uint32_t b1) {
        __LAPACK_int lw = lwork, err = 0;
        std::vector<float> work_a((size_t)M * wcols), tau(K), work(lwork);
        for (uint32_t b = b0; b < b1; ++b) {
            float* qb = QC ? q_out + (size_t)b * M * QC : nullptr;
            float* rb = r_out + (size_t)b * RR * N;
            if (!finite[b]) {
                if (qb) std::fill(qb, qb + (size_t)M * QC, NAN);
                std::fill(rb, rb + (size_t)RR * N, NAN);
                continue;
            }
            const float* ab = a.data + b * per;
            if (!wide) {
                vDSP_mtrans(ab, 1, work_a.data(), 1, N, M);   // column-major A, M x N
            } else {
                for (uint32_t j = 0; j < K; ++j)              // column-major A1, the leading K x K
                    for (uint32_t i = 0; i < M; ++i) work_a[i + (size_t)j * M] = ab[(size_t)i * N + j];
            }
            sgeqrf_(&lm, &ln, work_a.data(), &lm, tau.data(), work.data(), &lw, &err);
            if (err != 0) {
                throw std::runtime_error("[qr] LAPACK sgeqrf failed on matrix " + std::to_string(b) +
                                         " of " + std::to_string(batch) + " (" + std::to_string(M) +
                                         "x" + std::to_string(N) + "), info " +
                                         std::to_string((long long)err) + ".");
            }
            for (uint32_t i = 0; i < K; ++i)          // R, row-major K x N, upper trapezoidal (R1 if wide)
                for (uint32_t j = 0; j < lcols; ++j)
                    rb[(size_t)i * N + j] = j >= i ? work_a[i + (size_t)j * M] : 0.0f;
            std::fill(rb + (size_t)K * N, rb + (size_t)RR * N, 0.0f);   // a square Q's R: zero rows below K
            if (!qb) {   // R alone
                if (wide) {
                    // R2 = Q^T A2 needs Q: formed in the work array all the same
                    sorgqr_(&lm, &lk, &lk, work_a.data(), &lm, tau.data(), work.data(), &lw, &err);
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (__LAPACK_int)K, (__LAPACK_int)(N - K),
                                (__LAPACK_int)K, 1.0f, work_a.data(), (__LAPACK_int)M, ab + K, (__LAPACK_int)N, 0.0f,
                                rb + K, (__LAPACK_int)N);
                }
                continue;
            }
            sorgqr_(&lm, &lq, &lk, work_a.data(), &lm, tau.data(), work.data(), &lw, &err);
            if (err != 0) {
                throw std::runtime_error("[qr] LAPACK sorgqr failed on matrix " + std::to_string(b) +
                                         " of " + std::to_string(batch) + ", info " +
                                         std::to_string((long long)err) + ".");
            }
            vDSP_mtrans(work_a.data(), 1, qb, 1, M, QC);                  // row-major Q, M x QC
            if (wide) {
                // R2 = Q^T A2, K x (N - K), into R's last columns: the
                // column-major Q read row-major is Q^T.
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (__LAPACK_int)K, (__LAPACK_int)(N - K),
                            (__LAPACK_int)K, 1.0f, work_a.data(), (__LAPACK_int)M, ab + K, (__LAPACK_int)N, 0.0f,
                            rb + K, (__LAPACK_int)N);
            }
        }
    });
}

void qr_cpu(const Matrices& a, float* q, float* r) { qr_cpu(a, q, r, QrMode::reduced); }

} // namespace metal_linalg::core::detail
