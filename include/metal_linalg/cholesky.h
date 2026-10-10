#pragma once
#include <metal_linalg/core.h>   // CholeskyPolicy, cholesky_backend

#include <mlx/mlx.h>

namespace metal_linalg {

    // -------------------------------------------------------------------------
    // Cholesky factorization on the GPU (since 2.18.0)
    // -------------------------------------------------------------------------
    // For a batch of real symmetric positive definite matrices A [..., N, N],
    // returns L [..., N, N], lower triangular with a positive diagonal, such
    // that A = L L^T; with `upper`, U = L^T, so that A = U^T U. Same contract
    // as mlx::core::linalg::cholesky, which runs on the CPU alone.
    //
    // Only one triangle of the input is read (the lower; the upper with
    // `upper`), and the other triangle of the output is zero. Output is
    // float32. Batch dimensions are arbitrary.
    //
    // A matrix that is not positive definite (or holds a NaN or an infinity
    // that reaches a pivot) gives an all-NaN result for that matrix alone;
    // cholesky_ex_accelerated says which and where. Throws
    // std::invalid_argument for a non-square input.
    mlx::core::array cholesky_accelerated(const mlx::core::array& a, bool upper = false);

    struct CholeskyResult {
        mlx::core::array l;      // [..., N, N]
        mlx::core::array info;   // uint32 [...]: 0, or k where the leading minor of order k is not positive definite
    };

    // As cholesky_accelerated, with LAPACK's `info` for each matrix, as
    // torch.linalg.cholesky_ex returns it.
    CholeskyResult cholesky_ex_accelerated(const mlx::core::array& a, bool upper = false);

    // Each backend on its own, as described in core.h, for tests and tuning.
    namespace detail {
        CholeskyResult cholesky_simd(const mlx::core::array& a, bool upper = false);
        CholeskyResult cholesky_threadgroup(const mlx::core::array& a, bool upper = false);
        CholeskyResult cholesky_blocked(const mlx::core::array& a, bool upper = false);
        CholeskyResult cholesky_cpu(const mlx::core::array& a, bool upper = false);
    }

} // namespace metal_linalg
