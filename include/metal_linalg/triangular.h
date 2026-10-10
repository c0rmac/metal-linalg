#pragma once
#include <metal_linalg/core.h>   // TrsmPolicy, trsm_backend

#include <mlx/mlx.h>

namespace metal_linalg {

    // -------------------------------------------------------------------------
    // Triangular solve on the GPU (since 2.18.0)
    // -------------------------------------------------------------------------
    // X with A X = B for a batch of triangular A [..., N, N] and B [..., N, K]
    // (X the same) or [..., N] (one right-hand side a matrix; X [..., N]),
    // the batch shapes equal, as mlx::core::linalg::solve_triangular, which
    // runs on the CPU alone. Only A's lower triangle is read (the upper with
    // `upper`); with `unit_diagonal` its diagonal is taken as ones and not
    // read (as torch.linalg.solve_triangular's unitriangular). Output is
    // float32. Large problems (by N and K) go to the GPU (TrsmPolicy). A zero
    // on the diagonal gives infinities or NaN, as BLAS's strsm. Throws
    // std::invalid_argument for a non-square A or mismatched shapes.
    mlx::core::array solve_triangular_accelerated(const mlx::core::array& a, const mlx::core::array& b,
                                                  bool upper = false, bool unit_diagonal = false);

    // Each backend on its own, as described in core.h, for tests and tuning.
    namespace detail {
        mlx::core::array solve_triangular_cpu(const mlx::core::array& a, const mlx::core::array& b,
                                              bool upper = false, bool unit_diagonal = false);
        mlx::core::array solve_triangular_blocked(const mlx::core::array& a, const mlx::core::array& b,
                                                  bool upper = false, bool unit_diagonal = false);
    }

} // namespace metal_linalg
