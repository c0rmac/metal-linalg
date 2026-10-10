#pragma once
#include <metal_linalg/core.h>   // LuPolicy, lu_backend

#include <mlx/mlx.h>

namespace metal_linalg {

    // -------------------------------------------------------------------------
    // LU factorization, linear solve and inverse on the GPU (since 2.18.0)
    // -------------------------------------------------------------------------
    // For a batch of real square matrices A [..., N, N], with partial
    // pivoting (P A = L U), as mlx::core::linalg's lu_factor, solve and inv,
    // which run on the CPU alone. Output is float32. Batch dimensions are
    // arbitrary; b's must equal a's. Large matrices go to the GPU (LuPolicy).
    //
    // A singular matrix (U with an exactly zero diagonal entry) is factored as
    // LAPACK's sgetrf factors it, and its solve and inverse are all NaN, for
    // that matrix alone; the _ex functions say which. Throws
    // std::invalid_argument for a non-square input or mismatched shapes.

    // The packed factorization [..., N, N] (U on and above the diagonal, L
    // below it with its unit diagonal implied) and the pivots [..., N]
    // (uint32, 0-based: row i was swapped with row pivots[i], in order).
    std::pair<mlx::core::array, mlx::core::array> lu_factor_accelerated(const mlx::core::array& a);

    // X with A X = B, for B [..., N, K] (X the same) or [..., N] (one
    // right-hand side a matrix; X [..., N]).
    mlx::core::array solve_accelerated(const mlx::core::array& a, const mlx::core::array& b);

    // A^-1 [..., N, N].
    mlx::core::array inv_accelerated(const mlx::core::array& a);

    struct LuResult {
        mlx::core::array lu;       // [..., N, N]
        mlx::core::array pivots;   // uint32 [..., N]
        mlx::core::array info;     // uint32 [...]: 0, or k where U(k-1, k-1) is exactly zero
    };
    LuResult lu_factor_ex_accelerated(const mlx::core::array& a);

    struct SolveResult {
        mlx::core::array x;
        mlx::core::array info;     // uint32 [...], as LuResult's
    };
    SolveResult solve_ex_accelerated(const mlx::core::array& a, const mlx::core::array& b);
    SolveResult inv_ex_accelerated(const mlx::core::array& a);

    // Each backend on its own, as described in core.h, for tests and tuning.
    namespace detail {
        LuResult    lu_factor_cpu(const mlx::core::array& a);
        LuResult    lu_factor_blocked(const mlx::core::array& a);
        SolveResult solve_cpu(const mlx::core::array& a, const mlx::core::array& b);
        SolveResult solve_blocked(const mlx::core::array& a, const mlx::core::array& b);
        SolveResult inv_cpu(const mlx::core::array& a);
        SolveResult inv_blocked(const mlx::core::array& a);
    }

} // namespace metal_linalg
