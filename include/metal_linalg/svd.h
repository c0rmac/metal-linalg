#pragma once
#include <metal_linalg/core.h>   // SvdPolicy, SvdOptions, svd_backend and the info decoders

#include <mlx/mlx.h>
#include <tuple>

namespace metal_linalg {

    // -------------------------------------------------------------------------
    // Singular value decomposition on the GPU
    // -------------------------------------------------------------------------
    // For a batch of real matrices A [..., M, N], computes the thin SVD
    //
    //     A = U diag(S) Vt,     K = min(M, N)
    //
    // and returns {U, S, Vt}: U [..., M, K] with orthonormal columns, the
    // singular values S [..., K] non-negative and descending, and Vt
    // [..., K, N] with orthonormal rows.
    //
    // This is the economy form, like this library's QR and like numpy's
    // full_matrices=False. mlx::core::linalg::svd returns the full-size
    // factors (U is M x M and Vt is N x N), and as of MLX 0.31 only runs on
    // the CPU. The thin factors are the leading K columns of U and rows of Vt
    // of the full ones.
    //
    // Output is float32. Batch dimensions are arbitrary. Any shape is
    // accepted: wide inputs are handled by decomposing the transpose. Rank-
    // deficient input is handled, U stays orthonormal. Non-finite input yields
    // NaN output rather than an exception.
    //
    // Throws std::runtime_error if a finite matrix fails to converge, which
    // one-sided Jacobi does not do in practice.
    std::tuple<mlx::core::array, mlx::core::array, mlx::core::array>
    svd_accelerated(const mlx::core::array& a);

    // Singular values only. Skips V and U, roughly halving the work.
    mlx::core::array svdvals_accelerated(const mlx::core::array& a);

    // -------------------------------------------------------------------------
    // Lower-level entry points, for tests and tuning
    // -------------------------------------------------------------------------
    struct SvdResult {
        mlx::core::array U;     // [..., M, K]; a 0-element array if not requested
        mlx::core::array S;     // [..., K], descending
        mlx::core::array Vt;    // [..., K, N]; a 0-element array if not requested
        mlx::core::array info;  // uint32 [...]: sweeps | flags, see the decoders in core.h
    };

    // Each backend on its own, as described in core.h.
    namespace detail {
        SvdResult svd_jacobi(const mlx::core::array& a, bool compute_uv, const SvdOptions& opt);
        SvdResult svd_block_jacobi(const mlx::core::array& a, bool compute_uv, const SvdOptions& opt);
        SvdResult svd_qr_jacobi(const mlx::core::array& a, bool compute_uv, const SvdOptions& opt);
        SvdResult svd_cpu(const mlx::core::array& a, bool compute_uv);
        SvdResult svd_bidiag(const mlx::core::array& a, bool compute_uv);
        // Shapes that metal_linalg::detail::svd_gk_fits(); `info` counts QR
        // steps. With the QR first: svd_qr_jacobi with Kernel::golub_kahan.
        SvdResult svd_golub_kahan(const mlx::core::array& a, bool compute_uv);
        // golub_kahan and the CPU path sharing one batch (SvdPolicy::share_min_batch).
        SvdResult svd_golub_kahan_shared(const mlx::core::array& a, bool compute_uv);
    }

} // namespace metal_linalg
