#pragma once
#include <metal_linalg/core.h>   // EighPolicy, EighOptions, eigh_backend and the info decoders

#include <mlx/mlx.h>
#include <string>
#include <utility>

namespace metal_linalg {

    // -------------------------------------------------------------------------
    // Symmetric eigendecomposition on the GPU
    // -------------------------------------------------------------------------
    // For a batch of real symmetric matrices A [..., N, N], computes
    //
    //     A = V diag(w) V^T
    //
    // and returns {w, V}: eigenvalues w [..., N] in ascending order and the
    // corresponding eigenvectors as the columns of V [..., N, N]. Same contract
    // as mlx::core::linalg::eigh, which as of MLX 0.31 only runs on the CPU.
    //
    // Only one triangle of the input is read (`uplo` = "L" or "U", as in
    // LAPACK); the other is ignored, so the input need not be exactly
    // symmetric. Output is float32. Batch dimensions are arbitrary.
    //
    // Throws std::invalid_argument for a non-square or too-large input, and
    // std::runtime_error if any matrix with finite entries fails to converge
    // (which the Jacobi method does not do in practice; the bound exists so a
    // bug is loud rather than a hang). Non-finite input yields NaN output.
    std::pair<mlx::core::array, mlx::core::array>
    eigh_accelerated(const mlx::core::array& a, const std::string& uplo = "L");

    // Eigenvalues only. Skips the eigenvector accumulation, which is roughly a
    // third of the work per sweep.
    mlx::core::array
    eigvalsh_accelerated(const mlx::core::array& a, const std::string& uplo = "L");

    // -------------------------------------------------------------------------
    // Lower-level entry points, for tests and tuning
    // -------------------------------------------------------------------------
    struct EighResult {
        mlx::core::array eigenvalues;   // [..., N], ascending
        mlx::core::array eigenvectors;  // [..., N, N]; a 0-element array if not requested
        mlx::core::array info;          // uint32 [...]: sweeps | (converged << 16) | (non_finite << 17)
    };

    // Each backend on its own, as described in core.h.
    namespace detail {
        EighResult eigh_jacobi(const mlx::core::array& a, bool compute_vectors,
                               bool lower, const EighOptions& opt);

        EighResult eigh_block_jacobi(const mlx::core::array& a, bool compute_vectors,
                                     bool lower, const EighOptions& opt);

        EighResult eigh_cpu(const mlx::core::array& a, bool compute_vectors, bool lower);

        EighResult eigh_tridiag(const mlx::core::array& a, bool compute_vectors, bool lower);
        // The tridiag backend's batched form: a batch's reductions together.
        EighResult eigh_tridiag_batch(const mlx::core::array& a, bool compute_vectors, bool lower);
        // Eigenvalues alone by the two-stage reduction; `width` the band's,
        // 8, 16 or 32 (0: the default).
        EighResult eigh_band(const mlx::core::array& a, bool lower = true, uint32_t width = 0);
        // Eigenvalues and eigenvectors by the two-stage reduction (width 16).
        EighResult eigh_band_vectors(const mlx::core::array& a, bool lower = true);

        // N <= metal_linalg::detail::eigh_ql_max_n(); `info` counts QL iterations.
        EighResult eigh_ql(const mlx::core::array& a, bool compute_vectors, bool lower);
        // eigh_ql and the CPU path sharing one batch (EighPolicy::share_min_batch).
        EighResult eigh_ql_shared(const mlx::core::array& a, bool compute_vectors, bool lower);
    }

} // namespace metal_linalg
