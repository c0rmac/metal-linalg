#pragma once
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
    // Routing policy
    // -------------------------------------------------------------------------
    // As for QR and eigh, which path is fastest depends on the device, so the
    // routing is a per-device policy measured by tuning/tune_svd.py. With
    // k = min(M, N) and l = max(M, N):
    //
    //   GPU or CPU     GPU iff k <= gpu_max_k, batch * k >= gpu_min_batch_times_k
    //                  and batch >= gpu_min_batch
    //   precondition   with this library's QR iff l >= qr_min_rows, k >= qr_min_k and
    //                  l >= 2k; the kernel then runs on the k x k factor
    //   kernel         block Jacobi iff k >= block_min_k, or k >= block_min_k_batched
    //                  in a batch of at least block_min_batch; else the
    //                  whole-matrix kernel
    //
    // The CPU path is MLX's svd (Accelerate LAPACK), preceded by a CPU QR when
    // the matrix is tall, so that it too computes thin factors only.

    struct SvdPolicy {
        // --- which GPU backend ---
        // Tall matrices are reduced by this library's QR first and the Jacobi
        // kernel runs on the triangular factor, so the rotations act on k x k
        // instead of l x k. That path carries a fixed cost of a few
        // milliseconds, so it needs the direct kernel's work to be large: a
        // long matrix that is also wide enough. 1024 x 64 gains 1.8x from it
        // and 1024 x 16 loses 2.7x. It is never used unless l >= 2k, below
        // which there is little to reduce.
        unsigned qr_min_rows = 512;
        unsigned qr_min_k    = 64;

        // --- which kernel ---
        // The whole-matrix kernel gives a matrix one threadgroup, that is one
        // GPU core; the block kernel spreads it over the grid and applies the
        // rotations as tile products. Block from this short side on. With
        // preconditioning the kernel sees the k x k factor, so k decides
        // either way.
        unsigned block_min_k = 192;

        // Batch-dependent crossover: block also from block_min_k_batched once
        // the batch reaches block_min_batch. Zero disables it. The block
        // kernel issues several dispatches per round whatever the batch, and
        // a batch shares them, so it overtakes the whole-matrix kernel at a
        // smaller k. tuning/tune_svd.py emits it only when it is better on
        // held-out data.
        unsigned block_min_k_batched = 0;
        unsigned block_min_batch     = 0;

        // --- GPU or CPU ---
        // gpu_max_k = 0 means never, kSvdNoLimit means no cap. The minimum
        // batch exists because a lone matrix is faster on the CPU at every
        // k measured on every device so far (up to 4096 on an M5 Pro), and
        // the batch * k product alone cannot say so without also refusing
        // small batches of small matrices. 1 disables it.
        unsigned gpu_max_k             = 64;
        unsigned gpu_min_batch_times_k = 1024;
        unsigned gpu_min_batch         = 1;

        // Device this was resolved against; informational.
        unsigned gpu_cores = 0;
    };

    constexpr unsigned kSvdNoLimit = 0xFFFFFFFFu;

    // The policy in effect, where it came from ("user", "env:<variables>",
    // "tuned:<device>" or "default:untuned-device (<device>)"), and a way to
    // replace it.
    SvdPolicy   svd_policy();
    const char* svd_policy_source();
    void        set_svd_policy(const SvdPolicy& p);

    enum class SvdBackend { cpu, jacobi, block_jacobi, qr_jacobi, qr_block_jacobi };

    // What svd_accelerated does with a problem under the policy in effect.
    // SVD_DEVICE=gpu or SVD_DEVICE=cpu forces the first part of the decision.
    SvdBackend svd_backend(unsigned m, unsigned n, unsigned batch);

    // The GPU backend the policy picks, regardless of the CPU routing.
    SvdBackend svd_gpu_backend(unsigned m, unsigned n, unsigned batch);

    // True iff svd_backend(m, n, batch) is a GPU backend.
    bool svd_uses_gpu(unsigned m, unsigned n, unsigned batch);

    // -------------------------------------------------------------------------
    // Lower-level entry point, for tests and tuning
    // -------------------------------------------------------------------------
    struct SvdOptions {
        // A pair of columns is rotated while |g_p . g_q| exceeds
        // tol * |g_p| |g_q|. 0 selects sqrt(M) * float32 epsilon, which is
        // LAPACK xGESVJ's default.
        float tol = 0.0f;

        // Sweep bound. Typical convergence is 5-10 sweeps.
        unsigned max_sweeps = 30;

        // The row count the rotation tolerance refers to, when that is not
        // this matrix's own: the QR-preconditioned backend hands the kernel a
        // k x k factor of an l x k matrix, and the tolerance sqrt(rows) * eps
        // belongs to the matrix the user passed. 0 = this matrix's own rows.
        // Never taken below 64, under which the tolerance falls inside the
        // rounding of a float32 inner product.
        unsigned effective_rows = 0;

        // Which kernel the QR-preconditioned backend runs on the factor.
        // `automatic` follows the policy's kernel rule.
        enum class Kernel { automatic, jacobi, block };
        Kernel kernel = Kernel::automatic;

        // Block kernel: Jacobi sweeps per 32 x 32 subproblem. 0 = 1, or
        // SVD_INNER_SWEEPS.
        unsigned inner_sweeps = 0;

        // Simdgroups per matrix, 1 to 32. Each owns one or more column pairs
        // per round. 0 = automatic (see svd.mm).
        unsigned simdgroups = 0;
    };

    struct SvdResult {
        mlx::core::array U;     // [..., M, K]; a 0-element array if not requested
        mlx::core::array S;     // [..., K], descending
        mlx::core::array Vt;    // [..., K, N]; a 0-element array if not requested
        mlx::core::array info;  // uint32 [...]: sweeps | flags, see below
    };

    namespace detail {
        // One-sided Jacobi on the matrix itself: one threadgroup per matrix,
        // each simdgroup owning one or more column pairs. Always runs the
        // Metal kernel.
        SvdResult svd_jacobi(const mlx::core::array& a, bool compute_uv, const SvdOptions& opt);

        // Block one-sided Jacobi: each matrix spread over the grid, rotations
        // applied as tile products. The large-matrix kernel. min(M, N) <= 4096.
        SvdResult svd_block_jacobi(const mlx::core::array& a, bool compute_uv, const SvdOptions& opt);

        // QR-preconditioned: this library's QR, one of the two kernels on the
        // triangular factor (opt.kernel), and U = Q * U_R. For tall matrices.
        SvdResult svd_qr_jacobi(const mlx::core::array& a, bool compute_uv, const SvdOptions& opt);

        // Completes the zero columns of one row-major M x N matrix U to an
        // orthonormal set; shared by the kernels' host code.
        void svd_complete_columns(float* u, unsigned m, unsigned n);

        // Thin SVD on the CPU through MLX (Accelerate LAPACK), with a QR first
        // when the matrix is tall. `info` is returned as converged.
        SvdResult svd_cpu(const mlx::core::array& a, bool compute_uv);

        // Decodes an `info` word. The sweep count includes the final sweep
        // that found nothing to rotate.
        inline unsigned svd_sweeps(unsigned info)         { return info & 0xFFFFu; }
        inline bool     svd_converged(unsigned info)      { return (info >> 16) & 1u; }
        inline bool     svd_nonfinite(unsigned info)      { return (info >> 17) & 1u; }
        inline bool     svd_rank_deficient(unsigned info) { return (info >> 18) & 1u; }
    }

} // namespace metal_linalg
