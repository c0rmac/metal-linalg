#pragma once
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
    // Routing policy
    // -------------------------------------------------------------------------
    // Which path is fastest depends on the GPU and on the CPU beside it, so the
    // routing is a tuned policy rather than a set of constants. Values are
    // measured per device with tuning/tune_eigh.py; see docs/tuning.md for
    // how, and docs/studies/eigh-routing-apple-m1.md for the M1 study.
    //
    // The decision has two parts. GPU or CPU: the Metal kernels need a batch
    // to fill the GPU and Accelerate's LAPACK is quick, so the public
    // functions use the GPU only inside the region where it was measured
    // faster and call mlx::core::linalg::eigh on the CPU otherwise (same
    // decomposition; eigenvector signs may differ). Then, on the GPU, which
    // backend: the whole-matrix kernel in simd or threadgroup mode, or block
    // Jacobi.

    // gpu_max_n value meaning "no upper limit on N".
    constexpr unsigned kEighNoLimit = 0xFFFFFFFFu;

    struct EighPolicy {
        // --- which GPU backend ---
        unsigned simd_max_n  = 8;    // whole-matrix kernel in simd mode up to this N
        unsigned block_min_n = 96;   // block Jacobi from this N on

        // Optional batch-dependent crossover: block also from
        // block_min_n_batched once the batch reaches block_min_batch. Zero
        // disables it. Supported but not used on any measured device: on an
        // M1 it was better on held-out data in 89% of bootstrap resamples
        // against a 95% bar. tuning/tune_eigh.py re-tests it on every device
        // and emits it only when it clears that bar.
        unsigned block_min_n_batched = 0;
        unsigned block_min_batch     = 0;

        // --- GPU or CPU ---
        // GPU iff N <= gpu_max_n, batch * N >= gpu_min_batch_times_n and
        // batch >= gpu_min_batch. gpu_max_n = 0 means never, kEighNoLimit
        // means no cap on N. The minimum batch exists because a lone matrix
        // is faster on the CPU at every N measured on every device so far
        // (up to 4096 on an M5 Pro), and the batch * N product alone cannot
        // say so without also refusing small batches of small matrices.
        // 1 disables it.
        unsigned gpu_max_n             = 64;
        unsigned gpu_min_batch_times_n = 1024;
        unsigned gpu_min_batch         = 1;

        // Device this was resolved against. Informational: detected, not
        // assumed, and what a retune should be keyed on.
        unsigned gpu_cores = 0;   // 0 if it could not be detected
    };

    // The policy in effect. Resolved once, on first use.
    EighPolicy eigh_policy();

    // Where that policy came from: "user", "env:<variables>", "tuned:<device>",
    // or "default:untuned-device (<device>)".
    const char* eigh_policy_source();

    // Override the policy programmatically. Takes precedence over the
    // environment and the tuned table.
    void set_eigh_policy(const EighPolicy& p);

    enum class EighBackend { cpu, simd, threadgroup, block };

    // What eigh_accelerated does with a problem under the policy in effect.
    // EIGH_DEVICE=gpu or EIGH_DEVICE=cpu forces the first part of the decision.
    EighBackend eigh_backend(unsigned n, unsigned batch);

    // The GPU backend the policy picks, regardless of the CPU routing. This is
    // what a forced-GPU call runs.
    EighBackend eigh_gpu_backend(unsigned n, unsigned batch);

    // True iff eigh_backend(n, batch) is a GPU backend.
    bool eigh_uses_gpu(unsigned n, unsigned batch);

    // -------------------------------------------------------------------------
    // Lower-level entry point, for tests and tuning
    // -------------------------------------------------------------------------
    struct EighOptions {
        // How each matrix is mapped onto the GPU. `automatic` follows the
        // routing policy: simd mode for N <= simd_max_n, threadgroup mode up
        // to block_min_n, and the block backend above that.
        //   simd        one 32-lane simdgroup per matrix, several matrices per
        //               threadgroup, no threadgroup barriers.
        //   threadgroup one threadgroup (up to 1024 threads) per matrix.
        //   block       block Jacobi: each matrix spread over the grid, with
        //               the rotations applied as simdgroup_matrix products.
        //               The large-N backend; see Eigh_BlockJacobi.metal.
        enum class Mode { automatic, simd, threadgroup, block };
        Mode mode = Mode::automatic;

        // block mode: scalar Jacobi sweeps per 2b x 2b subproblem. 0 = the
        // default in eigh_block_jacobi.mm (EIGH_INNER_SWEEPS overrides).
        unsigned inner_sweeps = 0;

        // Stop when the off-diagonal Frobenius norm is at most tol * ||A||_F.
        // The default is about one float32 epsilon; the off-norm converges
        // quadratically, so tightening it costs at most a sweep.
        float tol = 1e-7f;

        // Sweep bound. Typical convergence is 5-9 sweeps for any N.
        unsigned max_sweeps = 30;

        // threadgroup mode: threads per matrix, multiple of 32. 0 = automatic.
        unsigned threads = 0;

        // simd mode: matrices per threadgroup. 0 = automatic.
        unsigned matrices_per_threadgroup = 0;
    };

    struct EighResult {
        mlx::core::array eigenvalues;   // [..., N], ascending
        mlx::core::array eigenvectors;  // [..., N, N]; a 0-element array if not requested
        mlx::core::array info;          // uint32 [...]: sweeps | (converged << 16) | (non_finite << 17)
    };

    namespace detail {
        // One team (threadgroup or simdgroup) per matrix. Honours opt.mode
        // only between simd and threadgroup; `block` falls back to threadgroup.
        EighResult eigh_jacobi(const mlx::core::array& a, bool compute_vectors,
                               bool lower, const EighOptions& opt);

        // Block Jacobi: each matrix spread over the grid. N <= 4096.
        EighResult eigh_block_jacobi(const mlx::core::array& a, bool compute_vectors,
                                     bool lower, const EighOptions& opt);

        // Shorthands for eigh_policy().simd_max_n and .block_min_n.
        unsigned eigh_simd_max_n();
        unsigned eigh_block_min_n();

        // Decodes an `info` word.
        inline unsigned eigh_sweeps(unsigned info)    { return info & 0xFFFFu; }
        inline bool     eigh_converged(unsigned info) { return (info >> 16) & 1u; }
        inline bool     eigh_nonfinite(unsigned info) { return (info >> 17) & 1u; }
    }

} // namespace metal_linalg
