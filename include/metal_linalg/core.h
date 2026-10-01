#pragma once
// The library without MLX: the decompositions on plain float buffers, the
// routing policies, and the device queries. Everything else is built on
// this: the MLX API (qr.h, eigh.h, svd.h), the C API (c_api.h) and the Swift
// package. Nothing here includes an MLX header.
#include <metal_linalg/device.h>

#include <cstdint>

namespace metal_linalg {

    // =========================================================================
    // QR
    // =========================================================================

    // -------------------------------------------------------------------------
    // Routing policy
    // -------------------------------------------------------------------------
    // As for eigh and the SVD, the decision has two parts, both measured per
    // device by tuning/tune_qr.py (see docs/tuning.md). GPU or CPU: the
    // Metal kernels need enough work to pay for a launch, and LAPACK on the
    // CPU (sgeqrf, sorgqr) is quick for a lone or small batch. Then, on the
    // GPU, which of the two kernels: a single-threadgroup one for short
    // matrices and a grid-parallel one for long. With k = min(M, N):
    //
    //   GPU or CPU   GPU iff k <= gpu_max_k, batch * k >= gpu_min_batch_times_k
    //                and batch >= gpu_min_batch
    //   kernel       grid-parallel iff M >= m_crossover_*, else single-threadgroup
    //
    // The sign of R's diagonal is the one each backend produces: LAPACK's
    // Householder convention for the CPU and the single-threadgroup kernel,
    // non-negative (and, for square input, det(Q) = +1) for the grid-parallel
    // one. A caller that needs one convention normalises it: flip column i of
    // Q and row i of R wherever R[i][i] < 0.

    // gpu_max_k value meaning "no upper limit on k".
    constexpr unsigned kQrNoLimit = 0xFFFFFFFFu;

    struct QrPolicy {
        // --- which GPU kernel ---
        // Matrices with at least this many ROWS use the grid-parallel backend.
        // Rows, not max(M, N): qr_unblocked sweeps M serially inside a single
        // threadgroup, while N parallelises across that threadgroup's threads.
        //
        // A batch-dependent split is supported but not used on any measured
        // device: set the two thresholds equal to disable it. On an M1 the split
        // looked like a 0.7% win on a square-heavy grid and then failed once
        // tall shapes were sampled properly, so it is off by default.
        unsigned m_crossover_small_batch = 384;  // batch <  batch_threshold
        unsigned m_crossover_large_batch = 384;  // batch >= batch_threshold
        unsigned batch_threshold         = 16;

        // --- GPU or CPU ---
        // GPU iff k <= gpu_max_k, batch * k >= gpu_min_batch_times_k and
        // batch >= gpu_min_batch, with k = min(M, N). gpu_max_k = 0 means
        // never, kQrNoLimit no cap; gpu_min_batch_times_k = 0 with
        // gpu_min_batch = 1 means always the GPU, which is what a device
        // measured before QR had a CPU path gets. The defaults, for an
        // untuned device, send lone and small-batch calls to the CPU, the
        // safe direction: LAPACK is never slow, while a GPU launch for one
        // small matrix is.
        unsigned gpu_max_k             = kQrNoLimit;
        unsigned gpu_min_batch_times_k = 1024;
        unsigned gpu_min_batch         = 1;

        // Device properties this was resolved against. Informational: they are
        // detected, not assumed, and are what a retune should be keyed on.
        unsigned gpu_cores = 0;           // 0 if it could not be detected
        unsigned concurrent_matrices = 0; // qr_unblocked threadgroups resident at once
    };

    // The policy in effect, resolved once on first use; where it came from
    // ("user", "env:<variables>", "tuned:<device>" or
    // "default:untuned-device (<device>)"); and a way to replace it, which
    // takes precedence over the environment and the tuned table.
    QrPolicy    qr_policy();
    const char* qr_policy_source();
    void        set_qr_policy(const QrPolicy& p);

    // `cpu` is last so that the values the GPU backends had before it existed
    // are unchanged.
    enum class QrBackend { unblocked, streaming_reduced, cpu };

    // What a QR call does with a problem under the policy in effect.
    // QR_DEVICE=gpu or QR_DEVICE=cpu forces the first part of the decision.
    QrBackend qr_backend(unsigned m, unsigned n, unsigned batch);

    // The GPU kernel the policy picks, regardless of the CPU routing. This is
    // what a forced-GPU call runs.
    QrBackend qr_gpu_backend(unsigned m, unsigned n, unsigned batch);

    // True iff qr_backend(m, n, batch) is a GPU backend.
    bool qr_uses_gpu(unsigned m, unsigned n, unsigned batch);

    // =========================================================================
    // Symmetric eigendecomposition
    // =========================================================================

    // -------------------------------------------------------------------------
    // Routing policy
    // -------------------------------------------------------------------------
    // Which path is fastest depends on the GPU and on the CPU beside it, so the
    // routing is a tuned policy rather than a set of constants. Values are
    // measured per device with tuning/tune_eigh.py; see docs/tuning.md for
    // how, and docs/studies/eigh-routing-apple-m1.md for the M1 study.
    //
    // The decision has two parts. GPU or CPU: the Metal kernels need a batch
    // to fill the GPU and Accelerate's LAPACK is quick, so the GPU is used
    // only inside the region where it was measured faster, and LAPACK's
    // ssyevd on the CPU otherwise (same decomposition; eigenvector signs may
    // differ). Then, on the GPU, which backend: the whole-matrix kernel in
    // simd or threadgroup mode, or block Jacobi.

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

    // What an eigh call does with a problem under the policy in effect.
    // EIGH_DEVICE=gpu or EIGH_DEVICE=cpu forces the first part of the decision.
    EighBackend eigh_backend(unsigned n, unsigned batch);

    // The GPU backend the policy picks, regardless of the CPU routing. This is
    // what a forced-GPU call runs.
    EighBackend eigh_gpu_backend(unsigned n, unsigned batch);

    // True iff eigh_backend(n, batch) is a GPU backend.
    bool eigh_uses_gpu(unsigned n, unsigned batch);

    // -------------------------------------------------------------------------
    // Options of the lower-level entry points, for tests and tuning
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

    // =========================================================================
    // Singular value decomposition
    // =========================================================================

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
    // The CPU path is LAPACK (Accelerate): sgesdd, preceded by a QR (sgeqrf,
    // sorgqr) when l >= 2k, so that it computes thin factors only.

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

    // What an SVD call does with a problem under the policy in effect.
    // SVD_DEVICE=gpu or SVD_DEVICE=cpu forces the first part of the decision.
    SvdBackend svd_backend(unsigned m, unsigned n, unsigned batch);

    // The GPU backend the policy picks, regardless of the CPU routing.
    SvdBackend svd_gpu_backend(unsigned m, unsigned n, unsigned batch);

    // True iff svd_backend(m, n, batch) is a GPU backend.
    bool svd_uses_gpu(unsigned m, unsigned n, unsigned batch);

    // -------------------------------------------------------------------------
    // Options of the lower-level entry points, for tests and tuning
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

    namespace detail {
        // Shorthands for eigh_policy().simd_max_n and .block_min_n.
        unsigned eigh_simd_max_n();
        unsigned eigh_block_min_n();

        // Decode an eigh `info` word: sweeps | (converged << 16) | (non_finite << 17).
        inline unsigned eigh_sweeps(unsigned info)    { return info & 0xFFFFu; }
        inline bool     eigh_converged(unsigned info) { return (info >> 16) & 1u; }
        inline bool     eigh_nonfinite(unsigned info) { return (info >> 17) & 1u; }

        // Decode an SVD `info` word. The sweep count includes the final sweep
        // that found nothing to rotate.
        inline unsigned svd_sweeps(unsigned info)         { return info & 0xFFFFu; }
        inline bool     svd_converged(unsigned info)      { return (info >> 16) & 1u; }
        inline bool     svd_nonfinite(unsigned info)      { return (info >> 17) & 1u; }
        inline bool     svd_rank_deficient(unsigned info) { return (info >> 18) & 1u; }

        // Completes the zero columns of one row-major M x N matrix U to an
        // orthonormal set; shared by the kernels' host code.
        void svd_complete_columns(float* u, unsigned m, unsigned n);
    }

    // =========================================================================
    // The decompositions on buffers
    // =========================================================================
    // Inputs and outputs are float32, row-major, and contiguous: a batch is its
    // matrices one after another. The caller allocates every output. Memory
    // that starts on a page boundary is read by the GPU in place; anything
    // else is copied once. Every call returns when its results are written.
    //
    // Errors are thrown: std::invalid_argument for a shape a backend cannot
    // take, std::runtime_error for a GPU failure or a finite matrix that did
    // not converge. Non-finite input gives NaN output for that matrix only.
    namespace core {

        // `batch` matrices of rows x cols.
        struct Matrices {
            const float* data;
            uint32_t     batch;
            uint32_t     rows;
            uint32_t     cols;
        };

        // A = Q R with K = min(M, N): q [batch, M, K] with orthonormal
        // columns, r [batch, K, N] upper triangular.
        void qr(const Matrices& a, float* q, float* r);

        // A = V diag(w) V^T for symmetric A (N x N), reading only the lower
        // triangle if `lower`, else the upper. w [batch, N] ascending; v
        // [batch, N, N] with the eigenvectors as columns, or null for the
        // eigenvalues alone; info [batch] as decoded above, or null.
        void eigh(const Matrices& a, bool lower, float* w, float* v, uint32_t* info = nullptr);

        // Thin SVD A = U diag(s) Vt, K = min(M, N): u [batch, M, K], s
        // [batch, K] descending, vt [batch, K, N]. u and vt both null for the
        // singular values alone. info [batch] as decoded above, or null.
        void svd(const Matrices& a, float* u, float* s, float* vt, uint32_t* info = nullptr);

        // Each backend on its own, whatever the policy says; same contracts
        // as above. For tests and tuning.
        namespace detail {
            // Standard Householder QR in a single kernel dispatch, one
            // threadgroup per matrix. Preferred for small matrices, where
            // multi-pass streaming does not pay for its launch overhead.
            void qr_unblocked(const Matrices& a, float* q, float* r);

            // Multi-pass streaming panel factorisation that accumulates Q
            // directly at its economic K-column width via a backward pass. The
            // grid-parallel path for large matrices.
            void qr_streaming_amx_reduced(const Matrices& a, float* q, float* r);

            // As above, but accumulates the full M x M orthogonal factor
            // before slicing Q down to K columns.
            //
            // Not reachable from the routing: benchmarking put it within noise
            // of qr_streaming_amx_reduced everywhere it was measured, while
            // allocating the full M x M Q. Kept and tested rather than
            // deleted, since it is the only backend that forms the complete
            // orthogonal factor.
            void qr_streaming_amx_complete(const Matrices& a, float* q, float* r);

            // LAPACK on the CPU (sgeqrf, sorgqr), one matrix at a time. A
            // matrix holding a NaN or an infinity gives NaN for its Q and R.
            void qr_cpu(const Matrices& a, float* q, float* r);

            // One team (threadgroup or simdgroup) per matrix. Honours opt.mode
            // only between simd and threadgroup; `block` falls back to
            // threadgroup.
            void eigh_jacobi(const Matrices& a, bool lower, const EighOptions& opt,
                             float* w, float* v, uint32_t* info);

            // Block Jacobi: each matrix spread over the grid. N <= 4096.
            void eigh_block_jacobi(const Matrices& a, bool lower, const EighOptions& opt,
                                   float* w, float* v, uint32_t* info);

            // LAPACK ssyevd on the CPU, one matrix at a time. `info` reports
            // every finite matrix as converged in one sweep.
            void eigh_cpu(const Matrices& a, bool lower, float* w, float* v, uint32_t* info);

            // One-sided Jacobi on the matrix itself: one threadgroup per
            // matrix, each simdgroup owning one or more column pairs.
            void svd_jacobi(const Matrices& a, const SvdOptions& opt,
                            float* u, float* s, float* vt, uint32_t* info);

            // Block one-sided Jacobi: each matrix spread over the grid,
            // rotations applied as tile products. The large-matrix kernel.
            // min(M, N) <= 4096.
            void svd_block_jacobi(const Matrices& a, const SvdOptions& opt,
                                  float* u, float* s, float* vt, uint32_t* info);

            // QR-preconditioned: this library's QR, one of the two kernels on
            // the triangular factor (opt.kernel), and U = Q * U_R on the GPU.
            // For tall matrices.
            void svd_qr_jacobi(const Matrices& a, const SvdOptions& opt,
                               float* u, float* s, float* vt, uint32_t* info);

            // LAPACK on the CPU, one matrix at a time: sgesdd, after a QR for
            // a matrix at least twice as tall as wide. `info` reports every
            // finite matrix as converged in one sweep.
            void svd_cpu(const Matrices& a, float* u, float* s, float* vt, uint32_t* info);
        }

    } // namespace core

} // namespace metal_linalg
