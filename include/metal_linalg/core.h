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
    //   GPU or CPU   GPU iff gpu_min_k <= k <= gpu_max_k, batch * k >=
    //                gpu_min_batch_times_k and batch >= gpu_min_batch, or k >=
    //                gpu_large_min_k in a batch of at most gpu_large_max_batch
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
        // GPU iff gpu_min_k <= k <= gpu_max_k, batch * k >=
        // gpu_min_batch_times_k and batch >= gpu_min_batch, with k = min(M, N)
        // (or by the large-matrix clause below). gpu_max_k = 0 means
        // never, kQrNoLimit no cap; gpu_min_batch_times_k = 0 with
        // gpu_min_batch = 1 means always the GPU, which is what a device
        // measured before QR had a CPU path gets. The defaults, for a device
        // with nothing to estimate its policy from (estimate.h), send lone
        // and small-batch calls to the CPU: LAPACK is never slow, while a GPU
        // launch for one small matrix is.
        unsigned gpu_max_k             = kQrNoLimit;
        unsigned gpu_min_batch_times_k = 1024;
        unsigned gpu_min_batch         = 1;
        // ... and k at least this (0: no lower bound). Since the CPU path
        // spreads a batch over every core, it wins the smallest matrices at
        // any batch, while a large batch of mid-size ones can still be the
        // GPU's: on an M5 Pro 10000 of 16x16 take 2.1 ms on the CPU and 3.7
        // on the GPU, and 1024 of 128x128 21 ms on the GPU and 24 on the CPU.
        unsigned gpu_min_k             = 0;

        // Large matrices: the GPU also for k >= gpu_large_min_k in a batch of
        // at most gpu_large_max_batch (0: any batch), whatever the rule above
        // says. Since the CPU path spreads a batch over every core, it beats
        // the GPU kernels for batches of small and mid-size matrices, while
        // one large matrix, which Accelerate threads only weakly, is still
        // faster on the GPU (on an M5 Pro 2x at 2048 x 2048); one product
        // rule cannot say both. 0 = never, which a device without
        // measurements has.
        unsigned gpu_large_min_k     = 0;
        unsigned gpu_large_max_batch = 0;

        // From this batch on, a batch that goes to the GPU is shared with the
        // CPU path: the GPU takes chunks from the front, the CPU from the
        // back, at once, as EighPolicy::share_min_batch describes. Where the
        // two are close (on an M5 Pro 1024 of 128 x 128: 23.7 ms on the GPU,
        // 24.4 on the CPU) the batch takes about half the time. 0 means never,
        // which is what a device without measurements has.
        unsigned share_min_batch = 0;

        // Device properties this was resolved against. Informational: they are
        // detected, not assumed, and are what a retune should be keyed on.
        unsigned gpu_cores = 0;           // 0 if it could not be detected
        unsigned concurrent_matrices = 0; // qr_unblocked threadgroups resident at once
    };

    // The policy in effect, resolved once on first use; where it came from
    // ("user", "env:<variables>", "tuned:<device>", "estimated:<device> (from
    // <measured device>, ...)" on a Mac nobody has measured, or
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

    // True iff a QR call of this shape shares its batch between the GPU and
    // the CPU (share_min_batch).
    bool qr_shares_batch(unsigned m, unsigned n, unsigned batch);

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
    // only inside the region where it was measured faster, and LAPACK on
    // the CPU otherwise: ssyevd, or for eigenvalues alone from N = 128 the
    // two-stage ssyevd_2stage (same decomposition; eigenvector signs may
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

        // --- GPU or CPU, for eigenvalues alone (eigvalsh) ---
        // The same rule with its own thresholds, since the CPU computes
        // eigenvalues alone by a faster method (the two-stage reduction from
        // N = 128), and the GPU's saving from skipping the eigenvectors is
        // smaller. values_gpu_min_batch = 0 means "as for eigenvectors": a
        // device measured before these fields existed, or a policy that does
        // not set them.
        unsigned values_gpu_max_n             = 0;
        unsigned values_gpu_min_batch_times_n = 0;
        unsigned values_gpu_min_batch         = 0;

        // --- the tridiag backend instead of the CPU ---
        // Where the rules above choose the CPU, N >= tridiag_min_n (with
        // eigenvectors) or N >= values_tridiag_min_n (eigenvalues alone) uses
        // the tridiag backend instead: LAPACK's method with its two expensive
        // steps on the GPU, faster than the CPU for one large matrix (on an
        // M5 Pro 2x at N = 2048, 5x at 4096, 6x at 8192 with eigenvectors).
        // 0 means never, which is what a device without measurements has.
        unsigned tridiag_min_n        = 0;
        unsigned values_tridiag_min_n = 0;
        // ... and only for a batch of at most this many matrices (0: any
        // batch). The backend solves a batch one matrix after another, while
        // the CPU path spreads it over every core, so beyond a few matrices
        // the CPU wins whatever N is.
        unsigned tridiag_max_batch        = 0;
        unsigned values_tridiag_max_batch = 0;
        // Eigenvalues alone, from N >= values_band_min_n (within
        // values_tridiag_max_batch), the band backend before tridiag: the
        // two-stage reduction, A to a band on the GPU in blocks whose work is
        // matrix products, the band to tridiagonal on the CPU's cores. 0
        // means never.
        unsigned values_band_min_n = 0;
        // ... and the band's width, 8, 16 or 32 (0: 16, or EIGH_BAND_WIDTH).
        // A wider band halves the GPU's panels and doubles each one's
        // columns, and makes the CPU's chase dearer; which is best depends on
        // the GPU and the CPU together (on an M5 Pro 16: at 4096 x 4096, 262,
        // 165 and 166 ms for 8, 16 and 32).
        unsigned values_band_width = 0;

        // Large batches: the GPU also for N above gpu_max_n, up to
        // gpu_big_batch_max_n, in a batch of at least gpu_big_batch_min (with
        // eigenvectors, and for eigenvalues alone while values_gpu_min_batch
        // is 0). With a batch shared between the GPU and the CPU
        // (share_min_batch) the GPU wins large batches of matrices that the
        // product rule cannot take without also taking their small batches,
        // which the CPU wins. 0 means never, which is what a device without
        // measurements has; so does gpu_max_n = 0, which is never the GPU.
        unsigned gpu_big_batch_max_n = 0;
        unsigned gpu_big_batch_min   = 0;

        // --- the ql backend, for N in [ql_min_n, ql_max_n] ---
        // On the GPU, inside this window, the ql backend (Householder
        // tridiagonalization and implicit QL, one threadgroup per matrix)
        // instead of the Jacobi backend the fields above pick, for
        // eigenvectors and eigenvalues alone. ql_max_n = 0 means never, which
        // is what a device without measurements has; the window is clipped to
        // what the backend takes on the device (detail::eigh_ql_max_n(), 87
        // with 32 KB of threadgroup memory).
        unsigned ql_min_n = 0;
        unsigned ql_max_n = 0;

        // From this batch on, a batch the ql backend gets is shared with the
        // CPU path: the GPU takes chunks from the front, the CPU from the
        // back, at once, each the share its speed earns (on an M5 Pro 1.4-1.7x
        // faster than either alone where they are close). 0 means never,
        // which is what a device without measurements has.
        unsigned share_min_batch = 0;
    };

    // The policy in effect. Resolved once, on first use.
    EighPolicy eigh_policy();

    // Where that policy came from: "user", "env:<variables>", "tuned:<device>",
    // "estimated:<device> (from <measured device>, ...)" on a Mac nobody has
    // measured, or "default:untuned-device (<device>)".
    const char* eigh_policy_source();

    // Override the policy programmatically. Takes precedence over the
    // environment and the tuned table.
    void set_eigh_policy(const EighPolicy& p);

    // tridiag: the hybrid large-N backend (Householder tridiagonalization and
    // back-transformation on the GPU, the tridiagonal eigenproblem on the
    // CPU). ql: the batched one (tridiagonalization and implicit QL, one
    // threadgroup per matrix). Each added last, so the others keep their numbers.
    enum class EighBackend { cpu, simd, threadgroup, block, tridiag, ql, band };

    // What an eigh call does with a problem under the policy in effect.
    // EIGH_DEVICE=gpu or EIGH_DEVICE=cpu forces the first part of the decision.
    EighBackend eigh_backend(unsigned n, unsigned batch);

    // The GPU backend the policy picks, regardless of the CPU routing. This is
    // what a forced-GPU call runs.
    EighBackend eigh_gpu_backend(unsigned n, unsigned batch);

    // True iff eigh_backend(n, batch) is a GPU backend.
    bool eigh_uses_gpu(unsigned n, unsigned batch);

    // The same for eigenvalues alone (eigvalsh), under the values_* boundary.
    EighBackend eigvalsh_backend(unsigned n, unsigned batch);
    bool eigvalsh_uses_gpu(unsigned n, unsigned batch);

    // True iff an eigh call (eigvalsh: eigh_shares_batch_values) of this
    // shape shares its batch between the GPU and the CPU (share_min_batch).
    bool eigh_shares_batch(unsigned n, unsigned batch);
    bool eigvalsh_shares_batch(unsigned n, unsigned batch);

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
    //   GPU or CPU     GPU iff k <= gpu_max_k, l <= gpu_max_l, batch * k >=
    //                  gpu_min_batch_times_k and batch >= gpu_min_batch (for
    //                  singular values alone, the values_* constants)
    //   precondition   with this library's QR iff l >= qr_min_rows, k >= qr_min_k and
    //                  l >= 2k; the kernel then runs on the k x k factor
    //   kernel         block Jacobi iff k >= block_min_k, or k >= block_min_k_batched
    //                  in a batch of at least block_min_batch; else the
    //                  whole-matrix kernel
    //   golub_kahan    instead of the two above, for k in [gk_min_k, gk_max_k]
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
        // ... and l = max(M, N) at most this (kSvdNoLimit: no cap). The CPU
        // path reduces a tall matrix by a QR first, which a batch spread over
        // every core does quickly, so on an M5 Pro every shape with a long
        // side of 256 or more is the CPU's at any batch (256 x 16, 4096 of
        // them: 0.72x on the GPU), while large batches of 32 x 32 are the
        // GPU's (1.65x); a cap on k alone cannot say both.
        unsigned gpu_max_l             = 0xFFFFFFFFu;
        // Large batches: the GPU also for k above gpu_max_k, up to
        // gpu_big_batch_max_k (and l <= gpu_max_l), in a batch of at least
        // gpu_big_batch_min, as EighPolicy::gpu_big_batch_max_n describes.
        // 0 means never; so does gpu_max_k = 0, which is never the GPU.
        unsigned gpu_big_batch_max_k   = 0;
        unsigned gpu_big_batch_min     = 0;

        // --- GPU or CPU for singular values alone (svdvals) ---
        // The same rule with constants of its own. Both sides skip the
        // vectors, by different amounts: the CPU path accumulates no
        // rotations and skips the back-transformation, while the GPU's
        // reduction, most of its work, stays. On an M5 Pro the GPU wins
        // large batches of 16 x 16 for singular values alone by 1.4x, and
        // loses 48 x 48 by 1.5x, which it wins with vectors.
        // values_gpu_min_batch = 0 means "as with vectors": a device
        // measured before these fields existed, or a policy that does not set
        // them.
        unsigned values_gpu_max_k             = 0;
        unsigned values_gpu_min_batch_times_k = 0;
        unsigned values_gpu_min_batch         = 0;
        unsigned values_gpu_max_l             = 0xFFFFFFFFu;

        // Device this was resolved against; informational.
        unsigned gpu_cores = 0;

        // --- the bidiag backend instead of the CPU ---
        // Where the rules above choose the CPU, k >= bidiag_min_k (with
        // singular vectors) or k >= values_bidiag_min_k (singular values
        // alone) uses the bidiag backend instead: LAPACK's method with the
        // bidiagonalization and the back-transformations on the GPU, faster
        // than the CPU for one large matrix (on an M5 Pro 2x at 4096 x 4096).
        // 0 means never, which is what a device without measurements has.
        unsigned bidiag_min_k        = 0;
        unsigned values_bidiag_min_k = 0;
        // ... and only for a batch of at most this many matrices (0: any
        // batch): the backend solves a batch one matrix after another, while
        // the CPU path spreads it over every core.
        unsigned bidiag_max_batch        = 0;
        unsigned values_bidiag_max_batch = 0;
        // Singular values alone, from k >= values_band_min_k (within
        // values_bidiag_max_batch), the band backend before bidiag: the
        // two-stage reduction, A to a band on the GPU in blocks whose work is
        // matrix products, then the band to bidiagonal on the CPU (on an M5
        // Pro 1.8x the bidiag backend at 4096 x 4096). 0 means never.
        unsigned values_band_min_k = 0;
        // ... and the band's width, 8, 16 or 32 (0: 16, or SVD_BAND_WIDTH),
        // as EighPolicy::values_band_width (on an M5 Pro 16: at 4096 x 4096,
        // 329, 235 and 275 ms for 8, 16 and 32).
        unsigned values_band_width = 0;
        // With singular vectors, from k >= band_min_k (within
        // bidiag_max_batch), the band backend before bidiag: the two-stage
        // reduction, its reflectors and the bulge chase's applied on the GPU
        // while the CPU solves the bidiagonal problem (width 16; on an M5 Pro
        // 2.3x the bidiag backend at 4096 x 4096). 0 means never, which is
        // what a device without measurements of it has.
        unsigned band_min_k = 0;

        // --- the golub_kahan backend, for k in [gk_min_k, gk_max_k] ---
        // On the GPU, inside this window, LAPACK's method in one threadgroup
        // per matrix (Householder bidiagonalization and implicit bidiagonal
        // QR) instead of the Jacobi backends the fields above pick, with
        // singular vectors and for singular values alone: on the matrix
        // itself where it fits in threadgroup memory, else after this
        // library's QR on the k x k factor (qr_golub_kahan). gk_max_k = 0
        // means never, which is what a device without measurements has; the
        // window is clipped to what the backend takes on the device
        // (detail::svd_gk_max_k(), 83 with 32 KB of threadgroup memory).
        unsigned gk_min_k = 0;
        unsigned gk_max_k = 0;

        // From this batch on, a batch the golub_kahan backend gets is shared
        // with the CPU path, as EighPolicy::share_min_batch describes. 0 means
        // never, which is what a device without measurements has.
        unsigned share_min_batch = 0;
    };

    constexpr unsigned kSvdNoLimit = 0xFFFFFFFFu;

    // The policy in effect, where it came from ("user", "env:<variables>",
    // "tuned:<device>", "estimated:<device> (from <measured device>, ...)" on a
    // Mac nobody has measured, or "default:untuned-device (<device>)"), and a
    // way to replace it.
    SvdPolicy   svd_policy();
    const char* svd_policy_source();
    void        set_svd_policy(const SvdPolicy& p);

    // bidiag: the hybrid large-k backend (bidiagonalization and
    // back-transformations on the GPU, the bidiagonal SVD on the CPU).
    // golub_kahan: the batched one (bidiagonalization and implicit QR, one
    // threadgroup per matrix); qr_golub_kahan: the same on the k x k factor of
    // this library's QR. Each added last, so the others keep their numbers.
    enum class SvdBackend { cpu, jacobi, block_jacobi, qr_jacobi, qr_block_jacobi, bidiag,
                            golub_kahan, qr_golub_kahan, band };

    // What an SVD call does with a problem under the policy in effect.
    // SVD_DEVICE=gpu or SVD_DEVICE=cpu forces the first part of the decision.
    SvdBackend svd_backend(unsigned m, unsigned n, unsigned batch);

    // The GPU backend the policy picks, regardless of the CPU routing.
    SvdBackend svd_gpu_backend(unsigned m, unsigned n, unsigned batch);

    // True iff svd_backend(m, n, batch) is one of the GPU backends other than
    // bidiag (the Jacobi and golub_kahan ones).
    bool svd_uses_gpu(unsigned m, unsigned n, unsigned batch);

    // The same for singular values alone (svdvals), under the values_* rule.
    bool svdvals_uses_gpu(unsigned m, unsigned n, unsigned batch);

    // True iff an SVD call (svdvals: svdvals_shares_batch) of this shape
    // shares its batch between the GPU and the CPU (share_min_batch).
    bool svd_shares_batch(unsigned m, unsigned n, unsigned batch);
    bool svdvals_shares_batch(unsigned m, unsigned n, unsigned batch);

    // What a call for singular values alone (svdvals) does: as svd_backend,
    // with values_bidiag_min_k for the bidiag backend.
    SvdBackend svdvals_backend(unsigned m, unsigned n, unsigned batch);

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
        // `automatic` follows the policy's kernel rule (between the two
        // Jacobi kernels).
        enum class Kernel { automatic, jacobi, block, golub_kahan };
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

        // The largest N the ql backend takes on this device: the whole matrix
        // lives in threadgroup memory. 0 without a Metal device.
        unsigned eigh_ql_max_n();

        // Decode an eigh `info` word: sweeps | (converged << 16) | (non_finite << 17).
        inline unsigned eigh_sweeps(unsigned info)    { return info & 0xFFFFu; }
        inline bool     eigh_converged(unsigned info) { return (info >> 16) & 1u; }
        inline bool     eigh_nonfinite(unsigned info) { return (info >> 17) & 1u; }

        // The largest min(M, N) of a square matrix the golub_kahan backend
        // takes on this device, and whether it takes rows x cols itself: the
        // matrix and V live in threadgroup memory. 0 / false without a Metal
        // device.
        unsigned svd_gk_max_k();
        bool     svd_gk_fits(unsigned rows, unsigned cols);

        // Decode an SVD `info` word. The sweep count includes the final sweep
        // that found nothing to rotate (golub_kahan: the QR steps).
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
            // grid-parallel path for large matrices. One matrix, or a few
            // large ones, go to qr_blocked instead (qr_blocked_preferred;
            // QR_BLOCKED=0 keeps them here).
            void qr_streaming_amx_reduced(const Matrices& a, float* q, float* r);

            // Large matrices one at a time by blocks of columns: the panels
            // by the band reduction's kernels, the updates and Q's formation
            // as MPS products, the last columns by LAPACK. For up to 16384
            // rows (qr_blocked_fits; throws otherwise).
            void qr_blocked(const Matrices& a, float* q, float* r);
            bool qr_blocked_fits(uint32_t m, uint32_t n);
            // Whether qr_streaming_amx_reduced hands the call to qr_blocked:
            // one matrix, or batch * 512 <= min(m, n), where it fits.
            bool qr_blocked_preferred(uint32_t m, uint32_t n, uint32_t batch);

            // As above, but accumulates the full M x M orthogonal factor
            // before slicing Q down to K columns.
            //
            // Not reachable from the routing: benchmarking put it within noise
            // of qr_streaming_amx_reduced everywhere it was measured, while
            // allocating the full M x M Q. Kept and tested rather than
            // deleted, since it is the only backend that forms the complete
            // orthogonal factor.
            void qr_streaming_amx_complete(const Matrices& a, float* q, float* r);

            // LAPACK on the CPU (sgeqrf, sorgqr), the matrices of a batch
            // spread over cpu_threads() threads. A matrix holding a NaN or an
            // infinity gives NaN for its Q and R.
            void qr_cpu(const Matrices& a, float* q, float* r);

            // The GPU kernel qr_gpu_backend() picks and qr_cpu on one batch at
            // once, sharing it as QrPolicy::share_min_batch describes,
            // whatever the policy.
            void qr_shared(const Matrices& a, float* q, float* r);

            // One team (threadgroup or simdgroup) per matrix. Honours opt.mode
            // only between simd and threadgroup; `block` falls back to
            // threadgroup.
            void eigh_jacobi(const Matrices& a, bool lower, const EighOptions& opt,
                             float* w, float* v, uint32_t* info);

            // Block Jacobi: each matrix spread over the grid. N <= 4096.
            void eigh_block_jacobi(const Matrices& a, bool lower, const EighOptions& opt,
                                   float* w, float* v, uint32_t* info);

            // LAPACK's method with the tridiagonalization and the
            // back-transformation on the GPU, one matrix at a time; any N.
            // See eigh_tridiag.mm.
            void eigh_tridiag(const Matrices& a, bool lower, float* w, float* v, uint32_t* info);

            // Eigenvalues alone by the two-stage reduction: A to a band on the
            // GPU, the band to tridiagonal on the CPU's cores, then LAPACK
            // ssterf. `width` the band's, 8, 16 or 32 (0: the default, or
            // EIGH_BAND_WIDTH). See eigh_band.mm.
            void eigh_band(const Matrices& a, bool lower, float* w, uint32_t* info, uint32_t width = 0);

            // Householder tridiagonalization and implicit QL, one threadgroup
            // per matrix, N <= metal_linalg::detail::eigh_ql_max_n(). `info`
            // counts QL iterations where the others count sweeps. See
            // eigh_ql.mm.
            void eigh_ql(const Matrices& a, bool lower, float* w, float* v, uint32_t* info);

            // eigh_ql and eigh_cpu on one batch at once, sharing it as
            // EighPolicy::share_min_batch describes, whatever the policy.
            void eigh_ql_shared(const Matrices& a, bool lower, float* w, float* v, uint32_t* info);

            // LAPACK on the CPU: ssyevd, or ssyevd_2stage for eigenvalues
            // alone (v == nullptr) from N = 128, the matrices of a batch
            // spread over cpu_threads() threads. `info` reports every finite
            // matrix as converged in one sweep.
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

            // LAPACK's method with the bidiagonalization and the
            // back-transformations on the GPU, one matrix at a time; any shape
            // (a wide matrix as its transpose, a much taller one after a QR).
            // See svd_bidiag.mm.
            void svd_bidiag(const Matrices& a, float* u, float* s, float* vt, uint32_t* info);

            // Singular values alone by the two-stage reduction: A to a band
            // on the GPU (blocked panels, the work as matrix products), the
            // band to bidiagonal (LAPACK sgbbrd) and its singular values
            // (sbdsqr) on the CPU. `width` the band's, 8, 16 or 32 (0: the
            // default, or SVD_BAND_WIDTH). See svd_bidiag.mm.
            void svd_band(const Matrices& a, float* s, uint32_t* info, uint32_t width = 0);

            // The SVD with vectors by the two-stage reduction: the band's
            // reflectors and the chase's kept and applied on the GPU while
            // the CPU solves the bidiagonal problem. Width 16. See
            // svd_bidiag.mm.
            void svd_band_vectors(const Matrices& a, float* u, float* s, float* vt, uint32_t* info);

            // Householder bidiagonalization and implicit bidiagonal QR, one
            // threadgroup per matrix, for shapes that
            // metal_linalg::detail::svd_gk_fits(); with the QR first, through
            // svd_qr_jacobi with SvdOptions::Kernel::golub_kahan. `info`
            // counts QR steps where the Jacobi backends count sweeps. See
            // svd_golub_kahan.mm.
            void svd_golub_kahan(const Matrices& a, float* u, float* s, float* vt, uint32_t* info);

            // golub_kahan (after the QR where the shape does not fit) and
            // svd_cpu on one batch at once, sharing it as
            // EighPolicy::share_min_batch describes, whatever the policy.
            void svd_golub_kahan_shared(const Matrices& a, float* u, float* s, float* vt, uint32_t* info);

            // LAPACK on the CPU: sgesdd, after a QR for a matrix at least
            // twice as tall as wide, the matrices of a batch spread over
            // cpu_threads() threads. `info` reports every finite matrix as
            // converged in one sweep.
            void svd_cpu(const Matrices& a, float* u, float* s, float* vt, uint32_t* info);
        }

    } // namespace core

} // namespace metal_linalg
