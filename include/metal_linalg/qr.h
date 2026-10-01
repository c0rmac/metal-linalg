#pragma once
#include <mlx/mlx.h>
#include <utility>

namespace metal_linalg {

    // -------------------------------------------------------------------------
    // QR decomposition on the GPU
    // -------------------------------------------------------------------------
    // For a batch of real matrices A [..., M, N], computes the economic
    // factorisation A = Q R with K = min(M, N): Q [..., M, K] with
    // orthonormal columns and R [..., K, N] upper triangular. Output is
    // float32; any shape and magnitude is accepted.
    std::pair<mlx::core::array, mlx::core::array> qr_accelerated(const mlx::core::array& a);

    // -------------------------------------------------------------------------
    // Routing policy
    // -------------------------------------------------------------------------
    // Two GPU backends, a single-threadgroup kernel for small matrices and a
    // grid-parallel one for large, and which is faster depends on the GPU, so
    // the crossover is a per-device policy measured by tuning/tune_qr.py; see
    // docs/tuning.md.
    struct QrPolicy {
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

        // Device properties this was resolved against. Informational: they are
        // detected, not assumed, and are what a retune should be keyed on.
        unsigned gpu_cores = 0;           // 0 if it could not be detected
        unsigned concurrent_matrices = 0; // qr_unblocked threadgroups resident at once
    };

    // The policy in effect, resolved once on first use; where it came from
    // ("user", "env:QR_M_CROSSOVER", "tuned:<device>" or
    // "default:untuned-device (<device>)"); and a way to replace it, which
    // takes precedence over the environment and the tuned table.
    QrPolicy    qr_policy();
    const char* qr_policy_source();
    void        set_qr_policy(const QrPolicy& p);

    enum class QrBackend { unblocked, streaming_reduced };

    // What qr_accelerated does with a problem under the policy in effect.
    QrBackend qr_backend(unsigned m, unsigned n, unsigned batch);

    // -------------------------------------------------------------------------
    // Lower-level entry points, for tests and tuning
    // -------------------------------------------------------------------------
    // Every backend returns the economic factorisation.
    namespace detail {
        // Standard Householder QR in a single kernel dispatch, one threadgroup
        // per matrix. Preferred for small matrices, where multi-pass streaming
        // does not pay for its launch overhead.
        std::pair<mlx::core::array, mlx::core::array>
        qr_unblocked(const mlx::core::array& a);

        // Multi-pass streaming panel factorisation that accumulates Q directly
        // at its economic K-column width via a backward pass. The grid-parallel
        // path for large matrices.
        std::pair<mlx::core::array, mlx::core::array>
        qr_streaming_amx_reduced(const mlx::core::array& a);

        // As above, but accumulates the full M x M orthogonal factor before
        // slicing Q down to K columns.
        //
        // Not reachable from qr_accelerated: benchmarking put it within noise of
        // qr_streaming_amx_reduced everywhere it was measured, while allocating
        // the full M x M Q. Kept and tested rather than deleted, since it is the
        // only backend that forms the complete orthogonal factor.
        std::pair<mlx::core::array, mlx::core::array>
        qr_streaming_amx_complete(const mlx::core::array& a);
    }

} // namespace metal_linalg
