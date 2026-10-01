#pragma once
#include <metal_linalg/core.h>   // QrPolicy, qr_backend and the rest of the routing

#include <mlx/mlx.h>
#include <utility>

namespace metal_linalg {

    // -------------------------------------------------------------------------
    // QR decomposition on the GPU
    // -------------------------------------------------------------------------
    // For a batch of real matrices A [..., M, N], computes the economic
    // factorisation A = Q R with K = min(M, N): Q [..., M, K] with
    // orthonormal columns and R [..., K, N] upper triangular. Output is
    // float32; any shape and magnitude is accepted. Routed to one of two GPU
    // kernels or to LAPACK on the CPU by the policy in core.h.
    std::pair<mlx::core::array, mlx::core::array> qr_accelerated(const mlx::core::array& a);

    // -------------------------------------------------------------------------
    // Lower-level entry points, for tests and tuning
    // -------------------------------------------------------------------------
    // Each backend on its own, as described in core.h. Every backend returns
    // the economic factorisation.
    namespace detail {
        std::pair<mlx::core::array, mlx::core::array>
        qr_unblocked(const mlx::core::array& a);

        std::pair<mlx::core::array, mlx::core::array>
        qr_streaming_amx_reduced(const mlx::core::array& a);

        std::pair<mlx::core::array, mlx::core::array>
        qr_streaming_amx_complete(const mlx::core::array& a);

        std::pair<mlx::core::array, mlx::core::array>
        qr_cpu(const mlx::core::array& a);
    }

} // namespace metal_linalg
