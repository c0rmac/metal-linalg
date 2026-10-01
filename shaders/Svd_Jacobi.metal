#include "eigh_jacobi_common.h"

// =============================================================================
// Batched singular value decomposition: one-sided Jacobi, one simdgroup per pair
// =============================================================================
//
// Computes the thin SVD A = U diag(sigma) V^T of a batch of M x N matrices,
// M >= N, one matrix per threadgroup. (The host transposes wide inputs.)
//
// The method is Hestenes' one-sided Jacobi. It works on the columns g_1..g_N
// of G = A directly: for a pair (p, q) it forms the three inner products
//
//     alpha = g_p . g_p      beta = g_q . g_q      gamma = g_p . g_q
//
// and applies the plane rotation that makes the two columns orthogonal, to
// the columns of G and to the same columns of V (V starts as I). When every
// pair is orthogonal to working precision, G = U diag(sigma): the column
// norms are the singular values and the normalised columns are U. This is
// two-sided Jacobi on A^T A without ever forming A^T A, which is what keeps
// the small singular values accurate (Demmel & Veselic 1992; Drmac & Veselic
// 2008 is the modern form and what LAPACK's xGESVJ implements).
//
// Why this is a better GPU algorithm than the two-sided method in
// Eigh_Jacobi.metal: a rotation touches only columns p and q. With the
// round-robin ordering the N/2 pairs of a round are disjoint, so if each pair
// is owned by one simdgroup, nothing a simdgroup reads or writes during a
// round is touched by any other. The inner products are a simd_sum across the
// 32 lanes, the rotation parameters are uniform across them, and the rotation
// is applied by the same lanes -- no barrier anywhere inside a round. The only
// synchronisation is one threadgroup barrier *between* rounds, where the two-
// sided method needs three (parameters, rows, columns) because its row and
// column updates of different pairs meet at shared elements.
//
// Layout: G and V are stored by column (G[c * M + i] is row i of column c), so
// a lane walking down a column walks contiguous memory and the 32 lanes of a
// simdgroup read 32 consecutive elements. Input and outputs are row-major.
//
// Scaling, non-finite input and the descending sort follow the eigensolver.
// Columns whose singular value is below null_tol * sigma_max carry no
// direction information (g / sigma is noise); their U columns are written as
// zero and flagged, and the host completes them to an orthonormal basis. The
// same threshold keeps such columns from being rotated against each other,
// which is what makes the method terminate on rank-deficient input.
//
// References:
//   Hestenes, "Inversion of matrices by biorthogonalization and related
//     results", J. SIAM 6 (1958).
//   Brent & Luk (1985) -- the parallel ordering; see eigh_jacobi_common.h.
//   Demmel & Veselic, "Jacobi's method is more accurate than QR", SIAM J.
//     Matrix Anal. Appl. 13 (1992).
//   Drmac & Veselic, "New fast and accurate Jacobi SVD algorithm I, II", SIAM
//     J. Matrix Anal. Appl. 29 (2008).
//
// Compile with -fno-fast-math, as for the eigensolver.

constant bool kComputeUV [[function_constant(0)]];

// Must match `Params` in svd.mm.
struct SvdParams {
    uint  m;           // rows, m >= n
    uint  n;           // columns
    uint  n_pairs;     // ceil(n / 2)
    uint  max_sweeps;
    float tol;         // a pair is rotated iff |gamma| > tol * sqrt(alpha * beta)
    float null_tol;    // a column below null_tol * (largest column) is numerically null
};

constant uint kSvdConverged     = 1u << 16;
constant uint kSvdNonFinite     = 1u << 17;
constant uint kSvdRankDeficient = 1u << 18;

// Above this |zeta| the closed form would square it past float32 range; the
// rotation there is t = 1 / (2 zeta) to working precision anyway.
constant float kZetaAsymptotic = 1e15f;

// Cosine above which a null column still counts as unclean; see the sweep.
constant float kCleanCos = 0.05f;

// A column below this fraction of the largest is under the rounding of
// everything else in the matrix and never keeps the sweeps going.
constant float kNegligible = 1.2e-7f;

kernel void svd_jacobi(
    device const float*  A_in [[buffer(0)]],  // [batch, m, n] row-major
    device float*        G    [[buffer(1)]],  // [batch, n, m] working columns
    device float*        Vw   [[buffer(2)]],  // [batch, n, n] working columns of V
    device float*        S    [[buffer(3)]],  // [batch, n] descending
    device float*        U    [[buffer(4)]],  // [batch, m, n] row-major
    device float*        Vt   [[buffer(5)]],  // [batch, n, n] row-major
    device uint*         info [[buffer(6)]],  // [batch] sweeps | flags
    constant SvdParams&  prm  [[buffer(7)]],
    threadgroup float*   tg   [[threadgroup(0)]],  // red[32] flags[32] sig[n] rank[n]
    uint b    [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint T    [[threads_per_threadgroup]],
    uint sg   [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint n_sg [[simdgroups_per_threadgroup]])
{
    const uint m = prm.m, n = prm.n, np = prm.n_pairs, n_even = 2 * np;
    const uint mn = m * n, nn = n * n;

    threadgroup float*  red   = tg;
    threadgroup float*  flags = tg + kRedFloats;
    threadgroup float*  sig   = tg + 2 * kRedFloats;
    threadgroup ushort* rank  = reinterpret_cast<threadgroup ushort*>(sig + n);

    device const float* a = A_in + (ulong)b * mn;
    device float*       g = G    + (ulong)b * mn;
    device float*       v = Vw   + (ulong)b * nn;

    // -------------------------------------------------------------------------
    // Load by column, V = I, largest entry, non-finite scan
    // -------------------------------------------------------------------------
    float amax = 0.0f, bad = 0.0f;
    for (uint idx = tid; idx < mn; idx += T) {
        const uint i = idx / n;
        const uint c = idx - i * n;
        const float x = a[idx];
        g[(ulong)c * m + i] = x;
        amax = fmax(amax, fabs(x));
        if (non_finite(x)) bad = 1.0f;
    }
    if (kComputeUV) {
        for (uint idx = tid; idx < nn; idx += T) {
            const uint c = idx / n;
            v[idx] = (idx - c * n == c) ? 1.0f : 0.0f;
        }
    }
    threadgroup_barrier(mem_flags::mem_device);
    amax = team_max(amax, false, red, sg, lane, n_sg);
    const bool nonfinite = team_sum(bad, false, red, sg, lane, n_sg) > 0.0f;

    // Scale by a power of two so the largest entry is in [0.5, 1): the inner
    // products square entries, which would overflow or underflow float32 for
    // inputs far from unit magnitude.
    int expo = 0;
    if (amax > 0.0f) frexp(amax, expo);
    if (!nonfinite) {
        for (uint idx = tid; idx < mn; idx += T) g[idx] = ldexp(g[idx], -expo);
        threadgroup_barrier(mem_flags::mem_device);
    }

    // -------------------------------------------------------------------------
    // Sweeps. A sweep that rotates nothing is the convergence test, so the
    // count includes that final verifying sweep.
    // -------------------------------------------------------------------------
    uint sweeps = 0;
    bool converged = false;
    while (!nonfinite && sweeps < prm.max_sweeps) {
        // Null columns. A column whose norm is below null_tol times the
        // largest column's has no direction of its own left: in a rank-
        // deficient matrix it is what remains after cancellation, rounding
        // noise. Rotating two such columns against each other is where one-
        // sided Jacobi fails to terminate. The rotation amplifies their
        // relative error, which puts them back out of line with the large
        // columns, whose correction changes their angles to one another, and
        // so on down to underflow. So:
        //
        //   null against null      never rotated
        //   null against the rest  rotated like any pair, which drives the
        //                          null column down to noise, but it only
        //                          keeps the sweeps going while the angle is
        //                          gross (kCleanCos) and the column is not
        //                          yet negligible (kNegligible). Past either
        //                          point the null column's norm, the one
        //                          thing about it the result uses, has
        //                          stopped mattering, and the angle itself is
        //                          noise that would never settle.
        //
        // null_tol is a few tens of epsilon: just above the rounding floor,
        // where null singular values are observed (1e-8 to 1e-7 of the
        // largest), and independent of the matrix's size. It is deliberately
        // not the rank tolerance max(M, N) * eps: that would discard the
        // direction of every singular value under about 1e-5 of the largest,
        // which one-sided Jacobi computes accurately.
        float cmax = 0.0f;
        for (uint c = sg; c < n; c += n_sg) {
            device float* gc = g + (ulong)c * m;
            float acc = 0.0f;
            for (uint i = lane; i < m; i += 32) {
                const float x = gc[i];
                acc += x * x;
            }
            cmax = fmax(cmax, simd_sum(acc));
        }
        const float cmax2 = team_max(cmax, false, red, sg, lane, n_sg);
        const float null2 = prm.null_tol * prm.null_tol * cmax2;
        const float negl2 = kNegligible * kNegligible * cmax2;

        // Only this simdgroup writes its flag, and it is only read after the
        // barrier that ends the sweep.
        if (lane == 0) flags[sg] = 0.0f;

        for (uint round = 0; round + 1 < n_even; ++round) {
            bool rotated = false;
            for (uint j = sg; j < np; j += n_sg) {
                uint p, q;
                tournament_pair(round, j, n_even, p, q);
                if (q >= n) continue;   // the dummy partner when n is odd

                device float* gp = g + (ulong)p * m;
                device float* gq = g + (ulong)q * m;

                float sa = 0.0f, sb = 0.0f, sc = 0.0f;
                for (uint i = lane; i < m; i += 32) {
                    const float x = gp[i], y = gq[i];
                    sa += x * x;
                    sb += y * y;
                    sc += x * y;
                }
                const float alpha = simd_sum(sa);
                const float beta  = simd_sum(sb);
                const float gamma = simd_sum(sc);

                // Two numerically null columns are left alone; see the note
                // on null columns at the top of the sweep.
                if (alpha <= null2 && beta <= null2) continue;

                // Relative test, so a column of any magnitude is judged on
                // its angle to the other; a zero column gives 0 > 0.
                const float ab = alpha * beta;
                if (fabs(gamma) > prm.tol * sqrt(ab)) {
                    const float zeta = (beta - alpha) / (2.0f * gamma);
                    float t;
                    if (fabs(zeta) > kZetaAsymptotic) {
                        t = 0.5f / zeta;
                    } else {
                        t = 1.0f / (fabs(zeta) + sqrt(1.0f + zeta * zeta));
                        if (zeta < 0.0f) t = -t;
                    }
                    const float c = rsqrt(1.0f + t * t);
                    const float s = c * t;

                    for (uint i = lane; i < m; i += 32) {
                        const float x = gp[i], y = gq[i];
                        gp[i] = c * x - s * y;
                        gq[i] = s * x + c * y;
                    }
                    if (kComputeUV) {
                        device float* vp = v + (ulong)p * n;
                        device float* vq = v + (ulong)q * n;
                        for (uint i = lane; i < n; i += 32) {
                            const float x = vp[i], y = vq[i];
                            vp[i] = c * x - s * y;
                            vq[i] = s * x + c * y;
                        }
                    }
                    const bool one_null = alpha <= null2 || beta <= null2;
                    if (!one_null) {
                        rotated = true;
                    } else if (min(alpha, beta) > negl2 &&
                               gamma * gamma > kCleanCos * kCleanCos * ab) {
                        rotated = true;
                    }
                }
            }
            if (rotated && lane == 0) flags[sg] = 1.0f;

            // The one synchronisation point of a round: the next round pairs
            // columns that other simdgroups have just written.
            threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        }
        ++sweeps;

        float any = 0.0f;
        for (uint k = 0; k < n_sg; ++k) any += flags[k];
        threadgroup_barrier(mem_flags::mem_threadgroup);   // readers done before the reset above
        if (any == 0.0f) { converged = true; break; }
    }

    // -------------------------------------------------------------------------
    // Output: sigma = column norms, descending; U = columns / sigma; V^T
    // -------------------------------------------------------------------------
    for (uint c = sg; c < n; c += n_sg) {
        device float* gc = g + (ulong)c * m;
        float acc = 0.0f;
        for (uint i = lane; i < m; i += 32) {
            const float x = gc[i];
            acc += x * x;
        }
        const float s2 = simd_sum(acc);
        if (lane == 0) sig[c] = sqrt(s2);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float pmax = 0.0f;
    for (uint c = tid; c < n; c += T) {
        const float sc = sig[c];
        pmax = fmax(pmax, sc);
        uint r = 0;
        for (uint k = 0; k < n; ++k) {
            const float sk = sig[k];
            r += (sk > sc || (sk == sc && k < c)) ? 1u : 0u;
        }
        rank[c] = (ushort)r;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float smax = team_max(pmax, false, red, sg, lane, n_sg);
    const float thr  = prm.null_tol * smax;

    float low = 0.0f;
    for (uint c = tid; c < n; c += T) if (!(sig[c] > thr)) low = 1.0f;
    const bool deficient = team_sum(low, false, red, sg, lane, n_sg) > 0.0f;

    device float* out_s = S + (ulong)b * n;
    if (nonfinite) {
        const float qnan = as_type<float>(0x7FC00000u);
        for (uint c = tid; c < n; c += T) out_s[c] = qnan;
        if (kComputeUV) {
            device float* out_u  = U  + (ulong)b * mn;
            device float* out_vt = Vt + (ulong)b * nn;
            for (uint idx = tid; idx < mn; idx += T) out_u[idx]  = qnan;
            for (uint idx = tid; idx < nn; idx += T) out_vt[idx] = qnan;
        }
    } else {
        for (uint c = tid; c < n; c += T) out_s[rank[c]] = ldexp(sig[c], expo);
        if (kComputeUV) {
            device float* out_u  = U  + (ulong)b * mn;
            device float* out_vt = Vt + (ulong)b * nn;
            for (uint idx = tid; idx < mn; idx += T) {
                const uint c = idx / m;
                const uint i = idx - c * m;
                const float sc = sig[c];
                out_u[(ulong)i * n + rank[c]] = (sc > thr) ? g[idx] / sc : 0.0f;
            }
            for (uint idx = tid; idx < nn; idx += T) {
                const uint c = idx / n;
                const uint i = idx - c * n;
                out_vt[(ulong)rank[c] * n + i] = v[idx];
            }
        }
    }

    if (tid == 0) {
        info[b] = sweeps | (converged ? kSvdConverged : 0u) | (nonfinite ? kSvdNonFinite : 0u)
                         | ((deficient && !nonfinite) ? kSvdRankDeficient : 0u);
    }
}
