// Shared pieces of the two Jacobi eigensolvers (Eigh_Jacobi.metal, one team
// per matrix; Eigh_BlockJacobi.metal, one team per 2b x 2b subproblem).
//
// The matrix being rotated lives either in device memory (the whole-matrix
// kernel) or in threadgroup memory (the block subproblem), so the phases are
// templated on the pointer type and the address space comes with it.
//
// Algorithm and references: see the header comment of Eigh_Jacobi.metal.

#pragma once

#include <metal_stdlib>

using namespace metal;

// A pair whose off-diagonal element is below this fraction of the diagonal
// magnitudes is not rotated. It bounds |theta| = |a_qq - a_pp| / (2 |a_pq|) at
// 5e11, keeping theta^2 far from float32 overflow, and drops a backward error
// five orders of magnitude below float32 epsilon.
constant float kSkipRelative = 1e-12f;

// Slots at the start of threadgroup memory for the cross-simdgroup reduction.
constant uint kRedFloats = 32;

inline bool non_finite(float x) {
    return ((as_type<uint>(x) >> 23) & 0xFFu) == 0xFFu;
}

// Round-robin tournament: element 0 stays put, elements 1..n_even-1 rotate one
// position per round, and position i pairs with position n_even-1-i. Over
// n_even-1 rounds every pair meets exactly once and the pairs within a round
// are disjoint, which is what lets them be rotated simultaneously.
inline void tournament_pair(uint round, uint j, uint n_even,
                            thread uint& p, thread uint& q) {
    const uint m = n_even - 1;
    uint a, b;
    if (j == 0) {
        a = 0;
        b = 1 + (m - 1 + round) % m;
    } else {
        a = 1 + (j - 1 + round) % m;
        b = 1 + (m - 1 - j + round) % m;
    }
    p = min(a, b);
    q = max(a, b);
}

// Team-wide barrier. In simd mode the team is one simdgroup, so a simdgroup
// barrier is both sufficient and much cheaper. `simd` is a function constant
// or a literal at every call site, so the branch folds away.
inline void team_barrier(bool simd, mem_flags flags) {
    if (simd) simdgroup_barrier(flags);
    else      threadgroup_barrier(flags);
}

// Sums `v` over the team. Uniform result on every thread.
inline float team_sum(float v, bool simd, threadgroup float* red,
                      uint sg_id, uint lane, uint n_sg) {
    v = simd_sum(v);
    if (simd) return v;

    // The previous call's readers may still be looking at `red`.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) red[sg_id] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float total = 0.0f;
    for (uint i = 0; i < n_sg; ++i) total += red[i];
    return total;
}

// Max of `v` (assumed >= 0) over the team. Uniform result on every thread.
inline float team_max(float v, bool simd, threadgroup float* red,
                      uint sg_id, uint lane, uint n_sg) {
    v = simd_max(v);
    if (simd) return v;

    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) red[sg_id] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float m = 0.0f;
    for (uint i = 0; i < n_sg; ++i) m = fmax(m, red[i]);
    return m;
}

// Per-pair rotation parameters for one round, in threadgroup memory.
struct JacobiScratch {
    threadgroup float*  c;
    threadgroup float*  s;
    threadgroup float*  dp;   // exact new a_pp
    threadgroup float*  dq;   // exact new a_qq
    threadgroup ushort* p;
    threadgroup ushort* q;
};

// Carves a JacobiScratch for `np` pairs out of `base`: 4*np floats then 2*np
// ushorts, i.e. 5*np floats' worth of space (pad to 4 floats for alignment).
inline JacobiScratch jacobi_scratch(threadgroup float* base, uint np) {
    JacobiScratch sc;
    sc.c  = base;
    sc.s  = base + np;
    sc.dp = base + 2 * np;
    sc.dq = base + 3 * np;
    sc.p  = reinterpret_cast<threadgroup ushort*>(base + 4 * np);
    sc.q  = sc.p + np;
    return sc;
}

// --- phase 0: rotation parameters, one pair per thread -----------------------
// `null2`, when not negative, is the squared norm at or below which a diagonal
// entry counts as numerically null; a pair of two such entries is not rotated.
// Used when the matrix is a Gram matrix of columns (the block SVD), whose null
// columns must not be rotated against each other; see Svd_Jacobi.metal. The
// eigensolvers leave it off: a symmetric matrix may have any diagonal.
template <typename WPtr>
inline void jacobi_phase0(uint round, uint n, uint np, uint n_even, uint t, uint T,
                          WPtr w, JacobiScratch sc, float null2 = -1.0f) {
    for (uint j = t; j < np; j += T) {
        uint p, q;
        tournament_pair(round, j, n_even, p, q);

        float c = 1.0f, s = 0.0f, dp = 0.0f, dq = 0.0f;
        if (q < n) {   // q == n is the dummy partner when n is odd
            const float app = w[p * n + p];
            const float aqq = w[q * n + q];
            const float apq = w[p * n + q];
            dp = app;
            dq = aqq;
            const bool both_null = null2 >= 0.0f && app <= null2 && aqq <= null2;
            if (!both_null && fabs(apq) > kSkipRelative * (fabs(app) + fabs(aqq))) {
                // Golub & Van Loan Algorithm 8.5.1: the smaller-angle root,
                // which keeps the rotation close to identity and is what
                // makes cyclic Jacobi converge quadratically.
                const float theta = (aqq - app) / (2.0f * apq);
                float tt = 1.0f / (fabs(theta) + sqrt(1.0f + theta * theta));
                if (theta < 0.0f) tt = -tt;
                c  = rsqrt(1.0f + tt * tt);
                s  = tt * c;
                dp = app - tt * apq;
                dq = aqq + tt * apq;
            }
        }
        sc.c[j] = c; sc.s[j] = s; sc.dp[j] = dp; sc.dq[j] = dq;
        sc.p[j] = (ushort)p; sc.q[j] = (ushort)q;
    }
}

// --- phase 1: A <- J^T A. Item = (pair, column); consecutive threads take
//     consecutive columns of the same two rows. -------------------------------
template <typename WPtr>
inline void jacobi_phase1(uint n, uint np, uint t, uint T, WPtr w, JacobiScratch sc) {
    for (uint idx = t; idx < np * n; idx += T) {
        const uint j = idx / n;
        const uint k = idx - j * n;
        const uint q = sc.q[j];
        if (q < n) {
            const uint p = sc.p[j];
            const float c = sc.c[j], s = sc.s[j];
            WPtr rp = w + p * n;
            WPtr rq = w + q * n;
            const float x = rp[k], y = rq[k];
            rp[k] = c * x - s * y;
            rq[k] = s * x + c * y;
        }
    }
}

// --- phase 2: A <- A J and V <- V J. Item = (row, pair); consecutive threads
//     take consecutive pairs of the same row, so a simdgroup walks one
//     contiguous row. The 2x2 block of each pair is overwritten with the
//     analytic (d_p, 0, 0, d_q) rather than rotated; see Eigh_Jacobi.metal. ---
template <typename WPtr, typename VPtr>
inline void jacobi_phase2(uint n, uint np, uint t, uint T, WPtr w, VPtr v,
                          bool vectors, JacobiScratch sc) {
    for (uint idx = t; idx < n * np; idx += T) {
        const uint k = idx / np;
        const uint j = idx - k * np;
        const uint q = sc.q[j];
        if (q < n) {
            const uint p = sc.p[j];
            const float c = sc.c[j], s = sc.s[j];
            WPtr row = w + k * n;
            if (k == p) {
                row[p] = sc.dp[j];
                row[q] = 0.0f;
            } else if (k == q) {
                row[p] = 0.0f;
                row[q] = sc.dq[j];
            } else {
                const float x = row[p], y = row[q];
                row[p] = c * x - s * y;
                row[q] = s * x + c * y;
            }
            if (vectors) {
                VPtr vrow = v + k * n;
                const float x = vrow[p], y = vrow[q];
                vrow[p] = c * x - s * y;
                vrow[q] = s * x + c * y;
            }
        }
    }
}

// One full sweep: n_even - 1 rounds of the tournament, three phases each.
// `w` and `v` are n x n row-major with leading dimension n. Both barriers
// after the rotation phases carry both flags, since w may be in either
// address space; the cost difference is not measurable.
template <typename WPtr, typename VPtr>
inline void jacobi_sweep(uint n, uint np, uint t, uint T, bool simd,
                         WPtr w, VPtr v, bool vectors, JacobiScratch sc, float null2 = -1.0f) {
    const uint n_even = 2 * np;
    for (uint round = 0; round + 1 < n_even; ++round) {
        jacobi_phase0(round, n, np, n_even, t, T, w, sc, null2);
        team_barrier(simd, mem_flags::mem_threadgroup);

        jacobi_phase1(n, np, t, T, w, sc);
        team_barrier(simd, mem_flags::mem_device | mem_flags::mem_threadgroup);

        jacobi_phase2(n, np, t, T, w, v, vectors, sc);
        team_barrier(simd, mem_flags::mem_device | mem_flags::mem_threadgroup);
    }
}

// Rank sort of the first n diagonal entries of `w` (leading dimension ld):
// stages them in `lam`, leaves rank[i] = position of entry i in ascending
// order. Stable and deterministic; O(n) threadgroup reads per thread. Both
// barriers are threadgroup-wide (or simdgroup-wide in simd mode).
inline void jacobi_rank_sort(uint n, uint ld, uint t, uint T, bool simd,
                             device const float* w,
                             threadgroup float* lam, threadgroup ushort* rank) {
    for (uint i = t; i < n; i += T) lam[i] = w[i * ld + i];
    team_barrier(simd, mem_flags::mem_threadgroup);

    for (uint i = t; i < n; i += T) {
        const float li = lam[i];
        uint r = 0;
        for (uint j = 0; j < n; ++j) {
            const float lj = lam[j];
            r += (lj < li || (lj == li && j < i)) ? 1u : 0u;
        }
        rank[i] = (ushort)r;
    }
    team_barrier(simd, mem_flags::mem_threadgroup);
}
