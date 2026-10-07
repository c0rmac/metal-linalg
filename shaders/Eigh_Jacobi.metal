#include "eigh_jacobi_common.h"

// =============================================================================
// Batched symmetric eigensolver: cyclic Jacobi with a parallel ordering
// =============================================================================
//
// Computes A = V diag(lambda) V^T for a batch of real symmetric N x N matrices,
// one matrix per "team" of threads. A team is either a whole threadgroup
// (kSimdMode = false) or a single 32-lane simdgroup (kSimdMode = true, for tiny
// matrices, where threadgroup barriers would dominate).
//
// Why Jacobi rather than tridiagonalisation + QL, which is what LAPACK's
// ssyev does on a CPU:
//
//   * Parallelism. One Jacobi step rotates N/2 disjoint (p, q) pairs at once,
//     touching every element of A, so a sweep is only N-1 barrier-separated
//     steps with N^2 / 2 independent work items each. Tridiagonal QL applies
//     ~N^2 dependent Givens rotations per matrix, each of which is a barrier.
//     For a threadgroup-per-matrix design that is the difference between
//     ~8N and ~1.5N^2 synchronisations.
//   * Accuracy. Two-sided Jacobi computes eigenvalues with high relative
//     accuracy for positive definite input (Demmel & Veselic 1992), which
//     matters in float32.
//   * It is the batched-small-matrix choice in production GPU libraries
//     (cuSOLVER syevjBatched is a Jacobi solver for exactly this regime).
//
// References:
//   Golub & Van Loan, Matrix Computations 4th ed., s8.5 (Jacobi methods;
//     Algorithm 8.5.1 for the 2x2 symmetric Schur decomposition and s8.5.8
//     for the parallel ordering).
//   Brent & Luk, "The solution of singular-value and symmetric eigenvalue
//     problems on multiprocessor arrays", SIAM J. Sci. Stat. Comput. 6 (1985)
//     -- the round-robin pairing.
//   Rutishauser, "The Jacobi method for real symmetric matrices", Numer. Math.
//     9 (1966) -- the analytic diagonal update that keeps rounding noise off
//     the off-diagonal.
//   Demmel & Veselic, "Jacobi's method is more accurate than QR", SIAM J.
//     Matrix Anal. Appl. 13 (1992).
//
// Layout: everything is row-major. W (the working copy of A) and V live in
// device memory; only the per-step rotation parameters live in threadgroup
// memory, so N is bounded by that small footprint (20 bytes per pair) rather
// than by N^2.
//
// One step, for every pair (p, q) in the current round of the tournament:
//
//   phase 0   each pair computes its rotation (c, s) from a_pp, a_qq, a_pq and
//             the exact new diagonal d_p = a_pp - t a_pq, d_q = a_qq + t a_pq
//   phase 1   A <- J^T A     rows p, q of A rotated, all columns
//   phase 2   A <- A J       columns p, q of A rotated, all rows; V <- V J
//
// In phase 2 the 2x2 block (p,p) (p,q) (q,p) (q,q) is overwritten with
// (d_p, 0, 0, d_q) rather than computed by rotation. The rotated value of a_pq
// is a difference of O(|a_pp|) terms that cancels to zero only in exact
// arithmetic, and leaving that rounding residue on the off-diagonal would put
// a floor of eps * |a_pp| under the off-norm. With the analytic update every
// off-diagonal element only ever mixes with other off-diagonal elements, so the
// off-norm converges quadratically to roundoff of *itself* and the stopping
// test is reachable.
//
// The phases themselves are in eigh_jacobi_common.h, shared with the block
// Jacobi backend, which runs them on 2b x 2b subproblems in threadgroup memory.
//
// Compile with -fno-fast-math (see CMakeLists.txt): the rotation must satisfy
// c^2 + s^2 = 1 to roundoff or V drifts from orthogonal, and the non-finite
// input check relies on IEEE semantics.

constant bool kComputeVectors [[function_constant(0)]];
constant bool kSimdMode       [[function_constant(1)]];

// Must match `Params` in eigh.mm.
struct EighParams {
    uint  n;                // matrix order
    uint  n_pairs;          // ceil(n / 2): pairs per tournament round
    uint  batch;            // matrices in the batch
    uint  max_sweeps;       // upper bound on sweeps; convergence normally stops earlier
    float tol;              // stop when off(A) <= tol * ||A||_F
    uint  lower;            // 1: symmetrise from the lower triangle, 0: from the upper
    uint  matrices_per_tg;  // simd mode: simdgroups (= matrices) per threadgroup
    uint  tg_stride_floats; // simd mode: per-matrix threadgroup scratch, in floats
    uint  round_budget;     // threadgroup mode: rounds per dispatch, resumed from `st` (0: the whole solve)
};

// Info word written per matrix.
constant uint kInfoConverged = 1u << 16;
constant uint kInfoNonFinite = 1u << 17;

kernel void eigh_jacobi(
    device const float*  A_in  [[buffer(0)]],  // [batch, n, n] input (row-major)
    device float*        W     [[buffer(1)]],  // [batch, n, n] working copy of A; receives V on output
    device float*        V     [[buffer(2)]],  // [batch, n, n] eigenvector accumulator (kComputeVectors)
    device float*        vals  [[buffer(3)]],  // [batch, n] eigenvalues, ascending
    device uint*         info  [[buffer(4)]],  // [batch] sweeps | kInfoConverged | kInfoNonFinite
    constant EighParams& prm   [[buffer(5)]],
    device JacobiState*  st    [[buffer(6)]],  // [batch] (round_budget > 0)
    threadgroup float*   tg    [[threadgroup(0)]],
    uint tg_id   [[threadgroup_position_in_grid]],
    uint tid     [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint sg_id   [[simdgroup_index_in_threadgroup]],
    uint lane    [[thread_index_in_simdgroup]],
    uint n_sg    [[simdgroups_per_threadgroup]])
{
    const uint n  = prm.n;
    const uint np = prm.n_pairs;
    const uint nn = n * n;

    // -------------------------------------------------------------------------
    // Team setup
    // -------------------------------------------------------------------------
    uint b, t, T;
    threadgroup float* red = tg;
    threadgroup float* fbase;
    if (kSimdMode) {
        b = tg_id * prm.matrices_per_tg + sg_id;
        if (b >= prm.batch) return;   // whole simdgroup leaves together; no threadgroup barriers follow
        t = lane;
        T = 32;
        fbase = tg + kRedFloats + sg_id * prm.tg_stride_floats;
    } else {
        b = tg_id;
        t = tid;
        T = tg_size;
        fbase = tg + kRedFloats;
    }
    const JacobiScratch sc = jacobi_scratch(fbase, np);

    device const float* a = A_in + (ulong)b * nn;
    device float*       w = W    + (ulong)b * nn;
    device float*       v = V    + (ulong)b * nn;

    // Split over dispatches (threadgroup mode): a finished matrix is left
    // alone, a started one resumes where it stopped.
    const bool split = !kSimdMode && prm.round_budget != 0;
    const JacobiState s0 = split ? st[b] : JacobiState{0, 0.0f, 0.0f, 0u, 0u, 0u};
    if (s0.flags & kJacobiDone) return;
    const bool resume = (s0.flags & kJacobiStarted) != 0;

    int expo = 0;
    float fro2 = 0.0f;
    bool nonfinite = false;
    if (resume) {
        expo = s0.expo;
        fro2 = s0.a;
    } else {
    // -------------------------------------------------------------------------
    // Load: symmetrise from the requested triangle, V = I, scan for the
    // largest entry and for non-finite ones
    // -------------------------------------------------------------------------
    float amax = 0.0f;
    float bad  = 0.0f;
    for (uint idx = t; idx < nn; idx += T) {
        const uint i = idx / n;
        const uint j = idx - i * n;
        const bool direct = prm.lower ? (i >= j) : (i <= j);
        const float x = direct ? a[idx] : a[j * n + i];
        w[idx] = x;
        amax = fmax(amax, fabs(x));   // fmax skips NaN, hence the separate flag
        if (non_finite(x)) bad = 1.0f;
        if (kComputeVectors) v[idx] = (i == j) ? 1.0f : 0.0f;
    }
    team_barrier(kSimdMode, mem_flags::mem_device);
    amax = team_max(amax, kSimdMode, red, sg_id, lane, n_sg);
    nonfinite = team_sum(bad, kSimdMode, red, sg_id, lane, n_sg) > 0.0f;

    // -------------------------------------------------------------------------
    // Scale so the largest entry is in [0.5, 1)
    // -------------------------------------------------------------------------
    // The stopping test squares entries, which overflows float32 above ~1e19
    // and -- worse, because it is silent -- underflows below ~1e-19, where
    // ||A||_F^2 rounds to zero and any matrix looks converged. Scaling by a
    // power of two is exact and makes the solver invariant to the input's
    // magnitude, which is what LAPACK's ssyev does too. Eigenvalues are scaled
    // back on output; eigenvectors are unaffected.
    if (amax > 0.0f) frexp(amax, expo);

    // ||A||_F is invariant under the orthogonal similarities that follow, so it
    // is computed once and the stopping test is relative to it.
    if (!nonfinite) {
        float acc = 0.0f;
        for (uint idx = t; idx < nn; idx += T) {
            const float x = ldexp(w[idx], -expo);
            w[idx] = x;
            acc += x * x;
        }
        team_barrier(kSimdMode, mem_flags::mem_device);
        fro2 = team_sum(acc, kSimdMode, red, sg_id, lane, n_sg);
    }
    }
    const float thr2 = prm.tol * prm.tol * fro2;

    // -------------------------------------------------------------------------
    // Sweeps (split: at most round_budget rounds this dispatch)
    // -------------------------------------------------------------------------
    uint sweeps = s0.sweeps, round = s0.round, budget = split ? prm.round_budget : 0xFFFFFFFFu;
    const uint last = 2 * np - 1;   // rounds a sweep
    bool converged = false, paused = false;

    while (!nonfinite) {
        if (round == 0) {
        // Off-diagonal norm. Costs one pass over A per sweep, against the
        // 2(N-1) passes the sweep itself makes.
        float off_acc = 0.0f;
        for (uint idx = t; idx < nn; idx += T) {
            const uint i = idx / n;
            const uint j = idx - i * n;
            if (i != j) {
                const float x = w[idx];
                off_acc += x * x;
            }
        }
        const float off2 = team_sum(off_acc, kSimdMode, red, sg_id, lane, n_sg);

        if (non_finite(off2) || non_finite(fro2)) { nonfinite = true; break; }
        if (off2 <= thr2)                          { converged = true; break; }
        if (sweeps >= prm.max_sweeps)              { break; }
        }
        if (budget == 0) { paused = true; break; }
        const uint r1 = last - round <= budget ? last : round + budget;
        jacobi_rounds(round, r1, n, np, t, T, kSimdMode, w, v, kComputeVectors, sc);
        if (split) budget -= r1 - round;
        round = r1;
        if (round == last) {
            ++sweeps;
            round = 0;
        }
    }
    if (paused) {   // the next dispatch resumes here
        if (t == 0) st[b] = JacobiState{expo, fro2, 0.0f, sweeps, round, kJacobiStarted};
        return;
    }

    // -------------------------------------------------------------------------
    // Output: eigenvalues ascending, eigenvector columns permuted to match
    // -------------------------------------------------------------------------
    // The scratch is dead, so its floats hold lambda (4*np >= n) and its
    // ushorts the ranks (2*np >= n).
    threadgroup float*  lam  = fbase;
    threadgroup ushort* rank = reinterpret_cast<threadgroup ushort*>(fbase + 4 * np);
    jacobi_rank_sort(n, n, t, T, kSimdMode, w, lam, rank);

    device float* out_vals = vals + (ulong)b * n;
    if (nonfinite) {
        // Comparisons against NaN cannot order anything, so say so outright
        // rather than emitting a permutation with holes in it.
        const float qnan = as_type<float>(0x7FC00000u);
        for (uint i = t; i < n; i += T) out_vals[i] = qnan;
        if (kComputeVectors) for (uint idx = t; idx < nn; idx += T) w[idx] = qnan;
    } else {
        for (uint i = t; i < n; i += T) out_vals[rank[i]] = ldexp(lam[i], expo);
        if (kComputeVectors) {
            // A is dead: W becomes the eigenvector output.
            for (uint idx = t; idx < nn; idx += T) {
                const uint k = idx / n;
                const uint i = idx - k * n;
                w[k * n + rank[i]] = v[idx];
            }
        }
    }

    if (t == 0) {
        info[b] = sweeps | (converged ? kInfoConverged : 0u) | (nonfinite ? kInfoNonFinite : 0u);
        if (split) st[b] = JacobiState{expo, fro2, 0.0f, sweeps, 0u, kJacobiStarted | kJacobiDone};
    }
}
