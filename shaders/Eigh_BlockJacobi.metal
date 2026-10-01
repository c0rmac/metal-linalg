#include "eigh_jacobi_common.h"
#include "block_jacobi_common.h"

// =============================================================================
// Block Jacobi symmetric eigensolver: grid-parallel, one matrix over many cores
// =============================================================================
//
// The whole-matrix kernel (Eigh_Jacobi.metal) gives each matrix one
// threadgroup, i.e. one GPU core, and streams the matrix through device memory
// twice per round at two flops per element. That is the right shape for a
// batch of small matrices and the wrong one for a large matrix, where it is
// both one-core-bound and memory-bound. This backend is the `qr_streaming_amx`
// analogue: the same Jacobi method, but on b x b blocks, so that
//
//   * the rotations of a block pair are computed on a 2b x 2b subproblem that
//     fits in threadgroup memory, by the same phases the whole-matrix kernel
//     runs (eigh_jacobi_common.h), and
//   * applying them to the rest of the matrix is a 2b x 2b times 2b x N
//     matrix product on simdgroup_matrix tiles, spread over the grid, with
//     2b flops per element read instead of two.
//
// One block round rotates all nb/2 disjoint block pairs (P, Q) of the same
// round-robin tournament the scalar kernel uses, in three launches:
//
//   bj_subproblem   per pair: S = [A_PP A_PQ; A_QP A_QQ] -> threadgroup memory,
//                   `inner_sweeps` scalar Jacobi sweeps on S accumulating the
//                   2b x 2b orthogonal U (S <- U^T S U), S and U written back
//   bj_update_rows  per (pair, 32-column group): rows of blocks P, Q <- U^T rows
//   bj_update_cols  per (pair, 32-row group):    columns P, Q <- columns U,
//                   and the same for V (V <- V U)
//
// Rows and columns are in separate launches for the same reason the scalar
// kernel has two barriers: pair 1's row update and pair 2's column update
// both touch the element at (row of P1, column of P2). Within a launch every
// threadgroup touches a disjoint region, so all updates are in place.
//
// The subproblem is not solved to convergence. One inner sweep on every block
// pair is a scalar cyclic sweep in a block-cyclic ordering, and cyclic Jacobi
// converges for such orderings; with the block updates applied by GEMM the
// flop count is about that of scalar Jacobi. Solving each subproblem fully
// would cost several times more subproblem work (the latency-bound part) for
// fewer outer sweeps. `inner_sweeps` is a tuning knob (see eigh.mm).
//
// Convergence is tested on the host once per outer sweep from bj_norm, on the
// same off(A) <= tol ||A||_F criterion as the scalar kernel. Subproblems whose
// own off-norm is already below the per-pair share of that budget are skipped
// (U = I, flagged in `active`), so late sweeps only pay for the ones that
// still matter.
//
// Block size b = 16: the subproblem order 2b = 32 is the simdgroup width, and
// the threadgroup footprint (S, U and scratch, ~8.5 KB) lets three subproblems
// share a core. N is padded to a multiple of 32 with zeros; the padded rows
// and columns never couple to the real ones (a zero off-diagonal is never
// rotated), so they are simply dropped on output.
//
// Compile with -fno-fast-math, as for the scalar kernel.

constant bool kComputeVectors [[function_constant(0)]];

// Must match `BlockParams` in eigh_block_jacobi.mm.
struct BlockParams {
    uint n;             // original order
    uint n_pad;         // padded order, multiple of BJ_GROUP
    uint nb;            // n_pad / BJ_B (even)
    uint n_pairs;       // nb / 2
    uint round;         // current tournament round
    uint inner_sweeps;  // scalar sweeps per subproblem
    uint lower;         // 1: symmetrise from the lower triangle
};

// -----------------------------------------------------------------------------
// bj_pack: W = scale * sym(A) zero-padded to n_pad, V = I
// -----------------------------------------------------------------------------
kernel void bj_pack(
    device const float*   A     [[buffer(0)]],  // [batch, n, n]
    device float*         W     [[buffer(1)]],  // [batch, n_pad, n_pad]
    device float*         V     [[buffer(2)]],  // [batch, n_pad, n_pad]
    device const float*   scale [[buffer(3)]],  // [batch] power-of-two scale
    constant BlockParams& prm   [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]])
{
    const uint j = gid.x, i = gid.y, b = gid.z;
    const uint n = prm.n, np = prm.n_pad;
    if (i >= np || j >= np) return;

    // A zero scale marks a non-finite matrix: it becomes the zero matrix,
    // converges in no sweeps, and the host writes NaN in its place. The test
    // is explicit because NaN * 0 is NaN.
    const float s = scale[b];
    float x = 0.0f;
    if (i < n && j < n && s != 0.0f) {
        device const float* a = A + (ulong)b * n * n;
        const bool direct = prm.lower ? (i >= j) : (i <= j);
        x = (direct ? a[i * n + j] : a[j * n + i]) * s;
    }
    const ulong o = (ulong)b * np * np + (ulong)i * np + j;
    W[o] = x;
    if (kComputeVectors) V[o] = (i == j) ? 1.0f : 0.0f;
}

// -----------------------------------------------------------------------------
// bj_norm: per matrix, sum of squares of W (all entries, or off-diagonal only)
// -----------------------------------------------------------------------------
kernel void bj_norm(
    device const float*   W            [[buffer(0)]],
    device float*         out          [[buffer(1)]],  // [batch]
    constant BlockParams& prm          [[buffer(2)]],
    constant uint&        include_diag [[buffer(3)]],
    uint b     [[threadgroup_position_in_grid]],
    uint t     [[thread_index_in_threadgroup]],
    uint T     [[threads_per_threadgroup]],
    uint sg_id [[simdgroup_index_in_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]],
    uint n_sg  [[simdgroups_per_threadgroup]])
{
    threadgroup float red[kRedFloats];
    const uint  np = prm.n_pad;
    const ulong nn = (ulong)np * np;
    device const float* w = W + (ulong)b * nn;

    float acc = 0.0f;
    for (ulong idx = t; idx < nn; idx += T) {
        const uint i = (uint)(idx / np);
        const uint j = (uint)(idx - (ulong)i * np);
        if (include_diag || i != j) {
            const float x = w[idx];
            acc += x * x;
        }
    }
    const float total = team_sum(acc, false, red, sg_id, lane, n_sg);
    if (t == 0) out[b] = total;
}

// -----------------------------------------------------------------------------
// bj_subproblem: one threadgroup per block pair
// -----------------------------------------------------------------------------
kernel void bj_subproblem(
    device float*         W      [[buffer(0)]],
    device float*         U      [[buffer(1)]],  // [batch, n_pairs, 32, 32]
    device uint*          active [[buffer(2)]],  // [batch, n_pairs] 1 if U != I
    device const float*   thr    [[buffer(3)]],  // [batch] per-pair off^2 below which a pair is skipped
    constant BlockParams& prm    [[buffer(4)]],
    uint2 tgpos [[threadgroup_position_in_grid]],   // (pair, matrix)
    uint  t     [[thread_index_in_threadgroup]],
    uint2 tgs   [[threads_per_threadgroup]],
    uint  sg_id [[simdgroup_index_in_threadgroup]],
    uint  lane  [[thread_index_in_simdgroup]],
    uint  n_sg  [[simdgroups_per_threadgroup]])
{
    const uint T = tgs.x;
    threadgroup float S[BJ_SUB * BJ_SUB];
    threadgroup float Uloc[BJ_SUB * BJ_SUB];
    threadgroup float scratch[5 * BJ_SUBP];
    threadgroup float red[kRedFloats];

    const uint j = tgpos.x, b = tgpos.y;
    uint P, Q;
    tournament_pair(prm.round, j, prm.nb, P, Q);
    if (Q >= prm.nb) return;

    const uint np = prm.n_pad;
    device float* w   = W + (ulong)b * np * np;
    device uint*  act = active + (ulong)b * prm.n_pairs + j;

    // Gather S and start U at the identity.
    float acc = 0.0f;
    for (uint idx = t; idx < BJ_SUB * BJ_SUB; idx += T) {
        const uint r = idx / BJ_SUB;
        const uint c = idx - r * BJ_SUB;
        const float x = w[(ulong)bj_row(r, P, Q) * np + bj_row(c, P, Q)];
        S[idx]    = x;
        Uloc[idx] = (r == c) ? 1.0f : 0.0f;
        if (r != c) acc += x * x;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float off2 = team_sum(acc, false, red, sg_id, lane, n_sg);

    if (off2 <= thr[b]) {   // uniform: nothing worth rotating here
        if (t == 0) *act = 0;
        return;
    }

    const JacobiScratch sc = jacobi_scratch(scratch, BJ_SUBP);
    threadgroup float* s_ptr = S;
    threadgroup float* u_ptr = Uloc;
    for (uint k = 0; k < prm.inner_sweeps; ++k) {
        jacobi_sweep(BJ_SUB, BJ_SUBP, t, T, false, s_ptr, u_ptr, true, sc);
    }

    device float* u = U + ((ulong)b * prm.n_pairs + j) * (BJ_SUB * BJ_SUB);
    for (uint idx = t; idx < BJ_SUB * BJ_SUB; idx += T) {
        const uint r = idx / BJ_SUB;
        const uint c = idx - r * BJ_SUB;
        w[(ulong)bj_row(r, P, Q) * np + bj_row(c, P, Q)] = S[idx];
        u[idx] = Uloc[idx];
    }
    if (t == 0) *act = 1;
}

// -----------------------------------------------------------------------------
// bj_update_rows: rows of blocks P and Q, columns of one 32-wide group
//                 [W_P; W_Q] <- U^T [W_P; W_Q]
// -----------------------------------------------------------------------------
// 128 threads = 4 simdgroups; simdgroup s produces output rows 8s..8s+7, which
// is an 8-row strip inside block P (s < 2) or Q (s >= 2), for all 32 columns.
kernel void bj_update_rows(
    device float*         W      [[buffer(0)]],
    device const float*   U      [[buffer(1)]],
    device const uint*    active [[buffer(2)]],
    constant BlockParams& prm    [[buffer(3)]],
    uint3 tgpos [[threadgroup_position_in_grid]],   // (pair, column group, matrix)
    uint  t     [[thread_index_in_threadgroup]],
    uint3 tgs   [[threads_per_threadgroup]],
    uint  sg_id [[simdgroup_index_in_threadgroup]])
{
    const uint T = tgs.x;
    threadgroup float X[BJ_SUB * BJ_GROUP];   // [P rows; Q rows] x 32 columns

    const uint j = tgpos.x, g = tgpos.y, b = tgpos.z;
    if (!active[(ulong)b * prm.n_pairs + j]) return;
    uint P, Q;
    tournament_pair(prm.round, j, prm.nb, P, Q);
    if (Q >= prm.nb) return;

    // The group's two block columns; the ones belonging to this pair are part
    // of S, already updated by bj_subproblem, and must be left alone.
    const uint cb0 = 2 * g, cb1 = 2 * g + 1;
    const bool do0 = (cb0 != P && cb0 != Q);
    const bool do1 = (cb1 != P && cb1 != Q);
    if (!do0 && !do1) return;

    const uint np = prm.n_pad;
    device float*       w    = W + (ulong)b * np * np;
    device const float* u    = U + ((ulong)b * prm.n_pairs + j) * (BJ_SUB * BJ_SUB);
    const uint          col0 = g * BJ_GROUP;

    for (uint idx = t; idx < BJ_SUB * BJ_GROUP; idx += T) {
        const uint r = idx / BJ_GROUP;
        const uint c = idx - r * BJ_GROUP;
        X[idx] = w[(ulong)bj_row(r, P, Q) * np + col0 + c];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_float8x8 acc[4];
    for (uint ct = 0; ct < 4; ++ct) acc[ct] = simdgroup_float8x8(0.0f);
    for (uint k = 0; k < 4; ++k) {
        // (U^T)[s, k] is the transpose of the U tile at rows 8k, columns 8s.
        simdgroup_float8x8 ut;
        simdgroup_load(ut, u + (k * 8) * BJ_SUB + sg_id * 8, BJ_SUB, ulong2(0, 0), true);
        for (uint ct = 0; ct < 4; ++ct) {
            simdgroup_float8x8 xt;
            simdgroup_load(xt, X + (k * 8) * BJ_GROUP + ct * 8, BJ_GROUP);
            simdgroup_multiply_accumulate(acc[ct], ut, xt, acc[ct]);
        }
    }

    device float* out = w + (ulong)bj_row(sg_id * 8, P, Q) * np + col0;
    for (uint ct = 0; ct < 4; ++ct) {
        if ((ct < 2) ? do0 : do1) simdgroup_store(acc[ct], out + ct * 8, np);
    }
}

// -----------------------------------------------------------------------------
// bj_update_cols: columns of blocks P and Q, rows of one 32-high group
//                 [M_P  M_Q] <- [M_P  M_Q] U     for M = W (rows outside P, Q)
//                                                and M = V (all rows)
// -----------------------------------------------------------------------------
// Simdgroup s produces output columns 8s..8s+7, an 8-column strip inside
// block P (s < 2) or Q (s >= 2), for all 32 rows of the group.
kernel void bj_update_cols(
    device float*         W      [[buffer(0)]],
    device float*         V      [[buffer(1)]],
    device const float*   U      [[buffer(2)]],
    device const uint*    active [[buffer(3)]],
    constant BlockParams& prm    [[buffer(4)]],
    uint3 tgpos [[threadgroup_position_in_grid]],   // (pair, row group, matrix)
    uint  t     [[thread_index_in_threadgroup]],
    uint3 tgs   [[threads_per_threadgroup]],
    uint  sg_id [[simdgroup_index_in_threadgroup]])
{
    const uint T = tgs.x;
    threadgroup float X[BJ_GROUP * BJ_SUB];   // 32 rows x [P columns  Q columns]

    const uint j = tgpos.x, g = tgpos.y, b = tgpos.z;
    if (!active[(ulong)b * prm.n_pairs + j]) return;
    uint P, Q;
    tournament_pair(prm.round, j, prm.nb, P, Q);
    if (Q >= prm.nb) return;

    const uint np = prm.n_pad;
    device const float* u    = U + ((ulong)b * prm.n_pairs + j) * (BJ_SUB * BJ_SUB);
    const uint          row0 = g * BJ_GROUP;

    // Block rows of this pair are part of S: rotate W's other rows only.
    const uint rb0 = 2 * g, rb1 = 2 * g + 1;
    const bool w0 = (rb0 != P && rb0 != Q);
    const bool w1 = (rb1 != P && rb1 != Q);
    if (w0 || w1) {
        bj_cols_apply(W + (ulong)b * np * np, u, np, row0, P, Q, w0, w1, X, sg_id, t, T);
    }
    if (kComputeVectors) {
        bj_cols_apply(V + (ulong)b * np * np, u, np, row0, P, Q, true, true, X, sg_id, t, T);
    }
}

// -----------------------------------------------------------------------------
// bj_unpack: eigenvalues ascending (scaled back), eigenvector columns permuted,
//            padding dropped
// -----------------------------------------------------------------------------
kernel void bj_unpack(
    device const float*   W    [[buffer(0)]],
    device const float*   V    [[buffer(1)]],
    device float*         vals [[buffer(2)]],  // [batch, n]
    device float*         vecs [[buffer(3)]],  // [batch, n, n]
    device const int*     expo [[buffer(4)]],  // [batch]
    constant BlockParams& prm  [[buffer(5)]],
    threadgroup float*    tg   [[threadgroup(0)]],   // n floats + n ushorts
    uint b [[threadgroup_position_in_grid]],
    uint t [[thread_index_in_threadgroup]],
    uint T [[threads_per_threadgroup]])
{
    const uint n = prm.n, np = prm.n_pad;
    threadgroup float*  lam  = tg;
    threadgroup ushort* rank = reinterpret_cast<threadgroup ushort*>(tg + n);

    device const float* w = W + (ulong)b * np * np;
    jacobi_rank_sort(n, np, t, T, false, w, lam, rank);

    device float* out_vals = vals + (ulong)b * n;
    const int e = expo[b];
    for (uint i = t; i < n; i += T) out_vals[rank[i]] = ldexp(lam[i], e);

    if (kComputeVectors) {
        device const float* v   = V + (ulong)b * np * np;
        device float*       out = vecs + (ulong)b * n * n;
        for (uint idx = t; idx < n * n; idx += T) {
            const uint k = idx / n;
            const uint i = idx - k * n;
            out[k * n + rank[i]] = v[(ulong)k * np + i];
        }
    }
}
