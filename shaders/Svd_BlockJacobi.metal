#include "eigh_jacobi_common.h"
#include "block_jacobi_common.h"

// =============================================================================
// Block one-sided Jacobi SVD: grid-parallel, one matrix over many cores
// =============================================================================
//
// Svd_Jacobi.metal gives each matrix one threadgroup, which is one GPU core.
// That is the right shape for a batch of small matrices and the wrong one for
// a large matrix: a lone 512 x 512 takes half a second. This is the large-
// matrix shader, the counterpart of QR_Streaming_AMX_Reduced and
// Eigh_BlockJacobi: the same one-sided Jacobi method on blocks of b = 16
// columns, so that a matrix is spread over the grid and the rotations are
// applied as simdgroup_matrix tile products.
//
// One block round handles the nb/2 disjoint pairs (P, Q) of column blocks of
// the usual round-robin tournament, in three launches:
//
//   sbj_gram_subproblem   per pair: the 32 x 32 Gram matrix S = X^T X of the
//                         pair's columns X = [G_P  G_Q], accumulated with tile
//                         products over all rows into threadgroup memory; then
//                         one Jacobi sweep on S, by the phases of
//                         eigh_jacobi_common.h, accumulating the orthogonal U
//   sbj_update_cols (G)   per (pair, 32-row group): X <- X U
//   sbj_update_cols (V)   the same for V
//
// One-sided Jacobi has no row update, so a round is cheaper than the block
// eigensolver's, and every threadgroup of a launch touches a region no other
// does.
//
// S is only used to find the rotation. The decomposition itself never depends
// on its accuracy: whatever U comes out of the sweep is a product of plane
// rotations, so G U and V U are exact orthogonal transformations to rounding,
// and A V = G holds throughout. What the accuracy of S decides is how far one
// round gets, and its entries are inner products of columns computed to the
// same relative accuracy as in the scalar kernel.
//
// Convergence. A pair is settled when every two of its 32 columns that are
// both not null have |g_i . g_j| <= tol |g_i| |g_j|, and no null column is
// grossly out of line with the rest; it then produces no U and its updates
// are skipped. The matrix has converged when a whole sweep
// settles every pair, which the host reads from a flag per pair.
//
// Null columns and scaling are as in Svd_Jacobi.metal.
// The matrix is zero-padded to multiples of 32 in both directions; a zero
// column has zero inner products with everything and is never rotated, and a
// zero row contributes nothing to any inner product.
//
// Compile with -fno-fast-math.

constant bool kComputeUV [[function_constant(0)]];

// Cosine above which a null column still counts as unclean; see Svd_Jacobi.metal.
constant float kCleanCos = 0.05f;

// A column below this fraction of the largest is under the rounding of
// everything else in the matrix and never counts towards convergence.
constant float kNegligible = 1.2e-7f;

// Must match `BlockParams` in svd_block_jacobi.mm.
struct SvdBlockParams {
    uint  m;            // rows
    uint  n;            // columns, n <= m
    uint  m_pad;        // rows padded to a multiple of 32
    uint  n_pad;        // columns padded to a multiple of 32
    uint  nb;           // n_pad / BJ_B (even)
    uint  n_pairs;      // nb / 2
    uint  round;        // current tournament round
    uint  inner_sweeps; // Jacobi sweeps per subproblem
    uint  rows_pad;     // sbj_update_cols: padded rows of the matrix being updated
    float tol;          // rotate while |g_i . g_j| > tol |g_i| |g_j|
    float null_tol;     // a column below null_tol * (largest column) is null
};

// -----------------------------------------------------------------------------
// sbj_pack: G = A zero-padded, V = I. `valid` = 0 marks a non-finite matrix,
// which becomes the zero matrix and is overwritten with NaN by the host.
// -----------------------------------------------------------------------------
kernel void sbj_pack(
    device const float*      A     [[buffer(0)]],  // [batch, m, n], already scaled
    device float*            G     [[buffer(1)]],  // [batch, m_pad, n_pad]
    device float*            V     [[buffer(2)]],  // [batch, n_pad, n_pad]
    device const uint*       valid [[buffer(3)]],  // [batch]
    constant SvdBlockParams& prm   [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]])
{
    const uint j = gid.x, i = gid.y, b = gid.z;
    const uint np = prm.n_pad;
    const uint rows = max(prm.m_pad, np);
    if (j >= np || i >= rows) return;

    if (i < prm.m_pad) {
        float x = 0.0f;
        if (i < prm.m && j < prm.n && valid[b]) {
            x = A[(ulong)b * prm.m * prm.n + (ulong)i * prm.n + j];
        }
        G[(ulong)b * prm.m_pad * np + (ulong)i * np + j] = x;
    }
    if (kComputeUV && i < np) {
        V[(ulong)b * np * np + (ulong)i * np + j] = (i == j) ? 1.0f : 0.0f;
    }
}

// -----------------------------------------------------------------------------
// sbj_colmax: per matrix, the largest squared column norm
// -----------------------------------------------------------------------------
kernel void sbj_colmax(
    device const float*      G    [[buffer(0)]],
    device float*            out  [[buffer(1)]],  // [batch]
    constant SvdBlockParams& prm  [[buffer(2)]],
    uint b     [[threadgroup_position_in_grid]],
    uint t     [[thread_index_in_threadgroup]],
    uint T     [[threads_per_threadgroup]],
    uint sg_id [[simdgroup_index_in_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]],
    uint n_sg  [[simdgroups_per_threadgroup]])
{
    threadgroup float red[kRedFloats];
    const uint np = prm.n_pad;
    device const float* g = G + (ulong)b * prm.m_pad * np;

    float cmax = 0.0f;
    for (uint c = t; c < prm.n; c += T) {
        float acc = 0.0f;
        for (uint i = 0; i < prm.m; ++i) {
            const float x = g[(ulong)i * np + c];
            acc += x * x;
        }
        cmax = fmax(cmax, acc);
    }
    const float total = team_max(cmax, false, red, sg_id, lane, n_sg);
    if (t == 0) out[b] = total;
}

// -----------------------------------------------------------------------------
// sbj_gram_subproblem: one threadgroup per block pair
// -----------------------------------------------------------------------------
// 128 threads = 4 simdgroups; simdgroup s accumulates row strip s of S, the
// four 8 x 8 tiles S[s, 0..3], over every 8-row tile of the pair's columns.
kernel void sbj_gram_subproblem(
    device const float*      G      [[buffer(0)]],
    device float*            U      [[buffer(1)]],  // [batch, n_pairs, 32, 32]
    device uint*             active [[buffer(2)]],  // [batch, n_pairs] this round: 1 if U != I
    device uint*             any    [[buffer(3)]],  // [batch, n_pairs] sticky over the sweep
    device const float*      cmax   [[buffer(4)]],  // [batch] largest squared column norm
    constant SvdBlockParams& prm    [[buffer(5)]],
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
    device const float* g   = G + (ulong)b * prm.m_pad * np;
    device uint*        act = active + (ulong)b * prm.n_pairs + j;

    // --- Gram matrix of the pair's 32 columns ---
    simdgroup_float8x8 acc[4];
    for (uint ct = 0; ct < 4; ++ct) acc[ct] = simdgroup_float8x8(0.0f);
    const uint col_s = bj_row(sg_id * 8, P, Q);
    for (uint r = 0; r < prm.m_pad; r += 8) {
        device const float* row = g + (ulong)r * np;
        simdgroup_float8x8 xs;
        simdgroup_load(xs, row + col_s, np, ulong2(0, 0), true);   // X[r, s]^T
        for (uint ct = 0; ct < 4; ++ct) {
            simdgroup_float8x8 xc;
            simdgroup_load(xc, row + bj_row(ct * 8, P, Q), np);
            simdgroup_multiply_accumulate(acc[ct], xs, xc, acc[ct]);
        }
    }
    threadgroup float* s_ptr = S;
    threadgroup float* u_ptr = Uloc;
    for (uint ct = 0; ct < 4; ++ct) {
        simdgroup_store(acc[ct], s_ptr + (sg_id * 8) * BJ_SUB + ct * 8, BJ_SUB);
    }
    for (uint idx = t; idx < BJ_SUB * BJ_SUB; idx += T) {
        const uint r = idx / BJ_SUB;
        Uloc[idx] = (idx - r * BJ_SUB == r) ? 1.0f : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // --- is there anything left to rotate? ---
    const float null2 = prm.null_tol * prm.null_tol * cmax[b];
    const float negl2 = kNegligible * kNegligible * cmax[b];
    const float tol2  = prm.tol * prm.tol;
    float worst = 0.0f;   // 1 if some pair of columns is out of tolerance
    for (uint idx = t; idx < BJ_SUB * BJ_SUB; idx += T) {
        const uint r = idx / BJ_SUB;
        const uint c = idx - r * BJ_SUB;
        if (r < c) {
            // Two null columns never count. One null column counts only
            // while its angle is gross, and not at all once it is negligible;
            // see Svd_Jacobi.metal. The last matters twice over here: the
            // Jacobi phases do not rotate a pair whose off-diagonal entry is
            // below 1e-12 of the diagonal, so counting a column that small
            // would ask for a rotation that is never made.
            const float a = S[r * BJ_SUB + r], d = S[c * BJ_SUB + c], x = S[idx];
            const bool na = a <= null2, nd = d <= null2;
            if (na && nd) continue;
            if (na || nd) {
                if (min(a, d) > negl2 && x * x > kCleanCos * kCleanCos * a * d) worst = 1.0f;
            } else if (x * x > tol2 * a * d) {
                worst = 1.0f;
            }
        }
    }
    const float pending = team_max(worst, false, red, sg_id, lane, n_sg);
    if (pending == 0.0f) {   // uniform
        if (t == 0) *act = 0;
        return;
    }

    const JacobiScratch sc = jacobi_scratch(scratch, BJ_SUBP);
    for (uint k = 0; k < prm.inner_sweeps; ++k) {
        jacobi_sweep(BJ_SUB, BJ_SUBP, t, T, false, s_ptr, u_ptr, true, sc, null2);
    }

    device float* u = U + ((ulong)b * prm.n_pairs + j) * (BJ_SUB * BJ_SUB);
    for (uint idx = t; idx < BJ_SUB * BJ_SUB; idx += T) u[idx] = Uloc[idx];
    if (t == 0) {
        *act = 1;
        any[(ulong)b * prm.n_pairs + j] = 1;   // only ever set during a sweep; the host clears it
    }
}

// -----------------------------------------------------------------------------
// sbj_update_cols: columns of blocks P and Q, rows of one 32-high group
// -----------------------------------------------------------------------------
// Launched once for G and once for V; prm.rows_pad is the padded row count of
// whichever is bound.
kernel void sbj_update_cols(
    device float*            Mtx    [[buffer(0)]],
    device const float*      U      [[buffer(1)]],
    device const uint*       active [[buffer(2)]],
    constant SvdBlockParams& prm    [[buffer(3)]],
    uint3 tgpos [[threadgroup_position_in_grid]],   // (pair, row group, matrix)
    uint  t     [[thread_index_in_threadgroup]],
    uint3 tgs   [[threads_per_threadgroup]],
    uint  sg_id [[simdgroup_index_in_threadgroup]])
{
    const uint T = tgs.x;
    threadgroup float X[BJ_GROUP * BJ_SUB];

    const uint j = tgpos.x, grp = tgpos.y, b = tgpos.z;
    if (!active[(ulong)b * prm.n_pairs + j]) return;
    uint P, Q;
    tournament_pair(prm.round, j, prm.nb, P, Q);
    if (Q >= prm.nb) return;

    const uint np = prm.n_pad;
    device const float* u = U + ((ulong)b * prm.n_pairs + j) * (BJ_SUB * BJ_SUB);
    bj_cols_apply(Mtx + (ulong)b * prm.rows_pad * np, u, np, grp * BJ_GROUP, P, Q,
                  true, true, X, sg_id, t, T);
}

// -----------------------------------------------------------------------------
// sbj_unpack: sigma = column norms, descending; U = columns / sigma; V^T;
//             padding dropped
// -----------------------------------------------------------------------------
kernel void sbj_unpack(
    device const float*      G     [[buffer(0)]],
    device const float*      V     [[buffer(1)]],
    device float*            S     [[buffer(2)]],  // [batch, n]
    device float*            Uo    [[buffer(3)]],  // [batch, m, n] row-major
    device float*            Vt    [[buffer(4)]],  // [batch, n, n] row-major
    device uint*             flags [[buffer(5)]],  // [batch] 1 if some column is null
    constant SvdBlockParams& prm   [[buffer(6)]],
    threadgroup float*       tg    [[threadgroup(0)]],   // red[32] sig[n] rank[n]
    uint b     [[threadgroup_position_in_grid]],
    uint t     [[thread_index_in_threadgroup]],
    uint T     [[threads_per_threadgroup]],
    uint sg_id [[simdgroup_index_in_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]],
    uint n_sg  [[simdgroups_per_threadgroup]])
{
    const uint m = prm.m, n = prm.n, np = prm.n_pad;
    threadgroup float*  red  = tg;
    threadgroup float*  sig  = tg + kRedFloats;
    threadgroup ushort* rank = reinterpret_cast<threadgroup ushort*>(sig + n);

    device const float* g = G + (ulong)b * prm.m_pad * np;
    device const float* v = V + (ulong)b * np * np;

    for (uint c = t; c < n; c += T) {
        float acc = 0.0f;
        for (uint i = 0; i < m; ++i) {
            const float x = g[(ulong)i * np + c];
            acc += x * x;
        }
        sig[c] = sqrt(acc);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float pmax = 0.0f;
    for (uint c = t; c < n; c += T) {
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
    const float smax = team_max(pmax, false, red, sg_id, lane, n_sg);
    const float thr  = prm.null_tol * smax;

    float low = 0.0f;
    for (uint c = t; c < n; c += T) if (!(sig[c] > thr)) low = 1.0f;
    const bool deficient = team_sum(low, false, red, sg_id, lane, n_sg) > 0.0f;

    device float* out_s = S + (ulong)b * n;
    for (uint c = t; c < n; c += T) out_s[rank[c]] = sig[c];

    if (kComputeUV) {
        device float* out_u  = Uo + (ulong)b * m * n;
        device float* out_vt = Vt + (ulong)b * n * n;
        for (uint idx = t; idx < m * n; idx += T) {
            const uint i = idx / n;
            const uint c = idx - i * n;
            const float sc = sig[c];
            out_u[(ulong)i * n + rank[c]] = (sc > thr) ? g[(ulong)i * np + c] / sc : 0.0f;
        }
        for (uint idx = t; idx < n * n; idx += T) {
            const uint i = idx / n;
            const uint c = idx - i * n;
            out_vt[(ulong)rank[c] * n + i] = v[(ulong)i * np + c];
        }
    }
    if (t == 0) flags[b] = deficient ? 1u : 0u;
}
