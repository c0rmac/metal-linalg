#include <metal_stdlib>
using namespace metal;

// =============================================================================
// Batched QR of small and mid-size matrices: Householder QR, a matrix to a
// simdgroup or a threadgroup
// =============================================================================
//
// Computes the thin QR A = Q R of a batch of real m x n matrices, by LAPACK's
// methods, built as Svd_GolubKahan.metal and Eigh_QL.metal are: the QR
// counterparts of those kernels, and the replacement for QR_Unblocked.metal's,
// which walked device memory with a barrier per phase of every column.
//
//   qr_householder_simd  up to 32 columns and 128 rows, a simdgroup a matrix,
//                        the matrix in registers: sgeqr2 and sorg2r
//   qr_householder_wy    up to 4096 rows, a threadgroup a matrix: sgeqrf and
//                        sorgqr, panels of 16 columns in registers, the
//                        updates by blocks of 32 columns as 8 x 8 simdgroup
//                        matrix products
//
// Both read the row-major input as it is, scan it for the largest entry and
// for non-finite ones, scale by a power of two so the largest entry lies in
// [0.5, 1) (NaN out for a non-finite matrix), and write R (min(m, n) x n) and
// Q (m x min(m, n)) row-major.
//
// References:
//   Golub & Van Loan, Matrix Computations 4th ed., s5.1-5.2 (Householder
//     reflections and QR), s5.1.6 (accumulating their product backward) and
//     s5.2.3 (the block representation, compact WY).
//   Schreiber & Van Loan, A storage-efficient WY representation for products
//     of Householder transformations, SIAM J. Sci. Stat. Comput. 10 (1989).
//   LAPACK sgeqr2, slarfg, sorg2r, sgeqrf, slarft, slarfb, sorgqr.
//
// Built with fast math, as Svd_GolubKahan.metal and Eigh_QL.metal are: the
// columns are a latency-bound chain. What needs the accuracy -- the
// reflector's norm and scalars -- takes one Newton step after the fast
// division or square root, and the non-finite input check reads the bits.

// The sum over the threadgroup of x, every thread contributing (0 if it has
// nothing). `red` is a set of slots, one per simdgroup; the caller alternates
// sets between consecutive sums, and puts a barrier between two uses of the
// same set.
inline float group_sum(float x, threadgroup float* red, uint sg, uint lane, uint nsg) {
    x = simd_sum(x);
    if (lane == 0) red[sg] = x;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float s = 0.0f;
    for (uint k = 0; k < nsg; ++k) s += red[k];
    return s;
}

inline float group_max(float x, threadgroup float* red, uint sg, uint lane, uint nsg) {
    x = simd_max(x);
    if (lane == 0) red[sg] = x;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float m = 0.0f;
    for (uint k = 0; k < nsg; ++k) m = fmax(m, red[k]);
    return m;
}

// Division and square root to about half an ulp, from the fast forms and one
// Newton step; see Eigh_QL.metal.
inline float div_nr(float a, float b) {
    const float r = 1.0f / b;
    const float q = a * r;
    return fma(fma(-b, q, a), r, q);
}

inline float sqrt_nr(float x) {
    if (x <= 0.0f) return 0.0f;
    const float y = rsqrt(x);
    const float s = x * y;
    return fma(fma(-s, s, x), 0.5f * y, s);
}

inline bool non_finite(float v) {
    return (as_type<uint>(v) & 0x7F800000u) == 0x7F800000u;
}

// =============================================================================
// In registers: a simdgroup a matrix (qr_householder_simd)
// =============================================================================
//
// For n <= 32 and m <= 128, a matrix fits in one simdgroup's registers -- R rows a lane (row s * 32 + lane in x[s]), B
// columns -- as the band reduction's panels keep theirs (qr_simd in
// Svd_Bidiag.metal). Then nothing waits on a barrier and no threadgroup
// memory is used, so a core holds as many matrices as its registers allow,
// several to a threadgroup. A step's dot products, one per column right of
// the current one, are simd_sums four at a time (on a float4).
// (Instances of 64 columns were slower than qr_householder_wy below; a column
// a lane instead, each lane's dot products its own FMAs and the reflector's
// vector shuffled across a row at a time, was 2.2x slower at 32 x 32 and 6x
// at 64 x 64: simd_sum is cheap here, and the column layout pays every row's
// FMAs on every step.)
//
// A row must only ever be indexed by constants to stay in registers, so the
// loops over a row are expanded by the preprocessor (QH_UNROLL), and the
// current column is kept at index 0 by rotating the row: left one place a
// step while factoring, right one place a step while Q is formed (and B - K
// places at the turns, so that both passes end where they began).
//
//   1. load each lane's rows, scan with simd_max, scale by a power of two
//   2. factor: per column the reflector from the tail's simd_sum, then the
//      dot products of the columns right of it; tau kept on lane j % 32
//   3. write R
//   4. form Q in place backward (sorg2r)
//   5. write Q
#define QH_UNROLL_c(...) { { constexpr uint c = 0; __VA_ARGS__ } { constexpr uint c = 1; __VA_ARGS__ } { constexpr uint c = 2; __VA_ARGS__ } { constexpr uint c = 3; __VA_ARGS__ } { constexpr uint c = 4; __VA_ARGS__ } { constexpr uint c = 5; __VA_ARGS__ } { constexpr uint c = 6; __VA_ARGS__ } { constexpr uint c = 7; __VA_ARGS__ } { constexpr uint c = 8; __VA_ARGS__ } { constexpr uint c = 9; __VA_ARGS__ } { constexpr uint c = 10; __VA_ARGS__ } { constexpr uint c = 11; __VA_ARGS__ } { constexpr uint c = 12; __VA_ARGS__ } { constexpr uint c = 13; __VA_ARGS__ } { constexpr uint c = 14; __VA_ARGS__ } { constexpr uint c = 15; __VA_ARGS__ } { constexpr uint c = 16; __VA_ARGS__ } { constexpr uint c = 17; __VA_ARGS__ } { constexpr uint c = 18; __VA_ARGS__ } { constexpr uint c = 19; __VA_ARGS__ } { constexpr uint c = 20; __VA_ARGS__ } { constexpr uint c = 21; __VA_ARGS__ } { constexpr uint c = 22; __VA_ARGS__ } { constexpr uint c = 23; __VA_ARGS__ } { constexpr uint c = 24; __VA_ARGS__ } { constexpr uint c = 25; __VA_ARGS__ } { constexpr uint c = 26; __VA_ARGS__ } { constexpr uint c = 27; __VA_ARGS__ } { constexpr uint c = 28; __VA_ARGS__ } { constexpr uint c = 29; __VA_ARGS__ } { constexpr uint c = 30; __VA_ARGS__ } { constexpr uint c = 31; __VA_ARGS__ } { constexpr uint c = 32; __VA_ARGS__ } { constexpr uint c = 33; __VA_ARGS__ } { constexpr uint c = 34; __VA_ARGS__ } { constexpr uint c = 35; __VA_ARGS__ } { constexpr uint c = 36; __VA_ARGS__ } { constexpr uint c = 37; __VA_ARGS__ } { constexpr uint c = 38; __VA_ARGS__ } { constexpr uint c = 39; __VA_ARGS__ } { constexpr uint c = 40; __VA_ARGS__ } { constexpr uint c = 41; __VA_ARGS__ } { constexpr uint c = 42; __VA_ARGS__ } { constexpr uint c = 43; __VA_ARGS__ } { constexpr uint c = 44; __VA_ARGS__ } { constexpr uint c = 45; __VA_ARGS__ } { constexpr uint c = 46; __VA_ARGS__ } { constexpr uint c = 47; __VA_ARGS__ } { constexpr uint c = 48; __VA_ARGS__ } { constexpr uint c = 49; __VA_ARGS__ } { constexpr uint c = 50; __VA_ARGS__ } { constexpr uint c = 51; __VA_ARGS__ } { constexpr uint c = 52; __VA_ARGS__ } { constexpr uint c = 53; __VA_ARGS__ } { constexpr uint c = 54; __VA_ARGS__ } { constexpr uint c = 55; __VA_ARGS__ } { constexpr uint c = 56; __VA_ARGS__ } { constexpr uint c = 57; __VA_ARGS__ } { constexpr uint c = 58; __VA_ARGS__ } { constexpr uint c = 59; __VA_ARGS__ } { constexpr uint c = 60; __VA_ARGS__ } { constexpr uint c = 61; __VA_ARGS__ } { constexpr uint c = 62; __VA_ARGS__ } { constexpr uint c = 63; __VA_ARGS__ } }
#define QH_UNROLL_q(...) { { constexpr uint q = 0; __VA_ARGS__ } { constexpr uint q = 1; __VA_ARGS__ } { constexpr uint q = 2; __VA_ARGS__ } { constexpr uint q = 3; __VA_ARGS__ } { constexpr uint q = 4; __VA_ARGS__ } { constexpr uint q = 5; __VA_ARGS__ } { constexpr uint q = 6; __VA_ARGS__ } { constexpr uint q = 7; __VA_ARGS__ } { constexpr uint q = 8; __VA_ARGS__ } { constexpr uint q = 9; __VA_ARGS__ } { constexpr uint q = 10; __VA_ARGS__ } { constexpr uint q = 11; __VA_ARGS__ } { constexpr uint q = 12; __VA_ARGS__ } { constexpr uint q = 13; __VA_ARGS__ } { constexpr uint q = 14; __VA_ARGS__ } { constexpr uint q = 15; __VA_ARGS__ } }
#define QH_UNROLL_s(...) { { constexpr uint s = 0; __VA_ARGS__ } { constexpr uint s = 1; __VA_ARGS__ } { constexpr uint s = 2; __VA_ARGS__ } { constexpr uint s = 3; __VA_ARGS__ } }
#define QH_UNROLL_f(...) { { constexpr uint f = 0; __VA_ARGS__ } { constexpr uint f = 1; __VA_ARGS__ } { constexpr uint f = 2; __VA_ARGS__ } { constexpr uint f = 3; __VA_ARGS__ } }
#define QH_UNROLL_e(...) { { constexpr uint e = 0; __VA_ARGS__ } { constexpr uint e = 1; __VA_ARGS__ } { constexpr uint e = 2; __VA_ARGS__ } { constexpr uint e = 3; __VA_ARGS__ } }
#define QH_UNROLL(N, v, ...) QH_UNROLL_##v(if (v < N) __VA_ARGS__)

// Must match `QsParams` in qr_householder.mm.
struct QsParams {
    uint m;        // rows, at most 32 R
    uint n;        // columns, at most B
    uint batch;    // matrices in this dispatch
    uint q_cols;   // Q's columns: K, or 0 for R alone (Q not formed)
};

// slarfg's reflector from alpha and the tail's sum of squares: beta, tau and
// the scale of the tail, to about half an ulp (div_nr, sqrt_nr).
inline void reflector(float alpha, float sumsq, thread float& beta, thread float& tau, thread float& scale) {
    beta = alpha;
    tau = 0.0f;
    scale = 1.0f;
    if (sumsq > 0.0f) {
        beta = -copysign(sqrt_nr(alpha * alpha + sumsq), alpha);
        tau = div_nr(beta - alpha, beta);
        scale = div_nr(1.0f, alpha - beta);
    }
}

#define QH_ROTATE_LEFT  QH_UNROLL(R, s, { const float x0 = x[s][0]; QH_UNROLL(B, c, { if (c + 1 < B) x[s][c] = x[s][c + 1]; }); x[s][B - 1] = x0; })
#define QH_ROTATE_RIGHT QH_UNROLL(R, s, { const float xl = x[s][B - 1]; QH_UNROLL(B, c, { if (c + 1 < B) x[s][B - 1 - c] = x[s][B - 2 - c]; }); x[s][0] = xl; })

// x[s][k] -= coef(k) v[s] for k = 1 .. B-1 with j + k < lim, coef(k) = t times
// the column's dot product with v (plus x(j, k) itself if `head`): the dot
// products four columns to a simd_sum.
#define QH_APPLY(head)                                                                                      \
    QH_UNROLL(B / 4, q, {                                                                                   \
        if (j + 4 * q + 1 < lim + 3) {                                                                      \
            float4 loc = 0.0f;                                                                              \
            QH_UNROLL(4, e, {                                                                               \
                if (4 * q + e > 0 && 4 * q + e < B) {                                                       \
                    QH_UNROLL(R, s, { loc[e] = fma(v[s], x[s][4 * q + e], loc[e]); });                      \
                }                                                                                           \
            });                                                                                             \
            const float4 dd = t * simd_sum(loc);                                                            \
            QH_UNROLL(4, e, {                                                                               \
                if (4 * q + e > 0 && 4 * q + e < B && j + 4 * q + e < lim) {                                \
                    QH_UNROLL(R, s, {                                                                       \
                        if (head) {                                                                         \
                            const uint row = s * 32 + lane;                                                 \
                            if (row == j) x[s][4 * q + e] = -dd[e];                                         \
                            else if (row > j) x[s][4 * q + e] = fma(-dd[e], v[s], x[s][4 * q + e]);         \
                        } else {                                                                            \
                            x[s][4 * q + e] = fma(-dd[e], v[s], x[s][4 * q + e]);                           \
                        }                                                                                   \
                    });                                                                                     \
                }                                                                                           \
            });                                                                                             \
        }                                                                                                   \
    })

template <uint B, uint R>
kernel void qr_householder_simd(
    device const float* A_in [[buffer(0)]],   // [batch, m, n] input, row-major
    device float*       Q    [[buffer(1)]],   // [batch, m, K], unless R alone
    device float*       Rout [[buffer(2)]],   // [batch, K, n]
    constant QsParams&  prm  [[buffer(3)]],
    uint tgi  [[threadgroup_position_in_grid]],
    uint sg   [[simdgroup_index_in_threadgroup]],
    uint nsg  [[simdgroups_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    const uint mat = tgi * nsg + sg;
    if (mat >= prm.batch) return;   // a whole simdgroup: nothing below waits on the others
    const uint m = prm.m, n = prm.n, K = min(m, n), QC = prm.q_cols;
    device const float* src = A_in + (ulong)mat * m * n;
    device float* out_q = Q + (ulong)mat * m * QC;
    device float* out_r = Rout + (ulong)mat * K * n;

    // 1. Load, scan, scale
    float x[R][B];
    float amax = 0.0f, bad = 0.0f;
    QH_UNROLL(R, s, {
        const uint row = s * 32 + lane;
        QH_UNROLL(B, c, {
            float v = 0.0f;
            if (row < m && c < n) {
                v = src[row * n + c];
                amax = fmax(amax, fabs(v));
                if (non_finite(v)) bad = 1.0f;
            }
            x[s][c] = v;
        });
    });
    amax = simd_max(amax);
    if (simd_max(bad) > 0.0f) {
        const float qnan = as_type<float>(0x7FC00000u);
        for (uint idx = lane; idx < m * QC; idx += 32) out_q[idx] = qnan;
        for (uint idx = lane; idx < K * n; idx += 32) out_r[idx] = qnan;
        return;
    }
    int expo = 0;
    if (amax > 0.0f) frexp(amax, expo);
    QH_UNROLL(R, s, { QH_UNROLL(B, c, { x[s][c] = ldexp(x[s][c], -expo); }); });

    // 2. Factor: column j at index 0, column j + k at index k
    float tau_lo = 0.0f, tau_hi = 0.0f;   // tau_j on lane j % 32: j < 32, then j >= 32
    for (uint j = 0; j < K; ++j) {
        float ss = 0.0f, alpha = 0.0f;
        QH_UNROLL(R, s, {
            const uint row = s * 32 + lane;
            const float y = row > j && row < m ? x[s][0] : 0.0f;
            ss = fma(y, y, ss);
            if (s == j / 32) alpha = simd_shuffle(x[s][0], (ushort)(j % 32));
        });
        float beta, t, scale;
        reflector(alpha, simd_sum(ss), beta, t, scale);
        float v[R];
        QH_UNROLL(R, s, {
            const uint row = s * 32 + lane;
            v[s] = 0.0f;
            if (row == j) { x[s][0] = beta; v[s] = 1.0f; }
            else if (row > j && row < m) { x[s][0] *= scale; v[s] = x[s][0]; }
        });
        if (lane == j % 32) { if (j < 32) tau_lo = t; else tau_hi = t; }
        const uint lim = n;
        if (t != 0.0f) { QH_APPLY(false) }   // uniform
        QH_ROTATE_LEFT
    }
    for (uint j = K; j < B; ++j) { QH_ROTATE_LEFT }

    // 3. R
    QH_UNROLL(R, s, {
        const uint row = s * 32 + lane;
        if (row < K) {
            QH_UNROLL(B, c, { if (c < n) out_r[row * n + c] = c >= row ? ldexp(x[s][c], expo) : 0.0f; });
        }
    });

    if (QC == 0) return;   // R alone

    // 4. Q in place, backward (sorg2r): before step j, columns j+1 .. K-1
    // hold H(j+1) ... H(K-1) and are zero in rows up to j
    for (uint u = 0; u + K <= B; ++u) { QH_ROTATE_RIGHT }
    for (int jj = (int)K - 1; jj >= 0; --jj) {
        const uint j = (uint)jj;
        const float t = simd_shuffle(j < 32 ? tau_lo : tau_hi, (ushort)(j % 32));
        float v[R];
        QH_UNROLL(R, s, {
            const uint row = s * 32 + lane;
            v[s] = row > j && row < m ? x[s][0] : 0.0f;
        });
        const uint lim = K;
        if (t != 0.0f) { QH_APPLY(true) }   // uniform
        QH_UNROLL(R, s, {
            const uint row = s * 32 + lane;
            x[s][0] = row < j ? 0.0f : (row == j ? 1.0f - t : -t * v[s]);
        });
        if (j > 0) { QH_ROTATE_RIGHT }
    }

    // 5. Q
    QH_UNROLL(R, s, {
        const uint row = s * 32 + lane;
        if (row < m) {
            QH_UNROLL(B, c, { if (c < K) out_q[row * K + c] = x[s][c]; });
        }
    });
}

#define QH_SIMD(B, R)                                                                                    \
    template [[host_name("qr_householder_simd_" #B "_" #R)]] kernel void qr_householder_simd<B, R>(     \
        device const float*, device float*, device float*, constant QsParams&, uint, uint, uint, uint);
QH_SIMD(8, 1)
QH_SIMD(8, 2)
QH_SIMD(8, 4)
QH_SIMD(16, 1)
QH_SIMD(16, 2)
QH_SIMD(16, 4)
QH_SIMD(32, 1)
QH_SIMD(32, 2)
QH_SIMD(32, 4)

// =============================================================================
// Blocked, for larger matrices: panels in registers, updates by 8x8 products
// (qr_householder_wy)
// =============================================================================
//
// LAPACK's blocked method (sgeqrf, sorgqr) in one threadgroup a matrix, for
// matrices too large for the kernel above, up to 4096 rows: panels of 16
// columns factored as LAPACK factors a matrix, R rows a thread, in blocks of
// 32 columns. Each panel's H = I - V T V^T is applied to the rest of its block,
// each block's (its T merged from its panels', slarft's recurrence by blocks)
// to the columns right of it: W = V^T C, C -= V (T^T W), by simdgroup 8 x 8
// matrix products, two column tiles a simdgroup so that each tile of V it
// loads serves two products. Q is formed from [I; 0] by the blocks backward,
// C -= V (T (V^T C)). The matrix, padded with zero rows and columns to
// multiples of 8 and at least to K rounded up to 16, is in a device
// workspace, as Q is (or the caller's Q, where it has the same shape) and the
// blocks' T.
//
// A panel's step j: the column's sum of squares and alpha (a barrier), the
// reflector, then the dot products of the reflector with every other column
// of the panel (a barrier): those right of it update the panel, those left of
// it are V^T v, the new column of T (lane i of every simdgroup holding T's
// row i).
//
//   1. scan the input; copy it zero-padded into the workspace, scaled by a
//      power of two if its largest entry is beyond 2^20 or 2^-20, unless it
//      needs neither (block 0 then reads the input itself)
//   2. per block: per panel, factor, T into the block's T, the panel's two
//      unit lower diagonal tiles of V into threadgroup memory, T merged, the
//      rest of the block updated; then the columns right of the block, and
//      the block's rows of R written
//   3. Q from [I; 0] by the blocks backward, the tiles of Q not yet written
//      made rather than read (so Q is written once)
//   4. Q copied out of the workspace, if it was not formed in place
//
// On an M5 Pro, 1024 of 128 x 128 in 4.2 ms of GPU time (the blocked QR's
// panels and MPS products took 17), 4096 of 64 x 64 in 3.6 (the register
// kernel 4.4). Wider blocks (64 columns) were no faster, one column tile a
// simdgroup 5-20% slower.

#define QW_B 16   // panel width
#ifndef QW_CT
#define QW_CT 2   // column tiles a simdgroup in the updates
#endif
#ifndef QW_NB
#define QW_NB 32  // block width, a multiple of QW_B, at most 64
#endif

// Must match `QwParams` in qr_householder.mm.
struct QwParams {
    uint m, n;     // the matrix
    uint mp, np;   // padded: multiples of 8, at least Kp
    uint Kp;       // min(m, n) rounded up to QW_B
    uint batch;    // matrices in this dispatch
    uint q_direct; // 1: Q formed in place in the output (mp == m, qn == qc)
    uint sc;       // the scratch's floats in threadgroup memory
    uint qn;       // Q's columns in the workspace: Kp, mp (Q square) or 0 (R alone)
    uint qc;       // ... in the output: K, m or 0
    uint rr;       // R's rows in the output: K, or m (Q square: zeros below K)
};

// -a, in place.
inline void wy_neg(thread simdgroup_float8x8& a) {
    a.thread_elements()[0] = -a.thread_elements()[0];
    a.thread_elements()[1] = -a.thread_elements()[1];
}

// V's 8 x 8 tile in row block r, column tile i of the block at column k (rows
// and columns relative to k): zero above the diagonal tiles (the caller skips
// those), unit lower on them (vd, 64 floats a tile), else the workspace's.
inline void wy_v(thread simdgroup_float8x8& v, uint r, uint i, device const float* V, uint ldv, uint k,
                 threadgroup const float* vd, bool transpose) {
    if (r == i) simdgroup_load(v, vd + 64 * i, 8, ulong2(0, 0), transpose);
    else simdgroup_load(v, V + (ulong)(k + 8 * r) * ldv + k + 8 * i, ldv, ulong2(0, 0), transpose);
}

// C(k:mp, c0:c1) -= V (M (V^T C(k:mp, c0:c1))): V the block of NT column
// tiles at column k, M = T^T (TRANS: H^T applied) or T (H), T upper
// triangular, 8 NT square, at tp (ld 64, device). CT column tiles of 8 a
// simdgroup (each V tile loaded once for all of them); with fewer such units
// than simdgroups, each unit's rows split among G of them, their partial V^T C
// summed through wpart (NT CT 8 x 8 tiles a simdgroup, at most wcap floats).
// Every thread of the threadgroup must call it.
#define QW_FOR(i, N) _Pragma("clang loop unroll(full)") for (uint i = 0; i < N; ++i)

// C's tile at p (row block r, column tile ct of C). In Q's formation (!TRANS,
// C = Q(k:, k:) for the block of nt column tiles at k), what is not yet
// written is made, not read: the block's own columns still [I; 0], and the
// later blocks' columns, which the next block's update wrote from its first
// row k + 8 nt on, zero above it.
template <bool TRANS>
inline void wy_c(thread simdgroup_float8x8& c, device const float* p, uint ldc, uint r, uint ct, uint nt,
                 threadgroup const float* eye) {
    if (!TRANS && r < nt) {
        if (r == ct) simdgroup_load(c, eye, 8);
        else c = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        return;
    }
    if (!TRANS && ct < nt) {
        c = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        return;
    }
    simdgroup_load(c, p, ldc);
}

template <uint NT, uint CT, bool TRANS>
inline void wy_apply(device float* C, device const float* Cin, uint ldc, device const float* V, uint ldv, uint k,
                     uint mp, uint c0, uint c1,
                     threadgroup const float* vd, device const float* tp, threadgroup float* wpart, uint wcap,
                     threadgroup const float* eye, uint sg, uint S) {
    const uint ntiles = (c1 - c0) / 8, nrb = (mp - k) / 8, ngroups = (ntiles + CT - 1) / CT;
    if (ntiles == 0) return;   // uniform
    uint G = 1;
    if (ngroups < S) G = max(1u, min(min(S / ngroups, nrb), wcap / (NT * CT * 64) / ngroups));
    const uint units = ngroups * G;
    for (uint u0 = 0; u0 < units; u0 += S) {   // uniform: rounds of S units
        const uint u = u0 + sg;
        const bool have = u < units;
        const uint cg = u / G, part = u % G;
        const uint r0 = part * nrb / G, r1 = (part + 1) * nrb / G;
        const uint nc = min(CT, ntiles - min(ntiles, cg * CT));   // this unit's column tiles
        device float* Ct = C + (ulong)k * ldc + c0 + 8 * CT * cg;
        device const float* Cti = Cin + (ulong)k * ldc + c0 + 8 * CT * cg;   // read from, Ct written
        simdgroup_float8x8 w[NT][CT];
        QW_FOR(i, NT) QW_FOR(j, CT) w[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        if (have) {
            // The diagonal row blocks: tiles right of the diagonal are zero
            for (uint r = r0; r < min(r1, NT); ++r) {
                simdgroup_float8x8 c[CT];
                QW_FOR(j, CT) if (j < nc) wy_c<TRANS>(c[j], Cti + (ulong)(8 * r) * ldc + 8 * j, ldc, r, CT * cg + j, NT, eye);
                QW_FOR(i, NT) {
                    if (i <= r) {
                        simdgroup_float8x8 v;
                        wy_v(v, r, i, V, ldv, k, vd, true);
                        QW_FOR(j, CT) simdgroup_multiply_accumulate(w[i][j], v, c[j], w[i][j]);
                    }
                }
            }
            device const float* vr = V + (ulong)(k + 8 * max(r0, NT)) * ldv + k;
            device const float* cr = Cti + (ulong)(8 * max(r0, NT)) * ldc;
            for (uint r = max(r0, NT); r < r1; ++r, vr += 8 * ldv, cr += 8 * ldc) {
                simdgroup_float8x8 c[CT];
                QW_FOR(j, CT) if (j < nc) wy_c<TRANS>(c[j], cr + 8 * j, ldc, r, CT * cg + j, NT, eye);
                QW_FOR(i, NT) {
                    simdgroup_float8x8 v;
                    simdgroup_load(v, vr + 8 * i, ldv, ulong2(0, 0), true);
                    QW_FOR(j, CT) simdgroup_multiply_accumulate(w[i][j], v, c[j], w[i][j]);
                }
            }
        }
        if (G > 1) {   // uniform
            if (have) QW_FOR(i, NT) QW_FOR(j, CT) simdgroup_store(w[i][j], wpart + ((u - u0) * NT * CT + i * CT + j) * 64, 8);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (have) {
            simdgroup_float8x8 y[NT][CT];
            QW_FOR(i, NT) QW_FOR(j, CT) y[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            for (uint g = 0; g < G; ++g) {
                if (G > 1)
                    QW_FOR(i, NT) QW_FOR(j, CT)
                        simdgroup_load(w[i][j], wpart + ((cg * G + g - u0) * NT * CT + i * CT + j) * 64, 8);
                QW_FOR(i, NT) {
                    QW_FOR(l, NT) {
                        if (TRANS ? l <= i : l >= i) {
                            simdgroup_float8x8 mt;
                            if (TRANS) simdgroup_load(mt, tp + (8 * l) * 64 + 8 * i, 64, ulong2(0, 0), true);
                            else simdgroup_load(mt, tp + (8 * i) * 64 + 8 * l, 64);
                            QW_FOR(j, CT) simdgroup_multiply_accumulate(y[i][j], mt, w[l][j], y[i][j]);
                        }
                    }
                }
            }
            QW_FOR(i, NT) QW_FOR(j, CT) wy_neg(y[i][j]);
            for (uint r = r0; r < min(r1, NT); ++r) {
                simdgroup_float8x8 c[CT];
                device float* p = Ct + (ulong)(8 * r) * ldc;
                device const float* pi = Cti + (ulong)(8 * r) * ldc;
                QW_FOR(j, CT) if (j < nc) wy_c<TRANS>(c[j], pi + 8 * j, ldc, r, CT * cg + j, NT, eye);
                QW_FOR(i, NT) {
                    if (i <= r) {
                        simdgroup_float8x8 v;
                        wy_v(v, r, i, V, ldv, k, vd, false);
                        QW_FOR(j, CT) simdgroup_multiply_accumulate(c[j], v, y[i][j], c[j]);
                    }
                }
                QW_FOR(j, CT) if (j < nc) simdgroup_store(c[j], p + 8 * j, ldc);
            }
            device const float* vr = V + (ulong)(k + 8 * max(r0, NT)) * ldv + k;
            device float* cr = Ct + (ulong)(8 * max(r0, NT)) * ldc;
            device const float* cri = Cti + (ulong)(8 * max(r0, NT)) * ldc;
            for (uint r = max(r0, NT); r < r1; ++r, vr += 8 * ldv, cr += 8 * ldc, cri += 8 * ldc) {
                simdgroup_float8x8 c[CT];
                QW_FOR(j, CT) if (j < nc) wy_c<TRANS>(c[j], cri + 8 * j, ldc, r, CT * cg + j, NT, eye);
                QW_FOR(i, NT) {
                    simdgroup_float8x8 v;
                    simdgroup_load(v, vr + 8 * i, ldv);
                    QW_FOR(j, CT) simdgroup_multiply_accumulate(c[j], v, y[i][j], c[j]);
                }
                QW_FOR(j, CT) if (j < nc) simdgroup_store(c[j], cr + 8 * j, ldc);
            }
        }
        if (G > 1) threadgroup_barrier(mem_flags::mem_threadgroup);   // wpart reused next round
    }
}

// wy_apply for a block of nt column tiles (2, 4, 6 or 8).
template <bool TRANS>
inline void wy_apply_n(uint nt, device float* C, device const float* Cin, uint ldc, device const float* V, uint ldv, uint k, uint mp, uint c0,
                       uint c1, threadgroup const float* vd, device const float* tp, threadgroup float* wpart,
                       uint wcap, threadgroup const float* eye, uint sg, uint S) {
    switch (nt) {   // uniform
        case 2: wy_apply<2, QW_CT, TRANS>(C, Cin, ldc, V, ldv, k, mp, c0, c1, vd, tp, wpart, wcap, eye, sg, S); break;
        case 4: wy_apply<4, QW_CT, TRANS>(C, Cin, ldc, V, ldv, k, mp, c0, c1, vd, tp, wpart, wcap, eye, sg, S); break;
        case 6: wy_apply<6, QW_CT, TRANS>(C, Cin, ldc, V, ldv, k, mp, c0, c1, vd, tp, wpart, wcap, eye, sg, S); break;
        default: wy_apply<8, QW_CT, TRANS>(C, Cin, ldc, V, ldv, k, mp, c0, c1, vd, tp, wpart, wcap, eye, sg, S); break;
    }
}

// The block's T after its panel p (p >= 1) at column k0 + 16 p, T's other
// columns done: T(0:16p, 16p:16p+16) = -T(0:16p, 0:16p) (V_prev^T V_p) T_p
// (slarft's recurrence by blocks). sc: 32 (QW_NB - 16) floats of scratch.
inline void wy_merge(device float* tp, device const float* V, uint ldv, uint k0, uint mp, uint p,
                     threadgroup const float* vd, threadgroup float* sc, uint sg, uint S) {
    const uint nrb = (mp - k0) / 8, na = 2 * p;
    threadgroup float* g = sc;          // 16p x 16, ld 16
    threadgroup float* y = sc + (QW_NB - QW_B) * 16;   // the same
    // G = V_prev^T V_p, from V_p's first row
    for (uint u = sg; u < 2 * na; u += S) {
        const uint a = u / 2, b = na + u % 2;
        simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        for (uint r = b; r < nrb; ++r) {
            simdgroup_float8x8 va, vb;
            wy_v(va, r, a, V, ldv, k0, vd, true);
            wy_v(vb, r, b, V, ldv, k0, vd, false);
            simdgroup_multiply_accumulate(acc, va, vb, acc);
        }
        simdgroup_store(acc, g + (8 * a) * 16 + 8 * (u % 2), 16);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Y = G T_p
    for (uint u = sg; u < 2 * na; u += S) {
        const uint a = u / 2, b = u % 2;
        simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        for (uint c = 0; c <= b; ++c) {
            simdgroup_float8x8 gt, tt;
            simdgroup_load(gt, g + (8 * a) * 16 + 8 * c, 16);
            simdgroup_load(tt, tp + (16 * p + 8 * c) * 64 + 16 * p + 8 * b, 64);
            simdgroup_multiply_accumulate(acc, gt, tt, acc);
        }
        simdgroup_store(acc, y + (8 * a) * 16 + 8 * b, 16);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // T's new columns = -T_prev Y
    for (uint u = sg; u < 2 * na; u += S) {
        const uint a = u / 2, b = u % 2;
        simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        for (uint c = a; c < na; ++c) {
            simdgroup_float8x8 tt, yt;
            simdgroup_load(tt, tp + (8 * a) * 64 + 8 * c, 64);
            simdgroup_load(yt, y + (8 * c) * 16 + 8 * b, 16);
            simdgroup_multiply_accumulate(acc, tt, yt, acc);
        }
        wy_neg(acc);
        simdgroup_store(acc, tp + (8 * a) * 64 + 16 * p + 8 * b, 64);
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
}

// The panel's step J, rows t + T s (x[s]), the panel at column k: the
// reflector of column J, applied to the columns right of it; T's column J on
// lane i < J of every simdgroup (trow). red holds a slot a simdgroup and
// alpha's at 32; red2 16 floats a simdgroup. Two barriers.
template <uint R, uint J>
inline void wy_step(thread float (&x)[R][QW_B], thread float (&trow)[QW_B], uint k, uint mp, uint t, uint T,
                    uint sg, uint lane, uint S, threadgroup float* red, threadgroup float4* red2) {
    const uint col = k + J;
    float ss = 0.0f;
    QH_UNROLL(R, s, {
        const uint row = t + T * s;
        if (row > col && row < mp) ss = fma(x[s][J], x[s][J], ss);
        if (row == col) red[32] = x[s][J];
    });
    ss = simd_sum(ss);
    if (lane == 0) red[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float sumsq = simd_sum(lane < S ? red[lane] : 0.0f);
    const float alpha = red[32];
    float beta, tau, scale;
    reflector(alpha, sumsq, beta, tau, scale);
    float v[R];
    QH_UNROLL(R, s, {
        const uint row = t + T * s;
        v[s] = 0.0f;
        if (row == col) { x[s][J] = beta; v[s] = 1.0f; }
        else if (row > col && row < mp) { x[s][J] *= scale; v[s] = x[s][J]; }
    });
    // v's dot products with every other column of the panel: right of J the
    // update's, left of it V^T v for T
    float4 loc[4];
    QH_UNROLL(4, f, {
        loc[f] = 0.0f;
        QH_UNROLL(4, e, {
            if (4 * f + e != J) { QH_UNROLL(R, s, { loc[f][e] = fma(v[s], x[s][4 * f + e], loc[f][e]); }); }
        });
        loc[f] = simd_sum(loc[f]);
    });
    if (lane < 4) {
        float4 mine = loc[0];
        QH_UNROLL(4, f, { if (f == lane) mine = loc[f]; });
        red2[sg * 4 + lane] = mine;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float4 acc = 0.0f;
    for (uint g = lane / 4; g < S; g += 8) acc += red2[g * 4 + lane % 4];
    acc += simd_shuffle_xor(acc, (ushort)4);
    acc += simd_shuffle_xor(acc, (ushort)8);
    acc += simd_shuffle_xor(acc, (ushort)16);
    float4 D[4];
    QH_UNROLL(4, f, { D[f] = simd_shuffle(acc, (ushort)f); });
    if (tau != 0.0f) {   // uniform
        QH_UNROLL(4, f, {
            QH_UNROLL(4, e, {
                if (4 * f + e > J) {
                    const float d = tau * D[f][e];
                    QH_UNROLL(R, s, { x[s][4 * f + e] = fma(-d, v[s], x[s][4 * f + e]); });
                }
            });
        });
    }
    float tij = 0.0f;
    QH_UNROLL(4, f, { QH_UNROLL(4, e, { if (4 * f + e < J) tij = fma(trow[4 * f + e], D[f][e], tij); }); });
    trow[J] = lane == J ? tau : (lane < J ? -tau * tij : 0.0f);
}

// Threadgroup memory, in floats: red 64, red2 16 S, vd 8 QW_NB (a block's
// diagonal tiles of V), eye 64 (the 8 x 8 identity), sc prm.sc (the merge's
// scratch, the updates' partial sums). qr_householder.mm computes the same
// size.

template <uint R>
kernel void qr_householder_wy(
    device const float* A_in [[buffer(0)]],   // [batch, m, n] input, row-major
    device float*       Q    [[buffer(1)]],   // [batch, m, K]
    device float*       Rout [[buffer(2)]],   // [batch, K, n]
    constant QwParams&  prm  [[buffer(3)]],
    device float*       W    [[buffer(4)]],   // [batch, mp, np] the matrix
    device float*       QW   [[buffer(5)]],   // [batch, mp, Kp] Q (unless q_direct)
    device float*       TW   [[buffer(6)]],   // [batch, Kp rounded up to 64, 64] the blocks' T
    threadgroup float*  tg   [[threadgroup(0)]],
    uint mat  [[threadgroup_position_in_grid]],
    uint t    [[thread_index_in_threadgroup]],
    uint T    [[threads_per_threadgroup]],
    uint sg   [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint S    [[simdgroups_per_threadgroup]])
{
    const uint m = prm.m, n = prm.n, K = min(m, n), mp = prm.mp, np = prm.np, Kp = prm.Kp;
    threadgroup float*  red  = tg;
    threadgroup float4* red2 = (threadgroup float4*)(tg + 64);
    threadgroup float*  vd   = tg + 64 + 16 * S;
    threadgroup float*  eye  = vd + 8 * QW_NB;
    threadgroup float*  sc   = eye + 64;

    device const float* src = A_in + (ulong)mat * m * n;
    device float* out_q = Q + (ulong)mat * m * prm.qc;
    device float* out_r = Rout + (ulong)mat * prm.rr * n;
    device float* A = W + (ulong)mat * mp * np;
    const uint qn = prm.qn;
    device float* Qw = prm.q_direct ? out_q : QW + (ulong)mat * mp * qn;
    device float* Tw = TW + (ulong)mat * ((Kp + 63) / 64) * 64 * 64;   // a block's T at Tw + k0 64, ld 64

    // 1. Scan. A matrix that needs neither padding nor scaling (its largest
    // entry within 2^-20 .. 2^20; the sums of squares need no more) is read
    // straight from the input by block 0, which writes the workspace; any
    // other is copied into it zero-padded, scaled by a power of two if need be.
    const bool aligned = mp == m && np == n;
    float amax = 0.0f, bad = 0.0f;
    if (aligned) {
        for (uint idx = t; idx < m * n; idx += T) {
            const float v = src[idx];
            amax = fmax(amax, fabs(v));   // fmax skips NaN, hence the separate flag
            if (non_finite(v)) bad = 1.0f;
        }
    } else {
        for (uint idx = t; idx < mp * np; idx += T) {
            const uint r = idx / np, c = idx - r * np;
            const float v = r < m && c < n ? src[r * n + c] : 0.0f;
            A[idx] = v;
            amax = fmax(amax, fabs(v));
            if (non_finite(v)) bad = 1.0f;
        }
    }
    for (uint idx = t; idx < 64; idx += T) eye[idx] = idx % 9 == 0 ? 1.0f : 0.0f;
    amax = group_max(amax, red, sg, lane, S);
    const bool nonfinite = group_sum(bad, red + 32, sg, lane, S) > 0.0f;
    if (nonfinite) {
        const float qnan = as_type<float>(0x7FC00000u);
        for (uint idx = t; idx < m * prm.qc; idx += T) out_q[idx] = qnan;
        for (uint idx = t; idx < prm.rr * n; idx += T) out_r[idx] = qnan;
        return;   // uniform
    }
    int expo = 0;
    if (amax > 0.0f) frexp(amax, expo);
    const bool scale = expo < -20 || expo > 20;
    if (!scale) expo = 0;
    const bool direct = aligned && !scale;   // uniform
    if (aligned && scale) {
        for (uint idx = t; idx < m * n; idx += T) A[idx] = ldexp(src[idx], -expo);
    } else if (scale) {
        threadgroup_barrier(mem_flags::mem_device);
        for (uint idx = t; idx < mp * np; idx += T) A[idx] = ldexp(A[idx], -expo);
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    // Where block 0 reads from: the input, read directly, or the workspace
    device const float* A0 = direct ? src : A;

    // 2. Blocks of QW_NB columns, panels of QW_B
    for (uint k0 = 0; k0 < Kp; k0 += QW_NB) {
        const uint nb = min((uint)QW_NB, Kp - k0);
        device float* tb = Tw + k0 * 64;
        for (uint p = 0; p < nb / QW_B; ++p) {
            const uint k = k0 + QW_B * p;
            float x[R][QW_B];
            QH_UNROLL(R, s, {
                const uint row = t + T * s;
                if (row >= k && row < mp) {
                    device const float4* q4 = (device const float4*)((k == 0 ? A0 : A) + (ulong)row * np + k);
                    QH_UNROLL(4, f, { const float4 g4 = q4[f]; QH_UNROLL(4, e, { x[s][4 * f + e] = g4[e]; }); });
                } else {
                    QH_UNROLL(QW_B, q, { x[s][q] = 0.0f; });
                }
            });
            float trow[QW_B];
            QH_UNROLL(QW_B, q, { trow[q] = 0.0f; });
            wy_step<R, 0>(x, trow, k, mp, t, T, sg, lane, S, red, red2);
            wy_step<R, 1>(x, trow, k, mp, t, T, sg, lane, S, red, red2);
            wy_step<R, 2>(x, trow, k, mp, t, T, sg, lane, S, red, red2);
            wy_step<R, 3>(x, trow, k, mp, t, T, sg, lane, S, red, red2);
            wy_step<R, 4>(x, trow, k, mp, t, T, sg, lane, S, red, red2);
            wy_step<R, 5>(x, trow, k, mp, t, T, sg, lane, S, red, red2);
            wy_step<R, 6>(x, trow, k, mp, t, T, sg, lane, S, red, red2);
            wy_step<R, 7>(x, trow, k, mp, t, T, sg, lane, S, red, red2);
            wy_step<R, 8>(x, trow, k, mp, t, T, sg, lane, S, red, red2);
            wy_step<R, 9>(x, trow, k, mp, t, T, sg, lane, S, red, red2);
            wy_step<R, 10>(x, trow, k, mp, t, T, sg, lane, S, red, red2);
            wy_step<R, 11>(x, trow, k, mp, t, T, sg, lane, S, red, red2);
            wy_step<R, 12>(x, trow, k, mp, t, T, sg, lane, S, red, red2);
            wy_step<R, 13>(x, trow, k, mp, t, T, sg, lane, S, red, red2);
            wy_step<R, 14>(x, trow, k, mp, t, T, sg, lane, S, red, red2);
            wy_step<R, 15>(x, trow, k, mp, t, T, sg, lane, S, red, red2);
            // The panel back (R above the diagonal, beta on it, V below), its
            // two diagonal tiles of V into vd, unit lower; T_p into T's
            // diagonal block
            QH_UNROLL(R, s, {
                const uint row = t + T * s;
                if (row >= k && row < mp) {
                    device float4* q4 = (device float4*)(A + (ulong)row * np + k);
                    QH_UNROLL(4, f, { q4[f] = float4(x[s][4 * f], x[s][4 * f + 1], x[s][4 * f + 2], x[s][4 * f + 3]); });
                    if (row < k + QW_B) {
                        const uint r = row - k, h = r / 8, rr = r % 8;
                        threadgroup float* d = vd + (2 * p + h) * 64 + rr * 8;
                        QH_UNROLL(QW_B, q, {
                            if (q / 8 == h) d[q % 8] = q < r ? x[s][q] : (q == r ? 1.0f : 0.0f);
                        });
                    }
                }
            });
            if (sg == 0 && lane < QW_B) {
                QH_UNROLL(QW_B, q, { tb[(QW_B * p + lane) * 64 + QW_B * p + q] = trow[q]; });
            }
            threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
            if (p > 0) wy_merge(tb, A, np, k0, mp, p, vd, sc, sg, S);
            // The rest of the block
            wy_apply<2, QW_CT, true>(A, k == 0 ? A0 : A, np, A, np, k, mp, k + QW_B, k0 + nb, vd + 128 * p,
                                     tb + (QW_B * p) * 64 + QW_B * p, sc, prm.sc, eye, sg, S);
            threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        }
        // The columns right of the block; then the block's rows of R are
        // final
        wy_apply_n<true>(nb / 8, A, k0 == 0 ? A0 : A, np, A, np, k0, mp, k0 + nb, np, vd, tb, sc, prm.sc, eye, sg, S);
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        const uint r1 = min(k0 + nb, K);
        for (uint idx = t + k0 * n; idx < r1 * n; idx += T) {
            const uint r = idx / n, c = idx - r * n;
            out_r[idx] = c >= r ? ldexp(A[(ulong)r * np + c], expo) : 0.0f;
        }
    }

    // R's rows below K, where Q is square
    for (uint idx = t + K * n; idx < prm.rr * n; idx += T) out_r[idx] = 0.0f;
    if (qn == 0) return;   // R alone

    // 3. Q from [I; 0] by the blocks backward, the identity and the zeros
    // made where Q is not yet written (wy_c). A square Q's columns beyond Kp
    // start as the identity, below row Kp (above it the blocks make zeros).
    if (qn > Kp) {
        for (uint idx = t; idx < (mp - Kp) * (qn - Kp); idx += T) {
            const uint r = Kp + idx / (qn - Kp), c = Kp + idx % (qn - Kp);
            Qw[(ulong)r * qn + c] = r == c ? 1.0f : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_device);
    }
    for (uint kb = (Kp + QW_NB - 1) / QW_NB; kb > 0; --kb) {
        const uint k0 = (kb - 1) * QW_NB, nb = min((uint)QW_NB, Kp - k0);
        for (uint idx = t; idx < nb * 8; idx += T) {
            const uint i = idx / 64, r = (idx / 8) % 8, c = idx % 8;
            vd[idx] = c < r ? A[(ulong)(k0 + 8 * i + r) * np + k0 + 8 * i + c] : (c == r ? 1.0f : 0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        wy_apply_n<false>(nb / 8, Qw, Qw, qn, A, np, k0, mp, k0, qn, vd, Tw + k0 * 64, sc, prm.sc, eye, sg, S);
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    }

    // 4. Q, from the workspace
    if (!prm.q_direct) {
        const uint qc = prm.qc;
        for (uint idx = t; idx < m * qc; idx += T) {
            const uint r = idx / qc, c = idx - r * qc;
            out_q[idx] = Qw[(ulong)r * qn + c];
        }
    }
}

#define QW_INST(R)                                                                                          \
    template [[host_name("qr_householder_wy_" #R)]] kernel void qr_householder_wy<R>(                      \
        device const float*, device float*, device float*, constant QwParams&, device float*, device float*, \
        device float*, threadgroup float*, uint, uint, uint, uint, uint, uint);
QW_INST(1)
QW_INST(2)
QW_INST(4)
