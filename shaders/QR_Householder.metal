#include <metal_stdlib>
using namespace metal;

// =============================================================================
// Batched QR of small matrices: Householder QR in threadgroup memory
// =============================================================================
//
// Computes the thin QR A = Q R of a batch of real m x n matrices, one
// threadgroup per matrix, the whole matrix in threadgroup memory (so the size
// is bounded by it: about 90 x 90 at 32 KB, longer when narrow). This is
// LAPACK's method -- sgeqr2's column-by-column Householder reflectors, then
// sorg2r's accumulation of Q in place -- built as Svd_GolubKahan.metal and
// Eigh_QL.metal are, whose steps 1, 2 and 4 it shares in kind: the QR
// counterpart of those kernels, and the replacement for QR_Unblocked.metal's,
// which walked device memory with a barrier per phase of every column.
//
// Threads: one per row, thread i owning row i; a phase that sums down columns
// gives each column a group of lanes (lanes_per_column), all the
// threadgroup's threads taking part, so that a wide matrix's many columns
// are covered too.
//
//   1. load the row-major input as it is, scan for the largest entry and for
//      non-finite ones; scale by a power of two so the largest entry lies in
//      [0.5, 1) (NaN out for a non-finite matrix)
//   2. factor (sgeqr2): per column k < min(m, n) the reflector H(k) = I -
//      tau v v^T zeroing column k below the diagonal (v(k) = 1, its tail kept
//      there), applied to the columns right of it: three barriers a column
//   3. write R (min(m, n) x n, row-major), scaled back
//   4. form Q = H(0) ... H(K-1), its first K columns, in place (sorg2r)
//   5. write Q (m x K, row-major)
//
// References:
//   Golub & Van Loan, Matrix Computations 4th ed., s5.1-5.2 (Householder
//     reflections and QR) and s5.1.6 (accumulating their product backward).
//   LAPACK sgeqr2, slarfg, sorg2r.
//
// Built with fast math, as Svd_GolubKahan.metal and Eigh_QL.metal are: the
// columns are a latency-bound chain. What needs the accuracy -- the
// reflector's norm and scalars -- takes one Newton step after the fast
// division or square root, and the non-finite input check reads the bits.

// Must match `QhParams` in qr_householder.mm.
struct QhParams {
    uint m;   // rows
    uint n;   // columns
};

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

// Lanes per column for a phase that sums down columns: `ncols` columns of
// `len` rows, over `nthreads` threads. A tall matrix has few columns and long
// ones, so walking each on one thread is a long dependent chain while most
// threads wait; a group of lanes splits it. A power of two up to 32, so a
// group never straddles a simdgroup; 1 keeps a column on one thread.
inline uint lanes_per_column(uint nthreads, uint ncols, uint len) {
    uint g = 1;
    while (g < 32 && 2 * g * ncols <= nthreads && 4 * g <= len) g *= 2;
    return g;
}

// The sum over each aligned group of g lanes (g a power of two), on every lane
// of it. Every thread of the simdgroup must call it, with the same g.
inline float group_reduce(float s, uint g) {
    for (uint off = g >> 1; off > 0; off >>= 1) s += simd_shuffle_xor(s, (ushort)off);
    return s;
}

// The sums w[c] = t (a[k][c] + sum_{r > k} a[r][k] a[r][c]) for c in c0 .. n-1
// (the first term only when `head`), by groups of g lanes over every thread,
// two chains a group: the walk is latency-bound.
inline void column_sums(threadgroup const float* a, uint ld, uint m, uint n, uint k, uint c0, float t,
                        bool head, threadgroup float* w, uint tid, uint ntg) {
    const uint g = lanes_per_column(ntg, n - c0, m - k - 1);
    const uint groups = ntg / g, lg = tid % g;
    // Every lane of a simdgroup runs the same number of rounds, for the
    // shuffles: the rounds are counted over the groups, not the columns left.
    const uint rounds = (n - c0 + groups - 1) / groups;
    for (uint q = 0; q < rounds; ++q) {
        const uint c = c0 + q * groups + tid / g;
        const bool active = c < n;
        float s0 = 0.0f, s1 = 0.0f;
        if (active) {
            uint r = k + 1 + lg;
            for (; r + g < m; r += 2 * g) {
                s0 += a[r * ld + k] * a[r * ld + c];
                s1 += a[(r + g) * ld + k] * a[(r + g) * ld + c];
            }
            if (r < m) s0 += a[r * ld + k] * a[r * ld + c];
        }
        const float s = group_reduce(s0 + s1, g);
        if (active && lg == 0) w[c] = t * (head ? s + a[k * ld + c] : s);
    }
}

// Threadgroup memory, in floats, with ld = n | 1 (an odd row stride, so a
// column walk touches every bank once):
//   a     m * ld   A, then V and R, then Q in its first K columns
//   beta  K        R's diagonal
//   tau   K        the reflectors' scalars
//   w     n        column sums
//   red   64       reductions: two sets of one slot per simdgroup (up to 32)
// qr_householder.mm computes the same size.
kernel void qr_householder(
    device const float* A_in [[buffer(0)]],   // [batch, m, n] input, row-major
    device float*       Q    [[buffer(1)]],   // [batch, m, K]
    device float*       R    [[buffer(2)]],   // [batch, K, n]
    constant QhParams&  prm  [[buffer(3)]],
    threadgroup float*  tg   [[threadgroup(0)]],
    uint mat  [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint ntg  [[threads_per_threadgroup]],
    uint sg   [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint nsg  [[simdgroups_per_threadgroup]])
{
    const uint m = prm.m, n = prm.n, K = min(m, n);
    const uint ld = n | 1u;
    threadgroup float* a    = tg;
    threadgroup float* beta = a + m * ld;
    threadgroup float* tau  = beta + K;
    threadgroup float* w    = tau + K;
    threadgroup float* red  = w + n;
    const uint i = tid;            // this thread's row, if it has one
    const bool row = i < m;

    device const float* src = A_in + (ulong)mat * m * n;
    device float* out_q = Q + (ulong)mat * m * K;
    device float* out_r = R + (ulong)mat * K * n;

    // -------------------------------------------------------------------------
    // 1. Load, scan, scale
    // -------------------------------------------------------------------------
    float amax = 0.0f, bad = 0.0f;
    for (uint idx = tid; idx < m * n; idx += ntg) {
        const uint r = idx / n, c = idx - r * n;
        const float v = src[idx];
        a[r * ld + c] = v;
        amax = fmax(amax, fabs(v));   // fmax skips NaN, hence the separate flag
        if (non_finite(v)) bad = 1.0f;
    }
    amax = group_max(amax, red, sg, lane, nsg);
    const bool nonfinite = group_sum(bad, red + 32, sg, lane, nsg) > 0.0f;
    if (nonfinite) {
        const float qnan = as_type<float>(0x7FC00000u);
        for (uint k = tid; k < m * K; k += ntg) out_q[k] = qnan;
        for (uint k = tid; k < K * n; k += ntg) out_r[k] = qnan;
        return;   // uniform: every thread has the same flag
    }
    // A power of two, so exact.
    int expo = 0;
    if (amax > 0.0f) frexp(amax, expo);
    for (uint idx = tid; idx < m * n; idx += ntg) {
        const uint r = idx / n, c = idx - r * n;
        a[r * ld + c] = ldexp(a[r * ld + c], -expo);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // -------------------------------------------------------------------------
    // 2. Factor (sgeqr2)
    // -------------------------------------------------------------------------
    // Column k: the norm of its tail (a barrier), the reflector, the column
    // sums of the columns right of it (a barrier), the row-local update; the
    // next column's sum waits for the update.
    for (uint k = 0; k < K; ++k) {
        float t = 0.0f;
        if (row && i > k) { const float v = a[i * ld + k]; t = v * v; }
        const float sumsq = group_sum(t, red + 32 * (k & 1u), sg, lane, nsg);
        const float alpha = a[k * ld + k];
        float tv = 0.0f, b = alpha;
        if (sumsq > 0.0f) {
            b = -copysign(sqrt_nr(alpha * alpha + sumsq), alpha);
            tv = div_nr(b - alpha, b);
            const float scal = div_nr(1.0f, alpha - b);
            if (row && i > k) a[i * ld + k] *= scal;
        }
        if (tid == 0) { beta[k] = b; tau[k] = tv; }
        if (k + 1 >= n) break;   // uniform; step 3 starts with a barrier
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tv != 0.0f) {   // uniform
            column_sums(a, ld, m, n, k, k + 1, tv, true, w, tid, ntg);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (row && i >= k) {
                threadgroup float* x = a + i * ld;
                const float vi = (i == k) ? 1.0f : x[k];
                for (uint j = k + 1; j < n; ++j) x[j] -= vi * w[j];
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // -------------------------------------------------------------------------
    // 3. R, before step 4 overwrites its upper triangle
    // -------------------------------------------------------------------------
    for (uint idx = tid; idx < K * n; idx += ntg) {
        const uint r = idx / n, c = idx - r * n;
        const float v = c > r ? a[r * ld + c] : (c == r ? beta[r] : 0.0f);
        out_r[idx] = ldexp(v, expo);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // -------------------------------------------------------------------------
    // 4. Q = H(0) H(1) ... H(K-1), its first K columns, in place (sorg2r)
    // -------------------------------------------------------------------------
    // Backward: before step j, columns j+1 .. K-1 hold H(j+1) ... H(K-1) and
    // are zero in rows up to j; H(j)'s vector is read from column j, which
    // step j itself overwrites last.
    for (int jj = (int)K - 1; jj >= 0; --jj) {
        const uint j = (uint)jj;
        const float tj = tau[j];
        const bool apply = tj != 0.0f && j + 1 < K;   // uniform
        if (apply) {
            column_sums(a, ld, m, K, j, j + 1, tj, false, w, tid, ntg);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (row) {
            threadgroup float* x = a + i * ld;
            if (i < j) {
                x[j] = 0.0f;
            } else if (i == j) {
                if (apply) for (uint c = j + 1; c < K; ++c) x[c] = -w[c];
                x[j] = 1.0f - tj;
            } else {
                const float v = x[j];
                if (apply) for (uint c = j + 1; c < K; ++c) x[c] -= v * w[c];
                x[j] = -tj * v;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // -------------------------------------------------------------------------
    // 5. Q
    // -------------------------------------------------------------------------
    for (uint idx = tid; idx < m * K; idx += ntg) {
        const uint r = idx / K, c = idx - r * K;
        out_q[idx] = a[r * ld + c];
    }
}

// =============================================================================
// The same in registers: a simdgroup a matrix (qr_householder_simd)
// =============================================================================
//
// For n <= 64 with m <= 64, and n <= 32 with m <= 128, a matrix fits in one
// simdgroup's registers -- R rows a lane (row s * 32 + lane in x[s]), B
// columns -- as the band reduction's panels keep theirs (qr_simd in
// Svd_Bidiag.metal). Then nothing waits on a barrier and no threadgroup
// memory is used, so a core holds as many matrices as its registers allow,
// several to a threadgroup. A step's dot products, one per column right of
// the current one, are simd_sums four at a time (on a float4).
//
// (A column a lane instead, each lane's dot products its own FMAs and the
// reflector's vector shuffled across a row at a time, was 2.2x slower at
// 32 x 32 and 6x at 64 x 64: simd_sum is cheap here, and the column layout
// pays every row's FMAs on every step.)
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
#define QH_UNROLL_e(...) { { constexpr uint e = 0; __VA_ARGS__ } { constexpr uint e = 1; __VA_ARGS__ } { constexpr uint e = 2; __VA_ARGS__ } { constexpr uint e = 3; __VA_ARGS__ } }
#define QH_UNROLL(N, v, ...) QH_UNROLL_##v(if (v < N) __VA_ARGS__)

// Must match `QsParams` in qr_householder.mm.
struct QsParams {
    uint m;       // rows, at most 32 R
    uint n;       // columns, at most B
    uint batch;   // matrices in this dispatch
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
    device float*       Q    [[buffer(1)]],   // [batch, m, K]
    device float*       Rout [[buffer(2)]],   // [batch, K, n]
    constant QsParams&  prm  [[buffer(3)]],
    uint tgi  [[threadgroup_position_in_grid]],
    uint sg   [[simdgroup_index_in_threadgroup]],
    uint nsg  [[simdgroups_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    const uint mat = tgi * nsg + sg;
    if (mat >= prm.batch) return;   // a whole simdgroup: nothing below waits on the others
    const uint m = prm.m, n = prm.n, K = min(m, n);
    device const float* src = A_in + (ulong)mat * m * n;
    device float* out_q = Q + (ulong)mat * m * K;
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
        for (uint idx = lane; idx < m * K; idx += 32) out_q[idx] = qnan;
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
QH_SIMD(64, 1)
QH_SIMD(64, 2)
