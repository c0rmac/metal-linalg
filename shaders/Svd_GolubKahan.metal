#include <metal_stdlib>
using namespace metal;

// =============================================================================
// Batched SVD: Householder bidiagonalization and implicit bidiagonal QR
// =============================================================================
//
// Computes the thin SVD A = U diag(s) V^T of a batch of real M x N matrices,
// one threadgroup per matrix, the whole matrix in threadgroup memory (so the
// size is bounded by it: 83 x 83 at 32 KB, longer when tall). This is LAPACK's
// method (sgebd2, sorg2r / sorgbr's accumulation, sbdsqr's implicit QR), the
// SVD counterpart of Eigh_QL.metal, and does several times fewer flops than
// the one-sided Jacobi kernels.
//
// V lives in device memory (a workspace), transposed, so that the threads
// owning consecutive rows of it touch consecutive addresses. Kept in
// threadgroup memory beside the matrix it halved how many matrices share a
// core, and the kernel is bound by that: on an M5 Pro moving it out made
// 4096 matrices of 32 x 32 1.4x faster and 1024 of 48 x 48 1.5x.
//
// The QR iteration works on the bidiagonal (d, e) alone, so one thread runs it
// and records each step's rotations -- the left ones for U, the right ones for
// V -- and then every thread applies the recorded sequence to its own row of U
// or of V. A step costs one barrier, and from kOverlap a dedicated simdgroup
// computes the next step while the rows apply this one, as in Eigh_QL.metal.
//
// The kernel works on B with m >= n rows: A itself, or A^T for a wide A
// (prm.transpose), whose factors are then swapped on the way out.
//
// Threads: one per row of B and, with vectors, of V (m + n), thread i owning
// row i of B for i < m and row i - m of V above; a phase that works by
// columns gives each column a group of lanes (lanes_per_column). With
// kOverlap, simdgroup 0 is the chaser (lane 0 runs the QR iteration) and
// thread 32 + i owns row i.
//
//   1. load (transposed if wide), scan for the largest entry and for
//      non-finite ones; scale by a power of two so the largest entry lies in
//      [0.5, 1)
//   2. bidiagonalize, B = Q Bd P^T (sgebd2, upper): per column a left
//      reflector, per row a right one; d and e hold the bidiagonal
//   3. with vectors, form V = P from the right reflectors
//   4. with vectors, form Q in place from the left reflectors (sorg2r)
//   5. implicit QR with sbdsqr's shift on (d, e), deflating from the bottom;
//      a negligible diagonal entry is chased out with rotations of its own
//   6. make the singular values non-negative, rank sort descending, write
//
// References:
//   Golub & Van Loan, Matrix Computations 4th ed., s5.4.8 (bidiagonalization)
//     and s8.6.2-3 (the Golub-Kahan SVD step, and zero diagonal entries).
//   LAPACK sgebd2, sorg2r, sbdsqr (its shifted QR sweep, slas2's shift);
//   Demmel & Kahan, SIAM J. Sci. Stat. Comput. 11, 1990.
//
// Built with fast math, like Eigh_QL.metal and for the same reason: the QR
// iteration is a latency-bound chain. The Householder vectors take one Newton
// step after the fast division or square root, the rotations come from one
// reciprocal square root, and the non-finite input check reads the bits.

constant bool kComputeVectors [[function_constant(0)]];
constant bool kOverlap        [[function_constant(1)]];   // a chaser simdgroup of its own
// Which part of the work this dispatch does. 0: all of it. From mid-size k the
// work is split in two dispatches, because the QR iteration is one thread's
// chain of dependent rotations, and in a threadgroup sized for the matrix too
// few matrices share a core to hide it:
//   1  steps 1-4: the reduction and Q and V, then d, e and U into a workspace
//   2  steps 5-6: the QR iteration and the output, with U and V in device
//      memory and almost no threadgroup memory, so many matrices overlap
constant uint kPart           [[function_constant(2)]];
constant bool kReduce = kPart != 2;

// Must match `GkParams` in svd_golub_kahan.mm.
struct GkParams {
    uint m;           // rows of B, m >= n
    uint n;           // columns of B: min(M, N)
    uint transpose;   // 1: B = A^T (A is wide), 0: B = A
    uint max_rots;    // rotations allowed per matrix in total (LAPACK: 6 n^2)
};

// Info word written per matrix: QR steps | converged | non-finite input |
// rank-deficient, as the Jacobi kernels report it.
constant uint kInfoConverged     = 1u << 16;
constant uint kInfoNonFinite     = 1u << 17;
constant uint kInfoRankDeficient = 1u << 18;

// A singular value below this many epsilon times the largest marks the
// matrix rank-deficient, as kNullEps does for the Jacobi kernels (svd.mm).
// U needs no completion here: its columns come from reflectors and rotations.
constant float kNullEps = 32.0f;

constant uint kChaser = kOverlap ? 32 : 0;   // threads before the first row thread

// Threadgroup memory, in floats, with ld = n | 1 (an odd row stride, so a
// column walk touches every bank once):
//   a        m * ld         B, then Q, then U (none in part 2, where U is in
//                           the workspace)
//   d, e     n each         the bidiagonal: d[i] = Bd(i, i), e[i] = Bd(i, i+1)
//   tq, tp   n each         the reflectors' scalars (steps 2-4)
//   wb       n              column sums (steps 2-4)
//   x        8n             step 5: two rotation buffers of (c, s) for U and
//                           (c, s) for V, 4n each; step 6: ranks
//   red      64             reductions: two sets of one slot per simdgroup (up to 32)
//   ctl      32 uints       step 5: per rotation buffer, at 0 and 16; see gk_step
// svd_golub_kahan.mm computes the same size.

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
// `len` rows, over `nrows` row threads. A tall matrix has few columns and long
// ones, so walking each on one thread is a long dependent chain while most
// threads wait; a group of lanes splits it. A power of two up to 32, so a
// group never straddles a simdgroup (row threads start on a simdgroup
// boundary); 1 keeps a column on one thread.
inline uint lanes_per_column(uint nrows, uint ncols, uint len) {
    uint g = 1;
    while (g < 32 && 2 * g * ncols <= nrows && 4 * g <= len) g *= 2;
    return g;
}

// The sum over each aligned group of g lanes (g a power of two), on every lane
// of it. Every thread of the simdgroup must call it, with the same g.
inline float group_reduce(float s, uint g) {
    for (uint off = g >> 1; off > 0; off >>= 1) s += simd_shuffle_xor(s, (ushort)off);
    return s;
}

// The plane rotation taking (f, g) to (r, 0): c = f / r, s = g / r, r =
// hypot(f, g) >= 0, from one reciprocal square root (scaled where f^2 + g^2
// would under- or overflow). f = g = 0 gives c = 1, s = 0, r = 0.
inline float rotation(float f, float g, thread float& c, thread float& s) {
    const float h = f * f + g * g;
    if (h >= FLT_MIN && h <= FLT_MAX) {
        const float inv = fast::rsqrt(h);
        c = f * inv;
        s = g * inv;
        return h * inv;
    }
    const float big = fmax(fabs(f), fabs(g));
    if (big == 0.0f) {
        c = 1.0f;
        s = 0.0f;
        return 0.0f;
    }
    const float fs = f / big, gs = g / big;
    const float inv = fast::rsqrt(fs * fs + gs * gs);
    c = fs * inv;
    s = gs * inv;
    return big / inv;
}

// The smaller singular value of the upper triangular [[f, g], [0, h]]
// (LAPACK slas2): the shift.
inline float smin_2x2(float f, float g, float h) {
    const float fa = fabs(f), ga = fabs(g), ha = fabs(h);
    const float fhmn = fmin(fa, ha), fhmx = fmax(fa, ha);
    if (fhmn == 0.0f) return 0.0f;
    if (ga < fhmx) {
        const float as = 1.0f + fhmn / fhmx;
        const float at = (fhmx - fhmn) / fhmx;
        const float au = (ga / fhmx) * (ga / fhmx);
        const float c = 2.0f / (sqrt(as * as + au) + sqrt(at * at + au));
        return fhmn * c;
    }
    const float au = fhmx / ga;
    if (au == 0.0f) return (fhmn * fhmx) / ga;
    const float as = 1.0f + fhmn / fhmx;
    const float at = (fhmx - fhmn) / fhmx;
    const float c = 1.0f / (sqrt(1.0f + (as * au) * (as * au)) + sqrt(1.0f + (at * au) * (at * au)));
    return 2.0f * (fhmn * c) * au;
}

// e is negligible against its two diagonal neighbours (sbdsqr's relative test).
inline bool negligible(float e, float da, float db) {
    const float ae = fabs(e);
    return ae <= FLT_EPSILON * (fabs(da) + fabs(db)) || ae < FLT_MIN;
}

// The chaser's state across steps.
struct GkState {
    uint  hi;       // d[hi+1 ..] have converged
    uint  rots;     // rotations so far, against the budget
    uint  steps;    // QR steps so far
    bool  failed;   // the budget ran out
    float dtol;     // a diagonal entry at most this is zero: eps * |Bd|
};

// One rotation list, as the rows apply it. Rotation r acts on columns (a, b)
// of a row z: (z[a], z[b]) <- (c z[a] + s z[b], c z[b] - s z[a]), with
//   kind 0  a = a0 + r, b = a + 1        (a QR step; ascending)
//   kind 1  a = a0 + r, b fixed          (a zero diagonal entry chased along its row)
//   kind 2  a = a0 - r, b fixed          (... chased up its column)
// meta, per rotation buffer: [0] done, [1..4] U's {kind, a0, count, b},
// [5..8] V's.
constant uint kMetaU = 1, kMetaV = 5;

// Finds the next step and runs it on (d, e), recording its rotations if
// `record`: the left ones in (uc, us), the right ones in (vc, vs). Deflates
// converged values at the bottom first; sets done once every value has
// converged, or when the budget runs out.
inline void gk_step(threadgroup float* d, threadgroup float* e,
                    threadgroup float* uc, threadgroup float* us,
                    threadgroup float* vc, threadgroup float* vs,
                    threadgroup uint* meta, thread GkState& st, uint max_rots, bool record) {
    meta[kMetaU + 2] = 0;
    meta[kMetaV + 2] = 0;
    uint hi = st.hi;
    while (hi > 0 && negligible(e[hi - 1], d[hi - 1], d[hi])) {
        e[hi - 1] = 0.0f;
        --hi;
    }
    st.hi = hi;
    if (hi == 0) {
        meta[0] = 1;
        return;
    }
    // The top of the unreduced block ending at hi.
    uint lo = hi - 1;
    while (lo > 0 && !negligible(e[lo - 1], d[lo - 1], d[lo])) --lo;
    if (lo > 0) e[lo - 1] = 0.0f;

    // A negligible diagonal entry: set it to zero and chase its row (or, at
    // the bottom of the block, its column) out of the block, which splits it
    // (Golub & Van Loan s8.6.2).
    for (uint z = lo; z <= hi; ++z) {
        if (fabs(d[z]) > st.dtol) continue;
        d[z] = 0.0f;
        float c, s;
        if (z < hi) {
            // Row z holds f = e[z] at column z+1. Left rotations of rows
            // (j, z), j = z+1 .. hi, each zeroing it against d[j] and moving
            // it one column right.
            float f = e[z];
            e[z] = 0.0f;
            for (uint j = z + 1; j <= hi; ++j) {
                d[j] = rotation(d[j], f, c, s);
                if (record) { uc[j - z - 1] = c; us[j - z - 1] = s; }
                if (j < hi) {
                    f = -s * e[j];
                    e[j] = c * e[j];
                }
            }
            meta[kMetaU + 0] = 1;
            meta[kMetaU + 1] = z + 1;
            meta[kMetaU + 2] = hi - z;
            meta[kMetaU + 3] = z;
        } else {
            // Column hi holds f = e[hi-1] at row hi-1. Right rotations of
            // columns (j, hi), j = hi-1 .. lo, each zeroing it against d[j]
            // and moving it one row up.
            float f = e[hi - 1];
            e[hi - 1] = 0.0f;
            for (uint j = hi; j-- > lo;) {
                d[j] = rotation(d[j], f, c, s);
                if (record) { vc[hi - 1 - j] = c; vs[hi - 1 - j] = s; }
                if (j > lo) {
                    f = -s * e[j - 1];
                    e[j - 1] = c * e[j - 1];
                }
            }
            meta[kMetaV + 0] = 2;
            meta[kMetaV + 1] = hi - 1;
            meta[kMetaV + 2] = hi - lo;
            meta[kMetaV + 3] = hi;
        }
        meta[0] = 0;
        return;
    }

    if (st.rots >= max_rots) {
        st.failed = true;
        meta[0] = 1;
        return;
    }
    st.rots += hi - lo;
    ++st.steps;

    // The shift: the smaller singular value of the trailing 2 x 2, or none
    // where it is negligible against the top of the block (sbdsqr).
    float shift = smin_2x2(d[hi - 1], e[hi - 1], d[hi]);
    const float sll = fabs(d[lo]);
    if ((shift / sll) * (shift / sll) < FLT_EPSILON) shift = 0.0f;

    // sbdsqr's chase from the top, shifted: the right rotation (cr, sr) of
    // columns (i, i+1) makes a bulge below the diagonal, the left one (cl, sl)
    // of rows (i, i+1) moves it to the right of the superdiagonal.
    // d[i] and e[i] as the previous rotation left them are carried in
    // registers; their final values are written one rotation later.
    float f = (sll - shift) * (copysign(1.0f, d[lo]) + shift / d[lo]);
    float g = e[lo];
    float di = d[lo], ei = e[lo];
    for (uint i = lo; i < hi; ++i) {
        const float dn = d[i + 1];                       // untouched until here:
        const float e1 = (i + 1 < hi) ? e[i + 1] : 0.0f;  // loaded off the chain
        float cr, sr, cl, sl;
        const float r = rotation(f, g, cr, sr);
        if (i > lo) e[i - 1] = r;
        f = cr * di + sr * ei;
        const float en = cr * ei - sr * di;
        g = sr * dn;
        const float dn2 = cr * dn;
        d[i] = rotation(f, g, cl, sl);
        f = cl * en + sl * dn2;
        di = cl * dn2 - sl * en;
        g = sl * e1;
        ei = cl * e1;
        if (record) {
            vc[i - lo] = cr; vs[i - lo] = sr;
            uc[i - lo] = cl; us[i - lo] = sl;
        }
    }
    d[hi] = di;
    e[hi - 1] = f;
    meta[kMetaU + 0] = 0; meta[kMetaU + 1] = lo; meta[kMetaU + 2] = hi - lo; meta[kMetaU + 3] = 0;
    meta[kMetaV + 0] = 0; meta[kMetaV + 1] = lo; meta[kMetaV + 2] = hi - lo; meta[kMetaV + 3] = 0;
    meta[0] = 0;
}

// Applies one recorded list to the row z; see gk_step for the encoding. The
// value at the column every rotation shares -- the next pair's first column
// for a QR step, the fixed column otherwise -- is carried in a register.
inline void apply_rotations(threadgroup float* z, threadgroup const uint* list,
                            threadgroup const float* cb, threadgroup const float* sb) {
    const uint kind = list[0], a0 = list[1], cnt = list[2], b = list[3];
    if (cnt == 0) return;
    if (kind == 0) {
        float carry = z[a0];
        for (uint r = 0; r < cnt; ++r) {
            const float y = z[a0 + r + 1], c = cb[r], s = sb[r];
            z[a0 + r] = c * carry + s * y;
            carry = c * y - s * carry;
        }
        z[a0 + cnt] = carry;
    } else {
        float carry = z[b];
        for (uint r = 0; r < cnt; ++r) {
            const uint a = kind == 1 ? a0 + r : a0 - r;
            const float x = z[a], c = cb[r], s = sb[r];
            z[a] = c * x + s * carry;
            carry = c * carry - s * x;
        }
        z[b] = carry;
    }
}

// The same for a row of V, which lives in device memory with element c at
// z[c * ld]: the next element is loaded before this one is stored, so that the
// loads stay off the chain.
inline void apply_rotations(device float* z, uint ld, threadgroup const uint* list,
                            threadgroup const float* cb, threadgroup const float* sb) {
    const uint kind = list[0], a0 = list[1], cnt = list[2], b = list[3];
    if (cnt == 0) return;
    if (kind == 0) {
        float carry = z[a0 * ld];
        float y = z[(a0 + 1) * ld];
        for (uint r = 0; r < cnt; ++r) {
            const float c = cb[r], s = sb[r];
            const float next = (r + 1 < cnt) ? z[(a0 + r + 2) * ld] : 0.0f;
            z[(a0 + r) * ld] = c * carry + s * y;
            carry = c * y - s * carry;
            y = next;
        }
        z[(a0 + cnt) * ld] = carry;
    } else {
        float carry = z[b * ld];
        uint a = a0;
        float x = z[a * ld];
        for (uint r = 0; r < cnt; ++r) {
            const float c = cb[r], s = sb[r];
            const uint an = kind == 1 ? a + 1 : a - 1;
            const float next = (r + 1 < cnt) ? z[an * ld] : 0.0f;
            z[a * ld] = c * x + s * carry;
            carry = c * carry - s * x;
            a = an;
            x = next;
        }
        z[b * ld] = carry;
    }
}

kernel void svd_golub_kahan(
    device const float* A_in [[buffer(0)]],   // [batch, M, N] input (row-major)
    device float*       S    [[buffer(1)]],   // [batch, n] singular values, descending
    device float*       U    [[buffer(2)]],   // [batch, M, n] (kComputeVectors)
    device float*       Vt   [[buffer(3)]],   // [batch, n, N] (kComputeVectors)
    device uint*        info [[buffer(4)]],   // [batch] steps | flags
    constant GkParams&  prm  [[buffer(5)]],
    device float*       Vw   [[buffer(6)]],   // [batch, n, n] workspace: V, transposed (kComputeVectors)
    device float*       Uw   [[buffer(7)]],   // [batch, n, m] workspace: U, transposed (parts 1-2, vectors)
    device float*       Hw   [[buffer(8)]],   // [batch, 2n + 2] workspace: d, e, the exponent, non-finite (parts 1-2)
    threadgroup float*  tg   [[threadgroup(0)]],
    uint mat  [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint ntg  [[threads_per_threadgroup]],
    uint sg   [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint nsg  [[simdgroups_per_threadgroup]])
{
    const uint m  = prm.m;
    const uint n  = prm.n;
    const uint ld = n | 1u;
    const uint rows = m + (kComputeVectors ? n : 0);   // threads with a row: of B, and of V
    threadgroup float* a    = tg;
    threadgroup float* d    = a + (kReduce ? m * ld : 0);
    threadgroup float* e    = d + n;
    threadgroup float* tq   = e + n;
    threadgroup float* tp   = tq + n;
    threadgroup float* wb   = tp + n;
    threadgroup float* x    = wb + n;
    threadgroup float* red  = x + 8 * n;
    threadgroup uint*  ctl  = reinterpret_cast<threadgroup uint*>(red + 64);
    threadgroup uint*  rank = reinterpret_cast<threadgroup uint*>(x);   // step 6

    const uint i   = tid - kChaser;            // this thread's row, if it has one
    const bool row = tid >= kChaser && i < rows;
    const bool wrow = row && i < m;            // a row of B / Q / U
    // V, in device memory and transposed: V(r, c) at vg[c * n + r], so that
    // the threads owning consecutive rows touch consecutive addresses.
    device float* vg = Vw + (ulong)mat * n * n;
    // Parts 1-2: U likewise, U(r, c) at ug[c * m + r], and the header.
    device float* ug = Uw + (ulong)mat * n * m;
    device float* hw = Hw + (ulong)mat * (2 * n + 2);
    constexpr auto both = mem_flags::mem_threadgroup | mem_flags::mem_device;

    const uint M = prm.transpose ? n : m, N = prm.transpose ? m : n;
    device float* out_s  = S + (ulong)mat * n;
    device float* out_u  = U + (ulong)mat * M * n;
    device float* out_vt = Vt + (ulong)mat * n * N;
    int expo = 0;

    if (kReduce) {

        // ---------------------------------------------------------------------
        // 1. Load, scan, scale
        // ---------------------------------------------------------------------
        device const float* src = A_in + (ulong)mat * m * n;
        float amax = 0.0f, bad = 0.0f;
        for (uint idx = tid; idx < m * n; idx += ntg) {
            // Read in the input's own order: row-major A, which is B^T if wide.
            uint r, k;
            if (prm.transpose) { k = idx / m; r = idx - k * m; }
            else               { r = idx / n; k = idx - r * n; }
            const float v = src[idx];
            a[r * ld + k] = v;
            amax = fmax(amax, fabs(v));   // fmax skips NaN, hence the separate flag
            if (non_finite(v)) bad = 1.0f;
        }
        amax = group_max(amax, red, sg, lane, nsg);
        const bool nonfinite = group_sum(bad, red + 32, sg, lane, nsg) > 0.0f;

        if (nonfinite) {
            const float qnan = as_type<float>(0x7FC00000u);
            for (uint k = tid; k < n; k += ntg) out_s[k] = qnan;
            if (kComputeVectors) {
                for (uint k = tid; k < M * n; k += ntg) out_u[k] = qnan;
                for (uint k = tid; k < n * N; k += ntg) out_vt[k] = qnan;
            }
            if (tid == 0) {
                info[mat] = kInfoNonFinite;
                if (kPart == 1) hw[2 * n + 1] = 1.0f;   // part 2 leaves this matrix alone
            }
            return;   // uniform: every thread has the same flag
        }
        // A power of two, so exact.
        if (amax > 0.0f) frexp(amax, expo);
        for (uint idx = tid; idx < m * n; idx += ntg) {
            const uint r = idx / n, k = idx - r * n;
            a[r * ld + k] = ldexp(a[r * ld + k], -expo);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ---------------------------------------------------------------------
        // 2. Bidiagonalize: B = Q Bd P^T (sgebd2, m >= n, upper)
        // ---------------------------------------------------------------------
        // Step k: the left reflector H(k) = I - tq v v^T zeroes column k below the
        // diagonal (v(k) = 1, its tail kept there for step 4), then the right one
        // G(k) = I - tp u u^T zeroes row k right of the superdiagonal (u(k+1) = 1,
        // its tail kept there for step 3).
        // Four barriers per step: the right reflector needs only row k, so the
        // thread owning it forms it as soon as its row is updated, and the right
        // update runs on into the next step's sum, whose barrier waits for it.
        for (uint k = 0; k < n; ++k) {
            // Left: the column sums v^T B(k:, j) are walked by the thread owning
            // column j; the update is then row-local.
            float t = 0.0f;
            if (wrow && i > k) { const float v = a[i * ld + k]; t = v * v; }
            const float sumsq = group_sum(t, red + 32 * (k & 1u), sg, lane, nsg);
            const float alpha = a[k * ld + k];
            float tv = 0.0f, beta = alpha;
            if (sumsq > 0.0f) {
                beta = -copysign(sqrt_nr(alpha * alpha + sumsq), alpha);
                tv = div_nr(beta - alpha, beta);
                const float scal = div_nr(1.0f, alpha - beta);
                if (wrow && i > k) a[i * ld + k] *= scal;
            }
            if (tid == 0) { d[k] = beta; tq[k] = tv; }
            if (k + 1 >= n) break;   // uniform; the phases after start with a barrier
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tv != 0.0f) {   // uniform
                // Column c > k, walked by a group of g lanes in two chains each:
                // the walk is latency-bound. Sized by B's rows alone, so that
                // singular values alone come out bit for bit as with vectors.
                const uint g = lanes_per_column(m, n - k - 1, m - k - 1);
                const uint c = k + 1 + i / g, lg = i % g;
                const bool active = row && c < n;
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
                if (active && lg == 0) wb[c] = tv * (s + a[k * ld + c]);
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (wrow && i >= k) {
                    const float vi = (i == k) ? 1.0f : a[i * ld + k];
                    for (uint j = k + 1; j < n; ++j) a[i * ld + j] -= vi * wb[j];
                }
            }

            // Right, by the thread owning row k, on its own row.
            if (wrow && i == k) {
                threadgroup float* w = a + k * ld;
                float s0 = 0.0f, s1 = 0.0f;   // two chains: the sum is on the critical path
                uint j = k + 2;
                for (; j + 1 < n; j += 2) { s0 += w[j] * w[j]; s1 += w[j + 1] * w[j + 1]; }
                if (j < n) s0 += w[j] * w[j];
                const float sumsq_r = s0 + s1;
                const float alpha_r = w[k + 1];
                float tu = 0.0f, beta_r = alpha_r;
                if (sumsq_r > 0.0f) {
                    beta_r = -copysign(sqrt_nr(alpha_r * alpha_r + sumsq_r), alpha_r);
                    tu = div_nr(beta_r - alpha_r, beta_r);
                    const float scal = div_nr(1.0f, alpha_r - beta_r);
                    for (uint c = k + 2; c < n; ++c) w[c] *= scal;
                }
                e[k] = beta_r;
                tp[k] = tu;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            // Every row below k, reading u from row k; the next step's sum waits
            // for it.
            const float tu = tp[k];
            if (tu != 0.0f && wrow && i > k) {
                threadgroup float* w = a + i * ld;
                threadgroup const float* u = a + k * ld;
                float s0 = w[k + 1], s1 = 0.0f;
                uint j = k + 2;
                for (; j + 1 < n; j += 2) { s0 += w[j] * u[j]; s1 += w[j + 1] * u[j + 1]; }
                if (j < n) s0 += w[j] * u[j];
                const float s = tu * (s0 + s1);
                w[k + 1] -= s;
                for (uint j = k + 2; j < n; ++j) w[j] -= s * u[j];
            }
        }
        if (tid == 0) e[n - 1] = 0.0f;

        if (kComputeVectors) {
            // -----------------------------------------------------------------
            // 3. V = P = G(0) G(1) ... G(n-3), in rows m .. m+n-1
            // -----------------------------------------------------------------
            // Accumulated backward from the identity: G(j) acts on rows and
            // columns j+1 .., with u read from row j of B, which step 4 has not
            // overwritten yet.
            if (row && i >= m) {
                const uint r = i - m;
                for (uint k = 0; k < n; ++k) vg[k * n + r] = (r == k) ? 1.0f : 0.0f;
            }
            threadgroup_barrier(both);
            for (int jj = (int)n - 3; jj >= 0; --jj) {
                const uint j = (uint)jj;
                const float tj = tp[j];
                if (tj == 0.0f) continue;   // uniform
                threadgroup const float* u = a + j * ld;
                // Column c > j of V (contiguous), by a group of lanes as in step 2.
                const uint g = lanes_per_column(rows, n - j - 1, n - j - 1);
                const uint c = j + 1 + i / g, lg = i % g;
                const bool active = row && c < n;
                float s = 0.0f;
                if (active) {
                    device const float* v = vg + c * n;
                    for (uint r = j + 1 + lg; r < n; r += g) s += (r == j + 1 ? 1.0f : u[r]) * v[r];
                }
                s = group_reduce(s, g);
                if (active && lg == 0) wb[c] = tj * s;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (row && i >= m + j + 1) {
                    const uint r = i - m;
                    const float ur = (r == j + 1) ? 1.0f : u[r];
                    for (uint k = j + 1; k < n; ++k) vg[k * n + r] -= ur * wb[k];
                }
                threadgroup_barrier(both);
            }

            // -----------------------------------------------------------------
            // 4. Q = H(0) H(1) ... H(n-1), its first n columns, in place (sorg2r)
            // -----------------------------------------------------------------
            // Backward: before step j, columns j+1 .. hold H(j+1) ... H(n-1) and
            // are zero in rows up to j; H(j)'s vector is read from column j, which
            // step j itself overwrites last.
            for (int jj = (int)n - 1; jj >= 0; --jj) {
                const uint j = (uint)jj;
                const float tj = tq[j];
                const bool apply = tj != 0.0f && j + 1 < n;   // uniform
                if (apply) {
                    // Column c > j, by a group of lanes as in step 2.
                    const uint g = lanes_per_column(rows, n - j - 1, m - j - 1);
                    const uint c = j + 1 + i / g, lg = i % g;
                    const bool active = row && c < n;
                    float s0 = 0.0f, s1 = 0.0f;
                    if (active) {
                        uint r = j + 1 + lg;
                        for (; r + g < m; r += 2 * g) {
                            s0 += a[r * ld + j] * a[r * ld + c];
                            s1 += a[(r + g) * ld + j] * a[(r + g) * ld + c];
                        }
                        if (r < m) s0 += a[r * ld + j] * a[r * ld + c];
                    }
                    const float s = group_reduce(s0 + s1, g);
                    if (active && lg == 0) wb[c] = tj * s;
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
                if (wrow) {
                    threadgroup float* w = a + i * ld;
                    if (i < j) {
                        w[j] = 0.0f;
                    } else if (i == j) {
                        if (apply) for (uint k = j + 1; k < n; ++k) w[k] = -wb[k];
                        w[j] = 1.0f - tj;
                    } else {
                        const float v = w[j];
                        if (apply) for (uint k = j + 1; k < n; ++k) w[k] -= v * wb[k];
                        w[j] = -tj * v;
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
    }   // kReduce

    if (kPart == 1) {
        // Hand d, e, the exponent and U to part 2. V is in its workspace.
        for (uint k = tid; k < n; k += ntg) { hw[k] = d[k]; hw[n + k] = e[k]; }
        if (tid == 0) { hw[2 * n] = as_type<float>(expo); hw[2 * n + 1] = 0.0f; }
        if (kComputeVectors) {
            for (uint idx = tid; idx < m * n; idx += ntg) {
                const uint c = idx / m, r = idx - c * m;
                ug[idx] = a[r * ld + c];
            }
        }
        return;
    }
    if (kPart == 2) {
        if (hw[2 * n + 1] != 0.0f) return;   // non-finite: part 1 wrote its output
        for (uint k = tid; k < n; k += ntg) { d[k] = hw[k]; e[k] = hw[n + k]; }
        expo = as_type<int>(hw[2 * n]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // -------------------------------------------------------------------------
    // 5. Implicit QR on (d, e); rotations applied to the rows of U and of V
    // -------------------------------------------------------------------------
    GkState st{n - 1, 0u, 0u, false, 0.0f};
    if (tid == 0) {
        float bnorm = 0.0f;
        for (uint k = 0; k < n; ++k) bnorm = fmax(bnorm, fabs(d[k]) + fabs(e[k]));
        st.dtol = FLT_EPSILON * bnorm;
    }
    if (kComputeVectors && !kOverlap) {
        // One thread chases, then every row applies: two barriers per step.
        threadgroup float* buf = x;
        while (true) {
            if (tid == 0) gk_step(d, e, buf, buf + n, buf + 2 * n, buf + 3 * n, ctl, st, prm.max_rots, true);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (ctl[0]) break;
            if (wrow && kReduce) apply_rotations(a + i * ld, ctl + kMetaU, buf, buf + n);
            else if (wrow)       apply_rotations(ug + i, m, ctl + kMetaU, buf, buf + n);
            else if (row)        apply_rotations(vg + (i - m), n, ctl + kMetaV, buf + 2 * n, buf + 3 * n);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    } else if (kComputeVectors) {
        // Two rotation buffers: while the rows apply step t from one, the
        // chaser computes step t+1 into the other. One barrier per step.
        threadgroup float* buf[2] = {x, x + 4 * n};
        if (tid == 0) gk_step(d, e, buf[0], buf[0] + n, buf[0] + 2 * n, buf[0] + 3 * n, ctl, st, prm.max_rots, true);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint slot = 0;
        while (ctl[slot * 16] == 0) {
            threadgroup uint* cur = ctl + slot * 16;
            if (tid == 0) {
                const uint next = slot ^ 1u;
                threadgroup float* nb = buf[next];
                gk_step(d, e, nb, nb + n, nb + 2 * n, nb + 3 * n, ctl + next * 16, st, prm.max_rots, true);
            } else if (wrow && kReduce) {
                apply_rotations(a + i * ld, cur + kMetaU, buf[slot], buf[slot] + n);
            } else if (wrow) {
                apply_rotations(ug + i, m, cur + kMetaU, buf[slot], buf[slot] + n);
            } else if (row) {
                apply_rotations(vg + (i - m), n, cur + kMetaV, buf[slot] + 2 * n, buf[slot] + 3 * n);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            slot ^= 1u;
        }
    } else {
        if (tid == 0) {
            do {
                gk_step(d, e, x, x, x, x, ctl, st, prm.max_rots, false);
            } while (ctl[0] == 0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // -------------------------------------------------------------------------
    // 6. Non-negative, sorted descending; write
    // -------------------------------------------------------------------------
    // A negative d[k] flips column k of V.
    if (kComputeVectors && row && i >= m) {
        device float* z = vg + (i - m);
        for (uint k = 0; k < n; ++k) if (d[k] < 0.0f) z[k * n] = -z[k * n];
    }
    for (uint q = tid; q < n; q += ntg) {
        const float dq = fabs(d[q]);
        uint rk = 0;
        for (uint k = 0; k < n; ++k) {
            const float dk = fabs(d[k]);
            rk += (dk > dq || (dk == dq && k < q)) ? 1u : 0u;
        }
        rank[rk] = q;
    }
    threadgroup_barrier(both);
    for (uint k = tid; k < n; k += ntg) out_s[k] = ldexp(fabs(d[rank[k]]), expo);
    if (kComputeVectors) {
        // B = U_B diag(s) V^T, U_B in rows 0 .. m-1: A = B, or A = B^T =
        // V diag(s) U_B^T if wide.
        for (uint idx = tid; idx < M * n; idx += ntg) {
            const uint r = idx / n, k = idx - r * n;
            out_u[idx] = prm.transpose ? vg[rank[k] * n + r]
                                       : (kReduce ? a[r * ld + rank[k]] : ug[rank[k] * m + r]);
        }
        for (uint idx = tid; idx < n * N; idx += ntg) {
            const uint k = idx / N, c = idx - k * N;
            out_vt[idx] = prm.transpose ? (kReduce ? a[c * ld + rank[k]] : ug[rank[k] * m + c])
                                        : vg[rank[k] * n + c];
        }
    }
    if (tid == 0) {   // the chaser, whose state this is
        const float smax = fabs(d[rank[0]]), smin = fabs(d[rank[n - 1]]);
        const bool deficient = !(smin > kNullEps * FLT_EPSILON * smax);
        info[mat] = min(st.steps, 0xFFFFu) | (st.failed ? 0u : kInfoConverged)
                  | (deficient ? kInfoRankDeficient : 0u);
    }
}

// =============================================================================
// In registers: a simdgroup a matrix (svd_gk_simd, since 2.17.0)
// =============================================================================
//
// For max(M, N) <= 32 the matrix, U and V fit in a simdgroup's registers, a
// row a lane, as Eigh_QL.metal's eigh_ql_simd keeps the eigensolver's. The
// kernel above holds B in threadgroup memory, a threadgroup a matrix sized for
// it, and its QR iteration is one thread's chain of dependent rotations: few
// matrices share a core to hide it. Here a matrix needs 1.3 KB of threadgroup
// memory (d, e, the reflectors' scalars, a step's rotations), several share a
// threadgroup, a simdgroup each, and nothing waits on a threadgroup barrier.
//
//   1. lane i loads row i of B (A, or A^T if wide); scan, scale
//   2. bidiagonalize (sgebd2, upper): column k's reflector from a simd_sum,
//      the columns right of it updated by a simd_sum each; row k's reflector
//      from lane k's registers, gathered by shuffles and formed on every lane
//      alike, the rows below updated row-locally. v kept in column k, u in
//      row k, for steps 3 and 4
//   3. with vectors, V = G(0) ... G(n-3), forward, a row a lane: row-local
//   4. with vectors, U = H(0) ... H(n-1) [I; 0], backward (sorg2r), a row a
//      lane: a simd_sum per column
//   5. with vectors, implicit QR: lane 0 runs gk_step, then every lane applies
//      its rotations to its rows of U and V; singular values alone by
//      bisection on the Golub-Kahan tridiagonal (zero diagonal, off-diagonal
//      d0, e0, d1, ..., d_{n-1}; its eigenvalues are +-s), a lane a value
//   6. with vectors, non-negative, rank sort descending, write
//
// A row is indexed by constants only, so that it stays in registers: the
// loops over a row are expanded by the preprocessor (GS_UNROLL).
#define GS_UNROLL_c(...) { { constexpr uint c = 0; __VA_ARGS__ } { constexpr uint c = 1; __VA_ARGS__ } { constexpr uint c = 2; __VA_ARGS__ } { constexpr uint c = 3; __VA_ARGS__ } { constexpr uint c = 4; __VA_ARGS__ } { constexpr uint c = 5; __VA_ARGS__ } { constexpr uint c = 6; __VA_ARGS__ } { constexpr uint c = 7; __VA_ARGS__ } { constexpr uint c = 8; __VA_ARGS__ } { constexpr uint c = 9; __VA_ARGS__ } { constexpr uint c = 10; __VA_ARGS__ } { constexpr uint c = 11; __VA_ARGS__ } { constexpr uint c = 12; __VA_ARGS__ } { constexpr uint c = 13; __VA_ARGS__ } { constexpr uint c = 14; __VA_ARGS__ } { constexpr uint c = 15; __VA_ARGS__ } { constexpr uint c = 16; __VA_ARGS__ } { constexpr uint c = 17; __VA_ARGS__ } { constexpr uint c = 18; __VA_ARGS__ } { constexpr uint c = 19; __VA_ARGS__ } { constexpr uint c = 20; __VA_ARGS__ } { constexpr uint c = 21; __VA_ARGS__ } { constexpr uint c = 22; __VA_ARGS__ } { constexpr uint c = 23; __VA_ARGS__ } { constexpr uint c = 24; __VA_ARGS__ } { constexpr uint c = 25; __VA_ARGS__ } { constexpr uint c = 26; __VA_ARGS__ } { constexpr uint c = 27; __VA_ARGS__ } { constexpr uint c = 28; __VA_ARGS__ } { constexpr uint c = 29; __VA_ARGS__ } { constexpr uint c = 30; __VA_ARGS__ } { constexpr uint c = 31; __VA_ARGS__ } }
#define GS_UNROLL_cd(...) { { constexpr uint c = 31; __VA_ARGS__ } { constexpr uint c = 30; __VA_ARGS__ } { constexpr uint c = 29; __VA_ARGS__ } { constexpr uint c = 28; __VA_ARGS__ } { constexpr uint c = 27; __VA_ARGS__ } { constexpr uint c = 26; __VA_ARGS__ } { constexpr uint c = 25; __VA_ARGS__ } { constexpr uint c = 24; __VA_ARGS__ } { constexpr uint c = 23; __VA_ARGS__ } { constexpr uint c = 22; __VA_ARGS__ } { constexpr uint c = 21; __VA_ARGS__ } { constexpr uint c = 20; __VA_ARGS__ } { constexpr uint c = 19; __VA_ARGS__ } { constexpr uint c = 18; __VA_ARGS__ } { constexpr uint c = 17; __VA_ARGS__ } { constexpr uint c = 16; __VA_ARGS__ } { constexpr uint c = 15; __VA_ARGS__ } { constexpr uint c = 14; __VA_ARGS__ } { constexpr uint c = 13; __VA_ARGS__ } { constexpr uint c = 12; __VA_ARGS__ } { constexpr uint c = 11; __VA_ARGS__ } { constexpr uint c = 10; __VA_ARGS__ } { constexpr uint c = 9; __VA_ARGS__ } { constexpr uint c = 8; __VA_ARGS__ } { constexpr uint c = 7; __VA_ARGS__ } { constexpr uint c = 6; __VA_ARGS__ } { constexpr uint c = 5; __VA_ARGS__ } { constexpr uint c = 4; __VA_ARGS__ } { constexpr uint c = 3; __VA_ARGS__ } { constexpr uint c = 2; __VA_ARGS__ } { constexpr uint c = 1; __VA_ARGS__ } { constexpr uint c = 0; __VA_ARGS__ } }
#define GS_UNROLL(N, v, ...) GS_UNROLL_##v(if (v < N) __VA_ARGS__)

// The rotations of one recorded list (see gk_step) applied to the row z, held
// in registers: as apply_rotations, every index a constant.
#define GS_APPLY(N, z, list, cb, sb)                                                                \
    {                                                                                               \
        const uint kind = (list)[0], a0 = (list)[1], cnt = (list)[2], fb = (list)[3];               \
        if (cnt != 0 && kind == 0) {                                                                \
            GS_UNROLL_c(if (c + 1 < N && c >= a0 && c < a0 + cnt) {                                 \
                const float cr = (cb)[c - a0], sr = (sb)[c - a0], t0 = z[c], t1 = z[c + 1];         \
                z[c] = fma(cr, t0, sr * t1);                                                        \
                z[c + 1] = fma(cr, t1, -sr * t0);                                                   \
            })                                                                                      \
        } else if (cnt != 0) {                                                                      \
            float carry = 0.0f;                                                                     \
            GS_UNROLL(N, c, { if (c == fb) carry = z[c]; })                                         \
            if (kind == 1) {                                                                        \
                GS_UNROLL_c(if (c < N && c >= a0 && c < a0 + cnt) {                                 \
                    const float cr = (cb)[c - a0], sr = (sb)[c - a0], t0 = z[c];                    \
                    z[c] = fma(cr, t0, sr * carry);                                                 \
                    carry = fma(cr, carry, -sr * t0);                                               \
                })                                                                                  \
            } else {                                                                                \
                GS_UNROLL_cd(if (c < N && c <= a0 && c + cnt > a0) {                                \
                    const float cr = (cb)[a0 - c], sr = (sb)[a0 - c], t0 = z[c];                    \
                    z[c] = fma(cr, t0, sr * carry);                                                 \
                    carry = fma(cr, carry, -sr * t0);                                               \
                })                                                                                  \
            }                                                                                       \
            GS_UNROLL(N, c, { if (c == fb) z[c] = carry; })                                         \
        }                                                                                           \
    }

// Must match `GksParams` in svd_golub_kahan.mm.
struct GksParams {
    uint m;           // rows of B, m >= n, at most 32
    uint n;           // columns of B: min(M, N), at most the instance's N
    uint transpose;   // 1: B = A^T (A is wide), 0: B = A
    uint max_rots;    // rotations allowed per matrix in total (LAPACK: 6 n^2)
    uint batch;       // matrices in this dispatch
};

// Threadgroup memory a simdgroup, in floats: d, e, tq, tp, then a step's
// rotations (uc, us, vc, vs), 32 each; meta (16 uints), pos and rank (32 each).
constant uint kGksFloats = 336;
// RUN: d, e, tq, tp, two rotation buffers (128 each), their metas (16 uints
// each), pos, rank, and the runner's count of steps and failure
constant uint kGksFloatsRun = 488;

// Sums and maxima over a group of G lanes (G = 8, 16 or 32, aligned), on
// every lane of it: a simdgroup holds 32 / G matrices, a group each.
template <uint G> inline float lanes_sum(float x) {
    if (G == 32) return simd_sum(x);
    for (ushort off = G / 2; off > 0; off >>= 1) x += simd_shuffle_xor(x, off);
    return x;
}
template <uint G> inline float lanes_max(float x) {
    if (G == 32) return simd_max(x);
    for (ushort off = G / 2; off > 0; off >>= 1) x = fmax(x, simd_shuffle_xor(x, off));
    return x;
}

// G (the lanes a matrix takes, at least its rows) of 8 and 16 pack four and
// two matrices into a simdgroup, as Eigh_QL.metal's eigh_ql_simd does: each
// group's first lane runs its matrix's QR iteration alongside the others',
// every cross-lane step stays inside a group, and the iteration's loop ends
// once every group's has.
// RUN (G = 32, with vectors, since 2.17.0): from 17 rows a matrix fills a
// simdgroup, and its QR iteration, one lane's chain of dependent work, is
// most of the kernel while 31 lanes wait. Here the threadgroup's last
// simdgroup is a runner: lane j runs matrix j's iteration, the matrices side
// by side, and computes step t + 1 into a second buffer while the other
// simdgroups, a matrix each, apply step t; a threadgroup barrier a step.
template <uint N, uint G, bool RUN>
kernel void svd_gk_simd(
    device const float* A_in [[buffer(0)]],   // [batch, M, N] input (row-major)
    device float*       S    [[buffer(1)]],   // [batch, n] singular values, descending
    device float*       U    [[buffer(2)]],   // [batch, M, n] (kComputeVectors)
    device float*       Vt   [[buffer(3)]],   // [batch, n, N] (kComputeVectors)
    device uint*        info [[buffer(4)]],   // [batch] steps | flags
    constant GksParams& prm  [[buffer(5)]],
    threadgroup float*  tg   [[threadgroup(0)]],
    uint tgi  [[threadgroup_position_in_grid]],
    uint sg   [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint nsg  [[simdgroups_per_threadgroup]])
{
    constexpr uint P = 32 / G;                         // matrices a simdgroup, a group of G lanes each
    constexpr uint F = RUN ? kGksFloatsRun : kGksFloats;
    const uint i = lane % G, gb = lane - i, slot = lane / G;
    const bool runner = RUN && sg + 1 == nsg;          // RUN: the last simdgroup runs the QR iterations
    const uint per_tg = RUN ? nsg - 1 : nsg * P;       // matrices a threadgroup
    const uint mloc = RUN ? (runner ? 0u : sg) : sg * P + slot;
    const uint mat = tgi * per_tg + mloc;
    if (!RUN && (tgi * nsg + sg) * P >= prm.batch) return;   // the whole simdgroup; nothing below waits on other simdgroups
    const uint m = prm.m, n = prm.n;
    const uint nl = runner ? 0u : n;                   // the runner takes no part in steps 1-4
    const bool active = !runner && mat < prm.batch;
    threadgroup float* d   = tg + mloc * F;
    threadgroup float* e   = d + 32;
    threadgroup float* tq  = d + 64;
    threadgroup float* tp  = d + 96;
    threadgroup float* rb  = d + 128;
    threadgroup uint*  ctl = reinterpret_cast<threadgroup uint*>(d + (RUN ? 384 : 256));
    threadgroup uint*  pos = reinterpret_cast<threadgroup uint*>(d + (RUN ? 416 : 272));
    threadgroup uint*  rank = reinterpret_cast<threadgroup uint*>(d + (RUN ? 448 : 304));

    const uint M = prm.transpose ? n : m, NC = prm.transpose ? m : n;
    device const float* src = A_in + (ulong)mat * m * n;
    device float* out_s  = S + (ulong)mat * n;
    device float* out_u  = U + (ulong)mat * M * n;
    device float* out_vt = Vt + (ulong)mat * n * NC;

    // 1. Load row i of B, scan, scale
    float b[N];
    float amax = 0.0f, bad = 0.0f;
    GS_UNROLL(N, c, {
        float v = 0.0f;
        if (active && i < m && c < n) v = prm.transpose ? src[c * m + i] : src[i * n + c];
        b[c] = v;
        amax = fmax(amax, fabs(v));   // fmax skips NaN, hence the separate flag
        if (non_finite(v)) bad = 1.0f;
    })
    amax = lanes_max<G>(amax);
    // A group with a non-finite matrix writes NaN and carries on with zeros,
    // so that the other groups' steps stay in step
    const bool live = active && lanes_max<G>(bad) == 0.0f;
    if (active && !live) {
        const float qnan = as_type<float>(0x7FC00000u);
        if (i < n) out_s[i] = qnan;
        if (kComputeVectors) {
            for (uint k = i; k < M * n; k += G) out_u[k] = qnan;
            for (uint k = i; k < n * NC; k += G) out_vt[k] = qnan;
        }
        if (i == 0) info[mat] = kInfoNonFinite;
    }
    const bool wrow = live && i < m;   // a row of B, then of U
    const bool vrow = live && i < n;   // a row of V
    int expo = 0;
    if (live && amax > 0.0f) frexp(amax, expo);
    GS_UNROLL(N, c, { b[c] = live ? ldexp(b[c], -expo) : 0.0f; })

    // 2. Bidiagonalize: B = Q Bd P^T
    for (uint k = 0; k < nl; ++k) {
        // Left: H(k) zeroes column k below the diagonal
        float x = 0.0f;   // B(i, k)
        GS_UNROLL(N, c, { if (c == k) x = b[c]; })
        const bool below = wrow && i > k;
        const float sumsq = lanes_sum<G>(below ? x * x : 0.0f);
        const float alpha = simd_shuffle(x, (ushort)(gb + k));
        float tv = 0.0f, beta = alpha, v = i == k ? 1.0f : 0.0f;
        if (sumsq > 0.0f) {
            beta = -copysign(sqrt_nr(alpha * alpha + sumsq), alpha);
            tv = div_nr(beta - alpha, beta);
            if (below) v = x * div_nr(1.0f, alpha - beta);
        }
        if (below) GS_UNROLL(N, c, { if (c == k) b[c] = v; })   // kept for step 4
        if (i == 0) {
            d[k] = beta;
            tq[k] = tv;
        }
        if (k + 1 >= n) break;
        if (tv != 0.0f) {   // uniform
            // B(k:, c) -= tv v (v^T B(k:, c)), c > k: a simd_sum a column
            const float va = wrow && i >= k ? v : 0.0f;
            GS_UNROLL(N, c, {
                if (c > k && c < n) {
                    const float s = lanes_sum<G>(va * b[c]);
                    b[c] = fma(-tv * s, va, b[c]);
                }
            })
        }
        // Right: G(k) zeroes row k right of the superdiagonal; row k gathered
        // from lane k, the reflector formed on every lane alike
        float uk[N];
        GS_UNROLL(N, c, { uk[c] = simd_shuffle(b[c], (ushort)(gb + k)); })
        float alpha_r = 0.0f, ss = 0.0f;
        GS_UNROLL(N, c, {
            if (c == k + 1) alpha_r = uk[c];
            if (c >= k + 2 && c < n) ss = fma(uk[c], uk[c], ss);
        })
        float tu = 0.0f, beta_r = alpha_r, scal = 0.0f;
        if (ss > 0.0f) {
            beta_r = -copysign(sqrt_nr(alpha_r * alpha_r + ss), alpha_r);
            tu = div_nr(beta_r - alpha_r, beta_r);
            scal = div_nr(1.0f, alpha_r - beta_r);
        }
        GS_UNROLL(N, c, { uk[c] = c == k + 1 ? 1.0f : (c >= k + 2 && c < n ? uk[c] * scal : 0.0f); })
        if (i == k) GS_UNROLL(N, c, { if (c >= k + 2) b[c] = uk[c]; })   // kept for step 3
        if (i == 0) {
            e[k] = beta_r;
            tp[k] = tu;
        }
        if (tu != 0.0f && wrow && i > k) {
            float s = 0.0f;
            GS_UNROLL(N, c, { s = fma(b[c], uk[c], s); })
            s *= tu;
            GS_UNROLL(N, c, { b[c] = fma(-s, uk[c], b[c]); })
        }
    }
    if (i == 0 && !runner) e[n - 1] = 0.0f;
    simdgroup_barrier(mem_flags::mem_threadgroup);

    if (!kComputeVectors) {
        // 5. By bisection, lane i the i-th largest: the count of the Golub-Kahan
        // tridiagonal's eigenvalues below x > 0 is n plus the count of singular
        // values below it (Sturm, its pivots L D L^T), on [0, Gershgorin's
        // bound], halved until float32 can halve it no more.
        float g = 0.0f;
        if (vrow) g = fabs(d[i]) + fmax(fabs(e[i]), i > 0 ? fabs(e[i - 1]) : 0.0f);
        float lo = 0.0f, hi = lanes_max<G>(g);
        hi += 2.0f * FLT_EPSILON * hi + FLT_MIN;
        const float pivmin = FLT_MIN * fmax(1.0f, hi * hi);
        float s = 0.0f;
        if (vrow) {
            const uint target = n - 1 - i;   // the i-th largest is the target-th smallest
            for (uint pass = 0; pass < 64; ++pass) {
                const float mid = 0.5f * (lo + hi);
                if (mid <= lo || mid >= hi) break;   // float32 can halve it no more
                float q = -mid;
                if (fabs(q) < pivmin) q = -pivmin;
                uint neg = q < 0.0f ? 1u : 0u;
                for (uint k = 0; k < n; ++k) {
                    const float dk = d[k];
                    float t = -mid - div_nr(dk * dk, q);
                    if (fabs(t) < pivmin) t = -pivmin;
                    q = t;
                    neg += t < 0.0f ? 1u : 0u;
                    if (k + 1 < n) {
                        const float ek = e[k];
                        t = -mid - div_nr(ek * ek, q);
                        if (fabs(t) < pivmin) t = -pivmin;
                        q = t;
                        neg += t < 0.0f ? 1u : 0u;
                    }
                }
                if ((int)neg - (int)n > (int)target) hi = mid;
                else                  lo = mid;
            }
            s = lo == 0.0f ? 0.0f : 0.5f * (lo + hi);
            out_s[i] = ldexp(s, expo);
        }
        const float smax = simd_shuffle(s, (ushort)gb), smin = simd_shuffle(s, (ushort)(gb + n - 1));
        if (live && i == 0) {
            const bool deficient = !(smin > kNullEps * FLT_EPSILON * smax);
            info[mat] = 1u | kInfoConverged | (deficient ? kInfoRankDeficient : 0u);
        }
        return;
    }

    // 3. V = G(0) G(1) ... G(n-3), forward: Z <- Z G(k), a row a lane, u
    // gathered from row k (lane k)
    float z[N];
    GS_UNROLL(N, c, { z[c] = c == i ? 1.0f : 0.0f; })
    for (uint k = 0; k + 2 < nl; ++k) {
        const float tk = tp[k];
        if (tk == 0.0f) continue;   // uniform
        float uk[N];
        GS_UNROLL(N, c, {
            const float y = simd_shuffle(b[c], (ushort)(gb + k));
            uk[c] = c == k + 1 ? 1.0f : (c >= k + 2 ? y : 0.0f);
        })
        float dot = 0.0f;
        GS_UNROLL(N, c, { dot = fma(z[c], uk[c], dot); })
        dot *= tk;
        GS_UNROLL(N, c, { z[c] = fma(-dot, uk[c], z[c]); })
    }

    // 4. U = H(0) H(1) ... H(n-1) [I; 0], backward (sorg2r): before step k
    // the columns before k are still the identity's, which H(k) leaves alone
    float x[N];
    GS_UNROLL(N, c, { x[c] = wrow && c == i ? 1.0f : 0.0f; })
    for (uint k = nl; k-- > 0;) {
        const float tk = tq[k];
        if (tk == 0.0f) continue;   // uniform
        float v = i == k ? 1.0f : 0.0f;
        if (wrow && i > k) GS_UNROLL(N, c, { if (c == k) v = b[c]; })
        GS_UNROLL(N, c, {
            if (c >= k && c < n) {
                const float s = lanes_sum<G>(v * x[c]);
                x[c] = fma(-tk * s, v, x[c]);
            }
        })
    }

    // 5. Implicit QR on (d, e): lane 0 runs a step and records its rotations,
    // then every lane applies them to its rows of U and V
    GkState st{n - 1, 0u, 0u, false, 0.0f};
    if (!RUN) {
        if (i == 0) {
            float bnorm = 0.0f;
            for (uint k = 0; k < n; ++k) bnorm = fmax(bnorm, fabs(d[k]) + fabs(e[k]));
            st.dtol = FLT_EPSILON * bnorm;
        }
        threadgroup float* uc = rb;
        threadgroup float* us = rb + 32;
        threadgroup float* vc = rb + 64;
        threadgroup float* vs = rb + 96;
        if (i == 0) ctl[0] = live ? 0u : 1u;   // a group with nothing to do is done
        while (true) {
            if (i == 0 && ctl[0] == 0) gk_step(d, e, uc, us, vc, vs, ctl, st, prm.max_rots, true);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            const bool done = ctl[0] != 0;
            if (simd_all(done)) break;
            if (!done) {
                GS_APPLY(N, x, ctl + kMetaU, uc, us)
                GS_APPLY(N, z, ctl + kMetaV, vc, vs)
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    } else {
        // Matrix j's two rotation buffers at rb + 128 s and their metas at
        // ctl + 16 s (s = 0, 1); the runner's lane j works on matrix j
        if (!runner && i == 0) ctl[0] = live ? 0u : 1u;   // a matrix with nothing to do is done
        threadgroup_barrier(mem_flags::mem_threadgroup);   // every matrix's d, e and start
        const uint nm = per_tg;
        const bool runs = runner && lane < nm;
        threadgroup float* dr = tg + min(lane, nm - 1) * F;
        threadgroup uint*  cr = reinterpret_cast<threadgroup uint*>(dr + 384);
        if (runs) {
            float bnorm = 0.0f;
            for (uint k = 0; k < n; ++k) bnorm = fmax(bnorm, fabs(dr[k]) + fabs(dr[32 + k]));
            st.dtol = FLT_EPSILON * bnorm;
            if (cr[0] == 0) gk_step(dr, dr + 32, dr + 128, dr + 160, dr + 192, dr + 224, cr, st, prm.max_rots, true);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint sl = 0;
        while (true) {
            bool all = true;   // the same in every thread
            for (uint j = 0; j < nm; ++j)
                all = all && reinterpret_cast<threadgroup uint*>(tg + j * F + 384 + 16 * sl)[0] != 0;
            if (all) break;
            if (runs) {
                threadgroup uint* nxt = cr + 16 * (sl ^ 1u);
                threadgroup float* nb = dr + 128 + 128 * (sl ^ 1u);
                if (cr[16 * sl] == 0) gk_step(dr, dr + 32, nb, nb + 32, nb + 64, nb + 96, nxt, st, prm.max_rots, true);
                else nxt[0] = 1u;
            } else if (!runner) {
                threadgroup uint* cur = ctl + 16 * sl;
                if (cur[0] == 0) {
                    threadgroup float* cb = rb + 128 * sl;
                    GS_APPLY(N, x, cur + kMetaU, cb, cb + 32)
                    GS_APPLY(N, z, cur + kMetaV, cb + 64, cb + 96)
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            sl ^= 1u;
        }
        if (runs) {
            threadgroup uint* stat = reinterpret_cast<threadgroup uint*>(dr + 480);
            stat[0] = st.steps;
            stat[1] = st.failed ? 1u : 0u;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (!runner) {   // the info word's: the runner's state
            threadgroup uint* stat = reinterpret_cast<threadgroup uint*>(d + 480);
            st.steps = stat[0];
            st.failed = stat[1] != 0;
        }
    }

    // 6. Non-negative (a negative d[c] flips column c of V), sorted
    // descending (pos[c]: where column c goes; rank[k]: which column is k-th);
    // write
    GS_UNROLL(N, c, { if (c < n && d[c] < 0.0f) z[c] = -z[c]; })
    if (vrow) {
        const float di = fabs(d[i]);
        uint rk = 0;
        for (uint k = 0; k < n; ++k) {
            const float dk = fabs(d[k]);
            rk += (dk > di || (dk == di && k < i)) ? 1u : 0u;
        }
        pos[i] = rk;
        rank[rk] = i;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    if (vrow) out_s[i] = ldexp(fabs(d[rank[i]]), expo);
    // B = U_B diag(s) V^T: A = B, or A = B^T = V diag(s) U_B^T if wide
    if (!prm.transpose) {
        if (wrow) GS_UNROLL(N, c, { if (c < n) out_u[i * n + pos[c]] = x[c]; })
        if (vrow) GS_UNROLL(N, c, { if (c < n) out_vt[pos[c] * n + i] = z[c]; })
    } else {
        if (vrow) GS_UNROLL(N, c, { if (c < n) out_u[i * n + pos[c]] = z[c]; })
        if (wrow) GS_UNROLL(N, c, { if (c < n) out_vt[pos[c] * m + i] = x[c]; })
    }
    if (live && i == 0) {   // the chaser, whose state this is
        const float smax = fabs(d[rank[0]]), smin = fabs(d[rank[n - 1]]);
        const bool deficient = !(smin > kNullEps * FLT_EPSILON * smax);
        info[mat] = min(st.steps, 0xFFFFu) | (st.failed ? 0u : kInfoConverged)
                  | (deficient ? kInfoRankDeficient : 0u);
    }
}

#define GS_INSTANCE(N, G, RUN, NAME)                                                                 \
    template [[host_name(NAME)]] kernel void svd_gk_simd<N, G, RUN>(                                 \
        device const float*, device float*, device float*, device float*, device uint*,             \
        constant GksParams&, threadgroup float*, uint, uint, uint, uint);
GS_INSTANCE(8, 8, false, "svd_gk_simd_8_8")
GS_INSTANCE(8, 16, false, "svd_gk_simd_8_16")
GS_INSTANCE(8, 32, false, "svd_gk_simd_8_32")
GS_INSTANCE(16, 16, false, "svd_gk_simd_16_16")
GS_INSTANCE(16, 32, false, "svd_gk_simd_16_32")
GS_INSTANCE(32, 32, false, "svd_gk_simd_32_32")
GS_INSTANCE(8, 32, true, "svd_gk_simd_8_32_run")
GS_INSTANCE(16, 32, true, "svd_gk_simd_16_32_run")
GS_INSTANCE(32, 32, true, "svd_gk_simd_32_32_run")
