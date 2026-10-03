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
