#include <metal_stdlib>
using namespace metal;

// =============================================================================
// Householder tridiagonalization on the GPU: the device half of the `tridiag`
// eigensolver backend (src/eigh_tridiag.mm)
// =============================================================================
//
// For one large matrix the Jacobi backends lose to LAPACK, whose ssyevd spends
// most of its time tridiagonalizing: half of that is a symmetric matrix-vector
// product per column, bound by memory bandwidth. This is LAPACK's blocked
// ssytrd (lower) with every step on the GPU, so the bandwidth is the GPU's:
// per column, slatrd's steps as three kernels launched back to back, and per
// panel of nb columns, the rank-2nb trailing update as one GEMM (MPS). The
// host waits once per matrix, not once per column: a GPU round trip costs
// ~0.13 ms, n of them more than the whole reduction below N ~ 3000.
//
// The matrix is column-major with leading dimension lda and only its lower
// triangle is read: the trailing update writes both triangles, but nothing
// reads the upper one. Each kernel addresses the trailing matrix Ak = A(k:, k:)
// of the current panel (bound at the buffer offset of A(k, k)), and the panel's
// W (column-major, ldw), whose row 0 is Ak's.
//
// Per column i of the panel (len = nn - i, lr = len - 1), slatrd's steps in
// three kernels, each step that needs a whole vector done before the next
// starts at a kernel boundary (a dispatch costs a few microseconds of GPU time
// even when it does nothing, and a column used to take seven):
//   td_update   W(i:, i-1) += -tau/2 (W . v) v, finishing the previous column
//               from td_apply's dot partials; then
//               Ak(i:, i) -= Ak(i:, 0:i) W(i, 0:i)^T + W(i:, 0:i) Ak(i, 0:i)^T,
//               and per-threadgroup partials of the norm of Ak(i+2:, i)
//   td_symv     the reflector annihilating Ak(i+2:, i) from those partials
//               (d, e, tau as ssytrd; every threadgroup computes it), then
//               W(i+1:, i) = Ak(i+1:, i+1:) v in 64 x 64 tiles of the lower
//               triangle, and in further threadgroups W(i+1:, 0:i)^T v and
//               Ak(i+1:, 0:i)^T v
//   td_apply    sums the tiles' partials and applies slatrd's corrections:
//               W(i+1:, i) = tau (W - Ak(i+1:, 0:i) . - W(i+1:, 0:i) .); writes
//               v back into Ak(i+1:, i); per-threadgroup partials of W . v
// and per panel:
//   td_pack     finishes the panel's last column, B = [V W], C = [W V]; then
//               A22 -= B C^T (MPS GEMM)
//   td_restore_e  Ak(j+1, j) = e(j) (the reflector's unit was stored there)
//
// The symmetric product reads the trailing matrix as of the panel's start,
// which slatrd's corrections account for. Every sum over threadgroups is
// taken in a fixed order, so results do not depend on scheduling.
//
// A batch of matrices of one order is reduced by the same dispatches, the
// grid's second (td_pack's third) dimension the matrix: each kernel steps its
// buffers by the per-matrix strides in its parameters (since 2.17.0). Every
// dispatch of a small matrix's reduction is mostly the GPU's fixed cost, so
// a batch pays it once rather than once a matrix.

constant constexpr uint TILE  = 64;    // td_symv's tiles
constant constexpr uint GROUP = 256;   // threads per threadgroup of td_update and td_apply
constant constexpr uint LANES = 8;     // their threads per row: each sums every LANES-th term

// The sum over a row's LANES consecutive lanes, in a fixed order; every lane
// gets it.
static float row_sum(float x) {
    x += simd_shuffle_xor(x, 4);
    x += simd_shuffle_xor(x, 2);
    x += simd_shuffle_xor(x, 1);
    return x;
}

struct TdParams {
    uint nn;      // order of Ak
    uint lda;
    uint ldw;
    uint i;       // column within the panel
    uint k;       // the panel's first column, for d, e, tau
    uint ng;      // td_update's threadgroups for column i: its norm partials
    uint ngp;     // td_apply's threadgroups for column i - 1: its dot partials
    uint tiles;   // td_symv's tile threadgroups; the dot products' come after
    // Per-matrix strides, in floats: A, W, P, the partials (red), d e tau, tmp
    uint sa, sw, sp, sr, sv, stm;
};

// tau/2 (W(:, i-1) . v) from td_apply's partials, the same in every simdgroup.
static float finish_coeff(device const float* dpart, uint ngp, float tau, uint lane) {
    float q = 0.0f;
    for (uint u = lane; u < ngp; u += 32) q += dpart[u];
    return -0.5f * tau * simd_sum(q);
}

// x / y and sqrt(x) to about an ulp: the fast approximations and a Newton
// step. This file is built with -fno-fast-math, which makes `/` and sqrt()
// the IEEE sequences, and a kernel with any of them in it compiles all of
// its arithmetic in IEEE mode (an untaken sqrt() was enough). In the kernels
// whose steps are chains of dependent scalar work (a reflector a column,
// computed in every threadgroup; bisection's Sturm counts) that cost from 5% to a
// third of the time on an M5 Pro,
// so those kernels use these and nothing IEEE. div1's y is never zero; sqrt1
// returns x for x <= 0 (and NaN).
__attribute__((always_inline)) static float div1(float x, float y) {
    const float r = fast::divide(1.0f, y), q = x * r;
    return fma(fma(-y, q, x), r, q);
}
__attribute__((always_inline)) static float sqrt1(float x) {
    if (!(x > 0.0f)) return x;
    const float s = fast::sqrt(x);
    return fma(fma(-s, s, x), fast::divide(0.5f, s), s);
}

// A Householder reflector from alpha and the norm of the rest, as LAPACK's
// slarfg: beta (alpha's replacement), tau, and the rest's scale
// 1 / (alpha - beta); tau = 0 for a zero rest.
__attribute__((always_inline)) static void householder(float alpha, float xnorm, thread float& beta,
                                                        thread float& tau, thread float& scale) {
    beta = alpha;
    tau = 0.0f;
    scale = 1.0f;
    if (xnorm != 0.0f) {
        const float big = max(fabs(alpha), xnorm), rb = div1(1.0f, big), ra = alpha * rb, rx = xnorm * rb;
        beta = -copysign(big * sqrt1(ra * ra + rx * rx), alpha);
        tau = div1(beta - alpha, beta);
        scale = div1(1.0f, alpha - beta);
    }
}

struct Reflector { float beta, tau, scale; };

// As LAPACK's slarfg, from td_update's (max, sum of squares / max^2) partials
// of x = Ak(i+2:, i) and alpha = Ak(i+1, i); the same in every simdgroup.
// The norm is scaled by the largest magnitude, so it neither over- nor
// underflows.
static Reflector reflector(device const float* npart, uint ng, float alpha, uint lane) {
    float m = 0.0f, ss = 0.0f;
    for (uint u = lane; u < ng; u += 32) {
        const float pm = npart[2 * u], ps = npart[2 * u + 1];
        if (pm > m) { const float f = div1(m, pm); ss = ss * f * f + ps; m = pm; }
        else if (pm > 0.0f) { const float f = div1(pm, m); ss += ps * f * f; }
    }
    const float amax = simd_max(m);
    const float f = amax > 0.0f ? div1(m, amax) : 0.0f;
    const float xnorm = amax * sqrt1(simd_sum(ss * f * f));
    Reflector r;
    householder(alpha, xnorm, r.beta, r.tau, r.scale);
    return r;
}

// LANES threads per row i + r of Ak(i:, i), GROUP / LANES rows per threadgroup. Only
// a row's threads write W(row, i-1) and Ak(row, i); W(i, i-1), which every row
// reads, is finished on the fly by each and never stored (nothing reads it
// later).
kernel void td_update(device float* Ak [[buffer(0)]], device float* W [[buffer(1)]],
                      device const float* tau [[buffer(2)]], device const float* dpart [[buffer(3)]],
                      device float* npart [[buffer(4)]], constant TdParams& p [[buffer(5)]],
                      uint2 gid2 [[thread_position_in_grid]], uint2 g2 [[threadgroup_position_in_grid]],
                      uint t [[thread_index_in_threadgroup]],
                      uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float part[GROUP / 32];
    const uint gid = gid2.x, g = g2.x;
    const ulong mat = g2.y;
    Ak += mat * p.sa; W += mat * p.sw; tau += mat * p.sv; dpart += mat * p.sr; npart += mat * p.sr;
    const uint i = p.i, row = i + gid / LANES, q = gid % LANES, lda = p.lda, ldw = p.ldw;
    float x = 0.0f;
    if (i > 0) {
        const float a = finish_coeff(dpart, p.ngp, tau[p.k + i - 1], lane);
        float acc = 0.0f;
        const uint c = i - 1;
        if (row < p.nn)
            for (uint j = q; j < c; j += LANES)
                acc += Ak[row + j * lda] * W[i + j * ldw] + W[row + j * ldw] * Ak[i + j * lda];
        acc = row_sum(acc);
        if (row < p.nn && q == 0) {
            const float wi = W[i + c * ldw] + a * Ak[i + c * lda];   // Ak(i, i-1) = v(0) = 1
            float wr = wi;
            if (row > i) {
                wr = W[row + c * ldw] + a * Ak[row + c * lda];
                W[row + c * ldw] = wr;
            }
            acc += Ak[row + c * lda] * wi + wr * Ak[i + c * lda];
            x = Ak[row + i * lda] - acc;
            Ak[row + i * lda] = x;
        }
    } else if (row < p.nn && q == 0) {
        x = Ak[row + i * lda];
    }

    // The norm partials of Ak(i+2:, i): the largest magnitude, and the sum of
    // squares relative to it (each row once, from its first lane).
    const float ax = q == 0 && row >= i + 2 && row < p.nn ? fabs(x) : 0.0f;
    const float m = simd_max(ax);
    if (lane == 0) part[sg] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float gm = 0.0f;
    for (uint u = 0; u < GROUP / 32; ++u) gm = max(gm, part[u]);
    const float z = gm > 0.0f ? ax * div1(1.0f, gm) : 0.0f;
    const float s = simd_sum(z * z);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) part[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t == 0) {
        float ss = 0.0f;
        for (uint u = 0; u < GROUP / 32; ++u) ss += part[u];
        npart[2 * g]     = gm;
        npart[2 * g + 1] = ss;
    }
}

// y = M v for the symmetric trailing matrix M = Ak(i+1:, i+1:) (m = lr),
// reading its lower triangle only. One threadgroup (256 threads) per 64 x 64
// tile (bi, bj), bi >= bj, numbered row by row:
//   rows part     y[block bi] += T v[block bj]       (on the diagonal tile: r >= c)
//   columns part  y[block bj] += T^T v[block bi]     (on the diagonal tile: r > c)
// Partials go to P[slot * m + row]: block b gets slot s <= b from tile (b, s)'s
// rows part and slot s > b from tile (s, b)'s columns part, so each (block,
// slot) is written exactly once; the diagonal tile adds its two parts first.
// Threadgroups from p.tiles on take the dot products: tmp[t] = W(i+1:, t)^T v
// and tmp[i + t] = Ak(i+1:, t)^T v, t < i. v is formed on the fly, v(0) = 1,
// v(r) = scale Ak(i+1+r, i); td_apply writes it back.
kernel void td_symv(device const float* Ak [[buffer(0)]], device const float* W [[buffer(1)]],
                    device float* P [[buffer(2)]], device float* tmp [[buffer(3)]],
                    device const float* npart [[buffer(4)]], device float* d [[buffer(5)]],
                    device float* e [[buffer(6)]], device float* tau [[buffer(7)]],
                    device float* scal [[buffer(8)]], constant TdParams& p [[buffer(9)]],
                    uint2 g2 [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]],
                    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    const uint g = g2.x;
    const ulong mat = g2.y;
    Ak += mat * p.sa; W += mat * p.sw; P += mat * p.sp; tmp += mat * p.stm; npart += mat * p.sr;
    d += mat * p.sv; e += mat * p.sv; tau += mat * p.sv; scal += mat * p.sr;
    const uint i = p.i, m = p.nn - i - 1, lda = p.lda;
    device const float* x = Ak + i * lda + i + 1;          // Ak(i+1:, i): alpha, then the rest
    const Reflector rf = reflector(npart, p.ng, x[0], lane);
    if (g == 0 && t == 0) {
        d[p.k + i]   = Ak[i + i * lda];
        e[p.k + i]   = rf.beta;
        tau[p.k + i] = rf.tau;
        scal[0]      = rf.scale;
    }

    if (g >= p.tiles) {                                   // a dot product
        threadgroup float part[8];
        const uint q = g - p.tiles, j = q < i ? q : q - i;
        device const float* c = q < i ? W + j * p.ldw + i + 1 : Ak + j * lda + i + 1;
        float s = 0.0f;
        for (uint r = t; r < m; r += 256) s = fma(c[r], r == 0 ? 1.0f : rf.scale * x[r], s);
        s = simd_sum(s);
        if (lane == 0) part[sg] = s;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (t == 0) {
            float v = 0.0f;
            for (uint u = 0; u < 8; ++u) v += part[u];
            tmp[q] = v;
        }
        return;
    }

    uint bi = (uint)((fast::sqrt(8.0f * (float)g + 1.0f) - 1.0f) * 0.5f);   // corrected below
    while (bi * (bi + 1) / 2 > g) --bi;
    while ((bi + 1) * (bi + 2) / 2 <= g) ++bi;
    const uint bj = g - bi * (bi + 1) / 2;
    device const float* A = Ak + (i + 1) * lda + i + 1;
    threadgroup float xi[TILE], xj[TILE], colsum[TILE], rows[8][TILE];
    if (t < TILE) {
        const uint r = bi * TILE + t;
        xi[t] = r < m ? (r == 0 ? 1.0f : rf.scale * x[r]) : 0.0f;
        const uint c = bj * TILE + t;
        xj[t] = c < m ? (c == 0 ? 1.0f : rf.scale * x[c]) : 0.0f;
        colsum[t] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const bool diag = bi == bj;
    const uint r0 = lane, r1 = lane + 32, g0 = bi * TILE + r0, g1 = bi * TILE + r1;
    float acc0 = 0.0f, acc1 = 0.0f;
    for (uint cc = 0; cc < 8; ++cc) {                 // this simdgroup's 8 of the 64 columns
        const uint c = sg * 8 + cc, gc = bj * TILE + c;
        if (gc >= m) break;
        device const float* col = A + (ulong)gc * lda + bi * TILE;
        const float a0 = (g0 < m && (!diag || r0 >= c)) ? col[r0] : 0.0f;
        const float a1 = (g1 < m && (!diag || r1 >= c)) ? col[r1] : 0.0f;
        acc0 = fma(a0, xj[c], acc0);
        acc1 = fma(a1, xj[c], acc1);
        const float s0 = (!diag || r0 > c) ? a0 : 0.0f;
        const float s1 = (!diag || r1 > c) ? a1 : 0.0f;
        const float u = simd_sum(s0 * xi[r0] + s1 * xi[r1]);
        if (lane == 0) {
            if (diag) colsum[c] = u;
            else      P[(ulong)bi * m + gc] = u;      // slot bi of block bj
        }
    }
    rows[sg][r0] = acc0;
    rows[sg][r1] = acc1;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t < TILE) {
        const uint gr = bi * TILE + t;
        if (gr < m) {
            float v = 0.0f;
            for (uint q = 0; q < 8; ++q) v += rows[q][t];
            if (diag) v += colsum[t];
            P[(ulong)bj * m + gr] = v;                // slot bj of block bi
        }
    }
}

// LANES threads per row i + 1 + r, GROUP / LANES rows per threadgroup.
kernel void td_apply(device float* Ak [[buffer(0)]], device float* W [[buffer(1)]],
                     device const float* P [[buffer(2)]], device const float* tmp [[buffer(3)]],
                     device const float* tau [[buffer(4)]], device const float* scal [[buffer(5)]],
                     device float* dpart [[buffer(6)]], constant TdParams& p [[buffer(7)]],
                     uint2 gid2 [[thread_position_in_grid]], uint2 g2 [[threadgroup_position_in_grid]],
                     uint t [[thread_index_in_threadgroup]],
                     uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float part[GROUP / 32];
    const uint gid = gid2.x, g = g2.x;
    const ulong mat = g2.y;
    Ak += mat * p.sa; W += mat * p.sw; P += mat * p.sp; tmp += mat * p.stm; tau += mat * p.sv;
    scal += mat * p.sr; dpart += mat * p.sr;
    const uint i = p.i, m = p.nn - i - 1, lda = p.lda, ldw = p.ldw;
    const uint r = gid / LANES, q = gid % LANES, row = i + 1 + r;
    float acc = 0.0f;
    if (r < m) {
        const uint slots = (m + TILE - 1) / TILE;
        for (uint s = q; s < slots; s += LANES) acc += P[(ulong)s * m + r];
        for (uint j = q; j < i; j += LANES)
            acc -= Ak[row + j * lda] * tmp[j] + W[row + j * ldw] * tmp[i + j];
    }
    acc = row_sum(acc);
    float w = 0.0f, v = 0.0f;
    if (r < m && q == 0) {
        v = r == 0 ? 1.0f : scal[0] * Ak[row + i * lda];
        Ak[row + i * lda] = v;
        w = tau[p.k + i] * acc;
        W[row + i * ldw] = w;
    }
    const float s = simd_sum(w * v);
    if (lane == 0) part[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t == 0) {
        float u = 0.0f;
        for (uint k = 0; k < GROUP / 32; ++k) u += part[k];
        dpart[g] = u;
    }
}

struct PackParams {
    uint m;     // rows of the trailing matrix below the panel
    uint nb;
    uint lda;
    uint ldw;
    uint ldb;
    uint k;     // the panel's first column
    uint ngp;   // td_apply's threadgroups for the panel's last column
    uint sa, sw, sb, sv, sr;   // per-matrix strides, in floats: A, W, B and C, d e tau, the partials
};

// B = [V W], C = [W V] (column-major, ldb), V = Ak(nb:, 0:nb), W(nb:, 0:nb),
// the last column of W finished here (td_update finishes the others).
kernel void td_pack(device const float* Ak [[buffer(0)]], device const float* W [[buffer(1)]],
                    device float* B [[buffer(2)]], device float* C [[buffer(3)]],
                    device const float* tau [[buffer(4)]], device const float* dpart [[buffer(5)]],
                    constant PackParams& q [[buffer(6)]], uint3 id [[thread_position_in_grid]]) {
    const uint r = id.x, j = id.y;
    if (r >= q.m || j >= q.nb) return;
    const ulong mat = id.z;
    Ak += mat * q.sa; W += mat * q.sw; B += mat * q.sb; C += mat * q.sb; tau += mat * q.sv; dpart += mat * q.sr;
    const float v = Ak[q.nb + r + j * q.lda];
    float w = W[q.nb + r + j * q.ldw];
    if (j == q.nb - 1) {
        float s = 0.0f;
        for (uint u = 0; u < q.ngp; ++u) s += dpart[u];
        w = fma(-0.5f * tau[q.k + j] * s, v, w);
    }
    B[r + j * q.ldb] = v;
    B[r + (q.nb + j) * q.ldb] = w;
    C[r + j * q.ldb] = w;
    C[r + (q.nb + j) * q.ldb] = v;
}

kernel void td_restore_e(device float* Ak [[buffer(0)]], device const float* e [[buffer(1)]],
                         constant PackParams& q [[buffer(2)]], constant uint& k [[buffer(3)]],
                         uint2 id [[thread_position_in_grid]]) {
    const uint j = id.x;
    const ulong mat = id.y;
    if (j < q.nb) Ak[mat * q.sa + j + 1 + j * q.lda] = e[mat * q.sv + k + j];
}

// =============================================================================
// A panel of the reduction for a batch of mid-size matrices, a threadgroup a
// matrix (since 2.17.0)
// =============================================================================
//
// The kernels above spread one large matrix's column over the whole GPU: per
// column three dispatches of 256-thread threadgroups, a threadgroup for each
// of the corrections' dot products. For a batch of matrices of a few hundred
// that is mostly idle threads (some 66 thousand threadgroups a dispatch for
// 1024 matrices of 128). Here one threadgroup takes a matrix's whole panel,
// slatrd's steps column by column with barriers between them, and the panel's
// B = [V W] and C = [W V] for the trailing update (an MPS product, batched).
//
// The symmetric product reads the trailing matrix whole, both triangles, as
// of the panel's start, a thread a row and the columns in turn (adjacent
// threads, adjacent rows: coalesced). The host symmetrizes each matrix on the
// way in, and the trailing update writes both triangles. Within the panel
// only its own columns change, below their diagonal, which the product never
// reads.

struct PanelParams {
    uint nn;    // order of Ak
    uint lda, ldw, ldb;
    uint k;     // the panel's first column, for d, e, tau
    uint nb;
    uint sa, sw, sb, sv;   // per-matrix strides, in floats: A, W, B and C, d e tau
};

// The threadgroup's sum (or max) of x, in every thread; part holds a value a
// simdgroup and is free again on return.
static float tg_sum(float x, threadgroup float* part, uint sg, uint lane, uint nsg) {
    x = simd_sum(x);
    if (lane == 0) part[sg] = x;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float s = 0.0f;
    for (uint u = 0; u < nsg; ++u) s += part[u];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return s;
}
static float tg_max(float x, threadgroup float* part, uint sg, uint lane, uint nsg) {
    x = simd_max(x);
    if (lane == 0) part[sg] = x;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float s = 0.0f;
    for (uint u = 0; u < nsg; ++u) s = max(s, part[u]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return s;
}

// Constant-indexed loops, so that a lane's tile row stays in registers.
#define TD_U32(...) { { constexpr uint c = 0; __VA_ARGS__ } { constexpr uint c = 1; __VA_ARGS__ } { constexpr uint c = 2; __VA_ARGS__ } { constexpr uint c = 3; __VA_ARGS__ } { constexpr uint c = 4; __VA_ARGS__ } { constexpr uint c = 5; __VA_ARGS__ } { constexpr uint c = 6; __VA_ARGS__ } { constexpr uint c = 7; __VA_ARGS__ } { constexpr uint c = 8; __VA_ARGS__ } { constexpr uint c = 9; __VA_ARGS__ } { constexpr uint c = 10; __VA_ARGS__ } { constexpr uint c = 11; __VA_ARGS__ } { constexpr uint c = 12; __VA_ARGS__ } { constexpr uint c = 13; __VA_ARGS__ } { constexpr uint c = 14; __VA_ARGS__ } { constexpr uint c = 15; __VA_ARGS__ } { constexpr uint c = 16; __VA_ARGS__ } { constexpr uint c = 17; __VA_ARGS__ } { constexpr uint c = 18; __VA_ARGS__ } { constexpr uint c = 19; __VA_ARGS__ } { constexpr uint c = 20; __VA_ARGS__ } { constexpr uint c = 21; __VA_ARGS__ } { constexpr uint c = 22; __VA_ARGS__ } { constexpr uint c = 23; __VA_ARGS__ } { constexpr uint c = 24; __VA_ARGS__ } { constexpr uint c = 25; __VA_ARGS__ } { constexpr uint c = 26; __VA_ARGS__ } { constexpr uint c = 27; __VA_ARGS__ } { constexpr uint c = 28; __VA_ARGS__ } { constexpr uint c = 29; __VA_ARGS__ } { constexpr uint c = 30; __VA_ARGS__ } { constexpr uint c = 31; __VA_ARGS__ } }
#define TD_U16(...) { { constexpr uint j = 0; __VA_ARGS__ } { constexpr uint j = 1; __VA_ARGS__ } { constexpr uint j = 2; __VA_ARGS__ } { constexpr uint j = 3; __VA_ARGS__ } { constexpr uint j = 4; __VA_ARGS__ } { constexpr uint j = 5; __VA_ARGS__ } { constexpr uint j = 6; __VA_ARGS__ } { constexpr uint j = 7; __VA_ARGS__ } { constexpr uint j = 8; __VA_ARGS__ } { constexpr uint j = 9; __VA_ARGS__ } { constexpr uint j = 10; __VA_ARGS__ } { constexpr uint j = 11; __VA_ARGS__ } { constexpr uint j = 12; __VA_ARGS__ } { constexpr uint j = 13; __VA_ARGS__ } { constexpr uint j = 14; __VA_ARGS__ } { constexpr uint j = 15; __VA_ARGS__ } }
#define TD_U8(...) { { constexpr uint j = 0; __VA_ARGS__ } { constexpr uint j = 1; __VA_ARGS__ } { constexpr uint j = 2; __VA_ARGS__ } { constexpr uint j = 3; __VA_ARGS__ } { constexpr uint j = 4; __VA_ARGS__ } { constexpr uint j = 5; __VA_ARGS__ } { constexpr uint j = 6; __VA_ARGS__ } { constexpr uint j = 7; __VA_ARGS__ } }
#define TD_U4(...) { { constexpr uint j = 0; __VA_ARGS__ } { constexpr uint j = 1; __VA_ARGS__ } { constexpr uint j = 2; __VA_ARGS__ } { constexpr uint j = 3; __VA_ARGS__ } }
#define TD_U2(...) { { constexpr uint j = 0; __VA_ARGS__ } { constexpr uint j = 1; __VA_ARGS__ } }

// *p += v for a float in threadgroup memory: Metal has no float atomics
// there, so a compare-and-swap on its bits (the lanes of a simdgroup add to
// distinct addresses; simdgroups rarely meet).
inline void tg_atomic_add(threadgroup atomic_uint* p, float v) {
    uint old = atomic_load_explicit(p, memory_order_relaxed);
    while (!atomic_compare_exchange_weak_explicit(p, &old, as_type<uint>(as_type<float>(old) + v),
                                                  memory_order_relaxed, memory_order_relaxed)) {}
}

// b[c] summed over the simdgroup's lanes, lane c taking column c's sum: five
// halving stages, each lane keeping the half of its columns that its lane bit
// selects and adding its partner's copy of it (31 shuffles in all).
#define TD_TR_STAGE(U, H)                                                        \
    {                                                                            \
        const bool up = (lane & H) != 0;                                         \
        U({ const float send = up ? b[j] : b[j + H], keep = up ? b[j + H] : b[j]; \
            b[j] = keep + simd_shuffle_xor(send, (ushort)H); })                  \
    }
#define TD_TRANSPOSE_SUM(b)                                                      \
    TD_TR_STAGE(TD_U16, 16) TD_TR_STAGE(TD_U8, 8) TD_TR_STAGE(TD_U4, 4)           \
    TD_TR_STAGE(TD_U2, 2) { const bool up = (lane & 1) != 0;                     \
        const float send = up ? b[0] : b[1], keep = up ? b[1] : b[0];           \
        b[0] = keep + simd_shuffle_xor(send, (ushort)1); }

// Threadgroup memory: col and wcol, nn floats each (the host sizes it, so
// that small matrices leave room for more threadgroups a core).
kernel void td_panel(device float* A [[buffer(0)]], device float* W [[buffer(1)]],
                     device float* d [[buffer(2)]], device float* e [[buffer(3)]], device float* tau [[buffer(4)]],
                     device float* Bm [[buffer(5)]], device float* Cm [[buffer(6)]],
                     constant PanelParams& p [[buffer(7)]], threadgroup float* shm [[threadgroup(0)]],
                     uint mat [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]],
                     uint nt [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                     uint lane [[thread_index_in_simdgroup]], uint nsg [[simdgroups_per_threadgroup]]) {
    threadgroup float* col = shm;
    threadgroup float* wcol = shm + p.nn;
    threadgroup float rowW[32], rowA[32], tmp[64], part[32];
    const ulong mt = mat;
    A += mt * p.sa; W += mt * p.sw; d += mt * p.sv; e += mt * p.sv; tau += mt * p.sv;
    Bm += mt * p.sb; Cm += mt * p.sb;
    const uint nn = p.nn, lda = p.lda, ldw = p.ldw;
    for (uint i = 0; i < p.nb; ++i) {
        // Row i of the panel's V and W so far
        if (t < i) {
            rowW[t] = W[i + t * ldw];
            rowA[t] = A[i + t * lda];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Ak(i:, i) -= Ak(i:, 0:i) W(i, 0:i)^T + W(i:, 0:i) Ak(i, 0:i)^T, into col
        // alone: v overwrites the column below the diagonal (by other threads:
        // two threads' writes to one address would race), and d takes the
        // diagonal
        for (uint r = i + t; r < nn; r += nt) {
            float x = A[r + i * lda];
            if (i > 0) {
                float acc = 0.0f;
                for (uint j = 0; j < i; ++j) acc = fma(A[r + j * lda], rowW[j], fma(W[r + j * ldw], rowA[j], acc));
                x -= acc;
            }
            col[r] = x;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // The reflector annihilating Ak(i+2:, i), as slarfg, its norm scaled
        // by the largest magnitude; every thread computes the same. Alpha and
        // the diagonal read now: v overwrites col[i + 1] once a thread is past
        // the last barrier
        if (i + 1 >= nn) {   // the last column of the last panel: its diagonal alone
            if (t == 0) d[p.k + i] = col[i];
            break;
        }
        const float alpha = col[i + 1], diag = col[i];
        float m = 0.0f;
        for (uint r = i + 2 + t; r < nn; r += nt) m = max(m, fabs(col[r]));
        const float amax = tg_max(m, part, sg, lane, nsg);
        float ss = 0.0f;
        if (amax > 0.0f) {
            const float inv = div1(1.0f, amax);
            for (uint r = i + 2 + t; r < nn; r += nt) {
                const float z = col[r] * inv;
                ss = fma(z, z, ss);
            }
        }
        ss = tg_sum(ss, part, sg, lane, nsg);
        float beta, ta, sc;
        householder(alpha, amax * sqrt1(ss), beta, ta, sc);
        if (t == 0) {
            d[p.k + i] = diag;
            e[p.k + i] = beta;
            tau[p.k + i] = ta;
        }
        // v = [1; sc x], into col and Ak(i+1:, i)
        for (uint r = i + 1 + t; r < nn; r += nt) {
            const float v = r == i + 1 ? 1.0f : sc * col[r];
            col[r] = v;
            A[r + i * lda] = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // The corrections' dot products: tmp[j] = W(i+1:, j)^T v and
        // tmp[i + j] = Ak(i+1:, j)^T v, j < i, a simdgroup each
        for (uint q = sg; q < 2 * i; q += nsg) {
            device const float* c = q < i ? W + q * ldw : A + (q - i) * lda;
            float s = 0.0f;
            for (uint r = i + 1 + lane; r < nn; r += 32) s = fma(c[r], col[r], s);
            s = simd_sum(s);
            if (lane == 0) tmp[q] = s;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Ak(i+1:, i+1:) v from its lower triangle alone, half the reads of the
        // whole: 32 x 32 tiles, a simdgroup each, a lane a row. A tile below the
        // diagonal gives its rows' terms and, transposed (summed across the
        // lanes), its columns'; a diagonal tile its rows' alone. Summed into
        // wcol by threadgroup atomics.
        threadgroup atomic_uint* wacc = reinterpret_cast<threadgroup atomic_uint*>(wcol);
        for (uint r = i + 1 + t; r < nn; r += nt) wcol[r] = 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {
            const uint b0 = i + 1, nbk = (nn - b0 + 31) / 32, ntiles = nbk * (nbk + 1) / 2;
            for (uint tl = sg; tl < ntiles; tl += nsg) {
                uint R = (uint)((sqrt(8.0f * (float)tl + 1.0f) - 1.0f) * 0.5f);   // tile tl = (R, C), C <= R
                while ((R + 1) * (R + 2) / 2 <= tl) ++R;
                while (R * (R + 1) / 2 > tl) --R;
                const uint C = tl - R * (R + 1) / 2;
                const uint r = b0 + 32 * R + lane, c0 = b0 + 32 * C;
                const bool rin = r < nn;
                const float vr = rin ? col[r] : 0.0f;
                device const float* ar = A + r;
                float rp = 0.0f, b[32];
                TD_U32({
                    const uint cc = c0 + c;
                    const bool in = cc < nn;
                    const float a = rin && in ? ar[(ulong)cc * lda] : 0.0f;
                    rp = fma(a, in ? col[cc] : 0.0f, rp);
                    b[c] = a * vr;
                })
                if (rin) tg_atomic_add(&wacc[r], rp);
                if (R != C) {   // uniform
                    TD_TRANSPOSE_SUM(b)
                    if (c0 + lane < nn) tg_atomic_add(&wacc[c0 + lane], b[0]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // W(i+1:, i) = tau (Ak(i+1:, i+1:) v - Ak(i+1:, 0:i) tmp - W(i+1:, 0:i) tmp'),
        // then W += -tau/2 (W . v) v
        float dot = 0.0f;
        for (uint r = i + 1 + t; r < nn; r += nt) {
            float acc = wcol[r];
            for (uint j = 0; j < i; ++j) acc -= A[r + j * lda] * tmp[j] + W[r + j * ldw] * tmp[i + j];
            const float w = ta * acc;
            wcol[r] = w;
            dot = fma(w, col[r], dot);
        }
        const float al = -0.5f * ta * tg_sum(dot, part, sg, lane, nsg);
        for (uint r = i + 1 + t; r < nn; r += nt) W[r + i * ldw] = fma(al, col[r], wcol[r]);
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    }
    // B = [V W], C = [W V] below the panel (column-major, ldb), then e back
    // where each reflector's unit was
    const uint m = nn - p.nb, nb = p.nb, ldb = p.ldb;
    for (uint idx = t; idx < m * nb; idx += nt) {
        const uint r = idx % m, j = idx / m;
        const float v = A[nb + r + j * lda], w = W[nb + r + j * ldw];
        Bm[r + j * ldb] = v;
        Bm[r + (nb + j) * ldb] = w;
        Cm[r + j * ldb] = w;
        Cm[r + (nb + j) * ldb] = v;
    }
    threadgroup_barrier(mem_flags::mem_device);
    if (t < nb && t + 1 < nn) A[t + 1 + t * lda] = e[p.k + t];
}

// The batch's matrices into the workspace: matrix index[c0 + j] of src
// (row-major, n x n, its lower or upper triangle) as matrix j of A,
// column-major, ld lda, scaled by scale[c0 + j], both triangles.
struct LoadParams { uint n, lda, sa, lower, c0; };

kernel void td_load(device const float* src [[buffer(0)]], device float* A [[buffer(1)]],
                    device const uint* index [[buffer(2)]], device const float* scale [[buffer(3)]],
                    constant LoadParams& p [[buffer(4)]], uint3 g [[threadgroup_position_in_grid]],
                    uint3 l [[thread_position_in_threadgroup]]) {
    // A 32 x 32 tile of A (rows r0.., columns c0..) through threadgroup
    // memory, so that the reads and the writes are both coalesced. Its
    // entries come from src(r, c) or src(c, r), whichever is in the valid
    // triangle: both tiles are read, each along src's rows. Threadgroups of
    // 32 x 8.
    threadgroup float t1[32][33], t2[32][33];
    const uint r0 = g.x * 32, c0 = g.y * 32, j = g.z, n = p.n;
    const ulong b = index[p.c0 + j];
    device const float* s = src + b * n * n;
    for (uint q = l.y; q < 32; q += 8) {
        const uint x = l.x;
        t1[q][x] = c0 + q < n && r0 + x < n ? s[(ulong)(c0 + q) * n + r0 + x] : 0.0f;   // src(c0 + q, r0 + x)
        t2[q][x] = r0 + q < n && c0 + x < n ? s[(ulong)(r0 + q) * n + c0 + x] : 0.0f;   // src(r0 + q, c0 + x)
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float sc = scale[p.c0 + j];
    for (uint q = l.y; q < 32; q += 8) {
        const uint i = l.x, r = r0 + i, c = c0 + q;    // A(r, c)
        if (r >= n || c >= n) continue;
        // src(r, c) = t2[i][q], src(c, r) = t1[q][i]; the lower triangle is row >= column
        const bool direct = p.lower ? r >= c : r <= c;
        A[(ulong)j * p.sa + (ulong)c * p.lda + r] = (direct ? t2[i][q] : t1[q][i]) * sc;
    }
}

// The eigenvectors out: matrix j of Z (column-major, n x n) as matrix
// index[c0 + j] of out, row-major, the vectors its columns.
struct StoreParams { uint n, c0; };

kernel void td_store(device const float* Z [[buffer(0)]], device float* out [[buffer(1)]],
                     device const uint* index [[buffer(2)]], constant StoreParams& p [[buffer(3)]],
                     uint3 g [[threadgroup_position_in_grid]], uint3 l [[thread_position_in_threadgroup]]) {
    // A 32 x 32 tile transposed through threadgroup memory: Z read down its
    // columns, out written along its rows. Threadgroups of 32 x 8.
    threadgroup float tile[32][33];
    const uint i0 = g.x * 32, j0 = g.y * 32, n = p.n;
    const ulong j = g.z, b = index[p.c0 + j], nn = (ulong)n * n;
    for (uint q = l.y; q < 32; q += 8) {
        const uint row = i0 + l.x, col = j0 + q;   // Z(row, col), column-major
        tile[q][l.x] = row < n && col < n ? Z[j * nn + (ulong)col * n + row] : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint q = l.y; q < 32; q += 8) {
        const uint row = i0 + q, col = j0 + l.x;   // out(row, col) = Z(row, col)
        if (row < n && col < n) out[b * nn + (ulong)row * n + col] = tile[l.x][q];
    }
}

// The back-transformation's blocks for a batch, on the GPU (since 2.17.0):
// a block of kb reflectors k0 .. k0 + kb - 1 of every matrix as
// H = I - V T V^T, V (m x bb, row-major, its columns from kb on zero) by
// td_make_v, G = V^T V by an MPS product, and T by td_make_t, from G and
// tau as compact_wy_t (metal_runtime.mm) builds it on the CPU.

struct WyParams {
    uint m, kb, bb, k0, lda;
    uint sa, sv, svb, stb;   // per-matrix strides, in floats: A, tau, V, G and T
};

kernel void td_make_v(device const float* A [[buffer(0)]], device float* V [[buffer(1)]],
                      constant WyParams& p [[buffer(2)]], uint3 id [[thread_position_in_grid]]) {
    const uint j = id.x, r = id.y;
    if (j >= p.bb || r >= p.m) return;
    const ulong mat = id.z;
    float v = 0.0f;
    if (j < p.kb) v = r < j ? 0.0f : r == j ? 1.0f : A[mat * p.sa + (ulong)(p.k0 + j) * p.lda + p.k0 + 1 + r];
    V[mat * p.svb + (ulong)r * p.bb + j] = v;
}

// T (row-major, upper triangular) for blocks of at most 64 reflectors, a
// threadgroup a matrix, T in threadgroup memory: T(i, j) = -tau_j T(i, 0:j)
// G(0:j, j), T(i, i) = tau_i, compact_wy_t's recurrence, in blocks (below).
// (With T in device memory, each column's barrier had to make the last one's
// writes visible through memory: 6 ms of a 1024 x 128^2 batch's
// back-transformation, for blocks of 128.)
constant constexpr uint WY_MAX = 64;

kernel void td_make_t(device const float* G [[buffer(0)]], device const float* tau [[buffer(1)]],
                      device float* T [[buffer(2)]], constant WyParams& p [[buffer(3)]],
                      uint mat [[threadgroup_position_in_grid]], uint i [[thread_index_in_threadgroup]],
                      uint nt [[threads_per_threadgroup]]) {
    threadgroup float Ts[WY_MAX][WY_MAX], Gs[WY_MAX][WY_MAX];   // 32 KB: G staged, T built (bb <= 64)
    const ulong mt = mat;
    G += mt * p.stb; T += mt * p.stb; tau += mt * p.sv + p.k0;
    const uint bb = p.bb, kb = p.kb;
    // T in blocks: the four 16 x 16 diagonal blocks a thread a row (row i of a
    // block needs only its own earlier entries and G), then merged in pairs,
    // T(a, b) = -T(a, a) G(a, b) T(b, b), at 16 and then 32: five barriers,
    // where column by column took 128 (2-4% of a batch's call)
    for (uint e = i; e < 64 * 64; e += nt) Gs[e / 64][e % 64] = e / 64 < bb && e % 64 < bb ? G[(e / 64) * bb + e % 64] : 0.0f;
    for (uint e = i; e < 64 * 64; e += nt) Ts[e / 64][e % 64] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (i < 64) {
        const uint q0 = i / 16 * 16, q1 = q0 + 16;
        Ts[i][i] = i < kb ? tau[i] : 0.0f;
        for (uint j = i + 1; j < q1; ++j) {
            float s = 0.0f;
            for (uint k = i; k < j; ++k) s = fma(Ts[i][k], Gs[k][j], s);
            Ts[i][j] = -(j < kb ? tau[j] : 0.0f) * s;
        }
        (void)q0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint w = 16; w < 64; w *= 2) {
        // pairs of w-blocks (a, b) = ([2 w p, 2 w p + w), [2 w p + w, 2 w p + 2 w)):
        // X = G(a, b) T(b, b) into Ts(a, b), then T(a, b) = -T(a, a) X
        threadgroup float (*X)[64] = Gs;   // G(b, b) blocks are done with; X over G(a, b)
        for (uint e = i; e < (64 / (2 * w)) * w * w; e += nt) {
            const uint pr = e / (w * w), r = e % (w * w) / w, c = e % w;
            const uint a0 = 2 * w * pr, b0 = a0 + w;
            float s = 0.0f;
            for (uint k = 0; k <= c; ++k) s = fma(Gs[a0 + r][b0 + k], Ts[b0 + k][b0 + c], s);
            Ts[a0 + r][b0 + c] = s;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint e = i; e < (64 / (2 * w)) * w * w; e += nt) {
            const uint pr = e / (w * w), r = e % (w * w) / w, c = e % w;
            const uint a0 = 2 * w * pr, b0 = a0 + w;
            float s = 0.0f;
            for (uint k = r; k < w; ++k) s = fma(Ts[a0 + r][a0 + k], Ts[a0 + k][b0 + c], s);
            X[a0 + r][b0 + c] = -s;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint e = i; e < (64 / (2 * w)) * w * w; e += nt) {
            const uint pr = e / (w * w), r = e % (w * w) / w, c = e % w;
            const uint a0 = 2 * w * pr, b0 = a0 + w;
            Ts[a0 + r][b0 + c] = X[a0 + r][b0 + c];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint e = i; e < bb * bb; e += nt) T[e] = Ts[e / bb][e % bb];
}

// =============================================================================
// Eigenvalues of a symmetric tridiagonal by bisection (src/bisect.mm): a
// thread an eigenvalue, the k-th smallest found by halving an interval with
// Sturm counts, the number of eigenvalues below x being the number of
// negative pivots of T - x I = L D L^T. Every thread reads the same diagonal
// entries at the same step, so they are staged through threadgroup memory a
// chunk at a time, and every thread runs the same number of halvings (enough
// for float32's precision on the interval), keeping the threadgroup in step.
// The singular values of an upper bidiagonal are the eigenvalues of its
// Golub-Kahan form, of order 2n with a zero diagonal and off-diagonal
// d0, e0, d1, e1, ..., in plus-minus pairs: `tgk` reads no diagonal.
// A pivot smaller than pivmin is replaced by -pivmin, as LAPACK's slaebz does.
// =============================================================================

struct SturmParams {
    uint  n;        // eigenvalues to find: indices offset .. offset + n - 1, ascending
    uint  size;     // the tridiagonal's order
    uint  offset;
    uint  tgk;      // no diagonal (the Golub-Kahan form)
    float lo, hi;   // an interval holding them all
    float pivmin;
    uint  passes;   // halvings
};

constant constexpr uint STURM_CHUNK = 1024;

kernel void sturm_bisect(device const float* d [[buffer(0)]], device const float* e2 [[buffer(1)]],
                         device float* out [[buffer(2)]], constant SturmParams& p [[buffer(3)]],
                         uint t [[thread_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                         uint nt [[threads_per_threadgroup]]) {
    threadgroup float sd[STURM_CHUNK], se[STURM_CHUNK];
    const uint k = t + p.offset;
    float lo = p.lo, hi = p.hi;
    for (uint pass = 0; pass < p.passes; ++pass) {
        const float mid = 0.5f * (lo + hi);
        uint  below = 0;
        float q = 1.0f;
        for (uint c0 = 0; c0 < p.size; c0 += STURM_CHUNK) {
            const uint m = min(STURM_CHUNK, p.size - c0);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint i = tid; i < m; i += nt) {
                sd[i] = p.tgk ? 0.0f : d[c0 + i];
                se[i] = c0 + i == 0 ? 0.0f : e2[c0 + i - 1];   // e(i-1)^2; none before the first pivot
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint i = 0; i < m; ++i) {
                float v = sd[i] - mid - div1(se[i], q);   // |q| >= pivmin
                if (fabs(v) < p.pivmin) v = -p.pivmin;
                q = v;
                below += v < 0.0f;
            }
        }
        if (below > k) hi = mid;
        else           lo = mid;
    }
    if (t < p.n) out[t] = 0.5f * (lo + hi);
}
