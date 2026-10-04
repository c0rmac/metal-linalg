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

constant constexpr uint TILE  = 64;    // td_symv's tiles
constant constexpr uint GROUP = 256;   // threads per threadgroup of td_update and td_apply
constant constexpr uint LANES = 8;     // their threads per row: each sums every LANES-th term
constant constexpr uint ROWS  = GROUP / LANES;   // rows per threadgroup

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
};

// tau/2 (W(:, i-1) . v) from td_apply's partials, the same in every simdgroup.
static float finish_coeff(device const float* dpart, uint ngp, float tau, uint lane) {
    float q = 0.0f;
    for (uint u = lane; u < ngp; u += 32) q += dpart[u];
    return -0.5f * tau * simd_sum(q);
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
        if (pm > m) { const float f = m / pm; ss = ss * f * f + ps; m = pm; }
        else if (pm > 0.0f) { const float f = pm / m; ss += ps * f * f; }
    }
    const float amax = simd_max(m);
    const float f = amax > 0.0f ? m / amax : 0.0f;
    const float xnorm = amax * sqrt(simd_sum(ss * f * f));
    Reflector r;
    if (xnorm == 0.0f) {
        r.beta = alpha; r.tau = 0.0f; r.scale = 1.0f;
    } else {
        const float big = max(fabs(alpha), xnorm);      // hypot, scaled
        const float ra = alpha / big, rx = xnorm / big;
        r.beta  = -copysign(big * sqrt(ra * ra + rx * rx), alpha);
        r.tau   = (r.beta - alpha) / r.beta;
        r.scale = 1.0f / (alpha - r.beta);
    }
    return r;
}

// LANES threads per row i + r of Ak(i:, i), ROWS rows per threadgroup. Only
// a row's threads write W(row, i-1) and Ak(row, i); W(i, i-1), which every row
// reads, is finished on the fly by each and never stored (nothing reads it
// later).
kernel void td_update(device float* Ak [[buffer(0)]], device float* W [[buffer(1)]],
                      device const float* tau [[buffer(2)]], device const float* dpart [[buffer(3)]],
                      device float* npart [[buffer(4)]], constant TdParams& p [[buffer(5)]],
                      uint gid [[thread_position_in_grid]], uint g [[threadgroup_position_in_grid]],
                      uint t [[thread_position_in_threadgroup]],
                      uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float part[GROUP / 32];
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
    const float z = gm > 0.0f ? ax / gm : 0.0f;
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
                    uint g [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]],
                    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
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

    uint bi = (uint)((sqrt(8.0f * (float)g + 1.0f) - 1.0f) * 0.5f);
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

// LANES threads per row i + 1 + r, ROWS rows per threadgroup.
kernel void td_apply(device float* Ak [[buffer(0)]], device float* W [[buffer(1)]],
                     device const float* P [[buffer(2)]], device const float* tmp [[buffer(3)]],
                     device const float* tau [[buffer(4)]], device const float* scal [[buffer(5)]],
                     device float* dpart [[buffer(6)]], constant TdParams& p [[buffer(7)]],
                     uint gid [[thread_position_in_grid]], uint g [[threadgroup_position_in_grid]],
                     uint t [[thread_position_in_threadgroup]],
                     uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float part[GROUP / 32];
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
};

// B = [V W], C = [W V] (column-major, ldb), V = Ak(nb:, 0:nb), W(nb:, 0:nb),
// the last column of W finished here (td_update finishes the others).
kernel void td_pack(device const float* Ak [[buffer(0)]], device const float* W [[buffer(1)]],
                    device float* B [[buffer(2)]], device float* C [[buffer(3)]],
                    device const float* tau [[buffer(4)]], device const float* dpart [[buffer(5)]],
                    constant PackParams& q [[buffer(6)]], uint2 id [[thread_position_in_grid]]) {
    const uint r = id.x, j = id.y;
    if (r >= q.m || j >= q.nb) return;
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
                         uint j [[thread_position_in_grid]]) {
    if (j < q.nb) Ak[j + 1 + j * q.lda] = e[k + j];
}
