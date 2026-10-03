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
// per column, slatrd's steps as small kernels launched back to back, and per
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
// Per column i of the panel (len = nn - i, lr = len - 1):
//   td_col_update     Ak(i:, i) -= Ak(i:, 0:i) W(i, 0:i)^T + W(i:, 0:i) Ak(i, 0:i)^T
//   td_larfg          the reflector annihilating Ak(i+2:, i): d, e, tau as ssytrd
//   td_symv_tiles,    W(i+1:, i) = Ak(i+1:, i+1:) v, reading the lower triangle
//   td_symv_reduce      only, in 64 x 64 tiles; the trailing matrix as of the
//                       panel's start, which slatrd's corrections account for
//   td_corr_dots      W(i+1:, 0:i)^T v and Ak(i+1:, 0:i)^T v
//   td_corr_apply     W(i+1:, i) = tau (W - Ak(i+1:, 0:i) . - W(i+1:, 0:i) .)
//   td_finish_w       W(i+1:, i) += -tau/2 (W . v) v
// and per panel:
//   td_pack           B = [V W], C = [W V]; then A22 -= B C^T (MPS GEMM)
//   td_restore_e      Ak(j+1, j) = e(j) (the reflector's unit was stored there)

struct TdParams {
    uint nn;    // order of Ak
    uint lda;
    uint ldw;
    uint i;     // column within the panel
    uint k;     // the panel's first column, for d, e, tau
};

kernel void td_col_update(device float* Ak [[buffer(0)]], device const float* W [[buffer(1)]],
                          constant TdParams& p [[buffer(2)]], uint r [[thread_position_in_grid]]) {
    const uint len = p.nn - p.i;
    if (r >= len || p.i == 0) return;
    const uint row = p.i + r;
    float acc = 0.0f;
    for (uint j = 0; j < p.i; ++j)
        acc += Ak[row + j * p.lda] * W[p.i + j * p.ldw] + W[row + j * p.ldw] * Ak[p.i + j * p.lda];
    Ak[row + p.i * p.lda] -= acc;
}

// One threadgroup. As LAPACK's slarfg, with the norm scaled by the largest
// magnitude so that it neither overflows nor underflows.
kernel void td_larfg(device float* Ak [[buffer(0)]], device float* d [[buffer(1)]], device float* e [[buffer(2)]],
                     device float* tau [[buffer(3)]], constant TdParams& p [[buffer(4)]],
                     uint t [[thread_position_in_threadgroup]], uint nt [[threads_per_threadgroup]],
                     uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float part[32];
    threadgroup float scale_s;
    device float* col = Ak + p.i * p.lda + p.i;
    device float* v = col + 1;
    const uint lr = p.nn - p.i - 1;
    const uint nsg = (nt + 31) / 32;

    float mx = 0.0f;
    for (uint r = 1 + t; r < lr; r += nt) mx = max(mx, fabs(v[r]));
    mx = simd_max(mx);
    if (lane == 0) part[sg] = mx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float q = lane < nsg ? part[lane] : 0.0f;
        q = simd_max(q);
        if (lane == 0) part[0] = q;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float amax = part[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float ss = 0.0f;
    if (amax > 0.0f)
        for (uint r = 1 + t; r < lr; r += nt) { const float z = v[r] / amax; ss = fma(z, z, ss); }
    ss = simd_sum(ss);
    if (lane == 0) part[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t == 0) {
        float s = 0.0f;
        for (uint q = 0; q < nsg; ++q) s += part[q];
        const float xnorm = amax * sqrt(s);
        const float alpha = lr >= 1 ? v[0] : 0.0f;
        d[p.k + p.i] = col[0];
        if (xnorm == 0.0f) {
            tau[p.k + p.i] = 0.0f;
            e[p.k + p.i] = alpha;
            scale_s = 1.0f;
        } else {
            const float big = max(fabs(alpha), xnorm);      // hypot, scaled
            const float ra = alpha / big, rx = xnorm / big;
            const float beta = -copysign(big * sqrt(ra * ra + rx * rx), alpha);
            tau[p.k + p.i] = (beta - alpha) / beta;
            e[p.k + p.i] = beta;
            scale_s = 1.0f / (alpha - beta);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float sc = scale_s;
    for (uint r = 1 + t; r < lr; r += nt) v[r] *= sc;
    if (t == 0 && lr >= 1) v[0] = 1.0f;
}

// y = M x for symmetric M (column-major, lda), reading its lower triangle only.
// One threadgroup (256 threads) per 64 x 64 tile (bi, bj), bi >= bj:
//   rows part     y[block bi] += T x[block bj]       (on the diagonal tile: r >= c)
//   columns part  y[block bj] += T^T x[block bi]     (on the diagonal tile: r > c)
// Partials go to P[slot * m + row]: block b gets slot s <= b from tile (b, s)'s
// rows part and slot s > b from tile (s, b)'s columns part, so each (block,
// slot) is written exactly once; the diagonal tile adds its two parts first.
constant constexpr uint TILE = 64;

struct SymvDims {
    uint m;
    uint lda;
};

kernel void td_symv_tiles(device const float* A [[buffer(0)]], device const float* x [[buffer(1)]],
                          device float* P [[buffer(2)]], constant SymvDims& dm [[buffer(3)]],
                          uint2 tg [[threadgroup_position_in_grid]],
                          uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    const uint m = dm.m, lda = dm.lda, bi = tg.y, bj = tg.x;
    if (bj > bi) return;
    threadgroup float xi[TILE], xj[TILE], colsum[TILE], rows[8][TILE];
    const uint tid = sg * 32 + lane;
    if (tid < TILE) {
        const uint r = bi * TILE + tid;
        xi[tid] = r < m ? x[r] : 0.0f;
        const uint c = bj * TILE + tid;
        xj[tid] = c < m ? x[c] : 0.0f;
        colsum[tid] = 0.0f;
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
        const float t = simd_sum(s0 * xi[r0] + s1 * xi[r1]);
        if (lane == 0) {
            if (diag) colsum[c] = t;
            else      P[(ulong)bi * m + gc] = t;      // slot bi of block bj
        }
    }
    rows[sg][r0] = acc0;
    rows[sg][r1] = acc1;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < TILE) {
        const uint g = bi * TILE + tid;
        if (g < m) {
            float v = 0.0f;
            for (uint q = 0; q < 8; ++q) v += rows[q][tid];
            if (diag) v += colsum[tid];
            P[(ulong)bj * m + g] = v;                 // slot bj of block bi
        }
    }
}

kernel void td_symv_reduce(device const float* P [[buffer(0)]], device float* y [[buffer(1)]],
                           constant SymvDims& dm [[buffer(2)]], uint r [[thread_position_in_grid]]) {
    const uint m = dm.m, slots = (m + TILE - 1) / TILE;
    if (r >= m) return;
    float v = 0.0f;
    for (uint s = 0; s < slots; ++s) v += P[(ulong)s * m + r];
    y[r] = v;
}

// tmp[t] = W(i+1:, t)^T v and tmp[i + t] = Ak(i+1:, t)^T v, t < i: one
// threadgroup each.
kernel void td_corr_dots(device const float* Ak [[buffer(0)]], device const float* W [[buffer(1)]],
                         device float* tmp [[buffer(2)]], constant TdParams& p [[buffer(3)]],
                         uint g [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]],
                         uint nt [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float part[32];
    const uint lr = p.nn - p.i - 1, j = g < p.i ? g : g - p.i;
    device const float* v = Ak + p.i * p.lda + p.i + 1;
    device const float* c = g < p.i ? W + j * p.ldw + p.i + 1 : Ak + j * p.lda + p.i + 1;
    float s = 0.0f;
    for (uint r = t; r < lr; r += nt) s = fma(c[r], v[r], s);
    s = simd_sum(s);
    if (lane == 0) part[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t == 0) {
        float q = 0.0f;
        for (uint u = 0; u < (nt + 31) / 32; ++u) q += part[u];
        tmp[g] = q;
    }
}

kernel void td_corr_apply(device const float* Ak [[buffer(0)]], device float* W [[buffer(1)]],
                          device const float* tmp [[buffer(2)]], device const float* tau [[buffer(3)]],
                          constant TdParams& p [[buffer(4)]], uint r [[thread_position_in_grid]]) {
    const uint lr = p.nn - p.i - 1;
    if (r >= lr) return;
    const uint row = p.i + 1 + r;
    float acc = W[row + p.i * p.ldw];
    for (uint j = 0; j < p.i; ++j)
        acc -= Ak[row + j * p.lda] * tmp[j] + W[row + j * p.ldw] * tmp[p.i + j];
    W[row + p.i * p.ldw] = tau[p.k + p.i] * acc;
}

// One threadgroup.
kernel void td_finish_w(device const float* Ak [[buffer(0)]], device float* W [[buffer(1)]],
                        device const float* tau [[buffer(2)]], constant TdParams& p [[buffer(3)]],
                        uint t [[thread_position_in_threadgroup]], uint nt [[threads_per_threadgroup]],
                        uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float part[32];
    threadgroup float alpha_s;
    const uint lr = p.nn - p.i - 1;
    device const float* v = Ak + p.i * p.lda + p.i + 1;
    device float* y = W + p.i * p.ldw + p.i + 1;
    float s = 0.0f;
    for (uint r = t; r < lr; r += nt) s = fma(y[r], v[r], s);
    s = simd_sum(s);
    if (lane == 0) part[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t == 0) {
        float q = 0.0f;
        for (uint u = 0; u < (nt + 31) / 32; ++u) q += part[u];
        alpha_s = -0.5f * tau[p.k + p.i] * q;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float a = alpha_s;
    for (uint r = t; r < lr; r += nt) y[r] = fma(a, v[r], y[r]);
}

struct PackParams {
    uint m;     // rows of the trailing matrix below the panel
    uint nb;
    uint lda;
    uint ldw;
    uint ldb;
};

// B = [V W], C = [W V] (column-major, ldb), V = Ak(nb:, 0:nb), W(nb:, 0:nb).
kernel void td_pack(device const float* Ak [[buffer(0)]], device const float* W [[buffer(1)]],
                    device float* B [[buffer(2)]], device float* C [[buffer(3)]],
                    constant PackParams& q [[buffer(4)]], uint2 id [[thread_position_in_grid]]) {
    const uint r = id.x, j = id.y;
    if (r >= q.m || j >= q.nb) return;
    const float v = Ak[q.nb + r + j * q.lda], w = W[q.nb + r + j * q.ldw];
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
