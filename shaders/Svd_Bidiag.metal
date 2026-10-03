#include <metal_stdlib>
using namespace metal;

// =============================================================================
// Householder bidiagonalization on the GPU: the device half of the SVD's
// `bidiag` backend (src/svd_bidiag.mm)
// =============================================================================
//
// For one large matrix the Jacobi SVD backends lose to LAPACK, whose sgesdd
// spends half its time reducing A to bidiagonal form B = Q^T A P (sgebrd),
// bound by memory bandwidth: two matrix-vector products with the trailing
// matrix per column. This is LAPACK's blocked sgebrd / slabrd (m >= n, upper
// bidiagonal) with every step on the GPU, queued so the host waits once per
// matrix, as Eigh_Tridiag.metal does for the symmetric reduction.

// GPU bidiagonalization, LAPACK's sgebrd / slabrd (m >= n, upper bidiagonal),
// every step on the GPU. Column-major A (lda); each kernel addresses the
// trailing block Ak = A(k:, k:) (mm x nn) of the current panel, bound at the
// offset of A(k, k). X (mm x nb, ld ldx) and Y (nn x nb, ld ldy) are the
// panel's slabrd matrices, row 0 = Ak's. Column i of the panel:
//
//   bd_col_update   Ak(i:, i) -= Ak(i:, 0:i) Y(i, 0:i)^T + X(i:, 0:i) Ak(0:i, i)
//   bd_larfg        (strided) reflector for Ak(i:, i): d(i), tauq(i); Ak(i,i) = 1
//   bd_gemv_t_*     Y(i+1:, i) = Ak(i:, i+1:)^T v
//   bd_dots_col     t1 = Ak(i:, 0:i)^T v,  t2 = X(i:, 0:i)^T v
//   bd_y_corr       Y(i+1:, i) = tauq (Y - Y(i+1:, 0:i) t1 - Ak(0:i, i+1:)^T t2)
//   bd_row_update   Ak(i, i+1:) -= Y(i+1:, 0:i+1) Ak(i, 0:i+1)^T + Ak(0:i, i+1:)^T X(i, 0:i)^T
//   bd_larfg        (strided) reflector for Ak(i, i+1:): e(i), taup(i); Ak(i,i+1) = 1
//   bd_gemv_n_*     X(i+1:, i) = Ak(i+1:, i+1:) u
//   bd_dots_row     t3 = Y(i+1:, 0:i+1)^T u,  t4 = Ak(0:i, i+1:) u
//   bd_x_corr       X(i+1:, i) = taup (X - Ak(i+1:, 0:i+1) t3 - X(i+1:, 0:i) t4)

struct BdParams {
    uint mm, nn;     // the trailing block
    uint lda, ldx, ldy;
    uint i;          // column within the panel
    uint k;          // the panel's first column, for d, e, tauq, taup
};

kernel void bd_col_update(device float* Ak [[buffer(0)]], device const float* X [[buffer(1)]],
                          device const float* Y [[buffer(2)]], constant BdParams& p [[buffer(3)]],
                          uint r [[thread_position_in_grid]]) {
    const uint row = p.i + r;
    if (row >= p.mm || p.i == 0) return;
    float acc = 0.0f;
    for (uint j = 0; j < p.i; ++j)
        acc += Ak[row + j * p.lda] * Y[p.i + j * p.ldy] + X[row + j * p.ldx] * Ak[j + p.i * p.lda];
    Ak[row + p.i * p.lda] -= acc;
}

// Rows r >= p.i + 1 of Ak(:, i+1:) (columns c >= i+1 of row i): the row update.
kernel void bd_row_update(device float* Ak [[buffer(0)]], device const float* X [[buffer(1)]],
                          device const float* Y [[buffer(2)]], constant BdParams& p [[buffer(3)]],
                          uint t [[thread_position_in_grid]]) {
    const uint c = p.i + 1 + t;
    if (c >= p.nn) return;
    float acc = 0.0f;
    for (uint j = 0; j <= p.i; ++j) acc += Y[c + j * p.ldy] * Ak[p.i + j * p.lda];
    for (uint j = 0; j < p.i; ++j)  acc += Ak[j + c * p.lda] * X[p.i + j * p.ldx];
    Ak[p.i + c * p.lda] -= acc;
}

// Householder reflector for the vector at x0 with stride `inc` and length len
// (alpha = x0[0]): beta to diag[idx], tau to tau[idx], x0[1:] scaled, x0[0] = 1.
struct LarfgParams { uint len, inc, idx; };
kernel void bd_larfg(device float* x0 [[buffer(0)]], device float* diag [[buffer(1)]], device float* tau [[buffer(2)]],
                     constant LarfgParams& q [[buffer(3)]],
                     uint t [[thread_position_in_threadgroup]], uint nt [[threads_per_threadgroup]],
                     uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float part[32];
    threadgroup float scale_s;
    const uint nsg = (nt + 31) / 32;
    float mx = 0.0f;
    for (uint r = 1 + t; r < q.len; r += nt) mx = max(mx, fabs(x0[(ulong)r * q.inc]));
    mx = simd_max(mx);
    if (lane == 0) part[sg] = mx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) { float v = lane < nsg ? part[lane] : 0.0f; v = simd_max(v); if (lane == 0) part[0] = v; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float amax = part[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float ss = 0.0f;
    if (amax > 0.0f)
        for (uint r = 1 + t; r < q.len; r += nt) { const float z = x0[(ulong)r * q.inc] / amax; ss = fma(z, z, ss); }
    ss = simd_sum(ss);
    if (lane == 0) part[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t == 0) {
        float s = 0.0f;
        for (uint v = 0; v < nsg; ++v) s += part[v];
        const float xnorm = amax * sqrt(s), alpha = x0[0];
        if (xnorm == 0.0f) {
            tau[q.idx] = 0.0f; diag[q.idx] = alpha; scale_s = 1.0f;
        } else {
            const float big = max(fabs(alpha), xnorm), ra = alpha / big, rx = xnorm / big;
            const float beta = -copysign(big * sqrt(ra * ra + rx * rx), alpha);
            tau[q.idx] = (beta - alpha) / beta;
            diag[q.idx] = beta;
            scale_s = 1.0f / (alpha - beta);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float sc = scale_s;
    for (uint r = 1 + t; r < q.len; r += nt) x0[(ulong)r * q.inc] *= sc;
    if (t == 0) x0[0] = 1.0f;
}

// y = M^T x for M (rows x cols, column-major ld): one threadgroup per 64-row
// x 64-column tile; partials P[rowblock * cols + c], then summed.
constant constexpr uint TILE = 64;
struct GemvDims { uint rows, cols, ld, xinc; };   // x read with stride xinc
kernel void bd_gemv_t_tiles(device const float* M [[buffer(0)]], device const float* x [[buffer(1)]],
                            device float* P [[buffer(2)]], constant GemvDims& g [[buffer(3)]],
                            uint2 tg [[threadgroup_position_in_grid]],
                            uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    const uint br = tg.y, bc = tg.x;
    threadgroup float xs[TILE];
    const uint tid = sg * 32 + lane;
    if (tid < TILE) { const uint r = br * TILE + tid; xs[tid] = r < g.rows ? x[(ulong)r * g.xinc] : 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint cc = 0; cc < 8; ++cc) {
        const uint c = bc * TILE + sg * 8 + cc;
        if (c >= g.cols) break;
        device const float* col = M + (ulong)c * g.ld + br * TILE;
        const uint r0 = lane, r1 = lane + 32;
        float s = (br * TILE + r0 < g.rows ? col[r0] * xs[r0] : 0.0f) +
                  (br * TILE + r1 < g.rows ? col[r1] * xs[r1] : 0.0f);
        s = simd_sum(s);
        if (lane == 0) P[(ulong)br * g.cols + c] = s;
    }
}
kernel void bd_gemv_t_reduce(device const float* P [[buffer(0)]], device float* y [[buffer(1)]],
                             constant GemvDims& g [[buffer(2)]], uint c [[thread_position_in_grid]]) {
    if (c >= g.cols) return;
    const uint blocks = (g.rows + TILE - 1) / TILE;
    float v = 0.0f;
    for (uint b = 0; b < blocks; ++b) v += P[(ulong)b * g.cols + c];
    y[c] = v;
}

// y = M x: one threadgroup per tile; each of its 256 threads sums 16 columns
// for its row (4 rows per... see below); partials P[colblock * rows + r].
// Thread layout: 64 rows x 4 column groups of 16.
kernel void bd_gemv_n_tiles(device const float* M [[buffer(0)]], device const float* x [[buffer(1)]],
                            device float* P [[buffer(2)]], constant GemvDims& g [[buffer(3)]],
                            uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    const uint br = tg.y, bc = tg.x;
    threadgroup float xs[TILE];
    threadgroup float part[4][TILE];
    if (tid < TILE) { const uint c = bc * TILE + tid; xs[tid] = c < g.cols ? x[(ulong)c * g.xinc] : 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint rl = tid % TILE, grp = tid / TILE, r = br * TILE + rl;
    float s = 0.0f;
    if (r < g.rows) {
        for (uint cc = grp * 16; cc < grp * 16 + 16; ++cc) {
            const uint c = bc * TILE + cc;
            if (c >= g.cols) break;
            s = fma(M[(ulong)c * g.ld + r], xs[cc], s);
        }
    }
    part[grp][rl] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < TILE && br * TILE + tid < g.rows)
        P[(ulong)bc * g.rows + br * TILE + tid] = part[0][tid] + part[1][tid] + part[2][tid] + part[3][tid];
}
kernel void bd_gemv_n_reduce(device const float* P [[buffer(0)]], device float* y [[buffer(1)]],
                             constant GemvDims& g [[buffer(2)]], uint r [[thread_position_in_grid]]) {
    if (r >= g.rows) return;
    const uint blocks = (g.cols + TILE - 1) / TILE;
    float v = 0.0f;
    for (uint b = 0; b < blocks; ++b) v += P[(ulong)b * g.rows + r];
    y[r] = v;
}

// One threadgroup per output: dot of a column (or row) segment with v.
//   bd_dots_col: g <  i: t[g]     = Ak(i:, g) . v,   v = Ak(i:, i)
//                g >= i: t[g]     = X(i:, g-i) . v
//   bd_dots_row: g <= i: t[g]     = Y(i+1:, g) . u,  u = Ak(i, i+1:) (stride lda)
//                g >  i: t[g]     = Ak(g-i-1, i+1:) . u
kernel void bd_dots_col(device const float* Ak [[buffer(0)]], device const float* X [[buffer(1)]],
                        device float* t [[buffer(2)]], constant BdParams& p [[buffer(3)]],
                        uint g [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
                        uint nt [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float part[32];
    const uint len = p.mm - p.i;
    device const float* v = Ak + p.i * p.lda + p.i;
    device const float* c = g < p.i ? Ak + g * p.lda + p.i : X + (g - p.i) * p.ldx + p.i;
    float s = 0.0f;
    for (uint r = tid; r < len; r += nt) s = fma(c[r], v[r], s);
    s = simd_sum(s);
    if (lane == 0) part[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) { float q = 0.0f; for (uint u = 0; u < (nt + 31) / 32; ++u) q += part[u]; t[g] = q; }
}
kernel void bd_dots_row(device const float* Ak [[buffer(0)]], device const float* Y [[buffer(1)]],
                        device float* t [[buffer(2)]], constant BdParams& p [[buffer(3)]],
                        uint g [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
                        uint nt [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float part[32];
    const uint len = p.nn - p.i - 1;
    device const float* u = Ak + (p.i + 1) * p.lda + p.i;            // stride lda
    float s = 0.0f;
    if (g <= p.i) {
        device const float* c = Y + g * p.ldy + p.i + 1;
        for (uint r = tid; r < len; r += nt) s = fma(c[r], u[(ulong)r * p.lda], s);
    } else {
        device const float* c = Ak + (p.i + 1) * p.lda + (g - p.i - 1);   // row g-i-1, stride lda
        for (uint r = tid; r < len; r += nt) s = fma(c[(ulong)r * p.lda], u[(ulong)r * p.lda], s);
    }
    s = simd_sum(s);
    if (lane == 0) part[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) { float q = 0.0f; for (uint w = 0; w < (nt + 31) / 32; ++w) q += part[w]; t[g] = q; }
}

kernel void bd_y_corr(device const float* Ak [[buffer(0)]], device float* Y [[buffer(1)]],
                      device const float* t [[buffer(2)]], device const float* tauq [[buffer(3)]],
                      constant BdParams& p [[buffer(4)]], uint r [[thread_position_in_grid]]) {
    const uint c = p.i + 1 + r;
    if (c >= p.nn) return;
    float acc = Y[c + p.i * p.ldy];
    for (uint j = 0; j < p.i; ++j) acc -= Y[c + j * p.ldy] * t[j] + Ak[j + c * p.lda] * t[p.i + j];
    Y[c + p.i * p.ldy] = tauq[p.k + p.i] * acc;
}
kernel void bd_x_corr(device const float* Ak [[buffer(0)]], device float* X [[buffer(1)]],
                      device const float* t [[buffer(2)]], device const float* taup [[buffer(3)]],
                      constant BdParams& p [[buffer(4)]], uint r0 [[thread_position_in_grid]]) {
    const uint r = p.i + 1 + r0;
    if (r >= p.mm) return;
    float acc = X[r + p.i * p.ldx];
    for (uint j = 0; j <= p.i; ++j) acc -= Ak[r + j * p.lda] * t[j];
    for (uint j = 0; j < p.i; ++j)  acc -= X[r + j * p.ldx] * t[p.i + 1 + j];
    X[r + p.i * p.ldx] = taup[p.k + p.i] * acc;
}

// After the panel's trailing update: Ak(j, j) = d, Ak(j, j+1) = e (the units
// were only for the reflectors).
struct RestoreParams { uint nb, lda, k; };
kernel void bd_restore(device float* Ak [[buffer(0)]], device const float* d [[buffer(1)]],
                       device const float* e [[buffer(2)]], constant RestoreParams& q [[buffer(3)]],
                       uint j [[thread_position_in_grid]]) {
    if (j >= q.nb) return;
    Ak[j + j * q.lda] = d[q.k + j];
    Ak[j + (j + 1) * q.lda] = e[q.k + j];
}
