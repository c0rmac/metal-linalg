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
// panel's slabrd matrices, row 0 = Ak's. Column i of the panel, slabrd's
// steps in four kernels, each step that needs a whole vector done before the
// next starts at a kernel boundary (a dispatch costs a few microseconds of GPU
// time even when it does nothing, and a column used to take twelve):
//
//   bd_col     X(i:, i-1) = taup (X - Ak(i:, 0:i) t3 - X(i:, 0:i-1) t4), from
//              bd_gemv_n's partials, finishing the previous column; writes
//              that column's row reflector u back into Ak(i-1, i:); then
//              Ak(i:, i) -= Ak(i:, 0:i) Y(i, 0:i)^T + X(i:, 0:i) Ak(0:i, i),
//              and per-threadgroup partials of the norm of Ak(i+1:, i)
//   bd_gemv_t  the reflector for Ak(i:, i) from those partials (d(i), tauq(i);
//              every threadgroup computes it), then Ak(i:, i+1:)^T v in 64 x 64
//              tiles, and in further threadgroups t1 = Ak(i:, 0:i)^T v and
//              t2 = X(i:, 0:i)^T v
//   bd_row     Y(i+1:, i) = tauq (Y - Y(i+1:, 0:i) t1 - Ak(0:i, i+1:)^T t2) from
//              the tiles' partials; writes v back into Ak(i:, i); then
//              Ak(i, i+1:) -= Y(i+1:, 0:i+1) Ak(i, 0:i+1)^T + Ak(0:i, i+1:)^T X(i, 0:i)^T,
//              and per-threadgroup partials of the norm of Ak(i, i+2:)
//   bd_gemv_n  the reflector for Ak(i, i+1:) (e(i), taup(i)), then
//              Ak(i+1:, i+1:) u in tiles, and t3 = Y(i+1:, 0:i+1)^T u and
//              t4 = Ak(0:i, i+1:) u
//
// After the panel, bd_col (with `last`) finishes the last column before the
// trailing update. The reflectors are formed on the fly (v(0) = 1, then the
// scaled column) until bd_row / bd_col write them back; an element that a
// kernel writes while others read it (a reflector's unit) is read as its
// value, 1. Every sum over threadgroups is taken in a fixed order, so results
// do not depend on scheduling.

constant constexpr uint TILE  = 64;    // the products' tiles
constant constexpr uint GROUP = 256;   // threads per threadgroup
constant constexpr uint LANES = 8;     // bd_col's and bd_row's threads per row: each sums every LANES-th term
constant constexpr uint ROWS  = GROUP / LANES;   // rows per threadgroup

struct BdParams {
    uint mm, nn;     // the trailing block
    uint lda, ldx, ldy;
    uint i;          // column within the panel
    uint k;          // the panel's first column, for d, e, tauq, taup
    uint ng;         // bd_gemv_t / bd_gemv_n: the norm partials of their reflector
    uint tiles;      // bd_gemv_t / bd_gemv_n: tile threadgroups; the dot products' come after
    uint last;       // bd_col: only finish column i - 1 (after the panel)
};

// The sum over a row's LANES consecutive lanes, in a fixed order; every lane
// gets it.
static float row_sum(float x) {
    x += simd_shuffle_xor(x, 4);
    x += simd_shuffle_xor(x, 2);
    x += simd_shuffle_xor(x, 1);
    return x;
}

struct Reflector { float beta, tau, scale; };

// As LAPACK's slarfg, from (max, sum of squares / max^2) partials of x and
// alpha; the same in every simdgroup. The norm is scaled by the largest
// magnitude, so it neither over- nor underflows.
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

// The threadgroup's (max, sum of squares / max^2) of x (0 where not part of
// the vector), to npart[2 g].
static void norm_partial(float ax, device float* npart, uint g, uint t, uint sg, uint lane,
                         threadgroup float* part) {
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

// LANES threads per row r = i + gid / LANES of Ak, which is also column
// i + gid / LANES for the write-back of u (nn <= mm).
kernel void bd_col(device float* Ak [[buffer(0)]], device float* X [[buffer(1)]],
                   device const float* Y [[buffer(2)]], device const float* P [[buffer(3)]],
                   device const float* t [[buffer(4)]], device const float* taup [[buffer(5)]],
                   device const float* scal [[buffer(6)]], device float* npart [[buffer(7)]],
                   constant BdParams& p [[buffer(8)]],
                   uint gid [[thread_position_in_grid]], uint g [[threadgroup_position_in_grid]],
                   uint tt [[thread_position_in_threadgroup]],
                   uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float part[GROUP / 32];
    const uint i = p.i, idx = gid / LANES, q = gid % LANES, r = i + idx;
    const uint lda = p.lda, ldx = p.ldx, ldy = p.ldy;
    float xr = 0.0f;   // X(r, i-1), finished
    if (i > 0) {
        // X(r, i-1) = taup (Ak(i:, i:) u - Ak(r, 0:i) t3 - X(r, 0:i-1) t4); bd_gemv_n
        // for column i - 1 had mm - i rows and nn - i columns.
        const uint c = i - 1, rows = p.mm - i, slots = (p.nn - i + TILE - 1) / TILE;
        float acc = 0.0f;
        if (r < p.mm) {
            for (uint s = q; s < slots; s += LANES) acc += P[(ulong)s * rows + idx];
            for (uint j = q; j <= c; j += LANES) acc -= Ak[r + j * lda] * t[j];
            for (uint j = q; j < c; j += LANES)  acc -= X[r + j * ldx] * t[c + 1 + j];
        }
        xr = taup[p.k + c] * row_sum(acc);
        if (r < p.mm && q == 0) {
            X[r + c * ldx] = xr;
            if (r < p.nn) {   // u(r - i) back into Ak(i-1, r)
                device float* u = Ak + c + (ulong)r * lda;
                *u = idx == 0 ? 1.0f : scal[1] * *u;
            }
        }
    }
    if (p.last) return;

    float x = 0.0f;
    if (i > 0) {
        // Ak(r, i) -= Ak(r, 0:i) Y(i, 0:i)^T + X(r, 0:i) Ak(0:i, i); Ak(i-1, i) = u(0) = 1
        const uint c = i - 1;
        float acc = 0.0f;
        if (r < p.mm)
            for (uint j = q; j < c; j += LANES)
                acc += Ak[r + j * lda] * Y[i + j * ldy] + X[r + j * ldx] * Ak[j + i * lda];
        acc = row_sum(acc);
        if (r < p.mm && q == 0) {
            acc += Ak[r + c * lda] * Y[i + c * ldy] + xr;
            x = Ak[r + i * lda] - acc;
            Ak[r + i * lda] = x;
        }
    } else if (r < p.mm && q == 0) {
        x = Ak[r + i * lda];
    }
    norm_partial(q == 0 && r > i && r < p.mm ? fabs(x) : 0.0f, npart, g, tt, sg, lane, part);
}

// Tile threadgroups first: tile (br, bc) of M = Ak(i:, i+1:) (mm - i rows,
// nn - i - 1 columns), M^T v partials to P[br * cols + c]; then t1, t2.
kernel void bd_gemv_t(device const float* Ak [[buffer(0)]], device const float* X [[buffer(1)]],
                      device float* P [[buffer(2)]], device float* t [[buffer(3)]],
                      device const float* npart [[buffer(4)]], device float* d [[buffer(5)]],
                      device float* tauq [[buffer(6)]], device float* scal [[buffer(7)]],
                      constant BdParams& p [[buffer(8)]],
                      uint g [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
                      uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    const uint i = p.i, lda = p.lda, rows = p.mm - i, cols = p.nn - i - 1;
    device const float* x = Ak + i * lda + i;             // Ak(i:, i): alpha, then the rest
    const Reflector rf = reflector(npart, p.ng, x[0], lane);
    if (g == 0 && tid == 0) {
        d[p.k + i] = rf.beta;
        tauq[p.k + i] = rf.tau;
        scal[0] = rf.scale;
    }

    if (g >= p.tiles) {                                   // t1 = Ak(i:, j)^T v, t2 = X(i:, j)^T v
        threadgroup float part[GROUP / 32];
        const uint j = g - p.tiles;
        device const float* c = j < i ? Ak + j * lda + i : X + (j - i) * p.ldx + i;
        float s = 0.0f;
        for (uint r = tid; r < rows; r += GROUP) s = fma(c[r], r == 0 ? 1.0f : rf.scale * x[r], s);
        s = simd_sum(s);
        if (lane == 0) part[sg] = s;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            float v = 0.0f;
            for (uint u = 0; u < GROUP / 32; ++u) v += part[u];
            t[j] = v;
        }
        return;
    }

    const uint cblocks = (cols + TILE - 1) / TILE, br = g / cblocks, bc = g % cblocks;
    device const float* M = Ak + (i + 1) * lda + i;
    threadgroup float xs[TILE];
    if (tid < TILE) {
        const uint r = br * TILE + tid;
        xs[tid] = r < rows ? (r == 0 ? 1.0f : rf.scale * x[r]) : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint cc = 0; cc < 8; ++cc) {
        const uint c = bc * TILE + sg * 8 + cc;
        if (c >= cols) break;
        device const float* col = M + (ulong)c * lda + br * TILE;
        const uint r0 = lane, r1 = lane + 32;
        float s = (br * TILE + r0 < rows ? col[r0] * xs[r0] : 0.0f) +
                  (br * TILE + r1 < rows ? col[r1] * xs[r1] : 0.0f);
        s = simd_sum(s);
        if (lane == 0) P[(ulong)br * cols + c] = s;
    }
}

// LANES threads per entry e = gid / LANES: column c = i + 1 + e of Y and of
// row i (c < nn), and row i + e for the write-back of v (i + e < mm).
kernel void bd_row(device float* Ak [[buffer(0)]], device const float* X [[buffer(1)]],
                   device float* Y [[buffer(2)]], device const float* P [[buffer(3)]],
                   device const float* t [[buffer(4)]], device const float* tauq [[buffer(5)]],
                   device const float* scal [[buffer(6)]], device float* npart [[buffer(7)]],
                   constant BdParams& p [[buffer(8)]],
                   uint gid [[thread_position_in_grid]], uint g [[threadgroup_position_in_grid]],
                   uint tt [[thread_position_in_threadgroup]],
                   uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float part[GROUP / 32];
    const uint i = p.i, e = gid / LANES, q = gid % LANES, c = i + 1 + e;
    const uint lda = p.lda, ldx = p.ldx, ldy = p.ldy;
    const uint cols = p.nn - i - 1, slots = (p.mm - i + TILE - 1) / TILE;
    const bool col = c < p.nn;

    // Y(c, i) = tauq (Ak(i:, c)^T v - Y(c, 0:i) t1 - Ak(0:i, c)^T t2)
    float acc = 0.0f;
    if (col) {
        for (uint s = q; s < slots; s += LANES) acc += P[(ulong)s * cols + e];
        for (uint j = q; j < i; j += LANES) acc -= Y[c + j * ldy] * t[j] + Ak[j + c * lda] * t[i + j];
    }
    const float y = tauq[p.k + i] * row_sum(acc);

    // Ak(i, c) -= Y(c, 0:i+1) Ak(i, 0:i+1)^T + Ak(0:i, c)^T X(i, 0:i)^T; Ak(i, i) = v(0) = 1
    acc = 0.0f;
    if (col)
        for (uint j = q; j < i; j += LANES) acc += Y[c + j * ldy] * Ak[i + j * lda] + Ak[j + c * lda] * X[i + j * ldx];
    acc = row_sum(acc) + y;
    float x = 0.0f;
    if (q == 0) {
        if (col) {
            Y[c + i * ldy] = y;
            x = Ak[i + c * lda] - acc;
            Ak[i + c * lda] = x;
        }
        if (i + e < p.mm) {   // v(e) back into Ak(i + e, i)
            device float* v = Ak + i + e + (ulong)i * lda;
            *v = e == 0 ? 1.0f : scal[0] * *v;
        }
    }
    norm_partial(q == 0 && col && e > 0 ? fabs(x) : 0.0f, npart, g, tt, sg, lane, part);
}

// Tile threadgroups first: tile (br, bc) of M = Ak(i+1:, i+1:) (mm - i - 1
// rows, nn - i - 1 columns), M u partials to P[bc * rows + r]; then t3, t4.
// Each tile's 256 threads: 64 rows x 4 groups of 16 columns.
kernel void bd_gemv_n(device const float* Ak [[buffer(0)]], device const float* Y [[buffer(1)]],
                      device float* P [[buffer(2)]], device float* t [[buffer(3)]],
                      device const float* npart [[buffer(4)]], device float* e [[buffer(5)]],
                      device float* taup [[buffer(6)]], device float* scal [[buffer(7)]],
                      constant BdParams& p [[buffer(8)]],
                      uint g [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
                      uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    const uint i = p.i, lda = p.lda, rows = p.mm - i - 1, cols = p.nn - i - 1;
    device const float* x = Ak + (i + 1) * lda + i;       // Ak(i, i+1:), stride lda: alpha, then the rest
    const Reflector rf = reflector(npart, p.ng, x[0], lane);
    if (g == 0 && tid == 0) {
        e[p.k + i] = rf.beta;
        taup[p.k + i] = rf.tau;
        scal[1] = rf.scale;
    }

    if (g >= p.tiles) {   // t3 = Y(i+1:, j)^T u (j <= i), t4 = Ak(j, i+1:) u (j < i)
        threadgroup float part[GROUP / 32];
        const uint j = g - p.tiles;
        float s = 0.0f;
        if (j <= i) {
            device const float* c = Y + j * p.ldy + i + 1;
            for (uint r = tid; r < cols; r += GROUP) s = fma(c[r], r == 0 ? 1.0f : rf.scale * x[(ulong)r * lda], s);
        } else {
            device const float* c = Ak + (i + 1) * lda + (j - i - 1);
            for (uint r = tid; r < cols; r += GROUP)
                s = fma(c[(ulong)r * lda], r == 0 ? 1.0f : rf.scale * x[(ulong)r * lda], s);
        }
        s = simd_sum(s);
        if (lane == 0) part[sg] = s;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            float v = 0.0f;
            for (uint u = 0; u < GROUP / 32; ++u) v += part[u];
            t[j] = v;
        }
        return;
    }

    const uint cblocks = (cols + TILE - 1) / TILE, br = g / cblocks, bc = g % cblocks;
    device const float* M = Ak + (i + 1) * lda + i + 1;
    threadgroup float xs[TILE];
    threadgroup float part[4][TILE];
    if (tid < TILE) {
        const uint c = bc * TILE + tid;
        xs[tid] = c < cols ? (c == 0 ? 1.0f : rf.scale * x[(ulong)c * lda]) : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint rl = tid % TILE, grp = tid / TILE, r = br * TILE + rl;
    float s = 0.0f;
    if (r < rows) {
        for (uint cc = grp * 16; cc < grp * 16 + 16; ++cc) {
            const uint c = bc * TILE + cc;
            if (c >= cols) break;
            s = fma(M[(ulong)c * lda + r], xs[cc], s);
        }
    }
    part[grp][rl] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < TILE && br * TILE + tid < rows)
        P[(ulong)bc * rows + br * TILE + tid] = part[0][tid] + part[1][tid] + part[2][tid] + part[3][tid];
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
