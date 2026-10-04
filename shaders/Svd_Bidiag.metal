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

// =============================================================================
// Two-stage reduction, stage 1, for singular values alone: A to an upper band
// of width b (b <= 32) by blocks of b columns, the bulk of the work as matrix
// products (src/svd_bidiag.mm). bd_panel_qr factors one panel; the products
// that apply it are MPS GEMMs.
// =============================================================================

struct PanelParams {
    uint rows, cols;   // rows >= 2 cols; cols is the kernel's B (8, 16 or 32)
    uint rs, cs;       // P(i, j) = P[i * rs + j * cs]
    uint ldv;          // V row-major, ld ldv
    uint flags;        // 1: V T into VT (ld 32); 2: V^T, rows i >= shift, into Vtr (ld ldt);
                       // 4: before factoring, P(i, :) -= W(i, :) V0^T (the row panel's own update);
                       // 8: V a second time, at Vout + dup
    uint shift, ldt;
    uint leaf;         // rows per TSQR leaf
    uint ldw;          // W's ld (flag 4)
    uint dup;          // flag 8: V again at Vout + dup
};

// Merges (max, sum of squares / max^2) pairs.
__attribute__((always_inline)) static float2 merge_ss(float2 a, float2 b) {
    if (b.x > a.x) { const float f = a.x / b.x; return float2(b.x, a.y * f * f + b.y); }
    if (b.x > 0.0f) { const float f = b.x / a.x; return float2(a.x, a.y + b.y * f * f); }
    return a;
}

// The panel kernels are templates on B = cols, the width of each thread's row
// in registers. A row must only ever be indexed by constants to stay in
// registers (indexed by a variable, a thread's array goes to memory, many
// times slower), and the compiler does not unroll loops that hold barriers or
// simdgroup reductions, so every loop over a row is unrolled by hand.
// Every loop over a row is expanded by the preprocessor, UNROLL_c(...) being
// its body 32 times with c the constant 0 .. 31 (bodies test c < B, which
// folds away; MSL before 4 has no lambdas to do it with templates).
#define UNROLL_c(...) { { constexpr uint c = 0; __VA_ARGS__ } { constexpr uint c = 1; __VA_ARGS__ } { constexpr uint c = 2; __VA_ARGS__ } { constexpr uint c = 3; __VA_ARGS__ } { constexpr uint c = 4; __VA_ARGS__ } { constexpr uint c = 5; __VA_ARGS__ } { constexpr uint c = 6; __VA_ARGS__ } { constexpr uint c = 7; __VA_ARGS__ } { constexpr uint c = 8; __VA_ARGS__ } { constexpr uint c = 9; __VA_ARGS__ } { constexpr uint c = 10; __VA_ARGS__ } { constexpr uint c = 11; __VA_ARGS__ } { constexpr uint c = 12; __VA_ARGS__ } { constexpr uint c = 13; __VA_ARGS__ } { constexpr uint c = 14; __VA_ARGS__ } { constexpr uint c = 15; __VA_ARGS__ } { constexpr uint c = 16; __VA_ARGS__ } { constexpr uint c = 17; __VA_ARGS__ } { constexpr uint c = 18; __VA_ARGS__ } { constexpr uint c = 19; __VA_ARGS__ } { constexpr uint c = 20; __VA_ARGS__ } { constexpr uint c = 21; __VA_ARGS__ } { constexpr uint c = 22; __VA_ARGS__ } { constexpr uint c = 23; __VA_ARGS__ } { constexpr uint c = 24; __VA_ARGS__ } { constexpr uint c = 25; __VA_ARGS__ } { constexpr uint c = 26; __VA_ARGS__ } { constexpr uint c = 27; __VA_ARGS__ } { constexpr uint c = 28; __VA_ARGS__ } { constexpr uint c = 29; __VA_ARGS__ } { constexpr uint c = 30; __VA_ARGS__ } { constexpr uint c = 31; __VA_ARGS__ } }
#define UNROLL_j(...) { { constexpr uint j = 0; __VA_ARGS__ } { constexpr uint j = 1; __VA_ARGS__ } { constexpr uint j = 2; __VA_ARGS__ } { constexpr uint j = 3; __VA_ARGS__ } { constexpr uint j = 4; __VA_ARGS__ } { constexpr uint j = 5; __VA_ARGS__ } { constexpr uint j = 6; __VA_ARGS__ } { constexpr uint j = 7; __VA_ARGS__ } { constexpr uint j = 8; __VA_ARGS__ } { constexpr uint j = 9; __VA_ARGS__ } { constexpr uint j = 10; __VA_ARGS__ } { constexpr uint j = 11; __VA_ARGS__ } { constexpr uint j = 12; __VA_ARGS__ } { constexpr uint j = 13; __VA_ARGS__ } { constexpr uint j = 14; __VA_ARGS__ } { constexpr uint j = 15; __VA_ARGS__ } { constexpr uint j = 16; __VA_ARGS__ } { constexpr uint j = 17; __VA_ARGS__ } { constexpr uint j = 18; __VA_ARGS__ } { constexpr uint j = 19; __VA_ARGS__ } { constexpr uint j = 20; __VA_ARGS__ } { constexpr uint j = 21; __VA_ARGS__ } { constexpr uint j = 22; __VA_ARGS__ } { constexpr uint j = 23; __VA_ARGS__ } { constexpr uint j = 24; __VA_ARGS__ } { constexpr uint j = 25; __VA_ARGS__ } { constexpr uint j = 26; __VA_ARGS__ } { constexpr uint j = 27; __VA_ARGS__ } { constexpr uint j = 28; __VA_ARGS__ } { constexpr uint j = 29; __VA_ARGS__ } { constexpr uint j = 30; __VA_ARGS__ } { constexpr uint j = 31; __VA_ARGS__ } }
#define UNROLL_k(...) { { constexpr uint k = 0; __VA_ARGS__ } { constexpr uint k = 1; __VA_ARGS__ } { constexpr uint k = 2; __VA_ARGS__ } { constexpr uint k = 3; __VA_ARGS__ } { constexpr uint k = 4; __VA_ARGS__ } { constexpr uint k = 5; __VA_ARGS__ } { constexpr uint k = 6; __VA_ARGS__ } { constexpr uint k = 7; __VA_ARGS__ } { constexpr uint k = 8; __VA_ARGS__ } { constexpr uint k = 9; __VA_ARGS__ } { constexpr uint k = 10; __VA_ARGS__ } { constexpr uint k = 11; __VA_ARGS__ } { constexpr uint k = 12; __VA_ARGS__ } { constexpr uint k = 13; __VA_ARGS__ } { constexpr uint k = 14; __VA_ARGS__ } { constexpr uint k = 15; __VA_ARGS__ } { constexpr uint k = 16; __VA_ARGS__ } { constexpr uint k = 17; __VA_ARGS__ } { constexpr uint k = 18; __VA_ARGS__ } { constexpr uint k = 19; __VA_ARGS__ } { constexpr uint k = 20; __VA_ARGS__ } { constexpr uint k = 21; __VA_ARGS__ } { constexpr uint k = 22; __VA_ARGS__ } { constexpr uint k = 23; __VA_ARGS__ } { constexpr uint k = 24; __VA_ARGS__ } { constexpr uint k = 25; __VA_ARGS__ } { constexpr uint k = 26; __VA_ARGS__ } { constexpr uint k = 27; __VA_ARGS__ } { constexpr uint k = 28; __VA_ARGS__ } { constexpr uint k = 29; __VA_ARGS__ } { constexpr uint k = 30; __VA_ARGS__ } { constexpr uint k = 31; __VA_ARGS__ } }
#define UNROLL_l(...) { { constexpr uint l = 0; __VA_ARGS__ } { constexpr uint l = 1; __VA_ARGS__ } { constexpr uint l = 2; __VA_ARGS__ } { constexpr uint l = 3; __VA_ARGS__ } { constexpr uint l = 4; __VA_ARGS__ } { constexpr uint l = 5; __VA_ARGS__ } { constexpr uint l = 6; __VA_ARGS__ } { constexpr uint l = 7; __VA_ARGS__ } { constexpr uint l = 8; __VA_ARGS__ } { constexpr uint l = 9; __VA_ARGS__ } { constexpr uint l = 10; __VA_ARGS__ } { constexpr uint l = 11; __VA_ARGS__ } { constexpr uint l = 12; __VA_ARGS__ } { constexpr uint l = 13; __VA_ARGS__ } { constexpr uint l = 14; __VA_ARGS__ } { constexpr uint l = 15; __VA_ARGS__ } { constexpr uint l = 16; __VA_ARGS__ } { constexpr uint l = 17; __VA_ARGS__ } { constexpr uint l = 18; __VA_ARGS__ } { constexpr uint l = 19; __VA_ARGS__ } { constexpr uint l = 20; __VA_ARGS__ } { constexpr uint l = 21; __VA_ARGS__ } { constexpr uint l = 22; __VA_ARGS__ } { constexpr uint l = 23; __VA_ARGS__ } { constexpr uint l = 24; __VA_ARGS__ } { constexpr uint l = 25; __VA_ARGS__ } { constexpr uint l = 26; __VA_ARGS__ } { constexpr uint l = 27; __VA_ARGS__ } { constexpr uint l = 28; __VA_ARGS__ } { constexpr uint l = 29; __VA_ARGS__ } { constexpr uint l = 30; __VA_ARGS__ } { constexpr uint l = 31; __VA_ARGS__ } }
#define UNROLL_s(...) { { constexpr uint s = 0; __VA_ARGS__ } { constexpr uint s = 1; __VA_ARGS__ } { constexpr uint s = 2; __VA_ARGS__ } { constexpr uint s = 3; __VA_ARGS__ } { constexpr uint s = 4; __VA_ARGS__ } { constexpr uint s = 5; __VA_ARGS__ } { constexpr uint s = 6; __VA_ARGS__ } { constexpr uint s = 7; __VA_ARGS__ } }
#define UNROLL(N, v, ...) UNROLL_##v(if (v < N) __VA_ARGS__)

// slarft's recurrence for T, T(0:j, j) = -tau_j T(0:j, 0:j) D(j, 0:j), with
// D(j, c) = V(:, c)^T V(:, j): a row per lane of simdgroup 0, each lane
// reading only its own row of T.
template <uint B>
__attribute__((always_inline)) static void form_t(threadgroup float (*Tm)[32], threadgroup float (*D)[32],
                                                   threadgroup float* taus, uint t) {
    if (t < B) {
        const uint r = t;
        for (uint c = 0; c < r; ++c) Tm[r][c] = 0.0f;
        Tm[r][r] = taus[r];
        for (uint j = r + 1; j < B; ++j) {
            float s = 0.0f;
            for (uint l = r; l < j; ++l) s = fma(Tm[r][l], D[j][l], s);
            Tm[r][j] = -taus[j] * s;
        }
    }
}

// Householder QR of `rows` rows (rows <= threads), one row per thread in x,
// for the TSQR's top (a few hundred rows): on return, as LAPACK leaves it,
// thread `row` holds R(row, c) for c >= row (row < B) and the reflectors'
// V(row, c) for c < row; T (upper triangular) in Tm. The column loop is not
// unrolled (B copies of its body would be more than the compiler takes in
// reasonable time); instead each thread's row turns one place left a column,
// so that column j is always x[0] and column (j + k) mod B is x[k], and only
// the loops over k are unrolled; after B turns the row is back in place. Two
// barriers a column: the simdgroups' partials of the norm, then of the dot
// products, each read once a simdgroup (a lane a partial) and passed round
// by shuffles. They have a buffer each: a column's writes to one come after
// the barrier that follows the last reads of the other column's.
template <uint B>
__attribute__((always_inline)) static void qr_rows(thread float (&x)[B], uint row, uint rows,
                                                    threadgroup float (*Tm)[32], threadgroup float (*red)[32][33],
                                                    threadgroup float (*D)[32], threadgroup float* bcast,
                                                    threadgroup float* taus, uint t, uint nt, uint sg, uint lane) {
    const uint nsg = (nt + 31) / 32;
    threadgroup float (*part)[33] = red[0];
    threadgroup float (*dp)[33] = red[1];
    for (uint j = 0; j < B; ++j) {
        // ||x(j+1:)||, scaled by its largest entry; alpha = x(j)
        const float ax = row > j && row < rows ? fabs(x[0]) : 0.0f;
        float2 ms = float2(ax, ax > 0.0f ? 1.0f : 0.0f);
        for (uint o = 16; o > 0; o >>= 1)
            ms = merge_ss(ms, float2(simd_shuffle_xor(ms.x, o), simd_shuffle_xor(ms.y, o)));
        if (lane == 0) { part[sg][0] = ms.x; part[sg][1] = ms.y; }
        if (row == j) bcast[0] = x[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float2 all = lane < nsg ? float2(part[lane][0], part[lane][1]) : float2(0.0f, 0.0f);
        for (uint o = 16; o > 0; o >>= 1)
            all = merge_ss(all, float2(simd_shuffle_xor(all.x, o), simd_shuffle_xor(all.y, o)));
        const float alpha = bcast[0], xnorm = all.x * sqrt(all.y);
        float beta = alpha, tau = 0.0f, scale = 1.0f;
        if (xnorm != 0.0f) {
            const float big = max(fabs(alpha), xnorm), ra = alpha / big, rx = xnorm / big;
            beta = -copysign(big * sqrt(ra * ra + rx * rx), alpha);
            tau = (beta - alpha) / beta;
            scale = 1.0f / (alpha - beta);
        }
        float v = 0.0f;
        if (row == j) { x[0] = beta; v = 1.0f; }
        else if (row > j && row < rows) { x[0] *= scale; v = x[0]; }
        // v . x(:, c) for the other columns: the update of the columns right
        // of j, and V(:, c)^T v for T (c < j)
        UNROLL(B, k, {
            if (k > 0) {
                const float s = simd_sum(v * x[k]);
                if (lane == 0) dp[sg][k] = s;
            }
        });
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (t == 0) taus[j] = tau;
        float mine = 0.0f;   // lane k: the column at k's dot product
        if (lane < B)
            for (uint q = 0; q < nsg; ++q) mine += dp[q][lane];
        UNROLL(B, k, {
            if (k > 0) {
                const float d = simd_shuffle(mine, (ushort)k);
                if (j + k < B) x[k] -= tau * d * v;
                else if (t == 0) D[j][j + k - B] = d;
            }
        });
        const float x0 = x[0];
        UNROLL(B, k, { if (k + 1 < B) x[k] = x[k + 1]; });
        x[B - 1] = x0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    form_t<B>(Tm, D, taus, t);
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// The same QR in one simdgroup, R rows a lane (row s * 32 + lane in x[s]), so
// with no threadgroup barrier at all: a TSQR leaf's, or a short panel's.
template <uint B, uint R>
__attribute__((always_inline)) static void qr_simd(thread float (&x)[R][B], uint rows, uint lane,
                                                    threadgroup float (*Tm)[32], threadgroup float (*D)[32],
                                                    threadgroup float* taus) {
    for (uint j = 0; j < B; ++j) {
        float2 ms = float2(0.0f, 0.0f);
        UNROLL(R, s, {
            const uint row = s * 32 + lane;
            const float ax = row > j && row < rows ? fabs(x[s][0]) : 0.0f;
            ms = merge_ss(ms, float2(ax, ax > 0.0f ? 1.0f : 0.0f));
        });
        for (uint o = 16; o > 0; o >>= 1)
            ms = merge_ss(ms, float2(simd_shuffle_xor(ms.x, o), simd_shuffle_xor(ms.y, o)));
        const float alpha = simd_shuffle(x[0][0], (ushort)j), xnorm = ms.x * sqrt(ms.y);
        float beta = alpha, tau = 0.0f, scale = 1.0f;
        if (xnorm != 0.0f) {
            const float big = max(fabs(alpha), xnorm), ra = alpha / big, rx = xnorm / big;
            beta = -copysign(big * sqrt(ra * ra + rx * rx), alpha);
            tau = (beta - alpha) / beta;
            scale = 1.0f / (alpha - beta);
        }
        float v[R];
        UNROLL(R, s, {
            const uint row = s * 32 + lane;
            v[s] = 0.0f;
            if (row == j) { x[s][0] = beta; v[s] = 1.0f; }
            else if (row > j && row < rows) { x[s][0] *= scale; v[s] = x[s][0]; }
        });
        if (lane == 0) taus[j] = tau;
        UNROLL(B, k, {
            if (k > 0) {
                float local = 0.0f;
                UNROLL(R, s, { local = fma(v[s], x[s][k], local); });
                const float d = simd_sum(local);
                if (j + k < B) { UNROLL(R, s, { x[s][k] -= tau * d * v[s]; }); }
                else if (lane == 0) D[j][j + k - B] = d;
            }
        });
        UNROLL(R, s, {
            const float x0 = x[s][0];
            UNROLL(B, k, { if (k + 1 < B) x[s][k] = x[s][k + 1]; });
            x[s][B - 1] = x0;
        });
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    form_t<B>(Tm, D, taus, lane);
    simdgroup_barrier(mem_flags::mem_threadgroup);
}

// Loads row i of the panel into x, applying flag 4's update.
template <uint B>
__attribute__((always_inline)) static void load_row(device const float* P, device const float* W,
                                                     device const float* V0, constant PanelParams& p, uint i,
                                                     thread float (&x)[B]) {
    UNROLL(B, c, { x[c] = P[i * p.rs + c * p.cs]; });
    if (p.flags & 4u) {
        float w[B];
        UNROLL(B, l, { w[l] = W[(ulong)i * p.ldw + l]; });
        UNROLL(B, c, {
            float s = 0.0f;
            UNROLL(B, l, { s = fma(w[l], V0[c * 32 + l], s); });
            x[c] -= s;
        });
    }
}

// Writes the explicit V row i (and V T, V^T as flags ask) of the panel's H.
template <uint B>
__attribute__((always_inline)) static void store_v(thread const float (&v)[B], uint i, constant PanelParams& p,
                                                    device float* Vout, device float* VT, device float* Vtr,
                                                    threadgroup float (*Tm)[32]) {
    UNROLL(B, c, { Vout[(ulong)i * p.ldv + c] = v[c]; });
    if (p.flags & 8u) UNROLL(B, c, { Vout[(ulong)i * p.ldv + p.dup + c] = v[c]; });
    if (p.flags & 1u)
        UNROLL(B, c, {
            float s = 0.0f;
            UNROLL(B, l, { if (l <= c) s = fma(v[l], Tm[l][c], s); });
            VT[(ulong)i * 32 + c] = s;
        });
    if ((p.flags & 2u) && i >= p.shift)
        UNROLL(B, c, { Vtr[(ulong)c * p.ldt + i - p.shift] = v[c]; });
}

// Rows a lane in the simdgroup kernels: a short panel or a TSQR leaf has at
// most 32 * PANEL_R rows.
constant constexpr uint PANEL_R = 4;

// A short panel (at most 32 * PANEL_R rows), one simdgroup: QR in registers;
// R in place, H's V (and V T, V^T) and T out.
template <uint B>
kernel void bd_panel_qr(device float* P [[buffer(0)]], device float* Vout [[buffer(1)]],
                        device float* VT [[buffer(2)]], device float* Vtr [[buffer(3)]],
                        device float* Tout [[buffer(4)]], constant PanelParams& p [[buffer(5)]],
                        device const float* W [[buffer(6)]], device const float* V0 [[buffer(7)]],
                        uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float Tm[32][32], D[32][32], taus[32];
    float x[PANEL_R][B];
    UNROLL(PANEL_R, s, {
        const uint row = s * 32 + lane;
        UNROLL(B, c, { x[s][c] = 0.0f; });
        if (row < p.rows) load_row<B>(P, W, V0, p, row, x[s]);
    });
    qr_simd<B, PANEL_R>(x, p.rows, lane, Tm, D, taus);
    UNROLL(PANEL_R, s, {
        const uint row = s * 32 + lane;
        if (row < p.rows) {
            float v[B];
            UNROLL(B, c, {
                v[c] = row > c ? x[s][c] : (row == c ? 1.0f : 0.0f);
                if (row <= c) P[row * p.rs + c * p.cs] = x[s][c];   // R
            });
            store_v<B>(v, row, p, Vout, VT, Vtr, Tm);
        }
    });
    for (uint q = lane; q < B * B; q += 32) Tout[(q / B) * 32 + q % B] = Tm[q / B][q % B];
}

// A taller panel: TSQR (Demmel, Grigori, Hoemmen and Langou), then Householder
// vectors rebuilt from its Q (Ballard, Demmel, Grigori, Jacquelin, Knight and
// Nguyen), so that the update is the same I - V T V^T as for a short panel.
// Leaves of p.leaf rows (at least B each), one per threadgroup. Scratch S:
// the leaves' V (the panel's rows, ld 32), then per leaf T_l, R_l and E_l
// (32 x 32 each), then L1 and U^{-1}.
struct TsqrLayout { uint sT, sR, sE, sL1, sUi; };
static TsqrLayout tsqr_layout(constant PanelParams& p) {
    const uint nl = (p.rows + p.leaf - 1) / p.leaf;
    TsqrLayout L;
    L.sT = p.rows * 32;
    L.sR = L.sT + nl * 32 * 32;
    L.sE = L.sR + nl * 32 * 32;
    L.sL1 = L.sE + nl * 32 * 32;
    L.sUi = L.sL1 + 32 * 32;
    return L;
}

// One leaf of rows (at most 32 * PANEL_R) per simdgroup-sized threadgroup:
// its QR; V_l, T_l and R_l to the scratch.
template <uint B>
kernel void bd_tsqr_leaf(device const float* P [[buffer(0)]], device float* S [[buffer(1)]],
                         constant PanelParams& p [[buffer(2)]], device const float* W [[buffer(3)]],
                         device const float* V0 [[buffer(4)]],
                         uint l [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float Tm[32][32], D[32][32], taus[32];
    const TsqrLayout L = tsqr_layout(p);
    const uint r0 = l * p.leaf, rows = min(p.leaf, p.rows - r0);
    float x[PANEL_R][B];
    UNROLL(PANEL_R, s, {
        const uint row = s * 32 + lane;
        UNROLL(B, c, { x[s][c] = 0.0f; });
        if (row < rows) load_row<B>(P, W, V0, p, r0 + row, x[s]);
    });
    qr_simd<B, PANEL_R>(x, rows, lane, Tm, D, taus);
    UNROLL(PANEL_R, s, {
        const uint row = s * 32 + lane;
        if (row < rows)
            UNROLL(B, c, {
                S[(ulong)(r0 + row) * 32 + c] = row > c ? x[s][c] : (row == c ? 1.0f : 0.0f);
                if (row < B) S[L.sR + (l * 32 + row) * 32 + c] = row <= c ? x[s][c] : 0.0f;
            });
    });
    for (uint q = lane; q < 32 * 32; q += 32) S[L.sT + l * 32 * 32 + q] = q / 32 < B && q % 32 < B ? Tm[q / 32][q % 32] : 0.0f;
}

// One threadgroup: the QR of the stacked R_l; the panel's R; E, the stacked
// blocks of the first B columns of that QR's Q; the top B rows of the
// panel's Q, Q1, and their LU with the signs that keep it stable,
// Q1 - S = L1 U; then T = -U S L1^{-T} and R_H = S R (into the panel), L1 and
// U^{-1} for bd_tsqr_rebuild.
template <uint B>
kernel void bd_tsqr_top(device float* P [[buffer(0)]], device float* S [[buffer(1)]],
                        device float* Tout [[buffer(2)]], constant PanelParams& p [[buffer(3)]],
                        uint t [[thread_position_in_threadgroup]], uint nt [[threads_per_threadgroup]],
                        uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[2][32][33];
    threadgroup float Tm[32][32], D[32][32], taus[32];
    threadgroup float bcast[1];
    threadgroup float A[32][33], Bm[32][33], C[32][33], sgn[32];
    const TsqrLayout L = tsqr_layout(p);
    const uint b = B, nl = (p.rows + p.leaf - 1) / p.leaf, rows = nl * b;
    float x[B];
    UNROLL(B, c, { x[c] = 0.0f; });
    if (t < rows) {
        const uint l = t / b, r = t % b;
        UNROLL(B, c, { x[c] = S[L.sR + (l * 32 + r) * 32 + c]; });
    }
    qr_rows<B>(x, t, rows, Tm, red, D, bcast, taus, t, nt, sg, lane);
    // M = T V(0:b, :)^T, with V(0:b, :) (unit lower) in A
    if (t < b) UNROLL(B, c, { A[t][c] = t > c ? x[c] : (t == c ? 1.0f : 0.0f); });
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint q = t; q < b * b; q += nt) {
        const uint r = q / b, c = q % b;
        float s = 0.0f;
        for (uint k = r; k < b; ++k) s = fma(Tm[r][k], A[c][k], s);
        Bm[r][c] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // E(t, :) = I(t, :) - V(t, :) M
    if (t < rows)
        UNROLL(B, c, {
            float s = 0.0f;
            UNROLL(B, k, { s = fma(t > k ? x[k] : (t == k ? 1.0f : 0.0f), Bm[k][c], s); });
            const float e = (t == c ? 1.0f : 0.0f) - s;
            S[L.sE + t * 32 + c] = e;
            if (t < b) C[t][c] = e;   // E_0
        });
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    // Q1 = E_0 - V_0(0:b, :) T_0 (V_0(0:b, :)^T E_0), with A = V_0's top, Tm = T_0
    for (uint q = t; q < b * b; q += nt) {
        const uint r = q / b, c = q % b;
        A[r][c] = S[(ulong)r * 32 + c];
        Tm[r][c] = S[L.sT + r * 32 + c];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint q = t; q < b * b; q += nt) {   // Bm = V_0^T E_0
        const uint r = q / b, c = q % b;
        float s = 0.0f;
        for (uint k = 0; k < b; ++k) s = fma(A[k][r], C[k][c], s);
        Bm[r][c] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint q = t; q < b * b; q += nt) {   // Q1, in place in C
        const uint r = q / b, c = q % b;
        float s = 0.0f;
        for (uint k = 0; k < b; ++k) {
            float tb = 0.0f;
            for (uint u = k; u < b; ++u) tb = fma(Tm[k][u], Bm[u][c], tb);
            s = fma(A[r][k], tb, s);
        }
        C[r][c] -= s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // The rest is b x b work, done by simdgroup 0 alone, a lane a row or a
    // column, so that its steps need no threadgroup barrier. LU of Q1 - S
    // without pivoting, in C: s_j = -sign(the pivot), so that every pivot is
    // at least 1 in magnitude.
    if (sg != 0) return;
    for (uint j = 0; j < b; ++j) {
        if (lane == 0) {
            const float d = C[j][j], s = d >= 0.0f ? -1.0f : 1.0f;
            sgn[j] = s;
            C[j][j] = d - s;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (lane > j && lane < b) {
            const float l = C[lane][j] / C[j][j];
            C[lane][j] = l;
            for (uint c = j + 1; c < b; ++c) C[lane][c] -= l * C[j][c];
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    // L1 (unit lower, below C's diagonal), U (C's upper triangle): L1^{-1}
    // into A, U^{-1} into Bm, a column a lane
    if (lane < b) {
        for (uint r = 0; r < b; ++r) {
            float s = r == lane ? 1.0f : 0.0f;
            for (uint k = lane; k < r; ++k) s -= C[r][k] * A[k][lane];
            A[r][lane] = r < lane ? 0.0f : s;
        }
        for (int r = (int)b - 1; r >= 0; --r) {
            float s = (uint)r == lane ? 1.0f : 0.0f;
            for (uint k = r + 1; k < b; ++k) s -= C[r][k] * Bm[k][lane];
            Bm[r][lane] = (uint)r > lane ? 0.0f : s / C[r][r];
        }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    // T_H = -U S L1^{-T}: T(r, c) = -sum_{r <= k <= c} U(r, k) s_k L1^{-1}(c, k)
    for (uint q = lane; q < b * b; q += 32) {
        const uint r = q / b, c = q % b;
        float s = 0.0f;
        for (uint k = r; k <= c; ++k) s = fma(C[r][k] * sgn[k], A[c][k], s);
        Tout[r * 32 + c] = r <= c ? -s : 0.0f;
        S[L.sL1 + r * 32 + c] = r > c ? C[r][c] : (r == c ? 1.0f : 0.0f);
        S[L.sUi + r * 32 + c] = Bm[r][c];
    }
    if (t < b)   // R_H = S R
        UNROLL(B, c, { if (c >= t) P[t * p.rs + c * p.cs] = sgn[t] * x[c]; });
}

// The panel's V, rebuilt: row i in leaf l, with E_l its block of E and
// Q(i, :) = E_l(i, :) - V_l(i, :) T_l (V_l(0:b, :)^T E_l) the panel's Q;
// V(i, :) = L1(i, :) for i < b, else Q(i, :) U^{-1}. One leaf a threadgroup.
template <uint B>
kernel void bd_tsqr_rebuild(device const float* S [[buffer(0)]], device float* Vout [[buffer(1)]],
                            device float* VT [[buffer(2)]], device float* Vtr [[buffer(3)]],
                            device const float* Tin [[buffer(4)]], constant PanelParams& p [[buffer(5)]],
                            uint l [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]],
                            uint nt [[threads_per_threadgroup]]) {
    threadgroup float A[32][33], E[32][33], M[32][33], Tm[32][32], Ui[32][33], L1[32][33];
    const TsqrLayout L = tsqr_layout(p);
    const uint b = B, r0 = l * p.leaf, rows = min(p.leaf, p.rows - r0);
    for (uint q = t; q < b * b; q += nt) {
        const uint r = q / b, c = q % b;
        A[r][c] = S[(ulong)(r0 + r) * 32 + c];               // V_l(0:b, :)
        E[r][c] = S[L.sE + (l * b + r) * 32 + c];             // E_l
        Ui[r][c] = S[L.sUi + r * 32 + c];
        L1[r][c] = S[L.sL1 + r * 32 + c];
    }
    for (uint q = t; q < 32 * 32; q += nt) Tm[q / 32][q % 32] = S[L.sT + l * 32 * 32 + q];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint q = t; q < b * b; q += nt) {   // M = T_l (V_l(0:b, :)^T E_l)
        const uint r = q / b, c = q % b;
        float s = 0.0f;
        for (uint k = r; k < b; ++k) {
            float g = 0.0f;
            for (uint u = 0; u < b; ++u) g = fma(A[u][k], E[u][c], g);
            s = fma(Tm[r][k], g, s);
        }
        M[r][c] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint q = t; q < 32 * 32; q += nt) Tm[q / 32][q % 32] = Tin[q];   // T_H, for V T
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t >= rows) return;
    const uint i = r0 + t;
    float v[B];
    if (i < b) {
        UNROLL(B, c, { v[c] = L1[i][c]; });
    } else {
        float vl[B], q[B];
        UNROLL(B, c, { vl[c] = S[(ulong)i * 32 + c]; });
        UNROLL(B, c, {
            float s = t < b ? E[t][c] : 0.0f;
            UNROLL(B, k, { s -= vl[k] * M[k][c]; });
            q[c] = s;
        });
        UNROLL(B, c, {
            float s = 0.0f;
            UNROLL(B, k, { if (k <= c) s = fma(q[k], Ui[k][c], s); });
            v[c] = s;
        });
    }
    store_v<B>(v, i, p, Vout, VT, Vtr, Tm);
}

#define BD_PANEL_KERNELS(B)                                                                                     \
    template [[host_name("bd_panel_qr_" #B)]] kernel void bd_panel_qr<B>(                                       \
        device float*, device float*, device float*, device float*, device float*, constant PanelParams&,      \
        device const float*, device const float*, uint);                                                       \
    template [[host_name("bd_tsqr_leaf_" #B)]] kernel void bd_tsqr_leaf<B>(                                     \
        device const float*, device float*, constant PanelParams&, device const float*, device const float*,   \
        uint, uint);                                                                                            \
    template [[host_name("bd_tsqr_top_" #B)]] kernel void bd_tsqr_top<B>(                                       \
        device float*, device float*, device float*, constant PanelParams&, uint, uint, uint, uint);           \
    template [[host_name("bd_tsqr_rebuild_" #B)]] kernel void bd_tsqr_rebuild<B>(                               \
        device const float*, device float*, device float*, device float*, device const float*,                 \
        constant PanelParams&, uint, uint, uint);
BD_PANEL_KERNELS(8)
BD_PANEL_KERNELS(16)
BD_PANEL_KERNELS(32)

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
