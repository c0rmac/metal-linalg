#include <metal_stdlib>
using namespace metal;
// Prototype: the bulge chase's left reflectors (Q2) applied to X, as a
// pipeline of groups in each threadgroup (docs/proposals/two-stage-vectors.md),
// which became bd_chase_apply in shaders/Svd_Bidiag.metal. This copy also has
// two variants the library's does not: the first simdgroup's next tile
// fetched a step ahead (61 to 60 ms at 4096), and device-memory fences only at
// the steps that need them (no change). Built by build_w.sh; q2w_gpu checks
// both directions against one reflector at a time up to n = 2048.
//
// Width-16 band. Block (G, j): the reflectors of sweeps 16 G .. 16 G + 15 at
// step j, I - V T V^T with V 32 x 16 (column c the reflector of sweep
// 16 G + c, at row offset c). It acts on rows 1 + 16 p .. 16 p + 31 of X,
// p = G + j: tiles p and p + 1, tile t being rows 1 + 16 t .. 16 + 16 t.
//
// up (X <- Q2 X): groups from the last to the first, in a group p ascending.
// Block (G - 1, p) needs (G, p) and (G, p + 1) done, so the groups of a
// pass of K run together, a simdgroup each, group G_top - k at p = G_top +
// sigma - 2 k at step sigma: disjoint tiles. Each simdgroup keeps its block's
// two tiles in registers; from one step to the next the upper becomes the
// lower, the new upper comes from the simdgroup before (its lower, through
// threadgroup memory) or, for the first, from X; the lower goes to the next
// simdgroup, or for the last back to X.
//
// down (X <- Q2^T X): the blocks in the reverse order, transposed: groups
// from the first to the last, in a group p descending, group G_0 + k at
// p = pmax - sigma + 2 k; the lower becomes the upper, and so on mirrored.
//
// A threadgroup owns a strip of 8 CT columns of X for the whole call.
// Vb: per block V (32 x 16, row-major); Tb: per block -T (16 x 16, row-major).

struct WParams {
    uint n;       // X's rows: 1 .. n - 1 are acted on
    uint rs, cs;  // X(r, c) at X[r rs + c cs]
    uint pmax;    // the last block position, (n - 2) / 16; groups 0 .. pmax
    uint down;
};

constant constexpr uint TP = 4;   // padding of a staged tile's rows

// Block index of (G, p): groups in order, a group's blocks p = G .. pmax.
inline uint block_index(uint G, uint p, uint pmax) {
    return G * (pmax + 1) - G * (G - 1) / 2 + (p - G);
}

template <uint CT>
inline void tile_from_dev(thread simdgroup_float8x8 (&m)[2][CT], device const float* X, constant WParams& q,
                          uint t, uint c0, threadgroup float* S, uint lane) {
    constexpr uint C = 8 * CT, LD = C + TP;
    const uint r0 = 1 + 16 * t;
    if (q.rs == 1) {   // column-major: rows contiguous
        for (uint e = lane; e < 16 * C; e += 32) {
            const uint i = e % 16, c = e / 16, r = r0 + i;
            S[i * LD + c] = r < q.n ? X[r + (ulong)(c0 + c) * q.cs] : 0.0f;
        }
    } else {
        for (uint e = lane; e < 16 * C; e += 32) {
            const uint i = e / C, c = e % C, r = r0 + i;
            S[i * LD + c] = r < q.n ? X[(ulong)r * q.rs + c0 + c] : 0.0f;
        }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (uint a = 0; a < 2; ++a)
        for (uint c = 0; c < CT; ++c) simdgroup_load(m[a][c], S + 8 * a * LD + 8 * c, LD);
    simdgroup_barrier(mem_flags::mem_threadgroup);
}

// The same in two halves: the loads into registers (a lane's 4 CT entries),
// issued a step ahead, then into the matrices through the staging area.
template <uint CT>
inline void tile_fetch(thread float (&r)[4 * CT], device const float* X, constant WParams& q, uint t, uint c0,
                       uint lane) {
    constexpr uint C = 8 * CT;
    const uint r0 = 1 + 16 * t;
    for (uint i = 0; i < 4 * CT; ++i) {
        const uint e = lane + 32 * i;
        if (q.rs == 1) {
            const uint row = r0 + e % 16, c = e / 16;
            r[i] = row < q.n ? X[row + (ulong)(c0 + c) * q.cs] : 0.0f;
        } else {
            const uint row = r0 + e / C, c = e % C;
            r[i] = row < q.n ? X[(ulong)row * q.rs + c0 + c] : 0.0f;
        }
    }
}

template <uint CT>
inline void tile_from_regs(thread simdgroup_float8x8 (&m)[2][CT], thread const float (&r)[4 * CT],
                           constant WParams& q, threadgroup float* S, uint lane) {
    constexpr uint C = 8 * CT, LD = C + TP;
    for (uint i = 0; i < 4 * CT; ++i) {
        const uint e = lane + 32 * i;
        if (q.rs == 1) S[(e % 16) * LD + e / 16] = r[i];
        else S[(e / C) * LD + e % C] = r[i];
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (uint a = 0; a < 2; ++a)
        for (uint c = 0; c < CT; ++c) simdgroup_load(m[a][c], S + 8 * a * LD + 8 * c, LD);
    simdgroup_barrier(mem_flags::mem_threadgroup);
}

template <uint CT>
inline void tile_to_dev(thread simdgroup_float8x8 (&m)[2][CT], device float* X, constant WParams& q, uint t,
                        uint c0, threadgroup float* S, uint lane) {
    constexpr uint C = 8 * CT, LD = C + TP;
    const uint r0 = 1 + 16 * t;
    for (uint a = 0; a < 2; ++a)
        for (uint c = 0; c < CT; ++c) simdgroup_store(m[a][c], S + 8 * a * LD + 8 * c, LD);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    if (q.rs == 1) {
        for (uint e = lane; e < 16 * C; e += 32) {
            const uint i = e % 16, c = e / 16, r = r0 + i;
            if (r < q.n) X[r + (ulong)(c0 + c) * q.cs] = S[i * LD + c];
        }
    } else {
        for (uint e = lane; e < 16 * C; e += 32) {
            const uint i = e / C, c = e % C, r = r0 + i;
            if (r < q.n) X[(ulong)r * q.rs + c0 + c] = S[i * LD + c];
        }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
}

template <uint CT>
inline void tile_to_tg(thread simdgroup_float8x8 (&m)[2][CT], threadgroup float* S) {
    constexpr uint LD = 8 * CT + TP;
    for (uint a = 0; a < 2; ++a)
        for (uint c = 0; c < CT; ++c) simdgroup_store(m[a][c], S + 8 * a * LD + 8 * c, LD);
}

template <uint CT>
inline void tile_from_tg(thread simdgroup_float8x8 (&m)[2][CT], threadgroup const float* S) {
    constexpr uint LD = 8 * CT + TP;
    for (uint a = 0; a < 2; ++a)
        for (uint c = 0; c < CT; ++c) simdgroup_load(m[a][c], S + 8 * a * LD + 8 * c, LD);
}

// The block on [lo; hi] (32 x 8 CT): X += V (-T) V^T X (up), X += V (-T)^T V^T X (down).
template <uint CT>
inline void apply_block(thread simdgroup_float8x8 (&lo)[2][CT], thread simdgroup_float8x8 (&hi)[2][CT],
                        device const float* V, device const float* T, bool down) {
    simdgroup_float8x8 vt00, vt10, vt20, vt11, vt21, vt31;   // V(r, a)^T
    simdgroup_load(vt00, V, 16, ulong2(0, 0), true);
    simdgroup_load(vt10, V, 16, ulong2(0, 8), true);
    simdgroup_load(vt20, V, 16, ulong2(0, 16), true);
    simdgroup_load(vt11, V, 16, ulong2(8, 8), true);
    simdgroup_load(vt21, V, 16, ulong2(8, 16), true);
    simdgroup_load(vt31, V, 16, ulong2(8, 24), true);
    simdgroup_float8x8 W0[CT], W1[CT];
    for (uint c = 0; c < CT; ++c) {
        simdgroup_float8x8 a = simdgroup_float8x8(0.0f), b = simdgroup_float8x8(0.0f);
        simdgroup_multiply_accumulate(a, vt00, lo[0][c], a);
        simdgroup_multiply_accumulate(a, vt10, lo[1][c], a);
        simdgroup_multiply_accumulate(a, vt20, hi[0][c], a);
        simdgroup_multiply_accumulate(b, vt11, lo[1][c], b);
        simdgroup_multiply_accumulate(b, vt21, hi[0][c], b);
        simdgroup_multiply_accumulate(b, vt31, hi[1][c], b);
        W0[c] = a;
        W1[c] = b;
    }
    simdgroup_float8x8 t00, t01, t11;   // (-T) or (-T)^T blocks: out = [t00 t01; 0 t11] (up), [t00 0; t01 t11] (down)
    simdgroup_load(t00, T, 16, ulong2(0, 0), down);
    simdgroup_load(t01, T, 16, ulong2(8, 0), down);
    simdgroup_load(t11, T, 16, ulong2(8, 8), down);
    simdgroup_float8x8 Z0[CT], Z1[CT];
    if (!down) {
        for (uint c = 0; c < CT; ++c) {
            simdgroup_float8x8 a = simdgroup_float8x8(0.0f), b = simdgroup_float8x8(0.0f);
            simdgroup_multiply_accumulate(a, t00, W0[c], a);
            simdgroup_multiply_accumulate(a, t01, W1[c], a);
            simdgroup_multiply_accumulate(b, t11, W1[c], b);
            Z0[c] = a;
            Z1[c] = b;
        }
    } else {
        for (uint c = 0; c < CT; ++c) {
            simdgroup_float8x8 a = simdgroup_float8x8(0.0f), b = simdgroup_float8x8(0.0f);
            simdgroup_multiply_accumulate(a, t00, W0[c], a);
            simdgroup_multiply_accumulate(b, t01, W0[c], b);
            simdgroup_multiply_accumulate(b, t11, W1[c], b);
            Z0[c] = a;
            Z1[c] = b;
        }
    }
    simdgroup_float8x8 v00, v10, v20, v11, v21, v31;
    simdgroup_load(v00, V, 16, ulong2(0, 0));
    simdgroup_load(v10, V, 16, ulong2(0, 8));
    simdgroup_load(v20, V, 16, ulong2(0, 16));
    simdgroup_load(v11, V, 16, ulong2(8, 8));
    simdgroup_load(v21, V, 16, ulong2(8, 16));
    simdgroup_load(v31, V, 16, ulong2(8, 24));
    for (uint c = 0; c < CT; ++c) {
        simdgroup_multiply_accumulate(lo[0][c], v00, Z0[c], lo[0][c]);
        simdgroup_multiply_accumulate(lo[1][c], v10, Z0[c], lo[1][c]);
        simdgroup_multiply_accumulate(lo[1][c], v11, Z1[c], lo[1][c]);
        simdgroup_multiply_accumulate(hi[0][c], v20, Z0[c], hi[0][c]);
        simdgroup_multiply_accumulate(hi[0][c], v21, Z1[c], hi[0][c]);
        simdgroup_multiply_accumulate(hi[1][c], v31, Z1[c], hi[1][c]);
    }
}

template <uint CT, uint K>
kernel void q2w(device float* X [[buffer(0)]], device const float* Vb [[buffer(1)]],
                device const float* Tb [[buffer(2)]], constant WParams& q [[buffer(3)]],
                uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint C = 8 * CT, TILE = 16 * (C + TP);
    threadgroup float hand[2][K][TILE];
    const uint c0 = tg * C, pmax = q.pmax, ng = pmax + 1;
    const int k = (int)sg;
    simdgroup_float8x8 lo[2][CT], hi[2][CT];
    float pf[4 * CT];   // the first simdgroup's next tile from X, fetched a step ahead
    for (uint pass = 0; pass * K < ng; ++pass) {
        const uint kpass = min(K, ng - pass * K);
        // up: group G = gtop - k at p = gtop + s - 2k; down: G = g0 + k at p = pmax - s + 2k
        const int gtop = (int)ng - 1 - (int)(pass * K), g0 = (int)(pass * K);
        const int G = q.down ? g0 + k : gtop - k;
        const bool mine = k < (int)kpass;
        const int steps = q.down ? (int)pmax - g0 + (int)kpass : (int)pmax - gtop + 2 * (int)kpass - 1;
        for (int s = 0; s < steps; ++s) {
            const int p = q.down ? (int)pmax - s + 2 * k : gtop + s - 2 * k;
            const bool active = mine && p >= G && p <= (int)pmax;
            threadgroup float* out = hand[s & 1][k];
            if (active) {
                if (!q.down) {
                    // lo: the previous upper, or X at the group's first step
                    if (p == G) tile_from_dev<CT>(lo, X, q, p, c0, out, lane);
                    else
                        for (uint a = 0; a < 2; ++a)
                            for (uint c = 0; c < CT; ++c) lo[a][c] = hi[a][c];
                    // hi: the previous simdgroup's lower, or X (the first simdgroup's
                    // fetched the step before, but at its first step)
                    if (k > 0 && p < (int)pmax) tile_from_tg<CT>(hi, hand[(s - 1) & 1][k - 1]);
                    else if (k == 0 && p > G) tile_from_regs<CT>(hi, pf, q, out, lane);
                    else tile_from_dev<CT>(hi, X, q, p + 1, c0, out, lane);
                    if (k == 0 && p < (int)pmax) tile_fetch<CT>(pf, X, q, p + 2, c0, lane);
                } else {
                    if (p == (int)pmax) tile_from_dev<CT>(hi, X, q, p + 1, c0, out, lane);
                    else
                        for (uint a = 0; a < 2; ++a)
                            for (uint c = 0; c < CT; ++c) hi[a][c] = lo[a][c];
                    if (k > 0) tile_from_tg<CT>(lo, hand[(s - 1) & 1][k - 1]);
                    else if (p < (int)pmax) tile_from_regs<CT>(lo, pf, q, out, lane);
                    else tile_from_dev<CT>(lo, X, q, p, c0, out, lane);
                    if (k == 0 && p > G) tile_fetch<CT>(pf, X, q, p - 1, c0, lane);
                }
                const ulong b = block_index((uint)G, (uint)p, pmax);
                apply_block<CT>(lo, hi, Vb + b * 512, Tb + b * 256, q.down != 0);
                if (!q.down) {   // (the staging area is the handoff's: the store first)
                    if (p == (int)pmax) tile_to_dev<CT>(hi, X, q, p + 1, c0, out, lane);
                    if (k + 1 < (int)kpass) tile_to_tg<CT>(lo, out);
                    else tile_to_dev<CT>(lo, X, q, p, c0, out, lane);
                } else {
                    if (p == G) tile_to_dev<CT>(lo, X, q, p, c0, out, lane);
                    if (k + 1 < (int)kpass && p < (int)pmax) tile_to_tg<CT>(hi, out);
                    else tile_to_dev<CT>(hi, X, q, p + 1, c0, out, lane);
                }
            }
            // X's tiles stored for another simdgroup to load in this pass: down,
            // each simdgroup's first upper tile (steps 2 k); up, its last (the
            // last 2 kpass steps); and every store before the next pass. Only
            // those steps fence device memory.
            const bool fence = q.down ? s < 2 * (int)kpass : s >= steps - 2 * (int)kpass;
            if (fence) threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
            else threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    }
}

#define Q2W(CT, K) \
    template [[host_name("q2w_" #CT "_" #K)]] kernel void q2w<CT, K>(device float*, device const float*, \
        device const float*, constant WParams&, uint, uint, uint);
Q2W(2, 4)
Q2W(2, 8)
Q2W(4, 4)
