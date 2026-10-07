#include <metal_stdlib>
using namespace metal;
// Prototype: the bulge chase's left reflectors (Q2) applied to X on the GPU,
// for docs/proposals/two-stage-vectors.md. Reflector (s, j), sweep s's j-th,
// acts on rows [s + 1 + j nb, min(s + (j + 1) nb, n - 1)]. Groups of ib
// consecutive sweeps are applied from the last group to the first, and in a
// group the steps j in increasing order, each step's ib reflectors as one
// block I - V T V^T (T from slarft's recurrence, on the CPU); that order
// gives the same result as one reflector at a time (check_q2.cpp). A
// threadgroup a strip of 32 columns of X, the blocks one after another.
//
// On an M5 Pro, n = 4096, nb = ib = 16 (32,896 blocks): 170 ms; one
// threadgroup alone takes 83 ms (2.5 us a block), and above about 40
// threadgroups they run in waves. Wider strips, smaller threadgroups, a
// sliding window kept in threadgroup memory, and a simdgroup a group with
// the groups staggered one block apart (registers spilled) were all slower.
struct Q2Params { uint n, nb, ib, J, nblocks, ldx; };
// Block b of the application order: (G, j). The chase's reflector (s, j) acts on rows
// [s + 1 + j nb, min(s + (j + 1) nb, n - 1)], stored at Lv[(s J + j) nb ...], tau Lt[s J + j].
// X (column-major, ld ldx) <- (I - V T V^T) X on the block's rows, for this threadgroup's 32 columns.
kernel void q2_apply(device float* X [[buffer(0)]], device const float* Lv [[buffer(1)]],
                     device const float* Tb [[buffer(2)]], device const uint2* blocks [[buffer(3)]],
                     constant Q2Params& q [[buffer(4)]], uint tg [[threadgroup_position_in_grid]],
                     uint sg [[simdgroup_index_in_threadgroup]], uint t [[thread_index_in_threadgroup]]) {
    threadgroup float Vs[32][20], Vn[32][20], Ts[16][20], Xs[32][36], Ws[16][36], W2[16][36];
    const uint c0 = tg * 32, ib = q.ib, nb = q.nb, n = q.n;
    for (uint bi = 0; bi < q.nblocks; ++bi) {
        const uint G = blocks[bi].x, j = blocks[bi].y;
        const uint g0 = G * ib, g1 = min(g0 + ib - 1, n - 2);
        const uint r0 = g0 + 1 + j * nb, rend = min(g1 + (j + 1) * nb, n - 1), rows = rend - r0 + 1;
        // V (rows x ib, column c the reflector of sweep g0 + c, at row offset c), -V, T, X's rows
        for (uint e = t; e < 32 * 16; e += 256) {
            const uint r = e / 16, c = e % 16, s = g0 + c;
            float v = 0.0f;
            if (c < ib && s <= g1 && r >= c) {
                const uint a = s + 1 + j * nb, b2 = min(s + (j + 1) * nb, n - 1);
                const uint len = b2 >= a ? b2 - a + 1 : 0, k = r - c;
                if (k < len) v = k == 0 ? 1.0f : Lv[((ulong)s * q.J + j) * nb + k];
            }
            Vs[r][c] = v;
            Vn[r][c] = -v;
        }
        for (uint e = t; e < 256; e += 256) Ts[e / 16][e % 16] = Tb[(ulong)bi * 256 + e];
        for (uint e = t; e < 32 * 32; e += 256) {
            const uint c = e / 32, r = e % 32;
            Xs[r][c] = r < rows ? X[(ulong)(c0 + c) * q.ldx + r0 + r] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {   // W = V^T X: 2 x 4 tiles, a simdgroup each
            const uint wr = sg / 4, wc = sg % 4;
            simdgroup_float8x8 acc = simdgroup_float8x8(0.0f);
            for (uint k = 0; k < 32; k += 8) {
                simdgroup_float8x8 vt, xm;
                simdgroup_load(vt, &Vs[k][wr * 8], 20, ulong2(0, 0), true);
                simdgroup_load(xm, &Xs[k][wc * 8], 36);
                simdgroup_multiply_accumulate(acc, vt, xm, acc);
            }
            simdgroup_store(acc, &Ws[wr * 8][wc * 8], 36);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {   // W2 = T W
            const uint wr = sg / 4, wc = sg % 4;
            simdgroup_float8x8 acc = simdgroup_float8x8(0.0f);
            for (uint k = 0; k < 16; k += 8) {
                simdgroup_float8x8 tm, wm;
                simdgroup_load(tm, &Ts[wr * 8][k], 20);
                simdgroup_load(wm, &Ws[k][wc * 8], 36);
                simdgroup_multiply_accumulate(acc, tm, wm, acc);
            }
            simdgroup_store(acc, &W2[wr * 8][wc * 8], 36);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint tt = sg; tt < 16; tt += 8) {   // X -= V W2: 4 x 4 tiles
            const uint xr = tt / 4, xc = tt % 4;
            simdgroup_float8x8 acc;
            simdgroup_load(acc, &Xs[xr * 8][xc * 8], 36);
            for (uint k = 0; k < 16; k += 8) {
                simdgroup_float8x8 vm, wm;
                simdgroup_load(vm, &Vn[xr * 8][k], 20);
                simdgroup_load(wm, &W2[k][xc * 8], 36);
                simdgroup_multiply_accumulate(acc, vm, wm, acc);
            }
            simdgroup_store(acc, &Xs[xr * 8][xc * 8], 36);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint e = t; e < 32 * 32; e += 256) {
            const uint c = e / 32, r = e % 32;
            if (r < rows) X[(ulong)(c0 + c) * q.ldx + r0 + r] = Xs[r][c];
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    }
}
