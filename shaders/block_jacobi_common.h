// Pieces shared by the two block Jacobi backends (Eigh_BlockJacobi.metal and
// Svd_BlockJacobi.metal): the block geometry, and the kernel body that applies
// a 2b x 2b orthogonal U to the columns of a block pair with simdgroup_matrix
// tile products.

#pragma once

#include <metal_stdlib>

using namespace metal;

// Block size b = 16: the subproblem order 2b = 32 is the simdgroup width, and
// the threadgroup footprint of a subproblem (S, U and scratch, ~8.5 KB) lets
// three of them share a core.
#define BJ_B      16          // block size
#define BJ_SUB    32          // subproblem order, 2 * BJ_B
#define BJ_SUBP   16          // pairs per subproblem round
#define BJ_GROUP  32          // rows / columns per update threadgroup (two blocks)

// Global row (or column) index of local index r in the stacked [P; Q] frame.
inline uint bj_row(uint r, uint P, uint Q) {
    return (r < BJ_B) ? P * BJ_B + r : Q * BJ_B + (r - BJ_B);
}

// [M_P  M_Q] <- [M_P  M_Q] U for the 32 rows starting at row0 of a row-major
// matrix with leading dimension np. 128 threads = 4 simdgroups; simdgroup s
// produces output columns 8s..8s+7, an 8-column strip inside block P (s < 2)
// or Q (s >= 2). w0 / w1 say whether the first / second 16 rows are written.
inline void bj_cols_apply(device float* m, device const float* u, uint np, uint row0,
                          uint P, uint Q, bool w0, bool w1,
                          threadgroup float* X, uint sg_id, uint t, uint T)
{
    for (uint idx = t; idx < BJ_GROUP * BJ_SUB; idx += T) {
        const uint r = idx / BJ_SUB;
        const uint c = idx - r * BJ_SUB;
        X[idx] = m[(ulong)(row0 + r) * np + bj_row(c, P, Q)];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_float8x8 acc[4];
    for (uint rt = 0; rt < 4; ++rt) acc[rt] = simdgroup_float8x8(0.0f);
    for (uint k = 0; k < 4; ++k) {
        simdgroup_float8x8 ut;
        simdgroup_load(ut, u + (k * 8) * BJ_SUB + sg_id * 8, BJ_SUB);
        for (uint rt = 0; rt < 4; ++rt) {
            simdgroup_float8x8 xt;
            simdgroup_load(xt, X + (rt * 8) * BJ_SUB + k * 8, BJ_SUB);
            simdgroup_multiply_accumulate(acc[rt], xt, ut, acc[rt]);
        }
    }

    device float* out = m + (ulong)row0 * np + bj_row(sg_id * 8, P, Q);
    for (uint rt = 0; rt < 4; ++rt) {
        if ((rt < 2) ? w0 : w1) simdgroup_store(acc[rt], out + (ulong)(rt * 8) * np, np);
    }
    // X is about to be refilled by the caller.
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
