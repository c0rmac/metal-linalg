#include <metal_stdlib>
using namespace metal;

// =============================================================================
// LU with partial pivoting: the row swaps of the large-matrix path
// =============================================================================
//
// lu_gpu.mm factors one panel of columns at a time on the CPU (LAPACK's
// sgetrf, in a column-major workspace the GPU shares), which swaps rows within
// the panel; the same swaps must reach every other column before the panel's
// update of the trailing matrix (by MPS products). The swaps of a panel come
// as a list of (row, source row) pairs, the sequential swaps composed into one
// permutation of the rows they touch (lu_gpu.mm), so that every entry of a
// column moves at once rather than swap after dependent swap down a column (6
// ms a matrix of 4096 x 4096, against 0.9).

// Must match `GatherParams` in lu_gpu.mm.
struct GatherParams {
    uint ld;      // the workspace's leading dimension (a column's stride)
    uint count;   // pairs in the list, at most 2 x the panel's width
    uint c0;      // the columns, [c0, c1)
    uint c1;
};

// Row e[i].x of each column gets row e[i].y's old value: a threadgroup a
// column, every source read before any destination is written.
kernel void lu_gather(device float*         W [[buffer(0)]],
                      device const uint2*   e [[buffer(1)]],
                      constant GatherParams& p [[buffer(2)]],
                      uint tg [[threadgroup_position_in_grid]],
                      uint t  [[thread_index_in_threadgroup]],
                      uint T  [[threads_per_threadgroup]])
{
    threadgroup float v[512];
    const uint col = p.c0 + tg;
    if (col >= p.c1) return;
    device float* w = W + (ulong)col * p.ld;
    for (uint i = t; i < p.count; i += T) v[i] = w[e[i].y];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = t; i < p.count; i += T) w[e[i].x] = v[i];
}

// =============================================================================
// Moving matrices in and out: on the GPU, from and to the caller's buffers
// =============================================================================

// Must match `MoveParams` in lu_gpu.mm.
struct MoveParams {
    uint rows, cols;   // the source's
    uint ldi, ldo;     // row strides (a column-major matrix's: its column stride)
};

// out[c][r] = in[r][c]: a row-major matrix into the column-major workspace,
// or the workspace's factors out as a row-major matrix. 32 x 32 tiles through
// threadgroup memory, so that both sides are read and written along rows.
// Grid: (cols, rows) rounded up to 32, threadgroups of 32 x 8.
kernel void lu_transpose(device const float* in  [[buffer(0)]],
                         device float*       out [[buffer(1)]],
                         constant MoveParams& p  [[buffer(2)]],
                         uint2 tg [[threadgroup_position_in_grid]],
                         uint2 t  [[thread_position_in_threadgroup]])
{
    threadgroup float tile[32][33];
    const uint c0 = tg.x * 32, r0 = tg.y * 32;
    for (uint i = t.y; i < 32; i += 8) {
        const uint r = r0 + i, c = c0 + t.x;
        if (r < p.rows && c < p.cols) tile[i][t.x] = in[(ulong)r * p.ldi + c];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = t.y; i < 32; i += 8) {
        const uint c = c0 + i, r = r0 + t.x;
        if (r < p.rows && c < p.cols) out[(ulong)c * p.ldo + r] = tile[t.x][i];
    }
}

// out[i][j] = in[perm[i]][j]: the right-hand sides with A's row swaps, P B.
// Grid: (cols, rows).
kernel void lu_permute_rows(device const float* in   [[buffer(0)]],
                            device float*       out  [[buffer(1)]],
                            device const uint*  perm [[buffer(2)]],
                            constant MoveParams& p   [[buffer(3)]],
                            uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= p.cols || gid.y >= p.rows) return;
    out[(ulong)gid.y * p.ldo + gid.x] = in[(ulong)perm[gid.y] * p.ldi + gid.x];
}

// out[i][j] = (perm[i] == j): the identity with A's row swaps, P I, for the
// inverse. Grid: (cols, rows).
kernel void lu_permuted_identity(device float*      out  [[buffer(0)]],
                                 device const uint* perm [[buffer(1)]],
                                 constant MoveParams& p  [[buffer(2)]],
                                 uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= p.cols || gid.y >= p.rows) return;
    out[(ulong)gid.y * p.ldo + gid.x] = perm[gid.y] == gid.x ? 1.0f : 0.0f;
}
