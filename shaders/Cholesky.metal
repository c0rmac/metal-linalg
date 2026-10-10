#include <metal_stdlib>
using namespace metal;

// =============================================================================
// Batched Cholesky factorization A = L L^T of symmetric positive definite
// matrices, a matrix to a simdgroup, a threadgroup, or the whole GPU
// =============================================================================
//
// Reads the lower triangle of each row-major n x n matrix (the upper one,
// transposed, with `upper`) and writes L, or U = L^T with `upper`, with zeros
// in the other triangle, as numpy.linalg.cholesky and torch.linalg.cholesky
// return it. A matrix that is not positive definite -- a pivot that is not a
// positive finite number, which includes any NaN or infinity that reaches
// one -- stops at that pivot, as LAPACK's spotrf does: its info is the
// pivot's index + 1 and its output is all NaN.
//
//   chol_simd    up to 32 x 32, a simdgroup a matrix (several below 16): a
//                row a lane in registers, right-looking, a column a step
//   chol_factor  a threadgroup a matrix (or the diagonal block of a larger
//                one), in a zero-padded workspace: panels of 32 columns, the
//                diagonal block in one simdgroup's registers, the rows below
//                it by forward substitution, a thread a row, and the trailing
//                lower triangle by 8 x 8 simdgroup matrix products
//   chol_panel   for the large-matrix path (cholesky_gpu.mm): a 32-column
//                sub-panel of a 128-column panel in one dispatch, every
//                threadgroup a strip of rows, each bringing the 32 x 32
//                diagonal block up to date and factoring it itself; the
//                trailing update between panels is MPS's
//   chol_put     the diagonal blocks chol_panel kept aside into place
//   chol_load    the input into the workspace, zero-padded
//   chol_store   the workspace's lower triangle out as L or U, NaN on failure
//
// References: Golub & Van Loan, Matrix Computations 4th ed., s4.2 (the
// Cholesky factorization, its gaxpy and outer-product forms, s4.2.8 the
// block form); LAPACK spotrf, spotf2.
//
// Built with fast math; the pivot's square root and reciprocal are precise.

// A pivot that is a positive finite number: by its bits, which fast math
// cannot fold away (NaN and infinities fail, as do zero and negatives).
inline bool good_pivot(float d) {
    const uint b = as_type<uint>(d);
    return (b & 0x80000000u) == 0u && (b & 0x7F800000u) != 0x7F800000u && b != 0u;
}

#define CH_U32_c(...) { { constexpr uint c = 0; __VA_ARGS__ } { constexpr uint c = 1; __VA_ARGS__ } { constexpr uint c = 2; __VA_ARGS__ } { constexpr uint c = 3; __VA_ARGS__ } { constexpr uint c = 4; __VA_ARGS__ } { constexpr uint c = 5; __VA_ARGS__ } { constexpr uint c = 6; __VA_ARGS__ } { constexpr uint c = 7; __VA_ARGS__ } { constexpr uint c = 8; __VA_ARGS__ } { constexpr uint c = 9; __VA_ARGS__ } { constexpr uint c = 10; __VA_ARGS__ } { constexpr uint c = 11; __VA_ARGS__ } { constexpr uint c = 12; __VA_ARGS__ } { constexpr uint c = 13; __VA_ARGS__ } { constexpr uint c = 14; __VA_ARGS__ } { constexpr uint c = 15; __VA_ARGS__ } { constexpr uint c = 16; __VA_ARGS__ } { constexpr uint c = 17; __VA_ARGS__ } { constexpr uint c = 18; __VA_ARGS__ } { constexpr uint c = 19; __VA_ARGS__ } { constexpr uint c = 20; __VA_ARGS__ } { constexpr uint c = 21; __VA_ARGS__ } { constexpr uint c = 22; __VA_ARGS__ } { constexpr uint c = 23; __VA_ARGS__ } { constexpr uint c = 24; __VA_ARGS__ } { constexpr uint c = 25; __VA_ARGS__ } { constexpr uint c = 26; __VA_ARGS__ } { constexpr uint c = 27; __VA_ARGS__ } { constexpr uint c = 28; __VA_ARGS__ } { constexpr uint c = 29; __VA_ARGS__ } { constexpr uint c = 30; __VA_ARGS__ } { constexpr uint c = 31; __VA_ARGS__ } }
#define CH_U32_u(...) { { constexpr uint u = 0; __VA_ARGS__ } { constexpr uint u = 1; __VA_ARGS__ } { constexpr uint u = 2; __VA_ARGS__ } { constexpr uint u = 3; __VA_ARGS__ } { constexpr uint u = 4; __VA_ARGS__ } { constexpr uint u = 5; __VA_ARGS__ } { constexpr uint u = 6; __VA_ARGS__ } { constexpr uint u = 7; __VA_ARGS__ } { constexpr uint u = 8; __VA_ARGS__ } { constexpr uint u = 9; __VA_ARGS__ } { constexpr uint u = 10; __VA_ARGS__ } { constexpr uint u = 11; __VA_ARGS__ } { constexpr uint u = 12; __VA_ARGS__ } { constexpr uint u = 13; __VA_ARGS__ } { constexpr uint u = 14; __VA_ARGS__ } { constexpr uint u = 15; __VA_ARGS__ } { constexpr uint u = 16; __VA_ARGS__ } { constexpr uint u = 17; __VA_ARGS__ } { constexpr uint u = 18; __VA_ARGS__ } { constexpr uint u = 19; __VA_ARGS__ } { constexpr uint u = 20; __VA_ARGS__ } { constexpr uint u = 21; __VA_ARGS__ } { constexpr uint u = 22; __VA_ARGS__ } { constexpr uint u = 23; __VA_ARGS__ } { constexpr uint u = 24; __VA_ARGS__ } { constexpr uint u = 25; __VA_ARGS__ } { constexpr uint u = 26; __VA_ARGS__ } { constexpr uint u = 27; __VA_ARGS__ } { constexpr uint u = 28; __VA_ARGS__ } { constexpr uint u = 29; __VA_ARGS__ } { constexpr uint u = 30; __VA_ARGS__ } { constexpr uint u = 31; __VA_ARGS__ } }
#define CH_U32_k(...) { { constexpr uint k = 0; __VA_ARGS__ } { constexpr uint k = 1; __VA_ARGS__ } { constexpr uint k = 2; __VA_ARGS__ } { constexpr uint k = 3; __VA_ARGS__ } { constexpr uint k = 4; __VA_ARGS__ } { constexpr uint k = 5; __VA_ARGS__ } { constexpr uint k = 6; __VA_ARGS__ } { constexpr uint k = 7; __VA_ARGS__ } { constexpr uint k = 8; __VA_ARGS__ } { constexpr uint k = 9; __VA_ARGS__ } { constexpr uint k = 10; __VA_ARGS__ } { constexpr uint k = 11; __VA_ARGS__ } { constexpr uint k = 12; __VA_ARGS__ } { constexpr uint k = 13; __VA_ARGS__ } { constexpr uint k = 14; __VA_ARGS__ } { constexpr uint k = 15; __VA_ARGS__ } { constexpr uint k = 16; __VA_ARGS__ } { constexpr uint k = 17; __VA_ARGS__ } { constexpr uint k = 18; __VA_ARGS__ } { constexpr uint k = 19; __VA_ARGS__ } { constexpr uint k = 20; __VA_ARGS__ } { constexpr uint k = 21; __VA_ARGS__ } { constexpr uint k = 22; __VA_ARGS__ } { constexpr uint k = 23; __VA_ARGS__ } { constexpr uint k = 24; __VA_ARGS__ } { constexpr uint k = 25; __VA_ARGS__ } { constexpr uint k = 26; __VA_ARGS__ } { constexpr uint k = 27; __VA_ARGS__ } { constexpr uint k = 28; __VA_ARGS__ } { constexpr uint k = 29; __VA_ARGS__ } { constexpr uint k = 30; __VA_ARGS__ } { constexpr uint k = 31; __VA_ARGS__ } }
#define CH_UNROLL(N, v, ...) CH_U32_##v(if (v < N) __VA_ARGS__)

// The factorization of an n x n matrix (n <= B <= G) held a row a lane in a
// group of G lanes (lane gb + i holds row i, x[c] = A(i, c) for c <= i, 0
// above): on return x[c] = L(i, c), 0 above the diagonal, and the result is 0,
// or k + 1 if pivot k failed (then x is garbage). Both loops unrolled, so the
// row is only ever indexed by constants and stays in registers: column k's
// pivot from lane k, its entries scaled, then each later column c of the rows
// below updated with L(c, k) from lane c, (B - k) shuffles a step. (A loop
// over k that rotated the row to keep column k at index 0 did three times
// the instructions.)
template <uint B, uint G>
inline uint chol_regs(thread float (&x)[B], uint n, uint i, uint gb) {
    uint fail = 0;
    CH_UNROLL(B, k, {
        if (k < n && fail == 0) {   // uniform within the group
            const float d = simd_shuffle(x[k], (ushort)(gb + k));
            if (!good_pivot(d)) {
                fail = k + 1;
            } else {
                const float r = precise::sqrt(d), rinv = precise::divide(1.0f, r);
                x[k] = i == k ? r : (i > k ? x[k] * rinv : 0.0f);
                CH_UNROLL(B, c, {
                    if (c > k) {
                        const float lc = simd_shuffle(x[k], (ushort)(gb + min(c, G - 1)));
                        if (i >= c && c < n) x[c] = fma(-x[k], lc, x[c]);
                    }
                });
            }
        }
    });
    return fail;
}

// Must match `CsParams` in cholesky.mm.
struct CsParams {
    uint n;       // the order, at most G
    uint batch;   // matrices
    uint upper;   // 1: read the upper triangle, write U = L^T
};

// chol_simd: a matrix to a group of G lanes, 32 / G matrices a simdgroup,
// each simdgroup's matrices staged through its own 32 x 33 floats of
// threadgroup memory both ways, so that device memory is read and written a
// whole row of consecutive floats at a time. (Each lane reading its own row
// touched 32 cache lines an instruction: 16384 matrices of 32 x 32 took
// 4.1 ms, against 0.5 for the arithmetic.)
template <uint B, uint G>
kernel void chol_simd(device const float* A     [[buffer(0)]],
                      device float*       L     [[buffer(1)]],
                      device uint*        info  [[buffer(2)]],
                      constant CsParams&  p     [[buffer(3)]],
                      threadgroup float*  stage [[threadgroup(0)]],
                      uint tgi  [[threadgroup_position_in_grid]],
                      uint sg   [[simdgroup_index_in_threadgroup]],
                      uint nsg  [[simdgroups_per_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]])
{
    constexpr uint P = 32 / G;   // matrices a simdgroup
    const uint i = lane % G, gb = lane - i, slot = lane / G;
    const uint first = (tgi * nsg + sg) * P;   // the simdgroup's first matrix
    if (first >= p.batch) return;              // the whole simdgroup
    const uint n = p.n, nn = n * n, count = min(P, p.batch - first);
    threadgroup float* st = stage + sg * (32 * 33);
    device const float* a = A + (ulong)first * nn;
    // in: consecutive floats, matrix m's row r at stage row m G + r
    for (uint e = lane; e < count * nn; e += 32) {
        const uint m = e / nn, w = e - m * nn, r = w / n;
        st[(m * G + r) * 33 + (w - r * n)] = a[e];
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    float x[B];
    CH_UNROLL(B, c, {
        float v = 0.0f;
        if (i < n && c <= i) v = p.upper ? st[(gb + c) * 33 + i] : st[(gb + i) * 33 + c];
        x[c] = v;
    });
    simdgroup_barrier(mem_flags::mem_threadgroup);
    const uint fail = chol_regs<B, G>(x, n, i, gb);
    // out the same way, U transposed in the stage
    if (i < n) {
        const float qnan = as_type<float>(0x7FC00000u);
        CH_UNROLL(B, c, {
            if (c < n) {
                const float v = fail ? qnan : (c <= i ? x[c] : 0.0f);
                if (p.upper) st[(gb + c) * 33 + i] = fail ? qnan : (c <= i ? x[c] : 0.0f);
                else         st[(gb + i) * 33 + c] = v;
            }
        });
    }
    if (i == 0 && slot < count) info[first + slot] = fail;
    simdgroup_barrier(mem_flags::mem_threadgroup);
    device float* l = L + (ulong)first * nn;
    for (uint e = lane; e < count * nn; e += 32) {
        const uint m = e / nn, w = e - m * nn, r = w / n;
        l[e] = st[(m * G + r) * 33 + (w - r * n)];
    }
}

#define CH_SIMD(B, G) \
    template [[host_name("chol_simd_" #B "_" #G)]] kernel void chol_simd<B, G>( \
        device const float*, device float*, device uint*, constant CsParams&, threadgroup float*, uint, uint, uint, uint);
CH_SIMD(8, 8)
CH_SIMD(16, 16)
CH_SIMD(32, 32)

// =============================================================================
// The workspace: matrices of np x np (np = n rounded up to 32), zero-padded,
// row-major, ld = np, the lower triangle factored in place
// =============================================================================

// Must match `CwParams` in cholesky.mm.
struct CwParams {
    uint n;       // the order
    uint np;      // padded, a multiple of 32: the workspace's ld
    uint upper;   // 1: read the upper triangle / write U
    uint batch;
};

// The input's lower triangle (or upper, transposed) into the workspace, zero
// above it and in the padding; fail cleared.
kernel void chol_load(device const float* A    [[buffer(0)]],
                      device float*       W    [[buffer(1)]],
                      device uint*        fail [[buffer(2)]],
                      constant CwParams&  p    [[buffer(3)]],
                      uint3 gid [[thread_position_in_grid]])
{
    const uint c = gid.x, r = gid.y, mat = gid.z;
    if (c >= p.np || r >= p.np || mat >= p.batch) return;
    const uint n = p.n;
    float v = 0.0f;
    if (r < n && c <= r) {
        device const float* a = A + (ulong)mat * n * n;
        v = p.upper ? a[c * n + r] : a[r * n + c];
    }
    W[(ulong)mat * p.np * p.np + r * p.np + c] = v;
    if (r == 0 && c == 0) fail[mat] = 0;
}

// The workspace's lower triangle out as L (or U = L^T), NaN where the
// factorization failed.
kernel void chol_store(device const float* W    [[buffer(0)]],
                       device float*       L    [[buffer(1)]],
                       device const uint*  fail [[buffer(2)]],
                       constant CwParams&  p    [[buffer(3)]],
                       uint3 gid [[thread_position_in_grid]])
{
    const uint c = gid.x, r = gid.y, mat = gid.z;
    const uint n = p.n;
    if (c >= n || r >= n || mat >= p.batch) return;
    float v;
    if (fail[mat]) {
        v = as_type<float>(0x7FC00000u);
    } else {
        // L(r, c) for the lower output, U(r, c) = L(c, r) for the upper
        const uint lr = p.upper ? c : r, lc = p.upper ? r : c;
        v = lc <= lr ? W[(ulong)mat * p.np * p.np + lr * p.np + lc] : 0.0f;
    }
    L[(ulong)mat * n * n + r * n + c] = v;
}

// Must match `CfParams` in cholesky.mm.
struct CfParams {
    uint n;      // the block's order
    uint ld;     // the workspace's leading dimension
    uint sw;     // floats between matrices in the workspace
    uint off;    // the block's first entry: j0 ld + j0
    uint base;   // j0, for the failing pivot's index
    uint np;     // the block's order rounded up to 32 (the trailing tiles' extent)
};

// The threadgroup's sum of x is not needed: every reduction here is a
// simdgroup's. The tile index t of the lower triangle of an nt x nt grid of
// tiles: (I, J), J <= I, row by row.
inline uint2 tile_of(uint t) {
    uint I = (uint)((sqrt(8.0f * (float)t + 1.0f) - 1.0f) * 0.5f);
    while ((I + 1) * (I + 2) / 2 <= t) ++I;
    while (I * (I + 1) / 2 > t) --I;
    return uint2(I, t - I * (I + 1) / 2);
}

// The 32 x 32 block at w (ld), rows below it from `r0` to `r1`: the block
// factored by simdgroup 0 into registers, written back and staged in Lt, its
// reciprocal pivots in dinv; then each row below by forward substitution
// against it, a thread a row. `fail_out` the failing pivot (+1) or 0.
// Threadgroup barriers inside: every thread of the threadgroup calls it.
inline uint chol_block_and_rows(device float* w, uint ld, uint jb, uint r0, uint r1,
                                threadgroup float (*Lt)[33], threadgroup float* dinv,
                                threadgroup uint* tfail, bool write_block,
                                uint t, uint T, uint sg, uint lane)
{
    if (sg == 0) {
        float x[32];
        CH_UNROLL(32, c, { x[c] = (lane < jb && c <= lane) ? w[lane * ld + c] : 0.0f; });
        const uint f = chol_regs<32, 32>(x, jb, lane, 0);
        if (lane == 0) *tfail = f;
        CH_UNROLL(32, c, {
            const float v = (c <= lane && lane < jb && c < jb) ? x[c] : 0.0f;
            Lt[lane][c] = v;
            if (c == lane) dinv[lane] = lane < jb && v != 0.0f ? precise::divide(1.0f, v) : 0.0f;
            if (write_block && lane < jb && c < jb && c <= lane) w[lane * ld + c] = v;
        });
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    const uint f = *tfail;
    if (f) return f;
    for (uint r = r0 + t; r < r1; r += T) {
        device float* row = w + r * ld;
        float y[32];
        CH_UNROLL(32, c, { y[c] = c < jb ? row[c] : 0.0f; });
        CH_UNROLL(32, c, {
            if (c < jb) {
                float s = y[c];
                CH_UNROLL(32, u, { if (u < c) s = fma(-y[u], Lt[c][u], s); });
                y[c] = s * dinv[c];
            }
        });
        CH_UNROLL(32, c, { if (c < jb) row[c] = y[c]; });
    }
    return 0;
}

// c -= a b^T over K = 32 (four 8 x 8 steps): the trailing tile (I, J) of a
// panel at column j0, its rows from s0 (the tiles row-major at w, ld).
inline void chol_tile(device float* w, uint ld, uint j0, uint s0, uint I, uint J) {
    simdgroup_float8x8 c, a, b;
    device float* cp = w + (ulong)(s0 + 8 * I) * ld + s0 + 8 * J;
    simdgroup_load(c, cp, ld);
    for (uint kk = 0; kk < 4; ++kk) {
        simdgroup_load(a, w + (ulong)(s0 + 8 * I) * ld + j0 + 8 * kk, ld);
        simdgroup_load(b, w + (ulong)(s0 + 8 * J) * ld + j0 + 8 * kk, ld, ulong2(0, 0), true);
        thread auto& e = a.thread_elements();
        e[0] = -e[0];
        e[1] = -e[1];
        simdgroup_multiply_accumulate(c, a, b, c);
    }
    simdgroup_store(c, cp, ld);
}

// chol_factor: the n x n block at off, a threadgroup a matrix (grid z), in
// panels of 32 columns. Skips a matrix that failed earlier (the large path
// calls it on each diagonal block).
kernel void chol_factor(device float*      W    [[buffer(0)]],
                        device uint*       fail [[buffer(1)]],
                        constant CfParams& p    [[buffer(2)]],
                        uint mat  [[threadgroup_position_in_grid]],
                        uint t    [[thread_index_in_threadgroup]],
                        uint T    [[threads_per_threadgroup]],
                        uint sg   [[simdgroup_index_in_threadgroup]],
                        uint nsg  [[simdgroups_per_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]])
{
    if (fail[mat]) return;
    threadgroup float Lt[32][33];
    threadgroup float dinv[32];
    threadgroup uint tfail;
    device float* w = W + (ulong)mat * p.sw + p.off;
    const uint n = p.n, ld = p.ld, np = p.np;
    for (uint j0 = 0; j0 < n; j0 += 32) {
        const uint jb = min(32u, n - j0);
        const uint f = chol_block_and_rows(w + (ulong)j0 * ld + j0, ld, jb, jb, n - j0, Lt, dinv, &tfail, true,
                                           t, T, sg, lane);
        if (f) {
            if (t == 0) fail[mat] = p.base + j0 + f;
            return;
        }
        threadgroup_barrier(mem_flags::mem_device);
        // A22 -= L21 L21^T on its lower triangle of 8 x 8 tiles (padding
        // rows of L21 are zero, so the padded tiles stay zero)
        const uint s0 = j0 + 32;
        if (s0 < n) {
            const uint nt = (np - s0) / 8, ntiles = nt * (nt + 1) / 2;
            for (uint tl = sg; tl < ntiles; tl += nsg) {
                const uint2 ij = tile_of(tl);
                chol_tile(w, ld, j0, s0, ij.x, ij.y);
            }
        }
        threadgroup_barrier(mem_flags::mem_device);
    }
}

// Must match `CpParams` in cholesky_gpu.mm.
struct CpParams {
    uint ld;     // the workspace's leading dimension (a multiple of 32)
    uint sw;     // floats between matrices
    uint j0;     // the panel's first column
    uint j;      // the sub-panel's first column (and its diagonal block's row)
    uint jb;     // the sub-panel's width: 32, or what is left of n
    uint rows;   // the (padded) rows below the diagonal block: ld - j - 32
    uint nblk;   // diagonal blocks a matrix (ld / 32): D's stride
};

constant constexpr uint kPanelStrip = 64;   // rows a chol_panel threadgroup solves (kStrip in cholesky_gpu.mm)

// chol_panel: a 32-column sub-panel at column j of a panel starting at j0, on
// the large path, in one dispatch. Every threadgroup (x) brings the 32 x 32
// diagonal block and its own strip of the rows below up to date with the
// panel's earlier columns (A -= L(:, j0:j) L(j:j+32, j0:j)^T, by 8 x 8
// simdgroup products, into threadgroup memory), factors the block in
// registers -- the same arithmetic in each, so the same result -- and solves
// its strip against it. The block is never written back to W, where other
// threadgroups still read it: the first threadgroup keeps it in D, and
// chol_put places every block once the factorization is done (nothing reads
// a diagonal block again: the later sub-panels' products and the trailing
// update read only the rows below it). Grid: (strips, matrices), 128
// threads.
kernel void chol_panel(device float*      W    [[buffer(0)]],
                       device uint*       fail [[buffer(1)]],
                       device float*      D    [[buffer(2)]],
                       constant CpParams& p    [[buffer(3)]],
                       uint3 tg   [[threadgroup_position_in_grid]],
                       uint3 tid  [[thread_position_in_threadgroup]],
                       uint3 tpt  [[threads_per_threadgroup]],
                       uint sg    [[simdgroup_index_in_threadgroup]],
                       uint nsg   [[simdgroups_per_threadgroup]],
                       uint lane  [[thread_index_in_simdgroup]])
{
    const uint t = tid.x, T = tpt.x, mat = tg.y;
    if (fail[mat]) return;
    threadgroup float Lt[32][33];
    threadgroup float Y[kPanelStrip][33];       // the strip's rows; first a 32-column chunk of them
    threadgroup float Bs[32][33];               // a 32-column chunk of the block's rows
    threadgroup float (*As)[33] = Y;
    threadgroup float dinv[32];
    threadgroup uint tfail[1];   // (an array: a scalar drew a spurious "uninitialized" warning)
    device float* w = W + (ulong)mat * p.sw;
    const uint ld = p.ld, j = p.j, j0 = p.j0, jb = p.jb;
    const uint s0 = tg.x * kPanelStrip;   // the strip's first row, below the block
    const uint nr = s0 < p.rows ? min(kPanelStrip, p.rows - s0) : 0;   // a multiple of 8
    const uint r0 = j + 32 + s0;
    // Simdgroup s (of 4) owns the block's row of tiles s (tiles (s, 0..s))
    // and the strip's rows of tiles 2s and 2s + 1 (four tiles each): its
    // accumulators start as A and take the panel's earlier columns 32 at a
    // time, every operand staged in threadgroup memory a row of consecutive
    // floats at a time. (Each tile loading its own operands from device
    // memory, a chain of dependent loads, took ~30 us a dispatch.)
    simdgroup_float8x8 cd[4], cs[2][4];
    const bool sr0 = 16 * sg < nr, sr1 = 16 * sg + 8 < nr;   // the strip's rows of tiles present
    // (every loop over J unrolled, so that the accumulators are indexed by
    // constants and stay in registers: indexed by sg they went to memory)
    _Pragma("unroll") for (uint J = 0; J < 4; ++J)
        if (J <= sg) simdgroup_load(cd[J], w + (ulong)(j + 8 * sg) * ld + j + 8 * J, ld);
    _Pragma("unroll") for (uint J = 0; J < 4; ++J) {
        if (sr0) simdgroup_load(cs[0][J], w + (ulong)(r0 + 16 * sg) * ld + j + 8 * J, ld);
        if (sr1) simdgroup_load(cs[1][J], w + (ulong)(r0 + 16 * sg + 8) * ld + j + 8 * J, ld);
    }
    for (uint kc = j0; kc < j; kc += 32) {
        for (uint e = t; e < 32 * 32; e += T) Bs[e / 32][e % 32] = w[(ulong)(j + e / 32) * ld + kc + e % 32];
        for (uint e = t; e < nr * 32; e += T) As[e / 32][e % 32] = w[(ulong)(r0 + e / 32) * ld + kc + e % 32];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        _Pragma("unroll") for (uint kk = 0; kk < 32; kk += 8) {
            simdgroup_float8x8 a, b[4];
            _Pragma("unroll") for (uint J = 0; J < 4; ++J) simdgroup_load(b[J], &Bs[8 * J][kk], 33, ulong2(0, 0), true);
            simdgroup_load(a, &Bs[8 * sg][kk], 33);
            thread auto& ea = a.thread_elements();
            ea[0] = -ea[0];
            ea[1] = -ea[1];
            _Pragma("unroll") for (uint J = 0; J < 4; ++J)
                if (J <= sg) simdgroup_multiply_accumulate(cd[J], a, b[J], cd[J]);
            if (sr0) {
                simdgroup_load(a, &As[16 * sg][kk], 33);
                thread auto& e0 = a.thread_elements();
                e0[0] = -e0[0];
                e0[1] = -e0[1];
                _Pragma("unroll") for (uint J = 0; J < 4; ++J) simdgroup_multiply_accumulate(cs[0][J], a, b[J], cs[0][J]);
            }
            if (sr1) {
                simdgroup_load(a, &As[16 * sg + 8][kk], 33);
                thread auto& e1 = a.thread_elements();
                e1[0] = -e1[0];
                e1[1] = -e1[1];
                _Pragma("unroll") for (uint J = 0; J < 4; ++J) simdgroup_multiply_accumulate(cs[1][J], a, b[J], cs[1][J]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    _Pragma("unroll") for (uint J = 0; J < 4; ++J)
        if (J <= sg) simdgroup_store(cd[J], &Lt[8 * sg][8 * J], 33);
    _Pragma("unroll") for (uint J = 0; J < 4; ++J) {
        if (sr0) simdgroup_store(cs[0][J], &Y[16 * sg][8 * J], 33);
        if (sr1) simdgroup_store(cs[1][J], &Y[16 * sg + 8][8 * J], 33);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float x[32];
        CH_UNROLL(32, c, { x[c] = (lane < jb && c <= lane) ? Lt[lane][c] : 0.0f; });
        const uint f = chol_regs<32, 32>(x, jb, lane, 0);
        if (lane == 0) tfail[0] = f;
        CH_UNROLL(32, c, {
            const float v = (c <= lane && lane < jb && c < jb) ? x[c] : 0.0f;
            Lt[lane][c] = v;
            if (c == lane) dinv[lane] = lane < jb && v != 0.0f ? precise::divide(1.0f, v) : 0.0f;
        });
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint f = tfail[0];
    if (f) {
        if (tg.x == 0 && t == 0) fail[mat] = j + f;
        return;
    }
    if (tg.x == 0) {
        device float* d = D + ((ulong)mat * p.nblk + j / 32) * 1024;
        for (uint e = t; e < 1024; e += T) d[e] = Lt[e / 32][e % 32];
    }
    // the strip's rows by forward substitution, a thread a row, back into Y
    if (t < nr) {
        float y[32];
        CH_UNROLL(32, c, { y[c] = Y[t][c]; });
        CH_UNROLL(32, c, {
            if (c < jb) {
                float s = y[c];
                CH_UNROLL(32, u, { if (u < c) s = fma(-y[u], Lt[c][u], s); });
                y[c] = s * dinv[c];
            }
        });
        CH_UNROLL(32, c, { Y[t][c] = y[c]; });
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // and out, a row of consecutive floats at a time (the columns past n stay 0)
    for (uint e = t; e < nr * 32; e += T) {
        const uint r = e / 32, c = e % 32;
        if (c < jb) w[(ulong)(r0 + r) * ld + j + c] = Y[r][c];
    }
}

// chol_put: the diagonal blocks chol_panel kept in D into place, from the
// panel at column `from` on, once the factorization is done. Grid: (32 x 32
// entries, blocks, matrices).
struct CdParams {
    uint ld, sw, nblk, from;
};

kernel void chol_put(device float*       W    [[buffer(0)]],
                     device const uint*  fail [[buffer(1)]],
                     device const float* D    [[buffer(2)]],
                     constant CdParams&  p    [[buffer(3)]],
                     uint3 gid [[thread_position_in_grid]])
{
    const uint e = gid.x, blk = p.from + gid.y, mat = gid.z;
    if (e >= 1024 || blk >= p.nblk || fail[mat]) return;
    const uint r = e / 32, c = e % 32;
    W[(ulong)mat * p.sw + (ulong)(32 * blk + r) * p.ld + 32 * blk + c] = D[((ulong)mat * p.nblk + blk) * 1024 + e];
}
