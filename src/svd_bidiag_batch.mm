// The SVD's `bidiag_batch` backend (since 2.17.0): the `bidiag` backend's
// method for a whole batch of mid-size matrices at once, as the eigensolver's
// `tridiag_batch` is `tridiag`'s.
//
//   1. Bidiagonalize every matrix by the same dispatches, A = Q B P^T: per
//      panel of 32 columns, bd_panel (shaders/Svd_Bidiag.metal), a threadgroup
//      a matrix, slabrd's steps for the panel's columns with barriers between
//      them, each step reading the trailing block once where slabrd reads it
//      twice (the kernel is bound by memory: 1.3-1.9x from 256 x 256); the
//      trailing update as two batched MPS products; the last columns a final
//      panel. The matrices are loaded and scaled on the GPU (bd_load), a wide
//      one as its transpose.
//   2. B = U_B diag(S) V_B^T on the CPU's cores, a matrix a core (sbdsdc's
//      divide and conquer; dqds for the singular values alone).
//   3. U = Q U_B and V^T = V_B^T P^T as batched products, blocks of 64
//      reflectors, their V and T built on the GPU (bd_make_v, V^T V,
//      bd_make_t); then U and V^T out (bd_store, or bd_copy for a wide
//      matrix, whose column-major factors are the outputs' layout already).
//
// From k = 160 for the singular values alone, and from 384 with vectors, the
// reduction is in two stages instead, as the `band` backend reduces one
// matrix: every matrix to an upper band of width 16 by blocks whose updates
// are batched products (bb_panel for the panels), then the bands to
// bidiagonal on the CPU's cores. With vectors both stages' reflectors are
// kept: the GPU blocks' straight into the layout the back-transformation's
// products read; the CPU, a matrix a core, applies the LAPACK tail's to
// Q = [I; 0] and P = I (the GPU then forms Q1 Q and P1 P while the CPU
// chases), chases keeping the chase's (bd_chase_apply's blocks) and solves
// the bidiagonal problem; the GPU applies the chase's to Q and P, a dispatch
// a matrix, and forms U = Q U_B and V^T = V_B^T P^T. On an M5 Pro 2-2.1x the
// CPU path for 1-4 matrices of 1024 x 1024 with vectors (29 ms for one
// against 57), 2.7x for 16, 2.7-3.4x the direct reduction.
//
// Why: for batches of mid-size matrices (about 96 to 512) every GPU backend
// lost to the CPU path, which spreads a batch over every core. `bidiag` solves
// a batch a matrix at a time, each reduction thousands of dispatches whose
// fixed cost is most of their time; here a batch pays it once. The batch goes
// through in chunks, pipelined over two workspace slots so that the GPU
// reduces chunk k + 1 and back-transforms chunk k - 1 while the CPU solves
// chunk k, the GPU kept a chunk ahead and the host doing nothing else.
// Up to 1024 rows and columns (the panel kernel's threadgroup memory).

#ifndef ACCELERATE_NEW_LAPACK
#define ACCELERATE_NEW_LAPACK
#endif
#include <Accelerate/Accelerate.h>

#include <metal_linalg/core.h>
#include "band_chase.h"
#include "divide_conquer.h"
#include "metal_runtime.h"
#include "shaders.h"

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

using metal_linalg::core::Matrices;
using metal_linalg::detail::AutoreleasePool;
using metal_linalg::detail::HostBuffer;
using metal_linalg::detail::MetalRuntime;
using metal_linalg::detail::Part;
using metal_linalg::detail::make_pipeline;
using metal_linalg::detail::scan;

namespace metal_linalg {
namespace {

constexpr uint32_t kPanel = 32;      // columns a panel; the last panel up to kPanel + 1
constexpr uint32_t kBack  = 64;      // reflectors a block of the back-transformations (BW_MAX in the shader)
constexpr uint32_t kMaxDim = 1024;   // bd_panel's threadgroup memory

// Must match Svd_Bidiag.metal.
struct BpParams { uint32_t mm, nn, lda, ldx, ldy, k, nb, sa, sx, sy, sv; };
struct BlParams { uint32_t rows, cols, m, n, lda, sa, c0; };
struct BsParams { uint32_t rows, cols, c0; };
struct BcParams { uint32_t per, c0; };
struct BwParams { uint32_t len, kb, bb, k0, lda, left, sa, sv, svb, stb; };
struct BqParams { uint32_t p, nb, rs, cs, sa, sv, st, ldv, ldt, dup, s2, ld2, merge, sz, ldt1, st1, s1, sw, ms, sy; };
struct MergeParams { uint32_t np, b, sg, stb, sta; };
struct ChaseParams { uint32_t n, rs, cs, pmax, pass0, pass1; };
struct SlParams { uint32_t n, lda, sa, lower, c0; };
struct SsParams { uint32_t n1, ld, sv, st, ldt; };

// Singular values alone from this k: every matrix to an upper band of width
// kBand by blocks on the GPU (bb_panel and batched products), the bands to
// bidiagonal on the CPU's cores (SVD_BIDIAG_BATCH_BAND=0: bidiagonalized
// directly throughout). On an M5 Pro 1.05-1.1x the direct reduction at
// 160-256, 1.5x at 512, 2-2.3x at 1024; 0.88x at 128, where the CPU's chase
// of the band is the longer stage.
constexpr uint32_t kBand = 16;
constexpr uint32_t kBandMinK = 160;

// With vectors from kBandVectorsMinK: two stages too, both stages'
// reflectors kept and applied (see direct()), the back-transformation's
// blocks aggregated kAgg panels at a time; from kBandVectorsSmallMinK for
// batches of up to as many matrices as the CPU's solve has threads, beyond
// which the CPU's chases bound it. On an M5 Pro against the direct reduction:
// 1.10-1.25x at 288-320 (16 to 256 matrices), 1.18x at 256 x 384^2, 1.6-2.1x
// at 768, 2.7-3.4x at 1024; at 256, 1.29-1.46x for 2-16 matrices, 1.03x for
// 32, 0.91x for 256; at 160, 1.09-1.83x for 1-8, 0.88x for 32; at 128,
// 1.05-1.23x for 2-8.
constexpr uint32_t kBandVectorsMinK = 288;
constexpr uint32_t kBandVectorsSmallMinK = 128;
constexpr uint32_t kAgg = 8;

using L = __LAPACK_int;

struct Cache {
    MetalRuntime& rt = MetalRuntime::shared(METAL_LINALG_SHADER(Svd_Bidiag), "svd_bidiag");
    // bd_panel's instances by rows: a lane's rows of a column in registers, up
    // to 128, 256, 512 and 1024 rows
    id<MTLComputePipelineState> panels[4] = {nil, nil, nil, nil};
    id<MTLComputePipelineState> load = nil, store = nil, copy = nil, make_v = nil, make_t = nil, band_panel = nil;
    id<MTLComputePipelineState> merge_t = nil, chase_wide = nil, chase_narrow = nil, sym_load = nil, sym_small = nil;

    void ensure() {
        if (load) return;
        auto mk = [&](NSString* n) { return make_pipeline(rt.device, rt.library, n, nil); };
        panels[0] = mk(@"bd_panel_4");
        panels[1] = mk(@"bd_panel_8");
        panels[2] = mk(@"bd_panel_16");
        panels[3] = mk(@"bd_panel_32");
        load = mk(@"bd_load");
        store = mk(@"bd_store");
        copy = mk(@"bd_copy");
        make_v = mk(@"bd_make_v");
        make_t = mk(@"bd_make_t");
        band_panel = mk(@"bb_panel");
        merge_t = mk(@"bd_merge_t");
        chase_wide = mk(@"bd_chase_apply_4_4");
        chase_narrow = mk(@"bd_chase_apply_2_8");
        sym_load = mk(@"sb_load");
        sym_small = mk(@"sb_small");
    }
};

MPSMatrix* mps(id<MTLBuffer> b, size_t off, uint32_t rows, uint32_t cols, uint32_t ld) {
    MPSMatrixDescriptor* d = [MPSMatrixDescriptor matrixDescriptorWithRows:rows columns:cols
                                                                  rowBytes:(size_t)ld * 4 dataType:MPSDataTypeFloat32];
    return [[MPSMatrix alloc] initWithBuffer:b offset:off * 4 descriptor:d];
}

// `count` matrices, `stride` floats apart, as band_reduce.mm's: MPS's batched
// products step from one left or result matrix to the next by rows x
// rowBytes, whatever matrixBytes says (macOS 27), so the descriptor is a whole
// stride tall; the buffer needs a stride of slack past the batch's last.
MPSMatrix* mps(id<MTLBuffer> b, size_t off, uint32_t rows, uint32_t cols, uint32_t ld, uint32_t count,
               size_t stride) {
    if (count <= 1) return mps(b, off, rows, cols, ld);
    if (stride % ld != 0 || stride / ld < rows) throw std::logic_error("[svd] bidiag_batch: a batched view's stride");
    MPSMatrixDescriptor* d = [MPSMatrixDescriptor matrixDescriptorWithRows:(uint32_t)(stride / ld) columns:cols
                                                                  matrices:count rowBytes:(size_t)ld * 4
                                                               matrixBytes:stride * 4 dataType:MPSDataTypeFloat32];
    return [[MPSMatrix alloc] initWithBuffer:b offset:off * 4 descriptor:d];
}

// The products' kernels by shape, reused; a thread's own (MPS kernels are not
// thread-safe).
void gemm(id<MTLDevice> dev, id<MTLCommandBuffer> cb, MPSMatrix* A, bool ta, MPSMatrix* B, bool tb, MPSMatrix* C,
          uint32_t m, uint32_t n, uint32_t k, double alpha, double beta, uint32_t batch) {
    using Key = std::tuple<uint32_t, uint32_t, uint32_t, int, double, double>;
    thread_local std::map<Key, MPSMatrixMultiplication*> kernels;
    const Key key{m, n, k, (ta ? 1 : 0) | (tb ? 2 : 0), alpha, beta};
    auto it = kernels.find(key);
    if (it == kernels.end()) {
        if (kernels.size() >= 4096) kernels.clear();
        MPSMatrixMultiplication* g = [[MPSMatrixMultiplication alloc] initWithDevice:dev transposeLeft:ta
            transposeRight:tb resultRows:m resultColumns:n interiorColumns:k alpha:alpha beta:beta];
        it = kernels.insert_or_assign(key, g).first;
    }
    it->second.batchStart = 0;
    it->second.batchSize = batch;
    [it->second encodeToCommandBuffer:cb leftMatrix:A rightMatrix:B resultMatrix:C];
}

// A pipeline slot's buffers for up to `capacity` matrices of m x n (m >= n),
// each buffer's matrices its stride apart, with a stride of slack past the
// last for the batched products' views.
struct Work {
    uint32_t m = 0, n = 0, capacity = 0;
    bool     vectors = false, band = false;
    uint32_t lda = 0, sa = 0, sx = 0, sy = 0, sv = 0, su = 0, svt = 0, svb = 0, stb = 0, sz = 0;
    id<MTLBuffer> A, X, Y, d, e, tq, tp;   // the reduction
    id<MTLBuffer> U, VT;                   // m x n and n x n, column-major
    id<MTLBuffer> V, G, T, Z, Z2;          // the back-transformations'
    // the reduction to bands: [V1 Q] (m + kBand rows, 2 kBand wide) and [W V2]
    // (n x 2 kBand), row-major, for the block's one rank-2 kBand update; the
    // panels' T; Z = A^T V1 and V2 T2
    uint32_t sb1 = 0, sb2 = 0, sbz = 0;
    id<MTLBuffer> BR, BL, BT1, BT2, BZ, BZ2;
    // with vectors in two stages: the blocks' reflectors, QY (m x n) and PY
    // (n x n) row-major, a panel's V at its row and column (PY's a block's
    // row panel at row k + kBand), zeros above, and their T's, 32 x 32 slots;
    // the Gram matrix and aggregated T of the back-transformation's blocks;
    // Q and P (column-major, ld ldq and ldp, multiples of 32 for
    // bd_chase_apply); U_B and V_B^T (column-major n x n); the chase's blocks
    // (bd_chase_apply's); the outputs; the products' scratch
    bool bandv = false;
    uint32_t blocks = 0, ldq = 0, ldp = 0;
    size_t sqy = 0, spy = 0, stq = 0, sq = 0, sp = 0, sub = 0, slv = 0, so = 0, sza = 0;
    id<MTLBuffer> QY, PY, QT, PT, GA, TA, Q, P, UB, VTB, LV, RV, OU, OV, ZA, ZA2;
};

// Floats a matrix of the two-stage reduction with vectors takes in a slot.
size_t bandv_floats(uint32_t m, uint32_t n) {
    const size_t pmax = n >= 2 ? (n - 2) / 16 : 0, nblocks = (pmax + 1) * (pmax + 2) / 2;
    const size_t ldq = (m + 31) / 32 * 32, ldp = (n + 31) / 32 * 32, blocks = n / kBand + 1;
    return (size_t)m * n * 4 + (size_t)n * n * 3 + 2 * blocks * 1024 + ldq * n + ldp * n +
           2 * nblocks * metal_linalg::detail::kChaseBlockFloats + 2 * (size_t)std::max(m, n) * kAgg * kBand +
           2 * 128 * 128;
}

Work& work(Cache& c, uint32_t m, uint32_t n, uint32_t capacity, bool vectors, int slot, bool band = false,
           bool bandv = false) {
    static Work ws[2];
    Work& w = ws[slot];
    if (w.A && w.m == m && w.n == n && w.capacity >= capacity && (w.vectors || !vectors) && (w.band || !band) &&
        (w.bandv || !bandv))
        return w;
    id<MTLDevice> dev = c.rt.device;
    auto make = [&](size_t floats, MTLResourceOptions opt) {
        id<MTLBuffer> b = [dev newBufferWithLength:std::max<size_t>(floats, 4) * 4 options:opt];
        if (!b) throw std::runtime_error("[svd] bidiag_batch: could not allocate " + std::to_string(floats * 4) + " bytes");
        return b;
    };
    const MTLResourceOptions shared = MTLResourceStorageModeShared, priv = MTLResourceStorageModePrivate;
    w = Work{};
    w.m = m;
    w.n = n;
    w.capacity = capacity;
    w.vectors = vectors;
    w.band = band;
    w.bandv = bandv;
    // The band reduction's blocks keep their W^T in 16 spare rows below each
    // matrix (encode_band)
    w.lda = (m + 7) / 8 * 8 + (band ? kBand : 0);
    w.sa = w.lda * n;
    w.sx = m * (kPanel + 1);
    w.sy = n * (kPanel + 1);
    w.sv = n;
    const size_t C = capacity + 1;
    w.A = make(C * w.sa, shared);
    w.X = make(C * w.sx, priv);
    w.Y = make(C * w.sy, priv);
    w.d = make(C * n, shared);
    w.e = make(C * n, shared);
    w.tq = make(C * n, shared);
    w.tp = make(C * n, shared);
    if (band) {
        w.sb1 = (m + kBand) * 2 * kBand;
        w.sb2 = n * 2 * kBand;
        w.sbz = m * kBand;
        w.BR = make(C * w.sb1, priv);
        w.BL = make(C * w.sb2, priv);
        w.BT1 = make(C * kBand * kBand, priv);
        w.BT2 = make(C * kBand * kBand, priv);
        w.BZ = make(C * w.sbz, priv);
        w.BZ2 = make(C * w.sbz, priv);
    }
    if (bandv) {
        const size_t pmax = n >= 2 ? (n - 2) / 16 : 0;
        w.blocks = n / kBand + 1;   // at least the GPU's blocks
        w.ldq = (m + 31) / 32 * 32;
        w.ldp = (n + 31) / 32 * 32;
        w.sqy = (size_t)m * n;
        w.spy = (size_t)n * n;
        w.stq = (size_t)w.blocks * 1024;
        w.sq = (size_t)w.ldq * n;
        w.sp = (size_t)w.ldp * n;
        w.sub = (size_t)n * n;
        w.slv = (pmax + 1) * (pmax + 2) / 2 * metal_linalg::detail::kChaseBlockFloats;
        w.so = (size_t)m * n;
        w.sza = (size_t)std::max(m, n) * kAgg * kBand;
        w.QY = make(C * w.sqy, shared);
        w.PY = make(C * w.spy, shared);
        std::memset(w.QY.contents, 0, C * w.sqy * 4);   // the zeros above each panel, never written
        std::memset(w.PY.contents, 0, C * w.spy * 4);
        w.QT = make(C * w.stq, priv);
        w.PT = make(C * w.stq, priv);
        w.GA = make(C * 128 * 128, priv);
        w.TA = make(C * 128 * 128, priv);
        w.Q = make(C * w.sq, shared);
        w.P = make(C * w.sp, shared);
        w.UB = make(C * w.sub, shared);
        w.VTB = make(C * w.sub, shared);
        w.LV = make(C * w.slv, shared);
        w.RV = make(C * w.slv, shared);
        w.OU = make(C * w.so, priv);
        w.OV = make(C * w.so, priv);
        w.ZA = make(C * w.sza, priv);
        w.ZA2 = make(C * w.sza, priv);
    }
    if (vectors) {
        w.su = m * n;
        w.svt = n * n;
        w.svb = m * kBack;
        w.stb = kBack * kBack;
        w.sz = n * kBack;
        w.U = make(C * w.su, shared);
        w.VT = make(C * w.svt, shared);
        w.V = make(C * w.svb, priv);
        w.G = make(C * w.stb, priv);
        w.T = make(C * w.stb, priv);
        w.Z = make(C * w.sz, priv);
        w.Z2 = make(C * w.sz, priv);
    }
    return w;
}

// Step 1 for `cnt` matrices from index[c0], encoded into cb: loaded, then the
// panels, the last columns a final panel.
void encode_reduce(Cache& c, id<MTLCommandBuffer> cb, Work& w, uint32_t cnt, id<MTLBuffer> src, id<MTLBuffer> index,
                   id<MTLBuffer> scale, uint32_t c0, uint32_t rows, uint32_t cols, uint32_t threads) {
    id<MTLDevice> dev = c.rt.device;
    const uint32_t m = w.m, n = w.n, lda = w.lda, nb = kPanel;
    id<MTLComputePipelineState> panel = c.panels[m <= 128 ? 0 : m <= 256 ? 1 : m <= 512 ? 2 : 3];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    const BlParams lp{rows, cols, m, n, lda, w.sa, c0};
    [enc setComputePipelineState:c.load];
    [enc setBuffer:src offset:0 atIndex:0];
    [enc setBuffer:w.A offset:0 atIndex:1];
    [enc setBuffer:index offset:0 atIndex:2];
    [enc setBuffer:scale offset:0 atIndex:3];
    [enc setBytes:&lp length:sizeof lp atIndex:4];
    [enc dispatchThreadgroups:MTLSizeMake((m + 31) / 32, (n + 31) / 32, cnt) threadsPerThreadgroup:MTLSizeMake(32, 8, 1)];
    for (uint32_t k = 0;; k += nb) {
        const uint32_t mm = m - k, nn = n - k, last = k + nb + 1 >= n, width = last ? nn : nb;
        const size_t ok = (size_t)k * lda + k;
        const BpParams pp{mm, nn, lda, m, n, k, width, w.sa, w.sx, w.sy, w.sv};
        [enc setComputePipelineState:panel];
        [enc setBuffer:w.A offset:ok * 4 atIndex:0];
        [enc setBuffer:w.X offset:0 atIndex:1];
        [enc setBuffer:w.Y offset:0 atIndex:2];
        [enc setBuffer:w.d offset:0 atIndex:3];
        [enc setBuffer:w.e offset:0 atIndex:4];
        [enc setBuffer:w.tq offset:0 atIndex:5];
        [enc setBuffer:w.tp offset:0 atIndex:6];
        [enc setBytes:&pp length:sizeof pp atIndex:7];
        [enc setThreadgroupMemoryLength:((size_t)(2 * mm + nn) * 4 + 15) / 16 * 16 atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(cnt, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(std::min<uint32_t>(threads, (uint32_t)panel.maxTotalThreadsPerThreadgroup),
                                              1, 1)];
        if (last) break;
        [enc endEncoding];
        // A22 -= V Y2^T + X2 U, as A22^T -= Y2 V^T + U^T X2^T on the row-major
        // views of the column-major buffers (as bidiagonalize's)
        const uint32_t m2 = mm - nb, n2 = nn - nb;
        MPSMatrix* A22t = mps(w.A, ok + (size_t)nb * lda + nb, n2, m2, lda, cnt, w.sa);
        gemm(dev, cb, mps(w.Y, nb, nb, n2, n, cnt, w.sy), true, mps(w.A, ok + nb, nb, m2, lda, cnt, w.sa), false,
             A22t, n2, m2, nb, -1, 1, cnt);
        gemm(dev, cb, mps(w.A, ok + (size_t)nb * lda, n2, nb, lda, cnt, w.sa), false, mps(w.X, nb, nb, m2, m, cnt, w.sx),
             false, A22t, n2, m2, nb, -1, 1, cnt);
        enc = [cb computeCommandEncoder];
    }
    [enc endEncoding];
}

// Step 1 in two stages' first: `cnt` matrices loaded (as encode_reduce's),
// then reduced to an upper band of width b = kBand by blocks while three
// blocks' columns remain, as band_reduce.mm reduces one matrix: per block the
// column panel's QR (bb_panel), H1 = I - V1 T1 V1^T, applied to the columns
// right of it, then the row panel's LQ (bb_panel on its transpose), G = I -
// V2 T2 V2^T, applied to the rows below; the products batched, on the
// row-major views of the column-major matrices. The two updates are merged,
// as slabrd merges a column's: Z = A(k:, k+b:)^T V1 (one read); the row
// panel's kernel forms W = Z T1, updates its b rows from W alone, factors
// them and forms V2 T2, writing W^T into the 16 spare rows below the matrix;
// then one product gives both A22 V2 T2 and W^T V2 T2 (one read), and with
// Q = A22 V2 T2 - V1b (W^T V2 T2), A22 -= [W V2] [V1b Q]^T is one rank-2b
// product (one read and write): three passes over the trailing matrix and
// five dispatches a block where applying H1 then G took four and eight. Returns
// the first column left to the CPU. With `keep` (vectors), each block's
// reflectors into QY, PY and their T's into QT, PT too, where the
// back-transformation reads them.
uint32_t encode_band(Cache& c, id<MTLCommandBuffer> cb, Work& w, uint32_t cnt, id<MTLBuffer> src, id<MTLBuffer> index,
                     id<MTLBuffer> scale, uint32_t c0, uint32_t rows, uint32_t cols, bool keep = false) {
    id<MTLDevice> dev = c.rt.device;
    const uint32_t m = w.m, n = w.n, lda = w.lda, b = kBand;
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    const BlParams lp{rows, cols, m, n, lda, w.sa, c0};
    [enc setComputePipelineState:c.load];
    [enc setBuffer:src offset:0 atIndex:0];
    [enc setBuffer:w.A offset:0 atIndex:1];
    [enc setBuffer:index offset:0 atIndex:2];
    [enc setBuffer:scale offset:0 atIndex:3];
    [enc setBytes:&lp length:sizeof lp atIndex:4];
    [enc dispatchThreadgroups:MTLSizeMake((m + 31) / 32, (n + 31) / 32, cnt) threadsPerThreadgroup:MTLSizeMake(32, 8, 1)];
    // A panel's V into V (ld 2b, sv apart) and, with keep, into V2 too (ld2,
    // s2); with merge (a row panel), the block's left update first and V2 T2
    // after (bb_panel)
    auto panel = [&](size_t off, uint32_t p, uint32_t rs, uint32_t cs, id<MTLBuffer> V, size_t voff, size_t sv,
                     id<MTLBuffer> T, size_t toff, size_t st, uint32_t ldt, id<MTLBuffer> V2, size_t v2off,
                     size_t s2, bool merge, uint32_t ms, id<MTLBuffer> T1) {
        const BqParams q{p, b, rs, cs, w.sa, (uint32_t)sv, (uint32_t)st, 2 * b, ldt, 0, (uint32_t)s2, V2 ? n : 0u,
                         merge ? 3u : 0u, w.sbz, ldt, (uint32_t)st, w.sb1, w.sb2, ms, w.sbz};
        [enc setComputePipelineState:c.band_panel];
        [enc setBuffer:w.A offset:off * 4 atIndex:0];
        [enc setBuffer:V offset:voff * 4 atIndex:1];
        [enc setBuffer:T offset:toff * 4 atIndex:2];
        [enc setBytes:&q length:sizeof q atIndex:3];
        [enc setBuffer:V2 ? V2 : V offset:(V2 ? v2off : voff) * 4 atIndex:4];   // (unused without keep)
        [enc setBuffer:w.BZ offset:0 atIndex:5];                                // Z
        [enc setBuffer:T1 ? T1 : T offset:toff * 4 atIndex:6];                  // T1: the column panel's
        [enc setBuffer:w.BR offset:0 atIndex:7];                                // V1
        [enc setBuffer:w.BL offset:0 atIndex:8];                                // [W .]
        [enc setBuffer:w.BZ2 offset:0 atIndex:9];                               // V2 T2
        [enc dispatchThreadgroups:MTLSizeMake(cnt, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(std::max<uint32_t>(32, (p + 31) / 32 * 32), 1, 1)];
    };
    uint32_t k = 0;
    for (; k + 3 * b <= n; k += b) {
        const uint32_t p = m - k, n2 = n - k - b, m2 = m - k - b, j = k / b;
        // T's: scratch, or (keep) QT/PT, 32 x 32 slots
        id<MTLBuffer> t1b = keep ? w.QT : w.BT1, t2b = keep ? w.PT : w.BT2;
        const size_t to = keep ? (size_t)j * 1024 : 0, st = keep ? w.stq : (size_t)b * b;
        const uint32_t ldt = keep ? 32 : b;
        MPSMatrix* S = mps(w.A, (size_t)(k + b) * lda + k, n2, p, lda, cnt, w.sa);   // A(k:, k+b:)^T
        MPSMatrix* S2 = mps(w.A, (size_t)(k + b) * lda + k + b, n2, m2, lda, cnt, w.sa);   // A22^T
        MPSMatrix* V1 = mps(w.BR, 0, p, b, 2 * b, cnt, w.sb1);
        MPSMatrix* Z = mps(w.BZ, 0, n2, b, b, cnt, w.sbz);
        // The column panel A(k:, k:k+b): R in place, V1 into [V1 .] (rows 0..p)
        panel((size_t)k * lda + k, p, 1, lda, w.BR, 0, w.sb1, t1b, to, st, ldt, keep ? w.QY : nil, (size_t)k * n + k,
              w.sqy, false, 0, nil);
        [enc endEncoding];
        // Z = S V1 (A not yet updated)
        gemm(dev, cb, S, false, V1, false, Z, n2, b, p, 1, 0, cnt);
        // The row panel A(k:k+b, k+b:) through its transpose (n2 x b): W = Z
        // T1 into [W .] and A's spare rows, its rows updated, L in place, V2
        // into [W V2], V2 T2
        enc = [cb computeCommandEncoder];
        panel((size_t)(k + b) * lda + k, n2, lda, 1, w.BL, b, w.sb2, t2b, to, st, ldt, keep ? w.PY : nil,
              (size_t)(k + b) * n + k, w.spy, true, m - k, t1b);
        [enc endEncoding];
        // [A22; W^T] V2 T2 into [. Q] and the 16 rows after it; Q -= V1b (W^T
        // V2 T2); A22^T -= [W V2] [V1b Q]^T
        gemm(dev, cb, mps(w.A, (size_t)(k + b) * lda + k + b, n2, m2 + b, lda, cnt, w.sa), true,
             mps(w.BZ2, 0, n2, b, b, cnt, w.sbz), false, mps(w.BR, (size_t)b * 2 * b + b, m2 + b, b, 2 * b, cnt, w.sb1),
             m2 + b, b, n2, 1, 0, cnt);
        gemm(dev, cb, mps(w.BR, (size_t)b * 2 * b, m2, b, 2 * b, cnt, w.sb1), false,
             mps(w.BR, (size_t)(b + m2) * 2 * b + b, b, b, 2 * b, cnt, w.sb1), false,
             mps(w.BR, (size_t)b * 2 * b + b, m2, b, 2 * b, cnt, w.sb1), m2, b, b, -1, 1, cnt);
        gemm(dev, cb, mps(w.BL, 0, n2, 2 * b, 2 * b, cnt, w.sb2), false,
             mps(w.BR, (size_t)b * 2 * b, m2, 2 * b, 2 * b, cnt, w.sb1), true, S2, n2, m2, 2 * b, -1, 1, cnt);
        enc = [cb computeCommandEncoder];
    }
    [enc endEncoding];
    return k;
}

// Step 3 for `cnt` matrices, encoded into cb: U <- Q U and V^T <- V^T P^T,
// blocks of kBack reflectors last first, their V and T built on the GPU.
void encode_back(Cache& c, id<MTLCommandBuffer> cb, Work& w, uint32_t cnt) {
    id<MTLDevice> dev = c.rt.device;
    const uint32_t m = w.m, n = w.n, bb = kBack;
    for (int side = 0; side < 2; ++side) {
        const bool left = side == 0;
        const uint32_t refl = left ? n : n - 1, off = left ? 0 : 1;
        if (refl == 0) continue;
        for (int k0 = (int)(((refl - 1) / bb) * bb); k0 >= 0; k0 -= (int)bb) {
            const uint32_t kb = std::min<uint32_t>(bb, refl - (uint32_t)k0);
            const uint32_t len = (left ? m : n) - (uint32_t)k0 - off;
            const BwParams bp{len, kb, bb, (uint32_t)k0, w.lda, left ? 1u : 0u, w.sa, w.sv, w.svb, w.stb};
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:c.make_v];
            [enc setBuffer:w.A offset:0 atIndex:0];
            [enc setBuffer:w.V offset:0 atIndex:1];
            [enc setBytes:&bp length:sizeof bp atIndex:2];
            [enc dispatchThreads:MTLSizeMake(bb, len, cnt) threadsPerThreadgroup:MTLSizeMake(bb, 1, 1)];
            [enc endEncoding];
            MPSMatrix* Vm = mps(w.V, 0, len, bb, bb, cnt, w.svb);
            gemm(dev, cb, Vm, true, Vm, false, mps(w.G, 0, bb, bb, bb, cnt, w.stb), bb, bb, len, 1, 0, cnt);
            enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:c.make_t];
            [enc setBuffer:w.G offset:0 atIndex:0];
            [enc setBuffer:left ? w.tq : w.tp offset:0 atIndex:1];
            [enc setBuffer:w.T offset:0 atIndex:2];
            [enc setBytes:&bp length:sizeof bp atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake(cnt, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc endEncoding];
            MPSMatrix* Tm = mps(w.T, 0, bb, bb, bb, cnt, w.stb);   // T, row-major
            if (left) {
                // U column-major (ld m) is U^T row-major: U^T(:, k0:) -= ((U^T(:, k0:) V) T^T) V^T
                MPSMatrix* S = mps(w.U, (size_t)k0, n, len, m, cnt, w.su);
                MPSMatrix* Z = mps(w.Z, 0, n, bb, bb, cnt, w.sz);
                MPSMatrix* Z2 = mps(w.Z2, 0, n, bb, bb, cnt, w.sz);
                gemm(dev, cb, S, false, Vm, false, Z, n, bb, len, 1, 0, cnt);
                gemm(dev, cb, Z, false, Tm, true, Z2, n, bb, bb, 1, 0, cnt);
                gemm(dev, cb, Z2, false, Vm, true, S, n, len, bb, -1, 1, cnt);
            } else {
                // V^T column-major is V^T^T row-major: its rows k0+1.. -= V T (V^T rows)
                MPSMatrix* S = mps(w.VT, (size_t)(k0 + 1) * n, len, n, n, cnt, w.svt);
                MPSMatrix* Z = mps(w.Z, 0, bb, n, n, cnt, w.sz);
                MPSMatrix* Z2 = mps(w.Z2, 0, bb, n, n, cnt, w.sz);
                gemm(dev, cb, Vm, true, S, false, Z, bb, n, len, 1, 0, cnt);
                gemm(dev, cb, Tm, false, Z, false, Z2, bb, n, bb, 1, 0, cnt);
                gemm(dev, cb, Vm, false, Z2, false, S, len, n, bb, -1, 1, cnt);
            }
        }
    }
}

// Step 3 with vectors in two stages, for `cnt` matrices whose `blocks` GPU
// blocks were kept (encode_band). encode_q1_p1, once solve_tail has left Q =
// Q_tail [I; 0] and P = P_tail in the slot: Q <- Q1 Q and P <- P1 P, kAgg
// panels an aggregate, last first (I - Y Ta Y^T, Ta merged by bd_merge_t
// from the Gram matrix Y^T Y and the panels' T's), on the row-major views Q^T
// and P^T: Z <- Z - ((Z Y) Ta^T) Y^T. encode_back_band, once solve_chase has
// left the chase's blocks, U_B and V_B^T: Q <- Q Q2 and P <- P P2
// (bd_chase_apply, a dispatch a matrix, the matrices at once); then the
// outputs: U = Q U_B and V^T = V_B^T P^T, or for a wide matrix's transpose
// U = P V_B and V^T = U_B^T Q^T, row-major, copied to matrix index[c0 + j].
void encode_q1_p1(Cache& c, id<MTLCommandBuffer> cb, Work& w, uint32_t cnt, uint32_t blocks) {
    id<MTLDevice> dev = c.rt.device;
    const uint32_t m = w.m, n = w.n, b = kBand, aggs = (blocks + kAgg - 1) / kAgg;
    for (int a = (int)aggs - 1; a >= 0; --a) {
        const uint32_t j0 = (uint32_t)a * kAgg, np = std::min(kAgg, blocks - j0), wa = np * b;
        for (int side = 0; side < 2; ++side) {
            const bool left = side == 0;
            const uint32_t r0 = j0 * b + (left ? 0 : b), len = (left ? m : n) - r0, cols = n - r0;
            const uint32_t ld = left ? w.ldq : w.ldp;
            MPSMatrix* Y = left ? mps(w.QY, (size_t)r0 * n + r0, len, wa, n, cnt, w.sqy)
                                : mps(w.PY, (size_t)r0 * n + j0 * b, len, wa, n, cnt, w.spy);
            gemm(dev, cb, Y, true, Y, false, mps(w.GA, 0, wa, wa, 128, cnt, 128 * 128), wa, wa, len, 1, 0, cnt);
            const MergeParams mp{np, b, 128 * 128, (uint32_t)w.stq, 128 * 128};
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:c.merge_t];
            [enc setBuffer:w.GA offset:0 atIndex:0];
            [enc setBuffer:left ? w.QT : w.PT offset:(size_t)j0 * 1024 * 4 atIndex:1];
            [enc setBuffer:w.TA offset:0 atIndex:2];
            [enc setBytes:&mp length:sizeof mp atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake(1, 1, cnt) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
            [enc endEncoding];
            MPSMatrix* Zm = mps(left ? w.Q : w.P, (size_t)r0 * ld + r0, cols, len, ld, cnt, left ? w.sq : w.sp);
            MPSMatrix* Ta = mps(w.TA, 0, wa, wa, 128, cnt, 128 * 128);
            MPSMatrix* W = mps(w.ZA, 0, cols, wa, kAgg * b, cnt, w.sza);
            MPSMatrix* W2 = mps(w.ZA2, 0, cols, wa, kAgg * b, cnt, w.sza);
            gemm(dev, cb, Zm, false, Y, false, W, cols, wa, len, 1, 0, cnt);
            gemm(dev, cb, W, false, Ta, true, W2, cols, wa, wa, 1, 0, cnt);
            gemm(dev, cb, W2, false, Y, true, Zm, cols, len, wa, -1, 1, cnt);
        }
    }
}

void encode_back_band(Cache& c, id<MTLCommandBuffer> cb, Work& w, uint32_t cnt, bool wide, id<MTLBuffer> uo,
                      id<MTLBuffer> vo, id<MTLBuffer> index, uint32_t c0) {
    id<MTLDevice> dev = c.rt.device;
    const uint32_t m = w.m, n = w.n;
    // Q <- Q Q2, P <- P P2: on X = Q^T (n rows, ldq columns) and P^T
    if (n >= 3) {
        const uint32_t pmax = (n - 2) / 16, groups = pmax + 1;
        for (int side = 0; side < 2; ++side) {
            const bool left = side == 0;
            const uint32_t ld = left ? w.ldq : w.ldp;
            const bool widek = ld / 32 >= 64;
            const uint32_t C = widek ? 32 : 16, K = widek ? 4 : 8;
            const ChaseParams q{n, ld, 1, pmax, 0, (groups + K - 1) / K};
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
            [enc setComputePipelineState:widek ? c.chase_wide : c.chase_narrow];
            [enc setBytes:&q length:sizeof q atIndex:2];
            for (uint32_t j = 0; j < cnt; ++j) {
                [enc setBuffer:left ? w.Q : w.P offset:(size_t)j * (left ? w.sq : w.sp) * 4 atIndex:0];
                [enc setBuffer:left ? w.LV : w.RV offset:(size_t)j * w.slv * 4 atIndex:1];
                [enc dispatchThreadgroups:MTLSizeMake(ld / C, 1, 1) threadsPerThreadgroup:MTLSizeMake(32 * K, 1, 1)];
            }
            [enc endEncoding];
        }
    }
    // The outputs, row-major, on the row-major views of the column-major
    // factors: Q^T (n x m, ld ldq), P^T (n x n, ld ldp), U_B^T and V_B
    MPSMatrix* Qr = mps(w.Q, 0, n, m, w.ldq, cnt, w.sq);
    MPSMatrix* Pr = mps(w.P, 0, n, n, w.ldp, cnt, w.sp);
    MPSMatrix* UBr = mps(w.UB, 0, n, n, n, cnt, w.sub);
    MPSMatrix* VBr = mps(w.VTB, 0, n, n, n, cnt, w.sub);
    const size_t pu = wide ? (size_t)n * n : (size_t)m * n, pv = wide ? (size_t)n * m : (size_t)n * n;
    if (!wide) {
        gemm(dev, cb, Qr, true, UBr, true, mps(w.OU, 0, m, n, n, cnt, pu), m, n, n, 1, 0, cnt);    // Q U_B
        gemm(dev, cb, VBr, true, Pr, false, mps(w.OV, 0, n, n, n, cnt, pv), n, n, n, 1, 0, cnt);   // V_B^T P^T
    } else {
        gemm(dev, cb, Pr, true, VBr, false, mps(w.OU, 0, n, n, n, cnt, pu), n, n, n, 1, 0, cnt);   // P V_B
        gemm(dev, cb, UBr, false, Qr, false, mps(w.OV, 0, n, m, m, cnt, pv), n, m, n, 1, 0, cnt);  // U_B^T Q^T
    }
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:c.copy];
    for (int f = 0; f < 2; ++f) {
        const BcParams cp{(uint32_t)(f ? pv : pu), c0};
        [enc setBuffer:f ? w.OV : w.OU offset:0 atIndex:0];
        [enc setBuffer:f ? vo : uo offset:0 atIndex:1];
        [enc setBuffer:index offset:0 atIndex:2];
        [enc setBytes:&cp length:sizeof cp atIndex:3];
        [enc dispatchThreads:MTLSizeMake(cp.per, cnt, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }
    [enc endEncoding];
}

// Step 2 with vectors in two stages, for the slot's `cnt` matrices on the
// CPU's cores, a matrix a core (their threads shared out when there are
// fewer matrices than threads), in two parts so that the GPU can form Q1
// and P1 (encode_q1_p1) while the CPU chases:
//   solve_tail: the last columns from `tail` on by LAPACK, keeping their
//   reflectors, and Q = Q_tail [I; 0], P = P_tail, column-major;
//   solve_chase: the band chased to bidiagonal, the chase's reflectors into
//   the slot's blocks (bd_chase_apply's, Y built beside V), and B = U_B S
//   V_B^T by the divide and conquer. done(j, d) takes matrix j's singular
//   values; which(j) names it in errors.
void solve_tail(Work& w, uint32_t cnt, uint32_t tail, unsigned threads) {
    const uint32_t m = w.m, n = w.n;
    float* A = static_cast<float*>(w.A.contents);
    float* Q = static_cast<float*>(w.Q.contents);
    float* P = static_cast<float*>(w.P.contents);
    metal_linalg::detail::lapack_batches(cnt, (size_t)m * n, threads, [&](uint32_t b0, uint32_t b1) {
        std::vector<float> work((size_t)std::max(m, n) * 64 + 64);
        for (uint32_t j = b0; j < b1; ++j) {
            float* Aj = A + (size_t)j * w.sa;
            metal_linalg::detail::BandKeep keep;
            metal_linalg::detail::band_general_tail(Aj, m, n, w.lda, kBand, tail, &keep);
            // Q = [I; 0] and P = I, then the tail's reflectors, last step first:
            // Q(k:, k:) <- H Q(k:, k:), P(k+bk:, k+bk:) <- G P(k+bk:, k+bk:)
            float* Qj = Q + (size_t)j * w.sq;
            float* Pj = P + (size_t)j * w.sp;
            for (uint32_t c = 0; c < n; ++c) {
                std::fill(Qj + (size_t)c * w.ldq, Qj + (size_t)(c + 1) * w.ldq, 0.0f);
                Qj[(size_t)c * w.ldq + c] = 1.0f;
                std::fill(Pj + (size_t)c * w.ldp, Pj + (size_t)(c + 1) * w.ldp, 0.0f);
                Pj[(size_t)c * w.ldp + c] = 1.0f;
            }
            L lw = (L)work.size(), info = 0;
            for (auto it = keep.steps.rbegin(); it != keep.steps.rend(); ++it) {
                const auto& st = *it;
                if (st.nr > 0) {
                    L N = st.nr, Kt = (L)st.tp.size(), LD = st.bk, LDP = w.ldp;
                    sormlq_("L", "T", &N, &N, &Kt, const_cast<float*>(st.lq.data()), &LD,
                            const_cast<float*>(st.tp.data()), Pj + (size_t)(st.k + st.bk) * w.ldp + st.k + st.bk, &LDP,
                            work.data(), &lw, &info);
                }
                L Mq = m - st.k, Nq = n - st.k, Kq = st.bk, LDA = w.lda, LDQ = w.ldq;
                sormqr_("L", "N", &Mq, &Nq, &Kq, Aj + (size_t)st.k * w.lda + st.k, &LDA, const_cast<float*>(st.tq.data()),
                        Qj + (size_t)st.k * w.ldq + st.k, &LDQ, work.data(), &lw, &info);
            }
        }
    });
}

template <class Done, class Which>
void solve_chase(Work& w, uint32_t cnt, unsigned threads, const Done& done, const Which& which) {
    const uint32_t m = w.m, n = w.n, b = kBand;
    const size_t pmax = n >= 2 ? (n - 2) / 16 : 0, nblocks = (pmax + 1) * (pmax + 2) / 2;
    const long groups = (long)pmax + 1;
    const unsigned per = std::max(1u, threads / std::max(1u, std::min(cnt, threads)));
    float* A = static_cast<float*>(w.A.contents);
    float* UB = static_cast<float*>(w.UB.contents);
    float* VTB = static_cast<float*>(w.VTB.contents);
    float* LV = static_cast<float*>(w.LV.contents);
    float* RV = static_cast<float*>(w.RV.contents);
    metal_linalg::detail::lapack_batches(cnt, (size_t)m * n, threads, [&](uint32_t b0, uint32_t b1) {
        std::vector<float> dj(n), ej(n), ltau(nblocks * 16), rtau(nblocks * 16);
        const size_t ld = 3 * (size_t)b + 1, ku = 2 * (size_t)b;
        std::vector<float> ab(ld * n);
        for (uint32_t j = b0; j < b1; ++j) {
            const float* Aj = A + (size_t)j * w.sa;
            std::fill(ab.begin(), ab.end(), 0.0f);
            for (uint32_t c = 0; c < n; ++c)
                for (uint32_t r = c > b ? c - b : 0; r <= c; ++r) ab[(size_t)c * ld + ku + r - c] = Aj[(size_t)c * w.lda + r];
            metal_linalg::detail::ChaseReflectors rec;
            rec.L = LV + (size_t)j * w.slv;
            rec.Ltau = ltau.data();
            rec.R = RV + (size_t)j * w.slv;
            rec.Rtau = rtau.data();
            rec.pmax = pmax;
            metal_linalg::detail::band_to_bidiagonal(n, b, ab.data(), ld, ku, dj.data(), ej.data(), per, &rec);
            // Both sides' blocks, by groups over the matrix's threads (single
            // threaded, 3 ms of one 1024 x 1024's solve)
            const long tasks = 2 * groups, chunks = std::min<long>(tasks, 4 * (long)per);
            auto build = [&](size_t t) {
                for (long g = tasks * (long)t / chunks; g < tasks * ((long)t + 1) / chunks; ++g) {
                    const bool left = g < groups;
                    const long G = left ? g : g - groups;
                    metal_linalg::detail::chase_build_blocks(left ? rec.L : rec.R, left ? rec.Ltau : rec.Rtau, n, G, G + 1);
                }
            };
            if (per > 1) metal_linalg::detail::parallel_for((size_t)chunks, build);
            else for (long t = 0; t < chunks; ++t) build((size_t)t);
            const long dinfo = metal_linalg::detail::bidiagonal_svd(n, dj.data(), ej.data(), UB + (size_t)j * w.sub, n,
                                                                    VTB + (size_t)j * w.sub, n, per);
            if (dinfo != 0)
                throw std::runtime_error("[svd] bidiag_batch: the divide and conquer failed on matrix " +
                                         std::to_string(which(j)) + ", info " + std::to_string(dinfo) + ".");
            done(j, dj.data());
        }
    });
}

Cache& shared_cache() {
    static Cache cache;
    cache.ensure();
    return cache;
}

// Every matrix bidiagonalized as it is (a wide one as its transpose).
void direct(const Matrices& a, float* u_out, float* s_out, float* vt_out, uint32_t* info_out) {
    const uint32_t M = a.rows, N = a.cols, batch = a.batch, K = std::min(M, N);
    const bool vectors = u_out || vt_out;
    if (vectors && !(u_out && vt_out)) throw std::invalid_argument("[svd] bidiag_batch: U and V^T, or neither.");
    if (K == 0 || batch == 0) {
        if (info_out) std::fill(info_out, info_out + batch, 0u);
        return;
    }
    if (std::max(M, N) > kMaxDim)
        throw std::invalid_argument("[svd] bidiag_batch: rows and columns at most " + std::to_string(kMaxDim));
    AutoreleasePool pool;
    Cache& cache = shared_cache();
    id<MTLDevice> dev = cache.rt.device;
    // Each matrix as a tall one, column-major m x n: A, or A^T if wide
    const bool wide = M < N;
    const uint32_t m = std::max(M, N), n = K;
    const size_t per = (size_t)M * N;

    std::vector<float> amax(batch);
    std::vector<char>  finite(batch);
    scan(a, Part::all, amax.data(), finite.data());
    std::vector<uint32_t> todo;
    for (uint32_t b = 0; b < batch; ++b) {
        if (finite[b]) { todo.push_back(b); continue; }
        std::fill(s_out + (size_t)b * K, s_out + (size_t)(b + 1) * K, NAN);
        if (vectors) {
            std::fill(u_out + (size_t)b * M * K, u_out + (size_t)(b + 1) * M * K, NAN);
            std::fill(vt_out + (size_t)b * K * N, vt_out + (size_t)(b + 1) * K * N, NAN);
        }
        if (info_out) info_out[b] = 1u << 17;
    }
    if (todo.empty()) return;
    const size_t total = todo.size();
    // Singular values alone from kBandMinK, and with vectors from
    // kBandVectorsMinK (kBandVectorsSmallMinK for small batches): in two
    // stages, a band first
    const unsigned solve_threads = metal_linalg::detail::cpu_threads_beside_gpu();
    const char* band_env = std::getenv("SVD_BIDIAG_BATCH_BAND");
    const bool band_ok = !(band_env && std::string(band_env) == "0");
    const bool band = !vectors && n >= kBandMinK && band_ok;
    const bool bandv = vectors && band_ok &&
                       (n >= kBandVectorsMinK || (n >= kBandVectorsSmallMinK && total <= solve_threads));
    // A thread a row, up to 1024 (on an M5 Pro 1.1x at 8 x 1024^2 against 512;
    // eigh's panel, with less work a column, gains nothing from it)
    const uint32_t threads = std::clamp((m + 31) / 32 * 32, 64u, 1024u);
    // Chunks: four, to overlap the stages (eight for matrices of up to 128 x
    // 128, whose pipeline's first and last stages, alone on the GPU, are a
    // larger share: 1.1x at 1024 x 128^2; from 256 four were faster), each at
    // least 8 MB of matrices, at most 2^25 floats (128 MB a slot)
    const size_t min_chunk = std::max<size_t>(16, ((size_t)1 << 21) / per);
    const size_t max_chunks = per <= (size_t)128 * 128 ? 8 : 4;
    const size_t chunks = std::clamp<size_t>(std::min<size_t>(max_chunks, total / std::max<size_t>(1, min_chunk)), 1, total);
    size_t chunk = (total + chunks - 1) / chunks;
    chunk = std::min(chunk, std::max<size_t>(1, ((size_t)1 << 25) / per));
    // With vectors in two stages a matrix takes about 13 of its own in a slot
    // (52 MB at 1024 x 1024): at most 2^26 floats (256 MB) a slot
    if (bandv) {
        const size_t most = std::max<size_t>(1, ((size_t)1 << 26) / bandv_floats(m, n));
        if (chunk > most) chunk = (total + (total + most - 1) / most - 1) / ((total + most - 1) / most);   // balanced
    }
    const size_t count = (total + chunk - 1) / chunk;

    id<MTLBuffer> src = metal_linalg::detail::input_buffer(dev, a);
    id<MTLBuffer> index = [dev newBufferWithBytes:todo.data() length:total * sizeof(uint32_t)
                                          options:MTLResourceStorageModeShared];
    id<MTLBuffer> scale = [dev newBufferWithLength:total * sizeof(float) options:MTLResourceStorageModeShared];
    float* sc = static_cast<float*>(scale.contents);
    for (size_t j = 0; j < total; ++j) {
        int ex = 0;
        if (amax[todo[j]] > 0.0f) std::frexp(amax[todo[j]], &ex);
        sc[j] = std::ldexp(1.0f, -ex);
    }
    // The outputs, used in place where they can be
    auto output = [&](float* p, size_t floats) -> id<MTLBuffer> {
        if (reinterpret_cast<uintptr_t>(p) % (uintptr_t)getpagesize() == 0) {
            try { return metal_linalg::detail::wrap_host(dev, p, floats); } catch (...) {}
        }
        return [dev newBufferWithLength:floats * sizeof(float) options:MTLResourceStorageModeShared];
    };
    id<MTLBuffer> uo = vectors ? output(u_out, (size_t)batch * M * K) : nil;
    id<MTLBuffer> vo = vectors ? output(vt_out, (size_t)batch * K * N) : nil;

    uint32_t band_tail = 0;   // the first column the CPU reduces (the same for every chunk)
    Work* ws[2] = {&work(cache, m, n, (uint32_t)chunk, vectors && !bandv, 0, band || bandv, bandv),
                   count > 1 ? &work(cache, m, n, (uint32_t)chunk, vectors && !bandv, 1, band || bandv, bandv) : nullptr};
    auto check = [](id<MTLCommandBuffer> cb) {
        [cb waitUntilCompleted];
        if (cb.error)
            throw std::runtime_error(std::string("[svd] bidiag_batch: GPU error: ") + cb.error.localizedDescription.UTF8String);
    };
    std::vector<id<MTLCommandBuffer>> committed;
    auto commit = [&](id<MTLCommandBuffer> cb) {
        [cb commit];
        committed.push_back(cb);
        return cb;
    };
    auto cnt_of = [&](size_t k) { return (uint32_t)std::min(chunk, total - k * chunk); };
    // Each chunk's reduction waits for the previous chunk's: queued together
    // they shared the GPU, and the first, which the CPU waits on, took twice
    // as long (on an M5 Pro 1.15x for eigh at 16 x 1024^2 and 256 x 256^2,
    // 1.04-1.1x for the SVD)
    id<MTLEvent> order = [dev newEvent];
    auto reduce = [&](size_t k) {
        id<MTLCommandBuffer> cb = [cache.rt.queue commandBufferWithUnretainedReferences];
        if (k > 0) [cb encodeWaitForEvent:order value:k];
        if (band || bandv)
            band_tail = encode_band(cache, cb, *ws[k % 2], cnt_of(k), src, index, scale, (uint32_t)(k * chunk), M, N, bandv);
        else      encode_reduce(cache, cb, *ws[k % 2], cnt_of(k), src, index, scale, (uint32_t)(k * chunk), M, N, threads);
        [cb encodeSignalEvent:order value:k + 1];
        return commit(cb);
    };
    auto back = [&](size_t k) {
        Work& w = *ws[k % 2];
        const uint32_t cnt = cnt_of(k), c0 = (uint32_t)(k * chunk);
        id<MTLCommandBuffer> cb = [cache.rt.queue commandBufferWithUnretainedReferences];
        if (bandv) {
            encode_back_band(cache, cb, w, cnt, wide, uo, vo, index, c0);
            return commit(cb);
        }
        encode_back(cache, cb, w, cnt);
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (!wide) {   // U (m x n) and V^T (n x n) column-major, transposed out
            [enc setComputePipelineState:cache.store];
            for (int f = 0; f < 2; ++f) {
                const BsParams sp{f ? n : m, n, c0};
                [enc setBuffer:f ? w.VT : w.U offset:0 atIndex:0];
                [enc setBuffer:f ? vo : uo offset:0 atIndex:1];
                [enc setBuffer:index offset:0 atIndex:2];
                [enc setBytes:&sp length:sizeof sp atIndex:3];
                [enc dispatchThreadgroups:MTLSizeMake((sp.rows + 31) / 32, (n + 31) / 32, cnt)
                    threadsPerThreadgroup:MTLSizeMake(32, 8, 1)];
            }
        } else {       // U' (m x n) is V^T's layout, V'^T (n x n) U's
            [enc setComputePipelineState:cache.copy];
            for (int f = 0; f < 2; ++f) {
                const BcParams cp{f ? n * n : m * n, c0};
                [enc setBuffer:f ? w.VT : w.U offset:0 atIndex:0];
                [enc setBuffer:f ? uo : vo offset:0 atIndex:1];
                [enc setBuffer:index offset:0 atIndex:2];
                [enc setBytes:&cp length:sizeof cp atIndex:3];
                [enc dispatchThreads:MTLSizeMake(cp.per, cnt, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            }
        }
        [enc endEncoding];
        return commit(cb);
    };
    // Step 2 for chunk k on the CPU's cores but two: the singular values into
    // s_out, the bidiagonal's vectors into U's first n rows and V^T
    auto solve = [&](size_t k) {
        Work& w = *ws[k % 2];
        const uint32_t cnt = cnt_of(k);
        if (bandv) {   // the tail and Q1 P1 queued, then the chase and the rest
            solve_tail(w, cnt, band_tail, solve_threads);
            id<MTLCommandBuffer> cq = [cache.rt.queue commandBufferWithUnretainedReferences];
            encode_q1_p1(cache, cq, w, cnt, band_tail / kBand);
            commit(cq);
            solve_chase(w, cnt, solve_threads, [&](uint32_t j, const float* dj) {
                const size_t at = k * chunk + j;
                const float unscale = 1.0f / sc[at];
                float* sb = s_out + (size_t)todo[at] * K;
                for (uint32_t i = 0; i < n; ++i) sb[i] = dj[i] * unscale;   // descending
                if (info_out) info_out[todo[at]] = 1u | (1u << 16);
            }, [&](uint32_t j) { return todo[k * chunk + j]; });
            return;
        }
        const float* dg = static_cast<const float*>(w.d.contents);
        const float* eg = static_cast<const float*>(w.e.contents);
        float* U = vectors ? static_cast<float*>(w.U.contents) : nullptr;
        float* VT = vectors ? static_cast<float*>(w.VT.contents) : nullptr;
        metal_linalg::detail::lapack_batches(cnt, (size_t)m * n, solve_threads, [&](uint32_t b0, uint32_t b1) {
            std::vector<float> dj(n), ej(n), work(4 * (size_t)n + 16);
            // the band, with room for the chase's bulges: B(i, j) at
            // ab[j * ld + ku + i - j], ku = 2 kBand above the diagonal, kBand below
            const size_t ld = 3 * (size_t)kBand + 1, ku = 2 * (size_t)kBand;
            std::vector<float> ab(band ? ld * n : 0);
            for (uint32_t j = b0; j < b1; ++j) {
                const size_t at = k * chunk + j;
                if (band) {   // the last columns by LAPACK, then the band to bidiagonal
                    float* Aj = static_cast<float*>(w.A.contents) + (size_t)j * w.sa;
                    metal_linalg::detail::band_general_tail(Aj, m, n, w.lda, kBand, band_tail);
                    std::fill(ab.begin(), ab.end(), 0.0f);
                    for (uint32_t c = 0; c < n; ++c)
                        for (uint32_t r = c > kBand ? c - kBand : 0; r <= c; ++r)
                            ab[(size_t)c * ld + ku + r - c] = Aj[(size_t)c * w.lda + r];
                    metal_linalg::detail::band_to_bidiagonal(n, kBand, ab.data(), ld, ku, dj.data(), ej.data(), 1);
                } else {
                    std::copy(dg + (size_t)j * w.sv, dg + (size_t)j * w.sv + n, dj.begin());
                    std::copy(eg + (size_t)j * w.sv, eg + (size_t)j * w.sv + n, ej.begin());
                }
                L info = 0;
                if (vectors) {
                    float* Uj = U + (size_t)j * w.su;
                    for (uint32_t c = 0; c < n; ++c) std::fill(Uj + (size_t)c * m + n, Uj + (size_t)(c + 1) * m, 0.0f);
                    info = (L)metal_linalg::detail::bidiagonal_svd(n, dj.data(), ej.data(), Uj, m, VT + (size_t)j * w.svt,
                                                                   n, 1);
                } else {
                    L nn = n, zero = 0, one = 1;
                    float qd = 0.0f;
                    sbdsqr_("U", &nn, &zero, &zero, &zero, dj.data(), ej.data(), &qd, &one, &qd, &one, &qd, &one,
                            work.data(), &info);
                }
                if (info != 0)
                    throw std::runtime_error(std::string("[svd] bidiag_batch: LAPACK ") + (vectors ? "sbdsdc" : "sbdsqr") +
                                             " failed on matrix " + std::to_string(todo[at]) + ", info " +
                                             std::to_string((long long)info) + ".");
                const float unscale = 1.0f / sc[at];
                float* sb = s_out + (size_t)todo[at] * K;
                for (uint32_t i = 0; i < n; ++i) sb[i] = dj[i] * unscale;   // descending
                if (info_out) info_out[todo[at]] = 1u | (1u << 16);
            }
        });
    };

    std::vector<id<MTLCommandBuffer>> reduced(count, nil), backed;
    try {
        reduced[0] = reduce(0);
        if (count > 1) reduced[1] = reduce(1);
        for (size_t k = 0; k < count; ++k) {
            check(reduced[k]);
            if (vectors && k >= 2) check(backed[k - 2]);   // done reading the slot solve(k) writes
            solve(k);
            if (vectors) backed.push_back(back(k));
            if (k + 2 < count) reduced[k + 2] = reduce(k + 2);   // queued behind back(k), which frees the slot
        }
        for (id<MTLCommandBuffer> cb : backed) check(cb);
    } catch (...) {
        for (id<MTLCommandBuffer> cb : committed) [cb waitUntilCompleted];
        throw;
    }
    if (vectors) {   // outputs that could not be used in place
        const size_t pu = (size_t)M * K, pv = (size_t)K * N;
        const float* ou = static_cast<const float*>(uo.contents);
        const float* ov = static_cast<const float*>(vo.contents);
        if (ou != u_out || ov != vt_out)
            metal_linalg::detail::for_each_matrix((uint32_t)total, pu + pv, [&](uint32_t j) {
                const size_t b = todo[j];
                if (ou != u_out) std::memcpy(u_out + b * pu, ou + b * pu, pu * sizeof(float));
                if (ov != vt_out) std::memcpy(vt_out + b * pv, ov + b * pv, pv * sizeof(float));
            });
    }
}

// A matrix at least twice as tall as wide, reduced as the CPU path reduces it:
// R of this library's QR (R alone for the singular values alone), the K x K
// R decomposed directly, and U = Q U_R, one batched product on the GPU; one
// at least twice as wide, the same through its transpose, A^T = Q R, so that
// U = V_R and V^T = U_R^T Q^T. Bidiagonalized as it is, such a matrix's
// trailing block is read once a column, and it lost to the CPU
// (256 x 1024 x 128: 0.78x).
void qr_first(const Matrices& a, float* u_out, float* s_out, float* vt_out, uint32_t* info_out) {
    const uint32_t M = a.rows, N = a.cols, batch = a.batch, L = std::max(M, N), K = std::min(M, N);
    const bool vectors = u_out || vt_out, wide = M < N;
    if (vectors && !(u_out && vt_out)) throw std::invalid_argument("[svd] bidiag_batch: U and V^T, or neither.");
    const size_t pq = (size_t)L * K, pr = (size_t)K * K;
    std::unique_ptr<HostBuffer> at;   // a wide matrix's transpose, tall
    Matrices t = a;
    if (wide) {
        at = std::make_unique<HostBuffer>(batch * pq);
        metal_linalg::detail::transpose_out(a.data, at->data(), batch, M, N);
        t = Matrices{at->data(), batch, L, K};
    }
    HostBuffer r(batch * pr);
    std::unique_ptr<HostBuffer> q = vectors ? std::make_unique<HostBuffer>(batch * pq) : nullptr;
    core::qr(t, q ? q->data() : nullptr, r.data(), vectors ? core::QrMode::reduced : core::QrMode::r);
    at.reset();
    const Matrices rm{r.data(), batch, K, K};
    if (!vectors) {
        direct(rm, nullptr, s_out, nullptr, info_out);
        return;
    }
    HostBuffer ur(batch * pr);
    std::unique_ptr<HostBuffer> vtr = wide ? std::make_unique<HostBuffer>(batch * pr) : nullptr;
    direct(rm, ur.data(), s_out, wide ? vtr->data() : vt_out, info_out);
    if (wide) metal_linalg::detail::transpose_out(vtr->data(), u_out, batch, K, K);   // U = V_R

    // The product into U (tall: Q U_R, L x K) or V^T (wide: U_R^T Q^T, K x L)
    float* out = wide ? vt_out : u_out;
    AutoreleasePool pool;
    Cache& cache = shared_cache();
    id<MTLDevice> dev = cache.rt.device;
    using metal_linalg::detail::wrap_host;
    std::unique_ptr<HostBuffer> staged;
    id<MTLBuffer> ob = nil;
    if (reinterpret_cast<uintptr_t>(out) % (uintptr_t)getpagesize() == 0) {
        try { ob = wrap_host(dev, out, batch * pq); } catch (...) {}
    }
    if (!ob) {
        staged = std::make_unique<HostBuffer>(batch * pq);
        ob = wrap_host(dev, staged->data(), batch * pq);
    }
    MPSMatrix* Qm = mps(wrap_host(dev, q->data(), batch * pq), 0, L, K, K, batch, pq);
    MPSMatrix* Um = mps(wrap_host(dev, ur.data(), batch * pr), 0, K, K, K, batch, pr);
    id<MTLCommandBuffer> cb = [cache.rt.queue commandBuffer];
    if (!wide) gemm(dev, cb, Qm, false, Um, false, mps(ob, 0, L, K, K, batch, pq), L, K, K, 1, 0, batch);
    else       gemm(dev, cb, Um, true, Qm, true, mps(ob, 0, K, L, L, batch, pq), K, L, K, 1, 0, batch);
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.error)
        throw std::runtime_error(std::string("[svd] bidiag_batch: GPU error: ") + cb.error.localizedDescription.UTF8String);
    if (staged) metal_linalg::detail::copy_out(staged->data(), out, batch, pq);
}

// --- eigh with vectors for batches in two stages ----------------------------
//
// eigh_band_batch, the eigensolver's `tridiag_batch` from kEighBandMinN with
// eigenvectors: every matrix (loaded from its triangle, mirrored, scaled:
// sb_load) to a lower band of width kBand by blocks on the GPU, as
// band_reduce_symmetric reduces one: per block the panel below the diagonal
// block (bb_panel, which forms V T too), then both sides of the trailing
// matrix on the whole of it, X = A22 V T (a batched product), Y = X - V (T^T
// V^T X) / 2 (sb_small), A22 -= [V Y] [Y V]^T (one rank-2b product), each
// block's reflectors kept as bidiag_batch's are;
// on the CPU's cores, a matrix a core, the last columns by LAPACK's
// ssytrd_sy2sb and Q = Q_tail, then (the GPU forming Q1 Q meanwhile) the band
// chased to tridiagonal keeping the chase's reflectors and T = Z diag(w) Z^T
// by the divide and conquer; on the GPU, Q <- Q Q2 (bd_chase_apply, a
// dispatch a matrix) and V = Q Z. Chunks pipelined over two slots, as
// direct()'s.

struct EWork {
    uint32_t n = 0, capacity = 0, lda = 0, ldq = 0, blocks = 0;
    size_t sa = 0, sqy = 0, stq = 0, sq = 0, sz = 0, slv = 0, sb = 0, sza = 0;
    id<MTLBuffer> A, QY, QT, VT, VYV, GA, TA, Q, Z, LV, OV, ZA, ZA2;
    size_t svyv = 0;
};

size_t eband_floats(uint32_t n) {
    const size_t pmax = n >= 2 ? (n - 2) / 16 : 0, nblocks = (pmax + 1) * (pmax + 2) / 2;
    const size_t lda = (n + 7) / 8 * 8, ldq = (n + 31) / 32 * 32;
    return lda * n + (size_t)n * n * 3 + ldq * n + (n / kBand + 1) * 1024 + nblocks * metal_linalg::detail::kChaseBlockFloats +
           4 * (size_t)n * kBand + 2 * (size_t)n * kAgg * kBand + 2 * 128 * 128;
}

EWork& ework(Cache& c, uint32_t n, uint32_t capacity, int slot) {
    static EWork ws[2];
    EWork& w = ws[slot];
    if (w.A && w.n == n && w.capacity >= capacity) return w;
    id<MTLDevice> dev = c.rt.device;
    auto make = [&](size_t floats, MTLResourceOptions opt) {
        id<MTLBuffer> b = [dev newBufferWithLength:std::max<size_t>(floats, 4) * 4 options:opt];
        if (!b) throw std::runtime_error("[eigh] tridiag_batch: could not allocate " + std::to_string(floats * 4) + " bytes");
        return b;
    };
    const MTLResourceOptions shared = MTLResourceStorageModeShared, priv = MTLResourceStorageModePrivate;
    w = EWork{};
    w.n = n;
    w.capacity = capacity;
    w.lda = (n + 7) / 8 * 8;
    w.ldq = (n + 31) / 32 * 32;
    w.blocks = n / kBand + 1;
    const size_t pmax = n >= 2 ? (n - 2) / 16 : 0;
    w.sa = (size_t)w.lda * n;
    w.sqy = (size_t)n * n;
    w.stq = (size_t)w.blocks * 1024;
    w.sq = (size_t)w.ldq * n;
    w.sz = (size_t)n * n;
    w.slv = (pmax + 1) * (pmax + 2) / 2 * metal_linalg::detail::kChaseBlockFloats;
    w.sb = (size_t)n * kBand;
    w.sza = (size_t)n * kAgg * kBand;
    const size_t C = capacity + 1;
    w.A = make(C * w.sa, shared);
    w.QT = make(C * w.stq, priv);
    w.VT = make(C * w.sb, priv);
    w.svyv = (size_t)n * 3 * kBand;
    w.VYV = make(C * w.svyv, priv);
    w.QY = make(C * w.sqy, shared);
    std::memset(w.QY.contents, 0, C * w.sqy * 4);   // the zeros above each panel, never written
    w.GA = make(C * 128 * 128, priv);
    w.TA = make(C * 128 * 128, priv);
    w.Q = make(C * w.sq, shared);
    w.Z = make(C * w.sz, shared);
    w.LV = make(C * w.slv, shared);
    w.OV = make(C * w.sz, priv);
    w.ZA = make(C * w.sza, priv);
    w.ZA2 = make(C * w.sza, priv);
    return w;
}

// Step 1 for `cnt` matrices from index[c0]: loaded, then reduced by blocks
// while three blocks' columns remain. Returns the first column left to the
// CPU.
uint32_t encode_sym_band(Cache& c, id<MTLCommandBuffer> cb, EWork& w, uint32_t cnt, id<MTLBuffer> src,
                         id<MTLBuffer> index, id<MTLBuffer> scale, uint32_t c0, bool lower) {
    id<MTLDevice> dev = c.rt.device;
    const uint32_t n = w.n, lda = w.lda, b = kBand;
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    const SlParams lp{n, lda, (uint32_t)w.sa, lower ? 1u : 0u, c0};
    [enc setComputePipelineState:c.sym_load];
    [enc setBuffer:src offset:0 atIndex:0];
    [enc setBuffer:w.A offset:0 atIndex:1];
    [enc setBuffer:index offset:0 atIndex:2];
    [enc setBuffer:scale offset:0 atIndex:3];
    [enc setBytes:&lp length:sizeof lp atIndex:4];
    [enc dispatchThreads:MTLSizeMake(n, n, cnt) threadsPerThreadgroup:MTLSizeMake(32, 8, 1)];
    uint32_t k = 0;
    for (; k + 3 * b <= n; k += b) {
        const uint32_t n1 = n - k - b, j = k / b;
        // The panel A(k+b:, k:k+b): R in place (the band), V into [V Y V]
        // (ld 3b) twice and into QY, T into QT, V T into VT
        const BqParams q{n1, b, 1, lda, (uint32_t)w.sa, (uint32_t)w.svyv, (uint32_t)w.stq, 3 * b, 32, 2 * b,
                         (uint32_t)w.sqy, n, 2u, 0, 0, 0, 0, 0, 0, (uint32_t)w.sb};
        [enc setComputePipelineState:c.band_panel];
        [enc setBuffer:w.A offset:((size_t)k * lda + k + b) * 4 atIndex:0];
        [enc setBuffer:w.VYV offset:0 atIndex:1];
        [enc setBuffer:w.QT offset:(size_t)j * 1024 * 4 atIndex:2];
        [enc setBytes:&q length:sizeof q atIndex:3];
        [enc setBuffer:w.QY offset:((size_t)(k + b) * n + k) * 4 atIndex:4];
        for (NSUInteger i = 5; i <= 8; ++i) [enc setBuffer:w.VYV offset:0 atIndex:i];   // (merge's, unused)
        [enc setBuffer:w.VT offset:0 atIndex:9];                                          // V T
        [enc dispatchThreadgroups:MTLSizeMake(cnt, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(std::max<uint32_t>(32, (n1 + 31) / 32 * 32), 1, 1)];
        [enc endEncoding];
        // A22 <- H^T A22 H on the whole symmetric A22 (its row-major view is
        // itself): X = A22 (V T) into Y's place; Y = X - V (T^T (V^T X)) / 2
        // (sb_small; as MPS products, four of them, it took 1.1-1.3x as long);
        // A22 -= [V Y] [Y V]^T, one product
        MPSMatrix* A22 = mps(w.A, (size_t)(k + b) * lda + k + b, n1, n1, lda, cnt, w.sa);
        gemm(dev, cb, A22, false, mps(w.VT, 0, n1, b, b, cnt, w.sb), false, mps(w.VYV, b, n1, b, 3 * b, cnt, w.svyv),
             n1, b, n1, 1, 0, cnt);
        const SsParams sp{n1, 3 * b, (uint32_t)w.svyv, (uint32_t)w.stq, 32};
        enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:c.sym_small];
        [enc setBuffer:w.VYV offset:0 atIndex:0];
        [enc setBuffer:w.QT offset:(size_t)j * 1024 * 4 atIndex:1];
        [enc setBytes:&sp length:sizeof sp atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake(cnt, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(std::max<uint32_t>(32, (n1 + 31) / 32 * 32), 1, 1)];
        [enc endEncoding];
        gemm(dev, cb, mps(w.VYV, 0, n1, 2 * b, 3 * b, cnt, w.svyv), false, mps(w.VYV, b, n1, 2 * b, 3 * b, cnt, w.svyv),
             true, A22, n1, n1, 2 * b, -1, 1, cnt);
        enc = [cb computeCommandEncoder];
    }
    [enc endEncoding];
    return k;
}

// Step 2's first part for the slot's `cnt` matrices: the trailing block from
// `tail` on by LAPACK's ssytrd_sy2sb (its band copied into A's lower band),
// and Q = I with its reflectors applied, last panel first.
void solve_sym_tail(EWork& w, uint32_t cnt, uint32_t tail, unsigned threads) {
    const uint32_t n = w.n, b = kBand, nt = n - tail;
    float* A = static_cast<float*>(w.A.contents);
    float* Q = static_cast<float*>(w.Q.contents);
    metal_linalg::detail::lapack_batches(cnt, (size_t)n * n, threads, [&](uint32_t b0, uint32_t b1) {
        std::vector<float> tau(std::max(nt, 1u)), work((size_t)std::max(nt, 1u) * 64 + 64);
        const L KD = nt > 1 ? std::min<L>(b, nt - 1) : 0, LDAB = KD + 1;
        std::vector<float> abt((size_t)LDAB * std::max(nt, 1u));
        for (uint32_t j = b0; j < b1; ++j) {
            float* Aj = A + (size_t)j * w.sa;
            float* Qj = Q + (size_t)j * w.sq;
            for (uint32_t c = 0; c < n; ++c) {
                std::fill(Qj + (size_t)c * w.ldq, Qj + (size_t)(c + 1) * w.ldq, 0.0f);
                Qj[(size_t)c * w.ldq + c] = 1.0f;
            }
            if (nt < 2) continue;
            L N = nt, LDA = w.lda, lw = -1, info = 0;
            float q = 0.0f;
            float* At = Aj + (size_t)tail * w.lda + tail;
            ssytrd_sy2sb_("L", &N, &KD, At, &LDA, abt.data(), &LDAB, tau.data(), &q, &lw, &info);
            if ((size_t)q > work.size()) work.resize((size_t)q);
            lw = (L)work.size();
            ssytrd_sy2sb_("L", &N, &KD, At, &LDA, abt.data(), &LDAB, tau.data(), work.data(), &lw, &info);
            if (info != 0) throw std::runtime_error("[eigh] tridiag_batch: LAPACK ssytrd_sy2sb failed, info " +
                                                    std::to_string((long long)info));
            // Q(tail:, tail:) <- H_tail, ssytrd_sy2sb's panels, last first
            // (LAPACK's I = 1, 1 + KD, ... while I <= N - KD, 1-based)
            std::vector<uint32_t> panels;
            for (uint32_t i = 0; i + (uint32_t)KD < nt; i += (uint32_t)KD) panels.push_back(i);
            L LDQ = w.ldq;
            for (auto it = panels.rbegin(); it != panels.rend(); ++it) {
                const uint32_t i = *it;
                L Mq = nt - i - KD, Nq = nt, Kq = std::min<L>(Mq, KD);
                sormqr_("L", "N", &Mq, &Nq, &Kq, At + (size_t)i * w.lda + i + KD, &LDA, tau.data() + i,
                        Qj + (size_t)tail * w.ldq + tail + i + KD, &LDQ, work.data(), &lw, &info);
            }
            // The tail's band into A's lower band
            for (uint32_t c = 0; c < nt; ++c)
                for (uint32_t r = c; r < nt && r <= c + (uint32_t)KD; ++r)
                    At[(size_t)c * w.lda + r] = abt[(size_t)c * LDAB + r - c];
        }
    });
}

// Step 2's second part: the band chased to tridiagonal, keeping the chase's
// reflectors (bd_chase_apply's blocks), then T = Z diag(w) Z^T into the
// slot's Z. done(j, d) takes matrix j's eigenvalues; which(j) names it.
template <class Done, class Which>
void solve_sym_chase(EWork& w, uint32_t cnt, unsigned threads, const Done& done, const Which& which) {
    const uint32_t n = w.n, b = kBand;
    const size_t pmax = n >= 2 ? (n - 2) / 16 : 0, nblocks = (pmax + 1) * (pmax + 2) / 2;
    const long groups = (long)pmax + 1;
    const unsigned per = std::max(1u, threads / std::max(1u, std::min(cnt, threads)));
    const float* A = static_cast<const float*>(w.A.contents);
    float* Z = static_cast<float*>(w.Z.contents);
    float* LV = static_cast<float*>(w.LV.contents);
    metal_linalg::detail::lapack_batches(cnt, (size_t)n * n, threads, [&](uint32_t b0, uint32_t b1) {
        std::vector<float> dj(n), ej(n), ltau(nblocks * 16);
        const size_t ld = 2 * (size_t)b + 1;
        std::vector<float> ab(ld * n);
        for (uint32_t j = b0; j < b1; ++j) {
            const float* Aj = A + (size_t)j * w.sa;
            std::fill(ab.begin(), ab.end(), 0.0f);
            for (uint32_t c = 0; c < n; ++c)
                for (uint32_t r = c; r < n && r <= c + b; ++r) ab[(size_t)c * ld + r - c] = Aj[(size_t)c * w.lda + r];
            metal_linalg::detail::ChaseReflectors rec;
            rec.L = LV + (size_t)j * w.slv;
            rec.Ltau = ltau.data();
            rec.pmax = pmax;
            metal_linalg::detail::band_to_tridiagonal(n, b, ab.data(), ld, dj.data(), ej.data(), per, &rec);
            const long chunks = std::min<long>(groups, 4 * (long)per);
            auto build = [&](size_t t) {
                metal_linalg::detail::chase_build_blocks(rec.L, rec.Ltau, n, groups * (long)t / chunks,
                                                         groups * ((long)t + 1) / chunks);
            };
            if (per > 1) metal_linalg::detail::parallel_for((size_t)chunks, build);
            else for (long t = 0; t < chunks; ++t) build((size_t)t);
            const long info = metal_linalg::detail::tridiagonal_eigensystem(n, dj.data(), ej.data(), Z + (size_t)j * w.sz,
                                                                            n, per);
            if (info != 0)
                throw std::runtime_error("[eigh] tridiag_batch: the divide and conquer failed on matrix " +
                                         std::to_string(which(j)) + ", info " + std::to_string(info) + ".");
            done(j, dj.data());
        }
    });
}

// Step 3: Q <- Q1 Q (once solve_sym_tail is done), as encode_q1_p1's left
// side with the reflectors a block lower; then (once solve_sym_chase is)
// Q <- Q Q2 and V = Q Z, row-major, copied to matrix index[c0 + j].
void encode_sym_q1(Cache& c, id<MTLCommandBuffer> cb, EWork& w, uint32_t cnt, uint32_t blocks) {
    id<MTLDevice> dev = c.rt.device;
    const uint32_t n = w.n, b = kBand, aggs = (blocks + kAgg - 1) / kAgg;
    for (int a = (int)aggs - 1; a >= 0; --a) {
        const uint32_t j0 = (uint32_t)a * kAgg, np = std::min(kAgg, blocks - j0), wa = np * b;
        const uint32_t r0 = j0 * b + b, len = n - r0;
        MPSMatrix* Y = mps(w.QY, (size_t)r0 * n + j0 * b, len, wa, n, cnt, w.sqy);
        gemm(dev, cb, Y, true, Y, false, mps(w.GA, 0, wa, wa, 128, cnt, 128 * 128), wa, wa, len, 1, 0, cnt);
        const MergeParams mp{np, b, 128 * 128, (uint32_t)w.stq, 128 * 128};
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:c.merge_t];
        [enc setBuffer:w.GA offset:0 atIndex:0];
        [enc setBuffer:w.QT offset:(size_t)j0 * 1024 * 4 atIndex:1];
        [enc setBuffer:w.TA offset:0 atIndex:2];
        [enc setBytes:&mp length:sizeof mp atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, cnt) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
        [enc endEncoding];
        MPSMatrix* Zm = mps(w.Q, (size_t)r0 * w.ldq + r0, len, len, w.ldq, cnt, w.sq);
        MPSMatrix* Ta = mps(w.TA, 0, wa, wa, 128, cnt, 128 * 128);
        MPSMatrix* W = mps(w.ZA, 0, len, wa, kAgg * b, cnt, w.sza);
        MPSMatrix* W2 = mps(w.ZA2, 0, len, wa, kAgg * b, cnt, w.sza);
        gemm(dev, cb, Zm, false, Y, false, W, len, wa, len, 1, 0, cnt);
        gemm(dev, cb, W, false, Ta, true, W2, len, wa, wa, 1, 0, cnt);
        gemm(dev, cb, W2, false, Y, true, Zm, len, len, wa, -1, 1, cnt);
    }
}

void encode_sym_back(Cache& c, id<MTLCommandBuffer> cb, EWork& w, uint32_t cnt, id<MTLBuffer> out,
                     id<MTLBuffer> index, uint32_t c0) {
    id<MTLDevice> dev = c.rt.device;
    const uint32_t n = w.n;
    if (n >= 3) {
        const uint32_t pmax = (n - 2) / 16, groups = pmax + 1, ld = w.ldq;
        const bool widek = ld / 32 >= 64;
        const uint32_t C = widek ? 32 : 16, K = widek ? 4 : 8;
        const ChaseParams q{n, ld, 1, pmax, 0, (groups + K - 1) / K};
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
        [enc setComputePipelineState:widek ? c.chase_wide : c.chase_narrow];
        [enc setBytes:&q length:sizeof q atIndex:2];
        for (uint32_t j = 0; j < cnt; ++j) {
            [enc setBuffer:w.Q offset:(size_t)j * w.sq * 4 atIndex:0];
            [enc setBuffer:w.LV offset:(size_t)j * w.slv * 4 atIndex:1];
            [enc dispatchThreadgroups:MTLSizeMake(ld / C, 1, 1) threadsPerThreadgroup:MTLSizeMake(32 * K, 1, 1)];
        }
        [enc endEncoding];
    }
    // V = Q Z, row-major: on the row-major views Q^T and Z^T
    gemm(dev, cb, mps(w.Q, 0, n, n, w.ldq, cnt, w.sq), true, mps(w.Z, 0, n, n, n, cnt, w.sz), true,
         mps(w.OV, 0, n, n, n, cnt, w.sz), n, n, n, 1, 0, cnt);
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    const BcParams cp{n * n, c0};
    [enc setComputePipelineState:c.copy];
    [enc setBuffer:w.OV offset:0 atIndex:0];
    [enc setBuffer:out offset:0 atIndex:1];
    [enc setBuffer:index offset:0 atIndex:2];
    [enc setBytes:&cp length:sizeof cp atIndex:3];
    [enc dispatchThreads:MTLSizeMake(cp.per, cnt, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [enc endEncoding];
}

} // namespace

namespace detail {

void eigh_band_batch(const Matrices& a, bool lower, float* w_out, float* v_out, uint32_t* info_out) {
    const uint32_t n = a.cols, batch = a.batch;
    if (a.rows != n) throw std::invalid_argument("[eigh] Input matrices must be square.");
    if (!v_out) throw std::invalid_argument("[eigh] tridiag_batch in two stages: eigenvectors only.");
    if (n == 0 || batch == 0) {
        if (info_out) std::fill(info_out, info_out + batch, 0u);
        return;
    }
    if (n > kMaxDim) throw std::invalid_argument("[eigh] tridiag_batch: N above " + std::to_string(kMaxDim));
    AutoreleasePool pool;
    Cache& cache = shared_cache();
    id<MTLDevice> dev = cache.rt.device;
    const size_t per = (size_t)n * n;
    std::vector<float> amax(batch);
    std::vector<char>  finite(batch);
    scan(a, lower ? Part::lower : Part::upper, amax.data(), finite.data());
    std::vector<uint32_t> todo;
    for (uint32_t b = 0; b < batch; ++b) {
        if (finite[b]) { todo.push_back(b); continue; }
        std::fill(w_out + (size_t)b * n, w_out + (size_t)(b + 1) * n, NAN);
        std::fill(v_out + b * per, v_out + (b + 1) * per, NAN);
        if (info_out) info_out[b] = 1u << 17;
    }
    if (todo.empty()) return;
    const size_t total = todo.size();
    // Chunks as direct()'s, at most 2^26 floats (256 MB) a slot
    const size_t min_chunk = std::max<size_t>(16, ((size_t)1 << 21) / per);
    const size_t chunks = std::clamp<size_t>(std::min<size_t>(4, total / std::max<size_t>(1, min_chunk)), 1, total);
    size_t chunk = (total + chunks - 1) / chunks;
    const size_t most = std::max<size_t>(1, ((size_t)1 << 26) / eband_floats(n));
    if (chunk > most) chunk = (total + (total + most - 1) / most - 1) / ((total + most - 1) / most);   // balanced
    const size_t count = (total + chunk - 1) / chunk;

    id<MTLBuffer> src = metal_linalg::detail::input_buffer(dev, a);
    id<MTLBuffer> index = [dev newBufferWithBytes:todo.data() length:total * sizeof(uint32_t)
                                          options:MTLResourceStorageModeShared];
    id<MTLBuffer> scale = [dev newBufferWithLength:total * sizeof(float) options:MTLResourceStorageModeShared];
    float* sc = static_cast<float*>(scale.contents);
    for (size_t j = 0; j < total; ++j) {
        int ex = 0;
        if (amax[todo[j]] > 0.0f) std::frexp(amax[todo[j]], &ex);
        sc[j] = std::ldexp(1.0f, -ex);
    }
    id<MTLBuffer> out = nil;
    if (reinterpret_cast<uintptr_t>(v_out) % (uintptr_t)getpagesize() == 0) {
        try { out = metal_linalg::detail::wrap_host(dev, v_out, (size_t)batch * per); } catch (...) { out = nil; }
    }
    if (!out) out = [dev newBufferWithLength:(size_t)batch * per * sizeof(float) options:MTLResourceStorageModeShared];

    EWork* ws[2] = {&ework(cache, n, (uint32_t)chunk, 0), count > 1 ? &ework(cache, n, (uint32_t)chunk, 1) : nullptr};
    auto check = [](id<MTLCommandBuffer> cb) {
        [cb waitUntilCompleted];
        if (cb.error)
            throw std::runtime_error(std::string("[eigh] tridiag_batch: GPU error: ") + cb.error.localizedDescription.UTF8String);
    };
    std::vector<id<MTLCommandBuffer>> committed;
    auto commit = [&](id<MTLCommandBuffer> cb) {
        [cb commit];
        committed.push_back(cb);
        return cb;
    };
    auto cnt_of = [&](size_t k) { return (uint32_t)std::min(chunk, total - k * chunk); };
    uint32_t tail = 0;
    // Each chunk's reduction waits for the previous chunk's: queued together
    // they shared the GPU, and the first, which the CPU waits on, took twice
    // as long (on an M5 Pro 1.15x for eigh at 16 x 1024^2 and 256 x 256^2,
    // 1.04-1.1x for the SVD)
    id<MTLEvent> order = [dev newEvent];
    auto reduce = [&](size_t k) {
        id<MTLCommandBuffer> cb = [cache.rt.queue commandBufferWithUnretainedReferences];
        if (k > 0) [cb encodeWaitForEvent:order value:k];
        tail = encode_sym_band(cache, cb, *ws[k % 2], cnt_of(k), src, index, scale, (uint32_t)(k * chunk), lower);
        [cb encodeSignalEvent:order value:k + 1];
        return commit(cb);
    };
    const unsigned solve_threads = metal_linalg::detail::cpu_threads_beside_gpu();
    std::vector<id<MTLCommandBuffer>> reduced(count, nil), backed;
    try {
        reduced[0] = reduce(0);
        if (count > 1) reduced[1] = reduce(1);
        for (size_t k = 0; k < count; ++k) {
            EWork& w = *ws[k % 2];
            const uint32_t cnt = cnt_of(k), c0 = (uint32_t)(k * chunk);
            check(reduced[k]);
            if (k >= 2) check(backed[k - 2]);   // done reading the slot this chunk writes
            solve_sym_tail(w, cnt, tail, solve_threads);
            id<MTLCommandBuffer> cq = [cache.rt.queue commandBufferWithUnretainedReferences];
            encode_sym_q1(cache, cq, w, cnt, tail / kBand);
            commit(cq);
            solve_sym_chase(w, cnt, solve_threads, [&](uint32_t j, const float* dj) {
                const size_t at = k * chunk + j;
                const float unscale = 1.0f / sc[at];
                float* wb = w_out + (size_t)todo[at] * n;
                for (uint32_t i = 0; i < n; ++i) wb[i] = dj[i] * unscale;   // ascending
                if (info_out) info_out[todo[at]] = 1u | (1u << 16);
            }, [&](uint32_t j) { return todo[k * chunk + j]; });
            id<MTLCommandBuffer> cb = [cache.rt.queue commandBufferWithUnretainedReferences];
            encode_sym_back(cache, cb, w, cnt, out, index, c0);
            backed.push_back(commit(cb));
            if (k + 2 < count) reduced[k + 2] = reduce(k + 2);
        }
        for (id<MTLCommandBuffer> cb : backed) check(cb);
    } catch (...) {
        for (id<MTLCommandBuffer> cb : committed) [cb waitUntilCompleted];
        throw;
    }
    if (out.contents != (void*)v_out) {
        const float* o = static_cast<const float*>(out.contents);
        metal_linalg::detail::for_each_matrix((uint32_t)total, per, [&](uint32_t j) {
            std::memcpy(v_out + (size_t)todo[j] * per, o + (size_t)todo[j] * per, per * sizeof(float));
        });
    }
}

} // namespace detail

namespace core::detail {

void svd_bidiag_batch(const Matrices& a, float* u_out, float* s_out, float* vt_out, uint32_t* info_out) {
    const uint32_t L = std::max(a.rows, a.cols), K = std::min(a.rows, a.cols);
    const char* env = std::getenv("SVD_BIDIAG_BATCH_QR");
    if (L >= 2 * K && K > 0 && a.batch > 0 && !(env && std::string(env) == "0")) qr_first(a, u_out, s_out, vt_out, info_out);
    else direct(a, u_out, s_out, vt_out, info_out);
}

} // namespace core::detail
} // namespace metal_linalg
