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
struct BqParams { uint32_t p, nb, rs, cs, sa, sv, st, ldv, ldt; };
struct MergeParams { uint32_t np, b, sg, stb, sta; };
struct ChaseParams { uint32_t n, rs, cs, pmax, pass0, pass1; };

// Singular values alone from this k: every matrix to an upper band of width
// kBand by blocks on the GPU (bb_panel and batched products), the bands to
// bidiagonal on the CPU's cores (SVD_BIDIAG_BATCH_BAND=0: bidiagonalized
// directly throughout). On an M5 Pro 1.05-1.1x the direct reduction at
// 160-256, 1.5x at 512, 2-2.3x at 1024; 0.88x at 128, where the CPU's chase
// of the band is the longer stage.
constexpr uint32_t kBand = 16;
constexpr uint32_t kBandMinK = 160;

// With vectors from this k: two stages too, both stages' reflectors kept and
// applied (see direct()), the back-transformation's blocks aggregated kAgg
// panels at a time. On an M5 Pro against the direct reduction: 1.07x at
// 16 x 384^2, 1.08-1.27x at 512, 1.6-2.1x at 768, 2.7-3.4x at 1024; 0.88-0.9x
// at 256.
constexpr uint32_t kBandVectorsMinK = 384;
constexpr uint32_t kAgg = 8;

using L = __LAPACK_int;

struct Cache {
    MetalRuntime& rt = MetalRuntime::shared(METAL_LINALG_SHADER(Svd_Bidiag), "svd_bidiag");
    // bd_panel's instances by rows: a lane's rows of a column in registers, up
    // to 128, 256, 512 and 1024 rows
    id<MTLComputePipelineState> panels[4] = {nil, nil, nil, nil};
    id<MTLComputePipelineState> load = nil, store = nil, copy = nil, make_v = nil, make_t = nil, band_panel = nil;
    id<MTLComputePipelineState> merge_t = nil, chase_wide = nil, chase_narrow = nil;

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
    // the reduction to bands: the column and row panels' V (m x kBand, n x
    // kBand) and T, and two m x kBand products
    uint32_t sb1 = 0, sb2 = 0, sbz = 0;
    id<MTLBuffer> BV1, BV2, BT1, BT2, BZ, BZ2;
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
    w.lda = (m + 7) / 8 * 8;
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
        w.sb1 = m * kBand;
        w.sb2 = n * kBand;
        w.sbz = m * kBand;
        w.BV1 = make(C * w.sb1, priv);
        w.BV2 = make(C * w.sb2, priv);
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

// Singular values alone, step 1 as two stages' first: `cnt` matrices loaded
// (as encode_reduce's), then reduced to an upper band of width kBand by blocks
// while three blocks' columns remain, as band_reduce.mm reduces one matrix:
// per block the column panel's QR (bb_panel) and Q^T applied to the columns
// right of it, then the row panel's LQ (bb_panel on its transpose) and its Q
// applied to the rows below, the products batched (on the row-major views of
// the column-major matrices). Returns the first column left to the CPU. With
// `keep` (vectors), each block's reflectors into QY, PY and their T's into QT,
// PT, where the back-transformation reads them, instead of scratch.
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
    auto panel = [&](size_t off, uint32_t p, uint32_t rs, uint32_t cs, id<MTLBuffer> V, size_t voff, size_t sv,
                     uint32_t ldv, id<MTLBuffer> T, size_t toff, size_t st, uint32_t ldt) {
        const BqParams q{p, b, rs, cs, w.sa, (uint32_t)sv, (uint32_t)st, ldv, ldt};
        [enc setComputePipelineState:c.band_panel];
        [enc setBuffer:w.A offset:off * 4 atIndex:0];
        [enc setBuffer:V offset:voff * 4 atIndex:1];
        [enc setBuffer:T offset:toff * 4 atIndex:2];
        [enc setBytes:&q length:sizeof q atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(cnt, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(std::max<uint32_t>(32, (p + 31) / 32 * 32), 1, 1)];
    };
    uint32_t k = 0;
    for (; k + 3 * b <= n; k += b) {
        const uint32_t p = m - k, n2 = n - k - b, m2 = m - k - b, j = k / b;
        // Where the panels' V and T go: scratch, or (keep) QY/PY and QT/PT
        id<MTLBuffer> v1b = keep ? w.QY : w.BV1, t1b = keep ? w.QT : w.BT1;
        id<MTLBuffer> v2b = keep ? w.PY : w.BV2, t2b = keep ? w.PT : w.BT2;
        const size_t v1o = keep ? (size_t)k * n + k : 0, v2o = keep ? (size_t)(k + b) * n + k : 0;
        const size_t t1o = keep ? (size_t)j * 1024 : 0, sv1 = keep ? w.sqy : w.sb1, sv2 = keep ? w.spy : w.sb2;
        const size_t st = keep ? w.stq : (size_t)b * b;
        const uint32_t ldv = keep ? n : b, ldt = keep ? 32 : b;
        // The column panel A(k:, k:k+b): R in place, V1, T1
        panel((size_t)k * lda + k, p, 1, lda, v1b, v1o, sv1, ldv, t1b, t1o, st, ldt);
        [enc endEncoding];
        // A(k:, k+b:) <- Q1^T A(k:, k+b:): on its transpose S (n2 x p),
        // S -= ((S V1) T1) V1^T
        MPSMatrix* S = mps(w.A, (size_t)(k + b) * lda + k, n2, p, lda, cnt, w.sa);
        MPSMatrix* V1 = mps(v1b, v1o, p, b, ldv, cnt, sv1);
        MPSMatrix* T1 = mps(t1b, t1o, b, b, ldt, cnt, st);
        MPSMatrix* Z = mps(w.BZ, 0, n2, b, b, cnt, w.sbz);
        MPSMatrix* Z2 = mps(w.BZ2, 0, n2, b, b, cnt, w.sbz);
        gemm(dev, cb, S, false, V1, false, Z, n2, b, p, 1, 0, cnt);
        gemm(dev, cb, Z, false, T1, false, Z2, n2, b, b, 1, 0, cnt);
        gemm(dev, cb, Z2, false, V1, true, S, n2, p, b, -1, 1, cnt);
        // The row panel A(k:k+b, k+b:) through its transpose (n2 x b): L in
        // place, V2, T2
        enc = [cb computeCommandEncoder];
        panel((size_t)(k + b) * lda + k, n2, lda, 1, v2b, v2o, sv2, ldv, t2b, t1o, st, ldt);
        [enc endEncoding];
        // A(k+b:, k+b:) <- A Q2, Q2 = I - V2 T2 V2^T: on its transpose S2
        // (n2 x m2), S2 -= V2 (T2^T (V2^T S2))
        MPSMatrix* S2 = mps(w.A, (size_t)(k + b) * lda + k + b, n2, m2, lda, cnt, w.sa);
        MPSMatrix* V2 = mps(v2b, v2o, n2, b, ldv, cnt, sv2);
        MPSMatrix* T2 = mps(t2b, t1o, b, b, ldt, cnt, st);
        MPSMatrix* Y = mps(w.BZ, 0, b, m2, m, cnt, w.sbz);
        MPSMatrix* Y2 = mps(w.BZ2, 0, b, m2, m, cnt, w.sbz);
        gemm(dev, cb, V2, true, S2, false, Y, b, m2, n2, 1, 0, cnt);
        gemm(dev, cb, T2, true, Y, false, Y2, b, m2, b, 1, 0, cnt);
        gemm(dev, cb, V2, false, Y2, false, S2, n2, m2, b, -1, 1, cnt);
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
            [enc dispatchThreadgroups:MTLSizeMake(cnt, 1, 1) threadsPerThreadgroup:MTLSizeMake(bb, 1, 1)];
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
// blocks were kept (encode_band) and whose CPU step left Q = Q_tail [I; 0],
// P = P_tail, U_B, V_B^T and the chase's blocks in the slot: Q <- Q1 Q and
// P <- P1 P, kAgg panels an aggregate, last first (I - Y Ta Y^T, Ta merged by
// bd_merge_t from the Gram matrix Y^T Y and the panels' T's), on the
// row-major views Q^T and P^T: Z <- Z - ((Z Y) Ta^T) Y^T; then Q <- Q Q2 and
// P <- P P2 (bd_chase_apply, a dispatch a matrix, the matrices at once); then
// the outputs: U = Q U_B and V^T = V_B^T P^T, or for a wide matrix's
// transpose U = P V_B and V^T = U_B^T Q^T, row-major, copied to matrix
// index[c0 + j].
void encode_back_band(Cache& c, id<MTLCommandBuffer> cb, Work& w, uint32_t cnt, uint32_t blocks, bool wide,
                      id<MTLBuffer> uo, id<MTLBuffer> vo, id<MTLBuffer> index, uint32_t c0) {
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
// fewer matrices than threads): the last columns from `tail` on by LAPACK,
// keeping their reflectors; the band chased to bidiagonal, the chase's
// reflectors into the slot's blocks (bd_chase_apply's, Y built beside V);
// Q = Q_tail [I; 0] and P = P_tail, column-major; B = U_B S V_B^T by the
// divide and conquer. done(j, d) takes matrix j's singular values; which(j)
// names it in errors.
template <class Done, class Which>
void solve_band_vectors(Work& w, uint32_t cnt, uint32_t tail, unsigned threads, const Done& done, const Which& which) {
    const uint32_t m = w.m, n = w.n, b = kBand;
    const size_t pmax = n >= 2 ? (n - 2) / 16 : 0, nblocks = (pmax + 1) * (pmax + 2) / 2;
    const long groups = (long)pmax + 1;
    const unsigned per = std::max(1u, threads / std::max(1u, std::min(cnt, threads)));
    float* A = static_cast<float*>(w.A.contents);
    float* Q = static_cast<float*>(w.Q.contents);
    float* P = static_cast<float*>(w.P.contents);
    float* UB = static_cast<float*>(w.UB.contents);
    float* VTB = static_cast<float*>(w.VTB.contents);
    float* LV = static_cast<float*>(w.LV.contents);
    float* RV = static_cast<float*>(w.RV.contents);
    metal_linalg::detail::lapack_batches(cnt, (size_t)m * n, threads, [&](uint32_t b0, uint32_t b1) {
        std::vector<float> dj(n), ej(n), ltau(nblocks * 16), rtau(nblocks * 16), work((size_t)std::max(m, n) * 64 + 64);
        const size_t ld = 3 * (size_t)b + 1, ku = 2 * (size_t)b;
        std::vector<float> ab(ld * n);
        for (uint32_t j = b0; j < b1; ++j) {
            float* Aj = A + (size_t)j * w.sa;
            metal_linalg::detail::BandKeep keep;
            metal_linalg::detail::band_general_tail(Aj, m, n, w.lda, b, tail, &keep);
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
            metal_linalg::detail::chase_build_blocks(rec.L, rec.Ltau, n, 0, groups);
            metal_linalg::detail::chase_build_blocks(rec.R, rec.Rtau, n, 0, groups);
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
    if (vectors && n >= kBandVectorsMinK)
        chunk = std::min(chunk, std::max<size_t>(1, ((size_t)1 << 26) / bandv_floats(m, n)));
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

    // Singular values alone from kBandMinK, and with vectors from
    // kBandVectorsMinK: in two stages, a band first
    const char* band_env = std::getenv("SVD_BIDIAG_BATCH_BAND");
    const bool band_ok = !(band_env && std::string(band_env) == "0");
    const bool band = !vectors && n >= kBandMinK && band_ok;
    const bool bandv = vectors && n >= kBandVectorsMinK && band_ok;
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
    auto reduce = [&](size_t k) {
        id<MTLCommandBuffer> cb = [cache.rt.queue commandBufferWithUnretainedReferences];
        if (band || bandv)
            band_tail = encode_band(cache, cb, *ws[k % 2], cnt_of(k), src, index, scale, (uint32_t)(k * chunk), M, N, bandv);
        else      encode_reduce(cache, cb, *ws[k % 2], cnt_of(k), src, index, scale, (uint32_t)(k * chunk), M, N, threads);
        return commit(cb);
    };
    auto back = [&](size_t k) {
        Work& w = *ws[k % 2];
        const uint32_t cnt = cnt_of(k), c0 = (uint32_t)(k * chunk);
        id<MTLCommandBuffer> cb = [cache.rt.queue commandBufferWithUnretainedReferences];
        if (bandv) {
            encode_back_band(cache, cb, w, cnt, band_tail / kBand, wide, uo, vo, index, c0);
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
    const unsigned solve_threads = metal_linalg::detail::cpu_threads_beside_gpu();
    auto solve = [&](size_t k) {
        Work& w = *ws[k % 2];
        const uint32_t cnt = cnt_of(k);
        if (bandv) {
            solve_band_vectors(w, cnt, band_tail, solve_threads, [&](uint32_t j, const float* dj) {
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

} // namespace

namespace core::detail {

void svd_bidiag_batch(const Matrices& a, float* u_out, float* s_out, float* vt_out, uint32_t* info_out) {
    const uint32_t L = std::max(a.rows, a.cols), K = std::min(a.rows, a.cols);
    const char* env = std::getenv("SVD_BIDIAG_BATCH_QR");
    if (L >= 2 * K && K > 0 && a.batch > 0 && !(env && std::string(env) == "0")) qr_first(a, u_out, s_out, vt_out, info_out);
    else direct(a, u_out, s_out, vt_out, info_out);
}

} // namespace core::detail
} // namespace metal_linalg
