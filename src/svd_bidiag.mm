// The SVD's `bidiag` backend: LAPACK's method with its two expensive steps on
// the GPU.
//
//   1. Bidiagonalize, A = Q B P^T, entirely on the GPU (shaders/Svd_Bidiag.metal):
//      blocked sgebrd (upper bidiagonal), per column slabrd's steps as small
//      kernels, per panel the trailing update as two MPS GEMMs; the panels'
//      command buffers are queued and the host waits once per matrix. The last
//      few columns, fewer than a panel, are reduced by LAPACK.
//   2. B = U_B diag(S) V_B^T on the CPU: LAPACK sbdsdc.
//   3. U = Q U_B and V^T = V_B^T P^T on the GPU, 128 reflectors at a time as
//      three MPS GEMMs each.
//
// A wide matrix is solved as its transpose; a matrix at least twice as tall as
// wide (and at least 64 wide) is first reduced by this library's QR, so the
// bidiagonalization sees the k x k factor, as LAPACK's sgesdd does.
//
// Why: for one large matrix the Jacobi backends lose to the CPU, and LAPACK's
// sgesdd spends about half its time in step 1, bound by memory bandwidth, and
// a quarter in step 3, which is matrix products. On an M5 Pro, one square
// N x N with singular vectors: 1.3x the CPU's speed at N = 2048, 2.1x at
// 4096. Below N ~ 1500 the launches and the per-panel work cost more than
// they save; the routing policy's bidiag_min_k is measured for that.
//
// Matrices are solved one after another; each is scaled by a power of two
// first (exact), so magnitudes whose products over- or underflow float32 work.

#ifndef ACCELERATE_NEW_LAPACK
#define ACCELERATE_NEW_LAPACK
#endif
#include <Accelerate/Accelerate.h>

#include <metal_linalg/core.h>
#include "metal_runtime.h"
#include "shaders.h"

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

using metal_linalg::core::Matrices;
using metal_linalg::detail::AutoreleasePool;
using metal_linalg::detail::MetalRuntime;
using metal_linalg::detail::Part;
using metal_linalg::detail::make_pipeline;
using metal_linalg::detail::scan;

namespace metal_linalg {
namespace {

constexpr uint32_t kPanel     = 32;    // columns per panel of the reduction
constexpr uint32_t kBackBlock = 128;   // reflectors per pass of the back-transformations
constexpr uint32_t kTile      = 64;    // must match TILE in Svd_Bidiag.metal
constexpr uint32_t kQrFirstMinK = 64;  // QR first for l >= 2k from this k

// Must match Svd_Bidiag.metal.
struct BdParams      { uint32_t mm, nn, lda, ldx, ldy, i, k; };
struct LarfgParams   { uint32_t len, inc, idx; };
struct GemvDims      { uint32_t rows, cols, ld, xinc; };
struct RestoreParams { uint32_t nb, lda, k; };

using L = __LAPACK_int;

struct Pipelines {
    id<MTLComputePipelineState> col_update, row_update, larfg, gt_tiles, gt_reduce, gn_tiles, gn_reduce,
        dots_col, dots_row, y_corr, x_corr, restore;
};

// Buffers for one (m, n), m >= n, reused across a batch; only the latest kept.
struct Workspace {
    uint32_t      m = 0, n = 0, lda = 0;
    id<MTLBuffer> A, X, Y, P, t, d, e, tq, tp;
    id<MTLBuffer> U, VT;                          // m x n and n x n, column-major
    id<MTLBuffer> V[2], T[2], Z, Z2;              // back-transformation blocks, two slots
};

struct Cache {
    MetalRuntime& rt = MetalRuntime::shared(METAL_LINALG_SHADER(Svd_Bidiag), "svd_bidiag");
    Pipelines p{};
    bool have = false;
    std::map<std::tuple<uint32_t, uint32_t, bool>, Workspace> workspaces;

    const Pipelines& pipelines() {
        if (!have) {
            auto mk = [&](NSString* n) { return make_pipeline(rt.device, rt.library, n, nil); };
            p.col_update = mk(@"bd_col_update");   p.row_update = mk(@"bd_row_update");
            p.larfg      = mk(@"bd_larfg");
            p.gt_tiles   = mk(@"bd_gemv_t_tiles"); p.gt_reduce  = mk(@"bd_gemv_t_reduce");
            p.gn_tiles   = mk(@"bd_gemv_n_tiles"); p.gn_reduce  = mk(@"bd_gemv_n_reduce");
            p.dots_col   = mk(@"bd_dots_col");     p.dots_row   = mk(@"bd_dots_row");
            p.y_corr     = mk(@"bd_y_corr");       p.x_corr     = mk(@"bd_x_corr");
            p.restore    = mk(@"bd_restore");
            have = true;
        }
        return p;
    }

    Workspace& workspace(uint32_t m, uint32_t n, bool vectors) {
        const auto key = std::make_tuple(m, n, vectors);
        if (auto it = workspaces.find(key); it != workspaces.end()) return it->second;
        // Several m x n buffers: only the latest shape is kept, so a program
        // solving many shapes does not accumulate them.
        workspaces.clear();
        auto shared = [&](size_t f) { return [rt.device newBufferWithLength:std::max<size_t>(f, 4) * 4
                                                                    options:MTLResourceStorageModeShared]; };
        auto priv = [&](size_t f) { return [rt.device newBufferWithLength:std::max<size_t>(f, 4) * 4
                                                                  options:MTLResourceStorageModePrivate]; };
        Workspace w;
        w.m = m; w.n = n; w.lda = (m + 7) / 8 * 8;
        w.A  = shared((size_t)w.lda * n);
        w.X  = priv((size_t)m * kPanel);
        w.Y  = priv((size_t)n * kPanel);
        w.P  = priv((size_t)((m + kTile - 1) / kTile) * m);
        w.t  = priv(2 * kPanel + 2);
        w.d  = shared(n); w.e = shared(n); w.tq = shared(n); w.tp = shared(n);
        if (vectors) {
            w.U = shared((size_t)m * n);
            w.VT = shared((size_t)n * n);
            for (int s = 0; s < 2; ++s) { w.V[s] = shared((size_t)m * kBackBlock); w.T[s] = shared(kBackBlock * kBackBlock); }
            w.Z = priv((size_t)std::max(m, n) * kBackBlock);
            w.Z2 = priv((size_t)std::max(m, n) * kBackBlock);
        }
        return workspaces[key] = w;
    }
};

MPSMatrix* mps(id<MTLBuffer> b, size_t off, uint32_t rows, uint32_t cols, uint32_t ld) {
    MPSMatrixDescriptor* d = [MPSMatrixDescriptor matrixDescriptorWithRows:rows columns:cols
                                                                  rowBytes:(size_t)ld * 4 dataType:MPSDataTypeFloat32];
    return [[MPSMatrix alloc] initWithBuffer:b offset:off * 4 descriptor:d];
}

void gemm(id<MTLDevice> dev, id<MTLCommandBuffer> cb, MPSMatrix* A, bool ta, MPSMatrix* B, bool tb, MPSMatrix* C,
          uint32_t m, uint32_t n, uint32_t k, double alpha, double beta) {
    MPSMatrixMultiplication* g = [[MPSMatrixMultiplication alloc] initWithDevice:dev transposeLeft:ta
        transposeRight:tb resultRows:m resultColumns:n interiorColumns:k alpha:alpha beta:beta];
    [g encodeToCommandBuffer:cb leftMatrix:A rightMatrix:B resultMatrix:C];
}

void run_threads(id<MTLComputeCommandEncoder> e, id<MTLComputePipelineState> ps, uint32_t n) {
    [e dispatchThreads:MTLSizeMake(std::max<uint32_t>(n, 1), 1, 1)
        threadsPerThreadgroup:MTLSizeMake(std::min<uint32_t>(256, (uint32_t)ps.maxTotalThreadsPerThreadgroup), 1, 1)];
}

void check(id<MTLCommandBuffer> cb) {
    if (cb.error) throw std::runtime_error(std::string("[svd] bidiag: GPU error: ") + cb.error.localizedDescription.UTF8String);
}

// Step 1 on ws.A (m x n column-major, m >= n): d, e, tauq, taup as sgebrd
// leaves them, and the reflectors in ws.A.
void bidiagonalize(Cache& c, Workspace& ws, float* d, float* e, float* tq, float* tp) {
    const Pipelines& p = c.pipelines();
    id<MTLDevice> dev = c.rt.device;
    const uint32_t m = ws.m, n = ws.n, lda = ws.lda, nb = kPanel;
    const uint32_t tg1024 = std::min<uint32_t>(1024, (uint32_t)p.larfg.maxTotalThreadsPerThreadgroup);
    id<MTLCommandBuffer> last = nil;
    uint32_t k = 0;
    for (; k + nb + 1 < n; k += nb) {
        id<MTLCommandBuffer> cb = [c.rt.queue commandBufferWithUnretainedReferences];
        const uint32_t mm = m - k, nn = n - k;
        const size_t ok = (size_t)k * lda + k;
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        for (uint32_t i = 0; i < nb; ++i) {
            const BdParams prm{mm, nn, lda, m, n, i, k};
            if (i > 0) {
                [enc setComputePipelineState:p.col_update];
                [enc setBuffer:ws.A offset:ok * 4 atIndex:0]; [enc setBuffer:ws.X offset:0 atIndex:1];
                [enc setBuffer:ws.Y offset:0 atIndex:2]; [enc setBytes:&prm length:sizeof prm atIndex:3];
                run_threads(enc, p.col_update, mm - i);
            }
            {   const LarfgParams lp{mm - i, 1, k + i};
                [enc setComputePipelineState:p.larfg];
                [enc setBuffer:ws.A offset:(ok + (size_t)i * lda + i) * 4 atIndex:0];
                [enc setBuffer:ws.d offset:0 atIndex:1]; [enc setBuffer:ws.tq offset:0 atIndex:2];
                [enc setBytes:&lp length:sizeof lp atIndex:3];
                [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg1024, 1, 1)]; }
            {   // Y(i+1:, i) = Ak(i:, i+1:)^T v
                const GemvDims gd{mm - i, nn - i - 1, lda, 1};
                [enc setComputePipelineState:p.gt_tiles];
                [enc setBuffer:ws.A offset:(ok + (size_t)(i + 1) * lda + i) * 4 atIndex:0];
                [enc setBuffer:ws.A offset:(ok + (size_t)i * lda + i) * 4 atIndex:1];
                [enc setBuffer:ws.P offset:0 atIndex:2]; [enc setBytes:&gd length:sizeof gd atIndex:3];
                [enc dispatchThreadgroups:MTLSizeMake((gd.cols + kTile - 1) / kTile, (gd.rows + kTile - 1) / kTile, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc setComputePipelineState:p.gt_reduce];
                [enc setBuffer:ws.P offset:0 atIndex:0]; [enc setBuffer:ws.Y offset:((size_t)i * n + i + 1) * 4 atIndex:1];
                [enc setBytes:&gd length:sizeof gd atIndex:2];
                run_threads(enc, p.gt_reduce, gd.cols); }
            if (i > 0) {
                [enc setComputePipelineState:p.dots_col];
                [enc setBuffer:ws.A offset:ok * 4 atIndex:0]; [enc setBuffer:ws.X offset:0 atIndex:1];
                [enc setBuffer:ws.t offset:0 atIndex:2]; [enc setBytes:&prm length:sizeof prm atIndex:3];
                [enc dispatchThreadgroups:MTLSizeMake(2 * i, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            }
            [enc setComputePipelineState:p.y_corr];
            [enc setBuffer:ws.A offset:ok * 4 atIndex:0]; [enc setBuffer:ws.Y offset:0 atIndex:1];
            [enc setBuffer:ws.t offset:0 atIndex:2]; [enc setBuffer:ws.tq offset:0 atIndex:3];
            [enc setBytes:&prm length:sizeof prm atIndex:4];
            run_threads(enc, p.y_corr, nn - i - 1);
            [enc setComputePipelineState:p.row_update];
            [enc setBuffer:ws.A offset:ok * 4 atIndex:0]; [enc setBuffer:ws.X offset:0 atIndex:1];
            [enc setBuffer:ws.Y offset:0 atIndex:2]; [enc setBytes:&prm length:sizeof prm atIndex:3];
            run_threads(enc, p.row_update, nn - i - 1);
            {   const LarfgParams lp{nn - i - 1, lda, k + i};
                [enc setComputePipelineState:p.larfg];
                [enc setBuffer:ws.A offset:(ok + (size_t)(i + 1) * lda + i) * 4 atIndex:0];
                [enc setBuffer:ws.e offset:0 atIndex:1]; [enc setBuffer:ws.tp offset:0 atIndex:2];
                [enc setBytes:&lp length:sizeof lp atIndex:3];
                [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg1024, 1, 1)]; }
            {   // X(i+1:, i) = Ak(i+1:, i+1:) u, u = Ak(i, i+1:) (stride lda)
                const GemvDims gd{mm - i - 1, nn - i - 1, lda, lda};
                [enc setComputePipelineState:p.gn_tiles];
                [enc setBuffer:ws.A offset:(ok + (size_t)(i + 1) * lda + i + 1) * 4 atIndex:0];
                [enc setBuffer:ws.A offset:(ok + (size_t)(i + 1) * lda + i) * 4 atIndex:1];
                [enc setBuffer:ws.P offset:0 atIndex:2]; [enc setBytes:&gd length:sizeof gd atIndex:3];
                [enc dispatchThreadgroups:MTLSizeMake((gd.cols + kTile - 1) / kTile, (gd.rows + kTile - 1) / kTile, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc setComputePipelineState:p.gn_reduce];
                [enc setBuffer:ws.P offset:0 atIndex:0]; [enc setBuffer:ws.X offset:((size_t)i * m + i + 1) * 4 atIndex:1];
                [enc setBytes:&gd length:sizeof gd atIndex:2];
                run_threads(enc, p.gn_reduce, gd.rows); }
            [enc setComputePipelineState:p.dots_row];
            [enc setBuffer:ws.A offset:ok * 4 atIndex:0]; [enc setBuffer:ws.Y offset:0 atIndex:1];
            [enc setBuffer:ws.t offset:0 atIndex:2]; [enc setBytes:&prm length:sizeof prm atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake(2 * i + 1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc setComputePipelineState:p.x_corr];
            [enc setBuffer:ws.A offset:ok * 4 atIndex:0]; [enc setBuffer:ws.X offset:0 atIndex:1];
            [enc setBuffer:ws.t offset:0 atIndex:2]; [enc setBuffer:ws.tp offset:0 atIndex:3];
            [enc setBytes:&prm length:sizeof prm atIndex:4];
            run_threads(enc, p.x_corr, mm - i - 1);
        }
        [enc endEncoding];
        // A22 -= V Y2^T + X2 U, as A22^T -= Y2 V^T + U^T X2^T on the row-major
        // views of the column-major buffers.
        const uint32_t m2 = mm - nb, n2 = nn - nb;
        MPSMatrix* A22t = mps(ws.A, ok + (size_t)nb * lda + nb, n2, m2, lda);
        gemm(dev, cb, mps(ws.Y, nb, nb, n2, n), true, mps(ws.A, ok + nb, nb, m2, lda), false, A22t, n2, m2, nb, -1, 1);
        gemm(dev, cb, mps(ws.A, ok + (size_t)nb * lda, n2, nb, lda), false, mps(ws.X, nb, nb, m2, m), false,
             A22t, n2, m2, nb, -1, 1);
        enc = [cb computeCommandEncoder];
        const RestoreParams rp{nb, lda, k};
        [enc setComputePipelineState:p.restore];
        [enc setBuffer:ws.A offset:ok * 4 atIndex:0]; [enc setBuffer:ws.d offset:0 atIndex:1];
        [enc setBuffer:ws.e offset:0 atIndex:2]; [enc setBytes:&rp length:sizeof rp atIndex:3];
        [enc dispatchThreads:MTLSizeMake(nb, 1, 1) threadsPerThreadgroup:MTLSizeMake(nb, 1, 1)];
        [enc endEncoding];
        [cb commit];
        last = cb;
    }
    if (last) {
        [last waitUntilCompleted];
        check(last);
        std::memcpy(d, ws.d.contents, k * 4);  std::memcpy(e, ws.e.contents, k * 4);
        std::memcpy(tq, ws.tq.contents, k * 4); std::memcpy(tp, ws.tp.contents, k * 4);
    }
    float* A = static_cast<float*>(ws.A.contents);
    L M = m - k, N = n - k, LDA = lda, info = 0, lw = -1;
    float q = 0;
    sgebrd_(&M, &N, A + (size_t)k * lda + k, &LDA, d + k, e + k, tq + k, tp + k, &q, &lw, &info);
    std::vector<float> w(std::max<L>(1, (L)q));
    lw = (L)w.size();
    sgebrd_(&M, &N, A + (size_t)k * lda + k, &LDA, d + k, e + k, tq + k, tp + k, w.data(), &lw, &info);
    if (info != 0) throw std::runtime_error("[svd] bidiag: LAPACK sgebrd failed, info " + std::to_string((long long)info));
}

// Step 3. Q's reflector j acts on rows j.. (offset 0), P's on columns j+1..
// (offset 1). `left`: Z <- Q Z for Z = ws.U (m x n); else Z <- Z P^T for
// Z = ws.VT (n x n). Blocks are applied last first; each block's V and T are
// built on the CPU (slarft) in one of two slots while the GPU applies the other.
void back_transform(Cache& c, Workspace& ws, const float* tau, bool left) {
    const uint32_t m = ws.m, n = ws.n, lda = ws.lda, bb = kBackBlock;
    const uint32_t refl = left ? n : n - 1, off = left ? 0 : 1;
    if (refl == 0) return;
    const float* A = static_cast<const float*>(ws.A.contents);
    id<MTLDevice> dev = c.rt.device;
    std::vector<float> vc, tc((size_t)bb * bb);
    id<MTLCommandBuffer> inflight[2] = {nil, nil};
    int slot = 0;
    for (int k0 = (int)(((refl - 1) / bb) * bb); k0 >= 0; k0 -= (int)bb) {
        const uint32_t kb = std::min<uint32_t>(bb, refl - (uint32_t)k0);
        const uint32_t len = (left ? m : n) - (uint32_t)k0 - off;   // the rows (Q) or columns (P) acted on
        if (inflight[slot]) [inflight[slot] waitUntilCompleted];
        vc.assign((size_t)len * kb, 0.0f);
        for (uint32_t j = 0; j < kb; ++j) {
            vc[(size_t)j * len + j] = 1.0f;
            for (uint32_t r = j + 1; r < len; ++r)
                vc[(size_t)j * len + r] = left ? A[(size_t)(k0 + j) * lda + k0 + r]         // column j, below the diagonal
                                               : A[(size_t)(k0 + 1 + r) * lda + k0 + j];    // row j, right of the superdiagonal
        }
        L M = len, KB = kb, LDT = bb;
        slarft_("F", "C", &M, &KB, vc.data(), &M, const_cast<float*>(tau) + k0, tc.data(), &LDT);
        float* V = static_cast<float*>(ws.V[slot].contents);
        float* T = static_cast<float*>(ws.T[slot].contents);
        for (uint32_t r = 0; r < len; ++r)
            for (uint32_t j = 0; j < bb; ++j) V[(size_t)r * bb + j] = j < kb ? vc[(size_t)j * len + r] : 0.0f;
        for (uint32_t i = 0; i < bb; ++i)
            for (uint32_t j = 0; j < bb; ++j)
                T[(size_t)i * bb + j] = (i < kb && j < kb && j >= i) ? tc[(size_t)j * bb + i] : 0.0f;

        id<MTLCommandBuffer> cb = [c.rt.queue commandBufferWithUnretainedReferences];
        MPSMatrix* Vm = mps(ws.V[slot], 0, len, bb, bb);
        MPSMatrix* Tm = mps(ws.T[slot], 0, bb, bb, bb);
        if (left) {
            // U column-major (ld m) is U^T row-major: U^T(:, k0:) -= ((U^T(:, k0:) V) T^T) V^T.
            MPSMatrix* S = mps(ws.U, (size_t)k0, n, len, m);
            MPSMatrix* Z = mps(ws.Z, 0, n, bb, bb);
            MPSMatrix* Z2 = mps(ws.Z2, 0, n, bb, bb);
            gemm(dev, cb, S, false, Vm, false, Z, n, bb, len, 1, 0);
            gemm(dev, cb, Z, false, Tm, true, Z2, n, bb, bb, 1, 0);
            gemm(dev, cb, Z2, false, Vm, true, S, n, len, bb, -1, 1);
        } else {
            // VT column-major is VT^T row-major: VT^T(k0+1:, :) -= V T (V^T VT^T(k0+1:, :)).
            MPSMatrix* S = mps(ws.VT, (size_t)(k0 + 1) * n, len, n, n);
            MPSMatrix* Z = mps(ws.Z, 0, bb, n, n);
            MPSMatrix* Z2 = mps(ws.Z2, 0, bb, n, n);
            gemm(dev, cb, Vm, true, S, false, Z, bb, n, len, 1, 0);
            gemm(dev, cb, Tm, false, Z, false, Z2, bb, n, bb, 1, 0);
            gemm(dev, cb, Vm, false, Z2, false, S, len, n, bb, -1, 1);
        }
        [cb commit];
        inflight[slot] = cb;
        slot ^= 1;
    }
    for (id<MTLCommandBuffer> cb : inflight) {
        if (!cb) continue;
        [cb waitUntilCompleted];
        check(cb);
    }
}

// One tall matrix (row-major l x k, l >= k), already scaled: S descending, and
// with vectors U (row-major l x k) and Vt (row-major k x k).
void tall_svd(Cache& c, const float* t, uint32_t l, uint32_t k, float* s, float* u, float* vt) {
    const bool vectors = u != nullptr;
    // Much taller than wide: R of a QR first, U = Q U_R at the end.
    if (l >= 2 * k && k >= kQrFirstMinK) {
        std::vector<float> q(vectors ? (size_t)l * k : 0), r((size_t)k * k), qtmp;
        if (!vectors) qtmp.resize((size_t)l * k);
        core::qr(Matrices{t, 1, l, k}, vectors ? q.data() : qtmp.data(), r.data());
        std::vector<float> ur(vectors ? (size_t)k * k : 0);
        tall_svd(c, r.data(), k, k, s, vectors ? ur.data() : nullptr, vt);
        if (vectors)
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, l, k, k, 1.0f, q.data(), k, ur.data(), k, 0.0f, u, k);
        return;
    }
    Workspace& ws = c.workspace(l, k, vectors);
    float* A = static_cast<float*>(ws.A.contents);
    for (uint32_t j = 0; j < k; ++j)
        for (uint32_t i = 0; i < l; ++i) A[(size_t)j * ws.lda + i] = t[(size_t)i * k + j];
    std::vector<float> d(k), e(k), tq(k), tp(k);
    bidiagonalize(c, ws, d.data(), e.data(), tq.data(), tp.data());

    char uplo = 'U', compq = vectors ? 'I' : 'N';
    L N = k, ld = k, info = 0, iq = 0;
    float qd = 0;
    std::vector<float> ub(vectors ? (size_t)k * k : 1), vb(vectors ? (size_t)k * k : 1);
    std::vector<float> work(3 * (size_t)k * k + 4 * (size_t)k + 8 * (size_t)k + 16);
    std::vector<L> iwork(8 * (size_t)k + 8);
    sbdsdc_(&uplo, &compq, &N, d.data(), e.data(), ub.data(), &ld, vb.data(), &ld, &qd, &iq, work.data(),
            iwork.data(), &info);
    if (info != 0) throw std::runtime_error("[svd] bidiag: LAPACK sbdsdc failed, info " + std::to_string((long long)info));
    std::copy(d.begin(), d.end(), s);   // descending
    if (!vectors) return;

    float* U = static_cast<float*>(ws.U.contents);
    std::fill(U, U + (size_t)l * k, 0.0f);
    for (uint32_t j = 0; j < k; ++j) std::copy(ub.begin() + (size_t)j * k, ub.begin() + (size_t)(j + 1) * k, U + (size_t)j * l);
    std::copy(vb.begin(), vb.end(), static_cast<float*>(ws.VT.contents));
    back_transform(c, ws, tq.data(), true);
    back_transform(c, ws, tp.data(), false);
    vDSP_mtrans(U, 1, u, 1, l, k);                                           // to row-major l x k
    vDSP_mtrans(static_cast<const float*>(ws.VT.contents), 1, vt, 1, k, k);  // to row-major k x k
}

} // namespace

namespace core::detail {

void svd_bidiag(const Matrices& a, float* u_out, float* s_out, float* vt_out, uint32_t* info_out) {
    const uint32_t M = a.rows, N = a.cols, batch = a.batch, K = std::min(M, N);
    const bool vectors = u_out || vt_out;
    if (K == 0 || batch == 0) {
        if (info_out) std::fill(info_out, info_out + batch, 0u);
        return;
    }
    AutoreleasePool pool;
    static Cache cache;
    std::vector<float> amax(batch);
    std::vector<char>  finite(batch);
    scan(a, Part::all, amax.data(), finite.data());

    const uint32_t l = std::max(M, N);
    const bool wide = M < N;
    std::vector<float> t((size_t)l * K), ut(vectors ? (size_t)l * K : 0), vtt(vectors ? (size_t)K * K : 0);
    for (uint32_t b = 0; b < batch; ++b) {
        float* s = s_out + (size_t)b * K;
        float* u = u_out ? u_out + (size_t)b * M * K : nullptr;
        float* vt = vt_out ? vt_out + (size_t)b * K * N : nullptr;
        if (!finite[b]) {
            std::fill(s, s + K, NAN);
            if (u) std::fill(u, u + (size_t)M * K, NAN);
            if (vt) std::fill(vt, vt + (size_t)K * N, NAN);
            if (info_out) info_out[b] = 1u << 17;
            continue;
        }
        int ex = 0;
        if (amax[b] > 0.0f) std::frexp(amax[b], &ex);
        const float scale = std::ldexp(1.0f, -ex);
        // The tall matrix (row-major l x K): A itself, or A^T for a wide one.
        const float* src = a.data + (size_t)b * M * N;
        if (wide) {
            for (uint32_t i = 0; i < N; ++i)
                for (uint32_t j = 0; j < M; ++j) t[(size_t)i * M + j] = src[(size_t)j * N + i] * scale;
        } else {
            for (size_t i = 0; i < (size_t)M * N; ++i) t[i] = src[i] * scale;
        }
        tall_svd(cache, t.data(), l, K, s, vectors ? ut.data() : nullptr, vectors ? vtt.data() : nullptr);
        for (uint32_t i = 0; i < K; ++i) s[i] /= scale;
        if (vectors) {
            if (!wide) {
                if (u) std::copy(ut.begin(), ut.end(), u);
                if (vt) std::copy(vtt.begin(), vtt.end(), vt);
            } else {
                // A^T = U' S V'^T  =>  A = V' S U'^T: U = V' = (Vt')^T, Vt = U'^T.
                if (u) vDSP_mtrans(vtt.data(), 1, u, 1, K, K);
                if (vt) vDSP_mtrans(ut.data(), 1, vt, 1, K, l);
            }
        }
        if (info_out) info_out[b] = 1u | (1u << 16);
    }
}

} // namespace core::detail
} // namespace metal_linalg
