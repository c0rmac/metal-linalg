// The SVD's `bidiag` backend: LAPACK's method with its two expensive steps on
// the GPU; and the `band` backend, in two stages (below, band_reduce), for
// singular values alone and, since 2.15.0, with vectors (band_vectors).
//
//   1. Bidiagonalize, A = Q B P^T, entirely on the GPU (shaders/Svd_Bidiag.metal):
//      blocked sgebrd (upper bidiagonal), per column slabrd's steps as four
//      kernels, per panel the trailing update as two MPS GEMMs; the panels'
//      command buffers are queued and the host waits once per matrix. The last
//      few columns, fewer than a panel, are reduced by LAPACK.
//   2. B = U_B diag(S) V_B^T on the CPU: sbdsdc's divide and conquer on
//      every core but two (divide_conquer.cpp); the singular values alone by
//      bisection on the GPU (bisect.mm) or sbdsqr (dqds).
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
// N x N with singular vectors: 1.4x the CPU's speed at N = 1024, 1.75x at
// 2048, 2.3x at 4096. Below N ~ 1000 the launches and the per-panel work cost
// more than they save; the routing policy's bidiag_min_k is measured for that.
//
// A batch is pipelined over two workspace slots, as the eigensolver's tridiag
// backend does: the CPU solves one matrix's bidiagonal problem while the GPU
// reduces the next, and solves the next while the GPU back-transforms this
// one: on an M5 Pro, per matrix of 2048 x 2048 with vectors, 253 ms alone,
// 168 ms in a batch of 4, 155 ms in a batch of 8. Each matrix is scaled by a
// power of two first (exact), so magnitudes whose products over- or underflow
// float32 work.
//
// `band`, for singular values alone: the reduction is in two stages, as
// LAPACK's ssyevd_2stage does for symmetric eigenvalues. A to an upper band of
// width b on the GPU (band_reduce.mm), a block of b columns at a time, the work
// matrix products that read the matrix three times a block where the one-stage
// reduction reads it twice a column; then the band to bidiagonal by bulge
// chasing on the CPU's cores (band_chase.cpp) and its singular values by
// bisection on the GPU (bisect.mm). On an M5 Pro, one 4096 x 4096: 234 ms
// against bidiag's 757; from about 2048 it is the faster. See
// docs/studies/two-stage-apple-m5-pro.md. With vectors, both stages'
// reflectors are kept and applied on the GPU while the CPU chases the band
// and solves the bidiagonal problem (band_vectors, below): about 380-400 ms
// against bidiag's 940 at 4096.

#ifndef ACCELERATE_NEW_LAPACK
#define ACCELERATE_NEW_LAPACK
#endif
#include <Accelerate/Accelerate.h>

#include <metal_linalg/core.h>
#include <metal_linalg/device.h>
#include "band_chase.h"
#include "divide_conquer.h"
#include "metal_runtime.h"
#include "shaders.h"

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <future>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using metal_linalg::core::Matrices;
using metal_linalg::detail::AutoreleasePool;
using metal_linalg::detail::MetalRuntime;
using metal_linalg::detail::Part;
using metal_linalg::detail::make_pipeline;
using metal_linalg::detail::scan;
using metal_linalg::detail::transpose_scaled;

namespace metal_linalg {
namespace {

constexpr uint32_t kPanel     = 32;    // columns per panel of the reduction
constexpr uint32_t kBackBlock = 128;   // reflectors per pass of the back-transformations
constexpr uint32_t kTile      = 64;    // must match TILE in Svd_Bidiag.metal
constexpr uint32_t kGroup     = 256;   // must match GROUP
constexpr uint32_t kRows      = 32;    // must match GROUP / LANES: bd_col's and bd_row's rows per threadgroup
constexpr uint32_t kQrFirstMinK = 64;  // QR first for l >= 2k from this k

// Must match Svd_Bidiag.metal.
struct BdParams      { uint32_t mm, nn, lda, ldx, ldy, i, k, ng, tiles, last; };
struct RestoreParams { uint32_t nb, lda, k; };

uint32_t groups(uint32_t rows) { return (rows + kRows - 1) / kRows; }
uint32_t blocks(uint32_t n) { return (n + kTile - 1) / kTile; }

using L = __LAPACK_int;

struct Pipelines {
    id<MTLComputePipelineState> col, gemv_t, row, gemv_n, restore;
};

// Buffers for one (m, n), m >= n, reused across a batch; only the latest kept.
// A, U and VT come in two pipeline slots (the second allocated for a batch).
struct Workspace {
    uint32_t      m = 0, n = 0, lda = 0;
    id<MTLBuffer> A[2], X, Y, P, t, d, e, tq, tp;
    id<MTLBuffer> npart, scal;                    // a reflector's norm partials; the two reflectors' scales
    id<MTLBuffer> U[2], VT[2];                    // m x n and n x n, column-major
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
            p.col = mk(@"bd_col");  p.gemv_t = mk(@"bd_gemv_t");
            p.row = mk(@"bd_row");  p.gemv_n = mk(@"bd_gemv_n");
            p.restore = mk(@"bd_restore");
            for (id<MTLComputePipelineState> ps : {p.col, p.gemv_t, p.row, p.gemv_n}) {
                if (ps.maxTotalThreadsPerThreadgroup < kGroup) {
                    throw std::runtime_error("[svd] bidiag: a pipeline allows fewer than 256 threads per threadgroup.");
                }
            }
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
        w.A[0] = shared((size_t)w.lda * n);
        w.X  = priv((size_t)m * kPanel);
        w.Y  = priv((size_t)n * kPanel);
        w.P  = priv((size_t)((m + kTile - 1) / kTile) * m);
        w.t  = priv(2 * kPanel + 2);
        w.npart = priv(2 * (groups(m) + 1));
        w.scal  = priv(2);
        w.d  = shared(n); w.e = shared(n); w.tq = shared(n); w.tp = shared(n);
        if (vectors) {
            w.U[0] = shared((size_t)m * n);
            w.VT[0] = shared((size_t)n * n);
            for (int s = 0; s < 2; ++s) { w.V[s] = shared((size_t)m * kBackBlock); w.T[s] = shared(kBackBlock * kBackBlock); }
            w.Z = priv((size_t)std::max(m, n) * kBackBlock);
            w.Z2 = priv((size_t)std::max(m, n) * kBackBlock);
        }
        return workspaces[key] = w;
    }

    // The second pipeline slot, for a batch.
    void second_slot(Workspace& w, bool vectors) {
        auto shared = [&](size_t f) { return [rt.device newBufferWithLength:std::max<size_t>(f, 4) * 4
                                                                    options:MTLResourceStorageModeShared]; };
        if (!w.A[1]) w.A[1] = shared((size_t)w.lda * w.n);
        if (vectors && !w.U[1]) { w.U[1] = shared((size_t)w.m * w.n); w.VT[1] = shared((size_t)w.n * w.n); }
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

void check(id<MTLCommandBuffer> cb) {
    if (cb.error) throw std::runtime_error(std::string("[svd] bidiag: GPU error: ") + cb.error.localizedDescription.UTF8String);
}

// Step 1 on Abuf (m x n column-major, m >= n): d, e, tauq, taup as sgebrd
// leaves them, and the reflectors in Abuf.
void bidiagonalize(Cache& c, Workspace& ws, id<MTLBuffer> Abuf, float* d, float* e, float* tq, float* tp) {
    const Pipelines& p = c.pipelines();
    id<MTLDevice> dev = c.rt.device;
    const uint32_t m = ws.m, n = ws.n, lda = ws.lda, nb = kPanel;
    id<MTLCommandBuffer> last = nil;
    uint32_t k = 0;
    for (; k + nb + 1 < n; k += nb) {
        id<MTLCommandBuffer> cb = [c.rt.queue commandBufferWithUnretainedReferences];
        const uint32_t mm = m - k, nn = n - k;
        const size_t ok = (size_t)k * lda + k;
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        // Column i's steps; with `last`, bd_col only finishes the panel's last column.
        auto col = [&](uint32_t i, uint32_t last) {
            const BdParams prm{mm, nn, lda, m, n, i, k, 0, 0, last};
            [enc setComputePipelineState:p.col];
            [enc setBuffer:Abuf offset:ok * 4 atIndex:0]; [enc setBuffer:ws.X offset:0 atIndex:1];
            [enc setBuffer:ws.Y offset:0 atIndex:2];      [enc setBuffer:ws.P offset:0 atIndex:3];
            [enc setBuffer:ws.t offset:0 atIndex:4];      [enc setBuffer:ws.tp offset:0 atIndex:5];
            [enc setBuffer:ws.scal offset:0 atIndex:6];   [enc setBuffer:ws.npart offset:0 atIndex:7];
            [enc setBytes:&prm length:sizeof prm atIndex:8];
            [enc dispatchThreadgroups:MTLSizeMake(groups(mm - i), 1, 1) threadsPerThreadgroup:MTLSizeMake(kGroup, 1, 1)];
        };
        for (uint32_t i = 0; i < nb; ++i) {
            // Finish X(:, i-1) and store u(i-1); update Ak(i:, i); its norm partials.
            col(i, 0);
            {   // The column reflector; Ak(i:, i+1:)^T v in tiles; t1, t2.
                const uint32_t tiles = blocks(mm - i) * blocks(nn - i - 1);
                const BdParams prm{mm, nn, lda, m, n, i, k, groups(mm - i), tiles, 0};
                [enc setComputePipelineState:p.gemv_t];
                [enc setBuffer:Abuf offset:ok * 4 atIndex:0]; [enc setBuffer:ws.X offset:0 atIndex:1];
                [enc setBuffer:ws.P offset:0 atIndex:2];      [enc setBuffer:ws.t offset:0 atIndex:3];
                [enc setBuffer:ws.npart offset:0 atIndex:4];  [enc setBuffer:ws.d offset:0 atIndex:5];
                [enc setBuffer:ws.tq offset:0 atIndex:6];     [enc setBuffer:ws.scal offset:0 atIndex:7];
                [enc setBytes:&prm length:sizeof prm atIndex:8];
                [enc dispatchThreadgroups:MTLSizeMake(tiles + 2 * i, 1, 1) threadsPerThreadgroup:MTLSizeMake(kGroup, 1, 1)];
            }
            {   // Y(i+1:, i); store v; update Ak(i, i+1:); its norm partials.
                const BdParams prm{mm, nn, lda, m, n, i, k, 0, 0, 0};
                [enc setComputePipelineState:p.row];
                [enc setBuffer:Abuf offset:ok * 4 atIndex:0]; [enc setBuffer:ws.X offset:0 atIndex:1];
                [enc setBuffer:ws.Y offset:0 atIndex:2];      [enc setBuffer:ws.P offset:0 atIndex:3];
                [enc setBuffer:ws.t offset:0 atIndex:4];      [enc setBuffer:ws.tq offset:0 atIndex:5];
                [enc setBuffer:ws.scal offset:0 atIndex:6];   [enc setBuffer:ws.npart offset:0 atIndex:7];
                [enc setBytes:&prm length:sizeof prm atIndex:8];
                [enc dispatchThreadgroups:MTLSizeMake(groups(mm - i), 1, 1) threadsPerThreadgroup:MTLSizeMake(kGroup, 1, 1)];
            }
            {   // The row reflector; Ak(i+1:, i+1:) u in tiles; t3, t4.
                const uint32_t tiles = blocks(mm - i - 1) * blocks(nn - i - 1);
                const BdParams prm{mm, nn, lda, m, n, i, k, groups(mm - i), tiles, 0};
                [enc setComputePipelineState:p.gemv_n];
                [enc setBuffer:Abuf offset:ok * 4 atIndex:0]; [enc setBuffer:ws.Y offset:0 atIndex:1];
                [enc setBuffer:ws.P offset:0 atIndex:2];      [enc setBuffer:ws.t offset:0 atIndex:3];
                [enc setBuffer:ws.npart offset:0 atIndex:4];  [enc setBuffer:ws.e offset:0 atIndex:5];
                [enc setBuffer:ws.tp offset:0 atIndex:6];     [enc setBuffer:ws.scal offset:0 atIndex:7];
                [enc setBytes:&prm length:sizeof prm atIndex:8];
                [enc dispatchThreadgroups:MTLSizeMake(tiles + 2 * i + 1, 1, 1) threadsPerThreadgroup:MTLSizeMake(kGroup, 1, 1)];
            }
        }
        col(nb, 1);
        [enc endEncoding];
        // A22 -= V Y2^T + X2 U, as A22^T -= Y2 V^T + U^T X2^T on the row-major
        // views of the column-major buffers.
        const uint32_t m2 = mm - nb, n2 = nn - nb;
        MPSMatrix* A22t = mps(Abuf, ok + (size_t)nb * lda + nb, n2, m2, lda);
        gemm(dev, cb, mps(ws.Y, nb, nb, n2, n), true, mps(Abuf, ok + nb, nb, m2, lda), false, A22t, n2, m2, nb, -1, 1);
        gemm(dev, cb, mps(Abuf, ok + (size_t)nb * lda, n2, nb, lda), false, mps(ws.X, nb, nb, m2, m), false,
             A22t, n2, m2, nb, -1, 1);
        enc = [cb computeCommandEncoder];
        const RestoreParams rp{nb, lda, k};
        [enc setComputePipelineState:p.restore];
        [enc setBuffer:Abuf offset:ok * 4 atIndex:0]; [enc setBuffer:ws.d offset:0 atIndex:1];
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
    float* A = static_cast<float*>(Abuf.contents);
    L M = m - k, N = n - k, LDA = lda, info = 0, lw = -1;
    float q = 0;
    sgebrd_(&M, &N, A + (size_t)k * lda + k, &LDA, d + k, e + k, tq + k, tp + k, &q, &lw, &info);
    std::vector<float> w(std::max<L>(1, (L)q));
    lw = (L)w.size();
    sgebrd_(&M, &N, A + (size_t)k * lda + k, &LDA, d + k, e + k, tq + k, tp + k, w.data(), &lw, &info);
    if (info != 0) throw std::runtime_error("[svd] bidiag: LAPACK sgebrd failed, info " + std::to_string((long long)info));
}

// Step 3. Q's reflector j acts on rows j.. (offset 0), P's on columns j+1..
// (offset 1), both read from Abuf. `left`: Z <- Q Z for Z = Ubuf (m x n);
// else Z <- Z P^T for Z = VTbuf (n x n). Blocks are applied last first; each
// block's V and T are built on the CPU (compact_wy_t, the copies on every
// core) in one of two slots while the GPU applies the other.
void back_transform(Cache& c, Workspace& ws, id<MTLBuffer> Abuf, id<MTLBuffer> Ubuf, id<MTLBuffer> VTbuf,
                    const float* tau, bool left) {
    const uint32_t m = ws.m, n = ws.n, lda = ws.lda, bb = kBackBlock;
    const uint32_t refl = left ? n : n - 1, off = left ? 0 : 1;
    if (refl == 0) return;
    const float* A = static_cast<const float*>(Abuf.contents);
    id<MTLDevice> dev = c.rt.device;
    std::vector<float> vc, tc((size_t)bb * bb);
    id<MTLCommandBuffer> inflight[2] = {nil, nil};
    int slot = 0;
    for (int k0 = (int)(((refl - 1) / bb) * bb); k0 >= 0; k0 -= (int)bb) {
        const uint32_t kb = std::min<uint32_t>(bb, refl - (uint32_t)k0);
        const uint32_t len = (left ? m : n) - (uint32_t)k0 - off;   // the rows (Q) or columns (P) acted on
        if (inflight[slot]) [inflight[slot] waitUntilCompleted];
        vc.resize((size_t)len * kb);
        metal_linalg::detail::parallel_for(kb, [&](size_t j) {
            float* col = vc.data() + j * len;
            std::fill(col, col + j, 0.0f);
            col[j] = 1.0f;
            for (uint32_t r = (uint32_t)j + 1; r < len; ++r)
                col[r] = left ? A[(size_t)(k0 + j) * lda + k0 + r]         // column j, below the diagonal
                              : A[(size_t)(k0 + 1 + r) * lda + k0 + j];    // row j, right of the superdiagonal
        });
        metal_linalg::detail::compact_wy_t(len, kb, vc.data(), tau + k0, tc.data(), bb);
        float* V = static_cast<float*>(ws.V[slot].contents);
        float* T = static_cast<float*>(ws.T[slot].contents);
        metal_linalg::detail::parallel_for((len + 255) / 256, [&](size_t t) {
            for (uint32_t r = (uint32_t)t * 256; r < std::min<uint32_t>(len, (uint32_t)t * 256 + 256); ++r)
                for (uint32_t j = 0; j < bb; ++j) V[(size_t)r * bb + j] = j < kb ? vc[(size_t)j * len + r] : 0.0f;
        });
        for (uint32_t i = 0; i < bb; ++i)
            for (uint32_t j = 0; j < bb; ++j)
                T[(size_t)i * bb + j] = (i < kb && j < kb && j >= i) ? tc[(size_t)j * bb + i] : 0.0f;

        id<MTLCommandBuffer> cb = [c.rt.queue commandBufferWithUnretainedReferences];
        MPSMatrix* Vm = mps(ws.V[slot], 0, len, bb, bb);
        MPSMatrix* Tm = mps(ws.T[slot], 0, bb, bb, bb);
        if (left) {
            // U column-major (ld m) is U^T row-major: U^T(:, k0:) -= ((U^T(:, k0:) V) T^T) V^T.
            MPSMatrix* S = mps(Ubuf, (size_t)k0, n, len, m);
            MPSMatrix* Z = mps(ws.Z, 0, n, bb, bb);
            MPSMatrix* Z2 = mps(ws.Z2, 0, n, bb, bb);
            gemm(dev, cb, S, false, Vm, false, Z, n, bb, len, 1, 0);
            gemm(dev, cb, Z, false, Tm, true, Z2, n, bb, bb, 1, 0);
            gemm(dev, cb, Z2, false, Vm, true, S, n, len, bb, -1, 1);
        } else {
            // VT column-major is VT^T row-major: VT^T(k0+1:, :) -= V T (V^T VT^T(k0+1:, :)).
            MPSMatrix* S = mps(VTbuf, (size_t)(k0 + 1) * n, len, n, n);
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

// The two-stage reduction with singular vectors (`band` with vectors): A =
// Q1 Q2 B P2^T P1^T, Q1 and P1 the GPU stage's block reflectors (BandKeep),
// Q2 and P2 the bulge chase's. U = (Q1 Q2) U_B and V = (P1 P2) V_B, with
// Q = Q1 Q2 and P = P1 P2 formed explicitly on the GPU while the CPU chases
// the band and solves the bidiagonal problem:
//   1. the band reduction, keeping its reflectors; on the CPU meanwhile, the
//      aggregated T of every kAgg panels as their command buffers complete;
//   2. Q1 (m x n, thin) and P1 (n x n) explicit: the LAPACK tail's
//      reflectors on the CPU, then the aggregated blocks on the GPU, three
//      MPS products each, last first; the CPU meanwhile chases the band to
//      bidiagonal, keeping its reflectors;
//   3. Q <- Q Q2 and P <- P P2 on the GPU (bd_chase_apply, its blocks' V and
//      T built on the CPU); the CPU meanwhile solves B = U_B S V_B^T;
//   4. U = Q U_B and V = P V_B, two MPS products.
// Width 16 only (bd_chase_apply's blocks).
constexpr uint32_t kAgg = 8;   // panels per aggregated block reflector

// Buffers for one (m, n), only the latest kept. The band reduction writes its
// blocks' V and U straight into the aggregated layout (qagg, pagg: per
// aggregate, row-major len x kb, a panel's at its row and column offset, the
// zeros above it never written); the chase writes its reflectors straight into
// bd_chase_apply's blocks (lv, rv), each block's Y built beside its V.
struct BandVectors {
    uint32_t m = 0, n = 0, ldq = 0, ldp = 0, blocks = 0, aggs = 0;
    metal_linalg::detail::BandKeep keep;
    id<MTLBuffer> Q, P, qagg, pagg, qtagg, ptagg, ub, vb, Z, Z2, lv, rv;
    std::vector<size_t> qaoff, paoff;
    std::vector<float> ltau, rtau;   // the chase's taus, 16 a block
    size_t nblocks = 0;
    id<MTLComputePipelineState> apply_wide, apply_narrow;
};

// T of H_0 ... H_{p-1} = I - V T V^T for p block reflectors I - V_i T_i V_i^T
// of b columns each, V = [V_0 ... V_{p-1}] (len x p b, row-major), from the
// Gram matrix: T(0:ib, ib:(i+1)b) = -T(0:ib, 0:ib) G(0:ib, ib:(i+1)b) T_i.
// T_i row-major (ld 32) at Ts + i 1024; T row-major, ld kb.
void merge_t(uint32_t len, uint32_t p, uint32_t b, const float* V, const float* Ts, float* T) {
    const uint32_t kb = p * b;
    std::vector<float> G((size_t)kb * kb), X((size_t)kb * b), Y((size_t)kb * b);
    cblas_ssyrk(CblasRowMajor, CblasUpper, CblasTrans, (int)kb, (int)len, 1.0f, V, (int)kb, 0.0f, G.data(), (int)kb);
    std::fill(T, T + (size_t)kb * kb, 0.0f);
    for (uint32_t i = 0; i < p; ++i)
        for (uint32_t r = 0; r < b; ++r)
            for (uint32_t c = r; c < b; ++c) T[(size_t)(i * b + r) * kb + i * b + c] = Ts[(size_t)i * 1024 + r * 32 + c];
    for (uint32_t i = 1; i < p; ++i) {
        const uint32_t ib = i * b;
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int)ib, (int)b, (int)b, 1.0f, G.data() + ib, (int)kb,
                    Ts + (size_t)i * 1024, 32, 0.0f, X.data(), (int)b);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int)ib, (int)b, (int)ib, -1.0f, T, (int)kb, X.data(),
                    (int)b, 0.0f, Y.data(), (int)b);
        for (uint32_t r = 0; r < ib; ++r)
            for (uint32_t c = 0; c < b; ++c) T[(size_t)r * kb + ib + c] = Y[(size_t)r * b + c];
    }
}

BandVectors& band_vectors(Cache& c, uint32_t m, uint32_t n) {
    static BandVectors bv;
    if (bv.Q && bv.m == m && bv.n == n) return bv;
    constexpr uint32_t b = 16;
    bv = BandVectors{};
    bv.m = m;
    bv.n = n;
    bv.ldq = (m + 31) / 32 * 32;
    bv.ldp = (n + 31) / 32 * 32;
    const uint32_t blocks = metal_linalg::detail::band_blocks(n, b);
    bv.blocks = blocks;
    bv.aggs = (blocks + kAgg - 1) / kAgg;
    bv.qaoff.assign(bv.aggs + 1, 0);
    bv.paoff.assign(bv.aggs + 1, 0);
    auto& keep = bv.keep;
    keep.qoff.resize(blocks);
    keep.poff.resize(blocks);
    keep.qld.resize(blocks);
    keep.pld.resize(blocks);
    for (uint32_t a = 0; a < bv.aggs; ++a) {
        const uint32_t kb = std::min(kAgg, blocks - a * kAgg) * b, r0 = a * kAgg * b;
        bv.qaoff[a + 1] = bv.qaoff[a] + (size_t)(m - r0) * kb;
        bv.paoff[a + 1] = bv.paoff[a] + (size_t)(n - r0 - b) * kb;
        for (uint32_t t = 0; t < kb / b; ++t) {   // panel a kAgg + t at row and column t b
            const uint32_t k = a * kAgg + t;
            keep.qoff[k] = bv.qaoff[a] + (size_t)t * b * kb + t * b;
            keep.poff[k] = bv.paoff[a] + (size_t)t * b * kb + t * b;
            keep.qld[k] = keep.pld[k] = kb;
        }
    }
    auto shared = [&](size_t f) {
        return [c.rt.device newBufferWithLength:std::max<size_t>(f, 4) * 4 options:MTLResourceStorageModeShared];
    };
    auto priv = [&](size_t f) {
        return [c.rt.device newBufferWithLength:std::max<size_t>(f, 4) * 4 options:MTLResourceStorageModePrivate];
    };
    const size_t kmax = kAgg * b;
    bv.Q = shared((size_t)bv.ldq * n);
    bv.P = shared((size_t)bv.ldp * n);
    bv.qagg = shared(bv.qaoff[bv.aggs]);
    bv.pagg = shared(bv.paoff[bv.aggs]);
    bv.qtagg = shared((size_t)std::max(bv.aggs, 1u) * kmax * kmax);
    bv.ptagg = shared((size_t)std::max(bv.aggs, 1u) * kmax * kmax);
    keep.qv = bv.qagg;
    keep.pv = bv.pagg;
    keep.qt = shared((size_t)std::max(blocks, 1u) * 1024);
    keep.pt = shared((size_t)std::max(blocks, 1u) * 1024);
    bv.ub = shared((size_t)n * n);
    bv.vb = shared((size_t)n * n);
    bv.Z = priv((size_t)std::max(m, n) * kmax);
    bv.Z2 = priv((size_t)std::max(m, n) * kmax);
    const size_t pmax = n >= 2 ? (n - 2) / 16 : 0;
    bv.nblocks = (pmax + 1) * (pmax + 2) / 2;
    bv.ltau.resize(bv.nblocks * 16);
    bv.rtau.resize(bv.nblocks * 16);
    bv.lv = shared(bv.nblocks * metal_linalg::detail::kChaseBlockFloats);
    bv.rv = shared(bv.nblocks * metal_linalg::detail::kChaseBlockFloats);
    bv.apply_wide = make_pipeline(c.rt.device, c.rt.library, @"bd_chase_apply_4_4", nil);
    bv.apply_narrow = make_pipeline(c.rt.device, c.rt.library, @"bd_chase_apply_2_8", nil);
    return bv;
}

// The aggregated T of aggregate a, for Q (left) or P, its V written by the
// band reduction.
void build_aggregate(BandVectors& bv, uint32_t a, bool left) {
    constexpr uint32_t b = 16;
    const uint32_t k0 = a * kAgg, p = std::min(kAgg, bv.blocks - k0);
    const uint32_t r0 = k0 * b + (left ? 0 : b), len = (left ? bv.m : bv.n) - r0;
    const float* V = static_cast<const float*>((left ? bv.qagg : bv.pagg).contents) + (left ? bv.qaoff : bv.paoff)[a];
    const float* Ts = static_cast<const float*>((left ? bv.keep.qt : bv.keep.pt).contents) + (size_t)k0 * 1024;
    float* T = static_cast<float*>((left ? bv.qtagg : bv.ptagg).contents) + (size_t)a * (kAgg * b) * (kAgg * b);
    merge_t(len, p, b, V, Ts, T);
}

// Q = [I; 0] and P = I, where step 2 starts: on the CPU while the GPU
// reduces the matrix to a band.
void init_q_p(BandVectors& bv) {
    const uint32_t n = bv.n, ldq = bv.ldq, ldp = bv.ldp;
    float* Q = static_cast<float*>(bv.Q.contents);
    float* P = static_cast<float*>(bv.P.contents);
    metal_linalg::detail::parallel_for(n, [&](size_t j) {
        std::fill(Q + j * ldq, Q + (j + 1) * ldq, 0.0f);
        Q[j * ldq + j] = 1.0f;
        std::fill(P + j * ldp, P + (j + 1) * ldp, 0.0f);
        P[j * ldp + j] = 1.0f;
    });
}

// Step 2: Q = Q1 (thin) and P = P1 explicit from init_q_p's identities: the
// tail's reflectors on the CPU (tail_q1_p1), then the GPU's blocks, last
// first (queue_q1_p1).
void tail_q1_p1(BandVectors& bv, const float* A, uint32_t lda) {
    const uint32_t m = bv.m, n = bv.n, ldq = bv.ldq, ldp = bv.ldp;
    float* Q = static_cast<float*>(bv.Q.contents);
    float* P = static_cast<float*>(bv.P.contents);
    // The tail, last step first: Q(k:, k:) <- H Q(k:, k:), P(k+bk:, k+bk:) <- G P(k+bk:, k+bk:)
    std::vector<float> work((size_t)std::max(m, n) * 64 + 64);
    L lw = (L)work.size(), info = 0;
    for (auto it = bv.keep.steps.rbegin(); it != bv.keep.steps.rend(); ++it) {
        const auto& st = *it;
        if (st.nr > 0) {
            L N = st.nr, K = (L)st.tp.size(), LD = st.bk, LDP = ldp;
            sormlq_("L", "T", &N, &N, &K, const_cast<float*>(st.lq.data()), &LD, const_cast<float*>(st.tp.data()),
                    P + (size_t)(st.k + st.bk) * ldp + st.k + st.bk, &LDP, work.data(), &lw, &info);
        }
        L M = m - st.k, N = n - st.k, K = st.bk, LDA = lda, LDQ = ldq;
        sormqr_("L", "N", &M, &N, &K, const_cast<float*>(A) + (size_t)st.k * lda + st.k, &LDA, st.tq.data(),
                Q + (size_t)st.k * ldq + st.k, &LDQ, work.data(), &lw, &info);
    }
}

// The GPU's blocks, queued while the GPU still reduces the matrix (encoding
// them took 4 ms), to start once `ready` reaches 1: once the CPU has built
// the aggregates' T and applied the tail. Z = X22^T (row-major view),
// Z <- Z - ((Z V) T^T) V^T.
id<MTLCommandBuffer> queue_q1_p1(Cache& c, BandVectors& bv, id<MTLSharedEvent> ready) {
    constexpr uint32_t b = 16;
    const uint32_t m = bv.m, n = bv.n, ldq = bv.ldq, ldp = bv.ldp;
    id<MTLDevice> dev = c.rt.device;
    id<MTLCommandBuffer> cb = [c.rt.queue commandBuffer];
    [cb encodeWaitForEvent:ready value:1];
    const uint32_t kmax = kAgg * b;
    for (int a = (int)bv.aggs - 1; a >= 0; --a) {
        const uint32_t kb = std::min(kAgg, bv.blocks - (uint32_t)a * kAgg) * b;
        for (int side = 0; side < 2; ++side) {
            const bool left = side == 0;
            const uint32_t r0 = (uint32_t)a * kAgg * b + (left ? 0 : b), len = (left ? m : n) - r0, cols = n - r0;
            const uint32_t ld = left ? ldq : ldp;
            MPSMatrix* Zm = mps(left ? bv.Q : bv.P, (size_t)r0 * ld + r0, cols, len, ld);
            MPSMatrix* Vm = mps(left ? bv.qagg : bv.pagg, (left ? bv.qaoff : bv.paoff)[a], len, kb, kb);
            MPSMatrix* Tm = mps(left ? bv.qtagg : bv.ptagg, (size_t)a * kmax * kmax, kb, kb, kb);
            MPSMatrix* W = mps(bv.Z, 0, cols, kb, kb);
            MPSMatrix* W2 = mps(bv.Z2, 0, cols, kb, kb);
            gemm(dev, cb, Zm, false, Vm, false, W, cols, kb, len, 1, 0);
            gemm(dev, cb, W, false, Tm, true, W2, cols, kb, kb, 1, 0);
            gemm(dev, cb, W2, false, Vm, true, Zm, cols, len, kb, -1, 1);
        }
    }
    [cb commit];
    return cb;
}

// bd_chase_apply's blocks of groups G0 .. G1 - 1, their V written by the
// chase (V's six tiles that are not zero): per block (G, p), T from V and
// the taus (slarft's recurrence), and the kernel's Y = -T^T V^T (16 x 32)
// into the block's seven tiles after V's; a column with no reflector (length
// 1, or past the last sweep) set to zero with tau 0.
void chase_blocks(BandVectors& bv, bool left, long G0, long G1) {
    constexpr size_t kb = metal_linalg::detail::kChaseBlockFloats;
    const long n = bv.n, pmax = (n - 2) / 16;
    float* Bp = static_cast<float*>((left ? bv.lv : bv.rv).contents);
    float* taus = (left ? bv.ltau : bv.rtau).data();
    // V's tile (row, column) as a slot of the block; -1 for a zero tile.
    auto vslot = [](long rt, long ct) -> long { return ct == 0 ? (rt <= 2 ? rt : -1) : (rt >= 1 ? rt + 2 : -1); };
    metal_linalg::detail::parallel_for((size_t)(G1 - G0), [&](size_t Gs) {
        const long G = G0 + (long)Gs;
        for (long p = G; p <= pmax; ++p) {
            const long j = p - G, blk = G * (pmax + 1) - G * (G - 1) / 2 + j;
            float* B = Bp + blk * (long)kb;
            float* tau = taus + blk * 16;
            for (long c = 0; c < 16; ++c) {
                const long s = 16 * G + c, a = s + 1 + 16 * j, e = std::min(s + 16 * (j + 1), n - 1);
                if (s <= n - 2 && e - a + 1 >= 2) continue;
                for (long r = 8 * (c / 8); r < 8 * (c / 8) + 24; ++r) B[vslot(r / 8, c / 8) * 64 + (r % 8) * 8 + c % 8] = 0.0f;
                tau[c] = 0.0f;
            }
            // Y = -T^T V^T, with T slarft's forward T, row by row: Y_i =
            // -tau_i (V_i + sum_{m < i} (V_m . V_i) Y_m). Reflector c spans
            // rows c .. c + 15 of the block (Y_i rows up to i + 15).
            float Vt[16][32] = {}, Y[16][32] = {};   // a reflector a row
            for (long c = 0; c < 16; ++c)
                for (long r = c; r < c + 16; ++r) Vt[c][r] = B[vslot(r / 8, c / 8) * 64 + (r % 8) * 8 + c % 8];
            for (long i = 0; i < 16; ++i) {
                if (tau[i] == 0.0f) continue;
                float* y = Y[i];
                for (long r = i; r < i + 16; ++r) y[r] = Vt[i][r];
                for (long m = std::max(0L, i - 15); m < i; ++m) {
                    float z = 0.0f;
                    for (long r = i; r <= m + 15; ++r) z += Vt[m][r] * Vt[i][r];
                    for (long r = 0; r <= m + 15; ++r) y[r] += z * Y[m][r];
                }
                for (long r = 0; r < i + 16; ++r) y[r] *= -tau[i];
            }
            static constexpr long yt[7][2] = {{0, 0}, {0, 1}, {0, 2}, {1, 0}, {1, 1}, {1, 2}, {1, 3}};
            for (long t = 0; t < 7; ++t)
                for (long ii = 0; ii < 8; ++ii)
                    for (long rr = 0; rr < 8; ++rr) B[(6 + t) * 64 + ii * 8 + rr] = Y[8 * yt[t][0] + ii][8 * yt[t][1] + rr];
        }
    });
}

// Must match ChaseParams in Svd_Bidiag.metal.
struct ChaseParams { uint32_t n, rs, cs, pmax, pass0, pass1; };

// Groups (of 16 sweeps) a chunk of step 3's GPU work: a multiple of both
// kernels' passes (4 and 8 groups), about an eighth of the groups.
uint32_t chase_chunk_groups(uint32_t groups) { return std::max(32u, (groups / 2 + 7) / 8 * 8); }

// Step 3's GPU side: Q <- Q Q2, P <- P P2, a command buffer a chunk of
// groups, each waiting for `ready` to reach its index + 1 (the CPU signals
// as the chase finishes the chunk's groups and their T is built); queued.
// Returns the last.
id<MTLCommandBuffer> apply_chase(Cache& c, BandVectors& bv, id<MTLSharedEvent> ready) {
    const uint32_t n = bv.n;
    if (n < 3) {
        id<MTLCommandBuffer> cb = [c.rt.queue commandBuffer];
        [cb commit];
        return cb;
    }
    const uint32_t groups = (n - 2) / 16 + 1, chunk = chase_chunk_groups(groups);
    id<MTLCommandBuffer> cb = nil;
    for (uint32_t g0 = 0, k = 1; g0 < groups; g0 += chunk, ++k) {
        const uint32_t g1 = std::min(groups, g0 + chunk);
        cb = [c.rt.queue commandBuffer];
        [cb encodeWaitForEvent:ready value:k];
        // One after the other: run concurrently, the two took 177 ms at
        // 4096 against 132 (their strips competing for the caches).
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        for (int side = 0; side < 2; ++side) {
            const bool left = side == 0;
            const uint32_t ld = left ? bv.ldq : bv.ldp;   // X = M^T: ncols = ld (padded), rows n
            const bool wide = ld / 32 >= 64;
            id<MTLComputePipelineState> ps = wide ? bv.apply_wide : bv.apply_narrow;
            const uint32_t C = wide ? 32 : 16, K = wide ? 4 : 8;
            const ChaseParams q{n, ld, 1, (n - 2) / 16, g0 / K, (g1 + K - 1) / K};
            [enc setComputePipelineState:ps];
            [enc setBuffer:left ? bv.Q : bv.P offset:0 atIndex:0];
            [enc setBuffer:left ? bv.lv : bv.rv offset:0 atIndex:1];
            [enc setBytes:&q length:sizeof q atIndex:2];
            [enc dispatchThreadgroups:MTLSizeMake(ld / C, 1, 1) threadsPerThreadgroup:MTLSizeMake(32 * K, 1, 1)];
        }
        [enc endEncoding];
        [cb commit];
    }
    return cb;
}

// Both backends: `band` (> 0, singular values alone) the two-stage reduction's
// band width, 0 the one-stage reduction.
void bidiag_impl(const Matrices& a, float* u_out, float* s_out, float* vt_out, uint32_t* info_out, uint32_t band) {
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

    // Non-finite matrices give NaN at once; the rest are pipelined.
    std::vector<uint32_t> todo;
    for (uint32_t b = 0; b < batch; ++b) {
        if (finite[b]) { todo.push_back(b); continue; }
        std::fill(s_out + (size_t)b * K, s_out + (size_t)(b + 1) * K, NAN);
        if (u_out) std::fill(u_out + (size_t)b * M * K, u_out + (size_t)(b + 1) * M * K, NAN);
        if (vt_out) std::fill(vt_out + (size_t)b * K * N, vt_out + (size_t)(b + 1) * K * N, NAN);
        if (info_out) info_out[b] = 1u << 17;
    }
    if (todo.empty()) return;

    // Each matrix as a tall one (row-major l x K: A itself, or A^T if wide);
    // much taller than wide, R of a QR first, and U = Q U_R at the end.
    const uint32_t l = std::max(M, N);
    const bool wide = M < N;
    const bool qr_first = l >= 2 * K && K >= kQrFirstMinK;
    const uint32_t rows = qr_first ? K : l;   // of the matrix bidiagonalized
    Workspace& ws = cache.workspace(rows, K, vectors);
    if (todo.size() > 1) cache.second_slot(ws, vectors);


    struct Slot {
        std::vector<float> t, q, d, e, tq, tp, ab;
        float scale = 1.0f;
        id<MTLCommandBuffer> pending = nil;   // two-stage with vectors: the GPU's Q and P
        bool chased = false;                  // singular values alone: the chase done under the reduction
    };
    bool chase_first = todo.size() == 1;   // see reduce
    // Two-stage with vectors: one matrix at a time, in one slot.
    BandVectors* bvp = band && vectors ? &band_vectors(cache, rows, K) : nullptr;

    Slot slots[2];
    for (Slot& sl : slots) {
        if (qr_first) sl.t.resize((size_t)l * K);
        if (qr_first && vectors) sl.q.resize((size_t)l * K);
        sl.d.resize(K); sl.e.resize(K); sl.tq.resize(K); sl.tp.resize(K);
    }

    // Step 1 for matrix b in slot s: scale, the QR if much taller than wide
    // (of A^T if wide), then the bidiagonalization into A[s] (column-major
    // rows x K: A, or A^T if wide, which is A's row-major storage).
    auto reduce = [&](uint32_t b, int s) {
        Slot& sl = slots[s];
        int ex = 0;
        if (amax[b] > 0.0f) std::frexp(amax[b], &ex);
        sl.scale = std::ldexp(1.0f, -ex);
        const float* src = a.data + (size_t)b * M * N;
        float* A = static_cast<float*>(ws.A[s].contents);
        if (qr_first) {
            float* t = sl.t.data();   // row-major l x K, scaled
            if (wide) transpose_scaled(src, N, t, K, M, N, sl.scale);
            else      for (size_t i = 0; i < (size_t)M * N; ++i) t[i] = src[i] * sl.scale;
            std::vector<float> r((size_t)K * K), qtmp(vectors ? 0 : (size_t)l * K);
            core::qr(Matrices{t, 1, l, K}, vectors ? sl.q.data() : qtmp.data(), r.data());
            transpose_scaled(r.data(), K, A, ws.lda, K, K, 1.0f);
        } else if (wide) {
            for (uint32_t j = 0; j < K; ++j) {
                const float* row = src + (size_t)j * N;
                float* col = A + (size_t)j * ws.lda;
                for (uint32_t i = 0; i < N; ++i) col[i] = row[i] * sl.scale;
            }
        } else {
            transpose_scaled(src, N, A, ws.lda, M, N, sl.scale);
        }
        if (!band) {
            bidiagonalize(cache, ws, ws.A[s], sl.d.data(), sl.e.data(), sl.tq.data(), sl.tp.data());
            return;
        }
        metal_linalg::detail::BandKeep* keep = bvp ? &bvp->keep : nullptr;
        // Q1 and P1's GPU work, queued during the reduction, released once
        // the CPU's part is done; released whatever happens, since a command
        // buffer left waiting would hold the queue.
        struct Release {
            id<MTLSharedEvent> event;
            ~Release() { if (event) event.signaledValue = 1; }
        } q1_ready{bvp ? [cache.rt.device newSharedEvent] : nil};
        if (keep)   // Q and P's start, their GPU work queued, the aggregates' T as the blocks complete
            keep->while_gpu = [&](metal_linalg::detail::BandWatch& kp) {
                init_q_p(*bvp);
                sl.pending = queue_q1_p1(cache, *bvp, q1_ready.event);
                for (uint32_t ag = 0; ag < bvp->aggs; ++ag) {
                    [kp.done[std::min((ag + 1) * kAgg, bvp->blocks) - 1] waitUntilCompleted];
                    build_aggregate(*bvp, ag, true);
                    build_aggregate(*bvp, ag, false);
                }
            };
        // The band, with room for the bulges of band_to_bidiagonal: B(i, j) at
        // ab[j * ld + ku + i - j], ku = 2 band above the diagonal, band below.
        const size_t ld = 3 * (size_t)band + 1, ku = 2 * (size_t)band;
        auto rows = [&](uint32_t r0, uint32_t r1) {   // the band's rows r0 .. r1 - 1
            for (uint32_t i = r0; i < std::min(r1, K); ++i)
                for (uint32_t j = i; j < K && j <= i + band; ++j)
                    sl.ab[(size_t)j * ld + ku + i - j] = A[(size_t)j * ws.lda + i];
        };
        // Singular values alone, the call's first matrix (whose CPU is idle
        // meanwhile): the chase on threads of its own, trailing the GPU down
        // the band as it finishes each block's rows.
        sl.chased = chase_first && !vectors && metal_linalg::detail::band_fit(ws.m, band) == band;
        chase_first = false;
        sl.ab.assign(ld * K, 0.0f);
        std::atomic<long> ready{0};
        metal_linalg::detail::ChaseReflectors trail;
        trail.ready_rows = &ready;
        std::exception_ptr failed;
        std::thread chaser;
        if (sl.chased)
            chaser = std::thread([&] {
                try {
                    metal_linalg::detail::band_to_bidiagonal(K, band, sl.ab.data(), ld, ku, sl.d.data(), sl.e.data(),
                                                             metal_linalg::detail::cpu_threads_beside_gpu(), &trail);
                } catch (...) {
                    failed = std::current_exception();
                }
            });
        // Released and joined whatever happens (on garbage, if the GPU failed).
        struct Join {
            std::thread& thread;
            std::atomic<long>& ready;
            long n;
            ~Join() {
                ready.store(n, std::memory_order_release);
                if (thread.joinable()) thread.join();
            }
        } join{chaser, ready, (long)K};
        metal_linalg::detail::BandWatch watch;
        if (sl.chased)
            watch.while_gpu = [&](metal_linalg::detail::BandWatch& w) {
                for (size_t k = 0; k < w.done.size(); ++k) {
                    [w.done[k] waitUntilCompleted];
                    rows((uint32_t)k * band, (uint32_t)(k + 1) * band);
                    ready.store((long)(k + 1) * band, std::memory_order_release);
                }
            };
        const bool banded = metal_linalg::detail::band_reduce_general(ws.A[s], ws.m, ws.n, ws.lda, band, keep,
                                                                      sl.chased ? &watch : nullptr);
        if (keep) keep->while_gpu = nullptr;   // it refers to this call's locals
        if (!banded) {
            sl.ab.clear();   // too tall for the band reduction's panels: one stage
            bidiagonalize(cache, ws, ws.A[s], sl.d.data(), sl.e.data(), sl.tq.data(), sl.tp.data());
            return;
        }
        rows(sl.chased ? (uint32_t)watch.done.size() * band : 0, K);   // (the last rows, LAPACK's)
        ready.store(K, std::memory_order_release);
        if (chaser.joinable()) chaser.join();
        if (failed) std::rethrow_exception(failed);
        if (bvp) {
            tail_q1_p1(*bvp, A, ws.lda);
            q1_ready.event.signaledValue = 1;
        }
    };

    // Step 2 for matrix b in slot s, on the CPU: the singular values (into
    // s_out) and the bidiagonal problem's vectors.
    // Singular values alone: the band to bidiagonal (band_to_bidiagonal, on
    // the CPU's cores but two, which the GPU's host work keeps) if two-stage,
    // then sbdsqr without vectors, which is dqds (slasq1): faster than sbdsdc,
    // and accurate to the bidiagonal's every singular value, however small.
    auto solve = [&](uint32_t b, int s) {
        Slot& sl = slots[s];
        L n = K, info = 0, zero = 0, one = 1;
        float qd = 0;
        const char* routine = vectors ? "sbdsdc" : "sbdsqr";
        if (vectors && !sl.ab.empty()) {
            // Two stages: the chase on threads of its own, keeping its
            // reflectors; as it finishes each chunk of groups their T is
            // built here and that chunk of Q <- Q Q2 and P <- P P2, queued
            // on the GPU behind Q1 and P1, released; then the divide and
            // conquer while the GPU finishes.
            BandVectors& bv = *bvp;
            std::atomic<long> frontier{0};
            std::atomic<bool> chased{false};
            const metal_linalg::detail::ChaseReflectors rec{
                static_cast<float*>(bv.lv.contents), bv.ltau.data(), static_cast<float*>(bv.rv.contents),
                bv.rtau.data(), K >= 2 ? (K - 2) / 16 : 0, &frontier};
            id<MTLSharedEvent> ready = [cache.rt.device newSharedEvent];
            sl.pending = apply_chase(cache, bv, ready);
            // Released whatever happens: a command buffer left waiting would hold the queue.
            struct Release {
                id<MTLSharedEvent> event;
                ~Release() { event.signaledValue = 1ull << 40; }
            } release{ready};
            const long groups = K >= 3 ? (long)(K - 2) / 16 + 1 : 0, chunk = chase_chunk_groups((uint32_t)groups);
            std::exception_ptr failed;
            std::thread chase([&] {
                try {
                    metal_linalg::detail::band_to_bidiagonal(K, band, sl.ab.data(), 3 * (size_t)band + 1,
                                                             2 * (size_t)band, sl.d.data(), sl.e.data(),
                                                             metal_linalg::detail::cpu_threads_beside_gpu(), &rec);
                } catch (...) {
                    failed = std::current_exception();
                }
                chased.store(true, std::memory_order_release);
            });
            long built = 0;          // groups whose T is built
            uint64_t released = 0;   // chunks released to the GPU
            for (;;) {
                const bool over = chased.load(std::memory_order_acquire);
                // Group G is chased once sweeps 16 G .. min(16 G + 15, K - 2) are.
                const long f = frontier.load(std::memory_order_acquire);
                const long g = over || f >= (long)K - 1 ? groups : std::min(groups, f / 16);
                if (g > built) {
                    chase_blocks(bv, true, built, g);
                    chase_blocks(bv, false, built, g);
                    built = g;
                    const uint64_t can = built >= groups ? (uint64_t)((groups + chunk - 1) / chunk)
                                                         : (uint64_t)(built / chunk);
                    if (can > released) ready.signaledValue = released = can;
                } else if (over) {
                    break;
                } else {
                    std::this_thread::sleep_for(std::chrono::microseconds(200));
                }
            }
            chase.join();
            if (failed) std::rethrow_exception(failed);
            // Its largest products on the GPU once Q2 and P2 are done (the
            // top merge's, at 4096: 13 ms there against about 40 here)
            metal_linalg::detail::MpsGemm gpu(cache.rt.device, cache.rt.queue, sl.pending);
            gpu.add_buffer(bv.ub);
            gpu.add_buffer(bv.vb);
            info = (L)metal_linalg::detail::bidiagonal_svd(K, sl.d.data(), sl.e.data(),
                                                           static_cast<float*>(bv.ub.contents), K,
                                                           static_cast<float*>(bv.vb.contents), K,
                                                           metal_linalg::detail::cpu_threads_beside_gpu(), &gpu);
        } else if (vectors) {
            // sbdsdc's divide and conquer on the CPU's cores but the two the
            // GPU's host work keeps (divide_conquer.cpp), into U's first K
            // rows and VT; for one matrix, whose GPU is idle meanwhile, its
            // largest products on the GPU
            std::unique_ptr<metal_linalg::detail::MpsGemm> gpu;
            if (todo.size() == 1) {
                gpu = std::make_unique<metal_linalg::detail::MpsGemm>(cache.rt.device, cache.rt.queue);
                gpu->add_buffer(ws.U[s]);
                gpu->add_buffer(ws.VT[s]);
            }
            info = (L)metal_linalg::detail::bidiagonal_svd(K, sl.d.data(), sl.e.data(),
                                                           static_cast<float*>(ws.U[s].contents), rows,
                                                           static_cast<float*>(ws.VT[s].contents), K,
                                                           metal_linalg::detail::cpu_threads_beside_gpu(), gpu.get());
        } else {
            std::vector<float> work(4 * (size_t)K + 16);
            if (!sl.ab.empty() && !sl.chased)
                metal_linalg::detail::band_to_bidiagonal(K, band, sl.ab.data(), 3 * (size_t)band + 1,
                                                         2 * (size_t)band, sl.d.data(), sl.e.data(),
                                                         metal_linalg::detail::cpu_threads_beside_gpu());
            // By bisection on the GPU where it is the faster, else sbdsqr.
            std::vector<float> sv(K);
            if (metal_linalg::detail::bidiagonal_singular_values(K, sl.d.data(), sl.e.data(), sv.data()))
                sl.d = std::move(sv);
            else
                sbdsqr_("U", &n, &zero, &zero, &zero, sl.d.data(), sl.e.data(), &qd, &one, &qd, &one, &qd, &one,
                    work.data(), &info);
        }
        if (info != 0) {
            throw std::runtime_error(std::string("[svd] bidiag: LAPACK ") + routine + " failed on matrix " +
                                     std::to_string(b) + ", info " + std::to_string((long long)info));
        }
        float* sv = s_out + (size_t)b * K;
        for (uint32_t i = 0; i < K; ++i) sv[i] = sl.d[i] / sl.scale;   // descending
    };

    // Step 3 for matrix b in slot s, and the output.
    std::vector<float> ut(vectors && (!bvp || qr_first) ? (size_t)l * K : 0), vtt(vectors && !bvp ? (size_t)K * K : 0);
    std::vector<float> urt(qr_first && vectors && !bvp ? (size_t)K * K : 0);
    auto finish = [&](uint32_t b, int s) {
        Slot& sl = slots[s];
        float* u = vectors && u_out ? u_out + (size_t)b * M * K : nullptr;
        float* vt = vectors && vt_out ? vt_out + (size_t)b * K * N : nullptr;
        if (vectors && !sl.ab.empty()) {
            // Two stages: U = Q U_B and V^T = V_B^T P^T on the GPU, row-major
            // into U and VT (on the row-major views, Q^T and U_B^T, V_B and
            // P^T); U copied out while the GPU forms V^T.
            [sl.pending waitUntilCompleted];
            check(sl.pending);
            sl.pending = nil;
            id<MTLDevice> dev = cache.rt.device;
            id<MTLCommandBuffer> cu = [cache.rt.queue commandBuffer], cv = [cache.rt.queue commandBuffer];
            gemm(dev, cu, mps(bvp->Q, 0, K, rows, bvp->ldq), true, mps(bvp->ub, 0, K, K, K), true,
                 mps(ws.U[s], 0, rows, K, K), rows, K, K, 1, 0);
            gemm(dev, cv, mps(bvp->vb, 0, K, K, K), true, mps(bvp->P, 0, K, K, bvp->ldp), false,
                 mps(ws.VT[s], 0, K, K, K), K, K, K, 1, 0);
            [cu commit];
            [cv commit];
            const float* Ur = static_cast<const float*>(ws.U[s].contents);    // rows x K
            const float* Vr = static_cast<const float*>(ws.VT[s].contents);   // K x K
            [cu waitUntilCompleted];
            check(cu);
            if (qr_first && (wide ? vt : u)) {   // U = Q U_R
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, l, K, K, 1.0f, sl.q.data(), K, Ur, K, 0.0f,
                            ut.data(), K);
                Ur = ut.data();
            }
            if (!wide && u) std::memcpy(u, Ur, (size_t)l * K * 4);
            if (wide && vt) vDSP_mtrans(Ur, 1, vt, 1, K, l);   // Vt = U'^T
            [cv waitUntilCompleted];
            check(cv);
            if (!wide && vt) std::memcpy(vt, Vr, (size_t)K * K * 4);
            if (wide && u) vDSP_mtrans(Vr, 1, u, 1, K, K);     // U = V' = (Vt')^T
        } else if (vectors) {
            if (vtt.empty()) {   // a two-stage call that fell back to one stage
                ut.resize((size_t)l * K);
                vtt.resize((size_t)K * K);
                if (qr_first) urt.resize((size_t)K * K);
            }
            float* U = static_cast<float*>(ws.U[s].contents);   // U_B in its first K rows, VT V_B^T
            if (rows > K)
                for (uint32_t j = 0; j < K; ++j) std::fill(U + (size_t)j * rows + K, U + (size_t)(j + 1) * rows, 0.0f);
            back_transform(cache, ws, ws.A[s], ws.U[s], ws.VT[s], sl.tq.data(), true);
            back_transform(cache, ws, ws.A[s], ws.U[s], ws.VT[s], sl.tp.data(), false);
            vDSP_mtrans(static_cast<const float*>(ws.VT[s].contents), 1, vtt.data(), 1, K, K);  // row-major K x K
            if (qr_first) {
                vDSP_mtrans(U, 1, urt.data(), 1, K, K);                                   // U_R, row-major
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, l, K, K, 1.0f, sl.q.data(), K,
                            urt.data(), K, 0.0f, ut.data(), K);                             // U = Q U_R
            } else {
                vDSP_mtrans(U, 1, ut.data(), 1, l, K);                                      // row-major l x K
            }
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
    };

    if (bvp) {   // two-stage with vectors: overlapped within each matrix instead
        for (uint32_t b : todo) {
            try {
                reduce(b, 0);
                solve(b, 0);
                finish(b, 0);
            } catch (...) {
                if (slots[0].pending) [slots[0].pending waitUntilCompleted];
                throw;
            }
        }
        return;
    }

    // The pipeline: matrix t in slot t % 2. The CPU solves t while the GPU
    // reduces t + 1, and solves t + 1 while the GPU back-transforms t.
    const size_t count = todo.size();
    reduce(todo[0], 0);
    std::future<void> solving = std::async(std::launch::async, solve, todo[0], 0);
    for (size_t t = 0; t < count; ++t) {
        const int s = (int)(t % 2);
        if (t + 1 < count) {
            try {
                reduce(todo[t + 1], s ^ 1);
            } catch (...) {
                solving.wait();
                throw;
            }
        }
        solving.get();
        if (t + 1 < count) solving = std::async(std::launch::async, solve, todo[t + 1], s ^ 1);
        try {
            finish(todo[t], s);
        } catch (...) {
            if (solving.valid()) solving.wait();
            throw;
        }
    }
}

} // namespace

namespace core::detail {

void svd_bidiag(const Matrices& a, float* u, float* s, float* vt, uint32_t* info) {
    bidiag_impl(a, u, s, vt, info, 0);
}

void svd_band_vectors(const Matrices& a, float* u, float* s, float* vt, uint32_t* info) {
    // Width 16 (bd_chase_apply's blocks), or if the matrix is too tall for
    // the panel kernels at 16, the one-stage reduction.
    const uint32_t M = a.rows, N = a.cols, K = std::min(M, N), l = std::max(M, N);
    const uint32_t rows = l >= 2 * K && K >= kQrFirstMinK ? K : l;
    bidiag_impl(a, u, s, vt, info, metal_linalg::detail::band_fit(rows, 16) == 16 ? 16u : 0u);
}

void svd_band(const Matrices& a, float* s, uint32_t* info, uint32_t width) {
    // The band's width: as asked, narrower if the matrix is too tall for the
    // panel kernels at that width (rows b <= 128 * 1024), and if even the
    // narrowest is, the one-stage reduction (band 0).
    const uint32_t M = a.rows, N = a.cols, K = std::min(M, N), l = std::max(M, N);
    const uint32_t rows = l >= 2 * K && K >= kQrFirstMinK ? K : l;
    bidiag_impl(a, nullptr, s, nullptr, info,
                metal_linalg::detail::band_fit(rows, metal_linalg::detail::band_width(width, "SVD_BAND_WIDTH")));
}

} // namespace core::detail
} // namespace metal_linalg
