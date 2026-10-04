// The SVD's `bidiag` backend: LAPACK's method with its two expensive steps on
// the GPU; and the `band` backend, singular values alone in two stages
// (below, band_reduce).
//
//   1. Bidiagonalize, A = Q B P^T, entirely on the GPU (shaders/Svd_Bidiag.metal):
//      blocked sgebrd (upper bidiagonal), per column slabrd's steps as four
//      kernels, per panel the trailing update as two MPS GEMMs; the panels'
//      command buffers are queued and the host waits once per matrix. The last
//      few columns, fewer than a panel, are reduced by LAPACK.
//   2. B = U_B diag(S) V_B^T on the CPU: LAPACK sbdsdc; the singular values
//      alone by sbdsqr (dqds), faster and accurate to the smallest.
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
// docs/studies/two-stage-apple-m5-pro.md.

#ifndef ACCELERATE_NEW_LAPACK
#define ACCELERATE_NEW_LAPACK
#endif
#include <Accelerate/Accelerate.h>

#include <metal_linalg/core.h>
#include <metal_linalg/device.h>
#include "band_chase.h"
#include "metal_runtime.h"
#include "shaders.h"

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <future>
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
// block's V and T are built on the CPU (slarft) in one of two slots while the
// GPU applies the other.
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
        std::vector<float> t, q, d, e, tq, tp, ub, vb, ab;
        float scale = 1.0f;
    };
    Slot slots[2];
    for (Slot& sl : slots) {
        if (qr_first) sl.t.resize((size_t)l * K);
        if (qr_first && vectors) sl.q.resize((size_t)l * K);
        sl.d.resize(K); sl.e.resize(K); sl.tq.resize(K); sl.tp.resize(K);
        if (vectors) { sl.ub.resize((size_t)K * K); sl.vb.resize((size_t)K * K); }
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
        if (!metal_linalg::detail::band_reduce_general(ws.A[s], ws.m, ws.n, ws.lda, band)) {
            sl.ab.clear();   // too tall for the band reduction's panels: one stage
            bidiagonalize(cache, ws, ws.A[s], sl.d.data(), sl.e.data(), sl.tq.data(), sl.tp.data());
            return;
        }
        // The band, with room for the bulges of band_to_bidiagonal: B(i, j) at
        // ab[j * ld + ku + i - j], ku = 2 band above the diagonal, band below.
        const size_t ld = 3 * (size_t)band + 1, ku = 2 * (size_t)band;
        sl.ab.assign(ld * K, 0.0f);
        for (uint32_t j = 0; j < K; ++j)
            for (uint32_t i = j > band ? j - band : 0; i <= j; ++i)
                sl.ab[(size_t)j * ld + ku + i - j] = A[(size_t)j * ws.lda + i];
    };

    // Step 2 for matrix b in slot s, on the CPU: the singular values (into
    // s_out) and the bidiagonal problem's vectors.
    // Singular values alone: the band to bidiagonal (band_to_bidiagonal, on
    // the CPU's cores but two, which the GPU's host work keeps) if two-stage,
    // then sbdsqr without vectors, which is dqds (slasq1): faster than sbdsdc,
    // and accurate to the bidiagonal's every singular value, however small.
    auto solve = [&](uint32_t b, int s) {
        Slot& sl = slots[s];
        L n = K, ld = K, info = 0, iq = 0, zero = 0, one = 1;
        float qd = 0;
        const char* routine = vectors ? "sbdsdc" : "sbdsqr";
        if (vectors) {
            char uplo = 'U', compq = 'I';
            std::vector<float> work(3 * (size_t)K * K + 4 * (size_t)K + 8 * (size_t)K + 16);
            std::vector<L> iwork(8 * (size_t)K + 8);
            sbdsdc_(&uplo, &compq, &n, sl.d.data(), sl.e.data(), sl.ub.data(), &ld, sl.vb.data(), &ld, &qd, &iq,
                    work.data(), iwork.data(), &info);
        } else {
            std::vector<float> work(4 * (size_t)K + 16);
            if (!sl.ab.empty())
                metal_linalg::detail::band_to_bidiagonal(K, band, sl.ab.data(), 3 * (size_t)band + 1,
                                                         2 * (size_t)band, sl.d.data(), sl.e.data(),
                                                         std::max(1u, cpu_threads() - 2));
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
    std::vector<float> ut(vectors ? (size_t)l * K : 0), vtt(vectors ? (size_t)K * K : 0);
    std::vector<float> urt(qr_first && vectors ? (size_t)K * K : 0);
    auto finish = [&](uint32_t b, int s) {
        Slot& sl = slots[s];
        if (vectors) {
            float* U = static_cast<float*>(ws.U[s].contents);
            std::fill(U, U + (size_t)rows * K, 0.0f);
            for (uint32_t j = 0; j < K; ++j)
                std::copy(sl.ub.begin() + (size_t)j * K, sl.ub.begin() + (size_t)(j + 1) * K, U + (size_t)j * rows);
            std::copy(sl.vb.begin(), sl.vb.end(), static_cast<float*>(ws.VT[s].contents));
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
            float* u = u_out ? u_out + (size_t)b * M * K : nullptr;
            float* vt = vt_out ? vt_out + (size_t)b * K * N : nullptr;
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
