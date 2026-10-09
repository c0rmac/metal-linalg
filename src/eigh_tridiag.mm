// The `tridiag` eigensolver backend: LAPACK's ssyevd with its two expensive
// steps on the GPU.
//
//   1. Tridiagonalize, A = Q T Q^T, entirely on the GPU (shaders/Eigh_Tridiag.metal):
//      blocked ssytrd (lower), per column slatrd's steps as three kernels, per
//      panel the rank-2nb trailing update as an MPS GEMM. The panels' command
//      buffers are queued back to back and the host waits once per matrix, so
//      no GPU round trip is paid per column. The last few columns, fewer than a
//      panel, are reduced by LAPACK on the CPU.
//   2. T = Z diag(w) Z^T on the CPU: sstedc's divide and conquer on every core
//      but two (divide_conquer.cpp) for eigenvectors; for eigenvalues alone,
//      bisection on the GPU (bisect.mm) or ssterf.
//   3. V = Q Z on the GPU: ssytrd's reflectors applied kBackBlock at a time as
//      blocked Householder transformations, three MPS GEMMs each.
//
// Why: for one large matrix the Jacobi backends lose to the CPU, and LAPACK's
// ssyevd spends most of its time in step 1, half of it a symmetric
// matrix-vector product per column, bound by memory bandwidth, and in the
// back-transformation, which is matrix products. On an M5 Pro, one N x N with
// eigenvectors: 1.6x the CPU's speed at N = 1024, 2.5x at 2048, 5.4x at 4096,
// 6.6x at 8192. Below N ~ 1000 the per-panel work and the launches cost more
// than they save, which is what the routing policy's tridiag_min_n is measured
// for.
//
// The batched form (eigh_tridiag_batch, the tridiag_batch backend, since
// 2.17.0) reduces a whole batch of mid-size matrices by the same dispatches
// instead (td_panel, a threadgroup a matrix and panel), and pipelines chunks
// of the batch over the three steps; see the comment above BatchWorkspace.
//
// A batch is pipelined over two workspace slots: while the CPU solves one
// matrix's tridiagonal problem (step 2), the GPU reduces the next (step 1),
// and while the GPU back-transforms one (step 3), the CPU solves the next. On
// an M5 Pro, per matrix of 2048 x 2048 with eigenvectors: 90 ms alone, 55 ms
// in a batch of 4, 50 ms in a batch of 8. Each matrix is scaled by a power of
// two first (exact), so magnitudes that would over- or underflow a float32
// product work as on the CPU.

#ifndef ACCELERATE_NEW_LAPACK
#define ACCELERATE_NEW_LAPACK   // LAPACK's current interface; before any Accelerate header
#endif
#include <Accelerate/Accelerate.h>

#include <metal_linalg/core.h>
#include "divide_conquer.h"
#include "metal_runtime.h"
#include "shaders.h"

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <future>
#include <map>
#include <memory>
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

constexpr uint32_t kPanel     = 32;    // columns per panel of the reduction (nb)
constexpr uint32_t kBackBlock = 128;   // reflectors per pass of the back-transformation
constexpr uint32_t kBatchBackBlock = 64;   // the same in the batched form; at most WY_MAX in the shader
constexpr uint32_t kTile      = 64;    // must match TILE in Eigh_Tridiag.metal
constexpr uint32_t kGroup     = 256;   // must match GROUP in Eigh_Tridiag.metal
constexpr uint32_t kRows      = 32;    // must match GROUP / LANES: td_update's and td_apply's rows per threadgroup

// Must match TdParams and PackParams in Eigh_Tridiag.metal.
struct TdParams   { uint32_t nn, lda, ldw, i, k, ng, ngp, tiles, sa, sw, sp, sr, sv, stm; };
struct PackParams { uint32_t m, nb, lda, ldw, ldb, k, ngp, sa, sw, sb, sv, sr; };

uint32_t groups(uint32_t rows) { return (rows + kRows - 1) / kRows; }

using L = __LAPACK_int;

struct Pipelines {
    id<MTLComputePipelineState> update, symv, apply, pack, restore, panel, make_v, make_t, load, store;
};

// Must match LoadParams and StoreParams in Eigh_Tridiag.metal.
struct LoadParams  { uint32_t n, lda, sa, lower, c0; };
struct StoreParams { uint32_t n, c0; };

// Must match WyParams in Eigh_Tridiag.metal.
struct WyParams { uint32_t m, kb, bb, k0, lda, sa, sv, svb, stb; };

// Must match PanelParams in Eigh_Tridiag.metal.
struct PanelParams { uint32_t nn, lda, ldw, ldb, k, nb, sa, sw, sb, sv; };
constexpr uint32_t kPanelMaxN = 1024;   // must match PANEL_MAX_N
// The batched form with eigenvectors in two stages instead (eigh_band_batch,
// svd_bidiag_batch.mm) from kBandMinN for batches of up to half as many
// matrices as the CPU's solve has threads, as many from kBandFullMinN and
// twice as many from kBandDoubleMinN: its band reduction is matrix products,
// but each matrix's band chase is the CPU's, which bounds larger batches. On
// an M5 Pro (16 solve threads) against the one-stage reduction: 1.4-2.4x at
// 64-96 for 2-8 matrices, 1.14-1.5x at 128-256 for 1-8 (level at 16), 1.23-1.25x
// at 384-512 for 8 (level at 16), 1.07x for 12 of 768 (0.90x for 32), 1.35x
// for 24 of 896, 1.48x for 24 of 1024 and 1.29x for 32; one 1024 x 1024 2.9x
// (17.4 ms against 49.3).
constexpr uint32_t kBandMinN = 64;
constexpr uint32_t kBandFullMinN = 640;
constexpr uint32_t kBandDoubleMinN = 896;

// Buffers for one N, reused across the matrices of a batch and across calls.
// A and Z come in two pipeline slots (the second allocated for a batch).
struct Workspace {
    uint32_t      lda = 0;
    id<MTLBuffer> A[2], W, B, C, P, tmp, d, e, tau;   // the reduction
    id<MTLBuffer> red;                                 // reflector scale, norm and dot partials
    size_t        npart_off = 0, dpart_off = 0;        // in red, bytes
    id<MTLBuffer> Z[2];                                // eigenvectors, column-major (n x n)
    id<MTLBuffer> V[2], T[2], Y, Y2;                   // the back-transformation, two slots
};

struct Cache {
    MetalRuntime& rt = MetalRuntime::shared(METAL_LINALG_SHADER(Eigh_Tridiag), "eigh_tridiag");
    Pipelines p{};
    bool have_pipelines = false;
    std::map<std::pair<uint32_t, bool>, Workspace> workspaces;   // (n, vectors): at most one

    const Pipelines& pipelines() {
        if (!have_pipelines) {
            auto make = [&](NSString* name) { return make_pipeline(rt.device, rt.library, name, nil); };
            p.update  = make(@"td_update");
            p.symv    = make(@"td_symv");
            p.apply   = make(@"td_apply");
            p.pack    = make(@"td_pack");
            p.restore = make(@"td_restore_e");
            p.panel   = make(@"td_panel");
            p.make_v  = make(@"td_make_v");
            p.make_t  = make(@"td_make_t");
            p.load    = make(@"td_load");
            p.store   = make(@"td_store");
            for (id<MTLComputePipelineState> ps : {p.update, p.symv, p.apply}) {
                if (ps.maxTotalThreadsPerThreadgroup < kGroup) {
                    throw std::runtime_error("[eigh] tridiag: a pipeline allows fewer than 256 threads per threadgroup.");
                }
            }
            have_pipelines = true;
        }
        return p;
    }

    Workspace& workspace(uint32_t n, bool vectors) {
        const auto key = std::make_pair(n, vectors);
        if (auto it = workspaces.find(key); it != workspaces.end()) return it->second;
        // A workspace holds about two N x N matrices (256 MB each at N = 8192),
        // so only the latest is kept: a program solving many sizes would
        // otherwise accumulate one per size for its whole life.
        workspaces.clear();
        auto shared = [&](size_t floats) {
            return [rt.device newBufferWithLength:std::max<size_t>(floats, 4) * sizeof(float)
                                          options:MTLResourceStorageModeShared];
        };
        auto priv = [&](size_t floats) {
            return [rt.device newBufferWithLength:std::max<size_t>(floats, 4) * sizeof(float)
                                          options:MTLResourceStorageModePrivate];
        };
        Workspace w;
        w.lda = (n + 7) / 8 * 8;
        w.A[0] = shared((size_t)w.lda * n);
        w.W   = priv((size_t)n * kPanel);
        w.B   = priv((size_t)n * 2 * kPanel);
        w.C   = priv((size_t)n * 2 * kPanel);
        w.P   = priv((size_t)((n + kTile - 1) / kTile) * n);
        w.tmp = priv(2 * kPanel);
        const size_t g = groups(n) + 1;
        w.red = priv(4 + 3 * g);
        w.npart_off = 4 * sizeof(float);
        w.dpart_off = (4 + 2 * g) * sizeof(float);
        w.d   = shared(n);
        w.e   = shared(n);
        w.tau = shared(n);
        if (vectors) {
            w.Z[0] = shared((size_t)n * n);
            for (int s = 0; s < 2; ++s) {
                w.V[s] = shared((size_t)n * kBackBlock);
                w.T[s] = shared((size_t)kBackBlock * kBackBlock);
            }
            w.Y  = priv((size_t)n * kBackBlock);
            w.Y2 = priv((size_t)n * kBackBlock);
        }
        return workspaces[key] = w;
    }

    // The second pipeline slot, for a batch.
    void second_slot(Workspace& w, uint32_t n, bool vectors) {
        auto shared = [&](size_t floats) {
            return [rt.device newBufferWithLength:std::max<size_t>(floats, 4) * sizeof(float)
                                          options:MTLResourceStorageModeShared];
        };
        if (!w.A[1]) w.A[1] = shared((size_t)w.lda * n);
        if (vectors && !w.Z[1]) w.Z[1] = shared((size_t)n * n);
    }
};

MPSMatrix* mps_matrix(id<MTLBuffer> b, size_t offset_floats, uint32_t rows, uint32_t cols, uint32_t ld) {
    MPSMatrixDescriptor* d = [MPSMatrixDescriptor matrixDescriptorWithRows:rows
                                                                   columns:cols
                                                                  rowBytes:(size_t)ld * sizeof(float)
                                                                  dataType:MPSDataTypeFloat32];
    return [[MPSMatrix alloc] initWithBuffer:b offset:offset_floats * sizeof(float) descriptor:d];
}

// `count` matrices, `stride` floats apart. MPS's batched products step from
// one left or result matrix to the next by rows x rowBytes, whatever
// matrixBytes says (macOS 27), so the descriptor is a whole stride tall
// (stride a multiple of ld): the product's own sizes say what it reads. The
// buffer needs a stride of slack past the batch's last matrix. (As
// band_reduce.mm's.)
MPSMatrix* mps_matrix(id<MTLBuffer> b, size_t offset_floats, uint32_t rows, uint32_t cols, uint32_t ld,
                      uint32_t count, size_t stride) {
    if (count <= 1) return mps_matrix(b, offset_floats, rows, cols, ld);
    if (stride % ld != 0 || stride / ld < rows) throw std::logic_error("[eigh] tridiag: a batched view's stride");
    MPSMatrixDescriptor* d = [MPSMatrixDescriptor matrixDescriptorWithRows:(uint32_t)(stride / ld)
                                                                   columns:cols
                                                                  matrices:count
                                                                  rowBytes:(size_t)ld * sizeof(float)
                                                               matrixBytes:stride * sizeof(float)
                                                                  dataType:MPSDataTypeFloat32];
    return [[MPSMatrix alloc] initWithBuffer:b offset:offset_floats * sizeof(float) descriptor:d];
}

// The products' kernels by shape, reused (making one costs more than
// encoding it). A thread's own: MPS kernels are not thread-safe.
void gemm(id<MTLDevice> dev, id<MTLCommandBuffer> cb, MPSMatrix* A, bool ta, MPSMatrix* B, bool tb,
          MPSMatrix* C, uint32_t m, uint32_t n, uint32_t k, double alpha, double beta, uint32_t batch = 1) {
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

uint32_t group_size(id<MTLComputePipelineState> ps, uint32_t want) {
    return std::min<uint32_t>(want, (uint32_t)ps.maxTotalThreadsPerThreadgroup);
}

// The reduction's buffers for `batch` matrices of one order, each buffer
// holding the matrices `s*` floats apart (A, B and C with a stride of slack
// past the last, for the batched products' views).
struct Reduce {
    id<MTLBuffer> A, W, B, C, P, tmp, red, d, e, tau;
    size_t   npart_off = 0, dpart_off = 0;   // in red, bytes
    uint32_t lda = 0, batch = 1;
    uint32_t sa = 0, sw = 0, sb = 0, sp = 0, sr = 0, sv = 0, stm = 0;
};

// Floats of the partials (red) a matrix of order n needs: the reflector's
// scale, then the norm partials (two a threadgroup), then the dot partials.
size_t red_floats(uint32_t n) { return 4 + 3 * ((size_t)groups(n) + 1); }

// Step 1 for the matrices in r.A (n x n each, column-major, lower triangle):
// leaves ssytrd('L')'s d, e, tau for matrix b at d + b n, e + b n, tau + b n
// and its reflectors in r.A; three dispatches a column, spread over the GPU.
// (The batched form reduces a batch with encode_reduce_batch instead.)
void tridiagonalize(Cache& cache, const Reduce& r, uint32_t n, float* d, float* e, float* tau) {
    const Pipelines& p = cache.pipelines();
    id<MTLDevice> dev = cache.rt.device;
    id<MTLBuffer> Abuf = r.A;
    const uint32_t lda = r.lda, ldw = n, nb = kPanel, B = r.batch;
    id<MTLCommandBuffer> last = nil;
    uint32_t k = 0;
    for (; k + nb + 1 < n; k += nb) {
        id<MTLCommandBuffer> cb = [cache.rt.queue commandBufferWithUnretainedReferences];
        const uint32_t nn = n - k;
        const size_t offk = (size_t)k * lda + k;
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        for (uint32_t i = 0; i < nb; ++i) {
            const uint32_t len = nn - i, lr = len - 1, blocks = (lr + kTile - 1) / kTile;
            const TdParams prm{nn, lda, ldw, i, k, groups(len), i > 0 ? groups(len) : 0,
                               blocks * (blocks + 1) / 2, r.sa, r.sw, r.sp, r.sr, r.sv, r.stm};
            // Finish W(:, i-1), update Ak(i:, i), the norm partials.
            [enc setComputePipelineState:p.update];
            [enc setBuffer:Abuf offset:offk * sizeof(float) atIndex:0];
            [enc setBuffer:r.W offset:0 atIndex:1];
            [enc setBuffer:r.tau offset:0 atIndex:2];
            [enc setBuffer:r.red offset:r.dpart_off atIndex:3];
            [enc setBuffer:r.red offset:r.npart_off atIndex:4];
            [enc setBytes:&prm length:sizeof prm atIndex:5];
            [enc dispatchThreadgroups:MTLSizeMake(prm.ng, B, 1) threadsPerThreadgroup:MTLSizeMake(kGroup, 1, 1)];

            // The reflector; W(i+1:, i) = Ak(i+1:, i+1:) v in tiles; the corrections' dot products.
            [enc setComputePipelineState:p.symv];
            [enc setBuffer:Abuf offset:offk * sizeof(float) atIndex:0];
            [enc setBuffer:r.W offset:0 atIndex:1];
            [enc setBuffer:r.P offset:0 atIndex:2];
            [enc setBuffer:r.tmp offset:0 atIndex:3];
            [enc setBuffer:r.red offset:r.npart_off atIndex:4];
            [enc setBuffer:r.d offset:0 atIndex:5];
            [enc setBuffer:r.e offset:0 atIndex:6];
            [enc setBuffer:r.tau offset:0 atIndex:7];
            [enc setBuffer:r.red offset:0 atIndex:8];
            [enc setBytes:&prm length:sizeof prm atIndex:9];
            [enc dispatchThreadgroups:MTLSizeMake(prm.tiles + 2 * i, B, 1)
                threadsPerThreadgroup:MTLSizeMake(kGroup, 1, 1)];

            // Sum the tiles, apply the corrections, store v; the dot partials.
            [enc setComputePipelineState:p.apply];
            [enc setBuffer:Abuf offset:offk * sizeof(float) atIndex:0];
            [enc setBuffer:r.W offset:0 atIndex:1];
            [enc setBuffer:r.P offset:0 atIndex:2];
            [enc setBuffer:r.tmp offset:0 atIndex:3];
            [enc setBuffer:r.tau offset:0 atIndex:4];
            [enc setBuffer:r.red offset:0 atIndex:5];
            [enc setBuffer:r.red offset:r.dpart_off atIndex:6];
            [enc setBytes:&prm length:sizeof prm atIndex:7];
            [enc dispatchThreadgroups:MTLSizeMake(groups(lr), B, 1) threadsPerThreadgroup:MTLSizeMake(kGroup, 1, 1)];
        }
        // A(k+nb:, k+nb:) -= [V W] [W V]^T, one GEMM. The column-major m x 2nb
        // B and C, seen row-major, are B^T and C^T.
        const uint32_t m = nn - nb;
        const PackParams pp{m, nb, lda, ldw, n, k, groups(m), r.sa, r.sw, r.sb, r.sv, r.sr};
        [enc setComputePipelineState:p.pack];
        [enc setBuffer:Abuf offset:offk * sizeof(float) atIndex:0];
        [enc setBuffer:r.W offset:0 atIndex:1];
        [enc setBuffer:r.B offset:0 atIndex:2];
        [enc setBuffer:r.C offset:0 atIndex:3];
        [enc setBuffer:r.tau offset:0 atIndex:4];
        [enc setBuffer:r.red offset:r.dpart_off atIndex:5];
        [enc setBytes:&pp length:sizeof pp atIndex:6];
        [enc dispatchThreads:MTLSizeMake(m, nb, B) threadsPerThreadgroup:MTLSizeMake(group_size(p.pack, 256), 1, 1)];
        [enc endEncoding];
        gemm(dev, cb, mps_matrix(r.B, 0, 2 * nb, m, n, B, r.sb), true, mps_matrix(r.C, 0, 2 * nb, m, n, B, r.sb), false,
             mps_matrix(Abuf, offk + (size_t)nb * lda + nb, m, m, lda, B, r.sa), m, m, 2 * nb, -1.0, 1.0, B);
        enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:p.restore];
        [enc setBuffer:Abuf offset:offk * sizeof(float) atIndex:0];
        [enc setBuffer:r.e offset:0 atIndex:1];
        [enc setBytes:&pp length:sizeof pp atIndex:2];
        [enc setBytes:&k length:sizeof k atIndex:3];
        [enc dispatchThreads:MTLSizeMake(nb, B, 1) threadsPerThreadgroup:MTLSizeMake(nb, 1, 1)];
        [enc endEncoding];
        [cb commit];   // queued behind the previous panel; nothing waits here
        last = cb;
    }
    if (last) {
        [last waitUntilCompleted];
        if (last.error) {
            throw std::runtime_error(std::string("[eigh] tridiag: GPU error: ") +
                                     last.error.localizedDescription.UTF8String);
        }
        for (uint32_t b = 0; b < B; ++b) {
            std::memcpy(d + (size_t)b * n, static_cast<const float*>(r.d.contents) + (size_t)b * r.sv, k * sizeof(float));
            std::memcpy(e + (size_t)b * n, static_cast<const float*>(r.e.contents) + (size_t)b * r.sv, k * sizeof(float));
            std::memcpy(tau + (size_t)b * n, static_cast<const float*>(r.tau.contents) + (size_t)b * r.sv,
                        k * sizeof(float));
        }
    }
    // The remaining n - k columns (fewer than a panel and one): LAPACK, the
    // matrices of a batch on the CPU's cores.
    auto tail = [&](uint32_t b) {
        float* A = static_cast<float*>(Abuf.contents) + (size_t)b * r.sa;
        float *db = d + (size_t)b * n, *eb = e + (size_t)b * n, *tb = tau + (size_t)b * n;
        char ul = 'L';
        L nr = n - k, LDA = lda, info = 0, lw = -1;
        float q = 0.0f;
        ssytrd_(&ul, &nr, A + (size_t)k * lda + k, &LDA, db + k, eb + k, tb + k, &q, &lw, &info);
        std::vector<float> work(std::max<L>(1, (L)q));
        lw = (L)work.size();
        ssytrd_(&ul, &nr, A + (size_t)k * lda + k, &LDA, db + k, eb + k, tb + k, work.data(), &lw, &info);
        if (info != 0)
            throw std::runtime_error("[eigh] tridiag: LAPACK ssytrd failed, info " + std::to_string((long long)info));
    };
    if (B == 1) tail(0);
    else metal_linalg::detail::lapack_batches(B, (size_t)(n - k) * (n - k), [&](uint32_t b0, uint32_t b1) {
        for (uint32_t b = b0; b < b1; ++b) tail(b);
    });
}

// Step 3: Z <- Q Z, Q = H(0) ... H(n-2) from ssytrd('L')'s reflectors in Abuf.
// Z is column-major (ld n), so Z^T row-major; per block of reflectors k0 ..
// k0 + kb - 1, acting on rows k0 + 1 ..:  Z^T(:, k0+1:) -= ((Z^T(:, k0+1:) V) T^T) V^T.
// The blocks are applied last first; each block's V and T are built on the CPU
// (compact_wy_t, the copies on every core) into one of two slots while the GPU
// works on the other.
void back_transform(Cache& cache, Workspace& ws, id<MTLBuffer> Abuf, id<MTLBuffer> Zbuf, uint32_t n,
                    const float* tau) {
    if (n < 2) return;
    id<MTLDevice> dev = cache.rt.device;
    const float* A = static_cast<const float*>(Abuf.contents);
    const uint32_t lda = ws.lda, bb = kBackBlock;
    std::vector<float> vcol, tcol((size_t)bb * bb);
    id<MTLCommandBuffer> inflight[2] = {nil, nil};
    int slot = 0;
    for (int k0 = (int)(((n - 2) / bb) * bb); k0 >= 0; k0 -= (int)bb) {
        const uint32_t kb = std::min<uint32_t>(bb, n - 1 - (uint32_t)k0), m = n - (uint32_t)k0 - 1;
        if (inflight[slot]) [inflight[slot] waitUntilCompleted];   // the slot's previous block is done
        vcol.resize((size_t)m * kb);
        metal_linalg::detail::parallel_for(kb, [&](size_t j) {
            float* col = vcol.data() + j * m;
            const float* src = A + (size_t)(k0 + j) * lda + k0 + 1;
            std::fill(col, col + j, 0.0f);
            col[j] = 1.0f;
            std::copy(src + j + 1, src + m, col + j + 1);
        });
        metal_linalg::detail::compact_wy_t(m, kb, vcol.data(), tau + k0, tcol.data(), bb);
        float* V = static_cast<float*>(ws.V[slot].contents);
        float* T = static_cast<float*>(ws.T[slot].contents);
        metal_linalg::detail::parallel_for((m + 255) / 256, [&](size_t t) {
            for (uint32_t r = (uint32_t)t * 256; r < std::min<uint32_t>(m, (uint32_t)t * 256 + 256); ++r)
                for (uint32_t j = 0; j < bb; ++j) V[(size_t)r * bb + j] = j < kb ? vcol[(size_t)j * m + r] : 0.0f;
        });
        for (uint32_t i = 0; i < bb; ++i)
            for (uint32_t j = 0; j < bb; ++j)
                T[(size_t)i * bb + j] = (i < kb && j < kb && j >= i) ? tcol[(size_t)j * bb + i] : 0.0f;

        id<MTLCommandBuffer> cb = [cache.rt.queue commandBufferWithUnretainedReferences];
        MPSMatrix* Zs = mps_matrix(Zbuf, (size_t)k0 + 1, n, m, n);
        MPSMatrix* Vm = mps_matrix(ws.V[slot], 0, m, bb, bb);
        MPSMatrix* Tm = mps_matrix(ws.T[slot], 0, bb, bb, bb);
        MPSMatrix* Y  = mps_matrix(ws.Y, 0, n, bb, bb);
        MPSMatrix* Y2 = mps_matrix(ws.Y2, 0, n, bb, bb);
        gemm(dev, cb, Zs, false, Vm, false, Y, n, bb, m, 1.0, 0.0);     // Y  = Z^T(:, k0+1:) V
        gemm(dev, cb, Y, false, Tm, true, Y2, n, bb, bb, 1.0, 0.0);     // Y2 = Y T^T
        gemm(dev, cb, Y2, false, Vm, true, Zs, n, m, bb, -1.0, 1.0);    // Z^T(:, k0+1:) -= Y2 V^T
        [cb commit];
        inflight[slot] = cb;
        slot ^= 1;
    }
    for (id<MTLCommandBuffer> cb : inflight) {
        if (!cb) continue;
        [cb waitUntilCompleted];
        if (cb.error) {
            throw std::runtime_error(std::string("[eigh] tridiag: GPU error: ") +
                                     cb.error.localizedDescription.UTF8String);
        }
    }
}

// --- A batch reduced together ------------------------------------------------
//
// For batches of mid-size matrices: the reduction above for every matrix at
// once (each dispatch's fixed cost paid once a batch, not once a matrix), the
// tridiagonal problems on the CPU's cores a matrix a core, and the
// back-transformation as batched products.

struct BatchWorkspace {
    uint32_t n = 0, capacity = 0;
    bool     vectors = false;
    Reduce   r;
    id<MTLBuffer> Z;                      // eigenvectors, column-major, n x n a matrix
    id<MTLBuffer> V, G, T, Y, Y2;         // the back-transformation's
};

// Two, for the pipeline over a batch's chunks (slot 0 or 1).
BatchWorkspace& batch_workspace(Cache& cache, uint32_t n, uint32_t batch, bool vectors, int slot) {
    static BatchWorkspace ws[2];
    BatchWorkspace& w = ws[slot];
    if (w.r.A && w.n == n && w.capacity >= batch && (w.vectors || !vectors)) return w;
    id<MTLDevice> dev = cache.rt.device;
    auto make = [&](size_t floats, MTLResourceOptions opt) {
        id<MTLBuffer> b = [dev newBufferWithLength:std::max<size_t>(floats, 4) * sizeof(float) options:opt];
        if (!b) throw std::runtime_error("[eigh] tridiag: could not allocate " + std::to_string(floats * 4) + " bytes");
        return b;
    };
    const MTLResourceOptions shared = MTLResourceStorageModeShared, priv = MTLResourceStorageModePrivate;
    w = BatchWorkspace{};
    w.n = n;
    w.capacity = batch;
    w.vectors = vectors;
    Reduce& r = w.r;
    r.lda = (n + 7) / 8 * 8;
    r.sa  = r.lda * n;
    r.sw  = n * kPanel;
    r.sb  = n * 2 * kPanel;
    r.sp  = (uint32_t)(((n + kTile - 1) / kTile) * (size_t)n);
    r.sr  = (uint32_t)red_floats(n);
    r.sv  = n;
    r.stm = 2 * kPanel;
    r.npart_off = 4 * sizeof(float);
    r.dpart_off = (4 + 2 * ((size_t)groups(n) + 1)) * sizeof(float);
    // A stride of slack past the last matrix, for the batched products' views
    r.A   = make((size_t)(batch + 1) * r.sa, shared);
    r.W   = make((size_t)batch * r.sw, priv);
    r.B   = make((size_t)(batch + 1) * r.sb, priv);
    r.C   = make((size_t)(batch + 1) * r.sb, priv);
    r.P   = make((size_t)batch * r.sp, priv);
    r.tmp = make((size_t)batch * r.stm, priv);
    r.red = make((size_t)batch * r.sr, priv);
    r.d   = make((size_t)batch * n, shared);
    r.e   = make((size_t)batch * n, shared);
    r.tau = make((size_t)batch * n, shared);
    if (vectors) {
        const uint32_t bb = kBatchBackBlock;
        w.Z = make((size_t)(batch + 1) * n * n, shared);
        w.V = make((size_t)(batch + 1) * n * bb, priv);
        w.G = make((size_t)(batch + 1) * bb * bb, priv);
        w.T = make((size_t)(batch + 1) * bb * bb, priv);
        w.Y  = make((size_t)(batch + 1) * n * bb, priv);
        w.Y2 = make((size_t)(batch + 1) * n * bb, priv);
    }
    return w;
}

// Step 3 for the batch, encoded into cb: Z_b <- Q_b Z_b, as back_transform,
// a block of reflectors for every matrix at once and every step on the GPU:
// V and T built there (td_make_v, V^T V, td_make_t), then the three
// products. tau in r.tau, n a matrix.
void encode_back_transform_batch(Cache& cache, id<MTLCommandBuffer> cb, BatchWorkspace& w, uint32_t n, uint32_t batch) {
    if (n < 2) return;
    const Pipelines& p = cache.pipelines();
    id<MTLDevice> dev = cache.rt.device;
    const Reduce& r = w.r;
    const uint32_t bb = kBatchBackBlock;
    const size_t sv = (size_t)n * bb, st = (size_t)bb * bb, sz = (size_t)n * n;
    for (int k0 = (int)(((n - 2) / bb) * bb); k0 >= 0; k0 -= (int)bb) {
        const uint32_t kb = std::min<uint32_t>(bb, n - 1 - (uint32_t)k0), m = n - (uint32_t)k0 - 1;
        const WyParams wp{m, kb, bb, (uint32_t)k0, r.lda, r.sa, r.sv, (uint32_t)sv, (uint32_t)st};
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:p.make_v];
        [enc setBuffer:r.A offset:0 atIndex:0];
        [enc setBuffer:w.V offset:0 atIndex:1];
        [enc setBytes:&wp length:sizeof wp atIndex:2];
        [enc dispatchThreads:MTLSizeMake(bb, m, batch) threadsPerThreadgroup:MTLSizeMake(bb, 1, 1)];
        [enc endEncoding];
        MPSMatrix* Vm = mps_matrix(w.V, 0, m, bb, bb, batch, sv);
        gemm(dev, cb, Vm, true, Vm, false, mps_matrix(w.G, 0, bb, bb, bb, batch, st), bb, bb, m, 1.0, 0.0, batch);
        enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:p.make_t];
        [enc setBuffer:w.G offset:0 atIndex:0];
        [enc setBuffer:r.tau offset:0 atIndex:1];
        [enc setBuffer:w.T offset:0 atIndex:2];
        [enc setBytes:&wp length:sizeof wp atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(batch, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
        MPSMatrix* Zs = mps_matrix(w.Z, (size_t)k0 + 1, n, m, n, batch, sz);
        MPSMatrix* Tm = mps_matrix(w.T, 0, bb, bb, bb, batch, st);   // T, row-major
        MPSMatrix* Y  = mps_matrix(w.Y, 0, n, bb, bb, batch, sv);
        MPSMatrix* Y2 = mps_matrix(w.Y2, 0, n, bb, bb, batch, sv);
        gemm(dev, cb, Zs, false, Vm, false, Y, n, bb, m, 1.0, 0.0, batch);    // Y  = Z^T(:, k0+1:) V
        gemm(dev, cb, Y, false, Tm, true, Y2, n, bb, bb, 1.0, 0.0, batch);    // Y2 = Y T^T
        gemm(dev, cb, Y2, false, Vm, true, Zs, n, m, bb, -1.0, 1.0, batch);   // Z^T(:, k0+1:) -= Y2 V^T
    }
}

// Step 1 for a chunk of the batch, encoded into cb, nothing on the CPU:
// the matrices loaded (td_load), the panels (td_panel and the trailing
// products), and the last columns by td_panel as one final panel. d, e and
// tau end in r's buffers.
void encode_reduce_batch(Cache& cache, id<MTLCommandBuffer> cb, const Reduce& r, uint32_t n, id<MTLBuffer> src,
                         id<MTLBuffer> index, id<MTLBuffer> scale, uint32_t c0, bool lower, uint32_t threads) {
    const Pipelines& p = cache.pipelines();
    id<MTLDevice> dev = cache.rt.device;
    const uint32_t lda = r.lda, nb = kPanel, B = r.batch;
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    const LoadParams lp{n, lda, r.sa, lower ? 1u : 0u, c0};
    [enc setComputePipelineState:p.load];
    [enc setBuffer:src offset:0 atIndex:0];
    [enc setBuffer:r.A offset:0 atIndex:1];
    [enc setBuffer:index offset:0 atIndex:2];
    [enc setBuffer:scale offset:0 atIndex:3];
    [enc setBytes:&lp length:sizeof lp atIndex:4];
    [enc dispatchThreadgroups:MTLSizeMake((n + 31) / 32, (n + 31) / 32, B) threadsPerThreadgroup:MTLSizeMake(32, 8, 1)];
    for (uint32_t k = 0;; k += nb) {
        const uint32_t nn = n - k, last = k + nb + 1 >= n, width = last ? nn : nb;
        const size_t offk = (size_t)k * lda + k;
        const PanelParams pp{nn, lda, n, n, k, width, r.sa, r.sw, r.sb, r.sv};
        [enc setComputePipelineState:p.panel];
        [enc setBuffer:r.A offset:offk * sizeof(float) atIndex:0];
        [enc setBuffer:r.W offset:0 atIndex:1];
        [enc setBuffer:r.d offset:0 atIndex:2];
        [enc setBuffer:r.e offset:0 atIndex:3];
        [enc setBuffer:r.tau offset:0 atIndex:4];
        [enc setBuffer:r.B offset:0 atIndex:5];
        [enc setBuffer:r.C offset:0 atIndex:6];
        [enc setBytes:&pp length:sizeof pp atIndex:7];
        [enc setThreadgroupMemoryLength:((size_t)2 * nn * sizeof(float) + 15) / 16 * 16 atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(B, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(group_size(p.panel, threads), 1, 1)];
        if (last) break;
        [enc endEncoding];
        const uint32_t m = nn - nb;
        gemm(dev, cb, mps_matrix(r.B, 0, 2 * nb, m, n, B, r.sb), true, mps_matrix(r.C, 0, 2 * nb, m, n, B, r.sb),
             false, mps_matrix(r.A, offk + (size_t)nb * lda + nb, m, m, lda, B, r.sa), m, m, 2 * nb, -1.0, 1.0, B);
        enc = [cb computeCommandEncoder];
    }
    [enc endEncoding];
}

} // namespace

namespace core::detail {

void eigh_tridiag(const Matrices& a, bool lower, float* w_out, float* v_out, uint32_t* info_out) {
    const uint32_t n = a.cols, batch = a.batch;
    if (a.rows != n) {
        throw std::invalid_argument("[eigh] Input matrices must be square.");
    }
    if (n == 0 || batch == 0) {
        if (info_out) std::fill(info_out, info_out + batch, 0u);
        return;
    }
    AutoreleasePool pool;
    static Cache cache;
    const bool vectors = v_out != nullptr;
    Workspace& ws = cache.workspace(n, vectors);
    const size_t per = (size_t)n * n;

    std::vector<float> amax(batch);
    std::vector<char>  finite(batch);
    scan(a, lower ? Part::lower : Part::upper, amax.data(), finite.data());

    // Non-finite matrices give NaN at once; the rest are pipelined.
    std::vector<uint32_t> todo;
    for (uint32_t b = 0; b < batch; ++b) {
        if (finite[b]) { todo.push_back(b); continue; }
        std::fill(w_out + (size_t)b * n, w_out + (size_t)(b + 1) * n, NAN);
        if (vectors) std::fill(v_out + b * per, v_out + (b + 1) * per, NAN);
        if (info_out) info_out[b] = 1u << 17;
    }
    if (todo.size() > 1) cache.second_slot(ws, n, vectors);

    struct Slot { std::vector<float> d, e, tau; float scale = 1.0f; };
    Slot slots[2];
    for (Slot& sl : slots) { sl.d.resize(n); sl.e.resize(n); sl.tau.resize(n); }

    // Step 1 for matrix b in slot s: scale, copy in, tridiagonalize.
    auto reduce = [&](uint32_t b, int s) {
        // A power of two putting the largest entry in [0.5, 1): exact, and keeps
        // every product of the reduction inside float32's range.
        int ex = 0;
        if (amax[b] > 0.0f) std::frexp(amax[b], &ex);
        const float scale = std::ldexp(1.0f, -ex);
        slots[s].scale = scale;

        // Column-major lower triangle: the row-major lower triangle is the
        // column-major upper one, so it is transposed in; the row-major upper
        // triangle is already the column-major lower one.
        const float* src = a.data + b * per;
        float* A = static_cast<float*>(ws.A[s].contents);
        if (lower) {
            transpose_scaled(src, n, A, ws.lda, n, n, scale);
        } else {
            for (uint32_t j = 0; j < n; ++j) {
                float* col = A + (size_t)j * ws.lda;
                for (uint32_t i = 0; i < n; ++i) col[i] = src[(size_t)j * n + i] * scale;
            }
        }
        Reduce r;
        r.A = ws.A[s]; r.W = ws.W; r.B = ws.B; r.C = ws.C; r.P = ws.P; r.tmp = ws.tmp; r.red = ws.red;
        r.d = ws.d; r.e = ws.e; r.tau = ws.tau;
        r.npart_off = ws.npart_off;
        r.dpart_off = ws.dpart_off;
        r.lda = ws.lda;
        tridiagonalize(cache, r, n, slots[s].d.data(), slots[s].e.data(), slots[s].tau.data());
    };

    // Step 2 for matrix b in slot s, on the CPU: the eigenvalues into w_out,
    // the tridiagonal problem's eigenvectors into Z[s].
    auto solve = [&](uint32_t b, int s) {
        Slot& sl = slots[s];
        L N = n, info = 0;
        if (!vectors) {
            // By bisection on the GPU where it is the faster, else ssterf.
            std::vector<float> w(n);
            if (metal_linalg::detail::tridiagonal_eigenvalues(n, sl.d.data(), sl.e.data(), w.data()))
                sl.d = std::move(w);
            else
                ssterf_(&N, sl.d.data(), sl.e.data(), &info);
        } else {
            // sstedc's divide and conquer on the CPU's cores but the two the
            // GPU's host work keeps (divide_conquer.cpp); for one matrix,
            // whose GPU is idle meanwhile, its largest products on the GPU
            float* Z = static_cast<float*>(ws.Z[s].contents);
            std::unique_ptr<metal_linalg::detail::MpsGemm> gpu;
            if (todo.size() == 1) {
                gpu = std::make_unique<metal_linalg::detail::MpsGemm>(cache.rt.device, cache.rt.queue);
                gpu->add_buffer(ws.Z[s]);
            }
            info = (L)metal_linalg::detail::tridiagonal_eigensystem(
                n, sl.d.data(), sl.e.data(), Z, n, metal_linalg::detail::cpu_threads_beside_gpu(), gpu.get());
        }
        if (info != 0) {
            throw std::runtime_error(std::string("[eigh] tridiag: LAPACK ") + (vectors ? "sstedc" : "ssterf") +
                                     " failed on matrix " + std::to_string(b) + " (N=" + std::to_string(n) +
                                     "), info " + std::to_string((long long)info) + ".");
        }
        const float unscale = 1.0f / sl.scale;
        float* w = w_out + (size_t)b * n;
        for (uint32_t i = 0; i < n; ++i) w[i] = sl.d[i] * unscale;   // ascending, as LAPACK leaves them
    };

    // Step 3 for matrix b in slot s, and the output.
    auto finish = [&](uint32_t b, int s) {
        if (vectors) {
            back_transform(cache, ws, ws.A[s], ws.Z[s], n, slots[s].tau.data());
            // Z is column-major: its columns are the eigenvectors, which the
            // row-major output wants as columns too, so it is transposed out.
            vDSP_mtrans(static_cast<const float*>(ws.Z[s].contents), 1, v_out + b * per, 1, n, n);
        }
        if (info_out) info_out[b] = 1u | (1u << 16);   // converged; one "sweep", as the CPU path
    };

    // The pipeline: matrix t in slot t % 2. The CPU solves t while the GPU
    // reduces t + 1, and solves t + 1 while the GPU back-transforms t.
    const size_t count = todo.size();
    if (count == 0) return;
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

// The batched form. The batch goes in chunks through three stages: the
// reduction (GPU), the tridiagonal problems (CPU, every core but two) and
// the back-transformation (GPU), pipelined over two workspace slots: the GPU
// is kept a chunk ahead, reducing chunk k + 1 while the CPU solves chunk k,
// and the host does nothing else (the loads, the last columns and the
// output on the GPU too). See the comment above BatchWorkspace.
void eigh_tridiag_batch(const Matrices& a, bool lower, float* w_out, float* v_out, uint32_t* info_out) {
    const uint32_t n = a.cols, batch = a.batch;
    if (a.rows != n) throw std::invalid_argument("[eigh] Input matrices must be square.");
    if (n == 0 || batch == 0) {
        if (info_out) std::fill(info_out, info_out + batch, 0u);
        return;
    }
    if (n > kPanelMaxN) throw std::invalid_argument("[eigh] tridiag (batched): N above " + std::to_string(kPanelMaxN));
    // With eigenvectors from kBandMinN, in two stages (eigh_band_batch);
    // EIGH_TRIDIAG_BATCH_BAND=0 keeps the one-stage reduction
    const unsigned solvers = metal_linalg::detail::cpu_threads_beside_gpu();
    const unsigned band_max_batch =
        std::max(1u, n >= kBandDoubleMinN ? 2 * solvers : n >= kBandFullMinN ? solvers : solvers / 2);
    if (v_out && n >= kBandMinN && batch <= band_max_batch) {
        const char* e = std::getenv("EIGH_TRIDIAG_BATCH_BAND");
        if (!(e && std::string(e) == "0")) {
            metal_linalg::detail::eigh_band_batch(a, lower, w_out, v_out, info_out);
            return;
        }
    }
    AutoreleasePool pool;
    static Cache cache;
    id<MTLDevice> dev = cache.rt.device;
    const bool vectors = v_out != nullptr;
    const size_t per = (size_t)n * n;

    std::vector<float> amax(batch);
    std::vector<char>  finite(batch);
    scan(a, lower ? Part::lower : Part::upper, amax.data(), finite.data());
    std::vector<uint32_t> todo;
    for (uint32_t b = 0; b < batch; ++b) {
        if (finite[b]) { todo.push_back(b); continue; }
        std::fill(w_out + (size_t)b * n, w_out + (size_t)(b + 1) * n, NAN);
        if (vectors) std::fill(v_out + b * per, v_out + (b + 1) * per, NAN);
        if (info_out) info_out[b] = 1u << 17;
    }
    if (todo.empty()) return;
    const size_t total = todo.size();
    // A thread a row, 64 to 512 threads
    const uint32_t threads = std::clamp((n + 31) / 32 * 32, 64u, 512u);
    // Chunks: four, to overlap the stages (eight for matrices of up to 128 x
    // 128, whose pipeline's first and last stages, alone on the GPU, are a
    // larger share: 1.07-1.15x at 1024 x 96-128^2; from 256 four were
    // faster), each large enough to fill the GPU, at most 2^25 floats of
    // matrices (128 MB a slot)
    size_t chunks = per <= (size_t)128 * 128 ? 8 : 4;
    const size_t min_chunk = std::max<size_t>(16, ((size_t)1 << 21) / per);   // 8 MB of matrices
    chunks = std::clamp<size_t>(std::min(chunks, total / std::max<size_t>(1, min_chunk)), 1, total);
    size_t chunk = (total + chunks - 1) / chunks;
    chunk = std::min(chunk, std::max<size_t>(1, ((size_t)1 << 25) / per));
    const size_t count = (total + chunk - 1) / chunk;

    // The input, the matrices to do and their scales, and the output, on the GPU
    id<MTLBuffer> src = metal_linalg::detail::input_buffer(dev, a);
    id<MTLBuffer> index = [dev newBufferWithBytes:todo.data() length:total * sizeof(uint32_t)
                                          options:MTLResourceStorageModeShared];
    id<MTLBuffer> scale = [dev newBufferWithLength:total * sizeof(float) options:MTLResourceStorageModeShared];
    float* sc = static_cast<float*>(scale.contents);
    for (size_t j = 0; j < total; ++j) {
        // A power of two putting the largest entry in [0.5, 1), as eigh_tridiag's
        int ex = 0;
        if (amax[todo[j]] > 0.0f) std::frexp(amax[todo[j]], &ex);
        sc[j] = std::ldexp(1.0f, -ex);
    }
    id<MTLBuffer> out = nil;
    std::vector<float> staged;
    if (vectors) {
        const size_t page = (size_t)getpagesize();
        if (reinterpret_cast<uintptr_t>(v_out) % page == 0) {
            try { out = metal_linalg::detail::wrap_host(dev, v_out, (size_t)batch * per); } catch (...) { out = nil; }
        }
        if (!out) out = [dev newBufferWithLength:(size_t)batch * per * sizeof(float) options:MTLResourceStorageModeShared];
    }
    const bool out_staged = vectors && out.contents != (void*)v_out;

    BatchWorkspace* ws[2] = {&batch_workspace(cache, n, (uint32_t)chunk, vectors, 0),
                             count > 1 ? &batch_workspace(cache, n, (uint32_t)chunk, vectors, 1) : nullptr};
    auto check = [](id<MTLCommandBuffer> cb) {
        [cb waitUntilCompleted];
        if (cb.error)
            throw std::runtime_error(std::string("[eigh] tridiag: GPU error: ") + cb.error.localizedDescription.UTF8String);
    };
    std::vector<id<MTLCommandBuffer>> committed;   // waited for whatever happens
    auto timed = [&](id<MTLCommandBuffer> cb) {
        [cb commit];
        committed.push_back(cb);
        return cb;
    };
    auto cnt_of = [&](size_t k) { return (uint32_t)std::min(chunk, total - k * chunk); };
    auto reduce = [&](size_t k) {
        BatchWorkspace& w = *ws[k % 2];
        Reduce r = w.r;
        r.batch = cnt_of(k);
        id<MTLCommandBuffer> cb = [cache.rt.queue commandBufferWithUnretainedReferences];
        encode_reduce_batch(cache, cb, r, n, src, index, scale, (uint32_t)(k * chunk), lower, threads);
        return timed(cb);
    };
    auto back = [&](size_t k) {
        BatchWorkspace& w = *ws[k % 2];
        const uint32_t cnt = cnt_of(k);
        id<MTLCommandBuffer> cb = [cache.rt.queue commandBufferWithUnretainedReferences];
        encode_back_transform_batch(cache, cb, w, n, cnt);
        const StoreParams sp{n, (uint32_t)(k * chunk)};
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:cache.pipelines().store];
        [enc setBuffer:w.Z offset:0 atIndex:0];
        [enc setBuffer:out offset:0 atIndex:1];
        [enc setBuffer:index offset:0 atIndex:2];
        [enc setBytes:&sp length:sizeof sp atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake((n + 31) / 32, (n + 31) / 32, cnt) threadsPerThreadgroup:MTLSizeMake(32, 8, 1)];
        [enc endEncoding];
        return timed(cb);
    };
    // Stage 2 for chunk k, on the calling thread and every core but two:
    // the eigenvalues into w_out, the vectors into Z
    const unsigned solve_threads = metal_linalg::detail::cpu_threads_beside_gpu();
    auto solve = [&](size_t k) {
        BatchWorkspace& w = *ws[k % 2];
        const uint32_t cnt = cnt_of(k);
        const float* dg = static_cast<const float*>(w.r.d.contents);
        const float* eg = static_cast<const float*>(w.r.e.contents);
        float* Z = vectors ? static_cast<float*>(w.Z.contents) : nullptr;
        metal_linalg::detail::lapack_batches(cnt, per, solve_threads, [&](uint32_t b0, uint32_t b1) {
            std::vector<float> dj(n), ej(n);
            for (uint32_t j = b0; j < b1; ++j) {
                const size_t at = k * chunk + j;
                std::copy(dg + (size_t)j * w.r.sv, dg + (size_t)j * w.r.sv + n, dj.begin());
                std::copy(eg + (size_t)j * w.r.sv, eg + (size_t)j * w.r.sv + n, ej.begin());
                L N = n, info = 0;
                if (vectors) info = (L)metal_linalg::detail::tridiagonal_eigensystem(n, dj.data(), ej.data(), Z + j * per, n, 1);
                else         ssterf_(&N, dj.data(), ej.data(), &info);
                if (info != 0)
                    throw std::runtime_error(std::string("[eigh] tridiag: LAPACK ") + (vectors ? "sstedc" : "ssterf") +
                                             " failed on matrix " + std::to_string(todo[at]) + " (N=" +
                                             std::to_string(n) + "), info " + std::to_string((long long)info) + ".");
                const float unscale = 1.0f / sc[at];
                float* wb = w_out + (size_t)todo[at] * n;
                for (uint32_t i = 0; i < n; ++i) wb[i] = dj[i] * unscale;
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
        for (id<MTLCommandBuffer> cb : committed) [cb waitUntilCompleted];   // nothing left using the workspaces
        throw;
    }
    if (out_staged) {
        const float* o = static_cast<const float*>(out.contents);
        metal_linalg::detail::for_each_matrix((uint32_t)total, per, [&](uint32_t j) {
            std::memcpy(v_out + (size_t)todo[j] * per, o + (size_t)todo[j] * per, per * sizeof(float));
        });
    }
}
} // namespace core::detail
} // namespace metal_linalg
