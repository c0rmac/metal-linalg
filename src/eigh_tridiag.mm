// The `tridiag` eigensolver backend: LAPACK's ssyevd with its two expensive
// steps on the GPU.
//
//   1. Tridiagonalize, A = Q T Q^T, entirely on the GPU (shaders/Eigh_Tridiag.metal):
//      blocked ssytrd (lower), per column slatrd's steps as three kernels, per
//      panel the rank-2nb trailing update as an MPS GEMM. The panels' command
//      buffers are queued back to back and the host waits once per matrix, so
//      no GPU round trip is paid per column. The last few columns, fewer than a
//      panel, are reduced by LAPACK on the CPU.
//   2. T = Z diag(w) Z^T on the CPU: sstedc (eigenvectors) or ssterf (eigenvalues).
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
#include "metal_runtime.h"
#include "shaders.h"

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <algorithm>
#include <cmath>
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

constexpr uint32_t kPanel     = 32;    // columns per panel of the reduction (nb)
constexpr uint32_t kBackBlock = 128;   // reflectors per pass of the back-transformation
constexpr uint32_t kTile      = 64;    // must match TILE in Eigh_Tridiag.metal
constexpr uint32_t kGroup     = 256;   // must match GROUP in Eigh_Tridiag.metal
constexpr uint32_t kRows      = 32;    // must match GROUP / LANES: td_update's and td_apply's rows per threadgroup

// Must match TdParams and PackParams in Eigh_Tridiag.metal.
struct TdParams   { uint32_t nn, lda, ldw, i, k, ng, ngp, tiles; };
struct PackParams { uint32_t m, nb, lda, ldw, ldb, k, ngp; };

uint32_t groups(uint32_t rows) { return (rows + kRows - 1) / kRows; }

using L = __LAPACK_int;

struct Pipelines {
    id<MTLComputePipelineState> update, symv, apply, pack, restore;
};

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

void gemm(id<MTLDevice> dev, id<MTLCommandBuffer> cb, MPSMatrix* A, bool ta, MPSMatrix* B, bool tb,
          MPSMatrix* C, uint32_t m, uint32_t n, uint32_t k, double alpha, double beta) {
    MPSMatrixMultiplication* g = [[MPSMatrixMultiplication alloc] initWithDevice:dev
                                                                   transposeLeft:ta
                                                                  transposeRight:tb
                                                                      resultRows:m
                                                                   resultColumns:n
                                                                 interiorColumns:k
                                                                           alpha:alpha
                                                                            beta:beta];
    [g encodeToCommandBuffer:cb leftMatrix:A rightMatrix:B resultMatrix:C];
}

uint32_t group_size(id<MTLComputePipelineState> ps, uint32_t want) {
    return std::min<uint32_t>(want, (uint32_t)ps.maxTotalThreadsPerThreadgroup);
}

// Step 1 for the matrix in Abuf (n x n, column-major, lower triangle): leaves
// ssytrd('L')'s d, e, tau in d, e, tau and its reflectors in Abuf.
void tridiagonalize(Cache& cache, Workspace& ws, id<MTLBuffer> Abuf, uint32_t n, float* d, float* e, float* tau) {
    const Pipelines& p = cache.pipelines();
    id<MTLDevice> dev = cache.rt.device;
    const uint32_t lda = ws.lda, ldw = n, nb = kPanel;
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
                               blocks * (blocks + 1) / 2};
            // Finish W(:, i-1), update Ak(i:, i), the norm partials.
            [enc setComputePipelineState:p.update];
            [enc setBuffer:Abuf offset:offk * sizeof(float) atIndex:0];
            [enc setBuffer:ws.W offset:0 atIndex:1];
            [enc setBuffer:ws.tau offset:0 atIndex:2];
            [enc setBuffer:ws.red offset:ws.dpart_off atIndex:3];
            [enc setBuffer:ws.red offset:ws.npart_off atIndex:4];
            [enc setBytes:&prm length:sizeof prm atIndex:5];
            [enc dispatchThreadgroups:MTLSizeMake(prm.ng, 1, 1) threadsPerThreadgroup:MTLSizeMake(kGroup, 1, 1)];

            // The reflector; W(i+1:, i) = Ak(i+1:, i+1:) v in tiles; the corrections' dot products.
            [enc setComputePipelineState:p.symv];
            [enc setBuffer:Abuf offset:offk * sizeof(float) atIndex:0];
            [enc setBuffer:ws.W offset:0 atIndex:1];
            [enc setBuffer:ws.P offset:0 atIndex:2];
            [enc setBuffer:ws.tmp offset:0 atIndex:3];
            [enc setBuffer:ws.red offset:ws.npart_off atIndex:4];
            [enc setBuffer:ws.d offset:0 atIndex:5];
            [enc setBuffer:ws.e offset:0 atIndex:6];
            [enc setBuffer:ws.tau offset:0 atIndex:7];
            [enc setBuffer:ws.red offset:0 atIndex:8];
            [enc setBytes:&prm length:sizeof prm atIndex:9];
            [enc dispatchThreadgroups:MTLSizeMake(prm.tiles + 2 * i, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(kGroup, 1, 1)];

            // Sum the tiles, apply the corrections, store v; the dot partials.
            [enc setComputePipelineState:p.apply];
            [enc setBuffer:Abuf offset:offk * sizeof(float) atIndex:0];
            [enc setBuffer:ws.W offset:0 atIndex:1];
            [enc setBuffer:ws.P offset:0 atIndex:2];
            [enc setBuffer:ws.tmp offset:0 atIndex:3];
            [enc setBuffer:ws.tau offset:0 atIndex:4];
            [enc setBuffer:ws.red offset:0 atIndex:5];
            [enc setBuffer:ws.red offset:ws.dpart_off atIndex:6];
            [enc setBytes:&prm length:sizeof prm atIndex:7];
            [enc dispatchThreadgroups:MTLSizeMake(groups(lr), 1, 1) threadsPerThreadgroup:MTLSizeMake(kGroup, 1, 1)];
        }
        // A(k+nb:, k+nb:) -= [V W] [W V]^T, one GEMM. The column-major m x 2nb
        // B and C, seen row-major, are B^T and C^T.
        const uint32_t m = nn - nb;
        const PackParams pp{m, nb, lda, ldw, n, k, groups(m)};
        [enc setComputePipelineState:p.pack];
        [enc setBuffer:Abuf offset:offk * sizeof(float) atIndex:0];
        [enc setBuffer:ws.W offset:0 atIndex:1];
        [enc setBuffer:ws.B offset:0 atIndex:2];
        [enc setBuffer:ws.C offset:0 atIndex:3];
        [enc setBuffer:ws.tau offset:0 atIndex:4];
        [enc setBuffer:ws.red offset:ws.dpart_off atIndex:5];
        [enc setBytes:&pp length:sizeof pp atIndex:6];
        [enc dispatchThreads:MTLSizeMake(m, nb, 1) threadsPerThreadgroup:MTLSizeMake(group_size(p.pack, 256), 1, 1)];
        [enc endEncoding];
        gemm(dev, cb, mps_matrix(ws.B, 0, 2 * nb, m, n), true, mps_matrix(ws.C, 0, 2 * nb, m, n), false,
             mps_matrix(Abuf, offk + (size_t)nb * lda + nb, m, m, lda), m, m, 2 * nb, -1.0, 1.0);
        enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:p.restore];
        [enc setBuffer:Abuf offset:offk * sizeof(float) atIndex:0];
        [enc setBuffer:ws.e offset:0 atIndex:1];
        [enc setBytes:&pp length:sizeof pp atIndex:2];
        [enc setBytes:&k length:sizeof k atIndex:3];
        [enc dispatchThreads:MTLSizeMake(nb, 1, 1) threadsPerThreadgroup:MTLSizeMake(nb, 1, 1)];
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
        std::memcpy(d, ws.d.contents, k * sizeof(float));
        std::memcpy(e, ws.e.contents, k * sizeof(float));
        std::memcpy(tau, ws.tau.contents, k * sizeof(float));
    }
    // The remaining n - k columns (fewer than a panel and one): LAPACK.
    float* A = static_cast<float*>(Abuf.contents);
    char ul = 'L';
    L nr = n - k, LDA = lda, info = 0, lw = -1;
    float q = 0.0f;
    ssytrd_(&ul, &nr, A + (size_t)k * lda + k, &LDA, d + k, e + k, tau + k, &q, &lw, &info);
    std::vector<float> work(std::max<L>(1, (L)q));
    lw = (L)work.size();
    ssytrd_(&ul, &nr, A + (size_t)k * lda + k, &LDA, d + k, e + k, tau + k, work.data(), &lw, &info);
    if (info != 0) throw std::runtime_error("[eigh] tridiag: LAPACK ssytrd failed, info " + std::to_string((long long)info));
}

// Step 3: Z <- Q Z, Q = H(0) ... H(n-2) from ssytrd('L')'s reflectors in Abuf.
// Z is column-major (ld n), so Z^T row-major; per block of reflectors k0 ..
// k0 + kb - 1, acting on rows k0 + 1 ..:  Z^T(:, k0+1:) -= ((Z^T(:, k0+1:) V) T^T) V^T.
// The blocks are applied last first; each block's V and T are built on the CPU
// (slarft) into one of two slots while the GPU works on the other.
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
        vcol.assign((size_t)m * kb, 0.0f);
        for (uint32_t j = 0; j < kb; ++j) {
            vcol[(size_t)j * m + j] = 1.0f;
            const float* src = A + (size_t)(k0 + j) * lda + k0 + 1;
            std::copy(src + j + 1, src + m, vcol.begin() + (size_t)j * m + j + 1);
        }
        L M = m, KB = kb, LDT = bb;
        slarft_("F", "C", &M, &KB, vcol.data(), &M, tau + k0, tcol.data(), &LDT);
        float* V = static_cast<float*>(ws.V[slot].contents);
        float* T = static_cast<float*>(ws.T[slot].contents);
        for (uint32_t r = 0; r < m; ++r)
            for (uint32_t j = 0; j < bb; ++j) V[(size_t)r * bb + j] = j < kb ? vcol[(size_t)j * m + r] : 0.0f;
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
        tridiagonalize(cache, ws, ws.A[s], n, slots[s].d.data(), slots[s].e.data(), slots[s].tau.data());
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
            float* Z = static_cast<float*>(ws.Z[s].contents);
            char compz = 'I';
            L lw = -1, liw = -1, iq = 0;
            float q = 0.0f;
            sstedc_(&compz, &N, sl.d.data(), sl.e.data(), Z, &N, &q, &lw, &iq, &liw, &info);
            std::vector<float> work(std::max<L>(1, (L)q));
            std::vector<L> iwork(std::max<L>(1, iq));
            lw = (L)work.size();
            liw = (L)iwork.size();
            sstedc_(&compz, &N, sl.d.data(), sl.e.data(), Z, &N, work.data(), &lw, iwork.data(), &liw, &info);
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

} // namespace core::detail
} // namespace metal_linalg
