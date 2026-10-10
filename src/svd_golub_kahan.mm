// The `golub_kahan` SVD backend: LAPACK's method -- Householder
// bidiagonalization, then implicit bidiagonal QR -- one threadgroup per matrix
// with the whole matrix and V in threadgroup memory
// (shaders/Svd_GolubKahan.metal). The SVD counterpart of the eigensolver's ql
// backend (eigh_ql.mm).
//
// Why: for batches of small and mid-size matrices the one-sided Jacobi kernels
// do several times the flops of LAPACK's method, and since the CPU path spreads
// a batch over every core it beat them almost everywhere. Recording each QR
// step's rotations and applying them a row per thread removes the barrier per
// rotation. Threadgroup memory bounds the size (83 x 83 at 32 KB, longer for
// tall matrices; V lives in a device-memory workspace); svd.mm sends a tall
// matrix that does not fit through the library's QR first, so the kernel sees
// the k x k factor. Up to 32 rows and columns (since 2.17.0) a second kernel
// keeps the matrix, U and V in a simdgroup's registers, a simdgroup a matrix
// (svd_gk_simd; SVD_GK_SIMD=0 turns it off).

#include <metal_linalg/core.h>
#include <metal_linalg/device.h>
#include "metal_runtime.h"
#include "shaders.h"

#import <Metal/Metal.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

using metal_linalg::core::Matrices;
using metal_linalg::detail::AutoreleasePool;
using metal_linalg::detail::MetalRuntime;
using metal_linalg::detail::copy_out;
using metal_linalg::detail::input_buffer;
using metal_linalg::detail::make_pipeline;
using metal_linalg::detail::pad_up;

namespace metal_linalg {
namespace {

// Must match `GkParams` in Svd_GolubKahan.metal.
struct GkParams {
    uint32_t m;
    uint32_t n;
    uint32_t transpose;
    uint32_t max_rots;
};

// Must match `GksParams` in Svd_GolubKahan.metal (svd_gk_simd, in registers).
struct GksParams {
    uint32_t m;
    uint32_t n;
    uint32_t transpose;
    uint32_t max_rots;
    uint32_t batch;
};
constexpr uint32_t kSimdMaxDim = 32;     // rows and columns: a row a lane
constexpr uint32_t kSimdFloats = 336;    // its threadgroup memory a matrix (kGksFloats)
constexpr uint32_t kSimdFloatsRun = 488; // the same with a runner (kGksFloatsRun)
constexpr uint32_t kSimdPerGroup = 4;    // simdgroups a threadgroup

// The register kernel's instance for n columns: 8, 16 or 32.
uint32_t simd_instance(uint32_t n) { return n <= 8 ? 8 : n <= 16 ? 16 : 32; }

constexpr uint32_t kChaserThreads = 32;   // simdgroup 0 in overlap mode; kChaser in the shader

// From this k the QR iteration runs on a simdgroup of its own, computing the
// next step while the rows apply this one (kOverlap in the shader), as the ql
// backend does from N = 33. Measured on an M5 Pro: 14% faster at 48 x 48,
// 6% slower at 32 x 32, within noise below.
constexpr uint32_t kOverlapMinK = 33;

// From this k, the work is split in two dispatches (kPart in the shader): the
// reduction, then the QR iteration in threadgroups with almost no threadgroup
// memory, so that many matrices overlap their iterations. Measured on an M5
// Pro: for singular values alone, where the iteration is most of the work,
// 1.3x faster at 48 x 48 and 1.55x at 64 x 64, about even at 40; with vectors,
// where forming Q and V in the matrix-sized threadgroups dominates, 9% faster
// at 64 x 64 and 80 x 80, even or slightly slower below 56.
constexpr uint32_t kSplitMinKValues  = 40;
constexpr uint32_t kSplitMinKVectors = 60;

// Threadgroup memory for B with m >= n rows, as laid out in
// Svd_GolubKahan.metal; V lives in device memory.
size_t tg_bytes(uint32_t m, uint32_t n) {
    const size_t ld = n | 1u;
    const size_t floats = (size_t)m * ld + 13 * (size_t)n + 64 + 32;
    return (floats * sizeof(float) + 15) / 16 * 16;
}

uint32_t threads_for(uint32_t m, uint32_t n, bool vectors, bool overlap) {
    return (overlap ? kChaserThreads : 0) + pad_up(m + (vectors ? n : 0), 32);
}

// Cost model for chunking: core-milliseconds per matrix, about this times
// (m + n) n^2, generous as the ql backend's is, so that a slower GPU still
// gets command buffers well inside the watchdog.
constexpr double kCoreMsPerMN2  = 4e-6;
constexpr double kChunkBudgetMs = 750.0;   // wall time per command buffer

struct Workspace {
    uint32_t      capacity = 0;        // matrices the buffers hold
    id<MTLBuffer> s, u, vt, info, v;   // v: the kernel's V, [batch, K, K]
    id<MTLBuffer> uw, hw;              // the split's U [batch, K, max(M, N)] and header [batch, 2K + 2]
};

struct Cache {
    MetalRuntime& rt = MetalRuntime::shared(METAL_LINALG_SHADER(Svd_GolubKahan), "svd_golub_kahan");
    std::map<std::tuple<bool, bool, uint32_t>, id<MTLComputePipelineState>> pipelines;   // (vectors, overlap, part)
    std::map<std::tuple<bool, uint32_t, uint32_t, bool>, id<MTLComputePipelineState>> simd;   // (vectors, columns, lanes, run)
    std::map<std::pair<uint32_t, uint32_t>, Workspace>               workspaces;   // (M, N)

    id<MTLComputePipelineState> simd_pipeline(bool vectors, uint32_t instance, uint32_t lanes, bool run = false) {
        const auto key = std::make_tuple(vectors, instance, lanes, run);
        if (auto it = simd.find(key); it != simd.end()) return it->second;
        MTLFunctionConstantValues* cv = [[MTLFunctionConstantValues alloc] init];
        const bool overlap = false;
        const uint32_t part = 0;
        [cv setConstantValue:&vectors type:MTLDataTypeBool atIndex:0];
        [cv setConstantValue:&overlap type:MTLDataTypeBool atIndex:1];
        [cv setConstantValue:&part type:MTLDataTypeUInt atIndex:2];
        NSString* name = [NSString stringWithFormat:@"svd_gk_simd_%u_%u%s", instance, lanes, run ? "_run" : ""];
        return simd[key] = make_pipeline(rt.device, rt.library, name, cv);
    }

    id<MTLComputePipelineState> pipeline(bool vectors, bool overlap, uint32_t part = 0) {
        const auto key = std::make_tuple(vectors, overlap, part);
        if (auto it = pipelines.find(key); it != pipelines.end()) return it->second;
        MTLFunctionConstantValues* cv = [[MTLFunctionConstantValues alloc] init];
        [cv setConstantValue:&vectors type:MTLDataTypeBool atIndex:0];
        [cv setConstantValue:&overlap type:MTLDataTypeBool atIndex:1];
        [cv setConstantValue:&part type:MTLDataTypeUInt atIndex:2];
        return pipelines[key] = make_pipeline(rt.device, rt.library, @"svd_golub_kahan", cv);
    }

    // Buffers for at least `batch` matrices of M x N: the latest shape's are
    // kept and grown, so that the chunks of a shared batch (share_batch) and
    // calls with varying batches reuse them.
    Workspace workspace(uint32_t batch, uint32_t M, uint32_t N) {
        const auto key = std::make_pair(M, N);
        if (auto it = workspaces.find(key); it != workspaces.end() && it->second.capacity >= batch) return it->second;
        const MTLResourceOptions opt = MTLResourceStorageModeShared;
        const size_t K = std::min(M, N), f = sizeof(float);
        auto buf = [&](size_t bytes) { return [rt.device newBufferWithLength:std::max<size_t>(16, bytes) options:opt]; };
        Workspace w;
        w.s    = buf((size_t)batch * K * f);
        w.u    = buf((size_t)batch * M * K * f);
        w.vt   = buf((size_t)batch * K * N * f);
        w.info = buf((size_t)batch * sizeof(uint32_t));
        w.v    = buf((size_t)batch * K * K * f);
        w.uw   = buf((size_t)batch * K * std::max(M, N) * f);
        w.hw   = buf((size_t)batch * (2 * K + 2) * f);
        w.capacity = batch;
        // One workspace per shape is kept; drop the rest so that a sweep over
        // many shapes does not hold them all.
        workspaces.clear();
        return workspaces[key] = w;
    }

    static Cache& shared() {
        static Cache c;
        return c;
    }
};

unsigned env_uint(const char* name, unsigned fallback) {
    if (const char* s = std::getenv(name)) {
        const long v = std::strtol(s, nullptr, 10);
        if (v > 0) return (unsigned)v;
    }
    return fallback;
}

// The device's limits: threadgroup memory, and threads per threadgroup as the
// least of the four pipelines allows. Zero without a Metal device.
struct Limits {
    size_t   memory  = 0;
    uint32_t threads = 0;
};

const Limits& limits() {
    static const Limits lim = [] {
        Limits l;
        @autoreleasepool {
            id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
            if (!dev) return l;
            l.memory = dev.maxThreadgroupMemoryLength;
            Cache& cache = Cache::shared();
            uint32_t t = 1024;
            for (bool v : {false, true})
                for (bool o : {false, true})
                    t = std::min<uint32_t>(t, (uint32_t)cache.pipeline(v, o).maxTotalThreadsPerThreadgroup);
            l.threads = t;
        }
        return l;
    }();
    return lim;
}

bool fits(uint32_t rows, uint32_t cols) {
    const uint32_t m = std::max(rows, cols), n = std::min(rows, cols);
    if (n == 0) return true;
    const Limits& l = limits();
    // With vectors, which needs more, so that one answer serves both.
    return tg_bytes(m, n) <= l.memory && threads_for(m, n, true, n >= kOverlapMinK) <= l.threads;
}

} // namespace

namespace detail {

unsigned svd_gk_max_k() {
    static const unsigned max_k = [] {
        unsigned k = 0;
        while (k < 1024 && fits(k + 1, k + 1)) ++k;
        return k;
    }();
    return max_k;
}

bool svd_gk_fits(unsigned rows, unsigned cols) { return fits(rows, cols); }

} // namespace detail

namespace core::detail {

void svd_golub_kahan(const Matrices& a, float* u_out, float* s_out, float* vt_out, uint32_t* info_out) {
    const uint32_t M = a.rows, N = a.cols, K = std::min(M, N), batch = a.batch;
    if (K == 0 || batch == 0) {
        if (info_out) std::fill(info_out, info_out + batch, 0u);
        return;
    }
    if (!fits(M, N)) {
        throw std::invalid_argument("[svd] The golub_kahan backend takes " + std::to_string(M) + "x" +
                                    std::to_string(N) + " only through the QR first on this device (the "
                                    "matrix and V live in threadgroup memory; the largest square is " +
                                    std::to_string(metal_linalg::detail::svd_gk_max_k()) + ").");
    }
    AutoreleasePool pool;
    Cache& cache = Cache::shared();
    const bool vectors = u_out || vt_out;
    const uint32_t m = std::max(M, N), n = K;
    // In registers, up to 32 rows and columns, four matrices a simdgroup up
    // to 8 rows and two up to 16 (SVD_GK_SIMD=0: the threadgroup-memory
    // kernel throughout); singular values alone by bisection there, a lane a
    // value. On an M5 Pro, with vectors 1.6-2.1x the threadgroup kernel up to
    // 16 x 16 and 1.0-1.35x at 32 x 32; values alone 1.6-2.7x.
    const char* simd_env = std::getenv("SVD_GK_SIMD");
    const bool in_registers = m <= kSimdMaxDim && !(simd_env && std::string(simd_env) == "0");
    const bool overlap = vectors && n >= kOverlapMinK;
    id<MTLComputePipelineState> pso = cache.pipeline(vectors, overlap);
    const uint32_t threads = threads_for(m, n, vectors, overlap);

    // From kSplitMinK, two dispatches (kPart in the shader): the reduction,
    // then the QR iteration in threadgroups with almost no threadgroup memory.
    // A pipeline that allows fewer threads than either needs keeps the fused
    // kernel.
    bool split = !in_registers && n >= (vectors ? kSplitMinKVectors : kSplitMinKValues);
    id<MTLComputePipelineState> pso1 = nil, pso2 = nil;
    const uint32_t threads1 = threads_for(m, n, vectors, false);
    const uint32_t threads2 = vectors ? threads : kChaserThreads;
    if (split) {
        pso1 = cache.pipeline(vectors, false, 1);
        pso2 = cache.pipeline(vectors, overlap, 2);
        split = threads1 <= pso1.maxTotalThreadsPerThreadgroup && threads2 <= pso2.maxTotalThreadsPerThreadgroup;
    }

    // Work per command buffer: a large batch is cut into chunks by a
    // conservative cost model, never smaller than the core count.
    unsigned cores = gpu_core_count();
    if (cores == 0) cores = 8;
    const double per_matrix = kCoreMsPerMN2 * ((double)m + n) * n * n + 0.002;
    const double budget = (double)env_uint("SVD_CHUNK_MS", (unsigned)kChunkBudgetMs);
    const uint32_t chunk = (uint32_t)std::min<double>(
        batch, std::max<double>(cores, std::floor(budget * cores / per_matrix)));

    Workspace ws = cache.workspace(batch, M, N);
    id<MTLBuffer> src = input_buffer(cache.rt.device, a);

    // LAPACK sbdsqr's budget: 6 n^2 rotations in all.
    GkParams prm{m, n, M < N ? 1u : 0u, 6u * std::max(n * n, 1u)};
    const size_t f = sizeof(float), mat = (size_t)M * N;
    const size_t tgm = tg_bytes(m, n);
    for (uint32_t b0 = 0; b0 < batch; b0 += chunk) {
        const uint32_t bc = std::min(chunk, batch - b0);
        id<MTLCommandBuffer> cmd = [cache.rt.queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setBuffer:src     offset:(size_t)b0 * mat * f atIndex:0];
        [enc setBuffer:ws.s    offset:(size_t)b0 * K * f atIndex:1];
        [enc setBuffer:ws.u    offset:(size_t)b0 * M * K * f atIndex:2];
        [enc setBuffer:ws.vt   offset:(size_t)b0 * K * N * f atIndex:3];
        [enc setBuffer:ws.info offset:(size_t)b0 * sizeof(uint32_t) atIndex:4];
        [enc setBytes:&prm length:sizeof prm atIndex:5];
        [enc setBuffer:ws.v    offset:(size_t)b0 * K * K * f atIndex:6];
        [enc setBuffer:ws.uw   offset:(size_t)b0 * K * m * f atIndex:7];
        [enc setBuffer:ws.hw   offset:(size_t)b0 * (2 * K + 2) * f atIndex:8];
        if (in_registers) {
            // 32 / lanes matrices a simdgroup (lanes: at least the rows, 8, 16
            // or 32); as many simdgroups a threadgroup as the pipeline allows,
            // up to kSimdPerGroup
            const uint32_t lanes = simd_instance(m);
            // From 17 rows with vectors, a runner simdgroup runs the QR
            // iterations of a threadgroup's matrices, a simdgroup each (on an
            // M5 Pro 1.1-1.35x; SVD_GK_RUN=0 turns it off): 8 matrices a
            // runner from 17 columns, 4 below (more leave too few threadgroups
            // for a batch of 1024, fewer too much runner)
            const char* run_env = std::getenv("SVD_GK_RUN");
            const bool run = lanes == 32 && vectors && !(run_env && std::string(run_env) == "0");
            id<MTLComputePipelineState> ps = cache.simd_pipeline(vectors, simd_instance(n), lanes, run);
            const uint32_t cap = (uint32_t)ps.maxTotalThreadsPerThreadgroup / 32;
            const uint32_t per = run ? std::clamp<uint32_t>((n > 16 ? 8u : 4u) + 1, 2, cap)
                                     : std::clamp<uint32_t>(cap, 1, kSimdPerGroup);
            const uint32_t mats = run ? per - 1 : per * (32 / lanes);   // matrices a threadgroup
            const GksParams q{m, n, prm.transpose, prm.max_rots, bc};
            [enc setComputePipelineState:ps];
            [enc setBytes:&q length:sizeof q atIndex:5];
            [enc setThreadgroupMemoryLength:(size_t)mats * (run ? kSimdFloatsRun : kSimdFloats) * sizeof(float) atIndex:0];
            [enc dispatchThreadgroups:MTLSizeMake((bc + mats - 1) / mats, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(32 * per, 1, 1)];
        } else if (!split) {
            [enc setComputePipelineState:pso];
            [enc setThreadgroupMemoryLength:tgm atIndex:0];
            [enc dispatchThreadgroups:MTLSizeMake(bc, 1, 1) threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        } else {
            // Serial dispatches in one encoder: part 2 sees part 1's writes.
            [enc setComputePipelineState:pso1];
            [enc setThreadgroupMemoryLength:tgm atIndex:0];
            [enc dispatchThreadgroups:MTLSizeMake(bc, 1, 1) threadsPerThreadgroup:MTLSizeMake(threads1, 1, 1)];
            [enc setComputePipelineState:pso2];
            [enc setThreadgroupMemoryLength:tg_bytes(0, n) atIndex:0];
            [enc dispatchThreadgroups:MTLSizeMake(bc, 1, 1) threadsPerThreadgroup:MTLSizeMake(threads2, 1, 1)];
        }
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        if (cmd.error) {
            throw std::runtime_error(std::string("[svd] golub_kahan: GPU error: ") +
                                     cmd.error.localizedDescription.UTF8String + " (" + std::to_string(M) +
                                     "x" + std::to_string(N) + ", " + std::to_string(bc) +
                                     " matrices in this command buffer).");
        }
    }

    // Non-finite input is reported through the output (NaN), as on the CPU; a
    // finite matrix that ran out of iterations is raised, as the Jacobi
    // backends do. U needs no completion: its columns are orthonormal by
    // construction, rank-deficient or not.
    const uint32_t* info = static_cast<const uint32_t*>([ws.info contents]);
    for (uint32_t b = 0; b < batch; ++b) {
        const uint32_t word = info[b];
        if (!metal_linalg::detail::svd_converged(word) && !metal_linalg::detail::svd_nonfinite(word)) {
            throw std::runtime_error("[svd] golub_kahan: matrix " + std::to_string(b) + " of " +
                                     std::to_string(batch) + " (" + std::to_string(M) + "x" +
                                     std::to_string(N) + ") did not converge in " +
                                     std::to_string(prm.max_rots) + " rotations.");
        }
    }
    copy_out(static_cast<const float*>([ws.s contents]), s_out, batch, K);
    if (u_out)  copy_out(static_cast<const float*>([ws.u contents]), u_out, batch, (size_t)M * K);
    if (vt_out) copy_out(static_cast<const float*>([ws.vt contents]), vt_out, batch, (size_t)K * N);
    if (info_out) std::memcpy(info_out, info, (size_t)batch * sizeof(uint32_t));
}

} // namespace core::detail
} // namespace metal_linalg
