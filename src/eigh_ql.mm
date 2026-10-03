// The `ql` eigensolver backend: LAPACK's method -- Householder
// tridiagonalization, then implicit QL -- one threadgroup per matrix with the
// whole matrix in threadgroup memory (shaders/Eigh_QL.metal).
//
// Why: for batches of small and mid-size matrices the Jacobi kernels do
// several times the flops of LAPACK's method, and on an M5 Pro a batch spread
// over the CPU's cores beat them from N = 32. Recording each QL sweep's
// rotations and applying them a row per thread removes the barrier per
// rotation that made QL look unsuited to the GPU. Threadgroup memory bounds N
// (87 at 32 KB) and also how many matrices share a core, which is what the
// latency-bound QL iteration needs, so it is sized for the N of the call.

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

// Must match `QlParams` in Eigh_QL.metal.
struct QlParams {
    uint32_t n;
    uint32_t lower;
    uint32_t max_iter;
};

constexpr uint32_t kChaserThreads = 32;   // simdgroup 0 in overlap mode; kChaser in the shader
constexpr uint32_t kMaxRowThreads = 96;   // with the chaser, at most four simdgroups

// From this N the QL iteration runs on a simdgroup of its own, computing the
// next sweep while the rows apply this one (kOverlap in the shader). Measured
// on an M5 Pro: 9-13% faster from N = 48, 4-11% slower at N = 16-32, where the
// extra simdgroup costs more in matrices per core than the overlap saves.
constexpr uint32_t kOverlapMinN = 33;

// Threadgroup memory for order n, as laid out in Eigh_QL.metal.
size_t tg_bytes(uint32_t n) {
    const size_t ld = n | 1u;
    const size_t floats = (size_t)n * ld + 6 * (size_t)n + 8 + 8;
    return (floats * sizeof(float) + 15) / 16 * 16;
}

// Cost model for chunking: core-milliseconds per matrix, about this times N^3.
// Measured at about 8e-7 on an M5 Pro at N = 64; five times that, so that a
// slower GPU still gets command buffers well inside the watchdog.
constexpr double kCoreMsPerN3   = 4e-6;
constexpr double kChunkBudgetMs = 750.0;   // wall time per command buffer

struct Workspace {
    id<MTLBuffer> vals, vecs, info;
};

struct Cache {
    MetalRuntime& rt = MetalRuntime::shared(METAL_LINALG_SHADER(Eigh_QL), "eigh_ql");
    std::map<std::pair<bool, bool>, id<MTLComputePipelineState>> pipelines;  // (vectors, overlap)
    std::map<std::pair<uint32_t, uint32_t>, Workspace>        workspaces;   // (batch, n)

    id<MTLComputePipelineState> pipeline(bool vectors, bool overlap) {
        const auto key = std::make_pair(vectors, overlap);
        if (auto it = pipelines.find(key); it != pipelines.end()) return it->second;
        MTLFunctionConstantValues* cv = [[MTLFunctionConstantValues alloc] init];
        [cv setConstantValue:&vectors type:MTLDataTypeBool atIndex:0];
        [cv setConstantValue:&overlap type:MTLDataTypeBool atIndex:1];
        return pipelines[key] = make_pipeline(rt.device, rt.library, @"eigh_ql", cv);
    }

    Workspace workspace(uint32_t batch, uint32_t n) {
        const auto key = std::make_pair(batch, n);
        if (auto it = workspaces.find(key); it != workspaces.end()) return it->second;
        const MTLResourceOptions opt = MTLResourceStorageModeShared;
        Workspace w;
        w.vals = [rt.device newBufferWithLength:std::max<size_t>(16, (size_t)batch * n * sizeof(float)) options:opt];
        w.vecs = [rt.device newBufferWithLength:std::max<size_t>(16, (size_t)batch * n * n * sizeof(float)) options:opt];
        w.info = [rt.device newBufferWithLength:std::max<size_t>(16, (size_t)batch * sizeof(uint32_t)) options:opt];
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

} // namespace

namespace detail {

unsigned eigh_ql_max_n() {
    static const unsigned max_n = [] {
        unsigned limit = 0;
        @autoreleasepool {
            id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
            if (!dev) return 0u;
            const size_t mem = dev.maxThreadgroupMemoryLength;
            for (unsigned n = 1; n <= kMaxRowThreads && tg_bytes(n) <= mem; ++n) limit = n;
        }
        return limit;
    }();
    return max_n;
}

} // namespace detail

namespace core::detail {

void eigh_ql(const Matrices& a, bool lower, float* w_out, float* v_out, uint32_t* info_out) {
    const uint32_t n = a.cols, batch = a.batch;
    if (a.rows != n) {
        throw std::invalid_argument("[eigh] Input matrices must be square.");
    }
    if (n > metal_linalg::detail::eigh_ql_max_n()) {
        throw std::invalid_argument("[eigh] The ql backend takes N <= " +
                                    std::to_string(metal_linalg::detail::eigh_ql_max_n()) +
                                    " on this device (the matrix lives in threadgroup memory); N = " +
                                    std::to_string(n) + ".");
    }
    if (n == 0 || batch == 0) {
        if (info_out) std::fill(info_out, info_out + batch, 0u);
        return;
    }
    AutoreleasePool pool;
    Cache& cache = Cache::shared();
    const bool vectors = v_out != nullptr;
    const bool overlap = n >= kOverlapMinN;
    id<MTLComputePipelineState> pso = cache.pipeline(vectors, overlap);
    const uint32_t threads = (overlap ? kChaserThreads : 0) + pad_up(n, 32);
    if (threads > pso.maxTotalThreadsPerThreadgroup) {
        throw std::runtime_error("[eigh] The ql pipeline allows " +
                                 std::to_string((unsigned)pso.maxTotalThreadsPerThreadgroup) +
                                 " threads per threadgroup; N = " + std::to_string(n) + " needs " +
                                 std::to_string(threads) + ".");
    }

    // Work per command buffer: macOS stops a command buffer that holds the
    // GPU for more than a couple of seconds, so a large batch is cut into
    // chunks by a conservative cost model, never smaller than the core count.
    unsigned cores = gpu_core_count();
    if (cores == 0) cores = 8;
    const double per_matrix = kCoreMsPerN3 * (double)n * n * n + 0.002;
    const double budget = (double)env_uint("EIGH_CHUNK_MS", (unsigned)kChunkBudgetMs);
    const uint32_t chunk = (uint32_t)std::min<double>(
        batch, std::max<double>(cores, std::floor(budget * cores / per_matrix)));

    Workspace ws = cache.workspace(batch, n);
    id<MTLBuffer> src = input_buffer(cache.rt.device, a);

    QlParams prm{n, lower ? 1u : 0u, 30u * std::max(n, 1u)};   // LAPACK ssteqr's budget
    const size_t mat_bytes = (size_t)n * n * sizeof(float);
    const size_t tgm = tg_bytes(n);
    for (uint32_t b0 = 0; b0 < batch; b0 += chunk) {
        const uint32_t bc = std::min(chunk, batch - b0);
        id<MTLCommandBuffer> cmd = [cache.rt.queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:src     offset:b0 * mat_bytes atIndex:0];
        [enc setBuffer:ws.vals offset:(size_t)b0 * n * sizeof(float) atIndex:1];
        [enc setBuffer:ws.vecs offset:b0 * mat_bytes atIndex:2];
        [enc setBuffer:ws.info offset:(size_t)b0 * sizeof(uint32_t) atIndex:3];
        [enc setBytes:&prm length:sizeof prm atIndex:4];
        [enc setThreadgroupMemoryLength:tgm atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(bc, 1, 1) threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        if (cmd.error) {
            throw std::runtime_error(std::string("[eigh] ql: GPU error: ") +
                                     cmd.error.localizedDescription.UTF8String + " (N=" + std::to_string(n) +
                                     ", " + std::to_string(bc) + " matrices in this command buffer).");
        }
    }

    // Non-finite input is reported through the output (NaN), as on the CPU; a
    // finite matrix that ran out of iterations is raised, as the Jacobi
    // backends do.
    const uint32_t* info = static_cast<const uint32_t*>([ws.info contents]);
    for (uint32_t b = 0; b < batch; ++b) {
        const uint32_t word = info[b];
        if (!metal_linalg::detail::eigh_converged(word) && !metal_linalg::detail::eigh_nonfinite(word)) {
            throw std::runtime_error("[eigh] ql: matrix " + std::to_string(b) + " of " + std::to_string(batch) +
                                     " (N=" + std::to_string(n) + ") did not converge in " +
                                     std::to_string(prm.max_iter) + " QL iterations.");
        }
    }
    copy_out(static_cast<const float*>([ws.vals contents]), w_out, batch, n);
    if (vectors) copy_out(static_cast<const float*>([ws.vecs contents]), v_out, batch, (size_t)n * n);
    if (info_out) std::memcpy(info_out, info, (size_t)batch * sizeof(uint32_t));
}

} // namespace core::detail
} // namespace metal_linalg
