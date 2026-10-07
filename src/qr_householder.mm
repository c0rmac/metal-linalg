// QR for batches of small matrices: LAPACK's method (sgeqr2, sorg2r), one
// threadgroup per matrix with the whole matrix in threadgroup memory
// (shaders/QR_Householder.metal), built as the SVD's golub_kahan and the
// eigensolver's ql backends are. qr_unblocked hands it every call it fits.
//
// Why: qr_unblocked's kernel walked device memory with a barrier per phase of
// every column, one thread building T, the matrix padded to 32 rows and a
// full padded square Q; and around it the CPU scanned and copied the input
// and copied R and Q back, which at 4096 x 16 x 16 took as long as the CPU
// path's whole call. Here the kernel reads the caller's row-major input as it
// is, scans and scales it itself, and writes Q and R row-major, into the
// caller's memory where it is page-aligned.

#include <metal_linalg/core.h>
#include <metal_linalg/device.h>
#include "metal_runtime.h"
#include "shaders.h"

#import <Metal/Metal.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <tuple>
#include <utility>
#include <stdexcept>
#include <string>
#include <unistd.h>

using metal_linalg::core::Matrices;
using metal_linalg::detail::AutoreleasePool;
using metal_linalg::detail::MetalRuntime;
using metal_linalg::detail::copy_out;
using metal_linalg::detail::input_buffer;
using metal_linalg::detail::make_pipeline;
using metal_linalg::detail::pad_up;

namespace metal_linalg {
namespace {

// Must match `QhParams` and `QsParams` in QR_Householder.metal.
struct QhParams {
    uint32_t m;
    uint32_t n;
};
struct QsParams {
    uint32_t m;
    uint32_t n;
    uint32_t batch;
};

// The register kernel's instance for m x n: B columns (8, 16, 32, 64) and R
// rows a lane (1, 2, 4), at most 128 floats a lane; {0, 0} if none fits. Must
// match the instances in QR_Householder.metal.
struct SimdShape {
    uint32_t b = 0, r = 0;
};

SimdShape simd_shape(uint32_t m, uint32_t n) {
    SimdShape sh;
    for (uint32_t b : {8u, 16u, 32u, 64u})
        if (n <= b) { sh.b = b; break; }
    for (uint32_t r : {1u, 2u, 4u})
        if (m <= 32 * r) { sh.r = r; break; }
    if (!sh.b || !sh.r || sh.b * sh.r > 128) return SimdShape{};
    return sh;
}

// Matrices (simdgroups) per threadgroup of the register kernel: one alone
// halved its speed at 8 columns, two to sixteen were alike (M5 Pro).
constexpr uint32_t kSimdPerGroup = 4;

// Threadgroup memory for an m x n matrix, as laid out in QR_Householder.metal.
size_t tg_bytes(uint32_t m, uint32_t n) {
    const size_t ld = n | 1u, K = std::min(m, n);
    const size_t floats = (size_t)m * ld + 2 * K + n + 64;
    return (floats * sizeof(float) + 15) / 16 * 16;
}

uint32_t threads_for(uint32_t m) { return pad_up(std::max(m, 1u), 32); }

// Cost model for chunking: core-milliseconds per matrix, about this times
// m n^2, generous, so that a slower GPU still gets command buffers well inside
// the watchdog.
constexpr double kCoreMsPerMN2  = 4e-6;
constexpr double kChunkBudgetMs = 750.0;   // wall time per command buffer

struct Workspace {
    uint32_t      capacity = 0;   // matrices the buffers hold
    id<MTLBuffer> q, r;
};

struct Cache {
    MetalRuntime& rt = MetalRuntime::shared(METAL_LINALG_SHADER(QR_Householder), "qr_householder");
    id<MTLComputePipelineState> pso = make_pipeline(rt.device, rt.library, @"qr_householder", nil);
    std::map<std::pair<uint32_t, uint32_t>, id<MTLComputePipelineState>> simd;
    uint32_t  m = 0, n = 0;
    Workspace ws;

    id<MTLComputePipelineState> simd_pipeline(SimdShape sh) {
        const auto key = std::make_pair(sh.b, sh.r);
        if (auto it = simd.find(key); it != simd.end()) return it->second;
        NSString* name = [NSString stringWithFormat:@"qr_householder_simd_%u_%u", sh.b, sh.r];
        return simd[key] = make_pipeline(rt.device, rt.library, name, nil);
    }

    // Output buffers for a caller's memory that is not page-aligned: the
    // latest shape's, grown as needed.
    Workspace& workspace(uint32_t batch, uint32_t M, uint32_t N) {
        if (ws.capacity >= batch && m == M && n == N) return ws;
        const size_t K = std::min(M, N);
        auto buf = [&](size_t floats) {
            return [rt.device newBufferWithLength:std::max<size_t>(16, floats * sizeof(float))
                                          options:MTLResourceStorageModeShared];
        };
        ws.q = buf((size_t)batch * M * K);
        ws.r = buf((size_t)batch * K * N);
        ws.capacity = batch;
        m = M;
        n = N;
        return ws;
    }

    static Cache& shared() {
        static Cache c;
        return c;
    }
};

bool fits_device(uint32_t m, uint32_t n) {
    static const size_t memory = [] {
        @autoreleasepool {
            id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
            return dev ? (size_t)dev.maxThreadgroupMemoryLength : (size_t)0;
        }
    }();
    if (memory == 0) return false;
    return tg_bytes(m, n) <= memory &&
           threads_for(m) <= (uint32_t)Cache::shared().pso.maxTotalThreadsPerThreadgroup;
}

unsigned env_uint(const char* name, unsigned fallback) {
    if (const char* s = std::getenv(name)) {
        const long v = std::strtol(s, nullptr, 10);
        if (v > 0) return (unsigned)v;
    }
    return fallback;
}

} // namespace

namespace core::detail {

bool qr_householder_fits(uint32_t m, uint32_t n) {
    if (std::min(m, n) == 0) return true;
    return simd_shape(m, n).b != 0 || fits_device(m, n);
}

// In registers wherever they fit; in threadgroup memory only for narrow
// matrices (n <= 32), where it beats qr_unblocked's kernel 3x and the blocked
// QR (on an M5 Pro, 1024 of 200 x 30: 3.2 ms against 9.5 and 5.3), and not
// for wider ones, where the blocked QR is the faster (256 of 80 x 80: 3.2
// against 1.8). QR_HOUSEHOLDER=0 turns it off.
bool qr_householder_preferred(uint32_t m, uint32_t n) {
    if (const char* e = std::getenv("QR_HOUSEHOLDER"); e && std::string(e) == "0") return false;
    if (std::min(m, n) == 0) return false;
    return simd_shape(m, n).b != 0 || (n <= 32 && fits_device(m, n));
}

void qr_householder(const Matrices& a, float* q_out, float* r_out) {
    const uint32_t M = a.rows, N = a.cols, K = std::min(M, N), batch = a.batch;
    if (K == 0 || batch == 0) return;
    AutoreleasePool pool;
    Cache& cache = Cache::shared();
    id<MTLDevice> dev = cache.rt.device;

    // Q and R straight into the caller's memory where it is page-aligned,
    // else into a workspace and copied.
    const size_t page = (size_t)getpagesize(), f = sizeof(float);
    const size_t qn = (size_t)batch * M * K, rn = (size_t)batch * K * N;
    const bool q_direct = reinterpret_cast<uintptr_t>(q_out) % page == 0;
    const bool r_direct = reinterpret_cast<uintptr_t>(r_out) % page == 0;
    id<MTLBuffer> qb = nil, rb = nil;
    if (!q_direct || !r_direct) {
        Workspace& ws = cache.workspace(batch, M, N);
        qb = ws.q;
        rb = ws.r;
    }
    if (q_direct) qb = metal_linalg::detail::wrap_host(dev, q_out, qn);
    if (r_direct) rb = metal_linalg::detail::wrap_host(dev, r_out, rn);
    id<MTLBuffer> src = input_buffer(dev, a);

    // Work per command buffer: a large batch is cut into chunks by a
    // conservative cost model, never smaller than the core count.
    unsigned cores = gpu_core_count();
    if (cores == 0) cores = 8;
    const double per_matrix = kCoreMsPerMN2 * (double)M * N * K + 0.002;
    const double budget = (double)env_uint("QR_CHUNK_MS", (unsigned)kChunkBudgetMs);
    const uint32_t chunk = (uint32_t)std::min<double>(
        batch, std::max<double>(cores, std::floor(budget * cores / per_matrix)));

    // In registers where the shape allows (QR_HOUSEHOLDER_SIMD=0 turns it
    // off), else in threadgroup memory
    const char* se = std::getenv("QR_HOUSEHOLDER_SIMD");
    const SimdShape sh = se && std::string(se) == "0" ? SimdShape{} : simd_shape(M, N);
    if (!sh.b && !fits_device(M, N))
        throw std::invalid_argument("[qr] householder: " + std::to_string(M) + "x" + std::to_string(N) +
                                    " does not fit in threadgroup memory on this device.");
    const QhParams prm{M, N};
    const uint32_t threads = threads_for(M);
    const size_t tgm = tg_bytes(M, N);
    id<MTLComputePipelineState> spso = sh.b ? cache.simd_pipeline(sh) : nil;
    for (uint32_t b0 = 0; b0 < batch; b0 += chunk) {
        const uint32_t bc = std::min(chunk, batch - b0);
        id<MTLCommandBuffer> cmd = [cache.rt.queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setBuffer:src offset:(size_t)b0 * M * N * f atIndex:0];
        [enc setBuffer:qb offset:(size_t)b0 * M * K * f atIndex:1];
        [enc setBuffer:rb offset:(size_t)b0 * K * N * f atIndex:2];
        if (sh.b) {
            const QsParams sp{M, N, bc};
            [enc setComputePipelineState:spso];
            [enc setBytes:&sp length:sizeof sp atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake((bc + kSimdPerGroup - 1) / kSimdPerGroup, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(32 * kSimdPerGroup, 1, 1)];
        } else {
            [enc setComputePipelineState:cache.pso];
            [enc setBytes:&prm length:sizeof prm atIndex:3];
            [enc setThreadgroupMemoryLength:tgm atIndex:0];
            [enc dispatchThreadgroups:MTLSizeMake(bc, 1, 1) threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        }
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        if (cmd.error) {
            throw std::runtime_error(std::string("[qr] householder: GPU error: ") +
                                     cmd.error.localizedDescription.UTF8String + " (" + std::to_string(M) + "x" +
                                     std::to_string(N) + ", " + std::to_string(bc) +
                                     " matrices in this command buffer).");
        }
    }
    if (!q_direct) copy_out(static_cast<const float*>([qb contents]), q_out, batch, (size_t)M * K);
    if (!r_direct) copy_out(static_cast<const float*>([rb contents]), r_out, batch, (size_t)K * N);
}

} // namespace core::detail
} // namespace metal_linalg
