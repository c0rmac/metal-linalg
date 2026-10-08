// QR for batches of small and mid-size matrices, one matrix to a simdgroup or
// a threadgroup (shaders/QR_Householder.metal), built as the SVD's
// golub_kahan and the eigensolver's ql backends are: the unblocked backend
// (qr_unblocked, at the end).
//
//   - in registers, a simdgroup a matrix, up to 32 columns and 128 rows:
//     LAPACK's unblocked method (sgeqr2, sorg2r);
//   - blocked, a threadgroup a matrix, up to 4096 rows: LAPACK's blocked
//     method (sgeqrf, sorgqr), panels of 16 columns in registers, the updates
//     by blocks of 32 columns as 8 x 8 simdgroup matrix products, the matrix
//     and Q in a device workspace.
//
// Why: the unblocked backend's own kernel (QR_Unblocked.metal, retired in
// 2.16.0) walked device memory with a barrier per phase of
// every column, one thread building T, the matrix padded to 32 rows and a
// full padded square Q; and around it the CPU scanned and copied the input
// and copied R and Q back, which at 4096 x 16 x 16 took as long as the CPU
// path's whole call. And the blocked QR (qr_blocked.mm), for a batch of
// mid-size matrices, spends its time in some forty dispatches a call. Here
// the kernels read the caller's row-major input as it is, scan and scale it
// themselves, and write Q and R row-major, into the caller's memory where it
// is page-aligned.

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
#include <stdexcept>
#include <string>
#include <utility>
#include <unistd.h>

using metal_linalg::core::Matrices;
using metal_linalg::core::QrMode;
using metal_linalg::detail::AutoreleasePool;
using metal_linalg::detail::MetalRuntime;
using metal_linalg::detail::copy_out;
using metal_linalg::detail::input_buffer;
using metal_linalg::detail::make_pipeline;

namespace metal_linalg {
namespace {

// Must match `QsParams` and `QwParams` in QR_Householder.metal.
struct QsParams {
    uint32_t m;
    uint32_t n;
    uint32_t batch;
    uint32_t q_cols;   // Q's columns: K, or 0 for R alone
};
struct QwParams {
    uint32_t m, n, mp, np, Kp, batch, q_direct, sc;
    uint32_t qn;   // Q's columns in its workspace: Kp, mp (Q square) or 0 (R alone)
    uint32_t qc;   // ... in the output: K, M or 0
    uint32_t rr;   // R's rows in the output: K, or M (Q square)
};

unsigned env_uint(const char* name, unsigned fallback) {
    if (const char* s = std::getenv(name)) {
        const long v = std::strtol(s, nullptr, 10);
        if (v > 0) return (unsigned)v;
    }
    return fallback;
}

uint32_t round_up(uint32_t x, uint32_t to) { return (x + to - 1) / to * to; }

// The register kernel's instance for m x n: B columns (8, 16, 32) and R rows
// a lane (1, 2, 4); {0, 0} if none fits. Must match the instances in
// QR_Householder.metal. (Instances of 64 columns lost to the blocked kernel:
// 4096 of 48 x 48 in 3.3 ms of GPU time against 2.3, of 64 x 64 4.4 against
// 3.8, on an M5 Pro.)
struct SimdShape {
    uint32_t b = 0, r = 0;
};

SimdShape simd_shape(uint32_t m, uint32_t n) {
    SimdShape sh;
    for (uint32_t b : {8u, 16u, 32u})
        if (n <= b) { sh.b = b; break; }
    for (uint32_t r : {1u, 2u, 4u})
        if (m <= 32 * r) { sh.r = r; break; }
    if (!sh.b || !sh.r) return SimdShape{};
    return sh;
}

// Matrices (simdgroups) per threadgroup of the register kernel: one alone
// halved its speed at 8 columns, two to sixteen were alike (M5 Pro).
constexpr uint32_t kSimdPerGroup = 4;

// The blocked kernel's shape for a batch of m x n: the matrix padded to mp x
// np (multiples of 8, at least Kp = K rounded up to the panel width 16), R
// rows a thread and S simdgroups (at least enough for the rows). Two rows a
// thread, four for narrow matrices (n <= 32, where the panels are most of the
// work) or where two would take more simdgroups than a threadgroup has; one
// was 1.1-1.2x slower at 96-512 in large batches, four 1.1-1.4x slower there
// and 1.4x faster at 1000 x 20 (M5 Pro). But a small batch leaves the GPU
// short of simdgroups, and each matrix's panels are a latency-bound chain:
// there a matrix gets more of them, one row a thread, so that the batch has
// about 12 a GPU core, up to 16 a matrix (one 384 x 384 in 2.0 ms against 3.0,
// 16 of 256 x 256 in 1.2 against 1.7, 32 of them 1.7 against 2.1).
constexpr uint32_t kWyPanel = 16;   // QW_B
constexpr uint32_t kWyBlock = 32;   // QW_NB
constexpr uint32_t kWySmallBatchSimdgroups = 12;   // a GPU core, for a small batch
struct WyShape {
    uint32_t mp = 0, np = 0, Kp = 0, r = 0, s = 0, sc = 0;
};

uint32_t pow2_at_least(uint32_t x) {
    uint32_t p = 1;
    while (p < x) p *= 2;
    return p;
}

WyShape wy_shape(uint32_t m, uint32_t n, uint32_t max_simdgroups, uint32_t batch = 0xFFFFFFFFu,
                 uint32_t cores = 0) {
    WyShape w;
    const uint32_t K = std::min(m, n);
    w.Kp = round_up(K, kWyPanel);
    w.mp = std::max(round_up(m, 8), w.Kp);
    w.np = std::max(round_up(n, 8), w.Kp);
    w.r = w.np <= 32 ? 4 : 2;
    w.s = (w.mp + 32 * w.r - 1) / (32 * w.r);
    if (w.s > max_simdgroups && w.r < 4) {
        w.r = 4;
        w.s = (w.mp + 127) / 128;
    }
    if (w.s > max_simdgroups) return WyShape{};
    // A small batch: more simdgroups a matrix, the fewest rows a thread that
    // the 16 cover
    const uint32_t want = kWySmallBatchSimdgroups * std::max(cores, 1u);
    if (cores && (uint64_t)batch * w.s < want) {
        const uint32_t s = std::min({16u, max_simdgroups, pow2_at_least((want + batch - 1) / batch)});
        for (uint32_t r : {1u, 2u, 4u}) {
            if (s > w.s && 32 * r * s >= w.mp) {
                w.r = r;
                w.s = s;
                break;
            }
        }
    }
    // A wide matrix's updates are its work: a simdgroup for every 128 columns
    // too, up to 8 (64 of 64 x 1024 in 1.3 ms against 2.7 with the one its
    // rows ask for)
    w.s = std::max(w.s, std::min({max_simdgroups, 8u, w.np / 128}));
    // The updates' scratch: room for partial sums when a narrow matrix's few
    // column tiles leave simdgroups to split their rows (tall and narrow),
    // else the T merge's
    w.sc = w.np <= 128 && w.mp >= 512 ? 4096 : 1024;
    return w;
}

// Threadgroup memory of the blocked kernel, as laid out in
// QR_Householder.metal.
size_t wy_tg_bytes(const WyShape& w) { return (64 + 16 * (size_t)w.s + 8 * kWyBlock + 64 + w.sc) * sizeof(float); }

// Cost model for chunking: core-milliseconds per matrix, about this times
// m n min(m, n), generous (some 10x what an M5 Pro takes), so that a slower GPU
// still gets command buffers well inside the watchdog.
constexpr double kSimdCoreMs = 1e-6;
constexpr double kWyCoreMs = 4e-7;
constexpr double kChunkBudgetMs = 750.0;   // wall time per command buffer

struct Workspace {
    uint32_t      capacity = 0;   // matrices the buffers hold
    id<MTLBuffer> q, r;
};

// The blocked kernel's: the matrix, Q and the blocks' T, for up to so many
// floats each, grown as needed.
struct WyWork {
    size_t        a = 0, q = 0, t = 0;
    id<MTLBuffer> ba, bq, bt;
};

struct Cache {
    MetalRuntime& rt = MetalRuntime::shared(METAL_LINALG_SHADER(QR_Householder), "qr_householder");
    std::map<std::pair<uint32_t, uint32_t>, id<MTLComputePipelineState>> simd;
    id<MTLComputePipelineState> wy[3] = {nil, nil, nil};
    uint32_t  m = 0, n = 0;
    Workspace ws;
    WyWork    wyw;

    id<MTLComputePipelineState> simd_pipeline(SimdShape sh) {
        const auto key = std::make_pair(sh.b, sh.r);
        if (auto it = simd.find(key); it != simd.end()) return it->second;
        NSString* name = [NSString stringWithFormat:@"qr_householder_simd_%u_%u", sh.b, sh.r];
        return simd[key] = make_pipeline(rt.device, rt.library, name, nil);
    }

    id<MTLComputePipelineState> wy_pipeline(uint32_t r) {
        const int i = r == 1 ? 0 : r == 2 ? 1 : 2;
        if (!wy[i])
            wy[i] = make_pipeline(rt.device, rt.library, [NSString stringWithFormat:@"qr_householder_wy_%u", r], nil);
        return wy[i];
    }

    // Simdgroups a threadgroup of the blocked kernel takes (its four-row
    // instance's limit, the tightest)
    uint32_t wy_max_simdgroups() {
        static const uint32_t s = (uint32_t)wy_pipeline(4).maxTotalThreadsPerThreadgroup / 32;
        return s;
    }

    WyWork& wy_work(size_t a, size_t q, size_t t) {
        auto grow = [&](size_t& have, id<MTLBuffer> __strong& b, size_t want) {
            if (b && have >= want) return;
            b = [rt.device newBufferWithLength:std::max<size_t>(16, want * sizeof(float))
                                       options:MTLResourceStorageModePrivate];
            if (!b) throw std::runtime_error("[qr] householder: could not allocate its workspace");
            have = want;
        };
        grow(wyw.a, wyw.ba, a);
        grow(wyw.q, wyw.bq, q);
        grow(wyw.t, wyw.bt, t);
        return wyw;
    }

    // Output buffers for a caller's memory that is not page-aligned: the
    // latest shape's (Q's and R's floats a matrix), grown as needed.
    Workspace& workspace(uint32_t batch, uint32_t q_floats, uint32_t r_floats) {
        if (ws.capacity >= batch && m == q_floats && n == r_floats) return ws;
        auto buf = [&](size_t floats) {
            return [rt.device newBufferWithLength:std::max<size_t>(16, floats * sizeof(float))
                                          options:MTLResourceStorageModeShared];
        };
        ws.q = buf((size_t)batch * q_floats);
        ws.r = buf((size_t)batch * r_floats);
        ws.capacity = batch;
        m = q_floats;
        n = r_floats;
        return ws;
    }

    static Cache& shared() {
        static Cache c;
        return c;
    }
};

// The kernel for m x n: in registers where the shape allows
// (QR_HOUSEHOLDER_SIMD=0 turns it off) and Q is not square beyond its
// columns, else blocked;
// QR_HOUSEHOLDER_KERNEL=simd|wy forces one where it fits. The register kernel
// is the faster up to 32 columns (4096 of 32 x 32: 0.82 ms of GPU time against
// 1.59, of 128 x 16 0.54 against 1.46), the blocked one from 48.
enum class Kind { none, simd, wy };

Kind choose(uint32_t m, uint32_t n, QrMode mode = QrMode::reduced) {
    const char* se = std::getenv("QR_HOUSEHOLDER_SIMD");
    const bool simd = !(se && std::string(se) == "0") && simd_shape(m, n).b != 0 &&
                      !(mode == QrMode::complete && m > n);
    const bool wy = wy_shape(m, n, Cache::shared().wy_max_simdgroups()).r != 0;
    if (const char* e = std::getenv("QR_HOUSEHOLDER_KERNEL")) {
        const std::string k = e;
        if (k == "simd" && simd) return Kind::simd;
        if (k == "wy" && wy) return Kind::wy;
    }
    if (simd) return Kind::simd;
    return wy ? Kind::wy : Kind::none;
}

} // namespace

namespace core::detail {

bool qr_householder_fits(uint32_t m, uint32_t n) {
    if (std::min(m, n) == 0) return true;
    return choose(m, n) != Kind::none;
}

// Wherever a kernel takes it: the routing (m_crossover) decides between it
// and the blocked QR.
bool qr_householder_preferred(uint32_t m, uint32_t n) {
    if (std::min(m, n) == 0) return false;
    return choose(m, n) != Kind::none;
}

void qr_householder(const Matrices& a, float* q_out, float* r_out, QrMode mode) {
    const uint32_t M = a.rows, N = a.cols, K = std::min(M, N), batch = a.batch;
    if (K == 0 || batch == 0) return;
    const Kind kind = choose(M, N, mode);
    if (kind == Kind::none)
        throw std::invalid_argument("[qr] householder: " + std::to_string(M) + "x" + std::to_string(N) +
                                    " is more than its kernels take on this device.");
    AutoreleasePool pool;
    Cache& cache = Cache::shared();
    id<MTLDevice> dev = cache.rt.device;
    // Q's columns (K, M, or none) and R's rows (K, or M with zeros below K)
    const uint32_t QC = core::qr_q_cols(mode, M, N), RR = core::qr_r_rows(mode, M, N);

    // Q and R straight into the caller's memory where it is page-aligned,
    // else into a workspace and copied.
    const size_t page = (size_t)getpagesize(), f = sizeof(float);
    const size_t qn = (size_t)batch * M * QC, rn = (size_t)batch * RR * N;
    const bool q_direct = QC && reinterpret_cast<uintptr_t>(q_out) % page == 0;
    const bool r_direct = reinterpret_cast<uintptr_t>(r_out) % page == 0;
    id<MTLBuffer> qb = nil, rb = nil;
    if ((QC && !q_direct) || !r_direct) {
        Workspace& ws = cache.workspace(batch, M * QC, RR * N);
        qb = ws.q;
        rb = ws.r;
    }
    if (q_direct) qb = metal_linalg::detail::wrap_host(dev, q_out, qn);
    if (r_direct) rb = metal_linalg::detail::wrap_host(dev, r_out, rn);
    if (!qb) qb = rb;   // R alone: bound, never written
    id<MTLBuffer> src = input_buffer(dev, a);

    // Work per command buffer: a large batch is cut into chunks by a
    // conservative cost model, never smaller than the core count; the blocked
    // kernel's workspace into chunks of at most about 256 MB.
    unsigned cores = gpu_core_count();
    if (cores == 0) cores = 8;
    const double per_matrix = (kind == Kind::wy ? kWyCoreMs : kSimdCoreMs) * (double)M * N * std::max(K, QC) + 0.002;
    const double budget = (double)env_uint("QR_CHUNK_MS", (unsigned)kChunkBudgetMs);
    uint32_t chunk = (uint32_t)std::min<double>(batch, std::max<double>(cores, std::floor(budget * cores / per_matrix)));
    const SimdShape sh = kind == Kind::simd ? simd_shape(M, N) : SimdShape{};
    const WyShape wy = kind == Kind::wy ? wy_shape(M, N, cache.wy_max_simdgroups(), batch, cores) : WyShape{};
    // The blocked kernel's Q workspace: Kp columns, mp for Q square, none for
    // R alone; none either where Q is formed in place in the output (the
    // output has the workspace's shape)
    const uint32_t wq = mode == QrMode::r ? 0 : mode == QrMode::complete && M > N ? wy.mp : wy.Kp;
    const bool q_in_place = wq && wy.mp == M && wq == QC;
    WyWork* ww = nullptr;
    if (kind == Kind::wy) {
        const size_t fa = (size_t)wy.mp * wy.np, fq = q_in_place ? 0 : (size_t)wy.mp * wq,
                     ft = (size_t)round_up(wy.Kp, 64) * 64;
        chunk = (uint32_t)std::clamp<size_t>(((size_t)1 << 26) / (fa + fq + ft), 1, chunk);
        ww = &cache.wy_work(fa * chunk, fq * chunk, ft * chunk);
    }
    id<MTLComputePipelineState> pso = kind == Kind::simd ? cache.simd_pipeline(sh) : cache.wy_pipeline(wy.r);
    for (uint32_t b0 = 0; b0 < batch; b0 += chunk) {
        const uint32_t bc = std::min(chunk, batch - b0);
        id<MTLCommandBuffer> cmd = [cache.rt.queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:src offset:(size_t)b0 * M * N * f atIndex:0];
        [enc setBuffer:qb offset:QC ? (size_t)b0 * M * QC * f : 0 atIndex:1];
        [enc setBuffer:rb offset:(size_t)b0 * RR * N * f atIndex:2];
        if (kind == Kind::simd) {
            const QsParams sp{M, N, bc, QC};
            [enc setBytes:&sp length:sizeof sp atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake((bc + kSimdPerGroup - 1) / kSimdPerGroup, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(32 * kSimdPerGroup, 1, 1)];
        } else {
            const QwParams wp{M, N, wy.mp, wy.np, wy.Kp, bc, q_in_place ? 1u : 0u, wy.sc, wq, QC, RR};
            [enc setBytes:&wp length:sizeof wp atIndex:3];
            [enc setBuffer:ww->ba offset:0 atIndex:4];
            [enc setBuffer:ww->bq offset:0 atIndex:5];
            [enc setBuffer:ww->bt offset:0 atIndex:6];
            [enc setThreadgroupMemoryLength:wy_tg_bytes(wy) atIndex:0];
            [enc dispatchThreadgroups:MTLSizeMake(bc, 1, 1) threadsPerThreadgroup:MTLSizeMake(32 * wy.s, 1, 1)];
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
    if (QC && !q_direct) copy_out(static_cast<const float*>([qb contents]), q_out, batch, (size_t)M * QC);
    if (!r_direct) copy_out(static_cast<const float*>([rb contents]), r_out, batch, (size_t)RR * N);
}

// The unblocked backend: these kernels wherever they take the matrix (up to
// 4096 rows), the blocked QR beyond. Its own kernel, a threadgroup a matrix
// walking device memory with the matrix padded to 32 rows and a full square
// Q, was 2.5-6x slower than these wherever it ran, and is gone (2.16.0).
void qr_unblocked(const Matrices& a, float* q, float* r, QrMode mode) {
    if (qr_householder_preferred(a.rows, a.cols)) qr_householder(a, q, r, mode);
    else if (qr_blocked_fits(a.rows, a.cols)) qr_blocked(a, q, r, mode);
    else qr_streaming_amx_reduced(a, q, r, mode);
}

void qr_householder(const Matrices& a, float* q, float* r) { qr_householder(a, q, r, QrMode::reduced); }
void qr_unblocked(const Matrices& a, float* q, float* r) { qr_unblocked(a, q, r, QrMode::reduced); }

} // namespace core::detail
} // namespace metal_linalg
