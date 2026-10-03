#ifndef ACCELERATE_NEW_LAPACK
#define ACCELERATE_NEW_LAPACK   // LAPACK's current interface; before any Accelerate header
#endif
#include <metal_linalg/core.h>
#include "calibration.h"
#include "metal_runtime.h"
#include "shaders.h"

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <Accelerate/Accelerate.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

using metal_linalg::core::Matrices;
using metal_linalg::detail::AutoreleasePool;
using metal_linalg::detail::HostBuffer;
using metal_linalg::detail::MetalRuntime;
using metal_linalg::detail::Part;
using metal_linalg::detail::copy_out;
using metal_linalg::detail::input_buffer;
using metal_linalg::detail::make_pipeline;
using metal_linalg::detail::pad_up;
using metal_linalg::detail::scan;
using metal_linalg::detail::transpose_out;
using metal_linalg::detail::wrap_host;

namespace metal_linalg {
namespace {

// Must match `SvdParams` in Svd_Jacobi.metal.
struct Params {
    uint  m;
    uint  n;
    uint  n_pairs;
    uint  max_sweeps;
    float tol;
    float null_tol;
};

constexpr uint kRedFloats = 32;   // kRedFloats in the shader

// Floor on the row count the rotation tolerance is computed from; see SvdOptions.
constexpr uint kMinEffectiveRows = 64;

// A column below this many epsilon times the largest column is numerically
// null: just above the rounding floor, and independent of the matrix's size.
// See the note on null columns in Svd_Jacobi.metal.
constexpr float kNullEps = 32.0f;

// Cost model for chunking, core-milliseconds per matrix as K * M * N^2.
// Conservative; it only bounds the work per command buffer (see eigh.mm).
constexpr double kCoreMsPerMN2  = 8e-6;
constexpr double kChunkBudgetMs = 750.0;

unsigned env_uint(const char* name, unsigned fallback) {
    if (const char* s = std::getenv(name)) {
        const long v = std::strtol(s, nullptr, 10);
        if (v > 0) return (unsigned)v;
    }
    return fallback;
}

struct Workspace {
    id<MTLBuffer> G, V, S, U, Vt, info;
};

struct Cache {
    MetalRuntime& rt = MetalRuntime::shared(METAL_LINALG_SHADER(Svd_Jacobi), "svd");

    std::map<bool, id<MTLComputePipelineState>>              pipelines;   // by compute_uv
    std::map<std::tuple<uint, uint, uint, bool>, Workspace>  workspaces;  // (batch, m, n, uv)

    id<MTLComputePipelineState> get_pipeline(bool uv) {
        if (auto it = pipelines.find(uv); it != pipelines.end()) return it->second;
        MTLFunctionConstantValues* cv = [[MTLFunctionConstantValues alloc] init];
        [cv setConstantValue:&uv type:MTLDataTypeBool atIndex:0];
        return pipelines[uv] = make_pipeline(rt.device, rt.library, @"svd_jacobi", cv);
    }

    Workspace get_workspace(uint batch, uint m, uint n, bool uv) {
        const auto key = std::make_tuple(batch, m, n, uv);
        if (auto it = workspaces.find(key); it != workspaces.end()) return it->second;

        const MTLResourceOptions opt = MTLResourceStorageModeShared;
        auto buf = [&](size_t bytes) { return [rt.device newBufferWithLength:std::max<size_t>(bytes, 16) options:opt]; };
        const size_t f = sizeof(float);
        Workspace w;
        w.G    = buf((size_t)batch * m * n * f);
        w.S    = buf((size_t)batch * n * f);
        w.info = buf((size_t)batch * sizeof(uint));
        w.V    = uv ? buf((size_t)batch * n * n * f) : nil;
        w.U    = uv ? buf((size_t)batch * m * n * f) : nil;
        w.Vt   = uv ? buf((size_t)batch * n * n * f) : nil;
        return workspaces[key] = w;
    }
};

// Simdgroups per matrix. Each owns one or more column pairs per round, so
// this trades parallelism inside a matrix against how many matrices fit on a
// core. Measured with `benchmark_svd --tune`; the same two regimes as the
// eigensolver's thread rule:
//
//   few matrices   as many simdgroups as there are pairs, up to the 32 a
//                  threadgroup allows: the matrix has the core to itself.
//   many matrices  only as many as the work needs, one per
//                  kRowPairsPerSimdgroup row-pairs rotated per round, so that
//                  threadgroups stay small and co-reside.
//
// The switch is where the batch would need more than kSimdgroupsPerCore
// simdgroups per core at one pair each.
constexpr uint kRowPairsPerSimdgroup = 512;
constexpr uint kSimdgroupsPerCore    = 32;

uint auto_simdgroups(uint n_pairs, uint m, uint batch, uint cores) {
    const uint work   = (n_pairs * m + kRowPairsPerSimdgroup - 1) / kRowPairsPerSimdgroup;
    const uint budget = std::max(1u, kSimdgroupsPerCore * cores / std::max(1u, batch));
    return std::max(work, std::min(n_pairs, budget));
}

// Completes the zero columns of one M x N matrix U (row-major) to an
// orthonormal set. The kernel writes a zero column wherever the singular value
// is too small for column / sigma to mean anything; any unit vector orthogonal
// to the others is then a valid left singular vector.
//
// Each candidate is a fixed pseudo-random vector, orthogonalised three times
// against the columns already in place, in double precision. A random vector
// always has a component in the orthogonal complement, however few dimensions
// that has left -- coordinate vectors do not, and an acceptance test that asks
// for a large residual fails exactly when the complement is nearly used up.
void complete_columns_impl(float* u, uint m, uint n) {
    std::vector<char> valid(n, 0);
    for (uint c = 0; c < n; ++c) {
        for (uint i = 0; i < m && !valid[c]; ++i) valid[c] = u[(size_t)i * n + c] != 0.0f;
    }
    std::vector<double> w(m);
    uint64_t state = 0x9E3779B97F4A7C15ull;   // deterministic, so results are reproducible
    auto next_unit = [&state]() {
        state ^= state << 13; state ^= state >> 7; state ^= state << 17;
        return (double)(state >> 11) / (double)(1ull << 53) - 0.5;
    };
    for (uint c = 0; c < n; ++c) {
        if (valid[c]) continue;
        for (int attempt = 0; attempt < 8 && !valid[c]; ++attempt) {
            double start = 0.0;
            for (uint i = 0; i < m; ++i) { w[i] = next_unit(); start += w[i] * w[i]; }
            start = std::sqrt(start);
            for (int pass = 0; pass < 3; ++pass) {
                for (uint k = 0; k < n; ++k) {
                    if (!valid[k]) continue;
                    double dot = 0.0;
                    for (uint i = 0; i < m; ++i) dot += (double)u[(size_t)i * n + k] * w[i];
                    for (uint i = 0; i < m; ++i) w[i] -= dot * (double)u[(size_t)i * n + k];
                }
            }
            double nrm = 0.0;
            for (uint i = 0; i < m; ++i) nrm += w[i] * w[i];
            nrm = std::sqrt(nrm);
            // In double, a residual of 1e-6 of the start still has ten good digits.
            if (nrm > 1e-6 * start) {
                for (uint i = 0; i < m; ++i) u[(size_t)i * n + c] = (float)(w[i] / nrm);
                valid[c] = 1;
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Routing policy
// -----------------------------------------------------------------------------
// One entry per device that has actually been measured, keyed on the Metal
// device name and GPU core count, exactly as in qr.mm and eigh.mm.
// To add a device: run `python3 tuning/tune_svd.py build/sweep_svd` on it, on
// an idle machine, and paste the row it prints.
struct TunedEntry {
    const char* device_name;
    unsigned    gpu_cores;
    unsigned    qr_min_rows;
    unsigned    qr_min_k;
    unsigned    block_min_k;
    unsigned    block_min_k_batched;   // 0, 0 = no batch-dependent block crossover
    unsigned    block_min_batch;
    unsigned    gpu_max_k;
    unsigned    gpu_min_batch_times_k;
    unsigned    gpu_min_batch;
    unsigned    calibration;   // kCalibration* (calibration.h); rows without it are current
};

// The rows are generated from every run submitted for a device (docs/results/)
// by tuning/generate_tables.py, which a GitHub Action reruns after each
// merge; see docs/tuning.md. Why the measured values are what they are is in
// docs/studies/. The last row keeps the array non-empty and matches nothing.
constexpr TunedEntry kTuned[] = {
#include "tuned/svd.inc"
    {"", 0,   0, 0,   0, 0, 0,   0, 0, 0,   0},
};

struct ResolvedPolicy {
    SvdPolicy   policy;
    std::string source;
    std::string device;
};

bool env_value(const char* name, unsigned& out) {
    const char* s = std::getenv(name);
    if (!s || !*s) return false;
    char* end = nullptr;
    const long long v = std::strtoll(s, &end, 10);
    if (end == s || v < 0) return false;
    out = v > 0xFFFFFFFFll ? kSvdNoLimit : (unsigned)v;
    return true;
}

ResolvedPolicy resolve_policy() {
    ResolvedPolicy r;
    r.device = device_name();
    r.policy.gpu_cores = gpu_core_count();

    for (const auto& e : kTuned) {
        if (e.device_name[0] != '\0' && r.device == e.device_name &&
            r.policy.gpu_cores == e.gpu_cores) {
            r.policy.qr_min_rows           = e.qr_min_rows;
            r.policy.qr_min_k              = e.qr_min_k;
            r.policy.block_min_k           = e.block_min_k;
            r.policy.block_min_k_batched   = e.block_min_k_batched;
            r.policy.block_min_batch       = e.block_min_batch;
            r.policy.gpu_max_k             = e.gpu_max_k;
            r.policy.gpu_min_batch_times_k = e.gpu_min_batch_times_k;
            r.policy.gpu_min_batch         = e.gpu_min_batch;
            r.source = detail::tuned_source_prefix(e.calibration) + r.device;
            detail::calibration_notice("SVD", e.calibration);
            break;
        }
    }
    if (r.source.empty()) {
        r.source = "default:untuned-device" + (r.device.empty() ? "" : " (" + r.device + ")");
        detail::calibration_notice("SVD", kUncalibrated);
    }

    std::string env;
    auto over = [&](const char* name, unsigned& field) {
        if (env_value(name, field)) env += (env.empty() ? "" : ",") + std::string(name);
    };
    over("SVD_QR_MIN_ROWS",           r.policy.qr_min_rows);
    over("SVD_QR_MIN_K",              r.policy.qr_min_k);
    over("SVD_BLOCK_MIN_K",           r.policy.block_min_k);
    over("SVD_BLOCK_MIN_K_BATCHED",   r.policy.block_min_k_batched);
    over("SVD_BLOCK_MIN_BATCH",       r.policy.block_min_batch);
    over("SVD_GPU_MAX_K",             r.policy.gpu_max_k);
    over("SVD_GPU_MIN_BATCH_TIMES_K", r.policy.gpu_min_batch_times_k);
    over("SVD_GPU_MIN_BATCH",         r.policy.gpu_min_batch);
    if (!env.empty()) r.source = "env:" + env;
    return r;
}

ResolvedPolicy& policy_state() {
    static ResolvedPolicy s = resolve_policy();
    return s;
}

// The kernel rule: which of the two kernels a k-column problem gets.
bool wants_block(unsigned k, unsigned batch) {
    const SvdPolicy& p = policy_state().policy;
    if (k >= p.block_min_k) return true;
    return p.block_min_batch && k >= p.block_min_k_batched && batch >= p.block_min_batch;
}

} // namespace

namespace detail {

void svd_complete_columns(float* u, unsigned m, unsigned n) { complete_columns_impl(u, m, n); }

} // namespace detail

namespace {

// C = A B for every matrix of a batch, on the GPU: A [m, k], B [k, n] and
// C [m, n]. A and B are page-aligned host memory, read in place.
void batched_matmul(float* a, float* b, float* c_out, uint32_t batch, uint32_t m, uint32_t k, uint32_t n) {
    AutoreleasePool pool;
    MetalRuntime& rt = MetalRuntime::shared(METAL_LINALG_SHADER(Svd_Jacobi), "svd");
    HostBuffer c((size_t)batch * m * n);
    auto matrix = [&](float* data, uint32_t rows, uint32_t cols) {
        MPSMatrixDescriptor* d =
            [MPSMatrixDescriptor matrixDescriptorWithRows:rows
                                                  columns:cols
                                                 matrices:batch
                                                 rowBytes:cols * sizeof(float)
                                              matrixBytes:(size_t)rows * cols * sizeof(float)
                                                 dataType:MPSDataTypeFloat32];
        return [[MPSMatrix alloc] initWithBuffer:wrap_host(rt.device, data, (size_t)batch * rows * cols)
                                      descriptor:d];
    };
    MPSMatrix* A = matrix(a, m, k);
    MPSMatrix* B = matrix(b, k, n);
    MPSMatrix* C = matrix(c.data(), m, n);
    MPSMatrixMultiplication* mm =
        [[MPSMatrixMultiplication alloc] initWithDevice:rt.device
                                          transposeLeft:NO
                                         transposeRight:NO
                                             resultRows:m
                                          resultColumns:n
                                        interiorColumns:k
                                                  alpha:1.0
                                                   beta:0.0];
    mm.batchStart = 0;
    mm.batchSize  = batch;

    id<MTLCommandBuffer> cmd = [rt.queue commandBuffer];
    [mm encodeToCommandBuffer:cmd leftMatrix:A rightMatrix:B resultMatrix:C];
    [cmd commit];
    [cmd waitUntilCompleted];
    if (cmd.error) {
        throw std::runtime_error(std::string("[svd] GPU matrix product failed: ") +
                                 cmd.error.localizedDescription.UTF8String);
    }
    copy_out(c.data(), c_out, batch, (size_t)m * n);
}

} // namespace

namespace core::detail {

void svd_jacobi(const Matrices& a, const SvdOptions& opt,
                float* u_out, float* s_out, float* vt_out, uint32_t* info_out) {
    const uint M = a.rows;
    const uint N = a.cols;
    const uint K = std::min(M, N);
    const uint batch = a.batch;
    const bool compute_uv = u_out || vt_out;
    if (K == 0 || batch == 0) {
        if (info_out) std::fill(info_out, info_out + batch, 0u);
        return;
    }
    AutoreleasePool pool;

    // The kernel wants m >= n. For a wide matrix decompose the transpose,
    // A^T = U' S V'^T, and read off A = V' S U'^T.
    const bool wide = M < N;
    const uint m = wide ? N : M;
    const uint n = K;
    if (n > 0xFFFFu) {
        throw std::invalid_argument("[svd] min(M, N) exceeds the 16-bit index used in threadgroup memory.");
    }

    static Cache cache;
    id<MTLDevice> dev = cache.rt.device;
    id<MTLComputePipelineState> pso = cache.get_pipeline(compute_uv);
    Workspace ws = cache.get_workspace(batch, m, n, compute_uv);

    const uint np       = (n + 1) / 2;
    const uint max_sg   = std::min(np, std::min(32u, (uint)pso.maxTotalThreadsPerThreadgroup / 32));
    uint cores = gpu_core_count();
    if (cores == 0) cores = 8;
    const uint n_sg     = std::max(1u, std::min(max_sg, opt.simdgroups ? opt.simdgroups
                                                          : auto_simdgroups(np, m, batch, cores)));
    const uint threads  = 32 * n_sg;

    const size_t tg_bytes = pad_up((uint)((2 * kRedFloats + n) * sizeof(float) + n * sizeof(uint16_t)), 16);
    if (tg_bytes > dev.maxThreadgroupMemoryLength) {
        throw std::invalid_argument("[svd] min(M, N)=" + std::to_string(n) + " needs " +
                                    std::to_string(tg_bytes) + " bytes of threadgroup memory but "
                                    "this device allows " +
                                    std::to_string((size_t)dev.maxThreadgroupMemoryLength) + ".");
    }

    Params prm;
    prm.m          = m;
    prm.n          = n;
    prm.n_pairs    = np;
    prm.max_sweeps = opt.max_sweeps;
    const float rows = (float)std::max({m, opt.effective_rows, kMinEffectiveRows});
    prm.tol        = opt.tol > 0.0f ? opt.tol
                                    : std::sqrt(rows) * std::numeric_limits<float>::epsilon();
    prm.null_tol   = kNullEps * std::numeric_limits<float>::epsilon();

    id<MTLBuffer> buf_src = input_buffer(dev, a, wide);

    uint chunk = batch;
    {
        const double per_matrix_core_ms = kCoreMsPerMN2 * (double)m * n * n;
        const double budget = (double)env_uint("SVD_CHUNK_MS", (unsigned)kChunkBudgetMs);
        const double fit = std::floor(budget * cores / per_matrix_core_ms);
        chunk = (uint)std::max((double)cores, std::min((double)batch, fit));
        chunk = std::min(chunk, batch);
    }

    const size_t f = sizeof(float);
    for (uint b0 = 0; b0 < batch; b0 += chunk) {
        const uint bc = std::min(chunk, batch - b0);

        id<MTLCommandBuffer> cmd = [cache.rt.queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:buf_src offset:((size_t)b0 * m * n * f) atIndex:0];
        [enc setBuffer:ws.G    offset:((size_t)b0 * m * n * f) atIndex:1];
        [enc setBuffer:(compute_uv ? ws.V  : ws.G) offset:(compute_uv ? (size_t)b0 * n * n * f : 0) atIndex:2];
        [enc setBuffer:ws.S    offset:((size_t)b0 * n * f) atIndex:3];
        [enc setBuffer:(compute_uv ? ws.U  : ws.G) offset:(compute_uv ? (size_t)b0 * m * n * f : 0) atIndex:4];
        [enc setBuffer:(compute_uv ? ws.Vt : ws.G) offset:(compute_uv ? (size_t)b0 * n * n * f : 0) atIndex:5];
        [enc setBuffer:ws.info offset:((size_t)b0 * sizeof(uint)) atIndex:6];
        [enc setBytes:&prm length:sizeof(prm) atIndex:7];
        [enc setThreadgroupMemoryLength:tg_bytes atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(bc, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        if (cmd.error) {
            throw std::runtime_error(std::string("[svd] GPU kernel error: ") +
                                     cmd.error.localizedDescription.UTF8String +
                                     " (" + std::to_string(m) + "x" + std::to_string(n) + ", " +
                                     std::to_string(bc) + " matrices in this command buffer).");
        }
    }

    const uint* info_ptr = static_cast<const uint*>([ws.info contents]);
    for (uint b = 0; b < batch; ++b) {
        const uint w = info_ptr[b];
        if (!metal_linalg::detail::svd_converged(w) && !metal_linalg::detail::svd_nonfinite(w)) {
            throw std::runtime_error("[svd] Matrix " + std::to_string(b) + " of " +
                                     std::to_string(batch) + " (" + std::to_string(M) + "x" +
                                     std::to_string(N) + ") did not converge in " +
                                     std::to_string(opt.max_sweeps) + " sweeps.");
        }
    }

    copy_out(static_cast<const float*>([ws.S contents]), s_out, batch, n);
    if (info_out) std::memcpy(info_out, info_ptr, (size_t)batch * sizeof(uint32_t));
    if (!compute_uv) return;

    // Rank-deficient matrices: give U an orthonormal completion.
    float* u_ptr = static_cast<float*>([ws.U contents]);
    for (uint b = 0; b < batch; ++b) {
        if (metal_linalg::detail::svd_rank_deficient(info_ptr[b])) {
            metal_linalg::detail::svd_complete_columns(u_ptr + (size_t)b * m * n, m, n);
        }
    }

    // Kernel factors: Uk [batch, m, n], Vtk [batch, n, n].
    const float* vt_ptr = static_cast<const float*>([ws.Vt contents]);
    if (!wide) {
        copy_out(u_ptr, u_out, batch, (size_t)m * n);
        copy_out(vt_ptr, vt_out, batch, (size_t)n * n);
        return;
    }
    // A = V' S U'^T:  U = V' = (Vtk)^T  [M, K],   Vt = U'^T = (Uk)^T  [K, N].
    transpose_out(vt_ptr, u_out, batch, n, n);
    transpose_out(u_ptr, vt_out, batch, m, n);
}

void svd_qr_jacobi(const Matrices& a, const SvdOptions& opt,
                   float* u_out, float* s_out, float* vt_out, uint32_t* info_out) {
    const uint32_t M = a.rows, N = a.cols, batch = a.batch;
    const bool compute_uv = u_out || vt_out;
    if (std::min(M, N) == 0 || batch == 0) {
        if (info_out) std::fill(info_out, info_out + batch, 0u);
        return;
    }
    if (M < N) {
        // A^T = U' S V'^T  =>  A = V' S U'^T.
        HostBuffer at((size_t)batch * M * N);
        transpose_out(a.data, at.data(), batch, M, N);
        const Matrices t{at.data(), batch, N, M};
        if (!compute_uv) {
            svd_qr_jacobi(t, opt, nullptr, s_out, nullptr, info_out);
            return;
        }
        HostBuffer ut((size_t)batch * N * M), vtt((size_t)batch * M * M);
        svd_qr_jacobi(t, opt, ut.data(), s_out, vtt.data(), info_out);
        transpose_out(vtt.data(), u_out, batch, M, M);   // U  = V'  = (Vt')^T  [M, M]
        transpose_out(ut.data(), vt_out, batch, N, M);   // Vt = U'^T           [M, N]
        return;
    }

    // A = Q R, R = U_R S V^T  =>  A = (Q U_R) S V^T. The tolerances are those
    // of A, not of the small factor.
    const uint32_t l = M, k = N;
    HostBuffer q((size_t)batch * l * k), r((size_t)batch * k * k);
    core::qr(a, q.data(), r.data());
    SvdOptions inner = opt;
    inner.effective_rows = std::max<unsigned>(opt.effective_rows, l);
    const bool block = opt.kernel == SvdOptions::Kernel::block ||
                       (opt.kernel == SvdOptions::Kernel::automatic && wants_block(k, batch));
    auto kernel = block ? svd_block_jacobi : svd_jacobi;
    const Matrices rm{r.data(), batch, k, k};
    if (!compute_uv) {
        kernel(rm, inner, nullptr, s_out, nullptr, info_out);
        return;
    }
    HostBuffer ur((size_t)batch * k * k);
    kernel(rm, inner, ur.data(), s_out, vt_out, info_out);
    if (u_out) batched_matmul(q.data(), ur.data(), u_out, batch, l, k, k);
}

void svd_cpu(const Matrices& a, float* u_out, float* s_out, float* vt_out, uint32_t* info_out) {
    const uint32_t M = a.rows, N = a.cols, K = std::min(M, N), batch = a.batch;
    if (K == 0 || batch == 0) {
        if (info_out) std::fill(info_out, info_out + batch, 0u);
        return;
    }
    const bool compute_uv = u_out || vt_out;
    const size_t per = (size_t)M * N;

    // LAPACK is column-major, so it sees each matrix transposed: A^T, N x M.
    // Its thin SVD A^T = U' S V'^T gives A = V' S U'^T, and the column-major
    // factors it writes are already the row-major ones wanted: U' (N x K) read
    // row-major is Vt (K x N), and V'^T (K x M) read row-major is U (M x K).
    //
    // A tall matrix is reduced first, by a QR of A itself, as MLX's CPU path
    // did before 2.0: sgesdd would reduce it too, but on the wide A^T it uses
    // an LQ factorisation, and Accelerate's LQ routines measured about half
    // the speed of its QR ones (M5 Pro: 4096 x 256 in 26 ms against 12 ms
    // for a transpose and a QR). So: transpose into column-major A, A = Q R
    // (sgeqrf, sorgqr), R = U_R S Vt_R (sgesdd on the N x N factor), and
    // U = Q U_R (one sgemm).
    const bool tall = M >= 2 * N;
    char jobz = compute_uv ? 'S' : 'N';
    __LAPACK_int lm = (__LAPACK_int)N, ln = (__LAPACK_int)M, lk = (__LAPACK_int)K, err = 0;

    std::vector<float> work_a(per), tau(K), r(tall ? (size_t)K * K : 1);
    std::vector<float> spare_u(compute_uv ? (size_t)M * K : 1), spare_vt(compute_uv ? (size_t)K * N : 1);
    std::vector<float> r_u(tall && compute_uv ? (size_t)K * K : 1), r_vt(tall && compute_uv ? (size_t)K * K : 1);
    std::vector<__LAPACK_int> iwork(8 * (size_t)K);

    // Workspace sizes, by query.
    __LAPACK_int lwork = 1, query = -1;
    auto grow = [&](float q) { lwork = std::max<__LAPACK_int>(lwork, (__LAPACK_int)std::ceil(q)); };
    float q = 0.0f;
    if (tall) {
        sgeqrf_(&ln, &lk, work_a.data(), &ln, tau.data(), &q, &query, &err);                         grow(q);
        if (compute_uv) { sorgqr_(&ln, &lk, &lk, work_a.data(), &ln, tau.data(), &q, &query, &err); grow(q); }
        sgesdd_(&jobz, &lk, &lk, r.data(), &lk, s_out, r_u.data(), &lk, r_vt.data(), &lk,
                &q, &query, iwork.data(), &err);                                                       grow(q);
    } else {
        sgesdd_(&jobz, &lm, &ln, work_a.data(), &lm, s_out, spare_vt.data(), &lm, spare_u.data(), &lk,
                &q, &query, iwork.data(), &err);                                                       grow(q);
    }
    std::vector<float> work(lwork);

    auto lapack_check = [&](const char* routine, uint32_t b) {
        if (err != 0) {
            throw std::runtime_error(std::string("[svd] LAPACK ") + routine + " failed on matrix " +
                                     std::to_string(b) + " of " + std::to_string(batch) + " (" +
                                     std::to_string(M) + "x" + std::to_string(N) + "), info " +
                                     std::to_string((long long)err) + ".");
        }
    };

    // Non-finite input gives NaN for that matrix, as on the GPU.
    std::vector<float> amax(batch);
    std::vector<char>  finite(batch);
    scan(a, Part::all, amax.data(), finite.data());

    for (uint32_t b = 0; b < batch; ++b) {
        float* s  = s_out + (size_t)b * K;
        float* u  = u_out  ? u_out  + (size_t)b * M * K : spare_u.data();
        float* vt = vt_out ? vt_out + (size_t)b * K * N : spare_vt.data();
        if (!finite[b]) {
            std::fill(s, s + K, NAN);
            if (u_out)  std::fill(u, u + (size_t)M * K, NAN);
            if (vt_out) std::fill(vt, vt + (size_t)K * N, NAN);
            if (info_out) info_out[b] = 1u << 17;
            continue;
        }
        if (!tall) {
            std::memcpy(work_a.data(), a.data + b * per, per * sizeof(float));
            sgesdd_(&jobz, &lm, &ln, work_a.data(), &lm, s, vt, &lm, u, &lk,
                    work.data(), &lwork, iwork.data(), &err);
            lapack_check("sgesdd", b);
        } else {
            vDSP_mtrans(a.data + b * per, 1, work_a.data(), 1, N, M);   // column-major A, M x N
            sgeqrf_(&ln, &lk, work_a.data(), &ln, tau.data(), work.data(), &lwork, &err);
            lapack_check("sgeqrf", b);
            for (uint32_t j = 0; j < K; ++j)          // R: the upper triangle, column-major
                for (uint32_t i = 0; i < K; ++i)
                    r[i + (size_t)j * K] = i <= j ? work_a[i + (size_t)j * M] : 0.0f;
            sgesdd_(&jobz, &lk, &lk, r.data(), &lk, s, r_u.data(), &lk, r_vt.data(), &lk,
                    work.data(), &lwork, iwork.data(), &err);
            lapack_check("sgesdd", b);
            if (compute_uv) {
                sorgqr_(&ln, &lk, &lk, work_a.data(), &ln, tau.data(), work.data(), &lwork, &err);
                lapack_check("sorgqr", b);
                // Row-major U = Q U_R, from the column-major Q and U_R (each
                // read row-major as its transpose).
                cblas_sgemm(CblasRowMajor, CblasTrans, CblasTrans, (__LAPACK_int)M, lk, lk, 1.0f,
                            work_a.data(), (__LAPACK_int)M, r_u.data(), lk, 0.0f, u, lk);
                vDSP_mtrans(r_vt.data(), 1, vt, 1, K, K);   // row-major Vt_R
            }
        }
        if (info_out) info_out[b] = 1u | (1u << 16);
    }
}

} // namespace core::detail

SvdPolicy   svd_policy()        { return policy_state().policy; }
const char* svd_policy_source() { return policy_state().source.c_str(); }

void set_svd_policy(const SvdPolicy& p) {
    policy_state().policy = p;
    policy_state().source = "user";
}

SvdBackend svd_gpu_backend(unsigned m, unsigned n, unsigned batch) {
    const SvdPolicy& p = policy_state().policy;
    const unsigned long long k = std::min(m, n), l = std::max(m, n);
    const bool precondition = l >= p.qr_min_rows && k >= p.qr_min_k && l >= 2 * k;
    const bool block = wants_block((unsigned)k, batch);
    if (precondition) return block ? SvdBackend::qr_block_jacobi : SvdBackend::qr_jacobi;
    return block ? SvdBackend::block_jacobi : SvdBackend::jacobi;
}

bool svd_uses_gpu(unsigned m, unsigned n, unsigned batch) {
    if (const char* e = std::getenv("SVD_DEVICE")) {
        const std::string s = e;
        if (s == "gpu") return true;
        if (s == "cpu") return false;
    }
    const SvdPolicy& p = policy_state().policy;
    const unsigned k = std::min(m, n);
    return k <= p.gpu_max_k && (unsigned long long)batch * k >= p.gpu_min_batch_times_k &&
           batch >= p.gpu_min_batch;
}

SvdBackend svd_backend(unsigned m, unsigned n, unsigned batch) {
    return svd_uses_gpu(m, n, batch) ? svd_gpu_backend(m, n, batch) : SvdBackend::cpu;
}

void core::svd(const Matrices& a, float* u, float* s, float* vt, uint32_t* info) {
    const unsigned m = a.rows, n = a.cols, batch = a.batch;
    if (m == 0 || n == 0 || batch == 0) {
        core::detail::svd_jacobi(a, SvdOptions{}, u, s, vt, info);   // handles empties
        return;
    }
    SvdOptions opt;
    switch (svd_backend(m, n, batch)) {
        case SvdBackend::cpu:
            core::detail::svd_cpu(a, u, s, vt, info);
            return;
        case SvdBackend::block_jacobi:
            core::detail::svd_block_jacobi(a, opt, u, s, vt, info);
            return;
        case SvdBackend::qr_jacobi:
            opt.kernel = SvdOptions::Kernel::jacobi;
            core::detail::svd_qr_jacobi(a, opt, u, s, vt, info);
            return;
        case SvdBackend::qr_block_jacobi:
            opt.kernel = SvdOptions::Kernel::block;
            core::detail::svd_qr_jacobi(a, opt, u, s, vt, info);
            return;
        default:
            core::detail::svd_jacobi(a, opt, u, s, vt, info);
            return;
    }
}

} // namespace metal_linalg
