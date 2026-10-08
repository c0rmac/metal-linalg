#ifndef ACCELERATE_NEW_LAPACK
#define ACCELERATE_NEW_LAPACK   // LAPACK's current interface; before any Accelerate header
#endif
#include <metal_linalg/core.h>
#include "calibration.h"
#include "estimate.h"
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
using metal_linalg::detail::lapack_batches;
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
    uint  round_budget;   // rounds per dispatch (0: the whole solve)
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
    id<MTLBuffer> G, V, S, U, Vt, info, state;   // state: a split solve's per-matrix JacobiState
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
        w.state = buf((size_t)batch * metal_linalg::detail::kJacobiStateBytes);
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
    unsigned    gpu_max_l;             // kSvdNoLimit in rows from before 2.10.0
    // The same for singular values alone; 0, 0, 0, kSvdNoLimit = as with
    // vectors, which rows measured before 2.11.0 leave.
    unsigned    values_gpu_max_k;
    unsigned    values_gpu_min_batch_times_k;
    unsigned    values_gpu_min_batch;
    unsigned    values_gpu_max_l;
    // The bidiag backend instead of the CPU from these k; 0 = never, which rows
    // measured before the backend existed leave.
    unsigned    bidiag_min_k;
    unsigned    values_bidiag_min_k;
    // ... for batches of at most this many matrices; 0 = any batch, which
    // rows measured before the CPU path used every core leave.
    unsigned    bidiag_max_batch;
    unsigned    values_bidiag_max_batch;
    // The golub_kahan window; 0, 0 = never, which rows measured before the
    // backend existed leave.
    unsigned    gk_min_k;
    unsigned    gk_max_k;
    unsigned    share_min_batch;       // 0 = never, which rows measured before 2.11.0 leave
    // The large-batch clause; 0, 0 = never, which rows from before 2.12.0 leave.
    unsigned    gpu_big_batch_max_k;
    unsigned    gpu_big_batch_min;
    unsigned    values_band_min_k;     // 0 = never, which rows from before 2.13.0 leave
    unsigned    values_band_width;     // 0 = 16, which rows from before 2.15.0 leave
    unsigned    band_min_k;            // 0 = never, which rows from before 2.15.0 leave
    // The bidiag_batch windows; rows from before 2.17.0 leave them 0: never
    // (and max_l no cap, as apply() reads it).
    unsigned    bidiag_batch_min_k;
    unsigned    bidiag_batch_max_k;
    unsigned    bidiag_batch_min_batch;
    unsigned    bidiag_batch_max_l;
    unsigned    values_bidiag_batch_min_k;
    unsigned    values_bidiag_batch_max_k;
    unsigned    values_bidiag_batch_min_batch;
    unsigned    values_bidiag_batch_max_l;
    unsigned    calibration;   // kCalibration* (calibration.h); rows without it are current
};

// The rows are generated from every run submitted for a device (docs/results/)
// by tuning/generate_tables.py, which a GitHub Action reruns after each
// merge; see docs/tuning.md. Why the measured values are what they are is in
// docs/studies/. The last row keeps the array non-empty and matches nothing.
constexpr TunedEntry kTuned[] = {
#include "tuned/svd.inc"
    {"", 0,   0, 0,   0, 0, 0,   0, 0, 0, 0,   0, 0, 0, 0,   0, 0, 0, 0,   0, 0,   0,   0, 0,   0, 0,   0,   0, 0, 0, 0,   0, 0, 0, 0,   0},
};

// A device with no entry gets an estimated row: a measured device's timings
// refitted for a GPU weaker against its CPU, for every slowdown pair on
// tuning/estimate.py's ladder (estimate.h). Generated with kTuned.
struct EstimatedEntry {
    unsigned   small_x100;        // batched kernels' slowdown x 100
    unsigned   large_x100;        // large-matrix backends' slowdown x 100
    unsigned   anchor_cpu_cores;
    TunedEntry row;
};
constexpr EstimatedEntry kEstimated[] = {
#include "tuned/svd_estimated.inc"
    {0, 0, 0, {"", 0,   0, 0,   0, 0, 0,   0, 0, 0, 0,   0, 0, 0, 0,   0, 0, 0, 0,   0, 0,   0,   0, 0,   0, 0,   0,   0, 0, 0, 0,   0, 0, 0, 0,   0}},
};

void apply(const TunedEntry& e, SvdPolicy& p) {
    p.qr_min_rows                  = e.qr_min_rows;
    p.qr_min_k                     = e.qr_min_k;
    p.block_min_k                  = e.block_min_k;
    p.block_min_k_batched          = e.block_min_k_batched;
    p.block_min_batch              = e.block_min_batch;
    p.gpu_max_k                    = e.gpu_max_k;
    p.gpu_min_batch_times_k        = e.gpu_min_batch_times_k;
    p.gpu_min_batch                = e.gpu_min_batch;
    p.gpu_max_l                    = e.gpu_max_l;
    p.values_gpu_max_k             = e.values_gpu_max_k;
    p.values_gpu_min_batch_times_k = e.values_gpu_min_batch_times_k;
    p.values_gpu_min_batch         = e.values_gpu_min_batch;
    p.values_gpu_max_l             = e.values_gpu_max_l;
    p.bidiag_min_k                 = e.bidiag_min_k;
    p.values_bidiag_min_k          = e.values_bidiag_min_k;
    p.bidiag_max_batch             = e.bidiag_max_batch;
    p.values_bidiag_max_batch      = e.values_bidiag_max_batch;
    p.gk_min_k                     = e.gk_min_k;
    p.gk_max_k                     = e.gk_max_k;
    p.share_min_batch              = e.share_min_batch;
    p.gpu_big_batch_max_k          = e.gpu_big_batch_max_k;
    p.gpu_big_batch_min            = e.gpu_big_batch_min;
    p.values_band_min_k            = e.values_band_min_k;
    p.values_band_width            = e.values_band_width;
    p.band_min_k                   = e.band_min_k;
    p.bidiag_batch_min_k            = e.bidiag_batch_min_k;
    p.bidiag_batch_max_k            = e.bidiag_batch_max_k;
    p.bidiag_batch_min_batch        = e.bidiag_batch_min_batch;
    p.bidiag_batch_max_l            = e.bidiag_batch_max_k ? e.bidiag_batch_max_l : kSvdNoLimit;
    p.values_bidiag_batch_min_k     = e.values_bidiag_batch_min_k;
    p.values_bidiag_batch_max_k     = e.values_bidiag_batch_max_k;
    p.values_bidiag_batch_min_batch = e.values_bidiag_batch_min_batch;
    p.values_bidiag_batch_max_l     = e.values_bidiag_batch_max_k ? e.values_bidiag_batch_max_l : kSvdNoLimit;
}

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

    bool forced = false;   // METAL_LINALG_ESTIMATE_AS: estimate even a measured Mac
    const detail::EstimateTarget target = detail::estimate_target(&forced);
    for (const auto& e : kTuned) {
        if (forced) break;
        if (e.device_name[0] != '\0' && r.device == e.device_name &&
            r.policy.gpu_cores == e.gpu_cores) {
            apply(e, r.policy);
            r.source = detail::tuned_source_prefix(e.calibration) + r.device;
            detail::calibration_notice("SVD", e.calibration);
            break;
        }
    }
    if (r.source.empty() && !target.device.empty()) {
        // No measurements for this GPU: a measured one's, refitted for this
        // one's GPU against its CPU (estimate.h).
        std::string source;
        if (const EstimatedEntry* e = detail::estimated_row(kEstimated, target, source)) {
            apply(e->row, r.policy);
            r.source = source;
            detail::calibration_notice("SVD", kUncalibrated);
        }
    }
    if (r.source.empty()) {
        // Nothing to estimate from (no Metal device, or no measured anchor).
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
    over("SVD_GPU_MAX_L",             r.policy.gpu_max_l);
    over("SVD_VALUES_GPU_MAX_K",             r.policy.values_gpu_max_k);
    over("SVD_VALUES_GPU_MIN_BATCH_TIMES_K", r.policy.values_gpu_min_batch_times_k);
    over("SVD_VALUES_GPU_MIN_BATCH",         r.policy.values_gpu_min_batch);
    over("SVD_VALUES_GPU_MAX_L",             r.policy.values_gpu_max_l);
    over("SVD_BIDIAG_MIN_K",          r.policy.bidiag_min_k);
    over("SVD_VALUES_BIDIAG_MIN_K",   r.policy.values_bidiag_min_k);
    over("SVD_BIDIAG_MAX_BATCH",        r.policy.bidiag_max_batch);
    over("SVD_VALUES_BIDIAG_MAX_BATCH", r.policy.values_bidiag_max_batch);
    over("SVD_GK_MIN_K",              r.policy.gk_min_k);
    over("SVD_GK_MAX_K",              r.policy.gk_max_k);
    over("SVD_SHARE_MIN_BATCH",       r.policy.share_min_batch);
    over("SVD_GPU_BIG_BATCH_MAX_K",   r.policy.gpu_big_batch_max_k);
    over("SVD_GPU_BIG_BATCH_MIN",     r.policy.gpu_big_batch_min);
    over("SVD_VALUES_BAND_MIN_K",     r.policy.values_band_min_k);
    over("SVD_VALUES_BAND_WIDTH",     r.policy.values_band_width);
    over("SVD_BAND_MIN_K",            r.policy.band_min_k);
    over("SVD_BIDIAG_BATCH_MIN_K",            r.policy.bidiag_batch_min_k);
    over("SVD_BIDIAG_BATCH_MAX_K",            r.policy.bidiag_batch_max_k);
    over("SVD_BIDIAG_BATCH_MIN_BATCH",        r.policy.bidiag_batch_min_batch);
    over("SVD_BIDIAG_BATCH_MAX_L",            r.policy.bidiag_batch_max_l);
    over("SVD_VALUES_BIDIAG_BATCH_MIN_K",     r.policy.values_bidiag_batch_min_k);
    over("SVD_VALUES_BIDIAG_BATCH_MAX_K",     r.policy.values_bidiag_batch_max_k);
    over("SVD_VALUES_BIDIAG_BATCH_MIN_BATCH", r.policy.values_bidiag_batch_min_batch);
    over("SVD_VALUES_BIDIAG_BATCH_MAX_L",     r.policy.values_bidiag_batch_max_l);
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
    const double per_matrix_core_ms = kCoreMsPerMN2 * (double)m * n * n;
    // A threadgroup's solve longer than a dispatch should run (the GPU's
    // watchdog; metal_runtime.h) is split over dispatches of a few rounds.
    const uint rounds = 2 * np - 1;
    prm.round_budget = metal_linalg::detail::jacobi_round_budget(per_matrix_core_ms, rounds, "SVD_DISPATCH_MS");
    {
        const double budget = (double)env_uint("SVD_CHUNK_MS", (unsigned)kChunkBudgetMs);
        const double fit = std::floor(budget * cores / per_matrix_core_ms);
        chunk = (uint)std::max((double)cores, std::min((double)batch, fit));
        chunk = std::min(chunk, batch);
    }

    const size_t f = sizeof(float);
    for (uint b0 = 0; b0 < batch; b0 += chunk) {
        const uint bc = std::min(chunk, batch - b0);

        auto encode = [&](id<MTLComputeCommandEncoder> enc) {
            [enc setComputePipelineState:pso];
            [enc setBuffer:buf_src offset:((size_t)b0 * m * n * f) atIndex:0];
            [enc setBuffer:ws.G    offset:((size_t)b0 * m * n * f) atIndex:1];
            [enc setBuffer:(compute_uv ? ws.V  : ws.G) offset:(compute_uv ? (size_t)b0 * n * n * f : 0) atIndex:2];
            [enc setBuffer:ws.S    offset:((size_t)b0 * n * f) atIndex:3];
            [enc setBuffer:(compute_uv ? ws.U  : ws.G) offset:(compute_uv ? (size_t)b0 * m * n * f : 0) atIndex:4];
            [enc setBuffer:(compute_uv ? ws.Vt : ws.G) offset:(compute_uv ? (size_t)b0 * n * n * f : 0) atIndex:5];
            [enc setBuffer:ws.info offset:((size_t)b0 * sizeof(uint)) atIndex:6];
            [enc setBytes:&prm length:sizeof(prm) atIndex:7];
            [enc setBuffer:ws.state offset:((size_t)b0 * metal_linalg::detail::kJacobiStateBytes) atIndex:8];
            [enc setThreadgroupMemoryLength:tg_bytes atIndex:0];
            [enc dispatchThreadgroups:MTLSizeMake(bc, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        };
        if (prm.round_budget) {
            // Command buffers of about kChunkBudgetMs, the chunk's matrices in
            // waves of one per core, until each has finished.
            const uint waves = (bc + cores - 1) / cores;
            const uint per_buffer = std::max(1u, (uint)(kChunkBudgetMs / (40.0 * waves)));
            const uint max_dispatches = (uint)std::min<uint64_t>(
                0xFFFFFFFFu, (uint64_t)opt.max_sweeps * rounds / prm.round_budget + 3);
            metal_linalg::detail::run_split_jacobi(cache.rt.queue, ws.state,
                                                   (size_t)b0 * metal_linalg::detail::kJacobiStateBytes, bc,
                                                   per_buffer, max_dispatches, encode,
                                                   "[svd] " + std::to_string(m) + "x" + std::to_string(n));
            continue;
        }

        id<MTLCommandBuffer> cmd = [cache.rt.queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        encode(enc);
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
    auto kernel = [&](const Matrices& m, float* u, float* s, float* vt, uint32_t* info) {
        if (opt.kernel == SvdOptions::Kernel::golub_kahan) svd_golub_kahan(m, u, s, vt, info);
        else (block ? svd_block_jacobi : svd_jacobi)(m, inner, u, s, vt, info);
    };
    const Matrices rm{r.data(), batch, k, k};
    if (!compute_uv) {
        kernel(rm, nullptr, s_out, nullptr, info_out);
        return;
    }
    HostBuffer ur((size_t)batch * k * k);
    kernel(rm, ur.data(), s_out, vt_out, info_out);
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

    // Workspace sizes, by query (LAPACK reads no array during one).
    __LAPACK_int lwork = 1, query = -1;
    {
        std::vector<__LAPACK_int> iq(8 * (size_t)K);
        float scratch = 0.0f;
        auto grow = [&](float q) { lwork = std::max<__LAPACK_int>(lwork, (__LAPACK_int)std::ceil(q)); };
        float q = 0.0f;
        if (tall) {
            sgeqrf_(&ln, &lk, &scratch, &ln, &scratch, &q, &query, &err);                              grow(q);
            if (compute_uv) { sorgqr_(&ln, &lk, &lk, &scratch, &ln, &scratch, &q, &query, &err); grow(q); }
            sgesdd_(&jobz, &lk, &lk, &scratch, &lk, s_out, &scratch, &lk, &scratch, &lk,
                    &q, &query, iq.data(), &err);                                                       grow(q);
        } else {
            sgesdd_(&jobz, &lm, &ln, &scratch, &lm, s_out, &scratch, &lm, &scratch, &lk,
                    &q, &query, iq.data(), &err);                                                       grow(q);
        }
    }

    // Non-finite input gives NaN for that matrix, as on the GPU.
    std::vector<float> amax(batch);
    std::vector<char>  finite(batch);
    scan(a, Part::all, amax.data(), finite.data());

    // Each chunk of the batch, on its own thread with its own workspace.
    lapack_batches(batch, per, [&](uint32_t b0, uint32_t b1) {
        char jz = jobz;
        __LAPACK_int lw = lwork, err = 0;
        std::vector<float> work_a(per), tau(K), r(tall ? (size_t)K * K : 1), work(lwork);
        std::vector<float> spare_u(compute_uv ? (size_t)M * K : 1), spare_vt(compute_uv ? (size_t)K * N : 1);
        std::vector<float> r_u(tall && compute_uv ? (size_t)K * K : 1), r_vt(tall && compute_uv ? (size_t)K * K : 1);
        std::vector<__LAPACK_int> iwork(8 * (size_t)K);

        auto lapack_check = [&](const char* routine, uint32_t b) {
            if (err != 0) {
                throw std::runtime_error(std::string("[svd] LAPACK ") + routine + " failed on matrix " +
                                         std::to_string(b) + " of " + std::to_string(batch) + " (" +
                                         std::to_string(M) + "x" + std::to_string(N) + "), info " +
                                         std::to_string((long long)err) + ".");
            }
        };

        for (uint32_t b = b0; b < b1; ++b) {
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
                sgesdd_(&jz, &lm, &ln, work_a.data(), &lm, s, vt, &lm, u, &lk,
                        work.data(), &lw, iwork.data(), &err);
                lapack_check("sgesdd", b);
            } else {
                vDSP_mtrans(a.data + b * per, 1, work_a.data(), 1, N, M);   // column-major A, M x N
                sgeqrf_(&ln, &lk, work_a.data(), &ln, tau.data(), work.data(), &lw, &err);
                lapack_check("sgeqrf", b);
                for (uint32_t j = 0; j < K; ++j)          // R: the upper triangle, column-major
                    for (uint32_t i = 0; i < K; ++i)
                        r[i + (size_t)j * K] = i <= j ? work_a[i + (size_t)j * M] : 0.0f;
                sgesdd_(&jz, &lk, &lk, r.data(), &lk, s, r_u.data(), &lk, r_vt.data(), &lk,
                        work.data(), &lw, iwork.data(), &err);
                lapack_check("sgesdd", b);
                if (compute_uv) {
                    sorgqr_(&ln, &lk, &lk, work_a.data(), &ln, tau.data(), work.data(), &lw, &err);
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
    });
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
    // The golub_kahan window, clipped to what the backend takes here: on the
    // matrix itself where it fits, else on the factor of a QR.
    if (p.gk_max_k != 0 && k >= p.gk_min_k && k <= std::min(p.gk_max_k, detail::svd_gk_max_k())) {
        return detail::svd_gk_fits(m, n) ? SvdBackend::golub_kahan : SvdBackend::qr_golub_kahan;
    }
    const bool precondition = l >= p.qr_min_rows && k >= p.qr_min_k && l >= 2 * k;
    const bool block = wants_block((unsigned)k, batch);
    if (precondition) return block ? SvdBackend::qr_block_jacobi : SvdBackend::qr_jacobi;
    return block ? SvdBackend::block_jacobi : SvdBackend::jacobi;
}

namespace {

// SVD_DEVICE=gpu or cpu decides alone; -1 when it does not say.
int forced_device() {
    if (const char* e = std::getenv("SVD_DEVICE")) {
        const std::string s = e;
        if (s == "gpu") return 1;
        if (s == "cpu") return 0;
    }
    return -1;
}

bool gpu_rule(unsigned m, unsigned n, unsigned batch, unsigned max_k, unsigned min_bk, unsigned min_batch,
              unsigned max_l) {
    const unsigned k = std::min(m, n), l = std::max(m, n);
    return k <= max_k && l <= max_l && (unsigned long long)batch * k >= min_bk && batch >= min_batch;
}

} // namespace

bool svd_uses_gpu(unsigned m, unsigned n, unsigned batch) {
    if (const int f = forced_device(); f >= 0) return f == 1;
    const SvdPolicy& p = policy_state().policy;
    const unsigned k = std::min(m, n);
    // gpu_max_k = 0 is never the GPU, the clause included.
    if (p.gpu_big_batch_min && p.gpu_max_k && k > p.gpu_max_k && k <= p.gpu_big_batch_max_k &&
        std::max(m, n) <= p.gpu_max_l && batch >= p.gpu_big_batch_min) {
        return true;   // the large-batch clause, above the product rule's cap
    }
    return gpu_rule(m, n, batch, p.gpu_max_k, p.gpu_min_batch_times_k, p.gpu_min_batch, p.gpu_max_l);
}

bool svdvals_uses_gpu(unsigned m, unsigned n, unsigned batch) {
    if (const int f = forced_device(); f >= 0) return f == 1;
    const SvdPolicy& p = policy_state().policy;
    if (p.values_gpu_min_batch == 0) return svd_uses_gpu(m, n, batch);   // not measured apart
    return gpu_rule(m, n, batch, p.values_gpu_max_k, p.values_gpu_min_batch_times_k, p.values_gpu_min_batch,
                    p.values_gpu_max_l);
}

namespace {

// Where the rules send a call to the CPU: a batch of mid-size matrices to the
// bidiag_batch backend inside its window; otherwise the band backend from its
// threshold (band_min_k with vectors, values_band_min_k without), then the
// bidiag backend from its own (0 = never), both up to the batch cap (0 =
// none). SVD_DEVICE=cpu keeps the CPU; SVD_DEVICE=bidiag, band or
// bidiag_batch forces that backend for every call.
SvdBackend route(unsigned m, unsigned n, unsigned batch, bool vectors) {
    if (const char* e = std::getenv("SVD_DEVICE"); e && std::string(e) == "bidiag") return SvdBackend::bidiag;
    if (const char* e = std::getenv("SVD_DEVICE"); e && std::string(e) == "band") return SvdBackend::band;
    if (const char* e = std::getenv("SVD_DEVICE"); e && std::string(e) == "bidiag_batch")
        return SvdBackend::bidiag_batch;
    if (vectors ? svd_uses_gpu(m, n, batch) : svdvals_uses_gpu(m, n, batch)) return svd_gpu_backend(m, n, batch);
    if (const char* e = std::getenv("SVD_DEVICE"); e && std::string(e) == "cpu") return SvdBackend::cpu;
    const SvdPolicy& p = policy_state().policy;
    const unsigned k = std::min(m, n), l = std::max(m, n);
    {
        const unsigned lo = vectors ? p.bidiag_batch_min_k : p.values_bidiag_batch_min_k;
        const unsigned hi = vectors ? p.bidiag_batch_max_k : p.values_bidiag_batch_max_k;
        const unsigned mb = vectors ? p.bidiag_batch_min_batch : p.values_bidiag_batch_min_batch;
        const unsigned ml = vectors ? p.bidiag_batch_max_l : p.values_bidiag_batch_max_l;
        // (the backend takes rows and columns up to 1024, its panel kernel's
        // threadgroup memory, but bidiagonalizes only R of a matrix at least
        // twice as tall as wide, or of its transpose if as wide)
        const bool fits = (l >= 2 * k && k <= 1024) || l <= 1024;
        if (hi != 0 && k >= lo && k <= hi && l <= ml && fits && batch >= std::max(mb, 1u))
            return SvdBackend::bidiag_batch;
    }
    const unsigned cap = vectors ? p.bidiag_max_batch : p.values_bidiag_max_batch;
    if (cap != 0 && batch > cap) return SvdBackend::cpu;
    const unsigned band = vectors ? p.band_min_k : p.values_band_min_k;
    if (band != 0 && k >= band) return SvdBackend::band;
    const unsigned from = vectors ? p.bidiag_min_k : p.values_bidiag_min_k;
    return from != 0 && k >= from ? SvdBackend::bidiag : SvdBackend::cpu;
}

} // namespace

SvdBackend svd_backend(unsigned m, unsigned n, unsigned batch) { return route(m, n, batch, true); }

SvdBackend svdvals_backend(unsigned m, unsigned n, unsigned batch) { return route(m, n, batch, false); }

namespace {

bool shares(SvdBackend b, unsigned batch) {
    const unsigned from = policy_state().policy.share_min_batch;
    return from != 0 && batch >= from && (b == SvdBackend::golub_kahan || b == SvdBackend::qr_golub_kahan);
}

// The smallest GPU chunk worth a dispatch: eight matrices per core. The CPU's
// chunks are a few matrices per worker (share_batch).
uint32_t gpu_share_chunk(uint32_t) {
    return 8 * std::max(1u, gpu_core_count());
}

} // namespace

bool svd_shares_batch(unsigned m, unsigned n, unsigned batch) { return shares(svd_backend(m, n, batch), batch); }

bool svdvals_shares_batch(unsigned m, unsigned n, unsigned batch) {
    return shares(svdvals_backend(m, n, batch), batch);
}

void core::detail::svd_golub_kahan_shared(const Matrices& a, float* u, float* s, float* vt, uint32_t* info) {
    const uint32_t M = a.rows, N = a.cols, K = std::min(M, N);
    if (K == 0 || a.batch == 0) {
        if (info) std::fill(info, info + a.batch, 0u);
        return;
    }
    const bool direct = metal_linalg::detail::svd_gk_fits(M, N);
    auto part = [&](uint32_t b0, uint32_t count) {
        return std::make_tuple(Matrices{a.data + (size_t)b0 * M * N, count, M, N},
                               u ? u + (size_t)b0 * M * K : nullptr, s + (size_t)b0 * K,
                               vt ? vt + (size_t)b0 * K * N : nullptr, info ? info + b0 : nullptr);
    };
    metal_linalg::detail::share_batch(
        a.batch, gpu_share_chunk(a.batch), std::clamp(a.batch / (16 * std::max(1u, cpu_threads())), 1u, 16u),
        [&](uint32_t b0, uint32_t count) {
            auto [m, pu, ps, pvt, pi] = part(b0, count);
            if (direct) {
                svd_golub_kahan(m, pu, ps, pvt, pi);
            } else {
                SvdOptions opt;
                opt.kernel = SvdOptions::Kernel::golub_kahan;
                svd_qr_jacobi(m, opt, pu, ps, pvt, pi);
            }
        },
        [&](uint32_t b0, uint32_t count) {
            auto [m, pu, ps, pvt, pi] = part(b0, count);
            svd_cpu(m, pu, ps, pvt, pi);
        });
}

void core::svd(const Matrices& a, float* u, float* s, float* vt, uint32_t* info) {
    const unsigned m = a.rows, n = a.cols, batch = a.batch;
    if (m == 0 || n == 0 || batch == 0) {
        core::detail::svd_jacobi(a, SvdOptions{}, u, s, vt, info);   // handles empties
        return;
    }
    SvdOptions opt;
    const SvdBackend backend = route(m, n, batch, u || vt);
    if (shares(backend, batch)) {
        core::detail::svd_golub_kahan_shared(a, u, s, vt, info);
        return;
    }
    switch (backend) {
        case SvdBackend::cpu:
            core::detail::svd_cpu(a, u, s, vt, info);
            return;
        case SvdBackend::bidiag:
            core::detail::svd_bidiag(a, u, s, vt, info);
            return;
        case SvdBackend::bidiag_batch:
            core::detail::svd_bidiag_batch(a, u, s, vt, info);
            return;
        case SvdBackend::band:
            if (u || vt) core::detail::svd_band_vectors(a, u, s, vt, info);
            else core::detail::svd_band(a, s, info, policy_state().policy.values_band_width);
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
        case SvdBackend::golub_kahan:
            core::detail::svd_golub_kahan(a, u, s, vt, info);
            return;
        case SvdBackend::qr_golub_kahan:
            opt.kernel = SvdOptions::Kernel::golub_kahan;
            core::detail::svd_qr_jacobi(a, opt, u, s, vt, info);
            return;
        default:
            core::detail::svd_jacobi(a, opt, u, s, vt, info);
            return;
    }
}

} // namespace metal_linalg
