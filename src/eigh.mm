#ifndef ACCELERATE_NEW_LAPACK
#define ACCELERATE_NEW_LAPACK   // LAPACK's current interface; before any Accelerate header
#endif
#include <metal_linalg/core.h>
#include <metal_linalg/device.h>
#include "calibration.h"
#include "divide_conquer.h"
#include "estimate.h"
#include "metal_runtime.h"
#include "shaders.h"

#import <Metal/Metal.h>
#include <Accelerate/Accelerate.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

using metal_linalg::core::Matrices;
using metal_linalg::detail::AutoreleasePool;
using metal_linalg::detail::MetalRuntime;
using metal_linalg::detail::Part;
using metal_linalg::detail::copy_out;
using metal_linalg::detail::input_buffer;
using metal_linalg::detail::lapack_batches;
using metal_linalg::detail::make_pipeline;
using metal_linalg::detail::pad_up;
using metal_linalg::detail::scan;

namespace metal_linalg {
namespace {

// Must match `EighParams` in Eigh_Jacobi.metal.
struct Params {
    uint  n;
    uint  n_pairs;
    uint  batch;
    uint  max_sweeps;
    float tol;
    uint  lower;
    uint  matrices_per_tg;
    uint  tg_stride_floats;
    uint  round_budget;   // threadgroup mode: rounds per dispatch (0: the whole solve)
};

// Reduction slots at the start of threadgroup memory (kRedFloats in the shader).
constexpr uint kRedFloats = 32;

// -----------------------------------------------------------------------------
// Launch parameters of the whole-matrix kernel
// -----------------------------------------------------------------------------
// Measured on an 8-core Apple M1 with `benchmark_eigh --tune`
// (docs/studies/eigh-launch-parameters-apple-m1.md). These scale with the
// detected core count or are conservative bounds, so unlike the routing
// policy below they are not per-device table entries.

// Simd mode: at most this many simdgroups (matrices) per threadgroup, and
// never so many that fewer than kSimdMinThreadgroupsPerCore threadgroups per
// GPU core remain.
constexpr unsigned kSimdMatricesPerTg = 8;
constexpr unsigned kSimdMinThreadgroupsPerCore = 4;

// Threadgroup mode: threads per matrix. Two regimes, measured on an M1 (see
// docs/studies/eigh-launch-parameters-apple-m1.md):
//
//   few matrices   one work item (element pair) per thread, up to the 1024
//                  cap: the matrix has the core to itself, so parallelise it.
//   many matrices  about kItemsPerThreadLargeBatch items per thread: smaller
//                  threadgroups co-reside on a core and hide each other's
//                  barriers, and total throughput is what matters.
//
// The switch is where the batch would need more than kThreadsPerCoreBudget
// threads per core at one item per thread.
constexpr unsigned kItemsPerThreadLargeBatch = 6;
constexpr unsigned kThreadsPerCoreBudget     = 1024;

// Cost model for chunking: core-milliseconds per matrix is about this times
// N^3 in threadgroup mode at full thread count, scaled up when fewer threads
// are used, and up to kSimdCostFactor more in simd mode at the top of its
// range. Only used to bound the work per command buffer, so a faster GPU
// simply gets smaller chunks than it needs.
constexpr double kCoreMsPerN3    = 8e-6;
constexpr double kSimdCostFactor = 4.0;
constexpr double kChunkBudgetMs  = 750.0;   // wall time per command buffer

unsigned env_uint(const char* name, unsigned fallback) {
    if (const char* s = std::getenv(name)) {
        const long v = std::strtol(s, nullptr, 10);
        if (v > 0) return (unsigned)v;
    }
    return fallback;
}

struct Workspace {
    id<MTLBuffer> W;      // working copy of A; eigenvectors on output
    id<MTLBuffer> V;      // eigenvector accumulator (nil when not computing vectors)
    id<MTLBuffer> vals;
    id<MTLBuffer> info;
    id<MTLBuffer> state;  // a split solve's per-matrix JacobiState
};

struct Cache {
    MetalRuntime& rt = MetalRuntime::shared(METAL_LINALG_SHADER(Eigh_Jacobi), "eigh");

    std::map<std::pair<bool, bool>, id<MTLComputePipelineState>> pipelines;  // (vectors, simd)
    std::map<std::tuple<uint, uint, bool>, Workspace>            workspaces; // (batch, n, vectors)

    id<MTLComputePipelineState> get_pipeline(bool vectors, bool simd) {
        const auto key = std::make_pair(vectors, simd);
        if (auto it = pipelines.find(key); it != pipelines.end()) return it->second;

        MTLFunctionConstantValues* cv = [[MTLFunctionConstantValues alloc] init];
        [cv setConstantValue:&vectors type:MTLDataTypeBool atIndex:0];
        [cv setConstantValue:&simd    type:MTLDataTypeBool atIndex:1];
        return pipelines[key] = make_pipeline(rt.device, rt.library, @"eigh_jacobi", cv);
    }

    Workspace get_workspace(uint batch, uint n, bool vectors) {
        const auto key = std::make_tuple(batch, n, vectors);
        if (auto it = workspaces.find(key); it != workspaces.end()) return it->second;

        const MTLResourceOptions opt = MTLResourceStorageModeShared;
        const size_t mat_bytes = (size_t)batch * n * n * sizeof(float);
        Workspace w;
        w.W    = [rt.device newBufferWithLength:mat_bytes options:opt];
        w.V    = vectors ? [rt.device newBufferWithLength:mat_bytes options:opt] : nil;
        w.vals = [rt.device newBufferWithLength:((size_t)batch * n * sizeof(float)) options:opt];
        w.info = [rt.device newBufferWithLength:((size_t)batch * sizeof(uint)) options:opt];
        w.state = [rt.device newBufferWithLength:((size_t)batch * metal_linalg::detail::kJacobiStateBytes)
                                         options:opt];
        return workspaces[key] = w;
    }
};

// -----------------------------------------------------------------------------
// Routing policy
// -----------------------------------------------------------------------------
// One entry per device that has actually been measured, keyed on the Metal
// device name and the GPU core count (same-name parts ship with different
// core counts). Extrapolating between entries is not safe: the GPU/CPU
// boundary depends on the ratio of GPU throughput to CPU throughput, both of
// which change across generations, and the block crossover on core count and
// launch latency. Hence a table, not a formula. A device with no entry gets
// an estimated row (kEstimated below): a measured device's timings refitted
// for a GPU weaker against its CPU by what published benchmarks say, with a
// margin (estimate.h).
//
// To add a device: run `python3 tuning/tune_eigh.py build/sweep_eigh` on it and
// paste the row it prints.
struct TunedEntry {
    const char* device_name;        // exact MTLDevice.name
    unsigned    gpu_cores;
    unsigned    simd_max_n;
    unsigned    block_min_n;
    unsigned    block_min_n_batched; // 0, 0 = no batch-dependent block crossover
    unsigned    block_min_batch;
    unsigned    gpu_max_n;
    unsigned    gpu_min_batch_times_n;
    unsigned    gpu_min_batch;
    // Eigenvalues alone. Rows measured before these existed leave them 0,
    // which means "as for eigenvectors" (values_gpu_min_batch = 0).
    unsigned    values_gpu_max_n;
    unsigned    values_gpu_min_batch_times_n;
    unsigned    values_gpu_min_batch;
    // The tridiag backend instead of the CPU from these N; 0 = never, which
    // rows measured before the backend existed leave.
    unsigned    tridiag_min_n;
    unsigned    values_tridiag_min_n;
    // ... for batches of at most this many matrices; 0 = any batch, which
    // rows measured before the CPU path used every core leave.
    unsigned    tridiag_max_batch;
    unsigned    values_tridiag_max_batch;
    // The ql backend for N in [ql_min_n, ql_max_n]; 0, 0 = never, which rows
    // measured before the backend existed leave.
    unsigned    ql_min_n;
    unsigned    ql_max_n;
    unsigned    share_min_batch;   // 0 = never, which rows measured before 2.11.0 leave
    // The large-batch clause; 0, 0 = never, which rows from before 2.12.0 leave.
    unsigned    gpu_big_batch_max_n;
    unsigned    gpu_big_batch_min;
    unsigned    values_band_min_n;   // 0 = never, which rows from before 2.13.0 leave
    unsigned    values_band_width;   // 0 = 16, which rows from before 2.15.0 leave
    // Rows from before 2.17.0 leave these 0: never.
    unsigned    band_min_n;
    unsigned    tridiag_batch_min_n;
    unsigned    tridiag_batch_max_n;
    unsigned    tridiag_batch_min_batch;
    unsigned    values_tridiag_batch_min_n;
    unsigned    values_tridiag_batch_max_n;
    unsigned    values_tridiag_batch_min_batch;
    unsigned    share_min_n;   // 0 = any N, which rows from before 2.17.0 leave
    unsigned    calibration;   // kCalibration* (calibration.h); rows without it are current
};

// The rows are generated from every run submitted for a device (docs/results/)
// by tuning/generate_tables.py, which a GitHub Action reruns after each
// merge; see docs/tuning.md. Why the measured values are what they are is in
// docs/studies/. The last row keeps the array non-empty and matches nothing.
constexpr TunedEntry kTuned[] = {
#include "tuned/eigh.inc"
    {"", 0,   0, 0, 0, 0,   0, 0, 0,   0, 0, 0,   0, 0, 0, 0,   0, 0,   0,   0, 0,   0, 0,   0,   0, 0, 0,   0, 0, 0,   0},
};

// The estimated rows, for every slowdown pair on tuning/estimate.py's ladder
// and every measured device that serves as an anchor. Generated with kTuned.
struct EstimatedEntry {
    unsigned   small_x100;        // batched kernels' slowdown x 100
    unsigned   large_x100;        // large-matrix backends' slowdown x 100
    unsigned   anchor_cpu_cores;
    TunedEntry row;
};
constexpr EstimatedEntry kEstimated[] = {
#include "tuned/eigh_estimated.inc"
    {0, 0, 0, {"", 0,   0, 0, 0, 0,   0, 0, 0,   0, 0, 0,   0, 0, 0, 0,   0, 0,   0,   0, 0,   0, 0,   0,   0, 0, 0,   0, 0, 0,   0}},
};

void apply(const TunedEntry& e, EighPolicy& p) {
    p.simd_max_n                   = e.simd_max_n;
    p.block_min_n                  = e.block_min_n;
    p.block_min_n_batched          = e.block_min_n_batched;
    p.block_min_batch              = e.block_min_batch;
    p.gpu_max_n                    = e.gpu_max_n;
    p.gpu_min_batch_times_n        = e.gpu_min_batch_times_n;
    p.gpu_min_batch                = e.gpu_min_batch;
    p.values_gpu_max_n             = e.values_gpu_max_n;
    p.values_gpu_min_batch_times_n = e.values_gpu_min_batch_times_n;
    p.values_gpu_min_batch         = e.values_gpu_min_batch;
    p.tridiag_min_n                = e.tridiag_min_n;
    p.values_tridiag_min_n         = e.values_tridiag_min_n;
    p.tridiag_max_batch            = e.tridiag_max_batch;
    p.values_tridiag_max_batch     = e.values_tridiag_max_batch;
    p.ql_min_n                     = e.ql_min_n;
    p.ql_max_n                     = e.ql_max_n;
    p.share_min_batch              = e.share_min_batch;
    p.gpu_big_batch_max_n          = e.gpu_big_batch_max_n;
    p.gpu_big_batch_min            = e.gpu_big_batch_min;
    p.values_band_min_n            = e.values_band_min_n;
    p.values_band_width            = e.values_band_width;
    p.band_min_n                     = e.band_min_n;
    p.tridiag_batch_min_n            = e.tridiag_batch_min_n;
    p.tridiag_batch_max_n            = e.tridiag_batch_max_n;
    p.tridiag_batch_min_batch        = e.tridiag_batch_min_batch;
    p.values_tridiag_batch_min_n     = e.values_tridiag_batch_min_n;
    p.values_tridiag_batch_max_n     = e.values_tridiag_batch_max_n;
    p.values_tridiag_batch_min_batch = e.values_tridiag_batch_min_batch;
    p.share_min_n                    = e.share_min_n;
}

struct ResolvedPolicy {
    EighPolicy  policy;
    std::string source;
    std::string device;
};

// Parses a non-negative integer from the environment. Zero is a valid value
// (simd_max_n = 0 disables simd mode), so this is not env_uint.
bool env_value(const char* name, unsigned& out) {
    const char* s = std::getenv(name);
    if (!s || !*s) return false;
    char* end = nullptr;
    const long long v = std::strtoll(s, &end, 10);
    if (end == s || v < 0) return false;
    out = v > 0xFFFFFFFFll ? kEighNoLimit : (unsigned)v;
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
            detail::calibration_notice("eigh", e.calibration);
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
            detail::calibration_notice("eigh", kUncalibrated);
        }
    }
    if (r.source.empty()) {
        // Nothing to estimate from (no Metal device, or no measured anchor):
        // the EighPolicy defaults.
        r.source = "default:untuned-device" + (r.device.empty() ? "" : " (" + r.device + ")");
        detail::calibration_notice("eigh", kUncalibrated);
    }

    // Environment overrides, for retuning without a rebuild.
    std::string env;
    auto over = [&](const char* name, unsigned& field) {
        if (env_value(name, field)) env += (env.empty() ? "" : ",") + std::string(name);
    };
    over("EIGH_SIMD_MAX_N",            r.policy.simd_max_n);
    over("EIGH_BLOCK_MIN_N",           r.policy.block_min_n);
    over("EIGH_BLOCK_MIN_N_BATCHED",   r.policy.block_min_n_batched);
    over("EIGH_BLOCK_MIN_BATCH",       r.policy.block_min_batch);
    over("EIGH_GPU_MAX_N",             r.policy.gpu_max_n);
    over("EIGH_GPU_MIN_BATCH_TIMES_N", r.policy.gpu_min_batch_times_n);
    over("EIGH_GPU_MIN_BATCH",         r.policy.gpu_min_batch);
    over("EIGH_VALUES_GPU_MAX_N",             r.policy.values_gpu_max_n);
    over("EIGH_VALUES_GPU_MIN_BATCH_TIMES_N", r.policy.values_gpu_min_batch_times_n);
    over("EIGH_VALUES_GPU_MIN_BATCH",         r.policy.values_gpu_min_batch);
    over("EIGH_TRIDIAG_MIN_N",                r.policy.tridiag_min_n);
    over("EIGH_VALUES_TRIDIAG_MIN_N",         r.policy.values_tridiag_min_n);
    over("EIGH_TRIDIAG_MAX_BATCH",            r.policy.tridiag_max_batch);
    over("EIGH_VALUES_TRIDIAG_MAX_BATCH",     r.policy.values_tridiag_max_batch);
    over("EIGH_QL_MIN_N",                     r.policy.ql_min_n);
    over("EIGH_QL_MAX_N",                     r.policy.ql_max_n);
    over("EIGH_SHARE_MIN_BATCH",              r.policy.share_min_batch);
    over("EIGH_GPU_BIG_BATCH_MAX_N",          r.policy.gpu_big_batch_max_n);
    over("EIGH_GPU_BIG_BATCH_MIN",            r.policy.gpu_big_batch_min);
    over("EIGH_VALUES_BAND_MIN_N",            r.policy.values_band_min_n);
    over("EIGH_VALUES_BAND_WIDTH",            r.policy.values_band_width);
    over("EIGH_BAND_MIN_N",                   r.policy.band_min_n);
    over("EIGH_TRIDIAG_BATCH_MIN_N",          r.policy.tridiag_batch_min_n);
    over("EIGH_TRIDIAG_BATCH_MAX_N",          r.policy.tridiag_batch_max_n);
    over("EIGH_TRIDIAG_BATCH_MIN_BATCH",      r.policy.tridiag_batch_min_batch);
    over("EIGH_VALUES_TRIDIAG_BATCH_MIN_N",     r.policy.values_tridiag_batch_min_n);
    over("EIGH_VALUES_TRIDIAG_BATCH_MAX_N",     r.policy.values_tridiag_batch_max_n);
    over("EIGH_VALUES_TRIDIAG_BATCH_MIN_BATCH", r.policy.values_tridiag_batch_min_batch);
    over("EIGH_SHARE_MIN_N",                  r.policy.share_min_n);
    if (!env.empty()) r.source = "env:" + env;
    return r;
}

ResolvedPolicy& policy_state() {
    static ResolvedPolicy s = resolve_policy();
    return s;
}

} // namespace

namespace detail {

unsigned eigh_simd_max_n()  { return policy_state().policy.simd_max_n; }
unsigned eigh_block_min_n() { return policy_state().policy.block_min_n; }

} // namespace detail

namespace core::detail {

void eigh_jacobi(const Matrices& a, bool lower, const EighOptions& opt,
                 float* w_out, float* v_out, uint32_t* info_out) {
    const uint n = a.cols;
    if (a.rows != n) {
        throw std::invalid_argument("[eigh] Input matrices must be square.");
    }
    if (n > 0xFFFFu) {
        throw std::invalid_argument("[eigh] N exceeds the 16-bit index used in threadgroup memory.");
    }
    const uint batch = a.batch;
    const bool compute_vectors = v_out != nullptr;
    if (n == 0 || batch == 0) {
        if (info_out) std::fill(info_out, info_out + batch, 0u);
        return;
    }
    AutoreleasePool pool;

    // 1. Resolve the launch geometry.
    static Cache cache;
    id<MTLDevice> dev = cache.rt.device;

    const uint np = (n + 1) / 2;
    bool simd;
    switch (opt.mode) {
        case EighOptions::Mode::simd:        simd = true;  break;
        case EighOptions::Mode::threadgroup: simd = false; break;
        case EighOptions::Mode::block:       simd = false; break;   // not this backend's call
        default:                             simd = n <= policy_state().policy.simd_max_n; break;
    }
    if (const char* m = std::getenv("EIGH_MODE")) {
        const std::string s = m;
        if (s == "simd") simd = true; else if (s == "threadgroup") simd = false;
    }

    id<MTLComputePipelineState> pso = cache.get_pipeline(compute_vectors, simd);
    const uint max_threads = (uint)pso.maxTotalThreadsPerThreadgroup;

    uint cores = gpu_core_count();
    if (cores == 0) cores = 8;

    // Per-matrix scratch: c, s, dp, dq (floats) and p, q (ushorts) per pair,
    // padded to 16 bytes.
    const uint stride_floats = pad_up(4 * np + (2 * np + 1) / 2, 4);

    uint threads, matrices_per_tg;
    if (simd) {
        // Enough threadgroups to occupy every core several times over, and
        // no more matrices per threadgroup than that allows: at batch 64 on
        // 8 cores, 8 matrices per threadgroup would leave one threadgroup per
        // core and most of each core idle.
        if (opt.matrices_per_threadgroup) {
            matrices_per_tg = opt.matrices_per_threadgroup;
        } else {
            const uint min_tgs = kSimdMinThreadgroupsPerCore * cores;
            matrices_per_tg = std::max(1u, std::min(kSimdMatricesPerTg, batch / min_tgs));
        }
        matrices_per_tg = std::max(1u, std::min(matrices_per_tg, max_threads / 32));
        threads = 32 * matrices_per_tg;
    } else {
        matrices_per_tg = 1;
        const uint items = np * n;   // work items per phase
        if (opt.threads) {
            threads = pad_up(opt.threads, 32);
        } else {
            const uint per_matrix_budget = std::max(32u, kThreadsPerCoreBudget * cores / batch);
            threads = std::max(items / kItemsPerThreadLargeBatch, std::min(items, per_matrix_budget));
            threads = pad_up(std::max(threads, 32u), 32);
        }
        threads = std::min(threads, max_threads);
    }

    const size_t tg_bytes = (size_t)(kRedFloats + matrices_per_tg * stride_floats) * sizeof(float);
    if (tg_bytes > dev.maxThreadgroupMemoryLength) {
        throw std::invalid_argument(
            "[eigh] N=" + std::to_string(n) + " needs " + std::to_string(tg_bytes) +
            " bytes of threadgroup memory but this device allows " +
            std::to_string((size_t)dev.maxThreadgroupMemoryLength) +
            (simd ? "; use threadgroup mode or fewer matrices per threadgroup."
                  : "."));
    }

    // Work per command buffer. macOS kills a command buffer that runs for
    // more than a couple of seconds ("Impacting Interactivity"), so a large
    // batch is split into chunks sized by a conservative cost model. A chunk
    // never goes below the core count: fewer matrices than cores leaves cores
    // idle without making the chunk finish sooner, since each matrix is one
    // threadgroup. That per-matrix time is what bounds the practical N for
    // this design.
    uint chunk = batch;
    // Fewer threads than the one-item-per-thread count stretch a matrix's
    // wall time roughly in proportion; assume the worst for the budget.
    const uint   full_threads  = std::min(max_threads, pad_up(std::max(np * n, 32u), 32));
    const double thread_factor = simd ? kSimdCostFactor
                                      : std::max(1.0, (double)full_threads / threads);
    const double per_matrix_core_ms = kCoreMsPerN3 * (double)n * n * n * thread_factor;
    // A threadgroup's solve longer than a dispatch should run (the GPU's
    // watchdog; metal_runtime.h) is split over dispatches of a few rounds.
    const uint rounds = 2 * np - 1;
    const uint round_budget =
        simd ? 0u : metal_linalg::detail::jacobi_round_budget(per_matrix_core_ms, rounds, "EIGH_DISPATCH_MS");
    {
        const double budget = (double)env_uint("EIGH_CHUNK_MS", (unsigned)kChunkBudgetMs);
        const double fit = std::floor(budget * cores / per_matrix_core_ms);
        chunk = (uint)std::max((double)cores, std::min((double)batch, fit));
        chunk = std::min(chunk, batch);
    }

    // 3. Buffers.
    Workspace ws = cache.get_workspace(batch, n, compute_vectors);

    id<MTLBuffer> buf_src = input_buffer(dev, a);

    // 4. Dispatch, one command buffer per chunk.
    const size_t mat_bytes = (size_t)n * n * sizeof(float);
    for (uint b0 = 0; b0 < batch; b0 += chunk) {
        const uint bc   = std::min(chunk, batch - b0);
        const uint n_tg = simd ? (bc + matrices_per_tg - 1) / matrices_per_tg : bc;

        Params prm;
        prm.n                = n;
        prm.n_pairs          = np;
        prm.batch            = bc;
        prm.max_sweeps       = opt.max_sweeps;
        prm.tol              = opt.tol;
        prm.lower            = lower ? 1u : 0u;
        prm.matrices_per_tg  = matrices_per_tg;
        prm.tg_stride_floats = stride_floats;
        prm.round_budget     = round_budget;

        auto encode = [&](id<MTLComputeCommandEncoder> enc) {
            [enc setComputePipelineState:pso];
            [enc setBuffer:buf_src offset:(b0 * mat_bytes) atIndex:0];
            [enc setBuffer:ws.W    offset:(b0 * mat_bytes) atIndex:1];
            [enc setBuffer:(compute_vectors ? ws.V : ws.W) offset:(b0 * mat_bytes) atIndex:2];
            [enc setBuffer:ws.vals offset:((size_t)b0 * n * sizeof(float)) atIndex:3];
            [enc setBuffer:ws.info offset:((size_t)b0 * sizeof(uint)) atIndex:4];
            [enc setBytes:&prm length:sizeof(prm) atIndex:5];
            [enc setBuffer:ws.state offset:((size_t)b0 * metal_linalg::detail::kJacobiStateBytes) atIndex:6];
            [enc setThreadgroupMemoryLength:tg_bytes atIndex:0];
            [enc dispatchThreadgroups:MTLSizeMake(n_tg, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        };
        if (round_budget) {
            // Command buffers of about kChunkBudgetMs, the chunk's matrices in
            // waves of one per core, until each has finished.
            const uint waves = (bc + cores - 1) / cores;
            const uint per_buffer = std::max(1u, (uint)(kChunkBudgetMs / (40.0 * waves)));
            const uint max_dispatches = (uint)std::min<uint64_t>(
                0xFFFFFFFFu, (uint64_t)opt.max_sweeps * rounds / round_budget + 3);
            metal_linalg::detail::run_split_jacobi(cache.rt.queue, ws.state,
                                                   (size_t)b0 * metal_linalg::detail::kJacobiStateBytes, bc,
                                                   per_buffer, max_dispatches, encode,
                                                   "[eigh] N=" + std::to_string(n));
            continue;
        }

        id<MTLCommandBuffer> cmd = [cache.rt.queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        encode(enc);
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        if (cmd.error) {
            throw std::runtime_error(
                std::string("[eigh] GPU kernel error: ") + cmd.error.localizedDescription.UTF8String +
                " (N=" + std::to_string(n) + ", " + std::to_string(bc) + " matrices in this "
                "command buffer; if this is the interactivity watchdog, the matrix is too "
                "large for one threadgroup on this device).");
        }
    }

    // 5. Convergence check. Non-finite input is reported through the output
    // (NaNs) rather than an exception, like the LAPACK path; a finite matrix
    // that did not converge is a bug and is raised.
    const uint* info_ptr = static_cast<const uint*>([ws.info contents]);
    for (uint b = 0; b < batch; ++b) {
        const uint w = info_ptr[b];
        if (!metal_linalg::detail::eigh_converged(w) && !metal_linalg::detail::eigh_nonfinite(w)) {
            throw std::runtime_error("[eigh] Matrix " + std::to_string(b) + " of " +
                                     std::to_string(batch) + " (N=" + std::to_string(n) +
                                     ") did not converge in " + std::to_string(opt.max_sweeps) +
                                     " sweeps.");
        }
    }

    // 6. Copy out of the recycled workspace.
    copy_out(static_cast<const float*>([ws.vals contents]), w_out, batch, n);
    copy_out(static_cast<const float*>([ws.W contents]), v_out, batch, (size_t)n * n);
    if (info_out) std::memcpy(info_out, info_ptr, (size_t)batch * sizeof(uint32_t));
}

// The smallest n whose eigenvalues alone go through the two-stage reduction.
constexpr uint32_t kEighTwoStageMinN = 128;

// The smallest n solved in ssyevd's steps where cores are idle (eigh_cpu).
constexpr uint32_t kEighCpuDcMinN = 192;

// Whether LAPACK's two-stage driver can be trusted here. Accelerate's
// ssyevd_2stage gives wrong eigenvalues on macOS 14 (off by 1.5-7% of the
// largest for every N from 128, on GitHub's macOS 14 runners) and right ones
// on 15 and later; ssyevd is right on both. So it is used from macOS 15 only,
// and only once it has matched ssyevd on a fixed matrix, checked once per
// process (well under a millisecond), in case another release regresses.
bool two_stage_trusted() {
    static const bool trusted = [] {
        if (@available(macOS 15.0, *)) {
        } else {
            return false;
        }
        const __LAPACK_int n = 160;
        std::vector<float> a((size_t)n * n);
        for (__LAPACK_int i = 0; i < n; ++i)
            for (__LAPACK_int j = 0; j < n; ++j)
                a[(size_t)i * n + j] = std::cos(0.37f * float(i + j)) + 1.0f / float(1 + std::abs(i - j)) +
                                       (i == j ? 0.05f * float(i) : 0.0f);
        auto values = [&](auto syevd, float* w) {
            std::vector<float> m(a);
            char jobz = 'N', uplo = 'L';
            __LAPACK_int N = n, lwork = -1, liwork = -1, err = 0, liq = 0;
            float lq = 0.0f;
            syevd(&jobz, &uplo, &N, m.data(), &N, w, &lq, &lwork, &liq, &liwork, &err);
            lwork = std::max<__LAPACK_int>(1, (__LAPACK_int)std::ceil(lq));
            liwork = std::max<__LAPACK_int>(1, liq);
            std::vector<float> work(lwork);
            std::vector<__LAPACK_int> iwork(liwork);
            syevd(&jobz, &uplo, &N, m.data(), &N, w, work.data(), &lwork, iwork.data(), &liwork, &err);
            return err == 0;
        };
        std::vector<float> w1(n), w2(n);
        if (!values(ssyevd_, w1.data()) || !values(ssyevd_2stage_, w2.data())) return false;
        float d = 0.0f, scale = 0.0f;
        for (__LAPACK_int i = 0; i < n; ++i) {
            d = std::max(d, std::fabs(w1[i] - w2[i]));
            scale = std::max(scale, std::fabs(w1[i]));
        }
        return d <= 1e-4f * scale;
    }();
    return trusted;
}

namespace {

// eigh_cpu with vectors in ssyevd's steps, the divide and conquer on
// `threads` threads per matrix. uplo is the triangle as LAPACK sees it.
void eigh_cpu_steps(const Matrices& a, char uplo, const float* amax, const char* finite, unsigned threads,
                    float* w_out, float* v_out, uint32_t* info_out) {
    const uint32_t n = a.cols, batch = a.batch;
    const size_t   per = (size_t)n * n;
    __LAPACK_int N = (__LAPACK_int)n, lwork = std::max<__LAPACK_int>(1, 64 * N), query = -1, err = 0;
    {
        float q = 0.0f, scratch = 0.0f;
        ssytrd_(&uplo, &N, &scratch, &N, &scratch, &scratch, &scratch, &q, &query, &err);
        lwork = std::max<__LAPACK_int>(lwork, (__LAPACK_int)std::ceil(q));
        sormtr_("L", &uplo, "N", &N, &N, &scratch, &N, &scratch, &scratch, &N, &q, &query, &err);
        lwork = std::max<__LAPACK_int>(lwork, (__LAPACK_int)std::ceil(q));
    }
    // ssyevd's scaling: a matrix whose largest entry is outside
    // [2^-51, 2^51] is solved scaled, here by a power of two, so exactly.
    const float lo = std::ldexp(1.0f, -51), hi = std::ldexp(1.0f, 51);

    lapack_batches(batch, per, [&](uint32_t b0, uint32_t b1) {
        std::vector<float> h(per), z(per), d(n), e(n), tau(n), work(lwork);
        __LAPACK_int lw = lwork, err = 0;
        auto check = [&](const char* routine, long info, uint32_t b) {
            if (info != 0) {
                throw std::runtime_error(std::string("[eigh] ") + routine + " failed on matrix " +
                                         std::to_string(b) + " of " + std::to_string(batch) + " (N=" +
                                         std::to_string(n) + "), info " + std::to_string(info) + ".");
            }
        };
        for (uint32_t b = b0; b < b1; ++b) {
            float* w = w_out + (size_t)b * n;
            if (!finite[b]) {
                std::fill(w, w + n, NAN);
                std::fill(v_out + b * per, v_out + (b + 1) * per, NAN);
                if (info_out) info_out[b] = 1u << 17;
                continue;
            }
            int exponent = 0;
            if (amax[b] > 0.0f && (amax[b] < lo || amax[b] > hi)) std::frexp(amax[b], &exponent);
            if (exponent != 0) {
                const float scale = std::ldexp(1.0f, -exponent);
                vDSP_vsmul(a.data + b * per, 1, &scale, h.data(), 1, per);
            } else {
                std::memcpy(h.data(), a.data + b * per, per * sizeof(float));
            }
            ssytrd_(&uplo, &N, h.data(), &N, d.data(), e.data(), tau.data(), work.data(), &lw, &err);
            check("LAPACK ssytrd", err, b);
            check("tridiagonal_eigensystem",
                  metal_linalg::detail::tridiagonal_eigensystem(n, d.data(), e.data(), z.data(), n, threads), b);
            sormtr_("L", &uplo, "N", &N, &N, h.data(), &N, tau.data(), z.data(), &N, work.data(), &lw, &err);
            check("LAPACK sormtr", err, b);
            for (uint32_t i = 0; i < n; ++i) w[i] = std::ldexp(d[i], exponent);
            vDSP_mtrans(z.data(), 1, v_out + b * per, 1, n, n);   // the vectors as columns, row-major
            if (info_out) info_out[b] = 1u | (1u << 16);
        }
    });
}

} // namespace

void eigh_cpu(const Matrices& a, bool lower, float* w_out, float* v_out, uint32_t* info_out) {
    const uint32_t n = a.cols, batch = a.batch;
    if (a.rows != n) {
        throw std::invalid_argument("[eigh] Input matrices must be square.");
    }
    if (n == 0 || batch == 0) {
        if (info_out) std::fill(info_out, info_out + batch, 0u);
        return;
    }

    // LAPACK is column-major, so it sees each matrix transposed: the lower
    // triangle of the row-major matrix is the upper triangle of the one
    // LAPACK reads, and its eigenvectors come back as rows.
    char jobz = v_out ? 'V' : 'N';
    char uplo = lower ? 'U' : 'L';
    // Eigenvalues alone go through LAPACK's two-stage reduction (dense to
    // band in matrix-matrix products, then band to tridiagonal), which the
    // one-stage reduction, bound by memory bandwidth in its matrix-vector
    // half, falls far behind as n grows: on an M5 Pro 1.2x faster at
    // n = 1024, 4x at 4096, 6x at 8192, and level from 128 down to 32.
    // LAPACK's two-stage driver does not compute eigenvectors.
    const bool two_stage = !v_out && n >= kEighTwoStageMinN && two_stage_trusted();
    auto syevd = two_stage ? ssyevd_2stage_ : ssyevd_;
    // The two-stage reduction runs about 1.5x faster on the lower triangle
    // than the upper (and loses to one-stage at N = 1024 on the upper), so it
    // is always given the lower: where that is the row-major lower triangle,
    // which LAPACK would see as its upper, the matrix is transposed on copy.
    const bool transpose_in = two_stage && lower;
    if (two_stage) uplo = 'L';
    const char* routine = two_stage ? "ssyevd_2stage" : "ssyevd";
    const size_t per = (size_t)n * n;

    // Non-finite input gives NaN for that matrix, as on the GPU, rather than
    // whatever LAPACK makes of it. Only the triangle that is read counts.
    std::vector<float> amax(batch);
    std::vector<char>  finite(batch);
    scan(a, lower ? Part::lower : Part::upper, amax.data(), finite.data());

    // With vectors, and cores idle beside each matrix (a quarter as many
    // matrices as cores, at most), ssyevd's steps one by one, its divide and
    // conquer (sstedc, on one core) on the idle cores too: ssytrd,
    // tridiagonal_eigensystem, sormtr. On an M5 Pro, one matrix 1.26x faster
    // at 256, 1.11x at 512, 1.2x at 1024, 1.17x at 2048; level at 128; 4 of
    // 1024 1.2x. EIGH_CPU_DC=0 keeps ssyevd.
    using metal_linalg::detail::kCpuDcMinThreads;
    const unsigned dc_threads = v_out ? metal_linalg::detail::lapack_threads_per_matrix(batch, per) : 1;
    const char*    dc_env = std::getenv("EIGH_CPU_DC");
    if (v_out && n >= kEighCpuDcMinN && dc_threads >= kCpuDcMinThreads && !(dc_env && std::string(dc_env) == "0")) {
        eigh_cpu_steps(a, uplo, amax.data(), finite.data(), dc_threads, w_out, v_out, info_out);
        return;
    }

    // Each chunk of the batch, on its own thread with its own workspace.
    lapack_batches(batch, per, [&](uint32_t b0, uint32_t b1) {
        char jz = jobz, ul = uplo;
        std::vector<float> work_a(per);
        __LAPACK_int N = (__LAPACK_int)n, lwork = -1, liwork = -1, err = 0;
        float lwork_query = 0.0f;
        __LAPACK_int liwork_query = 0;
        syevd(&jz, &ul, &N, work_a.data(), &N, w_out, &lwork_query, &lwork,
              &liwork_query, &liwork, &err);
        lwork  = std::max<__LAPACK_int>(1, (__LAPACK_int)std::ceil(lwork_query));
        liwork = std::max<__LAPACK_int>(1, liwork_query);
        std::vector<float>        work(lwork);
        std::vector<__LAPACK_int> iwork(liwork);

        for (uint32_t b = b0; b < b1; ++b) {
            float* w = w_out + (size_t)b * n;
            if (!finite[b]) {
                std::fill(w, w + n, NAN);
                if (v_out) std::fill(v_out + b * per, v_out + (b + 1) * per, NAN);
                if (info_out) info_out[b] = 1u << 17;
                continue;
            }
            if (transpose_in) vDSP_mtrans(a.data + b * per, 1, work_a.data(), 1, n, n);
            else              std::memcpy(work_a.data(), a.data + b * per, per * sizeof(float));
            syevd(&jz, &ul, &N, work_a.data(), &N, w, work.data(), &lwork,
                  iwork.data(), &liwork, &err);
            if (err != 0) {
                throw std::runtime_error(std::string("[eigh] LAPACK ") + routine + " failed on matrix " +
                                         std::to_string(b) + " of " + std::to_string(batch) + " (N=" +
                                         std::to_string(n) + "), info " + std::to_string((long long)err) + ".");
            }
            if (v_out) vDSP_mtrans(work_a.data(), 1, v_out + b * per, 1, n, n);
            if (info_out) info_out[b] = 1u | (1u << 16);
        }
    });
}

} // namespace core::detail

EighPolicy  eigh_policy()        { return policy_state().policy; }
const char* eigh_policy_source() { return policy_state().source.c_str(); }

void set_eigh_policy(const EighPolicy& p) {
    policy_state().policy = p;
    policy_state().source = "user";
}

EighBackend eigh_gpu_backend(unsigned n, unsigned batch) {
    const EighPolicy& p = policy_state().policy;
    if (p.ql_max_n && n >= p.ql_min_n && n <= std::min(p.ql_max_n, detail::eigh_ql_max_n())) {
        return EighBackend::ql;
    }
    if (n >= p.block_min_n) return EighBackend::block;
    if (p.block_min_batch && n >= p.block_min_n_batched && batch >= p.block_min_batch) {
        return EighBackend::block;
    }
    return n <= p.simd_max_n ? EighBackend::simd : EighBackend::threadgroup;
}

namespace {

// GPU iff N <= max_n, batch * N >= min_bn and batch >= min_batch, unless
// EIGH_DEVICE forces a side.
bool gpu_rule(unsigned n, unsigned batch, unsigned max_n, unsigned min_bn, unsigned min_batch) {
    if (const char* e = std::getenv("EIGH_DEVICE")) {
        const std::string s = e;
        if (s == "gpu") return true;
        if (s == "cpu") return false;
    }
    return n <= max_n && (unsigned long long)batch * n >= min_bn && batch >= min_batch;
}

} // namespace

bool eigh_uses_gpu(unsigned n, unsigned batch) {
    if (cpu_only()) return false;   // this thread's setting (device.h), ahead of EIGH_DEVICE
    const EighPolicy& p = policy_state().policy;
    // gpu_max_n = 0 is never the GPU, the clause included.
    if (!std::getenv("EIGH_DEVICE") && p.gpu_big_batch_min && p.gpu_max_n && n > p.gpu_max_n &&
        n <= p.gpu_big_batch_max_n && batch >= p.gpu_big_batch_min) {
        return true;   // the large-batch clause, above the product rule's cap
    }
    return gpu_rule(n, batch, p.gpu_max_n, p.gpu_min_batch_times_n, p.gpu_min_batch);
}

bool eigvalsh_uses_gpu(unsigned n, unsigned batch) {
    if (cpu_only()) return false;
    const EighPolicy& p = policy_state().policy;
    if (p.values_gpu_min_batch == 0) return eigh_uses_gpu(n, batch);   // not measured apart
    return gpu_rule(n, batch, p.values_gpu_max_n, p.values_gpu_min_batch_times_n, p.values_gpu_min_batch);
}

namespace {

// Where the rules send a call to the CPU: a batch of mid-size matrices to
// the tridiag_batch backend inside its window; otherwise the band backend
// from its threshold, then the tridiag backend from its own (0 = never), both
// up to the batch cap (0 = none). EIGH_DEVICE=cpu keeps the CPU;
// EIGH_DEVICE=tridiag, =band or =tridiag_batch forces that backend for every
// call.
EighBackend cpu_side(unsigned n, unsigned batch, bool vectors) {
    if (const char* e = std::getenv("EIGH_DEVICE")) {
        const std::string s = e;
        if (s == "cpu") return EighBackend::cpu;
        if (s == "tridiag") return EighBackend::tridiag;
    }
    const EighPolicy& p = policy_state().policy;
    const unsigned bmin = vectors ? p.tridiag_batch_min_n : p.values_tridiag_batch_min_n;
    const unsigned bmax = vectors ? p.tridiag_batch_max_n : p.values_tridiag_batch_max_n;
    const unsigned bbat = vectors ? p.tridiag_batch_min_batch : p.values_tridiag_batch_min_batch;
    // (the backend takes N up to 1024, its panel kernel's threadgroup memory)
    if (bmax != 0 && n >= bmin && n <= std::min(bmax, 1024u) && batch >= std::max(bbat, 1u))
        return EighBackend::tridiag_batch;
    const unsigned cap = vectors ? p.tridiag_max_batch : p.values_tridiag_max_batch;
    if (cap != 0 && batch > cap) return EighBackend::cpu;
    if (!vectors && p.values_band_min_n != 0 && n >= p.values_band_min_n) return EighBackend::band;
    if (vectors && p.band_min_n != 0 && n >= p.band_min_n) return EighBackend::band;
    const unsigned from = vectors ? p.tridiag_min_n : p.values_tridiag_min_n;
    return from != 0 && n >= from ? EighBackend::tridiag : EighBackend::cpu;
}

bool forced_tridiag() {
    const char* e = std::getenv("EIGH_DEVICE");
    return e && std::string(e) == "tridiag";
}

bool forced_band() {
    const char* e = std::getenv("EIGH_DEVICE");
    return e && std::string(e) == "band";
}

bool forced_tridiag_batch() {
    const char* e = std::getenv("EIGH_DEVICE");
    return e && std::string(e) == "tridiag_batch";
}

EighBackend route(unsigned n, unsigned batch, bool vectors) {
    // CPU only (device.h): LAPACK alone, not the backends that reduce on the GPU
    if (cpu_only()) return EighBackend::cpu;
    if (forced_tridiag()) return EighBackend::tridiag;
    if (forced_band()) return EighBackend::band;
    if (forced_tridiag_batch()) return EighBackend::tridiag_batch;
    const bool gpu = vectors ? eigh_uses_gpu(n, batch) : eigvalsh_uses_gpu(n, batch);
    return gpu ? eigh_gpu_backend(n, batch) : cpu_side(n, batch, vectors);
}

} // namespace

EighBackend eigh_backend(unsigned n, unsigned batch) { return route(n, batch, true); }

EighBackend eigvalsh_backend(unsigned n, unsigned batch) { return route(n, batch, false); }

namespace {

// Whether a ql batch is shared with the CPU path: from share_min_batch, for N
// from share_min_n.
bool ql_shared(unsigned n, unsigned batch) {
    const EighPolicy& p = policy_state().policy;
    return p.share_min_batch != 0 && batch >= p.share_min_batch && n >= p.share_min_n;
}

} // namespace

bool eigh_shares_batch(unsigned n, unsigned batch) {
    return ql_shared(n, batch) && eigh_backend(n, batch) == EighBackend::ql;
}

bool eigvalsh_shares_batch(unsigned n, unsigned batch) {
    return ql_shared(n, batch) && eigvalsh_backend(n, batch) == EighBackend::ql;
}

void core::detail::eigh_ql_shared(const Matrices& a, bool lower, float* w, float* v, uint32_t* info) {
    const uint32_t n = a.cols;
    if (a.rows != n) throw std::invalid_argument("[eigh] Input matrices must be square.");
    if (n == 0 || a.batch == 0) {
        if (info) std::fill(info, info + a.batch, 0u);
        return;
    }
    const size_t per = (size_t)n * n;
    auto sub = [&](uint32_t b0, uint32_t count) { return Matrices{a.data + b0 * per, count, n, n}; };
    // The smallest GPU chunk worth a dispatch: eight matrices per core. The
    // CPU's chunks are a few matrices per worker (share_batch).
    const uint32_t cores = std::max(1u, gpu_core_count());
    metal_linalg::detail::share_batch(
        a.batch, 8 * cores, std::clamp(a.batch / (16 * std::max(1u, cpu_threads())), 1u, 16u),
        [&](uint32_t b0, uint32_t count) {
            core::detail::eigh_ql(sub(b0, count), lower, w + (size_t)b0 * n, v ? v + b0 * per : nullptr,
                                  info ? info + b0 : nullptr);
        },
        [&](uint32_t b0, uint32_t count) {
            core::detail::eigh_cpu(sub(b0, count), lower, w + (size_t)b0 * n, v ? v + b0 * per : nullptr,
                                   info ? info + b0 : nullptr);
        });
}

void core::eigh(const Matrices& a, bool lower, float* w, float* v, uint32_t* info) {
    if (a.rows != a.cols) {
        throw std::invalid_argument("[eigh] Input matrices must be square.");
    }
    const unsigned n = a.cols, batch = a.batch;
    if (n == 0 || batch == 0) {
        core::detail::eigh_cpu(a, lower, w, v, info);   // handles the empty case
        return;
    }
    const EighBackend backend = route(n, batch, v != nullptr);
    if (backend == EighBackend::cpu) {
        core::detail::eigh_cpu(a, lower, w, v, info);
        return;
    }
    if (backend == EighBackend::tridiag) {
        core::detail::eigh_tridiag(a, lower, w, v, info);
        return;
    }
    if (backend == EighBackend::band) {
        if (v) core::detail::eigh_band_vectors(a, lower, w, v, info);
        else   core::detail::eigh_band(a, lower, w, info, policy_state().policy.values_band_width);
        return;
    }
    if (backend == EighBackend::tridiag_batch) {
        core::detail::eigh_tridiag_batch(a, lower, w, v, info);
        return;
    }
    if (backend == EighBackend::ql) {
        if (ql_shared(n, batch)) core::detail::eigh_ql_shared(a, lower, w, v, info);
        else                            core::detail::eigh_ql(a, lower, w, v, info);
        return;
    }
    // A Jacobi backend, as the policy splits them.
    EighOptions opt;
    switch (backend) {
        case EighBackend::block:
            opt.mode = EighOptions::Mode::block;
            core::detail::eigh_block_jacobi(a, lower, opt, w, v, info);
            return;
        case EighBackend::simd:
            opt.mode = EighOptions::Mode::simd;
            break;
        default:
            opt.mode = EighOptions::Mode::threadgroup;
            break;
    }
    core::detail::eigh_jacobi(a, lower, opt, w, v, info);
}

} // namespace metal_linalg
