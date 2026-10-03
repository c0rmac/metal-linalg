#ifndef ACCELERATE_NEW_LAPACK
#define ACCELERATE_NEW_LAPACK   // LAPACK's current interface; before any Accelerate header
#endif
#include <metal_linalg/core.h>
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
// launch latency. Hence a table, not a formula.
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
};

// The rows are generated from every run submitted for a device (docs/results/)
// by tuning/generate_tables.py, which a GitHub Action reruns after each
// merge; see docs/tuning.md. Why the measured values are what they are is in
// docs/studies/. The last row keeps the array non-empty and matches nothing.
constexpr TunedEntry kTuned[] = {
#include "tuned/eigh.inc"
    {"", 0,   0, 0, 0, 0,   0, 0, 0,   0, 0, 0,   0, 0},
};

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

    for (const auto& e : kTuned) {
        if (e.device_name[0] != '\0' && r.device == e.device_name &&
            r.policy.gpu_cores == e.gpu_cores) {
            r.policy.simd_max_n            = e.simd_max_n;
            r.policy.block_min_n           = e.block_min_n;
            r.policy.block_min_n_batched   = e.block_min_n_batched;
            r.policy.block_min_batch       = e.block_min_batch;
            r.policy.gpu_max_n             = e.gpu_max_n;
            r.policy.gpu_min_batch_times_n = e.gpu_min_batch_times_n;
            r.policy.gpu_min_batch         = e.gpu_min_batch;
            r.policy.values_gpu_max_n             = e.values_gpu_max_n;
            r.policy.values_gpu_min_batch_times_n = e.values_gpu_min_batch_times_n;
            r.policy.values_gpu_min_batch         = e.values_gpu_min_batch;
            r.policy.tridiag_min_n                = e.tridiag_min_n;
            r.policy.values_tridiag_min_n         = e.values_tridiag_min_n;
            r.source = "tuned:" + r.device;
            break;
        }
    }
    if (r.source.empty()) {
        // No measurements for this GPU: the EighPolicy defaults, which are the
        // M1 values. They err toward the CPU, and that is the safe direction,
        // since the CPU path is never catastrophic: the cost of being untuned
        // is a missed GPU win, not a call routed to a backend that takes
        // seconds. A GPU with more cores than an M1 will want a higher
        // gpu_max_n than this.
        r.source = "default:untuned-device" + (r.device.empty() ? "" : " (" + r.device + ")");
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
    {
        // Fewer threads than the one-item-per-thread count stretch a matrix's
        // wall time roughly in proportion; assume the worst for the budget.
        const uint   full_threads  = std::min(max_threads, pad_up(std::max(np * n, 32u), 32));
        const double thread_factor = simd ? kSimdCostFactor
                                          : std::max(1.0, (double)full_threads / threads);
        const double per_matrix_core_ms = kCoreMsPerN3 * (double)n * n * n * thread_factor;
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

        id<MTLCommandBuffer> cmd = [cache.rt.queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

        [enc setComputePipelineState:pso];
        [enc setBuffer:buf_src offset:(b0 * mat_bytes) atIndex:0];
        [enc setBuffer:ws.W    offset:(b0 * mat_bytes) atIndex:1];
        [enc setBuffer:(compute_vectors ? ws.V : ws.W) offset:(b0 * mat_bytes) atIndex:2];
        [enc setBuffer:ws.vals offset:((size_t)b0 * n * sizeof(float)) atIndex:3];
        [enc setBuffer:ws.info offset:((size_t)b0 * sizeof(uint)) atIndex:4];
        [enc setBytes:&prm length:sizeof(prm) atIndex:5];
        [enc setThreadgroupMemoryLength:tg_bytes atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(n_tg, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];

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
    const bool two_stage = !v_out && n >= kEighTwoStageMinN;
    auto syevd = two_stage ? ssyevd_2stage_ : ssyevd_;
    // The two-stage reduction runs about 1.5x faster on the lower triangle
    // than the upper (and loses to one-stage at N = 1024 on the upper), so it
    // is always given the lower: where that is the row-major lower triangle,
    // which LAPACK would see as its upper, the matrix is transposed on copy.
    const bool transpose_in = two_stage && lower;
    if (two_stage) uplo = 'L';
    const char* routine = two_stage ? "ssyevd_2stage" : "ssyevd";
    const size_t per = (size_t)n * n;
    std::vector<float> work_a(per);
    __LAPACK_int N = (__LAPACK_int)n, lwork = -1, liwork = -1, err = 0;
    float lwork_query = 0.0f;
    __LAPACK_int liwork_query = 0;
    syevd(&jobz, &uplo, &N, work_a.data(), &N, w_out, &lwork_query, &lwork,
          &liwork_query, &liwork, &err);
    lwork  = std::max<__LAPACK_int>(1, (__LAPACK_int)std::ceil(lwork_query));
    liwork = std::max<__LAPACK_int>(1, liwork_query);
    std::vector<float>        work(lwork);
    std::vector<__LAPACK_int> iwork(liwork);

    // Non-finite input gives NaN for that matrix, as on the GPU, rather than
    // whatever LAPACK makes of it. Only the triangle that is read counts.
    std::vector<float> amax(batch);
    std::vector<char>  finite(batch);
    scan(a, lower ? Part::lower : Part::upper, amax.data(), finite.data());

    for (uint32_t b = 0; b < batch; ++b) {
        float* w = w_out + (size_t)b * n;
        if (!finite[b]) {
            std::fill(w, w + n, NAN);
            if (v_out) std::fill(v_out + b * per, v_out + (b + 1) * per, NAN);
            if (info_out) info_out[b] = 1u << 17;
            continue;
        }
        if (transpose_in) vDSP_mtrans(a.data + b * per, 1, work_a.data(), 1, n, n);
        else              std::memcpy(work_a.data(), a.data + b * per, per * sizeof(float));
        syevd(&jobz, &uplo, &N, work_a.data(), &N, w, work.data(), &lwork,
              iwork.data(), &liwork, &err);
        if (err != 0) {
            throw std::runtime_error(std::string("[eigh] LAPACK ") + routine + " failed on matrix " + std::to_string(b) +
                                     " of " + std::to_string(batch) + " (N=" + std::to_string(n) +
                                     "), info " + std::to_string((long long)err) + ".");
        }
        if (v_out) vDSP_mtrans(work_a.data(), 1, v_out + b * per, 1, n, n);
        if (info_out) info_out[b] = 1u | (1u << 16);
    }
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
    const EighPolicy& p = policy_state().policy;
    return gpu_rule(n, batch, p.gpu_max_n, p.gpu_min_batch_times_n, p.gpu_min_batch);
}

bool eigvalsh_uses_gpu(unsigned n, unsigned batch) {
    const EighPolicy& p = policy_state().policy;
    if (p.values_gpu_min_batch == 0) return eigh_uses_gpu(n, batch);   // not measured apart
    return gpu_rule(n, batch, p.values_gpu_max_n, p.values_gpu_min_batch_times_n, p.values_gpu_min_batch);
}

namespace {

// Where the rules send a call to the CPU: the tridiag backend instead, from
// the policy's threshold (0 = never). EIGH_DEVICE=cpu keeps the CPU;
// EIGH_DEVICE=tridiag forces this backend for every call.
EighBackend cpu_side(unsigned n, bool vectors) {
    if (const char* e = std::getenv("EIGH_DEVICE")) {
        const std::string s = e;
        if (s == "cpu") return EighBackend::cpu;
        if (s == "tridiag") return EighBackend::tridiag;
    }
    const EighPolicy& p = policy_state().policy;
    const unsigned from = vectors ? p.tridiag_min_n : p.values_tridiag_min_n;
    return from != 0 && n >= from ? EighBackend::tridiag : EighBackend::cpu;
}

bool forced_tridiag() {
    const char* e = std::getenv("EIGH_DEVICE");
    return e && std::string(e) == "tridiag";
}

EighBackend route(unsigned n, unsigned batch, bool vectors) {
    if (forced_tridiag()) return EighBackend::tridiag;
    const bool gpu = vectors ? eigh_uses_gpu(n, batch) : eigvalsh_uses_gpu(n, batch);
    return gpu ? eigh_gpu_backend(n, batch) : cpu_side(n, vectors);
}

} // namespace

EighBackend eigh_backend(unsigned n, unsigned batch) { return route(n, batch, true); }

EighBackend eigvalsh_backend(unsigned n, unsigned batch) { return route(n, batch, false); }

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
