#include <metal_linalg/eigh.h>
#include <metal_linalg/device.h>
#include "metal_runtime.h"
#include "shaders.h"

#import <Metal/Metal.h>

#include <mlx/mlx.h>
#include <mlx/linalg.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <map>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

using namespace mlx::core;
using metal_linalg::detail::MetalRuntime;
using metal_linalg::detail::make_pipeline;
using metal_linalg::detail::pad_up;
using metal_linalg::detail::prepare_input;

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

Shape batch_shape(const Shape& s) { return Shape(s.begin(), s.end() - 2); }

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
};

// The rows are generated from every run submitted for a device (docs/results/)
// by tuning/generate_tables.py, which a GitHub Action reruns after each
// merge; see docs/tuning.md. Why the measured values are what they are is in
// docs/studies/. The last row keeps the array non-empty and matches nothing.
constexpr TunedEntry kTuned[] = {
#include "tuned/eigh.inc"
    {"", 0,   0, 0, 0, 0,   0, 0, 0},
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

EighResult eigh_jacobi(const array& a, bool compute_vectors, bool lower, const EighOptions& opt) {
    if (a.ndim() < 2) {
        throw std::invalid_argument("[eigh] Input must be at least a 2D matrix.");
    }
    const Shape& shape = a.shape();
    const uint n = (uint)shape[shape.size() - 1];
    if ((uint)shape[shape.size() - 2] != n) {
        throw std::invalid_argument("[eigh] Input matrices must be square.");
    }
    if (n > 0xFFFFu) {
        throw std::invalid_argument("[eigh] N exceeds the 16-bit index used in threadgroup memory.");
    }

    uint batch = 1;
    for (size_t i = 0; i + 2 < shape.size(); ++i) batch *= (uint)shape[i];

    Shape vals_shape = batch_shape(shape);  vals_shape.push_back((int)n);
    Shape vecs_shape = shape;
    Shape info_shape = batch_shape(shape);

    if (n == 0 || batch == 0) {
        return {zeros(vals_shape, float32),
                zeros(compute_vectors ? vecs_shape : Shape{0}, float32),
                zeros(info_shape, mlx::core::uint32)};
    }

    // 1. Input: float32, row-contiguous, materialised, page-aligned.
    array a_f32 = prepare_input(a);

    // 2. Resolve the launch geometry.
    static Cache cache;
    id<MTLDevice> dev = cache.rt.device;

    const uint np = (n + 1) / 2;
    bool simd;
    switch (opt.mode) {
        case EighOptions::Mode::simd:        simd = true;  break;
        case EighOptions::Mode::threadgroup: simd = false; break;
        case EighOptions::Mode::block:       simd = false; break;   // not this backend's call
        default:                             simd = n <= eigh_simd_max_n(); break;
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

    id<MTLBuffer> buf_src = [dev newBufferWithBytesNoCopy:(void*)a_f32.data<float>()
                                                   length:a_f32.nbytes()
                                                  options:MTLResourceStorageModeShared
                                              deallocator:nil];
    if (!buf_src) {
        throw std::runtime_error("[eigh] Could not wrap the input array as a Metal buffer.");
    }

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
        if (!eigh_converged(w) && !eigh_nonfinite(w)) {
            throw std::runtime_error("[eigh] Matrix " + std::to_string(b) + " of " +
                                     std::to_string(batch) + " (N=" + std::to_string(n) +
                                     ") did not converge in " + std::to_string(opt.max_sweeps) +
                                     " sweeps.");
        }
    }

    // 6. Hand off. Constructing from a host pointer makes MLX deep-copy, which
    // keeps the recycled workspace safe from later use.
    array vals(static_cast<const float*>([ws.vals contents]), vals_shape, float32);
    array info(info_ptr, info_shape, mlx::core::uint32);
    array vecs = compute_vectors
        ? array(static_cast<const float*>([ws.W contents]), vecs_shape, float32)
        : zeros(Shape{0}, float32);

    return {vals, vecs, info};
}

} // namespace detail

namespace {

bool parse_uplo(const std::string& uplo, const char* who) {
    if (uplo == "L" || uplo == "l") return true;
    if (uplo == "U" || uplo == "u") return false;
    throw std::invalid_argument(std::string("[") + who + "] uplo must be \"L\" or \"U\".");
}

void problem_size(const array& a, unsigned& n, unsigned& batch) {
    const Shape& s = a.shape();
    n = a.ndim() >= 2 ? (unsigned)s[s.size() - 1] : 0;
    batch = 1;
    for (size_t i = 0; i + 2 < s.size(); ++i) batch *= (unsigned)s[i];
}

} // namespace

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

bool eigh_uses_gpu(unsigned n, unsigned batch) {
    if (const char* e = std::getenv("EIGH_DEVICE")) {
        const std::string s = e;
        if (s == "gpu") return true;
        if (s == "cpu") return false;
    }
    const EighPolicy& p = policy_state().policy;
    return n <= p.gpu_max_n && (unsigned long long)batch * n >= p.gpu_min_batch_times_n &&
           batch >= p.gpu_min_batch;
}

EighBackend eigh_backend(unsigned n, unsigned batch) {
    return eigh_uses_gpu(n, batch) ? eigh_gpu_backend(n, batch) : EighBackend::cpu;
}

namespace {

// The GPU backend split, as the policy has it.
EighResult gpu_eigh(const array& a, bool vectors, bool lower, unsigned n, unsigned batch) {
    EighOptions opt;
    switch (eigh_gpu_backend(n, batch)) {
        case EighBackend::block:
            opt.mode = EighOptions::Mode::block;
            return detail::eigh_block_jacobi(a, vectors, lower, opt);
        case EighBackend::simd:
            opt.mode = EighOptions::Mode::simd;
            break;
        default:
            opt.mode = EighOptions::Mode::threadgroup;
            break;
    }
    return detail::eigh_jacobi(a, vectors, lower, opt);
}

} // namespace

std::pair<array, array> eigh_accelerated(const array& a, const std::string& uplo) {
    const bool lower = parse_uplo(uplo, "eigh");
    unsigned n, batch;
    problem_size(a, n, batch);
    if (n > 0 && batch > 0 && !eigh_uses_gpu(n, batch)) {
        return linalg::eigh(astype(a, float32), lower ? "L" : "U", Device::cpu);
    }
    EighResult r = gpu_eigh(a, true, lower, n, batch);
    return {r.eigenvalues, r.eigenvectors};
}

array eigvalsh_accelerated(const array& a, const std::string& uplo) {
    const bool lower = parse_uplo(uplo, "eigvalsh");
    unsigned n, batch;
    problem_size(a, n, batch);
    if (n > 0 && batch > 0 && !eigh_uses_gpu(n, batch)) {
        return linalg::eigvalsh(astype(a, float32), lower ? "L" : "U", Device::cpu);
    }
    return gpu_eigh(a, false, lower, n, batch).eigenvalues;
}

} // namespace metal_linalg
