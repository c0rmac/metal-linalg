#include <metal_linalg/svd.h>
#include <metal_linalg/device.h>
#include <metal_linalg/qr.h>                 // qr_accelerated(), for the preconditioned backend
#include "metal_runtime.h"
#include "shaders.h"

#import <Metal/Metal.h>

#include <mlx/mlx.h>
#include <mlx/linalg.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
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

Shape batch_shape(const Shape& s) { return Shape(s.begin(), s.end() - 2); }

array transpose_last_two(const array& x) {
    std::vector<int> axes(x.ndim());
    for (size_t i = 0; i < axes.size(); ++i) axes[i] = (int)i;
    std::swap(axes[axes.size() - 1], axes[axes.size() - 2]);
    return transpose(x, axes);
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
};

// The SvdPolicy defaults come from measurements on an M1 taken while the
// machine was heavily loaded by other jobs, which is good enough to place the
// crossovers roughly and not good enough for a table entry; see
// docs/studies/svd-design-notes.md. The M1 itself therefore has no row.
//
// Apple M5 Pro: tuning/tune_svd.py over 263 (shape, batch) points up to
// k = 1024, five backends, two randomised passes, idle machine on mains
// (docs/results/svd-apple-m5-pro/). Against the best GPU backend the
// QR-preconditioned path pays from 512 rows and k = 32, the block kernel
// from k = 192, and from k = 64 in batches of 64 or more; that batch term was
// better on held-out data in 100% of bootstrap resamples. Against the CPU the
// GPU is ahead up to the largest k measured (1024 is a lower bound), never
// for fewer than four matrices, and only when batch * k >= 512; the
// flat region is gpu_max_k 1024 or none, gpu_min_batch_times_k 512 alone,
// gpu_min_batch 4 alone.
constexpr TunedEntry kTuned[] = {
    {"Apple M5 Pro", 20,   512, 32,   192, 64, 64,   1024, 512, 4},
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
            r.source = "tuned:" + r.device;
            break;
        }
    }
    if (r.source.empty()) {
        r.source = "default:untuned-device" + (r.device.empty() ? "" : " (" + r.device + ")");
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

void problem_size(const array& a, unsigned& m, unsigned& n, unsigned& batch) {
    const Shape& s = a.shape();
    m = a.ndim() >= 2 ? (unsigned)s[s.size() - 2] : 0;
    n = a.ndim() >= 2 ? (unsigned)s[s.size() - 1] : 0;
    batch = 1;
    for (size_t i = 0; i + 2 < s.size(); ++i) batch *= (unsigned)s[i];
}

array converged_info(const Shape& shape) {
    return full(Shape(shape.begin(), shape.end() - 2), (uint32_t)(1u | (1u << 16)));
}

} // namespace

namespace detail {

void svd_complete_columns(float* u, unsigned m, unsigned n) { complete_columns_impl(u, m, n); }

SvdResult svd_jacobi(const array& a_in, bool compute_uv, const SvdOptions& opt) {
    if (a_in.ndim() < 2) {
        throw std::invalid_argument("[svd] Input must be at least a 2D matrix.");
    }
    const Shape& shape = a_in.shape();
    const uint M = (uint)shape[shape.size() - 2];
    const uint N = (uint)shape[shape.size() - 1];
    const uint K = std::min(M, N);

    uint batch = 1;
    for (size_t i = 0; i + 2 < shape.size(); ++i) batch *= (uint)shape[i];

    Shape u_shape = shape;                  u_shape[u_shape.size() - 1] = (int)K;
    Shape s_shape = batch_shape(shape);     s_shape.push_back((int)K);
    Shape vt_shape = shape;                 vt_shape[vt_shape.size() - 2] = (int)K;
    Shape info_shape = batch_shape(shape);

    if (K == 0 || batch == 0) {
        return {zeros(compute_uv ? u_shape : Shape{0}, float32), zeros(s_shape, float32),
                zeros(compute_uv ? vt_shape : Shape{0}, float32),
                zeros(info_shape, mlx::core::uint32)};
    }

    // The kernel wants m >= n. For a wide matrix decompose the transpose,
    // A^T = U' S V'^T, and read off A = V' S U'^T.
    const bool wide = M < N;
    const uint m = wide ? N : M;
    const uint n = K;
    if (n > 0xFFFFu) {
        throw std::invalid_argument("[svd] min(M, N) exceeds the 16-bit index used in threadgroup memory.");
    }

    array a_f32 = prepare_input(wide ? transpose_last_two(a_in) : a_in);

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

    id<MTLBuffer> buf_src = [dev newBufferWithBytesNoCopy:(void*)a_f32.data<float>()
                                                   length:a_f32.nbytes()
                                                  options:MTLResourceStorageModeShared
                                              deallocator:nil];
    if (!buf_src) {
        throw std::runtime_error("[svd] Could not wrap the input array as a Metal buffer.");
    }

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
        if (!svd_converged(w) && !svd_nonfinite(w)) {
            throw std::runtime_error("[svd] Matrix " + std::to_string(b) + " of " +
                                     std::to_string(batch) + " (" + std::to_string(M) + "x" +
                                     std::to_string(N) + ") did not converge in " +
                                     std::to_string(opt.max_sweeps) + " sweeps.");
        }
    }

    array S(static_cast<const float*>([ws.S contents]), s_shape, float32);
    array info(info_ptr, info_shape, mlx::core::uint32);
    if (!compute_uv) {
        return {zeros(Shape{0}, float32), S, zeros(Shape{0}, float32), info};
    }

    // Rank-deficient matrices: give U an orthonormal completion.
    float* u_ptr = static_cast<float*>([ws.U contents]);
    for (uint b = 0; b < batch; ++b) {
        if (svd_rank_deficient(info_ptr[b])) svd_complete_columns(u_ptr + (size_t)b * m * n, m, n);
    }

    // Kernel factors: Uk [batch, m, n], Vtk [batch, n, n].
    Shape uk_shape = batch_shape(shape);  uk_shape.push_back((int)m);  uk_shape.push_back((int)n);
    Shape vk_shape = batch_shape(shape);  vk_shape.push_back((int)n);  vk_shape.push_back((int)n);
    array Uk(static_cast<const float*>(u_ptr), uk_shape, float32);
    array Vtk(static_cast<const float*>([ws.Vt contents]), vk_shape, float32);

    if (!wide) return {Uk, S, Vtk, info};
    // A = V' S U'^T:  U = V' = (Vtk)^T  [M, K],   Vt = U'^T = (Uk)^T  [K, N].
    return {contiguous(transpose_last_two(Vtk)), S, contiguous(transpose_last_two(Uk)), info};
}

SvdResult svd_qr_jacobi(const array& a_in, bool compute_uv, const SvdOptions& opt) {
    if (a_in.ndim() < 2) {
        throw std::invalid_argument("[svd] Input must be at least a 2D matrix.");
    }
    const Shape& shape = a_in.shape();
    const bool wide = shape[shape.size() - 2] < shape[shape.size() - 1];
    if (wide) {
        // A^T = U' S V'^T  =>  A = V' S U'^T.
        SvdResult t = svd_qr_jacobi(transpose_last_two(a_in), compute_uv, opt);
        if (!compute_uv) return t;
        return {contiguous(transpose_last_two(t.Vt)), t.S, contiguous(transpose_last_two(t.U)), t.info};
    }

    // A = Q R, R = U_R S V^T  =>  A = (Q U_R) S V^T. The tolerances are those
    // of A, not of the small factor.
    auto [Q, R] = qr_accelerated(a_in);
    SvdOptions inner = opt;
    inner.effective_rows = std::max<unsigned>(opt.effective_rows, (unsigned)shape[shape.size() - 2]);
    const unsigned k = (unsigned)shape[shape.size() - 1];
    unsigned batch = 1;
    for (size_t i = 0; i + 2 < shape.size(); ++i) batch *= (unsigned)shape[i];
    const bool block = opt.kernel == SvdOptions::Kernel::block ||
                       (opt.kernel == SvdOptions::Kernel::automatic && wants_block(k, batch));
    SvdResult r = block ? svd_block_jacobi(R, compute_uv, inner) : svd_jacobi(R, compute_uv, inner);
    if (!compute_uv) return r;
    return {matmul(Q, r.U), r.S, r.Vt, r.info};
}

SvdResult svd_cpu(const array& a_in, bool compute_uv) {
    if (a_in.ndim() < 2) {
        throw std::invalid_argument("[svd] Input must be at least a 2D matrix.");
    }
    const Shape& shape = a_in.shape();
    const int M = shape[shape.size() - 2], N = shape[shape.size() - 1];
    const Device cpu = Device::cpu;

    if (M < N) {
        SvdResult t = svd_cpu(transpose(a_in, [&] {
            std::vector<int> ax(a_in.ndim());
            for (size_t i = 0; i < ax.size(); ++i) ax[i] = (int)i;
            std::swap(ax[ax.size() - 1], ax[ax.size() - 2]);
            return ax; }(), cpu), compute_uv);
        if (!compute_uv) return t;
        return {transpose_last_two(t.Vt), t.S, transpose_last_two(t.U), t.info};
    }

    array a = astype(a_in, float32, cpu);
    array info = converged_info(shape);
    if (M == 0 || N == 0) {
        Shape s_shape = batch_shape(shape);  s_shape.push_back(0);
        return {zeros(compute_uv ? shape : Shape{0}, float32), zeros(s_shape, float32),
                zeros(compute_uv ? shape : Shape{0}, float32), info};
    }

    // MLX only offers the full-size factors, whose U is M x M. For a tall
    // matrix that is most of the cost and none of what was asked for, so
    // reduce with a thin QR first, as LAPACK does for its own thin SVD.
    if (M >= 2 * N) {
        auto [Q, R] = linalg::qr(a, cpu);
        std::vector<array> f = linalg::svd(R, compute_uv, cpu);
        if (!compute_uv) return {zeros(Shape{0}, float32), f.back(), zeros(Shape{0}, float32), info};
        return {matmul(Q, f[0], cpu), f[1], f[2], info};
    }

    std::vector<array> f = linalg::svd(a, compute_uv, cpu);
    if (!compute_uv) return {zeros(Shape{0}, float32), f.back(), zeros(Shape{0}, float32), info};
    array U = f[0];
    if (M > N) {   // keep the leading N columns
        Shape start(U.ndim(), 0), stop = U.shape();
        stop.back() = N;
        U = slice(U, start, stop, cpu);
    }
    return {U, f[1], f[2], info};
}

} // namespace detail

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

namespace {

SvdResult routed(const array& a, bool compute_uv) {
    unsigned m, n, batch;
    problem_size(a, m, n, batch);
    if (a.ndim() < 2 || m == 0 || n == 0 || batch == 0) {
        return detail::svd_jacobi(a, compute_uv, SvdOptions{});   // validates, handles empties
    }
    SvdOptions opt;
    switch (svd_backend(m, n, batch)) {
        case SvdBackend::cpu:             return detail::svd_cpu(a, compute_uv);
        case SvdBackend::block_jacobi:    return detail::svd_block_jacobi(a, compute_uv, opt);
        case SvdBackend::qr_jacobi:       opt.kernel = SvdOptions::Kernel::jacobi;
                                          return detail::svd_qr_jacobi(a, compute_uv, opt);
        case SvdBackend::qr_block_jacobi: opt.kernel = SvdOptions::Kernel::block;
                                          return detail::svd_qr_jacobi(a, compute_uv, opt);
        default:                          return detail::svd_jacobi(a, compute_uv, opt);
    }
}

} // namespace

std::tuple<array, array, array> svd_accelerated(const array& a) {
    SvdResult r = routed(a, true);
    return {r.U, r.S, r.Vt};
}

array svdvals_accelerated(const array& a) {
    return routed(a, false).S;
}

} // namespace metal_linalg
