#include <metal_linalg/core.h>
#include "metal_runtime.h"
#include "shaders.h"

#import <Metal/Metal.h>

#include <algorithm>
#include <cmath>
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
using metal_linalg::detail::MetalRuntime;
using metal_linalg::detail::ScaledInput;
using metal_linalg::detail::copy_out;
using metal_linalg::detail::make_pipeline;
using metal_linalg::detail::pad_up;
using metal_linalg::detail::scaled_input;
using metal_linalg::detail::transpose_out;

namespace metal_linalg {
namespace {

// Must match the defines and `SvdBlockParams` in Svd_BlockJacobi.metal.
constexpr uint kB     = 16;
constexpr uint kSub   = 2 * kB;
constexpr uint kGroup = 32;
constexpr uint kRedFloats = 32;

struct BlockParams {
    uint  m;
    uint  n;
    uint  m_pad;
    uint  n_pad;
    uint  nb;
    uint  n_pairs;
    uint  round;
    uint  inner_sweeps;
    uint  rows_pad;
    float tol;
    float null_tol;
};

// The output sort stages min(M, N) singular values and ranks in threadgroup memory.
constexpr uint kBlockMaxK = 4096;
constexpr uint  kMinEffectiveRows = 64;   // as in svd.mm
constexpr float kNullEps = 32.0f;        // as in svd.mm

// Chunking cost model, core-milliseconds per block pair per round: a fixed
// subproblem cost, the Gram product and column update over the rows of G, and
// the column update over the rows of V. Conservative; it only bounds the work
// per command buffer.
constexpr double kSubCoreMs       = 0.3;
constexpr double kGramUpdMsPerRow = 4.1e-4;
constexpr double kVUpdMsPerRow    = 2.0e-4;
constexpr double kChunkBudgetMs   = 750.0;

constexpr uint kPairThreads   = 128;   // 4 simdgroups, one strip of the Gram matrix each
constexpr uint kUpdateThreads = 128;

unsigned env_uint(const char* name, unsigned fallback) {
    if (const char* s = std::getenv(name)) {
        const long v = std::strtol(s, nullptr, 10);
        if (v > 0) return (unsigned)v;
    }
    return fallback;
}

struct Pipelines {
    id<MTLComputePipelineState> pack, colmax, gram, cols, unpack;
};

struct Workspace {
    id<MTLBuffer> G, V, U, active, any, cmax, valid, S, Uo, Vt, flags;
};

struct Cache {
    MetalRuntime& rt = MetalRuntime::shared(METAL_LINALG_SHADER(Svd_BlockJacobi), "svd_block");

    std::map<bool, Pipelines>                               pipelines;
    std::map<std::tuple<uint, uint, uint, bool>, Workspace> workspaces;  // (batch, m, n, uv)

    Pipelines get_pipelines(bool uv) {
        if (auto it = pipelines.find(uv); it != pipelines.end()) return it->second;
        MTLFunctionConstantValues* cv = [[MTLFunctionConstantValues alloc] init];
        [cv setConstantValue:&uv type:MTLDataTypeBool atIndex:0];
        Pipelines p;
        p.pack   = make_pipeline(rt.device, rt.library, @"sbj_pack",            cv);
        p.colmax = make_pipeline(rt.device, rt.library, @"sbj_colmax",          nil);
        p.gram   = make_pipeline(rt.device, rt.library, @"sbj_gram_subproblem", nil);
        p.cols   = make_pipeline(rt.device, rt.library, @"sbj_update_cols",     nil);
        p.unpack = make_pipeline(rt.device, rt.library, @"sbj_unpack",          cv);
        return pipelines[uv] = p;
    }

    Workspace get_workspace(uint batch, uint m, uint n, uint m_pad, uint n_pad, uint n_pairs, bool uv) {
        const auto key = std::make_tuple(batch, m, n, uv);
        if (auto it = workspaces.find(key); it != workspaces.end()) return it->second;

        const MTLResourceOptions opt = MTLResourceStorageModeShared;
        auto buf = [&](size_t bytes) { return [rt.device newBufferWithLength:std::max<size_t>(bytes, 16) options:opt]; };
        const size_t f = sizeof(float), u = sizeof(uint);
        Workspace w;
        w.G      = buf((size_t)batch * m_pad * n_pad * f);
        w.V      = uv ? buf((size_t)batch * n_pad * n_pad * f) : nil;
        w.U      = buf((size_t)batch * n_pairs * kSub * kSub * f);
        w.active = buf((size_t)batch * n_pairs * u);
        w.any    = buf((size_t)batch * n_pairs * u);
        w.cmax   = buf((size_t)batch * f);
        w.valid  = buf((size_t)batch * u);
        w.S      = buf((size_t)batch * n * f);
        w.Uo     = uv ? buf((size_t)batch * m * n * f) : nil;
        w.Vt     = uv ? buf((size_t)batch * n * n * f) : nil;
        w.flags  = buf((size_t)batch * u);
        return workspaces[key] = w;
    }
};

} // namespace

namespace core::detail {

void svd_block_jacobi(const Matrices& a, const SvdOptions& opt,
                      float* u_out, float* s_out, float* vt_out, uint32_t* info_out) {
    const uint M = a.rows;
    const uint N = a.cols;
    const uint K = std::min(M, N);
    if (K > kBlockMaxK) {
        throw std::invalid_argument("[svd_block] min(M, N)=" + std::to_string(K) + " exceeds the " +
                                    std::to_string(kBlockMaxK) + " supported by the output sort.");
    }
    const uint batch = a.batch;
    const bool compute_uv = u_out || vt_out;
    if (K == 0 || batch == 0) {
        if (info_out) std::fill(info_out, info_out + batch, 0u);
        return;
    }
    AutoreleasePool pool;

    // The kernels want m >= n; a wide matrix is decomposed through its transpose.
    const bool wide = M < N;
    const uint m = wide ? N : M;
    const uint n = K;

    const uint m_pad   = pad_up(m, kGroup);
    const uint n_pad   = pad_up(n, kGroup);
    const uint nb      = n_pad / kB;      // even
    const uint n_pairs = nb / 2;
    const uint rounds  = nb - 1;

    // 1. GPU state.
    static Cache cache;
    id<MTLDevice> dev = cache.rt.device;

    // 2. Input, scaled by a power of two per matrix.
    ScaledInput in = scaled_input(dev, a, wide);
    Pipelines p  = cache.get_pipelines(compute_uv);
    Workspace ws = cache.get_workspace(batch, m, n, m_pad, n_pad, n_pairs, compute_uv);

    {
        uint* valid = static_cast<uint*>([ws.valid contents]);
        for (uint b = 0; b < batch; ++b) valid[b] = in.nonfinite[b] ? 0u : 1u;
    }

    id<MTLBuffer> buf_src = in.buffer;

    const float rows_eff = (float)std::max({m, opt.effective_rows, kMinEffectiveRows});
    const float eps = std::numeric_limits<float>::epsilon();

    BlockParams prm;
    prm.m = m; prm.n = n; prm.m_pad = m_pad; prm.n_pad = n_pad;
    prm.nb = nb; prm.n_pairs = n_pairs; prm.round = 0; prm.rows_pad = m_pad;
    prm.inner_sweeps = std::max(1u, opt.inner_sweeps ? opt.inner_sweeps
                                                     : env_uint("SVD_INNER_SWEEPS", 1));
    prm.tol      = opt.tol > 0.0f ? opt.tol : std::sqrt(rows_eff) * eps;
    prm.null_tol = kNullEps * eps;

    uint cores = gpu_core_count();
    if (cores == 0) cores = 8;
    const double per_round_core_ms =
        (double)n_pairs * (kSubCoreMs + kGramUpdMsPerRow * m_pad + kVUpdMsPerRow * n_pad);
    const double budget = (double)env_uint("SVD_CHUNK_MS", (unsigned)kChunkBudgetMs);
    const uint chunk = (uint)std::max(1.0, std::min((double)batch,
                                     std::floor(budget * cores / per_round_core_ms)));

    const size_t f = sizeof(float), u32 = sizeof(uint);
    const size_t in_bytes = (size_t)m * n * f;
    const size_t g_bytes  = (size_t)m_pad * n_pad * f;
    const size_t v_bytes  = (size_t)n_pad * n_pad * f;
    const size_t u_bytes  = (size_t)n_pairs * kSub * kSub * f;
    const size_t pr_bytes = (size_t)n_pairs * u32;
    const size_t unpack_tg = pad_up((uint)((kRedFloats + n) * f + n * sizeof(uint16_t)), 16);

    std::vector<uint> info(batch, 0);

    for (uint b0 = 0; b0 < batch; b0 += chunk) {
        const uint bc = std::min(chunk, batch - b0);
        const uint rounds_per_cmd = (uint)std::max(1.0, std::min((double)rounds,
            std::floor(budget * cores / (bc * per_round_core_ms))));

        auto run_cmd = [&](auto encode) {
            id<MTLCommandBuffer> cmd = [cache.rt.queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            encode(enc);
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
            if (cmd.error) {
                throw std::runtime_error(std::string("[svd_block] GPU kernel error: ") +
                                         cmd.error.localizedDescription.UTF8String);
            }
        };

        // --- pack ---
        run_cmd([&](id<MTLComputeCommandEncoder> enc) {
            [enc setComputePipelineState:p.pack];
            [enc setBuffer:buf_src  offset:(b0 * in_bytes) atIndex:0];
            [enc setBuffer:ws.G     offset:(b0 * g_bytes) atIndex:1];
            [enc setBuffer:(compute_uv ? ws.V : ws.G) offset:(compute_uv ? b0 * v_bytes : 0) atIndex:2];
            [enc setBuffer:ws.valid offset:((size_t)b0 * u32) atIndex:3];
            [enc setBytes:&prm length:sizeof(prm) atIndex:4];
            [enc dispatchThreads:MTLSizeMake(n_pad, std::max(m_pad, n_pad), bc)
                threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        });

        // --- sweeps: a sweep that settles every pair is the convergence test ---
        uint sweeps = 0;
        std::vector<char> conv(bc, 0);
        uint* any = static_cast<uint*>([ws.any contents]) + (size_t)b0 * n_pairs;
        while (sweeps < opt.max_sweeps) {
            std::memset(any, 0, (size_t)bc * pr_bytes);

            for (uint r0 = 0; r0 < rounds; r0 += rounds_per_cmd) {
                const uint r1 = std::min(rounds, r0 + rounds_per_cmd);
                run_cmd([&](id<MTLComputeCommandEncoder> enc) {
                    if (r0 == 0) {
                        // Largest column, for the null threshold of this sweep.
                        [enc setComputePipelineState:p.colmax];
                        [enc setBuffer:ws.G    offset:(b0 * g_bytes) atIndex:0];
                        [enc setBuffer:ws.cmax offset:((size_t)b0 * f) atIndex:1];
                        [enc setBytes:&prm length:sizeof(prm) atIndex:2];
                        [enc dispatchThreadgroups:MTLSizeMake(bc, 1, 1)
                            threadsPerThreadgroup:MTLSizeMake(std::min(1024u, pad_up(n, 32)), 1, 1)];
                    }
                    for (uint round = r0; round < r1; ++round) {
                        prm.round = round;

                        [enc setComputePipelineState:p.gram];
                        [enc setBuffer:ws.G      offset:(b0 * g_bytes) atIndex:0];
                        [enc setBuffer:ws.U      offset:(b0 * u_bytes) atIndex:1];
                        [enc setBuffer:ws.active offset:(b0 * pr_bytes) atIndex:2];
                        [enc setBuffer:ws.any    offset:(b0 * pr_bytes) atIndex:3];
                        [enc setBuffer:ws.cmax   offset:((size_t)b0 * f) atIndex:4];
                        [enc setBytes:&prm length:sizeof(prm) atIndex:5];
                        [enc dispatchThreadgroups:MTLSizeMake(n_pairs, bc, 1)
                            threadsPerThreadgroup:MTLSizeMake(kPairThreads, 1, 1)];

                        [enc setComputePipelineState:p.cols];
                        prm.rows_pad = m_pad;
                        [enc setBuffer:ws.G      offset:(b0 * g_bytes) atIndex:0];
                        [enc setBuffer:ws.U      offset:(b0 * u_bytes) atIndex:1];
                        [enc setBuffer:ws.active offset:(b0 * pr_bytes) atIndex:2];
                        [enc setBytes:&prm length:sizeof(prm) atIndex:3];
                        [enc dispatchThreadgroups:MTLSizeMake(n_pairs, m_pad / kGroup, bc)
                            threadsPerThreadgroup:MTLSizeMake(kUpdateThreads, 1, 1)];

                        if (compute_uv) {
                            prm.rows_pad = n_pad;
                            [enc setBuffer:ws.V offset:(b0 * v_bytes) atIndex:0];
                            [enc setBytes:&prm length:sizeof(prm) atIndex:3];
                            [enc dispatchThreadgroups:MTLSizeMake(n_pairs, n_pad / kGroup, bc)
                                threadsPerThreadgroup:MTLSizeMake(kUpdateThreads, 1, 1)];
                        }
                    }
                });
            }
            ++sweeps;

            bool all = true;
            for (uint b = 0; b < bc; ++b) {
                bool rotated = false;
                for (uint j = 0; j < n_pairs && !rotated; ++j) rotated = any[(size_t)b * n_pairs + j] != 0;
                conv[b] = !rotated;
                all = all && conv[b];
            }
            if (all) break;
        }

        // --- unpack ---
        run_cmd([&](id<MTLComputeCommandEncoder> enc) {
            [enc setComputePipelineState:p.unpack];
            [enc setBuffer:ws.G     offset:(b0 * g_bytes) atIndex:0];
            [enc setBuffer:(compute_uv ? ws.V  : ws.G) offset:(compute_uv ? b0 * v_bytes : 0) atIndex:1];
            [enc setBuffer:ws.S     offset:((size_t)b0 * n * f) atIndex:2];
            [enc setBuffer:(compute_uv ? ws.Uo : ws.S) offset:(compute_uv ? b0 * in_bytes : 0) atIndex:3];
            [enc setBuffer:(compute_uv ? ws.Vt : ws.S) offset:(compute_uv ? (size_t)b0 * n * n * f : 0) atIndex:4];
            [enc setBuffer:ws.flags offset:((size_t)b0 * u32) atIndex:5];
            [enc setBytes:&prm length:sizeof(prm) atIndex:6];
            [enc setThreadgroupMemoryLength:unpack_tg atIndex:0];
            [enc dispatchThreadgroups:MTLSizeMake(bc, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(std::min(1024u, pad_up(n, 32)), 1, 1)];
        });

        const uint* flags = static_cast<const uint*>([ws.flags contents]) + b0;
        for (uint b = 0; b < bc; ++b) {
            const uint gb = b0 + b;
            if (in.nonfinite[gb]) {
                info[gb] = sweeps | (1u << 17);
                continue;
            }
            if (!conv[b]) {
                throw std::runtime_error("[svd_block] Matrix " + std::to_string(gb) + " of " +
                                         std::to_string(batch) + " (" + std::to_string(M) + "x" +
                                         std::to_string(N) + ") did not converge in " +
                                         std::to_string(opt.max_sweeps) + " sweeps.");
            }
            info[gb] = sweeps | (1u << 16) | (flags[b] ? (1u << 18) : 0u);
        }
    }

    // 3. Non-finite matrices become NaN; rank-deficient ones get U completed.
    float* s_ptr = static_cast<float*>([ws.S contents]);
    float* u_ptr = compute_uv ? static_cast<float*>([ws.Uo contents]) : nullptr;
    float* v_ptr = compute_uv ? static_cast<float*>([ws.Vt contents]) : nullptr;
    for (uint b = 0; b < batch; ++b) {
        if (in.nonfinite[b]) {
            std::fill(s_ptr + (size_t)b * n, s_ptr + (size_t)(b + 1) * n, NAN);
            if (compute_uv) {
                std::fill(u_ptr + (size_t)b * m * n, u_ptr + (size_t)(b + 1) * m * n, NAN);
                std::fill(v_ptr + (size_t)b * n * n, v_ptr + (size_t)(b + 1) * n * n, NAN);
            }
        } else if (compute_uv && metal_linalg::detail::svd_rank_deficient(info[b])) {
            metal_linalg::detail::svd_complete_columns(u_ptr + (size_t)b * m * n, m, n);
        }
    }

    // 4. Copy out of the recycled workspace; the singular values are scaled back.
    copy_out(s_ptr, s_out, batch, n, in.scaled ? &in.unscale : nullptr);
    if (info_out) std::memcpy(info_out, info.data(), (size_t)batch * sizeof(uint32_t));
    if (!compute_uv) return;

    // Kernel factors: Uk [batch, m, n], Vtk [batch, n, n].
    if (!wide) {
        copy_out(u_ptr, u_out, batch, (size_t)m * n);
        copy_out(v_ptr, vt_out, batch, (size_t)n * n);
        return;
    }
    // A = V' S U'^T:  U = (Vtk)^T  [M, K],   Vt = (Uk)^T  [K, N].
    transpose_out(v_ptr, u_out, batch, n, n);
    transpose_out(u_ptr, vt_out, batch, m, n);
}

} // namespace core::detail
} // namespace metal_linalg
