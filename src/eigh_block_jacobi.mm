#include <metal_linalg/eigh.h>
#include <metal_linalg/device.h>
#include "metal_runtime.h"
#include "shaders.h"

#import <Metal/Metal.h>

#include <mlx/mlx.h>

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

// Must match the defines and `BlockParams` in Eigh_BlockJacobi.metal.
constexpr uint kB     = 16;
constexpr uint kSub   = 2 * kB;
constexpr uint kGroup = 32;

struct BlockParams {
    uint n;
    uint n_pad;
    uint nb;
    uint n_pairs;
    uint round;
    uint inner_sweeps;
    uint lower;
};

// The output sort stages N eigenvalues and N ranks in threadgroup memory.
constexpr uint kBlockMaxN = 4096;

// Chunking cost model, core-milliseconds per block pair per round: a fixed
// subproblem cost plus the three 32 x 32 x N_pad products. Conservative on
// purpose; it only bounds the work per command buffer (see eigh.mm for why).
constexpr double kSubCoreMs        = 0.3;
constexpr double kUpdCoreMsPerCol  = 8e-4;
constexpr double kChunkBudgetMs    = 750.0;

constexpr uint kSubThreads    = 128;   // 4 simdgroups; 512 items per phase
constexpr uint kUpdateThreads = 128;   // 4 simdgroups, one 8-strip each
constexpr uint kNormThreads   = 256;

unsigned env_uint(const char* name, unsigned fallback) {
    if (const char* s = std::getenv(name)) {
        const long v = std::strtol(s, nullptr, 10);
        if (v > 0) return (unsigned)v;
    }
    return fallback;
}

struct Pipelines {
    id<MTLComputePipelineState> pack, subproblem, rows, cols, norm, unpack;
};

struct Workspace {
    id<MTLBuffer> W, V, U, active, thr, norm, scale, expo, vals, vecs;
};

struct Cache {
    MetalRuntime& rt = MetalRuntime::shared(METAL_LINALG_SHADER(Eigh_BlockJacobi), "eigh_block");

    std::map<bool, Pipelines>                          pipelines;   // by compute_vectors
    std::map<std::tuple<uint, uint, bool>, Workspace>  workspaces;  // (batch, n, vectors)

    Pipelines get_pipelines(bool vectors) {
        if (auto it = pipelines.find(vectors); it != pipelines.end()) return it->second;

        MTLFunctionConstantValues* cv = [[MTLFunctionConstantValues alloc] init];
        [cv setConstantValue:&vectors type:MTLDataTypeBool atIndex:0];

        Pipelines p;
        p.pack       = make_pipeline(rt.device, rt.library, @"bj_pack",        cv);
        p.subproblem = make_pipeline(rt.device, rt.library, @"bj_subproblem",  nil);
        p.rows       = make_pipeline(rt.device, rt.library, @"bj_update_rows", nil);
        p.cols       = make_pipeline(rt.device, rt.library, @"bj_update_cols", cv);
        p.norm       = make_pipeline(rt.device, rt.library, @"bj_norm",        nil);
        p.unpack     = make_pipeline(rt.device, rt.library, @"bj_unpack",      cv);
        return pipelines[vectors] = p;
    }

    Workspace get_workspace(uint batch, uint n, uint n_pad, uint n_pairs, bool vectors) {
        const auto key = std::make_tuple(batch, n, vectors);
        if (auto it = workspaces.find(key); it != workspaces.end()) return it->second;

        const MTLResourceOptions opt = MTLResourceStorageModeShared;
        auto buf = [&](size_t bytes) { return [rt.device newBufferWithLength:std::max<size_t>(bytes, 16) options:opt]; };

        const size_t pad_bytes = (size_t)batch * n_pad * n_pad * sizeof(float);
        Workspace w;
        w.W      = buf(pad_bytes);
        w.V      = vectors ? buf(pad_bytes) : nil;
        w.U      = buf((size_t)batch * n_pairs * kSub * kSub * sizeof(float));
        w.active = buf((size_t)batch * n_pairs * sizeof(uint));
        w.thr    = buf((size_t)batch * sizeof(float));
        w.norm   = buf((size_t)batch * sizeof(float));
        w.scale  = buf((size_t)batch * sizeof(float));
        w.expo   = buf((size_t)batch * sizeof(int));
        w.vals   = buf((size_t)batch * n * sizeof(float));
        w.vecs   = vectors ? buf((size_t)batch * n * n * sizeof(float)) : nil;
        return workspaces[key] = w;
    }
};

Shape batch_shape(const Shape& s) { return Shape(s.begin(), s.end() - 2); }

} // namespace

namespace detail {

EighResult eigh_block_jacobi(const array& a, bool compute_vectors, bool lower, const EighOptions& opt) {
    if (a.ndim() < 2) {
        throw std::invalid_argument("[eigh_block] Input must be at least a 2D matrix.");
    }
    const Shape& shape = a.shape();
    const uint n = (uint)shape[shape.size() - 1];
    if ((uint)shape[shape.size() - 2] != n) {
        throw std::invalid_argument("[eigh_block] Input matrices must be square.");
    }
    if (n > kBlockMaxN) {
        throw std::invalid_argument("[eigh_block] N=" + std::to_string(n) + " exceeds the " +
                                    std::to_string(kBlockMaxN) + " supported by the output sort.");
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

    const uint n_pad   = pad_up(n, kGroup);
    const uint nb      = n_pad / kB;          // even, since n_pad is a multiple of 2b
    const uint n_pairs = nb / 2;
    const uint rounds  = nb - 1;

    // 1. Input, and per-matrix scale / finiteness from the triangle in use.
    //    Junk in the unused triangle must not set the scale.
    array a_f32 = prepare_input(a);
    std::vector<float> scale(batch, 1.0f);
    std::vector<int>   expo(batch, 0);
    std::vector<char>  nonfinite(batch, 0);
    {
        array tri  = lower ? tril(a_f32, 0) : triu(a_f32, 0);
        array amax = reshape(max(abs(tri), std::vector<int>{-2, -1}), {-1});
        array bad  = reshape(any(logical_or(isnan(tri), isinf(tri)), std::vector<int>{-2, -1}), {-1});
        eval({amax, bad});
        for (uint b = 0; b < batch; ++b) {
            nonfinite[b] = bad.data<bool>()[b] ? 1 : 0;
            const float m = amax.data<float>()[b];
            if (nonfinite[b] || !(m > 0.0f)) {
                scale[b] = 0.0f;   // a zero matrix converges in zero sweeps
                expo[b]  = 0;
            } else {
                int e = 0;
                std::frexp(m, &e);
                expo[b]  = e;
                scale[b] = std::ldexp(1.0f, -e);
            }
        }
    }

    // 2. GPU state.
    static Cache cache;
    id<MTLDevice> dev = cache.rt.device;
    Pipelines p  = cache.get_pipelines(compute_vectors);
    Workspace ws = cache.get_workspace(batch, n, n_pad, n_pairs, compute_vectors);

    std::copy(scale.begin(), scale.end(), static_cast<float*>([ws.scale contents]));
    std::copy(expo.begin(),  expo.end(),  static_cast<int*>([ws.expo contents]));

    id<MTLBuffer> buf_src = [dev newBufferWithBytesNoCopy:(void*)a_f32.data<float>()
                                                   length:a_f32.nbytes()
                                                  options:MTLResourceStorageModeShared
                                              deallocator:nil];
    if (!buf_src) {
        throw std::runtime_error("[eigh_block] Could not wrap the input array as a Metal buffer.");
    }

    const uint inner_sweeps = std::max(1u, opt.inner_sweeps ? opt.inner_sweeps
                                                            : env_uint("EIGH_INNER_SWEEPS", 1));

    uint cores = gpu_core_count();
    if (cores == 0) cores = 8;

    // Chunking: matrices per solve and rounds per command buffer.
    const double per_round_core_ms = (double)n_pairs * (kSubCoreMs + kUpdCoreMsPerCol * n_pad);
    const double budget = (double)env_uint("EIGH_CHUNK_MS", (unsigned)kChunkBudgetMs);
    const uint chunk = (uint)std::max(1.0, std::min((double)batch,
                                     std::floor(budget * cores / per_round_core_ms)));

    const size_t in_bytes  = (size_t)n * n * sizeof(float);
    const size_t pad_bytes = (size_t)n_pad * n_pad * sizeof(float);
    const size_t u_bytes   = (size_t)n_pairs * kSub * kSub * sizeof(float);
    const size_t unpack_tg = (size_t)n * (sizeof(float) + sizeof(uint16_t));

    std::vector<uint> info(batch, 0);

    for (uint b0 = 0; b0 < batch; b0 += chunk) {
        const uint bc = std::min(chunk, batch - b0);
        const uint rounds_per_cmd = (uint)std::max(1.0, std::min((double)rounds,
            std::floor(budget * cores / (bc * per_round_core_ms))));

        BlockParams prm;
        prm.n = n; prm.n_pad = n_pad; prm.nb = nb; prm.n_pairs = n_pairs;
        prm.round = 0; prm.inner_sweeps = inner_sweeps; prm.lower = lower ? 1u : 0u;

        auto run_cmd = [&](auto encode) {
            id<MTLCommandBuffer> cmd = [cache.rt.queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            encode(enc);
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
            if (cmd.error) {
                throw std::runtime_error(std::string("[eigh_block] GPU kernel error: ") +
                                         cmd.error.localizedDescription.UTF8String);
            }
        };

        auto encode_norm = [&](id<MTLComputeCommandEncoder> enc, uint include_diag) {
            [enc setComputePipelineState:p.norm];
            [enc setBuffer:ws.W    offset:(b0 * pad_bytes) atIndex:0];
            [enc setBuffer:ws.norm offset:((size_t)b0 * sizeof(float)) atIndex:1];
            [enc setBytes:&prm length:sizeof(prm) atIndex:2];
            [enc setBytes:&include_diag length:sizeof(uint) atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake(bc, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(kNormThreads, 1, 1)];
        };

        // --- pack, then ||A||_F^2 per matrix ---
        run_cmd([&](id<MTLComputeCommandEncoder> enc) {
            [enc setComputePipelineState:p.pack];
            [enc setBuffer:buf_src  offset:(b0 * in_bytes) atIndex:0];
            [enc setBuffer:ws.W     offset:(b0 * pad_bytes) atIndex:1];
            [enc setBuffer:(compute_vectors ? ws.V : ws.W) offset:(b0 * pad_bytes) atIndex:2];
            [enc setBuffer:ws.scale offset:((size_t)b0 * sizeof(float)) atIndex:3];
            [enc setBytes:&prm length:sizeof(prm) atIndex:4];
            [enc dispatchThreads:MTLSizeMake(n_pad, n_pad, bc)
                threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
            encode_norm(enc, 1u);
        });

        std::vector<float> thr2(bc);
        {
            const float* fro2 = static_cast<const float*>([ws.norm contents]) + b0;
            float* sub_thr = static_cast<float*>([ws.thr contents]) + b0;
            for (uint b = 0; b < bc; ++b) {
                thr2[b] = opt.tol * opt.tol * fro2[b];
                // A pair whose off-norm is under its share of the budget
                // cannot be what keeps the matrix from converging.
                sub_thr[b] = thr2[b] / (float)(n_pairs * rounds);
            }
        }

        // --- sweeps ---
        uint sweeps = 0;
        std::vector<char> conv(bc, 0);
        while (true) {
            run_cmd([&](id<MTLComputeCommandEncoder> enc) { encode_norm(enc, 0u); });
            const float* off2 = static_cast<const float*>([ws.norm contents]) + b0;
            bool all = true;
            for (uint b = 0; b < bc; ++b) {
                conv[b] = off2[b] <= thr2[b];
                all = all && conv[b];
            }
            if (all || sweeps >= opt.max_sweeps) break;

            for (uint r0 = 0; r0 < rounds; r0 += rounds_per_cmd) {
                const uint r1 = std::min(rounds, r0 + rounds_per_cmd);
                run_cmd([&](id<MTLComputeCommandEncoder> enc) {
                    for (uint round = r0; round < r1; ++round) {
                        prm.round = round;

                        [enc setComputePipelineState:p.subproblem];
                        [enc setBuffer:ws.W      offset:(b0 * pad_bytes) atIndex:0];
                        [enc setBuffer:ws.U      offset:(b0 * u_bytes) atIndex:1];
                        [enc setBuffer:ws.active offset:((size_t)b0 * n_pairs * sizeof(uint)) atIndex:2];
                        [enc setBuffer:ws.thr    offset:((size_t)b0 * sizeof(float)) atIndex:3];
                        [enc setBytes:&prm length:sizeof(prm) atIndex:4];
                        [enc dispatchThreadgroups:MTLSizeMake(n_pairs, bc, 1)
                            threadsPerThreadgroup:MTLSizeMake(kSubThreads, 1, 1)];

                        [enc setComputePipelineState:p.rows];
                        [enc setBuffer:ws.W      offset:(b0 * pad_bytes) atIndex:0];
                        [enc setBuffer:ws.U      offset:(b0 * u_bytes) atIndex:1];
                        [enc setBuffer:ws.active offset:((size_t)b0 * n_pairs * sizeof(uint)) atIndex:2];
                        [enc setBytes:&prm length:sizeof(prm) atIndex:3];
                        [enc dispatchThreadgroups:MTLSizeMake(n_pairs, n_pad / kGroup, bc)
                            threadsPerThreadgroup:MTLSizeMake(kUpdateThreads, 1, 1)];

                        [enc setComputePipelineState:p.cols];
                        [enc setBuffer:ws.W      offset:(b0 * pad_bytes) atIndex:0];
                        [enc setBuffer:(compute_vectors ? ws.V : ws.W) offset:(b0 * pad_bytes) atIndex:1];
                        [enc setBuffer:ws.U      offset:(b0 * u_bytes) atIndex:2];
                        [enc setBuffer:ws.active offset:((size_t)b0 * n_pairs * sizeof(uint)) atIndex:3];
                        [enc setBytes:&prm length:sizeof(prm) atIndex:4];
                        [enc dispatchThreadgroups:MTLSizeMake(n_pairs, n_pad / kGroup, bc)
                            threadsPerThreadgroup:MTLSizeMake(kUpdateThreads, 1, 1)];
                    }
                });
            }
            ++sweeps;
        }

        // --- unpack ---
        run_cmd([&](id<MTLComputeCommandEncoder> enc) {
            [enc setComputePipelineState:p.unpack];
            [enc setBuffer:ws.W    offset:(b0 * pad_bytes) atIndex:0];
            [enc setBuffer:(compute_vectors ? ws.V : ws.W) offset:(b0 * pad_bytes) atIndex:1];
            [enc setBuffer:ws.vals offset:((size_t)b0 * n * sizeof(float)) atIndex:2];
            [enc setBuffer:(compute_vectors ? ws.vecs : ws.vals) offset:(compute_vectors ? b0 * in_bytes : 0) atIndex:3];
            [enc setBuffer:ws.expo offset:((size_t)b0 * sizeof(int)) atIndex:4];
            [enc setBytes:&prm length:sizeof(prm) atIndex:5];
            [enc setThreadgroupMemoryLength:pad_up((uint)unpack_tg, 16) atIndex:0];
            [enc dispatchThreadgroups:MTLSizeMake(bc, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(std::min(1024u, pad_up(n, 32)), 1, 1)];
        });

        for (uint b = 0; b < bc; ++b) {
            const uint gb = b0 + b;
            if (nonfinite[gb]) {
                info[gb] = sweeps | (1u << 17);
            } else {
                if (!conv[b]) {
                    throw std::runtime_error("[eigh_block] Matrix " + std::to_string(gb) + " of " +
                                             std::to_string(batch) + " (N=" + std::to_string(n) +
                                             ") did not converge in " + std::to_string(opt.max_sweeps) +
                                             " sweeps.");
                }
                info[gb] = sweeps | (1u << 16);
            }
        }
    }

    // 3. Non-finite matrices: NaN everywhere, as the scalar kernel does.
    {
        float* vals_p = static_cast<float*>([ws.vals contents]);
        float* vecs_p = compute_vectors ? static_cast<float*>([ws.vecs contents]) : nullptr;
        for (uint b = 0; b < batch; ++b) {
            if (!nonfinite[b]) continue;
            std::fill(vals_p + (size_t)b * n, vals_p + (size_t)(b + 1) * n, NAN);
            if (vecs_p) std::fill(vecs_p + (size_t)b * n * n, vecs_p + (size_t)(b + 1) * n * n, NAN);
        }
    }

    // 4. Hand off (deep copies, so the recycled workspace stays private).
    array vals(static_cast<const float*>([ws.vals contents]), vals_shape, float32);
    array info_arr(info.data(), info_shape, mlx::core::uint32);
    array vecs = compute_vectors
        ? array(static_cast<const float*>([ws.vecs contents]), vecs_shape, float32)
        : zeros(Shape{0}, float32);
    return {vals, vecs, info_arr};
}

} // namespace detail
} // namespace metal_linalg
