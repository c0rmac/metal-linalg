#include <metal_linalg/qr.h>
#include "metal_runtime.h"
#include "shaders.h"

#include <mlx/mlx.h>

#include <algorithm>
#include <map>
#include <stdexcept>
#include <tuple>
#include <utility>

using namespace mlx::core;

namespace metal_linalg::detail {
namespace {

struct Pipelines {
    id<MTLComputePipelineState> pso_transpose;
    id<MTLComputePipelineState> pso_init_q;
    id<MTLComputePipelineState> pso_panel;
    id<MTLComputePipelineState> pso_t_mat;
    id<MTLComputePipelineState> pso_update;
    id<MTLComputePipelineState> pso_haar;
    id<MTLComputePipelineState> pso_unpk_R;
    id<MTLComputePipelineState> pso_unpk_Q;
};

struct Workspace {
    id<MTLBuffer> buf_A;
    id<MTLBuffer> buf_Q;
    id<MTLBuffer> buf_R_diag;
    id<MTLBuffer> buf_tau;
    id<MTLBuffer> buf_T;
    id<MTLBuffer> buf_R_out;
    id<MTLBuffer> buf_Q_out;
};

// Process-lifetime cache of everything that only depends on the input shape.
struct Cache {
    MetalRuntime& rt =
        MetalRuntime::shared(METAL_LINALG_SHADER(QR_Streaming_AMX_Complete), "qr_streaming_amx_complete");

    std::map<std::pair<uint, uint>, Pipelines>        pipelines;   // keyed by (M_pad, N_pad)
    std::map<std::tuple<uint, uint, uint>, Workspace> workspaces;  // keyed by (batch, M_pad, N_pad)

    Pipelines get_pipelines(uint M_pad, uint N_pad) {
        auto key = std::make_pair(M_pad, N_pad);
        if (auto it = pipelines.find(key); it != pipelines.end()) {
            return it->second;
        }

        MTLFunctionConstantValues* cv = [[MTLFunctionConstantValues alloc] init];
        [cv setConstantValue:&M_pad type:MTLDataTypeUInt atIndex:0];
        [cv setConstantValue:&N_pad type:MTLDataTypeUInt atIndex:1];

        Pipelines p;
        p.pso_transpose = make_pipeline(rt.device, rt.library, @"preprocess_transpose",   cv);
        p.pso_init_q    = make_pipeline(rt.device, rt.library, @"init_identity_q",        cv);
        p.pso_panel     = make_pipeline(rt.device, rt.library, @"panel_factorization",    cv);
        p.pso_t_mat     = make_pipeline(rt.device, rt.library, @"compute_t_matrix",       cv);
        p.pso_update    = make_pipeline(rt.device, rt.library, @"grid_parallel_update",   cv);
        p.pso_haar      = make_pipeline(rt.device, rt.library, @"postprocess_haar_fix",   cv);
        p.pso_unpk_R    = make_pipeline(rt.device, rt.library, @"unpack_batch_R",        nil);
        p.pso_unpk_Q    = make_pipeline(rt.device, rt.library, @"unpack_batch_Q",        nil);

        return pipelines[key] = p;
    }

    Workspace get_workspace(uint batch, uint M_pad, uint N_pad, uint M, uint N, uint K) {
        auto key = std::make_tuple(batch, M_pad, N_pad);
        if (auto it = workspaces.find(key); it != workspaces.end()) {
            return it->second;
        }

        const MTLResourceOptions opt = MTLResourceStorageModeShared;

        Workspace w;
        w.buf_A      = [rt.device newBufferWithLength:((size_t)batch * M_pad * N_pad * sizeof(float)) options:opt];

        // Unlike the reduced backend, Q here is the full M x M orthogonal factor.
        w.buf_Q      = [rt.device newBufferWithLength:((size_t)batch * M_pad * M_pad * sizeof(float)) options:opt];
        w.buf_R_diag = [rt.device newBufferWithLength:((size_t)batch * M_pad * sizeof(float))         options:opt];
        w.buf_tau    = [rt.device newBufferWithLength:((size_t)batch * N_pad * sizeof(float))         options:opt];

        // Q is accumulated inside the forward loop, so only the current block's
        // T-matrix is ever live.
        w.buf_T      = [rt.device newBufferWithLength:((size_t)batch * 32 * 32 * sizeof(float))       options:opt];

        // Exact-size output buffers.
        w.buf_R_out  = [rt.device newBufferWithLength:((size_t)batch * K * N * sizeof(float)) options:opt];
        w.buf_Q_out  = [rt.device newBufferWithLength:((size_t)batch * M * K * sizeof(float)) options:opt];

        return workspaces[key] = w;
    }
};

} // namespace

// =============================================================================
// MAIN ENTRY POINT
// =============================================================================

std::pair<array, array> qr_streaming_amx_complete(const array& a) {
    if (a.ndim() < 2)
        throw std::invalid_argument("[qr_streaming_amx_complete] Input must be at least a 2D matrix.");

    const Shape& shape       = a.shape();
    const uint   original_M  = static_cast<uint>(shape[shape.size() - 2]);
    const uint   original_N  = static_cast<uint>(shape[shape.size() - 1]);
    const uint   original_K  = std::min(original_M, original_N);

    uint batch = 1;
    for (size_t i = 0; i + 2 < shape.size(); ++i)
        batch *= static_cast<uint>(shape[i]);

    // Scaled by a power of two per matrix; see prepare_input_scaled.
    ScaledInput in = prepare_input_scaled(a);
    array a_f32 = in.a;

    const uint M_pad = pad_up(original_M, 32);
    const uint N_pad = pad_up(original_N, 32);

    static Cache cache;
    Pipelines p = cache.get_pipelines(M_pad, N_pad);
    Workspace w = cache.get_workspace(batch, M_pad, N_pad, original_M, original_N, original_K);

    id<MTLBuffer> buf_src = [cache.rt.device newBufferWithBytesNoCopy:(void*)a_f32.data<float>()
                                                              length:a_f32.nbytes()
                                                             options:MTLResourceStorageModeShared
                                                         deallocator:nil];

    id<MTLCommandBuffer>         cmd = [cache.rt.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

    [enc setComputePipelineState:p.pso_transpose];
    [enc setBuffer:buf_src  offset:0 atIndex:0];
    [enc setBuffer:w.buf_A  offset:0 atIndex:1];
    [enc setBytes:&original_M length:sizeof(uint) atIndex:2];
    [enc setBytes:&original_N length:sizeof(uint) atIndex:3];
    [enc dispatchThreads:MTLSizeMake(N_pad, M_pad, batch) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];

    [enc setComputePipelineState:p.pso_init_q];
    [enc setBuffer:w.buf_Q offset:0 atIndex:0];
    [enc dispatchThreads:MTLSizeMake(M_pad, M_pad, batch) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];

    for (uint block_start = 0; block_start < original_N; block_start += 32) {
        [enc setComputePipelineState:p.pso_panel];
        [enc setBuffer:w.buf_A      offset:0 atIndex:0];
        [enc setBuffer:w.buf_R_diag offset:0 atIndex:1];
        [enc setBuffer:w.buf_tau    offset:0 atIndex:2];
        [enc setBytes:&block_start length:sizeof(uint) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, batch) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];

        [enc setComputePipelineState:p.pso_t_mat];
        [enc setBuffer:w.buf_A   offset:0 atIndex:0];
        [enc setBuffer:w.buf_T   offset:0 atIndex:1];
        [enc setBuffer:w.buf_tau offset:0 atIndex:2];
        [enc setBytes:&block_start length:sizeof(uint) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, batch) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];

        uint update_cols_A = (N_pad - block_start);
        if (update_cols_A > 32) {
            uint groups_x    = (update_cols_A - 32 + 31) / 32;
            uint is_Q_update = 0;
            [enc setComputePipelineState:p.pso_update];
            [enc setBuffer:w.buf_A offset:0 atIndex:0];
            [enc setBuffer:w.buf_Q offset:0 atIndex:1];
            [enc setBuffer:w.buf_T offset:0 atIndex:2];
            [enc setBytes:&block_start length:sizeof(uint) atIndex:3];
            [enc setBytes:&is_Q_update  length:sizeof(uint) atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(groups_x, 1, batch) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        }

        uint is_Q_update = 1;
        uint groups_Q_x  = (M_pad + 31) / 32;
        [enc setComputePipelineState:p.pso_update];
        [enc setBuffer:w.buf_A offset:0 atIndex:0];
        [enc setBuffer:w.buf_Q offset:0 atIndex:1];
        [enc setBuffer:w.buf_T offset:0 atIndex:2];
        [enc setBytes:&block_start length:sizeof(uint) atIndex:3];
        [enc setBytes:&is_Q_update  length:sizeof(uint) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(groups_Q_x, 1, batch) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    }

    [enc setComputePipelineState:p.pso_haar];
    [enc setBuffer:w.buf_A offset:0 atIndex:0];
    [enc setBuffer:w.buf_Q offset:0 atIndex:1];
    [enc setBuffer:w.buf_R_diag offset:0 atIndex:2];
    [enc setBytes:&original_M length:sizeof(uint) atIndex:3];
    [enc setBytes:&original_K length:sizeof(uint) atIndex:4];
    [enc dispatchThreadgroups:MTLSizeMake(1, 1, batch) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];

    // --- Unpack R (Padded Col-major -> Exact Row-major + Triu) ---
    MTLSize memory_threads = MTLSizeMake(16, 16, 1);
    [enc setComputePipelineState:p.pso_unpk_R];
    [enc setBuffer:w.buf_A offset:0 atIndex:0];
    [enc setBuffer:w.buf_R_out offset:0 atIndex:1];
    [enc setBytes:&original_M length:sizeof(uint) atIndex:2];
    [enc setBytes:&original_N length:sizeof(uint) atIndex:3];
    [enc setBytes:&original_K length:sizeof(uint) atIndex:4];
    [enc setBytes:&M_pad length:sizeof(uint) atIndex:5];
    [enc setBytes:&N_pad length:sizeof(uint) atIndex:6];
    [enc dispatchThreads:MTLSizeMake(original_N, original_K, batch) threadsPerThreadgroup:memory_threads];

    // --- Unpack Q (Padded Col-major -> Exact Row-major) ---
    [enc setComputePipelineState:p.pso_unpk_Q];
    [enc setBuffer:w.buf_Q offset:0 atIndex:0];
    [enc setBuffer:w.buf_Q_out offset:0 atIndex:1];
    [enc setBytes:&original_M length:sizeof(uint) atIndex:2];
    [enc setBytes:&original_K length:sizeof(uint) atIndex:3];
    [enc setBytes:&M_pad length:sizeof(uint) atIndex:4];
    [enc dispatchThreads:MTLSizeMake(original_K, original_M, batch) threadsPerThreadgroup:memory_threads];

    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];

    if (cmd.error) {
        throw std::runtime_error(std::string("[qr_streaming_amx_complete] GPU error: ")
                                 + cmd.error.localizedDescription.UTF8String);
    }

    // 4. Instant MLX Handoff
    Shape R_shape(shape.begin(), shape.end());
    R_shape[R_shape.size() - 2] = original_K;

    Shape Q_shape(shape.begin(), shape.end());
    Q_shape[Q_shape.size() - 1] = original_K;

    const float* r_ptr = static_cast<const float*>([w.buf_R_out contents]);
    const float* q_ptr = static_cast<const float*>([w.buf_Q_out contents]);

    array R = array(r_ptr, R_shape, float32);
    if (in.scaled) R = multiply(R, in.unscale);
    array Q = array(q_ptr, Q_shape, float32);

    return {Q, R};
}

} // namespace metal_linalg::detail