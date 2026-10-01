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

constexpr size_t kQRSharedMemBytes = 5120;

// Compiled pipelines for one (M, N) shape.
struct QRPipelines {
    id<MTLComputePipelineState> pso_qr;
    id<MTLComputePipelineState> pso_pack;
    id<MTLComputePipelineState> pso_init_q;
    id<MTLComputePipelineState> pso_unpk_R;
    id<MTLComputePipelineState> pso_unpk_Q;
};

// Recycled GPU allocations, to keep repeat calls off the OS allocator.
struct Workspace {
    id<MTLBuffer> buf_A_pad;
    id<MTLBuffer> buf_Q_pad;
    id<MTLBuffer> buf_R_out;
    id<MTLBuffer> buf_Q_out;
};

// Process-lifetime cache of everything that only depends on the input shape.
struct Cache {
    MetalRuntime& rt = MetalRuntime::shared(METAL_LINALG_SHADER(QR_Unblocked), "qr_unblocked");

    std::map<std::pair<uint, uint>, QRPipelines>     pipelines;   // keyed by (M, N)
    std::map<std::tuple<uint, uint, uint>, Workspace> workspaces; // keyed by (batch, M, N)

    QRPipelines get_pipelines(uint M, uint N, uint M_pad, uint N_pad) {
        auto key = std::make_pair(M, N);
        if (auto it = pipelines.find(key); it != pipelines.end()) {
            return it->second;
        }

        // The monolithic QR kernel is specialised on the original and padded
        // dimensions; the memory-shuffling helpers take theirs as buffers.
        MTLFunctionConstantValues* cv = [[MTLFunctionConstantValues alloc] init];
        [cv setConstantValue:&M     type:MTLDataTypeUInt atIndex:0];
        [cv setConstantValue:&N     type:MTLDataTypeUInt atIndex:1];
        [cv setConstantValue:&M_pad type:MTLDataTypeUInt atIndex:2];
        [cv setConstantValue:&N_pad type:MTLDataTypeUInt atIndex:3];

        QRPipelines p;
        p.pso_qr     = make_pipeline(rt.device, rt.library, @"standard_householder_qr_float32", cv);
        p.pso_pack   = make_pipeline(rt.device, rt.library, @"pack_batch_memory",   nil);
        p.pso_init_q = make_pipeline(rt.device, rt.library, @"init_identity_batch", nil);
        p.pso_unpk_R = make_pipeline(rt.device, rt.library, @"unpack_batch_R",      nil);
        p.pso_unpk_Q = make_pipeline(rt.device, rt.library, @"unpack_batch_Q",      nil);

        return pipelines[key] = p;
    }

    Workspace get_workspace(uint batch, uint M, uint N, uint M_pad, uint N_pad, uint K) {
        auto key = std::make_tuple(batch, M, N);
        if (auto it = workspaces.find(key); it != workspaces.end()) {
            return it->second;
        }

        const MTLResourceOptions opt = MTLResourceStorageModeShared;
        Workspace w;
        w.buf_A_pad = [rt.device newBufferWithLength:((size_t)batch * M_pad * N_pad * sizeof(float)) options:opt];
        w.buf_Q_pad = [rt.device newBufferWithLength:((size_t)batch * M_pad * M_pad * sizeof(float)) options:opt];
        w.buf_R_out = [rt.device newBufferWithLength:((size_t)batch * K * N * sizeof(float))         options:opt];
        w.buf_Q_out = [rt.device newBufferWithLength:((size_t)batch * M * K * sizeof(float))         options:opt];

        return workspaces[key] = w;
    }
};

} // namespace

// =============================================================================
// MAIN ENTRY POINT
// =============================================================================

std::pair<array, array> qr_unblocked(const array& a) {
    if (a.ndim() < 2) {
        throw std::invalid_argument("[qr_unblocked] Input must be at least a 2D matrix.");
    }

    const Shape& shape = a.shape();
    const uint M = static_cast<uint>(shape[shape.size() - 2]);
    const uint N = static_cast<uint>(shape[shape.size() - 1]);
    const uint K = std::min(M, N);

    uint batch = 1;
    for (size_t i = 0; i + 2 < shape.size(); ++i) {
        batch *= static_cast<uint>(shape[i]);
    }

    const uint M_pad = pad_up(M, 32);
    const uint N_pad = pad_up(N, 16);

    // 1. MLX Array Prep
    // Scaled by a power of two per matrix; see prepare_input_scaled.
    ScaledInput in = prepare_input_scaled(a);
    array a_f32 = in.a;

    // 2. Retrieve Cached State & Workspaces
    static Cache cache;
    QRPipelines p = cache.get_pipelines(M, N, M_pad, N_pad);
    Workspace w   = cache.get_workspace(batch, M, N, M_pad, N_pad, K);

    // 3. Map Input Data (Zero-Copy)
    id<MTLBuffer> buf_src = [cache.rt.device newBufferWithBytesNoCopy:(void*)a_f32.data<float>()
                                                              length:a_f32.nbytes()
                                                             options:MTLResourceStorageModeShared
                                                         deallocator:nil];

    // 4. Encode Command Sequence
    id<MTLCommandBuffer> cmd = [cache.rt.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

    MTLSize memory_threads = MTLSizeMake(16, 16, 1);

    // --- Pass 1: Pack Memory ---
    [enc setComputePipelineState:p.pso_pack];
    [enc setBuffer:buf_src offset:0 atIndex:0];
    [enc setBuffer:w.buf_A_pad offset:0 atIndex:1];
    [enc setBytes:&M length:sizeof(uint) atIndex:2];
    [enc setBytes:&N length:sizeof(uint) atIndex:3];
    [enc setBytes:&M_pad length:sizeof(uint) atIndex:4];
    [enc setBytes:&N_pad length:sizeof(uint) atIndex:5];
    [enc dispatchThreads:MTLSizeMake(N, M, batch) threadsPerThreadgroup:memory_threads];

    // --- Pass 2: Init Q to Identity ---
    [enc setComputePipelineState:p.pso_init_q];
    [enc setBuffer:w.buf_Q_pad offset:0 atIndex:0];
    [enc setBytes:&M_pad length:sizeof(uint) atIndex:1];
    [enc setBytes:&N_pad length:sizeof(uint) atIndex:2];
    [enc dispatchThreads:MTLSizeMake(M_pad, M_pad, batch) threadsPerThreadgroup:memory_threads];

    // --- Pass 3: The Monolithic QR Factorization ---
    uint max_simd_groups = std::min((N_pad + 7) / 8, 32u);
    max_simd_groups = std::max(max_simd_groups, 1u);

    [enc setComputePipelineState:p.pso_qr];
    [enc setBuffer:w.buf_A_pad offset:0 atIndex:0];
    [enc setBuffer:w.buf_Q_pad offset:0 atIndex:1];
    [enc setThreadgroupMemoryLength:kQRSharedMemBytes atIndex:0];

    [enc dispatchThreadgroups:MTLSizeMake(1, 1, batch)
        threadsPerThreadgroup:MTLSizeMake(32, max_simd_groups, 1)];

    // --- Pass 4: Unpack R ---
    [enc setComputePipelineState:p.pso_unpk_R];
    [enc setBuffer:w.buf_A_pad offset:0 atIndex:0];
    [enc setBuffer:w.buf_R_out offset:0 atIndex:1];
    [enc setBytes:&M length:sizeof(uint) atIndex:2];
    [enc setBytes:&N length:sizeof(uint) atIndex:3];
    [enc setBytes:&K length:sizeof(uint) atIndex:4];
    [enc setBytes:&M_pad length:sizeof(uint) atIndex:5];
    [enc setBytes:&N_pad length:sizeof(uint) atIndex:6];
    [enc dispatchThreads:MTLSizeMake(N, K, batch) threadsPerThreadgroup:memory_threads];

    // --- Pass 5: Unpack Q ---
    [enc setComputePipelineState:p.pso_unpk_Q];
    [enc setBuffer:w.buf_Q_pad offset:0 atIndex:0];
    [enc setBuffer:w.buf_Q_out offset:0 atIndex:1];
    [enc setBytes:&M length:sizeof(uint) atIndex:2];
    [enc setBytes:&K length:sizeof(uint) atIndex:3];
    [enc setBytes:&M_pad length:sizeof(uint) atIndex:4];
    [enc dispatchThreads:MTLSizeMake(K, M, batch) threadsPerThreadgroup:memory_threads];

    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];

    if (cmd.error) {
        throw std::runtime_error(std::string("[qr_unblocked] GPU Kernel Error: ") +
                                 cmd.error.localizedDescription.UTF8String);
    }

    // 5. Flat Array Handoff to MLX
    Shape R_shape(shape.begin(), shape.end());
    R_shape[R_shape.size() - 2] = K;

    Shape Q_shape(shape.begin(), shape.end());
    Q_shape[Q_shape.size() - 1] = K;

    const float* r_ptr = static_cast<const float*>([w.buf_R_out contents]);
    const float* q_ptr = static_cast<const float*>([w.buf_Q_out contents]);

    // Note: Passing the pointer like this forces MLX to deep copy the result,
    // which protects our recycled `Workspace` buffers from being overwritten by MLX later.
    array final_R = array(r_ptr, R_shape, float32);
    if (in.scaled) final_R = multiply(final_R, in.unscale);
    array final_Q = array(q_ptr, Q_shape, float32);

    return {final_Q, final_R};
}

} // namespace metal_linalg::detail