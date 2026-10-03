#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

// C (M x N) = A (M x K) * B (K x N), row-major, M % TM == 0, N % TN == 0.
// One threadgroup per TM x TN tile of C; 4 simdgroups cooperate.
template <typename TA, typename TB, int TM, int TN, bool RELAXED>
inline void mm(device const TA* a, device const TB* b, device float* c,
               constant uint3& mnk, uint2 tg) {
    const int M = mnk.x, N = mnk.y, K = mnk.z;
    tensor<device TA, dextents<int32_t, 2>, tensor_inline> A((device TA*)a, dextents<int32_t, 2>(K, M));
    tensor<device TB, dextents<int32_t, 2>, tensor_inline> B((device TB*)b, dextents<int32_t, 2>(N, K));
    tensor<device float, dextents<int32_t, 2>, tensor_inline> C(c, dextents<int32_t, 2>(N, M));
    constexpr auto desc = matmul2d_descriptor(TM, TN, static_cast<int>(dynamic_extent), false, false, RELAXED);
    matmul2d<desc, execution_simdgroups<4>> op;
    auto mA = A.slice(0, tg.y * TM);
    auto mB = B.slice(tg.x * TN, 0);
    auto mC = C.slice(tg.x * TN, tg.y * TM);
    auto cT = op.template get_destination_cooperative_tensor<decltype(mA), decltype(mB), float>();
    #pragma unroll
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) if (cT.is_valid_element(i)) cT[i] = 0;
    op.run(mA, mB, cT);
    cT.store(mC);
}

#define KERNEL(name, TA, TB, TM, TN, R) \
kernel void name(device const TA* a [[buffer(0)]], device const TB* b [[buffer(1)]], \
                 device float* c [[buffer(2)]], constant uint3& mnk [[buffer(3)]], \
                 uint2 tg [[threadgroup_position_in_grid]]) { mm<TA, TB, TM, TN, R>(a, b, c, mnk, tg); }

KERNEL(mm_f32_64x32,        float, float, 64, 32, false)
KERNEL(mm_f32_64x64,        float, float, 64, 64, false)
KERNEL(mm_f32_128x64,       float, float, 128, 64, false)
KERNEL(mm_f32r_64x64,       float, float, 64, 64, true)
KERNEL(mm_f32r_128x64,      float, float, 128, 64, true)
KERNEL(mm_f16_64x64,        half,  half,  64, 64, false)
KERNEL(mm_f16_128x64,       half,  half,  128, 64, false)
KERNEL(mm_bf16_128x64,      bfloat, bfloat, 128, 64, false)

kernel void copy4(device const float4* src [[buffer(0)]], device float4* dst [[buffer(1)]],
                  uint i [[thread_position_in_grid]]) { dst[i] = src[i]; }
kernel void read4(device const float4* src [[buffer(0)]], device float* dst [[buffer(1)]],
                  constant uint& n [[buffer(2)]],
                  uint i [[thread_position_in_grid]], uint g [[threads_per_grid]]) {
    float4 s = 0;
    for (uint j = i; j < n; j += g) s += src[j];
    if (s.x == -1.2345f) dst[0] = s.y;   // keep the loads
}
