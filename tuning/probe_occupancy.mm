// Reports the real occupancy limits of each pipeline, so dispatch heuristics
// can use a measured saturation point instead of a fitted constant.
//
// The number that matters for dispatch is how many threadgroups the device can
// keep resident at once: qr_unblocked runs one simdgroup or threadgroup per
// matrix (qr_householder), so its parallelism is capped by that figure, while
// qr_streaming_amx_reduced spreads each matrix over many.

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include <cstdio>
#include "metal_runtime.h"
#include "shaders.h"

using namespace metal_linalg::detail;

namespace {

void describe(const char* label, id<MTLComputePipelineState> pso, id<MTLDevice> dev) {
    const NSUInteger maxThreads = pso.maxTotalThreadsPerThreadgroup;
    const NSUInteger width      = pso.threadExecutionWidth;
    const NSUInteger tgMem      = pso.staticThreadgroupMemoryLength;

    std::printf("  %-32s maxThreads/tg=%-5lu simdWidth=%-3lu staticTgMem=%lu B\n",
                label, (unsigned long)maxThreads, (unsigned long)width,
                (unsigned long)tgMem);
    (void)dev;
}

} // namespace

int main() {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) { std::fprintf(stderr, "no Metal device\n"); return 1; }

        std::printf("device: %s\n", dev.name.UTF8String);
        std::printf("  maxThreadgroupMemoryLength = %lu B\n",
                    (unsigned long)dev.maxThreadgroupMemoryLength);
        std::printf("  maxBufferLength            = %.1f GB\n",
                    (double)dev.maxBufferLength / (1024.0 * 1024.0 * 1024.0));
        std::printf("  hasUnifiedMemory           = %s\n",
                    dev.hasUnifiedMemory ? "yes" : "no");
        std::printf("  recommendedWorkingSetSize  = %.1f GB\n",
                    (double)dev.recommendedMaxWorkingSetSize / (1024.0 * 1024.0 * 1024.0));

        MetalRuntime& hh = MetalRuntime::shared(METAL_LINALG_SHADER(QR_Householder), "probe");
        MetalRuntime& red = MetalRuntime::shared(METAL_LINALG_SHADER(QR_Streaming_AMX_Reduced), "probe");
        uint M_pad = 128, N_pad = 128;

        std::printf("\nqr_householder pipelines:\n");
        for (NSString* name in @[@"qr_householder_simd_32_4", @"qr_householder_wy_1", @"qr_householder_wy_2",
                                 @"qr_householder_wy_4"]) {
            describe(name.UTF8String, make_pipeline(hh.device, hh.library, name, nil), hh.device);
        }

        MTLFunctionConstantValues* cv2 = [[MTLFunctionConstantValues alloc] init];
        [cv2 setConstantValue:&M_pad type:MTLDataTypeUInt atIndex:0];
        [cv2 setConstantValue:&N_pad type:MTLDataTypeUInt atIndex:1];

        std::printf("\nqr_streaming_amx_reduced pipelines:\n");
        for (NSString* name in @[@"panel_factorization", @"compute_t_matrix",
                                 @"grid_parallel_update", @"postprocess_haar_fix"]) {
            describe(name.UTF8String,
                     make_pipeline(red.device, red.library, name, cv2), red.device);
        }
    }
    return 0;
}
