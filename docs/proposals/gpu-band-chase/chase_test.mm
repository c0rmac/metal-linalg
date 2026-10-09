// chase <n> <batch> [rec [empty]]: sb_chase (or sb_chase2) on the GPU against
// band_to_tridiagonal on the CPU. LIB is a metallib of Svd_Bidiag.metal with
// sb_chase.metal (and sb_chase2.metal) appended, KER the kernel, TPT the
// threadgroup's threads (1024). From the repository's root:
//   clang++ -std=c++17 -O2 -fobjc-arc -Isrc chase_test.mm src/band_chase.cpp \
//       -framework Metal -framework Foundation -framework Accelerate -o chase
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#import <Metal/Metal.h>
#include "band_chase.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
using namespace metal_linalg::detail;
struct ChParams { uint32_t n, ld, sw, rec, pmax, sl, stau, steps; };
static int tasks(int s, int n) { int ed0 = std::min(s + 16, n - 1); int m = n - 1 > ed0 ? (n - 1 - ed0 + 15) / 16 : 0; return 2 * m + 1; }
int main(int argc, char** argv) {
    const int n = atoi(argv[1]), B = argc > 2 ? atoi(argv[2]) : 1; const bool rec = argc > 3; const bool empty = argc > 4;
    const int kd = 16, ld = 2 * kd + 1;
    std::mt19937 g(5); std::normal_distribution<float> nd;
    const size_t sw = (size_t)n * ld;
    std::vector<float> band(sw * B, 0.0f);
    for (int b = 0; b < B; ++b) for (int c = 0; c < n; ++c) for (int r = c; r < n && r <= c + kd; ++r) band[b * sw + (size_t)c * ld + r - c] = nd(g) * 0.5f;
    const size_t pmax = (n - 2) / 16, nblocks = (pmax + 1) * (pmax + 2) / 2, sl = nblocks * kChaseBlockFloats;
    // CPU
    std::vector<float> cband = band, cd(n * B), ce(n * B), cL(sl * B, 0.0f), cLt(nblocks * 16 * B, 0.0f);
    auto t0 = std::chrono::steady_clock::now();
    for (int b = 0; b < B; ++b) {
        ChaseReflectors r; r.L = cL.data() + b * sl; r.Ltau = cLt.data() + b * nblocks * 16; r.pmax = pmax;
        band_to_tridiagonal(n, kd, cband.data() + b * sw, ld, cd.data() + b * n, ce.data() + b * n, 1, rec ? &r : nullptr);
    }
    double cms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    // GPU
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice(); NSError* err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:getenv("LIB")]] error:&err];
    id<MTLComputePipelineState> ps = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:[NSString stringWithUTF8String:getenv("KER")]] error:&err];
    if (!ps) { printf("no pipeline %s\n", err.localizedDescription.UTF8String); return 1; }
    id<MTLBuffer> W = [dev newBufferWithBytes:band.data() length:band.size() * 4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> L = [dev newBufferWithLength:std::max<size_t>(sl * B, 1) * 4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> Lt = [dev newBufferWithLength:std::max<size_t>(nblocks * 16 * B, 1) * 4 options:MTLResourceStorageModeShared];
    memset(L.contents, 0, sl * B * 4); memset(Lt.contents, 0, nblocks * 16 * B * 4);
    int steps = 0; for (int s = 0; s <= n - 2; ++s) steps = std::max(steps, 3 * s + tasks(s, n));
    ChParams p{(uint32_t)n, (uint32_t)ld, (uint32_t)sw, (rec ? 1u : 0u) | (empty ? 2u : 0u), (uint32_t)pmax, (uint32_t)sl, (uint32_t)(nblocks * 16), (uint32_t)steps};
    id<MTLCommandQueue> q = [dev newCommandQueue];
    double gms = 1e9;
    for (int rep = 0; rep < 3; ++rep) {
        memcpy(W.contents, band.data(), band.size() * 4);
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:ps]; [e setBuffer:W offset:0 atIndex:0]; [e setBuffer:L offset:0 atIndex:1]; [e setBuffer:Lt offset:0 atIndex:2];
        [e setBytes:&p length:sizeof p atIndex:3];
        [e dispatchThreadgroups:MTLSizeMake(B, 1, 1) threadsPerThreadgroup:MTLSizeMake(getenv("TPT") ? atoi(getenv("TPT")) : 1024, 1, 1)];
        [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
        if (cb.error) { printf("GPU error %s\n", cb.error.localizedDescription.UTF8String); return 1; }
        gms = std::min(gms, (cb.GPUEndTime - cb.GPUStartTime) * 1e3);
    }
    const float* gw = (const float*)W.contents;
    // compare eigenvalues of the two tridiagonals
    double maxd = 0, scale = 0;
    for (int b = 0; b < B; ++b) {
        std::vector<float> d1(n), e1(n), d2(n), e2(n);
        for (int i = 0; i < n; ++i) { d1[i] = cd[b * n + i]; e1[i] = ce[b * n + i]; d2[i] = gw[b * sw + (size_t)i * ld]; e2[i] = i + 1 < n ? gw[b * sw + (size_t)i * ld + 1] : 0; }
        __LAPACK_int N = n, info; ssterf_(&N, d1.data(), e1.data(), &info); ssterf_(&N, d2.data(), e2.data(), &info);
        for (int i = 0; i < n; ++i) { maxd = std::max(maxd, (double)std::fabs(d1[i] - d2[i])); scale = std::max(scale, (double)std::fabs(d1[i])); }
    }
    double maxl = 0, maxt = 0;
    if (rec) {
        const float* gl = (const float*)L.contents; const float* gt = (const float*)Lt.contents;
        for (size_t i = 0; i < sl * B; ++i) maxl = std::max(maxl, (double)std::fabs(gl[i] - cL[i]));
        for (size_t i = 0; i < nblocks * 16 * B; ++i) maxt = std::max(maxt, (double)std::fabs(gt[i] - cLt[i]));
    }
    printf("max %lu  n=%d B=%d rec=%d  cpu %.2f ms (1 thread each, sequential)  gpu %.2f ms  |eig diff|/max %.2e  |L diff| %.2e |tau diff| %.2e\n",
           (unsigned long)ps.maxTotalThreadsPerThreadgroup, n, B, rec, cms, gms, maxd / scale, maxl, maxt);
}
