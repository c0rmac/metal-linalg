#import <Metal/Metal.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
// Phase profile of the tridiagonal-QL prototype: exp_stage <metallib> <N>...
int main(int argc, char** argv) { @autoreleasepool {
  if (argc < 3) { std::printf("usage: exp_stage <metallib> <N>...\n"); return 1; }
  id<MTLDevice> dev = MTLCreateSystemDefaultDevice(); id<MTLCommandQueue> q = [dev newCommandQueue];
  NSError* err = nil;
  id<MTLLibrary> lib = [dev newLibraryWithURL:[NSURL fileURLWithPath:@(argv[1])] error:&err];
  std::mt19937 rng(5); std::normal_distribution<float> nd;
  for (int arg = 2; arg < argc; ++arg) {
    uint32_t n = (uint32_t)atoi(argv[arg]);
    uint32_t batch = 2048, per = n*n;
    id<MTLBuffer> A = [dev newBufferWithLength:per*batch*4 options:0], W = [dev newBufferWithLength:n*batch*4 options:0],
                  V = [dev newBufferWithLength:per*batch*4 options:0], I = [dev newBufferWithLength:batch*4 options:0];
    float* a = (float*)A.contents;
    for (uint32_t b = 0; b < batch; ++b) for (uint32_t i = 0; i < n; ++i) for (uint32_t j = 0; j <= i; ++j) a[b*per+i*n+j] = a[b*per+j*n+i] = nd(rng);
    std::printf("N=%u batch %u:", n, batch);
    for (uint32_t stage : {1u, 2u, 5u, 4u}) {
      MTLFunctionConstantValues* cv = [MTLFunctionConstantValues new];
      [cv setConstantValue:&stage type:MTLDataTypeUInt atIndex:0];
      id<MTLFunction> f = [lib newFunctionWithName:@"eigh_tdql" constantValues:cv error:&err];
      id<MTLComputePipelineState> ps = [dev newComputePipelineStateWithFunction:f error:&err];
      double best = 1e9;
      for (int r = 0; r < 6; ++r) {
        id<MTLCommandBuffer> cb = [q commandBuffer]; id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:ps]; [e setBuffer:A offset:0 atIndex:0]; [e setBuffer:W offset:0 atIndex:1];
        [e setBuffer:V offset:0 atIndex:2]; [e setBuffer:I offset:0 atIndex:3]; [e setBytes:&n length:4 atIndex:4];
        [e dispatchThreadgroups:MTLSizeMake(batch,1,1) threadsPerThreadgroup:MTLSizeMake(n <= 32 ? 32 : 64,1,1)];
        [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
        if (r) best = std::min(best, cb.GPUEndTime - cb.GPUStartTime);
      }
      std::printf("  stage<=%u %.2f ms", stage, best*1e3);
      if (stage == 1) std::printf(" (maxTG %lu, TGmem %lu)", (unsigned long)ps.maxTotalThreadsPerThreadgroup, (unsigned long)ps.staticThreadgroupMemoryLength);
    }
    std::printf("\n");
  }
}}
