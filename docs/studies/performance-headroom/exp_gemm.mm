// Experiment 4: the matrix-multiply and bandwidth ceilings on this Mac.
//   - MPS FP32 GEMM (what the tridiag/bidiag backends use)
//   - Metal 4 tensor ops (matmul2d, the M5's per-core neural accelerators):
//     FP32 strict, FP32 relaxed, FP16 and BF16 inputs, with their accuracy
//   - Accelerate cblas_sgemm on the CPU (SME), and CPU + GPU at once
//   - device-memory read bandwidth
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <Accelerate/Accelerate.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <thread>
#include <vector>

using clk = std::chrono::steady_clock;
static double now_s() { return std::chrono::duration<double>(clk::now().time_since_epoch()).count(); }

static id<MTLDevice> dev;
static id<MTLCommandQueue> q;

static double gpu_time(id<MTLCommandBuffer> cb) { return cb.GPUEndTime - cb.GPUStartTime; }

// Max over sampled entries of |c - exact| / (|a_i| |b_j|), the normwise relative error of each dot product.
static double sample_err(const float* A, const float* B, const float* C, int n, std::mt19937& rng) {
    std::uniform_int_distribution<int> d(0, n - 1);
    double worst = 0;
    for (int s = 0; s < 256; ++s) {
        int i = d(rng), j = d(rng);
        double ex = 0, na = 0, nb = 0;
        for (int k = 0; k < n; ++k) {
            ex += (double)A[(size_t)i * n + k] * B[(size_t)k * n + j];
            na += (double)A[(size_t)i * n + k] * A[(size_t)i * n + k];
            nb += (double)B[(size_t)k * n + j] * B[(size_t)k * n + j];
        }
        worst = std::max(worst, std::fabs(C[(size_t)i * n + j] - ex) / std::sqrt(na * nb));
    }
    return worst;
}

int main() {
    @autoreleasepool {
        dev = MTLCreateSystemDefaultDevice();
        q = [dev newCommandQueue];
        std::printf("device %s\n", dev.name.UTF8String);
        NSError* err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithURL:[NSURL fileURLWithPath:@"gemm.metallib"] error:&err];
        if (!lib) { std::printf("lib: %s\n", err.localizedDescription.UTF8String); return 1; }
        std::mt19937 rng(3);
        std::normal_distribution<float> nd;

        // ---- bandwidth ----
        {
            const size_t bytes = 1ull << 30;
            id<MTLBuffer> src = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
            id<MTLBuffer> dst = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
            memset(src.contents, 1, bytes);
            id<MTLComputePipelineState> rd = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"read4"] error:&err];
            id<MTLComputePipelineState> cp = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"copy4"] error:&err];
            double best_r = 1e9, best_c = 1e9;
            uint32_t n4 = (uint32_t)(bytes / 16);
            for (int r = 0; r < 6; ++r) {
                id<MTLCommandBuffer> cb = [q commandBuffer];
                id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
                [e setComputePipelineState:rd];
                [e setBuffer:src offset:0 atIndex:0]; [e setBuffer:dst offset:0 atIndex:1];
                [e setBytes:&n4 length:4 atIndex:2];
                [e dispatchThreads:MTLSizeMake(20 * 1024 * 8, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
                best_r = std::min(best_r, gpu_time(cb));
                cb = [q commandBuffer];
                e = [cb computeCommandEncoder];
                [e setComputePipelineState:cp];
                [e setBuffer:src offset:0 atIndex:0]; [e setBuffer:dst offset:0 atIndex:1];
                [e dispatchThreads:MTLSizeMake(n4, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
                best_c = std::min(best_c, gpu_time(cb));
            }
            std::printf("GPU bandwidth: read %.0f GB/s, copy %.0f GB/s (read+write)\n", bytes / best_r / 1e9,
                        2.0 * bytes / best_c / 1e9);
            // CPU read bandwidth, all cores
            std::atomic<double> sink{0};
            const unsigned T = std::thread::hardware_concurrency();
            double best_cpu = 1e9;
            for (int r = 0; r < 4; ++r) {
                double t0 = now_s();
                std::vector<std::thread> th;
                for (unsigned t = 0; t < T; ++t) th.emplace_back([&, t] {
                    const float* p = (const float*)src.contents + (bytes / 4) * t / T;
                    size_t cnt = (bytes / 4) / T;
                    float s = 0; vDSP_sve(p, 1, &s, cnt); sink = sink + s;
                });
                for (auto& x : th) x.join();
                best_cpu = std::min(best_cpu, now_s() - t0);
            }
            std::printf("CPU bandwidth (%u threads, vDSP_sve): read %.0f GB/s\n", T, bytes / best_cpu / 1e9);
        }

        for (int n : {2048, 4096}) {
            const size_t nn = (size_t)n * n;
            id<MTLBuffer> A = [dev newBufferWithLength:nn * 4 options:MTLResourceStorageModeShared];
            id<MTLBuffer> B = [dev newBufferWithLength:nn * 4 options:MTLResourceStorageModeShared];
            id<MTLBuffer> C = [dev newBufferWithLength:nn * 4 options:MTLResourceStorageModeShared];
            id<MTLBuffer> Ah = [dev newBufferWithLength:nn * 2 options:MTLResourceStorageModeShared];
            id<MTLBuffer> Bh = [dev newBufferWithLength:nn * 2 options:MTLResourceStorageModeShared];
            id<MTLBuffer> Ab = [dev newBufferWithLength:nn * 2 options:MTLResourceStorageModeShared];
            id<MTLBuffer> Bb = [dev newBufferWithLength:nn * 2 options:MTLResourceStorageModeShared];
            float* a = (float*)A.contents; float* b = (float*)B.contents;
            for (size_t i = 0; i < nn; ++i) { a[i] = nd(rng); b[i] = nd(rng); }
            __fp16* ah = (__fp16*)Ah.contents; __fp16* bh = (__fp16*)Bh.contents;
            uint16_t* ab = (uint16_t*)Ab.contents; uint16_t* bb = (uint16_t*)Bb.contents;
            for (size_t i = 0; i < nn; ++i) {
                ah[i] = (__fp16)a[i]; bh[i] = (__fp16)b[i];
                uint32_t ua, ub; memcpy(&ua, &a[i], 4); memcpy(&ub, &b[i], 4);
                ab[i] = (uint16_t)((ua + 0x7FFF + ((ua >> 16) & 1)) >> 16);
                bb[i] = (uint16_t)((ub + 0x7FFF + ((ub >> 16) & 1)) >> 16);
            }
            const double flops = 2.0 * n * (double)n * n;
            std::printf("\n=== %d x %d x %d ===\n", n, n, n);

            // MPS FP32
            {
                MPSMatrixDescriptor* d = [MPSMatrixDescriptor matrixDescriptorWithRows:n columns:n rowBytes:n * 4 dataType:MPSDataTypeFloat32];
                MPSMatrix* mA = [[MPSMatrix alloc] initWithBuffer:A descriptor:d];
                MPSMatrix* mB = [[MPSMatrix alloc] initWithBuffer:B descriptor:d];
                MPSMatrix* mC = [[MPSMatrix alloc] initWithBuffer:C descriptor:d];
                MPSMatrixMultiplication* mm = [[MPSMatrixMultiplication alloc] initWithDevice:dev transposeLeft:NO transposeRight:NO resultRows:n resultColumns:n interiorColumns:n alpha:1.0 beta:0.0];
                double best = 1e9;
                for (int r = 0; r < 6; ++r) {
                    id<MTLCommandBuffer> cb = [q commandBuffer];
                    [mm encodeToCommandBuffer:cb leftMatrix:mA rightMatrix:mB resultMatrix:mC];
                    [cb commit]; [cb waitUntilCompleted];
                    if (r) best = std::min(best, gpu_time(cb));
                }
                std::printf("%-26s %7.2f ms  %6.2f TFLOP/s  err %.1e\n", "MPS fp32", best * 1e3, flops / best / 1e12,
                            sample_err(a, b, (float*)C.contents, n, rng));
            }
            // Metal 4 tensor ops
            struct K { const char* name; int tm, tn; int kind; };   // kind 0 f32, 1 f16, 2 bf16
            for (K k : {K{"mm_f32_64x32", 64, 32, 0}, K{"mm_f32_64x64", 64, 64, 0}, K{"mm_f32_128x64", 128, 64, 0},
                        K{"mm_f32r_64x64", 64, 64, 0}, K{"mm_f32r_128x64", 128, 64, 0},
                        K{"mm_f16_64x64", 64, 64, 1}, K{"mm_f16_128x64", 128, 64, 1}, K{"mm_bf16_128x64", 128, 64, 2}}) {
                id<MTLFunction> f = [lib newFunctionWithName:@(k.name)];
                id<MTLComputePipelineState> ps = [dev newComputePipelineStateWithFunction:f error:&err];
                if (!ps) { std::printf("%s: %s\n", k.name, err.localizedDescription.UTF8String); continue; }
                uint32_t mnk[4] = {(uint32_t)n, (uint32_t)n, (uint32_t)n, 0};
                double best = 1e9;
                for (int r = 0; r < 6; ++r) {
                    id<MTLCommandBuffer> cb = [q commandBuffer];
                    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
                    [e setComputePipelineState:ps];
                    [e setBuffer:(k.kind == 0 ? A : k.kind == 1 ? Ah : Ab) offset:0 atIndex:0];
                    [e setBuffer:(k.kind == 0 ? B : k.kind == 1 ? Bh : Bb) offset:0 atIndex:1];
                    [e setBuffer:C offset:0 atIndex:2];
                    [e setBytes:mnk length:16 atIndex:3];
                    [e dispatchThreadgroups:MTLSizeMake(n / k.tn, n / k.tm, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                    [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
                    if (cb.error) { std::printf("%s: %s\n", k.name, cb.error.localizedDescription.UTF8String); break; }
                    if (r) best = std::min(best, gpu_time(cb));
                }
                std::printf("%-26s %7.2f ms  %6.2f TFLOP/s  err %.1e\n", k.name, best * 1e3, flops / best / 1e12,
                            sample_err(a, b, (float*)C.contents, n, rng));
            }
            // CPU sgemm (Accelerate, default threading)
            std::vector<float> cc(nn);
            double best_cpu = 1e9;
            for (int r = 0; r < 4; ++r) {
                double t0 = now_s();
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0f, a, n, b, n, 0.0f, cc.data(), n);
                if (r) best_cpu = std::min(best_cpu, now_s() - t0);
            }
            std::printf("%-26s %7.2f ms  %6.2f TFLOP/s  err %.1e\n", "Accelerate sgemm (CPU)", best_cpu * 1e3,
                        flops / best_cpu / 1e12, sample_err(a, b, cc.data(), n, rng));

            // CPU and GPU at once: GPU runs MPS fp32 GEMMs back to back while the CPU runs sgemm.
            {
                MPSMatrixDescriptor* d = [MPSMatrixDescriptor matrixDescriptorWithRows:n columns:n rowBytes:n * 4 dataType:MPSDataTypeFloat32];
                MPSMatrix* mA = [[MPSMatrix alloc] initWithBuffer:A descriptor:d];
                MPSMatrix* mB = [[MPSMatrix alloc] initWithBuffer:B descriptor:d];
                MPSMatrix* mC = [[MPSMatrix alloc] initWithBuffer:C descriptor:d];
                MPSMatrixMultiplication* mm = [[MPSMatrixMultiplication alloc] initWithDevice:dev transposeLeft:NO transposeRight:NO resultRows:n resultColumns:n interiorColumns:n alpha:1.0 beta:0.0];
                const int G = n == 2048 ? 40 : 8, Cn = n == 2048 ? 12 : 3;
                std::vector<float> a2(a, a + nn), b2(b, b + nn);
                double t_gpu = 0, t_cpu = 0;
                std::thread cpu([&] {
                    double t0 = now_s();
                    for (int i = 0; i < Cn; ++i)
                        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0f, a2.data(), n, b2.data(), n, 0.0f, cc.data(), n);
                    t_cpu = now_s() - t0;
                });
                double t0 = now_s();
                id<MTLCommandBuffer> cb = [q commandBuffer];
                for (int i = 0; i < G; ++i) [mm encodeToCommandBuffer:cb leftMatrix:mA rightMatrix:mB resultMatrix:mC];
                [cb commit]; [cb waitUntilCompleted];
                t_gpu = now_s() - t0;
                cpu.join();
                std::printf("concurrent: GPU MPS %.2f TFLOP/s and CPU %.2f TFLOP/s at once -> %.2f TFLOP/s combined\n",
                            G * flops / t_gpu / 1e12, Cn * flops / t_cpu / 1e12,
                            (G * flops / t_gpu + Cn * flops / t_cpu) / 1e12);
            }
        }
    }
}
