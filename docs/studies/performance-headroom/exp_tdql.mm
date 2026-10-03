// Prototype check: the batched tridiagonal-QL kernel against LAPACK (accuracy)
// and against the library's GPU path and the batch-parallel CPU (time).
#import <Metal/Metal.h>
#include <metal_linalg/core.h>
#include <Accelerate/Accelerate.h>
#include <vecLib/thread_api.h>
#include <dispatch/dispatch.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <functional>
#include <random>
#include <vector>

using namespace metal_linalg;
using clk = std::chrono::steady_clock;
static double now_ms() { return std::chrono::duration<double, std::milli>(clk::now().time_since_epoch()).count(); }
static double tmin(const std::function<void()>& f, int reps) {
    double w0 = now_ms();
    do { f(); } while (now_ms() - w0 < 250.0);
    reps = std::max(reps, 7);
    double best = 1e30;
    for (int r = 0; r < reps; ++r) { double t0 = now_ms(); f(); best = std::min(best, now_ms() - t0); }
    return best;
}
static float* page_alloc(size_t floats, size_t* bytes) {
    *bytes = std::max<size_t>(16384, (floats * 4 + 16383) / 16384 * 16384);
    void* p = nullptr; posix_memalign(&p, 16384, *bytes); return (float*)p;
}

int main() {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> q = [dev newCommandQueue];
        NSError* err = nil;
        auto make = [&](NSString* path) {
            id<MTLLibrary> lib = [dev newLibraryWithURL:[NSURL fileURLWithPath:path] error:nil];
            MTLFunctionConstantValues* cv = [MTLFunctionConstantValues new];
            uint32_t stage = 4;
            [cv setConstantValue:&stage type:MTLDataTypeUInt atIndex:0];
            id<MTLFunction> f = [lib newFunctionWithName:@"eigh_tdql" constantValues:cv error:nil];
            return [dev newComputePipelineStateWithFunction:f error:nil];
        };
        id<MTLComputePipelineState> ps64 = make(@"tdql64.metallib"), ps32 = make(@"tdql32.metallib");

        auto e = eigh_policy();
        e.gpu_max_n = kEighNoLimit; e.gpu_min_batch_times_n = 0; e.gpu_min_batch = 1; e.tridiag_min_n = 0;
        set_eigh_policy(e);

        std::mt19937 rng(5);
        std::normal_distribution<float> nd;
        std::printf("%4s %6s | %9s %9s %9s %9s | %7s %7s | %8s %8s %8s %4s\n", "N", "batch", "proto", "lib gpu",
                    "cpu par", "cpu ser", "vs gpu", "vs par", "resid", "orth", "dw", "bad");
        struct C { uint32_t n, batch; };
        for (C c : {C{8, 4096}, C{16, 4096}, C{24, 4096}, C{32, 256}, C{32, 2048}, C{32, 4096}, C{48, 2048}, C{64, 256}, C{64, 2048}}) {
            const uint32_t n = c.n, batch = c.batch;
            const size_t per = (size_t)n * n;
            size_t ba, bw, bv, bi;
            float* a = page_alloc(per * batch, &ba);
            float* w = page_alloc((size_t)n * batch, &bw);
            float* v = page_alloc(per * batch, &bv);
            uint32_t* info = (uint32_t*)page_alloc(batch, &bi);
            for (uint32_t b = 0; b < batch; ++b)
                for (uint32_t i = 0; i < n; ++i)
                    for (uint32_t j = 0; j <= i; ++j) a[b * per + i * n + j] = a[b * per + j * n + i] = nd(rng);
            id<MTLBuffer> A = [dev newBufferWithBytesNoCopy:a length:ba options:MTLResourceStorageModeShared deallocator:nil];
            id<MTLBuffer> W = [dev newBufferWithBytesNoCopy:w length:bw options:MTLResourceStorageModeShared deallocator:nil];
            id<MTLBuffer> V = [dev newBufferWithBytesNoCopy:v length:bv options:MTLResourceStorageModeShared deallocator:nil];
            id<MTLBuffer> I = [dev newBufferWithBytesNoCopy:info length:bi options:MTLResourceStorageModeShared deallocator:nil];
            const uint32_t threads = n <= 32 ? 32 : 64;
            id<MTLComputePipelineState> ps = n <= 32 ? ps32 : ps64;
            auto run = [&] {
                id<MTLCommandBuffer> cb = [q commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:ps];
                [enc setBuffer:A offset:0 atIndex:0]; [enc setBuffer:W offset:0 atIndex:1];
                [enc setBuffer:V offset:0 atIndex:2]; [enc setBuffer:I offset:0 atIndex:3];
                [enc setBytes:&n length:4 atIndex:4];
                [enc dispatchThreadgroups:MTLSizeMake(batch, 1, 1) threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
                [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
                if (cb.error) std::printf("GPU error %s\n", cb.error.localizedDescription.UTF8String);
            };
            const int reps = 5;
            double t_proto = tmin(run, reps);

            // accuracy against LAPACK
            std::vector<float> wl((size_t)n * batch);
            core::Matrices m{a, batch, n, n};
            core::detail::eigh_cpu(m, true, wl.data(), nullptr, nullptr);
            double resid = 0, orth = 0, dw = 0; uint32_t bad = 0;
            for (uint32_t b = 0; b < std::min<uint32_t>(batch, 64); ++b) {
                const float* A0 = a + b * per; const float* V0 = v + b * per; const float* w0 = w + (size_t)b * n;
                double na = 0, r = 0, o = 0, wmax = 0;
                for (size_t k = 0; k < per; ++k) na += (double)A0[k] * A0[k];
                for (uint32_t i = 0; i < n; ++i)
                    for (uint32_t j = 0; j < n; ++j) {
                        double av = 0, vv = 0;
                        for (uint32_t k = 0; k < n; ++k) { av += (double)A0[i * n + k] * V0[k * n + j]; vv += (double)V0[k * n + i] * V0[k * n + j]; }
                        av -= (double)V0[i * n + j] * w0[j];
                        r += av * av; o += (vv - (i == j)) * (vv - (i == j));
                    }
                for (uint32_t i = 0; i < n; ++i) wmax = std::max(wmax, std::fabs((double)w0[i] - wl[(size_t)b * n + i]));
                resid = std::max(resid, std::sqrt(r / na));
                orth = std::max(orth, std::sqrt(o / n));
                dw = std::max(dw, wmax / std::sqrt(na));
            }
            for (uint32_t b = 0; b < batch; ++b) bad += info[b];

            // the library's GPU path and the CPU, same input
            std::vector<float> w2((size_t)n * batch), v2(per * batch);
            double t_lib = tmin([&] { core::eigh(m, true, w2.data(), v2.data(), nullptr); }, 3);
            double t_par = tmin([&] {
                const unsigned chunks = std::min<unsigned>(18, batch);
                dispatch_apply(chunks, dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0), ^(size_t ch) {
                    BLASSetThreading(BLAS_THREADING_SINGLE_THREADED);
                    uint32_t s = (uint32_t)((uint64_t)batch * ch / chunks), e2 = (uint32_t)((uint64_t)batch * (ch + 1) / chunks);
                    core::Matrices mm{a + s * per, e2 - s, n, n};
                    core::detail::eigh_cpu(mm, true, w2.data() + (size_t)s * n, v2.data() + s * per, nullptr);
                });
            }, 3);
            double t_ser = tmin([&] { core::detail::eigh_cpu(m, true, w2.data(), v2.data(), nullptr); }, 2);
            std::printf("%4u %6u | %9.2f %9.2f %9.2f %9.2f | %6.1fx %6.2fx | %8.1e %8.1e %8.1e %4u\n", n, batch, t_proto,
                        t_lib, t_par, t_ser, t_lib / t_proto, t_par / t_proto, resid, orth, dw, bad);
            std::fflush(stdout);
            free(a); free(w); free(v); free(info);
        }
    }
}
