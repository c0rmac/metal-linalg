// Prototype driver for q2.metal (docs/proposals/two-stage-vectors.md): a random
// upper band of width 16 chased to bidiagonal (chase_record.cpp, which keeps
// the reflectors), the blocks' T on the CPU, Q2 applied to a random n x n on
// the GPU, and for n <= 2048 checked against one reflector at a time.
//   q2_gpu <n>   (run where q2.metallib is)
#ifndef ACCELERATE_NEW_LAPACK
#define ACCELERATE_NEW_LAPACK
#endif
#include <Accelerate/Accelerate.h>
#import <Metal/Metal.h>
#include <dispatch/dispatch.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
namespace rec { extern float* L; extern float* Lt; extern float* R; extern float* Rt; extern long J, NB; }
namespace metal_linalg::detail {
void band_to_bidiagonal(uint32_t n, uint32_t nb, float* W, size_t ld, size_t ku, float* d, float* e, unsigned threads);
}
struct Q2Params { uint32_t n, nb, ib, J, nblocks, ldx; };
static double now() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static long rlen(long n, long nb, long s, long j) { long a = s + 1 + j * nb, b = std::min(s + (j + 1) * nb, n - 1); return b >= a ? b - a + 1 : 0; }
int main(int argc, char** argv) {
    const long n = argc > 1 ? atol(argv[1]) : 1000, nb = 16, ib = 16;
    const bool check = n <= 2048;
    std::mt19937 rng(5); std::normal_distribution<float> g;
    const long ld = 3 * nb + 1, ku = 2 * nb;
    std::vector<float> band(ld * n, 0.0f);
    for (long j = 0; j < n; ++j) for (long i = std::max(0L, j - nb); i <= j; ++i) band[j * ld + ku + i - j] = g(rng);
    const long J = (n + nb - 1) / nb + 2;
    std::vector<float> Lv(n * J * nb, 0.0f), Lt(n * J, 0.0f), Rv(n * J * nb, 0.0f), Rt(n * J, 0.0f);
    rec::L = Lv.data(); rec::Lt = Lt.data(); rec::R = Rv.data(); rec::Rt = Rt.data(); rec::J = J; rec::NB = nb;
    std::vector<float> d(n), e(n);
    double t0 = now();
    metal_linalg::detail::band_to_bidiagonal(n, nb, band.data(), ld, ku, d.data(), e.data(), 16);
    printf("chase with recording: %.1f ms\n", now() - t0);
    // blocks in application order, and their T
    std::vector<uint32_t> blk;
    const long ngroups = (n - 1 + ib - 1) / ib;
    for (long G = ngroups - 1; G >= 0; --G)
        for (long j = 0; G * ib + 1 + j * nb <= n - 1; ++j) { blk.push_back(G); blk.push_back(j); }
    const long nblocks = blk.size() / 2;
    std::vector<float> T(nblocks * 256, 0.0f);
    t0 = now();
    float* Tp = T.data(); const uint32_t* bk = blk.data(); const float* Lvp = Lv.data(); const float* Ltp = Lt.data();
    dispatch_apply(nblocks, DISPATCH_APPLY_AUTO, ^(size_t b) {
        const long G = bk[2 * b], j = bk[2 * b + 1], g0 = G * ib, g1 = std::min(g0 + ib - 1, n - 2);
        const long r0 = g0 + 1 + j * nb, rend = std::min(g1 + (j + 1) * nb, n - 1), rows = rend - r0 + 1;
        float V[32][16] = {}; float tau[16] = {};
        for (long c = 0; c < ib && g0 + c <= g1; ++c) {
            const long s = g0 + c, len = rlen(n, nb, s, j);
            if (!len) continue;
            tau[c] = Ltp[s * J + j];
            for (long k = 0; k < len; ++k) V[c + k][c] = k == 0 ? 1.0f : Lvp[(s * J + j) * nb + k];
        }
        float* Tb = Tp + b * 256;
        for (long c = 0; c < ib; ++c) {   // slarft forward columnwise: T(0:c, c) = -tau_c T(0:c, 0:c) V(:, 0:c)^T v_c
            float z[16] = {};
            for (long i = 0; i < c; ++i) { float s = 0; for (long r = 0; r < rows; ++r) s += V[r][i] * V[r][c]; z[i] = s; }
            for (long i = 0; i < c; ++i) { float s = 0; for (long k = i; k < c; ++k) s += Tb[i * 16 + k] * z[k]; Tb[i * 16 + c] = -tau[c] * s; }
            Tb[c * 16 + c] = tau[c];
        }
    });
    printf("blocks %ld, T on the CPU: %.1f ms\n", nblocks, now() - t0);
    // X: random n x n
    std::vector<float> X(n * n); for (auto& x : X) x = g(rng);
    std::vector<float> Xref = X;
    if (check) {   // one at a time, reverse sweep order
        for (long s = n - 2; s >= 0; --s)
            for (long j = 0; j < J; ++j) {
                const long len = rlen(n, nb, s, j); if (!len) continue;
                const float tau = Lt[s * J + j]; if (tau == 0.0f) continue;
                const long r0 = s + 1 + j * nb; const float* u = &Lv[(s * J + j) * nb];
                for (long c = 0; c < n; ++c) {
                    float* x = &Xref[c * n + r0];
                    float w = x[0]; for (long k = 1; k < len; ++k) w += u[k] * x[k];
                    w *= tau; x[0] -= w; for (long k = 1; k < len; ++k) x[k] -= w * u[k];
                }
            }
    }
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    id<MTLLibrary> lib = [dev newLibraryWithURL:[NSURL fileURLWithPath:@"q2.metallib"] error:nil];
    id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"q2_apply"] error:nil];
    const long ldx = n;
    std::vector<float> Xp((size_t)ldx * n, 0.0f);
    for (long c = 0; c < n; ++c) std::copy(&X[c * n], &X[c * n] + n, &Xp[c * ldx]);
    id<MTLBuffer> bX = [dev newBufferWithBytes:Xp.data() length:Xp.size() * 4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bL = [dev newBufferWithBytes:Lv.data() length:Lv.size() * 4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bT = [dev newBufferWithBytes:T.data() length:T.size() * 4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bB = [dev newBufferWithBytes:blk.data() length:blk.size() * 4 options:MTLResourceStorageModeShared];
    Q2Params p{(uint32_t)n, (uint32_t)nb, (uint32_t)ib, (uint32_t)J, (uint32_t)nblocks, (uint32_t)ldx};
    id<MTLCommandQueue> q = [dev newCommandQueue];
    double best = 1e30;
    for (int rep = 0; rep < (check ? 1 : 3); ++rep) {
        memcpy(bX.contents, Xp.data(), Xp.size() * 4);
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:bX offset:0 atIndex:0]; [enc setBuffer:bL offset:0 atIndex:1]; [enc setBuffer:bT offset:0 atIndex:2];
        [enc setBuffer:bB offset:0 atIndex:3]; [enc setBytes:&p length:sizeof p atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(n / 32, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
        if (cb.error) { printf("GPU error: %s\n", cb.error.localizedDescription.UTF8String); return 1; }
        best = std::min(best, (cb.GPUEndTime - cb.GPUStartTime) * 1e3);
    }
    printf("n=%ld: Q2 applied on the GPU in %.1f ms\n", n, best);
    if (check) {
        const float* r = (const float*)bX.contents;
        double diff = 0, nrm = 0;
        for (long c = 0; c < n; ++c) for (long i = 0; i < n; ++i) { diff = std::max(diff, (double)std::fabs(r[c * ldx + i] - Xref[c * n + i])); nrm = std::max(nrm, (double)std::fabs(Xref[c * n + i])); }
        printf("  vs one at a time on the CPU: max diff %.2e (max |X| %.2f)\n", diff, nrm);
    }
}
