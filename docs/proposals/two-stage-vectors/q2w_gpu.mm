// Prototype driver for q2w.metal (docs/proposals/two-stage-vectors.md): a
// random upper band of width 16 chased to bidiagonal (chase_record.cpp, which
// keeps the reflectors), each block's V and -T on the CPU, then Q2 applied
// to a random n x n on the GPU, X <- Q2 X (X column-major) and X <- Q2^T X
// (X row-major, as M <- M Q2 for M column-major), checked against one
// reflector at a time for n <= 2048.
//   q2w_gpu <n> [kernel ...]   (run where q2w.metallib is)
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
#include <cstring>
#include <random>
#include <string>
#include <vector>
namespace rec { extern float* L; extern float* Lt; extern float* R; extern float* Rt; extern long J, NB; }
namespace metal_linalg::detail {
void band_to_bidiagonal(uint32_t n, uint32_t nb, float* W, size_t ld, size_t ku, float* d, float* e, unsigned threads);
}
struct WParams { uint32_t n, rs, cs, pmax, down; };
static double now() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static long rlen(long n, long nb, long s, long j) { long a = s + 1 + j * nb, b = std::min(s + (j + 1) * nb, n - 1); return b >= a ? b - a + 1 : 0; }
int main(int argc, char** argv) {
    const long n = argc > 1 ? atol(argv[1]) : 1000, nb = 16;
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
    // blocks (G, p), p = G .. pmax, in that order; V (32 x 16) and -T (16 x 16), row-major
    const long pmax = (n - 2) / 16, nblocks = (pmax + 1) * (pmax + 2) / 2;
    std::vector<float> Vb(nblocks * 512, 0.0f), Tb(nblocks * 256, 0.0f);
    t0 = now();
    {
        float* Vp = Vb.data(); float* Tp = Tb.data(); const float* Lvp = Lv.data(); const float* Ltp = Lt.data();
        dispatch_apply(pmax + 1, DISPATCH_APPLY_AUTO, ^(size_t Gs) {
            const long G = (long)Gs, g0 = 16 * G, g1 = std::min(g0 + 15, n - 2);
            for (long p = G; p <= pmax; ++p) {
                const long j = p - G, b = G * (pmax + 1) - G * (G - 1) / 2 + j;
                float* V = Vp + b * 512; float* T = Tp + b * 256;
                float tau[16] = {};
                for (long c = 0; c < 16 && g0 + c <= g1; ++c) {
                    const long s = g0 + c, len = rlen(n, nb, s, j);
                    if (!len) continue;
                    tau[c] = Ltp[s * J + j];
                    for (long k = 0; k < len; ++k) V[(c + k) * 16 + c] = k == 0 ? 1.0f : Lvp[(s * J + j) * nb + k];
                }
                float Tt[16][16] = {};
                for (long c = 0; c < 16; ++c) {   // slarft forward columnwise
                    float z[16] = {};
                    for (long i = 0; i < c; ++i) { float s = 0; for (long r = 0; r < 32; ++r) s += V[r * 16 + i] * V[r * 16 + c]; z[i] = s; }
                    for (long i = 0; i < c; ++i) { float s = 0; for (long k = i; k < c; ++k) s += Tt[i][k] * z[k]; Tt[i][c] = -tau[c] * s; }
                    Tt[c][c] = tau[c];
                }
                for (long i = 0; i < 16; ++i) for (long c = 0; c < 16; ++c) T[i * 16 + c] = -Tt[i][c];
            }
        });
    }
    printf("blocks %ld, V and T on the CPU: %.1f ms\n", nblocks, now() - t0);
    const long ncols = (n + 31) / 32 * 32, ldx = ncols;   // X: n x ncols
    std::vector<float> X0((size_t)n * ncols); for (auto& x : X0) x = g(rng);   // column-major n x ncols, ld n
    std::vector<float> refU, refD;
    if (check) {
        refU = X0;   // Q2 X: reverse sweep order
        for (long s = n - 2; s >= 0; --s)
            for (long j = 0; j < J; ++j) {
                const long len = rlen(n, nb, s, j); if (!len) continue;
                const float tau = Lt[s * J + j]; if (tau == 0.0f) continue;
                const long r0 = s + 1 + j * nb; const float* u = &Lv[(s * J + j) * nb];
                for (long c = 0; c < ncols; ++c) {
                    float* x = &refU[c * n + r0];
                    float w = x[0]; for (long k = 1; k < len; ++k) w += u[k] * x[k];
                    w *= tau; x[0] -= w; for (long k = 1; k < len; ++k) x[k] -= w * u[k];
                }
            }
        refD = X0;   // Q2^T X: chase order
        for (long s = 0; s <= n - 2; ++s)
            for (long j = 0; j < J; ++j) {
                const long len = rlen(n, nb, s, j); if (!len) continue;
                const float tau = Lt[s * J + j]; if (tau == 0.0f) continue;
                const long r0 = s + 1 + j * nb; const float* u = &Lv[(s * J + j) * nb];
                for (long c = 0; c < ncols; ++c) {
                    float* x = &refD[c * n + r0];
                    float w = x[0]; for (long k = 1; k < len; ++k) w += u[k] * x[k];
                    w *= tau; x[0] -= w; for (long k = 1; k < len; ++k) x[k] -= w * u[k];
                }
            }
    }
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    NSError* err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithURL:[NSURL fileURLWithPath:@"q2w.metallib"] error:&err];
    if (!lib) { printf("no library: %s\n", err.localizedDescription.UTF8String); return 1; }
    std::vector<std::string> names;
    for (int i = 2; i < argc; ++i) names.push_back(argv[i]);
    if (names.empty()) names = {"q2w_2_4", "q2w_2_8", "q2w_4_4"};
    id<MTLBuffer> bV = [dev newBufferWithBytes:Vb.data() length:Vb.size() * 4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bT = [dev newBufferWithBytes:Tb.data() length:Tb.size() * 4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bX = [dev newBufferWithLength:(size_t)n * ncols * 4 options:MTLResourceStorageModeShared];
    id<MTLCommandQueue> q = [dev newCommandQueue];
    for (const auto& name : names) {
        id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:name.c_str()]];
        id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&err];
        if (!pso) { printf("%s: %s\n", name.c_str(), err.localizedDescription.UTF8String); continue; }
        const int CT = name[4] - '0', K = name[6] - '0', C = 8 * CT;
        for (int down = 0; down < 2; ++down) {
            float* Xg = (float*)bX.contents;
            auto fill = [&] {   // up: column-major (ld n); down: row-major n x ncols (ld ncols)
                if (!down) std::memcpy(Xg, X0.data(), X0.size() * 4);
                else for (long c = 0; c < ncols; ++c) for (long r = 0; r < n; ++r) Xg[r * ldx + c] = X0[c * n + r];
            };
            WParams p{(uint32_t)n, down ? (uint32_t)ldx : 1u, down ? 1u : (uint32_t)n, (uint32_t)pmax, (uint32_t)down};
            double best = 1e30;
            for (int rep = 0; rep < (check ? 1 : 3); ++rep) {
                fill();
                id<MTLCommandBuffer> cb = [q commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:pso];
                [enc setBuffer:bX offset:0 atIndex:0]; [enc setBuffer:bV offset:0 atIndex:1];
                [enc setBuffer:bT offset:0 atIndex:2]; [enc setBytes:&p length:sizeof p atIndex:3];
                [enc dispatchThreadgroups:MTLSizeMake(ncols / C, 1, 1) threadsPerThreadgroup:MTLSizeMake(32 * K, 1, 1)];
                [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
                if (cb.error) { printf("GPU error: %s\n", cb.error.localizedDescription.UTF8String); return 1; }
                best = std::min(best, (cb.GPUEndTime - cb.GPUStartTime) * 1e3);
            }
            printf("%s %s n=%ld: %.1f ms (max threads %lu)", name.c_str(), down ? "down" : "up  ", n, best,
                   (unsigned long)pso.maxTotalThreadsPerThreadgroup);
            if (check) {
                const std::vector<float>& ref = down ? refD : refU;
                double diff = 0, nrm = 0;
                for (long c = 0; c < ncols; ++c) for (long r = 0; r < n; ++r) {
                    const float v = down ? Xg[r * ldx + c] : Xg[c * n + r];
                    diff = std::max(diff, (double)std::fabs(v - ref[c * n + r])); nrm = std::max(nrm, (double)std::fabs(ref[c * n + r]));
                }
                printf("  max diff %.2e (max |X| %.2f)", diff, nrm);
            }
            printf("\n");
        }
    }
}
