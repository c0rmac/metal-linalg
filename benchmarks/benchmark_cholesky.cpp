// Cholesky benchmark: each GPU kernel against the library's CPU path (LAPACK's
// spotrf through Accelerate, a batch spread over every core) and MLX's own
// mlx::core::linalg::cholesky (CPU only), over a grid of (n, batch).
//
//   benchmark_cholesky          the grid
//   benchmark_cholesky N B      one point: N x N, a batch of B

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include <mlx/mlx.h>
#include <mlx/linalg.h>

#include <metal_linalg/cholesky.h>
#include <metal_linalg/device.h>

using namespace mlx::core;
using namespace metal_linalg;

namespace {

// (M M^T + n I) / n: symmetric positive definite, well conditioned.
array random_spd(int batch, int n, unsigned seed = 7) {
    std::mt19937 r(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> m((size_t)batch * n * n);
    for (auto& x : m) x = dist(r);
    array a = array(m.begin(), {batch, n, n}, float32);
    a = add(divide(matmul(a, transpose(a, {0, 2, 1}), Device::cpu), array((float)n), Device::cpu),
            eye(n, float32), Device::cpu);
    if (batch == 1) a = reshape(a, {n, n});
    eval({a});
    return a;
}

double quantile(std::vector<double> v, double q) {
    std::sort(v.begin(), v.end());
    const double pos = q * (v.size() - 1);
    const size_t lo = (size_t)std::floor(pos), hi = (size_t)std::ceil(pos);
    return v[lo] + (v[hi] - v[lo]) * (pos - lo);
}

// Median of adaptive repeats: at least 5, or until ~300 ms is spent.
template <typename Fn>
double median_ms(Fn fn, double budget_ms = 300.0, int max_reps = 25) {
    for (int i = 0; i < 2; ++i) fn();
    std::vector<double> s;
    double total = 0.0;
    while ((int)s.size() < max_reps && (total < budget_ms || s.size() < 5)) {
        auto t0 = std::chrono::high_resolution_clock::now();
        fn();
        auto t1 = std::chrono::high_resolution_clock::now();
        s.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        total += s.back();
    }
    return quantile(s, 0.5);
}

// ||L L^T - A||_F / ||A||_F
float residual(const array& a, const array& l) {
    const int d = (int)l.ndim();
    std::vector<int> axes(d);
    for (int i = 0; i < d; ++i) axes[i] = i;
    std::swap(axes[d - 1], axes[d - 2]);
    array r = sqrt(sum(square(subtract(matmul(l, transpose(l, axes), Device::cpu), a, Device::cpu), Device::cpu),
                       Device::cpu), Device::cpu);
    array na = sqrt(sum(square(a, Device::cpu), Device::cpu), Device::cpu);
    eval({r, na});
    return r.item<float>() / std::max(na.item<float>(), 1e-30f);
}

using Fn = CholeskyResult (*)(const array&, bool);

struct Kernel {
    const char* name;
    Fn fn;
    bool (*fits)(int n, int batch);
};

const Kernel kKernels[] = {
    {"simd", detail::cholesky_simd, [](int n, int) { return n <= 32; }},
    {"threadgroup", detail::cholesky_threadgroup, [](int n, int b) { return n > 8 && (double)b * n * n * n <= 2e10; }},
    {"blocked", detail::cholesky_blocked, [](int n, int b) { return n >= 128 && b <= 64; }},
};

void point(int n, int batch) {
    const array a = random_spd(batch, n);
    std::printf("%6d %7d |", n, batch);
    double best = 1e30;
    float worst = 0.0f;
    for (const auto& k : kKernels) {
        if (!k.fits(n, batch)) { std::printf(" %10s", "--"); continue; }
        const double ms = median_ms([&] { CholeskyResult r = k.fn(a, false); eval({r.l}); });
        CholeskyResult r = k.fn(a, false);
        eval({r.l});
        worst = std::max(worst, residual(a, r.l));
        best = std::min(best, ms);
        std::printf(" %10.3f", ms);
    }
    const double cpu = median_ms([&] { CholeskyResult r = detail::cholesky_cpu(a, false); eval({r.l}); });
    const double mlx_ms = median_ms([&] { array l = linalg::cholesky(a, false, Device::cpu); eval({l}); });
    const double routed = median_ms([&] { array l = cholesky_accelerated(a); eval({l}); });
    std::printf(" | %10.3f %10.3f | %10.3f %-12s | %7.2fx %7.2fx | %8.1e\n", cpu, mlx_ms, routed,
                (std::string("(") + [&] {
                    switch (cholesky_backend(n, batch)) {
                        case CholeskyBackend::simd: return "simd";
                        case CholeskyBackend::threadgroup: return "threadgroup";
                        case CholeskyBackend::blocked: return "blocked";
                        case CholeskyBackend::cpu: break;
                    }
                    return "cpu";
                }() + ")").c_str(),
                cpu / best, mlx_ms / routed, worst);
    std::fflush(stdout);
}

} // namespace

int main(int argc, char** argv) {
    std::printf("\nCholesky on %s (%u GPU cores) vs the CPU path (Accelerate LAPACK, %u threads) and MLX (CPU)\n",
                device_name(), gpu_core_count(), cpu_threads());
    std::printf("median ms of >=5 runs after 2 warmups; best/CPU = CPU over the fastest GPU kernel, "
                "MLX/auto = MLX's over the routed call's; resid = max ||L L^T - A||_F / ||A||_F of the kernels\n\n");
    std::printf("%6s %7s | %10s %10s %10s | %10s %10s | %10s %-12s | %8s %8s | %8s\n", "n", "batch", "simd",
                "threadgroup", "blocked", "CPU", "MLX", "auto", "(route)", "best/CPU", "MLX/auto", "resid");
    std::printf("%s\n", std::string(132, '-').c_str());
    if (argc == 3) {
        point(std::atoi(argv[1]), std::atoi(argv[2]));
        return 0;
    }
    const struct { int n; std::vector<int> batches; } grid[] = {
        {8, {1, 256, 16384}},   {16, {1, 256, 16384}},   {32, {1, 256, 16384}}, {64, {1, 64, 4096}},
        {128, {1, 64, 1024}},   {256, {1, 16, 256}},     {512, {1, 4, 64}},     {1024, {1, 4, 16}},
        {2048, {1, 4}},         {4096, {1}},
    };
    for (const auto& g : grid)
        for (int b : g.batches) point(g.n, b);
    std::printf("\n");
    return 0;
}
