// Triangular solve benchmark: the CPU path (BLAS strsm through Accelerate, a
// batch over every core), the blocked GPU path, and MLX's own
// mlx::core::linalg::solve_triangular (CPU only), over (n, k, batch).
//
//   benchmark_trsm            the grid
//   benchmark_trsm N K [B]    one point: N x N lower triangles, K right-hand sides, a batch of B

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include <mlx/linalg.h>
#include <mlx/mlx.h>

#include <metal_linalg/device.h>
#include <metal_linalg/triangular.h>

using namespace mlx::core;
using namespace metal_linalg;

namespace {

template <typename Fn>
double median_ms(Fn fn, double budget_ms = 300.0, int max_reps = 15) {
    for (int i = 0; i < 2; ++i) fn();
    std::vector<double> s;
    double total = 0.0;
    while ((int)s.size() < max_reps && (total < budget_ms || s.size() < 3)) {
        auto t0 = std::chrono::high_resolution_clock::now();
        fn();
        s.push_back(std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t0).count());
        total += s.back();
    }
    std::sort(s.begin(), s.end());
    return s[s.size() / 2];
}

void point(int n, int k, int batch) {
    std::mt19937 g(7);
    std::normal_distribution<float> nd;
    std::vector<float> av((size_t)batch * n * n, 0.0f), bv((size_t)batch * n * k);
    for (int b = 0; b < batch; ++b)
        for (int i = 0; i < n; ++i)
            for (int j = 0; j <= i; ++j) av[(size_t)b * n * n + (size_t)i * n + j] = i == j ? 2.0f : nd(g) / std::sqrt((float)n);
    for (auto& x : bv) x = nd(g);
    const array A(av.begin(), {batch, n, n}, float32), B(bv.begin(), {batch, n, k}, float32);
    eval({A, B});
    const double c = median_ms([&] { array x = detail::solve_triangular_cpu(A, B); eval({x}); });
    const double gpu = median_ms([&] { array x = detail::solve_triangular_blocked(A, B); eval({x}); });
    const double m = median_ms([&] { array x = linalg::solve_triangular(A, B, false, Device::cpu); eval({x}); });
    std::printf("%5d %5d %5d | %9.3f %9.3f %9.3f | %7.2fx %7.2fx | %s\n", n, k, batch, c, gpu, m, c / gpu,
                m / std::min(c, gpu), trsm_backend(n, k, batch) == TrsmBackend::blocked ? "blocked" : "cpu");
    std::fflush(stdout);
}

} // namespace

int main(int argc, char** argv) {
    std::printf("\nTriangular solve on %s (%u GPU cores): the CPU path (Accelerate strsm, %u threads), the blocked "
                "GPU path, MLX's own (CPU); median ms\n\n", device_name(), gpu_core_count(), cpu_threads());
    std::printf("%5s %5s %5s | %9s %9s %9s | %8s %8s | %s\n", "n", "k", "batch", "cpu", "gpu", "mlx", "cpu/gpu",
                "mlx/best", "route");
    std::printf("%s\n", std::string(82, '-').c_str());
    if (argc >= 3) {
        point(std::atoi(argv[1]), std::atoi(argv[2]), argc > 3 ? std::atoi(argv[3]) : 1);
        return 0;
    }
    const int grid[][3] = {{256, 256, 1}, {512, 512, 1}, {1024, 1, 1}, {1024, 64, 1}, {1024, 1024, 1},
                           {2048, 1, 1}, {2048, 256, 1}, {2048, 2048, 1}, {4096, 1, 1}, {4096, 512, 1},
                           {4096, 4096, 1}, {512, 512, 16}, {1024, 1024, 4}};
    for (const auto& p : grid) point(p[0], p[1], p[2]);
    std::printf("\n");
    return 0;
}
