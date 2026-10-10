// LU benchmark: lu_factor, solve and inv, each backend (the CPU path, LAPACK
// through Accelerate a batch over every core; the blocked GPU path) against
// MLX's own mlx::core::linalg functions (CPU only), over a grid of (n, batch).
//
//   benchmark_lu          the grid
//   benchmark_lu N B [K]  one point: N x N, a batch of B, K right-hand sides for solve (default 1)

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
#include <metal_linalg/lu.h>

using namespace mlx::core;
using namespace metal_linalg;

namespace {

array random_matrices(int batch, int n, int cols, unsigned seed, float diag) {
    std::mt19937 r(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> v((size_t)batch * n * cols);
    for (auto& x : v) x = dist(r);
    if (diag != 0.0f)
        for (int b = 0; b < batch; ++b)
            for (int i = 0; i < std::min(n, cols); ++i) v[(size_t)b * n * cols + (size_t)i * cols + i] += diag;
    array a(v.begin(), {batch, n, cols}, float32);
    eval({a});
    return a;
}

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

void point(int n, int batch, int k) {
    const array a = random_matrices(batch, n, n, 7, 2.0f * std::sqrt((float)n));
    const array b = random_matrices(batch, n, k, 8, 0.0f);
    const bool gpu = n >= 64;   // the blocked path one matrix at a time; small ones are the CPU's
    auto t = [&](auto fn) { return median_ms(fn); };
    const double lc = t([&] { LuResult r = detail::lu_factor_cpu(a); eval({r.lu}); });
    const double lg = gpu ? t([&] { LuResult r = detail::lu_factor_blocked(a); eval({r.lu}); }) : 0;
    const double lm = t([&] { auto r = linalg::lu_factor(a, Device::cpu); eval({r.first}); });
    const double sc = t([&] { SolveResult r = detail::solve_cpu(a, b); eval({r.x}); });
    const double sg = gpu ? t([&] { SolveResult r = detail::solve_blocked(a, b); eval({r.x}); }) : 0;
    const double sm = t([&] { array x = linalg::solve(a, b, Device::cpu); eval({x}); });
    const double ic = t([&] { SolveResult r = detail::inv_cpu(a); eval({r.x}); });
    const double ig = gpu ? t([&] { SolveResult r = detail::inv_blocked(a); eval({r.x}); }) : 0;
    const double im = t([&] { array x = linalg::inv(a, Device::cpu); eval({x}); });
    auto g = [&](double v) { return gpu ? v : NAN; };
    std::printf("%5d %6d %4d | %8.3f %8.3f %8.3f %6.2fx | %8.3f %8.3f %8.3f %6.2fx | %8.3f %8.3f %8.3f %6.2fx | %s\n", n,
                batch, k, lc, g(lg), lm, gpu ? lc / lg : NAN, sc, g(sg), sm, gpu ? sc / sg : NAN, ic, g(ig), im,
                gpu ? ic / ig : NAN, lu_backend(n, batch) == LuBackend::blocked ? "blocked" : "cpu");
    std::fflush(stdout);
}

} // namespace

int main(int argc, char** argv) {
    std::printf("\nLU on %s (%u GPU cores): the CPU path (Accelerate LAPACK, %u threads), the blocked GPU path, "
                "and MLX's own (CPU); median ms\n\n", device_name(), gpu_core_count(), cpu_threads());
    std::printf("%5s %6s %4s | %8s %8s %8s %7s | %8s %8s %8s %7s | %8s %8s %8s %7s | %s\n", "n", "batch", "k",
                "lu cpu", "lu gpu", "lu mlx", "cpu/gpu", "slv cpu", "slv gpu", "slv mlx", "cpu/gpu", "inv cpu",
                "inv gpu", "inv mlx", "cpu/gpu", "route");
    std::printf("%s\n", std::string(150, '-').c_str());
    if (argc >= 3) {
        point(std::atoi(argv[1]), std::atoi(argv[2]), argc > 3 ? std::atoi(argv[3]) : 1);
        return 0;
    }
    const struct { int n; std::vector<int> batches; } grid[] = {
        {8, {1, 4096}}, {32, {1, 1024}}, {128, {1, 64}}, {512, {1, 16}}, {1024, {1, 4}},
        {2048, {1, 4}}, {3072, {1}}, {4096, {1}},
    };
    for (const auto& gp : grid)
        for (int b : gp.batches) point(gp.n, b, 1);
    point(2048, 1, 256);
    point(4096, 1, 512);
    std::printf("\n");
    return 0;
}
