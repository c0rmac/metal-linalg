// SVD benchmark: the four Metal backends against a thin SVD on the CPU.
//
//   benchmark_svd                    all five over a grid of shapes and batches
//   benchmark_svd <batch> <M> <N>    one point
//   benchmark_svd --tune             simdgroups per matrix, for svd.mm
//
// The GPU columns are the whole-matrix kernel (jacobi) and the block kernel
// (block) on the matrix itself, then each after this library's QR on tall
// input (qr, qr+block). Both kernels are pinned, so no column follows the
// routing policy; the crossover between them is what the columns show.
//
// The CPU column is a fair one. mlx::core::linalg::svd only offers the
// full-size factors, whose U is M x M, which for a tall matrix is most of the
// cost and none of what was asked for. The CPU path here (detail::svd_cpu)
// reduces a tall matrix with a thin QR first, as LAPACK does for its own thin
// SVD, so every column computes the same thing. "CPU full" is what calling
// MLX's svd directly would cost, for reference.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include <mlx/mlx.h>
#include <mlx/linalg.h>

#include <metal_linalg/svd.h>

using namespace mlx::core;
using namespace metal_linalg;

namespace {

array random_matrix(int batch, int M, int N, unsigned seed = 7) {
    std::mt19937 r(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> data((size_t)batch * M * N);
    for (auto& v : data) v = dist(r);
    if (batch == 1) return array(data.begin(), {M, N}, float32);
    return array(data.begin(), {batch, M, N}, float32);
}

double quantile(std::vector<double> v, double q) {
    std::sort(v.begin(), v.end());
    const double pos = q * (v.size() - 1);
    const size_t lo = (size_t)std::floor(pos), hi = (size_t)std::ceil(pos);
    return v[lo] + (v[hi] - v[lo]) * (pos - lo);
}

template <typename Fn>
double median_ms(Fn fn, double budget_ms = 300.0, int max_reps = 25) {
    for (int i = 0; i < 2; ++i) fn();
    std::vector<double> s;
    double total = 0.0;
    int min_reps = 5;
    while ((int)s.size() < max_reps && (total < budget_ms || (int)s.size() < min_reps)) {
        auto t0 = std::chrono::high_resolution_clock::now();
        fn();
        auto t1 = std::chrono::high_resolution_clock::now();
        s.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        total += s.back();
        if (s.back() > 300.0) min_reps = 3;
    }
    return quantile(s, 0.5);
}

float reconstruction(const array& A, const SvdResult& r) {
    array US = multiply(r.U, expand_dims(r.S, -2));
    array e  = sqrt(sum(square(subtract(matmul(US, r.Vt), A))));
    array nA = sqrt(sum(square(A)));
    eval({e, nA});
    return e.item<float>() / std::max(nA.item<float>(), 1e-30f);
}

unsigned max_sweeps(const array& info) {
    array flat = reshape(info, {-1});
    eval({flat});
    unsigned m = 0;
    for (int i = 0; i < (int)flat.size(); ++i)
        m = std::max(m, detail::svd_sweeps(flat.data<uint32_t>()[i]));
    return m;
}

// A GPU column: 0 ms means not run.
struct Row {
    double jacobi_ms, block_ms, qr_ms, qr_block_ms, cpu_ms, cpu_full_ms;
    float recon;          // of the fastest GPU backend
    unsigned sweeps;      // of the whole-matrix kernel, or the block kernel if that alone ran
    const char* best;     // name of the fastest GPU backend
};

// The whole-matrix kernel is a single threadgroup per matrix and takes
// seconds on one large matrix; it is not timed above this k, and the block
// kernel not below the smallest k it is meant for.
constexpr int kJacobiMaxK = 512;
constexpr int kBlockMinK  = 32;

template <typename Fn>
double time_svd(Fn fn) {
    return median_ms([&] { SvdResult s = fn(); eval({s.U, s.S, s.Vt}); });
}

SvdOptions pinned(SvdOptions::Kernel k) {
    SvdOptions o;
    o.kernel = k;
    return o;
}

Row bench_point(int batch, int M, int N) {
    array A = random_matrix(batch, M, N);
    eval({A});
    const int k = std::min(M, N);
    const bool tall = std::max(M, N) >= 2 * k;
    Row r{};
    struct Gpu { const char* name; bool run; std::function<SvdResult()> fn; };
    const std::vector<Gpu> gpus = {
        {"jacobi",   k <= kJacobiMaxK,         [&] { return detail::svd_jacobi(A, true, {}); }},
        {"block",    k >= kBlockMinK,          [&] { return detail::svd_block_jacobi(A, true, {}); }},
        {"qr",       tall && k <= kJacobiMaxK, [&] { return detail::svd_qr_jacobi(A, true, pinned(SvdOptions::Kernel::jacobi)); }},
        {"qr+block", tall && k >= kBlockMinK,  [&] { return detail::svd_qr_jacobi(A, true, pinned(SvdOptions::Kernel::block)); }},
    };
    double* slots[] = {&r.jacobi_ms, &r.block_ms, &r.qr_ms, &r.qr_block_ms};
    double best = INFINITY;
    for (size_t i = 0; i < gpus.size(); ++i) {
        if (!gpus[i].run) continue;
        *slots[i] = time_svd(gpus[i].fn);
        if (*slots[i] < best) { best = *slots[i]; r.best = gpus[i].name; }
    }
    for (size_t i = 0; i < gpus.size(); ++i) {
        if (!gpus[i].run || std::string(gpus[i].name) != r.best) continue;
        SvdResult s = gpus[i].fn();
        eval({s.U, s.S, s.Vt, s.info});
        r.recon = reconstruction(A, s);
    }
    {
        const bool whole = k <= kJacobiMaxK;
        SvdResult s = whole ? detail::svd_jacobi(A, true, {}) : detail::svd_block_jacobi(A, true, {});
        eval({s.info});
        r.sweeps = max_sweeps(s.info);
    }
    r.cpu_ms = time_svd([&] { return detail::svd_cpu(A, true); });
    r.cpu_full_ms = median_ms([&] {
        std::vector<array> s = linalg::svd(A, true, Device::cpu);
        eval(s);
    });
    return r;
}

void print_header() {
    std::printf("%6s %6s %7s | %10s %10s | %10s %10s | %10s %10s | %8s %-8s | %8s %6s\n",
                "M", "N", "batch", "jacobi ms", "block ms", "qr ms", "qr+block", "CPU ms", "CPU full",
                "best/CPU", "best", "recon", "sweeps");
    std::printf("%s\n", std::string(136, '-').c_str());
}

void print_ms(double ms) {
    if (ms > 0) std::printf("%10.3f ", ms); else std::printf("%10s ", "--");
}

void print_row(int batch, int M, int N, const Row& r) {
    double best = INFINITY;
    for (double ms : {r.jacobi_ms, r.block_ms, r.qr_ms, r.qr_block_ms})
        if (ms > 0) best = std::min(best, ms);
    std::printf("%6d %6d %7d | ", M, N, batch);
    print_ms(r.jacobi_ms); print_ms(r.block_ms); std::printf("| ");
    print_ms(r.qr_ms);     print_ms(r.qr_block_ms); std::printf("| ");
    std::printf("%10.3f %10.3f | %7.2fx %-8s | %8.1e %6u\n", r.cpu_ms, r.cpu_full_ms,
                r.cpu_ms / best, r.best ? r.best : "--", r.recon, r.sweeps);
    std::fflush(stdout);
}

} // namespace

int main(int argc, char** argv) {
    set_default_device(Device::gpu);
    set_cache_limit(0);

    std::printf("\nThin SVD on an Apple GPU (one-sided Jacobi: whole-matrix and block kernels, each direct\n"
                "and QR-preconditioned) vs the CPU. median of >=5 runs after 2 warmups; recon = ||A - U S Vt||_F / ||A||_F\n"
                "of the fastest GPU backend; sweeps = the whole-matrix kernel's. CPU = thin factors through\n"
                "MLX / Accelerate, QR first when tall; CPU full = MLX svd as is. -- = not run: the whole-matrix\n"
                "kernel above k = %d, the block kernel below k = %d, the QR paths unless M >= 2N.\n\n",
                kJacobiMaxK, kBlockMinK);

    if (argc == 4) {
        const int batch = std::atoi(argv[1]), M = std::atoi(argv[2]), N = std::atoi(argv[3]);
        print_header();
        print_row(batch, M, N, bench_point(batch, M, N));
        return 0;
    }
    if (argc == 2 && std::string(argv[1]) == "--tune") {
        std::printf("[ simdgroups per matrix ]  median ms, full SVD; 'pairs' is pairs per round\n\n");
        std::printf("%6s %6s %7s %6s |", "M", "N", "batch", "pairs");
        for (int g : {1, 2, 4, 8, 16, 32}) std::printf(" %9s", ("sg=" + std::to_string(g)).c_str());
        std::printf("\n");
        struct P { int M, N, batch; };
        for (const P& p : std::vector<P>{{8, 8, 1}, {8, 8, 256}, {8, 8, 4096}, {16, 16, 1}, {16, 16, 256},
                                         {16, 16, 4096}, {32, 32, 1}, {32, 32, 16}, {32, 32, 256},
                                         {32, 32, 4096}, {64, 64, 1}, {64, 64, 16}, {64, 64, 256},
                                         {64, 64, 2048}, {128, 128, 1}, {128, 128, 16}, {128, 128, 128},
                                         {256, 256, 1}, {256, 256, 16}, {256, 32, 256}, {1024, 64, 16}}) {
            array A = random_matrix(p.batch, p.M, p.N);
            eval({A});
            const int pairs = (p.N + 1) / 2;
            std::printf("%6d %6d %7d %6d |", p.M, p.N, p.batch, pairs);
            for (int g : {1, 2, 4, 8, 16, 32}) {
                if (g > pairs) { std::printf(" %9s", "--"); continue; }
                SvdOptions o; o.simdgroups = g;
                const double ms = median_ms([&] {
                    SvdResult s = detail::svd_jacobi(A, true, o);
                    eval({s.U, s.S, s.Vt});
                });
                std::printf(" %9.3f", ms);
                std::fflush(stdout);
            }
            std::printf("\n");
        }
        return 0;
    }
    if (argc != 1) { std::fprintf(stderr, "usage: %s [--tune | <batch> <M> <N>]\n", argv[0]); return 2; }

    struct Cfg { int M, N, max_batch; };
    const std::vector<Cfg> cfgs = {
        {4, 4, 0}, {8, 8, 0}, {16, 16, 0}, {32, 32, 0}, {64, 64, 4096}, {128, 128, 256},
        {192, 192, 64}, {256, 256, 16}, {384, 384, 16}, {512, 512, 16}, {1024, 1024, 1},
        {64, 8, 0}, {128, 32, 1024}, {256, 32, 1024}, {1024, 16, 256}, {1024, 64, 64}, {2048, 64, 16},
        {1024, 256, 16}, {2048, 512, 1},
    };
    print_header();
    for (const auto& c : cfgs) {
        for (int b : {1, 16, 256, 4096}) {
            if (c.max_batch && b > c.max_batch) continue;
            std::printf("  running %dx%d batch=%d ...\r", c.M, c.N, b);
            std::fflush(stdout);
            print_row(b, c.M, c.N, bench_point(b, c.M, c.N));
        }
    }
    std::printf("\n");
    return 0;
}
