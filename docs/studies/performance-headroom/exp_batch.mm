// Experiment 1+2: is the CPU baseline using the machine, and does splitting a
// batch between the CPU and the GPU at the same time pay?
//
//   serial    the library's CPU path as shipped (one matrix at a time)
//   par       the same LAPACK calls, batch split over all CPU cores
//             (dispatch_apply, BLAS single-threaded per worker)
//   gpu       the library's GPU backend for that shape (policy forced to GPU)
//   hybrid    GPU on the first g matrices while the CPU workers take the rest,
//             best g of a small search around the throughput-balanced split
#include <metal_linalg/core.h>
#include <Accelerate/Accelerate.h>
#include <vecLib/thread_api.h>
#include <dispatch/dispatch.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace metal_linalg;
using core::Matrices;
using clk = std::chrono::steady_clock;

static double now_ms() { return std::chrono::duration<double, std::milli>(clk::now().time_since_epoch()).count(); }

static double time_min(const std::function<void()>& f, int reps) {
    // Warm up for 250 ms so the GPU and CPU clocks have ramped, then the min of
    // at least `reps` (7 when cheap) runs.
    double w0 = now_ms();
    do { f(); } while (now_ms() - w0 < 250.0);
    reps = std::max(reps, 7);
    double best = 1e30;
    for (int r = 0; r < reps; ++r) {
        double t0 = now_ms();
        f();
        best = std::min(best, now_ms() - t0);
    }
    return best;
}

static float* page_alloc(size_t floats) {
    void* p = nullptr;
    posix_memalign(&p, 16384, std::max<size_t>(16384, (floats * 4 + 16383) / 16384 * 16384));
    return static_cast<float*>(p);
}

enum class Op { eigh, svd, qr };

struct Problem {
    Op op; uint32_t m, n, batch;
    float *a, *o1, *o2, *o3;
};

static size_t out1(const Problem& p) { uint32_t k = std::min(p.m, p.n); return p.op == Op::eigh ? p.n : p.op == Op::svd ? (size_t)p.m * k : (size_t)p.m * k; }
static size_t out2(const Problem& p) { uint32_t k = std::min(p.m, p.n); return p.op == Op::eigh ? (size_t)p.n * p.n : p.op == Op::svd ? k : (size_t)k * p.n; }
static size_t out3(const Problem& p) { uint32_t k = std::min(p.m, p.n); return p.op == Op::svd ? (size_t)k * p.n : 0; }

// Run matrices [b0, b1) of the problem through `cpu` (true) or the library's GPU policy.
static void run_range(const Problem& p, uint32_t b0, uint32_t b1, bool cpu) {
    if (b1 <= b0) return;
    const size_t per = (size_t)p.m * p.n;
    Matrices a{p.a + b0 * per, b1 - b0, p.m, p.n};
    float* o1 = p.o1 + b0 * out1(p);
    float* o2 = p.o2 + b0 * out2(p);
    float* o3 = p.o3 ? p.o3 + b0 * out3(p) : nullptr;
    switch (p.op) {
        case Op::eigh:
            if (cpu) core::detail::eigh_cpu(a, true, o1, o2, nullptr);
            else     core::eigh(a, true, o1, o2, nullptr);
            break;
        case Op::svd:
            if (cpu) core::detail::svd_cpu(a, o1, o2, o3, nullptr);
            else     core::svd(a, o1, o2, o3, nullptr);
            break;
        case Op::qr:
            if (cpu) core::detail::qr_cpu(a, o1, o2);
            else     core::qr(a, o1, o2);
            break;
    }
}

// CPU over [b0, b1), split into `workers` chunks run concurrently.
static void run_cpu_parallel(const Problem& p, uint32_t b0, uint32_t b1, unsigned workers) {
    const uint32_t cnt = b1 - b0;
    if (cnt == 0) return;
    const unsigned chunks = std::min<unsigned>(workers, cnt);
    dispatch_apply(chunks, dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0), ^(size_t c) {
        BLASSetThreading(BLAS_THREADING_SINGLE_THREADED);
        uint32_t s = b0 + (uint32_t)((uint64_t)cnt * c / chunks);
        uint32_t e = b0 + (uint32_t)((uint64_t)cnt * (c + 1) / chunks);
        run_range(p, s, e, true);
    });
}

static void force_gpu() {
    auto e = eigh_policy();
    e.gpu_max_n = kEighNoLimit; e.gpu_min_batch_times_n = 0; e.gpu_min_batch = 1; e.tridiag_min_n = 0;
    set_eigh_policy(e);
    auto s = svd_policy();
    s.gpu_max_k = kSvdNoLimit; s.gpu_min_batch_times_k = 0; s.gpu_min_batch = 1; s.bidiag_min_k = 0;
    set_svd_policy(s);
    auto q = qr_policy();
    q.gpu_max_k = kQrNoLimit; q.gpu_min_batch_times_k = 0; q.gpu_min_batch = 1;
    set_qr_policy(q);
}

int main(int argc, char** argv) {
    const unsigned ncpu = std::thread::hardware_concurrency();
    const unsigned workers = argc > 1 ? (unsigned)atoi(argv[1]) : ncpu;
    force_gpu();
    struct Case { Op op; uint32_t m, n, batch; };
    std::vector<Case> cases = {
        {Op::eigh, 8, 8, 4096},     {Op::eigh, 16, 16, 4096},  {Op::eigh, 32, 32, 256},
        {Op::eigh, 32, 32, 4096},   {Op::eigh, 64, 64, 256},   {Op::eigh, 64, 64, 2048},
        {Op::eigh, 128, 128, 256},  {Op::eigh, 256, 256, 64},  {Op::eigh, 512, 512, 16},
        {Op::svd, 8, 8, 4096},      {Op::svd, 32, 32, 4096},   {Op::svd, 64, 64, 256},
        {Op::svd, 128, 128, 64},    {Op::svd, 256, 256, 16},   {Op::svd, 512, 512, 4},
        {Op::svd, 1024, 64, 64},    {Op::svd, 2048, 256, 16},
        {Op::qr, 16, 16, 10000},    {Op::qr, 64, 64, 1000},    {Op::qr, 256, 128, 1000},
        {Op::qr, 512, 512, 32},     {Op::qr, 1024, 512, 16},
    };
    if (argc > 2) {   // a single case: op m n batch
        cases.clear();
        std::string o = argv[2];
        cases.push_back({o == "eigh" ? Op::eigh : o == "svd" ? Op::svd : Op::qr,
                         (uint32_t)atoi(argv[3]), (uint32_t)atoi(argv[4]), (uint32_t)atoi(argv[5])});
    }
    std::printf("cpu workers %u (hardware %u)\n", workers, ncpu);
    std::printf("%-5s %11s %6s | %9s %9s %9s %9s | %6s %6s %6s | %s\n", "op", "shape", "batch",
                "serial", "par", "gpu", "hybrid", "par/s", "gpu/p", "hyb/b", "split");
    std::mt19937 rng(7);
    std::normal_distribution<float> nd;
    for (const Case& c : cases) {
        Problem p{c.op, c.m, c.n, c.batch};
        const size_t per = (size_t)c.m * c.n;
        p.a = page_alloc(per * c.batch);
        for (size_t i = 0; i < per * c.batch; ++i) p.a[i] = nd(rng);
        if (c.op == Op::eigh)   // symmetrise
            for (uint32_t b = 0; b < c.batch; ++b)
                for (uint32_t i = 0; i < c.n; ++i)
                    for (uint32_t j = 0; j < i; ++j) p.a[b * per + j * c.n + i] = p.a[b * per + i * c.n + j];
        p.o1 = page_alloc(out1(p) * c.batch);
        p.o2 = page_alloc(out2(p) * c.batch);
        p.o3 = out3(p) ? page_alloc(out3(p) * c.batch) : nullptr;

        const double work = (double)c.batch * c.m * c.n * std::min(c.m, c.n);
        const int reps = work > 4e9 ? 2 : work > 5e8 ? 3 : 5;
        double t_ser = time_min([&] { run_range(p, 0, c.batch, true); }, reps);
        double t_par = time_min([&] { run_cpu_parallel(p, 0, c.batch, workers); }, reps);
        double t_gpu = time_min([&] { run_range(p, 0, c.batch, false); }, reps);

        // Hybrid: GPU takes g matrices on its own thread, CPU workers the rest.
        double best = 1e30; uint32_t best_g = 0;
        const double f0 = t_par / (t_par + t_gpu);   // throughput-balanced share for the GPU
        std::vector<double> fracs = {f0 * 0.7, f0 * 0.85, f0, std::min(1.0, f0 * 1.15), std::min(1.0, f0 * 1.3)};
        for (double f : fracs) {
            uint32_t g = (uint32_t)std::lround(f * c.batch);
            g = std::min(g, c.batch);
            if (g == 0 || g == c.batch) continue;
            double t = time_min([&] {
                std::thread gt([&] { run_range(p, 0, g, false); });
                run_cpu_parallel(p, g, c.batch, workers);
                gt.join();
            }, std::max(2, reps - 1));
            if (t < best) { best = t; best_g = g; }
        }
        const char* name = c.op == Op::eigh ? "eigh" : c.op == Op::svd ? "svd" : "qr";
        char shape[32];
        std::snprintf(shape, sizeof shape, "%ux%u", c.m, c.n);
        const double best_alone = std::min(t_par, t_gpu);
        std::printf("%-5s %11s %6u | %9.2f %9.2f %9.2f %9.2f | %5.1fx %5.2fx %5.2fx | gpu %u/%u\n", name, shape,
                    c.batch, t_ser, t_par, t_gpu, best < 1e29 ? best : NAN, t_ser / t_par, t_par / t_gpu,
                    best < 1e29 ? best_alone / best : NAN, best_g, c.batch);
        std::fflush(stdout);
        free(p.a); free(p.o1); free(p.o2); free(p.o3);
    }
}
