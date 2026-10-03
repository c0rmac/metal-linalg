// Symmetric eigensolver benchmark: the GPU backends against the library's CPU
// path (LAPACK through Accelerate, a batch spread over every core).
//
//   benchmark_eigh              GPU vs CPU over a grid of (N, batch)
//   benchmark_eigh --tune [k]   the tuning tables (all, or just table k) that
//                               set the constants at the top of eigh.mm

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include <mlx/mlx.h>
#include <mlx/linalg.h>

#include <metal_linalg/device.h>
#include <metal_linalg/eigh.h>

using namespace mlx::core;
using namespace metal_linalg;

namespace {

array random_symmetric(int batch, int n, unsigned seed = 7) {
    std::mt19937 r(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> data((size_t)batch * n * n);
    for (int b = 0; b < batch; ++b) {
        float* m = data.data() + (size_t)b * n * n;
        for (int i = 0; i < n; ++i)
            for (int j = 0; j <= i; ++j) {
                const float x = dist(r);
                m[i * n + j] = x;
                m[j * n + i] = x;
            }
    }
    if (batch == 1) return array(data.begin(), {n, n}, float32);
    return array(data.begin(), {batch, n, n}, float32);
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

array transpose_last_two(const array& x) {
    std::vector<int> axes(x.ndim());
    for (size_t i = 0; i < axes.size(); ++i) axes[i] = (int)i;
    std::swap(axes[axes.size() - 1], axes[axes.size() - 2]);
    return transpose(x, axes);
}

float residual(const array& A, const array& w, const array& V) {
    array VL = multiply(V, expand_dims(w, -2));
    array r  = sqrt(sum(square(subtract(matmul(A, V), VL))));
    array nA = sqrt(sum(square(A)));
    eval({r, nA});
    return r.item<float>() / std::max(nA.item<float>(), 1e-30f);
}

unsigned max_sweeps(const array& info) {
    array flat = reshape(info, {-1});
    eval({flat});
    unsigned m = 0;
    for (int i = 0; i < flat.size(); ++i)
        m = std::max(m, detail::eigh_sweeps(flat.data<uint32_t>()[i]));
    return m;
}

struct Row {
    int n, batch;
    double scalar_ms, block_ms, ql_ms, cpu_ms;
    float resid_scalar, resid_block, resid_ql;
    unsigned sweeps_scalar, sweeps_block;
};

// Whole-matrix kernel, its automatic simd/threadgroup choice.
EighOptions scalar_opts() { EighOptions o; o.mode = EighOptions::Mode::threadgroup; return o; }
EighOptions block_opts()  { EighOptions o; o.mode = EighOptions::Mode::block;       return o; }

double time_opts(const array& A, const EighOptions& o) {
    if (o.mode == EighOptions::Mode::block)
        return median_ms([&] { EighResult r = detail::eigh_block_jacobi(A, true, true, o); eval({r.eigenvalues}); });
    return median_ms([&] { EighResult r = detail::eigh_jacobi(A, true, true, o); eval({r.eigenvalues}); });
}

Row bench_point(int n, int batch, bool with_scalar, bool with_block, bool with_ql) {
    array A = random_symmetric(batch, n);
    eval({A});

    Row r{n, batch, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    if (with_scalar) {
        EighOptions o; o.mode = EighOptions::Mode::automatic;   // simd or threadgroup by N
        r.scalar_ms = time_opts(A, o);
        EighResult res = detail::eigh_jacobi(A, true, true, o);
        eval({res.eigenvalues, res.eigenvectors, res.info});
        r.resid_scalar  = residual(A, res.eigenvalues, res.eigenvectors);
        r.sweeps_scalar = max_sweeps(res.info);
    }
    if (with_block) {
        r.block_ms = time_opts(A, block_opts());
        EighResult res = detail::eigh_block_jacobi(A, true, true, block_opts());
        eval({res.eigenvalues, res.eigenvectors, res.info});
        r.resid_block  = residual(A, res.eigenvalues, res.eigenvectors);
        r.sweeps_block = max_sweeps(res.info);
    }
    if (with_ql) {
        r.ql_ms = median_ms([&] { EighResult q = detail::eigh_ql(A, true, true); eval({q.eigenvalues}); });
        EighResult res = detail::eigh_ql(A, true, true);
        eval({res.eigenvalues, res.eigenvectors});
        r.resid_ql = residual(A, res.eigenvalues, res.eigenvectors);
    }
    r.cpu_ms = median_ms([&] { EighResult c = detail::eigh_cpu(A, true, true); eval({c.eigenvalues}); });
    return r;
}

void run_main() {
    // max_batch caps per backend: the whole-matrix kernel is hopeless at
    // large N and large batch (minutes), the block backend at tiny N is
    // pointless.
    struct Cfg { int n; int max_batch_scalar; int max_batch_block; };
    const std::vector<Cfg> cfgs = {
        {4,    0,  -1}, {8,    0, -1}, {16,   0,  -1}, {32,   0, 256}, {64, 4096, 4096},
        {128, 1024, 1024}, {256, 64, 256}, {512, 16, 64}, {1024, -1, 4},
    };
    const std::vector<int> batches = {1, 16, 256, 4096};

    const int ql_max = (int)detail::eigh_ql_max_n();
    std::printf("\nSymmetric eigensolver on an Apple GPU vs the CPU path (Accelerate LAPACK, %u threads)\n",
                cpu_threads());
    std::printf("scalar = whole-matrix Jacobi (one threadgroup per matrix), block = block Jacobi,\n"
                "ql = tridiagonalization and QL (one threadgroup per matrix, N <= %d); x/CPU = CPU time over x's\n",
                ql_max);
    std::printf("median of >=5 runs after 2 warmups; resid = ||A V - V diag(w)||_F / ||A||_F\n");
    std::printf("automatic dispatch: simd for N <= %u, block from N >= %u\n\n",
                detail::eigh_simd_max_n(), detail::eigh_block_min_n());

    std::printf("%6s %7s | %10s %10s %10s %10s | %8s %8s %8s | %9s %9s %9s | %5s %5s\n",
                "N", "batch", "scalar ms", "block ms", "ql ms", "CPU ms", "scal/CPU", "blk/CPU", "ql/CPU",
                "resid_s", "resid_b", "resid_q", "sw_s", "sw_b");
    std::printf("%s\n", std::string(141, '-').c_str());
    for (const auto& c : cfgs) {
        for (int b : batches) {
            const bool ws = c.max_batch_scalar >= 0 && (c.max_batch_scalar == 0 || b <= c.max_batch_scalar);
            const bool wb = c.max_batch_block  >= 0 && (c.max_batch_block  == 0 || b <= c.max_batch_block);
            const bool wq = c.n <= ql_max;
            if (!ws && !wb && !wq) continue;
            std::printf("  running N=%d batch=%d ...\r", c.n, b);
            std::fflush(stdout);
            Row r = bench_point(c.n, b, ws, wb, wq);
            auto ms = [](bool on, double v) { return on ? v : 0.0; };
            std::printf("%6d %7d | ", r.n, r.batch);
            if (ws) std::printf("%10.3f ", r.scalar_ms); else std::printf("%10s ", "--");
            if (wb) std::printf("%10.3f ", r.block_ms);  else std::printf("%10s ", "--");
            if (wq) std::printf("%10.3f ", r.ql_ms);     else std::printf("%10s ", "--");
            std::printf("%10.3f | ", r.cpu_ms);
            if (ws) std::printf("%7.2fx ", r.cpu_ms / ms(ws, r.scalar_ms)); else std::printf("%8s ", "--");
            if (wb) std::printf("%7.2fx ", r.cpu_ms / ms(wb, r.block_ms)); else std::printf("%8s ", "--");
            if (wq) std::printf("%7.2fx | ", r.cpu_ms / ms(wq, r.ql_ms)); else std::printf("%8s | ", "--");
            if (ws) std::printf("%9.1e ", r.resid_scalar); else std::printf("%9s ", "--");
            if (wb) std::printf("%9.1e ", r.resid_block); else std::printf("%9s ", "--");
            if (wq) std::printf("%9.1e | ", r.resid_ql); else std::printf("%9s | ", "--");
            if (ws) std::printf("%5u ", r.sweeps_scalar); else std::printf("%5s ", "--");
            if (wb) std::printf("%5u\n", r.sweeps_block); else std::printf("%5s\n", "--");
            std::fflush(stdout);
        }
    }
    std::printf("\n");
}

void run_tune(int only) {
    // Tables 1 and 2 set each mode's internal parameter; table 3 compares the
    // modes with those parameters in effect (the compiled-in defaults, so
    // rebuild between adjusting eigh.mm and reading table 3).

    if (only == 0 || only == 1) {
    std::printf("\n[ 1. simd mode: matrices per threadgroup ]\n");
    std::printf("median ms; 'auto' is the rule in eigh.mm (kSimdMatricesPerTg, kSimdMinThreadgroupsPerCore)\n\n");
    std::printf("%6s %7s |", "N", "batch");
    for (int g : {1, 2, 4, 8, 16}) std::printf(" %8s", ("g=" + std::to_string(g)).c_str());
    std::printf("     auto\n");
    for (int n : {4, 8, 16, 32}) {
        for (int batch : {16, 64, 256, 4096}) {
            array A = random_symmetric(batch, n);
            eval({A});
            std::printf("%6d %7d |", n, batch);
            for (int g : {1, 2, 4, 8, 16}) {
                EighOptions o; o.mode = EighOptions::Mode::simd; o.matrices_per_threadgroup = g;
                std::printf(" %8.3f", time_opts(A, o));
                std::fflush(stdout);
            }
            EighOptions o; o.mode = EighOptions::Mode::simd;
            std::printf(" %8.3f\n", time_opts(A, o));
        }
    }
    }

    if (only == 0 || only == 2) {
    std::printf("\n[ 2. threadgroup mode: threads per matrix ]\n");
    std::printf("median ms; 'auto' is the rule in eigh.mm (kItemsPerThreadLargeBatch, kThreadsPerCoreBudget)\n\n");
    std::printf("%6s %7s |", "N", "batch");
    for (int th : {32, 64, 128, 256, 512, 1024}) std::printf(" %8s", ("t=" + std::to_string(th)).c_str());
    std::printf("     auto\n");
    for (int n : {8, 16, 24, 32, 48, 64, 128, 256, 512}) {
        for (int batch : {1, 16, 256}) {
            if (n >= 256 && batch > 16) continue;
            if (n >= 512 && batch > 8) continue;
            array A = random_symmetric(batch, n);
            eval({A});
            std::printf("%6d %7d |", n, batch);
            for (int th : {32, 64, 128, 256, 512, 1024}) {
                // A 512x512 matrix on 32 threads would take ~20 s and trip
                // the interactivity watchdog on its own; skip the hopeless
                // corner rather than crash the sweep.
                if ((n >= 256 && th < 128) || (n >= 512 && th < 512)) {
                    std::printf(" %8s", "--");
                    continue;
                }
                EighOptions o; o.mode = EighOptions::Mode::threadgroup; o.threads = th;
                std::printf(" %8.3f", time_opts(A, o));
                std::fflush(stdout);
            }
            EighOptions o; o.mode = EighOptions::Mode::threadgroup;
            std::printf(" %8.3f\n", time_opts(A, o));
        }
    }
    }

    if (only == 0 || only == 3) {
    std::printf("\n[ 3. execution mode: simd vs threadgroup, each with its automatic parameters ]\n");
    std::printf("the crossover is simd_max_n in the routing policy (eigh.mm)\n\n");
    std::printf("%6s %7s | %10s %10s | %s\n", "N", "batch", "simd ms", "tg ms", "faster");
    for (int n : {2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64}) {
        for (int batch : {16, 64, 1024, 8192}) {
            if (n >= 48 && batch > 1024) continue;
            array A = random_symmetric(batch, n);
            eval({A});
            EighOptions s; s.mode = EighOptions::Mode::simd;
            EighOptions t; t.mode = EighOptions::Mode::threadgroup;
            const double ms_s = time_opts(A, s);
            const double ms_t = time_opts(A, t);
            std::printf("%6d %7d | %10.3f %10.3f | %s %.2fx\n", n, batch, ms_s, ms_t,
                        ms_s <= ms_t ? "simd" : "tg  ", ms_s <= ms_t ? ms_t / ms_s : ms_s / ms_t);
            std::fflush(stdout);
        }
    }
    }

    if (only == 0 || only == 4) {
    std::printf("\n[ 4. backend: whole-matrix (scalar) vs block Jacobi ]\n");
    std::printf("the crossover is block_min_n in the routing policy (eigh.mm); tuning/tune_eigh.py is the full study\n\n");
    std::printf("%6s %7s | %10s %10s | %s\n", "N", "batch", "scalar ms", "block ms", "faster");
    for (int n : {32, 48, 64, 96, 128, 192, 256, 384, 512}) {
        for (int batch : {1, 4, 16, 64}) {
            if (n >= 256 && batch > 16) continue;
            if (n >= 512 && batch > 4) continue;
            array A = random_symmetric(batch, n);
            eval({A});
            const double ms_s = time_opts(A, scalar_opts());
            const double ms_b = time_opts(A, block_opts());
            std::printf("%6d %7d | %10.3f %10.3f | %s %.2fx\n", n, batch, ms_s, ms_b,
                        ms_s <= ms_b ? "scalar" : "block ", ms_s <= ms_b ? ms_b / ms_s : ms_s / ms_b);
            std::fflush(stdout);
        }
    }
    }

    if (only == 0 || only == 5) {
    std::printf("\n[ 5. block backend: inner sweeps per subproblem ]\n");
    std::printf("ms and outer sweeps to convergence; sets the EIGH_INNER_SWEEPS default\n\n");
    std::printf("%6s %7s |", "N", "batch");
    for (int k : {1, 2, 3, 4}) std::printf(" %14s", ("inner=" + std::to_string(k)).c_str());
    std::printf("\n");
    for (int n : {64, 128, 256, 512}) {
        for (int batch : {1, 8}) {
            if (n >= 512 && batch > 1) continue;
            array A = random_symmetric(batch, n);
            eval({A});
            std::printf("%6d %7d |", n, batch);
            for (int k : {1, 2, 3, 4}) {
                EighOptions o = block_opts(); o.inner_sweeps = k;
                const double ms = time_opts(A, o);
                EighResult r = detail::eigh_block_jacobi(A, true, true, o);
                eval({r.info});
                std::printf(" %9.3f (%2u)", ms, max_sweeps(r.info));
                std::fflush(stdout);
            }
            std::printf("\n");
        }
    }
    }
    std::printf("\n");
}

} // namespace

int main(int argc, char** argv) {
    set_default_device(Device::gpu);
    set_cache_limit(0);
    // The public functions route to the CPU where the GPU was measured
    // slower; this is the measurement, so force the kernel.
    setenv("EIGH_DEVICE", "gpu", 1);
    if (argc > 1 && std::strcmp(argv[1], "--tune") == 0) {
        run_tune(argc > 2 ? std::atoi(argv[2]) : 0);
        return 0;
    }
    if (argc > 1) { std::fprintf(stderr, "usage: %s [--tune [1|2|3|4|5]]\n", argv[0]); return 2; }
    run_main();
    return 0;
}
