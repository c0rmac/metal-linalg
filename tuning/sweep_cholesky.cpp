// Cholesky routing harness: times every backend on one (batch, N) point.
//
//   usage: sweep_cholesky <batch> <N> <backend>[,<backend>...]
//   backends: cpu (the library's CPU path, LAPACK spotrf, the batch over every core), simd (a
//             matrix in a simdgroup's registers, N up to 32), tg (a matrix a threadgroup),
//             blocked (the large-matrix path: fused 32-column sub-panels and MPS products)
//   out:   batch,N,backend,ok,ms,p25,p75,reps   (one row per backend)
//
//   usage: sweep_cholesky --policy
//   out:   one JSON object: the device and the routing policy the library
//          resolved for it, which is what the harness compares against
//
// One point per process, so cached pipelines and recycled workspaces from an
// earlier shape cannot skew a later one. Within a point every backend sees the
// same fresh process, which is the fair comparison. Medians of adaptive
// repeats, quartiles for the spread, a correctness gate before anything is
// timed: sweep_timing.h.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include <mlx/mlx.h>

#include "sweep_timing.h"

#include <metal_linalg/cholesky.h>
#include <metal_linalg/device.h>

using namespace mlx::core;
using namespace metal_linalg;

namespace {

// (M M^T + N I) / N for N up to 1024, else a diagonally dominant matrix (the
// product is O(N^3) on the CPU here, longer than the sweep's point): both
// well-conditioned symmetric positive definite.
array random_spd(int batch, int N) {
    std::mt19937 rng(1234);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> m((size_t)N * N);
    std::vector<float> data((size_t)batch * N * N);
    for (int b = 0; b < batch; ++b) {
        for (auto& v : m) v = dist(rng);
        float* a = data.data() + (size_t)b * N * N;
        if (N > 1024) {
            const float s = 1.0f / std::sqrt((float)N);
            for (int i = 0; i < N; ++i)
                for (int j = 0; j <= i; ++j)
                    a[(size_t)i * N + j] = a[(size_t)j * N + i] = i == j ? 2.0f : m[(size_t)i * N + j] * s * 0.5f;
            continue;
        }
        for (int i = 0; i < N; ++i)
            for (int j = 0; j <= i; ++j) {
                double s = i == j ? N : 0.0;
                for (int t = 0; t < N; ++t) s += (double)m[(size_t)i * N + t] * m[(size_t)j * N + t];
                a[(size_t)i * N + j] = a[(size_t)j * N + i] = (float)(s / N);
            }
    }
    if (batch == 1) return array(data.begin(), {N, N}, float32);
    return array(data.begin(), {batch, N, N}, float32);
}

using Fn = CholeskyResult (*)(const array&, bool);

struct Solver {
    std::string name;
    Fn fn;
};

// ||L L^T - A||_F / ||A||_F on up to 8 matrices of the batch, infinity for a
// failure or anything non-finite.
float correctness(const Solver& s, const array& A, double& call_ms) {
    try {
        const auto t0 = std::chrono::high_resolution_clock::now();
        CholeskyResult r = s.fn(A, false);
        eval({r.l, r.info});
        call_ms = sweep::ms_since(t0);
        array failed = any(not_equal(r.info, array(0u, uint32)));
        array bad = any(logical_or(isnan(r.l), isinf(r.l)));
        eval({failed, bad});
        if (failed.item<bool>() || bad.item<bool>()) return INFINITY;
        array a = A, l = r.l;
        if (A.ndim() == 3 && A.shape(0) > 8) {
            a = slice(A, {0, 0, 0}, {8, A.shape(1), A.shape(2)});
            l = slice(r.l, {0, 0, 0}, {8, A.shape(1), A.shape(2)});
        }
        const int d = (int)a.ndim();
        std::vector<int> axes(d);
        for (int i = 0; i < d; ++i) axes[i] = i;
        std::swap(axes[d - 1], axes[d - 2]);
        array e = sqrt(sum(square(subtract(matmul(l, transpose(l, axes)), a))));
        array na = sqrt(sum(square(a)));
        eval({e, na});
        return e.item<float>() / std::max(na.item<float>(), 1e-30f);
    } catch (const std::exception&) {
        return INFINITY;
    }
}

} // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::string(argv[1]) == "--policy") {
        const CholeskyPolicy p = cholesky_policy();
        std::printf("{\"device\": \"%s\", \"gpu_cores\": %u, \"source\": \"%s\", \"cpu_threads\": %u, "
                    "\"simd_max_n\": %u, \"blocked_min_n\": %u, \"blocked_max_batch\": %u, "
                    "\"gpu_max_n\": %u, \"gpu_min_batch_times_n\": %u, \"gpu_min_batch\": %u, "
                    "\"gpu_min_n\": %u, \"gpu_large_min_n\": %u, \"gpu_large_max_batch\": %u}\n",
                    device_name(), p.gpu_cores, cholesky_policy_source(), cpu_threads(), p.simd_max_n,
                    p.blocked_min_n, p.blocked_max_batch, p.gpu_max_n, p.gpu_min_batch_times_n, p.gpu_min_batch,
                    p.gpu_min_n, p.gpu_large_min_n, p.gpu_large_max_batch);
        return 0;
    }
    if (argc != 4) {
        std::fprintf(stderr, "usage: %s <batch> <N> <cpu|simd|tg|blocked>[,...]\n"
                             "       %s --policy\n", argv[0], argv[0]);
        return 2;
    }
    const int batch = std::atoi(argv[1]);
    const int N     = std::atoi(argv[2]);

    std::vector<Solver> solvers;
    std::string list = argv[3];
    for (size_t pos = 0; pos <= list.size();) {
        size_t comma = list.find(',', pos);
        if (comma == std::string::npos) comma = list.size();
        const std::string name = list.substr(pos, comma - pos);
        if      (name == "cpu")     solvers.push_back({name, detail::cholesky_cpu});
        else if (name == "simd")    solvers.push_back({name, detail::cholesky_simd});
        else if (name == "tg")      solvers.push_back({name, detail::cholesky_threadgroup});
        else if (name == "blocked") solvers.push_back({name, detail::cholesky_blocked});
        else if (!name.empty()) { std::fprintf(stderr, "unknown backend: %s\n", name.c_str()); return 2; }
        pos = comma + 1;
    }

    set_default_device(Device::gpu);
    // MLX's buffer cache is left on, as an MLX program has it (see sweep_qr.cpp).

    array A = random_spd(batch, N);
    eval({A});

    std::vector<sweep::Backend> bs;
    for (const auto& s : solvers)
        bs.push_back({[&s, &A](double& call_ms) { return correctness(s, A, call_ms); },
                      [&s, &A] {
                          CholeskyResult r = s.fn(A, false);
                          eval({r.l, r.info});
                      },
                      0});
    const std::vector<sweep::Result> rs = sweep::measure_all(bs);
    for (size_t i = 0; i < solvers.size(); ++i) {
        const sweep::Timing& t = rs[i].t;
        std::printf("%d,%d,%s,%d,%.6f,%.6f,%.6f,%d\n",
                    batch, N, solvers[i].name.c_str(), rs[i].ok ? 1 : 0, t.median, t.p25, t.p75, t.reps);
    }
    std::fflush(stdout);
    return 0;
}
