// Dispatch tuning harness.
//
// Times every backend on one (batch, M, N) point and prints a CSV row per
// backend. One point per process, so cached pipelines and recycled workspaces
// from an earlier shape cannot skew a later one.
//
//   usage: sweep_qr <batch> <M> <N> <backend>
//   out:   batch,M,N,backend,ok,ms,p25,p75,reps
//
//   usage: sweep_qr --policy
//   out:   one JSON object: the device and the routing policy the library
//          resolved for it
//
// `ms` is the median, not the mean: GPU timings are right-skewed (frequency
// changes, other work on the device), so a mean chases outliers. p25/p75 give
// the spread, which is what tells us whether a difference between two backends
// is real or inside the noise floor.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include <mlx/mlx.h>

#include <metal_linalg/device.h>
#include <metal_linalg/qr.h>

using namespace mlx::core;
using namespace metal_linalg;

namespace {

array random_matrix(int batch, int M, int N) {
    std::mt19937 rng(1234);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> data((size_t)batch * M * N);
    for (auto& v : data) v = dist(rng);
    if (batch == 1) return array(data.begin(), {M, N}, float32);
    return array(data.begin(), {batch, M, N}, float32);
}

float frobenius(const array& x) {
    array n = sqrt(sum(square(x)));
    eval({n});
    return n.item<float>();
}

using QrFn = std::pair<array, array> (*)(const array&);

// Relative reconstruction error, or infinity if the backend cannot run here.
float correctness(QrFn qr, const array& A) {
    try {
        auto [Q, R] = qr(A);
        eval({Q, R});
        array bad = any(logical_or(isnan(Q), isinf(Q)));
        eval({bad});
        if (bad.item<bool>()) return INFINITY;
        return frobenius(subtract(matmul(Q, R), A)) / std::max(frobenius(A), 1.0f);
    } catch (const std::exception&) {
        return INFINITY;
    }
}

struct Timing {
    double median = 0, p25 = 0, p75 = 0;
    int reps = 0;
};

double quantile(const std::vector<double>& sorted, double q) {
    if (sorted.empty()) return 0.0;
    const double pos = q * (sorted.size() - 1);
    const size_t lo = (size_t)std::floor(pos), hi = (size_t)std::ceil(pos);
    return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}

// Adaptive timing: keep sampling until we have spent `budget_ms` or hit
// `max_reps`. Slow points settle for the minimum five reps.
Timing time_ms(QrFn qr, const array& A, double budget_ms = 150.0, int max_reps = 25) {
    for (int i = 0; i < 2; ++i) {  // warmup: pipeline compile + workspace alloc
        auto [Q, R] = qr(A);
        eval({Q, R});
    }

    std::vector<double> samples;
    double total = 0.0;
    while ((int)samples.size() < max_reps && (total < budget_ms || samples.size() < 5)) {
        auto t0 = std::chrono::high_resolution_clock::now();
        auto [Q, R] = qr(A);
        eval({Q, R});
        auto t1 = std::chrono::high_resolution_clock::now();
        const double dt = std::chrono::duration<double, std::milli>(t1 - t0).count();
        samples.push_back(dt);
        total += dt;
    }

    std::sort(samples.begin(), samples.end());
    return Timing{quantile(samples, 0.50), quantile(samples, 0.25),
                  quantile(samples, 0.75), (int)samples.size()};
}

void measure(const char* name, QrFn qr, const array& A, int batch, int M, int N) {
    const float err = correctness(qr, A);
    const bool ok = std::isfinite(err) && err <= 1e-3f;
    const Timing t = ok ? time_ms(qr, A) : Timing{};
    std::printf("%d,%d,%d,%s,%d,%.6f,%.6f,%.6f,%d\n",
                batch, M, N, name, ok ? 1 : 0, t.median, t.p25, t.p75, t.reps);
    std::fflush(stdout);
}

} // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::string(argv[1]) == "--policy") {
        const QrPolicy p = qr_policy();
        std::printf("{\"device\": \"%s\", \"gpu_cores\": %u, \"source\": \"%s\", "
                    "\"m_crossover_small_batch\": %u, \"m_crossover_large_batch\": %u, "
                    "\"batch_threshold\": %u, \"gpu_max_k\": %u, \"gpu_min_batch_times_k\": %u, "
                    "\"gpu_min_batch\": %u, \"concurrent_matrices\": %u, "
                    "\"gpu_large_min_k\": %u, \"gpu_large_max_batch\": %u}\n",
                    device_name(), p.gpu_cores, qr_policy_source(),
                    p.m_crossover_small_batch, p.m_crossover_large_batch, p.batch_threshold,
                    p.gpu_max_k, p.gpu_min_batch_times_k, p.gpu_min_batch,
                    p.concurrent_matrices, p.gpu_large_min_k, p.gpu_large_max_batch);
        return 0;
    }
    if (argc != 5) {
        std::fprintf(stderr, "usage: %s <batch> <M> <N> <unblocked|reduced|complete|cpu>\n"
                             "       %s --policy\n", argv[0], argv[0]);
        return 2;
    }
    const int batch = std::atoi(argv[1]);
    const int M     = std::atoi(argv[2]);
    const int N     = std::atoi(argv[3]);
    const std::string which = argv[4];

    QrFn qr = nullptr;
    if      (which == "unblocked") qr = detail::qr_unblocked;
    else if (which == "reduced")   qr = detail::qr_streaming_amx_reduced;
    else if (which == "complete")  qr = detail::qr_streaming_amx_complete;
    else if (which == "cpu")       qr = detail::qr_cpu;          // LAPACK, the CPU route
    else { std::fprintf(stderr, "unknown backend: %s\n", argv[4]); return 2; }

    set_default_device(Device::gpu);
    set_cache_limit(0);

    array A = random_matrix(batch, M, N);
    eval({A});

    measure(which.c_str(), qr, A, batch, M, N);
    return 0;
}
