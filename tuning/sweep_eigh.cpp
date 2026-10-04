// Eigensolver routing harness: times every backend on one (batch, N) point.
//
//   usage: sweep_eigh <batch> <N> <backend>[,<backend>...]
//   backends: cpu (the library's CPU path, LAPACK ssyevd, the batch over every core), simd,
//             tg (whole-matrix kernel in each execution mode), block (block Jacobi), tridiag
//             (the hybrid backend, eigh_tridiag.mm), ql (tridiagonalization and QL, one
//             threadgroup per matrix, eigh_ql.mm; N up to its device limit), ql_share (ql and the
//             CPU path sharing the batch, share_min_batch); each also as <name>_vals,
//             eigenvalues alone (eigvalsh: the CPU path is then LAPACK ssyevd_2stage
//             from N = 128), which has its own GPU-or-CPU boundary
//   out:   batch,N,backend,ok,ms,p25,p75,reps   (one row per backend)
//
//   usage: sweep_eigh --policy
//   out:   one JSON object: the device and the routing policy the library
//          resolved for it, which is what the harness compares against
//
// One point per process, so cached pipelines and recycled workspaces from an
// earlier shape cannot skew a later one. Within a point every backend sees the
// same fresh process, which is the fair comparison. Timing conventions are
// those of sweep_qr.cpp: median of adaptive repeats, quartiles for the
// spread, a correctness gate before anything is timed.

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

#include <metal_linalg/device.h>
#include <metal_linalg/eigh.h>

using namespace mlx::core;
using namespace metal_linalg;

namespace {

array random_symmetric(int batch, int n) {
    std::mt19937 rng(1234);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> data((size_t)batch * n * n);
    for (int b = 0; b < batch; ++b) {
        float* m = data.data() + (size_t)b * n * n;
        for (int i = 0; i < n; ++i)
            for (int j = 0; j <= i; ++j) {
                const float x = dist(rng);
                m[i * n + j] = x;
                m[j * n + i] = x;
            }
    }
    if (batch == 1) return array(data.begin(), {n, n}, float32);
    return array(data.begin(), {batch, n, n}, float32);
}

array transpose_last_two(const array& x) {
    std::vector<int> axes(x.ndim());
    for (size_t i = 0; i < axes.size(); ++i) axes[i] = (int)i;
    std::swap(axes[axes.size() - 1], axes[axes.size() - 2]);
    return transpose(x, axes);
}

// The solve under test, returning {w, V}; for eigenvalues alone, {w, w}.
struct Solver {
    std::string name;
    std::pair<array, array> (*fn)(const array&);
    bool vectors = true;
};

// What the library's routing sends to the CPU, so that is what it is measured against.
std::pair<array, array> solve_cpu(const array& A) {
    EighResult r = detail::eigh_cpu(A, true, true);
    return {r.eigenvalues, r.eigenvectors};
}
std::pair<array, array> solve_simd(const array& A) {
    EighOptions o; o.mode = EighOptions::Mode::simd;
    EighResult r = detail::eigh_jacobi(A, true, true, o);
    return {r.eigenvalues, r.eigenvectors};
}
std::pair<array, array> solve_tg(const array& A) {
    EighOptions o; o.mode = EighOptions::Mode::threadgroup;
    EighResult r = detail::eigh_jacobi(A, true, true, o);
    return {r.eigenvalues, r.eigenvectors};
}
std::pair<array, array> solve_block(const array& A) {
    EighOptions o; o.mode = EighOptions::Mode::block;
    EighResult r = detail::eigh_block_jacobi(A, true, true, o);
    return {r.eigenvalues, r.eigenvectors};
}

// Eigenvalues alone, as eigvalsh runs them. (The detail entry points take
// (A, compute_vectors, lower, ...).)
std::pair<array, array> vals_cpu(const array& A) {
    array w = detail::eigh_cpu(A, false, true).eigenvalues; return {w, w};
}
std::pair<array, array> vals_simd(const array& A) {
    EighOptions o; o.mode = EighOptions::Mode::simd;
    array w = detail::eigh_jacobi(A, false, true, o).eigenvalues; return {w, w};
}
std::pair<array, array> vals_tg(const array& A) {
    EighOptions o; o.mode = EighOptions::Mode::threadgroup;
    array w = detail::eigh_jacobi(A, false, true, o).eigenvalues; return {w, w};
}
std::pair<array, array> solve_tridiag(const array& A) {
    EighResult r = detail::eigh_tridiag(A, true, true);
    return {r.eigenvalues, r.eigenvectors};
}
std::pair<array, array> vals_tridiag(const array& A) {
    array w = detail::eigh_tridiag(A, false, true).eigenvalues; return {w, w};
}
std::pair<array, array> solve_ql(const array& A) {
    EighResult r = detail::eigh_ql(A, true, true);
    return {r.eigenvalues, r.eigenvectors};
}
std::pair<array, array> vals_ql(const array& A) {
    array w = detail::eigh_ql(A, false, true).eigenvalues; return {w, w};
}
std::pair<array, array> solve_ql_share(const array& A) {
    EighResult r = detail::eigh_ql_shared(A, true, true);
    return {r.eigenvalues, r.eigenvectors};
}
std::pair<array, array> vals_ql_share(const array& A) {
    array w = detail::eigh_ql_shared(A, false, true).eigenvalues; return {w, w};
}
std::pair<array, array> vals_block(const array& A) {
    EighOptions o; o.mode = EighOptions::Mode::block;
    array w = detail::eigh_block_jacobi(A, false, true, o).eigenvalues; return {w, w};
}

// Eigenvalues alone: the largest error against MLX's CPU eigvalsh (LAPACK,
// independent of every backend here), relative to ||A||.
float values_error(const Solver& s, const array& A) {
    try {
        array w = s.fn(A).first;
        array w_ref = linalg::eigvalsh(A, "L", Device::cpu);
        array e  = max(abs(subtract(w, w_ref)));
        array nA = sqrt(sum(square(A)));
        eval({e, nA});
        const float err = e.item<float>();
        if (!std::isfinite(err)) return INFINITY;
        return err / std::max(nA.item<float>(), 1e-30f);
    } catch (const std::exception&) {
        return INFINITY;
    }
}

// Relative eigen-residual, or infinity if the backend cannot run here.
float correctness(const Solver& s, const array& A) {
    if (!s.vectors) return values_error(s, A);
    try {
        auto [w, V] = s.fn(A);
        eval({w, V});
        array bad = any(logical_or(isnan(V), isinf(V)));
        eval({bad});
        if (bad.item<bool>()) return INFINITY;
        array VL = multiply(V, expand_dims(w, -2));
        array r  = sqrt(sum(square(subtract(matmul(A, V), VL))));
        array nA = sqrt(sum(square(A)));
        eval({r, nA});
        return r.item<float>() / std::max(nA.item<float>(), 1e-30f);
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

// Adaptive timing: at least `min_reps`, then until `budget_ms` is spent or
// `max_reps` reached. Slow points settle for the minimum.
Timing time_ms(const Solver& s, const array& A, double budget_ms = 150.0, int max_reps = 25) {
    for (int i = 0; i < 2; ++i) {   // warmup: pipeline compile + workspace alloc
        auto [w, V] = s.fn(A);
        eval({w, V});
    }
    std::vector<double> samples;
    double total = 0.0;
    int min_reps = 5;
    while ((int)samples.size() < max_reps && (total < budget_ms || (int)samples.size() < min_reps)) {
        auto t0 = std::chrono::high_resolution_clock::now();
        auto [w, V] = s.fn(A);
        eval({w, V});
        auto t1 = std::chrono::high_resolution_clock::now();
        const double dt = std::chrono::duration<double, std::milli>(t1 - t0).count();
        samples.push_back(dt);
        total += dt;
        if (dt > 300.0) min_reps = 3;   // a slow call is its own evidence
    }
    std::sort(samples.begin(), samples.end());
    return Timing{quantile(samples, 0.50), quantile(samples, 0.25),
                  quantile(samples, 0.75), (int)samples.size()};
}

void measure(const Solver& s, const array& A, int batch, int N) {
    const float err = correctness(s, A);
    const bool ok = std::isfinite(err) && err <= 1e-3f;
    const Timing t = ok ? time_ms(s, A) : Timing{};
    std::printf("%d,%d,%s,%d,%.6f,%.6f,%.6f,%d\n",
                batch, N, s.name.c_str(), ok ? 1 : 0, t.median, t.p25, t.p75, t.reps);
    std::fflush(stdout);
}

} // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::string(argv[1]) == "--policy") {
        const EighPolicy p = eigh_policy();
        std::printf("{\"device\": \"%s\", \"gpu_cores\": %u, \"source\": \"%s\", "
                    "\"simd_max_n\": %u, \"block_min_n\": %u, \"block_min_n_batched\": %u, "
                    "\"block_min_batch\": %u, \"gpu_max_n\": %u, \"gpu_min_batch_times_n\": %u, "
                    "\"gpu_min_batch\": %u, \"values_gpu_max_n\": %u, "
                    "\"values_gpu_min_batch_times_n\": %u, \"values_gpu_min_batch\": %u, "
                    "\"tridiag_min_n\": %u, \"values_tridiag_min_n\": %u, "
                    "\"ql_min_n\": %u, \"ql_max_n\": %u, \"ql_limit\": %u, \"cpu_threads\": %u, "
                    "\"tridiag_max_batch\": %u, \"values_tridiag_max_batch\": %u, \"share_min_batch\": %u, "
                    "\"gpu_big_batch_max_n\": %u, \"gpu_big_batch_min\": %u}\n",
                    device_name(), p.gpu_cores, eigh_policy_source(),
                    p.simd_max_n, p.block_min_n, p.block_min_n_batched, p.block_min_batch,
                    p.gpu_max_n, p.gpu_min_batch_times_n, p.gpu_min_batch,
                    p.values_gpu_max_n, p.values_gpu_min_batch_times_n, p.values_gpu_min_batch,
                    p.tridiag_min_n, p.values_tridiag_min_n, p.ql_min_n, p.ql_max_n,
                    metal_linalg::detail::eigh_ql_max_n(), cpu_threads(),
                    p.tridiag_max_batch, p.values_tridiag_max_batch, p.share_min_batch,
                    p.gpu_big_batch_max_n, p.gpu_big_batch_min);
        return 0;
    }
    if (argc != 4) {
        std::fprintf(stderr, "usage: %s <batch> <N> <cpu|simd|tg|block|tridiag|ql|ql_share>[_vals][,...]\n"
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
        if      (name == "cpu")   solvers.push_back({name, solve_cpu});
        else if (name == "simd")  solvers.push_back({name, solve_simd});
        else if (name == "tg")    solvers.push_back({name, solve_tg});
        else if (name == "block") solvers.push_back({name, solve_block});
        else if (name == "cpu_vals")   solvers.push_back({name, vals_cpu, false});
        else if (name == "simd_vals")  solvers.push_back({name, vals_simd, false});
        else if (name == "tg_vals")    solvers.push_back({name, vals_tg, false});
        else if (name == "block_vals") solvers.push_back({name, vals_block, false});
        else if (name == "tridiag")      solvers.push_back({name, solve_tridiag});
        else if (name == "tridiag_vals") solvers.push_back({name, vals_tridiag, false});
        else if (name == "ql")           solvers.push_back({name, solve_ql});
        else if (name == "ql_vals")      solvers.push_back({name, vals_ql, false});
        else if (name == "ql_share")      solvers.push_back({name, solve_ql_share});
        else if (name == "ql_share_vals") solvers.push_back({name, vals_ql_share, false});
        else if (!name.empty()) { std::fprintf(stderr, "unknown backend: %s\n", name.c_str()); return 2; }
        pos = comma + 1;
    }

    set_default_device(Device::gpu);
    set_cache_limit(0);
    setenv("EIGH_DEVICE", "gpu", 1);   // the detail entry points ignore routing anyway

    array A = random_symmetric(batch, N);
    eval({A});

    for (const auto& s : solvers) measure(s, A, batch, N);
    return 0;
}
