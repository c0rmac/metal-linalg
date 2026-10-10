// LU routing harness: times every backend on one (batch, N, K) point.
//
//   usage: sweep_lu <batch> <N> <K> <backend>[,<backend>...]
//   backends: cpu, blocked (lu_factor: LAPACK sgetrf a batch over every core; the GPU path),
//             inv_cpu, inv_blocked (the inverse), solve_cpu (sgetrf + sgetrs), solve_trsm (the
//             GPU path, its blocked triangular solves on the GPU), solve_getrs (the GPU path's
//             factorization, sgetrs on it) -- solve with K right-hand sides, the rest K = 1
//   out:   batch,N,K,backend,ok,ms,p25,p75,reps   (one row per backend)
//
//   usage: sweep_lu --policy
//   out:   one JSON object: the device and the routing policy the library resolved for it
//
// One point per process; medians of adaptive repeats, quartiles for the
// spread, a correctness gate before anything is timed: sweep_timing.h.

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

#include <metal_linalg/device.h>
#include <metal_linalg/lu.h>

using namespace mlx::core;
using namespace metal_linalg;

namespace {

// Gaussian, plus 2 sqrt(N) on the diagonal: well conditioned, pivoting still at work.
array random_matrices(int batch, int n, int cols, unsigned seed, float diag) {
    std::mt19937 r(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> v((size_t)batch * n * cols);
    for (auto& x : v) x = dist(r);
    if (diag != 0.0f)
        for (int b = 0; b < batch; ++b)
            for (int i = 0; i < std::min(n, cols); ++i) v[(size_t)b * n * cols + (size_t)i * cols + i] += diag;
    return array(v.begin(), {batch, n, cols}, float32);
}

float max_abs(const array& x) {
    array m = max(abs(x, Device::cpu), Device::cpu);
    eval({m});
    return m.item<float>();
}

// The first matrix's residual: |A X - B| / |B| (B = I for the inverse,
// P A for the factorization).
float residual(const std::string& name, const array& A, const array& B) {
    const array a = slice(A, {0, 0, 0}, {1, A.shape(1), A.shape(2)});
    if (name == "cpu" || name == "blocked") {
        LuResult r = name == "cpu" ? detail::lu_factor_cpu(a) : detail::lu_factor_blocked(a);
        eval({r.lu, r.pivots});
        const int n = A.shape(1);
        const float* lu = r.lu.data<float>();
        std::vector<int> perm(n);
        for (int i = 0; i < n; ++i) perm[i] = i;
        for (int i = 0; i < n; ++i) std::swap(perm[i], perm[r.pivots.data<uint32_t>()[i]]);
        eval({a});
        const float* av = a.data<float>();
        double err = 0, scale = 0;
        for (int i = 0; i < n; i += std::max(1, n / 16))   // sampled rows
            for (int j = 0; j < n; ++j) {
                double s = 0;
                for (int t = 0; t <= std::min(i, j); ++t) s += (t == i ? 1.0 : (double)lu[i * n + t]) * lu[t * n + j];
                err = std::max(err, std::fabs(s - av[perm[i] * n + j]));
                scale = std::max(scale, (double)std::fabs(av[perm[i] * n + j]));
            }
        return (float)(err / std::max(scale, 1e-30));
    }
    array x = name == "inv_cpu" ? detail::inv_cpu(a).x : name == "inv_blocked" ? detail::inv_blocked(a).x
            : name == "solve_cpu" ? detail::solve_cpu(a, slice(B, {0, 0, 0}, {1, B.shape(1), B.shape(2)})).x
                                  : detail::solve_blocked(a, slice(B, {0, 0, 0}, {1, B.shape(1), B.shape(2)})).x;
    eval({x});
    const array rhs = name.rfind("inv", 0) == 0 ? array(eye(A.shape(1), float32)) : slice(B, {0, 0, 0}, {1, B.shape(1), B.shape(2)});
    return max_abs(subtract(matmul(a, x, Device::cpu), rhs, Device::cpu)) / std::max(max_abs(rhs), 1e-30f);
}

} // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::string(argv[1]) == "--policy") {
        const LuPolicy p = lu_policy();
        std::printf("{\"device\": \"%s\", \"gpu_cores\": %u, \"source\": \"%s\", \"cpu_threads\": %u, "
                    "\"gpu_min_n\": %u, \"gpu_max_batch\": %u, \"gpu_solve_min_rhs\": %u}\n",
                    device_name(), p.gpu_cores, lu_policy_source(), cpu_threads(), p.gpu_min_n, p.gpu_max_batch,
                    p.gpu_solve_min_rhs);
        return 0;
    }
    if (argc != 5) {
        std::fprintf(stderr, "usage: %s <batch> <N> <K> <cpu|blocked|inv_cpu|inv_blocked|solve_cpu|solve_trsm|"
                             "solve_getrs>[,...]\n       %s --policy\n", argv[0], argv[0]);
        return 2;
    }
    const int batch = std::atoi(argv[1]), N = std::atoi(argv[2]), K = std::atoi(argv[3]);
    std::vector<std::string> names;
    const std::string list = argv[4];
    for (size_t pos = 0; pos <= list.size();) {
        size_t comma = list.find(',', pos);
        if (comma == std::string::npos) comma = list.size();
        const std::string name = list.substr(pos, comma - pos);
        if (name == "cpu" || name == "blocked" || name == "inv_cpu" || name == "inv_blocked" || name == "solve_cpu" ||
            name == "solve_trsm" || name == "solve_getrs")
            names.push_back(name);
        else if (!name.empty()) { std::fprintf(stderr, "unknown backend: %s\n", name.c_str()); return 2; }
        pos = comma + 1;
    }

    set_default_device(Device::gpu);
    const array A = random_matrices(batch, N, N, 1234, 2.0f * std::sqrt((float)N));
    const array B = random_matrices(batch, N, std::max(K, 1), 99, 0.0f);
    eval({A, B});
    const LuPolicy policy = lu_policy();

    auto run = [&](const std::string& name) {
        if (name == "cpu") { LuResult r = detail::lu_factor_cpu(A); eval({r.lu}); }
        else if (name == "blocked") { LuResult r = detail::lu_factor_blocked(A); eval({r.lu}); }
        else if (name == "inv_cpu") { SolveResult r = detail::inv_cpu(A); eval({r.x}); }
        else if (name == "inv_blocked") { SolveResult r = detail::inv_blocked(A); eval({r.x}); }
        else if (name == "solve_cpu") { SolveResult r = detail::solve_cpu(A, B); eval({r.x}); }
        else {   // the GPU path, its solve forced to one side of gpu_solve_min_rhs
            LuPolicy p = policy;
            p.gpu_solve_min_rhs = name == "solve_trsm" ? 1u : 0xFFFFFFFFu;
            set_lu_policy(p);
            SolveResult r = detail::solve_blocked(A, B);
            eval({r.x});
            set_lu_policy(policy);
        }
    };
    std::vector<sweep::Backend> bs;
    for (const auto& name : names)
        bs.push_back({[&, name](double& call_ms) {
                          try {
                              const auto t0 = std::chrono::high_resolution_clock::now();
                              run(name);
                              call_ms = sweep::ms_since(t0);
                              if (name == "solve_trsm" || name == "solve_getrs") {
                                  LuPolicy p = policy;
                                  p.gpu_solve_min_rhs = name == "solve_trsm" ? 1u : 0xFFFFFFFFu;
                                  set_lu_policy(p);
                                  const float r = residual("solve_blocked", A, B);
                                  set_lu_policy(policy);
                                  return r;
                              }
                              return residual(name, A, B);
                          } catch (const std::exception&) {
                              return INFINITY;
                          }
                      },
                      [&, name] { run(name); },
                      name.rfind("solve", 0) == 0 ? 2 : name.rfind("inv", 0) == 0 ? 1 : 0});
    const std::vector<sweep::Result> rs = sweep::measure_all(bs);
    for (size_t i = 0; i < names.size(); ++i) {
        const sweep::Timing& t = rs[i].t;
        std::printf("%d,%d,%d,%s,%d,%.6f,%.6f,%.6f,%d\n", batch, N, K, names[i].c_str(), rs[i].ok ? 1 : 0, t.median,
                    t.p25, t.p75, t.reps);
    }
    std::fflush(stdout);
    return 0;
}
