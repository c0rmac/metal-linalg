// Triangular solve routing harness: times each backend on one (batch, N, K) point.
//
//   usage: sweep_trsm <batch> <N> <K> <cpu|blocked>[,...]
//   out:   batch,N,K,backend,ok,ms,p25,p75,reps   (one row per backend)
//
//   usage: sweep_trsm --policy
//   out:   one JSON object: the device and the routing policy the library resolved for it
//
// Lower triangles with a diagonal of about 2 and small entries below it (well
// conditioned), K right-hand sides. One point per process; sweep_timing.h.

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
#include <metal_linalg/triangular.h>

using namespace mlx::core;
using namespace metal_linalg;

int main(int argc, char** argv) {
    if (argc == 2 && std::string(argv[1]) == "--policy") {
        const TrsmPolicy p = trsm_policy();
        std::printf("{\"device\": \"%s\", \"gpu_cores\": %u, \"source\": \"%s\", \"cpu_threads\": %u, "
                    "\"gpu_min_n\": %u, \"gpu_min_rhs\": %u, \"gpu_max_batch\": %u}\n",
                    device_name(), p.gpu_cores, trsm_policy_source(), cpu_threads(), p.gpu_min_n, p.gpu_min_rhs,
                    p.gpu_max_batch);
        return 0;
    }
    if (argc != 5) {
        std::fprintf(stderr, "usage: %s <batch> <N> <K> <cpu|blocked>[,...]\n       %s --policy\n", argv[0], argv[0]);
        return 2;
    }
    const int batch = std::atoi(argv[1]), n = std::atoi(argv[2]), k = std::atoi(argv[3]);
    std::vector<std::string> names;
    const std::string list = argv[4];
    for (size_t pos = 0; pos <= list.size();) {
        size_t comma = list.find(',', pos);
        if (comma == std::string::npos) comma = list.size();
        const std::string name = list.substr(pos, comma - pos);
        if (name == "cpu" || name == "blocked") names.push_back(name);
        else if (!name.empty()) { std::fprintf(stderr, "unknown backend: %s\n", name.c_str()); return 2; }
        pos = comma + 1;
    }
    set_default_device(Device::gpu);
    std::mt19937 g(1234);
    std::normal_distribution<float> nd;
    std::vector<float> av((size_t)batch * n * n, 0.0f), bv((size_t)batch * n * k);
    for (int b = 0; b < batch; ++b)
        for (int i = 0; i < n; ++i)
            for (int j = 0; j <= i; ++j) av[(size_t)b * n * n + (size_t)i * n + j] = i == j ? 2.0f : nd(g) / std::sqrt((float)n);
    for (auto& x : bv) x = nd(g);
    const array A(av.begin(), {batch, n, n}, float32), B(bv.begin(), {batch, n, k}, float32);
    eval({A, B});
    auto solve = [&](const std::string& name) {
        return name == "cpu" ? detail::solve_triangular_cpu(A, B) : detail::solve_triangular_blocked(A, B);
    };
    std::vector<sweep::Backend> bs;
    for (const auto& name : names)
        bs.push_back({[&, name](double& call_ms) {
                          try {
                              const auto t0 = std::chrono::high_resolution_clock::now();
                              array x = solve(name);
                              eval({x});
                              call_ms = sweep::ms_since(t0);
                              const array a1 = slice(A, {0, 0, 0}, {1, n, n}), x1 = slice(x, {0, 0, 0}, {1, n, k}),
                                          b1 = slice(B, {0, 0, 0}, {1, n, k});
                              array r = max(abs(subtract(matmul(a1, x1, Device::cpu), b1, Device::cpu), Device::cpu),
                                            Device::cpu);
                              array s = max(abs(b1, Device::cpu), Device::cpu);
                              eval({r, s});
                              return r.item<float>() / std::max(s.item<float>(), 1e-30f);
                          } catch (const std::exception&) {
                              return INFINITY;
                          }
                      },
                      [&, name] { array x = solve(name); eval({x}); }, 0});
    const std::vector<sweep::Result> rs = sweep::measure_all(bs);
    for (size_t i = 0; i < names.size(); ++i) {
        const sweep::Timing& t = rs[i].t;
        std::printf("%d,%d,%d,%s,%d,%.6f,%.6f,%.6f,%d\n", batch, n, k, names[i].c_str(), rs[i].ok ? 1 : 0, t.median,
                    t.p25, t.p75, t.reps);
    }
    std::fflush(stdout);
    return 0;
}
