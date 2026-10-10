// Correctness tests for the triangular solve: each backend directly (cpu,
// blocked) and through the routing, lower and upper, with and without a unit
// diagonal, one to many right-hand sides, a vector b; the residual A X - B,
// the other triangle never read, agreement with MLX's own; shapes and errors;
// the routing policy.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include <mlx/linalg.h>
#include <mlx/mlx.h>

#include <metal_linalg/triangular.h>

using namespace mlx::core;
using namespace metal_linalg;

namespace {

int g_failures = 0;
int g_checks   = 0;

void expect(const std::string& label, bool ok, const std::string& detail = "") {
    ++g_checks;
    if (!ok) {
        std::printf("  FAIL  %-60s %s\n", label.c_str(), detail.c_str());
        ++g_failures;
    }
}

std::string fmt(const char* f, double v) {
    char buf[64];
    std::snprintf(buf, sizeof buf, f, v);
    return buf;
}

float max_abs(const array& x) {
    array m = max(abs(x, Device::cpu), Device::cpu);
    eval({m});
    return m.item<float>();
}

// Triangular matrices with a well-scaled diagonal (2 +- 0.5) and small
// entries in the triangle, junk (1e3) in the other: never to be read.
std::vector<float> triangles(int batch, int n, bool upper, unsigned seed) {
    std::mt19937 g(seed);
    std::normal_distribution<float> nd;
    std::uniform_real_distribution<float> ud(1.5f, 2.5f);
    std::vector<float> v((size_t)batch * n * n);
    for (int b = 0; b < batch; ++b)
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) {
                float& e = v[(size_t)b * n * n + (size_t)i * n + j];
                if (i == j) e = ud(g);
                else if (upper ? j > i : j < i) e = nd(g) / std::sqrt((float)n);
                else e = 1e3f;
            }
    return v;
}

// The triangle alone (zeros elsewhere; ones on the diagonal with `unit`).
array clean(const std::vector<float>& v, int batch, int n, bool upper, bool unit) {
    std::vector<float> w = v;
    for (int b = 0; b < batch; ++b)
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) {
                float& e = w[(size_t)b * n * n + (size_t)i * n + j];
                if (i == j && unit) e = 1.0f;
                else if (i != j && (upper ? j < i : j > i)) e = 0.0f;
            }
    return array(w.begin(), {batch, n, n}, float32);
}

using Fn = std::function<array(const array&, const array&, bool, bool)>;

void run_backends() {
    std::printf("\n[ every backend, lower and upper, unit and not ]\n");
    const std::vector<std::pair<const char*, Fn>> backends = {
        {"cpu", [](const array& a, const array& b, bool u, bool d) { return detail::solve_triangular_cpu(a, b, u, d); }},
        {"blocked", [](const array& a, const array& b, bool u, bool d) { return detail::solve_triangular_blocked(a, b, u, d); }},
        {"dispatch", [](const array& a, const array& b, bool u, bool d) { return solve_triangular_accelerated(a, b, u, d); }},
    };
    const int cases[][3] = {{3, 1, 1}, {2, 5, 3}, {2, 127, 7}, {2, 128, 1}, {1, 129, 40}, {2, 300, 200},
                            {1, 700, 33}, {1, 1100, 300}};
    unsigned seed = 1;
    for (const auto& c : cases) {
        const int batch = c[0], n = c[1], k = c[2];
        std::mt19937 g(seed++);
        std::normal_distribution<float> nd;
        std::vector<float> bv((size_t)batch * n * k);
        for (auto& x : bv) x = nd(g);
        const array B(bv.begin(), {batch, n, k}, float32);
        for (bool upper : {false, true})
            for (bool unit : {false, true}) {
                const std::vector<float> av = triangles(batch, n, upper, seed++);
                const array A(av.begin(), {batch, n, n}, float32);
                const array Ac = clean(av, batch, n, upper, unit);
                const array ref = linalg::solve_triangular(Ac, B, upper, Device::cpu);
                for (const auto& [name, fn] : backends) {
                    const std::string tag = std::string(name) + " " + std::to_string(batch) + " x " +
                                            std::to_string(n) + " k=" + std::to_string(k) + (upper ? " U" : " L") +
                                            (unit ? " unit" : "");
                    array X = fn(A, B, upper, unit);
                    eval({X});
                    const float res = max_abs(subtract(matmul(Ac, X, Device::cpu), B, Device::cpu)) /
                                      std::max(max_abs(B), 1e-30f);
                    expect(tag + ": A X = B", X.shape() == B.shape() && res < 1e-4f, fmt("residual %.2e", res));
                    const float d = max_abs(subtract(X, ref, Device::cpu)) / std::max(max_abs(ref), 1e-30f);
                    expect(tag + ": agrees with mx::linalg::solve_triangular", d < 1e-4f, fmt("rel diff %.2e", d));
                }
            }
    }
}

void run_shapes() {
    std::printf("\n[ vectors, shapes, errors ]\n");
    const std::vector<float> av = triangles(6, 20, false, 5);
    const array A = reshape(array(av.begin(), {6, 20, 20}, float32), {2, 3, 20, 20});
    const array b = ones({2, 3, 20}, float32);
    array x = solve_triangular_accelerated(A, b);
    eval({x});
    expect("batch axes, a vector b", x.shape() == Shape{2, 3, 20});
    const array Ac = reshape(clean(av, 6, 20, false, false), {2, 3, 20, 20});
    const array Ax = squeeze(matmul(Ac, expand_dims(x, -1), Device::cpu), -1);
    expect("vector residual", max_abs(subtract(Ax, b, Device::cpu)) < 1e-4f);
    array e = solve_triangular_accelerated(zeros({0, 4, 4}, float32), zeros({0, 4, 2}, float32));
    eval({e});
    expect("empty batch", e.shape() == Shape{0, 4, 2});
    bool threw = false;
    try { solve_triangular_accelerated(zeros({4, 5}, float32), zeros({4, 1}, float32)); }
    catch (const std::invalid_argument&) { threw = true; }
    expect("non-square throws", threw);
    threw = false;
    try { solve_triangular_accelerated(A, zeros({2, 3, 19, 2}, float32)); } catch (const std::invalid_argument&) { threw = true; }
    expect("mismatched b throws", threw);
}

void run_routing() {
    std::printf("\n[ routing policy ]\n");
    const TrsmPolicy original = trsm_policy();
    TrsmPolicy p = original;
    p.gpu_min_n = 0;
    set_trsm_policy(p);
    expect("gpu_min_n = 0: never", trsm_backend(8192, 8192, 1) == TrsmBackend::cpu);
    expect("source user", std::string(trsm_policy_source()) == "user");
    p.gpu_min_n = 256;
    p.gpu_min_rhs = 64;
    p.gpu_max_batch = 2;
    set_trsm_policy(p);
    expect("256, 64 rhs, batch 2 -> blocked", trsm_backend(256, 64, 2) == TrsmBackend::blocked);
    expect("63 rhs -> cpu", trsm_backend(256, 63, 2) == TrsmBackend::cpu);
    expect("batch 3 -> cpu", trsm_backend(256, 64, 3) == TrsmBackend::cpu);
    setenv("TRSM_DEVICE", "gpu", 1);
    expect("TRSM_DEVICE=gpu", trsm_backend(4, 1, 100) == TrsmBackend::blocked);
    setenv("TRSM_DEVICE", "cpu", 1);
    expect("TRSM_DEVICE=cpu", trsm_backend(8192, 8192, 1) == TrsmBackend::cpu);
    unsetenv("TRSM_DEVICE");
    set_trsm_policy(original);
}

} // namespace

int main() {
    run_backends();
    run_shapes();
    run_routing();
    std::printf("\n%d checks, %d failures\n", g_checks, g_failures);
    return g_failures ? 1 : 0;
}
