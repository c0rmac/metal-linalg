// Correctness tests for LU, solve and inverse: each backend directly (cpu,
// blocked) and through the routing. lu_factor's P A = L U with |L| <= 1, its
// pivots a valid sequence of swaps; solve's residual for one and many
// right-hand sides (either side of the blocked path's switch to the GPU's
// triangular solves) and for a vector b; inv's A A^-1 = I; agreement with
// MLX's own CPU functions; a singular matrix's info and NaN results, alone in
// its batch; shapes, empties and errors; the routing policy.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include <mlx/linalg.h>
#include <mlx/mlx.h>

#include <metal_linalg/lu.h>

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

// Gaussian matrices, plus `diag` on the diagonal (0: general; a few times
// sqrt(n): well conditioned).
std::vector<float> matrices(int batch, int n, unsigned seed, float diag = 0.0f) {
    std::mt19937 g(seed);
    std::normal_distribution<float> nd;
    std::vector<float> v((size_t)batch * n * n);
    for (auto& x : v) x = nd(g);
    for (int b = 0; b < batch; ++b)
        for (int i = 0; i < n; ++i) v[(size_t)b * n * n + (size_t)i * n + i] += diag;
    return v;
}

array as_array(const std::vector<float>& v, Shape s) { return array(v.begin(), s, float32); }

array mm(const array& a, const array& b) { return matmul(a, b, Device::cpu); }

float max_abs(const array& x) {
    array m = max(abs(x, Device::cpu), Device::cpu);
    eval({m});
    return m.item<float>();
}

array transpose_last(const array& x) {
    std::vector<int> axes(x.ndim());
    for (size_t i = 0; i < axes.size(); ++i) axes[i] = (int)i;
    std::swap(axes[axes.size() - 1], axes[axes.size() - 2]);
    return transpose(x, axes, Device::cpu);
}

// P A = L U for one matrix [n, n], from the packed factor and the pivots.
void check_factor(const std::string& label, const float* A, const float* LU, const uint32_t* piv, int n) {
    std::vector<int> perm(n);
    for (int i = 0; i < n; ++i) perm[i] = i;
    bool valid = true;
    for (int k = 0; k < n; ++k) {
        if ((int)piv[k] < k || (int)piv[k] >= n) valid = false;
        else std::swap(perm[k], perm[piv[k]]);
    }
    expect(label + ": pivots are swaps with later rows", valid);
    if (!valid) return;
    double err = 0, scale = 0, lmax = 0;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            double s = 0;
            for (int t = 0; t <= std::min(i, j); ++t) s += (t == i ? 1.0 : (double)LU[i * n + t]) * LU[t * n + j];
            err = std::max(err, std::fabs(s - A[perm[i] * n + j]));
            scale = std::max(scale, (double)std::fabs(A[i * n + j]));
            if (j < i) lmax = std::max(lmax, (double)std::fabs(LU[i * n + j]));
        }
    expect(label + ": P A = L U", err <= 2e-6 * n * scale, fmt("max err / max|A| %.2e", err / scale));
    expect(label + ": |L| <= 1", lmax <= 1.0 + 1e-6, fmt("max |L| %.6f", lmax));
}

using LuF = std::function<LuResult(const array&)>;
using SolveF = std::function<SolveResult(const array&, const array&)>;
using InvF = std::function<SolveResult(const array&)>;

struct Backend {
    const char* name;
    LuF lu;
    SolveF solve;
    InvF inv;
};

std::vector<Backend> backends() {
    return {
        {"cpu", detail::lu_factor_cpu, detail::solve_cpu, detail::inv_cpu},
        {"blocked", detail::lu_factor_blocked, detail::solve_blocked, detail::inv_blocked},
        {"dispatch", lu_factor_ex_accelerated, solve_ex_accelerated, inv_ex_accelerated},
    };
}

void run_backends() {
    std::printf("\n[ every backend: lu_factor, solve, inv ]\n");
    const int cases[][2] = {{1, 1}, {5, 2}, {3, 7}, {4, 64}, {2, 127}, {2, 128}, {2, 129}, {1, 300},
                            {1, 520}, {1, 1100}};
    unsigned seed = 1;
    for (const auto& c : cases) {
        const int batch = c[0], n = c[1];
        const std::vector<float> v = matrices(batch, n, seed++);
        const array a = as_array(v, {batch, n, n});
        const std::vector<float> w = matrices(batch, n, seed++, 3.0f * std::sqrt((float)n));
        const array aw = as_array(w, {batch, n, n});
        for (const auto& be : backends()) {
            const std::string tag = std::string(be.name) + " " + std::to_string(batch) + " x " + std::to_string(n);
            LuResult r = be.lu(a);
            eval({r.lu, r.pivots, r.info});
            expect(tag + ": shapes", r.lu.shape() == Shape{batch, n, n} && r.pivots.shape() == Shape{batch, n} &&
                                         r.pivots.dtype() == uint32 && r.info.shape() == Shape{batch});
            for (int b = 0; b < batch; ++b)
                check_factor(tag + " #" + std::to_string(b), v.data() + (size_t)b * n * n,
                             r.lu.data<float>() + (size_t)b * n * n, r.pivots.data<uint32_t>() + (size_t)b * n, n);
            bool info0 = true;
            for (int b = 0; b < batch; ++b) info0 = info0 && r.info.data<uint32_t>()[b] == 0;
            expect(tag + ": info 0", info0);
            // solve: well conditioned, so the residual is the test
            for (int k : {1, 3, 16, 40}) {
                std::mt19937 g(seed + k);
                std::normal_distribution<float> nd;
                std::vector<float> rhs((size_t)batch * n * k);
                for (auto& x : rhs) x = nd(g);
                const array B = as_array(rhs, {batch, n, k});
                SolveResult s = be.solve(aw, B);
                eval({s.x, s.info});
                const float res = max_abs(subtract(mm(aw, s.x), B, Device::cpu)) / std::max(max_abs(B), 1e-30f);
                expect(tag + " solve k=" + std::to_string(k), s.x.shape() == B.shape() && res < 1e-4f,
                       fmt("residual %.2e", res));
            }
            {   // a vector b: x [batch, n]
                std::vector<float> rhs((size_t)batch * n, 1.0f);
                const array B = as_array(rhs, {batch, n});
                SolveResult s = be.solve(aw, B);
                eval({s.x});
                const array Ax = squeeze(mm(aw, expand_dims(s.x, -1)), -1);
                const float res = max_abs(subtract(Ax, B, Device::cpu));
                expect(tag + " solve vector", s.x.shape() == B.shape() && res < 1e-4f, fmt("residual %.2e", res));
            }
            SolveResult iv = be.inv(aw);
            eval({iv.x});
            std::vector<float> eye((size_t)n * n, 0.0f);
            for (int i = 0; i < n; ++i) eye[(size_t)i * n + i] = 1.0f;
            const float ie = max_abs(subtract(mm(aw, iv.x), as_array(eye, {n, n}), Device::cpu));
            expect(tag + " inv: A A^-1 = I", iv.x.shape() == aw.shape() && ie < 1e-4f, fmt("max err %.2e", ie));
        }
        // against MLX's own (CPU)
        const array mx_inv = linalg::inv(aw, Device::cpu);
        const array ours = inv_accelerated(aw);
        const float d = max_abs(subtract(ours, mx_inv, Device::cpu)) / std::max(max_abs(mx_inv), 1e-30f);
        expect("inv " + std::to_string(n) + " agrees with mx::linalg::inv", d < 1e-4f, fmt("rel diff %.2e", d));
    }
}

void run_singular() {
    std::printf("\n[ singular matrices ]\n");
    for (int n : {6, 140, 300}) {
        // three matrices; the middle one with a zero column at index n/2
        std::vector<float> v = matrices(3, n, 50 + n, 3.0f * std::sqrt((float)n));
        for (int i = 0; i < n; ++i) v[(size_t)n * n + (size_t)i * n + n / 2] = 0.0f;
        const array a = as_array(v, {3, n, n});
        std::vector<float> rhs((size_t)3 * n * 2, 1.0f);
        const array B = as_array(rhs, {3, n, 2});
        for (const auto& be : backends()) {
            const std::string tag = std::string(be.name) + " n=" + std::to_string(n);
            LuResult r = be.lu(a);
            eval({r.info});
            const uint32_t* info = r.info.data<uint32_t>();
            expect(tag + ": lu_factor info", info[0] == 0 && info[1] > 0 && info[2] == 0,
                   fmt("info[1] %.0f", info[1]));
            SolveResult s = be.solve(a, B);
            eval({s.x, s.info});
            const float* x = s.x.data<float>();
            bool nan1 = true, fin = true;
            for (int e = 0; e < n * 2; ++e) {
                nan1 = nan1 && std::isnan(x[n * 2 + e]);
                fin = fin && std::isfinite(x[e]) && std::isfinite(x[2 * n * 2 + e]);
            }
            expect(tag + ": solve NaN for it alone", nan1 && fin && s.info.data<uint32_t>()[1] > 0);
            SolveResult iv = be.inv(a);
            eval({iv.x, iv.info});
            bool inan = true;
            for (int e = 0; e < n * n; ++e) inan = inan && std::isnan(iv.x.data<float>()[(size_t)n * n + e]);
            expect(tag + ": inv NaN for it alone", inan && iv.info.data<uint32_t>()[0] == 0);
        }
    }
}

void run_shapes() {
    std::printf("\n[ shapes, empties, errors ]\n");
    const std::vector<float> v = matrices(6, 9, 3, 9.0f);
    const array a = reshape(as_array(v, {6, 9, 9}), {2, 3, 9, 9});
    auto [lu, piv] = lu_factor_accelerated(a);
    eval({lu, piv});
    expect("batch axes: lu_factor", lu.shape() == Shape{2, 3, 9, 9} && piv.shape() == Shape{2, 3, 9});
    array x = solve_accelerated(a, ones({2, 3, 9}, float32));
    eval({x});
    expect("batch axes: vector solve", x.shape() == Shape{2, 3, 9});
    array e = inv_accelerated(zeros({0, 4, 4}, float32));
    eval({e});
    expect("empty batch", e.shape() == Shape{0, 4, 4});
    array z = solve_accelerated(zeros({2, 0, 0}, float32), zeros({2, 0, 3}, float32));
    eval({z});
    expect("0 x 0", z.shape() == Shape{2, 0, 3});
    array t = inv_accelerated(transpose_last(a));   // a transposed view
    eval({t});
    const float d = max_abs(subtract(t, transpose_last(inv_accelerated(a)), Device::cpu));
    expect("a transposed view", d < 1e-4f, fmt("%.2e", d));
    bool threw = false;
    try { inv_accelerated(zeros({3, 4}, float32)); } catch (const std::invalid_argument&) { threw = true; }
    expect("non-square throws", threw);
    threw = false;
    try { solve_accelerated(a, ones({2, 3, 8}, float32)); } catch (const std::invalid_argument&) { threw = true; }
    expect("mismatched b throws", threw);
}

void run_routing() {
    std::printf("\n[ routing policy ]\n");
    const LuPolicy original = lu_policy();
    expect("a policy source", std::string(lu_policy_source()).size() > 0);
    LuPolicy p = original;
    p.gpu_min_n = 0;
    set_lu_policy(p);
    expect("gpu_min_n = 0: never the GPU", lu_backend(8192, 1) == LuBackend::cpu);
    expect("source user", std::string(lu_policy_source()) == "user");
    p.gpu_min_n = 200;
    p.gpu_max_batch = 2;
    set_lu_policy(p);
    expect("200 in a batch of 2 -> blocked", lu_backend(200, 2) == LuBackend::blocked);
    expect("200 in a batch of 3 -> cpu", lu_backend(200, 3) == LuBackend::cpu);
    expect("199 -> cpu", lu_backend(199, 1) == LuBackend::cpu);
    {
        const std::vector<float> v = matrices(2, 300, 9, 50.0f);
        const array a = as_array(v, {2, 300, 300});
        array x = inv_accelerated(a);
        eval({x});
        std::vector<float> eye((size_t)300 * 300, 0.0f);
        for (int i = 0; i < 300; ++i) eye[(size_t)i * 300 + i] = 1.0f;
        expect("routed to blocked, inv", max_abs(subtract(mm(a, x), as_array(eye, {300, 300}), Device::cpu)) < 1e-4f);
    }
    setenv("LU_DEVICE", "cpu", 1);
    expect("LU_DEVICE=cpu", lu_backend(4096, 1) == LuBackend::cpu);
    setenv("LU_DEVICE", "gpu", 1);
    expect("LU_DEVICE=gpu", lu_backend(4, 1) == LuBackend::blocked);
    unsetenv("LU_DEVICE");
    set_lu_policy(original);
}

} // namespace

int main() {
    run_backends();
    run_singular();
    run_shapes();
    run_routing();
    std::printf("\n%d checks, %d failures\n", g_checks, g_failures);
    return g_failures ? 1 : 0;
}
