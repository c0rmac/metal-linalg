// Correctness tests for Cholesky: every backend directly (simd, threadgroup,
// blocked, cpu) and through the routing, lower and upper. Each factorisation
// is checked for its shape, its reconstruction (L L^T == A), an exactly zero
// other triangle, a positive diagonal, info == 0, and agreement with LAPACK;
// a matrix that is not positive definite for its exact info and an all-NaN
// result that leaves the rest of its batch alone.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include <mlx/mlx.h>

#include <metal_linalg/cholesky.h>

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

// A batch of well-conditioned symmetric positive definite matrices,
// (M M^T + n I) / n, as float32 row-major.
std::vector<float> spd(int batch, int n, unsigned seed, float scale = 1.0f) {
    std::mt19937 g(seed);
    std::normal_distribution<float> nd;
    std::vector<float> out((size_t)batch * n * n), m((size_t)n * n);
    for (int b = 0; b < batch; ++b) {
        for (auto& v : m) v = nd(g);
        float* a = out.data() + (size_t)b * n * n;
        for (int i = 0; i < n; ++i)
            for (int j = 0; j <= i; ++j) {
                double s = 0;
                for (int t = 0; t < n; ++t) s += (double)m[(size_t)i * n + t] * m[(size_t)j * n + t];
                if (i == j) s += n;
                a[(size_t)i * n + j] = a[(size_t)j * n + i] = (float)(s / n) * scale;
            }
    }
    return out;
}

array as_array(const std::vector<float>& v, int batch, int n) {
    if (batch == 1) return array(v.begin(), {n, n}, float32);
    return array(v.begin(), {batch, n, n}, float32);
}

using Fn = std::function<CholeskyResult(const array&, bool)>;

// The checks on one factorisation of `a` (the batch `v` as floats).
void check(const std::string& label, const Fn& fn, const std::vector<float>& v, int batch, int n, bool upper,
           const std::vector<float>* reference = nullptr) {
    const array a = as_array(v, batch, n);
    CholeskyResult r = fn(a, upper);
    eval({r.l, r.info});
    const bool shape_ok = r.l.shape() == a.shape() && r.info.size() == (size_t)batch;
    expect(label + ": shape", shape_ok);
    if (!shape_ok || n == 0 || batch == 0) return;
    const float* l = r.l.data<float>();
    const uint32_t* info = r.info.data<uint32_t>();
    double worst = 0, other = 0, ref = 0;
    bool diag_ok = true, info_ok = true;
    for (int b = 0; b < batch; ++b) {
        const float* L = l + (size_t)b * n * n;
        const float* A = v.data() + (size_t)b * n * n;
        if (info[b] != 0) info_ok = false;
        double num = 0, den = 0;
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) {
                double s = 0;
                for (int t = 0; t < n; ++t)
                    s += upper ? (double)L[(size_t)t * n + i] * L[(size_t)t * n + j]
                               : (double)L[(size_t)i * n + t] * L[(size_t)j * n + t];
                num += (s - A[(size_t)i * n + j]) * (s - A[(size_t)i * n + j]);
                den += (double)A[(size_t)i * n + j] * A[(size_t)i * n + j];
                if (upper ? j < i : j > i) other = std::max(other, (double)std::fabs(L[(size_t)i * n + j]));
                if (reference) {
                    const double d = std::fabs((double)L[(size_t)i * n + j] - (*reference)[(size_t)b * n * n + i * n + j]);
                    ref = std::max(ref, d / std::max(1e-30, std::sqrt(std::fabs((double)A[(size_t)i * n + i]))));
                }
            }
        for (int i = 0; i < n; ++i)
            if (!(L[(size_t)i * n + i] > 0)) diag_ok = false;
        worst = std::max(worst, std::sqrt(num / std::max(den, 1e-300)));
    }
    char buf[96];
    std::snprintf(buf, sizeof buf, "rel err %.2e", worst);
    expect(label + ": L L^T = A", worst < 2e-6 * std::max(1.0, std::sqrt((double)n)), buf);
    std::snprintf(buf, sizeof buf, "max %.2e", other);
    expect(label + ": other triangle zero", other == 0, buf);
    expect(label + ": positive diagonal", diag_ok);
    expect(label + ": info 0", info_ok);
    if (reference) {
        std::snprintf(buf, sizeof buf, "max |L - L_cpu| / sqrt(a_ii) %.2e", ref);
        expect(label + ": agrees with LAPACK", ref < 1e-4, buf);
    }
}

// The CPU path's L for the same batch, the reference for the others.
std::vector<float> reference_of(const std::vector<float>& v, int batch, int n, bool upper) {
    CholeskyResult r = detail::cholesky_cpu(as_array(v, batch, n), upper);
    eval({r.l});
    return std::vector<float>(r.l.data<float>(), r.l.data<float>() + r.l.size());
}

struct Backend {
    const char* name;
    Fn fn;
    int max_n;   // the largest n the case list gives it
};

void run_backends() {
    std::printf("\n[ every backend, lower and upper, against LAPACK ]\n");
    const std::vector<Backend> backends = {
        {"simd", [](const array& a, bool u) { return detail::cholesky_simd(a, u); }, 32},
        {"threadgroup", [](const array& a, bool u) { return detail::cholesky_threadgroup(a, u); }, 520},
        {"blocked", [](const array& a, bool u) { return detail::cholesky_blocked(a, u); }, 1300},
        {"cpu", [](const array& a, bool u) { return detail::cholesky_cpu(a, u); }, 1300},
        {"dispatch", [](const array& a, bool u) { return cholesky_ex_accelerated(a, u); }, 1300},
    };
    const int cases[][2] = {{1, 1}, {7, 2}, {33, 3}, {5, 8},  {9, 12},  {64, 16},  {3, 17},  {40, 24},  {2, 31},
                            {100, 32}, {4, 33}, {3, 48}, {9, 64}, {2, 100}, {3, 128}, {2, 129}, {2, 200},
                            {1, 256}, {1, 300}, {1, 520}, {1, 700}, {1, 1029}, {1, 1300}};
    unsigned seed = 1;
    for (auto& c : cases) {
        const int batch = c[0], n = c[1];
        const std::vector<float> v = spd(batch, n, seed++);
        for (bool upper : {false, true}) {
            const std::vector<float> ref = reference_of(v, batch, n, upper);
            for (const auto& be : backends) {
                if (n > be.max_n) continue;
                char label[96];
                std::snprintf(label, sizeof label, "%s %d x %dx%d%s", be.name, batch, n, n, upper ? " U" : "");
                check(label, be.fn, v, batch, n, upper, std::string(be.name) == "cpu" ? nullptr : &ref);
            }
        }
    }
}

// A matrix whose leading minor of order k is the first that is not positive
// definite: the identity with -1 at (k-1, k-1).
std::vector<float> not_pd(int n, int k) {
    std::vector<float> a((size_t)n * n, 0.0f);
    for (int i = 0; i < n; ++i) a[(size_t)i * n + i] = 1.0f;
    a[(size_t)(k - 1) * n + (k - 1)] = -1.0f;
    return a;
}

void run_failures() {
    std::printf("\n[ not positive definite, non-finite input ]\n");
    const std::vector<std::pair<const char*, Fn>> backends = {
        {"simd", [](const array& a, bool u) { return detail::cholesky_simd(a, u); }},
        {"threadgroup", [](const array& a, bool u) { return detail::cholesky_threadgroup(a, u); }},
        {"blocked", [](const array& a, bool u) { return detail::cholesky_blocked(a, u); }},
        {"cpu", [](const array& a, bool u) { return detail::cholesky_cpu(a, u); }},
    };
    for (int n : {5, 30, 70, 300}) {
        for (int k : {1, n / 2 + 1, n}) {
            // a batch of three: positive definite, the failing one, positive definite
            std::vector<float> v = spd(3, n, 99 + n);
            const std::vector<float> bad = not_pd(n, k);
            std::copy(bad.begin(), bad.end(), v.begin() + (size_t)n * n);
            for (const auto& be : backends) {
                if (std::string(be.first) == "simd" && n > 32) continue;
                CholeskyResult r = be.second(as_array(v, 3, n), false);
                eval({r.l, r.info});
                const uint32_t* info = r.info.data<uint32_t>();
                const float* l = r.l.data<float>();
                bool nan_all = true, others_finite = true;
                for (size_t e = 0; e < (size_t)n * n; ++e) {
                    if (!std::isnan(l[(size_t)n * n + e])) nan_all = false;
                    if (!std::isfinite(l[e]) || !std::isfinite(l[2 * (size_t)n * n + e])) others_finite = false;
                }
                char label[96];
                std::snprintf(label, sizeof label, "%s n=%d minor %d", be.first, n, k);
                char got[64];
                std::snprintf(got, sizeof got, "info %u %u %u", info[0], info[1], info[2]);
                expect(std::string(label) + ": info", info[0] == 0 && info[1] == (uint32_t)k && info[2] == 0, got);
                expect(std::string(label) + ": its L all NaN", nan_all);
                expect(std::string(label) + ": the others' L finite", others_finite);
            }
        }
        // NaN in the triangle read fails; in the other one it is never read
        for (const auto& be : backends) {
            if (std::string(be.first) == "simd" && n > 32) continue;
            std::vector<float> v = spd(1, n, 7 + n);
            v[(size_t)(n - 1) * n + 0] = NAN;   // lower triangle, last row
            CholeskyResult r = be.second(as_array(v, 1, n), false);
            eval({r.l, r.info});
            expect(std::string(be.first) + " n=" + std::to_string(n) + ": NaN read -> info > 0",
                   r.info.data<uint32_t>()[0] > 0);
            std::vector<float> w = spd(1, n, 7 + n);
            if (n > 1) w[0 * n + (n - 1)] = NAN;   // upper triangle: ignored
            r = be.second(as_array(w, 1, n), false);
            eval({r.l, r.info});
            bool finite = true;
            for (size_t e = 0; e < r.l.size(); ++e) finite = finite && std::isfinite(r.l.data<float>()[e]);
            expect(std::string(be.first) + " n=" + std::to_string(n) + ": NaN in the unread triangle ignored",
                   r.info.data<uint32_t>()[0] == 0 && finite);
            std::vector<float> x = spd(1, n, 8 + n);
            x[0] = INFINITY;
            r = be.second(as_array(x, 1, n), false);
            eval({r.info});
            expect(std::string(be.first) + " n=" + std::to_string(n) + ": an infinite pivot -> info 1",
                   r.info.data<uint32_t>()[0] == 1);
        }
    }
}

void run_inputs() {
    std::printf("\n[ magnitudes, views, empties, the public calls ]\n");
    for (float scale : {1e-30f, 1e-10f, 1e10f, 1e30f}) {
        for (int n : {16, 100, 600}) {
            const std::vector<float> v = spd(2, n, 5, scale);
            char label[64];
            std::snprintf(label, sizeof label, "dispatch scale %.0e n=%d", (double)scale, n);
            check(label, [](const array& a, bool u) { return cholesky_ex_accelerated(a, u); }, v, 2, n, false);
        }
    }
    // a transposed view: the same SPD matrix, so the same L
    {
        const std::vector<float> v = spd(4, 40, 11);
        const array a = as_array(v, 4, 40);
        const array at = transpose(a, {0, 2, 1});
        array l1 = cholesky_accelerated(a), l2 = cholesky_accelerated(at);
        eval({l1, l2});
        array d = max(abs(subtract(l1, l2)));
        eval({d});
        expect("a transposed view", d.item<float>() < 1e-5f);
        // a slice that starts mid-buffer (not page-aligned)
        array s = slice(a, {1, 0, 0}, {3, 40, 40});
        array ls = cholesky_accelerated(s);
        array l1s = slice(l1, {1, 0, 0}, {3, 40, 40});
        array ds = max(abs(subtract(ls, l1s)));
        eval({ds});
        expect("an unaligned slice", ds.item<float>() < 1e-6f);
    }
    // empties and extra batch axes
    {
        array e = cholesky_accelerated(zeros({0, 5, 5}, float32));
        eval({e});
        expect("empty batch", e.shape() == Shape{0, 5, 5});
        array z = cholesky_accelerated(zeros({3, 0, 0}, float32));
        eval({z});
        expect("0 x 0 matrices", z.shape() == Shape{3, 0, 0});
        const std::vector<float> v = spd(6, 8, 3);
        array a = reshape(as_array(v, 6, 8), {2, 3, 8, 8});
        CholeskyResult r = cholesky_ex_accelerated(a);
        eval({r.l, r.info});
        expect("batch axes [2, 3]", r.l.shape() == Shape{2, 3, 8, 8} && r.info.shape() == Shape{2, 3} &&
                                       r.info.dtype() == uint32);
        bool threw = false;
        try { cholesky_accelerated(zeros({4, 5}, float32)); } catch (const std::invalid_argument&) { threw = true; }
        expect("a non-square matrix throws", threw);
    }
}

void run_routing() {
    std::printf("\n[ routing policy ]\n");
    const CholeskyPolicy original = cholesky_policy();
    const std::string source = cholesky_policy_source();
    expect("a policy source", !source.empty(), source);
    CholeskyPolicy p = original;
    p.gpu_max_n = 0;
    p.gpu_large_min_n = 0;
    set_cholesky_policy(p);
    expect("set_cholesky_policy -> source user", std::string(cholesky_policy_source()) == "user");
    expect("never the GPU -> cpu", cholesky_backend(16, 4096) == CholeskyBackend::cpu &&
                                       cholesky_backend(4096, 1) == CholeskyBackend::cpu);
    p.gpu_max_n = kCholeskyNoLimit;
    p.gpu_min_batch_times_n = 0;
    p.gpu_min_batch = 1;
    p.gpu_min_n = 0;
    p.simd_max_n = 32;
    p.blocked_min_n = 512;
    p.blocked_max_batch = 4;
    set_cholesky_policy(p);
    expect("n <= 32 -> simd", cholesky_backend(32, 1) == CholeskyBackend::simd);
    expect("33 -> threadgroup", cholesky_backend(33, 100) == CholeskyBackend::threadgroup);
    expect("512 in a batch of 4 -> blocked", cholesky_backend(512, 4) == CholeskyBackend::blocked);
    expect("512 in a batch of 5 -> threadgroup", cholesky_backend(512, 5) == CholeskyBackend::threadgroup);
    p.simd_max_n = 8;
    set_cholesky_policy(p);
    expect("simd_max_n = 8: 9 -> threadgroup", cholesky_backend(9, 64) == CholeskyBackend::threadgroup);
    p.gpu_max_n = 64;
    p.gpu_large_min_n = 1024;
    p.gpu_large_max_batch = 2;
    set_cholesky_policy(p);
    expect("gpu_max_n = 64: 100 -> cpu", cholesky_backend(100, 64) == CholeskyBackend::cpu);
    expect("the large clause: 1024 in a batch of 2 -> GPU", cholesky_backend(1024, 2) != CholeskyBackend::cpu);
    expect("the large clause: 1024 in a batch of 3 -> cpu", cholesky_backend(1024, 3) == CholeskyBackend::cpu);
    // each routed call still factors correctly
    for (int n : {8, 40, 700, 1100}) {
        const std::vector<float> v = spd(2, n, 21 + n);
        check("routed n=" + std::to_string(n), [](const array& a, bool u) { return cholesky_ex_accelerated(a, u); },
              v, 2, n, false);
    }
    setenv("CHOLESKY_DEVICE", "cpu", 1);
    expect("CHOLESKY_DEVICE=cpu", cholesky_backend(8, 1 << 16) == CholeskyBackend::cpu);
    setenv("CHOLESKY_DEVICE", "blocked", 1);
    expect("CHOLESKY_DEVICE=blocked", cholesky_backend(64, 1) == CholeskyBackend::blocked);
    unsetenv("CHOLESKY_DEVICE");
    set_cholesky_policy(original);
}

} // namespace

int main() {
    run_backends();
    run_failures();
    run_inputs();
    run_routing();
    std::printf("\n%d checks, %d failures\n", g_checks, g_failures);
    return g_failures ? 1 : 0;
}
