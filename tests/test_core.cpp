// Correctness tests for the buffer core (core.h), with no MLX: every backend
// called on plain float buffers, checked in double precision.
//
//   QR     reconstruction, orthonormal Q, R exactly upper triangular
//   eigh   residual A V - V diag(w), orthonormal V, ascending w, the unread
//          triangle ignored, values-only equal to values-with-vectors
//   SVD    reconstruction, orthonormal U and Vt, descending non-negative S,
//          values-only equal to values-with-vectors
//   Cholesky  reconstruction, the other triangle exactly zero, info, lower
//          and upper
//   CPU only  set_cpu_only / CpuOnly: every router answers the CPU on this
//          thread, ahead of policies and *_DEVICE, and on no other thread
//
// plus a NaN matrix in a batch (NaN there, nothing elsewhere), input that is
// not page-aligned (the copy path), rank deficiency, and two shapes that pad
// alike in a row (the streaming QR workspace once overran on that). Sizes are
// small: the test checks the plumbing around the kernels, which the MLX
// suites (test_qr, test_eigh, test_svd, test_cholesky) test in depth.

#include <metal_linalg/core.h>
#include <metal_linalg/device.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace metal_linalg;
using core::Matrices;

namespace {

constexpr double kTol = 2e-5;   // relative Frobenius bounds, float32

int g_failures = 0;
int g_checks   = 0;

void report(const std::string& label, const std::string& what, bool ok) {
    ++g_checks;
    if (!ok) {
        ++g_failures;
        std::printf("  FAIL  %-48s %s\n", label.c_str(), what.c_str());
    }
}

void report_value(const std::string& label, const char* name, double value, double tol) {
    char buf[96];
    std::snprintf(buf, sizeof buf, "%s %.2e > %.0e", name, value, tol);
    report(label, buf, value <= tol);
}

std::vector<float> random_matrices(uint32_t batch, uint32_t m, uint32_t n, unsigned seed, float scale = 1.0f) {
    std::mt19937 r(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> v((size_t)batch * m * n);
    for (auto& x : v) x = dist(r) * scale;
    return v;
}

std::vector<float> symmetric_matrices(uint32_t batch, uint32_t n, unsigned seed) {
    std::vector<float> a = random_matrices(batch, n, n, seed);
    for (uint32_t b = 0; b < batch; ++b) {
        float* m = a.data() + (size_t)b * n * n;
        for (uint32_t i = 0; i < n; ++i)
            for (uint32_t j = 0; j < i; ++j) m[(size_t)i * n + j] = m[(size_t)j * n + i];
    }
    return a;
}

// ||X||_F of one row-major matrix.
double frobenius(const float* x, size_t count) {
    double s = 0.0;
    for (size_t i = 0; i < count; ++i) s += (double)x[i] * x[i];
    return std::sqrt(s);
}

// ||A - B C||_F for row-major A (m x n), B (m x k), C (k x n); C scaled by
// column factor d (diag between B and C) when given.
double product_error(const float* a, const float* b, const float* d, const float* c,
                     uint32_t m, uint32_t k, uint32_t n) {
    double err = 0.0;
    for (uint32_t i = 0; i < m; ++i)
        for (uint32_t j = 0; j < n; ++j) {
            double s = 0.0;
            for (uint32_t t = 0; t < k; ++t) s += (double)b[(size_t)i * k + t] * (d ? d[t] : 1.0f) * c[(size_t)t * n + j];
            const double e = a[(size_t)i * n + j] - s;
            err += e * e;
        }
    return std::sqrt(err);
}

// ||X^T X - I||_F / sqrt(k) for X (m x k): orthonormal columns.
double column_orthogonality(const float* x, uint32_t m, uint32_t k) {
    double err = 0.0;
    for (uint32_t p = 0; p < k; ++p)
        for (uint32_t q = 0; q < k; ++q) {
            double s = 0.0;
            for (uint32_t i = 0; i < m; ++i) s += (double)x[(size_t)i * k + p] * x[(size_t)i * k + q];
            const double e = s - (p == q ? 1.0 : 0.0);
            err += e * e;
        }
    return std::sqrt(err / std::max(1u, k));
}

// The same for the rows of X (k x n).
double row_orthogonality(const float* x, uint32_t k, uint32_t n) {
    std::vector<float> t((size_t)n * k);
    for (uint32_t i = 0; i < k; ++i)
        for (uint32_t j = 0; j < n; ++j) t[(size_t)j * k + i] = x[(size_t)i * n + j];
    return column_orthogonality(t.data(), n, k);
}

bool all_finite(const float* x, size_t count) {
    for (size_t i = 0; i < count; ++i) if (!std::isfinite(x[i])) return false;
    return true;
}

bool all_nan(const float* x, size_t count) {
    for (size_t i = 0; i < count; ++i) if (!std::isnan(x[i])) return false;
    return true;
}

std::string dims(uint32_t batch, uint32_t m, uint32_t n) {
    return std::to_string(batch) + "x" + std::to_string(m) + "x" + std::to_string(n);
}

// -----------------------------------------------------------------------------
// QR
// -----------------------------------------------------------------------------

using QrFn = std::function<void(const Matrices&, float*, float*)>;
// The three-argument (reduced) overloads, by name
using QrPtr = void (*)(const Matrices&, float*, float*);

void check_qr(const std::string& name, const QrFn& fn, uint32_t batch, uint32_t m, uint32_t n,
              unsigned seed, float scale = 1.0f) {
    const std::string label = name + " " + dims(batch, m, n) + (scale != 1.0f ? " scaled" : "");
    const uint32_t k = std::min(m, n);
    std::vector<float> a = random_matrices(batch, m, n, seed, scale);
    std::vector<float> q((size_t)batch * m * k, -1.0f), r((size_t)batch * k * n, -1.0f);
    try {
        fn({a.data(), batch, m, n}, q.data(), r.data());
    } catch (const std::exception& e) {
        report(label, std::string("threw: ") + e.what(), false);
        return;
    }
    double recon = 0.0, ortho = 0.0;
    bool upper = true;
    for (uint32_t b = 0; b < batch; ++b) {
        const float* ab = a.data() + (size_t)b * m * n;
        const float* qb = q.data() + (size_t)b * m * k;
        const float* rb = r.data() + (size_t)b * k * n;
        recon = std::max(recon, product_error(ab, qb, nullptr, rb, m, k, n) / frobenius(ab, (size_t)m * n));
        ortho = std::max(ortho, column_orthogonality(qb, m, k));
        for (uint32_t i = 0; i < k; ++i)
            for (uint32_t j = 0; j < std::min(i, n); ++j) upper = upper && rb[(size_t)i * n + j] == 0.0f;
    }
    report_value(label, "recon", recon, kTol);
    report_value(label, "ortho", ortho, kTol);
    report(label, "R not upper triangular", upper);
}

// -----------------------------------------------------------------------------
// eigh
// -----------------------------------------------------------------------------

using EighFn = std::function<void(const Matrices&, bool, float*, float*, uint32_t*)>;

void check_eigh(const std::string& name, const EighFn& fn, uint32_t batch, uint32_t n, unsigned seed) {
    const std::string label = name + " " + dims(batch, n, n);
    std::vector<float> a = symmetric_matrices(batch, n, seed);
    std::vector<float> w((size_t)batch * n), v((size_t)batch * n * n), w_only((size_t)batch * n);
    std::vector<uint32_t> info(batch);
    try {
        fn({a.data(), batch, n, n}, true, w.data(), v.data(), info.data());
        fn({a.data(), batch, n, n}, true, w_only.data(), nullptr, nullptr);
    } catch (const std::exception& e) {
        report(label, std::string("threw: ") + e.what(), false);
        return;
    }
    double resid = 0.0, ortho = 0.0, values = 0.0;
    bool ascending = true, converged = true;
    for (uint32_t b = 0; b < batch; ++b) {
        const float* ab = a.data() + (size_t)b * n * n;
        const float* vb = v.data() + (size_t)b * n * n;
        const float* wb = w.data() + (size_t)b * n;
        // A V - V diag(w), column by column.
        double e2 = 0.0;
        for (uint32_t i = 0; i < n; ++i)
            for (uint32_t j = 0; j < n; ++j) {
                double s = 0.0;
                for (uint32_t t = 0; t < n; ++t) s += (double)ab[(size_t)i * n + t] * vb[(size_t)t * n + j];
                const double e = s - (double)vb[(size_t)i * n + j] * wb[j];
                e2 += e * e;
            }
        const double fa = std::max(1e-30, frobenius(ab, (size_t)n * n));
        resid  = std::max(resid, std::sqrt(e2) / fa);
        ortho  = std::max(ortho, column_orthogonality(vb, n, n));
        for (uint32_t j = 0; j < n; ++j) values = std::max(values, std::fabs((double)w_only[(size_t)b * n + j] - wb[j]) / fa);
        for (uint32_t j = 1; j < n; ++j) ascending = ascending && wb[j - 1] <= wb[j];
        converged = converged && detail::eigh_converged(info[b]);
    }
    report_value(label, "resid", resid, kTol);
    report_value(label, "ortho", ortho, kTol);
    report_value(label, "|w_only - w|", values, kTol);
    report(label, "not ascending", ascending);
    report(label, "info not converged", converged);

    // Only one triangle is read: junk in the other changes nothing.
    for (bool lower : {true, false}) {
        std::vector<float> junk = a;
        for (uint32_t b = 0; b < batch; ++b)
            for (uint32_t i = 0; i < n; ++i)
                for (uint32_t j = 0; j < n; ++j)
                    if (lower ? j > i : j < i) junk[((size_t)b * n + i) * n + j] = 1e3f;
        std::vector<float> wj((size_t)batch * n);
        fn({junk.data(), batch, n, n}, lower, wj.data(), nullptr, nullptr);
        double diff = 0.0;
        for (size_t i = 0; i < wj.size(); ++i) diff = std::max(diff, std::fabs((double)wj[i] - w[i]));
        report_value(label + (lower ? " uplo=L" : " uplo=U"), "|w_junk - w|", diff / std::sqrt((double)n), 1e-4);
    }
}

// -----------------------------------------------------------------------------
// SVD
// -----------------------------------------------------------------------------

using SvdFn = std::function<void(const Matrices&, float*, float*, float*, uint32_t*)>;

void check_svd(const std::string& name, const SvdFn& fn, uint32_t batch, uint32_t m, uint32_t n,
               unsigned seed, bool rank_deficient = false) {
    const std::string label = name + " " + dims(batch, m, n) + (rank_deficient ? " rank-deficient" : "");
    const uint32_t k = std::min(m, n);
    std::vector<float> a = random_matrices(batch, m, n, seed);
    if (rank_deficient) {   // repeat the first two columns, and the first two rows
        for (uint32_t b = 0; b < batch; ++b) {
            float* x = a.data() + (size_t)b * m * n;
            for (uint32_t i = 0; i < m; ++i)
                for (uint32_t j = 2; j < 4 && j < n; ++j) x[(size_t)i * n + j] = x[(size_t)i * n + j - 2];
            for (uint32_t i = 2; i < 4 && i < m; ++i)
                for (uint32_t j = 0; j < n; ++j) x[(size_t)i * n + j] = x[(size_t)(i - 2) * n + j];
        }
    }
    std::vector<float> u((size_t)batch * m * k), s((size_t)batch * k), vt((size_t)batch * k * n), s_only((size_t)batch * k);
    std::vector<uint32_t> info(batch);
    try {
        fn({a.data(), batch, m, n}, u.data(), s.data(), vt.data(), info.data());
        fn({a.data(), batch, m, n}, nullptr, s_only.data(), nullptr, nullptr);
    } catch (const std::exception& e) {
        report(label, std::string("threw: ") + e.what(), false);
        return;
    }
    double recon = 0.0, ortho_u = 0.0, ortho_v = 0.0, values = 0.0;
    bool descending = true, converged = true;
    for (uint32_t b = 0; b < batch; ++b) {
        const float* ab = a.data() + (size_t)b * m * n;
        const float* ub = u.data() + (size_t)b * m * k;
        const float* sb = s.data() + (size_t)b * k;
        const float* vb = vt.data() + (size_t)b * k * n;
        const double fa = frobenius(ab, (size_t)m * n);
        recon   = std::max(recon, product_error(ab, ub, sb, vb, m, k, n) / fa);
        ortho_u = std::max(ortho_u, column_orthogonality(ub, m, k));
        ortho_v = std::max(ortho_v, row_orthogonality(vb, k, n));
        for (uint32_t j = 0; j < k; ++j) values = std::max(values, std::fabs((double)s_only[(size_t)b * k + j] - sb[j]) / fa);
        for (uint32_t j = 0; j < k; ++j) descending = descending && sb[j] >= 0.0f && (j == 0 || sb[j - 1] >= sb[j]);
        converged = converged && detail::svd_converged(info[b]);
    }
    report_value(label, "recon", recon, kTol);
    report_value(label, "ortho U", ortho_u, kTol);
    report_value(label, "ortho Vt", ortho_v, kTol);
    report_value(label, "|s_only - s|", values, kTol);
    report(label, "S not descending and non-negative", descending);
    report(label, "info not converged", converged);
}

// -----------------------------------------------------------------------------
// A NaN matrix stays in its own slot
// -----------------------------------------------------------------------------

void check_nan_eigh(const std::string& name, const EighFn& fn, uint32_t n) {
    const std::string label = name + " NaN in 1 of 3, n=" + std::to_string(n);
    std::vector<float> a = symmetric_matrices(3, n, 77);
    a[(size_t)n * n + 1 + n] = NAN;   // a diagonal entry of matrix 1: read whichever the triangle
    std::vector<float> w((size_t)3 * n), v((size_t)3 * n * n);
    std::vector<uint32_t> info(3);
    try {
        fn({a.data(), 3, n, n}, true, w.data(), v.data(), info.data());
    } catch (const std::exception& e) {
        report(label, std::string("threw: ") + e.what(), false);
        return;
    }
    const size_t nn = (size_t)n * n;
    report(label, "matrix 1 not NaN", all_nan(w.data() + n, n) && all_nan(v.data() + nn, nn));
    report(label, "NaN leaked into matrices 0, 2",
           all_finite(w.data(), n) && all_finite(w.data() + 2 * n, n) &&
           all_finite(v.data(), nn) && all_finite(v.data() + 2 * nn, nn));
    report(label, "info: matrix 1 not flagged non-finite", detail::eigh_nonfinite(info[1]));
}

void check_nan_svd(const std::string& name, const SvdFn& fn, uint32_t m, uint32_t n) {
    const std::string label = name + " NaN in 1 of 3, " + std::to_string(m) + "x" + std::to_string(n);
    const uint32_t k = std::min(m, n);
    std::vector<float> a = random_matrices(3, m, n, 78);
    a[(size_t)m * n + 3] = NAN;
    std::vector<float> u((size_t)3 * m * k), s((size_t)3 * k), vt((size_t)3 * k * n);
    std::vector<uint32_t> info(3);
    try {
        fn({a.data(), 3, m, n}, u.data(), s.data(), vt.data(), info.data());
    } catch (const std::exception& e) {
        report(label, std::string("threw: ") + e.what(), false);
        return;
    }
    report(label, "matrix 1 not NaN", all_nan(s.data() + k, k));
    report(label, "NaN leaked into matrices 0, 2",
           all_finite(s.data(), k) && all_finite(s.data() + 2 * k, k) &&
           all_finite(u.data(), (size_t)m * k) && all_finite(vt.data() + 2 * (size_t)k * n, (size_t)k * n));
    report(label, "info: matrix 1 not flagged non-finite", detail::svd_nonfinite(info[1]));
}

} // namespace

// -----------------------------------------------------------------------------
// Cholesky
// -----------------------------------------------------------------------------

using CholeskyFn = std::function<void(const Matrices&, bool, float*, uint32_t*)>;

// A batch of symmetric positive definite matrices, (M M^T + n I) / n.
std::vector<float> spd_matrices(uint32_t batch, uint32_t n, unsigned seed) {
    const std::vector<float> m = random_matrices(batch, n, n, seed);
    std::vector<float> a((size_t)batch * n * n);
    for (uint32_t b = 0; b < batch; ++b) {
        const float* mb = m.data() + (size_t)b * n * n;
        float* ab = a.data() + (size_t)b * n * n;
        for (uint32_t i = 0; i < n; ++i)
            for (uint32_t j = 0; j < n; ++j) {
                double s = i == j ? n : 0.0;
                for (uint32_t t = 0; t < n; ++t) s += (double)mb[(size_t)i * n + t] * mb[(size_t)j * n + t];
                ab[(size_t)i * n + j] = (float)(s / n);
            }
    }
    return a;
}

void check_cholesky(const std::string& name, const CholeskyFn& fn, uint32_t batch, uint32_t n, unsigned seed) {
    for (bool upper : {false, true}) {
        const std::string label = name + " " + dims(batch, n, n) + (upper ? " upper" : "");
        std::vector<float> a = spd_matrices(batch, n, seed);
        std::vector<float> l((size_t)batch * n * n, -1.0f);
        std::vector<uint32_t> info(batch, 99);
        try {
            fn({a.data(), batch, n, n}, upper, l.data(), info.data());
        } catch (const std::exception& e) {
            report(label, std::string("threw: ") + e.what(), false);
            continue;
        }
        double recon = 0.0;
        bool zero = true, ok = true;
        std::vector<float> t((size_t)n * n);
        for (uint32_t b = 0; b < batch; ++b) {
            const float* lb = l.data() + (size_t)b * n * n;
            for (uint32_t i = 0; i < n; ++i)
                for (uint32_t j = 0; j < n; ++j) {
                    t[(size_t)j * n + i] = lb[(size_t)i * n + j];
                    if (upper ? j < i : j > i) zero = zero && lb[(size_t)i * n + j] == 0.0f;
                }
            const float* ab = a.data() + (size_t)b * n * n;
            const double e = upper ? product_error(ab, t.data(), nullptr, lb, n, n, n)
                                   : product_error(ab, lb, nullptr, t.data(), n, n, n);
            recon = std::max(recon, e / frobenius(ab, (size_t)n * n));
            ok = ok && info[b] == 0;
        }
        report_value(label, "recon", recon, kTol);
        report(label, "the other triangle not zero", zero);
        report(label, "info not 0", ok);
    }
}

// The second of three matrices is not positive definite at its third pivot:
// info 3 there, NaN for it alone.
void check_not_pd(const std::string& name, const CholeskyFn& fn, uint32_t n) {
    const std::string label = name + " not positive definite, " + dims(3, n, n);
    std::vector<float> a = spd_matrices(3, n, 60 + n);
    float* bad = a.data() + (size_t)n * n;
    std::fill(bad, bad + (size_t)n * n, 0.0f);
    for (uint32_t i = 0; i < n; ++i) bad[(size_t)i * n + i] = i == 2 ? -1.0f : 1.0f;
    std::vector<float> l((size_t)3 * n * n);
    std::vector<uint32_t> info(3, 99);
    fn({a.data(), 3, n, n}, false, l.data(), info.data());
    report(label, "info " + std::to_string(info[0]) + " " + std::to_string(info[1]) + " " + std::to_string(info[2]),
           info[0] == 0 && info[1] == 3 && info[2] == 0);
    report(label, "matrix 1 not NaN", all_nan(l.data() + (size_t)n * n, (size_t)n * n));
    report(label, "NaN leaked into matrices 0, 2",
           all_finite(l.data(), (size_t)n * n) && all_finite(l.data() + 2 * (size_t)n * n, (size_t)n * n));
}

int main() {
    namespace cd = core::detail;
    std::printf("\ncore (buffer API) correctness tests, %s\n", device_name());

    std::printf("\n[ QR ]\n");
    check_qr("qr_unblocked", QrPtr(cd::qr_unblocked), 3, 20, 12, 1);
    check_qr("qr_unblocked", QrPtr(cd::qr_unblocked), 2, 12, 20, 2);
    check_qr("qr_unblocked", QrPtr(cd::qr_unblocked), 1, 1, 1, 3);
    check_qr("qr_unblocked", QrPtr(cd::qr_unblocked), 2, 16, 16, 4, 1e-6f);
    check_qr("qr_unblocked", QrPtr(cd::qr_unblocked), 2, 16, 16, 5, 1e20f);
    check_qr("qr_streaming_amx_reduced", QrPtr(cd::qr_streaming_amx_reduced), 2, 100, 40, 6);
    check_qr("qr_streaming_amx_reduced", QrPtr(cd::qr_streaming_amx_reduced), 1, 40, 100, 7);
    // These two pad to the same 64 x 64: the second must not reuse the first's buffers.
    check_qr("qr_streaming_amx_reduced", QrPtr(cd::qr_streaming_amx_reduced), 1, 60, 60, 8);
    check_qr("qr_streaming_amx_reduced", QrPtr(cd::qr_streaming_amx_reduced), 1, 64, 64, 9);
    check_qr("qr_streaming_amx_complete", cd::qr_streaming_amx_complete, 2, 70, 40, 10);
    check_qr("qr_streaming_amx_complete", cd::qr_streaming_amx_complete, 1, 60, 60, 11);
    check_qr("qr_streaming_amx_complete", cd::qr_streaming_amx_complete, 1, 64, 64, 12);
    check_qr("qr_cpu (LAPACK)", QrPtr(cd::qr_cpu), 3, 20, 12, 14);
    check_qr("qr_cpu (LAPACK)", QrPtr(cd::qr_cpu), 2, 12, 20, 15);
    check_qr("qr_cpu (LAPACK)", QrPtr(cd::qr_cpu), 1, 1, 1, 16);
    check_qr("qr_cpu (LAPACK)", QrPtr(cd::qr_cpu), 2, 16, 16, 17, 1e-6f);
    check_qr("qr_cpu (LAPACK)", QrPtr(cd::qr_cpu), 1, 300, 40, 18);
    check_qr("core::qr", QrPtr(core::qr), 4, 30, 30, 13);
    {   // A NaN gives NaN for its matrix alone.
        const std::string label = "qr_cpu (LAPACK) NaN in 1 of 3, 10x6";
        std::vector<float> a = random_matrices(3, 10, 6, 79);
        a[60 + 7] = NAN;
        std::vector<float> q(3 * 10 * 6), r(3 * 6 * 6);
        cd::qr_cpu({a.data(), 3, 10, 6}, q.data(), r.data());
        report(label, "matrix 1 not NaN", all_nan(q.data() + 60, 60) && all_nan(r.data() + 36, 36));
        report(label, "NaN leaked into matrices 0, 2",
               all_finite(q.data(), 60) && all_finite(q.data() + 120, 60) &&
               all_finite(r.data(), 36) && all_finite(r.data() + 72, 36));
    }

    std::printf("\n[ eigh ]\n");
    auto with_mode = [](EighOptions::Mode mode) {
        return [mode](const Matrices& a, bool lower, float* w, float* v, uint32_t* info) {
            EighOptions opt;
            opt.mode = mode;
            if (mode == EighOptions::Mode::block) core::detail::eigh_block_jacobi(a, lower, opt, w, v, info);
            else                                  core::detail::eigh_jacobi(a, lower, opt, w, v, info);
        };
    };
    check_eigh("eigh_jacobi simd", with_mode(EighOptions::Mode::simd), 64, 6, 20);
    check_eigh("eigh_jacobi threadgroup", with_mode(EighOptions::Mode::threadgroup), 8, 33, 21);
    check_eigh("eigh_block_jacobi", with_mode(EighOptions::Mode::block), 2, 70, 22);
    check_eigh("eigh_cpu (LAPACK)", cd::eigh_cpu, 3, 30, 23);
    check_eigh("eigh_cpu (LAPACK)", cd::eigh_cpu, 2, 1, 24);
    check_eigh("core::eigh", [](const Matrices& a, bool l, float* w, float* v, uint32_t* i) { core::eigh(a, l, w, v, i); },
               4, 10, 25);
    check_nan_eigh("eigh_jacobi", with_mode(EighOptions::Mode::threadgroup), 12);
    check_nan_eigh("eigh_block_jacobi", with_mode(EighOptions::Mode::block), 40);
    check_nan_eigh("eigh_cpu (LAPACK)", cd::eigh_cpu, 12);

    std::printf("\n[ SVD ]\n");
    auto sv = [](void (*f)(const Matrices&, const SvdOptions&, float*, float*, float*, uint32_t*),
                 SvdOptions::Kernel kernel = SvdOptions::Kernel::automatic) {
        return [f, kernel](const Matrices& a, float* u, float* s, float* vt, uint32_t* info) {
            SvdOptions opt;
            opt.kernel = kernel;
            f(a, opt, u, s, vt, info);
        };
    };
    check_svd("svd_jacobi", sv(cd::svd_jacobi), 4, 30, 12, 30);
    check_svd("svd_jacobi", sv(cd::svd_jacobi), 3, 12, 30, 31);
    check_svd("svd_jacobi", sv(cd::svd_jacobi), 2, 20, 8, 32, true);
    check_svd("svd_block_jacobi", sv(cd::svd_block_jacobi), 2, 48, 40, 33);
    check_svd("svd_block_jacobi", sv(cd::svd_block_jacobi), 2, 40, 48, 34);
    check_svd("svd_block_jacobi", sv(cd::svd_block_jacobi), 2, 40, 24, 35, true);
    check_svd("svd_qr_jacobi (jacobi)", sv(cd::svd_qr_jacobi, SvdOptions::Kernel::jacobi), 2, 200, 24, 36);
    check_svd("svd_qr_jacobi (block)", sv(cd::svd_qr_jacobi, SvdOptions::Kernel::block), 2, 160, 40, 37);
    check_svd("svd_qr_jacobi (jacobi)", sv(cd::svd_qr_jacobi, SvdOptions::Kernel::jacobi), 2, 24, 200, 38);
    check_svd("svd_cpu (LAPACK)", cd::svd_cpu, 3, 30, 12, 39);
    check_svd("svd_cpu (LAPACK)", cd::svd_cpu, 3, 12, 30, 40);
    check_svd("svd_cpu (LAPACK)", cd::svd_cpu, 1, 1, 1, 41);
    check_svd("svd_cpu (LAPACK)", cd::svd_cpu, 2, 20, 8, 42, true);
    check_svd("core::svd", [](const Matrices& a, float* u, float* s, float* vt, uint32_t* i) { core::svd(a, u, s, vt, i); },
              4, 25, 10, 43);
    check_nan_svd("svd_jacobi", sv(cd::svd_jacobi), 10, 6);
    check_nan_svd("svd_block_jacobi", sv(cd::svd_block_jacobi), 40, 34);
    check_nan_svd("svd_cpu (LAPACK)", cd::svd_cpu, 10, 6);

    std::printf("\n[ Cholesky ]\n");
    check_cholesky("cholesky_simd", cd::cholesky_simd, 64, 12, 44);
    check_cholesky("cholesky_simd", cd::cholesky_simd, 9, 32, 45);
    check_cholesky("cholesky_threadgroup", cd::cholesky_threadgroup, 5, 70, 46);
    check_cholesky("cholesky_blocked", cd::cholesky_blocked, 1, 300, 47);
    check_cholesky("cholesky_cpu (LAPACK)", cd::cholesky_cpu, 3, 40, 48);
    check_cholesky("cholesky_cpu (LAPACK)", cd::cholesky_cpu, 2, 1, 49);
    check_cholesky("core::cholesky", core::cholesky, 4, 20, 50);
    check_not_pd("cholesky_simd", cd::cholesky_simd, 8);
    check_not_pd("cholesky_threadgroup", cd::cholesky_threadgroup, 40);
    check_not_pd("cholesky_blocked", cd::cholesky_blocked, 200);
    check_not_pd("cholesky_cpu (LAPACK)", cd::cholesky_cpu, 40);

    std::printf("\n[ LU, solve, inverse ]\n");
    for (auto [name, blocked] : {std::pair<const char*, bool>{"cpu", false}, {"blocked", true}}) {
        const uint32_t n = 150, batch = 2, k = 20;
        std::vector<float> a = random_matrices(batch, n, n, 61);
        for (uint32_t b = 0; b < batch; ++b)
            for (uint32_t i = 0; i < n; ++i) a[(size_t)b * n * n + i * n + i] += 40.0f;
        std::vector<float> rhs = random_matrices(batch, n, k, 62), x((size_t)batch * n * k), inv((size_t)batch * n * n);
        std::vector<uint32_t> info(batch, 9);
        const Matrices m{a.data(), batch, n, n};
        if (blocked) cd::solve_blocked(m, rhs.data(), k, x.data(), info.data());
        else cd::solve_cpu(m, rhs.data(), k, x.data(), info.data());
        double res = 0;
        for (uint32_t b = 0; b < batch; ++b)
            res = std::max(res, product_error(rhs.data() + (size_t)b * n * k, a.data() + (size_t)b * n * n, nullptr,
                                              x.data() + (size_t)b * n * k, n, n, k) /
                                    frobenius(rhs.data() + (size_t)b * n * k, (size_t)n * k));
        report_value(std::string("solve_") + name, "residual", res, kTol);
        report(std::string("solve_") + name, "info not 0", info[0] == 0 && info[1] == 0);
        if (blocked) cd::inv_blocked(m, inv.data(), info.data());
        else cd::inv_cpu(m, inv.data(), info.data());
        std::vector<float> eye((size_t)n * n, 0.0f);
        for (uint32_t i = 0; i < n; ++i) eye[(size_t)i * n + i] = 1.0f;
        double ie = 0;
        for (uint32_t b = 0; b < batch; ++b)
            ie = std::max(ie, product_error(eye.data(), a.data() + (size_t)b * n * n, nullptr,
                                            inv.data() + (size_t)b * n * n, n, n, n) / std::sqrt((double)n));
        report_value(std::string("inv_") + name, "A A^-1 - I", ie, kTol);
    }

    std::printf("\n[ input that is not page-aligned ]\n");
    {
        std::vector<float> storage(1 + 3 * 20 * 12);
        std::vector<float> a = random_matrices(3, 20, 12, 50);
        std::copy(a.begin(), a.end(), storage.begin() + 1);
        check_qr("qr_unblocked from data+1", [&](const Matrices&, float* q, float* r) {
            cd::qr_unblocked({storage.data() + 1, 3, 20, 12}, q, r);
        }, 3, 20, 12, 50);
    }

    // Calibration: each policy source names its state, and the notice text
    // agrees with it -- on any Mac, current, stale, incomplete, estimated or untuned.
    std::printf("\n[ calibration ]\n");
    {
        set_calibration_notices(false);
        const struct { const char* what; const char* source; } solvers[] = {
            {"QR", qr_policy_source()}, {"eigh", eigh_policy_source()}, {"SVD", svd_policy_source()},
            {"Cholesky", cholesky_policy_source()}, {"LU", lu_policy_source()},
            {"triangular solve", trsm_policy_source()}};
        for (const auto& s : solvers) {
            const std::string src = s.source, msg = calibration_message(s.what);
            const bool untuned = src.rfind("default:untuned-device", 0) == 0 ||
                                 src.rfind("estimated:", 0) == 0;
            const bool stale = src.rfind("tuned-stale:", 0) == 0;
            const bool incomplete = src.rfind("tuned-incomplete:", 0) == 0;
            const bool current = src.rfind("tuned:", 0) == 0;
            const bool has_device = device_name()[0] != '\0';
            bool ok = untuned || stale || incomplete || current || src.rfind("env:", 0) == 0;
            if (current) ok = ok && msg.empty();
            if (has_device && untuned) ok = ok && msg.find("is not calibrated") != std::string::npos;
            if (has_device && stale) ok = ok && msg.find("older kernels") != std::string::npos;
            if (has_device && incomplete) ok = ok && msg.find("newer optimisations") != std::string::npos;
            if (!msg.empty()) ok = ok && msg.find("CONTRIBUTING.md") != std::string::npos &&
                                    msg.find("METAL_LINALG_NO_CALIBRATION_NOTICE") != std::string::npos;
            report(std::string(s.what) + " source and notice agree", src + " / " + msg, ok);
            std::printf("  %s  %-6s %s\n", ok ? "ok  " : "FAIL", s.what, src.c_str());
        }
        report("calibration_notices() reflects the switch", "", !calibration_notices());
        set_calibration_notices(true);
        report("calibration_notices() back on", "", calibration_notices());
    }

    std::printf("\n[ CPU only, this thread ]\n");
    {
        // A Cholesky policy and *_DEVICE settings that send every call of
        // these shapes to the GPU, which CPU only must override
        const CholeskyPolicy measured = cholesky_policy();
        CholeskyPolicy all = measured;
        all.gpu_max_n = 0xFFFFFFFFu;
        all.gpu_min_batch_times_n = 0;
        all.gpu_min_batch = 1;
        all.gpu_min_n = 0;
        set_cholesky_policy(all);
        const char* vars[] = {"QR_DEVICE", "EIGH_DEVICE", "SVD_DEVICE", "LU_DEVICE", "TRSM_DEVICE"};
        for (const char* v : vars) setenv(v, "gpu", 1);
        auto all_cpu = [] {
            return qr_backend(64, 64, 4096) == QrBackend::cpu && eigh_backend(32, 4096) == EighBackend::cpu &&
                   eigvalsh_backend(4096, 1) == EighBackend::cpu && svd_backend(1024, 64, 64) == SvdBackend::cpu &&
                   svdvals_backend(4096, 4096, 1) == SvdBackend::cpu &&
                   cholesky_backend(300, 2) == CholeskyBackend::cpu && lu_backend(64, 1) == LuBackend::cpu &&
                   trsm_backend(64, 64, 1) == TrsmBackend::cpu;
        };
        auto none_cpu = [] {
            return qr_backend(64, 64, 4096) != QrBackend::cpu && eigh_backend(32, 4096) != EighBackend::cpu &&
                   svd_backend(1024, 64, 64) != SvdBackend::cpu && cholesky_backend(300, 2) != CholeskyBackend::cpu &&
                   lu_backend(64, 1) == LuBackend::blocked && trsm_backend(64, 64, 1) == TrsmBackend::blocked;
        };
        report("CPU only is off by default", "", !cpu_only());
        report("the GPU without it, under these settings", "", none_cpu());
        {
            CpuOnly scope;
            report("on in a CpuOnly scope", "", cpu_only());
            report("every router answers the CPU", "", all_cpu());
            check_cholesky("cholesky routed, CPU only", core::cholesky, 2, 300, 91);
            bool other_on = true, other_gpu = false;
            std::thread t([&] {
                other_on = cpu_only();
                other_gpu = cholesky_backend(300, 2) != CholeskyBackend::cpu;
            });
            t.join();
            report("another thread is not affected", "", !other_on && other_gpu);
            {
                CpuOnly off(false);
                report("CpuOnly(false) inside turns it off", "", !cpu_only() && none_cpu());
            }
            report("and the scope's setting is back after it", "", cpu_only());
        }
        report("off again after the scope", "", !cpu_only() && none_cpu());
        set_cpu_only(true);
        report("set_cpu_only(true)", "", cpu_only() && all_cpu());
        set_cpu_only(false);
        report("set_cpu_only(false)", "", !cpu_only() && none_cpu());
        for (const char* v : vars) unsetenv(v);
        set_cholesky_policy(measured);
        std::printf("  %s\n", g_failures ? "see FAIL lines" : "ok    routers, scopes and threads");
    }

    std::printf("\n%d checks, %d failed\n", g_checks, g_failures);
    return g_failures ? 1 : 0;
}
