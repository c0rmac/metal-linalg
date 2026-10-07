// Correctness tests for the Metal singular value decomposition.
//
// Every decomposition is checked six ways: thin output shapes, finiteness,
// reconstruction ||A - U diag(S) Vt||_F / ||A||_F, orthonormality of U's
// columns and of Vt's rows, and S non-negative and descending. The singular
// values are also compared against MLX's CPU svd (LAPACK), which is an
// independent reference rather than a self-consistency check.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <mlx/mlx.h>
#include <mlx/linalg.h>

#include <metal_linalg/svd.h>

using namespace mlx::core;
using namespace metal_linalg;

namespace {

// Relative Frobenius bounds, float32. One-sided Jacobi is backward stable
// with a modest constant; these hold with a wide margin up to 512.
constexpr float kReconTol = 2e-5f;
constexpr float kOrthoTol = 2e-5f;
constexpr float kSvalTol  = 2e-5f;   // |S - S_lapack| relative to ||A||_F

int g_failures = 0;
int g_checks   = 0;

std::mt19937& rng() { static std::mt19937 r(4321); return r; }

array from_values(std::vector<float> v, Shape shape) {
    return array(v.begin(), std::move(shape), float32);
}

array random_matrix(int batch, int M, int N, unsigned seed) {
    std::mt19937 r(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> data((size_t)batch * M * N);
    for (auto& v : data) v = dist(r);
    if (batch == 1) return from_values(data, {M, N});
    return from_values(data, {batch, M, N});
}

float frobenius(const array& x) {
    array v = sqrt(sum(square(x)));
    eval({v});
    return v.item<float>();
}

float max_abs(const array& x) {
    array m = max(abs(x));
    eval({m});
    return m.item<float>();
}

bool has_non_finite(const array& x) {
    array bad = any(logical_or(isnan(x), isinf(x)));
    eval({bad});
    return bad.item<bool>();
}

array transpose_last_two(const array& x) {
    std::vector<int> axes(x.ndim());
    for (size_t i = 0; i < axes.size(); ++i) axes[i] = (int)i;
    std::swap(axes[axes.size() - 1], axes[axes.size() - 2]);
    return transpose(x, axes);
}

void fail(const std::string& label, const std::string& what) {
    std::printf("  FAIL  %-44s %s\n", label.c_str(), what.c_str());
    ++g_failures;
}

// The full battery on one result.
void check(const std::string& label, const array& A_in, const SvdResult& r_in) {
    ++g_checks;
    array A = astype(A_in, float32);
    const auto& shape = A.shape();
    const int M = shape[shape.size() - 2];
    const int N = shape[shape.size() - 1];
    const int K = std::min(M, N);

    Shape want_u = shape;   want_u.back() = K;
    Shape want_s(shape.begin(), shape.end() - 2);   want_s.push_back(K);
    Shape want_vt = shape;  want_vt[want_vt.size() - 2] = K;
    if (r_in.U.shape()  != want_u)  { fail(label, "U has the wrong shape");  return; }
    if (r_in.S.shape()  != want_s)  { fail(label, "S has the wrong shape");  return; }
    if (r_in.Vt.shape() != want_vt) { fail(label, "Vt has the wrong shape"); return; }

    if (has_non_finite(r_in.U) || has_non_finite(r_in.S) || has_non_finite(r_in.Vt)) {
        fail(label, "output contains NaN/Inf");
        return;
    }

    array info = reshape(r_in.info, {-1});
    eval({info});
    unsigned max_sweeps = 0;
    const int nmat = (int)info.size();
    for (int b = 0; b < nmat; ++b) {
        const unsigned w = info.data<uint32_t>()[b];
        if (!detail::svd_converged(w)) { fail(label, "info says not converged"); return; }
        max_sweeps = std::max(max_sweeps, detail::svd_sweeps(w));
    }

    // Normalise by max|A| first: the checks square entries, and the solver
    // handles magnitudes that ||A||_F in float32 does not.
    SvdResult r = r_in;
    {
        const float amax = max_abs(A);
        if (amax > 0.0f) {
            A   = multiply(A, array(1.0f / amax));
            r.S = multiply(r.S, array(1.0f / amax));
        }
    }

    // S non-negative and descending.
    float order = 0.0f, smin = 0.0f;
    {
        array mn = min(r.S);
        eval({mn});
        smin = mn.item<float>();
        if (K > 1) {
            Shape start(want_s.size(), 0);
            Shape stop = want_s;
            Shape stop_lo = stop;   stop_lo.back() = K - 1;
            Shape start_hi = start; start_hi.back() = 1;
            array d = max(subtract(slice(r.S, start_hi, stop), slice(r.S, start, stop_lo)));
            eval({d});
            order = d.item<float>();   // S[i+1] - S[i], must be <= 0
        }
    }

    const float scale = std::max(frobenius(A), 1e-30f);
    array US = multiply(r.U, expand_dims(r.S, -2));
    const float recon  = frobenius(subtract(matmul(US, r.Vt), A)) / scale;
    const float norm   = std::sqrt((float)K * (float)std::max(1, nmat));
    const float orthoU = frobenius(subtract(matmul(transpose_last_two(r.U), r.U), eye(K))) / norm;
    const float orthoV = frobenius(subtract(matmul(r.Vt, transpose_last_two(r.Vt)), eye(K))) / norm;

    float sval_err = 0.0f;
    {
        std::vector<array> ref = linalg::svd(A, false, Device::cpu);
        array s_ref = ref.back();
        eval({s_ref});
        sval_err = max_abs(subtract(r.S, s_ref)) / scale;
    }

    // Orthogonality is limited by float32 inner products of length max(M, N),
    // whose rounding grows like the square root of their length; the bound is
    // flat up to 256 and follows that growth beyond.
    const float ortho_tol = kOrthoTol * std::max(1.0f, std::sqrt((float)std::max(M, N) / 256.0f));
    const bool ok = smin >= 0.0f && order <= 0.0f && recon <= kReconTol &&
                    orthoU <= ortho_tol && orthoV <= ortho_tol && sval_err <= kSvalTol;
    std::printf("  %s  %-44s recon=%.1e orthoU=%.1e orthoV=%.1e |S-lapack|=%.1e sweeps=%u\n",
                ok ? "ok  " : "FAIL", label.c_str(), recon, orthoU, orthoV, sval_err, max_sweeps);
    if (smin < 0.0f)  std::printf("        negative singular value %.3e\n", smin);
    if (order > 0.0f) std::printf("        singular values not descending (max rise %.3e)\n", order);
    if (!ok) ++g_failures;
}

void run(const std::string& label, const array& A, SvdOptions opt = {}) {
    SvdResult r = detail::svd_jacobi(A, true, opt);
    eval({r.U, r.S, r.Vt, r.info});
    check(label, A, r);
}

void run_qr(const std::string& label, const array& A) {
    SvdOptions o;
    o.kernel = SvdOptions::Kernel::jacobi;
    SvdResult r = detail::svd_qr_jacobi(A, true, o);
    eval({r.U, r.S, r.Vt, r.info});
    check(label, A, r);
}

void run_block(const std::string& label, const array& A, unsigned inner = 0) {
    SvdOptions o;
    o.inner_sweeps = inner;
    SvdResult r = detail::svd_block_jacobi(A, true, o);
    eval({r.U, r.S, r.Vt, r.info});
    check(label, A, r);
}

// QR-preconditioned, with the block kernel on the factor.
void run_qr_block(const std::string& label, const array& A) {
    SvdOptions o;
    o.kernel = SvdOptions::Kernel::block;
    SvdResult r = detail::svd_qr_jacobi(A, true, o);
    eval({r.U, r.S, r.Vt, r.info});
    check(label, A, r);
}

void run_bidiag(const std::string& label, const array& A) {
    SvdResult r = detail::svd_bidiag(A, true);
    eval({r.U, r.S, r.Vt, r.info});
    check(label, A, r);
}

void run_band_vectors(const std::string& label, const array& A) {
    SvdResult r = detail::svd_band_vectors(A);
    eval({r.U, r.S, r.Vt, r.info});
    check(label, A, r);
}

void run_cpu(const std::string& label, const array& A) {
    SvdResult r = detail::svd_cpu(A, true);
    eval({r.U, r.S, r.Vt, r.info});
    check(label, A, r);
}

// Q1 diag(spec) Q2^T for random orthogonal Q1 (M x K), Q2 (N x K).
array with_singular_values(int M, int N, const std::vector<float>& spec) {
    const int K = (int)spec.size();
    array G1 = random::normal({M, K}, float32, 0.0f, 1.0f, std::nullopt, Device::cpu);
    array G2 = random::normal({N, K}, float32, 0.0f, 1.0f, std::nullopt, Device::cpu);
    auto [Q1, R1] = linalg::qr(G1, Device::cpu);
    auto [Q2, R2] = linalg::qr(G2, Device::cpu);
    array A = matmul(multiply(Q1, from_values(spec, {1, K}), Device::cpu),
                     transpose(Q2, Device::cpu), Device::cpu);
    eval({A});
    return A;
}

std::string dims(int b, int M, int N) {
    return (b > 1 ? "batch " + std::to_string(b) + " x " : "") +
           std::to_string(M) + "x" + std::to_string(N);
}

} // namespace

int main() {
    set_default_device(Device::gpu);

    std::printf("\nsvd correctness tests\n");
    std::printf("tolerances: recon<=%.0e ortho<=%.0e |S-lapack|<=%.0e   (relative to ||A||_F)\n",
                kReconTol, kOrthoTol, kSvalTol);

    // -------------------------------------------------------------------------
    // Shapes: square around the simdgroup and pair-count boundaries (N = 64 is
    // the last size with one pair per simdgroup), odd N (dummy partner), tall,
    // wide, and batches of each.
    // -------------------------------------------------------------------------
    std::printf("\n[ square ]\n");
    for (int n : {1, 2, 3, 4, 5, 7, 8, 9, 16, 17, 31, 32, 33, 63, 64, 65, 66, 100, 128, 129, 200, 256})
        run(dims(1, n, n), random_matrix(1, n, n, 100 + n));
    run(dims(1, 512, 512), random_matrix(1, 512, 512, 199));

    std::printf("\n[ tall ]\n");
    for (auto [M, N] : std::vector<std::pair<int, int>>{{2, 1}, {5, 3}, {33, 1}, {40, 7}, {64, 32},
                                                        {100, 10}, {129, 65}, {256, 64}, {512, 8},
                                                        {1000, 33}, {2048, 64}})
        run(dims(1, M, N), random_matrix(1, M, N, 300 + M + N));

    std::printf("\n[ wide ]\n");
    for (auto [M, N] : std::vector<std::pair<int, int>>{{1, 2}, {3, 5}, {1, 33}, {7, 40}, {32, 64},
                                                        {65, 129}, {8, 512}})
        run(dims(1, M, N), random_matrix(1, M, N, 500 + M + N));

    std::printf("\n[ batched ]\n");
    run(dims(37, 3, 3),     random_matrix(37, 3, 3, 700));
    run(dims(1000, 6, 6),   random_matrix(1000, 6, 6, 701));
    run(dims(9, 33, 33),    random_matrix(9, 33, 33, 702));
    run(dims(16, 64, 64),   random_matrix(16, 64, 64, 703));
    run(dims(5, 100, 20),   random_matrix(5, 100, 20, 704));
    run(dims(5, 20, 100),   random_matrix(5, 20, 100, 705));
    run(dims(4, 200, 200),  random_matrix(4, 200, 200, 706));
    run(dims(300, 17, 5),   random_matrix(300, 17, 5, 707));
    {
        std::vector<float> data(2 * 3 * 9 * 7);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        for (auto& v : data) v = dist(rng());
        run("batch [2,3] x 9x7", from_values(data, {2, 3, 9, 7}));
    }

    // -------------------------------------------------------------------------
    // Simdgroups per matrix: every count must give the same decomposition.
    // -------------------------------------------------------------------------
    std::printf("\n[ simdgroups per matrix ]\n");
    for (unsigned g : {1u, 2u, 3u, 5u, 16u, 32u}) {
        SvdOptions o; o.simdgroups = g;
        run("65x65, " + std::to_string(g) + " simdgroups", random_matrix(1, 65, 65, 750), o);
    }
    { SvdOptions o; o.simdgroups = 1; run("batch 50 x 40x12, 1 simdgroup", random_matrix(50, 40, 12, 751), o); }

    // -------------------------------------------------------------------------
    // QR-preconditioned backend: long thin input, where it is used, and
    // shapes where it is not, since it must be correct wherever it is forced.
    // -------------------------------------------------------------------------
    std::printf("\n[ backend: qr_jacobi ]\n");
    for (auto [M, N] : std::vector<std::pair<int, int>>{{512, 8}, {600, 32}, {1024, 64}, {2048, 64},
                                                        {100, 10}, {64, 64}, {33, 32}, {5, 1}})
        run_qr(dims(1, M, N), random_matrix(1, M, N, 900 + M + N));
    run_qr(dims(1, 8, 512) + " (wide)",  random_matrix(1, 8, 512, 950));
    run_qr(dims(5, 700, 20),              random_matrix(5, 700, 20, 951));
    run_qr(dims(3, 20, 700) + " (wide)", random_matrix(3, 20, 700, 952));
    run_qr("zeros 600x8",                 zeros({600, 8}));
    {
        array L = random::normal({600, 5}, float32, 0.0f, 1.0f, std::nullopt, Device::cpu);
        array R = random::normal({5, 16}, float32, 0.0f, 1.0f, std::nullopt, Device::cpu);
        array A = matmul(L, R, Device::cpu);
        eval({A});
        run_qr("rank 5 of 600x16", A);
    }
    {
        array A = random_matrix(1, 800, 12, 953);
        run_qr("scaled 1e-30 (800x12)", multiply(A, array(1e-30f)));
    }

    // -------------------------------------------------------------------------
    // Block backend: sizes around the 16-block and 32-group boundaries, padding
    // in both directions, tall and wide shapes, batches, and the structured
    // cases. It must be correct at every size, not only where it is dispatched.
    // -------------------------------------------------------------------------
    std::printf("\n[ backend: block_jacobi ]\n");
    for (int n : {1, 5, 17, 31, 32, 33, 48, 64, 96, 100, 128, 129, 200, 256, 512})
        run_block(dims(1, n, n), random_matrix(1, n, n, 1100 + n));
    for (auto [M, N] : std::vector<std::pair<int, int>>{{40, 7}, {100, 10}, {129, 65}, {256, 64},
                                                        {1000, 33}, {600, 200}, {2048, 64}})
        run_block(dims(1, M, N), random_matrix(1, M, N, 1200 + M + N));
    run_block(dims(1, 7, 40) + " (wide)",   random_matrix(1, 7, 40, 1300));
    run_block(dims(1, 65, 129) + " (wide)", random_matrix(1, 65, 129, 1301));
    run_block(dims(5, 64, 64),              random_matrix(5, 64, 64, 1302));
    run_block(dims(3, 300, 100),            random_matrix(3, 300, 100, 1303));
    run_block(dims(40, 33, 17),             random_matrix(40, 33, 17, 1304));
    run_block("128x128, 2 inner sweeps",    random_matrix(1, 128, 128, 1305), 2);
    run_block("128x128, 3 inner sweeps",    random_matrix(1, 128, 128, 1306), 3);
    run_block("identity 64x64",             eye(64));
    run_block("zeros 48x48",                zeros({48, 48}));
    run_block("zeros 100x20",               zeros({100, 20}));
    run_block("diagonal 128x128",           diag(arange(128, float32)));
    run_block("ones 40x40 (rank 1)",        ones({40, 40}));
    {
        std::vector<float> spec(96);
        for (int i = 0; i < 96; ++i) spec[i] = std::pow(10.0f, 4.0f - 8.0f * i / 95.0f);
        run_block("singular values 1e+4 .. 1e-4 (120x96)", with_singular_values(120, 96, spec));
    }
    {
        std::vector<float> spec(128);
        for (int i = 0; i < 128; ++i) spec[i] = (float)(1 + i % 4);
        run_block("repeated singular values (128x128)", with_singular_values(128, 128, spec));
    }
    {
        array A = random_matrix(1, 80, 40, 1307);
        eval({A});
        std::vector<float> d(A.data<float>(), A.data<float>() + 80 * 40);
        for (int i = 0; i < 80; ++i)
            for (int c = 0; c < 40; ++c) d[i * 40 + c] *= std::pow(10.0f, -(float)c * 0.125f);
        run_block("graded columns 1 .. 1e-5 (80x40)", from_values(d, {80, 40}));
    }
    {
        array A = random_matrix(1, 64, 64, 1308);
        run_block("scaled 1e-30 (64x64)", multiply(A, array(1e-30f)));
        run_block("scaled 1e+37 (64x64)", multiply(A, array(1e37f)));
    }
    {
        // Matrices of different magnitude in one batch, and a NaN among them.
        array good = random_matrix(3, 48, 40, 1309);
        eval({good});
        std::vector<float> data(good.data<float>(), good.data<float>() + 3 * 48 * 40);
        for (int i = 0; i < 48 * 40; ++i) data[i] *= 1e-12f;
        data[1 * 48 * 40 + 5 * 40 + 2] = NAN;
        array A = from_values(data, {3, 48, 40});
        ++g_checks;
        try {
            SvdResult r = detail::svd_block_jacobi(A, true, {});
            eval({r.U, r.S, r.Vt, r.info});
            const uint32_t* info = r.info.data<uint32_t>();
            const bool flags_ok = detail::svd_converged(info[0]) && detail::svd_nonfinite(info[1]) &&
                                  detail::svd_converged(info[2]);
            array s1 = reshape(slice(r.S, {1, 0}, {2, 40}), {40});
            const bool all_nan = !any(logical_not(isnan(s1))).item<bool>();
            if (!flags_ok || !all_nan) fail("block: NaN in a batch", "flags or NaN propagation wrong");
            else {
                std::printf("  ok    block: NaN matrix flagged and all-NaN; neighbours converged\n");
                for (int b : {0, 2}) {
                    check("block: mixed batch, matrix " + std::to_string(b) + " intact",
                          reshape(slice(A, {b, 0, 0}, {b + 1, 48, 40}), {48, 40}),
                          SvdResult{reshape(slice(r.U, {b, 0, 0}, {b + 1, 48, 40}), {48, 40}),
                                    reshape(slice(r.S, {b, 0}, {b + 1, 40}), {40}),
                                    reshape(slice(r.Vt, {b, 0, 0}, {b + 1, 40, 40}), {40, 40}),
                                    array((uint32_t)(1u | (1u << 16)))});
                }
            }
        } catch (const std::exception& e) {
            fail("block: NaN in a batch", std::string("threw: ") + e.what());
        }
    }
    {
        array A = random_matrix(2, 60, 44, 1310);
        SvdResult full = detail::svd_block_jacobi(A, true, {});
        SvdResult vals = detail::svd_block_jacobi(A, false, {});
        eval({full.S, vals.S});
        const float d = max_abs(subtract(full.S, vals.S)) / std::max(frobenius(A), 1.0f);
        ++g_checks;
        if (d > 1e-6f) fail("block: values-only == with vectors", "differ by " + std::to_string(d));
        else std::printf("  ok    %-44s |dS|=%.1e\n", "block: values-only == with vectors", d);
    }

    std::printf("\n[ backend: qr then block_jacobi ]\n");
    for (auto [M, N] : std::vector<std::pair<int, int>>{{600, 128}, {1024, 200}, {300, 130}, {100, 10}})
        run_qr_block(dims(1, M, N), random_matrix(1, M, N, 1400 + M + N));
    run_qr_block(dims(1, 130, 700) + " (wide)", random_matrix(1, 130, 700, 1450));
    run_qr_block(dims(3, 500, 140),              random_matrix(3, 500, 140, 1451));

    // -------------------------------------------------------------------------
    // Rank deficiency, repeatedly. Columns that cancel to rounding noise are
    // where one-sided Jacobi can rotate for ever at the tolerance; this once
    // failed to converge in one run out of three through the QR path.
    // -------------------------------------------------------------------------
    std::printf("\n[ rank-deficient, 12 random instances per shape and path ]\n");
    {
        enum Path { kJacobi, kQr, kBlock, kQrBlock };
        const char* names[] = {"jacobi", "qr_jacobi", "block_jacobi", "qr_block_jacobi"};
        struct Case { int M, N, rank; Path path; };
        for (const Case& c : std::vector<Case>{
                 {16, 16, 5, kJacobi},  {24, 24, 1, kJacobi},   {40, 16, 5, kJacobi},  {64, 64, 20, kJacobi},
                 {600, 16, 5, kJacobi}, {600, 16, 5, kQr},      {1024, 32, 3, kQr},    {16, 600, 5, kQr},
                 {33, 32, 31, kQr},
                 {16, 16, 5, kBlock},   {64, 64, 20, kBlock},   {128, 128, 40, kBlock}, {200, 96, 7, kBlock},
                 {40, 16, 5, kBlock},   {96, 200, 7, kBlock},
                 {600, 130, 50, kQrBlock}, {400, 160, 3, kQrBlock}}) {
            ++g_checks;
            float worst_recon = 0.0f, worst_ortho = 0.0f;
            unsigned worst_sweeps = 0;
            std::string err;
            for (int k = 0; k < 12 && err.empty(); ++k) {
                array L = random::normal({c.M, c.rank}, float32, 0.0f, 1.0f, std::nullopt, Device::cpu);
                array R = random::normal({c.rank, c.N}, float32, 0.0f, 1.0f, std::nullopt, Device::cpu);
                array A = matmul(L, R, Device::cpu);
                eval({A});
                try {
                    SvdOptions o;
                    o.kernel = c.path == kQrBlock ? SvdOptions::Kernel::block : SvdOptions::Kernel::jacobi;
                    SvdResult r = c.path == kJacobi ? detail::svd_jacobi(A, true, o)
                                : c.path == kBlock  ? detail::svd_block_jacobi(A, true, o)
                                :                     detail::svd_qr_jacobi(A, true, o);
                    eval({r.U, r.S, r.Vt, r.info});
                    const int K = std::min(c.M, c.N);
                    const float nA = frobenius(A);
                    worst_recon = std::max(worst_recon, frobenius(subtract(
                        matmul(multiply(r.U, expand_dims(r.S, -2)), r.Vt), A)) / nA);
                    worst_ortho = std::max(worst_ortho, frobenius(subtract(
                        matmul(transpose_last_two(r.U), r.U), eye(K))) / std::sqrt((float)K));
                    worst_sweeps = std::max(worst_sweeps, detail::svd_sweeps(r.info.item<uint32_t>()));
                } catch (const std::exception& e) {
                    err = e.what();
                }
            }
            const std::string label = "rank " + std::to_string(c.rank) + " of " + dims(1, c.M, c.N) +
                                      " (" + names[c.path] + ")";
            if (!err.empty()) fail(label, err);
            else if (worst_recon > kReconTol || worst_ortho > kOrthoTol)
                fail(label, "recon " + std::to_string(worst_recon) + " ortho " + std::to_string(worst_ortho));
            else std::printf("  ok    %-44s worst recon=%.1e orthoU=%.1e sweeps=%u\n",
                             label.c_str(), worst_recon, worst_ortho, worst_sweeps);
        }
    }

    // -------------------------------------------------------------------------
    // CPU backend: thin factors from MLX, each of its three branches.
    // -------------------------------------------------------------------------
    std::printf("\n[ backend: cpu ]\n");
    run_cpu("40x40 (direct)",            random_matrix(1, 40, 40, 960));
    run_cpu("70x50 (full, then sliced)", random_matrix(1, 70, 50, 961));
    run_cpu("300x20 (QR first)",         random_matrix(1, 300, 20, 962));
    run_cpu("20x300 (wide)",             random_matrix(1, 20, 300, 963));
    run_cpu(dims(6, 90, 30),             random_matrix(6, 90, 30, 964));
    run_cpu("rank one 30x20",            matmul(random_matrix(1, 30, 1, 965), random_matrix(1, 1, 20, 966)));
    // A batch is spread over cpu_threads() threads; one thread must agree,
    // through the direct branch and the QR-first one.
    for (auto [m, n] : {std::pair{24, 24}, std::pair{200, 30}}) {
        array A = random_matrix(37, m, n, 967 + m);
        SvdResult all_threads = detail::svd_cpu(A, true);
        set_cpu_threads(1);
        SvdResult one = detail::svd_cpu(A, true);
        set_cpu_threads(0);
        eval({all_threads.U, all_threads.S, all_threads.Vt, all_threads.info, one.S});
        check(dims(37, m, n) + " every thread", A, all_threads);
        array dS = max(abs(subtract(all_threads.S, one.S)));
        array nS = max(abs(one.S));
        eval({dS, nS});
        ++g_checks;
        const float d = dS.item<float>() / std::max(nS.item<float>(), 1e-30f);
        if (d > 1e-6f) fail(dims(37, m, n) + " one thread == every thread", "differ by " + std::to_string(d));
        else std::printf("  ok    %-44s |dS|=%.1e\n", (dims(37, m, n) + " one thread == every thread").c_str(), d);
    }

    // The bidiag backend reduces on the GPU in panels of 32 columns while more
    // than 33 remain, LAPACK takes the rest; a matrix at least twice as tall
    // as wide and 64 wide goes through a QR first; a wide one is its transpose.
    // Above 1024 rows, a column has more threadgroups' norm partials than a
    // simdgroup has lanes.
    std::printf("\n[ backend: bidiag ]\n");
    for (auto [M, N] : std::vector<std::pair<int, int>>{{1, 1}, {3, 3}, {33, 33}, {34, 34}, {35, 35},
                                                         {65, 64}, {64, 65}, {100, 97}, {129, 130},
                                                         {300, 300}, {513, 500}, {300, 20}, {20, 300},
                                                         {600, 100}, {100, 600}, {1024, 1024}, {1100, 1060}})
        run_bidiag("bidiag " + dims(1, M, N), random_matrix(1, M, N, 1000 + M * 3 + N));
    run_bidiag("bidiag " + dims(3, 150, 120), random_matrix(3, 150, 120, 1100));
    // Batches are pipelined over two workspace slots: odd and even counts,
    // through the QR first and the transpose.
    run_bidiag("bidiag " + dims(4, 300, 80) + " (QR first)", random_matrix(4, 300, 80, 1101));
    run_bidiag("bidiag " + dims(2, 70, 200) + " (wide)", random_matrix(2, 70, 200, 1102));
    run_bidiag("bidiag " + dims(5, 40, 40), random_matrix(5, 40, 40, 1103));
    {   // singular values alone, batched, == with vectors
        array A = random_matrix(3, 160, 130, 1104);
        SvdResult rv = detail::svd_bidiag(A, false), rw = detail::svd_bidiag(A, true);
        eval({rv.S, rw.S});
        ++g_checks;
        const float d = max_abs(subtract(rv.S, rw.S)) / std::max(max_abs(rw.S), 1e-30f);
        if (d > 2e-5f) fail("bidiag batch 3 values-only == with vectors", "differ by " + std::to_string(d));
        else std::printf("  ok    %-44s |ds|=%.1e\n", "bidiag batch 3 values-only == with vectors", d);
    }
    for (float s : {1e-30f, 1e20f, 1e37f}) {
        char label[64];
        std::snprintf(label, sizeof label, "bidiag scaled by %.0e 160x140", s);
        run_bidiag(label, multiply(random_matrix(1, 160, 140, 1200), array(s / 5.0f)));
    }
    run_bidiag("bidiag rank one 200x150", matmul(random_matrix(1, 200, 1, 1300), random_matrix(1, 1, 150, 1301)));
    run_bidiag("bidiag zero 120x100", zeros({120, 100}));
    run_bidiag("bidiag identity 100x100", eye(100));
    {
        std::vector<float> spec(90);
        for (int i = 0; i < 90; ++i) spec[i] = i < 40 ? 3.0f : 1e-3f * (90 - i);
        run_bidiag("bidiag repeated and tiny values 150x90", with_singular_values(150, 90, spec));
    }
    // From k = 129 the singular vectors come from the parallel divide and
    // conquer (divide_conquer.cpp): its deflations on matrices large enough
    // to be divided several times.
    run_bidiag("bidiag rank one 500x400", matmul(random_matrix(1, 500, 1, 1310), random_matrix(1, 1, 400, 1311)));
    run_bidiag("bidiag zero 320x300", zeros({320, 300}));
    run_bidiag("bidiag identity 300x300", eye(300));
    {
        std::vector<float> spec(400), close(350);
        for (int i = 0; i < 400; ++i) spec[i] = i < 200 ? 3.0f : 1e-3f * (float)(400 - i);
        run_bidiag("bidiag repeated and tiny values 450x400", with_singular_values(450, 400, spec));
        for (int i = 0; i < 350; ++i) close[i] = 1.0f + 1e-6f * (float)i;
        run_bidiag("bidiag clustered values 350x350", with_singular_values(350, 350, close));
    }
    {   // singular values alone == with vectors
        array A = random_matrix(1, 300, 260, 1400);
        SvdResult rv = detail::svd_bidiag(A, false), rw = detail::svd_bidiag(A, true);
        eval({rv.S, rw.S});
        ++g_checks;
        const float d = max_abs(subtract(rv.S, rw.S)) / std::max(max_abs(rw.S), 1e-30f);
        // LAPACK computes them by different methods (dqds, divide and conquer),
        // so they agree to float32 precision, not bit for bit.
        if (d > 2e-5f) fail("bidiag values-only == with vectors", "differ by " + std::to_string(d));
        else std::printf("  ok    %-44s |ds|=%.1e\n", "bidiag values-only == with vectors", d);
    }
    {   // NaN in one matrix of a batch
        const int M = 70, N = 50;
        array A = random_matrix(2, M, N, 1500);
        eval({A});
        std::vector<float> data(A.data<float>(), A.data<float>() + 2 * M * N);
        data[(size_t)M * N + 7] = NAN;
        SvdResult r = detail::svd_bidiag(from_values(data, {2, M, N}), true);
        array info = reshape(r.info, {-1});
        array s1 = slice(r.S, {1, 0}, {2, N}), s0 = slice(r.S, {0, 0}, {1, N});
        eval({info, s0, s1});
        ++g_checks;
        const bool ok = all(isnan(s1)).item<bool>() && !has_non_finite(s0) &&
                        !detail::svd_converged(info.data<uint32_t>()[1]) && detail::svd_converged(info.data<uint32_t>()[0]);
        if (!ok) fail("bidiag NaN in one matrix of a batch", "not isolated");
        else std::printf("  ok    %-44s\n", "bidiag NaN in one matrix of a batch");
    }

    // The band backend, singular values alone by the two-stage reduction: the
    // GPU's blocks of b columns while 2b remain, LAPACK the rest; a panel of
    // up to 128 rows in one simdgroup, a taller one by TSQR (leaves of up to
    // 128 rows) with its Householder vectors rebuilt, so the sizes straddle
    // those boundaries, for each band width. Against the CPU's values.
    std::printf("\n[ backend: band ]\n");
    {
        auto run_band = [&](const std::string& label, const array& A, uint32_t width) {
            SvdResult r = detail::svd_band(A, width);
            SvdResult c = detail::svd_cpu(A, false);
            array info = reshape(r.info, {-1});
            eval({r.S, c.S, info});
            ++g_checks;
            const float d = max_abs(subtract(r.S, c.S)) / std::max(max_abs(c.S), 1e-30f);
            bool converged = true;
            for (size_t i = 0; i < info.size(); ++i) converged &= detail::svd_converged(info.data<uint32_t>()[i]);
            if (!(d <= 2e-5f) || !converged) fail(label, "|s - cpu| / s_max " + std::to_string(d));
            else std::printf("  ok    %-44s |ds|=%.1e\n", label.c_str(), d);
        };
        for (uint32_t w : {8u, 16u, 32u})
            for (auto [M, N] : std::vector<std::pair<int, int>>{{1, 1}, {3, 3}, {20, 20}, {64, 64}, {65, 64},
                                                                 {128, 128}, {129, 129}, {300, 300}, {513, 500},
                                                                 {300, 20}, {20, 300}, {600, 100}, {100, 600},
                                                                 {1100, 1060}})
                run_band("band b=" + std::to_string(w) + " " + dims(1, M, N), random_matrix(1, M, N, 2000 + M + N), w);
        run_band("band " + dims(3, 150, 120), random_matrix(3, 150, 120, 2100), 8);
        run_band("band zero 120x100", zeros({120, 100}), 8);
        run_band("band identity 100x100", eye(100), 16);
        run_band("band rank one 200x150",
                 matmul(random_matrix(1, 200, 1, 2200), random_matrix(1, 1, 150, 2201)), 8);
        for (float scale : {1e-30f, 1e30f}) {
            char label[64];
            std::snprintf(label, sizeof label, "band scaled by %.0e 160x140", scale);
            run_band(label, multiply(random_matrix(1, 160, 140, 2300), array(scale / 5.0f)), 8);
        }
        {
            std::vector<float> spec(90);
            for (int i = 0; i < 90; ++i) spec[i] = i < 40 ? 3.0f : 1e-3f * (90 - i);
            run_band("band repeated and tiny values 150x90", with_singular_values(150, 90, spec), 16);
        }
        {   // NaN in one matrix of a batch
            const int M = 300, N = 260;
            array A = random_matrix(2, M, N, 2400);
            eval({A});
            std::vector<float> data(A.data<float>(), A.data<float>() + 2 * M * N);
            data[(size_t)M * N + 7] = NAN;
            SvdResult r = detail::svd_band(from_values(data, {2, M, N}));
            array info = reshape(r.info, {-1});
            array s1 = slice(r.S, {1, 0}, {2, N}), s0 = slice(r.S, {0, 0}, {1, N});
            eval({info, s0, s1});
            ++g_checks;
            const bool ok = all(isnan(s1)).item<bool>() && !has_non_finite(s0) &&
                            !detail::svd_converged(info.data<uint32_t>()[1]) &&
                            detail::svd_converged(info.data<uint32_t>()[0]);
            if (!ok) fail("band NaN in one matrix of a batch", "not isolated");
            else std::printf("  ok    %-44s\n", "band NaN in one matrix of a batch");
        }
    }

    // The band backend with vectors: the GPU's blocks' reflectors (16 columns
    // a block, aggregated 8 at a time), the LAPACK tail's (the last 16 to 31
    // columns), and the chase's (bd_chase_apply: blocks of 16 sweeps, the
    // groups pipelined 4 or 8 to a threadgroup). Sizes straddle the tail
    // alone (k < 32), a partial aggregate, the chase's tiles (k - 1 a multiple
    // of 16 or not), the narrow and wide chase kernels (2048 columns), and
    // the TSQR panels (over 128 rows).
    std::printf("\n[ backend: band, with vectors ]\n");
    for (auto [M, N] : std::vector<std::pair<int, int>>{{1, 1}, {2, 2}, {3, 3}, {17, 17}, {31, 31}, {32, 32},
                                                        {33, 33}, {47, 47}, {48, 48}, {49, 49}, {64, 64},
                                                        {65, 64}, {64, 65}, {129, 129}, {161, 161}, {300, 300},
                                                        {513, 500}, {300, 20}, {20, 300}, {600, 100},
                                                        {100, 600}, {700, 450}, {450, 700}, {1024, 1024},
                                                        {1100, 1060}, {2049, 2049}})
        run_band_vectors("band " + dims(1, M, N), random_matrix(1, M, N, 3000 + M * 3 + N));
    run_band_vectors("band " + dims(3, 150, 120), random_matrix(3, 150, 120, 3100));
    run_band_vectors("band " + dims(2, 300, 80) + " (QR first)", random_matrix(2, 300, 80, 3101));
    run_band_vectors("band " + dims(2, 70, 200) + " (wide)", random_matrix(2, 70, 200, 3102));
    for (float s : {1e-30f, 1e20f, 1e37f}) {
        char label[64];
        std::snprintf(label, sizeof label, "band scaled by %.0e 160x140", s);
        run_band_vectors(label, multiply(random_matrix(1, 160, 140, 3200), array(s / 5.0f)));
    }
    run_band_vectors("band rank one 500x400", matmul(random_matrix(1, 500, 1, 3300), random_matrix(1, 1, 400, 3301)));
    run_band_vectors("band zero 320x300", zeros({320, 300}));
    run_band_vectors("band identity 300x300", eye(300));
    {
        std::vector<float> spec(400), close(350);
        for (int i = 0; i < 400; ++i) spec[i] = i < 200 ? 3.0f : 1e-3f * (float)(400 - i);
        run_band_vectors("band repeated and tiny values 450x400", with_singular_values(450, 400, spec));
        for (int i = 0; i < 350; ++i) close[i] = 1.0f + 1e-6f * (float)i;
        run_band_vectors("band clustered values 350x350", with_singular_values(350, 350, close));
    }
    {   // NaN in one matrix of a batch
        const int M = 300, N = 260;
        array A = random_matrix(2, M, N, 3400);
        eval({A});
        std::vector<float> data(A.data<float>(), A.data<float>() + 2 * M * N);
        data[(size_t)M * N + 7] = NAN;
        SvdResult r = detail::svd_band_vectors(from_values(data, {2, M, N}));
        array info = reshape(r.info, {-1});
        array s1 = slice(r.S, {1, 0}, {2, N}), s0 = slice(r.S, {0, 0}, {1, N});
        eval({info, s0, s1});
        ++g_checks;
        const bool ok = all(isnan(s1)).item<bool>() && !has_non_finite(s0) &&
                        !detail::svd_converged(info.data<uint32_t>()[1]) && detail::svd_converged(info.data<uint32_t>()[0]);
        if (!ok) fail("band with vectors NaN in one matrix of a batch", "not isolated");
        else std::printf("  ok    %-44s\n", "band with vectors NaN in one matrix of a batch");
    }

    // The golub_kahan backend keeps the matrix in threadgroup memory, so it
    // takes squares up to svd_gk_max_k() (longer matrices when tall); a wide
    // matrix is decomposed as its transpose, and through the QR first
    // (svd_qr_jacobi with Kernel::golub_kahan) any length with k up to the
    // limit. With 33 columns and more the QR iteration runs on a simdgroup of
    // its own.
    std::printf("\n[ backend: golub_kahan ]\n");
    {
        const int kmax = (int)metal_linalg::detail::svd_gk_max_k();
        std::printf("  this device: squares up to %d\n", kmax);
        auto run_gk = [&](const std::string& label, const array& A) {
            SvdResult r = detail::svd_golub_kahan(A, true);
            eval({r.U, r.S, r.Vt, r.info});
            check(label, A, r);
        };
        auto run_qr_gk = [&](const std::string& label, const array& A) {
            SvdOptions o;
            o.kernel = SvdOptions::Kernel::golub_kahan;
            SvdResult r = detail::svd_qr_jacobi(A, true, o);
            eval({r.U, r.S, r.Vt, r.info});
            check(label, A, r);
        };
        for (auto [M, N] : std::vector<std::pair<int, int>>{{1, 1}, {2, 2}, {3, 3}, {2, 5}, {5, 2}, {8, 8},
                                                             {16, 16}, {32, 32}, {33, 33}, {47, 47}, {64, 64},
                                                             {kmax, kmax}, {65, 64}, {64, 65}, {100, 20},
                                                             {20, 100}, {300, 12}, {12, 300}, {200, 1}, {1, 200}})
            run_gk("golub_kahan " + dims(1, M, N), random_matrix(1, M, N, 2000 + M * 3 + N));
        run_gk("golub_kahan " + dims(64, 24, 24), random_matrix(64, 24, 24, 2100));
        run_gk("golub_kahan " + dims(16, 40, 33), random_matrix(16, 40, 33, 2101));
        run_qr_gk("qr_golub_kahan " + dims(1, 2000, 48), random_matrix(1, 2000, 48, 2102));
        run_qr_gk("qr_golub_kahan " + dims(4, 600, 64), random_matrix(4, 600, 64, 2103));
        run_qr_gk("qr_golub_kahan " + dims(1, 40, 900), random_matrix(1, 40, 900, 2104));
        for (float sc : {1e-30f, 1e20f, 1e37f}) {
            char label[64];
            std::snprintf(label, sizeof label, "golub_kahan scaled by %.0e 40x30", sc);
            run_gk(label, multiply(random_matrix(1, 40, 30, 2200), array(sc / 5.0f)));
        }
        run_gk("golub_kahan zero 30x20", zeros({30, 20}));
        run_gk("golub_kahan identity 50x50", eye(50));
        run_gk("golub_kahan rank one 60x40", matmul(random_matrix(1, 60, 1, 2201), random_matrix(1, 1, 40, 2202)));
        {
            std::vector<float> spec(48);
            for (int i = 0; i < 48; ++i) spec[i] = std::pow(10.0f, 4.0f - 8.0f * i / 47.0f);
            run_gk("golub_kahan singular values 1e+4 .. 1e-4 (60x48)", with_singular_values(60, 48, spec));
            for (int i = 0; i < 48; ++i) spec[i] = i < 20 ? 3.0f : 1e-3f * (48 - i);
            run_gk("golub_kahan repeated and tiny values (60x48)", with_singular_values(60, 48, spec));
        }
        // Rank deficiency: zero diagonal entries in the bidiagonal, chased out
        // with rotations of their own; U stays orthonormal without completion.
        for (auto [M, N, rank] : std::vector<std::tuple<int, int, int>>{{16, 16, 5}, {48, 40, 5}, {40, 48, 7},
                                                                         {60, 60, 1}, {200, 24, 3}}) {
            for (int k = 0; k < 6; ++k) {
                array L = random::normal({M, rank}, float32, 0.0f, 1.0f, std::nullopt, Device::cpu);
                array R = random::normal({rank, N}, float32, 0.0f, 1.0f, std::nullopt, Device::cpu);
                array A = matmul(L, R, Device::cpu);
                eval({A});
                SvdResult r = detail::svd_golub_kahan(A, true);
                eval({r.U, r.S, r.Vt, r.info});
                if (k == 0) check("golub_kahan rank " + std::to_string(rank) + " of " + dims(1, M, N), A, r);
                ++g_checks;
                if (!detail::svd_rank_deficient(r.info.item<uint32_t>()))
                    fail("golub_kahan rank " + std::to_string(rank) + " of " + dims(1, M, N), "not flagged rank-deficient");
            }
        }
        {   // singular values alone == with vectors
            array A = random_matrix(3, 50, 37, 2300);
            SvdResult rv = detail::svd_golub_kahan(A, false), rw = detail::svd_golub_kahan(A, true);
            eval({rv.S, rw.S});
            ++g_checks;
            const float d = max_abs(subtract(rv.S, rw.S)) / std::max(max_abs(rw.S), 1e-30f);
            if (d > 1e-6f) fail("golub_kahan values-only == with vectors", "differ by " + std::to_string(d));
            else std::printf("  ok    %-44s |ds|=%.1e\n", "golub_kahan values-only == with vectors", d);
        }
        {   // NaN in one matrix of a batch
            const int M = 30, N = 20;
            array A = random_matrix(3, M, N, 2400);
            eval({A});
            std::vector<float> data(A.data<float>(), A.data<float>() + 3 * M * N);
            data[(size_t)M * N + 11] = NAN;
            SvdResult r = detail::svd_golub_kahan(from_values(data, {3, M, N}), true);
            array info = reshape(r.info, {-1});
            array s1 = slice(r.S, {1, 0}, {2, N}), s0 = slice(r.S, {0, 0}, {1, N}), s2 = slice(r.S, {2, 0}, {3, N});
            array u1 = slice(r.U, {1, 0, 0}, {2, M, N});
            eval({info, s0, s1, s2, u1});
            ++g_checks;
            const uint32_t* w = info.data<uint32_t>();
            const bool ok = all(isnan(s1)).item<bool>() && all(isnan(u1)).item<bool>() && !has_non_finite(s0) &&
                            !has_non_finite(s2) && detail::svd_nonfinite(w[1]) && detail::svd_converged(w[0]) &&
                            detail::svd_converged(w[2]);
            if (!ok) fail("golub_kahan NaN in one matrix of a batch", "not isolated");
            else std::printf("  ok    %-44s\n", "golub_kahan NaN in one matrix of a batch");
        }
        // A batch shared with the CPU path, directly and through the QR first.
        {
            array A = random_matrix(600, 24, 20, 2450);
            SvdResult r = detail::svd_golub_kahan_shared(A, true);
            eval({r.U, r.S, r.Vt, r.info});
            check("golub_kahan shared with the CPU 600 x 24x20", A, r);
            SvdResult rv = detail::svd_golub_kahan_shared(A, false);
            eval({rv.S});
            ++g_checks;
            const float d = max_abs(subtract(rv.S, r.S)) / std::max(max_abs(r.S), 1e-30f);
            if (d > 2e-5f) fail("golub_kahan shared values-only == with vectors", "differ by " + std::to_string(d));
            else std::printf("  ok    %-44s |ds|=%.1e\n", "golub_kahan shared values-only == with vectors", d);
            array B = random_matrix(100, 2000, 12, 2451);
            SvdResult rb = detail::svd_golub_kahan_shared(B, true);
            eval({rb.U, rb.S, rb.Vt, rb.info});
            check("qr_golub_kahan shared with the CPU 100 x 2000x12", B, rb);
        }
        ++g_checks;
        try {
            detail::svd_golub_kahan(random_matrix(1, kmax + 1, kmax + 1, 2500), true);
            fail("golub_kahan above its limit", "did not throw");
        } catch (const std::invalid_argument&) {
            std::printf("  ok    %-44s\n", "golub_kahan above its limit throws");
        }
    }

    // -------------------------------------------------------------------------
    // Routing policy. Device-tuned, so nothing here may assume the values
    // measured on any one GPU: each check installs the policy it needs, and
    // the device's own policy is restored at the end.
    // -------------------------------------------------------------------------
    std::printf("\n[ routing policy ]\n");
    {
        const SvdPolicy original = svd_policy();
        const std::string original_source = svd_policy_source();
        std::printf("  this device: gpu_cores=%u  source=%s\n", original.gpu_cores, original_source.c_str());
        std::printf("  QR-preconditioned from %u rows and k >= %u; block kernel from k >= %u; "
                    "GPU iff k<=%u and batch*k>=%u\n",
                    original.qr_min_rows, original.qr_min_k, original.block_min_k, original.gpu_max_k,
                    original.gpu_min_batch_times_k);

        auto expect = [&](const std::string& label, bool ok) {
            ++g_checks;
            if (ok) std::printf("  ok    %s\n", label.c_str());
            else    fail(label, "unexpected routing");
        };
        auto api = [&](const std::string& label, const array& A) {
            auto [U, S, Vt] = svd_accelerated(A);
            eval({U, S, Vt});
            const auto& sh = A.shape();
            check(label, A, SvdResult{U, S, Vt, full(Shape(sh.begin(), sh.end() - 2),
                                                     (uint32_t)(1u | (1u << 16)))});
        };

        SvdPolicy known;          // fixed here, so the checks hold on any device
        known.qr_min_rows = 512;  known.qr_min_k = 64;
        known.block_min_k = 128;  known.block_min_k_batched = 0;  known.block_min_batch = 0;
        known.gpu_max_k = 64;     known.gpu_min_batch_times_k = 1024;  known.gpu_min_batch = 1;
        set_svd_policy(known);
        expect("set_svd_policy -> source \"user\"", std::string(svd_policy_source()) == "user");

        unsetenv("SVD_DEVICE");
        expect("CPU/GPU: 40x40 b1 -> cpu, 8x8 b256 -> gpu, 128x128 b4096 -> cpu, 2048x16 b64 -> gpu",
               svd_backend(40, 40, 1) == SvdBackend::cpu && svd_uses_gpu(8, 8, 256) &&
               svd_backend(128, 128, 4096) == SvdBackend::cpu && svd_uses_gpu(2048, 16, 64));
        expect("preconditioning: 256x32, 1024x16, 300x200 no; 1024x64, 64x1024, 512x256 yes",
               svd_gpu_backend(256, 32, 1) == SvdBackend::jacobi &&          // too short, too narrow
               svd_gpu_backend(1024, 16, 1) == SvdBackend::jacobi &&         // long but narrow
               svd_gpu_backend(300, 200, 1) == SvdBackend::block_jacobi &&   // l < 2k
               svd_gpu_backend(1024, 64, 1) == SvdBackend::qr_jacobi &&
               svd_gpu_backend(64, 1024, 1) == SvdBackend::qr_jacobi &&      // wide: by its transpose
               svd_gpu_backend(512, 256, 1) == SvdBackend::qr_block_jacobi);
        expect("kernel: 64x64, 127x127 whole-matrix; 128x128, 512x512 block",
               svd_gpu_backend(64, 64, 1) == SvdBackend::jacobi &&
               svd_gpu_backend(127, 127, 1) == SvdBackend::jacobi &&
               svd_gpu_backend(128, 128, 1) == SvdBackend::block_jacobi &&
               svd_gpu_backend(512, 512, 1) == SvdBackend::block_jacobi);
        api("routed to CPU (40x40)",              random_matrix(1, 40, 40, 970));
        api("routed to GPU, jacobi (256 x 8x8)",  random_matrix(256, 8, 8, 971));
        api("routed to GPU, qr (16 x 1024x64)",   random_matrix(16, 1024, 64, 972));

        setenv("SVD_DEVICE", "cpu", 1);
        expect("SVD_DEVICE=cpu forces the CPU", !svd_uses_gpu(8, 8, 256));
        setenv("SVD_DEVICE", "gpu", 1);
        expect("SVD_DEVICE=gpu forces the GPU", svd_uses_gpu(512, 512, 1));

        // Force each GPU backend onto shapes it would not normally get.
        SvdPolicy forced = known;
        forced.qr_min_rows = 1;  forced.qr_min_k = 1;
        set_svd_policy(forced);
        expect("qr from 1 row and k = 1 -> qr at 40x20, never at 20x20",
               svd_gpu_backend(40, 20, 1) == SvdBackend::qr_jacobi &&
               svd_gpu_backend(20, 20, 1) == SvdBackend::jacobi);
        api("forced -> qr_jacobi  40x20 b3", random_matrix(3, 40, 20, 973));
        api("forced -> qr_jacobi  9x30 b2",  random_matrix(2, 9, 30, 974));

        forced = known;
        forced.qr_min_rows = 1u << 30;
        set_svd_policy(forced);
        expect("no qr -> jacobi at 2048x8", svd_gpu_backend(2048, 8, 1) == SvdBackend::jacobi);
        api("forced -> jacobi     900x12", random_matrix(1, 900, 12, 975));

        forced = known;
        forced.block_min_k = 1;
        forced.qr_min_rows = 1u << 30;
        set_svd_policy(forced);
        expect("block from k = 1 -> block at 20x20", svd_gpu_backend(20, 20, 1) == SvdBackend::block_jacobi);
        api("forced -> block       20x20 b4", random_matrix(4, 20, 20, 977));
        api("forced -> block       9x50",     random_matrix(1, 9, 50, 978));

        forced = known;
        forced.block_min_k = 1;  forced.qr_min_rows = 1;  forced.qr_min_k = 1;
        set_svd_policy(forced);
        expect("qr and block from 1 -> qr_block at 60x20",
               svd_gpu_backend(60, 20, 1) == SvdBackend::qr_block_jacobi);
        api("forced -> qr_block    60x20 b2", random_matrix(2, 60, 20, 979));

        forced = known;
        forced.block_min_k = 1u << 30;
        set_svd_policy(forced);
        expect("no block -> jacobi at 300x300", svd_gpu_backend(300, 300, 1) == SvdBackend::jacobi);

        // The batch-dependent crossover: block from k = 48 in batches of 8 up.
        forced = known;
        forced.qr_min_rows = 1;  forced.qr_min_k = 1;
        forced.block_min_k_batched = 48;  forced.block_min_batch = 8;
        set_svd_policy(forced);
        expect("batched block: 48x48 b7 and 47x47 b8 whole-matrix; 48x48 b8 block; 128x128 b1 block",
               svd_gpu_backend(48, 48, 7) == SvdBackend::jacobi &&
               svd_gpu_backend(47, 47, 8) == SvdBackend::jacobi &&
               svd_gpu_backend(48, 48, 8) == SvdBackend::block_jacobi &&
               svd_gpu_backend(128, 128, 1) == SvdBackend::block_jacobi);
        expect("batched block applies to the factor: 200x48 b8 -> qr_block, b4 -> qr",
               svd_gpu_backend(200, 48, 8) == SvdBackend::qr_block_jacobi &&
               svd_gpu_backend(200, 48, 4) == SvdBackend::qr_jacobi);
        api("forced -> block       8 x 48x48",  random_matrix(8, 48, 48, 980));
        api("forced -> qr_block    8 x 200x48", random_matrix(8, 200, 48, 981));
        forced.block_min_batch = 0;
        set_svd_policy(forced);
        expect("block_min_batch = 0 disables it",
               svd_gpu_backend(48, 48, 4096) == SvdBackend::jacobi);

        // The CPU boundary.
        unsetenv("SVD_DEVICE");
        forced = known;
        forced.gpu_max_k = 0;
        set_svd_policy(forced);
        expect("gpu_max_k = 0 -> never GPU", !svd_uses_gpu(8, 8, 4096) && !svd_uses_gpu(2048, 8, 4096));
        forced = known;
        forced.gpu_max_k = kSvdNoLimit;
        forced.gpu_min_batch_times_k = 0;
        set_svd_policy(forced);
        expect("no cap, no floor -> always GPU", svd_uses_gpu(1, 1, 1) && svd_uses_gpu(4096, 4096, 1));
        api("always GPU           12x12 b1", random_matrix(1, 12, 12, 976));
        forced.gpu_min_batch = 4;
        set_svd_policy(forced);
        expect("gpu_min_batch = 4 -> 512x512 b3 cpu, b4 gpu, 4x4 b3 cpu",
               !svd_uses_gpu(512, 512, 3) && svd_uses_gpu(512, 512, 4) && !svd_uses_gpu(4, 4, 3));
        forced.gpu_min_batch = 1;
        forced.gpu_max_l = 64;
        set_svd_policy(forced);
        expect("gpu_max_l = 64 -> 64x64 and 64x8 gpu, 65x8 and 8x65 cpu (the long side, either way round)",
               svd_uses_gpu(64, 64, 4096) && svd_uses_gpu(64, 8, 4096) && !svd_uses_gpu(65, 8, 4096) &&
               !svd_uses_gpu(8, 65, 4096));
        // The large-batch clause: above gpu_max_k, up to its own cap, from its batch.
        forced = known;
        forced.gpu_max_k = 48;  forced.gpu_min_batch_times_k = 16384;  forced.gpu_max_l = 256;
        forced.gpu_big_batch_max_k = 80;  forced.gpu_big_batch_min = 1024;
        set_svd_policy(forced);
        expect("big-batch clause (49..80 from 1024): 64x64 b1024 and 80x80 b1024 gpu, 64x64 b1023 and 81x81 b4096 cpu, "
               "64x300 b4096 cpu (l > 256), 32x32 b512 by the product rule",
               svd_uses_gpu(64, 64, 1024) && svd_uses_gpu(80, 80, 1024) && !svd_uses_gpu(64, 64, 1023) &&
               !svd_uses_gpu(81, 81, 4096) && !svd_uses_gpu(64, 300, 4096) && svd_uses_gpu(32, 32, 512) &&
               !svd_uses_gpu(16, 16, 512));
        forced.gpu_big_batch_min = 0;
        set_svd_policy(forced);
        expect("gpu_big_batch_min = 0 -> no clause (64x64 b4096 cpu)", !svd_uses_gpu(64, 64, 4096));

        set_svd_policy(original);
        const SvdPolicy back = svd_policy();
        expect("policy restored",
               back.qr_min_rows == original.qr_min_rows && back.qr_min_k == original.qr_min_k &&
               back.block_min_k == original.block_min_k &&
               back.block_min_k_batched == original.block_min_k_batched &&
               back.block_min_batch == original.block_min_batch &&
               back.gpu_max_k == original.gpu_max_k &&
               back.gpu_min_batch_times_k == original.gpu_min_batch_times_k &&
               back.gpu_min_batch == original.gpu_min_batch && back.gpu_max_l == original.gpu_max_l &&
               back.gk_min_k == original.gk_min_k && back.gk_max_k == original.gk_max_k);
    }

    // The bidiag backend replaces the CPU from its thresholds; 0 = never.
    std::printf("\n[ routing: bidiag ]\n");
    {
        const SvdPolicy original = svd_policy();
        auto expect = [&](const std::string& label, bool ok) {
            ++g_checks;
            if (ok) std::printf("  ok    %s\n", label.c_str()); else fail(label, "");
        };
        unsetenv("SVD_DEVICE");
        SvdPolicy p = original;
        p.gpu_max_k = 0;              // the CPU unless bidiag
        p.bidiag_min_k = 0;
        p.values_bidiag_min_k = 0;
        p.values_band_min_k = 0;      // band, tested below
        set_svd_policy(p);
        expect("thresholds 0 -> never (4096x4096 -> cpu)",
               svd_backend(4096, 4096, 1) == SvdBackend::cpu && svdvals_backend(4096, 4096, 1) == SvdBackend::cpu);
        p.bidiag_min_k = 256;
        p.values_bidiag_min_k = 1024;
        set_svd_policy(p);
        expect("bidiag_min_k = 256: k=255 cpu, k=256 bidiag (tall 2000x256 too)",
               svd_backend(255, 400, 1) == SvdBackend::cpu && svd_backend(256, 400, 1) == SvdBackend::bidiag &&
               svd_backend(2000, 256, 1) == SvdBackend::bidiag);
        expect("values_bidiag_min_k = 1024: svdvals k=512 cpu, k=1024 bidiag",
               svdvals_backend(512, 512, 1) == SvdBackend::cpu && svdvals_backend(1024, 1024, 1) == SvdBackend::bidiag);
        {
            array A = random_matrix(1, 400, 300, 1600);
            auto [U, S, Vt] = svd_accelerated(A);
            eval({U, S, Vt});
            check("routed to bidiag (400x300)", A, SvdResult{U, S, Vt, full({}, (uint32_t)(1u | (1u << 16)))});
        }
        {
            SvdPolicy c = svd_policy();
            c.bidiag_max_batch = 2;
            c.values_bidiag_max_batch = 1;
            set_svd_policy(c);
            expect("bidiag_max_batch = 2: k=256 batch 2 bidiag, batch 3 cpu; values cap 1: svdvals k=1024 b2 cpu",
                   svd_backend(256, 256, 2) == SvdBackend::bidiag && svd_backend(256, 256, 3) == SvdBackend::cpu &&
                   svdvals_backend(1024, 1024, 1) == SvdBackend::bidiag &&
                   svdvals_backend(1024, 1024, 2) == SvdBackend::cpu);
            c.bidiag_max_batch = 0;
            c.values_bidiag_max_batch = 0;
            set_svd_policy(c);
        }
        {   // the band backend, singular values alone, before bidiag
            SvdPolicy c = svd_policy();
            c.values_band_min_k = 2048;
            set_svd_policy(c);
            expect("values_band_min_k = 2048: svdvals k=1024 bidiag, k=2048 band; svd k=2048 bidiag",
                   svdvals_backend(1024, 1024, 1) == SvdBackend::bidiag &&
                   svdvals_backend(2048, 3000, 1) == SvdBackend::band &&
                   svd_backend(2048, 2048, 1) == SvdBackend::bidiag);
            c.values_bidiag_max_batch = 1;
            set_svd_policy(c);
            expect("band within values_bidiag_max_batch: svdvals k=2048 batch 2 cpu",
                   svdvals_backend(2048, 2048, 2) == SvdBackend::cpu);
            c.values_bidiag_max_batch = 0;
            c.values_band_min_k = 256;
            set_svd_policy(c);
            {
                array A = random_matrix(1, 400, 300, 1700);
                array s = svdvals_accelerated(A);
                SvdResult ref = detail::svd_cpu(A, false);
                eval({s, ref.S});
                ++g_checks;
                const float d = max_abs(subtract(s, ref.S)) / max_abs(ref.S);
                if (!(d <= 2e-5f)) fail("svdvals routed to band (400x300)", std::to_string(d));
                else std::printf("  ok    %-44s |ds|=%.1e\n", "svdvals routed to band (400x300)", d);
                for (unsigned width : {8u, 32u}) {   // the policy's band width reaches the backend
                    c.values_band_width = width;
                    set_svd_policy(c);
                    array sb = svdvals_accelerated(A);
                    eval({sb});
                    ++g_checks;
                    const float db = max_abs(subtract(sb, ref.S)) / max_abs(ref.S);
                    const std::string label = "svdvals band, values_band_width = " + std::to_string(width);
                    if (!(db <= 2e-5f) || !(max_abs(subtract(sb, s)) > 0.0f)) fail(label, std::to_string(db));
                    else std::printf("  ok    %-44s |ds|=%.1e\n", label.c_str(), db);
                }
                c.values_band_width = 0;
            }
            c.values_band_min_k = 0;
            set_svd_policy(c);
            expect("values_band_min_k = 0: never", svdvals_backend(4096, 4096, 1) == SvdBackend::bidiag);
            // band_min_k: the SVD with vectors, within bidiag's cap; svd_accelerated
            // reaches the backend
            const SvdPolicy before = c;
            c.gpu_max_k = 0;   // no GPU kernels: the CPU's region, then bidiag, then band
            c.gpu_big_batch_max_k = 0;
            c.bidiag_min_k = 128;
            c.band_min_k = 256;
            set_svd_policy(c);
            expect("band_min_k: with vectors from its k", svd_backend(400, 300, 1) == SvdBackend::band &&
                                                          svd_backend(200, 200, 1) == SvdBackend::bidiag &&
                                                          svdvals_backend(400, 300, 1) != SvdBackend::band);
            {
                array A = random_matrix(1, 400, 300, 1710);
                auto [U, S, Vt] = svd_accelerated(A);
                eval({U, S, Vt});
                check("svd routed to band (400x300)", A, SvdResult{U, S, Vt, full({}, (uint32_t)(1u | (1u << 16)))});
            }
            c = before;
            set_svd_policy(c);
            expect("band_min_k = 0: never", svd_backend(4096, 4096, 1) == SvdBackend::bidiag);
        }
        setenv("SVD_DEVICE", "band", 1);
        expect("SVD_DEVICE=band: svdvals and svd band",
               svdvals_backend(8, 8, 1) == SvdBackend::band && svd_backend(8, 8, 1) == SvdBackend::band);
        setenv("SVD_DEVICE", "cpu", 1);
        expect("SVD_DEVICE=cpu keeps the CPU over bidiag", svd_backend(4096, 4096, 1) == SvdBackend::cpu);
        setenv("SVD_DEVICE", "bidiag", 1);
        expect("SVD_DEVICE=bidiag forces it", svd_backend(16, 16, 4096) == SvdBackend::bidiag &&
                                              svdvals_backend(8, 8, 1) == SvdBackend::bidiag);
        unsetenv("SVD_DEVICE");
        set_svd_policy(original);
    }

    // The golub_kahan window replaces the Jacobi backends on the GPU; it is
    // clipped to what the backend takes, and the CPU rule still decides first.
    std::printf("\n[ routing: golub_kahan ]\n");
    {
        const SvdPolicy original = svd_policy();
        auto expect = [&](const std::string& label, bool ok) {
            ++g_checks;
            if (ok) std::printf("  ok    %s\n", label.c_str()); else fail(label, "");
        };
        unsetenv("SVD_DEVICE");
        SvdPolicy p;              // fixed here, as in the routing checks above
        p.gpu_max_k = 64;  p.gpu_min_batch_times_k = 1024;  p.gpu_min_batch = 1;
        set_svd_policy(p);
        expect("gk_max_k = 0 -> never (32x32 b4096 -> jacobi)", svd_backend(32, 32, 4096) == SvdBackend::jacobi);
        p.gk_min_k = 8;  p.gk_max_k = 48;
        set_svd_policy(p);
        expect("window 8 .. 48: 7x7 jacobi, 8x8 and 48x48 golub_kahan, 49x49 jacobi",
               svd_gpu_backend(7, 7, 1) == SvdBackend::jacobi && svd_gpu_backend(8, 8, 1) == SvdBackend::golub_kahan &&
               svd_gpu_backend(48, 48, 1) == SvdBackend::golub_kahan && svd_gpu_backend(49, 49, 1) == SvdBackend::jacobi);
        expect("tall that fits golub_kahan, too long qr_golub_kahan; wide by its transpose",
               svd_gpu_backend(300, 16, 1) == SvdBackend::golub_kahan &&
               svd_gpu_backend(4000, 16, 1) == SvdBackend::qr_golub_kahan &&
               svd_gpu_backend(16, 300, 1) == SvdBackend::golub_kahan &&
               svd_gpu_backend(16, 4000, 1) == SvdBackend::qr_golub_kahan);
        expect("the CPU rule first: 32x32 b1 -> cpu; svdvals the same window: 32x32 b4096 -> golub_kahan",
               svd_backend(32, 32, 1) == SvdBackend::cpu && svdvals_backend(32, 32, 4096) == SvdBackend::golub_kahan);
        {
            array A = random_matrix(256, 24, 24, 2600);
            auto [U, S, Vt] = svd_accelerated(A);
            array S2 = svdvals_accelerated(A);
            eval({U, S, Vt, S2});
            check("routed to golub_kahan (256 x 24x24)", A, SvdResult{U, S, Vt, full({256}, (uint32_t)(1u | (1u << 16)))});
            ++g_checks;
            const float d = max_abs(subtract(S, S2)) / std::max(max_abs(S), 1e-30f);
            if (d > 1e-6f) fail("routed svdvals == svd (golub_kahan)", "differ by " + std::to_string(d));
            else std::printf("  ok    %-44s |dS|=%.1e\n", "routed svdvals == svd (golub_kahan)", d);
            array B = random_matrix(64, 4000, 16, 2601);
            auto [U2, S3, Vt2] = svd_accelerated(B);
            eval({U2, S3, Vt2});
            check("routed to qr_golub_kahan (64 x 4000x16)", B,
                  SvdResult{U2, S3, Vt2, full({64}, (uint32_t)(1u | (1u << 16)))});
        }
        // svdvals' own GPU-or-CPU rule; values_gpu_min_batch = 0 follows svd's.
        p.values_gpu_max_k = 16;  p.values_gpu_min_batch_times_k = 0;  p.values_gpu_min_batch = 1;
        set_svd_policy(p);
        expect("svdvals' own rule (k <= 16): 16x16 b4096 golub_kahan, 32x32 b4096 cpu while svd keeps the GPU",
               svdvals_uses_gpu(16, 16, 4096) && !svdvals_uses_gpu(32, 32, 4096) && svd_uses_gpu(32, 32, 4096) &&
               svdvals_backend(16, 16, 4096) == SvdBackend::golub_kahan &&
               svdvals_backend(32, 32, 4096) == SvdBackend::cpu);
        p.values_gpu_min_batch = 0;
        set_svd_policy(p);
        expect("values_gpu_min_batch = 0 -> as with vectors: 32x32 b4096 golub_kahan",
               svdvals_uses_gpu(32, 32, 4096) && svdvals_backend(32, 32, 4096) == SvdBackend::golub_kahan);
        // Sharing a batch with the CPU, from share_min_batch on, golub_kahan only.
        p.share_min_batch = 512;
        set_svd_policy(p);
        expect("share_min_batch = 512: 32x32 b512 shared, b511 not, 49x49 b4096 (jacobi) not",
               svd_shares_batch(32, 32, 512) && !svd_shares_batch(32, 32, 511) && !svd_shares_batch(49, 49, 4096) &&
               svdvals_shares_batch(32, 32, 4096));
        {
            array A = random_matrix(700, 24, 24, 2602);
            auto [U, S, Vt] = svd_accelerated(A);
            eval({U, S, Vt});
            check("routed, shared with the CPU (700 x 24x24)", A, SvdResult{U, S, Vt, full({700}, (uint32_t)(1u | (1u << 16)))});
        }
        p.share_min_batch = 0;
        set_svd_policy(p);
        expect("share_min_batch = 0 -> never", !svd_shares_batch(32, 32, 1 << 20));
        const unsigned kmax = metal_linalg::detail::svd_gk_max_k();
        p.gk_min_k = 1;  p.gk_max_k = kSvdNoLimit;
        set_svd_policy(p);
        expect("window clipped to the device's limit (" + std::to_string(kmax) + ")",
               svd_gpu_backend(kmax, kmax, 1) == SvdBackend::golub_kahan &&
               svd_gpu_backend(kmax + 1, kmax + 1, 1) == SvdBackend::jacobi);
        set_svd_policy(original);
    }

    // -------------------------------------------------------------------------
    // Public API. The Metal kernels are what is under test, so force the GPU.
    // -------------------------------------------------------------------------
    setenv("SVD_DEVICE", "gpu", 1);
    std::printf("\n[ public API ]\n");
    {
        array A = random_matrix(3, 40, 24, 800);
        auto [U, S, Vt] = svd_accelerated(A);
        eval({U, S, Vt});
        check("svd_accelerated batch 3 x 40x24", A,
              SvdResult{U, S, Vt, full({3}, (uint32_t)(1u | (1u << 16)))});

        array S2 = svdvals_accelerated(A);
        eval({S2});
        const float d = max_abs(subtract(S, S2)) / std::max(frobenius(A), 1.0f);
        ++g_checks;
        if (d > 1e-6f) fail("svdvals == svd singular values", "differ by " + std::to_string(d));
        else std::printf("  ok    %-44s |dS|=%.1e\n", "svdvals == svd singular values", d);
    }
    {
        // Non-contiguous input: a transposed view.
        array G = random_matrix(1, 30, 50, 801);
        run("transposed (non-contiguous) view 50x30", transpose(G));
    }
    {
        // One slice of a batch: contiguous but not page-aligned.
        array B = random_matrix(4, 30, 20, 802);
        run("batch slice (unaligned) 30x20", reshape(slice(B, {2, 0, 0}, {3, 30, 20}), {30, 20}));
    }
    run("int32 input 12x9", astype(multiply(random_matrix(1, 12, 9, 803), array(10.0f)), int32));

    // -------------------------------------------------------------------------
    // Structured input, where the answer is known or the rank is deficient.
    // -------------------------------------------------------------------------
    std::printf("\n[ structured ]\n");
    run("identity 32x32",       eye(32));
    run("zeros 16x16",          zeros({16, 16}));
    run("zeros 20x6",           zeros({20, 6}));
    run("diagonal 64x64",       diag(arange(64, float32)));
    run("ones 24x24 (rank 1)",  ones({24, 24}));
    {
        array x = random::normal({30, 1}, float32, 0.0f, 1.0f, std::nullopt, Device::cpu);
        array y = random::normal({1, 20}, float32, 0.0f, 1.0f, std::nullopt, Device::cpu);
        array A = matmul(x, y, Device::cpu);
        eval({A});
        run("rank one 30x20", A);
    }
    {
        // Rank 5 in a 40 x 16 matrix: eleven singular values are zero.
        array L = random::normal({40, 5}, float32, 0.0f, 1.0f, std::nullopt, Device::cpu);
        array R = random::normal({5, 16}, float32, 0.0f, 1.0f, std::nullopt, Device::cpu);
        array A = matmul(L, R, Device::cpu);
        eval({A});
        run("rank 5 of 40x16", A);
        run("rank 5 of 16x40 (wide)", transpose(A));
    }
    {
        // Two identical columns and one zero column.
        array A = random_matrix(1, 12, 6, 810);
        eval({A});
        std::vector<float> d(A.data<float>(), A.data<float>() + 12 * 6);
        for (int i = 0; i < 12; ++i) { d[i * 6 + 4] = d[i * 6 + 1]; d[i * 6 + 5] = 0.0f; }
        run("duplicate and zero columns 12x6", from_values(d, {12, 6}));
    }
    {
        std::vector<float> spec(48);
        for (int i = 0; i < 48; ++i) spec[i] = std::pow(10.0f, 4.0f - 8.0f * i / 47.0f);
        run("singular values 1e+4 .. 1e-4 (60x48)", with_singular_values(60, 48, spec));
    }
    {
        std::vector<float> spec(64);
        for (int i = 0; i < 64; ++i) spec[i] = (float)(1 + i % 4);   // four 16-fold values
        run("repeated singular values (64x64)", with_singular_values(64, 64, spec));
    }
    {
        // Columns of wildly different magnitude: the relative rotation test
        // and the asymptotic rotation must both hold up.
        array A = random_matrix(1, 40, 8, 811);
        eval({A});
        std::vector<float> d(A.data<float>(), A.data<float>() + 40 * 8);
        for (int i = 0; i < 40; ++i)
            for (int c = 0; c < 8; ++c) d[i * 8 + c] *= std::pow(10.0f, -(float)c * 0.75f);
        run("graded columns 1 .. 1e-5 (40x8)", from_values(d, {40, 8}));
    }
    {
        array A = random_matrix(1, 24, 24, 812);
        run("scaled 1e-30 (24x24)", multiply(A, array(1e-30f)));
        run("scaled 1e+37 (24x24)", multiply(A, array(1e37f)));
    }
    {
        // 2x2 with a closed form: [[3, 0], [4, 5]] has singular values
        // sqrt(45) and sqrt(5).
        array A = from_values({3.0f, 0.0f, 4.0f, 5.0f}, {2, 2});
        SvdResult r = detail::svd_jacobi(A, true, {});
        eval({r.S});
        ++g_checks;
        const float* s = r.S.data<float>();
        if (std::fabs(s[0] - std::sqrt(45.0f)) > 1e-5f || std::fabs(s[1] - std::sqrt(5.0f)) > 1e-5f)
            fail("2x2 [[3,0],[4,5]]", "got " + std::to_string(s[0]) + ", " + std::to_string(s[1]));
        else std::printf("  ok    %-44s S={%.6f, %.6f}\n", "2x2 [[3,0],[4,5]] -> {sqrt45, sqrt5}", s[0], s[1]);
    }

    // -------------------------------------------------------------------------
    // Non-finite input: NaN out, flagged, no hang, no exception; and it must
    // not contaminate its neighbours in a batch.
    // -------------------------------------------------------------------------
    std::printf("\n[ non-finite input ]\n");
    {
        array good = random_matrix(3, 12, 8, 820);
        eval({good});
        std::vector<float> data(good.data<float>(), good.data<float>() + 3 * 12 * 8);
        data[1 * 12 * 8 + 5 * 8 + 2] = NAN;
        array A = from_values(data, {3, 12, 8});
        ++g_checks;
        try {
            SvdResult r = detail::svd_jacobi(A, true, {});
            eval({r.U, r.S, r.Vt, r.info});
            const uint32_t* info = r.info.data<uint32_t>();
            const bool flags_ok = detail::svd_converged(info[0]) && detail::svd_nonfinite(info[1]) &&
                                  detail::svd_converged(info[2]);
            array s1 = reshape(slice(r.S, {1, 0}, {2, 8}), {8});
            const bool all_nan = !any(logical_not(isnan(s1))).item<bool>();
            if (!flags_ok || !all_nan) fail("NaN in a batch", "flags or NaN propagation wrong");
            else {
                std::printf("  ok    NaN matrix flagged and all-NaN; neighbours converged\n");
                check("NaN in a batch, matrix 0 intact",
                      reshape(slice(A, {0, 0, 0}, {1, 12, 8}), {12, 8}),
                      SvdResult{reshape(slice(r.U, {0, 0, 0}, {1, 12, 8}), {12, 8}),
                                reshape(slice(r.S, {0, 0}, {1, 8}), {8}),
                                reshape(slice(r.Vt, {0, 0, 0}, {1, 8, 8}), {8, 8}),
                                array((uint32_t)(1u | (1u << 16)))});
            }
        } catch (const std::exception& e) {
            fail("NaN in a batch", std::string("threw: ") + e.what());
        }
    }

    // -------------------------------------------------------------------------
    // Errors.
    // -------------------------------------------------------------------------
    std::printf("\n[ errors ]\n");
    {
        ++g_checks;
        try { svd_accelerated(zeros({4})); fail("1-D input", "did not throw"); }
        catch (const std::invalid_argument&) { std::printf("  ok    1-D input throws\n"); }
        ++g_checks;
        try {
            SvdOptions o; o.max_sweeps = 1;
            detail::svd_jacobi(random_matrix(1, 16, 16, 830), true, o);
            fail("max_sweeps=1", "did not throw");
        } catch (const std::runtime_error&) { std::printf("  ok    max_sweeps=1 on a full matrix throws\n"); }
    }

    std::printf("\n%d checks, %d failures\n\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
