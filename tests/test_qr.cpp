// Correctness tests for the Metal QR backends.
//
// Every case is checked four ways: output shapes, reconstruction (Q*R == A),
// orthogonality (Q^T*Q == I) and upper-triangularity of R. The backends are
// also exercised directly, not just through the dispatcher, so a backend that
// is only reachable for a narrow shape range still gets covered.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include <mlx/mlx.h>

#include <metal_linalg/qr.h>

using namespace mlx::core;
using namespace metal_linalg;

namespace {

// float32 Householder QR loses roughly sqrt(n) digits; these bounds are on the
// *relative* Frobenius error, so they hold across the whole size range.
constexpr float kReconTol = 1e-4f;
constexpr float kOrthoTol = 1e-4f;
constexpr float kTriuTol  = 1e-5f;

int g_failures = 0;
int g_checks   = 0;

array random_matrix(int batch, int M, int N, unsigned seed) {
    std::mt19937 rng(seed);
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

// Transposes the last two axes of a 2D or 3D array.
array transpose_last_two(const array& x) {
    std::vector<int> axes(x.ndim());
    for (size_t i = 0; i < axes.size(); ++i) axes[i] = (int)i;
    std::swap(axes[axes.size() - 1], axes[axes.size() - 2]);
    return transpose(x, axes);
}

void fail(const std::string& label, const std::string& what) {
    std::printf("  FAIL  %-46s %s\n", label.c_str(), what.c_str());
    ++g_failures;
}

// Runs the full battery of checks on one factorisation.
void check_factorisation(const std::string& label, const array& A,
                         const array& Q, const array& R) {
    ++g_checks;

    const auto& shape = A.shape();
    const int M = shape[shape.size() - 2];
    const int N = shape[shape.size() - 1];
    const int K = std::min(M, N);

    // --- Shapes: economic QR, Q is [..., M, K] and R is [..., K, N] ---
    Shape want_Q(shape.begin(), shape.end()); want_Q[want_Q.size() - 1] = K;
    Shape want_R(shape.begin(), shape.end()); want_R[want_R.size() - 2] = K;
    if (Q.shape() != want_Q) { fail(label, "Q has the wrong shape"); return; }
    if (R.shape() != want_R) { fail(label, "R has the wrong shape"); return; }

    if (has_non_finite(Q) || has_non_finite(R)) {
        fail(label, "Q or R contains NaN/Inf");
        return;
    }

    // --- Reconstruction: ||Q*R - A||_F / ||A||_F ---
    const float norm_A = std::max(frobenius(A), 1.0f);
    const float recon  = frobenius(subtract(matmul(Q, R), A)) / norm_A;

    // --- Orthogonality: ||Q^T*Q - I||_F, normalised by the identity's norm ---
    array QtQ   = matmul(transpose_last_two(Q), Q);
    const float ortho = frobenius(subtract(QtQ, eye(K))) / std::sqrt((float)K);

    // --- Structure: everything strictly below R's diagonal must be zero ---
    const float triu = max_abs(tril(R, -1));

    const bool ok = recon <= kReconTol && ortho <= kOrthoTol && triu <= kTriuTol;
    std::printf("  %s  %-46s recon=%.2e ortho=%.2e subdiag=%.2e\n",
                ok ? "ok  " : "FAIL", label.c_str(), recon, ortho, triu);
    if (!ok) ++g_failures;
}

using QrFn = std::pair<array, array> (*)(const array&);

void run(const std::string& label, QrFn qr, const array& A) {
    auto [Q, R] = qr(A);
    eval({Q, R});
    check_factorisation(label, A, Q, R);
}

array from_values(std::vector<float> v, Shape shape) {
    return array(v.begin(), std::move(shape), float32);
}

} // namespace


// The modes (core::QrMode) of a raw backend: R alone matches the reduced R,
// and a square Q is orthogonal, its first K columns the reduced Q, R's rows
// below K zero. `fn` takes (a, q, r, mode).
using ModeFn = void (*)(const core::Matrices&, float*, float*, core::QrMode);
void check_modes(const std::string& label, ModeFn fn, uint32_t batch, uint32_t M, uint32_t N, uint32_t seed) {
    ++g_checks;
    const uint32_t K = std::min(M, N);
    std::mt19937 gen(seed);
    std::normal_distribution<float> dist;
    std::vector<float> a((size_t)batch * M * N);
    for (float& x : a) x = dist(gen);
    const core::Matrices A{a.data(), batch, M, N};
    std::vector<float> q((size_t)batch * M * K), r((size_t)batch * K * N);
    std::vector<float> rr((size_t)batch * K * N, -7.0f);
    std::vector<float> qc((size_t)batch * M * M, -7.0f), rc((size_t)batch * M * N, -7.0f);
    std::string what;
    try {
        fn(A, q.data(), r.data(), core::QrMode::reduced);
        fn(A, nullptr, rr.data(), core::QrMode::r);
        fn(A, qc.data(), rc.data(), core::QrMode::complete);
    } catch (const std::exception& e) {
        what = e.what();
    }
    double r_diff = 0, q_diff = 0, orth = 0, rec = 0, below = 0;
    if (what.empty()) {
        for (uint32_t b = 0; b < batch; ++b) {
            const float* Ab = a.data() + (size_t)b * M * N;
            const float* Rr = r.data() + (size_t)b * K * N;
            const float* Ro = rr.data() + (size_t)b * K * N;
            const float* Qc = qc.data() + (size_t)b * M * M;
            const float* Rc = rc.data() + (size_t)b * M * N;
            const float* Qr = q.data() + (size_t)b * M * K;
            for (size_t i = 0; i < (size_t)K * N; ++i) r_diff = std::max(r_diff, (double)std::fabs(Ro[i] - Rr[i]));
            for (uint32_t i = 0; i < M; ++i)
                for (uint32_t j = 0; j < K; ++j)
                    q_diff = std::max(q_diff, (double)std::fabs(Qc[(size_t)i * M + j] - Qr[(size_t)i * K + j]));
            for (uint32_t i = K; i < M; ++i)
                for (uint32_t j = 0; j < N; ++j) below = std::max(below, (double)std::fabs(Rc[(size_t)i * N + j]));
            for (uint32_t i = 0; i < M; ++i)       // Q^T Q - I over all M columns
                for (uint32_t j = 0; j < M; ++j) {
                    double d = 0;
                    for (uint32_t l = 0; l < M; ++l) d += (double)Qc[(size_t)l * M + i] * Qc[(size_t)l * M + j];
                    orth = std::max(orth, std::fabs(d - (i == j ? 1.0 : 0.0)));
                }
            double num = 0, den = 0;              // A - Q R with the square factors
            for (uint32_t i = 0; i < M; ++i)
                for (uint32_t j = 0; j < N; ++j) {
                    double d = 0;
                    for (uint32_t l = 0; l < M; ++l) d += (double)Qc[(size_t)i * M + l] * Rc[(size_t)l * N + j];
                    num += (d - Ab[(size_t)i * N + j]) * (d - Ab[(size_t)i * N + j]);
                    den += (double)Ab[(size_t)i * N + j] * Ab[(size_t)i * N + j];
                }
            rec = std::max(rec, std::sqrt(num / std::max(den, 1e-30)));
        }
    }
    char buf[200];
    std::snprintf(buf, sizeof buf, "R alone %.1e  Q's first K %.1e  orth %.1e  recon %.1e  below K %.1e", r_diff, q_diff,
                  orth, rec, below);
    const std::string name = label + " " + std::to_string(batch) + " x " + std::to_string(M) + "x" + std::to_string(N);
    if (!what.empty()) fail(name, what);
    else if (r_diff > 1e-5 || q_diff > 1e-5 || orth > 1e-4 || rec > 1e-4 || below != 0.0) fail(name, buf);
    else std::printf("  ok    %-46s %s\n", name.c_str(), buf);
}

int main() {
    set_default_device(Device::gpu);

    std::printf("\nQR correctness tests\n");
    std::printf("tolerances: recon<=%.0e ortho<=%.0e subdiag<=%.0e\n",
                kReconTol, kOrthoTol, kTriuTol);

    // -------------------------------------------------------------------------
    // Each backend directly. These do not depend on the dispatch heuristics, so
    // they stay meaningful even if the thresholds in src/qr.mm change.
    // -------------------------------------------------------------------------
    std::printf("\n[ backend: qr_unblocked ]\n");
    run("2x2",             detail::qr_unblocked, random_matrix(1, 2, 2, 1));
    run("8x8",             detail::qr_unblocked, random_matrix(1, 8, 8, 2));
    run("64x64",           detail::qr_unblocked, random_matrix(1, 64, 64, 3));
    run("64x32 (tall)",    detail::qr_unblocked, random_matrix(1, 64, 32, 4));
    run("32x64 (wide)",    detail::qr_unblocked, random_matrix(1, 32, 64, 5));
    run("batch 32 x 8x8",  detail::qr_unblocked, random_matrix(32, 8, 8, 6));
    run("batch 16 x 64x64",detail::qr_unblocked, random_matrix(16, 64, 64, 7));

    run("batch 3 x 5000x40 (beyond the Householder kernels)", detail::qr_unblocked, random_matrix(3, 5000, 40, 11));

    // The Householder kernels: in registers, B columns (8, 16, 32) by R rows a
    // lane (1, 2, 4), every instance and the edges of each; blocked beyond
    // them and when forced: padding to 8 rows and columns and to 16 for K,
    // one block of 32 columns or several and a part block, one to four rows a
    // thread, wide and tall.
    std::printf("\n[ backend: qr_householder ]\n");
    {
        const std::vector<std::pair<int, int>> shapes = {
            {1, 1}, {2, 1}, {1, 2}, {5, 3}, {3, 5}, {8, 8}, {33, 8}, {64, 8}, {65, 8}, {128, 8},
            {9, 9}, {16, 16}, {17, 16}, {40, 16}, {100, 16}, {128, 16}, {17, 17}, {31, 31}, {32, 32},
            {32, 20}, {60, 32}, {128, 32}, {33, 33}, {48, 40}, {64, 64}, {40, 64}, {20, 64}, {8, 30},
            {129, 8}, {200, 30}, {80, 80}, {20, 90}, {96, 96}, {100, 100}, {128, 128}, {130, 70},
            {200, 300}, {17, 300}, {300, 200}, {256, 256}, {513, 77}, {1000, 64}, {2048, 48}, {4096, 16},
            {64, 2048}, {520, 520}};
        int seed = 600;
        for (auto [M, N] : shapes) {
            if (!core::detail::qr_householder_fits(M, N)) { std::printf("  skip  %dx%d (does not fit)\n", M, N); continue; }
            run(std::to_string(M) + "x" + std::to_string(N), detail::qr_householder, random_matrix(1, M, N, seed++));
        }
        run("batch 37 x 16x16 (4 a threadgroup)", detail::qr_householder, random_matrix(37, 16, 16, 650));
        run("batch 5 x 64x64",                   detail::qr_householder, random_matrix(5, 64, 64, 651));
        run("batch 3 x 200x30",                  detail::qr_householder, random_matrix(3, 200, 30, 652));
        run("batch slice (unaligned) 30x20",     detail::qr_householder,
            reshape(slice(random_matrix(3, 30, 20, 653), {1, 0, 0}, {2, 30, 20}), {30, 20}));
        run("transposed view 40x24",             detail::qr_householder, transpose(random_matrix(1, 24, 40, 654)));
        run("batch 9 x 128x128",                 detail::qr_householder, random_matrix(9, 128, 128, 658));
        run("batch 3 x 300x90",                  detail::qr_householder, random_matrix(3, 300, 90, 659));
        run("batch slice (unaligned) 70x50",     detail::qr_householder,
            reshape(slice(random_matrix(3, 70, 50, 661), {1, 0, 0}, {2, 70, 50}), {70, 50}));
        run("transposed view 90x60",             detail::qr_householder, transpose(random_matrix(1, 60, 90, 662)));
        setenv("QR_HOUSEHOLDER_SIMD", "0", 1);
        run("32x32 (blocked kernel)",            detail::qr_householder, random_matrix(1, 32, 32, 655));
        run("batch 7 x 48x40 (blocked kernel)",  detail::qr_householder, random_matrix(7, 48, 40, 656));
        run("20x64 (blocked kernel)",            detail::qr_householder, random_matrix(1, 20, 64, 657));
        run("batch 37 x 16x16 (blocked kernel)", detail::qr_householder, random_matrix(37, 16, 16, 663));
        run("1x1 (blocked kernel)",              detail::qr_householder, random_matrix(1, 1, 1, 664));
        run("5x3 (blocked kernel)",              detail::qr_householder, random_matrix(1, 5, 3, 665));
        unsetenv("QR_HOUSEHOLDER_SIMD");
        {
            std::vector<float> eye(48 * 48, 0.0f);
            for (int i = 0; i < 48; ++i) eye[i * 48 + i] = 1.0f;
            run("identity 48x48",   detail::qr_householder, from_values(eye, {48, 48}));
            run("zeros 32x32",      detail::qr_householder, from_values(std::vector<float>(32 * 32, 0.0f), {32, 32}));
            run("zeros 100x60",     detail::qr_householder, from_values(std::vector<float>(100 * 60, 0.0f), {100, 60}));
            run("constant 50x10",   detail::qr_householder, from_values(std::vector<float>(50 * 10, 0.5f), {50, 10}));
            run("constant 90x70",   detail::qr_householder, from_values(std::vector<float>(90 * 70, 0.5f), {90, 70}));
            run("constant 600x64",  detail::qr_householder, from_values(std::vector<float>(600 * 64, 1.0f), {600, 64}));
            run("constant 2048x96", detail::qr_householder, from_values(std::vector<float>(2048 * 96, 1.0f), {2048, 96}));
        }
        for (bool simd : {true, false}) {   // NaN in one matrix of a batch: that matrix NaN, the others not
            if (!simd) setenv("QR_HOUSEHOLDER_SIMD", "0", 1);   // the blocked kernel
            ++g_checks;
            const int B = 6, M = 24, N = 20, K = 20;
            array A = random_matrix(B, M, N, 660);
            eval({A});
            std::vector<float> v(A.data<float>(), A.data<float>() + B * M * N);
            v[3 * M * N + 17] = INFINITY;
            auto [Q, R] = detail::qr_householder(from_values(v, {B, M, N}));
            eval({Q, R});
            const float* q = Q.data<float>();
            bool ok = true;
            for (int b = 0; b < B; ++b)
                for (int i = 0; i < M * K; ++i) ok &= (b == 3) == std::isnan(q[b * M * K + i]);
            const std::string label = std::string("non-finite in one matrix of 6 (") + (simd ? "registers" : "blocked") + ")";
            if (!ok) fail(label, "not isolated");
            else std::printf("  ok    %-46s\n", label.c_str());
            unsetenv("QR_HOUSEHOLDER_SIMD");
        }
        // Up to 16 rows the register kernel packs four or two matrices into a
        // simdgroup: the same factors as one a simdgroup (QR_SIMD_PACK=0), to
        // rounding (a group's sums add in another order than simd_sum), and a
        // non-finite matrix leaves its neighbours alone
        for (auto [B, M, N] : std::vector<std::tuple<int, int, int>>{{37, 5, 5}, {23, 12, 9}, {19, 6, 10}, {9, 16, 16},
                                                                     {11, 3, 30}}) {
            array A = random_matrix(B, M, N, 670 + M * 3 + N);
            auto [Qp, Rp] = detail::qr_householder(A);
            setenv("QR_SIMD_PACK", "0", 1);
            auto [Qu, Ru] = detail::qr_householder(A);
            unsetenv("QR_SIMD_PACK");
            eval({Qp, Rp, Qu, Ru});
            ++g_checks;
            const float d = std::max(max_abs(subtract(Qp, Qu)), max_abs(subtract(Rp, Ru)));
            const std::string label = "packed == one a simdgroup, batch " + std::to_string(B) + " x " + std::to_string(M) + "x" +
                                      std::to_string(N);
            if (!(d < 1e-5f)) fail(label, "differ by " + std::to_string(d));
            else std::printf("  ok    %-46s\n", label.c_str());
        }
        for (auto [M, N] : std::vector<std::pair<int, int>>{{6, 5}, {13, 9}}) {
            ++g_checks;
            const int B = 9, K = std::min(M, N);
            array A = random_matrix(B, M, N, 680 + M);
            eval({A});
            std::vector<float> v(A.data<float>(), A.data<float>() + B * M * N);
            v[(size_t)5 * M * N + 2] = NAN;
            auto [Q, R] = detail::qr_householder(from_values(v, {B, M, N}));
            eval({Q, R});
            const float* q = Q.data<float>();
            bool ok = true;
            for (int b = 0; b < B; ++b)
                for (int i = 0; i < M * K; ++i) ok &= (b == 5) == std::isnan(q[b * M * K + i]);
            const std::string label = "non-finite in one of 9 packed " + std::to_string(M) + "x" + std::to_string(N);
            if (!ok) fail(label, "not isolated");
            else std::printf("  ok    %-46s\n", label.c_str());
        }
        ++g_checks;
        if (!core::detail::qr_householder_preferred(64, 64) || !core::detail::qr_householder_preferred(128, 32) ||
            !core::detail::qr_householder_preferred(200, 30) || !core::detail::qr_householder_preferred(80, 80) ||
            !core::detail::qr_householder_preferred(512, 512) || !core::detail::qr_householder_preferred(4096, 16) ||
            core::detail::qr_householder_preferred(4097, 16) || core::detail::qr_householder_preferred(0, 5))
            fail("qr_householder_preferred", "wrong");
        else std::printf("  ok    %-46s\n", "qr_householder_preferred: up to 4096 rows");
    }

    std::printf("\n[ backend: qr_streaming_amx_complete ]\n");
    run("4x4",             detail::qr_streaming_amx_complete, random_matrix(1, 4, 4, 8));
    run("8x8",             detail::qr_streaming_amx_complete, random_matrix(1, 8, 8, 9));
    run("batch 4 x 8x8",   detail::qr_streaming_amx_complete, random_matrix(4, 8, 8, 10));
    run("batch 16 x 8x8",  detail::qr_streaming_amx_complete, random_matrix(16, 8, 8, 11));

    std::printf("\n[ backend: qr_streaming_amx_reduced ]\n");
    run("128x128",         detail::qr_streaming_amx_reduced, random_matrix(1, 128, 128, 12));
    run("512x512",         detail::qr_streaming_amx_reduced, random_matrix(1, 512, 512, 13));
    run("512x256 (tall)",  detail::qr_streaming_amx_reduced, random_matrix(1, 512, 256, 14));
    run("256x512 (wide)",  detail::qr_streaming_amx_reduced, random_matrix(1, 256, 512, 15));
    run("batch 2 x 512x256", detail::qr_streaming_amx_reduced, random_matrix(2, 512, 256, 16));
    run("600x600 (unaligned)", detail::qr_streaming_amx_reduced, random_matrix(1, 600, 600, 17));

    std::printf("\n[ backend: qr_streaming_amx_reduced, its own kernels for one matrix ]\n");
    setenv("QR_BLOCKED", "0", 1);
    run("512x512 (streaming)",            detail::qr_streaming_amx_reduced, random_matrix(1, 512, 512, 18));
    run("600x600 (streaming, unaligned)", detail::qr_streaming_amx_reduced, random_matrix(1, 600, 600, 19));
    unsetenv("QR_BLOCKED");

    // The blocked QR: panels of 16 columns, by TSQR from 129 rows (a tree of
    // leaves up to 2^15); aggregates of 128 columns, a short last one; the
    // matrix padded with zero rows and columns to whole panels of twice
    // their width in rows.
    std::printf("\n[ backend: qr_blocked ]\n");
    run("1x1",                      detail::qr_blocked, from_values({3.0f}, {1, 1}));
    run("5x3",                      detail::qr_blocked, random_matrix(1, 5, 3, 39));
    run("3x5 (wide)",               detail::qr_blocked, random_matrix(1, 3, 5, 38));
    run("32x32",                    detail::qr_blocked, random_matrix(1, 32, 32, 40));
    run("40x40",                    detail::qr_blocked, random_matrix(1, 40, 40, 41));
    run("128x128",                  detail::qr_blocked, random_matrix(1, 128, 128, 42));
    run("300x300",                  detail::qr_blocked, random_matrix(1, 300, 300, 43));
    run("1000x300 (tall)",          detail::qr_blocked, random_matrix(1, 1000, 300, 44));
    run("300x1000 (wide)",          detail::qr_blocked, random_matrix(1, 300, 1000, 45));
    run("1023x517",                 detail::qr_blocked, random_matrix(1, 1023, 517, 46));
    run("1024x1024",                detail::qr_blocked, random_matrix(1, 1024, 1024, 47));
    run("2100x2048",                detail::qr_blocked, random_matrix(1, 2100, 2048, 48));
    run("8192x48 (tall, 16-wide)",  detail::qr_blocked, random_matrix(1, 8192, 48, 49));
    run("9000x40",                  detail::qr_blocked, random_matrix(1, 9000, 40, 50));
    run("20000x40 (129 leaves, 8 tree levels)", detail::qr_blocked, random_matrix(1, 20000, 40, 54));
    run("batch 2 x 17000x24",       detail::qr_blocked, random_matrix(2, 17000, 24, 55));
    run("batch 3 x 200x150",        detail::qr_blocked, random_matrix(3, 200, 150, 51));
    run("batch slice (unaligned) 300x200", detail::qr_blocked,
        reshape(slice(random_matrix(3, 300, 200, 52), {1, 0, 0}, {2, 300, 200}), {300, 200}));
    run("transposed view 600x64",   detail::qr_blocked, transpose(random_matrix(1, 64, 600, 53)));
    {
        std::vector<float> eye(256 * 256, 0.0f);
        for (int i = 0; i < 256; ++i) eye[i * 256 + i] = 1.0f;
        run("identity 256x256",     detail::qr_blocked, from_values(eye, {256, 256}));
        run("zeros 256x256",        detail::qr_blocked, from_values(std::vector<float>(256 * 256, 0.0f), {256, 256}));
        run("constant 300x200",     detail::qr_blocked, from_values(std::vector<float>(300 * 200, 0.5f), {300, 200}));
        // A constant matrix's trailing columns are rounding noise, then noise
        // of that, down to entries whose squares underflow some and not
        // others: before 2.17.0 the panels' reflectors there were not
        // orthogonal (Q off by 5e4 at 600x64).
        for (auto [M, N] : std::vector<std::pair<int, int>>{{600, 64}, {200, 100}, {1024, 1024}, {2048, 256}})
            run("constant " + std::to_string(M) + "x" + std::to_string(N), detail::qr_blocked,
                from_values(std::vector<float>((size_t)M * N, 1.0f), {M, N}));
        run("constant batch 4 x 600x64", detail::qr_blocked, full({4, 600, 64}, 1.0f));
    }
    // Panels of 8 columns for up to 4 matrices of 768-3072 rows, 16 for more:
    // one shape's workspace, kept between calls, must follow the width.
    run("batch 2 x 1024x800 (8-wide panels)", detail::qr_blocked, random_matrix(2, 1024, 800, 56));
    run("batch 6 x 1024x800 (16-wide panels)", detail::qr_blocked, random_matrix(6, 1024, 800, 57));
    run("batch 2 x 1024x800 again (8-wide)", detail::qr_blocked, random_matrix(2, 1024, 800, 58));
    ++g_checks;
    {
        std::vector<float> v(300 * 300, 1.0f);
        v[7] = NAN;
        auto [Q, R] = detail::qr_blocked(from_values(v, {300, 300}));
        eval({Q, R});
        if (!all(isnan(Q)).item<bool>() || !all(isnan(R)).item<bool>()) fail("blocked NaN input", "not NaN");
        else std::printf("  ok    %-46s\n", "NaN input gives NaN");
    }
    ++g_checks;
    if (core::detail::qr_blocked_fits(4194305, 1) || !core::detail::qr_blocked_fits(4194304, 1) ||
        !core::detail::qr_blocked_fits(1, 1))
        fail("qr_blocked_fits", "wrong");
    else std::printf("  ok    %-46s\n", "qr_blocked_fits: up to 2^22 rows");

    std::printf("\n[ backend: qr_cpu (LAPACK) ]\n");
    run("1x1",             detail::qr_cpu, random_matrix(1, 1, 1, 90));
    run("8x8",             detail::qr_cpu, random_matrix(1, 8, 8, 91));
    run("64x64",           detail::qr_cpu, random_matrix(1, 64, 64, 92));
    run("64x32 (tall)",    detail::qr_cpu, random_matrix(1, 64, 32, 93));
    run("32x64 (wide)",    detail::qr_cpu, random_matrix(1, 32, 64, 94));
    run("512x512",         detail::qr_cpu, random_matrix(1, 512, 512, 95));
    run("2048x64 (tall)",  detail::qr_cpu, random_matrix(1, 2048, 64, 96));
    run("batch 32 x 8x8",  detail::qr_cpu, random_matrix(32, 8, 8, 97));
    run("batch 4 x 100x60",detail::qr_cpu, random_matrix(4, 100, 60, 98));
    // Wide: the leading block's QR, then R2 = Q^T A2 (see qr_cpu.mm).
    run("64x2048 (wide)",  detail::qr_cpu, random_matrix(1, 64, 2048, 99));
    run("batch 5 x 30x45 (wide)", detail::qr_cpu, random_matrix(5, 30, 45, 100));
    run("1x7 (wide)",      detail::qr_cpu, random_matrix(1, 1, 7, 101));
    run("rank 3 of 40x90 (wide)", detail::qr_cpu, matmul(random_matrix(1, 40, 3, 102), random_matrix(1, 3, 90, 103)));

    // A batch shared between a GPU kernel and the CPU path: the GPU's chunks
    // from the front, the CPU's from the back; every matrix solved once.
    std::printf("\n[ shared with the CPU ]\n");
    run("shared 700 x 24x24 (unblocked)",  detail::qr_shared, random_matrix(700, 24, 24, 104));
    run("shared 300 x 600x64 (reduced)",   detail::qr_shared, random_matrix(300, 600, 64, 105));
    run("shared 3 x 40x30",                detail::qr_shared, random_matrix(3, 40, 30, 106));
    run("shared 500 x 20x50 (wide)",       detail::qr_shared, random_matrix(500, 20, 50, 107));
    // A batch is spread over cpu_threads() threads; one thread must agree.
    {
        array A = random_matrix(41, 48, 20, 99);
        auto [Q, R] = detail::qr_cpu(A);
        set_cpu_threads(1);
        auto [Q1, R1] = detail::qr_cpu(A);
        set_cpu_threads(0);
        eval({Q, R, Q1, R1});
        check_factorisation("batch 41 x 48x20, every thread", A, Q, R);
        array d = maximum(max(abs(subtract(Q, Q1))), max(abs(subtract(R, R1))));
        eval({d});
        ++g_checks;
        if (d.item<float>() > 1e-5f) fail("batch 41 x 48x20, one thread == every thread", "differ by " + std::to_string(d.item<float>()));
        else std::printf("  ok    %-44s |d|=%.1e\n", "batch 41 x 48x20, one thread == every thread", d.item<float>());
    }

    // -------------------------------------------------------------------------
    // Through the public dispatcher, wherever this device's policy sends them.
    // -------------------------------------------------------------------------
    std::printf("\n[ dispatcher: metal_linalg::qr_accelerated ]\n");
    run("micro square 8x8",          qr_accelerated, random_matrix(1, 8, 8, 20));
    run("small batched 32 x 64x64",  qr_accelerated, random_matrix(32, 64, 64, 21));
    run("small unbatched 64x64",     qr_accelerated, random_matrix(1, 64, 64, 22));
    run("mid unbatched 128x128",     qr_accelerated, random_matrix(1, 128, 128, 23));
    run("large 1024x512",            qr_accelerated, random_matrix(1, 1024, 512, 24));

    // -------------------------------------------------------------------------
    // Edge cases.
    // -------------------------------------------------------------------------

    // -------------------------------------------------------------------------
    // Modes: R alone and Q square, every backend
    // -------------------------------------------------------------------------
    std::printf("\n[ modes: R alone and Q square ]\n");
    {
        const std::vector<std::array<uint32_t, 3>> shapes = {
            {3, 20, 12}, {2, 12, 20}, {2, 16, 16}, {1, 1, 1}, {2, 5, 3}, {2, 3, 5}, {4, 64, 32}, {3, 100, 40},
            {2, 130, 70}, {1, 300, 200}, {2, 40, 100}, {1, 257, 33}};
        const std::vector<std::pair<const char*, ModeFn>> backends = {
            {"core::qr", core::qr},
            {"qr_cpu", core::detail::qr_cpu},
            {"qr_householder", core::detail::qr_householder},
            {"qr_blocked", core::detail::qr_blocked},
            {"qr_unblocked", core::detail::qr_unblocked},
            {"qr_shared", core::detail::qr_shared},
            {"qr_streaming_amx_reduced", core::detail::qr_streaming_amx_reduced},
        };
        uint32_t seed = 900;
        for (auto [label, fn] : backends)
            for (auto [b, m, n] : shapes) {
                if (std::string(label) == "qr_householder" && !core::detail::qr_householder_fits(m, n)) continue;
                check_modes(label, fn, b, m, n, seed++);
            }
        setenv("QR_HOUSEHOLDER_SIMD", "0", 1);   // the blocked kernel where the register one would take it
        for (auto [b, m, n] : shapes) check_modes("qr_householder (blocked kernel)", core::detail::qr_householder, b, m, n, seed++);
        unsetenv("QR_HOUSEHOLDER_SIMD");
        // The MLX API's mode: shapes, and a bad mode refused
        ++g_checks;
        {
            array A = random_matrix(3, 50, 20, 950);
            auto [Q0, R0] = qr_accelerated(A, "r");
            auto [Q1, R1] = qr_accelerated(A, "complete");
            auto [Q2, R2] = qr_accelerated(A);
            eval({Q0, R0, Q1, R1, Q2, R2});
            const bool shapes_ok = Q0.size() == 0 && R0.shape() == Shape{3, 20, 20} && Q1.shape() == Shape{3, 50, 50} &&
                                   R1.shape() == Shape{3, 50, 20} && Q2.shape() == Shape{3, 50, 20};
            const bool same_r = max_abs(subtract(R0, R2)) < 1e-5f;
            bool refused = false;
            try { qr_accelerated(A, "full"); } catch (const std::invalid_argument&) { refused = true; }
            if (!shapes_ok || !same_r || !refused) fail("qr_accelerated(a, mode)", "wrong shapes, R, or a bad mode taken");
            else std::printf("  ok    %-46s\n", "qr_accelerated(a, \"r\" | \"complete\"), a bad mode refused");
        }
    }

    std::printf("\n[ edge cases ]\n");

    // Identity: the shader must not trip over the exact zeros below the diagonal.
    std::vector<float> eye64(64 * 64, 0.0f);
    for (int i = 0; i < 64; ++i) eye64[i * 64 + i] = 1.0f;
    run("identity 64x64", qr_accelerated, from_values(eye64, {64, 64}));

    // All zeros: exercises the division-by-zero guard in the reflection.
    run("zeros 32x32", qr_accelerated, from_values(std::vector<float>(32 * 32, 0.0f), {32, 32}));

    // Rank deficient: row 2 is 2 * row 1.
    run("rank deficient 3x3", qr_accelerated, from_values(
        {1.0f, 2.0f, 3.0f,
         2.0f, 4.0f, 6.0f,
         0.0f, 1.0f, 5.0f}, {3, 3}));

    // Degenerate extents.
    run("column vector 100x1", qr_accelerated, from_values(std::vector<float>(100, 2.5f), {100, 1}));
    run("row vector 1x100",    qr_accelerated, from_values(std::vector<float>(100, -1.5f), {1, 100}));
    run("1x1",                 qr_accelerated, from_values({3.0f}, {1, 1}));

    // Constant matrix: every column is identical, so all but the first
    // Householder reflection acts on a zero vector.
    run("constant 50x10", qr_accelerated, from_values(std::vector<float>(50 * 10, 0.5f), {50, 10}));

    // Batched non-square, to check batch striding on unaligned extents.
    run("batch 3 x 40x20", qr_accelerated, random_matrix(3, 40, 20, 30));

    // Lazy transposed views. An unevaluated MLX array reports itself as
    // row-contiguous, so a backend that checks flags before evaluating would
    // factorise the un-transposed buffer instead. Every backend is covered.
    run("transposed view 40x20 -> unblocked", detail::qr_unblocked,
        transpose(random_matrix(1, 20, 40, 31)));
    run("transposed view 600x64 -> reduced", detail::qr_streaming_amx_reduced,
        transpose(random_matrix(1, 64, 600, 32)));
    run("transposed view 8x8 -> complete", detail::qr_streaming_amx_complete,
        transpose(random_matrix(1, 8, 8, 33)));
    run("transposed view 40x20 -> cpu", detail::qr_cpu,
        transpose(random_matrix(1, 20, 40, 36)));
    run("transposed batch 3 x 40x20", qr_accelerated,
        transpose(random_matrix(3, 20, 40, 34), {0, 2, 1}));

    // One slice of a batch: contiguous, but not page-aligned.
    run("batch slice (unaligned) 30x30", qr_accelerated,
        reshape(slice(random_matrix(4, 30, 30, 35), {1, 0, 0}, {2, 30, 30}), {30, 30}));

    // -------------------------------------------------------------------------
    // Magnitude. The kernels compare squared norms with an absolute threshold,
    // so every backend used to lose accuracy from entries around 1e-3, fail
    // below 1e-5 and return NaN above 1e+18. Inputs are now scaled by a power
    // of two per matrix. The check is made on the unit-scale problem, since
    // ||A||_F itself overflows float32 at the extremes.
    // -------------------------------------------------------------------------
    std::printf("\n[ magnitude ]\n");
    {
        auto scaled = [&](const std::string& label, QrFn qr, int b, int M, int N, float scale, unsigned seed) {
            ++g_checks;
            array A1 = random_matrix(b, M, N, seed);
            array A  = multiply(A1, array(scale));
            auto [Q, R] = qr(A);
            eval({Q, R});
            array Rn = multiply(R, array(1.0f / scale));
            if (has_non_finite(Q) || has_non_finite(Rn)) { fail(label, "Q or R contains NaN/Inf"); return; }
            const int K = std::min(M, N);
            const float recon = frobenius(subtract(matmul(Q, Rn), A1)) / frobenius(A1);
            const float ortho = frobenius(subtract(matmul(transpose_last_two(Q), Q), eye(K))) /
                                std::sqrt((float)K * b);
            const bool ok = recon <= kReconTol && ortho <= kOrthoTol;
            std::printf("  %s  %-46s recon=%.2e ortho=%.2e\n", ok ? "ok  " : "FAIL", label.c_str(), recon, ortho);
            if (!ok) ++g_failures;
        };
        for (float sc : {1e-3f, 1e-5f, 1e-20f, 1e-30f, 1e+18f, 1e+30f, 1e+37f}) {
            char tag[32];
            std::snprintf(tag, sizeof tag, "x %.0e", sc);
            scaled(std::string("unblocked 64x64 ") + tag,  detail::qr_unblocked, 1, 64, 64, sc, 50);
            scaled(std::string("reduced 512x64 ") + tag,   detail::qr_streaming_amx_reduced, 2, 512, 64, sc, 51);
            scaled(std::string("blocked 512x200 ") + tag,  detail::qr_blocked, 1, 512, 200, sc, 56);
            scaled(std::string("householder 48x40 ") + tag, detail::qr_householder, 3, 48, 40, sc, 57);
            scaled(std::string("householder 200x30 ") + tag, detail::qr_householder, 2, 200, 30, sc, 58);
            scaled(std::string("householder 20x16 ") + tag, detail::qr_householder, 3, 20, 16, sc, 59);
            scaled(std::string("householder 150x100 ") + tag, detail::qr_householder, 2, 150, 100, sc, 60);
            scaled(std::string("householder 128x64 ") + tag, detail::qr_householder, 2, 128, 64, sc, 61);
            scaled(std::string("complete 8x8 ") + tag,     detail::qr_streaming_amx_complete, 1, 8, 8, sc, 52);
            scaled(std::string("cpu 64x64 ") + tag,        detail::qr_cpu, 1, 64, 64, sc, 55);
        }
        scaled("batch 6 x 40x20 x 1e-6 (dispatcher)", qr_accelerated, 6, 40, 20, 1e-6f, 53);

        // Matrices of different magnitude in one batch get their own factor.
        ++g_checks;
        {
            array A1 = random_matrix(3, 24, 24, 54);
            array f  = array({1e-12f, 1.0f, 1e+12f}, {3, 1, 1});
            auto [Q, R] = qr_accelerated(multiply(A1, f));
            eval({Q, R});
            array Rn = divide(R, f);
            const float recon = frobenius(subtract(matmul(Q, Rn), A1)) / frobenius(A1);
            if (recon > kReconTol) fail("mixed magnitudes in one batch", "recon " + std::to_string(recon));
            else std::printf("  ok    %-46s recon=%.2e\n", "mixed magnitudes 1e-12, 1, 1e+12 in one batch", recon);
        }
    }

    // -------------------------------------------------------------------------
    // Nearly dependent columns. The reflection threshold is compared with a
    // squared norm; at 1e-7 it treated every column tail shorter than 3e-4 as
    // zero and dropped it. The bound here is tighter than the general one,
    // because that loss sat just under it.
    // -------------------------------------------------------------------------
    std::printf("\n[ nearly dependent columns ]\n");
    {
        constexpr float kTight = 2e-6f;
        auto tight = [&](const std::string& label, QrFn qr, const array& A) {
            ++g_checks;
            auto [Q, R] = qr(A);
            eval({Q, R});
            const float recon = frobenius(subtract(matmul(Q, R), A)) / frobenius(A);
            const bool ok = !has_non_finite(Q) && !has_non_finite(R) && recon <= kTight;
            std::printf("  %s  %-46s recon=%.2e (bound %.0e)\n", ok ? "ok  " : "FAIL", label.c_str(), recon, kTight);
            if (!ok) ++g_failures;
        };
        auto low_rank = [&](int M, int N, int rank, float noise, unsigned seed) {
            array A = matmul(random_matrix(1, M, rank, seed), random_matrix(1, rank, N, seed + 1));
            if (noise > 0.0f) A = add(A, multiply(random_matrix(1, M, N, seed + 2), array(noise)));
            eval({A});
            return A;
        };
        for (int k = 0; k < 6; ++k) {
            tight("rank 5 of 600x16, instance " + std::to_string(k) + " (reduced)",
                  detail::qr_streaming_amx_reduced, low_rank(600, 16, 5, 0.0f, 60 + 3 * k));
        }
        tight("rank 5 of 64x16 (unblocked)",             detail::qr_unblocked, low_rank(64, 16, 5, 0.0f, 80));
        tight("rank 3 of 8x8 (complete)",                detail::qr_streaming_amx_complete, low_rank(8, 8, 3, 0.0f, 83));
        tight("rank 5 + 1e-5 noise, 600x16 (reduced)",   detail::qr_streaming_amx_reduced, low_rank(600, 16, 5, 1e-5f, 86));
        tight("rank 5 + 1e-4 noise, 64x16 (unblocked)",  detail::qr_unblocked, low_rank(64, 16, 5, 1e-4f, 89));
        tight("rank 5 of 600x16 (cpu)",                  detail::qr_cpu, low_rank(600, 16, 5, 0.0f, 92));
        for (int k = 0; k < 3; ++k)
            tight("rank 20 of 600x300, instance " + std::to_string(k) + " (blocked)", detail::qr_blocked,
                  low_rank(600, 300, 20, 0.0f, 95 + 3 * k));
        tight("rank 20 + 1e-5 noise, 600x300 (blocked)", detail::qr_blocked, low_rank(600, 300, 20, 1e-5f, 104));
        for (int k = 0; k < 3; ++k)
            tight("rank 5 of 64x40, instance " + std::to_string(k) + " (householder)", detail::qr_householder,
                  low_rank(64, 40, 5, 0.0f, 110 + 3 * k));
        tight("rank 5 + 1e-5 noise, 120x16 (householder)", detail::qr_householder, low_rank(120, 16, 5, 1e-5f, 120));
        tight("rank 3 of 200x24 (householder)", detail::qr_householder, low_rank(200, 24, 3, 0.0f, 123));
        for (int k = 0; k < 3; ++k)
            tight("rank 20 of 300x200, instance " + std::to_string(k) + " (householder)", detail::qr_householder,
                  low_rank(300, 200, 20, 0.0f, 126 + 3 * k));
        tight("rank 20 + 1e-5 noise, 300x200 (householder)", detail::qr_householder, low_rank(300, 200, 20, 1e-5f, 135));
        tight("rank 7 of 1000x64 (householder)", detail::qr_householder, low_rank(1000, 64, 7, 0.0f, 138));
    }

    // -------------------------------------------------------------------------
    // Dispatch policy. The crossover is hardware-tuned, so the tests must not
    // assume the value measured on any one GPU: they force it in both
    // directions and check that each backend is still correct where it lands.
    // -------------------------------------------------------------------------
    std::printf("\n[ routing policy ]\n");
    {
        const QrPolicy p = qr_policy();
        std::printf("  source=%s  m_crossover=%u (batch<%u) / %u (batch>=%u)"
                    "  GPU iff %u<=k<=%u, batch*k>=%u, batch>=%u"
                    "  gpu_cores=%u  concurrent_matrices=%u\n",
                    qr_policy_source(), p.m_crossover_small_batch,
                    p.batch_threshold, p.m_crossover_large_batch, p.batch_threshold,
                    p.gpu_min_k, p.gpu_max_k, p.gpu_min_batch_times_k, p.gpu_min_batch,
                    p.gpu_cores, p.concurrent_matrices);
        if (p.gpu_cores == 0)
            std::printf("  note: GPU core count undetected; crossover is the untuned default\n");

        const QrPolicy original = p;

        // Force every shape onto the GPU, then onto the grid-parallel kernel,
        // then onto the single-threadgroup one, so both are covered on any
        // hardware whatever its CPU routing.
        QrPolicy forced = original;
        forced.gpu_max_k = kQrNoLimit;
        forced.gpu_min_batch_times_k = 0;
        forced.gpu_min_batch = 1;
        forced.gpu_min_k = 0;
        forced.m_crossover_small_batch = forced.m_crossover_large_batch = 1;
        set_qr_policy(forced);
        if (qr_backend(64, 64, 1) != QrBackend::streaming_reduced) fail("qr_backend", "crossover 1 -> unblocked");
        else { std::printf("  ok    qr_backend: crossover 1 -> streaming_reduced at 64x64\n"); ++g_checks; }
        run("forced -> reduced   64x64",   qr_accelerated, random_matrix(1, 64, 64, 40));
        run("forced -> reduced   40x20 b3", qr_accelerated, random_matrix(3, 40, 20, 41));

        forced.m_crossover_small_batch = forced.m_crossover_large_batch = 1u << 30;
        set_qr_policy(forced);
        if (qr_backend(4096, 64, 1) != QrBackend::unblocked) fail("qr_backend", "crossover 2^30 -> reduced");
        else { std::printf("  ok    qr_backend: crossover 2^30 -> unblocked at 4096x64\n"); ++g_checks; }
        run("forced -> unblocked 512x512", qr_accelerated, random_matrix(1, 512, 512, 42));
        run("forced -> unblocked 600x128", qr_accelerated, random_matrix(1, 600, 128, 43));

        // GPU or CPU: never the GPU, then only from a batch * k threshold.
        forced = original;
        forced.gpu_max_k = 0;
        forced.gpu_large_min_k = 0;   // the large-matrix clause, tested below
        set_qr_policy(forced);
        if (qr_backend(4096, 512, 64) != QrBackend::cpu) fail("qr_backend", "gpu_max_k 0 -> GPU");
        else { std::printf("  ok    qr_backend: gpu_max_k 0 -> cpu at 64 x 4096x512\n"); ++g_checks; }
        if (qr_gpu_backend(4096, 512, 64) == QrBackend::cpu) fail("qr_gpu_backend", "returned cpu");
        else { std::printf("  ok    qr_gpu_backend ignores the CPU routing\n"); ++g_checks; }
        run("forced -> cpu       64x64",     qr_accelerated, random_matrix(1, 64, 64, 44));
        run("forced -> cpu       40x20 b3",  qr_accelerated, random_matrix(3, 40, 20, 45));

        forced.gpu_max_k = kQrNoLimit;
        forced.gpu_min_batch_times_k = 1000;
        forced.gpu_min_batch = 1;
        forced.gpu_min_k = 0;
        set_qr_policy(forced);
        const bool lone_cpu = qr_backend(64, 64, 1) == QrBackend::cpu;            // 1 * 64  < 1000
        const bool batch_gpu = qr_backend(64, 64, 16) != QrBackend::cpu;          // 16 * 64 >= 1000
        if (!lone_cpu || !batch_gpu) fail("qr_backend", "batch * k threshold not applied");
        else { std::printf("  ok    qr_backend: batch*k >= 1000 -> GPU at 16 x 64x64, CPU at 1 x 64x64\n"); ++g_checks; }
        forced.gpu_min_batch = 32;
        set_qr_policy(forced);
        if (qr_backend(64, 64, 16) != QrBackend::cpu) fail("qr_backend", "gpu_min_batch not applied");
        else { std::printf("  ok    qr_backend: gpu_min_batch 32 -> CPU at 16 x 64x64\n"); ++g_checks; }
        forced.gpu_min_batch = 1;
        forced.gpu_min_k = 128;
        set_qr_policy(forced);
        {
            // The smallest matrices stay on the CPU at any batch; their size
            // is w = floor(sqrt(M k)), so a tall one counts by its rows too.
            const bool ok = qr_backend(16, 16, 100000) == QrBackend::cpu && qr_backend(128, 16, 1000) == QrBackend::cpu &&
                            qr_backend(4096, 64, 1000) != QrBackend::cpu && qr_backend(128, 128, 1024) != QrBackend::cpu &&
                            qr_backend(512, 128, 16) != QrBackend::cpu && qr_backend(127, 127, 1024) == QrBackend::cpu;
            ++g_checks;
            if (!ok) fail("qr_backend", "gpu_min_k not applied");
            else std::printf("  ok    qr_backend: gpu_min_k 128 on sqrt(M k) -> CPU at 100000 x 16x16, 1000 x 128x16 "
                             "and 1024 x 127^2, GPU at 1000 x 4096x64 and 1024 x 128x128\n");
        }
        run("gpu_min_k -> cpu    16x16 b64", qr_accelerated, random_matrix(64, 16, 16, 47));

        // Sharing a GPU batch with the CPU from share_min_batch; never a CPU batch.
        forced.share_min_batch = 512;
        set_qr_policy(forced);
        {
            const bool ok = qr_shares_batch(128, 128, 512) && !qr_shares_batch(128, 128, 511) &&
                            !qr_shares_batch(16, 16, 100000);
            ++g_checks;
            if (!ok) fail("qr_shares_batch", "share_min_batch not applied");
            else std::printf("  ok    qr_shares_batch: from 512 at 128x128, never on the CPU (16x16)\n");
        }
        run("routed, shared       600 x 128x128", qr_accelerated, random_matrix(600, 128, 128, 48));
        forced.share_min_batch = 0;

        // Large matrices: the GPU from gpu_large_min_k in a batch up to the
        // cap, whatever the product rule says.
        forced = original;
        forced.gpu_max_k = 8;
        forced.gpu_min_k = 0;
        forced.gpu_min_batch_times_k = 1u << 30;
        forced.gpu_large_min_k = 1024;
        forced.gpu_large_max_batch = 4;
        set_qr_policy(forced);
        {
            const bool ok = qr_backend(2048, 2048, 1) != QrBackend::cpu && qr_backend(1024, 4096, 4) != QrBackend::cpu &&
                            qr_backend(2048, 2048, 5) == QrBackend::cpu && qr_backend(1023, 1023, 1) == QrBackend::cpu &&
                            qr_backend(8192, 512, 1) != QrBackend::cpu && qr_backend(2048, 256, 1) == QrBackend::cpu;
            ++g_checks;
            if (!ok) fail("qr_backend", "large-matrix clause not applied");
            else std::printf("  ok    qr_backend: large clause sqrt(M k) >= 1024, batch<=4 -> GPU at 1 x 2048^2, 4 x 1024x4096 "
                             "and 8192x512 (2048), CPU at 5 x 2048^2, 1023^2 and 2048x256 (724)\n");
            forced.gpu_large_max_batch = 0;
            set_qr_policy(forced);
            ++g_checks;
            if (qr_backend(2048, 2048, 64) == QrBackend::cpu) fail("qr_backend", "large clause, cap 0 -> any batch");
            else std::printf("  ok    qr_backend: large clause with no cap -> GPU at 64 x 2048^2\n");
        }
        run("large clause -> GPU 1100x1100", qr_accelerated, random_matrix(1, 1100, 1100, 46));

        set_qr_policy(original);
        if (qr_policy().m_crossover_small_batch != original.m_crossover_small_batch) {
            fail("routing policy", "restoring the original policy did not take effect");
        } else {
            std::printf("  ok    policy override and restore\n");
            ++g_checks;
        }
    }

    // -------------------------------------------------------------------------
    std::printf("\n%d checks, %d failures\n\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
