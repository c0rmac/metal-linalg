// Correctness tests for the Metal QR backends.
//
// Every case is checked four ways: output shapes, reconstruction (Q*R == A),
// orthogonality (Q^T*Q == I) and upper-triangularity of R. The backends are
// also exercised directly, not just through the dispatcher, so a backend that
// is only reachable for a narrow shape range still gets covered.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
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
            scaled(std::string("reduced 512x64 ") + tag,   detail::qr_streaming_amx_reduced, 1, 512, 64, sc, 51);
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
                    "  GPU iff k<=%u, batch*k>=%u, batch>=%u"
                    "  gpu_cores=%u  concurrent_matrices=%u\n",
                    qr_policy_source(), p.m_crossover_small_batch,
                    p.batch_threshold, p.m_crossover_large_batch, p.batch_threshold,
                    p.gpu_max_k, p.gpu_min_batch_times_k, p.gpu_min_batch,
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
        set_qr_policy(forced);
        const bool lone_cpu = qr_backend(64, 64, 1) == QrBackend::cpu;            // 1 * 64  < 1000
        const bool batch_gpu = qr_backend(64, 64, 16) != QrBackend::cpu;          // 16 * 64 >= 1000
        if (!lone_cpu || !batch_gpu) fail("qr_backend", "batch * k threshold not applied");
        else { std::printf("  ok    qr_backend: batch*k >= 1000 -> GPU at 16 x 64x64, CPU at 1 x 64x64\n"); ++g_checks; }
        forced.gpu_min_batch = 32;
        set_qr_policy(forced);
        if (qr_backend(64, 64, 16) != QrBackend::cpu) fail("qr_backend", "gpu_min_batch not applied");
        else { std::printf("  ok    qr_backend: gpu_min_batch 32 -> CPU at 16 x 64x64\n"); ++g_checks; }

        // Large matrices: the GPU from gpu_large_min_k in a batch up to the
        // cap, whatever the product rule says.
        forced = original;
        forced.gpu_max_k = 8;
        forced.gpu_min_batch_times_k = 1u << 30;
        forced.gpu_large_min_k = 1024;
        forced.gpu_large_max_batch = 4;
        set_qr_policy(forced);
        {
            const bool ok = qr_backend(2048, 2048, 1) != QrBackend::cpu && qr_backend(1024, 4096, 4) != QrBackend::cpu &&
                            qr_backend(2048, 2048, 5) == QrBackend::cpu && qr_backend(1023, 1023, 1) == QrBackend::cpu;
            ++g_checks;
            if (!ok) fail("qr_backend", "large-matrix clause not applied");
            else std::printf("  ok    qr_backend: large clause k>=1024, batch<=4 -> GPU at 1 x 2048^2 and 4 x 1024x4096, "
                             "CPU at 5 x 2048^2 and 1 x 1023^2\n");
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
