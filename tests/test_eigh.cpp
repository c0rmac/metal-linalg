// Correctness tests for the Metal symmetric eigensolver.
//
// Every decomposition is checked five ways: output shapes, finiteness, the
// eigen-residual ||A V - V diag(w)||_F / ||A||_F, orthogonality of V, and
// ascending order of w. Where MLX's CPU eigh (LAPACK) is available the
// eigenvalues are also compared against it directly, which is an independent
// reference rather than a self-consistency check.
//
// Both execution modes (simd and threadgroup) are forced explicitly across
// the size range so each is covered regardless of the crossover in eigh.mm.

#include <algorithm>
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

// Relative Frobenius bounds. Jacobi is backward stable with a modest
// constant; these hold with a wide margin across N = 1..600 in float32.
constexpr float kResidTol = 2e-5f;
constexpr float kOrthoTol = 2e-5f;
constexpr float kEigTol   = 2e-5f;   // |w - w_lapack| relative to ||A||_F

int g_failures = 0;
int g_checks   = 0;

std::mt19937& rng() { static std::mt19937 r(1234); return r; }

array from_values(std::vector<float> v, Shape shape) {
    return array(v.begin(), std::move(shape), float32);
}

// Symmetric Gaussian matrix, A = (G + G^T) / 2.
array random_symmetric(int batch, int n, unsigned seed) {
    std::mt19937 r(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> data((size_t)batch * n * n);
    for (int b = 0; b < batch; ++b) {
        float* m = data.data() + (size_t)b * n * n;
        for (int i = 0; i < n; ++i)
            for (int j = 0; j <= i; ++j) {
                const float x = dist(r);
                m[i * n + j] = x;
                m[j * n + i] = x;
            }
    }
    if (batch == 1) return from_values(data, {n, n});
    return from_values(data, {batch, n, n});
}

// Q diag(spec) Q^T for a random orthogonal Q, so the spectrum is known.
array with_spectrum(const std::vector<float>& spec) {
    const int n = (int)spec.size();
    array G = random::normal({n, n}, float32, 0.0f, 1.0f, std::nullopt, Device::cpu);
    auto [Q, R] = linalg::qr(G, Device::cpu);
    array A = matmul(matmul(Q, diag(from_values(spec, {n}), 0, Device::cpu), Device::cpu),
                     transpose(Q, Device::cpu), Device::cpu);
    A = multiply(add(A, transpose(A, Device::cpu), Device::cpu), array(0.5f), Device::cpu);
    eval({A});
    return A;
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

// Symmetrises `A` from one triangle, as the solver sees it.
array symmetrised(const array& A, bool lower) {
    array t = lower ? tril(A, 0) : triu(A, 0);
    array strict = lower ? tril(A, -1) : triu(A, 1);
    return add(t, transpose_last_two(strict));
}

// The full battery on one result. `lower` says which triangle the solver was
// asked to read, so the reference is built the same way.
void check(const std::string& label, const array& A_in, const EighResult& r_in,
           bool lower = true, bool vectors = true) {
    ++g_checks;
    array A = symmetrised(A_in, lower);
    const auto& shape = A.shape();
    const int n = shape[shape.size() - 1];

    // The checks below square entries, so normalise by max|A| first: the
    // solver handles 1e+37 fine but ||A||_F in float32 does not.
    EighResult r = r_in;
    {
        const float amax = max_abs(A);
        if (amax > 0.0f) {
            A = multiply(A, array(1.0f / amax));
            r.eigenvalues = multiply(r.eigenvalues, array(1.0f / amax));
        }
    }

    Shape want_w(shape.begin(), shape.end() - 1);
    if (r.eigenvalues.shape() != want_w) { fail(label, "eigenvalues have the wrong shape"); return; }
    if (vectors && r.eigenvectors.shape() != shape) { fail(label, "eigenvectors have the wrong shape"); return; }

    if (has_non_finite(r.eigenvalues) || (vectors && has_non_finite(r.eigenvectors))) {
        fail(label, "output contains NaN/Inf");
        return;
    }

    // info: every matrix converged
    array info = reshape(r.info, {-1});
    eval({info});
    unsigned max_sweeps = 0;
    for (int b = 0; b < (int)info.size(); ++b) {
        const unsigned w = info.data<uint32_t>()[b];
        if (!detail::eigh_converged(w)) { fail(label, "info says not converged"); return; }
        max_sweeps = std::max(max_sweeps, detail::eigh_sweeps(w));
    }

    // Ascending order: max over i of w[i] - w[i+1] must be <= 0.
    float order = 0.0f;
    if (n > 1) {
        Shape start(shape.size() - 1, 0);
        Shape stop(shape.begin(), shape.end() - 1);
        Shape stop_lo = stop;   stop_lo.back() = n - 1;
        Shape start_hi = start; start_hi.back() = 1;
        array w_lo = slice(r.eigenvalues, start, stop_lo);
        array w_hi = slice(r.eigenvalues, start_hi, stop);
        array m = max(subtract(w_lo, w_hi));
        eval({m});
        order = m.item<float>();
    }

    const float norm_A = std::max(frobenius(A), 1e-30f);
    const float scale  = std::max(norm_A, 1e-30f);

    float resid = 0.0f, ortho = 0.0f;
    if (vectors) {
        // A V - V diag(w): scale each column of V by w (broadcast over the last axis).
        array VL   = multiply(r.eigenvectors, expand_dims(r.eigenvalues, -2));
        resid = frobenius(subtract(matmul(A, r.eigenvectors), VL)) / scale;
        array VtV  = matmul(transpose_last_two(r.eigenvectors), r.eigenvectors);
        ortho = frobenius(subtract(VtV, eye(n))) / std::sqrt((float)n * (float)std::max<int>(1, (int)info.size()));
    }

    // Independent reference: LAPACK through MLX's CPU eigh.
    float eig_err = 0.0f;
    {
        array w_ref = linalg::eigvalsh(A, lower ? "L" : "U", Device::cpu);
        eval({w_ref});
        eig_err = max_abs(subtract(r.eigenvalues, w_ref)) / scale;
    }

    const bool ok = order <= 0.0f && resid <= kResidTol && ortho <= kOrthoTol && eig_err <= kEigTol;
    std::printf("  %s  %-44s resid=%.1e ortho=%.1e |w-lapack|=%.1e sweeps=%u\n",
                ok ? "ok  " : "FAIL", label.c_str(), resid, ortho, eig_err, max_sweeps);
    if (order > 0.0f) std::printf("        eigenvalues not ascending (max drop %.3e)\n", order);
    if (!ok) ++g_failures;
}

EighOptions mode_opts(EighOptions::Mode m) { EighOptions o; o.mode = m; return o; }

using EighFn = EighResult (*)(const array&, bool, bool, const EighOptions&);

void run(const std::string& label, const array& A, EighOptions opt = {}, bool lower = true,
         EighFn fn = detail::eigh_jacobi) {
    EighResult r = fn(A, true, lower, opt);
    eval({r.eigenvalues, r.eigenvectors, r.info});
    check(label, A, r, lower, true);
}

// The block backend, with a given number of inner sweeps (0 = default).
void run_block(const std::string& label, const array& A, unsigned inner = 0, bool lower = true) {
    EighOptions o = mode_opts(EighOptions::Mode::block);
    o.inner_sweeps = inner;
    run(label, A, o, lower, detail::eigh_block_jacobi);
}

} // namespace

int main() {
    set_default_device(Device::gpu);

    // The public functions route small problems to MLX's CPU eigh; the GPU
    // kernel is what is under test here, so force it (the routing itself is
    // checked at the end).
    setenv("EIGH_DEVICE", "gpu", 1);

    std::printf("\neigh correctness tests\n");
    std::printf("tolerances: resid<=%.0e ortho<=%.0e |w-lapack|<=%.0e   (relative to ||A||_F)\n",
                kResidTol, kOrthoTol, kEigTol);
    {
        const EighPolicy p = eigh_policy();
        std::printf("device: %s (%u GPU cores)   policy source: %s\n",
                    device_name(), p.gpu_cores, eigh_policy_source());
        std::printf("policy: simd N<=%u, block N>=%u, batched block N>=%u at batch>=%u, "
                    "GPU iff N<=%u and batch*N>=%u\n",
                    p.simd_max_n, p.block_min_n, p.block_min_n_batched, p.block_min_batch,
                    p.gpu_max_n, p.gpu_min_batch_times_n);
    }

    // -------------------------------------------------------------------------
    // Each execution mode across the size range, including odd N (dummy
    // partner in the tournament) and N around the simdgroup width.
    // -------------------------------------------------------------------------
    std::printf("\n[ mode: simd ]\n");
    for (int n : {1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 48, 64})
        run("N=" + std::to_string(n), random_symmetric(1, n, 100 + n), mode_opts(EighOptions::Mode::simd));
    run("batch 37 x N=3",  random_symmetric(37, 3, 200), mode_opts(EighOptions::Mode::simd));
    run("batch 1000 x N=6", random_symmetric(1000, 6, 201), mode_opts(EighOptions::Mode::simd));
    run("batch 9 x N=33",  random_symmetric(9, 33, 202), mode_opts(EighOptions::Mode::simd));

    std::printf("\n[ mode: threadgroup ]\n");
    for (int n : {1, 2, 3, 4, 5, 8, 9, 16, 17, 32, 33, 63, 64, 65, 100, 127, 128, 129, 200, 256, 257, 384, 512})
        run("N=" + std::to_string(n), random_symmetric(1, n, 300 + n), mode_opts(EighOptions::Mode::threadgroup));
    run("batch 5 x N=3",    random_symmetric(5, 3, 400),   mode_opts(EighOptions::Mode::threadgroup));
    run("batch 16 x N=64",  random_symmetric(16, 64, 401), mode_opts(EighOptions::Mode::threadgroup));
    run("batch 4 x N=200",  random_symmetric(4, 200, 402), mode_opts(EighOptions::Mode::threadgroup));
    {
        EighOptions o = mode_opts(EighOptions::Mode::threadgroup);
        o.threads = 256;
        run("N=300, 256 threads", random_symmetric(1, 300, 403), o);
        o.threads = 64;
        run("N=100, 64 threads",  random_symmetric(1, 100, 404), o);
    }

    // -------------------------------------------------------------------------
    // Block Jacobi backend: sizes around the 16-block and 32-group boundaries,
    // padding (N not a multiple of 32), batches, inner-sweep counts, and the
    // structured cases the scalar backend gets below.
    // -------------------------------------------------------------------------
    std::printf("\n[ backend: block jacobi ]\n");
    for (int n : {1, 5, 17, 31, 32, 33, 48, 64, 96, 100, 128, 129, 200, 256, 257, 384, 512})
        run_block("N=" + std::to_string(n), random_symmetric(1, n, 900 + n));
    run_block("batch 3 x N=100",   random_symmetric(3, 100, 950));
    run_block("batch 5 x N=64",    random_symmetric(5, 64, 951));
    run_block("N=128, 2 inner sweeps", random_symmetric(1, 128, 952), 2);
    run_block("N=128, 3 inner sweeps", random_symmetric(1, 128, 953), 3);
    {
        std::vector<float> data(2 * 2 * 40 * 40);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        for (auto& v : data) v = dist(rng());
        array A = from_values(data, {2, 2, 40, 40});
        EighOptions o = mode_opts(EighOptions::Mode::block);
        EighResult r = detail::eigh_block_jacobi(A, true, true, o);
        eval({r.eigenvalues, r.eigenvectors, r.info});
        check("batch [2,2] x 40x40 (nonsymmetric, L)", A, r, true);
        EighResult ru = detail::eigh_block_jacobi(A, true, false, o);
        eval({ru.eigenvalues, ru.eigenvectors, ru.info});
        check("batch [2,2] x 40x40 (nonsymmetric, U)", A, ru, false);

        // Eigenvalues only must match.
        EighResult rv = detail::eigh_block_jacobi(A, false, true, o);
        eval({rv.eigenvalues});
        const float d = max_abs(subtract(r.eigenvalues, rv.eigenvalues)) / std::max(frobenius(A), 1.0f);
        ++g_checks;
        if (d > 1e-6f) fail("block: values-only == with vectors", "differ by " + std::to_string(d));
        else std::printf("  ok    %-44s |dw|=%.1e\n", "block: values-only == with vectors", d);
    }
    run_block("identity 64x64",   eye(64));
    run_block("zeros 48x48",      zeros({48, 48}));
    run_block("diagonal 128x128 (already converged)", diag(arange(128, float32)));
    {
        std::vector<float> spec(128);
        for (int i = 0; i < 128; ++i) spec[i] = (float)(i % 4);
        run_block("repeated eigenvalues 128x128", with_spectrum(spec));
    }
    {
        std::vector<float> spec(96);
        for (int i = 0; i < 96; ++i) spec[i] = std::pow(10.0f, -4.0f + 8.0f * i / 95.0f);
        run_block("spectrum 1e-4 .. 1e+4 (96x96)", with_spectrum(spec));
    }
    {
        std::vector<float> spec(72);
        for (int i = 0; i < 72; ++i) spec[i] = -(float)i - 1.0f;
        run_block("negative definite 72x72", with_spectrum(spec));
    }
    {
        array x = random::normal({100, 1}, float32, 0.0f, 1.0f, std::nullopt, Device::cpu);
        array A = matmul(x, transpose(x, Device::cpu), Device::cpu);
        eval({A});
        run_block("rank one 100x100", A);
    }
    {
        array A = random_symmetric(1, 64, 960);
        run_block("scaled 1e-30 (64x64)", multiply(A, array(1e-30f)));
        run_block("scaled 1e+37 (64x64)", multiply(A, array(1e37f)));
    }
    {
        array A = random_symmetric(1, 100, 961);
        array junk = multiply(triu(random_symmetric(1, 100, 962), 1), array(1e6f));
        run_block("lower triangle only (junk above)", add(A, junk), 0, true);
        array junk_l = multiply(tril(random_symmetric(1, 100, 963), -1), array(1e6f));
        run_block("upper triangle only (junk below)", add(A, junk_l), 0, false);
    }
    {
        array G = random_symmetric(1, 100, 964);
        run_block("transposed view 100x100", transpose(add(G, multiply(triu(G, 1), array(3.0f)))));
    }
    {
        std::vector<float> data(64 * 64, 1.0f);
        data[3 * 64 + 5] = data[5 * 64 + 3] = NAN;
        array A = from_values(data, {64, 64});
        ++g_checks;
        try {
            EighResult r = detail::eigh_block_jacobi(A, true, true, mode_opts(EighOptions::Mode::block));
            eval({r.eigenvalues, r.eigenvectors, r.info});
            const unsigned info = r.info.item<uint32_t>();
            const bool all_nan = !any(logical_not(isnan(r.eigenvalues))).item<bool>();
            if (!detail::eigh_nonfinite(info) || !all_nan)
                fail("block: NaN input 64x64", "expected non-finite flag and all-NaN eigenvalues");
            else std::printf("  ok    %-44s info=0x%x\n", "block: NaN input 64x64 -> NaN, flagged", info);
        } catch (const std::exception& e) {
            fail("block: NaN input 64x64", std::string("threw: ") + e.what());
        }
    }
    {
        // A batch mixing a NaN matrix with good ones: the good ones must be
        // unaffected. The NaN goes in the lower triangle, the one that is read.
        array good = random_symmetric(3, 48, 965);
        std::vector<float> data(3 * 48 * 48);
        eval({good});
        std::copy(good.data<float>(), good.data<float>() + data.size(), data.begin());
        data[1 * 48 * 48 + 7 * 48 + 2] = NAN;
        array A = from_values(data, {3, 48, 48});
        EighResult r = detail::eigh_block_jacobi(A, true, true, mode_opts(EighOptions::Mode::block));
        eval({r.eigenvalues, r.eigenvectors, r.info});
        ++g_checks;
        const uint32_t* info = r.info.data<uint32_t>();
        const bool flags_ok = detail::eigh_converged(info[0]) && detail::eigh_nonfinite(info[1]) &&
                              detail::eigh_converged(info[2]);
        array w0 = reshape(slice(r.eigenvalues, {0, 0}, {1, 48}), {48});
        array V0 = reshape(slice(r.eigenvectors, {0, 0, 0}, {1, 48, 48}), {48, 48});
        array A0 = reshape(slice(A, {0, 0, 0}, {1, 48, 48}), {48, 48});
        if (!flags_ok) {
            char buf[96];
            std::snprintf(buf, sizeof buf, "info = 0x%x 0x%x 0x%x", info[0], info[1], info[2]);
            fail("block: mixed NaN batch", buf);
        }
        else {
            std::printf("  ok    block: mixed NaN batch flags (conv, nonfinite, conv)\n");
            check("block: mixed NaN batch, matrix 0 intact", A0,
                  EighResult{w0, V0, array((uint32_t)(1u | (1u << 16)))});
        }
    }

    // -------------------------------------------------------------------------
    // Public API and batch handling.
    // -------------------------------------------------------------------------
    std::printf("\n[ public API ]\n");
    {
        array A = random_symmetric(1, 40, 500);
        auto [w, V] = eigh_accelerated(A);
        eval({w, V});
        EighResult r{w, V, zeros({}, uint32)};
        r.info = array((uint32_t)(1u | (1u << 16)));
        check("eigh_accelerated 40x40", A, r);

        array w2 = eigvalsh_accelerated(A);
        eval({w2});
        const float d = max_abs(subtract(w, w2)) / std::max(frobenius(A), 1.0f);
        ++g_checks;
        if (d > 1e-6f) fail("eigvalsh == eigh eigenvalues", "differ by " + std::to_string(d));
        else std::printf("  ok    %-44s |dw|=%.1e\n", "eigvalsh == eigh eigenvalues", d);
    }
    {
        // 4-D batch shape, N odd, through the public API.
        std::vector<float> data(2 * 3 * 7 * 7);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        for (auto& v : data) v = dist(rng());
        array A = from_values(data, {2, 3, 7, 7});
        auto [w, V] = eigh_accelerated(A);
        eval({w, V});
        EighResult r{w, V, full({2, 3}, (uint32_t)(1u | (1u << 16)))};
        check("batch [2,3] x 7x7 (nonsymmetric input, L)", A, r, true);
        auto [wu, Vu] = eigh_accelerated(A, "U");
        eval({wu, Vu});
        EighResult ru{wu, Vu, full({2, 3}, (uint32_t)(1u | (1u << 16)))};
        check("batch [2,3] x 7x7 (nonsymmetric input, U)", A, ru, false);
    }
    {
        // Non-contiguous input: a transposed view.
        array G = random_symmetric(1, 50, 501);
        array A = transpose(add(G, multiply(triu(G, 1), array(3.0f))));  // asymmetric, transposed view
        auto [w, V] = eigh_accelerated(A);
        eval({w, V});
        EighResult r{w, V, array((uint32_t)(1u | (1u << 16)))};
        check("transposed (non-contiguous) view 50x50", A, r);
    }
    {
        // One slice of a batch: contiguous but not page-aligned.
        array B = random_symmetric(4, 30, 503);
        array A = reshape(slice(B, {2, 0, 0}, {3, 30, 30}), {30, 30});
        auto [w, V] = eigh_accelerated(A);
        eval({w, V});
        EighResult r{w, V, array((uint32_t)(1u | (1u << 16)))};
        check("batch slice (unaligned) 30x30", A, r);
    }
    {
        // Integer input is cast.
        array A = astype(multiply(random_symmetric(1, 12, 502), array(10.0f)), int32);
        auto [w, V] = eigh_accelerated(A);
        eval({w, V});
        EighResult r{w, V, array((uint32_t)(1u | (1u << 16)))};
        check("int32 input 12x12", astype(A, float32), r);
    }

    // -------------------------------------------------------------------------
    // Structured spectra, where the answer is known.
    // -------------------------------------------------------------------------
    std::printf("\n[ structured ]\n");
    run("identity 32x32",   eye(32));
    run("zeros 16x16",      zeros({16, 16}));
    run("diagonal 64x64 (already converged)", diag(arange(64, float32)));
    {
        std::vector<float> spec(64);
        for (int i = 0; i < 64; ++i) spec[i] = (float)(i % 4);   // four 16-fold eigenvalues
        run("repeated eigenvalues 64x64", with_spectrum(spec));
    }
    {
        std::vector<float> spec(48);
        for (int i = 0; i < 48; ++i) spec[i] = std::pow(10.0f, -4.0f + 8.0f * i / 47.0f);
        run("spectrum 1e-4 .. 1e+4 (48x48)", with_spectrum(spec));
    }
    {
        std::vector<float> spec(40);
        for (int i = 0; i < 40; ++i) spec[i] = -(float)i - 1.0f;
        run("negative definite 40x40", with_spectrum(spec));
    }
    {
        // Rank one: x x^T.
        array x = random::normal({30, 1}, float32, 0.0f, 1.0f, std::nullopt, Device::cpu);
        array A = matmul(x, transpose(x, Device::cpu), Device::cpu);
        eval({A});
        run("rank one 30x30", A);
    }
    {
        // 2x2 with a closed form.
        array A = from_values({2.0f, 1.0f, 1.0f, 2.0f}, {2, 2});
        EighResult r = detail::eigh_jacobi(A, true, true, {});
        eval({r.eigenvalues});
        ++g_checks;
        const float* w = r.eigenvalues.data<float>();
        if (std::fabs(w[0] - 1.0f) > 1e-6f || std::fabs(w[1] - 3.0f) > 1e-6f)
            fail("2x2 [[2,1],[1,2]] -> {1, 3}", "got " + std::to_string(w[0]) + ", " + std::to_string(w[1]));
        else std::printf("  ok    %-44s w={%.6f, %.6f}\n", "2x2 [[2,1],[1,2]] -> {1, 3}", w[0], w[1]);
    }
    {
        // Scale invariance. Squared entries overflow float32 above ~1e19 and
        // underflow below ~1e-19; the solver rescales so neither can happen.
        array A = random_symmetric(1, 24, 603);
        run("scaled 1e-18 (24x24)", multiply(A, array(1e-18f)));
        run("scaled 1e-30 (24x24)", multiply(A, array(1e-30f)));
        run("scaled 1e+18 (24x24)", multiply(A, array(1e18f)));
        run("scaled 1e+37 (24x24)", multiply(A, array(1e37f)));
    }
    {
        // Garbage in the unread triangle must not leak in.
        array A = random_symmetric(1, 20, 604);
        array junk = multiply(triu(random_symmetric(1, 20, 605), 1), array(1e6f));
        run("lower triangle only (junk above)", add(A, junk), {}, true);
        array junk_l = multiply(tril(random_symmetric(1, 20, 606), -1), array(1e6f));
        run("upper triangle only (junk below)", add(A, junk_l), {}, false);
    }

    // -------------------------------------------------------------------------
    // Non-finite input: NaN out, no hang, no exception.
    // -------------------------------------------------------------------------
    std::printf("\n[ non-finite input ]\n");
    {
        std::vector<float> data(8 * 8, 1.0f);
        data[3 * 8 + 5] = data[5 * 8 + 3] = NAN;
        array A = from_values(data, {8, 8});
        ++g_checks;
        try {
            EighResult r = detail::eigh_jacobi(A, true, true, {});
            eval({r.eigenvalues, r.eigenvectors, r.info});
            const unsigned info = r.info.item<uint32_t>();
            const bool all_nan = !any(logical_not(isnan(r.eigenvalues))).item<bool>();
            if (!detail::eigh_nonfinite(info) || !all_nan)
                fail("NaN input 8x8", "expected non-finite flag and all-NaN eigenvalues");
            else std::printf("  ok    %-44s info=0x%x\n", "NaN input 8x8 -> NaN, flagged", info);
        } catch (const std::exception& e) {
            fail("NaN input 8x8", std::string("threw: ") + e.what());
        }
    }

    // -------------------------------------------------------------------------
    // Routing policy. The policy is device-tuned, so nothing here may assume
    // the values measured on any one GPU: each check installs the policy it
    // needs, and the device's own policy is restored at the end.
    // -------------------------------------------------------------------------
    std::printf("\n[ routing policy ]\n");
    {
        const EighPolicy original = eigh_policy();
        const std::string original_source = eigh_policy_source();
        auto expect = [&](const std::string& label, bool ok, const std::string& why = "") {
            ++g_checks;
            if (ok) std::printf("  ok    %s\n", label.c_str());
            else    fail(label, why);
        };
        auto api = [&](const std::string& label, const array& A) {
            auto [w, V] = eigh_accelerated(A);
            eval({w, V});
            const auto& sh = A.shape();
            Shape info_shape(sh.begin(), sh.end() - 2);
            check(label, A, EighResult{w, V, full(info_shape, (uint32_t)(1u | (1u << 16)))});
        };

        // A known policy: the library defaults.
        EighPolicy known;
        set_eigh_policy(known);
        expect("set_eigh_policy -> source \"user\"", std::string(eigh_policy_source()) == "user");

        unsetenv("EIGH_DEVICE");
        expect("CPU/GPU: N=40 b=1 -> cpu, N=8 b=256 -> gpu, N=128 b=4096 -> cpu",
               eigh_backend(40, 1) == EighBackend::cpu && eigh_uses_gpu(8, 256) &&
               eigh_backend(128, 4096) == EighBackend::cpu);
        expect("GPU split: N=8 simd, N=40 threadgroup, N=96 block",
               eigh_gpu_backend(8, 1) == EighBackend::simd &&
               eigh_gpu_backend(40, 1) == EighBackend::threadgroup &&
               eigh_gpu_backend(96, 1) == EighBackend::block);
        api("routed to CPU (40x40)",       random_symmetric(1, 40, 800));
        api("routed to GPU (256 x 8x8)",   random_symmetric(256, 8, 801));

        setenv("EIGH_DEVICE", "cpu", 1);
        expect("EIGH_DEVICE=cpu forces the CPU", !eigh_uses_gpu(8, 256));
        setenv("EIGH_DEVICE", "gpu", 1);
        expect("EIGH_DEVICE=gpu forces the GPU", eigh_uses_gpu(512, 1));

        // Either side of the block crossover, through the public function.
        for (unsigned nn : {known.block_min_n - 1, known.block_min_n, known.block_min_n + 1})
            api("GPU, N=" + std::to_string(nn), random_symmetric(1, (int)nn, 810 + nn));

        // Force every size onto each GPU backend in turn, so all three are
        // exercised through the public function on any hardware.
        EighPolicy forced = known;
        forced.block_min_n = 1;
        set_eigh_policy(forced);
        expect("block_min_n = 1 -> block at N=20", eigh_gpu_backend(20, 1) == EighBackend::block);
        api("forced -> block        20x20 b3", random_symmetric(3, 20, 820));

        forced = known;
        forced.block_min_n = 1u << 30;
        forced.simd_max_n  = 0;
        set_eigh_policy(forced);
        expect("no block, no simd -> threadgroup at N=4 and N=200",
               eigh_gpu_backend(4, 1) == EighBackend::threadgroup &&
               eigh_gpu_backend(200, 1) == EighBackend::threadgroup);
        api("forced -> threadgroup  200x200", random_symmetric(1, 200, 821));
        api("forced -> threadgroup  4x4 b50", random_symmetric(50, 4, 822));

        forced = known;
        forced.block_min_n = 1u << 30;
        forced.simd_max_n  = 1u << 30;
        set_eigh_policy(forced);
        expect("simd everywhere -> simd at N=48", eigh_gpu_backend(48, 1) == EighBackend::simd);
        api("forced -> simd         48x48 b5", random_symmetric(5, 48, 823));

        // The batch-dependent block crossover.
        forced = known;
        forced.block_min_n_batched = 32;
        forced.block_min_batch     = 8;
        set_eigh_policy(forced);
        expect("batched block term: N=40 b=8 block, N=40 b=4 threadgroup, N=24 b=64 threadgroup",
               eigh_gpu_backend(40, 8) == EighBackend::block &&
               eigh_gpu_backend(40, 4) == EighBackend::threadgroup &&
               eigh_gpu_backend(24, 64) == EighBackend::threadgroup);
        api("batched block term     40x40 b8", random_symmetric(8, 40, 824));

        // The CPU boundary.
        unsetenv("EIGH_DEVICE");
        forced = known;
        forced.gpu_max_n = 0;
        set_eigh_policy(forced);
        expect("gpu_max_n = 0 -> never GPU", !eigh_uses_gpu(8, 4096) && !eigh_uses_gpu(64, 4096));
        forced = known;
        forced.gpu_max_n = kEighNoLimit;
        forced.gpu_min_batch_times_n = 0;
        set_eigh_policy(forced);
        expect("no cap, no floor -> always GPU", eigh_uses_gpu(1, 1) && eigh_uses_gpu(4096, 1));
        api("always GPU             12x12 b1", random_symmetric(1, 12, 825));
        forced.gpu_min_batch = 4;
        set_eigh_policy(forced);
        expect("gpu_min_batch = 4 -> N=512 b3 cpu, b4 gpu, N=4 b3 cpu",
               !eigh_uses_gpu(512, 3) && eigh_uses_gpu(512, 4) && !eigh_uses_gpu(4, 3));
        forced.gpu_min_batch = 1;
        set_eigh_policy(forced);
        setenv("EIGH_DEVICE", "gpu", 1);

        set_eigh_policy(original);
        const EighPolicy back = eigh_policy();
        expect("policy restored",
               back.simd_max_n == original.simd_max_n && back.block_min_n == original.block_min_n &&
               back.gpu_max_n == original.gpu_max_n &&
               back.gpu_min_batch_times_n == original.gpu_min_batch_times_n &&
               back.gpu_min_batch == original.gpu_min_batch);
        std::printf("  note: this device's policy came from \"%s\"\n", original_source.c_str());
    }

    // -------------------------------------------------------------------------
    // Errors.
    // -------------------------------------------------------------------------
    std::printf("\n[ errors ]\n");
    {
        ++g_checks;
        try { eigh_accelerated(zeros({4, 5})); fail("non-square input", "did not throw"); }
        catch (const std::invalid_argument&) { std::printf("  ok    non-square input throws\n"); }
        ++g_checks;
        try { eigh_accelerated(zeros({4, 4}), "X"); fail("bad uplo", "did not throw"); }
        catch (const std::invalid_argument&) { std::printf("  ok    bad uplo throws\n"); }
        ++g_checks;
        try {
            EighOptions o; o.max_sweeps = 0;
            detail::eigh_jacobi(random_symmetric(1, 8, 700), true, true, o);
            fail("max_sweeps=0", "did not throw");
        } catch (const std::runtime_error&) { std::printf("  ok    max_sweeps=0 on a non-diagonal matrix throws\n"); }
    }

    std::printf("\n%d checks, %d failures\n\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
