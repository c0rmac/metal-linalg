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
    // tridiag: the reduction runs on the GPU in panels of 32 columns while more
    // than 33 remain, and LAPACK takes the rest, so the sizes straddle every
    // boundary of that: no GPU panel at all (N <= 33), exactly one, a partial
    // last one, many; and above 1024, more threadgroups' norm and dot partials
    // per column than a simdgroup has lanes. The back-transformation goes 128
    // reflectors at a time.
    // -------------------------------------------------------------------------
    std::printf("\n[ tridiag backend ]\n");
    {
        auto tri = [](const array& A, bool vectors, bool lower) {
            EighResult r = detail::eigh_tridiag(A, vectors, lower);
            eval({r.eigenvalues, r.eigenvectors, r.info});
            return r;
        };
        for (int n : {1, 2, 3, 31, 33, 34, 35, 64, 65, 66, 97, 129, 130, 257, 300, 513, 1024, 1100}) {
            array A = random_symmetric(1, n, 1000 + n);
            check("tridiag " + std::to_string(n) + "x" + std::to_string(n), A, tri(A, true, true));
        }
        // Eigenvalues alone, against the same matrix with eigenvectors.
        for (int n : {2, 65, 300, 1024}) {
            array A = random_symmetric(1, n, 1100 + n);
            EighResult rv = tri(A, false, true), rw = tri(A, true, true);
            check("tridiag values only " + std::to_string(n) + "x" + std::to_string(n), A, rv, true, false);
            const float d = max_abs(subtract(rv.eigenvalues, rw.eigenvalues)) / std::max(frobenius(A), 1.0f);
            ++g_checks;
            if (d > 1e-6f) fail("tridiag values-only == with vectors", "differ by " + std::to_string(d));
        }
        // One triangle read: junk in the other, both ways.
        {
            const int n = 200;
            array S = random_symmetric(1, n, 1200);
            array junk = full({n, n}, 1e30f);
            check("tridiag lower, junk above 200x200", add(tril(S), triu(junk, 1)),
                  tri(add(tril(S), triu(junk, 1)), true, true), true);
            check("tridiag upper, junk below 200x200", add(triu(S), tril(junk, -1)),
                  tri(add(triu(S), tril(junk, -1)), true, false), false);
        }
        // A batch, pipelined over two workspace slots: odd and even counts.
        {
            array A = random_symmetric(3, 150, 1300);
            check("tridiag batch 3 x 150x150", A, tri(A, true, true));
            array B = random_symmetric(4, 70, 1301);
            check("tridiag batch 4 x 70x70", B, tri(B, true, true));
            EighResult bv = tri(B, false, true);
            check("tridiag batch 4 x 70x70 values only", B, bv, true, false);
        }
        // Magnitudes a float32 product would over- or underflow without scaling.
        for (float s : {1e-30f, 1e-20f, 1e20f, 1e37f}) {
            array A = multiply(random_symmetric(1, 160, 1400), array(s / 5.0f));
            char label[64];
            std::snprintf(label, sizeof label, "tridiag scaled by %.0e 160x160", s);
            check(label, A, tri(A, true, true));
        }
        // Degenerate inputs: every reflector trivial (tau = 0), or eigenvalues repeated.
        check("tridiag zero matrix 100x100", zeros({100, 100}), tri(zeros({100, 100}), true, true));
        check("tridiag identity 100x100", eye(100), tri(eye(100), true, true));
        {
            std::vector<float> dvals(120);
            for (int i = 0; i < 120; ++i) dvals[i] = (float)(i % 7) - 3.0f;
            array D = diag(from_values(dvals, {120}));
            check("tridiag diagonal 120x120", D, tri(D, true, true));
            std::vector<float> spec(150);
            for (int i = 0; i < 150; ++i) spec[i] = i < 100 ? 1.0f : 2.0f + i;
            array R = with_spectrum(spec);
            check("tridiag repeated eigenvalues 150x150", R, tri(R, true, true));
        }
        // From N = 129 the eigenvectors come from the parallel divide and
        // conquer (divide_conquer.cpp): its deflations (a zero z entry, close
        // values rotated together, a split tridiagonal) on matrices large
        // enough to be divided several times.
        check("tridiag zero matrix 300x300", zeros({300, 300}), tri(zeros({300, 300}), true, true));
        check("tridiag identity 300x300", eye(300), tri(eye(300), true, true));
        {
            std::vector<float> dvals(400);
            for (int i = 0; i < 400; ++i) dvals[i] = (float)(i % 7) - 3.0f;
            array D = diag(from_values(dvals, {400}));
            check("tridiag diagonal 400x400", D, tri(D, true, true));
            std::vector<float> spec(500), close(450);
            for (int i = 0; i < 500; ++i) spec[i] = i < 300 ? 1.0f : 2.0f + (float)(i % 5);
            array R = with_spectrum(spec);
            check("tridiag repeated eigenvalues 500x500", R, tri(R, true, true));
            for (int i = 0; i < 450; ++i) close[i] = 1.0f + 1e-6f * (float)i;
            array C = with_spectrum(close);
            check("tridiag clustered eigenvalues 450x450", C, tri(C, true, true));
            // Blocks coupled by nothing: the tridiagonal splits into
            // independent ones
            std::vector<float> blk((size_t)600 * 600, 0.0f);
            array S = random_symmetric(1, 150, 1450);
            eval({S});
            for (int b = 0; b < 4; ++b)
                for (int i = 0; i < 150; ++i)
                    for (int j = 0; j < 150; ++j)
                        blk[(size_t)(b * 150 + i) * 600 + b * 150 + j] = S.data<float>()[i * 150 + j] * (float)(b + 1);
            array B = from_values(blk, {600, 600});
            check("tridiag block diagonal 4 x 150 in 600x600", B, tri(B, true, true));
        }
        // A NaN in one matrix of a batch: that matrix NaN and flagged, the rest intact.
        {
            const int n = 70;
            std::vector<float> data(2 * n * n);
            array S = random_symmetric(2, n, 1500);
            eval({S});
            std::copy(S.data<float>(), S.data<float>() + 2 * n * n, data.begin());
            data[(size_t)n * n + 5 * n + 3] = NAN;     // matrix 1, lower triangle
            array A = from_values(data, {2, n, n});
            EighResult r = tri(A, true, true);
            array info = reshape(r.info, {-1});
            array w1 = slice(r.eigenvalues, {1, 0}, {2, n});
            array w0 = slice(r.eigenvalues, {0, 0}, {1, n});
            eval({info, w1, w0});
            ++g_checks;
            const bool nan1 = all(isnan(w1)).item<bool>(), fin0 = !has_non_finite(w0);
            const bool flags = !detail::eigh_converged(info.data<uint32_t>()[1]) &&
                               detail::eigh_converged(info.data<uint32_t>()[0]);
            if (!(nan1 && fin0 && flags)) fail("tridiag NaN in one matrix of a batch", "not isolated");
            else std::printf("  ok    %-44s\n", "tridiag NaN in one matrix of a batch");
        }
    }

    // -------------------------------------------------------------------------
    // band, eigenvalues alone: the GPU's blocks while 3b columns remain,
    // LAPACK the rest; a panel of up to 128 rows in one simdgroup, a taller one
    // by TSQR; the band chased to tridiagonal on the CPU's cores, then
    // bisection on the GPU from N = 512. Each band width, either triangle,
    // against LAPACK's eigenvalues.
    // -------------------------------------------------------------------------
    std::printf("\n[ band backend (eigenvalues alone) ]\n");
    {
        auto band = [&](const std::string& label, const array& A, bool lower, uint32_t width) {
            EighResult r = detail::eigh_band(A, lower, width);
            array ref = linalg::eigvalsh(symmetrised(A, lower), "L", Device::cpu);
            array info = reshape(r.info, {-1});
            eval({r.eigenvalues, ref, info});
            ++g_checks;
            const float d = max_abs(subtract(r.eigenvalues, ref)) / std::max(max_abs(ref), 1e-30f);
            bool converged = true;
            for (size_t i = 0; i < info.size(); ++i) converged &= detail::eigh_converged(info.data<uint32_t>()[i]);
            if (!(d <= 2e-5f) || !converged) fail(label, "|w - LAPACK| / |w|max " + std::to_string(d));
            else std::printf("  ok    %-44s |dw|=%.1e\n", label.c_str(), d);
        };
        for (uint32_t w : {8u, 16u, 32u})
            for (int n : {1, 2, 3, 20, 47, 48, 49, 64, 100, 128, 129, 200, 300, 513, 1100})
                band("band b=" + std::to_string(w) + " " + std::to_string(n) + "x" + std::to_string(n),
                     random_symmetric(1, n, 3000 + n), true, w);
        {
            array S = random_symmetric(1, 300, 3100), junk = random_symmetric(1, 300, 3101);
            band("band lower, junk above 300x300", add(tril(S), triu(junk, 1)), true, 16);
            band("band upper, junk below 300x300", add(triu(S), tril(junk, -1)), false, 16);
        }
        band("band batch 3 x 150x150", random_symmetric(3, 150, 3200), true, 16);
        band("band zero 100x100", zeros({100, 100}), true, 16);
        band("band identity 600x600", eye(600), true, 16);
        {
            std::vector<float> spec(600);
            for (int i = 0; i < 600; ++i) spec[i] = i < 200 ? 1.0f : i < 400 ? -2.0f : 1e-4f * i;
            band("band repeated eigenvalues 600x600", with_spectrum(spec), true, 16);
        }
        for (float s : {1e-30f, 1e30f}) {
            char label[64];
            std::snprintf(label, sizeof label, "band scaled by %.0e 160x160", s);
            band(label, multiply(random_symmetric(1, 160, 3300), array(s)), true, 8);
        }
        {   // NaN in one matrix of a batch
            array A = random_symmetric(2, 200, 3400);
            eval({A});
            std::vector<float> data(A.data<float>(), A.data<float>() + 2 * 200 * 200);
            data[200 * 200 + 5 * 200] = NAN;   // row 5, column 0: in the lower triangle, which is read
            EighResult r = detail::eigh_band(from_values(data, {2, 200, 200}), true);
            array info = reshape(r.info, {-1});
            array w1 = slice(r.eigenvalues, {1, 0}, {2, 200}), w0 = slice(r.eigenvalues, {0, 0}, {1, 200});
            eval({info, w0, w1});
            ++g_checks;
            const bool ok = all(isnan(w1)).item<bool>() && !has_non_finite(w0) &&
                            detail::eigh_nonfinite(info.data<uint32_t>()[1]) &&
                            detail::eigh_converged(info.data<uint32_t>()[0]);
            if (!ok) fail("band NaN in one matrix of a batch", "not isolated");
            else std::printf("  ok    %-44s\n", "band NaN in one matrix of a batch");
        }
    }

    // -------------------------------------------------------------------------
    // ql: one threadgroup per matrix, the matrix in threadgroup memory. The
    // sizes straddle the simdgroup boundaries (32, 64) and the switch to a
    // chaser simdgroup of its own (N = 33), up to the largest N the device's
    // threadgroup memory takes.
    // -------------------------------------------------------------------------
    std::printf("\n[ ql backend (N <= %u on this device) ]\n", metal_linalg::detail::eigh_ql_max_n());
    {
        const int max_n = (int)metal_linalg::detail::eigh_ql_max_n();
        auto ql = [](const array& A, bool vectors, bool lower) {
            EighResult r = detail::eigh_ql(A, vectors, lower);
            eval({r.eigenvalues, r.eigenvectors, r.info});
            return r;
        };
        for (int n : {1, 2, 3, 4, 7, 8, 16, 31, 32, 33, 34, 47, 48, 63, 64, 65, 80, 86, 87}) {
            if (n > max_n) continue;
            array A = random_symmetric(5, n, 2000 + n);
            check("ql 5 x " + std::to_string(n) + "x" + std::to_string(n), A, ql(A, true, true));
        }
        // Eigenvalues alone, against the same matrices with eigenvectors.
        for (int n : {1, 9, 33, 64}) {
            if (n > max_n) continue;
            array A = random_symmetric(4, n, 2100 + n);
            EighResult rv = ql(A, false, true), rw = ql(A, true, true);
            check("ql values only 4 x " + std::to_string(n) + "x" + std::to_string(n), A, rv, true, false);
            const float d = max_abs(subtract(rv.eigenvalues, rw.eigenvalues)) / std::max(frobenius(A), 1.0f);
            ++g_checks;
            if (d > 1e-6f) fail("ql values-only == with vectors", "differ by " + std::to_string(d));
        }
        // One triangle read: junk in the other, both ways.
        {
            const int n = 40;
            array S = random_symmetric(1, n, 2200);
            array junk = full({n, n}, 1e30f);
            check("ql lower, junk above 40x40", add(tril(S), triu(junk, 1)),
                  ql(add(tril(S), triu(junk, 1)), true, true), true);
            check("ql upper, junk below 40x40", add(triu(S), tril(junk, -1)),
                  ql(add(triu(S), tril(junk, -1)), true, false), false);
        }
        // Batch dimensions beyond one, and a batch big enough to need several
        // command buffers when each is given a millisecond.
        check("ql batch [2,3] x 20x20", reshape(random_symmetric(6, 20, 2300), {2, 3, 20, 20}),
              ql(reshape(random_symmetric(6, 20, 2300), {2, 3, 20, 20}), true, true));
        {
            setenv("EIGH_CHUNK_MS", "1", 1);
            array A = random_symmetric(3000, 24, 2310);
            check("ql 3000 x 24x24, 1 ms per command buffer", A, ql(A, true, true));
            unsetenv("EIGH_CHUNK_MS");
        }
        // A batch shared with the CPU path: the GPU's chunks from the front,
        // the CPU's from the back; every matrix solved once, either way.
        {
            auto shared = [](const array& A, bool vectors) {
                EighResult r = detail::eigh_ql_shared(A, vectors, true);
                eval({r.eigenvalues, r.eigenvectors, r.info});
                return r;
            };
            array A = random_symmetric(700, 24, 2320);
            check("ql shared with the CPU 700 x 24x24", A, shared(A, true));
            EighResult rv = shared(A, false);
            check("ql shared with the CPU, values only 700 x 24x24", A, rv, true, false);
            array B = random_symmetric(3, 40, 2321);
            check("ql shared with the CPU 3 x 40x40", B, shared(B, true));
        }
        // Magnitudes a float32 product would over- or underflow without scaling.
        for (float s : {1e-30f, 1e-20f, 1e20f, 1e37f}) {
            array A = multiply(random_symmetric(2, 50, 2400), array(s / 5.0f));
            char label[64];
            std::snprintf(label, sizeof label, "ql scaled by %.0e 2 x 50x50", s);
            check(label, A, ql(A, true, true));
        }
        // Structured spectra: trivial reflectors, repeated, clustered, graded,
        // rank one, negative definite.
        check("ql zero matrix 30x30", zeros({30, 30}), ql(zeros({30, 30}), true, true));
        check("ql identity 45x45", eye(45), ql(eye(45), true, true));
        {
            std::vector<float> dvals(60);
            for (int i = 0; i < 60; ++i) dvals[i] = (float)(i % 7) - 3.0f;
            array D = diag(from_values(dvals, {60}));
            check("ql diagonal 60x60", D, ql(D, true, true));
            std::vector<float> rep(64), graded(48), cluster(40), neg(32);
            for (int i = 0; i < 64; ++i) rep[i] = i < 48 ? 1.0f : 2.0f + i;
            for (int i = 0; i < 48; ++i) graded[i] = std::pow(10.0f, -4.0f + 8.0f * i / 47.0f);
            for (int i = 0; i < 40; ++i) cluster[i] = 1.0f + 1e-6f * i;
            for (int i = 0; i < 32; ++i) neg[i] = -1.0f - i;
            array R = with_spectrum(rep);   // a fresh random basis each call: build it once
            check("ql repeated eigenvalues 64x64", R, ql(R, true, true));
            array G = with_spectrum(graded);
            check("ql graded spectrum 1e-4..1e4 48x48", G, ql(G, true, true));
            array C = with_spectrum(cluster);
            check("ql clustered eigenvalues 40x40", C, ql(C, true, true));
            array Nn = with_spectrum(neg);
            check("ql negative definite 32x32", Nn, ql(Nn, true, true));
            check("ql rank one (ones) 50x50", full({50, 50}, 1.0f), ql(full({50, 50}, 1.0f), true, true));
        }
        // A NaN in one matrix of a batch: that matrix NaN and flagged, the rest intact.
        {
            const int n = 33;
            std::vector<float> data(3 * n * n);
            array S = random_symmetric(3, n, 2500);
            eval({S});
            std::copy(S.data<float>(), S.data<float>() + 3 * n * n, data.begin());
            data[(size_t)n * n + 5 * n + 3] = NAN;     // matrix 1, lower triangle
            array A = from_values(data, {3, n, n});
            EighResult r = ql(A, true, true);
            array info = reshape(r.info, {-1});
            array w1 = slice(r.eigenvalues, {1, 0}, {2, n});
            array rest = concatenate({slice(r.eigenvalues, {0, 0}, {1, n}), slice(r.eigenvalues, {2, 0}, {3, n})});
            eval({info, w1, rest});
            ++g_checks;
            const uint32_t* iw = info.data<uint32_t>();
            const bool nan1 = all(isnan(w1)).item<bool>(), fin = !has_non_finite(rest);
            const bool flags = detail::eigh_nonfinite(iw[1]) && detail::eigh_converged(iw[0]) &&
                               detail::eigh_converged(iw[2]);
            if (!(nan1 && fin && flags)) fail("ql NaN in one matrix of a batch", "not isolated");
            else std::printf("  ok    %-44s\n", "ql NaN in one matrix of a batch");
        }
        // Beyond the threadgroup memory: an error, not a wrong answer.
        ++g_checks;
        try {
            detail::eigh_ql(random_symmetric(1, max_n + 1, 2600), true, true);
            fail("ql N beyond its limit", "did not throw");
        } catch (const std::invalid_argument&) {
            std::printf("  ok    %-44s\n", ("ql N=" + std::to_string(max_n + 1) + " throws invalid_argument").c_str());
        }
    }

    // -------------------------------------------------------------------------
    // The CPU path spreads a batch over cpu_threads() threads. One thread and
    // every thread must agree, and a NaN must stay in its own matrix.
    // -------------------------------------------------------------------------
    std::printf("\n[ CPU path: a batch over %u threads ]\n", cpu_threads());
    {
        const unsigned threads = cpu_threads();
        ++g_checks;
        set_cpu_threads(3);
        const bool set_ok = cpu_threads() == 3;
        set_cpu_threads(0);
        if (!set_ok || cpu_threads() != threads || threads < 1) fail("set_cpu_threads(3), then 0", "not honoured");
        else std::printf("  ok    %-44s\n", "set_cpu_threads(3), then 0 restores the default");
        for (int n : {6, 70}) {
            array A = random_symmetric(41, n, 2700 + n);
            EighResult all_threads = detail::eigh_cpu(A, true, true);
            set_cpu_threads(1);
            EighResult one = detail::eigh_cpu(A, true, true);
            set_cpu_threads(0);
            eval({all_threads.eigenvalues, all_threads.eigenvectors, one.eigenvalues, one.eigenvectors});
            check("CPU 41 x " + std::to_string(n) + "x" + std::to_string(n) + ", every thread", A, all_threads);
            const float d = max_abs(subtract(all_threads.eigenvalues, one.eigenvalues)) / frobenius(A);
            ++g_checks;
            if (d > 1e-6f) fail("CPU one thread == every thread", "differ by " + std::to_string(d));
            else std::printf("  ok    %-44s |dw|=%.1e\n", ("CPU one thread == every thread, N=" + std::to_string(n)).c_str(), d);
        }
        {
            const int n = 12, batch = 64;
            array S = random_symmetric(batch, n, 2800);
            eval({S});
            std::vector<float> data(S.data<float>(), S.data<float>() + (size_t)batch * n * n);
            data[(size_t)37 * n * n + 4 * n + 1] = NAN;   // matrix 37, lower triangle
            array A = from_values(data, {batch, n, n});
            EighResult r = detail::eigh_cpu(A, true, true);
            array w = r.eigenvalues;
            eval({w, r.info});
            const float* wp = w.data<float>();
            bool ok = true;
            for (int b = 0; b < batch; ++b)
                for (int i = 0; i < n; ++i) ok = ok && (std::isnan(wp[b * n + i]) == (b == 37));
            ++g_checks;
            if (!ok || !detail::eigh_nonfinite(r.info.data<uint32_t>()[37])) fail("CPU NaN in matrix 37 of 64", "not isolated");
            else std::printf("  ok    %-44s\n", "CPU NaN in matrix 37 of 64 stays there");
        }
    }

    // -------------------------------------------------------------------------
    // CPU eigenvalues alone: the two-stage reduction from N = 128, the
    // one-stage one below. Each is checked against a known spectrum and
    // against the eigenvalues of the one-stage path with eigenvectors.
    // -------------------------------------------------------------------------
    std::printf("\n[ CPU eigenvalues: one-stage below N=128, two-stage from it ]\n");
    {
        setenv("EIGH_DEVICE", "cpu", 1);
        for (int n : {127, 128, 300, 1024}) {
            std::vector<float> spec(n);
            for (int i = 0; i < n; ++i) spec[i] = -1.0f + 2.0f * i / (n - 1) + (i % 7 == 0 ? 0.5f : 0.0f);
            std::sort(spec.begin(), spec.end());
            array A = with_spectrum(spec);
            array w = eigvalsh_accelerated(A);
            auto [w_v, V] = eigh_accelerated(A);
            eval({w, w_v});
            const float d_spec = max_abs(subtract(w, from_values(spec, {n})));
            const float d_vec  = max_abs(subtract(w, w_v));
            char label[64];
            std::snprintf(label, sizeof label, "N=%d %s", n, n >= 128 ? "two-stage" : "one-stage");
            ++g_checks;
            if (d_spec > 1e-4f || d_vec > 1e-4f) {
                fail(label, "|w - spectrum| " + std::to_string(d_spec) + ", |w - eigh's w| " + std::to_string(d_vec));
            } else {
                std::printf("  ok    %-44s |w-spec|=%.1e |w-eigh|=%.1e\n", label, d_spec, d_vec);
            }
        }
        // Only the named triangle is read: junk in the other must not matter.
        {
            const int n = 300;
            array S = random_symmetric(1, n, 900);
            array junk = full({n, n}, 7.0f);
            array lower_only = add(tril(S), triu(junk, 1));
            array upper_only = add(triu(S), tril(junk, -1));
            array w  = eigvalsh_accelerated(S);
            array wl = eigvalsh_accelerated(lower_only, "L");
            array wu = eigvalsh_accelerated(upper_only, "U");
            eval({w, wl, wu});
            const float d = std::max(max_abs(subtract(w, wl)), max_abs(subtract(w, wu))) / frobenius(S);
            ++g_checks;
            if (d > 1e-6f) fail("N=300 two-stage reads only its triangle (L, U)", "differ by " + std::to_string(d));
            else std::printf("  ok    %-44s |dw|=%.1e\n", "N=300 two-stage reads only its triangle (L, U)", d);
        }
        unsetenv("EIGH_DEVICE");
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

        // Eigenvalues alone: values_gpu_min_batch = 0 means "as for
        // eigenvectors"; set, the values_* fields decide eigvalsh on their own.
        {
            bool same = true;
            for (unsigned nn : {4u, 32u, 64u, 65u, 512u})
                for (unsigned bb : {1u, 8u, 64u, 4096u})
                    same = same && eigvalsh_uses_gpu(nn, bb) == eigh_uses_gpu(nn, bb);
            expect("values boundary unset -> eigvalsh routed as eigh", same);
            EighPolicy v = known;
            v.values_gpu_max_n = 16;
            v.values_gpu_min_batch_times_n = 0;
            v.values_gpu_min_batch = 1;
            set_eigh_policy(v);
            expect("values boundary set -> each side decided apart "
                   "(N=32 b=256: eigh GPU, eigvalsh CPU; N=8 b=1: eigh CPU, eigvalsh GPU)",
                   eigh_uses_gpu(32, 256) && eigvalsh_backend(32, 256) == EighBackend::cpu &&
                   !eigh_uses_gpu(8, 1) && eigvalsh_uses_gpu(8, 1));
            for (auto [bb, nn] : {std::pair{256, 32}, std::pair{1, 8}}) {
                array A = random_symmetric(bb, nn, 805 + nn);
                array w = eigvalsh_accelerated(A);
                auto [w_ref, V] = eigh_accelerated(A);
                eval({w, w_ref});
                const float d = max_abs(subtract(w, w_ref)) / std::max(frobenius(A), 1.0f);
                expect("eigvalsh " + std::to_string(bb) + " x " + std::to_string(nn) + "x" + std::to_string(nn) +
                       " on " + (eigvalsh_uses_gpu(nn, bb) ? "GPU" : "CPU") + " == eigh's eigenvalues",
                       d < 1e-5f, "differ by " + std::to_string(d));
            }
            set_eigh_policy(known);
        }

        // The tridiag backend replaces the CPU from its thresholds; 0 = never.
        {
            expect("tridiag thresholds unset -> never (N=4096 b=1 -> cpu)",
                   eigh_backend(4096, 1) == EighBackend::cpu && eigvalsh_backend(4096, 1) == EighBackend::cpu);
            EighPolicy t = known;
            t.tridiag_min_n = 256;
            t.values_tridiag_min_n = 1024;
            set_eigh_policy(t);
            expect("tridiag_min_n = 256: eigh N=255 cpu, N=256 tridiag; GPU-routed calls unchanged",
                   eigh_backend(255, 1) == EighBackend::cpu && eigh_backend(256, 1) == EighBackend::tridiag &&
                   eigh_backend(8, 4096) == eigh_gpu_backend(8, 4096));
            expect("values_tridiag_min_n = 1024: eigvalsh N=512 cpu, N=1024 tridiag",
                   eigvalsh_backend(512, 1) == EighBackend::cpu && eigvalsh_backend(1024, 1) == EighBackend::tridiag);
            api("routed to tridiag (300x300)", random_symmetric(1, 300, 830));
            {
                array A = random_symmetric(1, 1100, 831);
                array w = eigvalsh_accelerated(A);
                array w_ref = linalg::eigvalsh(A, "L", Device::cpu);
                eval({w, w_ref});
                const float d = max_abs(subtract(w, w_ref)) / std::max(frobenius(A), 1.0f);
                expect("eigvalsh routed to tridiag (1100x1100) == LAPACK", d < kEigTol, "differ by " + std::to_string(d));
            }
            {
                EighPolicy c = t;
                c.tridiag_max_batch = 2;
                c.values_tridiag_max_batch = 1;
                set_eigh_policy(c);
                expect("tridiag_max_batch = 2: N=300 b2 tridiag, b3 cpu; values cap 1: eigvalsh N=1024 b2 cpu",
                       eigh_backend(300, 2) == EighBackend::tridiag && eigh_backend(300, 3) == EighBackend::cpu &&
                       eigvalsh_backend(1024, 1) == EighBackend::tridiag &&
                       eigvalsh_backend(1024, 2) == EighBackend::cpu);
                set_eigh_policy(t);
            }
            {   // the band backend, eigenvalues alone, before tridiag
                EighPolicy c = t;
                c.values_band_min_n = 2048;
                set_eigh_policy(c);
                expect("values_band_min_n = 2048: eigvalsh N=1024 tridiag, N=2048 band; eigh N=2048 tridiag",
                       eigvalsh_backend(1024, 1) == EighBackend::tridiag &&
                       eigvalsh_backend(2048, 1) == EighBackend::band && eigh_backend(2048, 1) == EighBackend::tridiag);
                c.values_tridiag_max_batch = 1;
                set_eigh_policy(c);
                expect("band within values_tridiag_max_batch: eigvalsh N=2048 batch 2 cpu",
                       eigvalsh_backend(2048, 2) == EighBackend::cpu);
                c.values_tridiag_max_batch = 0;
                c.values_band_min_n = 256;
                set_eigh_policy(c);
                {
                    array A = random_symmetric(1, 700, 833);
                    array w = eigvalsh_accelerated(A);
                    array w_ref = linalg::eigvalsh(A, "L", Device::cpu);
                    eval({w, w_ref});
                    const float d = max_abs(subtract(w, w_ref)) / std::max(frobenius(A), 1.0f);
                    expect("eigvalsh routed to band (700x700) == LAPACK", d < kEigTol, "differ by " + std::to_string(d));
                    for (unsigned width : {8u, 32u}) {   // the policy's band width reaches the backend
                        c.values_band_width = width;
                        set_eigh_policy(c);
                        array wb = eigvalsh_accelerated(A);
                        eval({wb});
                        const float db = max_abs(subtract(wb, w_ref)) / std::max(frobenius(A), 1.0f);
                        expect("eigvalsh band, values_band_width = " + std::to_string(width) + " == LAPACK",
                               db < kEigTol && max_abs(subtract(wb, w)) > 0.0f, "differ by " + std::to_string(db));
                    }
                    c.values_band_width = 0;
                }
                c.values_band_min_n = 0;
                set_eigh_policy(c);
                expect("values_band_min_n = 0: never", eigvalsh_backend(4096, 1) == EighBackend::tridiag);
                set_eigh_policy(t);
            }
            setenv("EIGH_DEVICE", "band", 1);
            expect("EIGH_DEVICE=band: eigvalsh band, eigh tridiag",
                   eigvalsh_backend(8, 1) == EighBackend::band && eigh_backend(8, 1) == EighBackend::tridiag);
            setenv("EIGH_DEVICE", "cpu", 1);
            expect("EIGH_DEVICE=cpu keeps the CPU over tridiag", eigh_backend(4096, 1) == EighBackend::cpu);
            setenv("EIGH_DEVICE", "tridiag", 1);
            expect("EIGH_DEVICE=tridiag forces it, even where the GPU rule applies",
                   eigh_backend(8, 4096) == EighBackend::tridiag && eigvalsh_backend(8, 1) == EighBackend::tridiag);
            api("EIGH_DEVICE=tridiag, 64 x 16x16", random_symmetric(64, 16, 832));
            unsetenv("EIGH_DEVICE");
            set_eigh_policy(known);
        }

        // The ql window overrides the Jacobi split inside [ql_min_n, ql_max_n],
        // clipped to what the backend takes on the device.
        {
            expect("ql window unset -> never", eigh_gpu_backend(32, 1024) != EighBackend::ql);
            EighPolicy q = known;
            q.ql_min_n = 8;
            q.ql_max_n = 48;
            set_eigh_policy(q);
            expect("ql window [8, 48]: N=7 not ql, N=8 and N=48 ql, N=49 not ql",
                   eigh_gpu_backend(7, 64) != EighBackend::ql && eigh_gpu_backend(8, 64) == EighBackend::ql &&
                   eigh_gpu_backend(48, 64) == EighBackend::ql && eigh_gpu_backend(49, 64) != EighBackend::ql);
            expect("ql only where the GPU rule applies (N=40 b=1 -> cpu)", eigh_backend(40, 1) == EighBackend::cpu);
            q.ql_max_n = 100000;
            set_eigh_policy(q);
            const unsigned lim = metal_linalg::detail::eigh_ql_max_n();
            expect("ql window clipped to the device's limit (" + std::to_string(lim) + ")",
                   eigh_gpu_backend(lim, 64) == EighBackend::ql && eigh_gpu_backend(lim + 1, 64) != EighBackend::ql);
            q.gpu_max_n = kEighNoLimit;
            q.gpu_min_batch_times_n = 0;
            set_eigh_policy(q);
            api("routed to ql (16 x 30x30)", random_symmetric(16, 30, 840));
            {
                array A = random_symmetric(16, 30, 841);
                array w = eigvalsh_accelerated(A);
                array w_ref = linalg::eigvalsh(A, "L", Device::cpu);
                eval({w, w_ref});
                const float d = max_abs(subtract(w, w_ref)) / std::max(frobenius(A), 1.0f);
                expect("eigvalsh routed to ql (16 x 30x30) == LAPACK", d < kEigTol, "differ by " + std::to_string(d));
            }
            // The large-batch clause: above gpu_max_n, up to its own cap, from its batch.
            {
                EighPolicy c = q;
                c.gpu_max_n = 48;  c.gpu_min_batch_times_n = 8192;  c.gpu_min_batch = 1;
                c.gpu_big_batch_max_n = 64;  c.gpu_big_batch_min = 1024;
                set_eigh_policy(c);
                expect("big-batch clause (49..64 from 1024): N=64 b1024 gpu, b1023 cpu, N=65 b4096 cpu, N=32 b256 "
                       "by the product rule",
                       eigh_uses_gpu(64, 1024) && !eigh_uses_gpu(64, 1023) && !eigh_uses_gpu(65, 4096) &&
                       eigh_uses_gpu(32, 256) && !eigh_uses_gpu(16, 256));
                c.values_gpu_max_n = 48;  c.values_gpu_min_batch_times_n = 16384;  c.values_gpu_min_batch = 1;
                set_eigh_policy(c);
                expect("a rule of its own for eigvalsh has no clause (N=64 b4096 cpu)", !eigvalsh_uses_gpu(64, 4096));
                c.values_gpu_min_batch = 0;
                set_eigh_policy(c);
                expect("eigvalsh as for eigenvectors takes the clause (N=64 b1024 gpu)", eigvalsh_uses_gpu(64, 1024));
                set_eigh_policy(q);
            }
            // Sharing a batch with the CPU, from share_min_batch on, ql only.
            q.share_min_batch = 256;
            set_eigh_policy(q);
            expect("share_min_batch = 256: N=30 b256 shared, b255 not; eigvalsh too",
                   eigh_shares_batch(30, 256) && !eigh_shares_batch(30, 255) && eigvalsh_shares_batch(30, 256));
            api("routed to ql, shared with the CPU (300 x 30x30)", random_symmetric(300, 30, 842));
            set_eigh_policy(known);
            expect("share_min_batch unset -> never", !eigh_shares_batch(30, 1 << 20));
        }

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
        try { eigh_accelerated(astype(eye(4), complex64)); fail("complex input", "did not throw"); }
        catch (const std::invalid_argument&) { std::printf("  ok    complex input throws\n"); }
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
