// SVD routing harness: times every backend on one (batch, M, N) point.
//
//   usage: sweep_svd <batch> <M> <N> <backend>[,<backend>...]
//   backends: cpu      thin SVD through MLX (Accelerate LAPACK), QR first when tall
//             jacobi   whole-matrix one-sided Jacobi kernel on the matrix itself
//             block    block one-sided Jacobi kernel on the matrix itself
//             qr       this library's QR, then the whole-matrix kernel on R
//             qrblock  this library's QR, then the block kernel on R
//             bidiag   GPU bidiagonalization, LAPACK's bidiagonal SVD (svd_bidiag.mm)
//             band     the two-stage reduction with vectors (svd_bidiag.mm), its band 16 wide
//             bidiag_batch  a batch bidiagonalized at once (svd_bidiag_batch.mm), rows and
//                      columns up to 1024
//             gk       Householder bidiagonalization and implicit QR, one
//                      threadgroup per matrix (svd_golub_kahan.mm): on the
//                      matrix itself where it fits, else on R after this
//                      library's QR, as svd.mm routes it; k up to its device limit
//             gk_share golub_kahan and the CPU path sharing the batch (share_min_batch)
//             cpu_vals, bidiag_vals, bidiag_batch_vals, gk_vals, gk_share_vals   the same for
//                      singular values alone
//             band_vals  singular values alone by the two-stage reduction (svd_bidiag.mm), its band
//                        16 wide; band8_vals, band32_vals the same 8 and 32 wide
//   out:   batch,M,N,backend,ok,ms,p25,p75,reps   (one row per backend)
//
//   usage: sweep_svd --policy
//   out:   one JSON object: the device and the routing policy the library
//          resolved for it
//
// One point per process, as in sweep_qr.cpp and sweep_eigh.cpp, so nothing
// cached for an earlier shape can skew a later one; its backends timed as
// sweep_timing.h says. Every backend computes the
// same thing, the thin factors, so the comparison is like for like. The two
// QR backends pin their kernel, so neither follows the policy in effect.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include <mlx/mlx.h>

#include "sweep_timing.h"

#include <metal_linalg/device.h>
#include <metal_linalg/eigh.h>   // device_name()
#include <metal_linalg/svd.h>

using namespace mlx::core;
using namespace metal_linalg;

namespace {

array random_matrix(int batch, int M, int N) {
    std::mt19937 rng(1234);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> data((size_t)batch * M * N);
    for (auto& v : data) v = dist(rng);
    if (batch == 1) return array(data.begin(), {M, N}, float32);
    return array(data.begin(), {batch, M, N}, float32);
}

struct Solver {
    std::string name;
    SvdResult (*fn)(const array&);
    bool vectors = true;   // false: singular values alone (svdvals), checked against MLX's CPU svd
};

SvdOptions with_kernel(SvdOptions::Kernel k) {
    SvdOptions o;
    o.kernel = k;
    return o;
}

SvdResult solve_cpu(const array& A)    { return detail::svd_cpu(A, true); }
SvdResult solve_jacobi(const array& A) { return detail::svd_jacobi(A, true, {}); }
SvdResult solve_block(const array& A)  { return detail::svd_block_jacobi(A, true, {}); }
SvdResult solve_qr(const array& A) {
    return detail::svd_qr_jacobi(A, true, with_kernel(SvdOptions::Kernel::jacobi));
}
SvdResult solve_qr_block(const array& A) {
    return detail::svd_qr_jacobi(A, true, with_kernel(SvdOptions::Kernel::block));
}
SvdResult solve_bidiag(const array& A)  { return detail::svd_bidiag(A, true); }
// Singular values alone, as svdvals runs them.
SvdResult vals_cpu(const array& A)      { return detail::svd_cpu(A, false); }
SvdResult vals_bidiag(const array& A)   { return detail::svd_bidiag(A, false); }
SvdResult vals_band(const array& A)     { return detail::svd_band(A, 16); }
SvdResult solve_band(const array& A)    { return detail::svd_band_vectors(A); }
SvdResult solve_bidiag_batch(const array& A) { return detail::svd_bidiag_batch(A, true); }
SvdResult vals_bidiag_batch(const array& A)  { return detail::svd_bidiag_batch(A, false); }
SvdResult vals_band8(const array& A)    { return detail::svd_band(A, 8); }
SvdResult vals_band32(const array& A)   { return detail::svd_band(A, 32); }
// golub_kahan as svd.mm routes it inside its window.
SvdResult gk(const array& A, bool uv) {
    const int M = A.shape(-2), N = A.shape(-1);
    if (metal_linalg::detail::svd_gk_fits(M, N)) return detail::svd_golub_kahan(A, uv);
    return detail::svd_qr_jacobi(A, uv, with_kernel(SvdOptions::Kernel::golub_kahan));
}
SvdResult solve_gk(const array& A)      { return gk(A, true); }
SvdResult vals_gk(const array& A)       { return gk(A, false); }
// The same, its batch shared with the CPU path (share_min_batch).
SvdResult solve_gk_share(const array& A) { return detail::svd_golub_kahan_shared(A, true); }
SvdResult vals_gk_share(const array& A)  { return detail::svd_golub_kahan_shared(A, false); }

// MLX's CPU svd of the point's matrix (singular values alone), computed once
// for every backend that needs it.
const array& values_reference(const array& A) {
    static std::optional<array> ref;
    if (!ref) {
        ref = linalg::svd(A, false, Device::cpu)[0];
        eval({*ref});
    }
    return *ref;
}

// Singular values alone: the largest error against MLX's CPU svd, relative to S_max.
float values_error(const Solver& s, const array& A, double& call_ms) {
    try {
        const auto t0 = std::chrono::high_resolution_clock::now();
        array S = s.fn(A).S;
        eval({S});
        call_ms = sweep::ms_since(t0);
        // The reference on its own, after the backend's work has finished: queued
        // behind it, the comparison's GPU work outlasted the GPU's watchdog.
        const array& ref = values_reference(A);
        array e = max(abs(subtract(S, ref)));
        array top = max(abs(ref));
        eval({e, top});
        const float err = e.item<float>();
        return std::isfinite(err) ? err / std::max(top.item<float>(), 1e-30f) : INFINITY;
    } catch (const std::exception&) {
        return INFINITY;
    }
}

// Relative reconstruction error, or infinity if the backend cannot run here.
float correctness(const Solver& s, const array& A, double& call_ms) {
    if (!s.vectors) return values_error(s, A, call_ms);
    try {
        const auto t0 = std::chrono::high_resolution_clock::now();
        SvdResult r = s.fn(A);
        eval({r.U, r.S, r.Vt});
        call_ms = sweep::ms_since(t0);
        array bad = any(logical_or(isnan(r.U), isinf(r.U)));
        eval({bad});
        if (bad.item<bool>()) return INFINITY;
        array e  = sqrt(sum(square(subtract(matmul(multiply(r.U, expand_dims(r.S, -2)), r.Vt), A))));
        array nA = sqrt(sum(square(A)));
        eval({e, nA});
        return e.item<float>() / std::max(nA.item<float>(), 1e-30f);
    } catch (const std::exception&) {
        return INFINITY;
    }
}

} // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::string(argv[1]) == "--policy") {
        const SvdPolicy p = svd_policy();
        std::printf("{\"device\": \"%s\", \"gpu_cores\": %u, \"source\": \"%s\", "
                    "\"qr_min_rows\": %u, \"qr_min_k\": %u, "
                    "\"block_min_k\": %u, \"block_min_k_batched\": %u, \"block_min_batch\": %u, "
                    "\"gpu_max_k\": %u, \"gpu_min_batch_times_k\": %u, \"gpu_min_batch\": %u, \"gpu_max_l\": %u, "
                    "\"bidiag_min_k\": %u, \"values_bidiag_min_k\": %u, "
                    "\"bidiag_max_batch\": %u, \"values_bidiag_max_batch\": %u, "
                    "\"gk_min_k\": %u, \"gk_max_k\": %u, \"gk_limit\": %u, "
                    "\"values_gpu_max_k\": %u, \"values_gpu_min_batch_times_k\": %u, "
                    "\"values_gpu_min_batch\": %u, \"values_gpu_max_l\": %u, \"share_min_batch\": %u, "
                    "\"gpu_big_batch_max_k\": %u, \"gpu_big_batch_min\": %u, \"values_band_min_k\": %u, "
                    "\"values_band_width\": %u, \"band_min_k\": %u, \"bidiag_batch_min_k\": %u, "
                    "\"bidiag_batch_max_k\": %u, \"bidiag_batch_min_batch\": %u, \"bidiag_batch_max_l\": %u, "
                    "\"values_bidiag_batch_min_k\": %u, \"values_bidiag_batch_max_k\": %u, "
                    "\"values_bidiag_batch_min_batch\": %u, \"values_bidiag_batch_max_l\": %u, \"share_min_k\": %u}\n",
                    device_name(), p.gpu_cores, svd_policy_source(),
                    p.qr_min_rows, p.qr_min_k,
                    p.block_min_k, p.block_min_k_batched, p.block_min_batch,
                    p.gpu_max_k, p.gpu_min_batch_times_k, p.gpu_min_batch, p.gpu_max_l,
                    p.bidiag_min_k, p.values_bidiag_min_k, p.bidiag_max_batch, p.values_bidiag_max_batch,
                    p.gk_min_k, p.gk_max_k, metal_linalg::detail::svd_gk_max_k(),
                    p.values_gpu_max_k, p.values_gpu_min_batch_times_k, p.values_gpu_min_batch,
                    p.values_gpu_max_l, p.share_min_batch, p.gpu_big_batch_max_k, p.gpu_big_batch_min,
                    p.values_band_min_k, p.values_band_width, p.band_min_k, p.bidiag_batch_min_k,
                    p.bidiag_batch_max_k, p.bidiag_batch_min_batch, p.bidiag_batch_max_l,
                    p.values_bidiag_batch_min_k, p.values_bidiag_batch_max_k, p.values_bidiag_batch_min_batch,
                    p.values_bidiag_batch_max_l, p.share_min_k);
        return 0;
    }
    if (argc != 5) {
        std::fprintf(stderr, "usage: %s <batch> <M> <N> <cpu|jacobi|block|qr|qrblock|bidiag|bidiag_batch|band|gk|gk_share|cpu_vals|bidiag_vals|bidiag_batch_vals|band_vals|band8_vals|band32_vals|gk_vals|gk_share_vals>[,...]\n"
                             "       %s --policy\n", argv[0], argv[0]);
        return 2;
    }
    const int batch = std::atoi(argv[1]);
    const int M     = std::atoi(argv[2]);
    const int N     = std::atoi(argv[3]);

    std::vector<Solver> solvers;
    const std::string list = argv[4];
    for (size_t pos = 0; pos <= list.size();) {
        size_t comma = list.find(',', pos);
        if (comma == std::string::npos) comma = list.size();
        const std::string name = list.substr(pos, comma - pos);
        if      (name == "cpu")     solvers.push_back({name, solve_cpu});
        else if (name == "jacobi")  solvers.push_back({name, solve_jacobi});
        else if (name == "block")   solvers.push_back({name, solve_block});
        else if (name == "qr")      solvers.push_back({name, solve_qr});
        else if (name == "qrblock") solvers.push_back({name, solve_qr_block});
        else if (name == "bidiag")      solvers.push_back({name, solve_bidiag});
        else if (name == "bidiag_batch")      solvers.push_back({name, solve_bidiag_batch});
        else if (name == "bidiag_batch_vals") solvers.push_back({name, vals_bidiag_batch, false});
        else if (name == "band")        solvers.push_back({name, solve_band});
        else if (name == "cpu_vals")    solvers.push_back({name, vals_cpu, false});
        else if (name == "bidiag_vals") solvers.push_back({name, vals_bidiag, false});
        else if (name == "band_vals")   solvers.push_back({name, vals_band, false});
        else if (name == "band8_vals")  solvers.push_back({name, vals_band8, false});
        else if (name == "band32_vals") solvers.push_back({name, vals_band32, false});
        else if (name == "gk")          solvers.push_back({name, solve_gk});
        else if (name == "gk_vals")     solvers.push_back({name, vals_gk, false});
        else if (name == "gk_share")      solvers.push_back({name, solve_gk_share});
        else if (name == "gk_share_vals") solvers.push_back({name, vals_gk_share, false});
        else if (!name.empty()) { std::fprintf(stderr, "unknown backend: %s\n", name.c_str()); return 2; }
        pos = comma + 1;
    }

    set_default_device(Device::gpu);
    // MLX's buffer cache is left on, as an MLX program has it (2.16.0; off
    // before): with it off, every call's outputs are fresh pages, which the
    // GPU maps at some 12 us a MB and the CPU faults in, so the sweeps timed
    // allocation as much as the decompositions (4096 of 64 x 64 QR on an M5
    // Pro: 6.4 ms against 4.0 on the GPU, 10.5 against 9.0 on the CPU).

    array A = random_matrix(batch, M, N);
    eval({A});
    std::vector<sweep::Backend> bs;
    for (const auto& s : solvers)
        bs.push_back({[&s, &A](double& call_ms) { return correctness(s, A, call_ms); },
                      [&s, &A] {
                          SvdResult r = s.fn(A);
                          if (s.vectors) eval({r.U, r.S, r.Vt}); else eval({r.S});
                      },
                      s.vectors ? 0 : 1});
    const std::vector<sweep::Result> rs = sweep::measure_all(bs);
    for (size_t i = 0; i < solvers.size(); ++i) {
        const sweep::Timing& t = rs[i].t;
        std::printf("%d,%d,%d,%s,%d,%.6f,%.6f,%.6f,%d\n",
                    batch, M, N, solvers[i].name.c_str(), rs[i].ok ? 1 : 0, t.median, t.p25, t.p75, t.reps);
    }
    std::fflush(stdout);
    return 0;
}
