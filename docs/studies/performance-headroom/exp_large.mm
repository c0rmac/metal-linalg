// Experiment 3: where one large eigh goes, step by step, against the ceilings.
//   lib tridiag     the library's tridiag backend, with and without vectors
//   lib cpu         the library's CPU path (ssyevd / ssyevd_2stage), multithreaded Accelerate
//   cpu steps       ssytrd (one stage), ssytrd_sy2sb + ssytrd_sb2st (two stage), ssterf, sstedc
#include <metal_linalg/core.h>
#include <Accelerate/Accelerate.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <functional>
#include <random>
#include <vector>

using namespace metal_linalg;
using core::Matrices;
using L = __LAPACK_int;
using clk = std::chrono::steady_clock;
static double now_ms() { return std::chrono::duration<double, std::milli>(clk::now().time_since_epoch()).count(); }

static double tmin(const std::function<void()>& f, int reps) {
    double best = 1e30;
    for (int r = 0; r < reps; ++r) { double t0 = now_ms(); f(); best = std::min(best, now_ms() - t0); }
    return best;
}

int main(int argc, char** argv) {
    std::vector<int> sizes = {1024, 2048, 4096, 8192};
    if (argc > 1) { sizes.clear(); for (int i = 1; i < argc; ++i) sizes.push_back(atoi(argv[i])); }
    std::mt19937 rng(11);
    std::normal_distribution<float> nd;
    std::printf("%6s | %9s %9s | %9s %9s | %9s %9s %9s %9s %9s %9s\n", "N", "trid vec", "trid val", "cpu vec",
                "cpu val", "ssytrd", "sy2sb", "sb2st", "ssterf", "sstedc", "floor");
    for (int n : sizes) {
        const size_t nn = (size_t)n * n;
        float* a = nullptr;
        posix_memalign((void**)&a, 16384, nn * 4);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j <= i; ++j) a[(size_t)i * n + j] = a[(size_t)j * n + i] = nd(rng);
        std::vector<float> w(n), v(nn), work_a(nn);
        Matrices m{a, 1, (uint32_t)n, (uint32_t)n};
        const int reps = n >= 8192 ? 1 : n >= 4096 ? 2 : 3;

        core::detail::eigh_tridiag(m, true, w.data(), v.data(), nullptr);   // warm up
        double t_tv = tmin([&] { core::detail::eigh_tridiag(m, true, w.data(), v.data(), nullptr); }, reps);
        double t_tn = tmin([&] { core::detail::eigh_tridiag(m, true, w.data(), nullptr, nullptr); }, reps);
        double t_cv = n <= 4096 ? tmin([&] { core::detail::eigh_cpu(m, true, w.data(), v.data(), nullptr); }, reps) : NAN;
        double t_cn = tmin([&] { core::detail::eigh_cpu(m, true, w.data(), nullptr, nullptr); }, reps);

        // CPU steps, column-major lower = row-major upper; a is symmetric anyway.
        std::vector<float> d(n), e(n), tau(n);
        char ul = 'L';
        L N = n, info = 0, lw = -1;
        float q = 0;
        ssytrd_(&ul, &N, work_a.data(), &N, d.data(), e.data(), tau.data(), &q, &lw, &info);
        std::vector<float> work((size_t)q + 1);
        lw = (L)work.size();
        double t_trd = tmin([&] {
            std::memcpy(work_a.data(), a, nn * 4);
            ssytrd_(&ul, &N, work_a.data(), &N, d.data(), e.data(), tau.data(), work.data(), &lw, &info);
        }, reps);

        // Two stage, kd = 64 (LAPACK's ilaenv2stage default for this size range is 64 or so)
        L kd = getenv("KD") ? atoi(getenv("KD")) : 64, ldab = kd + 1;
        std::vector<float> ab((size_t)ldab * n), tau2(n);
        lw = -1;
        ssytrd_sy2sb_(&ul, &N, &kd, work_a.data(), &N, ab.data(), &ldab, tau2.data(), &q, &lw, &info);
        std::vector<float> w2((size_t)q + 1);
        L lw2 = (L)w2.size();
        double t_sy2sb = tmin([&] {
            std::memcpy(work_a.data(), a, nn * 4);
            ssytrd_sy2sb_(&ul, &N, &kd, work_a.data(), &N, ab.data(), &ldab, tau2.data(), w2.data(), &lw2, &info);
        }, reps);
        char st1 = 'N', vect = 'N';
        L lhous = -1, lw3 = -1;
        float hq = 0, wq = 0;
        ssytrd_sb2st_(&st1, &vect, &ul, &N, &kd, ab.data(), &ldab, d.data(), e.data(), &hq, &lhous, &wq, &lw3, &info);
        std::vector<float> hous((size_t)hq + 1), w3((size_t)wq + 1);
        lhous = (L)hous.size(); lw3 = (L)w3.size();
        std::vector<float> ab_copy = ab;
        double t_sb2st = tmin([&] {
            ab = ab_copy;
            ssytrd_sb2st_(&st1, &vect, &ul, &N, &kd, ab.data(), &ldab, d.data(), e.data(), hous.data(), &lhous,
                          w3.data(), &lw3, &info);
        }, reps);
        if (info) std::printf("sb2st info %lld\n", (long long)info);

        std::vector<float> d0 = d, e0 = e;
        double t_sterf = tmin([&] { d = d0; e = e0; ssterf_(&N, d.data(), e.data(), &info); }, reps);
        char compz = 'I';
        L lwz = -1, liw = -1, iq = 0;
        float zq = 0;
        sstedc_(&compz, &N, d.data(), e.data(), v.data(), &N, &zq, &lwz, &iq, &liw, &info);
        std::vector<float> wz((size_t)zq + 1);
        std::vector<L> iw(iq + 1);
        lwz = (L)wz.size(); liw = (L)iw.size();
        double t_stedc = tmin([&] {
            d = d0; e = e0;
            sstedc_(&compz, &N, d.data(), e.data(), v.data(), &N, wz.data(), &lwz, iw.data(), &liw, &info);
        }, reps);
        // Bandwidth floor of a one-stage reduction: the trailing lower triangle read once per column,
        // n^3/6 floats, at the 282 GB/s measured by exp_gemm.
        double floor_ms = (double)n * n * n / 6 * 4 / 282e9 * 1e3;
        std::printf("%6d | %9.1f %9.1f | %9.1f %9.1f | %9.1f %9.1f %9.1f %9.1f %9.1f %9.1f\n", n, t_tv, t_tn, t_cv,
                    t_cn, t_trd, t_sy2sb, t_sb2st, t_sterf, t_stedc, floor_ms);
        std::fflush(stdout);
        free(a);
    }
}
