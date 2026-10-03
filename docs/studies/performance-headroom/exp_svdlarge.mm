// Where one large SVD goes: the bidiag backend with and without vectors, the CPU
// path, and LAPACK's bidiagonal SVD (sbdsdc) alone on the CPU.
#include <metal_linalg/core.h>
#include <Accelerate/Accelerate.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <functional>
#include <random>
#include <vector>
using namespace metal_linalg; using L = __LAPACK_int;
using clk = std::chrono::steady_clock;
static double now_ms() { return std::chrono::duration<double, std::milli>(clk::now().time_since_epoch()).count(); }
static double tmin(const std::function<void()>& f, int reps) { f(); double b = 1e30; for (int r = 0; r < reps; ++r) { double t0 = now_ms(); f(); b = std::min(b, now_ms() - t0); } return b; }
int main() {
  std::mt19937 rng(2); std::normal_distribution<float> nd;
  std::printf("%6s | %9s %9s | %9s %9s | %9s %9s %9s\n", "k", "bid vec", "bid val", "cpu vec", "cpu val", "sbdsdc I", "sbdsdc N", "floor");
  for (int n : {2048, 4096}) {
    size_t nn = (size_t)n * n; float* a; posix_memalign((void**)&a, 16384, nn * 4);
    for (size_t i = 0; i < nn; ++i) a[i] = nd(rng);
    std::vector<float> u(nn), s(n), vt(nn);
    core::Matrices m{a, 1, (uint32_t)n, (uint32_t)n};
    int reps = n >= 4096 ? 2 : 3;
    double tbv = tmin([&]{ core::detail::svd_bidiag(m, u.data(), s.data(), vt.data(), nullptr); }, reps);
    double tbn = tmin([&]{ core::detail::svd_bidiag(m, nullptr, s.data(), nullptr, nullptr); }, reps);
    double tcv = tmin([&]{ core::detail::svd_cpu(m, u.data(), s.data(), vt.data(), nullptr); }, reps);
    double tcn = tmin([&]{ core::detail::svd_cpu(m, nullptr, s.data(), nullptr, nullptr); }, reps);
    std::vector<float> d(n), e(n), d0(n), e0(n);
    {   // the matrix's own bidiagonal, from LAPACK's sgebrd
      std::vector<float> ac(a, a + nn), tq(n), tp(n), wq(1); L lw = -1, inf = 0, NN = n;
      sgebrd_(&NN, &NN, ac.data(), &NN, d0.data(), e0.data(), tq.data(), tp.data(), wq.data(), &lw, &inf);
      std::vector<float> wk((size_t)wq[0] + 1); lw = (L)wk.size();
      double t0 = now_ms();
      sgebrd_(&NN, &NN, ac.data(), &NN, d0.data(), e0.data(), tq.data(), tp.data(), wk.data(), &lw, &inf);
      std::printf("(cpu sgebrd %d: %.0f ms) ", n, now_ms() - t0);
    }
    char up = 'U', ci = 'I', cn = 'N'; L N = n, info = 0, ld = n, one = 1;
    std::vector<float> work((size_t)3 * n * n + 4 * n + 100); std::vector<L> iwork(8 * n);
    std::vector<float> U(nn), VT(nn), q(1); std::vector<L> iq(1);
    double tdi = tmin([&]{ d = d0; e = e0; sbdsdc_(&up, &ci, &N, d.data(), e.data(), U.data(), &ld, VT.data(), &ld, q.data(), iq.data(), work.data(), iwork.data(), &info); }, reps);
    double tdn = tmin([&]{ d = d0; e = e0; sbdsdc_(&up, &cn, &N, d.data(), e.data(), U.data(), &one, VT.data(), &one, q.data(), iq.data(), work.data(), iwork.data(), &info); }, reps);
    // one-stage bidiagonalization reads the trailing matrix twice per column (A v and A^T u): 2 * n^3/3 floats
    double floor_ms = 2.0 * n * (double)n * n / 3 * 4 / 282e9 * 1e3;
    std::printf("%6d | %9.1f %9.1f | %9.1f %9.1f | %9.1f %9.1f %9.1f\n", n, tbv, tbn, tcv, tcn, tdi, tdn, floor_ms);
    free(a);
  }
}
