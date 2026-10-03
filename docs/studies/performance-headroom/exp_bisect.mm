// ssterf (serial, what eigvalsh's tridiag backend uses) against bisection
// (sstebz) split by eigenvalue index over all CPU cores.
#include <Accelerate/Accelerate.h>
#include <vecLib/thread_api.h>
#include <dispatch/dispatch.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <functional>
#include <random>
#include <vector>
using L = __LAPACK_int;
using clk = std::chrono::steady_clock;
static double now_ms() { return std::chrono::duration<double, std::milli>(clk::now().time_since_epoch()).count(); }
static double tmin(const std::function<void()>& f, int reps) { double b = 1e30; for (int r = 0; r < reps; ++r) { double t0 = now_ms(); f(); b = std::min(b, now_ms() - t0); } return b; }
int main() {
  std::mt19937 rng(9); std::normal_distribution<float> nd;
  std::printf("%6s | %9s %9s %9s | %9s\n", "N", "ssterf", "stebz x1", "stebz x18", "max|dw|");
  for (int n : {2048, 4096, 8192}) {
    std::vector<float> a((size_t)n * n);
    for (int i = 0; i < n; ++i) for (int j = 0; j <= i; ++j) a[(size_t)i * n + j] = a[(size_t)j * n + i] = nd(rng);
    std::vector<float> d0(n), e0(n), tau(n), hq(1), wq(1);
    char vn = 'N', ul = 'L'; L N = n, info = 0, lh = -1, lw = -1;
    ssytrd_2stage_(&vn, &ul, &N, a.data(), &N, d0.data(), e0.data(), tau.data(), hq.data(), &lh, wq.data(), &lw, &info);
    std::vector<float> h((size_t)hq[0] + 1), w((size_t)wq[0] + 1); lh = (L)h.size(); lw = (L)w.size();
    ssytrd_2stage_(&vn, &ul, &N, a.data(), &N, d0.data(), e0.data(), tau.data(), h.data(), &lh, w.data(), &lw, &info);
    std::vector<float> d, e;
    double t_sterf = tmin([&] { d = d0; e = e0; ssterf_(&N, d.data(), e.data(), &info); }, 3);
    std::vector<float> wref = d; std::sort(wref.begin(), wref.end());
    std::vector<float> wb(n);
    auto bisect = [&](unsigned parts) {
      dispatch_apply(parts, dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0), ^(size_t p) {
        BLASSetThreading(BLAS_THREADING_SINGLE_THREADED);
        L il = 1 + (L)((int64_t)n * p / parts), iu = (L)((int64_t)n * (p + 1) / parts);
        if (iu < il) return;
        char ra = 'I', ord = 'E'; float vl = 0, vu = 0, abstol = 0; L m = 0, nsplit = 0, inf = 0, NN = n;
        std::vector<float> wl(n), work(4 * n); std::vector<L> iblock(n), isplit(n), iwork(3 * n);
        sstebz_(&ra, &ord, &NN, &vl, &vu, &il, &iu, &abstol, d0.data(), e0.data(), &m, &nsplit, wl.data(),
                iblock.data(), isplit.data(), work.data(), iwork.data(), &inf);
        std::copy(wl.begin(), wl.begin() + m, wb.begin() + (il - 1));
      });
    };
    double t1 = tmin([&] { bisect(1); }, 1);
    double t18 = tmin([&] { bisect(18); }, 3);
    double dw = 0, sc = 0; for (int i = 0; i < n; ++i) { dw = std::max(dw, (double)std::fabs(wb[i] - wref[i])); sc = std::max(sc, (double)std::fabs(wref[i])); }
    std::printf("%6d | %9.1f %9.1f %9.1f | %9.1e\n", n, t_sterf, t1, t18, dw / sc);
    std::fflush(stdout);
  }
}
