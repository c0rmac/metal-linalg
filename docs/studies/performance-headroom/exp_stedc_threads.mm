#include <Accelerate/Accelerate.h>
#include <vecLib/thread_api.h>
#include <algorithm>
#include <chrono>
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
  for (int n : {2048, 4096}) {
    std::vector<float> a((size_t)n * n);
    for (int i = 0; i < n; ++i) for (int j = 0; j <= i; ++j) a[(size_t)i * n + j] = a[(size_t)j * n + i] = nd(rng);
    std::vector<float> d0(n), e0(n), tau(n), hq(1), wq(1);
    char vn = 'N', ul = 'L'; L N = n, info = 0, lh = -1, lw = -1;
    ssytrd_2stage_(&vn, &ul, &N, a.data(), &N, d0.data(), e0.data(), tau.data(), hq.data(), &lh, wq.data(), &lw, &info);
    std::vector<float> h((size_t)hq[0] + 1), w((size_t)wq[0] + 1); lh = (L)h.size(); lw = (L)w.size();
    ssytrd_2stage_(&vn, &ul, &N, a.data(), &N, d0.data(), e0.data(), tau.data(), h.data(), &lh, w.data(), &lw, &info);
    std::vector<float> d, e, z((size_t)n * n);
    char ci = 'I'; L lwz = -1, liw = -1, iq = 0; float zq = 0;
    d = d0; e = e0; sstedc_(&ci, &N, d.data(), e.data(), z.data(), &N, &zq, &lwz, &iq, &liw, &info);
    std::vector<float> wz((size_t)zq + 1); std::vector<L> iw(iq + 1); lwz = (L)wz.size(); liw = (L)iw.size();
    auto run = [&] { d = d0; e = e0; sstedc_(&ci, &N, d.data(), e.data(), z.data(), &N, wz.data(), &lwz, iw.data(), &liw, &info); };
    BLASSetThreading(BLAS_THREADING_MULTI_THREADED);   double tm = tmin(run, 3);
    BLASSetThreading(BLAS_THREADING_SINGLE_THREADED);  double ts = tmin(run, 3);
    BLASSetThreading(BLAS_THREADING_MULTI_THREADED);
    std::printf("N=%d sstedc: multi-threaded %.1f ms, single-threaded %.1f ms (%.1fx)\n", n, tm, ts, ts / tm);
  }
}
