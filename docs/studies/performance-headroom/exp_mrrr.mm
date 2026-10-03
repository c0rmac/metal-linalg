// sstedc (divide and conquer, what the tridiag backend calls) against sstemr
// (MRRR) on the tridiagonal of a Gaussian symmetric matrix: time, orthogonality
// of the eigenvectors, and eigenvalue agreement.
#include <Accelerate/Accelerate.h>
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
static double orth(const std::vector<float>& z, int n) {
  std::vector<float> g((size_t)n * n);
  cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, n, n, n, 1.0f, z.data(), n, z.data(), n, 0.0f, g.data(), n);
  double s = 0; for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) { double x = g[(size_t)i * n + j] - (i == j); s += x * x; }
  return std::sqrt(s / n);
}
int main() {
  std::mt19937 rng(9); std::normal_distribution<float> nd;
  std::printf("%6s | %9s %9s | %9s %9s | %9s\n", "N", "sstedc", "sstemr", "orth dc", "orth mr", "max|dw|");
  for (int n : {2048, 4096, 8192}) {
    std::vector<float> a((size_t)n * n);
    for (int i = 0; i < n; ++i) for (int j = 0; j <= i; ++j) a[(size_t)i * n + j] = a[(size_t)j * n + i] = nd(rng);
    // the tridiagonal, by the fast two-stage reduction
    std::vector<float> d0(n), e0(n), tau(n), hq(1), wq(1);
    char vn = 'N', ul = 'L'; L N = n, info = 0, lh = -1, lw = -1;
    ssytrd_2stage_(&vn, &ul, &N, a.data(), &N, d0.data(), e0.data(), tau.data(), hq.data(), &lh, wq.data(), &lw, &info);
    std::vector<float> h((size_t)hq[0] + 1), w((size_t)wq[0] + 1); lh = (L)h.size(); lw = (L)w.size();
    ssytrd_2stage_(&vn, &ul, &N, a.data(), &N, d0.data(), e0.data(), tau.data(), h.data(), &lh, w.data(), &lw, &info);
    const int reps = n >= 8192 ? 1 : 2;
    // sstedc
    std::vector<float> d, e, zdc((size_t)n * n), wdc;
    char ci = 'I'; L lwz = -1, liw = -1, iq = 0; float zq = 0;
    d = d0; e = e0;
    sstedc_(&ci, &N, d.data(), e.data(), zdc.data(), &N, &zq, &lwz, &iq, &liw, &info);
    std::vector<float> wz((size_t)zq + 1); std::vector<L> iw(iq + 1); lwz = (L)wz.size(); liw = (L)iw.size();
    double t_dc = tmin([&] { d = d0; e = e0; sstedc_(&ci, &N, d.data(), e.data(), zdc.data(), &N, wz.data(), &lwz, iw.data(), &liw, &info); }, reps);
    wdc = d;
    // sstemr, all eigenpairs
    std::vector<float> zmr((size_t)n * n), wmr(n); std::vector<L> isuppz(2 * n);
    char jv = 'V', ra = 'A'; float vl = 0, vu = 0; L il = 0, iu = 0, m = 0, nzc = n; __LAPACK_bool tryrac = 1;
    L lwm = -1, liwm = -1, iqm = 0; float wqm = 0;
    d = d0; e = e0; e.push_back(0);
    sstemr_(&jv, &ra, &N, d.data(), e.data(), &vl, &vu, &il, &iu, &m, wmr.data(), zmr.data(), &N, &nzc, isuppz.data(), &tryrac, &wqm, &lwm, &iqm, &liwm, &info);
    std::vector<float> wm((size_t)wqm + 1); std::vector<L> iwm(iqm + 1); lwm = (L)wm.size(); liwm = (L)iwm.size();
    double t_mr = tmin([&] { d = d0; e = e0; e.push_back(0); tryrac = 1;
      sstemr_(&jv, &ra, &N, d.data(), e.data(), &vl, &vu, &il, &iu, &m, wmr.data(), zmr.data(), &N, &nzc, isuppz.data(), &tryrac, wm.data(), &lwm, iwm.data(), &liwm, &info); }, reps);
    if (info) std::printf("sstemr info %lld\n", (long long)info);
    double dw = 0, scale = 0; for (int i = 0; i < n; ++i) { dw = std::max(dw, (double)std::fabs(wdc[i] - wmr[i])); scale = std::max(scale, (double)std::fabs(wdc[i])); }
    std::printf("%6d | %9.1f %9.1f | %9.1e %9.1e | %9.1e\n", n, t_dc, t_mr, orth(zdc, n), orth(zmr, n), dw / scale);
    std::fflush(stdout);
  }
}
