// Checks for docs/proposals/two-stage-vectors.md: the chase's recorded
// reflectors (chase_record.cpp) give Q2^T A P2 = the bidiagonal; and the
// grouped order q2.metal applies them in gives the same Q2 as one at a time,
// bit for bit.   check_q2 <n> <ib>
#ifndef ACCELERATE_NEW_LAPACK
#define ACCELERATE_NEW_LAPACK
#endif
#include <Accelerate/Accelerate.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
namespace rec { extern float* L; extern float* Lt; extern float* R; extern float* Rt; extern long J, NB; }
namespace metal_linalg::detail {
void band_to_bidiagonal(uint32_t n, uint32_t nb, float* W, size_t ld, size_t ku, float* d, float* e, unsigned threads);
}
using L_ = __LAPACK_int;
// X (n x m, column-major, ld n) <- H X for H = I - tau u u^T on rows [r0, r0 + len)
static void refl_left(float* X, long n, long m, long r0, long len, const float* u, float tau) {
    if (tau == 0.0f) return;
    for (long c = 0; c < m; ++c) {
        float* x = X + c * n + r0;
        double w = 0; for (long i = 0; i < len; ++i) w += (double)u[i] * x[i];
        const float t = (float)(w * tau);
        for (long i = 0; i < len; ++i) x[i] -= t * u[i];
    }
}
// rows of reflector (s, j): [s + 1 + j nb, min(s + (j + 1) nb, n - 1)]
static long rlen(long n, long nb, long s, long j) { long a = s + 1 + j * nb, b = std::min(s + (j + 1) * nb, n - 1); return b >= a ? b - a + 1 : 0; }
int main(int argc, char** argv) {
    const long n = argc > 1 ? atol(argv[1]) : 300, nb = 16, ib = argc > 2 ? atol(argv[2]) : 16;
    std::mt19937 rng(5); std::normal_distribution<float> g;
    const long ld = 3 * nb + 1, ku = 2 * nb;
    std::vector<float> band(ld * n, 0.0f), A0(n * n, 0.0f);
    for (long j = 0; j < n; ++j) for (long i = std::max(0L, j - nb); i <= j; ++i) { float x = g(rng); band[j * ld + ku + i - j] = x; A0[j * n + i] = x; }
    const long J = (n + nb - 1) / nb + 2;
    std::vector<float> Lv(n * J * nb, 0.0f), Lt(n * J, 0.0f), Rv(n * J * nb, 0.0f), Rt(n * J, 0.0f);
    rec::L = Lv.data(); rec::Lt = Lt.data(); rec::R = Rv.data(); rec::Rt = Rt.data(); rec::J = J; rec::NB = nb;
    std::vector<float> d(n), e(n);
    metal_linalg::detail::band_to_bidiagonal(n, nb, band.data(), ld, ku, d.data(), e.data(), 8);
    // Q2 = U_1 ... U_last: apply to I in reverse sweep order; P2 likewise
    std::vector<float> Q(n * n, 0.0f), P(n * n, 0.0f);
    for (long i = 0; i < n; ++i) Q[i * n + i] = P[i * n + i] = 1.0f;
    for (long s = n - 2; s >= 0; --s)
        for (long j = 0; j < J; ++j) {
            const long len = rlen(n, nb, s, j); if (!len) continue;
            refl_left(Q.data(), n, n, s + 1 + j * nb, len, &Lv[(s * J + j) * nb], Lt[s * J + j]);
            refl_left(P.data(), n, n, s + 1 + j * nb, len, &Rv[(s * J + j) * nb], Rt[s * J + j]);
        }
    // B = Q^T A0 P vs bidiag(d, e)
    std::vector<float> T(n * n), B(n * n);
    cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, n, n, n, 1, Q.data(), n, A0.data(), n, 0, T.data(), n);
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1, T.data(), n, P.data(), n, 0, B.data(), n);
    double err = 0, nrm = 0;
    for (long c = 0; c < n; ++c) for (long r = 0; r < n; ++r) {
        float want = r == c ? d[r] : (c == r + 1 ? e[r] : 0.0f);
        err = std::max(err, (double)std::fabs(B[c * n + r] - want)); nrm = std::max(nrm, (double)std::fabs(A0[c * n + r]));
    }
    printf("n=%ld: max |Q2^T A P2 - bidiag| = %.2e (max |A| %.2f)\n", n, err, nrm);
    // Grouped order: groups G of ib sweeps from last to first; within a group,
    // steps j ascending; within a block, sweeps from last to first.
    std::vector<float> Qb(n * n, 0.0f);
    for (long i = 0; i < n; ++i) Qb[i * n + i] = 1.0f;
    const long ngroups = (n - 1 + ib - 1) / ib;
    for (long G = ngroups - 1; G >= 0; --G)
        for (long j = 0; j < J; ++j)
            for (long s = std::min(n - 2, G * ib + ib - 1); s >= G * ib; --s) {
                const long len = rlen(n, nb, s, j); if (!len) continue;
                refl_left(Qb.data(), n, n, s + 1 + j * nb, len, &Lv[(s * J + j) * nb], Lt[s * J + j]);
            }
    double diff = 0; for (long i = 0; i < n * n; ++i) diff = std::max(diff, (double)std::fabs(Qb[i] - Q[i]));
    printf("grouped (ib=%ld) vs one at a time: max diff %.2e\n", ib, diff);
}
