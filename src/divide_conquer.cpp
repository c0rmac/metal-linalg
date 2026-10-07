// Divide and conquer for the symmetric tridiagonal eigenproblem and the
// bidiagonal SVD on the CPU's cores: what LAPACK's sstedc and sbdsdc do, with
// the work spread over threads. The tridiag and bidiag backends' step 2 with
// vectors (eigh_tridiag.mm, svd_bidiag.mm).
//
// Both split the problem in two, solve the halves, and merge: the merged
// problem is the halves' solutions plus a rank-one correction, whose
// eigenvalues (or singular values) are the roots of a secular equation and
// whose vectors follow from them by Gu and Eisenstat's formulas; the halves'
// vectors are then multiplied by the correction's. Recursively, the problem
// is a tree of merges over leaves of at most 25 rows. LAPACK walks that tree
// on one core: on an M5 Pro sstedc takes 177 ms on a 4096 x 4096 tridiagonal
// from ssytrd, sbdsdc 718 ms on a 4096 bidiagonal from sgebrd, 1.1-1.2 cores'
// worth of CPU time.
//
// Here the same tree, with LAPACK's own routines for everything numerical:
//
//   leaves     ssteqr or slasdq, one per task, all at once
//   merges     below kOwnMergeMinN rows, LAPACK's slaed1 or slasd1, one per
//              task where a level has many; from it, slaed1's and slasd1's
//              steps here, their loops spread over the threads: LAPACK's
//              deflation (slaed2; slasd2's, rewritten to move VT's rows a
//              column at a time, below), the secular equation's roots
//              (slaed4, slasd4, a call per root), the corrected z (a row of
//              the roots' differences per entry), the vectors (a column per
//              root), and the products with the halves' vectors (by blocks of
//              columns, Accelerate's sgemm each)
//
// The eigenvalues and singular values are bit for bit LAPACK's; the vectors
// differ from LAPACK's in the last bits, where the products' sums are blocked
// differently. Neither depends on the number of threads. The threads are
// GCD's, each with Accelerate's own threading off (blas_threading.h).
//
// On an M5 Pro, 16 threads, the tridiagonal of a 4096 x 4096 Gaussian matrix
// in 50 ms (sstedc 176), its bidiagonal in 100 ms (sbdsdc 753); at 2048, 10
// and 22 ms (43 and 133). The top merges are then bound by their matrix
// products, which the CPU's matrix units run at about 2 TFLOP/s whatever the
// split.
//
// Below kSerialMaxN, and with one thread, LAPACK's own solvers are called.

#ifndef ACCELERATE_NEW_LAPACK
#define ACCELERATE_NEW_LAPACK
#endif
#include <Accelerate/Accelerate.h>

#include "blas_threading.h"
#include "divide_conquer.h"

#include <dispatch/dispatch.h>

#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <new>
#include <numeric>
#include <vector>

namespace metal_linalg::detail {
namespace {

using L = __LAPACK_int;

constexpr L        kLeaf       = 25;    // LAPACK's SMLSIZ: the largest subproblem not divided further
constexpr uint32_t kSerialMaxN = 128;   // up to this order, LAPACK's own solver
constexpr L        kRowBlock   = 64;    // rows a task in the row loops
constexpr L        kColBlock   = 16;    // columns (roots) a task in the column loops
constexpr L        kGemmCols   = 128;   // columns of a product a task

// Merges of at least this order are this file's (merge_symmetric,
// merge_bidiagonal), smaller ones LAPACK's (slaed1, slasd1). By order alone,
// so that the results do not depend on the number of threads: the two
// differ in the last bits, where the matrix products' sums are blocked
// differently.
constexpr L kOwnMergeMinN = 256;

// Merges per level from which each merge is one task; below, the merges run
// one after another, each over every thread.
unsigned concurrent_merges_min(unsigned threads) { return std::max(2u, threads / 2); }

// Runs f(t) for t in [0, tasks) on up to `workers` threads, each with
// Accelerate's threading off; the tasks are taken in order, so that the
// faster cores take more of them.
template <class F>
void run(unsigned workers, size_t tasks, const F& f) {
    if (tasks == 0) return;
    struct Ctx {
        const F*            f;
        std::atomic<size_t> next{0};
        size_t              tasks;
    } ctx;
    ctx.f = &f;
    ctx.tasks = tasks;
    auto body = [](void* p, size_t) {
        Ctx& c = *static_cast<Ctx*>(p);
        SingleThreadedBlas single;
        for (size_t t; (t = c.next.fetch_add(1, std::memory_order_relaxed)) < c.tasks;) (*c.f)(t);
    };
    const size_t w = std::min<size_t>(std::max(1u, workers), tasks);
    if (w == 1) body(&ctx, 0);
    else dispatch_apply_f(w, DISPATCH_APPLY_AUTO, &ctx, body);
}

size_t blocks(L n, L per) { return n <= 0 ? 0 : (size_t)((n + per - 1) / per); }

// The GPU's products for this thread's solve, if the caller gave them
// (GpuGemm): the large merges run on the thread that called the solve,
// the small ones, which stay on the CPU, on the workers.
thread_local GpuGemm* tl_gpu = nullptr;
constexpr double kGpuMinFlops = 1e9;   // a product's 2 m n k, at least, for the GPU

// Page-aligned, zeroed floats, which tl_gpu's products can use in place.
class Pages {
public:
    explicit Pages(size_t floats) {
        const size_t page = (size_t)getpagesize(), bytes = std::max(page, (floats * 4 + page - 1) / page * page);
        void* p = nullptr;
        if (posix_memalign(&p, page, bytes) != 0) throw std::bad_alloc();
        p_ = static_cast<float*>(p);
        std::memset(p_, 0, bytes);
        if ((gpu_ = tl_gpu)) gpu_->add(p_, bytes / 4);
    }
    ~Pages() {
        if (gpu_) gpu_->remove(p_);
        std::free(p_);
    }
    Pages(const Pages&) = delete;
    Pages& operator=(const Pages&) = delete;
    float* data() { return p_; }
private:
    float* p_ = nullptr;
    GpuGemm* gpu_ = nullptr;
};

bool on_gpu(L m, L n, L k, const float* A, L lda, const float* B, L ldb, float* C, L ldc, bool accumulate) {
    return tl_gpu && 2.0 * m * n * k >= kGpuMinFlops && tl_gpu->gemm(m, n, k, A, lda, B, ldb, C, ldc, accumulate);
}

// C (m x n) = A (m x k) B (k x n), column-major, by blocks of C's columns
// (or on the GPU).
void gemm(unsigned workers, L m, L n, L k, const float* A, L lda, const float* B, L ldb, float* C, L ldc) {
    if (m <= 0 || n <= 0) return;
    if (k <= 0) {
        for (L j = 0; j < n; ++j) std::fill(C + (size_t)j * ldc, C + (size_t)j * ldc + m, 0.0f);
        return;
    }
    if (on_gpu(m, n, k, A, lda, B, ldb, C, ldc, false)) return;
    run(workers, blocks(n, kGemmCols), [&](size_t t) {
        const L j0 = (L)t * kGemmCols, nj = std::min(kGemmCols, n - j0);
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, (int)m, (int)nj, (int)k, 1.0f, A, (int)lda,
                    B + (size_t)j0 * ldb, (int)ldb, 0.0f, C + (size_t)j0 * ldc, (int)ldc);
    });
}

// dst (m x n, ld ldd) = src (ld lds), by blocks of columns.
void copy(unsigned workers, L m, L n, const float* src, L lds, float* dst, L ldd) {
    run(workers, blocks(n, kGemmCols), [&](size_t t) {
        const L j0 = (L)t * kGemmCols, j1 = std::min(n, j0 + kGemmCols);
        for (L j = j0; j < j1; ++j) std::copy(src + (size_t)j * lds, src + (size_t)j * lds + m, dst + (size_t)j * ldd);
    });
}

// The n x n identity, column-major.
void identity(unsigned workers, L n, float* a, L lda) {
    run(workers, blocks(n, kGemmCols), [&](size_t t) {
        const L j0 = (L)t * kGemmCols, j1 = std::min(n, j0 + kGemmCols);
        for (L j = j0; j < j1; ++j) {
            std::fill(a + (size_t)j * lda, a + (size_t)j * lda + n, 0.0f);
            a[(size_t)j * lda + j] = 1.0f;
        }
    });
}

// Records the first nonzero LAPACK info from any thread.
struct Info {
    std::atomic<L> v{0};
    void set(L x) {
        L z = 0;
        if (x) v.compare_exchange_strong(z, x);
    }
    L get() const { return v.load(); }
};

// ---------------------------------------------------------------------------
// The symmetric tridiagonal (sstedc, slaed0, slaed1, slaed3)
// ---------------------------------------------------------------------------

// slaed3, its loops over the threads: the K eigenvalues of the deflated
// rank-one problem into d, its eigenvectors, and Q's first K columns, the
// halves' eigenvectors times those.
L secular_symmetric(unsigned workers, L k, L n, L n1, float* d, float* q, L ldq, float rho, const float* dlamda,
                    const float* q2, const L* indx, const L* ctot, float* w, float* s) {
    if (k == 0) return 0;
    Info info;
    run(workers, blocks(k, kColBlock), [&](size_t t) {   // the roots; column j of Q gets dlamda - lambda_j
        const L j0 = (L)t * kColBlock, j1 = std::min(k, j0 + kColBlock);
        for (L j = j0; j < j1; ++j) {
            L J = j + 1, i = 0;
            slaed4_(&k, &J, dlamda, w, q + (size_t)j * ldq, &rho, d + j, &i);
            info.set(i);
        }
    });
    if (info.get()) return info.get();
    if (k == 2) {
        for (L j = 0; j < 2; ++j) {
            float* qj = q + (size_t)j * ldq;
            const float t0 = qj[0], t1 = qj[1];
            qj[0] = indx[0] == 1 ? t0 : t1;
            qj[1] = indx[1] == 1 ? t0 : t1;
        }
    } else if (k > 2) {
        // w_i = sign(z_i) sqrt(-(d_i - lambda_i) prod_{j != i} (d_i - lambda_j) / (d_i - d_j)),
        // each row's product in LAPACK's order (j ascending)
        std::vector<float> z(w, w + k);
        run(workers, blocks(k, kRowBlock), [&](size_t t) {
            const L i0 = (L)t * kRowBlock, i1 = std::min(k, i0 + kRowBlock);
            for (L i = i0; i < i1; ++i) w[i] = q[(size_t)i * ldq + i];
            for (L j = 0; j < k; ++j) {
                const float* qj = q + (size_t)j * ldq;
                const float dj = dlamda[j];
                for (L i = i0; i < i1; ++i)
                    if (i != j) w[i] *= qj[i] / (dlamda[i] - dj);
            }
            for (L i = i0; i < i1; ++i) w[i] = std::copysign(std::sqrt(-w[i]), z[i]);
        });
        // The eigenvectors, a column a root, normalised and permuted back
        run(workers, blocks(k, kColBlock), [&](size_t t) {
            const L j0 = (L)t * kColBlock, j1 = std::min(k, j0 + kColBlock);
            std::vector<float> v(k);
            for (L j = j0; j < j1; ++j) {
                float* qj = q + (size_t)j * ldq;
                for (L i = 0; i < k; ++i) v[i] = w[i] / qj[i];
                const float nrm = cblas_snrm2((int)k, v.data(), 1);
                for (L i = 0; i < k; ++i) qj[i] = v[indx[i] - 1] / nrm;
            }
        });
    }
    // Q(:, 0:K) = [Q2_1 0; 0 Q2_2] times them, Q2's columns by type
    const L n2 = n - n1, n12 = ctot[0] + ctot[1], n23 = ctot[1] + ctot[2];
    copy(workers, n23, k, q + ctot[0], ldq, s, std::max<L>(1, n23));
    gemm(workers, n2, k, n23, q2 + (size_t)n1 * n12, std::max<L>(1, n2), s, std::max<L>(1, n23), q + n1, ldq);
    copy(workers, n12, k, q, ldq, s, std::max<L>(1, n12));
    gemm(workers, n1, k, n12, q2, std::max<L>(1, n1), s, std::max<L>(1, n12), q, ldq);
    return 0;
}

// slaed1 with slaed3's work over the threads: merges the eigensystems of the
// two halves, order n1 and n - n1, in d and Q, into that of the whole.
L merge_symmetric(unsigned workers, L n, float* d, float* q, L ldq, L* indxq, float rho, L n1) {
    std::vector<float> z(n), dlamda(n), w(n);
    Pages q2((size_t)n * n);   // slaed2 puts the deflated vectors there too
    std::vector<L> indx(n), indxc(n), indxp(n), coltyp(n);
    for (L j = 0; j < n1; ++j) z[j] = q[(size_t)j * ldq + n1 - 1];   // Q1's last row
    for (L j = n1; j < n; ++j) z[j] = q[(size_t)j * ldq + n1];       // Q2's first
    L k = 0, info = 0;
    slaed2_(&k, &n, &n1, d, q, &ldq, indxq, &rho, z.data(), dlamda.data(), w.data(), q2.data(), indx.data(),
            indxc.data(), indxp.data(), coltyp.data(), &info);
    if (info) return info;
    if (k == 0) {
        for (L i = 0; i < n; ++i) indxq[i] = i + 1;
        return 0;
    }
    const L* ctot = coltyp.data();   // slaed2 leaves the column counts by type there
    Pages s((size_t)std::max<L>(1, std::max(ctot[0] + ctot[1], ctot[1] + ctot[2])) * k);
    info = secular_symmetric(workers, k, n, n1, d, q, ldq, rho, dlamda.data(), q2.data(), indxc.data(), ctot,
                             w.data(), s.data());
    if (info) return info;
    L rest = n - k, one = 1, back = -1;
    slamrg_(&k, &rest, d, &one, &back, indxq);
    return 0;
}

// slaed0 (COMPQ = 2) on the threads: the eigensystem of one unreduced
// tridiagonal of order n > kLeaf into d and Q (the identity on entry).
L dc_tridiagonal(unsigned workers, L n, float* d, float* e, float* q, L ldq) {
    // Halve until every piece has at most kLeaf rows, as slaed0 does.
    std::vector<L> size{n};
    while (size.back() > kLeaf) {
        std::vector<L> half(2 * size.size());
        for (size_t j = 0; j < size.size(); ++j) {
            half[2 * j] = size[j] / 2;
            half[2 * j + 1] = (size[j] + 1) / 2;
        }
        size.swap(half);
    }
    std::vector<L> start(size.size(), 0);
    for (size_t j = 1; j < size.size(); ++j) start[j] = start[j - 1] + size[j - 1];
    // Cuppen's tearing: T = diag(T_1, ..., T_p) + the rank-one cuts
    for (size_t j = 1; j < size.size(); ++j) {
        const L c = start[j];
        d[c - 1] -= std::fabs(e[c - 1]);
        d[c] -= std::fabs(e[c - 1]);
    }
    std::vector<L> indxq(n);
    Info info;
    run(workers, size.size(), [&](size_t j) {
        L m = size[j], s0 = start[j], i = 0;
        std::vector<float> work(std::max<L>(1, 2 * m - 2));
        ssteqr_("I", &m, d + s0, e + s0, q + (size_t)s0 * ldq + s0, &ldq, work.data(), &i);
        info.set(i);
        for (L r = 0; r < m; ++r) indxq[s0 + r] = r + 1;
    });
    if (info.get()) return info.get();
    // The merges, a level of the tree at a time
    while (size.size() > 1) {
        const size_t pairs = size.size() / 2;
        std::vector<L> msize(pairs), mstart(pairs), cut(pairs);
        for (size_t p = 0; p < pairs; ++p) {
            msize[p] = size[2 * p] + size[2 * p + 1];
            mstart[p] = start[2 * p];
            cut[p] = size[2 * p];
        }
        auto merge = [&](size_t p, unsigned inner) {
            L m = msize[p], s0 = mstart[p], c = cut[p], i = 0;
            if (m >= kOwnMergeMinN)
                return merge_symmetric(inner, m, d + s0, q + (size_t)s0 * ldq + s0, ldq, indxq.data() + s0,
                                       e[s0 + c - 1], c);
            std::vector<float> work(4 * (size_t)m + (size_t)m * m);
            std::vector<L> iwork(4 * (size_t)m);
            slaed1_(&m, d + s0, q + (size_t)s0 * ldq + s0, &ldq, indxq.data() + s0, e + s0 + c - 1, &c, work.data(),
                    iwork.data(), &i);
            return i;
        };
        if (pairs >= concurrent_merges_min(workers)) {
            run(workers, pairs, [&](size_t p) { info.set(merge(p, 1)); });
            if (info.get()) return info.get();
        } else {
            for (size_t p = 0; p < pairs; ++p)
                if (const L i = merge(p, workers)) return i;
        }
        size.swap(msize);
        start.swap(mstart);
    }
    // The last merge's order: d(i) = d(indxq(i)), Q(:, i) = Q(:, indxq(i))
    bool in_order = true;
    for (L i = 0; i < n && in_order; ++i) in_order = indxq[i] == i + 1;
    if (!in_order) {
        std::vector<float> dd(n), qq((size_t)n * n);
        for (L i = 0; i < n; ++i) dd[i] = d[indxq[i] - 1];
        run(workers, blocks(n, kGemmCols), [&](size_t t) {
            const L j0 = (L)t * kGemmCols, j1 = std::min(n, j0 + kGemmCols);
            for (L j = j0; j < j1; ++j) {
                const float* src = q + (size_t)(indxq[j] - 1) * ldq;
                std::copy(src, src + n, qq.begin() + (size_t)j * n);
            }
        });
        std::copy(dd.begin(), dd.end(), d);
        copy(workers, n, n, qq.data(), n, q, ldq);
    }
    return 0;
}

// Sorts d into ascending (or descending) order, permuting the columns of U
// and, if given, the rows of VT alike.
void sort_with_vectors(unsigned workers, L n, float* d, float* u, L ldu, float* vt, L ldvt, bool descending) {
    bool sorted = true;
    for (L i = 1; i < n && sorted; ++i) sorted = descending ? d[i - 1] >= d[i] : d[i - 1] <= d[i];
    if (sorted) return;
    std::vector<L> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](L a, L b) { return descending ? d[a] > d[b] : d[a] < d[b]; });
    std::vector<float> dd(n), uu((size_t)n * n);
    for (L i = 0; i < n; ++i) dd[i] = d[order[i]];
    std::copy(dd.begin(), dd.end(), d);
    run(workers, blocks(n, kGemmCols), [&](size_t t) {
        const L j0 = (L)t * kGemmCols, j1 = std::min(n, j0 + kGemmCols);
        for (L j = j0; j < j1; ++j) {
            const float* src = u + (size_t)order[j] * ldu;
            std::copy(src, src + n, uu.begin() + (size_t)j * n);
        }
    });
    copy(workers, n, n, uu.data(), n, u, ldu);
    if (!vt) return;
    run(workers, blocks(n, kGemmCols), [&](size_t t) {   // VT's rows: a gather within each column
        const L j0 = (L)t * kGemmCols, j1 = std::min(n, j0 + kGemmCols);
        std::vector<float> col(n);
        for (L j = j0; j < j1; ++j) {
            float* c = vt + (size_t)j * ldvt;
            for (L i = 0; i < n; ++i) col[i] = c[order[i]];
            std::copy(col.begin(), col.end(), c);
        }
    });
}

// ---------------------------------------------------------------------------
// The bidiagonal (sbdsdc, slasd0, slasd1, slasd3)
// ---------------------------------------------------------------------------

// slasd3, its loops over the threads: the K singular values of the deflated
// rank-one problem into d, its singular vectors, and the first K columns of
// U and rows of VT, the halves' vectors times those. As in LAPACK, U2 and
// VT2 are the deflated halves' vectors (slasd2), Q is K x K workspace and
// ctot the column counts by type.
L secular_bidiagonal(unsigned workers, L nl, L nr, L sqre, L k, float* d, float* q, L ldq, float* dsigma, float* u,
                     L ldu, const float* u2, L ldu2, float* vt, L ldvt, float* vt2, L ldvt2, const L* idxc,
                     const L* ctot, float* z) {
    const L n = nl + nr + 1, m = n + sqre;
    if (k == 1) {
        d[0] = std::fabs(z[0]);
        for (L j = 0; j < m; ++j) vt[(size_t)j * ldvt] = vt2[(size_t)j * ldvt2];
        for (L i = 0; i < n; ++i) u[i] = z[0] > 0.0f ? u2[i] : -u2[i];
        return 0;
    }
    std::copy(z, z + k, q);   // z kept, in Q's first column
    float rho = cblas_snrm2((int)k, z, 1);
    {
        L info = 0, zero = 0, one = 1;
        float unit = 1.0f;
        slascl_("G", &zero, &zero, &rho, &unit, &k, &one, z, &k, &info);
    }
    rho = rho * rho;
    Info info;
    run(workers, blocks(k, kColBlock), [&](size_t t) {   // the roots; U's column j the differences, VT's the sums
        const L j0 = (L)t * kColBlock, j1 = std::min(k, j0 + kColBlock);
        for (L j = j0; j < j1; ++j) {
            L J = j + 1, i = 0;
            slasd4_(&k, &J, dsigma, z, u + (size_t)j * ldu, &rho, d + j, vt + (size_t)j * ldvt, &i);
            info.set(i);
        }
    });
    if (info.get()) return info.get();
    // The corrected z, each entry's product in LAPACK's order
    run(workers, blocks(k, kRowBlock), [&](size_t t) {
        const L i0 = (L)t * kRowBlock, i1 = std::min(k, i0 + kRowBlock);
        for (L i = i0; i < i1; ++i) z[i] = u[(size_t)(k - 1) * ldu + i] * vt[(size_t)(k - 1) * ldvt + i];
        for (L j = 0; j + 1 < k; ++j) {
            const float* uj = u + (size_t)j * ldu;
            const float* vj = vt + (size_t)j * ldvt;
            for (L i = i0; i < i1; ++i) {
                const float sj = j < i ? dsigma[j] : dsigma[j + 1];
                z[i] = z[i] * (uj[i] * vj[i] / (dsigma[i] - sj) / (dsigma[i] + sj));
            }
        }
        for (L i = i0; i < i1; ++i) z[i] = std::copysign(std::sqrt(std::fabs(z[i])), q[i]);
    });
    // The left singular vectors, a column a root, into Q; VT's columns keep
    // what the right ones need
    run(workers, blocks(k, kColBlock), [&](size_t t) {
        const L i0 = (L)t * kColBlock, i1 = std::min(k, i0 + kColBlock);
        for (L i = i0; i < i1; ++i) {
            float* ui = u + (size_t)i * ldu;
            float* vi = vt + (size_t)i * ldvt;
            vi[0] = z[0] / ui[0] / vi[0];
            ui[0] = -1.0f;
            for (L j = 1; j < k; ++j) {
                vi[j] = z[j] / ui[j] / vi[j];
                ui[j] = dsigma[j] * vi[j];
            }
            const float temp = cblas_snrm2((int)k, ui, 1);
            float* qi = q + (size_t)i * ldq;
            qi[0] = ui[0] / temp;
            for (L j = 1; j < k; ++j) qi[j] = ui[idxc[j] - 1] / temp;
        }
    });
    // U = U2 Q, by blocks of U2's columns as their types allow
    auto gemm_acc = [&](L rows, L cols, L inner, const float* A, L lda, const float* B, L ldb, float* C, L ldc,
                        bool accumulate) {
        if (!accumulate) {
            gemm(workers, rows, cols, inner, A, lda, B, ldb, C, ldc);
            return;
        }
        if (rows <= 0 || cols <= 0 || inner <= 0) return;
        if (on_gpu(rows, cols, inner, A, lda, B, ldb, C, ldc, true)) return;
        run(workers, blocks(cols, kGemmCols), [&](size_t t) {
            const L j0 = (L)t * kGemmCols, nj = std::min(kGemmCols, cols - j0);
            cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, (int)rows, (int)nj, (int)inner, 1.0f, A, (int)lda,
                        B + (size_t)j0 * ldb, (int)ldb, 1.0f, C + (size_t)j0 * ldc, (int)ldc);
        });
    };
    if (k == 2) {
        gemm(workers, n, k, k, u2, ldu2, q, ldq, u, ldu);
    } else {
        const L kt3 = 1 + ctot[0] + ctot[1];   // the first column of type 3 (0-based)
        if (ctot[0] > 0) {
            gemm_acc(nl, k, ctot[0], u2 + ldu2, ldu2, q + 1, ldq, u, ldu, false);
            if (ctot[2] > 0)
                gemm_acc(nl, k, ctot[2], u2 + (size_t)kt3 * ldu2, ldu2, q + kt3, ldq, u, ldu, true);
        } else if (ctot[2] > 0) {
            gemm_acc(nl, k, ctot[2], u2 + (size_t)kt3 * ldu2, ldu2, q + kt3, ldq, u, ldu, false);
        } else {
            copy(workers, nl, k, u2, ldu2, u, ldu);
        }
        for (L j = 0; j < k; ++j) u[(size_t)j * ldu + nl] = q[(size_t)j * ldq];   // Q's first row
        const L kt = 1 + ctot[0], ct = ctot[1] + ctot[2];
        gemm_acc(nr, k, ct, u2 + (size_t)kt * ldu2 + nl + 1, ldu2, q + kt, ldq, u + nl + 1, ldu, false);
    }
    // The right singular vectors, a row of Q a root
    run(workers, blocks(k, kColBlock), [&](size_t t) {
        const L i0 = (L)t * kColBlock, i1 = std::min(k, i0 + kColBlock);
        for (L i = i0; i < i1; ++i) {
            const float* vi = vt + (size_t)i * ldvt;
            const float temp = cblas_snrm2((int)k, vi, 1);
            q[i] = vi[0] / temp;
            for (L j = 1; j < k; ++j) q[(size_t)j * ldq + i] = vi[idxc[j] - 1] / temp;
        }
    });
    // VT = Q VT2, likewise
    if (k == 2) {
        gemm(workers, k, m, k, q, ldq, vt2, ldvt2, vt, ldvt);
        return 0;
    }
    const L nlp1 = nl + 1;
    gemm_acc(k, nlp1, 1 + ctot[0], q, ldq, vt2, ldvt2, vt, ldvt, false);
    const L kt3 = 1 + ctot[0] + ctot[1];   // 0-based; LAPACK's KTEMP = 2 + CTOT(1) + CTOT(2)
    if (kt3 + 1 <= ldvt2)
        gemm_acc(k, nlp1, ctot[2], q + (size_t)kt3 * ldq, ldq, vt2 + kt3, ldvt2, vt, ldvt, true);
    const L kt = ctot[0];   // 0-based; LAPACK's KTEMP = CTOT(1) + 1
    const L nrp1 = nr + sqre;
    if (kt > 0) {
        std::copy(q, q + k, q + (size_t)kt * ldq);
        for (L i = nlp1; i < m; ++i) vt2[(size_t)i * ldvt2 + kt] = vt2[(size_t)i * ldvt2];
    }
    const L ct = 1 + ctot[1] + ctot[2];
    gemm_acc(k, nrp1, ct, q + (size_t)kt * ldq, ldq, vt2 + (size_t)nlp1 * ldvt2 + kt, ldvt2,
             vt + (size_t)nlp1 * ldvt, ldvt, false);
    return 0;
}

// LAPACK's slasd2, the deflation, with VT's rows moved a column at a time.
// slasd2 copies and rotates VT's rows one row at a time, a stride of ldvt
// between their entries, which in a 4096 merge took 130 ms of its 190; here
// the rotations are recorded during the scan and applied afterwards, and the
// rows gathered, a column of VT (and a block of U's rows) a task. The
// rotations' angles depend on d and z alone, so applying them later, in the
// same order, gives the same result. The loops are LAPACK's, 1-based.
L deflate_bidiagonal(unsigned workers, L nl, L nr, L sqre, L& k, float* d_, float* z_, float alpha, float beta,
                     float* u_, L ldu, float* vt_, L ldvt, float* dsigma_, float* u2_, L ldu2, float* vt2_, L ldvt2,
                     L* idxp_, L* idx_, L* idxc_, L* idxq_, L* coltyp_) {
    auto D = [&](L i) -> float& { return d_[i - 1]; };
    auto Z = [&](L i) -> float& { return z_[i - 1]; };
    auto DSIGMA = [&](L i) -> float& { return dsigma_[i - 1]; };
    auto U = [&](L i, L j) -> float& { return u_[(size_t)(j - 1) * ldu + i - 1]; };
    auto VT = [&](L i, L j) -> float& { return vt_[(size_t)(j - 1) * ldvt + i - 1]; };
    auto U2 = [&](L i, L j) -> float& { return u2_[(size_t)(j - 1) * ldu2 + i - 1]; };
    auto VT2 = [&](L i, L j) -> float& { return vt2_[(size_t)(j - 1) * ldvt2 + i - 1]; };
    auto IDXP = [&](L i) -> L& { return idxp_[i - 1]; };
    auto IDX = [&](L i) -> L& { return idx_[i - 1]; };
    auto IDXC = [&](L i) -> L& { return idxc_[i - 1]; };
    auto IDXQ = [&](L i) -> L& { return idxq_[i - 1]; };
    auto COLTYP = [&](L i) -> L& { return coltyp_[i - 1]; };

    const L n = nl + nr + 1, m = n + sqre, nlp1 = nl + 1, nlp2 = nl + 2;
    // z, the joining row times the halves' right vectors; d moved one place on
    const float z1 = alpha * VT(nlp1, nlp1);
    Z(1) = z1;
    for (L i = nl; i >= 1; --i) {
        Z(i + 1) = alpha * VT(i, nlp1);
        D(i + 1) = D(i);
        IDXQ(i + 1) = IDXQ(i) + 1;
    }
    for (L i = nlp2; i <= m; ++i) Z(i) = beta * VT(i, nlp2);
    for (L i = 2; i <= nlp1; ++i) COLTYP(i) = 1;
    for (L i = nlp2; i <= n; ++i) COLTYP(i) = 2;
    for (L i = nlp2; i <= n; ++i) IDXQ(i) = IDXQ(i) + nlp1;
    // Sorted into increasing order, DSIGMA, IDXC and U2's first column as storage
    for (L i = 2; i <= n; ++i) {
        DSIGMA(i) = D(IDXQ(i));
        U2(i, 1) = Z(IDXQ(i));
        IDXC(i) = COLTYP(IDXQ(i));
    }
    L one = 1;
    slamrg_(&nl, &nr, &DSIGMA(2), &one, &one, &IDX(2));
    for (L i = 2; i <= n; ++i) {
        const L idxi = 1 + IDX(i);
        D(i) = DSIGMA(idxi);
        Z(i) = U2(idxi, 1);
        COLTYP(i) = IDXC(idxi);
    }
    // The deflation: small z entries, and close values rotated together
    const float eps = slamch_("Epsilon");
    float tol = std::max(std::fabs(alpha), std::fabs(beta));
    tol = 8.0f * eps * std::max(std::fabs(D(n)), tol);
    struct Rot { L a, b; float c, s; };
    std::vector<Rot> rots;
    k = 1;
    L k2 = n + 1, jprev = 0;
    bool all_small = true;
    for (L j = 2; j <= n; ++j) {
        if (std::fabs(Z(j)) <= tol) {
            k2 = k2 - 1;
            IDXP(k2) = j;
            COLTYP(j) = 4;
        } else {
            jprev = j;
            all_small = false;
            break;
        }
    }
    if (!all_small) {
        for (L j = jprev + 1; j <= n; ++j) {
            if (std::fabs(Z(j)) <= tol) {
                k2 = k2 - 1;
                IDXP(k2) = j;
                COLTYP(j) = 4;
            } else if (std::fabs(D(j) - D(jprev)) <= tol) {
                float s = Z(jprev), c = Z(j);
                const float tau = slapy2_(&c, &s);
                c = c / tau;
                s = -s / tau;
                Z(j) = tau;
                Z(jprev) = 0.0f;
                L idxjp = IDXQ(IDX(jprev) + 1), idxj = IDXQ(IDX(j) + 1);
                if (idxjp <= nlp1) idxjp = idxjp - 1;
                if (idxj <= nlp1) idxj = idxj - 1;
                rots.push_back({idxjp, idxj, c, s});
                if (COLTYP(j) != COLTYP(jprev)) COLTYP(j) = 3;
                COLTYP(jprev) = 4;
                k2 = k2 - 1;
                IDXP(k2) = jprev;
                jprev = j;
            } else {
                k = k + 1;
                U2(k, 1) = Z(jprev);
                DSIGMA(k) = D(jprev);
                IDXP(k) = jprev;
                jprev = j;
            }
        }
        k = k + 1;
        U2(k, 1) = Z(jprev);
        DSIGMA(k) = D(jprev);
        IDXP(k) = jprev;
    }
    // The rotations (srot's arithmetic): on U's column pairs a block of rows
    // a task, on VT's row pairs a column a task
    if (!rots.empty()) {
        run(workers, blocks(n, kRowBlock), [&](size_t t) {
            const L i0 = (L)t * kRowBlock + 1, i1 = std::min(n, i0 + kRowBlock - 1);
            for (const Rot& r : rots)
                for (L i = i0; i <= i1; ++i) {
                    const float x = U(i, r.a), y = U(i, r.b);
                    U(i, r.a) = r.c * x + r.s * y;
                    U(i, r.b) = r.c * y - r.s * x;
                }
        });
        run(workers, blocks(m, kColBlock), [&](size_t t) {
            const L c0 = (L)t * kColBlock + 1, c1 = std::min(m, c0 + kColBlock - 1);
            for (L c = c0; c <= c1; ++c)
                for (const Rot& r : rots) {
                    const float x = VT(r.a, c), y = VT(r.b, c);
                    VT(r.a, c) = r.c * x + r.s * y;
                    VT(r.b, c) = r.c * y - r.s * x;
                }
        });
    }
    // The column types' counts, and IDXC, which groups the columns by type
    L ctot[4] = {0, 0, 0, 0};
    for (L j = 2; j <= n; ++j) ctot[COLTYP(j) - 1] += 1;
    L psm[4] = {2, 2 + ctot[0], 2 + ctot[0] + ctot[1], 2 + ctot[0] + ctot[1] + ctot[2]};
    for (L j = 2; j <= n; ++j) {
        const L jp = IDXP(j), ct = COLTYP(jp);
        IDXC(psm[ct - 1]) = j;
        psm[ct - 1] = psm[ct - 1] + 1;
    }
    // DSIGMA, U2 and VT2 in that order: U's columns, a column a task; VT's
    // rows, gathered a column of VT a task
    std::vector<L> src(n + 1, 0);
    for (L j = 2; j <= n; ++j) {
        DSIGMA(j) = D(IDXP(j));
        L idxj = IDXQ(IDX(IDXP(IDXC(j))) + 1);
        if (idxj <= nlp1) idxj = idxj - 1;
        src[j] = idxj;
    }
    run(workers, blocks(n - 1, kColBlock), [&](size_t t) {
        const L j0 = (L)t * kColBlock + 2, j1 = std::min(n, j0 + kColBlock - 1);
        for (L j = j0; j <= j1; ++j) std::copy(&U(1, src[j]), &U(1, src[j]) + n, &U2(1, j));
    });
    run(workers, blocks(m, kColBlock), [&](size_t t) {
        const L c0 = (L)t * kColBlock + 1, c1 = std::min(m, c0 + kColBlock - 1);
        for (L c = c0; c <= c1; ++c)
            for (L j = 2; j <= n; ++j) VT2(j, c) = VT(src[j], c);
    });
    // DSIGMA(1), DSIGMA(2), Z(1)
    DSIGMA(1) = 0.0f;
    const float hlftol = tol / 2.0f;
    if (std::fabs(DSIGMA(2)) <= hlftol) DSIGMA(2) = hlftol;
    float c = 1.0f, s = 0.0f;
    if (m > n) {
        float zm = Z(m), zz1 = z1;
        Z(1) = slapy2_(&zz1, &zm);
        if (Z(1) <= tol) {
            c = 1.0f;
            s = 0.0f;
            Z(1) = tol;
        } else {
            c = z1 / Z(1);
            s = Z(m) / Z(1);
        }
    } else {
        Z(1) = std::fabs(z1) <= tol ? tol : z1;
    }
    for (L i = 2; i <= k; ++i) Z(i) = U2(i, 1);
    // U2's first column, VT2's first row, VT's last
    for (L i = 1; i <= n; ++i) U2(i, 1) = 0.0f;
    U2(nlp1, 1) = 1.0f;
    if (m > n) {
        for (L i = 1; i <= nlp1; ++i) {
            VT(m, i) = -s * VT(nlp1, i);
            VT2(1, i) = c * VT(nlp1, i);
        }
        for (L i = nlp2; i <= m; ++i) {
            VT2(1, i) = s * VT(m, i);
            VT(m, i) = c * VT(m, i);
        }
    } else {
        for (L i = 1; i <= m; ++i) VT2(1, i) = VT(nlp1, i);
    }
    if (m > n)
        for (L i = 1; i <= m; ++i) VT2(m, i) = VT(m, i);
    // The deflated values and vectors to the back of D, U and VT
    if (n > k) {
        for (L i = k + 1; i <= n; ++i) D(i) = DSIGMA(i);
        run(workers, blocks(n - k, kColBlock), [&](size_t t) {
            const L j0 = k + 1 + (L)t * kColBlock, j1 = std::min(n, j0 + kColBlock - 1);
            for (L j = j0; j <= j1; ++j) std::copy(&U2(1, j), &U2(1, j) + n, &U(1, j));
        });
        run(workers, blocks(m, kColBlock), [&](size_t t) {
            const L c0 = (L)t * kColBlock + 1, c1 = std::min(m, c0 + kColBlock - 1);
            for (L cc = c0; cc <= c1; ++cc) std::copy(&VT2(k + 1, cc), &VT2(n, cc) + 1, &VT(k + 1, cc));
        });
    }
    for (L j = 1; j <= 4; ++j) COLTYP(j) = ctot[j - 1];
    return 0;
}

// slasd1 with slasd2's copies and slasd3's work over the threads: merges the
// SVDs of the upper (nl rows, nl + 1 columns) and lower (nr rows, nr + sqre
// columns) blocks, joined by the row (alpha, beta), into that of the whole.
L merge_bidiagonal(unsigned workers, L nl, L nr, L sqre, float* d, float alpha, float beta, float* u, L ldu,
                   float* vt, L ldvt, L* idxq) {
    const L n = nl + nr + 1, m = n + sqre, ldu2 = n, ldvt2 = m;
    std::vector<float> z(m), dsigma(n);
    Pages u2((size_t)ldu2 * n), vt2((size_t)ldvt2 * m);
    std::vector<L> idx(n), idxc(n), coltyp(n), idxp(n);
    float orgnrm = std::max(std::fabs(alpha), std::fabs(beta));
    d[nl] = 0.0f;
    for (L i = 0; i < n; ++i) orgnrm = std::max(orgnrm, std::fabs(d[i]));
    L info = 0, zero = 0, one = 1;
    float unit = 1.0f;
    slascl_("G", &zero, &zero, &orgnrm, &unit, &n, &one, d, &n, &info);
    alpha /= orgnrm;
    beta /= orgnrm;
    L k = 0;
    deflate_bidiagonal(workers, nl, nr, sqre, k, d, z.data(), alpha, beta, u, ldu, vt, ldvt, dsigma.data(), u2.data(),
                       ldu2, vt2.data(), ldvt2, idxp.data(), idx.data(), idxc.data(), idxq, coltyp.data());
    Pages q((size_t)k * k);
    info = secular_bidiagonal(workers, nl, nr, sqre, k, d, q.data(), k, dsigma.data(), u, ldu, u2.data(), ldu2, vt,
                              ldvt, vt2.data(), ldvt2, idxc.data(), coltyp.data(), z.data());
    if (info) return info;
    slascl_("G", &zero, &zero, &unit, &orgnrm, &n, &one, d, &n, &info);
    L rest = n - k, back = -1;
    slamrg_(&k, &rest, d, &one, &back, idxq);
    return 0;
}

// slasd0 on the threads: the SVD of one upper bidiagonal of order n (n + sqre
// columns), U and VT the identity on entry.
L dc_bidiagonal(unsigned workers, L n, L sqre, float* d, float* e, float* u, L ldu, float* vt, L ldvt) {
    L m = n + sqre, smlsiz = kLeaf, ncc = 0, info0 = 0;
    if (n <= kLeaf) {
        std::vector<float> work(4 * (size_t)m + 4);
        slasdq_("U", &sqre, &n, &m, &n, &ncc, d, e, vt, &ldvt, u, &ldu, u, &ldu, work.data(), &info0);
        return info0;
    }
    std::vector<L> inode(n), ndiml(n), ndimr(n), idxq(n);
    L nlvl = 0, nd = 0;
    slasdt_(&n, &nlvl, &nd, inode.data(), ndiml.data(), ndimr.data(), &smlsiz);
    Info info;
    // The bottom level's nodes: each its two halves by slasdq
    const L ndb1 = (nd + 1) / 2;   // 1-based
    run(workers, (size_t)(nd - ndb1 + 1), [&](size_t t) {
        const L i1 = ndb1 - 1 + (L)t, ic = inode[i1] - 1;   // 0-based centre row
        L nl = ndiml[i1], nr = ndimr[i1], nlp1 = nl + 1, sq = 1, i = 0;
        const L nlf = ic - nl, nrf = ic + 1;
        std::vector<float> work(4 * (size_t)(std::max(nl, nr) + 1) + 4);
        slasdq_("U", &sq, &nl, &nlp1, &nl, &ncc, d + nlf, e + nlf, vt + (size_t)nlf * ldvt + nlf, &ldvt,
                u + (size_t)nlf * ldu + nlf, &ldu, u + (size_t)nlf * ldu + nlf, &ldu, work.data(), &i);
        info.set(i);
        for (L j = 0; j < nl; ++j) idxq[nlf + j] = j + 1;
        sq = i1 == nd - 1 ? sqre : 1;
        L nrp1 = nr + sq;
        slasdq_("U", &sq, &nr, &nrp1, &nr, &ncc, d + nrf, e + nrf, vt + (size_t)nrf * ldvt + nrf, &ldvt,
                u + (size_t)nrf * ldu + nrf, &ldu, u + (size_t)nrf * ldu + nrf, &ldu, work.data(), &i);
        info.set(i);
        for (L j = 0; j < nr; ++j) idxq[ic + 1 + j] = j + 1;
    });
    if (info.get()) return info.get();
    // The merges, bottom up, a level at a time
    for (L lvl = nlvl; lvl >= 1; --lvl) {
        const L lf = lvl == 1 ? 1 : (L)1 << (lvl - 1), ll = lvl == 1 ? 1 : 2 * lf - 1;
        const size_t nodes = (size_t)(ll - lf + 1);
        auto node = [&](size_t t, unsigned inner) {
            const L i = lf + (L)t, im1 = i - 1, ic = inode[im1] - 1;
            L nl = ndiml[im1], nr = ndimr[im1], sq = (sqre == 0 && i == ll) ? sqre : 1;
            const L nlf = ic - nl;
            float alpha = d[ic], beta = e[ic];
            if (nl + nr + 1 >= kOwnMergeMinN)
                return merge_bidiagonal(inner, nl, nr, sq, d + nlf, alpha, beta, u + (size_t)nlf * ldu + nlf, ldu,
                                        vt + (size_t)nlf * ldvt + nlf, ldvt, idxq.data() + nlf);
            const L nn = nl + nr + 1, mm = nn + sq;
            std::vector<float> work(3 * (size_t)mm * mm + 2 * (size_t)mm);
            std::vector<L> iwork(4 * (size_t)nn);
            L r = 0;
            slasd1_(&nl, &nr, &sq, d + nlf, &alpha, &beta, u + (size_t)nlf * ldu + nlf, &ldu,
                    vt + (size_t)nlf * ldvt + nlf, &ldvt, idxq.data() + nlf, iwork.data(), work.data(), &r);
            return r;
        };
        if (nodes >= concurrent_merges_min(workers)) {
            run(workers, nodes, [&](size_t t) { info.set(node(t, 1)); });
            if (info.get()) return info.get();
        } else {
            for (size_t t = 0; t < nodes; ++t)
                if (const L r = node(t, workers)) return r;
        }
    }
    return 0;
}

} // namespace

// Sets tl_gpu for one solve.
struct WithGpu {
    explicit WithGpu(GpuGemm* g) : was(tl_gpu) { tl_gpu = g; }
    ~WithGpu() { tl_gpu = was; }
    GpuGemm* was;
};

long tridiagonal_eigensystem(uint32_t n, float* d, float* e, float* z, size_t ldz, unsigned threads,
                             GpuGemm* gpu) {
    const WithGpu with(gpu);
    L N = n, LDZ = (L)ldz, info = 0;
    if (n == 0) return 0;
    if (n <= kSerialMaxN || threads <= 1) {
        L lw = -1, liw = -1, iq = 0;
        float q = 0.0f;
        sstedc_("I", &N, d, e, z, &LDZ, &q, &lw, &iq, &liw, &info);
        std::vector<float> work(std::max<L>(1, (L)q));
        std::vector<L> iwork(std::max<L>(1, iq));
        lw = (L)work.size();
        liw = (L)iwork.size();
        sstedc_("I", &N, d, e, z, &LDZ, work.data(), &lw, iwork.data(), &liw, &info);
        return info;
    }
    identity(threads, N, z, LDZ);
    if (slanst_("M", &N, d, e) == 0.0f) return 0;
    const float eps = slamch_("Epsilon");
    // sstedc's split into unreduced blocks, each scaled to norm 1 and solved
    for (L start = 0; start < N;) {
        L finish = start;
        while (finish + 1 < N) {
            const float tiny = eps * std::sqrt(std::fabs(d[finish])) * std::sqrt(std::fabs(d[finish + 1]));
            if (std::fabs(e[finish]) > tiny) ++finish;
            else break;
        }
        L m = finish - start + 1;
        float* dz = z + (size_t)start * ldz + start;
        if (m > kLeaf) {
            L zero = 0, one = 1, m1 = m - 1;
            float nrm = slanst_("M", &m, d + start, e + start), unit = 1.0f;
            if (nrm != 0.0f) {
                slascl_("G", &zero, &zero, &nrm, &unit, &m, &one, d + start, &m, &info);
                slascl_("G", &zero, &zero, &nrm, &unit, &m1, &one, e + start, &m1, &info);
            }
            info = dc_tridiagonal(threads, m, d + start, e + start, dz, LDZ);
            if (info) return info;
            if (nrm != 0.0f) slascl_("G", &zero, &zero, &unit, &nrm, &m, &one, d + start, &m, &info);
        } else if (m > 1) {
            std::vector<float> work(std::max<L>(1, 2 * m - 2));
            ssteqr_("I", &m, d + start, e + start, dz, &LDZ, work.data(), &info);
            if (info) return info;
        }
        start = finish + 1;
    }
    sort_with_vectors(threads, N, d, z, LDZ, nullptr, 0, false);
    return 0;
}

long bidiagonal_svd(uint32_t n, float* d, float* e, float* u, size_t ldu, float* vt, size_t ldvt, unsigned threads,
                    GpuGemm* gpu) {
    const WithGpu with(gpu);
    L N = n, LDU = (L)ldu, LDVT = (L)ldvt, info = 0;
    if (n == 0) return 0;
    if (n <= kSerialMaxN || threads <= 1) {
        L iq = 0;
        float qd = 0.0f;
        std::vector<float> work(3 * (size_t)n * n + 4 * (size_t)n + 8 * (size_t)n + 16);
        std::vector<L> iwork(8 * (size_t)n + 8);
        sbdsdc_("U", "I", &N, d, e, u, &LDU, vt, &LDVT, &qd, &iq, work.data(), iwork.data(), &info);
        return info;
    }
    identity(threads, N, u, LDU);
    identity(threads, N, vt, LDVT);
    L zero = 0, one = 1, nm1 = N - 1;
    float orgnrm = slanst_("M", &N, d, e), unit = 1.0f;
    if (orgnrm == 0.0f) return 0;
    slascl_("G", &zero, &zero, &orgnrm, &unit, &N, &one, d, &N, &info);
    slascl_("G", &zero, &zero, &orgnrm, &unit, &nm1, &one, e, &nm1, &info);
    const float eps = 0.9f * slamch_("Epsilon");
    for (L i = 0; i < N; ++i)
        if (std::fabs(d[i]) < eps) d[i] = std::copysign(eps, d[i]);
    // sbdsdc's split where the superdiagonal is negligible
    for (L i = 0, start = 0; i < nm1; ++i) {
        if (!(std::fabs(e[i]) < eps || i == nm1 - 1)) continue;
        L nsize;
        if (i < nm1 - 1) {
            nsize = i - start + 1;
        } else if (std::fabs(e[i]) >= eps) {
            nsize = N - start;
        } else {   // e(n-2) negligible: d(n-1) alone
            nsize = i - start + 1;
            u[(size_t)(N - 1) * ldu + N - 1] = std::copysign(1.0f, d[N - 1]);
            vt[(size_t)(N - 1) * ldvt + N - 1] = 1.0f;
            d[N - 1] = std::fabs(d[N - 1]);
        }
        info = dc_bidiagonal(threads, nsize, 0, d + start, e + start, u + (size_t)start * ldu + start, LDU,
                             vt + (size_t)start * ldvt + start, LDVT);
        if (info) return info;
        start = i + 1;
    }
    slascl_("G", &zero, &zero, &unit, &orgnrm, &N, &one, d, &N, &info);
    sort_with_vectors(threads, N, d, u, LDU, vt, LDVT, true);
    return 0;
}

} // namespace metal_linalg::detail
