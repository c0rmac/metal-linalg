// The second stage of the two-stage reductions, on the CPU's cores: a band
// matrix to bidiagonal (the SVD's `band` backend), or a symmetric band to
// tridiagonal (the eigensolver's), by bulge chasing with Householder
// reflectors, the sweeps pipelined over threads.
//
// Sweep s makes row s bidiagonal: a reflector from the right on columns
// s+1 .. s+nb annihilates the row past its superdiagonal and fills the
// diagonal block below the diagonal; one from the left annihilates that
// block's first column and, applied to the block to its right, fills it past
// the band; a reflector from the right annihilates that block's first row,
// and so on down the matrix, a block of nb at a time. Each reflector clears
// one row or column of its bulge; the rest is cleared by the sweeps that
// follow, so the matrix stays within nb below and 2 nb above the diagonal.
// These are PLASMA's kernels for the upper case (Haidar, Ltaief and Dongarra,
// "Parallel reduction to condensed forms for symmetric eigenvalue problems
// using aggregated fine-grained and memory-aware kernels", SC '11, for the
// symmetric one that LAPACK's ssytrd_sb2st runs).
//
// The symmetric case, a lower band: sweep s annihilates column s below the
// subdiagonal with a reflector applied to both sides of the diagonal block
// below it; applied from the right to the block under that, it fills it past
// the band; that block's first column's reflector, applied from the left to
// the rest of it and to both sides of the next diagonal block, moves the bulge
// down, and so on. These are LAPACK's ssb2st kernels.
//
// Why: LAPACK's sgbbrd does it by rotations on one core: on an M5 Pro 91 ms
// for a 4096 x 4096 band of width 8, 196 ms at 16, 401 ms at 32 (and for a
// symmetric band ssytrd_sb2st 112 ms at 16, ssbtrd 178). This is 1.6x faster
// on one core at width 16, and the sweeps overlap: sweep s can run its t-th
// task once sweep s - 1 has finished its (t + 2)-th, the blocks they touch
// being one row and column apart. With 16 threads, 35 ms at width 16 and 39
// at 32 (the symmetric one 35 and 34).
//
// For the SVD with vectors the chase keeps its reflectors (ChaseReflectors),
// straight into the blocks svd_bidiag.mm's bd_chase_apply applies.
//
// Each sweep runs on one thread (sweep s on thread s mod P) and publishes how
// many tasks it has done; a thread waits for the previous sweep by spinning
// on that count. The threads are std::threads, all running at once, because a
// pool that ran fewer of them than the sweeps they wait on would deadlock.

#include "band_chase.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <memory>
#include <cmath>
#include <thread>
#include <vector>

namespace metal_linalg::detail {
namespace {

struct Band {
    float* W;
    long   ld, ku;
    float* at(long i, long j) const { return W + j * ld + ku + i - j; }   // a column's rows are contiguous
};

// Householder reflector for x (len entries, stride inc): x[0] becomes beta,
// x[1:] zero, v = (1, v[1:]); returns tau. The norm in double, so that float
// inputs neither over- nor underflow.
float reflector(long len, float* x, long inc, float* v) {
    v[0] = 1.0f;
    if (len <= 1) return 0.0f;
    double ss = 0.0;
    for (long k = 1; k < len; ++k) {
        const double t = x[k * inc];
        ss += t * t;
    }
    if (ss == 0.0) {
        for (long k = 1; k < len; ++k) v[k] = 0.0f;
        return 0.0f;
    }
    const double alpha = x[0];
    const double beta = -std::copysign(std::sqrt(alpha * alpha + ss), alpha);
    const double scale = 1.0 / (alpha - beta);
    x[0] = (float)beta;
    for (long k = 1; k < len; ++k) {
        v[k] = (float)(x[k * inc] * scale);
        x[k * inc] = 0.0f;
    }
    return (float)((beta - alpha) / beta);
}

// A(r0:r1, c0:c1) = (I - tau u u^T) A(r0:r1, c0:c1)
void apply_left(const Band& A, long r0, long r1, long c0, long c1, const float* u, float tau) {
    if (tau == 0.0f) return;
    const long m = r1 - r0 + 1;
    for (long j = c0; j <= c1; ++j) {
        float* p = A.at(r0, j);
        float w = 0.0f;
        for (long k = 0; k < m; ++k) w += u[k] * p[k];
        w *= tau;
        for (long k = 0; k < m; ++k) p[k] -= w * u[k];
    }
}

// A(r0:r1, c0:c1) = A(r0:r1, c0:c1) (I - tau v v^T), by columns (contiguous)
void apply_right(const Band& A, long r0, long r1, long c0, long c1, const float* v, float tau, float* w) {
    if (tau == 0.0f || r0 > r1) return;
    const long m = r1 - r0 + 1, nc = c1 - c0 + 1;
    std::fill(w, w + m, 0.0f);
    for (long k = 0; k < nc; ++k) {
        const float* p = A.at(r0, c0 + k);
        for (long i = 0; i < m; ++i) w[i] += v[k] * p[i];
    }
    for (long k = 0; k < nc; ++k) {
        float* p = A.at(r0, c0 + k);
        const float t = tau * v[k];
        for (long i = 0; i < m; ++i) p[i] -= t * w[i];
    }
}

// One sweep of the upper band to bidiagonal, a task at a time. Task 0 (type
// 1): row s's reflector, applied to the diagonal block st..ed, then that
// block's first column's, applied to the block. Then alternately (type 2) the
// column reflector applied to the block to the right and that block's first
// row's reflector, applied to the rest of it, and (type 3) the row reflector
// applied to the diagonal block below and its first column's.
struct Sweep {
    long  s, st, ed, task = 0;
    bool  done;
    float tq = 0.0f, tp = 0.0f;
    std::vector<float> v, u, w;
    const ChaseReflectors* rec;

    Sweep(long nb, const ChaseReflectors* r) : v(nb + 1), u(nb + 1), w(nb + 1), rec(r) {}

    // Keeps reflector (s, j): left (rows) or right (columns).
    void keep(bool left, long j, const float* x, long len, float tau, long) const {
        if (!rec || !rec->L || len < 2) return;
        const size_t G = (size_t)s / 16, c = (size_t)s % 16, pm = rec->pmax;
        const size_t b = G * (pm + 1) - G * (G - 1) / 2 + (size_t)j;
        // Column c lies in V's column tile c / 8, in its three row tiles from
        // c / 8 (tiles 0-2 or 3-5), 24 rows from 8 (c / 8).
        const size_t ct = c / 8, r0 = 8 * ct;
        float* blk = (left ? rec->L : rec->R) + b * kChaseBlockFloats + ct * 3 * 64 + c % 8;
        for (size_t r = r0; r < r0 + 24; ++r)
            blk[(r - r0) / 8 * 64 + (r % 8) * 8] = r >= c && r < c + (size_t)len ? x[r - c] : 0.0f;
        (left ? rec->Ltau : rec->Rtau)[b * 16 + c] = tau;
    }

    void start(long sweep, long n, long nb) {
        s = sweep;
        st = s + 1;
        ed = std::min(s + nb, n - 1);
        task = 0;
        done = st > n - 1;
    }

    // Runs the next task; false when the sweep is over.
    bool step(const Band& A, long n, long nb) {
        if (done) return false;
        if (task == 0) {
            const long len = ed - st + 1;
            tq = reflector(len, A.at(s, st), A.ld - 1, v.data());
            keep(false, 0, v.data(), len, tq, nb);
            apply_right(A, st, ed, st, ed, v.data(), tq, w.data());
            tp = reflector(len, A.at(st, st), 1, u.data());
            keep(true, 0, u.data(), len, tp, nb);
            apply_left(A, st, ed, st + 1, ed, u.data(), tp);
        } else if (task % 2 == 1) {
            const long j1 = ed + 1, j2 = std::min(ed + nb, n - 1);
            if (j1 > n - 1) {
                done = true;
                return false;
            }
            apply_left(A, st, ed, j1, j2, u.data(), tp);
            if (j2 == j1) {
                done = true;
                ++task;
                return true;
            }
            tq = reflector(j2 - j1 + 1, A.at(st, j1), A.ld - 1, v.data());
            keep(false, (task + 1) / 2, v.data(), j2 - j1 + 1, tq, nb);
            apply_right(A, st + 1, ed, j1, j2, v.data(), tq, w.data());
            st = j1;
            ed = j2;
        } else {
            apply_right(A, st, ed, st, ed, v.data(), tq, w.data());
            tp = reflector(ed - st + 1, A.at(st, st), 1, u.data());
            keep(true, task / 2, u.data(), ed - st + 1, tp, nb);
            apply_left(A, st, ed, st + 1, ed, u.data(), tp);
        }
        ++task;
        return true;
    }
};

// The lower half of a symmetric band: A(r, c), r >= c, at W[c * ld + r - c].
struct SymBand {
    float* W;
    long   ld;
    float* at(long r, long c) const { return W + c * ld + r - c; }
};

// The diagonal block A(st:ed, st:ed) (its lower triangle) = H A H, H = I - tau u u^T:
// w = tau A u - tau/2 (tau u^T A u) u, A -= u w^T + w u^T.
void two_sided(const SymBand& A, long st, long ed, const float* u, float tau, float* w) {
    if (tau == 0.0f) return;
    const long m = ed - st + 1;
    std::fill(w, w + m, 0.0f);
    for (long c = 0; c < m; ++c) {
        const float* col = A.at(st + c, st + c);
        float acc = col[0] * u[c];
        for (long r = c + 1; r < m; ++r) {
            acc += col[r - c] * u[r];
            w[r] += col[r - c] * u[c];
        }
        w[c] += acc;
    }
    double dot = 0.0;
    for (long i = 0; i < m; ++i) {
        w[i] *= tau;
        dot += (double)w[i] * u[i];
    }
    const float alpha = (float)(-0.5 * tau * dot);
    for (long i = 0; i < m; ++i) w[i] += alpha * u[i];
    for (long c = 0; c < m; ++c) {
        float* col = A.at(st + c, st + c);
        for (long r = c; r < m; ++r) col[r - c] -= u[r] * w[c] + w[r] * u[c];
    }
}

// One sweep of the symmetric band to tridiagonal (ssb2st's kernels). Task 0:
// column s's reflector, then both sides of the diagonal block st..ed. Then
// alternately the block below from the right and its first column's
// reflector from the left, and that reflector on both sides of its diagonal
// block.
struct SymSweep {
    long  s, st, ed, task = 0;
    bool  done;
    float tau = 0.0f;
    std::vector<float> u, w;
    const ChaseReflectors* rec;

    SymSweep(long kd, const ChaseReflectors* r) : u(kd + 1), w(kd + 1), rec(r) {}

    // Keeps reflector (s, j), on rows s + 1 + kd j .., in the bidiagonal
    // chase's left layout (kd = 16): Q2 = the reflectors' product in the
    // order applied, as for the SVD's left side.
    void keep(long j, const float* x, long len, float t) const {
        if (!rec || !rec->L || len < 2) return;
        const size_t G = (size_t)s / 16, c = (size_t)s % 16, pm = rec->pmax;
        const size_t b = G * (pm + 1) - G * (G - 1) / 2 + (size_t)j;
        const size_t ct = c / 8, r0 = 8 * ct;
        float* blk = rec->L + b * kChaseBlockFloats + ct * 3 * 64 + c % 8;
        for (size_t r = r0; r < r0 + 24; ++r)
            blk[(r - r0) / 8 * 64 + (r % 8) * 8] = r >= c && r < c + (size_t)len ? x[r - c] : 0.0f;
        rec->Ltau[b * 16 + c] = t;
    }

    void start(long sweep, long n, long kd) {
        s = sweep;
        st = s + 1;
        ed = std::min(s + kd, n - 1);
        task = 0;
        done = st > n - 1;
    }

    bool step(const SymBand& A, long n, long kd) {
        if (done) return false;
        if (task == 0) {
            tau = reflector(ed - st + 1, A.at(st, s), 1, u.data());
            keep(0, u.data(), ed - st + 1, tau);
            two_sided(A, st, ed, u.data(), tau, w.data());
        } else if (task % 2 == 1) {
            const long j1 = ed + 1, j2 = std::min(ed + kd, n - 1);
            if (j1 > n - 1) {
                done = true;
                return false;
            }
            const long m = j2 - j1 + 1, nc = ed - st + 1;   // the block A(j1:j2, st:ed) times H, by columns
            if (tau != 0.0f) {
                std::fill(w.begin(), w.begin() + m, 0.0f);
                for (long k = 0; k < nc; ++k) {
                    const float* p = A.at(j1, st + k);
                    for (long i = 0; i < m; ++i) w[i] += u[k] * p[i];
                }
                for (long k = 0; k < nc; ++k) {
                    float* p = A.at(j1, st + k);
                    const float t = tau * u[k];
                    for (long i = 0; i < m; ++i) p[i] -= t * w[i];
                }
            }
            tau = reflector(m, A.at(j1, st), 1, u.data());
            keep((task + 1) / 2, u.data(), m, tau);
            for (long j = st + 1; j <= ed; ++j) {   // the rest of the block from the left
                if (tau == 0.0f) break;
                float* p = A.at(j1, j);
                float t = 0.0f;
                for (long k = 0; k < m; ++k) t += u[k] * p[k];
                t *= tau;
                for (long k = 0; k < m; ++k) p[k] -= t * u[k];
            }
            st = j1;
            ed = j2;
        } else {
            two_sided(A, st, ed, u.data(), tau, w.data());
        }
        ++task;
        return true;
    }
};

// Sweeps 0 .. n - 2 pipelined over P threads, as described above.
// Waits for ok(): spinning, then yielding, and after a long wait (as for the
// GPU's band reduction, which the chase can trail) sleeping between looks.
template <class F>
void wait_until(const F& ok) {
    for (int spins = 0, yields = 0; !ok();)
        if (++spins == 64) {
            spins = 0;
            if (++yields < 256) std::this_thread::yield();
            else std::this_thread::sleep_for(std::chrono::microseconds(20));
        }
}

template <class S, class M>
void pipeline(long n, long nb, long P, const M& A, const ChaseReflectors* rec = nullptr) {
    std::vector<std::atomic<long>> done(n);   // per sweep: tasks done, LONG_MAX when over
    for (auto& x : done) x.store(0, std::memory_order_relaxed);
    const std::atomic<long>* ready = rec ? rec->ready_rows : nullptr;
    // Whether sweep s's next task (`task`) may run: the sweep before has run
    // two tasks further, or, for sweep 0 trailing the band reduction, the
    // rows its task reaches (about (task / 2 + 2) nb) are final.
    auto may = [&](long s, long task) {
        if (s > 0) return done[s - 1].load(std::memory_order_acquire) >= task + 3;
        return !ready || ready->load(std::memory_order_acquire) >= std::min(n, (task / 2 + 2) * nb + 1);
    };
    auto over = [&](long s) {
        done[s].store(LONG_MAX, std::memory_order_release);
        if (rec && rec->frontier) {   // the leading sweeps finished
            long f = rec->frontier->load(std::memory_order_acquire);
            while (f < n - 1 && done[f].load(std::memory_order_acquire) == LONG_MAX)
                if (rec->frontier->compare_exchange_weak(f, f + 1, std::memory_order_acq_rel)) ++f;
        }
    };
    // A sweep a thread at a time, each thread's sweeps s = p, p + P, ...
    auto worker = [&](long p) {
        S sweep(nb, rec);
        for (long s = p; s < n - 1; s += P) {
            sweep.start(s, n, nb);
            for (;;) {
                wait_until([&] { return may(s, sweep.task); });
                if (!sweep.step(A, n, nb)) break;
                done[s].store(sweep.task, std::memory_order_release);
            }
            over(s);
        }
    };
    // Trailing the band reduction, every sweep stops where the band is not
    // final yet, so a thread keeps several of its sweeps open and runs
    // whichever may go on: the later sweeps' work on the finished rows runs
    // while the first ones wait for the GPU, where one sweep a thread would
    // have held all the threads at the GPU's frontier.
    auto trailing = [&](long p) {
        std::vector<std::unique_ptr<S>> open;
        long next = p;
        for (int idle = 0;;) {
            bool moved = false;
            for (size_t i = 0; i < open.size();) {
                S& sw = *open[i];
                bool ended = false;
                while (may(sw.s, sw.task)) {
                    moved = true;
                    if (!sw.step(A, n, nb)) {
                        ended = true;
                        break;
                    }
                    done[sw.s].store(sw.task, std::memory_order_release);
                }
                if (ended) {
                    over(sw.s);
                    open.erase(open.begin() + (long)i);
                } else {
                    ++i;
                }
            }
            if (next < n - 1 && may(next, 0)) {
                open.push_back(std::make_unique<S>(nb, rec));
                open.back()->start(next, n, nb);
                next += P;
                moved = true;
            }
            if (open.empty() && next >= n - 1) break;
            if (moved) {
                idle = 0;
            } else if (++idle < 256) {
                std::this_thread::yield();
            } else {
                std::this_thread::sleep_for(std::chrono::microseconds(20));
            }
        }
    };
    if (P == 1) {
        ready ? trailing(0) : worker(0);
        return;
    }
    std::vector<std::thread> pool;
    for (long p = 0; p < P; ++p) {
        if (ready) pool.emplace_back(trailing, p);
        else pool.emplace_back(worker, p);
    }
    for (auto& t : pool) t.join();
}

// Threads: none past what the pipeline can keep busy (a sweep's tasks are
// 2 n / nb, three apart), and one for every 256 rows at most.
long pipeline_threads(long n, long nb, unsigned threads) {
    return std::max<long>(1, std::min<long>({(long)threads, n / 256, (2 * n / nb) / 3}));
}

} // namespace

void band_to_bidiagonal(uint32_t n, uint32_t nb, float* W, size_t ld, size_t ku, float* d, float* e,
                        unsigned threads, const ChaseReflectors* rec) {
    const Band A{W, (long)ld, (long)ku};
    const long N = n, NB = std::max<uint32_t>(nb, 1);
    if (N > 1 && NB > 1) pipeline<Sweep>(N, NB, pipeline_threads(N, NB, threads), A, rec);
    if (rec && rec->ready_rows) wait_until([&] { return rec->ready_rows->load(std::memory_order_acquire) >= N; });
    for (long i = 0; i < N; ++i) {
        d[i] = *A.at(i, i);
        if (i + 1 < N) e[i] = *A.at(i, i + 1);
    }
}

void band_to_tridiagonal(uint32_t n, uint32_t kd, float* W, size_t ld, float* d, float* e, unsigned threads,
                         const ChaseReflectors* rec) {
    const SymBand A{W, (long)ld};
    const long N = n, KD = std::max<uint32_t>(kd, 1);
    if (N > 1 && KD > 1) pipeline<SymSweep>(N, KD, pipeline_threads(N, KD, threads), A, rec);
    if (rec && rec->ready_rows) wait_until([&] { return rec->ready_rows->load(std::memory_order_acquire) >= N; });
    for (long i = 0; i < N; ++i) {
        d[i] = *A.at(i, i);
        if (i + 1 < N) e[i] = *A.at(i + 1, i);
    }
}

} // namespace metal_linalg::detail
