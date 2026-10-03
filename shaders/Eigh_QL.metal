#include <metal_stdlib>
using namespace metal;

// =============================================================================
// Batched symmetric eigensolver: Householder tridiagonalization and implicit QL
// =============================================================================
//
// Computes A = V diag(lambda) V^T for a batch of real symmetric N x N matrices,
// one threadgroup per matrix, the whole matrix in threadgroup memory (so N is
// bounded by it: 87 at 32 KB). This is LAPACK's method (ssytd2, sorg2l's
// accumulation, ssteqr's implicit QL), which does several times fewer flops
// than the Jacobi kernels: about 9N^3 per Jacobi sweep, over 7 sweeps at
// N = 64, against roughly 5-10 N^3 in all.
//
// Jacobi was the GPU choice because the QL phase applies about N^2 dependent
// Givens rotations, and applied one at a time each would need a barrier. They
// need not be applied one at a time. The QL iteration itself works on the
// tridiagonal (d, e) alone and never reads Z, so one thread runs it and records
// a sweep's rotations; then every thread applies the whole recorded sequence to
// its own row of Z. Rows are independent, so a sweep costs one barrier, and a
// dedicated simdgroup computes the next sweep while the rows apply this one.
//
// Threads: one per row, thread i owning row i. With kOverlap (eigh_ql.mm sets
// it from N = 33), simdgroup 0 is a chaser of its own, lane 0 running the QL
// iteration, and thread 32 + i owns row i; the chaser then takes no part in
// the row work of the other phases. Without it, thread 0 runs the QL
// iteration between sweeps, and the rows wait for it.
//
//   1. load the requested triangle, mirror it, scan for the largest entry and
//      for non-finite ones; scale by a power of two so the largest entry lies
//      in [0.5, 1), as the other eigensolver kernels do
//   2. tridiagonalize, A = Q T Q^T (ssytd2, lower): per column, a Householder
//      vector, p = tau A v, and the symmetric rank-2 update of the trailing rows
//   3. with eigenvectors, form Q in place from the reflectors
//   4. implicit QL with Wilkinson-type shifts on (d, e), the rotations applied
//      to the rows of Z = Q
//   5. rank sort; eigenvalues ascending, eigenvectors as columns
//
// References:
//   Golub & Van Loan, Matrix Computations 4th ed., s8.3 (tridiagonalization,
//     implicit symmetric QR step) and s5.1.6 (accumulating Householder products).
//   LAPACK ssytd2, sorg2l, ssteqr; the QL step is that of EISPACK's tql2
//     (Bowdler, Martin, Reinsch & Wilkinson, Numer. Math. 11, 1968).
//
// Built with fast math, unlike the Jacobi kernels (see CMakeLists.txt): the
// QL iteration is a latency-bound chain, and IEEE mode made the kernel 1.4x
// slower. What needs the accuracy -- the Householder vectors and the shift --
// takes one Newton step after the fast division or square root (div_nr,
// sqrt_nr), the rotations come from one reciprocal square root, and the
// non-finite input check reads the bits.

constant bool kComputeVectors [[function_constant(0)]];
constant bool kOverlap        [[function_constant(1)]];   // a chaser simdgroup of its own

// Must match `QlParams` in eigh_ql.mm.
struct QlParams {
    uint n;          // matrix order, at most the device's limit (eigh_ql.mm)
    uint lower;      // 1: read the lower triangle, 0: the upper
    uint max_iter;   // QL iterations allowed per matrix in total (LAPACK: 30 N)
};

// Info word written per matrix: QL iterations | converged | non-finite input.
constant uint kInfoConverged = 1u << 16;
constant uint kInfoNonFinite = 1u << 17;

constant uint kChaser = kOverlap ? 32 : 0;   // threads before the first row thread

// Threadgroup memory, in floats, with ld = n | 1 (an odd row stride, so a
// column walk touches every bank once):
//   a     n * ld   the matrix, then Q, then the eigenvectors (rows)
//   d, e  n each   the tridiagonal: diagonal, and e[i] = T(i+1, i)
//   x     4n       tau, v, w (steps 2-3); two rotation buffers of (c, s)
//                  (step 4); ranks (step 5)
//   red   8        reductions: two sets of one slot per simdgroup (at most 4)
//   ctl   8 uints  step 4: per rotation buffer {first column, count, done}, at 0 and 4
// eigh_ql.mm computes the same size.

// The sum over the threadgroup of x, every thread contributing (0 if it has
// nothing). `red` is a set of four slots; the caller alternates sets between
// consecutive sums, and puts a barrier between two uses of the same set.
inline float group_sum(float x, threadgroup float* red, uint sg, uint lane, uint nsg) {
    x = simd_sum(x);
    if (lane == 0) red[sg] = x;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float s = 0.0f;
    for (uint k = 0; k < nsg; ++k) s += red[k];
    return s;
}

inline float group_max(float x, threadgroup float* red, uint sg, uint lane, uint nsg) {
    x = simd_max(x);
    if (lane == 0) red[sg] = x;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float m = 0.0f;
    for (uint k = 0; k < nsg; ++k) m = fmax(m, red[k]);
    return m;
}

// Division and square root to about half an ulp, from the fast forms and one
// Newton step. The kernel is built with fast math: one IEEE-mode (precise)
// division or square root anywhere in it made the compiler build the whole
// kernel that way, 1.4x slower, and only the Householder vectors and the
// shift need the accuracy.
inline float div_nr(float a, float b) {
    const float r = 1.0f / b;
    const float q = a * r;
    return fma(fma(-b, q, a), r, q);   // q + (a - b q) / b
}

inline float sqrt_nr(float x) {
    if (x <= 0.0f) return 0.0f;
    const float y = rsqrt(x);
    const float s = x * y;
    return fma(fma(-s, s, x), 0.5f * y, s);   // s + (x - s^2) / (2 s)
}

// NaN or infinity, from the bits: fast math may assume neither exists.
inline bool non_finite(float v) {
    return (as_type<uint>(v) & 0x7F800000u) == 0x7F800000u;
}

// The chaser's state across QL sweeps.
struct QlState {
    uint l;        // the eigenvalue being found; d[0 .. l-1] are final
    uint total;    // QL iterations so far
    bool failed;   // the iteration budget ran out
};

// Finds the next QL sweep and runs it on (d, e), recording its rotations in
// (c, s) if `record`: rotation r acts on columns (lo - r, lo - r + 1).
// meta = {lo, count, done}. Advances past eigenvalues that have converged;
// sets done once every one has, or when the budget runs out.
inline void ql_sweep(threadgroup float* d, threadgroup float* e,
                     threadgroup float* c_out, threadgroup float* s_out,
                     threadgroup uint* meta, thread QlState& st,
                     uint n, uint max_iter, bool record) {
    while (st.l < n) {
        const uint l = st.l;
        // The first negligible off-diagonal at or after l, by ssteqr's test
        // relative to its two diagonal neighbours.
        uint m = l;
        for (; m + 1 < n; ++m) {
            const float em = fabs(e[m]);
            if (em <= FLT_EPSILON * (fabs(d[m]) + fabs(d[m + 1])) || em < FLT_MIN) break;
        }
        if (m == l) {           // d[l] has converged
            st.l = l + 1;
            continue;
        }
        if (st.total >= max_iter) {
            st.failed = true;
            break;
        }
        ++st.total;

        // The shift: the eigenvalue of the leading 2 x 2 block nearer d[l].
        const float el = e[l];
        float g = div_nr(d[l + 1] - d[l], 2.0f * el);
        float r = fabs(g) > 1.0e18f ? fabs(g) : sqrt_nr(g * g + 1.0f);
        g = d[m] - d[l] + div_nr(el, g + copysign(r, g));

        float s = 1.0f, c = 1.0f, p = 0.0f;
        float dn = d[m];   // d[i + 1], untouched by this sweep until written below
        uint cnt = 0;
        bool underflow = false;
        for (int i = (int)m - 1; i >= (int)l; --i) {
            const float di = d[i], ei = e[i];   // not on the chain: issued early
            const float f = s * ei, b = c * ei;
            const float af = fabs(f), ag = fabs(g);
            const float big = fmax(af, ag);
            if (big == 0.0f) {
                // f = g = 0: the rest of the sweep would divide by zero.
                // tql2 deflates here and starts the next sweep.
                e[i + 1] = 0.0f;
                d[i + 1] = dn - p;
                e[m] = 0.0f;
                underflow = true;
                break;
            }
            // r = hypot(f, g), s = f / r, c = g / r from one reciprocal
            // square root: the chain is latency-bound, and a square root and
            // two divides are its longest link. c^2 + s^2 = h / r^2 stays
            // within an ulp or two of 1. Scaled where h would under- or
            // overflow (or flush to zero).
            const float h = f * f + g * g;
            if (h >= FLT_MIN && h <= FLT_MAX) {
                const float inv = fast::rsqrt(h);
                r = h * inv;
                s = f * inv;
                c = g * inv;
            } else {
                const float fs = f / big, gs = g / big;
                const float inv = fast::rsqrt(fs * fs + gs * gs);
                r = big / inv;
                s = fs * inv;
                c = gs * inv;
            }
            e[i + 1] = r;
            g = dn - p;
            r = (di - g) * s + 2.0f * c * b;
            p = s * r;
            d[i + 1] = g + p;
            g = c * r - b;
            if (record) { c_out[cnt] = c; s_out[cnt] = s; }
            ++cnt;
            dn = di;
        }
        if (!underflow) {
            d[l] -= p;
            e[l] = g;
            e[m] = 0.0f;
        }
        meta[0] = m - 1;
        meta[1] = cnt;
        meta[2] = 0;
        return;
    }
    meta[2] = 1;
}

kernel void eigh_ql(
    device const float* A_in [[buffer(0)]],   // [batch, n, n] input (row-major)
    device float*       vals [[buffer(1)]],   // [batch, n] eigenvalues, ascending
    device float*       vecs [[buffer(2)]],   // [batch, n, n] eigenvectors as columns (kComputeVectors)
    device uint*        info [[buffer(3)]],   // [batch] iterations | kInfoConverged | kInfoNonFinite
    constant QlParams&  prm  [[buffer(4)]],
    threadgroup float*  tg   [[threadgroup(0)]],
    uint mat  [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint ntg  [[threads_per_threadgroup]],
    uint sg   [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint nsg  [[simdgroups_per_threadgroup]])
{
    const uint n  = prm.n;
    const uint ld = n | 1u;
    threadgroup float* a    = tg;
    threadgroup float* d    = a + n * ld;
    threadgroup float* e    = d + n;
    threadgroup float* x    = e + n;
    threadgroup float* red  = x + 4 * n;
    threadgroup uint*  ctl  = reinterpret_cast<threadgroup uint*>(red + 8);
    threadgroup float* tau  = x;           // steps 2-3
    threadgroup float* vb   = x + n;
    threadgroup float* wb   = x + 2 * n;
    threadgroup uint*  rank = reinterpret_cast<threadgroup uint*>(x);   // step 5

    const uint i   = tid - kChaser;        // this thread's row, if it has one
    const bool row = tid >= kChaser && i < n;

    // -------------------------------------------------------------------------
    // 1. Load, scan, scale
    // -------------------------------------------------------------------------
    device const float* src = A_in + (ulong)mat * n * n;
    float amax = 0.0f, bad = 0.0f;
    for (uint idx = tid; idx < n * n; idx += ntg) {
        const uint r = idx / n, k = idx - r * n;
        if (prm.lower ? (k <= r) : (k >= r)) {
            const float v = src[idx];
            a[r * ld + k] = v;
            a[k * ld + r] = v;
            amax = fmax(amax, fabs(v));   // fmax skips NaN, hence the separate flag
            if (non_finite(v)) bad = 1.0f;
        }
    }
    amax = group_max(amax, red, sg, lane, nsg);
    const bool nonfinite = group_sum(bad, red + 4, sg, lane, nsg) > 0.0f;

    device float* out_vals = vals + (ulong)mat * n;
    device float* out_vecs = vecs + (ulong)mat * n * n;
    if (nonfinite) {
        const float qnan = as_type<float>(0x7FC00000u);
        for (uint k = tid; k < n; k += ntg) out_vals[k] = qnan;
        if (kComputeVectors) for (uint k = tid; k < n * n; k += ntg) out_vecs[k] = qnan;
        if (tid == 0) info[mat] = kInfoNonFinite;
        return;   // uniform: every thread has the same flag
    }
    // A power of two, so exact: the products of the reduction stay inside
    // float32's range whatever the input's magnitude.
    int expo = 0;
    if (amax > 0.0f) frexp(amax, expo);
    for (uint idx = tid; idx < n * n; idx += ntg) {
        const uint r = idx / n, k = idx - r * n;
        a[r * ld + k] = ldexp(a[r * ld + k], -expo);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // -------------------------------------------------------------------------
    // 2. Tridiagonalize: H(n-3) ... H(0) A H(0) ... H(n-3) = T
    // -------------------------------------------------------------------------
    // Step k reduces column k. The reflector H(k) = I - tau v v^T acts on rows
    // and columns k+1 .., with v(k+1) = 1; its tail is kept below the
    // subdiagonal of column k for step 3. Each row thread keeps its whole row
    // of the trailing matrix, both triangles, so the products are row-local.
    for (uint k = 0; k + 2 < n; ++k) {
        float t = 0.0f;
        if (row && i >= k + 2) { const float v = a[i * ld + k]; t = v * v; }
        const float sumsq = group_sum(t, red, sg, lane, nsg);
        const float alpha = a[(k + 1) * ld + k];
        float tv = 0.0f, beta = alpha;
        if (sumsq > 0.0f) {
            beta = -copysign(sqrt_nr(alpha * alpha + sumsq), alpha);
            tv = div_nr(beta - alpha, beta);
            const float scal = div_nr(1.0f, alpha - beta);
            if (row && i >= k + 2) {
                const float v = a[i * ld + k] * scal;
                a[i * ld + k] = v;
                vb[i] = v;
            }
        } else if (row && i >= k + 2) {
            vb[i] = 0.0f;   // already zero below the subdiagonal: H = I
        }
        if (row && i == k + 1) vb[i] = 1.0f;
        if (tid == 0) {
            d[k] = a[k * ld + k];
            e[k] = beta;
            tau[k] = tv;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tv == 0.0f) continue;   // uniform

        // p = tau A22 v, then w = p - (tau/2)(p . v) v
        float p = 0.0f;
        if (row && i >= k + 1) {
            for (uint j = k + 1; j < n; ++j) p += a[i * ld + j] * vb[j];
            p *= tv;
        }
        const float vi = (row && i >= k + 1) ? vb[i] : 0.0f;
        const float pv = group_sum(p * vi, red + 4, sg, lane, nsg);
        const float w = p - 0.5f * tv * pv * vi;
        if (row && i >= k + 1) wb[i] = w;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // A22 -= v w^T + w v^T, row i
        if (row && i >= k + 1) {
            for (uint j = k + 1; j < n; ++j) a[i * ld + j] -= vi * wb[j] + w * vb[j];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        if (n >= 2) {
            d[n - 2] = a[(n - 2) * ld + n - 2];
            e[n - 2] = a[(n - 1) * ld + n - 2];
        }
        d[n - 1] = a[(n - 1) * ld + n - 1];
        e[n - 1] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // -------------------------------------------------------------------------
    // 3. Q = H(0) H(1) ... H(n-3), formed in place
    // -------------------------------------------------------------------------
    // Accumulated backward, as sorg2r does: after step j the matrix holds
    // H(j) ... H(n-3), which is the identity outside rows and columns j+1 ..
    // Column j+1 of it overwrites the stale entries there, while H(j)'s vector
    // is read from column j, which is not overwritten until step j-1.
    if (kComputeVectors) {
        if (n >= 3) {
            if (tid == 0) a[(n - 1) * ld + n - 1] = 1.0f;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (int jj = (int)n - 3; jj >= 0; --jj) {
                const uint j = (uint)jj;
                const float tj = tau[j];
                // Row j+1 of the current product is e_{j+1}^T.
                if (row && i == j + 1) for (uint k = j + 2; k < n; ++k) a[i * ld + k] = 0.0f;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                // s_k = v^T Q(:, k) for k >= j+2, by the thread owning row k
                // (column sums: the threads walk a column each).
                if (row && i >= j + 2) {
                    float s = 0.0f;
                    for (uint r = j + 2; r < n; ++r) s += a[r * ld + j] * a[r * ld + i];
                    wb[i] = s;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (row && i >= j + 1) {
                    const float vi = (i == j + 1) ? 1.0f : a[i * ld + j];
                    for (uint k = j + 2; k < n; ++k) a[i * ld + k] -= tj * vi * wb[k];
                    a[i * ld + j + 1] = (i == j + 1) ? 1.0f - tj : -tj * vi;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            // Row and column 0 are e_0.
            if (row) a[i * ld] = (i == 0) ? 1.0f : 0.0f;
            if (tid == 0) for (uint k = 1; k < n; ++k) a[k] = 0.0f;
        } else if (row) {
            for (uint k = 0; k < n; ++k) a[i * ld + k] = (i == k) ? 1.0f : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // -------------------------------------------------------------------------
    // 4. Implicit QL on (d, e); rotations applied to the rows of Z
    // -------------------------------------------------------------------------
    QlState st{0u, 0u, false};
    if (kComputeVectors && !kOverlap) {
        // One thread chases, then every row applies: two barriers per sweep.
        while (true) {
            if (tid == 0) ql_sweep(d, e, x, x + n, ctl, st, n, prm.max_iter, true);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (ctl[2]) break;
            const uint lo = ctl[0], cnt = ctl[1];
            if (row && cnt) {
                threadgroup float* z = a + i * ld;
                float carry = z[lo + 1];
                for (uint r = 0; r < cnt; ++r) {
                    const uint col = lo - r;
                    const float zc = z[col], cr = x[r], sr = x[n + r];
                    z[col + 1] = sr * zc + cr * carry;
                    carry = cr * zc - sr * carry;
                }
                z[lo + 1 - cnt] = carry;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    } else if (kComputeVectors) {
        // Two rotation buffers: while the rows apply sweep t from one, the
        // chaser computes sweep t+1 into the other. One barrier per sweep.
        threadgroup float* buf[2] = {x, x + 2 * n};   // (c, s) at [0, n) and [n, 2n)
        if (tid == 0) ql_sweep(d, e, buf[0], buf[0] + n, ctl, st, n, prm.max_iter, true);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint slot = 0;
        while (ctl[slot * 4 + 2] == 0) {
            if (tid == 0) {
                const uint next = slot ^ 1u;
                ql_sweep(d, e, buf[next], buf[next] + n, ctl + next * 4, st, n, prm.max_iter, true);
            } else if (row) {
                const uint lo = ctl[slot * 4], cnt = ctl[slot * 4 + 1];
                if (cnt) {
                    threadgroup const float* cb = buf[slot];
                    threadgroup const float* sbuf = buf[slot] + n;
                    threadgroup float* z = a + i * ld;
                    // Rotation r: (z[col], z[col+1]) <- (c z[col] - s z[col+1],
                    // s z[col] + c z[col+1]) with col = lo - r. The new z[col]
                    // is the next rotation's z[col+1], so it is carried.
                    float carry = z[lo + 1];
                    for (uint r = 0; r < cnt; ++r) {
                        const uint col = lo - r;
                        const float zc = z[col], cr = cb[r], sr = sbuf[r];
                        z[col + 1] = sr * zc + cr * carry;
                        carry = cr * zc - sr * carry;
                    }
                    z[lo + 1 - cnt] = carry;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            slot ^= 1u;
        }
    } else {
        if (tid == 0) {
            do {
                ql_sweep(d, e, x, x + n, ctl, st, n, prm.max_iter, false);
            } while (ctl[2] == 0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // -------------------------------------------------------------------------
    // 5. Sort ascending; write
    // -------------------------------------------------------------------------
    if (row) {
        const float di = d[i];
        uint rk = 0;
        for (uint k = 0; k < n; ++k) rk += (d[k] < di || (d[k] == di && k < i)) ? 1u : 0u;
        rank[rk] = i;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint k = tid; k < n; k += ntg) out_vals[k] = ldexp(d[rank[k]], expo);
    if (kComputeVectors) {
        for (uint idx = tid; idx < n * n; idx += ntg) {
            const uint r = idx / n, k = idx - r * n;
            out_vecs[idx] = a[r * ld + rank[k]];
        }
    }
    if (tid == 0) {   // the chaser, whose state this is
        info[mat] = min(st.total, 0xFFFFu) | (st.failed ? 0u : kInfoConverged);
    }
}
