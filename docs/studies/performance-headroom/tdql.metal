// Prototype: batched symmetric eigensolver by LAPACK's method, one
// threadgroup per matrix, N <= TDQL_NMAX, everything in threadgroup memory.
// Built twice (see build.sh): TDQL_NMAX = 64 (17.7 KB of threadgroup memory
// per matrix) and 32 (4.7 KB), which is the occupancy experiment.
//   1. Householder tridiagonalization (ssytd2, lower), thread i owns row i
//   2. Q formed in place from the reflectors (sorg2r-style backward accumulation)
//   3. implicit QL with shifts on (d, e): one thread chases the bulge and
//      records the sweep's rotations; then every thread applies the whole
//      sequence to its own row of Z, so a sweep costs two barriers, not one
//      per rotation
//   4. rank sort, write w ascending and V (eigenvectors as columns, row-major)
#include <metal_stdlib>
using namespace metal;

#ifndef TDQL_NMAX
#define TDQL_NMAX 64
#endif
constant constexpr uint NMAX = TDQL_NMAX;
constant constexpr uint LD = NMAX + 1;   // padded rows: column walks hit different banks

// Profiling only: 1 stops after the tridiagonalization, 2 after forming Q,
// 3 after the QL iteration, 5 runs the QL iteration without updating Z;
// anything else (4) runs the whole kernel.
constant uint STAGE [[function_constant(0)]];

kernel void eigh_tdql(device const float* A    [[buffer(0)]],
                      device float*       W    [[buffer(1)]],
                      device float*       V    [[buffer(2)]],
                      device uint*        info [[buffer(3)]],
                      constant uint&      n    [[buffer(4)]],
                      uint mat  [[threadgroup_position_in_grid]],
                      uint tid  [[thread_index_in_threadgroup]],
                      uint ntg  [[threads_per_threadgroup]],
                      uint sg   [[simdgroup_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float a[NMAX * LD];
    threadgroup float d[NMAX], e[NMAX], vb[NMAX], pb[NMAX], cc[NMAX], ss[NMAX];
    threadgroup float red1[2], red2[2];
    threadgroup int   ql_lo, ql_cnt, ql_flag;
    threadgroup uint  perm[NMAX];

    device const float* Am = A + (size_t)mat * n * n;
    const uint nsg = (ntg + 31) / 32;

    // ---- load, lower triangle mirrored ----
    for (uint i = 0; i < n; ++i)
        for (uint j = tid; j < n; j += ntg) a[i * LD + j] = Am[i * n + j];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < n) for (uint j = tid + 1; j < n; ++j) a[tid * LD + j] = a[j * LD + tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint i = tid;   // this thread's row
    const bool row = i < n;

    // ---- 1. tridiagonalize ----
    for (uint k = 0; k + 2 < n; ++k) {
        float t = (row && i >= k + 2) ? a[i * LD + k] * a[i * LD + k] : 0.0f;
        t = simd_sum(t);
        if (lane == 0) red1[sg] = t;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float sumsq = 0.0f;
        for (uint s = 0; s < nsg; ++s) sumsq += red1[s];
        const float alpha = a[(k + 1) * LD + k];
        if (tid == 0) d[k] = a[k * LD + k];
        float tau = 0.0f, beta = alpha;
        if (sumsq > 0.0f) {
            beta = -copysign(sqrt(alpha * alpha + sumsq), alpha);
            tau = (beta - alpha) / beta;
            const float scal = 1.0f / (alpha - beta);
            if (row && i >= k + 2) { float v = a[i * LD + k] * scal; a[i * LD + k] = v; vb[i] = v; }
        } else if (row && i >= k + 2) {
            vb[i] = 0.0f;   // H = I; the column is already zero below the subdiagonal
        }
        if (tid == k + 1) vb[i] = 1.0f;
        if (tid == 0) { e[k] = beta; cc[k] = tau; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tau == 0.0f) continue;   // uniform
        // p = tau * A22 v, then w = p - (tau/2)(p.v) v
        float p = 0.0f;
        if (row && i >= k + 1) {
            for (uint j = k + 1; j < n; ++j) p += a[i * LD + j] * vb[j];
            p *= tau;
        }
        float pv = (row && i >= k + 1) ? p * vb[i] : 0.0f;
        pv = simd_sum(pv);
        if (lane == 0) red2[sg] = pv;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float pvs = 0.0f;
        for (uint s = 0; s < nsg; ++s) pvs += red2[s];
        const float w = p - 0.5f * tau * pvs * ((row && i >= k + 1) ? vb[i] : 0.0f);
        if (row && i >= k + 1) pb[i] = w;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (row && i >= k + 1) {
            const float vi = vb[i];
            for (uint j = k + 1; j < n; ++j) a[i * LD + j] -= vi * pb[j] + w * vb[j];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        if (n >= 2) { d[n - 2] = a[(n - 2) * LD + n - 2]; e[n - 2] = a[(n - 1) * LD + n - 2]; }
        d[n - 1] = a[(n - 1) * LD + n - 1];
        e[n - 1] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (STAGE == 1) { if (tid < n) W[(size_t)mat * n + tid] = d[tid] + e[tid]; return; }
    // ---- 2. Q in place: Q = H(0) ... H(n-3), backward ----
    // cc[k] holds tau_k (0 where the column needed no reflection).
    if (n >= 3) {
        if (tid == 0) a[(n - 1) * LD + n - 1] = 1.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int jj = (int)n - 3; jj >= 0; --jj) {
            const uint j = (uint)jj;
            const float tau = cc[j];
            if (row && i == j + 1) for (uint c = j + 2; c < n; ++c) a[i * LD + c] = 0.0f;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            // s_c = sum_{r >= j+2} v_r Q[r][c], c >= j+2 (Q[j+1][c] = 0)
            if (row && i >= j + 2) {
                const uint c = i;
                float s = 0.0f;
                for (uint r = j + 2; r < n; ++r) s += a[r * LD + j] * a[r * LD + c];
                pb[c] = s;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (row && i >= j + 1) {
                const float vi = (i == j + 1) ? 1.0f : a[i * LD + j];
                for (uint c = j + 2; c < n; ++c) a[i * LD + c] -= tau * vi * pb[c];
                a[i * LD + j + 1] = (i == j + 1) ? 1.0f - tau : -tau * vi;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (row) a[i * LD + 0] = (i == 0) ? 1.0f : 0.0f;
        if (tid == 0) for (uint c = 1; c < n; ++c) a[c] = 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    } else {
        if (row) for (uint c = 0; c < n; ++c) a[i * LD + c] = (i == c) ? 1.0f : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (STAGE == 2) { if (tid < n) W[(size_t)mat * n + tid] = a[tid * LD + tid] + a[tid * LD + (n - 1 - tid)]; return; }
    // ---- 3. implicit QL on (d, e), rotations applied to the rows of Z = a ----
    uint bad = 0;
    for (uint l = 0; l < n; ++l) {
        uint iter = 0;
        while (true) {
            if (tid == 0) {
                uint m = l;
                for (; m + 1 < n; ++m) {
                    const float dd = fabs(d[m]) + fabs(d[m + 1]);
                    if (fabs(e[m]) <= 1.2e-7f * dd) break;
                }
                if (m == l || iter >= 40) {
                    ql_flag = 1;
                    if (m != l) bad = 1;
                } else {
                    float g = (d[l + 1] - d[l]) / (2.0f * e[l]);
                    float r = sqrt(g * g + 1.0f);
                    g = d[m] - d[l] + e[l] / (g + copysign(r, g));
                    float s = 1.0f, c = 1.0f, p = 0.0f;
                    int ii = (int)m - 1;
                    int cnt = 0;
                    bool underflow = false;
                    for (; ii >= (int)l; --ii) {
                        const float f = s * e[ii], b = c * e[ii];
                        const float h = f * f + g * g;
                        if (h == 0.0f) { e[ii + 1] = 0.0f; d[ii + 1] -= p; e[m] = 0.0f; underflow = true; break; }
                        const float inv = rsqrt(h);
                        e[ii + 1] = h * inv;
                        s = f * inv; c = g * inv;
                        g = d[ii + 1] - p;
                        r = (d[ii] - g) * s + 2.0f * c * b;
                        p = s * r;
                        d[ii + 1] = g + p;
                        g = c * r - b;
                        cc[cnt] = c; ss[cnt] = s; ++cnt;
                    }
                    if (!underflow) { d[l] -= p; e[l] = g; e[m] = 0.0f; }
                    ql_lo = (int)m - 1;   // first rotation acts on columns (m-1, m)
                    ql_cnt = cnt;
                    ql_flag = 0;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (ql_flag) break;
            const int lo = ql_lo, cnt = ql_cnt;
            if (row && STAGE != 5) {
                threadgroup float* z = a + i * LD;
                for (int r = 0; r < cnt; ++r) {
                    const int col = lo - r;
                    const float c = cc[r], s = ss[r];
                    const float f = z[col + 1];
                    z[col + 1] = s * z[col] + c * f;
                    z[col] = c * z[col] - s * f;
                }
            }
            ++iter;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (STAGE == 3) { if (tid < n) W[(size_t)mat * n + tid] = d[tid] + a[tid * LD + tid]; return; }
    // ---- 4. sort ascending, write ----
    if (row) {
        const float di = d[i];
        uint rank = 0;
        for (uint k = 0; k < n; ++k) rank += (d[k] < di || (d[k] == di && k < i)) ? 1u : 0u;
        perm[rank] = i;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device float* Wm = W + (size_t)mat * n;
    device float* Vm = V + (size_t)mat * n * n;
    for (uint r = tid; r < n; r += ntg) Wm[r] = d[perm[r]];
    for (uint rr = 0; rr < n; ++rr)
        for (uint c = tid; c < n; c += ntg) Vm[rr * n + c] = a[rr * LD + perm[c]];
    if (tid == 0) info[mat] = bad;
}
