// The GPU chase tried on 2026-10-09 (docs/proposals/gpu-band-chase.md), as it
// stood when set aside; it needs Svd_Bidiag.metal's helpers (householder,
// sqrt1, kTinySumsq) above it.

// =============================================================================
// The symmetric band's bulge chase on the GPU (sb_chase, for the eigensolver's
// batches in two stages: svd_bidiag_batch.mm), band_chase.cpp's SymSweep a
// threadgroup a matrix. The lower half of the band (width 16, room for the
// bulges below it), A(r, c) for r >= c at W[c ld + r - c], ld >= 33, zeros
// outside the band. Sweep s's task t may run once sweep s - 1 has finished
// task t + 2 (the blocks they touch one apart), so the sweeps go in lockstep:
// at step tau sweep s runs task tau - 3 s, every sweep active at a step on a
// simdgroup of its own (or a second turn), a barrier between steps. Lanes take
// a block's rows; a reflector's norm and the dot products are simd sums. With
// rec, each reflector into bd_chase_apply's blocks as band_chase.cpp's
// SymSweep::keep writes it (L, kChaseBlockFloats = 832 a block, Ltau 16 a
// block). At the end d is the diagonal, e the subdiagonal, in W.
// =============================================================================
constant constexpr uint CH_KD = 16;
constant constexpr uint CH_SLOTS = 128;   // sweeps active at once, at most (n up to about 6000)
struct ChParams { uint n, ld, sw, rec, pmax, sl, stau, steps; };

// Sweep s's tasks: task 0, then for each block below, an odd task (into it)
// and an even one (its two sides).
static int ch_tasks(int s, int n) {
    const int ed0 = min(s + (int)CH_KD, n - 1);
    const int m = n - 1 > ed0 ? (n - 1 - ed0 + (int)CH_KD - 1) / (int)CH_KD : 0;
    return 2 * m + 1;
}

// x (lane k < len holds x_k) -> the reflector: u (u_0 = 1) into lanes'
// returned value, tau; x_0 becomes beta, the rest zero. A rest whose sum of
// squares is below kTinySumsq is zero (tau 0).
static float ch_reflector(thread float& x, int len, uint lane, thread float& tau) {
    const float alpha = simd_shuffle(x, (ushort)0);
    const float ss = simd_sum(lane >= 1 && (int)lane < len ? x * x : 0.0f);
    float u = lane == 0 ? 1.0f : 0.0f;
    tau = 0.0f;
    if (ss < kTinySumsq) {
        if (lane >= 1) x = 0.0f;
        return u;
    }
    float beta, scale;
    householder(alpha, sqrt1(ss), beta, tau, scale);
    if (lane == 0) x = beta;
    else if ((int)lane < len) { u = x * scale; x = 0.0f; }
    return u;
}

static void ch_keep(device float* L, device float* Lt, uint pmax, int s, int j, float u, int len, float tau, uint lane) {
    if (len < 2) return;
    const uint G = (uint)s / 16, c = (uint)s % 16;
    const uint b = G * (pmax + 1) - G * (G - 1) / 2 + (uint)j;
    const uint ct = c / 8, r0 = 8 * ct;
    device float* blk = L + (ulong)b * 832 + ct * 3 * 64 + c % 8;
    // lane l writes row r0 + l of the column, u_{r - c}
    const int idx = (int)(r0 + lane) - (int)c;
    const float ur = simd_shuffle(u, (ushort)clamp(idx, 0, 31));
    if (lane < 24) blk[(lane / 8) * 64 + (lane % 8) * 8] = idx >= 0 && idx < len ? ur : 0.0f;
    if (lane == 0) Lt[(ulong)b * 16 + c] = tau;
}

// The diagonal block A(st:ed, st:ed) = H A H, H = I - tau u u^T (u in lane k):
// w = tau A u - tau/2 (tau u^T A u) u, A -= u w^T + w u^T; lane r its row.
static void ch_two_sided(device float* W, uint ld, int st, int ed, float u, float tau, uint lane) {
    if (tau == 0.0f) return;
    const int m = ed - st + 1, r = (int)lane;
    const bool live = r < m;
    float a[CH_KD];
    for (int c = 0; c < (int)CH_KD; ++c) {
        float v = 0.0f;
        if (live && c < m) v = c <= r ? W[(ulong)(st + c) * ld + (uint)(r - c)] : W[(ulong)(st + r) * ld + (uint)(c - r)];
        a[c] = v;
    }
    float w = 0.0f;
    for (int c = 0; c < (int)CH_KD; ++c) w = fma(a[c], simd_shuffle(u, (ushort)c), w);
    w = live ? tau * w : 0.0f;
    const float dot = simd_sum(live ? w * u : 0.0f);
    w += -0.5f * tau * dot * u;
    for (int c = 0; c < (int)CH_KD; ++c) {
        const float wc = simd_shuffle(w, (ushort)c), uc = simd_shuffle(u, (ushort)c);
        if (live && c <= r) W[(ulong)(st + c) * ld + (uint)(r - c)] = a[c] - (u * wc + w * uc);
    }
}

kernel void sb_chase(device float* W0 [[buffer(0)]], device float* L0 [[buffer(1)]], device float* Lt0 [[buffer(2)]],
                     constant ChParams& p [[buffer(3)]], uint mat [[threadgroup_position_in_grid]],
                     uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                     uint nsg [[simdgroups_per_threadgroup]]) {
    device float* W = W0 + (ulong)mat * p.sw;
    device float* L = L0 + (ulong)mat * p.sl;
    device float* Lt = Lt0 + (ulong)mat * p.stau;
    threadgroup float U[CH_SLOTS][CH_KD + 1];
    threadgroup float TAU[CH_SLOTS];
    threadgroup int ST[CH_SLOTS], ED[CH_SLOTS];
    const int n = (int)p.n, ld = (int)p.ld;
    int s_lo = 0;
    for (int step = 0; step < (int)p.steps; ++step) {
        while (s_lo <= n - 2 && 3 * s_lo + ch_tasks(s_lo, n) <= step) ++s_lo;
        const int s_hi = min(step / 3, n - 2);
        for (int s = s_lo + (int)sg; s <= s_hi && !(p.rec & 2u); s += (int)nsg) {
            const int task = step - 3 * s, slot = s % (int)CH_SLOTS;
            if (task == 0) {
                const int st = s + 1, ed = min(s + (int)CH_KD, n - 1), len = ed - st + 1;
                float x = (int)lane < len ? W[(ulong)s * ld + (uint)(st - s) + lane] : 0.0f;
                float tau;
                const float u = ch_reflector(x, len, lane, tau);
                if ((int)lane < len) W[(ulong)s * ld + (uint)(st - s) + lane] = x;
                if (p.rec & 1u) ch_keep(L, Lt, p.pmax, s, 0, u, len, tau, lane);
                ch_two_sided(W, (uint)ld, st, ed, u, tau, lane);
                if (lane < CH_KD) U[slot][lane] = u;
                if (lane == 0) { TAU[slot] = tau; ST[slot] = st; ED[slot] = ed; }
            } else if (task % 2 == 1) {
                const int st = ST[slot], ed = ED[slot];
                const int j1 = ed + 1, j2 = min(ed + (int)CH_KD, n - 1), m = j2 - j1 + 1, nc = ed - st + 1;
                const float tau = TAU[slot];
                const float uk = lane < CH_KD ? U[slot][lane] : 0.0f;   // u_k in lane k
                // B = A(j1:j2, st:ed), lane i row i: B <- B H, H = I - tau u u^T
                const int i = (int)lane;
                const bool live = i < m;
                float b[CH_KD];
                for (int k = 0; k < (int)CH_KD; ++k)
                    b[k] = live && k < nc ? W[(ulong)(st + k) * ld + (uint)(j1 + i - st - k)] : 0.0f;
                if (tau != 0.0f) {
                    float w = 0.0f;
                    for (int k = 0; k < (int)CH_KD; ++k) w = fma(b[k], simd_shuffle(uk, (ushort)k), w);
                    for (int k = 0; k < (int)CH_KD; ++k) b[k] = fma(-tau * simd_shuffle(uk, (ushort)k), w, b[k]);
                }
                // B's first column's reflector, from the left on the rest of B
                float x = b[0], t2;
                const float u2 = ch_reflector(x, m, lane, t2);
                b[0] = live ? x : 0.0f;
                if (p.rec & 1u) ch_keep(L, Lt, p.pmax, s, (task + 1) / 2, u2, m, t2, lane);
                if (t2 != 0.0f)
                    for (int k = 1; k < (int)CH_KD; ++k) {
                        const float tk = simd_sum(live ? u2 * b[k] : 0.0f);
                        if (live) b[k] = fma(-t2 * tk, u2, b[k]);
                    }
                for (int k = 0; k < (int)CH_KD; ++k)
                    if (live && k < nc) W[(ulong)(st + k) * ld + (uint)(j1 + i - st - k)] = b[k];
                if (lane < CH_KD) U[slot][lane] = u2;
                if (lane == 0) { TAU[slot] = t2; ST[slot] = j1; ED[slot] = j2; }
            } else {
                const float uk = lane < CH_KD ? U[slot][lane] : 0.0f;
                ch_two_sided(W, (uint)ld, ST[slot], ED[slot], uk, TAU[slot], lane);
            }
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    }
}
