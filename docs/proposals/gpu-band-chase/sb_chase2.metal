// Two sweeps a simdgroup, tried after sb_chase on 2026-10-09 and slower
// (docs/proposals/gpu-band-chase.md); it needs sb_chase.metal above it.

// Two sweeps a simdgroup (16 lanes each: sweeps s and s + 2, the same task parity),
// sums within a half by shuffles, the odd task's 15 column sums transposed.
static float hs(float x) {
    x += simd_shuffle_xor(x, (ushort)8); x += simd_shuffle_xor(x, (ushort)4);
    x += simd_shuffle_xor(x, (ushort)2); x += simd_shuffle_xor(x, (ushort)1);
    return x;
}
static float c2_reflector(thread float& x, int len, uint r, uint base, thread float& tau) {
    const float alpha = simd_shuffle(x, (ushort)base);
    const float ss = hs(r >= 1 && (int)r < len ? x * x : 0.0f);
    float u = r == 0 ? 1.0f : 0.0f;
    tau = 0.0f;
    if (ss < kTinySumsq) { if (r >= 1) x = 0.0f; return u; }
    float beta, scale;
    householder(alpha, sqrt1(ss), beta, tau, scale);
    if (r == 0) x = beta;
    else if ((int)r < len) { u = x * scale; x = 0.0f; }
    return u;
}
static void c2_keep(device float* L, device float* Lt, uint pmax, int s, int j, float u, int len, float tau, uint r, uint base, bool act) {
    const uint G = (uint)s / 16, c = (uint)s % 16;
    const uint b = G * (pmax + 1) - G * (G - 1) / 2 + (uint)j;
    const uint ct = c / 8, r0 = 8 * ct;
    device float* blk = L + (ulong)b * 832 + ct * 3 * 64 + c % 8;
    for (uint l = r; l < 32; l += 16) {
        const int idx = (int)(r0 + l) - (int)c;
        const float ur = simd_shuffle(u, (ushort)(base | (uint)clamp(idx, 0, 15)));
        if (act && len >= 2 && l < 24) blk[(l / 8) * 64 + (l % 8) * 8] = idx >= 0 && idx < len ? ur : 0.0f;
    }
    if (act && len >= 2 && r == 0) Lt[(ulong)b * 16 + c] = tau;
}
static void c2_two_sided(device float* W, uint ld, int st, int ed, float u, float tau, uint r, uint base, bool act) {
    const int m = ed - st + 1;
    const bool live = act && (int)r < m && tau != 0.0f;
    float a[CH_KD];
    for (int c = 0; c < (int)CH_KD; ++c) {
        float v = 0.0f;
        if (live && c < m) v = c <= (int)r ? W[(ulong)(st + c) * ld + (r - c)] : W[(ulong)(st + r) * ld + (c - r)];
        a[c] = v;
    }
    float w = 0.0f;
    for (int c = 0; c < (int)CH_KD; ++c) w = fma(a[c], simd_shuffle(u, (ushort)(base | c)), w);
    w = live ? tau * w : 0.0f;
    const float dot = hs(w * u);
    w += -0.5f * tau * dot * u;
    for (int c = 0; c < (int)CH_KD; ++c) {
        const float wc = simd_shuffle(w, (ushort)(base | c)), uc = simd_shuffle(u, (ushort)(base | c));
        if (live && c <= (int)r) W[(ulong)(st + c) * ld + (r - c)] = a[c] - (u * wc + w * uc);
    }
}
kernel void sb_chase2(device float* W0 [[buffer(0)]], device float* L0 [[buffer(1)]], device float* Lt0 [[buffer(2)]],
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
    const uint h = lane >> 4, r = lane & 15u, base = lane & 16u;
    const bool rec = p.rec & 1u;
    int s_lo = 0;
    for (int step = 0; step < (int)p.steps; ++step) {
        while (s_lo <= n - 2 && 3 * s_lo + ch_tasks(s_lo, n) <= step) ++s_lo;
        const int s_hi = min(step / 3, n - 2);
        for (int P = (int)sg; ; P += (int)nsg) {
            const int s0 = s_lo + 4 * (P / 2) + (P % 2);
            if (s0 > s_hi) break;
            const int s = s0 + 2 * (int)h;
            const bool act = s <= s_hi;
            const int task = step - 3 * s, slot = s % (int)CH_SLOTS;
            // the pair's tasks share a parity; task 0 (only the first sweep's
            // first task, or the second's) branches on its own
            const int kind = !act ? -1 : task == 0 ? 0 : (task & 1) ? 1 : 2;
            const int k0 = simd_shuffle(kind, (ushort)0), k1 = simd_shuffle(kind, (ushort)16);
            for (int pass = 0; pass < 2; ++pass) {
                const int want = pass == 0 ? k0 : k1;
                if (pass == 1 && (k1 == k0 || k1 < 0)) break;
                if (want < 0) continue;
                const bool mine = act && kind == want;
                if (want == 0) {
                    const int st = s + 1, ed = min(s + (int)CH_KD, n - 1), len = ed - st + 1;
                    float x = mine && (int)r < len ? W[(ulong)s * ld + (uint)(st - s) + r] : 0.0f;
                    float tau;
                    const float u = c2_reflector(x, mine ? len : 0, r, base, tau);
                    if (mine && (int)r < len) W[(ulong)s * ld + (uint)(st - s) + r] = x;
                    if (rec) c2_keep(L, Lt, p.pmax, s, 0, u, len, tau, r, base, mine);
                    c2_two_sided(W, (uint)ld, st, ed, u, tau, r, base, mine);
                    if (mine) { U[slot][r] = u; if (r == 0) { TAU[slot] = tau; ST[slot] = st; ED[slot] = ed; } }
                } else if (want == 1) {
                    const int st = mine ? ST[slot] : 0, ed = mine ? ED[slot] : -1;
                    const int j1 = ed + 1, j2 = min(ed + (int)CH_KD, n - 1), m = mine ? j2 - j1 + 1 : 0, nc = ed - st + 1;
                    const float tau = mine ? TAU[slot] : 0.0f;
                    const float uk = mine ? U[slot][r] : 0.0f;
                    const int i = (int)r;
                    const bool live = i < m;
                    float b[CH_KD], uu[CH_KD];
                    for (int k = 0; k < (int)CH_KD; ++k) uu[k] = simd_shuffle(uk, (ushort)(base | k));
                    for (int k = 0; k < (int)CH_KD; ++k)
                        b[k] = live && k < nc ? W[(ulong)(st + k) * ld + (uint)(j1 + i - st - k)] : 0.0f;
                    float w = 0.0f;
                    for (int k = 0; k < (int)CH_KD; ++k) w = fma(b[k], uu[k], w);
                    w *= tau;
                    for (int k = 0; k < (int)CH_KD; ++k) b[k] = fma(-uu[k], w, b[k]);
                    float x = b[0], t2;
                    const float u2 = c2_reflector(x, m, r, base, t2);
                    b[0] = live ? x : 0.0f;
                    if (rec) c2_keep(L, Lt, p.pmax, s, (task + 1) / 2, u2, m, t2, r, base, mine);
                    // t_k = sum_i u2_i b_i[k], reduce-scattered: lane r ends with t_r
                    float v[CH_KD];
                    for (int k = 0; k < (int)CH_KD; ++k) v[k] = live ? u2 * b[k] : 0.0f;
                    {
                        const bool b3 = r & 8u, b2 = r & 4u, b1 = r & 2u, b0 = r & 1u;
                        for (int q = 0; q < 8; ++q) { const float snd = b3 ? v[q] : v[q + 8], kp = b3 ? v[q + 8] : v[q]; v[q] = kp + simd_shuffle_xor(snd, (ushort)8); }
                        for (int q = 0; q < 4; ++q) { const float snd = b2 ? v[q] : v[q + 4], kp = b2 ? v[q + 4] : v[q]; v[q] = kp + simd_shuffle_xor(snd, (ushort)4); }
                        for (int q = 0; q < 2; ++q) { const float snd = b1 ? v[q] : v[q + 2], kp = b1 ? v[q + 2] : v[q]; v[q] = kp + simd_shuffle_xor(snd, (ushort)2); }
                        { const float snd = b0 ? v[0] : v[1], kp = b0 ? v[1] : v[0]; v[0] = kp + simd_shuffle_xor(snd, (ushort)1); }
                    }
                    const float tr = -t2 * v[0];
                    for (int k = 1; k < (int)CH_KD; ++k) b[k] = fma(simd_shuffle(tr, (ushort)(base | k)), u2, b[k]);
                    for (int k = 0; k < (int)CH_KD; ++k)
                        if (live && k < nc) W[(ulong)(st + k) * ld + (uint)(j1 + i - st - k)] = b[k];
                    if (mine) { U[slot][r] = u2; if (r == 0) { TAU[slot] = t2; ST[slot] = j1; ED[slot] = j2; } }
                } else {
                    const float uk = mine ? U[slot][r] : 0.0f;
                    c2_two_sided(W, (uint)ld, mine ? ST[slot] : 0, mine ? ED[slot] : -1, uk, mine ? TAU[slot] : 0.0f, r, base, mine);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    }
}
