// QR on the GPU by panels of b columns (8 or 16, see shape()), a batch at
// once, the work matrix products:
//
//   1. each panel factored by the band reduction's kernels (in one
//      simdgroup, or by TSQR for a tall one; shaders/Svd_Bidiag.metal), its
//      H = I - V T V^T applied to the rest of its aggregate of 128 columns as
//      two MPS products, W = (V T)^T C and C -= V W;
//   2. each aggregate's T merged from its panels' (bd_merge_t, from the Gram
//      matrix Y^T Y), and I - Y Ta Y^T applied to the columns right of it as
//      three: Z = Y^T C, W = Ta^T Z, C -= Y W;
//   3. Q formed from [I; 0] by the aggregates backwards, Q <- (I - Y Ta Y^T)
//      Q on Q's trailing rows and columns, three products each.
//
// The matrices are padded with zero rows and columns to whole panels with
// twice their width in rows (which changes neither R nor Q), so every column
// is the GPU's. Each is factored row-major in place: the panels read a row's
// b entries contiguously, the products take the row-major views as they
// are, and neither the input nor Q is transposed. Every kernel takes the
// batch as its grid's z and every product is one batched MPS product. The
// forward pass is committed a panel, then an aggregate, at a time, so that
// the GPU starts while the CPU encodes the rest; step 3 is queued at once,
// behind an event the CPU signals once it has written Q's start. Q (one
// matrix) and R go straight to the caller's memory where it is page-aligned.
//
// On an M5 Pro one 4096 x 4096 in 62 ms against the streaming kernels' 231
// (and 634 on the CPU), 16 x 1024 x 1024 in 25 against 52 (CPU 48).
#ifndef ACCELERATE_NEW_LAPACK
#define ACCELERATE_NEW_LAPACK
#endif
#include <metal_linalg/core.h>
#include "metal_runtime.h"

#include <Accelerate/Accelerate.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

using metal_linalg::detail::Part;
using metal_linalg::core::QrMode;
using metal_linalg::detail::QrStore;

namespace metal_linalg::core::detail {
namespace {

// The padded problem: panels of b columns over K rounded up to b (Kp), and
// at least Kp + b rows, so that every panel has twice its width in rows (the
// panel kernels' need); zero rows and columns, which change neither R nor Q.
struct Shape {
    uint32_t b = 0, Kp = 0, mp = 0, np = 0;
};

// Panels of 8 columns for up to 4 matrices of 768 to 3072 rows and at least
// 768 columns, else 16. A tall panel's TSQR top is a tree of b-step chains
// whose steps are up to b long, a third of the call at 1024 x 1024 with 16;
// with 8 its chains are a quarter as long, for twice as many panels and
// in-aggregate updates. On an M5 Pro 8 took 1.05x less time at 768^2 and
// 1024^2, 1.1x at 1536^2 and 2048^2, 1.04x at 3072^2, 1.08x at 2048 x 1024
// and 1.06x for 4 of 1024^2; it lost at 512^2 (0.96x), 4096 x 2048 (0.98x),
// 4096 x 512 (0.92x) and for batches of 16 or more (0.94x at 64 x 256^2).
// (32 took 1.4-1.5x 16's time at 512-2048.)
Shape shape(uint32_t m, uint32_t n, uint32_t batch) {
    const uint32_t K = std::min(m, n);
    Shape s;
    s.b = batch <= 4 && K >= 768 && m <= 3072 ? 8 : 16;
    if (const char* e = std::getenv("QR_PANEL_WIDTH")) {
        const std::string w = e;
        if (w == "8" || w == "16") s.b = (uint32_t)std::stoul(w);
    }
    s.Kp = (K + s.b - 1) / s.b * s.b;
    s.mp = std::max(m, s.Kp + s.b);
    s.np = std::max(n, s.Kp);
    return metal_linalg::detail::qr_block_width(s.mp) ? s : Shape{};
}

// The latest shape's buffers, for up to `capacity` matrices, a row of slack
// past the products' operands.
struct Work {
    uint32_t      m = 0, n = 0, qc = 0, capacity = 0, b = 0;
    id<MTLBuffer> A, Q, scale, up, bits;   // shared: the CPU writes Q's start and reads R when they are not the caller's
    QrStore       st{};
};

// Floats a matrix of the batch takes in the workspace, Q qc columns.
size_t per_matrix(uint32_t m, uint32_t n, uint32_t qc, const Shape& sh) {
    return (size_t)sh.mp * sh.np + (size_t)sh.mp * sh.Kp + (size_t)sh.mp * 32 + (size_t)m * qc +
           2 * (size_t)128 * std::max(sh.np, qc) + metal_linalg::detail::qr_scratch_floats(sh.mp) +
           (size_t)sh.Kp * 1024 / sh.b + ((size_t)sh.Kp / 128 + 1) * 128 * 128 + 128 * 128;
}

// qc: Q's columns (K, M for Q square, 0 for R alone). A workspace for more
// columns serves fewer (ldw is a minimum), so calls alternating modes on a
// shape do not rebuild it each time.
Work& workspace(uint32_t m, uint32_t n, uint32_t qc, const Shape& sh, uint32_t batch) {
    static Work w;
    if (w.A && w.m == m && w.n == n && w.qc >= qc && w.capacity >= batch && w.b == sh.b) return w;
    id<MTLDevice> dev = metal_linalg::detail::qr_device();
    const uint32_t K = std::min(m, n), blocks = sh.Kp / sh.b;
    auto buffer = [&](size_t floats, MTLResourceOptions opt) {
        id<MTLBuffer> buf = [dev newBufferWithLength:std::max<size_t>(floats, 1) * 4 options:opt];
        if (!buf) throw std::runtime_error("[qr] blocked: could not allocate " + std::to_string(floats * 4) + " bytes");
        return buf;
    };
    w = Work{};
    w.m = m;
    w.n = n;
    w.qc = qc;
    w.capacity = batch;
    w.b = sh.b;
    QrStore& st = w.st;
    st.ldv = sh.Kp;
    st.ldw = std::max(sh.np, qc);   // the products' scratch, as wide as Q's columns too
    st.sa = (size_t)sh.mp * sh.np;
    st.sv = (size_t)sh.mp * sh.Kp;
    st.svt = (size_t)sh.mp * 32;
    st.st = (size_t)blocks * 1024;
    st.sta = ((size_t)sh.Kp / 128 + 1) * 128 * 128;
    st.sg = 128 * 128;
    st.sz = (size_t)128 * st.ldw;
    st.ssc = metal_linalg::detail::qr_scratch_floats(sh.mp);
    const MTLResourceOptions priv = MTLResourceStorageModePrivate, shared = MTLResourceStorageModeShared;
    // A stride of slack past the products' operands (mps()'s batched views)
    w.A = buffer((batch + 1) * st.sa, shared);
    w.Q = buffer((size_t)(batch + 1) * m * qc, shared);
    w.scale = buffer(batch, shared);
    w.up = buffer(batch, shared);
    w.bits = buffer(batch, shared);
    st.v = buffer((batch + 1) * st.sv, priv);
    st.vt = buffer((batch + 1) * st.svt, priv);
    st.t = buffer(batch * st.st, priv);
    st.ta = buffer((batch + 1) * st.sta, priv);
    st.g = buffer((batch + 1) * st.sg, priv);
    st.z = buffer((batch + 1) * st.sz, priv);
    st.z2 = buffer((batch + 1) * st.sz, priv);
    st.sc = buffer(batch * st.ssc, priv);
    return w;
}

void check(id<MTLCommandBuffer> cb) {
    [cb waitUntilCompleted];
    if (cb.error)
        throw std::runtime_error(std::string("[qr] blocked: GPU error: ") + cb.error.localizedDescription.UTF8String);
}

// `batch` matrices at once.
void run(const float* a, uint32_t batch, uint32_t m, uint32_t n, float* q_out, float* r_out, QrMode mode) {
    const uint32_t K = std::min(m, n);
    // Q's columns (K, M, or none) and R's rows (K, or M with zeros below K)
    const uint32_t QC = core::qr_q_cols(mode, m, n), RR = core::qr_r_rows(mode, m, n);
    const Shape sh = shape(m, n, batch);
    const size_t per = (size_t)m * n;
    Work& w = workspace(m, n, QC, sh, batch);
    w.st.batch = batch;
    // Each matrix scaled by a power of two into [0.5, 1), as the panel
    // kernels' plain sums of squares need; R scaled back on the way out. A
    // matrix holding a NaN or an infinity gives NaN, written at the end.
    // Where the input is used in place, its scan is the GPU's too (qr_scan):
    // the GPU starts at once (QR_GPU_SCAN=0 keeps it on the CPU).
    const size_t page = (size_t)getpagesize();
    static const bool cpu_scan = std::getenv("QR_GPU_SCAN") && std::string(std::getenv("QR_GPU_SCAN")) == "0";
    const bool gpu_scan = reinterpret_cast<uintptr_t>(a) % page == 0 && !cpu_scan;
    std::vector<char> finite(batch, 1);
    float* down = static_cast<float*>(w.scale.contents);
    float* up = static_cast<float*>(w.up.contents);
    if (gpu_scan) {
        std::memset(w.bits.contents, 0, batch * sizeof(uint32_t));
    } else {
        std::vector<float> amax(batch);
        metal_linalg::detail::scan(Matrices{a, batch, m, n}, Part::all, amax.data(), finite.data());
        for (uint32_t i = 0; i < batch; ++i) {
            down[i] = up[i] = 1.0f;
            if (finite[i] && amax[i] > 0.0f) {
                int e = 0;
                std::frexp(amax[i], &e);
                down[i] = std::ldexp(1.0f, -e);
                up[i] = std::ldexp(1.0f, e);
            }
        }
    }
    id<MTLCommandQueue> queue = metal_linalg::detail::qr_queue();
    id<MTLDevice> dev = metal_linalg::detail::qr_device();
    float* A = static_cast<float*>(w.A.contents);

    // Q formed in the caller's memory if it is page-aligned and one matrix
    // (a batch's views need the slack past it), R too below
    id<MTLBuffer> Q = w.Q;
    if (QC && reinterpret_cast<uintptr_t>(q_out) % page == 0 && batch == 1)
        Q = metal_linalg::detail::wrap_host(dev, q_out, (size_t)batch * m * QC);
    float* Qh = static_cast<float*>(Q.contents);

    @autoreleasepool {
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        if (reinterpret_cast<uintptr_t>(a) % page == 0) {
            id<MTLBuffer> in = [dev newBufferWithBytesNoCopy:(void*)a
                                                      length:((size_t)batch * per * 4 + page - 1) / page * page
                                                     options:MTLResourceStorageModeShared deallocator:nil];
            if (!in) throw std::runtime_error("[qr] blocked: could not wrap the input");
            if (gpu_scan) metal_linalg::detail::qr_scan_scales(cb, in, w.bits, w.scale, w.up, per, batch);
            metal_linalg::detail::qr_scale_copy(cb, in, w.A, m, n, sh.mp, sh.np, w.st.sa, batch, w.scale);
        } else {
            metal_linalg::detail::for_each_rows(batch, sh.mp, sh.np, [&](uint32_t i, uint32_t r0, uint32_t r1) {
                for (uint32_t r = r0; r < r1; ++r) {
                    float* ar = A + i * w.st.sa + (size_t)r * sh.np;
                    if (r < m) vDSP_vsmul(a + i * per + (size_t)r * n, 1, &down[i], ar, 1, n);
                    std::fill(ar + (r < m ? n : 0), ar + sh.np, 0.0f);
                }
            });
        }
        std::vector<id<MTLCommandBuffer>> forward;
        metal_linalg::detail::qr_blocks(cb, w.A, sh.mp, sh.np, sh.np, sh.b, w.st, forward);
        [cb commit];
        forward.push_back(cb);
        // R's copy and Q's formation queued behind an event that the CPU
        // signals once Q's start is written (whatever happens: a command
        // buffer left waiting would hold the queue)
        id<MTLSharedEvent> ready = [dev newSharedEvent];
        struct Release {
            id<MTLSharedEvent> event;
            ~Release() { event.signaledValue = 1; }
        } release{ready};
        id<MTLCommandBuffer> back = [queue commandBuffer];
        [back encodeWaitForEvent:ready value:1];
        const bool r_gpu = reinterpret_cast<uintptr_t>(r_out) % page == 0;
        if (r_gpu)
            metal_linalg::detail::qr_r_out(back, w.A,
                                           metal_linalg::detail::wrap_host(dev, r_out, (size_t)batch * RR * n), K, n,
                                           sh.np, w.st.sa, batch, w.up, (size_t)RR * n);
        // Q from [I; 0], its QC columns (none for R alone)
        if (QC) metal_linalg::detail::qr_blocks_apply(back, Q, m, QC, QC, sh.b, sh.Kp / sh.b, w.st);
        [back commit];
        // Q's start, [I; 0], while the GPU factors; a square Q's R's zero
        // rows below K
        if (QC)
            metal_linalg::detail::for_each_rows(batch, m, QC, [&](uint32_t i, uint32_t r0, uint32_t r1) {
                float* qi = Qh + (size_t)i * m * QC;
                std::memset(qi + (size_t)r0 * QC, 0, (size_t)(r1 - r0) * QC * 4);
                for (uint32_t r = r0; r < std::min(r1, QC); ++r) qi[(size_t)r * QC + r] = 1.0f;
            });
        if (RR > K)
            for (uint32_t i = 0; i < batch; ++i)
                std::memset(r_out + (size_t)i * RR * n + (size_t)K * n, 0, (size_t)(RR - K) * n * 4);
        ready.signaledValue = 1;
        for (id<MTLCommandBuffer> f : forward) check(f);
        if (!r_gpu)   // R, unscaled, while the GPU forms Q
            metal_linalg::detail::for_each_rows(batch, K, n, [&](uint32_t i, uint32_t r0, uint32_t r1) {
                for (uint32_t r = r0; r < r1; ++r) {
                    float* rr = r_out + (size_t)i * RR * n + (size_t)r * n;
                    std::fill(rr, rr + r, 0.0f);
                    vDSP_vsmul(A + i * w.st.sa + (size_t)r * sh.np + r, 1, &up[i], rr + r, 1, n - r);
                }
            });
        check(back);
    }
    if (QC && Qh != q_out)
        std::memcpy(q_out, Qh, (size_t)batch * m * QC * 4);
    if (gpu_scan) {
        const uint32_t* bits = static_cast<const uint32_t*>(w.bits.contents);
        for (uint32_t i = 0; i < batch; ++i) finite[i] = bits[i] < 0x7f800000u;
    }
    for (uint32_t i = 0; i < batch; ++i)
        if (!finite[i]) {
            if (QC) std::fill(q_out + (size_t)i * m * QC, q_out + (size_t)(i + 1) * m * QC, NAN);
            std::fill(r_out + (size_t)i * RR * n, r_out + (size_t)(i + 1) * RR * n, NAN);
        }
}

} // namespace

bool qr_blocked_fits(uint32_t m, uint32_t n) { return std::min(m, n) >= 1 && shape(m, n, 1).b != 0; }

// A batch at once, the blocked QR beats the streaming kernels at every shape
// and batch measured on an M5 Pro (1.8-3.5x: 16 x 1024 x 1024 in 25 ms
// against 52, 1024 x 128 x 128 in 18 against 52), so they keep only what it
// cannot take. QR_BLOCKED=0 turns it off.
bool qr_blocked_preferred(uint32_t m, uint32_t n, uint32_t batch) {
    if (const char* e = std::getenv("QR_BLOCKED"); e && std::string(e) == "0") return false;
    (void)batch;
    return qr_blocked_fits(m, n);
}

void qr_blocked(const Matrices& a, float* q, float* r, QrMode mode) {
    const uint32_t M = a.rows, N = a.cols, K = std::min(M, N);
    if (K == 0 || a.batch == 0) return;
    if (!qr_blocked_fits(M, N))
        throw std::invalid_argument("[qr] blocked: " + std::to_string(M) + " rows is more than its panels take");
    const uint32_t QC = core::qr_q_cols(mode, M, N), RR = core::qr_r_rows(mode, M, N);
    // In chunks of at most about 1 GB of workspace
    const size_t per = per_matrix(M, N, QC, shape(M, N, 1));   // 8-wide panels: the larger
    const uint32_t chunk = (uint32_t)std::clamp<size_t>(((size_t)1 << 28) / per, 1, a.batch);
    for (uint32_t b0 = 0; b0 < a.batch; b0 += chunk) {
        const uint32_t count = std::min(chunk, a.batch - b0);
        run(a.data + (size_t)b0 * M * N, count, M, N, QC ? q + (size_t)b0 * M * QC : nullptr,
            r + (size_t)b0 * RR * N, mode);
    }
}

void qr_blocked(const Matrices& a, float* q, float* r) { qr_blocked(a, q, r, QrMode::reduced); }

} // namespace metal_linalg::core::detail
