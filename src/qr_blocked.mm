// QR of large matrices on the GPU by panels of b columns (16 up to 8192 rows,
// 8 up to 16384), the work matrix products:
//
//   1. each panel factored by the band reduction's kernels (in one
//      simdgroup, or by TSQR for a tall one; shaders/Svd_Bidiag.metal), its
//      H = I - V T V^T applied to the rest of its aggregate of 128 columns as
//      two MPS products, W = (V T)^T C and C -= V W;
//   2. each aggregate's T merged from its panels' (bd_merge_t, from the Gram
//      matrix Y^T Y), and I - Y Ta Y^T applied to the columns right of it as
//      three: Z = Y^T C, W = Ta^T Z, C -= Y W;
//   3. the last columns (fewer than b, or fewer than 2 b rows) by LAPACK on
//      the CPU, sgeqrf and sorgqr;
//   4. Q formed from [I 0; 0 Q_tail] by the aggregates backwards, Q <- (I -
//      Y Ta Y^T) Q on Q's trailing rows and columns, three products each.
//
// The matrix is factored row-major in place: the panels read each row's b
// entries contiguously, the products take the row-major views as they are,
// and neither the input nor Q is transposed. The forward pass is committed an
// aggregate or two at a time, so that the GPU starts while the CPU encodes
// the rest; step 4 is queued at once, behind an event the CPU signals after
// step 3; Q's initial zeros are written while the GPU factors. Q and R go
// straight to the caller's memory where it is page-aligned.
//
// Where qr_streaming_amx_reduced's grid streams a 32-column tile of the
// trailing matrix through a threadgroup for every panel, the products here
// run at MPS's rate: on an M5 Pro one 4096 x 4096 in 63 ms against 231 (and
// 634 on the CPU), 1024 x 1024 in 7 against 15.
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
#include <cstring>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

using metal_linalg::detail::Part;
using metal_linalg::detail::QrStore;

namespace metal_linalg::core::detail {
namespace {

using L = __LAPACK_int;

// The latest shape's buffers, a row of slack past the products' operands.
struct Work {
    uint32_t      m = 0, n = 0;
    id<MTLBuffer> A, Q;   // shared: the CPU writes the tail and reads R
    QrStore       st{};
};

Work& workspace(uint32_t m, uint32_t n, uint32_t b) {
    static Work w;
    if (w.A && w.m == m && w.n == n && w.st.ldv % b == 0) return w;
    id<MTLDevice> dev = metal_linalg::detail::qr_device();
    const uint32_t K = std::min(m, n), Kp = (K + b - 1) / b * b, blocks = Kp / b;
    auto buffer = [&](size_t floats, MTLResourceOptions opt) {
        id<MTLBuffer> buf = [dev newBufferWithLength:std::max<size_t>(floats, 1) * 4 options:opt];
        if (!buf) throw std::runtime_error("[qr] blocked: could not allocate " + std::to_string(floats * 4) + " bytes");
        return buf;
    };
    w = Work{};
    w.m = m;
    w.n = n;
    w.A = buffer((size_t)m * n + n, MTLResourceStorageModeShared);
    w.Q = buffer((size_t)m * K + K, MTLResourceStorageModeShared);
    w.st.ldv = Kp;
    w.st.ldw = std::max(n, K);
    w.st.v = buffer((size_t)m * Kp + Kp, MTLResourceStorageModePrivate);
    w.st.t = buffer((size_t)blocks * 1024, MTLResourceStorageModePrivate);
    w.st.ta = buffer(((size_t)Kp / 128 + 1) * 128 * 128, MTLResourceStorageModePrivate);
    w.st.g = buffer(128 * 128 + 128, MTLResourceStorageModePrivate);
    w.st.z = buffer((size_t)128 * w.st.ldw + w.st.ldw, MTLResourceStorageModePrivate);
    w.st.z2 = buffer((size_t)128 * w.st.ldw + w.st.ldw, MTLResourceStorageModePrivate);
    return w;
}

void check(id<MTLCommandBuffer> cb) {
    [cb waitUntilCompleted];
    if (cb.error)
        throw std::runtime_error(std::string("[qr] blocked: GPU error: ") + cb.error.localizedDescription.UTF8String);
}

// The columns from kt on, A(kt:, kt:), by LAPACK: R into A's rows, and Q's
// block Q(kt:, kt:K) (ld ldq).
void tail(float* A, uint32_t m, uint32_t n, uint32_t kt, float* Q, uint32_t ldq) {
    const uint32_t K = std::min(m, n);
    if (kt >= K) return;
    L mt = (L)(m - kt), nt = (L)(n - kt), kk = (L)(K - kt), info = 0, lwork = -1;
    std::vector<float> c((size_t)mt * nt), tau(kk);
    for (L i = 0; i < mt; ++i)
        for (L j = 0; j < nt; ++j) c[i + (size_t)j * mt] = A[(size_t)(kt + i) * n + kt + j];
    float q = 0.0f;
    sgeqrf_(&mt, &nt, c.data(), &mt, tau.data(), &q, &lwork, &info);
    float q2 = 0.0f;
    sorgqr_(&mt, &kk, &kk, c.data(), &mt, tau.data(), &q2, &lwork, &info);
    lwork = std::max<L>(1, (L)std::max(q, q2));
    std::vector<float> work(lwork);
    sgeqrf_(&mt, &nt, c.data(), &mt, tau.data(), work.data(), &lwork, &info);
    if (info) throw std::runtime_error("[qr] blocked: LAPACK sgeqrf failed, info " + std::to_string((long long)info));
    for (L i = 0; i < kk; ++i)
        for (L j = i; j < nt; ++j) A[(size_t)(kt + i) * n + kt + j] = c[i + (size_t)j * mt];
    sorgqr_(&mt, &kk, &kk, c.data(), &mt, tau.data(), work.data(), &lwork, &info);
    if (info) throw std::runtime_error("[qr] blocked: LAPACK sorgqr failed, info " + std::to_string((long long)info));
    for (L i = 0; i < mt; ++i)
        for (L j = 0; j < kk; ++j) Q[(size_t)(kt + i) * ldq + kt + j] = c[i + (size_t)j * mt];
}

void one(const float* a, uint32_t m, uint32_t n, float* q_out, float* r_out) {
    const uint32_t K = std::min(m, n), b = metal_linalg::detail::qr_block_width(m);
    float amax = 0.0f;
    char finite = 1;
    const Matrices am{a, 1, m, n};
    metal_linalg::detail::scan(am, Part::all, &amax, &finite);
    if (!finite) {
        std::fill(q_out, q_out + (size_t)m * K, NAN);
        std::fill(r_out, r_out + (size_t)K * n, NAN);
        return;
    }
    // Scaled by a power of two into [0.5, 1), as the panel kernels' plain
    // sums of squares need; R scaled back on the way out.
    float down = 1.0f, up = 1.0f;
    if (amax > 0.0f) {
        int e = 0;
        std::frexp(amax, &e);
        down = std::ldexp(1.0f, -e);
        up = std::ldexp(1.0f, e);
    }
    Work& w = workspace(m, n, b);
    id<MTLCommandQueue> queue = metal_linalg::detail::qr_queue();
    id<MTLDevice> dev = metal_linalg::detail::qr_device();
    const size_t page = (size_t)getpagesize(), per = (size_t)m * n;
    float* A = static_cast<float*>(w.A.contents);

    // Q formed in the caller's memory if it is page-aligned (R too, below)
    id<MTLBuffer> Q = w.Q;
    if (reinterpret_cast<uintptr_t>(q_out) % page == 0)
        Q = metal_linalg::detail::wrap_host(dev, q_out, (size_t)m * K);
    float* Qh = static_cast<float*>(Q.contents);

    @autoreleasepool {
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        if (reinterpret_cast<uintptr_t>(a) % page == 0) {
            id<MTLBuffer> in = [dev newBufferWithBytesNoCopy:(void*)a length:(per * 4 + page - 1) / page * page
                                                     options:MTLResourceStorageModeShared deallocator:nil];
            if (!in) throw std::runtime_error("[qr] blocked: could not wrap the input");
            metal_linalg::detail::qr_scale_copy(cb, in, w.A, per, down);
        } else {
            metal_linalg::detail::for_each_rows(1, m, n, [&](uint32_t, uint32_t r0, uint32_t r1) {
                vDSP_vsmul(a + (size_t)r0 * n, 1, &down, A + (size_t)r0 * n, 1, (size_t)(r1 - r0) * n);
            });
        }
        std::vector<id<MTLCommandBuffer>> forward;
        const uint32_t done = metal_linalg::detail::qr_blocks(cb, w.A, m, n, n, b, w.st, forward);
        [cb commit];
        forward.push_back(cb);
        // R's copy and Q's formation queued behind an event that the CPU
        // signals once it has done the last columns (whatever happens: a
        // command buffer left waiting would hold the queue)
        id<MTLSharedEvent> ready = [dev newSharedEvent];
        struct Release {
            id<MTLSharedEvent> event;
            ~Release() { event.signaledValue = 1; }
        } release{ready};
        id<MTLCommandBuffer> back = [queue commandBuffer];
        [back encodeWaitForEvent:ready value:1];
        const bool r_gpu = reinterpret_cast<uintptr_t>(r_out) % page == 0;
        if (r_gpu)
            metal_linalg::detail::qr_r_out(back, w.A, metal_linalg::detail::wrap_host(dev, r_out, (size_t)K * n), K,
                                           n, n, up);
        metal_linalg::detail::qr_blocks_apply(back, Q, m, K, K, b, done / b, w.st);
        [back commit];
        // Q's start, [I 0; 0 *], while the GPU factors
        metal_linalg::detail::for_each_rows(1, m, K, [&](uint32_t, uint32_t r0, uint32_t r1) {
            std::memset(Qh + (size_t)r0 * K, 0, (size_t)(r1 - r0) * K * 4);
            for (uint32_t i = r0; i < std::min(r1, done); ++i) Qh[(size_t)i * K + i] = 1.0f;
        });
        for (id<MTLCommandBuffer> f : forward) check(f);
        tail(A, m, n, done, Qh, K);
        ready.signaledValue = 1;
        if (!r_gpu)   // R, unscaled, while the GPU forms Q
            metal_linalg::detail::for_each_rows(1, K, n, [&](uint32_t, uint32_t r0, uint32_t r1) {
                for (uint32_t i = r0; i < r1; ++i) {
                    float* ri = r_out + (size_t)i * n;
                    std::fill(ri, ri + i, 0.0f);
                    vDSP_vsmul(A + (size_t)i * n + i, 1, &up, ri + i, 1, n - i);
                }
            });
        check(back);
    }
    if (Qh != q_out)
        metal_linalg::detail::for_each_rows(1, m, K, [&](uint32_t, uint32_t r0, uint32_t r1) {
            std::memcpy(q_out + (size_t)r0 * K, Qh + (size_t)r0 * K, (size_t)(r1 - r0) * K * 4);
        });
}

} // namespace

bool qr_blocked_fits(uint32_t m, uint32_t n) {
    const uint32_t b = metal_linalg::detail::qr_block_width(m);
    return std::min(m, n) >= 1 && b != 0 && m >= 2 * b;
}

// A matrix at a time, the blocked QR beats the streaming kernels, which take
// a batch at once, for one matrix from 256 x 256 (on an M5 Pro 1.5 ms against
// 2.7; 6.9 against 14.8 at 1024, 18.7 against 49 at 2048), two from 1024 and
// four from 2048. QR_BLOCKED=0 turns it off.
bool qr_blocked_preferred(uint32_t m, uint32_t n, uint32_t batch) {
    if (const char* e = std::getenv("QR_BLOCKED"); e && std::string(e) == "0") return false;
    return qr_blocked_fits(m, n) && (batch == 1 || (size_t)batch * 512 <= std::min(m, n));
}

void qr_blocked(const Matrices& a, float* q, float* r) {
    const uint32_t M = a.rows, N = a.cols, K = std::min(M, N);
    if (K == 0 || a.batch == 0) return;
    if (!qr_blocked_fits(M, N))
        throw std::invalid_argument("[qr] blocked: " + std::to_string(M) + " rows is more than its panels take");
    for (uint32_t b = 0; b < a.batch; ++b)
        one(a.data + (size_t)b * M * N, M, N, q + (size_t)b * M * K, r + (size_t)b * K * N);
}

} // namespace metal_linalg::core::detail
