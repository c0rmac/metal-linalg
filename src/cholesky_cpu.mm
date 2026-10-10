// Cholesky on the CPU: LAPACK's spotrf (Accelerate), the matrices of a batch
// spread over the cores (lapack_batches); one matrix gets Accelerate's own
// threads. The route for what the GPU does not win; see CholeskyPolicy.
//
// Always spotrf('L') on a column-major copy of the lower triangle, in a
// scratch whose leading dimension is padded off a power of two. Accelerate's
// spotrf('U') was 4-5x slower than 'L' up to n = 64 and 1.3-1.4x from 256 to
// 3072 (macOS 27, M5 Pro); either at a leading dimension of 1024, 2048, 4096
// or 8192 lost 10-45% to cache-set conflicts ('L' at 4096: 44 ms at ld 4096,
// 24 at 4112). spotrf also writes into the triangle it does not reference
// (scratch from n = 65), which the copy out leaves behind.
//
// Row-major A read column-major is A^T = A, so the scratch's column-major lower
// triangle (row-major upper) takes the upper triangle as it is, row by row, and
// the lower one transposed; L comes back in the same place, and goes out the
// same way.
#ifndef ACCELERATE_NEW_LAPACK
#define ACCELERATE_NEW_LAPACK   // LAPACK's current interface; before any Accelerate header
#endif
#include <metal_linalg/core.h>
#include "metal_runtime.h"

#include <Accelerate/Accelerate.h>
#include <arm_neon.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using metal_linalg::core::Matrices;
using metal_linalg::detail::lapack_batches;

namespace metal_linalg::core::detail {
namespace {

// The 4 x 4 block at in (row stride ldi), transposed, to out (row stride ldo).
inline void transpose4(const float* in, size_t ldi, float* out, size_t ldo) {
    const float32x4_t r0 = vld1q_f32(in), r1 = vld1q_f32(in + ldi), r2 = vld1q_f32(in + 2 * ldi),
                      r3 = vld1q_f32(in + 3 * ldi);
    const float32x4_t t0 = vtrn1q_f32(r0, r1), t1 = vtrn2q_f32(r0, r1), t2 = vtrn1q_f32(r2, r3),
                      t3 = vtrn2q_f32(r2, r3);
    vst1q_f32(out, vreinterpretq_f32_f64(vtrn1q_f64(vreinterpretq_f64_f32(t0), vreinterpretq_f64_f32(t2))));
    vst1q_f32(out + ldo, vreinterpretq_f32_f64(vtrn1q_f64(vreinterpretq_f64_f32(t1), vreinterpretq_f64_f32(t3))));
    vst1q_f32(out + 2 * ldo, vreinterpretq_f32_f64(vtrn2q_f64(vreinterpretq_f64_f32(t0), vreinterpretq_f64_f32(t2))));
    vst1q_f32(out + 3 * ldo, vreinterpretq_f32_f64(vtrn2q_f64(vreinterpretq_f64_f32(t1), vreinterpretq_f64_f32(t3))));
}

// out[c][r] = in[r][c] for r >= c: in's lower triangle, transposed, into
// out's upper. 4 x 4 blocks in 64 x 64 tiles, along in's rows; a block on
// the diagonal goes whole, so there it also reads in's upper triangle and
// writes out's lower (never read: spotrf('L') reads out's column-major lower).
void lower_into_upper(const float* in, size_t ldi, float* out, size_t ldo, uint32_t n) {
    constexpr uint32_t T = 64;
    const uint32_t n4 = n & ~3u;
    for (uint32_t r0 = 0; r0 < n4; r0 += T)
        for (uint32_t c0 = 0; c0 <= r0; c0 += T)
            for (uint32_t r = r0; r < std::min(r0 + T, n4); r += 4)
                for (uint32_t c = c0; c < std::min({c0 + T, n4, r + 4}); c += 4)
                    transpose4(in + (size_t)r * ldi + c, ldi, out + (size_t)c * ldo + r, ldo);
    for (uint32_t r = n4; r < n; ++r)   // the last rows (n not a multiple of 4)
        for (uint32_t c = 0; c <= r; ++c) out[(size_t)c * ldo + r] = in[(size_t)r * ldi + c];
}

// out[c][r] = in[r][c] for r <= c: in's upper triangle, transposed, into
// out's lower. 8 x 8 blocks in 128 x 128 tiles, along out's rows: out is the
// caller's matrix, its row stride often a power of two, and 4 x 4 blocks
// along in's rows wrote it 3x slower (at n = 2048). A block on the diagonal
// goes whole, so there it also writes out's upper: the caller zeroes that.
void upper_into_lower(const float* in, size_t ldi, float* out, size_t ldo, uint32_t n) {
    constexpr uint32_t T = 128;
    const uint32_t n8 = n & ~7u;
    for (uint32_t c0 = 0; c0 < n8; c0 += T)
        for (uint32_t r0 = 0; r0 <= c0; r0 += T)
            for (uint32_t c = c0; c < std::min(c0 + T, n8); c += 8)
                for (uint32_t r = r0; r < std::min({r0 + T, n8, c + 8}); r += 8) {
                    const float* i0 = in + (size_t)r * ldi + c;
                    float* o0 = out + (size_t)c * ldo + r;
                    transpose4(i0, ldi, o0, ldo);
                    transpose4(i0 + 4, ldi, o0 + 4 * ldo, ldo);
                    transpose4(i0 + 4 * ldi, ldi, o0 + 4, ldo);
                    transpose4(i0 + 4 * ldi + 4, ldi, o0 + 4 * ldo + 4, ldo);
                }
    for (uint32_t c = n8; c < n; ++c)   // the last columns (n not a multiple of 8)
        for (uint32_t r = 0; r <= c; ++r) out[(size_t)c * ldo + r] = in[(size_t)r * ldi + c];
}

} // namespace

void cholesky_cpu(const Matrices& a, bool upper, float* l, uint32_t* info) {
    if (a.rows != a.cols)
        throw std::invalid_argument("[cholesky] cpu: the matrices must be square, got " + std::to_string(a.rows) +
                                    "x" + std::to_string(a.cols) + ".");
    const uint32_t n = a.cols, batch = a.batch;
    if (n == 0 || batch == 0) return;
    const size_t per = (size_t)n * n;
    const uint32_t ld = n >= 128 ? (n + 15) / 16 * 16 + 16 : n;   // off a power of two (see the top)
    auto solve = [&](uint32_t b0, uint32_t b1) {
        // kept between calls, as the GPU paths keep their workspaces: a fresh
        // one's page faults cost as much as the copies
        thread_local std::vector<float> scratch;
        if (scratch.size() < (size_t)n * ld) scratch.resize((size_t)n * ld);
        float* w = scratch.data();
        for (uint32_t b = b0; b < b1; ++b) {
            const float* src = a.data + b * per;
            float* dst = l + b * per;
            // the column-major lower triangle: row j of w from column j on
            if (upper) {
                for (uint32_t j = 0; j < n; ++j)
                    std::memcpy(w + (size_t)j * ld + j, src + (size_t)j * n + j, (n - j) * sizeof(float));
            } else {
                lower_into_upper(src, n, w, ld, n);
            }
            __LAPACK_int ln = (__LAPACK_int)n, lld = (__LAPACK_int)ld, err = 0;
            char uplo = 'L';
            spotrf_(&uplo, &ln, w, &lld, &err);
            if (err < 0) throw std::runtime_error("[cholesky] cpu: spotrf rejected argument " + std::to_string(-err));
            // a non-finite pivot is a failure, as the GPU kernels report it
            // (spotrf lets an infinite one through): the first one's index + 1
            if (err == 0)
                for (uint32_t k = 0; k < n; ++k)
                    if (!std::isfinite(w[(size_t)k * ld + k])) { err = (__LAPACK_int)k + 1; break; }
            if (err > 0) {
                std::fill(dst, dst + per, std::numeric_limits<float>::quiet_NaN());
            } else if (upper) {
                for (uint32_t j = 0; j < n; ++j) {
                    std::memset(dst + (size_t)j * n, 0, j * sizeof(float));
                    std::memcpy(dst + (size_t)j * n + j, w + (size_t)j * ld + j, (n - j) * sizeof(float));
                }
            } else {
                // w's row-major upper triangle holds L^T: into dst's lower,
                // then dst's upper zeroed (the diagonal blocks wrote into it)
                upper_into_lower(w, ld, dst, n, n);
                for (uint32_t i = 0; i + 1 < n; ++i)
                    std::memset(dst + (size_t)i * n + i + 1, 0, (n - i - 1) * sizeof(float));
            }
            if (info) info[b] = (uint32_t)err;
        }
    };
    // Above kSerialMinN the matrices go one at a time, each with Accelerate's
    // own threads: spotrf calls running at once on several threads with its
    // threading off corrupted each other's factors from n ~ 1536 (macOS 27,
    // M5 Pro: 2 of 2048 failed at pivots 833 and 897, either alone did not),
    // and a matrix that large is the faster for its threads anyway.
    constexpr uint32_t kSerialMinN = 1024;
    if (n > kSerialMinN) solve(0, batch);
    else lapack_batches(batch, per, solve);
}

} // namespace metal_linalg::core::detail
