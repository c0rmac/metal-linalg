// Transposes between a row-major matrix and LAPACK's column-major layout, for
// the CPU paths that hand LAPACK a padded column-major copy (lu_cpu.mm).
// 8 x 8 blocks of NEON 4 x 4 transposes, walked along the destination's rows:
// the destination is often the caller's matrix, its row stride a power of two,
// and blocks walked along the source's rows wrote it 3x slower (2048 x 2048).
#pragma once

#include <arm_neon.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace metal_linalg::detail {

// The 4 x 4 block at in (row stride ldi), transposed, to out (row stride ldo).
inline void transpose4x4(const float* in, size_t ldi, float* out, size_t ldo) {
    const float32x4_t r0 = vld1q_f32(in), r1 = vld1q_f32(in + ldi), r2 = vld1q_f32(in + 2 * ldi),
                      r3 = vld1q_f32(in + 3 * ldi);
    const float32x4_t t0 = vtrn1q_f32(r0, r1), t1 = vtrn2q_f32(r0, r1), t2 = vtrn1q_f32(r2, r3),
                      t3 = vtrn2q_f32(r2, r3);
    vst1q_f32(out, vreinterpretq_f32_f64(vtrn1q_f64(vreinterpretq_f64_f32(t0), vreinterpretq_f64_f32(t2))));
    vst1q_f32(out + ldo, vreinterpretq_f32_f64(vtrn1q_f64(vreinterpretq_f64_f32(t1), vreinterpretq_f64_f32(t3))));
    vst1q_f32(out + 2 * ldo, vreinterpretq_f32_f64(vtrn2q_f64(vreinterpretq_f64_f32(t0), vreinterpretq_f64_f32(t2))));
    vst1q_f32(out + 3 * ldo, vreinterpretq_f32_f64(vtrn2q_f64(vreinterpretq_f64_f32(t1), vreinterpretq_f64_f32(t3))));
}

// out[c][r] = in[r][c] for an in of rows x cols (row stride ldi) into out
// (row stride ldo): out is cols x rows.
inline void transpose(const float* in, size_t ldi, float* out, size_t ldo, uint32_t rows, uint32_t cols) {
    constexpr uint32_t T = 128;
    const uint32_t r8 = rows & ~7u, c8 = cols & ~7u;
    for (uint32_t c0 = 0; c0 < c8; c0 += T)          // out's rows
        for (uint32_t r0 = 0; r0 < r8; r0 += T)
            for (uint32_t c = c0; c < std::min(c0 + T, c8); c += 8)
                for (uint32_t r = r0; r < std::min(r0 + T, r8); r += 8) {
                    const float* i0 = in + (size_t)r * ldi + c;
                    float* o0 = out + (size_t)c * ldo + r;
                    transpose4x4(i0, ldi, o0, ldo);
                    transpose4x4(i0 + 4, ldi, o0 + 4 * ldo, ldo);
                    transpose4x4(i0 + 4 * ldi, ldi, o0 + 4, ldo);
                    transpose4x4(i0 + 4 * ldi + 4, ldi, o0 + 4 * ldo + 4, ldo);
                }
    for (uint32_t c = 0; c < cols; ++c)   // the edges past the 8 x 8 blocks
        for (uint32_t r = (c < c8 ? r8 : 0); r < rows; ++r) out[(size_t)c * ldo + r] = in[(size_t)r * ldi + c];
}

// LAPACK's leading dimension for an n-row column-major copy: off a power of
// two from 128 rows. Accelerate's spotrf and sgetrf lost 10-65% at a leading
// dimension of 1024, 2048, 4096 or 8192 to cache-set conflicts (sgetrf at
// 4096: 169 ms at 4096, 62 at 4112; macOS 27, M5 Pro).
inline uint32_t padded_ld(uint32_t n) { return n >= 128 ? (n + 15) / 16 * 16 + 16 : std::max(n, 1u); }

} // namespace metal_linalg::detail
