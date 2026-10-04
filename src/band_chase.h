#pragma once
// The second stages of the two-stage reductions, on the CPU (band_chase.cpp).

#include <cstddef>
#include <cstdint>

namespace metal_linalg::detail {

// The second stage of the two-stage reductions (band_chase.cpp): an upper
// band of width nb, n x n, to bidiagonal, d (n) and e (n - 1), by Householder
// bulge chasing on up to `threads` threads. W holds the band column-major, A(i,
// j) at W[j * ld + ku + i - j], with room for the bulges: ku >= 2 nb above the
// diagonal and ld - 1 - ku >= nb below, all of it zero outside the band.
void band_to_bidiagonal(uint32_t n, uint32_t nb, float* W, size_t ld, size_t ku, float* d, float* e,
                        unsigned threads);

// A symmetric band, its lower half of width kd, n x n, to tridiagonal, d (n)
// and e (n - 1), likewise. W holds the lower half column-major, A(r, c) for
// r >= c at W[c * ld + r - c], with room for the bulges: ld >= 2 kd + 1, all
// of it zero outside the band.
void band_to_tridiagonal(uint32_t n, uint32_t kd, float* W, size_t ld, float* d, float* e, unsigned threads);

} // namespace metal_linalg::detail
