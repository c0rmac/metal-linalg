#pragma once
// The second stages of the two-stage reductions, on the CPU (band_chase.cpp).

#include <atomic>
#include <cstddef>
#include <cstdint>

namespace metal_linalg::detail {

// The bulge chase's reflectors, kept for the singular vectors (width 16), in
// the block layout of svd_bidiag.mm's bd_chase_apply: sweep s's step-j
// reflector from the left (on rows s + 1 + 16 j .. s + 16 (j + 1)) is column
// c = s % 16 of block (G, p) = (s / 16, s / 16 + j)'s V (32 x 16, at row
// offset c, with its 1 and zeros explicit), its tau at Ltau[16 b + c], b =
// G (pmax + 1) - G (G - 1) / 2 + j; from the right (on the same columns)
// likewise in R, Rtau. A block is kChaseBlockFloats at L[kChaseBlockFloats b],
// V's six tiles of 8 x 8 that are not zero first (row-major; (row, column)
// tiles (0,0) (1,0) (2,0) (1,1) (2,1) (3,1)), each column's three written
// whole. pmax = (n - 2) / 16. Reflectors of length 1 are not written. With `frontier`, the
// number of leading sweeps finished (all of 0 .. frontier - 1), as they
// finish, so that their reflectors can be used while the chase goes on.
// With `ready_rows`, the chase starts before the band is complete: rows (or,
// for a symmetric band, columns) 0 .. ready_rows - 1 are final, raised as
// the band reduction finishes them; the chase does not look past them.
constexpr size_t kChaseBlockFloats = 13 * 64;   // V's six tiles, then the kernel's Y's seven

struct ChaseReflectors {
    float* L = nullptr;
    float* Ltau = nullptr;
    float* R = nullptr;
    float* Rtau = nullptr;
    size_t pmax = 0;
    std::atomic<long>* frontier = nullptr;
    const std::atomic<long>* ready_rows = nullptr;
};

// The second stage of the two-stage reductions (band_chase.cpp): an upper
// band of width nb, n x n, to bidiagonal, d (n) and e (n - 1), by Householder
// bulge chasing on up to `threads` threads. W holds the band column-major, A(i,
// j) at W[j * ld + ku + i - j], with room for the bulges: ku >= 2 nb above the
// diagonal and ld - 1 - ku >= nb below, all of it zero outside the band.
// With `rec`, its reflectors are kept there (nb = 16), and its progress as
// rec says.
void band_to_bidiagonal(uint32_t n, uint32_t nb, float* W, size_t ld, size_t ku, float* d, float* e,
                        unsigned threads, const ChaseReflectors* rec = nullptr);

// A symmetric band, its lower half of width kd, n x n, to tridiagonal, d (n)
// and e (n - 1), likewise. W holds the lower half column-major, A(r, c) for
// r >= c at W[c * ld + r - c], with room for the bulges: ld >= 2 kd + 1, all
// of it zero outside the band.
// With `rec`, its reflectors are kept in its L and Ltau (kd = 16; as the
// bidiagonal chase's left ones: Q2 is their product in the order applied),
// and its progress as rec says.
void band_to_tridiagonal(uint32_t n, uint32_t kd, float* W, size_t ld, float* d, float* e, unsigned threads,
                         const ChaseReflectors* rec = nullptr);

} // namespace metal_linalg::detail
