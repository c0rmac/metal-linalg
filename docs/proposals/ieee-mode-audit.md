# IEEE arithmetic in the remaining shaders

Status: **done** in 2.15.0 (2026-10-07); see [Done](#done-2026-10-07).

## What

Six shader files are built with `-fno-fast-math` (`CMakeLists.txt`):
`Eigh_Jacobi`, `Eigh_BlockJacobi`, `Eigh_Tridiag`, `Svd_Jacobi`,
`Svd_BlockJacobi`, `Svd_Bidiag`. That makes `/` and `sqrt()` IEEE sequences,
and a kernel with any of them in it compiles all of its arithmetic in IEEE
mode, even if that division or square root is never executed (2.15.0 found
an untaken `sqrt()` costing the TSQR top a third of its time). In 2.15.0 the
band panels, the one-stage reductions' reflectors and bisection were moved to
fast division and square root with a Newton step (`div1`, `sqrt1`). Do the
same, kernel by kernel, where it pays, in the rest.

## Why: the measurements

The whole library built once with every `-fno-fast-math` removed, against the
normal build, same session, M5 Pro (before the 2.15.0 changes to the
one-stage reductions):

| point | IEEE | fast math |
|---|---|---|
| SVD Jacobi, 1024 x 32x32 | 7.38 ms | 6.10 ms (1.21x) |
| SVD Jacobi, 64 x 128x128 | 33.6 | 33.5 |
| SVD block Jacobi, 16 x 256x256 | 25.2 | 24.6 (1.02x) |
| eigh threadgroup mode, 1024 x 32x32 | 4.67 | 4.61 |
| eigh threadgroup mode, 256 x 64x64 | 10.40 | 10.44 |
| eigh simd mode, 1024 x 16x16 | 0.906 | 0.904 |
| eigh block Jacobi, 16 x 256x256 | 20.7 | 20.5 |
| eigvalsh `tridiag`, 2048 | 39.8 | 35.2 (1.13x) |
| svdvals `bidiag`, 2048 | 93.8 | 85.4 (1.10x) |

After 2.15.0's `div1`/`sqrt1` in the one-stage reductions, eigvalsh `tridiag`
at 2048 is 36.0 ms (most of the gain), svdvals `bidiag` 91.8: about 7% is
left there, which division and square root alone do not account for (IEEE
`max`, comparisons and the like, which only fast math relaxes).

## Plan

1. `Svd_Jacobi.metal`: the rotation's angle (a square root and divisions per
   pair) to `div1`/`sqrt1`; check its accuracy on the suite's hard cases
   (graded, nearly rank-deficient, tiny values) before timing.
2. The bidiag kernels' remaining 7%: compile one kernel at a time with fast
   math (a second metallib) to find which; then decide whether that kernel
   can live with fast math (its NaN and infinity handling is the reason the
   file is IEEE).
3. The eigh Jacobi kernels showed nothing; skip unless a profile says
   otherwise.
4. The batched-kernel routing is fitted per device: re-measure eigh or the
   SVD after a change.

## Effort

About a day, mostly checking accuracy.

## Expected gain

1.2x for the SVD's Jacobi kernel on batches of small matrices; up to 7% for
svdvals on `bidiag`. On the M5 Pro the Jacobi kernel is rarely chosen
(`golub_kahan` and the CPU take most of its region), but estimated policies
on other Macs use it more.

## Where to start

`div1` and `sqrt1` in `shaders/Svd_Bidiag.metal` (and their copy in
`shaders/Eigh_Tridiag.metal`); the rotation in `shaders/Svd_Jacobi.metal` and
`shaders/eigh_jacobi_common.h`.

## Done (2026-10-07)

**The SVD's Jacobi kernel.** Its rotation (a square root, three divisions
and an `rsqrt` a pair) and its output (a square root a column, a division an
entry of U) now use the fast approximations with Newton steps, `j_div`,
`j_sqrt` and `j_rsqrt` in `eigh_jacobi_common.h`. With one Newton step for
`rsqrt`, c came out a little low every time (c^2 + s^2 < 1) and V's
orthogonality was 10x worse (8.9e-5 at 512 x 512, against 7.1e-6): four
cases of the suite failed. With two steps it is better than the IEEE
sequence's (4.5e-6 at 512, 2.3e-6 at 200 against 3.2e-6), and the suite
passes. Against the build before, alternating runs, M5 Pro:

| point | before | after | |
|---|---|---|---|
| 1024 x 32x32 | 7.58 ms | 6.46 | 1.17x |
| 4096 x 16x16 | 5.87 | 5.09 | 1.15x |
| 256 x 48x48 | 5.52 | 4.86 | 1.14x |
| 64 x 128x128 | 33.7 | 32.5 | 1.03x |
| 16 x 256x256 | 66.1 | 66.2 | 1.00x |
| 1 x 512x512 | 262 | 249 | 1.06x |

(Fast math for the whole library had given 1.21x at 1024 x 32x32.)

**The one-stage reductions' 7%: gone already.** `Svd_Bidiag.metal` built
entirely with fast math, against the current build: svdvals on `bidiag`
85.4 ms at 2048 either way (the 91.8 above was measured before the one-stage
reflectors' `div1`), and no difference beyond 2% for `bidiag` with vectors,
`band` with or without; `Eigh_Tridiag.metal` likewise (eigvalsh on
`tridiag` 35.1 ms at 2048 against 35.4, eigh 54.3 against 53.3). Nothing
left there to find.

**The eigensolver's Jacobi kernels**: skipped, as planned (within 1-2% with
fast math).

