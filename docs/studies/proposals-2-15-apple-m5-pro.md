# The 2.13 proposals, done, on an Apple M5 Pro

After 2.13.0 the work left on eigh and the SVD was written up as seven
proposals ([docs/proposals/](../proposals/README.md)). This is what came of
them in 2.15.0, measured on an Apple M5 Pro (20 GPU cores, 18 CPU cores,
48 GB), and what turned up along the way. All seven were built; the
seventh, the two-stage SVD with vectors, after a first prototype was parked
the same day.

| | before (2.14.1) | 2.15.0 | |
|---|---|---|---|
| eigh with vectors, `tridiag`, 4096 | 438 ms | 306 ms | 1.43x |
| SVD with vectors, `bidiag`, 4096 | 1535 ms | 943 ms | 1.63x |
| eigvalsh, `band`, 4096 | 161 ms | 140 ms | 1.15x |
| svdvals, `band`, 4096 | 226 ms | 197 ms | 1.15x |
| SVD with vectors, `band` (new), 4096 | 1535 ms (`bidiag`) | 404 ms | 3.80x |

(One matrix; 2.15.0 is the median of `sweep_eigh` and `sweep_svd`, section
9; before, the routing sweep `20261004-06bc11` and section 1.)

## 1. The divide and conquer on every core

With vectors, `tridiag` and `bidiag` hand the tridiagonal or bidiagonal
problem to LAPACK's `sstedc` and `sbdsdc`, which work on one core: 177 of
eigh's 438 ms at 4096, and 718 of the SVD's 1535.

`src/divide_conquer.cpp` walks the same tree as LAPACK's `slaed0` and
`slasd0`, with LAPACK's routines for everything numerical. The leaves
(`ssteqr`, `slasdq`) and the many small merges low in the tree (`slaed1`,
`slasd1`) run one per task; the few large merges at the top run their own
steps, `slaed1`'s and `slasd1`'s, with their loops spread over the threads:
the secular equation's roots (`slaed4`, `slasd4`, a call a root), the
corrected z (each entry a product over the roots, in LAPACK's order), the
vectors (a column a root), and the products with the halves' vectors (by
blocks of 128 columns). The threads are GCD's, each with Accelerate's own
threading off; `cpu_threads() - 2` of them, as the band chase, since a batch
overlaps one matrix's solve with the next's reduction on the GPU.

The first cut made the tridiagonal 3.5x faster and the bidiagonal only 2.9x.
A profile of the bidiagonal's top merge at 4096 showed why:

| top merge, n = 4096 | time |
|---|---|
| `slasd2` (deflation) | 127 ms |
| roots | 3.6 ms |
| corrected z | 1.3 ms |
| left vectors | 1.0 ms |
| U's products | 28 ms |
| right vectors | 2.6 ms |
| VT's products | 20 ms |

`slasd2` copies and rotates VT's rows one row at a time, each a stride of
`ldvt` floats between entries. It is rewritten (`deflate_bidiagonal`) with
LAPACK's loops but VT's rows moved a column at a time: the Givens rotations
are recorded during the scan (their angles depend on d and z alone) and
applied afterwards in the same order, and the rows are gathered a column of
VT a task. Every output was checked against LAPACK's `slasd2` bit for bit
on Gaussian, random, repeated and glued-block bidiagonals, rotations
included.

Merges of 256 rows or more always take this file's path and smaller ones
LAPACK's, by size, not by the number of threads: the two differ in the last
bits of the vectors, where the products are blocked differently, and the
results must not depend on how many threads ran.

The tridiagonal of `ssytrd` and the bidiagonal of `sgebrd` of a Gaussian
matrix, best of three, 16 threads:

| | LAPACK | here | |
|---|---|---|---|
| `sstedc`, 1024 | 12.0 ms | 3.8 ms | 3.1x |
| `sstedc`, 2048 | 42.6 | 10.1 | 4.2x |
| `sstedc`, 4096 | 175.6 | 50.5 | 3.5x |
| `sbdsdc`, 1024 | 27.9 | 7.8 | 3.6x |
| `sbdsdc`, 2048 | 133.3 | 21.9 | 6.1x |
| `sbdsdc`, 4096 | 757.3 | 100.0 | 7.6x |

The eigenvalues and singular values are LAPACK's bit for bit; residuals and
orthogonality match LAPACK's. The top merges are now bound by their
products, which the CPU's matrix units run at about 2 TFLOP/s whether split
over the threads or left to Accelerate's threading (within 10%); see
[divide-and-conquer-gpu-products.md](../proposals/divide-and-conquer-gpu-products.md).
From n = 128 the parallel version is the faster; below, LAPACK's own is
called.

## 2. The TSQR top kernel, and IEEE arithmetic

A band panel taller than 128 rows is factored by TSQR: leaves of 128 rows a
simdgroup each, the stacked R's factored in one threadgroup (the top), and
the Householder vectors rebuilt. The top took about 95 us a panel. Timing
its phases (a scratch build that returns after each, 32 leaves at b = 16):

| phase | cumulative |
|---|---|
| the QR of the 512 stacked rows, a thread a row | 56.8 us |
| M and E | 59.2 |
| Q1 | 67.3 |
| the LU of Q1 - S | 75.6 |
| L1^{-1}, U^{-1} | 87.2 |
| T_H, the writes | 90.8 |

The proposal had guessed the last steps cost nothing, having seen no change
when they moved to one simdgroup; they cost 24 us. And Q1 and the rebuild
each recomputed a b x b product for every entry of another.

**The tree.** The stacked R's are triangles, so their QR is a binary tree of
pair QRs (LAPACK's `stpqrt2` with a triangular second block): reflector j acts on row j of the upper
triangle and rows 0..j of the lower, so a simdgroup with a lane a column
needs about b^2 / 2 shuffles a pair and no reduction or barrier. Up the tree
the pairs of a level run side by side, a simdgroup each; E comes down it
from E = I at the root, each node's Q applied to [E; 0]. The LU with chosen
signs, U^{-1} and T_H run in simdgroup 0's registers with shuffles.

That version was slower: 118 us. The tree alone, up, took 104 us, about 20 a
pair, where a pair is a few hundred dependent instructions. Building the
same kernel with `-ffast-math` took 15.

**IEEE mode.** `Svd_Bidiag.metal` is built with `-fno-fast-math` (the
one-stage kernels' NaN handling wants it), which makes `/` and `sqrt()`
correctly rounded sequences. And a kernel with any of them in it compiles all
of its arithmetic in that mode: the tree's `sqrt1`, with an IEEE `sqrt()`
only in a branch for x <= 0 that never ran, took 33 us; without that branch,
24. The panel kernels now use `div1` and `sqrt1`: the fast reciprocal or
square root and one Newton step, within about an ulp. Their norms are plain
sums of squares rather than slarfg's scaled pairs (the matrix is scaled into
[0.5, 1) first, so nothing in a panel nears overflow, and what underflows is
far below the reduction's rounding and is left by a skipped reflector).

One panel, 4096 x 16 (32 leaves):

| | before | after |
|---|---|---|
| top | 91.6 us | 42.9 |
| a leaf | 32.6 | 20.8 |
| rebuild | 15.6 | 12.0 |
| b = 8: top, leaf | 27.2, 11.9 | 10.3, 7.2 |
| b = 32: top, leaf, rebuild | 430, 114, 129 | 127, 92, 86 |

The whole call, one matrix, side by side with 2.14.1:

| | 1024 | 2048 | 3072 | 4096 |
|---|---|---|---|---|
| svdvals, `band` | 27.0 to 20.7 ms | 69.8 to 57.5 | 126 to 110 | 226 to 204 |
| eigvalsh, `band` | 17.2 to 14.5 | 47.8 to 41.3 | 88.3 to 79.4 | 159 to 149 |

The leaves' and the rebuild's share at 4096 is now about 30 ms of svdvals'
200. Merging the leaf and top dispatches (step 3 of the proposal) was not
tried.

## 3. The small products

Each block also made three small MPS products. Timed alone, MPS took 10-20 us
for the b x b product summed over the trailing rows, whatever their number,
and about 2 us for each of the others. They are now two kernels:
`bd_small_partial` (each threadgroup's partial of the b x b product over 256
rows, staged in threadgroup memory 64 rows at a time) and `bd_sy_apply` or
`bd_ge_apply` (every threadgroup sums the partials in order, issuing their
loads four at a time, then does the b-wide work in one pass). A first
version, a thread an entry reading its rows from device memory, took 28 and
19 us: latency, not work. eigvalsh on `band` 2-5% faster at 1024-2048,
svdvals 1-3% at 2048, nothing measurable at 4096, as the proposal estimated.

## 4. The symmetric trailing update

MPS has no symmetric products, so the eigensolver's band reduction kept both
triangles and updated all of A22 each block. MPS's two products of a block,
alone:

| n | X = A22 W | A22 -= [V Y][Y V]^T |
|---|---|---|
| 1024 | 19 us (221 GB/s) | 19 (443) |
| 2048 | 62 (269) | 71 (474) |
| 4096 | 365 (184) | 590 (227) |

Below 4096 the matrix sits in the GPU's cache (16 MB at 2048); at 4096 it
does not (64 MB).

`sb_update` updates A22's lower 64 x 64 tiles: a tile staged through
threadgroup memory with `float4` loads, eight simdgroups each a 16 x 32
part of it as simdgroup products with [V Y] and [Y V]'s pieces loaded from
the small, cached W, the tile written back. Loading the tiles straight from
device memory with strided `simdgroup_load`s reached only 65 GB/s; staged,
220-420. With the lower triangle alone the product X = A22 (V T) has to read
it too, and four versions of that lost to MPS on the full matrix: each tile
read twice (708 us at 4096), split over threadgroups by chunks of columns
(469), loaded straight from device memory (1386), double-buffered (471),
against MPS's 365; at 2048, 148 against 62. So `sb_update` also writes each
off-diagonal tile's transpose over its mirror (1.5 n^2 of memory a block
rather than 2 n^2), the upper triangle stays whole, and MPS keeps X:

| n | MPS update | `sb_update`, mirrored |
|---|---|---|
| 1024 | 19.0 us | 14.8 |
| 2048 | 70.4 | 60.5 |
| 2560 | 182.5 | 92.7 |
| 3072 | 360.3 | 210.5 |
| 4096 | 610.5 | 461.6 |
| 6000 | 1379.3 | 1025.5 |

eigvalsh on `band`: 4096 152 to 141 ms, 8192 872 to 744. The same tiles over
all of the SVD's trailing matrix (no symmetry to save on) were no faster
than MPS and were dropped.

## 5. The band's width

`values_band_width` (0: 16) is now part of both policies, and the sweeps time
the band at 8, 16 and 32 at the band points. Stages 3b and 4b choose the
width with the lowest geometric mean of each width's time over the best
width's at each point, 16 unless another wins by more than 1%, and fit the
threshold on that width's times. On the M5 Pro, side by side (section 9),
16 is the fastest width or within 2% of it at every point but one: eigvalsh
at 4096, where 32 takes 130 ms against 140. 32 loses 9% at 3072 and ties at
8192, and 8 loses 15-125% from 1536 on, so the fit should keep 16 for both;
the routing re-measure, which runs stages 3b and 4b, decides.

## 6. The thresholds' tie-break

The threshold fits keep, among thresholds within 0.5% of the best geometric
mean over all the band points, the one with the lowest worst case, then the
highest. On run `20261004-06bc11` that chose 4096 for eigvalsh although
`band` was 7% faster at the one point between 3072 and 4095: scored over all
28 points, that one point barely moved the mean. Now a threshold also counts
as near-optimal only if it is within 3% of the best on the points where the
two choose differently (`near_on_disagreement` in `tuning/tune_eigh.py`), and
the grids gain N = 2560 and 3584 (eigh) and k = 1280 and 1792 (SVD).
Re-analysed, `20261004-06bc11` chooses 3072, which `src/tuned/eigh.inc` now
carries. Side by side in 2.15.0, `band` is 18% faster than `tridiag` at 3072
(77 ms against 91) and 10% slower at 2048, so 3072 still stands; the routing
re-measure will place it between them.

## 7. The two-stage SVD with vectors

Built, after a first prototype was parked; see [the proposal](../proposals/two-stage-vectors.md#done-2026-10-07)
and [svd.md](../svd.md#with-singular-vectors-since-2150). The chase's
reflectors can be recorded at no cost and grouped into blocks of 16 sweeps
whose order gives Q2 bit for bit. The first kernel applied them one group
after another, every 32-column strip a chain of 32,896 dependent blocks at
4096: 170 ms a side, about twice what would pay. Block (G - 1, p) needs only
(G, p) and (G, p + 1), so `bd_chase_apply` runs four groups at once in a
threadgroup, a simdgroup each, two tiles apart, the tiles handed down
through threadgroup memory: 60-64 ms a side. And instead of applying Q2 to
U_B after the divide and conquer, Q1 Q2 and P1 P2 are formed explicitly while
the CPU runs it (and Q1, P1 while it chases the band), then U and V^T are one
product each. At 4096: 404 ms against `bidiag`'s 942 (2.33x), 2.59x at 8192,
1.17x at 1024. Routed from `band_min_k`, which stage 3c of `tune_svd.py`
fits; 0 until the M5 Pro is re-measured.

## 8. Found along the way

**The back-transformations' block reflectors.** `tridiag` and `bidiag` build
each block of 128 reflectors' V and T on the CPU while the GPU applies the
previous block. Per block at 4096: `slarft` 1.06 ms, the transposed copy into
V 0.94, the gather 0.5, against about 1.2 ms for the GPU's three products, so
the CPU's side was the bound (at 8192, 2.16 and 1.69 ms). T now comes from
the Gram matrix V^T V (one `ssyrk` on the matrix units, 0.1 ms) by slarft's
recurrence (`compact_wy_t`), and the copies are spread over the cores. eigh
with vectors on `tridiag`: 4096 331 to 302 ms. The SVD's, bound by the GPU
there, did not move.

**The one-stage reductions and bisection** compute a reflector a column in
every threadgroup and divide once a Sturm step, all in IEEE mode; and
`td_symv` computed its tile index with an IEEE `sqrt()`. Built once with
fast math throughout, against the normal build:

| | IEEE | fast math |
|---|---|---|
| eigvalsh, `tridiag`, 2048 | 39.8 ms | 35.2 |
| eigvalsh, `tridiag`, 4096 | 219 | 203 |
| eigh, `tridiag`, 2048 | 56.5 | 53.1 |
| svdvals, `bidiag`, 2048 | 93.8 | 85.4 |
| SVD, `bidiag`, 2048 | 128.9 | 122.4 |
| SVD Jacobi, 1024 x 32x32 | 7.38 | 6.10 |
| eigh Jacobi kernels (threadgroup, simd, block) | | no change |

With `div1` and `sqrt1` in those kernels: eigvalsh on `tridiag` 36.0 and 204
ms at 2048 and 4096, most of fast math's gain; svdvals on `bidiag` 91.8 at
2048, about 7% short of it. The rest is in
[ieee-mode-audit.md](../proposals/ieee-mode-audit.md).

**The sweeps' gate from N ~ 6500.** The correctness gate compared each
backend with MLX's `eigvalsh` or `svd` on the CPU, combined with the result
on the GPU in one expression: MLX queued that GPU work behind the CPU's, and
from N ~ 6500 the wait outlasted the GPU's watchdog, a timeout that the gate
read as a failed backend (2.14.1 too). The reference is now evaluated on its
own first.

**A quarter-second watchdog.** In the afternoon of the measurements, with
the display busy, macOS ended single-threadgroup Jacobi kernels after about
250 ms (eigh in threadgroup mode from N = 448, the SVD's at 512 x 512), which
had run that morning. Fixed in 2.15.0: the two kernels now split a long
solve over dispatches of a few rounds, about 15 ms each, the result bit for
bit the same; see
[gpu-watchdog-long-kernels.md](../proposals/gpu-watchdog-long-kernels.md#done-2026-10-07).

**`kernels.py` did not watch the band files.** The epoch check's `PATHS`
missed `src/band_*`, `src/bisect`, and `shaders/Svd_Bidiag` for eigh, all
added in 2.13; they, and `src/divide_conquer`, are now watched.

**The unsigned `cpu_threads() - 2`** wrapped around when the thread cap was
1 or 2, and the band chase then ignored the cap (it still stopped at one
thread per 256 rows); `cpu_threads_beside_gpu()` replaces it.

## 9. Results

One matrix, 2.15.0 against the CPU path, measured side by side on the
afternoon of 2026-10-07: the median of `sweep_eigh` and `sweep_svd` (at
least three calls, more up to 150 ms). The band columns are its widths 8, 16
(the default) and 32; "best" is the fastest GPU backend at the default
width.

Eigenvalues, in ms:

| N | eigh: CPU | `tridiag` | speedup | eigvalsh: CPU | `tridiag` | band 8 | band 16 | band 32 | best / CPU |
|---|---|---|---|---|---|---|---|---|---|
| 1024 | 40.7 | 17.6 | 2.30x | 17.6 | 11.8 | 15.0 | 13.7 | 16.9 | 1.49x |
| 1536 | 101 | 33.2 | 3.04x | 42.0 | 21.7 | 27.6 | 22.9 | 27.4 | 1.94x |
| 2048 | 243 | 55.5 | 4.39x | 83.3 | 35.9 | 52.0 | 39.4 | 47.9 | 2.32x |
| 3072 | 730 | 133 | 5.49x | 209 | 91.2 | 115 | 77.2 | 83.9 | 2.71x |
| 4096 | 2494 | 306 | 8.15x | 483 | 213 | 222 | 140 | 130 | 3.44x |
| 8192 | 18293 | 2170 | 8.43x | 2644 | 1656 | 1632 | 726 | 726 | 3.64x |

Singular values, in ms:

| k | SVD: CPU | `bidiag` | speedup | svdvals: CPU | `bidiag` | band 8 | band 16 | band 32 | best / CPU |
|---|---|---|---|---|---|---|---|---|---|
| 512 | 16.7 | 13.2 | 1.27x | 7.6 | 8.9 | | | | 0.86x |
| 1024 | 78.5 | 34.2 | 2.29x | 36.0 | 21.5 | 20.6 | 20.9 | 29.1 | 1.72x |
| 1536 | 182 | 69.1 | 2.64x | 86.2 | 47.5 | 38.7 | 33.7 | 46.3 | 2.56x |
| 2048 | 472 | 125 | 3.79x | 218 | 87.6 | 66.6 | 54.8 | 77.4 | 3.97x |
| 3072 | 1416 | 386 | 3.67x | 773 | 307 | 153 | 113 | 132 | 6.85x |
| 4096 | 3527 | 943 | 3.74x | 2070 | 712 | 314 | 197 | 205 | 10.5x |
| 8192 | | | | 12643 | 6028 | 2306 | 1095 | 1094 | 11.5x |

The CPU path is 2.14's; its svdvals measured 6-8% slower from 2048 than in
run `20261004-06bc11` (2070 ms at 4096 against 1949; PyTorch's `sgesdd` took
2.01 s, as on 2026-10-05), which flatters those svdvals ratios by as much.

The routing has not been re-measured: the epochs (eigh 6, SVD 8) mark every
Mac's eigh and SVD rows stale, the M5 Pro's included, and they apply as
they are, at width 16, until `tuning/run.py` is run on an idle Mac. That run
fills sections 5 and 6's choices.
