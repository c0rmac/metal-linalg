# SVD: design notes and the measurements so far

What was built, why it looks the way it does, and what was measured along the
way. Unlike [`qr-routing-apple-m1.md`](qr-routing-apple-m1.md) and
[`eigh-routing-apple-m1.md`](eigh-routing-apple-m1.md), this is **not** a routing study: every
timing here was taken on an Apple M1 while the machine was heavily loaded by
other jobs (load average 12 to 120 on 8 CPUs, on battery). Ratios between GPU
paths measured back to back are usable. Absolute times and every GPU-against-
CPU ratio are not, because other jobs slow the CPU backend most. The routing
defaults in `svd.mm` were placed from these numbers and are marked untuned.
The study is `python3 tuning/tune_svd.py build/sweep_svd` on an idle machine;
it has since been run on an Apple M5 Pro, see
[`routing-apple-m5-pro.md`](routing-apple-m5-pro.md).

## What was built

| piece | file |
|---|---|
| whole-matrix one-sided Jacobi kernel | `shaders/Svd_Jacobi.metal` |
| block one-sided Jacobi kernel | `shaders/Svd_BlockJacobi.metal`, `shaders/block_jacobi_common.h` (shared with the eigensolver) |
| host code, the QR-preconditioned and CPU backends, routing policy | `src/svd.mm`, `src/svd_block_jacobi.mm`, `include/metal_linalg/svd.h` |
| correctness tests, 197 checks | `tests/test_svd.cpp` |
| benchmark and launch-parameter sweep | `benchmarks/benchmark_svd.cpp` |
| routing harness | `tuning/sweep_svd.cpp`, `tuning/tune_svd.py` |

## Design decisions

**One simdgroup per column pair.** One-sided Jacobi rotates two columns and
touches nothing else, and the pairs of a round are disjoint. Giving each pair
to one simdgroup means the three inner products are a `simd_sum`, the rotation
parameters are uniform across the lanes, and the rotation is applied by the
same lanes, with no barrier inside a round. The only synchronisation is one
threadgroup barrier between rounds. The eigensolver's two-sided method needs
three per round, because row and column updates of different pairs meet at
shared elements.

**Columns stored by column.** `G[c * M + i]`, so a lane walking down a column
walks contiguous memory and a simdgroup's 32 lanes read 32 consecutive
elements.

**Thin factors.** `U` is M×K, not M×M. MLX's own `svd` returns the full-size
factors, and for a tall matrix the M×M `U` is most of its cost.

**A fair CPU path.** Because of that, comparing against `linalg::svd` directly
flatters the GPU on every tall shape. `detail::svd_cpu` reduces a tall matrix
with a thin QR on the CPU first, as LAPACK does for its own thin SVD, and that
is what the routing and the benchmark compare against.

**A second kernel for large matrices.** The whole-matrix kernel gives a matrix
one threadgroup, that is one GPU core, so a single 512×512 takes half a second
on an M1 whatever the core count. The block kernel (`Svd_BlockJacobi.metal`)
is the eigensolver's block Jacobi turned one-sided: the columns are cut into
blocks of 16, a round takes every disjoint pair of blocks, and for each pair
one threadgroup forms the 32×32 Gram matrix $S = X^T X$ of the two column
blocks, runs the eigensolver's two-sided Jacobi on it in threadgroup memory
(`jacobi_sweep` from `eigh_jacobi_common.h`, one inner sweep), and applies the
resulting 32×32 rotation to the columns of $G$ and $V$ as tile products
(`bj_cols_apply` from `block_jacobi_common.h`, `simdgroup_matrix` 8×8). Every
block pair of a round is a separate threadgroup, so one matrix spreads over
the grid, and a batch shares each round's three dispatches. The subproblem is
inexact, so it takes a few more outer sweeps than the whole-matrix kernel (11
against 10 at 128×128), and each sweep is cheap.

## What the measurements showed

### 1. The estimate was too optimistic by about a factor of two

Before building, the estimate for 32×32 at batch 4096 was a 6x win over the
CPU, with 3x named as the threshold below which the case for SVD would fall.
Measured: **3.0x**, on a loaded machine whose CPU timings are inflated, so the
true figure is probably lower. The kernel takes about 37 µs per 32×32 matrix
at its best launch configuration, against 23 µs for the eigensolver kernel,
despite needing a third of the barriers. The reason is the sweep count:
one-sided Jacobi needs 10 sweeps at N = 32 where the two-sided method needs 7,
and the saving in barriers does not make up for it. The expected win region is
therefore about the eigensolver's, batched small matrices, not larger.

### 2. Simdgroups per matrix: the eigensolver's two regimes again

Median ms for the full SVD; `pairs` is column pairs per round.

```
     M      N   batch  pairs |      sg=1      sg=2      sg=4      sg=8     sg=16     sg=32
     8      8       1      4 |     0.511     0.380     0.328        --        --        --
     8      8    4096      4 |     7.366     7.075     7.694        --        --        --
    16     16       1      8 |     0.943     0.623     0.507     0.343        --        --
    16     16     256      8 |     2.711     3.210     3.659     3.994        --        --
    16     16    4096      8 |    33.658    34.004    34.916    38.032        --        --
    32     32       1     16 |     3.272     1.830     1.033     0.785     0.733        --
    32     32      16     16 |     4.137     2.364     1.575     1.272     1.264        --
    32     32     256     16 |    10.343    10.803    10.501    11.011    12.958        --
    32     32    4096     16 |   158.564   152.336   163.240   178.328   193.644        --
    64     64       1     32 |    21.789    11.647     6.288     4.795     2.832     3.220
    64     64      16     32 |    24.935    12.059     8.460     4.775     4.331     4.674
    64     64     256     32 |   134.340    67.673    53.585    55.901    65.111    81.010
    64     64    2048     32 |  1061.009   562.555   451.962   467.853   451.849   552.200
   128    128       1     64 |   155.282    77.510    54.549    22.697    16.078    13.379
   128    128      16     64 |   219.615   103.790    64.272    35.697    24.937    24.663
   128    128     128     64 |   653.770   632.154   526.667   294.603   187.197   202.411
   256    256       1    128 |  1085.708   550.059   273.299   156.272    89.980    62.812
   256    256      16    128 |  2157.342  1246.831   761.684   608.571   605.911   240.282
   256     32     256     16 |    73.824    43.802    25.984    24.312    25.107        --
  1024     64      16     32 |   233.950   118.388    68.975    42.013    36.817    34.268
```

With few matrices, more simdgroups win up to the 32 a threadgroup allows. With
many, fewer win: threadgroups stay small and co-reside on a core. The rule in
`svd.mm` is the larger of a work floor, one simdgroup per 512 row-pairs rotated
per round, and a core budget, `32 * cores / batch`, capped by the pair count.
It picks the best or second-best column in every row above.

### 3. QR preconditioning: structural for long thin input, useless for square

Same machine, back to back. `J(R)` is the Jacobi kernel on the triangular
factor.

| shape | batch | direct | QR | J(R) | QR path | sweeps direct / on R |
|---|---|---|---|---|---|---|
| 32×32 | 4096 | 152 ms | 27 ms | 147 ms | 167 ms | 10 / 9 |
| 64×64 | 256 | 52 ms | 8 ms | 50 ms | 56 ms | 11 / 10 |
| 128×128 | 16 | 24 ms | 2 ms | 25 ms | 23 ms | 11 / 11 |
| 64×8 | 4096 | 8 ms | 14 ms | 7 ms | 22 ms | 7 / 7 |
| 128×16 | 1024 | 11 ms | 10 ms | 8 ms | 19 ms | 7 / 8 |
| 256×32 | 256 | 24 ms | 15 ms | 11 ms | 26 ms | 8 / 8 |
| 1024×64 | 64 | 153 ms | 25 ms | 13 ms | 39 ms | 8 / 9 |
| 2048×64 | 16 | 81 ms | 15 ms | 5 ms | 21 ms | 8 / 8 |

The literature's case for preconditioning is a lower sweep count, and that
depends on column pivoting, which this library's QR deliberately does not do.
Without it the sweep count barely moves, so for square input the QR is pure
cost. For tall input the gain is that the rotations act on K×K instead of
L×K.

The first rule tried keyed this on rows and aspect ratio, and the benchmark
showed that to be the wrong shape. At batch 16:

| shape | aspect | direct | QR path | |
|---|---|---|---|---|
| 1024×16 | 64 | 3.2 ms | 8.7 ms | direct wins 2.7x |
| 256×32 | 8 | 2.7 ms | 6.4 ms | direct wins 2.3x |
| 1024×32 | 32 | 8.3 ms | 8.1 ms | tie |
| 1024×64 | 16 | 33.3 ms | 18.3 ms | QR path wins 1.8x |
| 2048×64 | 32 | 69.3 ms | 31.8 ms | QR path wins 2.2x |

The most elongated shape is the one where preconditioning loses most. The QR
path has a fixed cost of a few milliseconds (its smallest measured time is
2.4 ms), so it pays only when the direct kernel has a lot of work, and that
work grows with the square of the short side. The policy is therefore a
threshold on rows and a threshold on the short side, `qr_min_rows = 512` and
`qr_min_k = 64` until measured, and never unless L ≥ 2K. At batch 1 the two
paths tie even at 2048×64, so batch may belong in the rule as well; the
harness will show it.

Decomposing R or its transpose made no consistent difference.

### 4. Rank-deficient input needed a change to the method

The first version passed a rank-deficient test and then failed to converge in
nearly every one of 108 random rank-deficient instances. A single passing
instance had been luck. Columns that cancel to rounding noise keep being
rotated against each other; each rotation amplifies their relative error,
which puts them out of line with the large columns; correcting that changes
their angles to one another; and the cycle runs down to underflow, far past
any sweep limit. Two fixes that did not work: skipping a pair whose norms
differ by more than the rank tolerance (the noise columns sit right at that
ratio, so it helped in two runs out of three), and loosening the tolerance.

What works is deciding nullity against the matrix, not the pair. A column
whose norm is below `rank_tol` times the largest column's is still
orthogonalised against every column that is not null, but never against
another null column. After that, all 108 instances converge in 3 to 11 sweeps
with reconstruction no worse than 1.2e-06. The tolerances are computed from
the row count of the matrix the user passed, not of the small factor the
QR-preconditioned backend hands the kernel, and never from fewer than 64 rows.

The block kernel then showed the rule was still not quite right, in two ways.

*The null threshold was the rank tolerance,* `max(M, N) * eps`, and at
512 rows that is 6e-05 of the largest column: real data. Singular values 1e+4
to 1e-4 through the block paths reconstructed to 7e-05 instead of 2e-06,
because columns four decades down were being treated as null and left
unrotated against each other. The null threshold is now its own constant,
`32 * eps` of the largest column, well clear of the rank tolerance. A probe
at 128 to 512 rows over ranks 1 to 500 put the noise that null columns settle
at between 0 and 3 eps of the largest singular value, so the threshold has
ten times that as margin.

*A null column kept the sweeps going without ever being rotated.* Against a
large column, a null column's angle is noise that never settles under the
tolerance, so it only counts as pending while the angle is gross (cosine
above 0.05). But for `ones(40, 40)`, rank 1, the block kernel ran 30 sweeps
with nothing changing after the second: the null columns were exact multiples
of the large one, cosine 1, so they counted as pending, while the rotation
phase skips a pair whose off-diagonal element is below 1e-12 of its diagonal,
which at those magnitudes it was. The pending test and the rotation
disagreed. A column below 1.2e-7 of the largest (`kNegligible`) is now under
the rounding of everything else and never counts as pending. Both kernels
carry both rules; `ones(40, 40)` converges in two sweeps and 204 random
rank-deficient instances through the four GPU paths reconstruct to 2.3e-06
or better.

### 5. Two defects in the existing QR, found by the SVD tests

Both are in shipped code and both are fixed, with regression tests in
`tests/test_qr.cpp`.

**QR was not scale-invariant.** Reconstruction error by input magnitude,
before the fix:

| input scale | `qr_unblocked` 64×64 | `qr_streaming_amx_reduced` 512×64 | `qr_streaming_amx_complete` 8×8 |
|---|---|---|---|
| 1 | 4e-07 | 3e-07 | 2e-07 |
| 1e-3 | 4e-03 | 3e-07 | 2e-07 |
| 1e-4 | 13% | 3e-07 | 62% |
| 1e-5 and below | 70% | 97% | 62% |
| 1e+18 | 4e-07 | NaN | 3e-07 |
| 1e+30 | NaN | NaN | NaN |

The kernels compare a squared column norm with an absolute threshold. Every
matrix is now scaled by an exact power of two on the way in and R is scaled
back on the way out (`prepare_input_scaled`). After the fix every backend is at
2e-07 to 4e-07 from 1e-30 to 1e+37.

**The reflection threshold discarded real data.** At `1e-7` on a squared norm,
any column tail shorter than 3e-4 of the matrix's scale was treated as already
zero. For nearly dependent columns that is data: a rank-deficient 600×16
reconstructed to 3e-05 instead of 3e-07. With inputs at unit scale the
threshold is now `1e-30`, which only guards the division.

These change QR's timings slightly, by one scan of the input and one multiply
of R. Both QR backends pay the same, so the crossover should not move, but the
QR routing study predates the change and is worth repeating with it.

### 6. The two kernels' crossover, on the loaded M1

Single matrix, median ms, load average 80 to 110 on 8 CPUs, so the CPU column
is inflated and only the two GPU columns are comparable:

| k | whole-matrix | block | CPU |
|---|---|---|---|
| 96 | 5.7 | 7.8 | 1.8 |
| 128 | 10.6 | 11.5 | 1.3 |
| 192 | 28.2 | 18.3 | 6.5 |
| 256 | 61.4 | 26.1 | 9.6 |
| 384 | 211.5 | 48.8 | 15.1 |
| 512 | 508.4 | 75.5 | 34.2 |

The block kernel overtakes the whole-matrix kernel between 128 and 192 for a
single matrix, and earlier in batches: at batch 64 and above it is ahead from
k = 96 (36 ms against 40 ms; 75 against 89 at 128; 248 against 317 at 192),
because the per-round dispatches are shared by the batch. That is why
`SvdPolicy` carries a batch-dependent crossover like the eigensolver's. The
default `block_min_k = 192` is from this table.

## Open items

1. **The routing study on the M1**, on an idle machine. The M5 Pro has been
   measured ([`routing-apple-m5-pro.md`](routing-apple-m5-pro.md)) and has a
   row in `svd.mm`; the M1 is still untuned, and its numbers above are the
   loaded-machine ones.
2. **The eigensolver could use this kernel's structure.** One barrier per
   round instead of three is a property of one-sided Jacobi, not of SVD. A
   symmetric matrix's SVD gives its eigenvectors, and the signs of the
   eigenvalues follow from one Rayleigh quotient each. Whether the higher
   sweep count eats the saving, as it did here, is a measurement.
3. **Fewer sweeps.** Column pivoting in the QR, or de Rijk's column
   interchanges inside the kernel, is what the literature uses to cut them.
