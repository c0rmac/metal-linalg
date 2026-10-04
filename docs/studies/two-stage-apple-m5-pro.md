# A two-stage reduction on an Apple M5 Pro

Can the eigenvalues alone or the singular values alone of a large matrix be
computed faster by reducing it in two stages, first to a band on the GPU,
then to tridiagonal or bidiagonal form? After 2.12.0 the one-stage `tridiag`
and `bidiag` backends were held to memory bandwidth: their reductions read the
trailing matrix once (eigh) or twice (SVD) a column, at about 290 GB/s. A
reduction to a band reads it a few times per block of columns, as matrix
products. This study measures what the second stage costs on this Mac, and
builds both two-stage reductions, the second stage on every CPU core and the
last step, the eigenvalues or singular values of the tridiagonal or
bidiagonal, by bisection on the GPU.

**What came of it (2.13.0):** a `band` backend for the singular values alone,
3.2x the `bidiag` backend and 8.6x the CPU at 4096×4096, 10.4x the CPU at
8192 ([svd.md](../svd.md)); one for the eigenvalues alone, 2.9x the CPU's own
two-stage `ssyevd_2stage` at 4096 and 3.0x at 8192 ([eigh.md](../eigh.md));
and bisection on the GPU in the one-stage backends' value paths, which made
`tridiag`'s eigenvalues alone 1.4-1.5x faster at 2048-4096.

## 1. Where the time goes, one stage

One matrix, M5 Pro (20 GPU cores, 18 CPU cores), 2.12.0:

| | total | GPU reduction | CPU |
|---|---|---|---|
| eigvalsh 4096 (`tridiag`) | 292 ms | about 210 ms | `ssterf` 80 ms |
| svdvals 4096 (`bidiag`) | 839 ms | 740 ms, 650 of them the two products a column | `sbdsdc` 93 ms |

## 2. What LAPACK's second stages cost here

Accelerate, one call, best of one (the band from `ssytrd_sy2sb` or random):

| N = 4096 | band width 16 | 32 | 64 |
|---|---|---|---|
| `ssytrd_sb2st` (band to tridiagonal, Householder) | 112 ms | 125 ms | 328 ms |
| `ssbtrd` (the same by rotations) | 172 ms | 349 ms | 726 ms |
| `sgbbrd` (band to bidiagonal, rotations) | 196 ms | 401 ms | 850 ms |

and `ssterf` 80 ms, `sbdsqr` without vectors 73-78 ms. For comparison, the
CPU's whole eigvalsh (`ssyevd_2stage`) takes 459 ms, `ssyevd` 1827 ms, and its
whole svdvals (`sgesdd`) 2032 ms. At 2048 everything is about four times
cheaper (`sb2st` 28 ms, `sgbbrd` 43 ms at width 16).

**The eigensolver.** Its second stage and `ssterf` alone take about 190 ms at
4096, against the one-stage path's 292 ms in all: however fast the GPU's band
reduction, at most about 1.2x, and nothing at 2048 (28 + 21 ms of CPU stage
against 58 ms in all). Worth building only with the second stage on every
core and the tridiagonal's eigenvalues off the single core (sections 6, 7).

**The SVD.** `sgbbrd` at width 8 to 16 (91-196 ms) and `sbdsqr` (78 ms) leave
room under the one-stage path's 792-839 ms: built first with them, then with
the replacements below.

## 3. The GPU's stage: a block's work

Block $k$ of $b$ columns: the QR of the column panel, $H = I - V T V^T$; with
$C$ the columns right of it, $W = T^T V^T C$; the row panel $C(0{:}b,:) - V(0{:}b,:) W$
and its LQ, $G = I - U S U^T$; $X = C_{low} U$; then
$C_{low} \mathrel{-}= V_{low} W + (X - V_{low} W U) S U^T$ as one product. $C$
is read three times and written once a block: at $b = 16$ and 4096 about 23
GB in all, 76 ms at 300 GB/s. Six MPS GEMMs a block (the row panel's own
update is folded into its kernel), measured at 104 ms; encoding one costs the
CPU 25-30 µs (126 µs the first time a shape is seen).

## 4. The panels

A block's two panels are a QR each of a tall, thin matrix (4096×16 at first):
every column needs a reduction over all rows before the next can start, so
the panel is a chain of small dependent steps, and at 512 panels per 4096
matrix (256 blocks) its latency decides. In order:

| version | per panel (b = 16) | panels at 4096 |
|---|---|---|
| one threadgroup, the panel streamed from device memory | 0.5 ms | 256 ms |
| TSQR, a thread a row in registers, 1024 threads a leaf | 0.66 ms | 339 ms |
| the same, 256 threads a leaf | 0.2 ms | 103 ms |
| two barriers a column instead of four | 0.37 ms | 190 ms |
| rows in registers for real (see below), rotated | 0.3 ms | 154 ms |
| one simdgroup a leaf, 4 rows a lane, no threadgroup barrier | 0.15 ms | 78 ms |

What made the difference:

- **A thread's row indexed by a variable lives in memory.** `x[j]` in the
  column loop put each thread's row on its stack, many times slower than
  registers, and the compiler does not unroll a loop that holds a barrier or
  a simdgroup reduction, so `#pragma unroll` changed nothing. Unrolling by
  hand removed every stack array, but the 32-wide kernels then took the GPU
  compiler over ten minutes to build. Instead each thread's row turns one
  place left a column: column $j$ is always `x[0]`, column $(j + k) \bmod b$
  is `x[k]`, only the loops over $k$ are unrolled ($b$ copies of a body, not
  $b^2$), and after $b$ turns the row is back in place.
- **Barriers cost more with more threads**, and two kinds of work hid behind
  them: T's recurrence run by one thread at the end (32 µs), and every thread
  summing every simdgroup's partials for every column (34 µs). A lane per
  partial and shuffles fixed the second, a row of T per lane the first; a
  leaf of one simdgroup has no threadgroup barrier at all.
- **TSQR** ([Demmel, Grigori, Hoemmen and Langou](https://doi.org/10.1137/080731992))
  splits a tall panel into leaves factored in parallel, then factors their
  stacked $R$'s; the update then needs the panel's Householder vectors, which
  an LU of $Q - S$ with chosen signs rebuilds ([Ballard et al.](https://doi.org/10.1016/j.jpdc.2015.06.003)),
  stably for any panel, rank-deficient ones included. Leaves must not have
  fewer rows than the band is wide (equal leaves see to it).

Left: the TSQR's top (80 µs a panel, one threadgroup of up to 1024 threads)
and its leaves (40 µs); about 15% of the GPU's stage.

## 5. The first results, LAPACK's second stage

svdvals, one square matrix, `bidiag` (one stage, `sbdsqr`) against `band`
with `sgbbrd` and `sbdsqr` on one core:

| k | bidiag | band, b = 8 | b = 16 |
|---|---|---|---|
| 512 | 9.6 ms | 13.7 ms | 14.1 ms |
| 1024 | 25.5 ms | 32.7 ms | 36.9 ms |
| 2048 | 106 ms | 98 ms | 120 ms |
| 3072 | 315 ms | 220 ms | 246 ms |
| 4096 | 792 ms | 450 ms | 455 ms |

At 4096, b = 8: the GPU's stage 277 ms (panels 64), `sgbbrd` 91 ms, `sbdsqr`
78 ms; b = 16: 182, 193, 78. The CPU's stage was now 170-270 ms of the 450,
on one core, and a wider band, which makes the GPU's stage cheaper, made it
dearer.

## 6. The second stage on every core

`src/band_chase.cpp`: the band to bidiagonal or tridiagonal by bulge chasing
with Householder reflectors, PLASMA's kernels for the general band ([Haidar,
Kurzak and Luszczek](https://doi.org/10.1145/2503210.2503292)) and LAPACK's
`ssb2st` kernels for the symmetric one. Sweep $s$ clears row (or column) $s$
and chases the bulge it makes down the band, a block of $b$ at a time; each
reflector clears one row or column of its bulge, the rest left to the sweeps
that follow, so the work stays within $2b$ of the diagonal. Sweep $s$ may run
its $t$-th step once sweep $s - 1$ has finished its $(t + 2)$-th, the blocks
they touch being one apart ([Lang](https://doi.org/10.1137/0914078)), so the
sweeps pipeline: sweep $s$ on thread $s \bmod P$, each publishing a count of
finished steps that the next spins on. The threads are `std::thread`s all
running at once; a pool that ran fewer than the sweeps waited on would
deadlock.

| N = 4096 | b = 8 | 16 | 32 |
|---|---|---|---|
| `sgbbrd` (rotations, one core) | 91 ms | 196 ms | 401 ms |
| band to bidiagonal, one thread | 103 ms | 119 ms | 188 ms |
| band to bidiagonal, 16 threads | 53 ms | 35 ms | 39 ms |
| `ssytrd_sb2st` (one core) | | 112 ms | 125 ms |
| band to tridiagonal, one thread | 94 ms | 112 ms | 152 ms |
| band to tridiagonal, 16 threads | 52 ms | 35 ms | 34 ms |

At $b = 8$ the steps are too small for the threads' hand-offs to pay; from
16 the chase is 3-5x faster than one core.

Sixteen threads, not eighteen: two cores are left to the GPU's host work
(encoding the next matrix's GEMMs in a batch), as when a batch is shared
between the GPU and the CPU. Fewer threads are used when
the band has too few blocks to keep them busy ($P \le 2N / 3b$) or the
matrix is small ($P \le N / 256$).

## 7. Bisection on the GPU

With both reductions on the GPU and the chase on every core, the last
sequential step was LAPACK's: `ssterf` 79 ms and `sbdsqr` 77 at 4096, 305 and
300 at 8192. Bisection (`sturm_bisect` in `shaders/Eigh_Tridiag.metal`) gives
each thread one eigenvalue: from the Gershgorin interval, a fixed number of
halvings, each a Sturm count of the tridiagonal's sign changes with LAPACK's
`pivmin` guard, the tridiagonal read from threadgroup memory 1024 entries at a
time. Singular values are the non-negative eigenvalues of the $2k \times 2k$
Golub-Kahan tridiagonal. The answer is as accurate as `ssterf`'s in absolute
terms, a few float32 ulps of the norm (not to high relative accuracy for tiny
singular values, which `sbdsqr` gives and nothing downstream of a float32
reduction can use).

| | `ssterf` | bisection | `sbdsqr` | bisection (2k) |
|---|---|---|---|---|
| 4096 | 79 ms | 6 ms | 77 ms | 12 ms |
| 8192 | 305 ms | 16 ms | 300 ms | 31 ms |

Below $N = 512$ (eigenvalues) and $k = 1024$ (singular values) LAPACK is the
faster and is kept. The one-stage `tridiag` and `bidiag` backends use the
bisection in their value paths too.

## 8. Results

From the routing sweep [`20261004-06bc11`](../results/apple-m5-pro-20gpu/20261004-06bc11/summary.md)
to 4096, 8192 measured alone; one square matrix, eigenvalues alone (CPU:
`ssyevd_2stage` on every core; `tridiag` in 2.12.0 from run `20261004-fd9bd8`):

| N | CPU | tridiag (2.12.0) | tridiag | band | best / CPU |
|---|---|---|---|---|---|
| 1024 | 18 ms | 17 ms | 12 ms | 17 ms | 1.47x |
| 2048 | 82 ms | 58 ms | 39 ms | 47 ms | 2.12x |
| 3072 | 207 ms | 141 ms | 96 ms | 90 ms | 2.30x |
| 4096 | 467 ms | 295 ms | 209 ms | 161 ms | 2.91x |
| 8192 | 2588 ms | 2016 ms | 1749 ms | 872 ms | 2.97x |

Singular values alone (CPU: `sgesdd`):

| k | CPU | bidiag (2.12.0) | bidiag | band | best / CPU |
|---|---|---|---|---|---|
| 1024 | 36 ms | 26 ms | 23 ms | 28 ms | 1.55x |
| 1536 | 86 ms | 59 ms | 50 ms | 44 ms | 1.93x |
| 2048 | 201 ms | 111 ms | 93 ms | 70 ms | 2.87x |
| 3072 | 721 ms | 330 ms | 281 ms | 127 ms | 5.68x |
| 4096 | 1949 ms | 810 ms | 733 ms | 226 ms | 8.62x |
| 8192 | 11.95 s | | 6.16 s | 1.15 s | 10.4x |

At 4096 the SVD's band path is now the GPU's stage, about 180 ms (panels 78,
GEMMs about 100), the chase 35 and bisection 12. Eigenvalues within
$6 \times 10^{-6}$ of LAPACK's relative to the largest, singular values within
$1 \times 10^{-5}$. The routing sweep puts `band` from N = 4096 for the
eigenvalues (at 3072 it is 7% ahead of `tridiag`, inside the fit's tolerance)
and from k = 1536 for the singular values.

## 9. What is left

- **The GPU's stage** is now about three quarters of the time. Its GEMMs,
  about 100 ms at 4096, are near what reading the matrix three times a block
  costs; the panels, 78 ms, are latency: 512 a matrix, each a chain of
  dependent column steps. A wider band halves their number but doubles each
  one's columns, and on the M5 Pro $b = 16$ was the best of 8, 16 and 32
  (svdvals at 4096: 329, 235 and 275 ms).
- **Look-ahead was tried and lost.** Updating the next block's columns first
  and factoring its panel while the rest of the trailing matrix is updated,
  with the two on separate buffers so that Metal need not order them, gave
  nothing in one command buffer (the GPU ran them one after the other: 1-4%
  slower, for the extra copies) and lost 14-54% with the panel on a second
  command queue, synchronised by shared events: a hand-off between queues
  costs more than the panel it would hide.
- **The small products.** Three of a block's six GEMMs are $b$ wide on one
  side; one kernel for them would save some of their encoding and launches.
- **Eigenvectors and singular vectors.** The two-stage reduction's
  back-transformation has two stages too, and the second one's reflectors are
  small and many; LAPACK does not offer it in its two-stage drivers either.
  Not attempted.

The experiments' sources are not kept; the measurements above are from the
library as built at each step.
