# Parallel divide and conquer for the tridiagonal and bidiagonal solvers

Status: **done in 2.15.0** (2026-10-07), the CPU version, and faster than
estimated. `src/divide_conquer.cpp` walks LAPACK's tree with its leaves and
small merges one per task, and its large merges (from 256 rows) with
`slaed3`'s and `slasd3`'s loops over the threads. Profiling the first cut
found that `slasd2`, the bidiagonal merge's deflation, took 130 of a 4096
merge's 190 ms: it moves VT's rows one strided row at a time. It is
rewritten to move them a column at a time, and checked against LAPACK's
`slasd2` bit for bit, every output, on four kinds of matrix. On an M5 Pro, one
4096 problem: `sstedc` 176 to 50 ms (estimate 40-50), `sbdsdc` 753 to 100 ms
(estimate 150-200); eigh with vectors on `tridiag` 438 to about 300 ms, the
SVD with vectors on `bidiag` 1535 to about 920 ms. The values are bit for
bit LAPACK's; the vectors differ in the last bits; neither depends on the
number of threads. See [the 2.15.0 study](../studies/proposals-2-15-apple-m5-pro.md),
section 1. Left: the CPU path for one matrix
([cpu-path-divide-and-conquer.md](cpu-path-divide-and-conquer.md)) and the
top merges' products on the GPU
([divide-and-conquer-gpu-products.md](divide-and-conquer-gpu-products.md)).

What follows is the proposal as written on 2026-10-04.

## What

With vectors, the `tridiag` (eigh) and `bidiag` (SVD) backends hand the
tridiagonal or bidiagonal problem to LAPACK's divide-and-conquer solvers,
`sstedc` and `sbdsdc`, which run on one CPU core. Replace those calls with a
divide and conquer whose subproblems and merges run on all the cores, built
from LAPACK's own merge routines.

## Why: the measurements

One matrix on the M5 Pro, Accelerate, the tridiagonal from `ssytrd` and the
bidiagonal from `sgebrd` of a dense Gaussian matrix (a random tridiagonal
deflates far more and is not representative: `sstedc` took 32 ms on one at
4096), best of three:

| | wall | CPU time | cores used |
|---|---|---|---|
| `sstedc`, n = 2048 | 42.7 ms | 47.1 ms | 1.1 |
| `sstedc`, n = 4096 | 176.8 ms | 220.2 ms | 1.2 |
| `sbdsdc` ('I'), n = 2048 | 143.7 ms | 152.4 ms | 1.1 |
| `sbdsdc` ('I'), n = 4096 | 718.2 ms | 798.6 ms | 1.1 |

Against the backends' totals for one matrix (routing sweep `20261004-06bc11`):

| | total | the solver's share |
|---|---|---|
| eigh, `tridiag`, 2048 | 91 ms | 47% |
| eigh, `tridiag`, 4096 | 438 ms | 40% |
| SVD, `bidiag`, 2048 | 248 ms | 58% |
| SVD, `bidiag`, 4096 | 1535 ms | 47% |

The rest is the GPU's reduction and back-transformation, already near memory
bandwidth. In a batch the two-slot pipeline overlaps one matrix's solve with
the next one's reduction, but then the solver bounds the throughput: for the
SVD at 4096 it is as slow as the whole GPU reduction.

## Plan

The CPU version first; it reuses LAPACK's numerics, which is where divide and
conquer is subtle. Accelerate exports every routine needed (checked by
linking): `slaed0`-`slaed4` and `slamrg` for the tridiagonal, `slasd0`-`slasd4`
and `slasdq` for the bidiagonal.

1. **Tear** the tridiagonal into P pieces by Cuppen's rank-one tearing,
   $T = \operatorname{diag}(T_1, T_2) + \beta v v^T$, recursively, as `slaed0`
   does; only the scheduling changes.
2. **Leaves**: `sstedc` on each piece, the pieces spread over the cores with
   Accelerate's own threading off (as the batched CPU path does), on
   `cpu_threads() - 2` workers, leaving two cores to the GPU's host work, as
   the band chase does.
3. **Merges**, bottom up. Low in the tree there are many independent merges:
   one per worker, each a call of `slaed1`. High in the tree there are few,
   large ones: split each merge's work. Deflation (`slaed2`) is O(n) and stays
   serial; the secular equation's roots (`slaed4`, one call per root) are
   independent, as are the Gu-Eisenstat $\hat z$ and the eigenvector columns
   (the body of `slaed3`), so they go over the workers; the product of the two
   halves' eigenvectors with the merge's (`sgemm`) is split by column blocks,
   or left to Accelerate's threaded `sgemm`.
4. **The bidiagonal** the same way with `slasd0`/`slasd1`'s structure: deflation
   `slasd2`, roots `slasd4`, vectors from `slasd3`'s formulas, and two
   products, for U and for V^T.
5. Wire it in where `src/eigh_tridiag.mm` calls `sstedc` and
   `src/svd_bidiag.mm` calls `sbdsdc`, behind a size threshold (below a few
   hundred LAPACK's serial solver is the faster). Optionally use it in the
   CPU path for one matrix too (`ssytrd`, this, `sormtr`), which would make
   the CPU baseline faster for mid-size matrices.

A first cut can stop at step 3's "one `slaed1` per worker" plus Accelerate's
threaded `sgemm` at the top, and measure before splitting the top merges.

A GPU version (one thread a secular root, the merges' products on MPS, the
bottom levels on the CPU) could go further at the top levels, where the
merges are large, but is a larger job; only worth it if the CPU version's top
merges turn out to bound it.

## Effort

3-5 days for the CPU version: a day for the tridiagonal skeleton on top of
`slaed1`, a day to split the top merges, a day for the bidiagonal, and the
rest for testing (clustered and repeated spectra, deflation-heavy and
deflation-free cases), the re-measure (the eigh and SVD epochs go up, since
`tridiag` and `bidiag` with vectors get faster: about 80 minutes unattended)
and the docs.

## Expected gain

An estimate, not a measurement: with 16 workers the merges, which are product-
and root-bound, should run 4-5x faster, so `sstedc` at 4096 from 177 ms to
about 40-50 and `sbdsdc` from 718 to about 150-200. Then, for one 4096 × 4096:
eigh from 438 ms to about 300 (5.6x the CPU to about 8x), the SVD from 1535 ms
to about 1000 (2.3x to about 3.5x). At 2048: eigh 91 ms to about 60, the SVD
248 ms to about 140.

## Risks and open questions

- Load balance: deflation varies a lot between merges and matrices.
- Orthogonality: keep LAPACK's routines for the deflation, the roots and the
  vectors (`slaed3`'s and `slasd3`'s care with $\hat z$) rather than
  rewriting them; the parallel layer should only schedule.
- Workspace: each concurrent merge needs its own; size it once per call.
- Accelerate's threads: the leaves and merges run with its threading off; the
  top-level `sgemm`, if left to Accelerate, with it on. Check that switching
  per call is cheap and safe from worker threads.
- The CPU path for batches already spreads matrices over the cores; this is
  for one matrix or a few, which is the only place `tridiag` and `bidiag` are
  used.

## Where to start

- `src/eigh_tridiag.mm` and `src/svd_bidiag.mm`: the `solve` lambdas.
- A new plain C++ file beside `src/band_chase.cpp`, with the same threading
  pattern (`cpu_threads()`, `detail::parallel_for`).
- Tests: the `tridiag` and `bidiag` sections of `tests/test_eigh.cpp` and
  `tests/test_svd.cpp` already check residuals and orthogonality against
  LAPACK across sizes and structured spectra.
- `tuning/kernels.py`: bump the eigh and SVD epochs with a HISTORY line.

## References

- J. J. M. Cuppen, ["A divide and conquer method for the symmetric tridiagonal eigenproblem"](https://doi.org/10.1007/BF01396757), *Numerische Mathematik* 36 — the tearing and the rank-one merge.
- M. Gu and S. C. Eisenstat, ["A divide-and-conquer algorithm for the symmetric tridiagonal eigenproblem"](https://doi.org/10.1137/S0895479892241287), *SIAM J. Matrix Anal. Appl.* 16(1), 1995 — the stable eigenvectors LAPACK's `slaed3` computes.
- M. Gu and S. C. Eisenstat, ["A divide-and-conquer algorithm for the bidiagonal SVD"](https://doi.org/10.1137/S0895479892242232), *SIAM J. Matrix Anal. Appl.* 16(1), 1995 — LAPACK's `sbdsdc`.
- F. Tisseur and J. Dongarra, ["A parallel divide and conquer algorithm for the symmetric eigenvalue problem on distributed memory architectures"](https://doi.org/10.1137/S1064827598336951), *SIAM J. Sci. Comput.* 20(6), 1999 — parallelising inside the merges.
- G. Pichon, A. Haidar, M. Faverge and J. Kurzak, ["Divide and conquer symmetric tridiagonal eigensolver for multicore architectures"](https://doi.org/10.1109/IPDPS.2015.51), IPDPS 2015 — the task-based version for multicore CPUs (PLASMA), the closest to this plan.
