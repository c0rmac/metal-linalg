# The CPU path's divide and conquer on every core

Status: proposal, not started (2026-10-07).

## What

For one matrix, or a batch smaller than the cores, the CPU path calls
LAPACK's drivers whole: `ssyevd` for eigh with vectors, `sgesdd` for the SVD.
Their divide and conquer (`sstedc`, `sbdsdc`) runs on one core. The `tridiag`
and `bidiag` backends now use `src/divide_conquer.cpp`, which runs it on
every core (2.15.0). Call the drivers' steps one by one in the CPU path too:
`ssytrd`, `tridiagonal_eigensystem`, `sormtr` for eigh, and `sgebrd`,
`bidiagonal_svd`, `sormbr` twice for the SVD, keeping the QR first for tall
matrices as now.

## Why: the measurements

One matrix on an M5 Pro, Accelerate (sweep timings, and the divide and
conquer alone on the tridiagonal of `ssytrd` and the bidiagonal of `sgebrd`
of a Gaussian matrix):

| | CPU path | its divide and conquer | on every core |
|---|---|---|---|
| eigh, 1024 | 41 ms | `sstedc` 12 ms | 3.8 ms |
| eigh, 2048 | 239 ms | 43 ms | 10 ms |
| SVD, 1024 | 78 ms | `sbdsdc` 28 ms | 7.8 ms |
| SVD, 2048 | 439 ms | 133 ms | 22 ms |

So, if the rest of the driver stays as it is: eigh 1.25x at 1024 and 1.16x at
2048; the SVD 1.34x at both. Above 2048 the GPU backends take one matrix
anyway; below about 1024 `tridiag` and `bidiag` are not used, and these are
the calls the CPU path serves.

## Plan

1. `eigh_cpu` for batches below `cpu_threads()`: query the workspace,
   `ssytrd`, the divide and conquer with `cpu_threads()` threads, `sormtr`.
   Match `ssyevd`'s scaling (it scales A when its norm is outside the safe
   range; the library already scales every matrix by a power of two, so
   this may be moot).
2. `svd_cpu` likewise: `sgebrd`, `bidiagonal_svd`, `sormbr("Q")` on U and
   `sormbr("P")` on VT, inside the existing QR-first wrapper for tall input.
3. Measure against the drivers at 128-2048 and keep them where they win
   (below a few hundred the drivers' own small paths are hard to beat).
4. The CPU path is every boundary's baseline: re-measure eigh and the SVD
   (epochs up).

## Effort

About two days: a day for the two drivers and their workspaces, the rest for
tests (the suites' CPU sections and the structured cases) and the
re-measure.

## Expected gain

1.15-1.35x for one matrix of 1024-2048 on the CPU path (an estimate: it
assumes `ssyevd` and `sgesdd` spend the rest as `ssytrd`/`sormtr` and
`sgebrd`/`sormbr` would alone). Batches gain only while there are fewer
matrices than cores; `lapack_batches` already gives each core a matrix.

## Where to start

`core::detail::eigh_cpu` in `src/eigh.mm` and `svd_cpu` in `src/svd.mm`;
`tridiagonal_eigensystem` and `bidiagonal_svd` in `src/divide_conquer.h`.
