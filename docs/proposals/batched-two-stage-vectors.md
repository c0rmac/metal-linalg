# Batches of a few large matrices with vectors in two stages

Status: proposal, not started (2026-10-09).

## What

`tridiag_batch` and `bidiag_batch` reduce every matrix of a batch by the same
dispatches, a threadgroup a matrix and panel. For a few matrices of 768-1024
that leaves most of the GPU idle: 8 matrices of 1024 x 1024 run on 8 of an M5
Pro's 20 cores, and with vectors the CPU path is 1.15x ahead (0.87x for
`bidiag_batch`). The singular values alone of such batches already go in two
stages (2.17.0): every matrix to a band by blocks whose updates are batched
products (`bb_panel` for the panels), then the bands to bidiagonal on the
CPU's cores, 2.0-2.3x the direct reduction at 1024^2 (1.5x the CPU path at
8 x 1024^2, 3.3x at 16). With vectors the same
needs the transformations kept and applied, as the `band` backend does for
one matrix:

1. Keep each block's panel reflectors and T (the values path discards them).
2. The chase per matrix on the CPU's cores with its reflectors kept
   (`band_to_bidiagonal` with vectors, as `band` calls it), then the
   divide and conquer per matrix.
3. The chase's reflectors applied to U and V^T on the GPU (`band`'s kernels,
   batched), then the first stage's blocks as batched products.

The eigensolver's counterpart is the same with `tridiag_batch`'s pieces.

## Why: the measurements

M5 Pro, 2.17.0, against the CPU path (LAPACK's drivers, a matrix a core):

| | CPU path | batch backend, direct | two stages, values alone |
|---|---|---|---|
| SVD, 8 x 1024^2, vectors | 1.0 | 0.87x | — |
| SVD, 8 x 1024^2, values | 1.0 | about 0.7x | 1.5x |
| SVD, 16 x 1024^2, values | 1.0 | 1.6x | 3.3x |

One matrix: `band` with vectors is 1.36x `tridiag` at 4096 and 2.3x `bidiag`
for the SVD, the same method.

## Plan

1. `svd_bidiag_batch`'s band path with vectors: keep `bb_panel`'s V and T
   per block (a buffer of k x k per matrix, as `band` keeps), chase with
   vectors per matrix on `cpu_threads_beside_gpu()`, divide and conquer, then
   the back-transformations as `band`'s, over the batch.
2. Measure 4-32 matrices of 512-2048 against the CPU path and `bidiag`; keep
   it where it wins (a window in the policy, as `values_band`).
3. The eigensolver's `tridiag_batch` likewise.
4. Tests (the suites' batch sections), sweep grid points, re-measure.

## Effort

Three to five days: the SVD's first, then the eigensolver's.

## Expected gain

An estimate: the reduction is about half of the time at 1024 with vectors,
and two stages took the values-alone reduction 1.5-2.3x faster; the chase
with vectors and the extra back-transformation take some of that back. About
1.2-1.5x the CPU path for 8-16 matrices of 768-1024.

## Where to start

`direct()` and the band flag in `src/svd_bidiag_batch.mm`; the `band`
backend's chase with vectors and back-transformation kernels
(`src/band_reduce.mm`, `src/band_chase.cpp`, `src/svd_bidiag.mm`, `src/eigh_band.mm`).
