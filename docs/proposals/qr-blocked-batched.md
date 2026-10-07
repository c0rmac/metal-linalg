# The blocked QR for a batch at once

Status: **done** in 2.15.0 (2026-10-07): a batch is one pass of kernels and
batched products, and the blocked QR beats the streaming kernels at every
shape and batch measured; see [Done](#done-2026-10-07).

## What

The blocked QR ([qr-blocked.md](qr-blocked.md)) takes a batch one matrix
after another: its cost is the batch times one matrix's, so it only pays
for one matrix or a few large ones. Its work for a batch could instead be
issued once for all the matrices:

- the panel kernels with the matrix as the grid's third dimension (their
  TSQR scratch, V, T and the matrix's offset a matrix apart);
- every product as one batched MPS product (`MPSMatrix` over a batch of
  matrices, `matrixBytes` apart; `MPSMatrixMultiplication`'s `batchSize`);
- the merge kernel a threadgroup a matrix; the last columns by LAPACK, a
  matrix a core.

The panels' latency, which sets the time at 1024-2048, would then be paid
once a batch rather than once a matrix.

## Why: the measurements

M5 Pro, `sweep_qr`, median:

| shape | CPU | streaming | blocked (a matrix at a time) |
|---|---|---|---|
| 4 x 512 x 512 | 4.3 ms | 6.8 ms | 12.6 ms |
| 16 x 512 x 512 | 7.3 ms | 11.9 ms | 50.6 ms |
| 4 x 1024 x 1024 | 21.7 ms | 20.5 ms | 28.7 ms |
| 16 x 1024 x 1024 | 46.3 ms | 51.2 ms | 115 ms |
| 4 x 2048 x 2048 | 142 ms | 99 ms | 76 ms |

One 1024 x 1024 takes 6.9 ms, of which the panels are most: 16 of them, at
the latency of one, would put 16 x 1024 x 1024 at roughly 15-20 ms, 2.5-3x
the CPU's 46.

## Plan

1. The panel kernels (`shaders/Svd_Bidiag.metal`) take a matrix stride for
   A, V, T and the scratch, and the grid's z as the matrix; the band
   reductions pass a batch of one.
2. `mps()` with a batch: `MPSMatrixDescriptor` with `matrices` and
   `matrixBytes`.
3. The tail and the copies per matrix on the CPU's threads.
4. The routing: `qr_blocked_preferred` measured again, the batch up to which
   the blocked QR beats the streaming kernels and the CPU.

## Effort

2-3 days: the panel kernels are shared with the band reductions.

## Expected gain

2-3x the CPU path on batches of 4-16 matrices of 512-2048, which the CPU
and the streaming kernels share now.

## Where to start

`panel()`, `qr_blocks` and `qr_blocks_apply` in `src/band_reduce.mm`;
`one()` in `src/qr_blocked.mm`.

## Done (2026-10-07)

**Built as planned**, after one step first: each matrix is padded with zero
rows and columns to whole panels with twice their width in rows, so that
the GPU takes every column and the CPU's LAPACK tail, and its round trip
between the two passes, are gone (one 256 x 256: 2.5 to 1.5 ms; 300 x 1000
2.6 to 1.8). Then the panel kernels take a matrix stride for A, V, V T, T
and their scratch, and the grid's z as the matrix (0 and 1 for the band
reductions, whose timings are unchanged); the merge, scale and R kernels
likewise; every product is one batched MPS product.

**An MPS surprise.** A batched `MPSMatrixMultiplication` (macOS 27) steps
from one left or result matrix to the next by rows x rowBytes, whatever the
descriptor's `matrixBytes` says; only the right matrix honours it. The
blocked QR's views are submatrices (rows below the panel) of matrices a
fixed stride apart, so each batched view's descriptor is made a whole stride
tall, the product's own sizes saying what it reads, and each buffer has a
stride of slack past the last matrix (`mps()` in `src/band_reduce.mm`).

**Measured** (M5 Pro, `sweep_qr`, median, ms):

| shape | CPU | streaming | blocked |
|---|---|---|---|
| 4 x 512 x 512 | 4.24 | 7.15 | 3.75 |
| 16 x 512 x 512 | 7.47 | 11.9 | 6.37 |
| 64 x 512 x 512 | 24.3 | 41.6 | 20.0 |
| 4 x 1024 x 1024 | 21.4 | 20.9 | 10.5 |
| 16 x 1024 x 1024 | 47.7 | 52.0 | 25.3 |
| 4 x 2048 x 2048 | 142 | 97.8 | 37.6 |
| 1024 x 128 x 128 | 25.0 | 52.3 | 17.8 |
| 256 x 1024 x 64 | 16.3 | 24.9 | 13.6 |
| 64 x 256 x 256 | 5.19 | 12.2 | 5.10 |

It beats the streaming kernels everywhere (1.8-3.5x), and the
one-threadgroup kernel on large batches of 64-128-row matrices too (1024 x
128 x 128: 17.8 against 24.6), so the reduced backend hands it every call
it takes; the kernel crossover and the GPU-or-CPU boundary are re-measured
(kernel epoch qr 5). Batches of 128 x 128 to 256 x 256 under 64 matrices
stay the CPU's.
