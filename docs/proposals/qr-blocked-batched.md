# The blocked QR for a batch at once

Status: proposal (2026-10-07).

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
