# A QR kernel for batches of small matrices

Status: proposal (2026-10-07).

## What

QR's GPU path for small matrices, `qr_unblocked`, gives each matrix a
threadgroup and works column by column through device memory: per column a
norm, a broadcast, dot products and an update, each behind a threadgroup
barrier, one thread building the block reflector's T, and the matrix padded
to 32 rows with a full padded square Q. Around it the CPU scans and copies
the input, and copies R and Q back. Build it as the SVD's `golub_kahan` and
the eigensolver's `ql` kernels are built, which batches of small matrices
made several times faster:

1. One threadgroup per matrix, the whole matrix in threadgroup memory at an
   odd row stride (`n | 1`, so a column walk touches every bank once), a
   thread per row.
2. Householder QR column by column as LAPACK's `sgeqr2`: the norm by a
   simdgroup sum and one slot per simdgroup, the reflector from a fast
   division and square root with one Newton step (`div_nr`, `sqrt_nr`), the
   column sums `v^T A(:, c)` split over groups of lanes (`lanes_per_column`),
   the update row-local; three barriers a column.
3. R written out, then Q formed in place from the reflectors as `sorg2r`
   does (`golub_kahan`'s step 4).
4. The input read row-major straight from the caller's buffer, scanned and
   scaled by a power of two on the GPU, NaN for a non-finite matrix; Q and R
   written row-major straight to the caller's buffers: no CPU pass at all.

The backend `unblocked` hands it every call it fits (threadgroup memory: up
to about 90 x 90), so the routing's kernel crossover is re-measured rather
than a backend added.

## Why: the measurements

M5 Pro, 2.15.0, buffers already touched (ms):

| 4096 matrices of | CPU | `qr_unblocked` | blocked QR | GPU + CPU shared |
|---|---|---|---|---|
| 16 x 16 | 1.08 | 2.64 | 1.77 | 0.98 |
| 32 x 32 | 2.66 | 5.02 | 3.53 | 1.86 |
| 64 x 64 | 8.89 | 17.3 | 13.2 | 7.36 |

At 4096 x 16 x 16, `qr_unblocked`'s kernel took 1.07 ms and its host passes
0.9 (the input scanned and copied 0.4, R and Q copied out 0.5): as much as
the CPU path's whole call. About 120 matrices are resident at a time (six a
core), each a latency-bound chain.

## Plan

`shaders/QR_Householder.metal` and `src/qr_householder.mm`, after
`Svd_GolubKahan.metal` and `svd_golub_kahan.mm`; tests beside
`qr_unblocked`'s; the kernel epoch, the re-measure, README's tables.

## Effort

A day.

## Expected gain

Several times the CPU path on large batches of 16 x 16 to 64 x 64 (a
32 x 32's QR is about 60 thousand flops; a kernel of `golub_kahan`'s kind
runs the SVD of 4096 of them in about 8 ms, and QR is a fraction of that
work).

## Where to start

`svd_golub_kahan`'s steps 1, 2 (the left reflectors) and 4.
