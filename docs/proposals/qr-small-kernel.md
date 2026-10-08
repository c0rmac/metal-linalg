# A QR kernel for batches of small matrices

Status: **done** in 2.16.0 (2026-10-08), with a kernel in registers beside
the one proposed, which a blocked kernel then replaced; see
[Done](#done-2026-10-08).

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

## Done (2026-10-08)

**Built as planned** (`qr_householder` in `shaders/QR_Householder.metal`,
`src/qr_householder.mm`), and beside it a kernel with the matrix in one
simdgroup's registers, as the band reduction's panels keep theirs: rows a
lane, the current column kept at index 0 by rotating the row, a step's dot
products four columns to a `simd_sum` on a float4, no barrier and no
threadgroup memory, four matrices to a threadgroup. GPU time, 4096 matrices
(M5 Pro):

| shape | `unblocked`'s kernel | threadgroup memory | registers |
|---|---|---|---|
| 16 x 16 | 1.07 ms | 0.81 | 0.19 |
| 32 x 32 | 2.6 | 1.41 | 0.64-0.82 |
| 64 x 64 | 15.9 | 15.8 | 4.4 |

A column a lane instead (each lane's dot products its own FMAs, the
reflector's vector shuffled across) was 2.2x slower at 32 x 32 and 6x at
64 x 64: `simd_sum` is cheap on this GPU.

**Then** the blocked kernel for mid-size matrices
([qr-mid-size-kernel.md](qr-mid-size-kernel.md)) beat both from 48 columns
up: the register kernel keeps up to 32 columns and 128 rows, and the
threadgroup-memory kernel, which no longer won anywhere (1024 of 200 x 30:
2.2 ms against 1.3), was removed before release, as was `unblocked`'s own
kernel.
