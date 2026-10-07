# Blocked QR on the band reduction's panels

Status: **done** in 2.15.0 (2026-10-07); see [Done](#done-2026-10-07).

## What

QR's GPU path for large matrices, `qr_streaming_amx_reduced`, factors each
32-column panel in one threadgroup (1024 threads, a reduction over all the
rows for every column), then updates the trailing matrix and forms Q with a
grid of threadgroups, each streaming a 32-column tile of the matrix through
threadgroup memory. Both stages run far below the GPU's rate. The band
reduction (2.13.0) already has what a QR needs: panels factored by TSQR
across threadgroups, their reflectors in the compact form H = I - V T V^T,
and updates as MPS products. Factor the matrix with those instead:

1. Column panels by the band reduction's panel kernels (`bd_panel_qr`, or
   `bd_tsqr_leaf`, `_top`, `_rebuild` for a tall one), on the row-major
   matrix in place: a panel's rows are read contiguously, and neither the
   input nor Q needs transposing.
2. Panels gathered into aggregates of 128 columns: inside one, each panel's
   H applied to the aggregate's columns right of it (two products); the
   aggregate's T merged from its panels' (from the Gram matrix Y^T Y, a
   small kernel); then I - Y Ta Y^T applied to the rest of the matrix by
   three products, rank-128 updates rather than rank-32.
3. The last columns (fewer than a panel, or rows fewer than twice its width)
   by LAPACK on the CPU.
4. Q from the identity by the aggregates backwards, three products each.

## Why: the measurements

M5 Pro, one matrix, 2.14 (`sweep_qr`, median):

| shape | CPU (LAPACK) | `streaming_reduced` |
|---|---|---|
| 1024 x 1024 | 16.9 ms | 14.8 ms |
| 2048 x 2048 | 94.0 ms | 49.3 ms |
| 4096 x 4096 | 634 ms | 231 ms |
| 4096 x 1024 | 67.1 ms | 63.3 ms |
| 8192 x 512 | 48.8 ms | 63.6 ms |

A 4096 x 4096 QR with Q is about 183 GFLOP: 231 ms is 0.8 TFLOP/s, where
the MPS products of the band reduction run at 5-6. Tall matrices fare worst:
the panel's single threadgroup walks every row for each of its columns.

## Plan

As above; the routing unchanged, the reduced backend handing one matrix (or
a few large ones) to the new path, so that the routing's re-measure (kernel
epoch qr 4) finds where the GPU now wins.

## Effort

1-2 days.

## Expected gain

2-4x on single large matrices, more on tall ones.

## Done (2026-10-07)

**Built as planned** (`src/qr_blocked.mm`; the GPU part, `qr_blocks` and
`qr_blocks_apply`, in `src/band_reduce.mm` beside the panels it reuses).
What the measurements decided along the way, at 4096 x 4096:

| step | ms |
|---|---|
| rank-32 updates, a command buffer, CPU-side copies in and out | 135 |
| aggregates of 128 columns (rank-128 updates, Q's formation 37.5 to 16 ms) | 95 |
| panels 16 wide inside the aggregates (32: 1.15x slower; 8: no faster, twice the dispatches) | 83 |
| committed an aggregate or two at a time (the GPU had waited 15-28 ms for the encoding), Q formed in the caller's memory | 63 |
| MPS's product kernels cached by shape (encoding 16 to 11 ms), R written by the GPU, Q's formation queued behind an event | 62 |

Where the 62 ms go: the forward pass 42 on the GPU (its panels about 25,
the rank-128 updates 14, the updates inside aggregates 4), Q's formation 16
(6 TFLOP/s), the rest the CPU's scan of the input, the first aggregate's
encoding and the last columns' round trip.

**Measured** (M5 Pro, `sweep_qr`, median, one matrix unless a batch is
given):

| shape | CPU | streaming | blocked | blocked / best before |
|---|---|---|---|---|
| 256 x 256 | 0.85 ms | 2.68 ms | 1.54 ms | CPU still faster |
| 512 x 512 | 3.68 ms | 5.85 ms | 3.00 ms | 1.23x |
| 1024 x 1024 | 16.9 ms | 14.8 ms | 6.9 ms | 2.1x |
| 2048 x 2048 | 93.8 ms | 49.3 ms | 18.7 ms | 2.6x |
| 4096 x 4096 | 634 ms | 231 ms | 62.3 ms | 3.7x |
| 4096 x 1024 | 67.1 ms | 63.3 ms | 12.4 ms | 5.1x |
| 8192 x 512 | 48.8 ms | 63.6 ms | 9.7 ms | 5.0x |
| 2 x 1024 x 1024 | 19.5 ms | 16.8 ms | 14.2 ms | 1.2x |
| 4 x 2048 x 2048 | 142 ms | 99 ms | 76 ms | 1.3x |

It takes a batch one matrix after another, so the streaming kernels (a
batch at once) and the CPU (a matrix a core) win batches of small and
mid-size matrices: the reduced backend hands a call to it for one matrix,
or when batch x 512 <= min(m, n) (`qr_blocked_preferred`; `QR_BLOCKED=0`
turns it off). Accuracy is LAPACK's or a little better (reconstruction and
orthogonality 2.1e-6 at 4096 against LAPACK's 2.4e-6: the TSQR panels).
Up to 16384 rows (the TSQR's leaves fit one threadgroup's top: 8-column
panels above 8192 rows); taller matrices stay on the streaming kernels.

The routing is re-measured at the new kernels (epoch qr 4): until then the
M5 Pro's row is 2.14's, which sends one matrix to the GPU only from k =
1024, so a 512 x 512 or a tall 8192 x 512 still goes to the CPU.

**Left**: a batch at once, as a proposal ([qr-blocked-batched.md](qr-blocked-batched.md)).
The panels under the trailing update on a second queue were tried and gave
nothing ([qr-look-ahead.md](qr-look-ahead.md#tried-2026-10-07)).
