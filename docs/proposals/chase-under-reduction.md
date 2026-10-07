# The bulge chase under the band reduction

Status: proposal, not started (2026-10-07).

## What

The `band` backends (eigenvalues or singular values alone) run their two
stages one after the other: the GPU reduces the matrix to a band, then the
CPU chases the band to tridiagonal or bidiagonal. But the chase does not
need the whole band at once. Sweep s's t-th task touches rows and columns
from about s + t b; the reduction finishes rows k b .. (k + 1) b of the band
with block k. So the chase can start as soon as the first blocks are done
and trail the GPU down the matrix, its sweeps pipelined as they are now,
each task waiting for the block it reaches.

## Why: the measurements

At 4096 on an M5 Pro (2.15.0): svdvals on `band` 197 ms, of which the GPU's
reduction about 150, the chase 35, bisection 12; eigvalsh 140, the chase
about 35. The chase's work on rows r .. r + dr comes from every sweep that
reaches them, so it grows down the matrix: what the last tenth of the rows
holds, which can only be chased once the GPU has finished, is about a fifth
of the chase. The rest could run under the reduction, with the CPU otherwise
idle then.

## Plan

1. `band_reduce_general` and `band_reduce_symmetric` already queue a command
   buffer a block (`BandKeep::done` keeps them for the SVD with vectors);
   expose the last completed block (a completion handler a block, or the
   command buffers' status).
2. Copy the band into the chase's storage (`sl.ab`) as blocks complete,
   rather than after the reduction.
3. In `band_chase.cpp`'s pipeline, a task that reaches row r waits for the
   block holding r + 2 b (its bulge's reach) to be complete; sweep 0 does the
   waiting, the others already wait on the sweep before.
4. The LAPACK tail (the last columns, fewer than 2 b) is the reduction's
   last step: the chase's last rows wait for it as for a block.

## Effort

2-3 days: the waiting is simple; the band copies and the tail are the
details.

## Expected gain

About four fifths of the chase off the critical path: svdvals on `band`
about 1.15x at 4096 (197 to about 170 ms), eigvalsh about 1.2x (140 to
about 115). Less at 2048 and below, where the chase is a smaller share. The
SVD with vectors on `band` gains nothing: its GPU is the bottleneck
([band-vectors-overlap.md](band-vectors-overlap.md#done-2026-10-07)).

## Where to start

`bidiag_impl`'s `reduce` and `solve` (`src/svd_bidiag.mm`), the eigensolver's
in `src/eigh_band.mm`; `pipeline()` in `src/band_chase.cpp`.
