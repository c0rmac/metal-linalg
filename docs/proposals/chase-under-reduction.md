# The bulge chase under the band reduction

Status: **done** in 2.15.0 (2026-10-07), for 1.04-1.05x rather than the
1.15-1.2x estimated below: the chase's sweeps cannot get far ahead of the
GPU; see [Done](#done-2026-10-07).

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

## Done (2026-10-07)

**Built as planned.** Both band reductions report their blocks' command
buffers (`BandWatch`); the caller starts the chase on its own threads,
copies each block's finished rows (general) or columns (symmetric) into the
chase's band as the block completes, and raises `ChaseReflectors::ready_rows`;
sweep 0 waits for the rows its task reaches, the others already wait on the
sweep before. Only for one matrix: in a batch the chase of one matrix
already runs under the next one's reduction, and moving the first one's
chase made a batch of two 0.87x.

**Why the estimate was wrong.** The first version gained nothing: the GPU's
reduction ended at 96 ms (eigvalsh, 4096) and the chase threads at 132, as
before. Each thread ran one sweep at a time, and the first sixteen sweeps,
waiting at the GPU's frontier for rows they would need only further down,
held all the threads. Letting a thread keep several sweeps open and run
whichever may go on (the trailing mode of `pipeline` in `band_chase.cpp`)
fixed that, but not the rest: every sweep runs to the band's end, which the
reduction finishes last, and each sweep trails the one before by about a
block (its task t needs the previous sweep's task t + 2). With rows 0 .. F
final, at most about F / 16 sweeps can have started, each stopped near F,
which at F = n is about 6% of the chase's work (at 4096, some 256 sweeps of
4095, each partway). The estimate counted the work on the upper rows as free
to go early; it is not, because a sweep's work on them waits for the sweep
before it, and so on back to sweep 0 at the frontier.

**Measured** (M5 Pro, one matrix, alternating builds): eigvalsh on `band`
1.04x at 2048, 1.05x at 4096 (136 to 129 ms); svdvals 1.05x at both (204
to 195 ms at 4096); a batch of two unchanged. The gain is that 6% and the
band's copy moving off the critical path. The values are the same bit for
bit as with the chase after the reduction (tests in both suites).

