# The blocked QR's panels under its trailing update

Status: proposal (2026-10-07).

## What

The blocked QR ([qr-blocked.md](qr-blocked.md)) runs its forward pass in
one order: an aggregate's panels, then its update of every column right of
it, then the next aggregate's panels. The panels are latency-bound (a TSQR's
leaves are a simdgroup each, its top one threadgroup), the trailing update
is MPS products that fill the GPU. With a look-ahead the next aggregate's
128 columns are updated first, so its panels can start while the rest of
the trailing update runs:

    stream P:  update(a-1 -> cols of a)  panels(a)  merge(a)   update(a -> cols of a+1)  panels(a+1) ...
    stream T:                                      trail(a-1)                            trail(a) ...

`trail(a)` (the columns from aggregate a + 2 on) waits for `panels(a)`;
the look-ahead update of aggregate a + 1's columns waits for `trail(a - 1)`,
which brought them up to date with the aggregates before.

## Why: the measurements

At 4096 x 4096 on an M5 Pro the forward pass takes 42 ms of the call's 62:
the panels about 25, the trailing updates 14, the updates inside aggregates
4. Run side by side, the panels and the trailing updates could take about
max(25, 14) instead of their sum.

## Plan

1. Two command queues; the matrix and V as two MTLBuffers over the same
   memory (newBufferWithBytesNoCopy twice), one per stream, so that Metal's
   hazard tracking orders each stream's own work and does not serialize the
   two; their dependencies by MTLSharedEvents, two an aggregate.
2. Scratch for the products (Z, W) per stream.
3. Measure first with the two streams' work as it is; the band reduction's
   version of this (its panels on a second queue, 2.13.0) was 14-54%
   slower, but its panels were both sides' and its updates bandwidth-bound,
   where QR's rank-128 updates are compute-bound.

## Effort

About a day.

## Expected gain

Up to 14 ms of 62 at 4096 (1.2x), less at 1024-2048 where the panels are a
larger share.

## Where to start

`qr_blocks` in `src/band_reduce.mm`.
