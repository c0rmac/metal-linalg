# The band SVD with vectors, overlapped further

Status: proposal, not started (2026-10-07).

## What

The `band` backend with singular vectors (2.15.0, [svd.md](../svd.md#with-singular-vectors-since-2150))
overlaps the GPU's work with the CPU's twice: Q1 and P1 are formed while the
CPU chases the band to bidiagonal, and Q1 Q2 and P1 P2 while it solves the
bidiagonal problem. Two gaps are left at 4096 on an M5 Pro:

| step | GPU | CPU |
|---|---|---|
| the band reduction | 158 ms | idle but the aggregates' T |
| Q1, P1 / the chase | 33 | 40 |
| Q Q2, P P2 / the divide and conquer | 133 | 113 |
| U, V^T | 43 | idle |

1. The GPU's Q2 and P2 (133 ms) outlast the divide and conquer (113): the
   CPU then waits about 20 ms.
2. Q2's blocks only start once the chase is over, though a group of 16
   sweeps is complete long before the chase is: the groups finish in order,
   and group G's share of Q2's work, like its share of the chase's, is
   proportional to n - 16 G, so Q2's work becomes available at the rate the
   chase runs.

## Plan

1. Split `bd_chase_apply`'s dispatch by passes of groups (it already loops
   over passes inside the kernel; a dispatch per few passes changes nothing
   else) and queue each behind an `MTLSharedEvent` wait for the value
   "groups up to G chased".
2. In `band_chase.cpp`, publish the last sweep completed in order (the
   pipeline already tracks each sweep's progress in `done[]`); a watcher
   builds the finished groups' T (`chase_blocks`, now one pass after the
   chase) and signals the event.
3. Q1 and P1 must come first on the GPU: they take 33 ms of the chase's 40,
   so the first passes would start about when the chase ends anyway unless
   Q1 and P1's aggregates are interleaved with them. Measure before
   reordering.
4. With Q2 and P2 off the divide and conquer's window, the GPU is free
   there: [the divide and conquer's top products on the GPU](divide-and-conquer-gpu-products.md)
   (48 of `sbdsdc`'s 100 ms at 4096) is the natural next step, and only pays
   once this is done.

## Effort

2-3 days: the event plumbing and the watcher are the work; the kernel is
unchanged.

## Expected gain

About 20 ms of 400 at 4096 (1.05x) from removing the wait; with the divide
and conquer's products on the freed GPU, about 60 ms (1.15x). Less at 2048,
where the steps are closer to balanced.

## Also open here

- Memory: the chase's blocks keep V (32 x 16) and T (16 x 16) dense, 3 KB a
  block, 400 MB a side at 8192. V's two empty tiles and T's lower one could
  go (576 floats a block instead of 768), or T be built on the GPU in the
  kernel's own pass.
- A batch of two at 1024 is still `bidiag`'s (54 ms against 59): its two
  slots overlap one matrix's divide and conquer with the next's reduction.
  The band backend takes a batch one matrix after another; a second slot
  would cost the memory above twice.

## Where to start

`apply_chase` and `bidiag_impl`'s `solve` in `src/svd_bidiag.mm`;
`pipeline()` in `src/band_chase.cpp`; `bd_chase_apply`'s pass loop in
`shaders/Svd_Bidiag.metal`.
