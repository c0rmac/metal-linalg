# The bulge chase on the GPU for batches

Status: in progress (2026-10-09).

## What

The batch backends' two stages (`bidiag_batch`, `tridiag_batch`;
`src/svd_bidiag_batch.mm`) chase each matrix's band to bidiagonal or
tridiagonal on the CPU, a matrix a core (`band_chase.cpp`). Chase every
matrix of a chunk at once on the GPU instead: a threadgroup a matrix, the
sweeps in lockstep. Sweep $s$'s task $t$ may run once sweep $s - 1$ has
finished task $t + 2$ (the blocks they touch are one apart, as the CPU's
pipeline relies on), so at step $\tau$ sweep $s$ runs task $\tau - 3s$: every
sweep active at a step on a simdgroup of its own, a barrier between steps,
about $3n$ steps of 16 x 16 work. The reflectors, with vectors, go into
`bd_chase_apply`'s blocks as the CPU's chase writes them.

## Why: the measurements

M5 Pro, eigh with eigenvectors in two stages, 16 x 1024^2 (65 ms, two chunks
of 8): the CPU's solve of a chunk takes 22-23 ms and is the critical path,
the GPU idle about 20 ms of the call. Per matrix (two threads each): the
chase 11.7 ms, the divide and conquer 7.4, the chase's blocks 0.8. The SVD's
16 x 1024^2: chase 132 ms of 287 summed over the matrices. With singular or
eigen values alone the chase is almost all of the CPU's work, and the reason
eigvalsh's two stages lost to LAPACK (0.52-0.91x).

## Plan

1. `sb_chase`, the symmetric chase (ssb2st's tasks), on the band as the CPU
   builds it; test against `band_to_tridiagonal` (d, e, reflectors).
2. Into `eigh_band_batch`: the chunk's tail on the CPU, the chase committed
   with Q1's products, the divide and conquer once it is done.
3. The general (bidiagonal) chase likewise for `bidiag_batch`.
4. Revisit the batch limits, and eigvalsh and svdvals in two stages.

## Expected gain

Several ms a matrix of the CPU's critical path: about 1.4x at 16 x 1024^2
with vectors if the chase runs in 3-6 ms a chunk; more for values alone.
