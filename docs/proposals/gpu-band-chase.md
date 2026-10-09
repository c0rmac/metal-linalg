# The bulge chase on the GPU for batches

Status: **tried and rejected** (2026-10-09): the chase on the GPU is no
faster than the CPU's cores; see [Tried](#tried-2026-10-09).

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

## Tried (2026-10-09)

Step 1 was built ([`sb_chase.metal`](gpu-band-chase/sb_chase.metal),
[`chase_test.mm`](gpu-band-chase/chase_test.mm) to time it against
`band_to_tridiagonal`) and gives the same tridiagonal (eigenvalues within
3-8e-6 of the CPU's). What removing the CPU's chase could save, measured by
skipping it in `eigh_band_batch` (the wrong answer, the right timing):

| eigh with vectors | as now | no chase |
|---|---|---|
| 8 x 512^2 | 11.6 ms | 6.5 |
| 16 x 768^2 | 37.4 | 24.3 |
| 8 x 1024^2 | 46.4 | 29.4 |
| 16 x 1024^2 | 68.9 | 55.1 |
| 32 x 1024^2 | 114.8 | 101.8 |

The GPU's chase, a threadgroup of 1024 threads a matrix, with the
reflectors kept:

| n | 1 matrix | batch | CPU, a thread a matrix |
|---|---|---|---|
| 512 | | 4.2 ms (16) | 1.8 |
| 768 | | 7.3 ms (8) | 4.0 |
| 1024 | 11.0 ms | 11.3 ms (16) | 7.0 |

A matrix's chase is one core's work and the batch runs beside it for free,
but one core is not enough: the 3,200 steps at 1024 cost 3.4 us each, of
which the barrier is 0.3 (the steps with their work skipped: 0.85 ms). The
steps are bound by the core's issue rate, not by their latency: 512 threads
(three sweeps a simdgroup at the widest step) take 11.5 ms, 1024 (one or
two) 11.0. Each sweep's 16 x 16 block leaves half a simdgroup idle, so two
sweeps a simdgroup were tried (sweeps $s$ and $s + 2$, whose tasks share
their parity; sums within the half by shuffles, the odd task's 15 column
sums as one transposed reduction,
[`sb_chase2.metal`](gpu-band-chase/sb_chase2.metal)): 16.4 ms, slower, the
half-simdgroup reductions costing more than the native `simd_sum`s they
replaced.

At 11 ms a chunk the GPU's chase costs the GPU about what the CPU's costs
the CPU (the CPU's chase of a chunk runs under the GPU's work for Q1), so
the table's gains would be spent on it: at best about 1.0-1.1x at 1024.
For eigenvalues alone the CPU's cores already run the chases of a batch in
parallel (16 x 1024^2: 16 x 7 ms over 14 cores, 8-9 ms, against the GPU's
11.3). The chase would need to run 3x faster on one core, or a matrix's
chase spread over cores, neither of which this kernel's design (lockstep
sweeps, a barrier a step) can reach.

## If picked up again

Spreading a matrix over several cores would help one matrix, not a batch:
a batch already fills the cores, and the GPU's 20 cores at 11 ms of a core
a matrix chase fewer matrices a second than the CPU's 14 free cores at 7 ms.
What would change the verdict is a chase with fewer instructions a step,
about a third of today's, for which this design (16 x 16 blocks on 32
lanes, the band in device memory) leaves no obvious room; keeping the
band's window in threadgroup memory (32 KB holds 240 columns of the 33-row
band, the sweeps in tiles of columns as PLASMA's `ssb2st` groups them)
would save the loads' issue slots, not the arithmetic's. A project of days
for an uncertain gain.
