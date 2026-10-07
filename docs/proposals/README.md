# Proposals

Work that was scoped but not done, written down so that it can be picked up
later. Each file says what the change is, the measurements behind it, a plan,
the effort and the expected gain, and where to start. The numbers are an
Apple M5 Pro's (20 GPU cores, 18 CPU cores).

## Open

| proposal | affects | time at stake | effort | expected gain |
|---|---|---|---|---|
| [The blocked QR's fixed costs and panels](qr-fixed-costs.md#done-2026-10-07) (what is left) | QR, one matrix of 512-4096 | the TSQR's leaves, top and rebuild about 3.8 of 6.6 ms at 1024; updates inside aggregates 12% | 1-2 days | a few % each: the leaves and top as one dispatch, the in-aggregate updates as kernels of their own |
| [The CPU path's divide and conquer](cpu-path-divide-and-conquer.md) | eigh and SVD with vectors on the CPU, one matrix | `sstedc` 43 of 239 ms, `sbdsdc` 133 of 439 at 2048 | about 2 days | eigh 1.15-1.25x, SVD ~1.34x at 1024-2048 (estimate) |

The CPU path's divide and conquer matters less on the M5 Pro since 2.15.0,
where the GPU takes single matrices from about 512, but more on Macs whose
GPU is weaker against their CPU.

**Not code, but open:**

- Measurements from other Macs: every Mac but the M5 Pro runs an estimated
  policy (refitted from the M5 Pro's timings). Each needs someone with that
  Mac to run `python3 tuning/run.py` on an idle machine.

## Done

In 2.15.0 (2026-10-07); see [the study](../studies/proposals-2-15-apple-m5-pro.md):

| proposal | outcome |
|---|---|
| [Parallel divide and conquer](parallel-divide-and-conquer.md) | `sstedc` 176 to 50 ms, `sbdsdc` 753 to 100 at 4096; eigh with vectors 1.4-1.5x, the SVD's 1.7-1.9x |
| [A faster TSQR top kernel](tsqr-top-kernel.md) | the top a tree of triangle pairs, and the panels off IEEE arithmetic: a panel 140 to 76 us; svdvals on `band` 1.13-1.30x, eigvalsh 1.08-1.18x |
| [Fused small products](fused-small-products.md) | two kernels a block instead of three products: 2-5% at 1024-2048 |
| [Symmetric trailing update](symmetric-trailing-update.md) | the update on the lower triangle, mirrored: eigvalsh on `band` 1.08x at 4096, 1.17x at 8192; X from the lower triangle lost to MPS |
| [Band width per device](band-width-per-device.md) | `values_band_width`, measured by stages 3b and 4b |
| [Band threshold tie-break](band-threshold-tie-break.md) | a finer grid, and thresholds compared on the points where they disagree |
| [The bulge chase under the band reduction](chase-under-reduction.md) | eigvalsh and svdvals on `band`, one matrix: 1.04-1.05x, not the 1.15-1.2x estimated (each sweep runs to the band's end, which the GPU finishes last, so only about 6% of the chase can go before it) |
| [The band SVD with vectors, overlapped further](band-vectors-overlap.md) | the GPU found to be the bottleneck; its two idle gaps closed (Q1 and P1 queued during the reduction, Q2 and P2 released in two chunks as the chase finishes them): 1.04x at 4096, 1.06x at 2048 (alternating runs) |
| [The divide and conquer's top products on the GPU](divide-and-conquer-gpu-products.md) | for one matrix, products of 1 GFLOP or more as MPS products on the merges' memory in place: eigh with vectors 1.075x at 4096, the SVD on `bidiag` 1.046x |
| [IEEE arithmetic in the remaining shaders](ieee-mode-audit.md) | the SVD's Jacobi kernel's rotation and output on fast division and square roots (`rsqrt` with two Newton steps, for V's orthogonality): 1.14-1.17x on batches of small matrices; nothing left in the one-stage reductions |
| [Long single-threadgroup kernels and the watchdog](gpu-watchdog-long-kernels.md) | the whole-matrix Jacobi kernels split a long solve over dispatches of a few rounds (about 15 ms each, from 1.7-2.8 s for one threadgroup), bit for bit the same and in the same time |
| [The blocked QR for a batch at once](qr-blocked-batched.md) | a batch is one pass of kernels and batched MPS products (whose left and result matrices ignore matrixBytes: views a whole stride tall); beats the streaming kernels everywhere, 1.8-3.5x; 16 x 1024^2 in 25 ms against the CPU's 48 |
| [QR's large-matrix clause by rows and k](qr-large-clause.md) | the clause on sqrt(M k) rather than k (it fitted better than the work's own size), tall large shapes in the grid, the fit in both orders: 8192 x 512 to the GPU (8.5 ms against 48); the M5 Pro's QR row 1.013x regret |
| [The blocked QR's fixed costs and panels](qr-fixed-costs.md) | padded to whole panels, no CPU round trip: one 256^2 2.5 to 1.5 ms; 32-column panels and 256-row leaves tried, slower |
| [Blocked QR on the band reduction's panels](qr-blocked.md) | one large matrix 2.1x the streaming kernels at 1024, 2.6x at 2048, 3.7x at 4096 (62 ms against 231; the CPU 634), 5x on tall 4096 x 1024 and 8192 x 512 |
| [Less GPU work for the band SVD with vectors](band-vectors-gpu-work.md) | `bd_chase_apply`'s blocks carry Y = -T^T V^T instead of T (two products a step, not three): Q2 and P2 98 ms instead of 135 at 4096; the CPU then the bottleneck, the divide and conquer's top products on the GPU once Q2 and P2 are done: 1.04-1.05x at 2048-4096 |
| [Two-stage reduction with vectors](two-stage-vectors.md) | the SVD with vectors on `band`: 1.21x `bidiag` at 1024, 1.45x at 2048, 2.35x at 4096, about 2.7x at 8192 with the overlap above (Q2 applied by a pipeline of groups, 60-64 ms a side at 4096; Q = Q1 Q2 and P = P1 P2 formed under the CPU's chase and divide and conquer) |

Found along the way, also in 2.15.0: the tridiag and bidiag backends' block
reflectors' T from a Gram matrix (eigh with vectors 1.1x at 4096); the
one-stage reductions and bisection off IEEE arithmetic (eigvalsh on
`tridiag` 1.07-1.11x); the sweeps' correctness gate failing from N ~ 6500
(the CPU reference outlasting the GPU's watchdog).

## Tried and rejected

So that they are not retried without a new idea:

- Look-ahead in the band reduction, the next block's panel factored while the
  rest of the trailing matrix is updated: no overlap within one command
  buffer, and 14-54% slower with the panel on a second queue. See
  [the two-stage study](../studies/two-stage-apple-m5-pro.md), section 9.
- The blocked QR's panels 32 columns wide (1.4-1.5x slower at 512-2048), or
  their leaves 256 rows (slower from 512 on, and a rank-one matrix lost its
  orthogonality): [qr-fixed-costs.md](qr-fixed-costs.md).
- The blocked QR's panels on a second queue beside its trailing update
  (2.15.0): the queues overlap, but the panels ran half as fast beside MPS's
  products; no gain ([qr-look-ahead.md](qr-look-ahead.md#tried-2026-10-07)).
- LAPACK's `sbdsvdx` (bisection by index range) in place of `sbdsqr`: 30x
  slower on one core, and wrong when the bidiagonal splits.
- X = A22 V T from the lower triangle only (2.15.0), in four versions: each
  tile read twice, staged or straight from device memory, split over
  threadgroups by column chunks, double-buffered. The best took 470 us at
  4096 against MPS's 365, and 150 against 62 at 2048 (MPS reads the cached
  matrix better). The update writes the mirror instead.
- `sb_update`'s tiles for the general reduction's update (2.15.0): no
  symmetry to save memory on, and no faster than MPS (svdvals unchanged).
- The small products as a thread an entry over its rows from device memory:
  28 and 19 us a block, slower than MPS's 22 (latency).
- For the two-stage SVD with vectors' Q2: a sliding window in threadgroup
  memory, wider strips, smaller threadgroups, and a simdgroup a group with
  the groups staggered all lost to the plain kernel; see
  [two-stage-vectors.md](two-stage-vectors.md#prototype-2026-10-07). For
  the pipelined kernel that replaced it: the next tile prefetched, fewer
  device fences, less threadgroup memory, and the two sides at once (slower:
  177 ms against 132); see [Done](two-stage-vectors.md#done-2026-10-07).

**How the profile numbers in the 2.13 proposals were taken.** The per-kernel
times came from a temporary build of `src/band_reduce.mm` that committed
every panel kernel and every matrix product in a command buffer of its own
and summed `GPUEndTime - GPUStartTime` per kind of operation. Each operation
then carries about 10 us of launch overhead it does not have in the real,
single command buffer a block: the profiled stage summed to 204 ms for
svdvals at 4096, against about 180 ms in the real run. In 2.15.0 kernels were
timed instead with scratch harnesses that dispatch one kernel many times in
one command buffer (GPU time per dispatch), and changes with the whole call
(`sweep_eigh`, `sweep_svd`), side by side with the previous build.
