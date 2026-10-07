# Proposals

Work that was scoped but not done, written down so that it can be picked up
later. Each file says what the change is, the measurements behind it, a plan,
the effort and the expected gain, and where to start. The numbers are an
Apple M5 Pro's (20 GPU cores, 18 CPU cores).

## Open

| proposal | affects | time at stake | effort | expected gain |
|---|---|---|---|---|
| [Two-stage reduction with vectors](two-stage-vectors.md) (prototyped, parked) | SVD with vectors | the reduction, ~720 of ~920 ms at 4096 | 1-2 weeks | 1.2x with the prototype's kernel; 1.5x+ if Q2 is applied in 80 ms a side |
| [The CPU path's divide and conquer](cpu-path-divide-and-conquer.md) | eigh and SVD with vectors on the CPU, one matrix | `sstedc` 43 of 239 ms, `sbdsdc` 133 of 439 at 2048 | about 2 days | eigh 1.15-1.25x, SVD ~1.34x at 1024-2048 (estimate) |
| [IEEE arithmetic in the remaining shaders](ieee-mode-audit.md) | the SVD's Jacobi kernel, svdvals on `bidiag` | 1.21x measured with fast math; 7% left on `bidiag` | about a day | 1.2x and up to 7% |
| [The divide and conquer's top products on the GPU](divide-and-conquer-gpu-products.md) | eigh and SVD with vectors, one large matrix | 23 of 50 ms, 48 of 100 at 4096 | about 2 days | ~1.05x (estimate) |
| [Long single-threadgroup kernels and the watchdog](gpu-watchdog-long-kernels.md) | forced and estimated routings, the tests | calls that fail while the display is busy | about a day | robustness, not speed |

Suggested order: the CPU path's divide and conquer (the calls most people
make, at sizes the GPU does not take); then the IEEE audit; the two-stage
SVD with vectors when there is time for its kernel. The watchdog one is
small and makes the suites and the sweeps robust on a Mac in use.

**Not code, but open:**

- Measurements from other Macs: every Mac but the M5 Pro runs an estimated
  policy (refitted from the M5 Pro's timings), which does not cover the
  band's width. Each needs someone with that Mac to run `python3
  tuning/run.py` on an idle machine.

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
  [two-stage-vectors.md](two-stage-vectors.md#prototype-2026-10-07).

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
