# Proposals

Work that was scoped but not done, written down so that it can be picked up
later. Each file says what the change is, the measurements behind it, a plan,
the effort and the expected gain, and where to start. Written 2026-10-04,
against the `two-stage-values` branch (2.13.0) on an Apple M5 Pro (20 GPU
cores, 18 CPU cores); the numbers are that machine's.

| proposal | affects | time at stake (one 4096 × 4096) | effort | expected gain |
|---|---|---|---|---|
| [Parallel divide and conquer](parallel-divide-and-conquer.md) | eigh and SVD with vectors (`tridiag`, `bidiag`) | `sstedc` 177 of 438 ms; `sbdsdc` 718 of 1535 ms, on one core | 3-5 days (CPU) | eigh ~1.45x, SVD ~1.5x (estimate) |
| [Two-stage reduction with vectors](two-stage-vectors.md) | SVD with vectors | the reduction, ~720 of 1535 ms | 1-2 weeks | SVD ~1.35x, ~1.65x once the D&C is parallel (estimate) |
| [A faster TSQR top kernel](tsqr-top-kernel.md) | eigvalsh and svdvals (`band`) | 24 and 49 ms, ~95 µs a panel | about 1 day | 1.1-1.15x at 4096, up to ~1.3x at 2048 |
| [Symmetric trailing update](symmetric-trailing-update.md) | eigvalsh (`band`) | the trailing update's products, 73 of 161 ms | 2-6 days | 1.2-1.3x at 4096, more at 8192 (estimate) |
| [Fused small products](fused-small-products.md) | eigvalsh and svdvals (`band`) | 8 and 21 ms (profiled, launch overhead included) | about half a day | 2-4% |
| [Band width per device](band-width-per-device.md) | `band` on other Macs | none on the M5 Pro | about 1 day | unknown until other Macs are measured |
| [Band threshold tie-break](band-threshold-tie-break.md) | eigvalsh from 3072 to 4095 | 7% at N = 3072 | about 2 hours | 7% in that range |

Suggested order: the divide and conquer first (the largest win, and for the
calls most people make, with vectors); then the TSQR top kernel (a day, and it
may move the `band` thresholds down); the rest as wanted. The tie-break is
cheap enough to fold into whichever of these next needs an eigh re-measure.

**Not code, but open:**

- Releasing 2.13.0 (GitHub release, PyPI wheels, Homebrew formula) once the
  branch is merged.
- PyPI's trusted publisher for `metal-linalg-torch`, if it is still missing:
  only the account owner can add it on pypi.org, then the Release workflow is
  rerun with `pypi_version`.
- Measurements from other Macs: the M1's eigh and QR runs are stale and it has
  no SVD row; every other chip runs the untuned default. Each needs someone
  with that Mac to run `python3 tuning/run.py` on an idle machine.

**Tried and rejected** (so that they are not retried without a new idea):

- Look-ahead in the band reduction, the next block's panel factored while the
  rest of the trailing matrix is updated: no overlap within one command
  buffer, and 14-54% slower with the panel on a second queue. See
  [the two-stage study](../studies/two-stage-apple-m5-pro.md), section 9.
- LAPACK's `sbdsvdx` (bisection by index range) in place of `sbdsqr`: 30x
  slower on one core, and wrong when the bidiagonal splits.

**How the profile numbers were taken.** The per-kernel times in these files
come from a temporary build of `src/band_reduce.mm` that committed every panel
kernel and every matrix product in a command buffer of its own and summed
`GPUEndTime - GPUStartTime` per kind of operation. Each operation then carries
about 10 µs of launch overhead it does not have in the real, single command
buffer a block: the profiled stage summed to 204 ms for svdvals at 4096,
against about 180 ms in the real run. The temporary build was not kept; it is
a few dozen lines (a `cut(cb, tag)` that commits and starts a new command
buffer after each operation, and a report at the end of each reduction).
