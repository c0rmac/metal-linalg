# The band width as part of the per-device policy

Status: proposal, not started (2026-10-04). Only matters once other Macs are
measured.

## What

The `band` backends reduce to a band 16 wide, a constant in `band_width()`
(`src/band_reduce.mm`), overridable with `EIGH_BAND_WIDTH` and
`SVD_BAND_WIDTH` (8, 16 or 32). Make the width a policy field per device,
measured by the tuning harness like the thresholds.

## Why: the measurements

On the M5 Pro, one matrix, the whole call (`bench band|eigband`, best of 4-5):

| | b = 8 | b = 16 | b = 32 |
|---|---|---|---|
| svdvals, 2048 | 77.8 ms | 70.0 ms | 108.3 ms |
| svdvals, 4096 | 329.3 ms | 235.0 ms | 275.1 ms |
| eigvalsh, 2048 | 61.8 ms | 46.3 ms | 61.6 ms |
| eigvalsh, 4096 | 262.2 ms | 164.9 ms | 165.7 ms |

The width trades the GPU stage's cost (a wider band: half the panels, each
twice as wide; fewer, larger products) against the CPU chase's (at 4096 on 16
threads, 53, 35 and 39 ms for b = 8, 16, 32 to bidiagonal). Both sides move
with the chip: a GPU with fewer cores, or a CPU with fewer, would shift the
best width, and nothing so far says by how much.

## Plan

1. Policy fields `values_band_width` (eigensolver and SVD), 0 meaning 16, at
   the end of the tuned rows (before the calibration field), with the C API,
   Python and PyTorch bindings, and the environment variables kept as
   overrides.
2. The sweeps time `band_vals` at 8, 16 and 32 on the band points only (28
   for eigh, 18 for the SVD on the M5 Pro's grid, a few minutes in all), as
   three backends.
3. Stages 3b and 4b choose the width with the lowest geometric-mean regret on
   those points, then fit the threshold with it.
4. `tuning/kernels.py`: the new backends go into `REQUIRED`/`ADDED`, so that
   other devices' runs become incomplete rather than stale.

## Effort

About a day, plus an eigh and SVD re-measure (about 80 minutes unattended),
following the usual checklist for a new policy field (`core.h`, `eigh.mm`,
`svd.mm`, `c_api.h`/`.cpp`, bindings, `combine.py` FIELDS, the sweeps and
harnesses, docs).

## Expected gain

None on the M5 Pro (16 is already its best). Elsewhere, unknown until another
Mac is measured; the spread above says the wrong width costs up to 1.6x.
A cheaper alternative until then: have the harness time 8 and 32 at a couple
of points and only print a warning when 16 is not the best.

## Where to start

`band_width()` and `band_fit()` in `src/band_reduce.mm`; their callers
`svd_band` (`src/svd_bidiag.mm`) and `eigh_band` (`src/eigh_band.mm`);
stage 3b in `tuning/tune_svd.py` and 4b in `tuning/tune_eigh.py`.
