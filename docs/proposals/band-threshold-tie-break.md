# The band thresholds' tie-break

Status: **done in 2.15.0** (2026-10-07), both options. The eigh grid gains N
= 2560 and 3584, the SVD's k = 1280 and 1792; and in stages 3b and 4b a
threshold counts as near-optimal only if, on the points where it chooses
differently from the best, it is within 3% of the best there
(`near_on_disagreement` in `tuning/tune_eigh.py`). Re-analysed, run
`20261004-06bc11` now chooses 3072 for eigvalsh, as it should have. See
[the study](../studies/proposals-2-15-apple-m5-pro.md), section 6, and
[reading-reports.md](../reading-reports.md).

What follows is the proposal as written on 2026-10-04.

## What

On the M5 Pro the routing sweep sends eigenvalues alone to the `band` backend
from N = 4096, although at the one point measured between 3072 and 4095,
N = 3072, `band` is 7% faster than `tridiag`. The fit is working as designed;
the design picks the higher threshold here because there is too little data
in the band region. Give the fit more points there, so that the conservative
tie-break is not deciding on one.

## Why: the measurements

Stage 4b of run `20261004-06bc11` (eigh), scored on the 28 points where
`band_vals` was timed (N ≥ 512), geometric-mean regret by threshold:

| threshold | 1536 | 2048 | 3072 | 4096 | never |
|---|---|---|---|---|---|
| geomean | 1.0344 | 1.0261 | 1.0187 | 1.0212 | 1.0308 |

3072 is the best, 4096 is within the fit's 0.5% tolerance, and the
tie-break (lowest worst case, then the highest threshold, as the `tridiag`
and `bidiag` thresholds' stages also do) takes 4096. The difference is one
point of 28: (3072, batch 1), `tridiag` 96.2 ms, `band` 89.9 ms. The SVD's
stage 3b had more separation and chose 1536, the best.

## Plan

Options, cheapest first:

1. **More points where it matters.** Add N = 2560 and 3584 to the eigh
   grid's large sizes (`N_EXTRA`, which `--max-n` adds; batches 1, 2 and 4
   up to `HUGE_N`, 1 above), so that the band region has several
   points. Costs a minute or two of sweep. The tie-break then has data rather
   than a single point.
2. **Score each threshold on the points it changes.** A threshold of 3072
   against 4096 only differs at 3072 ≤ N < 4096; scoring both over all 28
   points dilutes the difference by the other 27. Comparing candidates on the
   points where they disagree (with the same tolerance) would choose 3072.
   This changes the method, so apply it to stages 3b and 4b only and say so
   in `docs/reading-reports.md`.
3. Leave it: the cost is 7% for N between 3072 and 4095, one or two matrices,
   eigenvalues alone.

Option 1 is the least intrusive; fold it into the next eigh re-measure.

## Effort

About two hours for option 1 (the grid in `tuning/tune_eigh.py`, check the
cost model's estimate for the new points, re-run or reuse the next eigh
re-measure, 28 minutes unattended); half a day for option 2, with its tests
and docs.

## Where to start

`tuning/tune_eigh.py`: the grid (`N_EXTRA`, `LARGE_BATCHES`, `HUGE_N`,
`BAND_MIN_GRID_N`) and stage 4b (`band_choice`, `near_b`, the
`min(..., key=...)` tie-break); the SVD's counterparts in `tuning/tune_svd.py`
(stage 3b) if option 2 is taken.
