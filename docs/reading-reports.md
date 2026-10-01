# Reading a routing report

Every run of `tuning/run.py` writes a `report.md` for each decomposition, under
`docs/results/<device>/<id>/{qr,eigh,svd}/`, and `tuning/combine.py` writes
the same kind of report for a device's combined runs. This page explains
every section and every number in them. The definitions are those of the
code in `tuning/`, and the examples are from the Apple M5 Pro run
(`docs/results/apple-m5-pro-20gpu/20260930-27b6c2/`).

## The idea

At each measured point (a matrix shape and a batch size) every backend that
could run there is timed. A routing rule, the policy for this device, picks
one backend per point. The rule is scored by how much slower its pick is than
the fastest backend at each point, and its constants are chosen to make that
as small as possible across all points. Then the result is checked: is the
optimum sharp or flat, does a richer rule really do better or only on the
points it was fitted to, and was the machine steady enough to trust any of it.

## Regret and the numbers in every table

The **regret** of a rule at a point is

    regret = time of the backend the rule picks / time of the fastest backend measured there

so 1.00 is the best possible and 1.25 means the rule's pick is 25% slower than
the best one available. "The fastest backend measured" is the **oracle**, and
a rule that always picked it would score 1.00 everywhere; that is the
`oracle` row of the tables. Each backend's time at a point is the fastest of
its passes.

Every rule table has the same columns:

| column | meaning |
|---|---|
| **geomean regret** | the geometric mean of the regret over all points, exp(mean of ln regret). This is the score being minimised. Geometric, because regrets are ratios: a 2x loss counts the same on a 0.1 ms call as on a 100 ms one, and a single very bad point cannot dominate it the way it would an arithmetic mean |
| **worst** | the largest regret at any single point. A rule with a good geomean and a bad worst case is fine on average and terrible somewhere |
| **>10%** | how many points have a regret above 1.10 |
| **total time / oracle** | the time to run every point through the rule, divided by the time the oracle would take. Unlike the geomean, this is dominated by the slowest points, so the two together say whether the losses are on small or large problems |
| **est. picks** | points where the rule picks a backend that was not timed there, because the cost model judged it too slow to measure (see Calibration). Its regret is then the model's estimate: it counts in the geomean, >10% and total time, but not in **worst**, and the warnings list each such point |

For example, a rule that is perfect at one point and 21% slow at another has a
geomean regret of √(1.00 × 1.21) = 1.10 and a worst case of 1.21x.

## The flat region, and the value that is shipped

The candidate values for a threshold are the sizes that were measured:
between two measured sizes every threshold makes the same choices, so nothing
in between can score differently.

The report gives the **flat region**: every candidate scoring within a
tolerance of the best, 0.5% of the best geomean for the eigensolver and SVD
and 0.003 of regret for QR. It reports the range rather than the single best
value because the best value of a flat curve is noise: run the sweep again and
it moves within the region. A wide region means the constant barely matters on
this device; a region of one value means it is sharp. On the M5 Pro the
eigensolver's block crossover is sharp (only 96 is in the region) and its CPU
cap is flat (1024 to no cap).

Which value is shipped:

- **eigh and SVD**: the value already in the table, if it is inside the region
  and its worst case is within 10% of the best worst case there, since a gain
  smaller than the noise is not worth a constant that changes from run to run;
  otherwise the value in the region with the smallest worst case.
- **QR**: the middle of the band.

## Regret curves

Each curve varies one constant over its candidate values with every other
constant held at its chosen value, and plots the geomean regret (the chart)
and lists it with the worst case (the table below the chart).

- The **floor** of the curve is the flat region.
- The **slope** on either side is what a wrong value costs. Curves are often
  asymmetric: for QR on the M1, erring low cost little and erring high lost
  most on tall matrices.
- The **worst** row can jump while the geomean barely moves: one point going
  badly wrong. On the M5 Pro, an eigensolver block crossover of 128 scores
  almost the same geomean as 96 (1.0419 against 1.0202 for 96) but a 1.88x
  worst case against 1.52x.

The QR report draws one curve per kind of shape (square, tall, wide,
near-square) as well as the pooled one, because a threshold can look fine
pooled and fail on one aspect ratio: on the M1 a square-heavy grid suggested
512, which lost up to 1.95x on tall input.

## Two stages: which GPU kernel, then GPU or CPU

The eigensolver and SVD reports fit their rule in two stages.

**Stage 1** chooses between the GPU kernels, scored against the best *GPU*
backend at each point, as if there were no CPU. Fitting this jointly with the
CPU boundary would hide it: wherever the CPU wins, every GPU choice scores the
same, so the kernel crossover could not be seen. Stage 1 is also exactly the
rule a forced-GPU call follows (`EIGH_DEVICE=gpu`).

**Stage 2** fits the GPU/CPU boundary given stage 1's choice, scored against
the best of all backends.

## Held-out checks of richer rules

Each harness also tries rules with more constants, which a single run would
always seem to favour: a batch-dependent kernel crossover (all three), a
per-size GPU/CPU boundary (eigh, SVD), a narrow-matrix special case (QR). To
tell a real improvement from fitting noise, the points are split in two, the
richer rule and the plain one are both fitted on one half, and both are scored
on the other:

- **eigh, SVD**: the split gives each half half of the batch sizes at every
  matrix size. The comparison is a **bootstrap** over the held-out points: they
  are resampled with replacement 2000 times and the mean log-regret of the two
  rules compared each time. "Better in 93% of resamples" is how often the
  richer rule won; the "median gain" is its typical improvement. It is
  **justified**, and goes into the row, if it wins in at least 95% of
  resamples and its held-out worst case is no more than 2% worse. A justified
  refinement is then refitted on all points.
- **QR**: the measured points, in order, alternate between the halves, and a refinement is adopted if
  its held-out geomean is lower by more than 0.002 and its held-out worst case
  no more than 0.01 higher.

It is normal for a rule to score better on the half it was fitted to than on
the other; a refinement that wins on its own half and loses on the other is
fitting noise, which is what the check is for. On the M5 Pro, the SVD's
batch-dependent block crossover was better in 100% of resamples (held-out
geomean 1.0449 against 1.1363) and is in the row; the eigensolver's reached
93% and is not.

## Decision surfaces

Grids of matrix size against batch size, one cell per measured point.

**Eigensolver and SVD**, as letters: the best backend at each point, then
what the rule picks; once for the GPU kernels alone (stage 1) and once for all
backends (stage 2). Where the two grids differ, the rule is losing. The legend
is in the report: `c` is the CPU, `.` a point not measured, and for the
eigensolver `s`, `t` and `B` its kernels (simd, threadgroup, block); for the
SVD `J` and `B` the whole-matrix and block kernels and `j`, `b` the same after
QR. The third grid is the **speedup**: the CPU's time divided by the best GPU
backend's, so above 1 the GPU wins; `gpu` marks a point where the CPU was not
timed because it would take seconds.

**QR**, as numbers with a glyph: the ratio of the grid-parallel backend's time
to the single-threadgroup backend's, for batch 1 and batch 16. Below 1 the
grid-parallel backend wins: `###` below 0.60, `##` below 0.85, `#` below 0.95,
`~` a tie (0.95 to 1.05), `.` up to 1.30, blank above. The arrow marks the
first row at or above the chosen threshold.

## QR only: which feature, and candidate rules

**Which feature decides** fits a threshold on each of three quantities, the
row count M, max(M, N) and min(M, N), and lists each at its best threshold. M
wins by a wide margin on every device measured, because the single-threadgroup
backend sweeps the rows serially. If another quantity ever won, the right rule
would have a different shape, not a different number.

**Candidate rules** scores several complete rules, including the original
heuristic the crossover replaced, overall and per kind of shape.

## Noise floor

Every point is measured in two or more passes, in a different random order
each time so that thermal drift during the run is not mistaken for an effect
of size. For each backend at each point the **pass-to-pass ratio** is its
slowest pass divided by its fastest. The report gives the median, 90th
percentile and maximum of that ratio overall and by how long the call takes.

A difference between two backends smaller than the p90 for its time bucket is
not a result. Calls under a millisecond are always the noisiest (the
submission of the work dominates), which is why the fitting uses the fastest
pass and the geometric mean rather than single comparisons. For QR, a median
ratio above 1.10 marks the run untrustworthy. In a combined report the ratio
is computed within each run, so it measures noise and not the spread between
machines.

## Conditions: machine state, probe drift, calibration

The lines at the top of the eigensolver and SVD reports describe the machine.

- **Machine state**: the load average and power source at the start and the
  end. A load average above half the CPU count, or Low Power Mode, marks the
  run untrustworthy: other work slows the CPU backend most, which would bias
  the routing toward the GPU.
- **Probe point**: one point timed before the sweep and again after it. The
  ratio, after over before, should be close to 1. If any backend moved by more
  than 25%, the machine changed state during the run (it throttled, or a job
  started) and timings from either side cannot be compared; the run is marked
  untrustworthy.
- **Cost model calibration**: the harness estimates each call's time from a
  simple model of an M1, so it can skip calls that would take more than 2.5
  seconds each. The probe point rescales the model to this device: "block
  x0.30" means the block kernel ran in 30% of the M1 model's estimate there.
  A faster device is therefore measured further out, where its crossovers have
  moved. The same model prices the estimated picks.

## The answer

In the eigensolver and SVD reports the **Answer** section gives the row for
`kTuned[]` and the same policy as environment variables, to try before
rebuilding. It says **"Indicative only; do not paste this row"** instead when
the run was a single pass (`--quick`), the machine was busy, or the probe
point drifted. The QR report gives its row under **The band**. The **Warnings** that follow
are explained in [`tuning-details.md`](tuning-details.md#5-reading-a-report).

## Combined reports

A report written by `tuning/combine.py` says so near the top, listing the
runs it used. Each backend's time at a point is then the median, over the runs, of
each run's fastest pass, so one unusual machine cannot move it; everything
else reads as above.
