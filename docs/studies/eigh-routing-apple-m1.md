# Eigensolver routing on an Apple M1

Which backend should `eigh_accelerated` use for a batch of `batch` symmetric
N × N matrices? There are four:

| backend | what it is |
|---|---|
| `cpu` | MLX's `eigh` on the CPU (Accelerate LAPACK) |
| `simd` | whole-matrix Jacobi, one simdgroup per matrix (`Eigh_Jacobi.metal`) |
| `tg` | whole-matrix Jacobi, one threadgroup per matrix (`Eigh_Jacobi.metal`) |
| `block` | block Jacobi, one matrix spread over the grid (`Eigh_BlockJacobi.metal`) |

`eigh.mm` encodes the decision as a per-device routing policy (`EighPolicy`).
This document is the measurement behind the M1 entry, in the same form as the QR study
([`qr-routing-apple-m1.md`](qr-routing-apple-m1.md)); the generated report with
every table is [`results/apple-m1-8gpu/legacy/eigh/report.md`](../results/apple-m1-8gpu/legacy/eigh/report.md).

## The answer

The M1 row of `kTuned[]` in `src/eigh.mm`:

```cpp
// device, GPU cores,   simd_max_n, block_min_n, block_min_n_batched, block_min_batch,   gpu_max_n, gpu_min_batch_times_n, gpu_min_batch
{"Apple M1", 8,   8, 96, 0, 0,   64, 1024, 1},
```

The last field, a minimum batch for the GPU, was added after this study for
the M5 Pro ([`routing-apple-m5-pro.md`](routing-apple-m5-pro.md)); refitting
this study's data with it gives 1, i.e. no minimum, and leaves the rest of the
row unchanged.

| policy field | value | flat region |
|---|---|---|
| `simd_max_n` (simd up to here, threadgroup above) | 8 | 0 .. 8 |
| `block_min_n` (block from here on) | 96 | 96 only |
| `block_min_n_batched`, `block_min_batch` | off | tested, not adopted |
| `gpu_max_n` (GPU only up to here) | 64 | 64 only |
| `gpu_min_batch_times_n` (GPU only if batch × N is at least this) | 1024 | 512 .. 1024 |

Against the best measured backend at each of 174 (N, batch) points this policy
scores 1.023 geometric-mean regret, 1.70x at its worst point, 13 points losing
more than 10%, and 1.038x the oracle's total time. One value moved: the block
crossover was 128 from a single-pass table and is 96 from this study.

## What is being decided, and why in two stages

The rule has two parts, applied in this order:

1. **GPU or CPU**: GPU iff `N <= gpu_max_n` and `batch * N >= gpu_min_batch_times_n`.
2. **Which GPU backend**: `block` if `N >= block_min_n`, else `simd` if
   `N <= simd_max_n`, else `tg`.

They are *fitted* in the opposite order, and separately. On this GPU the CPU
wins above N = 64, so under the full rule every block crossover from 96 to
"never" scores identically: the CPU routing hides it. Fitting the four
constants jointly, which is what the first version of the harness did, gave a
band of 96 .. never for `block_min_n`, i.e. no information. So stage 1 fits the
GPU split against the best *GPU* backend at each point, as if there were no
CPU; that is the rule a forced-GPU call follows, and it is what a GPU with
enough cores to beat LAPACK will actually use. Stage 2 then fits the CPU
routing given that split, against the best of all four.

## Method

- **Grid.** N in {2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512},
  batch in {1, 2, ..., 4096} by powers of two. A backend is skipped at a point
  where a pessimistic cost model puts one call above 2.5 s (the whole-matrix
  kernel at 512² × 16, say); a rule that picks a skipped backend is scored by
  that model, since a backend known to take seconds is not a candidate. 174
  points survive with at least one GPU backend; 526 backend timings per pass.
- **Timing.** `tuning/sweep_eigh.cpp`, one point per process so no cached
  pipeline or workspace from an earlier shape leaks into a later one; every
  backend at a point in the same fresh process; a correctness gate before
  anything is timed; median of adaptive repeats.
- **Two passes**, each in an independent random order (a size-ordered sweep
  would make thermal drift look like size dependence), combined by
  min-of-repeats. The pass-to-pass ratio gives the noise floor: median 4.7%,
  p90 19%, worst 2.6x on a sub-millisecond point.
- **Scoring.** Regret per point is the chosen backend's time over the best
  measured one. Every combination of the constants is scored; the reported
  band is every combination within 0.5% of the best geometric mean, and the
  chosen value is the compiled-in one if it lies in the band (a noise-level
  gain is not worth a constant that moves between runs), otherwise the band
  member with the best worst case.
- **Refinements** are fitted on one half of the points and scored on the
  other, split so both halves see every N. The verdict is a bootstrap over the
  held-out points: a refinement is adopted only if its geometric-mean regret
  is lower in at least 95% of resamples and its worst case is no worse. This
  replaces the first version's test, which compared a mean over 81 points to
  single-measurement noise and could not have accepted anything.

## Findings

### 1. The block crossover is 96, and it is sharp

Against the best GPU backend, varying `block_min_n` with the other constants fixed:

| block_min_n | 32 | 48 | 64 | 96 | 128 | 192 | 256 | 384 | never |
|---|---|---|---|---|---|---|---|---|---|
| geomean regret | 1.235 | 1.131 | 1.051 | **1.024** | 1.045 | 1.083 | 1.135 | 1.203 | 1.380 |
| worst | 7.61x | 5.37x | 2.88x | **1.49x** | 2.64x | 2.91x | 3.12x | 3.86x | 11.8x |

The best GPU backend per point (`s` simd, `t` threadgroup, `B` block):

```
  N \ batch     1     2     4     8    16    32    64   128   256   512  1024  2048  4096
         32     t     t     t     t     t     t     t     t     s     t     t     t     t
         48     t     t     t     t     t     t     t     t     t     t     t     t     t
         64     t     t     t     t     t     t     t     t     B     B     B     B     B
         96     t     t     t     t     B     B     B     B     B     B     B     B     B
        128     B     B     B     B     B     B     B     B     B     B     B     B     .
        192     B     B     B     B     B     B     B     B     B     B     .     .     .
```

Block wins at N = 128 for every batch, including batch 1, and at N = 96 from
batch 16. The single-pass table in
[`eigh-launch-parameters-apple-m1.md`](eigh-launch-parameters-apple-m1.md) had called 128 a tie at
small batch and 96 a threadgroup win up to batch 16; two passes with
min-of-repeats disagree, and 96's only losses are at N = 96 with batch ≤ 8
(threadgroup mode ahead by 1.37-1.49x), which is what the worst case above is.

### 2. A batch-dependent block crossover is suggestive, not justified

At N = 64 block wins from batch 256 on, by 1.16x, 1.25x, 1.27x and 1.30x at
256, 1024, 2048 and 4096: consistent, monotone, and above the noise floor. The
mechanism is plausible, too: at large batch the block backend's small
subproblem threadgroups co-reside on a core and hide each other's latency,
while the whole-matrix kernel's threadgroups do not. So the harness tries
"block from `block_min_n_batched` once batch ≥ `block_min_batch`" as a refinement.
Fitted on the training half it chose 64 and 64 (not 256, so it is trading
losses at N = 64, batch 64-128 against wins above) and on the held-out half it
was better in 89% of bootstrap resamples with a median gain of 1.2%. The bar is
95%. It stays out, and the constants stay one-dimensional; the harness will
re-test it on every run, and a denser batch grid around N = 48-96 and batch
128-512 with a third pass is what would settle it.

### 3. The CPU boundary is `N <= 64` with `batch * N >= 1024`, and a per-N table overfits

Given the split, varying each routing constant with the other fixed:

| gpu_min_batch_times_n | 0 | 64 | 128 | 256 | 512 | 1024 | 2048 | 4096 | never |
|---|---|---|---|---|---|---|---|---|---|
| geomean regret | 1.527 | 1.189 | 1.100 | 1.044 | **1.021** | **1.023** | 1.044 | 1.081 | 1.371 |
| worst | 12.4x | 6.89x | 4.45x | 2.81x | 1.68x | 1.70x | 2.30x | 2.84x | 8.06x |

| gpu_max_n | 16 | 24 | 32 | 48 | 64 | 96 | 128 | 256 | never |
|---|---|---|---|---|---|---|---|---|---|
| geomean regret | 1.139 | 1.090 | 1.056 | 1.031 | **1.023** | 1.029 | 1.051 | 1.109 | 1.183 |
| worst | 3.72x | 2.47x | 1.90x | 1.70x | **1.70x** | 1.79x | 2.58x | 2.77x | 3.62x |

512 and 1024 are indistinguishable on the product; 1024 is kept because it was
compiled in. The rule's worst points all sit on that boundary: N = 24 at batch
32 (product 768) is 1.7x faster on the GPU, N = 48 at batch 16 (768) 1.33x,
N = 12 at batch 64 (768) 1.32x. A per-N lookup table of the smallest winning
batch fixes exactly those on the training half (1.009 against 1.016) and then
loses on the held-out half (1.073 against 1.031, worst 4.5x, better in 2% of
resamples). The product rule is the one that generalises.

Speedup of the best GPU backend over the CPU, so the shape of the boundary is
visible; the rule takes the GPU where N ≤ 64 and batch × N ≥ 1024:

```
  N \ batch     1     2     4     8    16    32    64   128   256   512  1024  2048  4096
          4  0.11  0.13  0.12  0.14  0.16  0.27  0.47  0.73  1.28  2.17  3.01  3.69  4.47
          8  0.08  0.10  0.12  0.21  0.29  0.55  1.03  1.73  2.84  3.56  4.45  5.91  8.06
         16  0.09  0.13  0.20  0.36  0.63  1.07  1.55  2.00  2.92  4.22  5.11  5.51  5.70
         32  0.19  0.24  0.44  0.76  1.27  1.70  1.79  1.59  2.00  2.08  2.28  2.33  2.47
         64  0.15  0.26  0.54  0.85  0.98  1.16  1.26  1.18  1.31  1.40  1.47  1.52  1.53
         96  0.11  0.17  0.32  0.54  0.56  0.67  0.86  0.91  0.98  1.02  1.06  1.08  1.06
        128  0.12  0.15  0.25  0.39  0.51  0.66  0.67  0.72  0.74  0.78  0.79  0.82     .
        256  0.13  0.21  0.36  0.44  0.47  0.48  0.48  0.51  0.51     .     .     .     .
        512  0.19  0.28  0.28  0.31  0.31  0.33     .     .     .     .     .     .     .
```

### 4. On this GPU the block backend is unreachable from the public API

With `gpu_max_n = 64` and `block_min_n = 96` no public call reaches the block
backend on an M1: at N = 96 the GPU only ties the CPU at large batch, and
above that it loses. The block crossover is still tuned, and worth tuning,
because it governs `EIGH_DEVICE=gpu`, the `detail` entry points, and any GPU
with enough cores for the block backend's grid to beat Accelerate. That is
also why stage 1 must be fitted without the CPU.

## What the routing buys, and what changed

Every rule scored against the best measured backend at each of the 174 points:

| rule | geomean regret | worst | points >10% | total time / oracle |
|---|---|---|---|---|
| oracle (best backend per point) | 1.000 | 1.00x | 0 | 1.000 |
| always CPU | 1.371 | 8.06x | 63 | 1.140 |
| always GPU, block from 128 | 2.092 | 12.44x | 114 | 1.892 |
| always GPU, block from 96 | 2.048 | 12.44x | 108 | 1.566 |
| routing before the study (block from 128) | 1.023 | 1.70x | 13 | 1.038 |
| routing after the study (block from 96) | 1.023 | 1.70x | 13 | 1.038 |

The public routing is unchanged on an M1: the only constant that moved is the
block crossover, and every N it affects is routed to the CPU here. The change
acts on forced-GPU calls, where it is scored against the best GPU backend:

| GPU split | geomean regret | worst | points >10% | total time / oracle |
|---|---|---|---|---|
| block from 128 | 1.045 | 2.64x | 18 | 1.227 |
| block from 96 | 1.023 | 1.49x | 14 | 1.016 |

The two differ only at N = 96, where the old split ran the whole-matrix
kernel and the new one runs block: 1.4-1.5x slower for batch ≤ 8, and 1.1x
faster at batch 16 rising to 2.1x at batch 2048 (at 4096 the whole-matrix
kernel was not measured, its estimate being over the 2.5 s cap). The 1.227x
total-time figure for the old split is that unmeasured point.

## Reproducing on other hardware

```sh
cmake --build build --target sweep_eigh
python3 tuning/tune_eigh.py build/sweep_eigh        # ~17 minutes; --quick for ~5
```

Run it on an idle machine, on mains power, with Low Power Mode off. This
writes `eigh-tune-results/{raw.csv,results.json,report.md,policy.json}` and
prints the row for `kTuned[]`. It records the load average and power state at
both ends of the sweep and times a probe point before and after; a busy
machine, a probe that drifted by more than 25%, or a single pass marks the
report "indicative only". The M1 run above predates that recording: its two
passes agree to 3% at the probe point and its CPU timings are the fastest
seen on this machine, which is what a quiet machine looks like, but its state
was not logged, so it is worth repeating once with the current harness. The harness reads the device and the policy
in effect from the binary, calibrates its skip model to the device's speed, and
takes its candidate values from the grid it measured, so nothing above is
assumed on the new machine. The report says whether the policy in effect is
inside the flat region, re-tests the batch-dependent block crossover and the
per-N table, lists every point where the rule loses more than 25%, and warns
if the GPU is still ahead at the largest N measured, in which case
`--max-n 1024` extends the grid and the search. On a GPU with many more cores
expect `gpu_max_n` to rise first, and the block crossover to enter the public
routing once it does.

Until a device has a row, `eigh_policy_source()` reports
`default:untuned-device (<name>)` and the M1 values apply. They err toward the
CPU, which costs a missed GPU win rather than a slow call.

To re-render from the committed M1 data without measuring:

```sh
python3 tuning/tune_eigh.py --reanalyse docs/results/apple-m1-8gpu/legacy/eigh/raw.csv
```
