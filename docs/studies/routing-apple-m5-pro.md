# Routing on an Apple M5 Pro

The measurements behind the Apple M5 Pro rows of `kTuned[]` in `src/qr.mm`,
`src/eigh.mm` and `src/svd.mm`: the first device measured for all three
solvers, and the first SVD row on any device. The machine has a 20-core GPU,
18 CPU cores (6 super, 12 performance) and 48 GB. The method is that of the M1
studies ([QR](qr-routing-apple-m1.md), [eigh](eigh-routing-apple-m1.md)); the
generated reports with every table are in
[`results/apple-m5-pro-20gpu/20260930-27b6c2/qr/`](../results/apple-m5-pro-20gpu/20260930-27b6c2/qr/report.md),
[`results/apple-m5-pro-20gpu/20260930-27b6c2/eigh/`](../results/apple-m5-pro-20gpu/20260930-27b6c2/eigh/report.md) and
[`results/apple-m5-pro-20gpu/20260930-27b6c2/svd/`](../results/apple-m5-pro-20gpu/20260930-27b6c2/svd/report.md).

## The answer

```cpp
// src/qr.mm:    device, GPU cores, m_crossover (batch < 16), m_crossover (batch >= 16), batch_threshold
{"Apple M5 Pro", 20, 512, 512, 16},

// src/eigh.mm:  device, GPU cores,   simd_max_n, block_min_n, block_min_n_batched, block_min_batch,
//                                    gpu_max_n, gpu_min_batch_times_n, gpu_min_batch
{"Apple M5 Pro", 20,   0, 96, 0, 0,   1024, 512, 16},

// src/svd.mm:   device, GPU cores,   qr_min_rows, qr_min_k,   block_min_k, block_min_k_batched, block_min_batch,
//                                    gpu_max_k, gpu_min_batch_times_k, gpu_min_batch
{"Apple M5 Pro", 20,   512, 32,   192, 64, 64,   1024, 512, 4},
```

| solver | rule | geomean regret | worst | points > 10% off | total time / oracle |
|---|---|---|---|---|---|
| QR | GPU backend by rows | 1.0115 | 1.22x | 5 of 143 | — |
| eigh | full rule, against the best of four backends | 1.0243 | 1.93x | 15 of 197 | 1.017 |
| SVD | full rule, against the best of five backends | 1.0418 | 2.03x | 31 of 263 | 1.130 |

What the rows buy over the untuned default, which on this device is the M1's
eigensolver policy and the SVD's loaded-machine defaults:

| solver | policy | geomean regret | worst measured | points > 10% off | total time / oracle | points routed to the GPU |
|---|---|---|---|---|---|---|
| eigh | untuned default | 1.2503 | 3.67x | 64 of 197 | 2.42 | 60 of 197 |
| eigh | M5 Pro row | 1.0243 | 1.93x | 15 of 197 | 1.02 | 116 of 197 |
| SVD | untuned default | 1.6000 | 9.10x | 104 of 263 | 8.40 | 103 of 263 |
| SVD | M5 Pro row | 1.0418 | 2.03x | 31 of 263 | 1.13 | 181 of 263 |
| QR | untuned default (384) | 1.0312 | 1.30x | — | — | all |
| QR | M5 Pro row (512) | 1.0115 | 1.22x | 5 of 143 | — | all |

The untuned defaults send most batched work to the CPU here: they cap the GPU
at N = 64 (eigh) and k = 64 (SVD), where on this device the GPU is ahead up to
1024.

## Conditions

- Idle machine on mains power: load average 1.6 to 1.8 on 18 CPUs at the start
  and end of each sweep. Low Power Mode was off when checked by hand after the
  sweeps; the harness's own check did not recognise this macOS's name for the
  setting (`powermode`) and recorded nothing, which is fixed since.
- Two passes per sweep in randomised order, combined by min-of-repeats. A probe
  point timed before and after each sweep moved by at most 7% (eigh x1.00 to
  x1.01, SVD x0.93 to x1.00), so no run was marked untrustworthy.
- Noise floor, pass to pass: median 1.015 / p90 1.175 (QR), 1.010 / 1.247
  (eigh), 1.010 / 1.207 (SVD), concentrated in calls under a millisecond.
- Grids: QR the standard 143 shapes; eigh N from 2 to 1024 (`--max-n 1024`)
  at batches 1 to 4096; SVD square k from 4 to 1024 (`--max-k 1024`) and tall
  shapes of aspect 2 to 32, at batches 1 to 4096. QR took 2 minutes, eigh 12,
  SVD 27.
- The library was built from the metallibs committed under
  `shaders/prebuilt/`, compiled on the M1; a metallib does not depend on the
  GPU that compiled it.

## Findings

### 1. QR: the crossover moved up, from 384 to 512

The optimum is flat from 480 to 512 rows, 1.0115 geometric-mean regret with a
1.22x worst case; the M1's 384 costs 1.95% here (1.0312, worst 1.30x). M
remains the right feature by a wide margin (max(M, N): 1.0411; K: 1.1160 with
a 5.81x worst case), and both refinements failed again on the held-out half:
a batch-dependent split scored 1.0142 and a narrow-N special case 1.0162,
against 1.0125 for the plain threshold.

A GPU with 2.5x the cores might be expected to favour the grid-parallel
backend and pull the crossover *down*. It went up, because the faster core
helps the single-threadgroup backend, which is bound by one core's serial
sweep, more than the extra cores help the grid-parallel one, which is bound
by launch latency. This is the case the comment above `kTuned[]` in
`src/qr.mm` anticipated, and the reason the crossover is a measured table
rather than a formula.

### 2. Eigh: the same kernel split, a much larger GPU region

Against the best GPU backend alone:

- **Block from N = 96**, the same as on the M1, and again the only value
  within 0.5% of the best (1.0202, worst 1.52x); 64 scores 1.0418 (worst
  2.94x) and 128 scores 1.0419 (worst 1.88x).
- **Simd mode never wins.** The flat region for `simd_max_n` is 0 to 2; the
  M1's 8 costs 1.0571 with a 2.98x worst case. The threadgroup mode is faster
  at every N on this GPU.
- **The batch-dependent block crossover** (block from N = 64 at batch 256 and
  up) was better on the held-out half in 93% of bootstrap resamples, against a
  95% bar; on the M1 it was 89%. Not adopted, on either.

Against the CPU, the GPU is ahead for batches up to the largest N measured,
so `gpu_max_n` is flat from 1024 to no cap, and a lone matrix is the CPU's at
every size (section 4). The product rule alone cannot express "batches of
large matrices yes, one large matrix no" without also refusing small batches
of small ones, so the policy gained a third constant, a minimum batch:

| `gpu_min_batch` | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| geomean regret | 1.0948 | 1.0768 | 1.0561 | 1.0365 | **1.0243** | 1.0380 |
| worst | 3.34x | 3.34x | 3.34x | 2.48x | **1.93x** | 1.93x |

16 is the only value in the flat region and the curve turns on both sides of
it. Given it, `gpu_min_batch_times_n` is flat from 512 to 1024. A per-N CPU
boundary was rejected on held-out data (better in 18% of resamples).

### 3. SVD: the first measured row

Against the best GPU backend alone:

- **Preconditioning with QR** pays from 512 rows and a short side of 32
  (flat region 256 to 512 rows, k from 32 to 64). The loaded-M1 default was
  k = 64. The cost of the lower bound is visible in the warnings:
  1024×32 and 512×32 at batches 16 to 64 run 1.7 to 2.0x slower through QR
  than through the whole-matrix kernel directly. Raising `qr_min_k` to 64
  removes those but scores worse overall (1.0433 against 1.0416, worst case
  3.80x against 1.98x).
- **Block kernel from k = 192 for a single matrix, from k = 64 in batches of
  64 or more.** This is the one refinement that cleared the bar on any
  device: on the held-out half it was better in 100% of resamples, with a
  median gain of 8.7% (held-out geomean 1.0449, worst 1.61x, against 1.1363
  and 3.80x for one crossover alone). A batch shares the block kernel's
  per-round dispatches, so the kernel pays from a smaller k when there are
  many matrices.

Against the CPU, as for the eigensolver: no cap up to 1024, a product floor
of 512, and a minimum batch, here 4:

| `gpu_min_batch` | 1 | 4 | 16 |
|---|---|---|---|
| geomean regret | 1.0484 | **1.0418** | 1.0649 |
| worst | 2.03x | **2.03x** | 2.29x |

The SVD's minimum is lower than the eigensolver's, most likely because its
CPU path is slower: a thin SVD on the CPU costs about twice an `eigh` of the
same size (16.6 against 9.1 ms at 512, 79 against 40 ms at 1024), so the GPU
catches up with fewer matrices. A per-k CPU
boundary was rejected (better in 0% of resamples).

One point is priced by the cost model rather than measured: 256×128 at batch
4096, where the rule picks the block kernel directly and only the
QR-then-block path was timed (estimated 3.0x).

### 4. A single matrix stays on the CPU at every size

Single matrices, median ms, block kernel against the CPU (MLX's `eigh`;
a thin SVD through MLX with a QR first when tall). The 512 to 1024 rows are
from the sweeps; the larger sizes were timed separately, the same way.

| N | eigh block | eigh CPU | ratio | SVD block | SVD CPU | ratio |
|---|---|---|---|---|---|---|
| 512 | 30.2 | 9.1 | 0.30x | 32.3 | 16.6 | 0.51x |
| 1024 | 100 | 40.4 | 0.40x | 114 | 79.1 | 0.69x |
| 1536 | 246 | 99.7 | 0.41x | 275 | 182 | 0.66x |
| 2048 | 565 | 230 | 0.41x | 680 | 457 | 0.67x |
| 3072 | 2150 | 690 | 0.32x | 2164 | 1355 | 0.63x |
| 4096 | 4932 | 2375 | 0.48x | 4742 | 3743 | 0.79x |

The ratio does not improve with N. Both sides are O(N³) and Accelerate on 18
CPU cores has the better constant. The likely reason is latency: the block
kernels issue a few dispatches per round, over about N/16 rounds per sweep
and 10 to 16 sweeps, and a lone matrix gives each round too little work to
hide them. The block kernels are still 16x (eigh, at 512) and 25x (SVD, at
1024) faster than the whole-matrix kernels on one matrix, which is what makes
batches of large matrices the GPU's.

## What changed relative to the M1

| constant | M1 | M5 Pro | |
|---|---|---|---|
| QR row crossover | 384 | 512 | faster core favours the single-threadgroup backend |
| eigh `simd_max_n` | 8 | 0 | simd mode never wins |
| eigh `block_min_n` | 96 | 96 | unchanged, sharp on both |
| eigh `gpu_max_n` | 64 | 1024 (lower bound) | the GPU region grows with the GPU |
| eigh `gpu_min_batch_times_n` | 1024 | 512 | |
| eigh `gpu_min_batch` | 1 | 16 | new; a lone matrix is the CPU's on both |
| SVD | no row | first row | the M1 needs a run on an idle machine |

## Update, 2026-10-02: QR's GPU-or-CPU boundary

The QR row above predates QR's CPU path, so it sent every QR call to the GPU,
up to ~100x slower than LAPACK on a lone 16×16. A QR-only run
([`20261002-9d19ba`](../results/apple-m5-pro-20gpu/20261002-9d19ba/qr/report.md),
`python3 tuning/run.py --only qr`) timed the CPU on 173 shapes, including lone
matrices up to 3072×3072, and gives the boundary

```cpp
{"Apple M5 Pro", 20, 512, 512, 16,   kQrNoLimit, 512, 1},   // GPU iff batch * k >= 512
```

1.08x geomean regret over the measured shapes, against 1.67x for always the
GPU. The kernel crossover (512) is unchanged. Its misroutes are near the
boundary: one 512×512 goes to the GPU at 1.7x the CPU's time, and 64 of
16×16 at 1.9x.

## Update, 2026-10-02: eigenvalues alone, and eigh past N = 1024

Since 2.3.0 `eigvalsh` runs on the CPU by LAPACK's two-stage reduction, up to
3x faster than the `ssyevd` the eigh row was fitted against, so routing
`eigvalsh` by eigh's boundary sent it to the GPU where the CPU had become
faster. An eigh-only run
([`20261002-153352`](../results/apple-m5-pro-20gpu/20261002-153352/eigh/report.md),
`python3 tuning/run.py --only eigh`) times every backend both ways (with
eigenvectors and, as `<backend>_vals`, without) and gives `eigvalsh` its own
boundary:

```cpp
// ..., gpu_max_n, gpu_min_batch_times_n, gpu_min_batch,   values_gpu_max_n, values_gpu_min_batch_times_n, values_gpu_min_batch
{"Apple M5 Pro", 20,   0, 96, 0, 0,   1024, 512, 16,   256, 2048, 32},
```

| eigenvalues alone, 194 points | geomean regret | worst |
|---|---|---|
| eigh's boundary (as before) | 1.0610 | 3.90x |
| its own, fitted | 1.0075 | 1.28x |
| its own, fitted on half, scored on the other half | 1.0068 | 1.26x |

The run also measures N = 1536 and 2048 for lone matrices and batches of 2 and
4, where the CPU is 2.4-2.6x faster than block Jacobi, so `gpu_max_n = 1024` is
now a measured cap rather than the edge of the grid. eigh's own row is
unchanged: combined with the earlier runs, its fitted values stay within the
near-optimal region.

## Update, 2026-10-03: the tridiag backend

The eigh sweep now times the `tridiag` backend (GPU tridiagonalization and
back-transformation, LAPACK's tridiagonal solver) and reaches N = 4096 for lone
matrices
([`20261003-d26059`](../results/apple-m5-pro-20gpu/20261003-d26059/eigh/report.md)).
Stage 4 of the tuner fits where it replaces the CPU:

```cpp
// ..., values_gpu_max_n, values_gpu_min_batch_times_n, values_gpu_min_batch,   tridiag_min_n, values_tridiag_min_n
{"Apple M5 Pro", 20,   0, 96, 0, 0,   1024, 512, 16,   256, 2048, 32,   1024, 0},
```

| tridiag over the CPU, one matrix | 512 | 768 | 1024 | 1536 | 2048 | 3072 |
|---|---|---|---|---|---|---|
| with eigenvectors | 0.69x | 0.81x | 1.13x | 1.47x | 1.93x | 2.73x |
| eigenvalues alone | 0.40x | 0.54x | 0.68x | 0.85x | 0.93x | 1.13x |

With eigenvectors it wins from N = 1024 at every batch measured (the GPU
Jacobi backends take larger batches below 1024); the threshold is the same
fitted on half the shapes, and scores 1.069x geomean regret on the other half
against 1.114x without the backend. For eigenvalues alone the CPU's two-stage
reduction stays ahead up to 2048, so the backend is off (0).

## Update, 2026-10-03: the bidiag backend, and the SVD remeasured

The SVD's measurements were stale (kernel version 1: from before QR, which
the QR-preconditioned backends call, routed small problems to the CPU). A new
SVD-only run
([`20261003-064803`](../results/apple-m5-pro-20gpu/20261003-064803/svd/report.md),
`python3 tuning/run.py --only svd`) remeasures the four Jacobi backends and
the CPU, and times the new `bidiag` backend (GPU bidiagonalization and
back-transformation, LAPACK's bidiagonal solver), with vectors and for
singular values alone, up to 4096×4096 for lone matrices. Stage 3 of the
tuner fits where it replaces the CPU:

```cpp
// ..., gpu_max_k, gpu_min_batch_times_k, gpu_min_batch,   bidiag_min_k, values_bidiag_min_k
{"Apple M5 Pro", 20,   512, 32,   192, 64, 64,   1024, 256, 4,   2048, 2048},
```

The GPU split is unchanged; the CPU boundary moves from batch × k >= 512 to
256 (1.0364 geomean regret against 1.0389 for the old row, worst 1.83x
against 2.43x).

| bidiag over the CPU, one matrix | 512 | 1024 | 1536 | 2048 | 3072 | 4096 |
|---|---|---|---|---|---|---|
| with vectors | 0.65x | 1.02x | 1.12x | 1.43x | 1.83x | 1.95x |
| singular values alone | 0.41x | 0.75x | 0.89x | 1.16x | 1.65x | 1.88x |

With vectors it breaks even at 1024 and wins from 1536, by 5-12% there
(batches of 1 to 4); the fit places the threshold at 2048: 1024, 1536 and
2048 all score within the 0.5% tolerance with the same worst case, and the
tie goes to the larger threshold, which keeps the backend off where it gains
only a few percent. Fitted
on half the shapes it gives the same 2048, scoring 1.0099 geomean regret on
the other half against 1.0229 without the backend. For singular values alone
it wins from 2048 too, where the eigensolver's CPU two-stage reduction keeps
`eigvalsh` on the CPU: the SVD's CPU path has no two-stage driver.

## Reproducing

```sh
python3 tuning/run.py
```

which writes a new submission beside this one; with several, the device's
rows come from all of them (`python3 tuning/combine.py
docs/results/apple-m5-pro-20gpu`). See [`../tuning.md`](../tuning.md).
