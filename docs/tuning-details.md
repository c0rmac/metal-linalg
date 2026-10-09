# Tuning: details

How the measurements behind the per-device routing work, for maintainers and
the curious. Contributors only need [`tuning.md`](tuning.md).

## 1. What `tuning/run.py` does

1. Checks the Mac: Apple Silicon, `cmake` and MLX installed, on mains power,
   load average under half the CPU count, Low Power Mode off. It stops if any
   check fails; `--anyway` measures regardless and the results are marked
   untrustworthy.
2. Builds the sweep binaries and correctness tests in `build-tuning/`.
3. Runs `test_qr`, `test_eigh` and `test_svd`, and stops if any fails: timings
   of a wrong answer mean nothing.
4. Creates `docs/results/<device>/<id>/` and runs the three harnesses into it,
   one after another:

   | step | harness | measures | time on an M5 Pro |
   |---|---|---|---|
   | QR | `tuning/tune_qr.py` | both GPU backends and the CPU on 185 shapes, square, tall, wide and near-square, batch 1 to 16384 | 4 min |
   | eigh | `tuning/tune_eigh.py --max-n 4096` | CPU, the whole-matrix kernel in both modes and block Jacobi up to N = 96, ql, tridiag, tridiag_batch, band, each again for eigenvalues alone; N from 2 to 4096, batch 1 to 4096 (small batches above 1024) | 15 min |
   | SVD | `tuning/tune_svd.py --max-k 4096` | CPU, both Jacobi kernels, each with and without QR, up to k = 128, square and tall shapes, batch 1 to 4096; golub_kahan up to its limit; bidiag_batch from k = 32; bidiag, band and the CPU, with and without vectors, from k = 128 up to 4096 (small batches above 1024) | 25 min |

   Each runs a first pass over every point, then a second over the points
   whose choice the first left open, both in random order, so that thermal
   drift is not mistaken for a size effect and a noise floor can be measured.
   A point needs no second pass when, with vectors and for the values alone,
   the fastest backend is at least 1.3x ahead of the next: any rule that picks
   a loser there pays at least that, however noisy its one timing (about half
   the points on an M5 Pro). The Jacobi kernels, which do several times the
   flops of the GPU's LAPACK-style backends timed beside them, are timed only
   where they can win (N up to 96, k up to 128; on the M5 Pro they won at no
   point above), and at a canary point beyond, 256 x 256 in batches of 1 and
   64: a report warns if one wins there. Re-analysed on the M5 Pro's earlier
   runs, the two cuts gave the full grid's routing, scored by the library's
   own router (since 2.17.0, when they took a sweep from 75 minutes to about
   40).
5. Writes `submission.json` and `summary.md`, and prints the rows and how to
   send them.

`--quick` runs coarser grids, one pass each, into `build-tuning/quick/`: a
smoke test of the pipeline, never a submission.


**Memory.** Every sweep skips a shape whose estimated peak memory, six copies
of its arrays in float32 (input, outputs, workspaces, the correctness check),
exceeds 35% of the Mac's RAM: about 2.8 GB on an 8 GB Mac, 17 GB on a 48 GB
one (`tuning/submissions.py`, `MEMORY_FRACTION`). A shape that swaps would
time the disk, or stop the run. A smaller Mac therefore measures a smaller
grid; the fitted thresholds are chosen among the shapes measured, as always.
`METAL_LINALG_TUNING_MEMORY_GB=<n>` sets the budget instead.

## 2. Submissions

```
docs/results/<device>/<id>/
    submission.json     the spec, versions, conditions and fitted rows
    summary.md          the same, readable
    qr/ eigh/ svd/      per decomposition: raw.csv, results.json, report.md, policy.json
    qr.log eigh.log svd.log
```

`<device>` is the chip and GPU core count, e.g. `apple-m5-pro-20gpu`: the
same pair `kTuned[]` is keyed on, so a binned part with fewer cores is its own
device. `<id>` is the UTC date and six random hex digits, e.g.
`20260930-27b6c2`, unique without any coordination between contributors.

`submission.json` records:

| field | contents |
|---|---|
| `device` | the chip and GPU core count, as Metal reports them; the key for `kTuned[]` |
| `machine` | the exact model: product name ("MacBook Pro (16-inch, M5 Pro)"), model identifier (`Mac17,8`) and built-in display resolution. One chip ships in machines that cool it differently, a 14-inch and a 16-inch MacBook Pro for instance, and only this tells them apart |
| `cpu`, `memory_gb` | CPU cores per performance level, memory |
| `macos`, `macos_build`, `mlx`, `metal_linalg`, `epoch` | the software it ran |
| `conditions` | at the start and after each decomposition: load average, CPU count, power source, charger and its wattage, battery charge, power mode, Low Power Mode, and macOS's thermal and performance warnings |
| `results` | per decomposition: trustworthy or not and why, the fitted row, and the probe point's drift over the sweep |
| `minutes` | how long each decomposition took |

It records nothing that identifies the person or the particular Mac: no
serial numbers, hardware UUIDs, hostnames or user names.

The M1's runs predate this layout and are kept as `apple-m1-8gpu/legacy/`
(QR and eigh only).

## 3. From runs to the library's tables

The rows the library uses are generated, never edited by hand:

```
docs/results/<device>/<id>/        every run submitted (any number per device)
        |  tuning/combine.py       one device's runs -> its rows
        v
docs/results/<device>/combined/    the combined report per decomposition, summary.md
        |  tuning/generate_tables.py   every device
        v
src/tuned/{qr,eigh,svd}.inc        included by kTuned[] in src/qr.mm, src/eigh.mm, src/svd.mm
```

**What a timing is.** Each sweep tool times one shape in a process of its
own, a call through the MLX API as a program makes it, the median of its
reps after two warm-ups, with MLX's buffer cache on as MLX has it by default
(since 2.16.0; off before, which made every call's outputs fresh pages for
the GPU to map, and timed the allocation as much as the decomposition).

**How runs combine** (`tuning/submissions.py`): for each decomposition,
every run whose measurements of it are trustworthy is used, and smoke tests,
interrupted runs, untrustworthy results and runs from an older measurement
epoch are skipped (the combined `summary.md` says which and why). Within a
run, each backend's time at a point is the fastest of its passes, since
interference only ever slows a run down; across runs, the median of those, so
one unusual machine cannot move the result. The analysis is then the same as
for a single run, on the combined times. The noise floor stays per run, so it
measures run-to-run noise rather than the spread between machines.

**The epoch** (`EPOCH` in `tuning/submissions.py`, recorded in every
`submission.json`). When a kernel or its launch parameters change enough that
earlier timings no longer describe the library, bump it: older runs then stop
counting, and a device is estimated from the measured ones (and its own
measurements stop being an anchor for the others) until it is measured again.

**Diagnosing disagreement.** The combined `summary.md` for each device ends
with a table of its runs: each run's machine, memory, macOS and conditions,
and for each decomposition whether the settings that run fitted on its own
match the combined ones, naming every setting that differs. The Action's
summary shows the same table on every results pull request. Runs from one
model that consistently disagree with another model of the same chip point to
a real difference between the machines rather than noise: a 14-inch MacBook
Pro that throttles where the 16-inch does not, for instance. The tables are
keyed on the chip and GPU core count only, so such machines share a row
today; if the difference is real, the key would have to include the model
identifier, which the library can read at run time (`sysctl hw.model`).

**The Action** (`.github/workflows/tuned-policies.yml`) runs on every pull
request and every push to `main` that touches `docs/results/` or `tuning/`,
on a Linux runner: the analysis is plain Python and needs no GPU.

- On a pull request it runs `tuning/validate_submissions.py` (the folder and
  file layout, the ID, that the device folder matches the spec, the CSV
  columns and values, no smoke tests, no oversized files), then
  `tuning/generate_tables.py`, and lists the resulting rows and the change to
  `src/tuned/` in the run's summary. A malformed submission fails the check.
- After a merge to `main` it does the same and commits `src/tuned/` and the
  combined reports if they changed, as `github-actions[bot]`. If `main` is
  protected against direct pushes, allow the Action to push or change this
  step to open a pull request.

**By hand**, the same steps are:

```sh
python3 tuning/validate_submissions.py
python3 tuning/generate_tables.py            # rewrites src/tuned/ and the combined reports
python3 tuning/generate_tables.py --check    # changes nothing; fails if they are out of date
python3 tuning/combine.py docs/results/apple-m5-pro-20gpu   # one device, to look at
```

After the tables change, `cmake --build build && ctest --test-dir build` and
`./build/sweep_qr --policy` (or `sweep_eigh`, `sweep_svd`) on a measured Mac
should report `tuned:<device name>`.

## 4. The rows

**QR** (`src/qr.mm`): device name, GPU cores, the row crossover for small
batches, the row crossover for large batches, and the batch that separates
them. The two crossovers are equal unless a batch-dependent split survived
held-out validation. Then the CPU boundary, `gpu_max_k`,
`gpu_min_batch_times_k`, `gpu_min_batch` and `gpu_min_k` (the GPU only from
this k, so that the smallest matrices stay on the CPU at any batch; 0 in a run
from before 2.10.0), the large-matrix clause,
`gpu_large_min_k` and `gpu_large_max_batch`: the GPU also from the first in
a batch of at most the second (0: any; `0, 0`: never, which a run from
before 2.9.0 gives), the size being `k` before 2.15.0 and `sqrt(M k)` since,
so that a tall matrix counts by its rows too. Since the CPU path spreads a batch over every core
it wins batches of small and mid-size matrices, while one large matrix is
still faster on the GPU, and one product rule cannot say both. Last,
`share_min_batch`: from this batch a GPU batch is shared with the CPU path
(0: never, which a run from before 2.12.0 gives), fitted on the `share`
timings before the GPU-or-CPU boundary, which is then fitted with it in
effect.

**Eigensolver** (`src/eigh.mm`): device name, GPU cores, then `simd_max_n`,
`block_min_n`, `block_min_n_batched`, `block_min_batch`, `gpu_max_n`,
`gpu_min_batch_times_n`, `gpu_min_batch`, then `values_gpu_max_n`,
`values_gpu_min_batch_times_n`, `values_gpu_min_batch`, then `tridiag_min_n` and
`values_tridiag_min_n`, `tridiag_max_batch` and `values_tridiag_max_batch`,
then `ql_min_n` and `ql_max_n`, as documented on
`EighPolicy` in `include/metal_linalg/core.h`. The first four choose the GPU
backend, fitted against the best GPU backend alone; the next three are the CPU
boundary: GPU iff N is at most `gpu_max_n`, batch × N at least
`gpu_min_batch_times_n` and the batch at least `gpu_min_batch`. The last
three are the same boundary for eigenvalues alone (`eigvalsh`), fitted on the
`_vals` timings; `0, 0, 0` (a run from before those were measured) means "as
for eigenvectors". The last two are where the `tridiag` backend takes over from
the CPU, with eigenvectors and without (0: never, which a run from before the
backend existed gives), for batches up to the two caps after them (0: any
batch; the backend solves a batch one matrix after another, the CPU path
spreads one over every core); stage 4 of `tune_eigh.py` fits each threshold
with its cap on the points where `tridiag` was timed (the backend pipelines a
batch over two slots since 2.11.0, so the caps grew). Then the window of N in
which the `ql` backend replaces the Jacobi backend the first four pick (`0, 0`:
never, which a run from before 2.9.0 gives); stage 1b fits it over the finished
split, against the best GPU backend, on the points where `ql` was timed (N up
to the device's limit, 87 with 32 KB of threadgroup memory). The very last is
`share_min_batch`: from this batch, a batch that goes to `ql` is shared with
the CPU path, the GPU and the CPU solving it at once (0: never, which a run
from before 2.11.0 gives); stage 1c fits it against the best GPU backend, the
shared one (`ql_share`, timed from batch 64) included. After it,
`gpu_big_batch_max_n` and `gpu_big_batch_min`: the GPU also for N above
`gpu_max_n` up to the first in a batch of at least the second (0, 0: never,
which a run from before 2.12.0 gives), fitted in stage 2 together with the
product rule: for each `gpu_max_n`, the rule fitted alone, the clause over it,
and the rule fitted again given the clause; the best combination is kept.
Last, `values_band_min_n`: for eigenvalues alone, from this N the `band`
backend (the two-stage reduction) instead of `tridiag` or the CPU, within
`values_tridiag_max_batch` (0: never, which a run from before 2.13.0 gives);
stage 4b fits it, after `tridiag`'s thresholds, on the points where
`band_vals` was timed (N >= 512). Then `values_band_width`, the band's width
(8, 16 or 32; 0: 16, which a run from before 2.15.0 gives): stage 4b chooses
it first, from `band8_vals`, `band_vals` and `band32_vals` at those points,
and fits the threshold on its times. Then (since 2.17.0; 0 in a run from
before) `band_min_n`, the same backend with eigenvectors from this N, which
stage 4c fits on the points where `band` was timed (N >= 512) together with
`tridiag_max_batch`, which `band` shares: since 2.17.0 `band` takes a batch
of two or more (384-1024) through `tridiag_batch`'s two stages, every matrix
at once, so the cap that suits `tridiag`, a matrix at a time, need not suit
it (scored over `band`'s points and `tridiag`'s; with `band` never, stage 4's
cap stands); and the
`tridiag_batch` windows, `tridiag_batch_min_n`, `tridiag_batch_max_n`,
`tridiag_batch_min_batch` and the three `values_` ones: N in the window from
that batch on, where the rules give the CPU, the `tridiag_batch` backend (a
batch reduced together). Stage 5 fits each over every window of measured N and
batch on the points where `tridiag_batch` was timed (N 48-1024, batches from
16), against the CPU and whatever else the CPU's side would pick; inside the
flat region the window in effect stays, otherwise the smallest worst case,
then the largest batch and the narrowest window.

**SVD** (`src/svd.mm`): device name, GPU cores, then `qr_min_rows`,
`qr_min_k`, `block_min_k`, `block_min_k_batched`, `block_min_batch`,
`gpu_max_k`, `gpu_min_batch_times_k`, `gpu_min_batch` and `gpu_max_l`, then
the same four for singular values alone (`values_gpu_max_k`,
`values_gpu_min_batch_times_k`, `values_gpu_min_batch`, `values_gpu_max_l`;
`values_gpu_min_batch = 0`, which a run from before 2.11.0 gives, means "as
with vectors"), then `bidiag_min_k`, `values_bidiag_min_k`, `bidiag_max_batch`
and `values_bidiag_max_batch`, then `gk_min_k` and `gk_max_k`, then
`share_min_batch`, as documented on `SvdPolicy` in
`include/metal_linalg/core.h`: whether to precondition with QR, which kernel,
then the CPU boundary in the eigensolver's form (with a cap on the long side,
`l = max(M, N)`), and again for `svdvals` (stage 2b of `tune_svd.py`, fitted on
`gk_vals` and `cpu_vals` where `gk` is timed), then where the `bidiag`
backend takes over from the CPU, with vectors and for singular values alone
(0: never, which a run from before the backend existed gives), and up to which
batch (0: any; as `tridiag`, it solves a batch one matrix after another);
stage 3 of `tune_svd.py` fits each threshold with its cap on the points where
`bidiag` was timed. Last, `gk_min_k` and `gk_max_k`: the window of k in which
the `golub_kahan` backend replaces the Jacobi backends on the GPU (0, 0: never,
which a run from before 2.10.0 gives). Stage 1b of `tune_svd.py` fits it over
the Jacobi split, against the best GPU backend, on the points where `gk` was
timed (k up to the device's limit, 83 with 32 KB of threadgroup memory), as
`tune_eigh.py` fits the `ql` window; the CPU boundary is then fitted with it
in place. And `share_min_batch`, as for the eigensolver: from this batch a
`golub_kahan` batch is shared with the CPU path (stage 1c, on `gk_share`),
and `gpu_big_batch_max_k` and `gpu_big_batch_min`, the large-batch clause, as
for the eigensolver. Last, `values_band_min_k`: for singular values alone,
from this k the `band` backend (the two-stage reduction) instead of `bidiag`
or the CPU, within `values_bidiag_max_batch` (0: never, which a run from
before 2.13.0 gives); stage 3b fits it, after `bidiag`'s threshold, on the
points where `band_vals` was timed (k >= 512, where `bidiag_vals` is). Then
`values_band_width`, chosen by stage 3b as stage 4b does for the
eigensolver. And `band_min_k` (since 2.15.0): with singular vectors, from
this k the `band` backend instead of `bidiag` or the CPU, within
`bidiag_max_batch` (0: never, which a run from before 2.15.0 gives); stage 3c
fits it on the points where `band` was timed with vectors (k >= 512),
together with that cap since 2.17.0, as stage 4c does for the eigensolver
(`band` takes a batch of two or more through `bidiag_batch`'s two stages).
Then (since 2.17.0; 0 in a run from before) the `bidiag_batch`
windows, `bidiag_batch_min_k`, `bidiag_batch_max_k`, `bidiag_batch_min_batch`,
`bidiag_batch_max_l` and the four `values_` ones: k in the window, l up to the
cap, from that batch on, where the rules give the CPU, the `bidiag_batch`
backend (a batch bidiagonalized together). Stage 4 fits each over every
window of measured k, batch and cap on the points where `bidiag_batch` was
timed (k from 32, l up to 1024, batches from 16), against the CPU and whatever
else the CPU's side would pick, as the eigensolver's stage 5 fits
`tridiag_batch`'s; a tall matrix, which the CPU reduces by a QR first, is what
the cap on l is for.

## 5. Reading a report

What every section and number in a report means (regret, the flat region,
the curves, the held-out checks, the decision surfaces, the noise floor) is
explained in [`reading-reports.md`](reading-reports.md). In short, whether a
run is usable:

| warning | meaning | what to do |
|---|---|---|
| the machine was not idle | load average above half the CPU count, or Low Power Mode on, at the start or end | stop the other jobs and rerun |
| the machine changed state during the sweep | a probe point timed before and after moved by more than 25%: thermal throttling or a job that started midway | rerun, on mains, once the machine has cooled |
| single pass | `--quick` was used, so there is no noise floor | rerun without `--quick` |
| (QR) median pass-to-pass ratio above 1.10 | the machine was not stable | rerun |

Any of these makes `run.py` mark that decomposition untrustworthy, and
`combine.py` leaves it out. The other warnings do not invalidate a run:

| warning | what to do |
|---|---|
| the GPU is still ahead of the CPU at the largest N (or k) measured | the cap reported is a lower bound. Harmless at 1024: a lone matrix stays on the CPU through the minimum batch, and batches above 1024 take seconds either way |
| this device has no entry in `kTuned[]` | expected on a new device |
| the fitted policy differs from the one in effect | update the device's row |
| chosen rule loses more than 25% at ... | informational: where a simple rule is furthest from the best backend |
| chosen rule picks a backend that was not timed at ... | see "est. picks" in [`reading-reports.md`](reading-reports.md#regret-and-the-numbers-in-every-table) |
| the best feature is no longer M (QR) | a structural change, not a moved threshold; worth an issue rather than a new row |

## 6. Running one step by hand

The harnesses run on their own too, which is useful when changing one of
them:

```sh
cmake --build build --target sweep_qr sweep_eigh sweep_svd
python3 tuning/tune_qr.py   build/sweep_qr                 # writes qr-tune-results/
python3 tuning/tune_eigh.py build/sweep_eigh --max-n 4096  # writes eigh-tune-results/
python3 tuning/tune_svd.py  build/sweep_svd  --max-k 4096  # writes svd-tune-results/
```

| option | harness | use |
|---|---|---|
| `--out DIR` | all | where to write |
| `--passes N` | all | more than two passes, for a noisy machine |
| `--full-passes` | eigh, SVD | repeat every point in every pass, not only those without a clear winner |
| `--full-grid` | eigh, SVD | time the Jacobi kernels at every size (what a canary warning asks for) |
| `--full` | QR | a denser grid, about three times longer |
| `--max-n`, `--max-k` | eigh, SVD | the largest size on the grid (default 512; larger sizes up to 4096 added up to this) |
| `--quick` | eigh, SVD | one pass on a coarse grid; a smoke test only |
| `--limit S` | all | per-point timeout in seconds |

To try a policy before rebuilding, copy the environment line from a report,
e.g. `EIGH_GPU_MAX_N=128 ./build/benchmark_eigh` or
`QR_M_CROSSOVER=320 ./build/benchmark_qr`.

**Re-analysing without the GPU**, for example after a harness change:

```sh
python3 tuning/tune_eigh.py --reanalyse docs/results/apple-m5-pro-20gpu/20260930-27b6c2/eigh/raw.csv --out /tmp/eigh
```

Several `raw.csv` files may be given: they combine as in section 3 (files in
one submission merge by min-of-passes, so a sweep can be topped up). Passing
the sweep binary as well makes the report describe the policy that binary
resolves; without it, the policy in effect when the sweep ran. `--from
results.json` re-renders a report from its JSON.

## 7. Launch parameters and benchmarks

The routing decides *which* backend runs. How each eigensolver and SVD backend
is launched (threads per matrix, inner sweeps, simdgroups per matrix) scales
with the detected core count and is not part of the per-device table, but it
can be inspected:

```sh
./build/benchmark_eigh --tune      # five tables; --tune 1..5 for one
./build/benchmark_svd  --tune      # simdgroups per matrix
./build/probe_occupancy            # threadgroup-memory limits for QR
```

The M1 tables are in
[`studies/eigh-launch-parameters-apple-m1.md`](studies/eigh-launch-parameters-apple-m1.md)
and [`studies/svd-design-notes.md`](studies/svd-design-notes.md).

## 8. Troubleshooting

| symptom | cause and fix |
|---|---|
| `Impacting Interactivity` in an error message | macOS stopped a GPU command buffer that ran for several seconds. The library splits large batches to avoid this; during a sweep the point is retried and then recorded as failed. Lower `EIGH_CHUNK_MS` or `SVD_CHUNK_MS` (default 750) if it recurs |
| `GPU Hang Error` in an error message | with the display busy, macOS stopped a threadgroup that ran for more than about a quarter of a second. Since 2.15.0 the whole-matrix Jacobi kernels split a long solve over dispatches (`EIGH_DISPATCH_MS`, `SVD_DISPATCH_MS`, default 40); if another kernel shows it, measure with the Mac idle |
| rows with `ok` = 0 in `raw.csv` | the backend failed its correctness gate or timed out at that point, and the point is excluded. A few are harmless; many at small sizes indicate a real fault |
| `sweep_eigh --policy failed` (or another sweep) | the binary is older than the harness; rebuild it |
| `missing Metal Toolchain` | only matters when changing a shader; the build otherwise uses `shaders/prebuilt/`. `xcodebuild -downloadComponent MetalToolchain` installs it |
| very different answers from two runs | the machine was not in the same state for both; compare the noise floor and machine-state lines of the two reports |

## 9. Reference

**Studies**, the reasoning behind each harness and the results in full:

- QR on an M1: [`studies/qr-routing-apple-m1.md`](studies/qr-routing-apple-m1.md)
- eigensolver routing on an M1: [`studies/eigh-routing-apple-m1.md`](studies/eigh-routing-apple-m1.md)
- all three on an M5 Pro: [`studies/routing-apple-m5-pro.md`](studies/routing-apple-m5-pro.md)
- SVD design and the measurements behind it: [`studies/svd-design-notes.md`](studies/svd-design-notes.md)

**Environment overrides.** All take effect without a rebuild and are reported
in the policy source.

| variable | effect |
|---|---|
| `QR_M_CROSSOVER` | QR: rows at which the grid-parallel backend takes over |
| `QR_GPU_MAX_K`, `QR_GPU_MIN_K`, `QR_GPU_MIN_BATCH_TIMES_K`, `QR_GPU_MIN_BATCH` | QR: the GPU/CPU boundary |
| `QR_SHARE_MIN_BATCH` | QR: a GPU batch shared with the CPU from this batch (0: never) |
| `QR_GPU_LARGE_MIN_K`, `QR_GPU_LARGE_MAX_BATCH` | QR: the GPU also from this k, for batches up to this (0: never / any batch) |
| `QR_DEVICE=gpu` or `cpu` | QR: bypass the GPU/CPU boundary |
| `QR_PANEL_WIDTH=8` or `16` | QR: the blocked QR's panel width (default 8 for up to 4 matrices of 768-3072 rows and 768+ columns, else 16) |
| `EIGH_SIMD_MAX_N`, `EIGH_BLOCK_MIN_N` | eigensolver: the GPU backend split |
| `EIGH_BLOCK_MIN_N_BATCHED`, `EIGH_BLOCK_MIN_BATCH` | eigensolver: batch-dependent block crossover, 0 for off |
| `EIGH_GPU_MAX_N`, `EIGH_GPU_MIN_BATCH_TIMES_N`, `EIGH_GPU_MIN_BATCH` | eigensolver: the GPU/CPU boundary |
| `EIGH_VALUES_GPU_MAX_N`, `EIGH_VALUES_GPU_MIN_BATCH_TIMES_N`, `EIGH_VALUES_GPU_MIN_BATCH` | eigensolver, eigenvalues alone: the GPU/CPU boundary |
| `EIGH_TRIDIAG_MIN_N`, `EIGH_VALUES_TRIDIAG_MIN_N` | eigensolver: the tridiag backend instead of the CPU from this N (0: never) |
| `EIGH_TRIDIAG_MAX_BATCH`, `EIGH_VALUES_TRIDIAG_MAX_BATCH` | eigensolver: the tridiag backend only for batches up to this (0: any) |
| `EIGH_QL_MIN_N`, `EIGH_QL_MAX_N` | eigensolver: the ql backend on the GPU for N in this window (`EIGH_QL_MAX_N=0`: never) |
| `EIGH_QL_SIMD=0` | eigensolver: the ql backend in threadgroup memory at every N (off: up to 32 in registers) |
| `EIGH_CPU_DC=0` | eigensolver: the CPU path with eigenvectors calls `ssyevd` whole, not its steps with the divide and conquer on idle cores |
| `EIGH_TRIDIAG_BATCH_BAND=0` | eigensolver, with eigenvectors: tridiag_batch reduces in one stage, not two for small batches from N = 384 |
| `EIGH_BAND_BATCH=0` | eigensolver, with eigenvectors: the band backend solves a batch a matrix at a time, not through tridiag_batch's two stages |
| `EIGH_SHARE_MIN_BATCH` | eigensolver: a ql batch shared with the CPU from this batch (0: never) |
| `EIGH_GPU_BIG_BATCH_MAX_N`, `EIGH_GPU_BIG_BATCH_MIN` | eigensolver: the GPU also for N above `gpu_max_n` up to this, in batches of at least this (0: never) |
| `EIGH_VALUES_BAND_MIN_N` | eigenvalues alone: the `band` backend (the two-stage reduction) from this N (0: never) |
| `EIGH_VALUES_BAND_WIDTH` | eigenvalues alone: the `band` backend's band width, 8, 16 or 32 (0: 16) |
| `EIGH_BAND_WIDTH` | the eigensolver's `band` backend's band width where the policy's is 0: 8, 16 (default) or 32 |
| `EIGH_BAND_MIN_N` | eigensolver, with eigenvectors: the `band` backend from this N (0: never) |
| `EIGH_TRIDIAG_BATCH_MIN_N`, `EIGH_TRIDIAG_BATCH_MAX_N`, `EIGH_TRIDIAG_BATCH_MIN_BATCH` | eigensolver: the `tridiag_batch` backend for N in this window from this batch (max 0: never); `EIGH_VALUES_TRIDIAG_BATCH_*` for eigenvalues alone |
| `METAL_LINALG_CPU_THREADS` | every decomposition: CPU threads a batch is spread over (default: every core) |
| `EIGH_DEVICE=tridiag` | eigensolver: every call on the tridiag backend |
| `EIGH_DEVICE=band` | eigensolver: every call on the band backend |
| `EIGH_DEVICE=tridiag_batch` | eigensolver: every call on the tridiag_batch backend |
| `EIGH_DEVICE=gpu` or `cpu` | eigensolver: bypass the GPU/CPU boundary |
| `SVD_QR_MIN_ROWS`, `SVD_QR_MIN_K` | SVD: when the QR-preconditioned backends are used |
| `SVD_BLOCK_MIN_K` | SVD: the short side from which the block kernel is used |
| `SVD_BLOCK_MIN_K_BATCHED`, `SVD_BLOCK_MIN_BATCH` | SVD: batch-dependent block crossover, 0 for off |
| `SVD_GPU_MAX_K`, `SVD_GPU_MIN_BATCH_TIMES_K`, `SVD_GPU_MIN_BATCH` | SVD: the GPU/CPU boundary |
| `SVD_BIDIAG_MIN_K`, `SVD_VALUES_BIDIAG_MIN_K` | SVD: the bidiag backend instead of the CPU from this k (0: never) |
| `SVD_BIDIAG_MAX_BATCH`, `SVD_VALUES_BIDIAG_MAX_BATCH` | SVD: the bidiag backend only for batches up to this (0: any) |
| `SVD_GK_MIN_K`, `SVD_GK_MAX_K` | SVD: the golub_kahan backend on the GPU for k in this window (`SVD_GK_MAX_K=0`: never) |
| `SVD_SHARE_MIN_BATCH` | SVD: a golub_kahan batch shared with the CPU from this batch (0: never) |
| `SVD_GPU_BIG_BATCH_MAX_K`, `SVD_GPU_BIG_BATCH_MIN` | SVD: the GPU also for k above `gpu_max_k` up to this, in batches of at least this (0: never) |
| `SVD_VALUES_BAND_MIN_K` | singular values alone: the `band` backend (the two-stage reduction) from this k (0: never) |
| `SVD_VALUES_BAND_WIDTH` | singular values alone: the `band` backend's band width, 8, 16 or 32 (0: 16) |
| `SVD_BAND_WIDTH` | the `band` backend's band width where the policy's is 0: 8, 16 (default) or 32 |
| `SVD_VALUES_GPU_MAX_K`, `SVD_VALUES_GPU_MIN_BATCH_TIMES_K`, `SVD_VALUES_GPU_MIN_BATCH`, `SVD_VALUES_GPU_MAX_L` | SVD, singular values alone: the GPU/CPU boundary (`SVD_VALUES_GPU_MIN_BATCH=0`: as with vectors) |
| `SVD_BAND_MIN_K` | SVD, with vectors: the `band` backend from this k (0: never) |
| `SVD_BIDIAG_BATCH_MIN_K`, `SVD_BIDIAG_BATCH_MAX_K`, `SVD_BIDIAG_BATCH_MIN_BATCH`, `SVD_BIDIAG_BATCH_MAX_L` | SVD: the `bidiag_batch` backend for k in this window, l up to the cap, from this batch (max k 0: never); `SVD_VALUES_BIDIAG_BATCH_*` for singular values alone |
| `SVD_GK_SIMD=0` | SVD: the golub_kahan backend in threadgroup memory at every size (off: up to 32 x 32 in registers) |
| `SVD_BIDIAG_BATCH_QR=0` | SVD: the bidiag_batch backend bidiagonalizes a tall or wide matrix as it is, not R of a QR first |
| `SVD_BIDIAG_BATCH_BAND=0` | SVD: the bidiag_batch backend reduces directly, not to a band first (singular values alone from k = 160, with vectors from 384) |
| `SVD_GK_RUN=0` | SVD: the register kernel's QR iterations a simdgroup each, no runner simdgroup |
| `SVD_CPU_DC=0` | SVD: the CPU path with vectors calls `sgesdd` whole, not its steps with the divide and conquer on idle cores |
| `SVD_BAND_BATCH=0` | SVD, with vectors: the band backend solves a batch a matrix at a time, not through bidiag_batch's two stages |
| `SVD_DEVICE=bidiag` | SVD: every call on the bidiag backend |
| `SVD_DEVICE=band` | SVD: every call on the band backend |
| `SVD_DEVICE=bidiag_batch` | SVD: every call on the bidiag_batch backend |
| `SVD_DEVICE=gpu` or `cpu` | SVD: bypass the GPU/CPU boundary |

**Programmatic overrides.** `set_qr_policy()`, `set_eigh_policy()` and
`set_svd_policy()` take precedence over both the environment and the table;
`qr_policy_source()`, `eigh_policy_source()` and `svd_policy_source()` report
which is in effect.
