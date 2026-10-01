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
   | QR | `tuning/tune_qr.py` | both GPU backends on 143 shapes, square, tall, wide and near-square | 2 min |
   | eigh | `tuning/tune_eigh.py --max-n 1024` | CPU, the whole-matrix kernel in both modes, block Jacobi; N from 2 to 1024, batch 1 to 4096 | 12 min |
   | SVD | `tuning/tune_svd.py --max-k 1024` | CPU, both kernels, each with and without QR; square k up to 1024 and tall shapes, batch 1 to 4096 | 27 min |

   Each runs two passes in random order, so thermal drift is not mistaken for
   a size effect and a noise floor can be measured.
5. Writes `submission.json` and `summary.md`, and prints the rows and how to
   send them.

`--quick` runs coarser grids, one pass each, into `build-tuning/quick/`: a
smoke test of the pipeline, never a submission.

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

`submission.json` records the chip, its CPU core counts per performance level,
its memory, the macOS, MLX and metal-linalg versions, the measurement epoch,
the load average, power source and Low Power Mode at the start and end, the
minutes each step took, and for each decomposition whether it is trustworthy
and its fitted row. It records nothing that identifies the person or the
machine.

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
counting, and a device falls back to the untuned default until it is measured
again.

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
held-out validation.

**Eigensolver** (`src/eigh.mm`): device name, GPU cores, then `simd_max_n`,
`block_min_n`, `block_min_n_batched`, `block_min_batch`, `gpu_max_n`,
`gpu_min_batch_times_n` and `gpu_min_batch`, as documented on `EighPolicy` in
`include/metal_linalg/eigh.h`. The first four choose the GPU backend, fitted
against the best GPU backend alone; the last three are the CPU boundary: GPU
iff N is at most `gpu_max_n`, batch × N at least `gpu_min_batch_times_n` and
the batch at least `gpu_min_batch`.

**SVD** (`src/svd.mm`): device name, GPU cores, then `qr_min_rows`,
`qr_min_k`, `block_min_k`, `block_min_k_batched`, `block_min_batch`,
`gpu_max_k`, `gpu_min_batch_times_k` and `gpu_min_batch`, as documented on
`SvdPolicy` in `include/metal_linalg/svd.h`: whether to precondition with QR,
which kernel, then the CPU boundary in the eigensolver's form.

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
python3 tuning/tune_eigh.py build/sweep_eigh --max-n 1024  # writes eigh-tune-results/
python3 tuning/tune_svd.py  build/sweep_svd  --max-k 1024  # writes svd-tune-results/
```

| option | harness | use |
|---|---|---|
| `--out DIR` | all | where to write |
| `--passes N` | all | more than two passes, for a noisy machine |
| `--full` | QR | a denser grid, about three times longer |
| `--max-n`, `--max-k` | eigh, SVD | the largest size on the grid (default 512; 768 and 1024 added up to this) |
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
| `EIGH_SIMD_MAX_N`, `EIGH_BLOCK_MIN_N` | eigensolver: the GPU backend split |
| `EIGH_BLOCK_MIN_N_BATCHED`, `EIGH_BLOCK_MIN_BATCH` | eigensolver: batch-dependent block crossover, 0 for off |
| `EIGH_GPU_MAX_N`, `EIGH_GPU_MIN_BATCH_TIMES_N`, `EIGH_GPU_MIN_BATCH` | eigensolver: the GPU/CPU boundary |
| `EIGH_DEVICE=gpu` or `cpu` | eigensolver: bypass the GPU/CPU boundary |
| `SVD_QR_MIN_ROWS`, `SVD_QR_MIN_K` | SVD: when the QR-preconditioned backends are used |
| `SVD_BLOCK_MIN_K` | SVD: the short side from which the block kernel is used |
| `SVD_BLOCK_MIN_K_BATCHED`, `SVD_BLOCK_MIN_BATCH` | SVD: batch-dependent block crossover, 0 for off |
| `SVD_GPU_MAX_K`, `SVD_GPU_MIN_BATCH_TIMES_K`, `SVD_GPU_MIN_BATCH` | SVD: the GPU/CPU boundary |
| `SVD_DEVICE=gpu` or `cpu` | SVD: bypass the GPU/CPU boundary |

**Programmatic overrides.** `set_qr_policy()`, `set_eigh_policy()` and
`set_svd_policy()` take precedence over both the environment and the table;
`qr_policy_source()`, `eigh_policy_source()` and `svd_policy_source()` report
which is in effect.
