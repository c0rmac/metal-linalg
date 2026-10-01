# Tuning on another Apple device

Which backend is fastest depends on the GPU, and for the eigensolver and the
SVD on the CPU beside it. The library therefore ships a table of measured
policies per device instead of constants, and a device without an entry runs
an untuned default. This page is the procedure for measuring a device and
adding its entry.

| what is tuned | measured by | time on an M1 | on an M5 Pro | entry goes in |
|---|---|---|---|---|
| QR backend crossover | `tuning/tune_qr.py` + `build/sweep_qr` | about 3 minutes | 2 minutes | `kTuned[]` in `src/qr.mm` |
| eigensolver routing | `tuning/tune_eigh.py` + `build/sweep_eigh` | about 17 minutes | 12 minutes with `--max-n 1024` | `kTuned[]` in `src/eigh.mm` |
| SVD routing | `tuning/tune_svd.py` + `build/sweep_svd` | about 20 minutes | 27 minutes with `--max-k 1024` | `kTuned[]` in `src/svd.mm` |

Devices measured so far:

| device | QR | eigensolver | SVD | study |
|---|---|---|---|---|
| Apple M1, 8 GPU cores | measured | measured | not yet: the M1 timings were taken on a busy machine | [QR](studies/qr-routing-apple-m1.md), [eigh](studies/eigh-routing-apple-m1.md) |
| Apple M5 Pro, 20 GPU cores | measured | measured | measured | [all three](studies/routing-apple-m5-pro.md) |

## 1. Before you start

**Requirements**

- Apple Silicon Mac with Xcode or its command line tools (clang)
- [MLX](https://github.com/ml-explore/mlx) (`brew install mlx`) and CMake 3.25 or later
- Python 3.8 or later; the harnesses use the standard library only

The Metal shader compiler is not needed: without it the build uses the
compiled shaders committed under `shaders/prebuilt/`. It is needed only to
change a shader (from Xcode 26 it is a separate download,
`xcodebuild -downloadComponent MetalToolchain`).

**Conditions.** These are timing experiments, so the state of the machine is
part of the measurement:

- plugged into mains power, Low Power Mode off
- no other heavy jobs running; quit builds, training runs, video calls
- run the sweeps one after the other, never together
- leave the machine alone until the sweep finishes

To check before you begin:

```sh
uptime              # load averages should be low single digits
pmset -g batt       # should say 'AC Power'
```

The eigensolver and SVD harnesses record and judge these themselves (section
6). The QR harness measures a noise floor but does not inspect the machine, so
for QR the conditions are on you.

## 2. Build and run the correctness tests

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/opt/homebrew
cmake --build build -j
ctest --test-dir build --output-on-failure
```

All three suites must pass before any timing means anything. They do not depend on
the device's tuning: every routing check installs the policy it needs and
restores the device's own afterwards. If a test fails on your device, stop and
report it with the output of `./build/test_qr`, `./build/test_eigh` or
`./build/test_svd`; do not tune around a correctness failure.

To see what your device is running today:

```sh
./build/sweep_qr   --policy
./build/sweep_eigh --policy
./build/sweep_svd  --policy
```

Each prints the device name, its GPU core count, the policy in effect and its
source. A source of `default:untuned-device (<name>)` means there is no entry
for this device yet, which is the normal starting point.

## 3. QR crossover

```sh
python3 tuning/tune_qr.py build/sweep_qr
```

This measures both QR backends on 143 shapes balanced across square, tall,
wide and near-square inputs, twice, in random order. It writes
`qr-tune-results/`:

| file | contents |
|---|---|
| `report.md` | the written report: the band, the regret curve, rule comparison, noise floor |
| `results.json` | the same analysis as plot-ready data |
| `raw.csv` | every timing, so the analysis can be redone without remeasuring |

and ends by printing the entry:

```
band 480..512   ship M >= 512   (1.0115x geomean regret)
kTuned entry:  {"Apple M5 Pro", 20, 512, 512, 16},
```

The fields are device name, GPU cores, the row crossover for small batches,
the row crossover for large batches, and the batch that separates them. The
two crossovers are equal unless a batch-dependent split survived held-out
validation on your device.

Options: `--full` for a denser grid (about three times longer), `--passes N`
for more than two passes.

To try a crossover before rebuilding:

```sh
QR_M_CROSSOVER=320 ./build/benchmark_qr
```

## 4. Eigensolver routing

```sh
python3 tuning/tune_eigh.py build/sweep_eigh --max-n 1024
```

This measures four backends (CPU, the whole-matrix kernel in both modes, block
Jacobi) on a grid of matrix size N and batch, twice, in random order. It
writes `eigh-tune-results/`:

| file | contents |
|---|---|
| `report.md` | the written report: the entry, both fitting stages, decision surfaces, held-out verdicts, noise floor |
| `results.json` | the same analysis as plot-ready data |
| `raw.csv` | every timing |
| `policy.json` | the device, the policy it was running, the cost-model calibration, machine state, probe drift |

and ends by printing the entry and the same policy as environment variables:

```
kTuned[] row:  {"Apple M5 Pro", 20,   0, 96, 0, 0,   1024, 512, 16},
```

The fields are device name, GPU cores, then `simd_max_n`, `block_min_n`,
`block_min_n_batched`, `block_min_batch`, `gpu_max_n`,
`gpu_min_batch_times_n` and `gpu_min_batch`, as documented on `EighPolicy` in
`include/metal_linalg/eigh.h`. The first four choose the GPU backend, fitted
against the best GPU backend alone. The last three are the CPU boundary: GPU
iff N is at most `gpu_max_n`, batch * N at least `gpu_min_batch_times_n` and
the batch at least `gpu_min_batch`. The minimum batch is 1 (no minimum) where
the product alone already keeps lone matrices on the CPU, as on the M1; on the
M5 Pro, whose GPU is ahead up to N = 1024 in batches and behind on every lone
matrix, it is 16.

Options:

| option | use |
|---|---|
| `--max-n 1024` | extend the grid to N = 768 and 1024. Use it on any GPU with more cores than an M1, and whenever the report warns that the GPU is still ahead at the largest N measured |
| `--passes N` | more than two passes, for a noisy machine |
| `--quick` | one pass on a coarse grid, about five minutes. A smoke test of the pipeline only; its report is marked indicative and its row must not be pasted |
| `--limit S` | per-point timeout in seconds, default 240 |

To try a policy before rebuilding, copy the environment line from the report:

```sh
EIGH_GPU_MAX_N=128 EIGH_BLOCK_MIN_N=96 ./build/benchmark_eigh
```

## 5. SVD routing

```sh
python3 tuning/tune_svd.py build/sweep_svd --max-k 1024
```

This measures five backends: a thin SVD on the CPU; the whole-matrix Jacobi
kernel and the block Jacobi kernel on the matrix itself; and each of the two
after this library's QR, on the triangular factor. The grid is square shapes
from 4 to 512 and tall shapes of aspect 2 to 32 with short side 8 to 256, at
batches from 1 to 4096, twice, in random order. The block kernel is timed from
a short side of 32. Only shapes with M >= N are measured: every backend
decomposes a wide matrix through its transpose, so it costs what the
transposed shape costs. It writes `svd-tune-results/` with the same four files
as the eigensolver harness and ends by printing the row:

```
kTuned[] row:  {"Apple M5 Pro", 20,   512, 32,   192, 64, 64,   1024, 512, 4},
```

The fields are device name, GPU cores, then `qr_min_rows`, `qr_min_k`,
`block_min_k`, `block_min_k_batched`, `block_min_batch`, `gpu_max_k`,
`gpu_min_batch_times_k` and `gpu_min_batch`, as documented on `SvdPolicy` in
`include/metal_linalg/svd.h`. The first five are the GPU backend, fitted
against the best GPU backend alone: whether to precondition with QR, and which
kernel. The batch-dependent kernel crossover (`block_min_k_batched`,
`block_min_batch`) is fitted on half the points and kept only if it is better
on the other half in at least 95% of bootstrap resamples, exactly as the
eigensolver's; otherwise the row carries `0, 0`. The last three are the CPU
boundary, in the eigensolver's form.

Options are those of the eigensolver harness, with `--max-k` in place of
`--max-n`: the square shapes go up to 512 by default and `--max-k 1024` adds
768 and 1024. Above 512 only the block kernel and the CPU are timed, since the
whole-matrix kernel takes seconds there.

To try a policy before rebuilding, copy the environment line from the report:

```sh
SVD_BLOCK_MIN_K=128 SVD_GPU_MAX_K=256 ./build/benchmark_svd
```

## 6. Reading the report before you trust it

**Is the run usable?** The eigensolver and SVD reports' "Answer" section either shows
the row, or says "Indicative only; do not paste this row". It is marked
indicative when any of these is true:

| warning | meaning | what to do |
|---|---|---|
| the machine was not idle | load average above half the CPU count at the start or end | stop the other jobs and rerun |
| the machine changed state during the sweep | a probe point timed before and after moved by more than 25%: thermal throttling or a job that started midway | rerun, on mains, once the machine has cooled |
| single pass | `--quick` was used, so there is no noise floor | rerun without `--quick` |

For QR, check the "Noise floor" section instead: a median pass-to-pass ratio
above about 1.10 means the machine was not stable, and the run should be
repeated.

**Other warnings** do not invalidate a run; they tell you what to do next:

| warning | what to do |
|---|---|
| the GPU is still ahead of the CPU at the largest N (or k) measured | rerun with `--max-n 1024` (eigensolver) or `--max-k 1024` (SVD); the cap reported is only a lower bound. At 1024 on an M5 Pro this is expected, and harmless: a lone matrix stays on the CPU through the minimum batch, and batches above 1024 take seconds either way |
| this device has no entry in `kTuned[]` | expected on a new device; paste the row |
| the fitted policy differs from the one in effect | update this device's row |
| chosen rule loses more than 25% at ... | informational: the points where a simple rule is furthest from the best backend. Worth a look if they cluster in a region you care about |
| chosen rule picks a backend that was not timed at ... | the cost model skipped a point it judged too slow, and the rule chose that backend there. The model's guess counts in the geometric mean and is kept out of the worst case |
| the best feature is no longer M (QR) | a structural change, not a moved threshold. Open an issue with the report rather than pasting the entry |

**The flat region, not the single best value.** Each constant is reported with
the range of values within 0.5% (eigensolver, SVD) or 0.3% (QR) of the best
score. A wide range means the constant barely matters on your device; a range
of one value means it is sharp. If the value already in the table lies inside
the range, the harness keeps it, since a gain smaller than the noise is not
worth a constant that changes from run to run.

**Refinements.** Each harness also tries richer rules (a batch-dependent
crossover, a per-N boundary) on half the points and scores them on the other
half, against the plain fit on the same half. One marked "justified" is
included in the printed entry automatically. One marked "rejected" is not,
however good it looked on the half it was fitted to.

## 7. Apply the entries

1. Paste the QR entry into `kTuned[]` in `src/qr.mm`, the eigensolver row
   into `kTuned[]` in `src/eigh.mm` and the SVD row into `kTuned[]` in
   `src/svd.mm`. Keep the existing rows.
2. Rebuild and rerun the tests:

   ```sh
   cmake --build build -j
   ctest --test-dir build --output-on-failure
   ```

3. Confirm the device now resolves to its own entry:

   ```sh
   ./build/sweep_qr   --policy
   ./build/sweep_eigh --policy
   ./build/sweep_svd  --policy
   ```

   All three should report `tuned:<your device name>`.

## 8. Contribute the measurements

So that others can diff their runs against yours, commit the results next to
the others under `docs/results/`, with the core count in the directory name
when the chip ships in more than one configuration:

```sh
cp -r qr-tune-results   docs/results/qr-apple-m4-max-40c
cp -r eigh-tune-results docs/results/eigh-apple-m4-max-40c
cp -r svd-tune-results  docs/results/svd-apple-m4-max-40c
```

Then add the device to the per-device tables in [`qr.md`](qr.md),
[`eigh.md`](eigh.md), [`svd.md`](svd.md), the summary in the top-level
`README.md`, and the table at the top of this page, and open a pull request
containing the three `kTuned[]` rows, the three results directories and the
table rows. Only commit runs whose reports are not marked indicative. A
written study like [`studies/routing-apple-m5-pro.md`](studies/routing-apple-m5-pro.md)
is welcome but not required.

## 9. Optional: launch parameters and benchmarks

The routing above decides *which* backend runs. How each eigensolver backend is
launched (threads per matrix, inner sweeps per block subproblem) scales with
the detected core count and is not part of the per-device table, but it can be
inspected:

```sh
./build/benchmark_eigh --tune      # five tables; --tune 1..5 for one
./build/benchmark_svd  --tune      # simdgroups per matrix
```

The M1 tables and what they showed are in
[`studies/eigh-launch-parameters-apple-m1.md`](studies/eigh-launch-parameters-apple-m1.md)
and [`studies/svd-design-notes.md`](studies/svd-design-notes.md). For the
headline GPU-against-CPU tables:

```sh
./build/benchmark_qr
./build/benchmark_eigh
./build/benchmark_svd
```

`./build/probe_occupancy` reports the threadgroup-memory limits that set how
many matrices `qr_unblocked` keeps resident, which governs whether batch count
can matter to the QR crossover at all.

## 10. Working with existing data

Neither of these touches the GPU.

Redo the analysis from raw timings, for example after a harness update.
Passing the sweep binary as well makes the report record the policy that
binary resolves:

```sh
python3 tuning/tune_qr.py   build/sweep_qr   --reanalyse qr-tune-results/raw.csv
python3 tuning/tune_eigh.py build/sweep_eigh --reanalyse eigh-tune-results/raw.csv
python3 tuning/tune_svd.py  build/sweep_svd  --reanalyse svd-tune-results/raw.csv
```

`tune_qr.py --reanalyse` accepts several `raw.csv` files and merges them, so a
sweep can be topped up rather than repeated. Re-render a report from its JSON:

```sh
python3 tuning/tune_qr.py   --from qr-tune-results/results.json
python3 tuning/tune_eigh.py --from eigh-tune-results/results.json
python3 tuning/tune_svd.py  --from svd-tune-results/results.json
```

Compare with the committed runs in [`results/`](results/).

## 11. Troubleshooting

| symptom | cause and fix |
|---|---|
| `Impacting Interactivity` in an error message | macOS stopped a GPU command buffer that ran for several seconds. The library splits large batches to avoid this; if it appears during a sweep the point is retried and then recorded as failed. Lower `EIGH_CHUNK_MS` or `SVD_CHUNK_MS` (default 750) if it recurs |
| rows with `ok` = 0 in `raw.csv` | the backend failed its correctness gate or timed out at that point, and the point is excluded from the analysis. A few are harmless. Many at small sizes indicate a real fault: run the correctness tests |
| the sweep seems stuck | the largest points take several seconds per call and are repeated. Progress and an estimate are printed every 20 points |
| `sweep_eigh --policy failed` (or `sweep_svd`) | the binary is older than the harness; rebuild it |
| `Could NOT find MLX` from CMake | MLX is not installed or not on CMake's path: `brew install mlx`, and pass `-DCMAKE_PREFIX_PATH=/opt/homebrew` |
| `missing Metal Toolchain` | only matters when changing a shader; the build otherwise uses `shaders/prebuilt/`. `xcodebuild -downloadComponent MetalToolchain` installs it |
| very different answers from two runs | the machine was not in the same state for both. Compare the noise floor and machine-state lines of the two reports |

## 12. Reference

**Where the studies are.** The reasoning behind each harness, and the results
in full:

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
