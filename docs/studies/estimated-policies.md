# Estimated policies for the Macs nobody has measured

Every Mac's routing comes from measurements of that Mac, and so far one Mac
(an M5 Pro) has current ones. Every other Mac ran the untuned default: the
M1's 2.0-era thresholds, with every backend added since switched off. This
study replaces that with an estimate, derived per Mac from the M5 Pro's
measurements and what is published about the two chips, and checks how close
the estimate comes.

**What came of it (2.14.0):** a Mac nobody has measured is routed by the M5
Pro's own timings refitted for its GPU against its CPU
([`tuning/estimate.py`](../../tuning/estimate.py), [`src/estimate.cpp`](../../src/estimate.cpp)).
On Macs simulated from the M5 Pro's timings with a GPU 1 to 4 times weaker
against its CPU, the estimate comes within 0-3% of the best routing on
geometric mean, worst points 1.0-2.1x, where the untuned default was 1.3-3.0x,
worst 4-54x; and within the same 0-3% when the benchmarks it relies on
are 1.5x off. Along the way, a fault in the SVD harness's fit for singular
values alone, which the M5 Pro's row carried too: its fix makes that row's
`svdvals` routing 1.4% faster on geometric mean, up to 1.9x on tall batches.

## 1. What the untuned default costs

On the M5 Pro's own measurements (the first row of each table in section 4),
the untuned default is 1.30x the best routing for eigh on geometric mean,
1.42x for eigvalsh, 1.35x for the SVD, 1.85x for svdvals and 1.34x for QR, its
worst points 4-16x. Two things make it so. It predates the CPU path that
spreads a batch over every core (2.9.0), so it sends batches to the GPU long
before the GPU wins: `batch * N >= 1024`, where the M5 Pro's measurements say
16384; the eigenvalues of 64 matrices of 64x64 go to a Jacobi kernel at
2.9 ms instead of the CPU's 0.3. And it has none of the backends added since 2.7.0 (`ql`,
`golub_kahan`, `tridiag`, `bidiag`, `band`, sharing a batch with the CPU), so
4096 matrices of 64x64 take 155 ms on a Jacobi kernel instead of 24 on `ql`
shared with the CPU, and one 4096x4096 matrix's singular values 2 s on the
CPU instead of 229 ms on `band`. On a slower GPU it is worse (section 4):
the Jacobi kernels it picks lose by more.

## 2. The estimate

What decides a GPU-or-CPU threshold is how fast the GPU is against the CPU.
The estimate takes a measured Mac (the anchor) and refits its timings as if
its GPU were `s` times slower against its CPU: every GPU backend's time
multiplied by `s`, a batch shared between the GPU and the CPU rescaled as two
workers in parallel (1/T = 1/T_GPU + 1/T_CPU, keeping the share's measured
overhead over that ideal), the CPU's times unchanged. The refit is the
anchor's own analysis (`tune_*.py --reanalyse`), so an estimated row is a
fitted row like any other, for a weaker GPU. Two slowdowns, since the
backends are limited by different things:

- `small`, for the batched kernels (one threadgroup a matrix) and the Jacobi
  block kernels: the GPU's compute against the CPU's;
- `large`, for the large-matrix reductions (eigh `tridiag` and `band`, SVD
  `bidiag` and `band`, QR's streaming kernel), which stream the matrix
  through memory: the memory bandwidth against the CPU as well.

For a Mac in [`tuning/chip_specs.py`](../../tuning/chip_specs.py), from the
medians of Geekbench 7's multi-core and Metal scores across the Macs that ship
it, and the memory bandwidth from Apple's specifications:

    q_small = (metal / cpu) / (the anchor's metal / cpu)
    q_large = min(q_small, (bandwidth / cpu) / (the anchor's bandwidth / cpu))
    s       = max(1, margin / q)

with a margin of 1.25 within the anchor's generation and 1.5 for another. A
Mac the table does not list is estimated from its core counts alone, `q =
(GPU cores / CPU cores)` over the anchor's, with a margin of 2: against the
benchmarks that model is 0.71-1.86x off for the listed Macs (section 5), the
overrating of the GPU at most 1.86x, for the M1 Ultra. `s` is never below 1:
a GPU stronger than the anchor's gets the anchor's own crossovers, which
leave some of its wins unused but send nothing to a backend that loses.

The rows are refitted for every pair of `s` on a ladder, 1, 1.15, 1.3, 1.5,
1.75, 2, 2.5, 3 and 4 (81 refits a decomposition, about a minute in all), and
a Mac gets the row of the smallest ladder values at least its own. With more
than one measured Mac, the anchor is one of the target's generation if there
is one, else the nearest in GPU core count; a measured Mac is an anchor once
its runs are at the current kernels. `tuning/generate_tables.py` refits the
rows whenever the measurements change, so each new measurement improves its
neighbours' estimates too. `METAL_LINALG_ESTIMATE_AS="<chip>[:<GPU cores>[:<CPU
cores>]]"` shows what another Mac gets.

## 3. Every listed Mac's estimate

From the M5 Pro (`python3 tuning/estimate.py`):

| Mac | GPU cores | CPU cores | basis | slowdown, batched | slowdown, large | ladder |
|---|---|---|---|---|---|---|
| Apple M1 | 7 | 8 | benchmarks | 1.86 | 1.86 | 2.0, 2.0 |
| Apple M1 | 8 | 8 | benchmarks | 1.80 | 1.80 | 2.0, 2.0 |
| Apple M1 Pro | 14 | 8 | benchmarks | 1.11 | 1.11 | 1.15, 1.15 |
| Apple M1 Pro | 16 | 10 | benchmarks | 1.29 | 1.29 | 1.3, 1.3 |
| Apple M1 Max | 24 | 10 | benchmarks | 1.07 | 1.07 | 1.15, 1.15 |
| Apple M1 Max | 32 | 10 | benchmarks | 1.00 | 1.00 | 1.0, 1.0 |
| Apple M1 Ultra | 48 | 20 | benchmarks | 1.29 | 1.29 | 1.3, 1.3 |
| Apple M1 Ultra | 64 | 20 | benchmarks | 1.00 | 1.00 | 1.0, 1.0 |
| Apple M2 | 8 | 8 | benchmarks | 1.70 | 1.70 | 1.75, 1.75 |
| Apple M2 | 10 | 8 | benchmarks | 1.36 | 1.36 | 1.5, 1.5 |
| Apple M2 Pro | 16 | 10 | benchmarks | 1.13 | 1.13 | 1.15, 1.15 |
| Apple M2 Pro | 19 | 12 | benchmarks | 1.22 | 1.22 | 1.3, 1.3 |
| Apple M2 Max | 30 | 12 | benchmarks | 1.00 | 1.00 | 1.0, 1.0 |
| Apple M2 Max | 38 | 12 | benchmarks | 1.00 | 1.00 | 1.0, 1.0 |
| Apple M2 Ultra | 60 | 24 | benchmarks | 1.00 | 1.00 | 1.0, 1.0 |
| Apple M2 Ultra | 76 | 24 | benchmarks | 1.00 | 1.00 | 1.0, 1.0 |
| Apple M3 | 8 | 8 | benchmarks | 1.65 | 1.65 | 1.75, 1.75 |
| Apple M3 | 10 | 8 | benchmarks | 1.48 | 1.60 | 1.5, 1.75 |
| Apple M3 Pro | 14 | 11 | benchmarks | 1.32 | 1.40 | 1.5, 1.5 |
| Apple M3 Pro | 18 | 12 | benchmarks | 1.33 | 1.54 | 1.5, 1.75 |
| Apple M3 Max | 30 | 14 | benchmarks | 1.02 | 1.02 | 1.15, 1.15 |
| Apple M3 Max | 40 | 16 | benchmarks | 1.00 | 1.00 | 1.0, 1.0 |
| Apple M3 Ultra | 60 | 28 | benchmarks | 1.00 | 1.00 | 1.0, 1.0 |
| Apple M3 Ultra | 80 | 32 | benchmarks | 1.00 | 1.00 | 1.0, 1.0 |
| Apple M4 | 8 | 8 | benchmarks | 1.72 | 1.72 | 1.75, 1.75 |
| Apple M4 | 8 | 10 | benchmarks | 1.92 | 1.92 | 2.0, 2.0 |
| Apple M4 | 10 | 10 | benchmarks | 1.73 | 1.74 | 1.75, 1.75 |
| Apple M4 Pro | 16 | 12 | benchmarks | 1.29 | 1.29 | 1.3, 1.3 |
| Apple M4 Pro | 20 | 14 | benchmarks | 1.31 | 1.31 | 1.5, 1.5 |
| Apple M4 Max | 32 | 14 | benchmarks | 1.00 | 1.00 | 1.0, 1.0 |
| Apple M4 Max | 40 | 16 | benchmarks | 1.00 | 1.00 | 1.0, 1.0 |
| Apple M5 | 8 | 10 | benchmarks | 1.31 | 1.31 | 1.5, 1.5 |
| Apple M5 | 10 | 10 | benchmarks | 1.22 | 1.28 | 1.3, 1.3 |
| Apple M5 Pro | 16 | 15 | benchmarks | 1.20 | 1.20 | 1.3, 1.3 |
| Apple M5 Pro | 20 | 18 | benchmarks | 1.25 | 1.25 | 1.3, 1.3 |
| Apple M5 Max | 32 | 18 | benchmarks | 1.00 | 1.00 | 1.0, 1.0 |
| Apple M5 Max | 40 | 18 | benchmarks | 1.00 | 1.00 | 1.0, 1.0 |
| Apple M5 Ultra | 64 | 36 | benchmarks | 1.02 | 1.02 | 1.15, 1.15 |
| Apple M5 Ultra | 80 | 36 | benchmarks | 1.00 | 1.00 | 1.0, 1.0 |
| Apple A18 Pro | 5 | 6 | benchmarks | 1.42 | 1.67 | 1.5, 1.75 |

The Max and Ultra chips, whose GPUs are stronger against their CPUs than the
M5 Pro's, get its own row. The base chips need the most work before the GPU
pays: 1.4-1.9x for M1 to M4, 1.2-1.3x for the M5. The bandwidth term raises
`large` above `small` where memory is narrow for the chip's CPU: the M3, the
M3 Pro and the A18 Pro.

## 4. Simulated Macs

There is no second current measurement to test against, so the test is a
simulation: the M5 Pro's timings with every GPU backend `s` times slower (the
batched kernels and the large-matrix backends alike), for `s` from 1 to 4.
Each candidate is scored against the best backend timed at each point:
geometric-mean time over the best, the worst point, and the share of points
more than 1.25x off. The candidates are the untuned default, the anchor's own
row copied as it is, the estimated row for the true `s`, for `s` with the
1.25 margin, and for `s` underestimated 1.5x, as if the benchmarks overrated
the GPU by that much (`python3 tuning/estimate.py --study`).

#### eigh

| true s | untuned default | anchor's own row | estimated, s exact | estimated, s x 1.25 (the margin) | estimated, s / 1.5 (benchmarks 1.5x off) |
|---|---|---|---|---|---|
| 1.0 | 1.30x, worst 6.6x, 30% | 1.01x, worst 1.4x, 2% | 1.01x, worst 1.4x, 2% | 1.02x, worst 1.7x, 3% | 1.01x, worst 1.4x, 2% |
| 1.3 | 1.36x, worst 7.7x, 31% | 1.01x, worst 1.7x, 1% | 1.01x, worst 1.3x, 0% | 1.04x, worst 1.9x, 8% | 1.01x, worst 1.7x, 1% |
| 1.75 | 1.45x, worst 9.2x, 31% | 1.02x, worst 2.2x, 4% | 1.02x, worst 1.6x, 5% | 1.02x, worst 1.6x, 6% | 1.01x, worst 1.5x, 2% |
| 2.5 | 1.58x, worst 11.9x, 30% | 1.05x, worst 3.2x, 7% | 1.01x, worst 1.3x, 1% | 1.01x, worst 1.3x, 2% | 1.01x, worst 1.3x, 1% |
| 4.0 | 1.79x, worst 17.8x, 30% | 1.09x, worst 5.1x, 8% | 1.00x, worst 1.1x, 0% | 1.00x, worst 1.1x, 0% | 1.00x, worst 1.3x, 0% |

#### eigvalsh

| true s | untuned default | anchor's own row | estimated, s exact | estimated, s x 1.25 (the margin) | estimated, s / 1.5 (benchmarks 1.5x off) |
|---|---|---|---|---|---|
| 1.0 | 1.42x, worst 11.2x, 32% | 1.02x, worst 1.5x, 4% | 1.02x, worst 1.5x, 4% | 1.02x, worst 1.7x, 5% | 1.02x, worst 1.5x, 4% |
| 1.3 | 1.51x, worst 13.4x, 31% | 1.02x, worst 1.8x, 4% | 1.01x, worst 1.5x, 3% | 1.01x, worst 1.5x, 3% | 1.02x, worst 1.8x, 4% |
| 1.75 | 1.63x, worst 16.8x, 30% | 1.03x, worst 2.4x, 5% | 1.01x, worst 1.3x, 1% | 1.01x, worst 1.3x, 1% | 1.01x, worst 1.4x, 1% |
| 2.5 | 1.79x, worst 22.3x, 29% | 1.05x, worst 3.5x, 7% | 1.00x, worst 1.1x, 0% | 1.00x, worst 1.2x, 0% | 1.01x, worst 1.4x, 0% |
| 4.0 | 2.04x, worst 34.6x, 29% | 1.08x, worst 5.6x, 9% | 1.00x, worst 1.0x, 0% | 1.00x, worst 1.0x, 0% | 1.00x, worst 1.0x, 0% |

#### SVD

| true s | untuned default | anchor's own row | estimated, s exact | estimated, s x 1.25 (the margin) | estimated, s / 1.5 (benchmarks 1.5x off) |
|---|---|---|---|---|---|
| 1.0 | 1.35x, worst 6.2x, 36% | 1.02x, worst 1.6x, 3% | 1.02x, worst 1.6x, 3% | 1.02x, worst 1.6x, 3% | 1.02x, worst 1.6x, 3% |
| 1.3 | 1.43x, worst 8.1x, 37% | 1.01x, worst 1.4x, 1% | 1.01x, worst 1.4x, 1% | 1.01x, worst 1.4x, 2% | 1.01x, worst 1.4x, 1% |
| 1.75 | 1.56x, worst 10.9x, 37% | 1.01x, worst 1.6x, 0% | 1.00x, worst 1.3x, 0% | 1.01x, worst 1.3x, 1% | 1.01x, worst 1.6x, 1% |
| 2.5 | 1.76x, worst 15.6x, 37% | 1.02x, worst 2.2x, 2% | 1.01x, worst 1.4x, 1% | 1.01x, worst 1.3x, 1% | 1.01x, worst 1.4x, 1% |
| 4.0 | 2.07x, worst 24.9x, 37% | 1.04x, worst 3.6x, 5% | 1.00x, worst 1.1x, 0% | 1.00x, worst 1.1x, 0% | 1.00x, worst 1.1x, 0% |

#### svdvals

| true s | untuned default | anchor's own row | estimated, s exact | estimated, s x 1.25 (the margin) | estimated, s / 1.5 (benchmarks 1.5x off) |
|---|---|---|---|---|---|
| 1.0 | 1.85x, worst 16.2x, 44% | 1.03x, worst 2.1x, 5% | 1.03x, worst 2.1x, 5% | 1.03x, worst 2.1x, 6% | 1.03x, worst 2.1x, 5% |
| 1.3 | 1.99x, worst 19.2x, 43% | 1.02x, worst 1.7x, 3% | 1.02x, worst 1.7x, 3% | 1.04x, worst 1.9x, 8% | 1.02x, worst 1.7x, 3% |
| 1.75 | 2.18x, worst 23.7x, 42% | 1.01x, worst 1.6x, 2% | 1.02x, worst 1.4x, 1% | 1.02x, worst 1.4x, 1% | 1.01x, worst 1.6x, 2% |
| 2.5 | 2.48x, worst 33.8x, 41% | 1.02x, worst 2.3x, 4% | 1.00x, worst 1.2x, 0% | 1.00x, worst 1.2x, 0% | 1.00x, worst 1.3x, 0% |
| 4.0 | 2.97x, worst 54.1x, 41% | 1.04x, worst 3.6x, 7% | 1.00x, worst 1.0x, 0% | 1.00x, worst 1.0x, 0% | 1.00x, worst 1.0x, 0% |

#### QR

| true s | untuned default | anchor's own row | estimated, s exact | estimated, s x 1.25 (the margin) | estimated, s / 1.5 (benchmarks 1.5x off) |
|---|---|---|---|---|---|
| 1.0 | 1.34x, worst 4.3x, 44% | 1.00x, worst 1.2x, 0% | 1.00x, worst 1.2x, 0% | 1.01x, worst 1.3x, 1% | 1.00x, worst 1.2x, 0% |
| 1.3 | 1.51x, worst 5.6x, 45% | 1.00x, worst 1.2x, 0% | 1.00x, worst 1.1x, 0% | 1.00x, worst 1.1x, 0% | 1.00x, worst 1.2x, 0% |
| 1.75 | 1.74x, worst 7.6x, 49% | 1.02x, worst 1.6x, 4% | 1.00x, worst 1.1x, 0% | 1.00x, worst 1.2x, 0% | 1.00x, worst 1.3x, 1% |
| 2.5 | 2.09x, worst 10.8x, 51% | 1.04x, worst 2.3x, 6% | 1.00x, worst 1.0x, 0% | 1.00x, worst 1.0x, 0% | 1.01x, worst 1.3x, 1% |
| 4.0 | 2.66x, worst 17.3x, 52% | 1.08x, worst 3.7x, 8% | 1.00x, worst 1.0x, 0% | 1.00x, worst 1.0x, 0% | 1.00x, worst 1.0x, 0% |

The estimated rows are within 0-3% of the best throughout, and as good when
`s` is underestimated 1.5x: the refit's thresholds degrade gradually.
Copying the anchor's row is close on average too, but its worst points grow
with `s` (to 5.6x at `s = 4`), where the estimates' stay at 1.0-1.6x: they
are the anchor's crossovers moved for the weaker GPU. The margin costs up to
3% on average at `s = 1.3`, the price of not relying on the benchmarks
exactly.

What the simulation cannot show: a real Mac differs from the M5 Pro in more
than one ratio. Its kernels' launch latency and occupancy, its CPU's speed
per matrix size, and its memory system all change the crossovers in ways a
single slowdown per kind of backend does not capture. The margins are there
for that, and the first measurement of another chip will say how well they
cover it.

## 5. Core counts against benchmarks

For every listed Mac, the GPU-against-CPU ratio the core-count fallback gives
over the one the benchmarks give (above 1: the fallback overrates the GPU):

| Mac | GPU cores | CPU cores | from benchmarks | from core counts | ratio |
|---|---|---|---|---|---|
| Apple M1 | 7 | 8 | 0.81 | 0.79 | 0.98 |
| Apple M1 | 8 | 8 | 0.83 | 0.90 | 1.08 |
| Apple M1 Pro | 14 | 8 | 1.35 | 1.57 | 1.17 |
| Apple M1 Pro | 16 | 10 | 1.16 | 1.44 | 1.24 |
| Apple M1 Max | 24 | 10 | 1.40 | 2.16 | 1.54 |
| Apple M1 Max | 32 | 10 | 1.87 | 2.88 | 1.54 |
| Apple M1 Ultra | 48 | 20 | 1.16 | 2.16 | 1.86 |
| Apple M1 Ultra | 64 | 20 | 1.55 | 2.88 | 1.86 |
| Apple M2 | 8 | 8 | 0.88 | 0.90 | 1.02 |
| Apple M2 | 10 | 8 | 1.10 | 1.12 | 1.02 |
| Apple M2 Pro | 16 | 10 | 1.33 | 1.44 | 1.08 |
| Apple M2 Pro | 19 | 12 | 1.23 | 1.42 | 1.16 |
| Apple M2 Max | 30 | 12 | 1.63 | 2.25 | 1.38 |
| Apple M2 Max | 38 | 12 | 2.05 | 2.85 | 1.39 |
| Apple M2 Ultra | 60 | 24 | 1.53 | 2.25 | 1.47 |
| Apple M2 Ultra | 76 | 24 | 1.93 | 2.85 | 1.47 |
| Apple M3 | 8 | 8 | 0.91 | 0.90 | 0.99 |
| Apple M3 | 10 | 8 | 1.02 | 1.12 | 1.11 |
| Apple M3 Pro | 14 | 11 | 1.13 | 1.15 | 1.01 |
| Apple M3 Pro | 18 | 12 | 1.13 | 1.35 | 1.20 |
| Apple M3 Max | 30 | 14 | 1.47 | 1.93 | 1.31 |
| Apple M3 Max | 40 | 16 | 1.66 | 2.25 | 1.36 |
| Apple M3 Ultra | 60 | 28 | 1.59 | 1.93 | 1.22 |
| Apple M3 Ultra | 80 | 32 | 1.57 | 2.25 | 1.43 |
| Apple M4 | 8 | 8 | 0.87 | 0.90 | 1.03 |
| Apple M4 | 8 | 10 | 0.78 | 0.72 | 0.92 |
| Apple M4 | 10 | 10 | 0.86 | 0.90 | 1.04 |
| Apple M4 Pro | 16 | 12 | 1.16 | 1.20 | 1.03 |
| Apple M4 Pro | 20 | 14 | 1.15 | 1.29 | 1.12 |
| Apple M4 Max | 32 | 14 | 1.65 | 2.06 | 1.25 |
| Apple M4 Max | 40 | 16 | 1.76 | 2.25 | 1.28 |
| Apple M5 | 8 | 10 | 0.95 | 0.72 | 0.76 |
| Apple M5 | 10 | 10 | 1.02 | 0.90 | 0.88 |
| Apple M5 Pro | 16 | 15 | 1.04 | 0.96 | 0.92 |
| Apple M5 Pro | 20 | 18 | 1.00 | 1.00 | 1.00 |
| Apple M5 Max | 32 | 18 | 1.38 | 1.60 | 1.16 |
| Apple M5 Max | 40 | 18 | 1.70 | 2.00 | 1.18 |
| Apple M5 Ultra | 64 | 36 | 1.23 | 1.60 | 1.30 |
| Apple M5 Ultra | 80 | 36 | 1.54 | 2.00 | 1.30 |
| Apple A18 Pro | 5 | 6 | 1.06 | 0.75 | 0.71 |

Ratio from 0.71 to 1.86.

The fallback overrates the GPUs of the big chips most (the M1 Ultra 1.86x),
whose CPUs gain less from their many cores than their GPUs do, and underrates
the A18 Pro's and the M5's. A margin of 2 covers the worst case.

## 6. A fault in the SVD harness, found on the way

The first estimated `svdvals` rows were worse than the anchor's row copied
as it is (1.08-1.12x against 1.02-1.04x on geometric mean), which a refit on
the very timings it is scored on should never be. The cause was in
`tune_svd.py`'s stage 2b, the GPU-or-CPU rule for singular values alone: it
scored candidate rules only at the points where `gk` alone was the GPU's
choice, leaving out the batches shared with the CPU. On the M5 Pro, which
shares from 1024 matrices, that still left batches up to 512 to fit on; a
weaker simulated GPU shares from 64, which left only batches of 1, 4 and 16,
where the CPU always wins, every rule tied, and the tie-break picked the most
GPU-eager rule, which then sent batches of 64-256 to `gk` shared with the CPU
at up to 3.6x the CPU's time. The fix scores the rule at the shared points as
well (as `values_choice` already routed them). It changes one field of the
M5 Pro's measured row, `values_gpu_max_l` from 56 to 256: tall batches of
singular values alone, 64-256 rows of 8-32 columns in 1024 and more, now go to
`gk` shared with the CPU, 1.15-1.9x faster than the CPU path, and one shape
(1024 of 64x16) 0.82x; on the 273 `svdvals` points of the run, 1.029x the best
on geometric mean against 1.043x, 5.2% of points more than 1.25x off against
8.4%.

## 7. Limits, and what next

- The estimates are only as good as the anchor and the benchmarks: Geekbench
  measures image processing and mixed CPU work, not linear algebra, and its
  scores move between runs and macOS versions.
- A stale measured row still takes precedence over an estimate, except one
  measured before 2.9.0, when the CPU path ran on one core and every
  GPU-or-CPU boundary sat elsewhere (`MIN_EPOCHS` in `tuning/kernels.py`).
  The M1's QR and eigh rows were such, and are no longer used: the M1 is
  estimated like any other Mac until it is measured again.
- Every new chip needs a line in `tuning/chip_specs.py` (the core-count
  fallback covers it until then, with the larger margin).
- The cure remains a measurement: `python3 tuning/run.py` on the Mac
  ([how to contribute](../../CONTRIBUTING.md)), which replaces the estimate and
  becomes an anchor for its neighbours.
