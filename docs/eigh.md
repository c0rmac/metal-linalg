# Symmetric eigensolver (`eigh`)

```cpp
#include <metal_linalg/eigh.h>

// a: MLX array of shape [N, N] or [..., N, N], real symmetric (real: float32, or cast to it; complex input throws)
// Returns: {w, V} with w [..., N] ascending and V [..., N, N], A = V diag(w) V^T
auto [w, V] = metal_linalg::eigh_accelerated(a);        // reads the lower triangle
auto [w2, V2] = metal_linalg::eigh_accelerated(a, "U"); // or the upper
array w3 = metal_linalg::eigvalsh_accelerated(a);       // eigenvalues only, ~1/3 less work
```

Same contract as `mlx::core::linalg::eigh`, which as of MLX 0.31 refuses to
run on the GPU (`"This op is not yet supported on the GPU"`). Only the
requested triangle is read, so the input need not be exactly symmetric. Batch
dimensions are arbitrary. Non-finite input yields NaN output rather than an
exception, as LAPACK does.

**Routing.** Two Metal backends cover the size range (see
[Dispatch](#dispatch)), but Accelerate's LAPACK on the CPU is quick (a single
512×512 in 18 ms on an M1), so the public functions run on the GPU only where
it was measured faster, and call LAPACK's `ssyevd` (Accelerate) on the CPU
otherwise: on an M1 for
`N <= 64` with `batch * N >= 1024`, on an M5 Pro for `N <= 1024` with
`batch * N >= 512` and at least 16 matrices. The boundary is part of the
per-device policy (see [Tuning](#tuning));
`eigh_backend(n, batch)` reports what a given
problem will run on. `EIGH_DEVICE=gpu` or `cpu` forces a path; the
`metal_linalg::detail` entry points always run their kernel. See
[Performance](#performance) for the numbers behind the rule.

## The problem

For a real symmetric $A \in \mathbb{R}^{N \times N}$ there is an orthogonal
$V$ and a real diagonal $\Lambda$ with

$$A = V \Lambda V^T, \qquad V^T V = I .$$

The columns of $V$ are the eigenvectors and the diagonal of $\Lambda$ holds
the eigenvalues. This is the decomposition behind PCA (covariance matrices),
spectral clustering (graph Laplacians), Hessian analysis, and the SVD of a
rectangular $B$ via $B^T B$.

## Algorithm: cyclic Jacobi with a parallel ordering

LAPACK's `ssyev` reduces $A$ to tridiagonal form with Householder reflections
and then runs implicit QL. That is the right choice on a CPU but a poor fit
for one threadgroup per matrix: the QL phase applies roughly $N^2$ dependent
Givens rotations, each of which would be a barrier. The GPU choice, and what
cuSOLVER's `syevjBatched` uses for this same batched-small regime, is the
**Jacobi eigenvalue method**: repeatedly pick a pair $(p, q)$ and apply a
plane rotation $J$ chosen so that $(J^T A J)_{pq} = 0$.

**The rotation.** With $\theta = (a_{qq} - a_{pp}) / (2 a_{pq})$, the
smaller-angle root

$$t = \frac{\operatorname{sign}(\theta)}{|\theta| + \sqrt{1 + \theta^2}},
\qquad c = \frac{1}{\sqrt{1 + t^2}}, \qquad s = t c$$

zeros $a_{pq}$ and moves the new diagonal entries to
$a_{pp} - t\,a_{pq}$ and $a_{qq} + t\,a_{pq}$ (Golub & Van Loan, Algorithm
8.5.1). Choosing the smaller angle keeps each rotation close to the identity,
which is what makes the cyclic method converge quadratically.

**The parallel ordering.** Rotations on disjoint pairs commute, so $N/2$ of
them can be applied at once. The kernel uses the round-robin tournament of
Brent & Luk: element 0 stays fixed while the rest rotate one position per
round, and position $i$ is paired with position $N-1-i$. After $N-1$ rounds
every pair has met exactly once, which is one *sweep*. Each round is three
phases separated by barriers:

| phase | work | what each thread touches |
|---|---|---|
| 0 | rotation $(c, s)$ and new diagonal for each pair | 3 reads per pair |
| 1 | $A \leftarrow J^T A$: rotate rows $p, q$ | consecutive columns of two rows |
| 2 | $A \leftarrow A J$, $V \leftarrow V J$: rotate columns $p, q$ | consecutive pairs within one row |

So a sweep costs $3(N-1)$ barriers with $N^2/2$ independent work items in
each of phases 1 and 2, against $\sim 1.5 N^2$ barriers for tridiagonal QL.

**Keeping roundoff off the off-diagonal.** Phase 2 does not compute the
$2 \times 2$ block $(p,p), (p,q), (q,p), (q,q)$ by rotation. The rotated
$a_{pq}$ is a difference of $O(|a_{pp}|)$ terms that cancels only in exact
arithmetic, and leaving that residue on the off-diagonal puts a floor of
$\varepsilon |a_{pp}|$ under the off-norm that the stopping test can never
get below. The block is overwritten analytically with
$(a_{pp} - t a_{pq},\ 0,\ 0,\ a_{qq} + t a_{pq})$ instead (Rutishauser's
formulation), after which every off-diagonal element only ever mixes with
other off-diagonal elements and the off-norm converges quadratically to
roundoff *of itself*.

**Stopping.** After each sweep the off-diagonal Frobenius norm is compared to
$\text{tol} \cdot \|A\|_F$, with $\text{tol} = 10^{-7}$. $\|A\|_F$ is invariant
under the similarities, so it is computed once. Convergence takes 1 sweep for
$N=2$, 6-7 for $N \approx 64$ and 9 for $N = 512$; a matrix with sixteen-fold
repeated eigenvalues took 15.

**Scaling.** The working copy is scaled by a power of two so its largest
entry lies in $[0.5, 1)$, as `ssyev` does. Without this the norms overflow
float32 above $\sim 10^{19}$ and, worse because it is silent, underflow below
$\sim 10^{-19}$, where $\|A\|_F^2$ rounds to zero and any matrix looks
converged. The tests cover $10^{-30}$ through $10^{37}$.

**Output.** Eigenvalues are sorted ascending with a rank sort (each of $N$
threads counts how many eigenvalues precede its own; stable, deterministic,
and negligible next to the sweeps) and the columns of $V$ are permuted to
match, written into the buffer that held the working copy of $A$.

## Backend 1: whole-matrix kernel (`Eigh_Jacobi.metal`)

Each matrix is handled by a *team* of threads; the kernel is compiled two
ways via a function constant.

- **simd mode** — the team is one 32-lane simdgroup, several matrices per
  threadgroup, synchronised with `simdgroup_barrier` only. For tiny $N$ the
  three threadgroup barriers per round would otherwise be the whole cost.
- **threadgroup mode** — the team is a whole threadgroup of up to 1024
  threads, one matrix per threadgroup.

Only the rotation parameters live in threadgroup memory (20 bytes per pair),
so $N$ is not bounded by the 32 KB limit until $N \approx 3200$. The real
bound is that one matrix is one GPU core streaming the whole matrix through
device memory twice per round at two flops per element: a lone $512 \times
512$ takes a second, and eight of them run concurrently take 4.4 s because
their working sets no longer fit in cache. That is the `qr_unblocked`
situation, and the second backend is the `qr_streaming_amx` answer to it.

## Backend 2: block Jacobi (`Eigh_BlockJacobi.metal`)

The same method on $b \times b$ blocks, $b = 16$, so that one matrix is
spread over the grid and the rotations are applied as tile products. A block
round handles all $n_b/2$ disjoint block pairs $(P, Q)$ of the same round-robin
tournament, in three launches:

| launch | threadgroups | work |
|---|---|---|
| `bj_subproblem` | one per pair | $S = \begin{bmatrix} A_{PP} & A_{PQ} \\ A_{QP} & A_{QQ} \end{bmatrix}$ into threadgroup memory; one scalar Jacobi sweep on it, accumulating the $32 \times 32$ orthogonal $U$; $S \leftarrow U^T S U$ |
| `bj_update_rows` | one per (pair, 32-column group) | rows of blocks $P, Q$ $\leftarrow U^T \cdot$ rows |
| `bj_update_cols` | one per (pair, 32-row group) | columns $P, Q$ of $A$ and of $V$ $\leftarrow$ columns $\cdot\, U$ |

The subproblem runs exactly the phases of backend 1 (shared through
`eigh_jacobi_common.h`) on a $32 \times 32$ matrix in threadgroup memory,
where 32 is both $2b$ and the simdgroup width. The updates are
`simdgroup_matrix` 8×8 tile products with $U$ resident, so each element read
gets 64 flops instead of two. Rows and columns are separate launches for the
same reason backend 1 has two barriers: pair 1's row update and pair 2's
column update meet at the element (row of $P_1$, column of $P_2$).

**Inexact subproblems.** Each subproblem gets one scalar sweep, not a full
solve. One sweep on every block pair is a scalar cyclic sweep in a
block-cyclic ordering, for which cyclic Jacobi converges, and it keeps the
flop count near scalar Jacobi's; solving every subproblem fully costs several
times more of the latency-bound work for at most one fewer outer sweep
(table 5 in the tuning doc). Subproblems whose own off-norm is already under
their share of the convergence budget are skipped (`U = I`), so late sweeps
only pay for the pairs that still matter.

Convergence is checked on the host once per outer sweep from a norm kernel,
on the same criterion. $N$ is zero-padded to a multiple of 32; padded rows
never couple to real ones (a zero off-diagonal is never rotated) and are
dropped on output. The output sort caps $N$ at 4096.

## Dispatch

On the GPU, by the policy for this device (`metal_linalg::eigh_policy()`); on an M1:

```
N <= 8    ->  backend 1, simd mode         (tie with threadgroup mode; kept as the natural tiny-N design)
N <  96   ->  backend 1, threadgroup mode
N >= 96   ->  backend 2, block Jacobi
```

The crossover comes from the routing study, [`studies/eigh-routing-apple-m1.md`](studies/eigh-routing-apple-m1.md):
174 (N, batch) points, every backend timed twice in randomised order, and the
split scored against the best GPU backend at each point. 96 is the only value
within 0.5% of the best geometric-mean regret (1.024, worst 1.49x); 128 costs
1.045 with a 2.64x worst case. Block also wins at N = 64 once the batch
reaches 256, but a batch-dependent term did not clear held-out validation
(better in 89% of bootstrap resamples against a 95% bar), so it is supported
by the policy and switched off. The M5 Pro repeats both findings: block from
96 is again the only near-optimal crossover, and the batch term (block from 64
at batch 256) reached 93% against the same bar. Launch-parameter tuning for
each backend is in [`studies/eigh-launch-parameters-apple-m1.md`](studies/eigh-launch-parameters-apple-m1.md).

Both backends split large batches across command buffers. macOS kills a
command buffer that monopolises the GPU for more than a couple of seconds
("Impacting Interactivity"), so the host bounds each one with a conservative
cost model, never going below one matrix per core.

## Accuracy

From `tests/test_eigh.cpp`, relative to $\|A\|_F$, on Gaussian symmetric input:

| $N$ | $\|A V - V\Lambda\|_F$ | $\|V^T V - I\|_F / \sqrt{N}$ | $\max \lvert w - w_\text{LAPACK} \rvert$ | sweeps |
|---|---|---|---|---|
| 8 | 2.3e-07 | 2.9e-07 | 2.0e-07 | 4 |
| 64 | 1.1e-06 | 1.2e-06 | 1.9e-07 | 7 |
| 256 | 2.8e-06 | 3.9e-06 | 4.3e-07 | 9 |
| 512 | 5.2e-06 | 6.5e-06 | 4.8e-07 | 9 |

Eigenvalues agree with LAPACK (through MLX's CPU `eigvalsh`) to a few
float32 ulps at every size tested. The residual grows roughly as
$\sqrt{N}\,\varepsilon$, as expected for a backward-stable method.

## Performance

Apple M1, median of at least five runs, against MLX's CPU `eigh` (Accelerate
LAPACK). Each cell is the speedup of the *dispatched* GPU backend (backend 1
for N < 96, block for N >= 96; no row falls between) over the CPU, with the GPU time in
parentheses; bold is where the GPU wins. CPU timings on this machine vary by
up to 1.5x between runs, so treat ratios near 1 as ties.

| N | backend | batch 1 | batch 16 | batch 256 | batch 4096 |
|---|---|---|---|---|---|
| 4 | 1 | 0.19x (0.31 ms) | 0.40x (0.37 ms) | **2.44x** (0.56 ms) | **4.35x** (2.14 ms) |
| 8 | 1 | 0.24x (0.42 ms) | 0.49x (0.56 ms) | **5.26x** (0.68 ms) | **7.44x** (4.20 ms) |
| 16 | 1 | 0.10x (0.71 ms) | 0.50x (1.09 ms) | **3.03x** (2.59 ms) | **7.81x** (13.4 ms) |
| 32 | 1 | 0.15x (1.56 ms) | **1.34x** (1.87 ms) | **2.73x** (7.78 ms) | **3.46x** (94.2 ms) |
| 64 | 1 | 0.20x (4.05 ms) | **1.52x** (4.76 ms) | **1.67x** (52.2 ms) | **1.72x** (865 ms) |
| 128 | block | 0.08x (19.7 ms) | 0.52x (34.3 ms) | 1.05x (314 ms) | -- |
| 256 | block | 0.23x (34.4 ms) | 0.68x (150 ms) | 0.66x (2280 ms) | -- |
| 512 | block | 0.31x (92.9 ms) | 0.44x (1150 ms) | -- | -- |
| 1024 | block | 0.33x (585 ms) | -- | -- | -- |

The same on an Apple M5 Pro (20 GPU cores, 18 CPU cores), from the routing
sweep in [`results/apple-m5-pro-20gpu/20260930-27b6c2/eigh/`](results/apple-m5-pro-20gpu/20260930-27b6c2/eigh/):
min of two randomised passes, idle machine on mains, against the same thin CPU
path. The cell names the GPU backend that won at that point (`tg` is backend 1
in threadgroup mode; simd mode never won on this device).

| N | batch 1 | batch 16 | batch 256 | batch 4096 |
|---|---|---|---|---|
| 4 | 0.10x (0.25 ms, tg) | 0.18x (0.18 ms, tg) | **1.21x** (0.20 ms, tg) | **11.8x** (0.29 ms, tg) |
| 8 | 0.10x (0.19 ms, tg) | 0.31x (0.18 ms, tg) | **3.31x** (0.21 ms, tg) | **19.9x** (0.54 ms, tg) |
| 16 | 0.10x (0.25 ms, tg) | 0.67x (0.25 ms, tg) | **6.51x** (0.37 ms, tg) | **15.9x** (2.40 ms, tg) |
| 32 | 0.14x (0.34 ms, tg) | **1.34x** (0.38 ms, tg) | **5.53x** (1.60 ms, tg) | **8.58x** (16.5 ms, tg) |
| 64 | 0.09x (1.90 ms, tg) | **1.14x** (1.93 ms, tg) | **3.96x** (8.82 ms, block) | **5.56x** (101 ms, block) |
| 128 | 0.08x (6.42 ms, block) | **1.08x** (7.10 ms, block) | **2.87x** (42.5 ms, block) | **2.84x** (686 ms, block) |
| 256 | 0.19x (12.2 ms, block) | **1.68x** (20.8 ms, block) | **1.72x** (323 ms, block) | -- |
| 512 | 0.30x (30.2 ms, block) | **1.10x** (128 ms, block) | -- | -- |
| 1024 | 0.40x (100 ms, block) | GPU only (1150 ms) | -- | -- |

The shape of the result is the M1's, with the GPU's region much larger: the
block backend is 3x faster than on the M1 at N = 512 (30 ms against 93) and
wins from batch 16 up to N = 512, and small batched matrices reach 20x. A
single matrix is still the CPU's at every size, and stays so as far as it was
measured: 0.41x at N = 1536, 0.48x at 4096 (4.9 s against 2.4 s). Both
methods are O(N³) and Accelerate on 18 CPU cores has the better constant.

What the block backend changed, same run on the M1, whole-matrix kernel vs block:

| N | batch | whole-matrix | block | gain |
|---|---|---|---|---|
| 256 | 1 | 113 ms | 34 ms | 3.3x |
| 512 | 1 | 1108 ms | 93 ms | 11.9x |
| 512 | 16 | 12845 ms | 1150 ms | 11.2x |
| 1024 | 1 | (watchdog) | 585 ms | -- |

The residual is 1e-7 to 1e-5 relative throughout, 2e-5 at N = 1024 (see
Accuracy; the block backend's is about 1.5x the whole-matrix kernel's at the
same N). Regenerate with `./build/benchmark_eigh`; the full per-backend
columns are printed there.

Batch is what the GPU needs at small $N$: one matrix is one threadgroup, and
Accelerate's LAPACK on a performance core is quick. At large $N$ the block
backend is 9x faster than the whole-matrix kernel and is the design that
scales with GPU core count, since every launch is a grid of independent
threadgroups; but on eight cores it does not close the gap to LAPACK, which
does a lone $512 \times 512$ in 18 ms. Where the remaining time goes, for a
$512 \times 512$: 10 outer sweeps of 31 rounds, three launches each, about
110 µs per round, of which the subproblem launch (16 threadgroups, 93
barriers each) is roughly half. Two things would move it:

1. **Fewer, fatter subproblem launches.** $b = 32$ halves the round count
   and doubles tile-product efficiency at the cost of one subproblem per
   core (its 17 KB of threadgroup memory), which matters for batches.
2. **Exploiting symmetry in the updates**, which halves the tile products;
   the row update of the upper block triangle is the transpose of the column
   update of the lower one.

The general (non-symmetric) eigenproblem is not attempted. It needs
Hessenberg reduction, a shifted QR iteration with deflation, complex
eigenvalues and back-substitution for eigenvectors; the QR iteration is
inherently sequential and is a different project rather than an extension of
this one.

## Tuning

**The routing is hardware-specific.** Both halves of the decision move with
the device: the GPU/CPU boundary with the ratio of GPU to CPU throughput, the
block crossover with core count and launch latency. So the library ships a
table of measured policies, keyed on the Metal device name and GPU core count,
rather than constants:

| GPU | cores | simd up to | block from | GPU iff | status |
|---|---|---|---|---|---|
| Apple M1 | 8 | N = 8 | N = 96 | N <= 64 and batch * N >= 1024 | measured — see [`studies/eigh-routing-apple-m1.md`](studies/eigh-routing-apple-m1.md) |
| Apple M5 Pro | 20 | never | N = 96 | N <= 1024, batch * N >= 512 and batch >= 16 | measured — see [`studies/routing-apple-m5-pro.md`](studies/routing-apple-m5-pro.md) |
| anything else | — | N = 8 | N = 96 | N <= 64 and batch * N >= 1024 | **untuned default** |

The two measured devices show what moves and what does not. The block
crossover is 96 on both. The GPU/CPU boundary is not: on the M5 Pro the GPU
stays ahead of the CPU up to the largest N measured, 1024 (the M1's cap is
64), at up to 20x for batches of small matrices. What does not move is that a
lone matrix is faster on the CPU at every size, on both devices, up to 4096 on
the M5 Pro. The M1's product rule already keeps lone matrices off its GPU,
whose cap is 64; on the M5 Pro that needs a third constant, a minimum batch of
16, which is why the policy has one. Simd mode, which wins on the M1 up to
N = 8, never wins on the M5 Pro.

The GPU/CPU rule is `N <= gpu_max_n`, `batch * N >= gpu_min_batch_times_n` and
`batch >= gpu_min_batch`; the last is 1 (no minimum) on the M1.
`eigh_policy_source()` reports `default:untuned-device (<name>)` on any GPU
without a table entry, so an untuned device is visible rather than silent. The
default errs toward the CPU, which is the safe direction: the CPU path is
never catastrophic, so being untuned costs a missed GPU win, not a call routed
to a backend that takes seconds. A GPU with more cores than an M1 will want a
higher `gpu_max_n` than this, as the M5 Pro row shows.

**To measure another Mac**, run `python3 tuning/run.py`, which measures all
three decompositions in one go (about 40 minutes); see [`tuning.md`](tuning.md).
The eigensolver part works as follows.

The conditions matter more here than for QR, because one of the four backends
is the CPU and other jobs slow it most, which biases the routing toward the
GPU. The harness checks rather than trusts: it records load average, power
source and power mode at both ends of the sweep and times a probe point before
and after it. A busy machine, a probe that moved by more than 25%, or a single
pass (`--quick`, a five-minute smoke test of the pipeline) marks the report
"indicative only" and says not to paste its row.

It writes a report, the timings and the fitted row for `kTuned[]` in
`src/eigh.mm` into the submission's `eigh/` folder. Nothing in it assumes the
machine it was written on:

- The device name, core count and policy in effect are read from the binary
  (`sweep_eigh --policy`).
- A calibration point rescales the cost model that decides which points are
  too slow to measure, so a faster GPU is probed further out, where its
  crossovers will have moved.
- The candidate values of each constant are the N and batch values measured.
  If the GPU is still ahead of the CPU at the largest N on the grid the report
  says so, and `--max-n 1024` extends both the grid and the search.
- The decision is fitted in two stages, the GPU backend split first against
  the best GPU backend alone, then the CPU routing given the split, because on
  a GPU where the CPU wins above some N the block crossover is otherwise
  invisible to the fit.
- The refinements that failed on an M1 (a batch-dependent block crossover, a
  per-N CPU boundary) are re-tested on every device and adopted only if they
  beat the plain policy on a held-out half of the points in at least 95% of
  bootstrap resamples.

The M1 and M5 Pro runs are committed under
[`results/apple-m1-8gpu/`](results/apple-m1-8gpu/) and
[`results/apple-m5-pro-20gpu/`](results/apple-m5-pro-20gpu/) so a new run can
be diffed against them.

To override the policy without rebuilding, set the environment variables
below, or call `set_eigh_policy()`:

```cpp
auto p = metal_linalg::eigh_policy();
p.gpu_max_n = 128;
metal_linalg::set_eigh_policy(p);
```

**Launch parameters** (threads per matrix, matrices per threadgroup, inner
sweeps) are measured by

```sh
cmake --build build --target benchmark_eigh
./build/benchmark_eigh --tune        # or --tune 1..5 for one table
```

with the M1 tables in [`studies/eigh-launch-parameters-apple-m1.md`](studies/eigh-launch-parameters-apple-m1.md).
To probe another GPU without a rebuild:

| variable | effect |
|---|---|
| `EIGH_SIMD_MAX_N`, `EIGH_BLOCK_MIN_N` | the GPU backend split |
| `EIGH_BLOCK_MIN_N_BATCHED`, `EIGH_BLOCK_MIN_BATCH` | the batch-dependent block crossover (0 = off) |
| `EIGH_GPU_MAX_N`, `EIGH_GPU_MIN_BATCH_TIMES_N`, `EIGH_GPU_MIN_BATCH` | the GPU/CPU boundary |
| `EIGH_DEVICE=gpu` / `cpu` | bypass the GPU/CPU boundary |
| `EIGH_MODE=simd` / `threadgroup` | force the execution mode of backend 1 |
| `EIGH_INNER_SWEEPS=<k>` | scalar sweeps per block subproblem |
| `EIGH_CHUNK_MS=<ms>` | wall-time budget per command buffer |

`EighOptions` (in `include/metal_linalg/eigh.h`) exposes the same knobs programmatically, plus the
tolerance and sweep bound, through the `metal_linalg::detail` entry points, which
also return a per-matrix `info` word with the sweep count.

## Tests

```sh
cmake --build build --target test_eigh
./build/test_eigh          # or: ctest --test-dir build
```

135 checks: both backends and both modes of backend 1 across
$N = 1 \ldots 512$ (odd sizes, sizes straddling the 16-block and 32-group
boundaries, several thread and inner-sweep counts), batched and 4-D inputs,
both triangles with junk in the other, transposed and unaligned views, integer
input, structured spectra (identity, zero, diagonal, repeated, $10^{-4}$ to
$10^4$, negative definite, rank one), scaling from $10^{-30}$ to $10^{37}$,
NaN input alone and inside a batch, and the error paths. Every eigenvalue is
also compared against LAPACK. The routing policy is tested without assuming
any device's values: each check installs the policy it needs, forces every
size onto each backend in turn through the public function, and restores the
device's own policy at the end.

## References

- G. H. Golub and C. F. Van Loan, [*Matrix Computations*](https://jhupbooks.press.jhu.edu/title/matrix-computations), 4th ed., 2013. §8.5 — Jacobi methods; Algorithm 8.5.1 (symmetric Schur decomposition of a 2×2), §8.5.8 (parallel Jacobi).
- R. P. Brent and F. T. Luk, ["The solution of singular-value and symmetric eigenvalue problems on multiprocessor arrays"](https://epubs.siam.org/doi/10.1137/0906007), *SIAM J. Sci. Stat. Comput.* 6(1), 1985 — the round-robin parallel ordering.
- H. Rutishauser, ["The Jacobi method for real symmetric matrices"](https://doi.org/10.1007/BF02165223), *Numerische Mathematik* 9, 1966 — the analytic diagonal update.
- J. Demmel and K. Veselić, ["Jacobi's method is more accurate than QR"](https://epubs.siam.org/doi/10.1137/0613074), *SIAM J. Matrix Anal. Appl.* 13(4), 1992.
- NVIDIA, [cuSOLVER `syevjBatched`](https://docs.nvidia.com/cuda/cusolver/index.html#cusolverdn-t-syevjbatch) — Jacobi as the production batched symmetric eigensolver on GPUs.
