# Symmetric eigensolver (`eigh`)

```cpp
#include <metal_linalg/eigh.h>

// a: MLX array of shape [N, N] or [..., N, N], real symmetric (real: float32, or cast to it; complex input throws)
// Returns: {w, V} with w [..., N] ascending and V [..., N, N], A = V diag(w) V^T
auto [w, V] = metal_linalg::eigh_accelerated(a);        // reads the lower triangle
auto [w2, V2] = metal_linalg::eigh_accelerated(a, "U"); // or the upper
array w3 = metal_linalg::eigvalsh_accelerated(a);       // eigenvalues only: less work (see Routing)
```

Same contract as `mlx::core::linalg::eigh`, which as of MLX 0.31 refuses to
run on the GPU (`"This op is not yet supported on the GPU"`). Only the
requested triangle is read, so the input need not be exactly symmetric. Batch
dimensions are arbitrary. Non-finite input yields NaN output rather than an
exception, as LAPACK does.

**Routing.** Four Metal backends cover the size range (see
[Dispatch](#dispatch)), but Accelerate's LAPACK on the CPU is quick (a single
512×512 in 18 ms on an M1), and since 2.9.0 a batch is spread over every CPU
core, so the public functions run on the GPU only where it was measured
faster, and call LAPACK (Accelerate) on the CPU otherwise: on an M5 Pro for
large batches of matrices up to N = 48 (`batch * N >= 8192`), and for one or
two matrices from N = 1536 on the `tridiag` backend. The boundary is part of the
per-device policy (see [Tuning](#tuning));
on the CPU, eigenvectors come from `ssyevd`, and eigenvalues alone
(`eigvalsh`) from N = 128 come from `ssyevd_2stage`, the two-stage
reduction (dense to band in matrix-matrix products, then band to
tridiagonal), which on an M5 Pro is 1.2x faster at N = 1024, 3.8x at 4096
and 5.7x at 8192 (2.6 s rather than 14.8 s for one 8192×8192). LAPACK's
two-stage driver does not return eigenvectors. On macOS 14, Accelerate's
`ssyevd_2stage` returns wrong eigenvalues (off by 1.5-7% of the largest),
so there `ssyevd` is used at every N; on later releases the two-stage
driver is used once it has matched `ssyevd` on a fixed matrix, checked once
per process.
Because the CPU's method differs, `eigvalsh` has its own GPU-or-CPU boundary
(the policy's `values_gpu_*` fields), measured from eigenvalue-only timings;
a device measured before it existed routes `eigvalsh` as `eigh`.
`eigh_backend(n, batch)` and `eigvalsh_backend(n, batch)` report what a given
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

## Backend 3: tridiagonalization on the GPU (`Eigh_Tridiag.metal`)

For one large matrix both Jacobi backends lose to LAPACK, whose `ssyevd`
spends most of its time in two places: reducing $A$ to tridiagonal form
$A = Q T Q^T$ (`ssytrd`), half of which is a symmetric matrix-vector product
per column and so bound by memory bandwidth, and forming $V = Q Z$ from the
eigenvectors $Z$ of $T$, which is matrix products. The `tridiag` backend
(`src/eigh_tridiag.mm`) keeps LAPACK's method and moves those two steps to the
GPU, leaving the tridiagonal eigenproblem, $O(N^2)$ for eigenvalues and cheap
next to the rest for eigenvectors, to LAPACK on the CPU:

1. **Reduction**, blocked `ssytrd` (lower) in panels of 32 columns, as
   [Dongarra, Hammarling and Sorensen](https://doi.org/10.1016/0377-0427(89)90367-1)
   block it and LAPACK's `slatrd` implements it, every step on the GPU. Per
   column: the panel's earlier reflectors applied to the column, the
   Householder vector (norm scaled by the largest entry, as `slarfg`), the
   product of the trailing matrix with it, reading the lower triangle only in
   64 x 64 tiles, and `slatrd`'s corrections. Per panel: the rank-64 update of
   the trailing matrix as one MPS GEMM. The panels' command buffers are queued
   back to back and the host waits once per matrix: a GPU round trip costs
   about 0.13 ms, and one per column, as a CPU-driven panel needs, cost more
   than the whole reduction below $N \approx 3000$. The last 33 columns or
   fewer are reduced by LAPACK.
2. **Tridiagonal eigenproblem**: LAPACK `sstedc` (eigenvectors) or `ssterf`
   (eigenvalues), on the CPU.
3. **Back-transformation**: `ssytrd`'s reflectors applied to $Z$ 128 at a
   time, each block as $I - V T V^T$ (`slarft`), three MPS GEMMs; the next
   block's $V$ and $T$ are built on the CPU while the GPU applies this one.

The split is that of hybrid CPU/GPU libraries such as MAGMA
([Tomov, Nath and Dongarra](https://doi.org/10.1016/j.parco.2010.06.001)),
except that the panel, which they factor on the CPU, stays on the GPU here:
on Apple Silicon the round trip, not the panel's arithmetic, is what costs.

Each matrix is scaled by a power of two first (exact), so magnitudes whose
products over- or underflow float32 work as on the CPU, and only the requested
triangle is read. A batch is pipelined over two workspace slots (since 2.11.0):
while the CPU solves one matrix's tridiagonal problem, the GPU reduces the
next, and while the GPU back-transforms one, the CPU solves the next. On an M5
Pro, per matrix of 2048×2048 with eigenvectors, that is 117 ms alone, 84 ms in
a batch of 4 and 79 ms in a batch of 8 (1.48x); the GPU then stays ahead of the
CPU path up to batches of 8 at that size, where before it was ahead only up to
2. Because the CPU path spreads a batch over every core, the backend still
wins only for a lone matrix or a few, and the policy caps the batch
(`tridiag_max_batch`). With eigenvectors, on an M5 Pro, one $N \times N$
(eigh, then eigvalsh, against the CPU path):

| $N$ | eigh: CPU | tridiag | speedup | eigvalsh: CPU | tridiag | speedup |
|---|---|---|---|---|---|---|
| 1024 | 0.040 s | 0.035 s | 1.14x | 0.018 s | 0.027 s | 0.66x |
| 2048 | 0.235 s | 0.119 s | 1.98x | 0.081 s | 0.088 s | 0.92x |
| 3072 | 0.759 s | 0.265 s | 2.86x | 0.222 s | 0.187 s | 1.18x |
| 4096 | 2.570 s | 0.532 s | 4.83x | 0.471 s | 0.403 s | 1.17x |
| 8192 | 18.68 s | 3.159 s | 5.91x | 2.606 s | 2.406 s | 1.08x |

With eigenvectors the gain grows with $N$, because the CPU's reduction
falls further behind memory bandwidth; for eigenvalues alone the CPU already
uses the two-stage reduction, and the GPU path gains only a little, from about
$N = 3000$. Accuracy matches LAPACK's: residual and orthogonality about
$10^{-6}$ at every size tested, eigenvalues within $3 \times 10^{-7}$ of
LAPACK's relative to $\|A\|_F$.

## Backend 4: tridiagonalization and QL in one threadgroup (`Eigh_QL.metal`)

The Jacobi kernels do several times the flops of LAPACK's method: about
$9N^3$ per sweep, over 7 sweeps at $N = 64$, against roughly $5$-$10N^3$ in
all for Householder tridiagonalization, implicit QL and the eigenvectors. Once
the CPU path spread a batch over every core (2.9.0), that difference decided
most batched calls in the CPU's favour, so the `ql` backend
(`src/eigh_ql.mm`) runs LAPACK's method on the GPU, one threadgroup per
matrix, the whole matrix in threadgroup memory:

1. **Load and scale**: the requested triangle, mirrored, scaled by a power of
   two as in the other kernels; a non-finite entry gives NaN output for that
   matrix and the `info` flag.
2. **Tridiagonalization** (`ssytd2`): thread $i$ owns row $i$ and keeps its
   whole row of the trailing matrix, so each column's product $p = \tau A v$
   and the symmetric rank-2 update are row-local; three barriers per column.
3. **Q**, formed in place from the reflectors by backward accumulation
   (`sorg2r`).
4. **Implicit QL** with shifts on $(d, e)$, the method of EISPACK's `tql2`.
   The objection that kept QL off the GPU, about $N^2$ dependent rotations
   each needing a barrier, holds only if they are applied one at a time. The
   QL iteration reads only $(d, e)$, never the eigenvectors, so one thread
   runs it and records a sweep's rotations, and then every thread applies the
   whole recorded sequence to its own row of $Z$: rows are independent, so a
   sweep costs one barrier. From $N = 33$ the iteration runs on a simdgroup of
   its own and computes the next sweep while the rows apply this one (9-13%
   faster from $N = 48$ on an M5 Pro; 4-11% slower at 16 to 32, where the extra
   simdgroup costs more in matrices per core than it saves).
5. **Output**: eigenvalues ascending by a rank sort, eigenvectors permuted
   to match.

The matrix in threadgroup memory bounds $N$ at 87 with 32 KB, and the memory
is sized for the $N$ of the call, which matters: the QL iteration is a chain
of dependent operations, so the kernel's speed is set by how many matrices
share a core, and that is set by threadgroup memory (sizing it for $N = 32$
rather than a fixed 64 made the same kernel 3x faster at $N = 32$). For the
same reason the kernel is built with fast math, unlike the Jacobi kernels: an
IEEE-mode (`precise::`) division or square root anywhere in it made the
compiler build the whole kernel that way, 1.4x slower. What needs the
accuracy, the Householder vectors and the shift, takes one Newton step after
the fast operation; each rotation comes from one reciprocal square root, and
the non-finite check reads the bits. The accuracy is that of the other
backends: across $N = 1 \ldots 87$, residual and orthogonality at most
$1.5 \times 10^{-6}$ relative to $\|A\|_F$ and eigenvalues within
$8 \times 10^{-7}$ of LAPACK's.

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

Inside a window of N, `[ql_min_n, ql_max_n]`, the `ql` backend (backend 4,
below) replaces whichever Jacobi backend the split would pick. The window is
fitted per device on top of the split (stage 1b of `tuning/tune_eigh.py`),
is clipped to the largest N the backend takes on the device (87 with 32 KB of
threadgroup memory), and is off (`ql_max_n = 0`) on a device without
measurements:

```
ql_min_n <= N <= ql_max_n  ->  backend 4, ql      (eigh and eigvalsh alike)
otherwise                  ->  the Jacobi split above
```

**Sharing a batch with the CPU** (since 2.11.0). From a batch of
`share_min_batch` (0: never), a batch that goes to `ql` is solved by the GPU
and the CPU path at once: the GPU takes chunks from the front of the batch,
cpu_threads() − 2 CPU workers take a few matrices at a time from the back with
Accelerate's threading off, and they meet wherever their speeds put them, so
no split has to be measured (`detail::share_batch` in
`src/metal_runtime.mm`). Two cores are left to the GPU's host work: with a
worker on every core the GPU's chunks took 5-7x their time alone. On an M5 Pro,
against the faster of the two alone: 1.37x for 4096 matrices of 16×16, 1.47x
for 256 of 48×48, 1.51x for 1024 of 64×64. Below a few hundred matrices the
threads cost more than they save, which is what the fitted threshold
(stage 1c of `tuning/tune_eigh.py`) says. `eigh_shares_batch(n, batch)` and
`eigvalsh_shares_batch` report it; `EIGH_SHARE_MIN_BATCH` overrides it.

All three split large batches across command buffers. macOS kills a
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

The M1 table above is from before 2.9.0, against MLX's CPU `eigh`, which
solves a batch one matrix at a time on one core.

On an Apple M5 Pro (20 GPU cores, 18 CPU cores) with 2.11.0, from the routing
sweep in [`results/apple-m5-pro-20gpu/20261003-c0878c/eigh/`](results/apple-m5-pro-20gpu/20261003-c0878c/eigh/):
min of two randomised passes, on mains, against the CPU path as the library
runs it, a batch spread over all 18 cores. Each cell is the speedup of the
fastest GPU backend over the CPU, with its time and name (`ql+CPU`: `ql`
sharing the batch with the CPU path, see [Dispatch](#dispatch)):

| N | batch 1 | batch 16 | batch 256 | batch 4096 |
|---|---|---|---|---|
| 4 | 0.01x (0.15 ms, ql) | 0.11x (0.16 ms, simd) | 0.66x (0.17 ms, tg) | **1.19x** (0.29 ms, ql) |
| 8 | 0.02x (0.20 ms, ql) | 0.22x (0.19 ms, simd) | 0.64x (0.20 ms, tg) | **1.36x** (0.56 ms, simd) |
| 16 | 0.04x (0.23 ms, tg) | 0.26x (0.24 ms, tg) | 0.84x (0.31 ms, ql) | **1.80x** (1.44 ms, ql+CPU) |
| 32 | 0.11x (0.33 ms, tg) | 0.39x (0.37 ms, tg) | **1.08x** (0.63 ms, ql) | **2.07x** (4.51 ms, ql+CPU) |
| 64 | 0.12x (1.25 ms, tg) | 0.22x (1.27 ms, tg) | 0.90x (2.90 ms, ql+CPU) | **1.58x** (23.4 ms, ql+CPU) |
| 128 | 0.09x (5.79 ms, block) | 0.12x (6.54 ms, block) | 0.25x (40.4 ms, block) | 0.22x (654.3 ms, block) |
| 256 | 0.20x (11.6 ms, block) | 0.17x (20.4 ms, block) | 0.14x (313.5 ms, block) | -- |
| 512 | 0.30x (29.9 ms, block) | 0.13x (124.4 ms, block) | -- | -- |
| 1024 | 0.40x (100.9 ms, block) | 0.10x (1.13 s, block) | -- | -- |

Against a CPU that uses its cores the GPU's region is small: large batches
of matrices up to N = 64, where `ql`, shared with the CPU from about 1024
matrices, is up to 2.1x faster than the CPU alone, and
one large matrix, which the `tridiag` backend takes (backend 3; 1.45x at
N = 1536, 1.9x at 2048, 4.6x at 4096). Without `ql` the GPU would win almost
nowhere: at 4096 matrices of 32×32 the whole-matrix Jacobi kernel takes
16.8 ms and the CPU 9.6 ms. `ql`, alone and sharing the batch with the CPU,
against the best Jacobi kernel and against the CPU:

| N | batch | ql | shared with the CPU | best Jacobi | CPU | Jacobi / ql | CPU / ql | CPU / shared |
|---|---|---|---|---|---|---|---|---|
| 16 | 4096 | 1.64 ms | 1.44 ms | 2.39 ms | 2.60 ms | 1.45x | 1.58x | 1.80x |
| 24 | 4096 | 3.21 ms | 2.65 ms | 7.50 ms | 5.78 ms | 2.34x | 1.80x | 2.18x |
| 32 | 256 | 0.63 ms | 0.77 ms | 1.61 ms | 0.69 ms | 2.54x | 1.08x | 0.89x |
| 32 | 4096 | 5.38 ms | 4.51 ms | 16.5 ms | 9.34 ms | 3.06x | 1.74x | 2.07x |
| 48 | 1024 | 4.84 ms | 3.93 ms | 15.2 ms | 5.79 ms | 3.14x | 1.20x | 1.47x |
| 48 | 4096 | 17.1 ms | 11.6 ms | 59.5 ms | 22.2 ms | 3.49x | 1.30x | 1.92x |
| 64 | 1024 | 11.6 ms | 6.81 ms | 24.9 ms | 10.1 ms | 2.15x | 0.87x | 1.48x |
| 64 | 4096 | 44.7 ms | 23.4 ms | 95.8 ms | 37.0 ms | 2.15x | 0.83x | 1.58x |

For a lone small matrix and small batches `ql` is slower than the
whole-matrix kernel (0.63x at 32×32 alone), whose many threads per matrix
shorten a lone matrix's critical path, but those calls go to the CPU, which
is 10-100x faster than either.

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

| GPU | cores | simd up to | block from | ql for | GPU iff | tridiag | status |
|---|---|---|---|---|---|---|---|
| Apple M1 | 8 | N = 8 | N = 96 | never | N <= 64 and batch * N >= 1024 | never | measured before 2.9.0 (incomplete) — see [`studies/eigh-routing-apple-m1.md`](studies/eigh-routing-apple-m1.md) |
| Apple M5 Pro | 20 | never | N = 96 | N = 12-64, shared with the CPU from batch 1024 | N <= 48 and batch * N >= 8192 (eigvalsh: batch * N >= 16384) | from N = 1536, batch <= 2 | measured — run [`20261003-c0878c`](results/apple-m5-pro-20gpu/20261003-c0878c/eigh/report.md) |
| anything else | — | N = 8 | N = 96 | never | N <= 64 and batch * N >= 1024 | never | **untuned default** |

The M5 Pro row is the first measured against the CPU path that spreads a
batch over every core (2.9.0). Against it the GPU keeps two regions: large
batches of matrices up to N = 48 (batch × N at least 8192, so 256 matrices of
32×32 or 1024 of 8×8), on the `ql` backend from N = 12, shared with the
CPU path from 1024 matrices (since 2.11.0), and one or two large matrices on
`tridiag`. Eigenvalues alone go to the GPU from twice the batch (batch × N at
least 16384), shared likewise; before 2.11.0 they never did, the CPU's
eigenvalue paths being faster than the GPU alone everywhere measured except
`tridiag` for one matrix from N = 3072. Before 2.9.0 the
same machine routed batches up to N = 1024 to the GPU, measured against one
CPU core ([study](studies/routing-apple-m5-pro.md)); against every core that
routing is 1.71x slower than the oracle on geometric mean, worst 14x, and
the new row 1.003x. The block crossover, 96, is unchanged, and simd mode
still never wins on the M5 Pro.

The GPU/CPU rule is `N <= gpu_max_n`, `batch * N >= gpu_min_batch_times_n` and
`batch >= gpu_min_batch`; the last is 1 (no minimum) on the M1. Eigenvalues
alone follow the same rule with `values_gpu_max_n`,
`values_gpu_min_batch_times_n` and `values_gpu_min_batch`, fitted on the same
sweep's eigenvalue-only timings (`<backend>_vals`); `values_gpu_min_batch = 0`
means "as for eigenvectors". The sweep reaches N = 2048 for lone matrices and
small batches, so `gpu_max_n` is a measured cap rather than the edge of the
grid.
`eigh_policy_source()` reports `default:untuned-device (<name>)` on any GPU
without a table entry, so an untuned device is visible rather than silent. The
default errs toward the CPU, which is the safe direction: the CPU path is
never catastrophic, so being untuned costs a missed GPU win, not a call routed
to a backend that takes seconds. A GPU with more cores than an M1 will want a
higher `gpu_max_n` than this, as the M5 Pro row shows.

**To measure another Mac**, run `python3 tuning/run.py`, which measures all
three decompositions in one go (about an hour and a half); see [`tuning.md`](tuning.md).
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
| `EIGH_VALUES_GPU_MAX_N`, `EIGH_VALUES_GPU_MIN_BATCH_TIMES_N`, `EIGH_VALUES_GPU_MIN_BATCH` | the GPU/CPU boundary for eigenvalues alone (`eigvalsh`) |
| `EIGH_TRIDIAG_MIN_N`, `EIGH_VALUES_TRIDIAG_MIN_N` | the tridiag backend instead of the CPU from this N (0: never) |
| `EIGH_TRIDIAG_MAX_BATCH`, `EIGH_VALUES_TRIDIAG_MAX_BATCH` | ... only for batches up to this (0: any) |
| `EIGH_QL_MIN_N`, `EIGH_QL_MAX_N` | the ql backend on the GPU for N in this window (`EIGH_QL_MAX_N=0`: never) |
| `METAL_LINALG_CPU_THREADS=<n>` | CPU threads a batch is spread over (default: every core; all three decompositions) |
| `EIGH_DEVICE=gpu` / `cpu` / `tridiag` | bypass the GPU/CPU boundary; `tridiag` forces that backend |
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

About 250 checks: every backend, and both modes of backend 1, across
$N = 1 \ldots 512$ (odd sizes, sizes straddling the 16-block and 32-group
boundaries, several thread and inner-sweep counts; for `ql` every simdgroup
boundary up to its limit, 87, and the switch to a chaser of its own at 33),
the `tridiag` backend to 1024, batched and 4-D inputs, a batch split over
many command buffers,
both triangles with junk in the other, transposed and unaligned views, integer
input, structured spectra (identity, zero, diagonal, repeated, $10^{-4}$ to
$10^4$, negative definite, rank one), scaling from $10^{-30}$ to $10^{37}$,
NaN input alone and inside a batch, and the error paths. The CPU path is
checked spread over every thread against one thread, with a NaN kept in its
own matrix. Every eigenvalue is also compared against LAPACK. The routing policy is tested without assuming
any device's values: each check installs the policy it needs, forces every
size onto each backend in turn through the public function, and restores the
device's own policy at the end.

## References

- G. H. Golub and C. F. Van Loan, [*Matrix Computations*](https://jhupbooks.press.jhu.edu/title/matrix-computations), 4th ed., 2013. §8.5 — Jacobi methods; Algorithm 8.5.1 (symmetric Schur decomposition of a 2×2), §8.5.8 (parallel Jacobi).
- R. P. Brent and F. T. Luk, ["The solution of singular-value and symmetric eigenvalue problems on multiprocessor arrays"](https://epubs.siam.org/doi/10.1137/0906007), *SIAM J. Sci. Stat. Comput.* 6(1), 1985 — the round-robin parallel ordering.
- H. Rutishauser, ["The Jacobi method for real symmetric matrices"](https://doi.org/10.1007/BF02165223), *Numerische Mathematik* 9, 1966 — the analytic diagonal update.
- J. Demmel and K. Veselić, ["Jacobi's method is more accurate than QR"](https://epubs.siam.org/doi/10.1137/0613074), *SIAM J. Matrix Anal. Appl.* 13(4), 1992.
- J. J. Dongarra, S. J. Hammarling and D. C. Sorensen, ["Block reduction of matrices to condensed forms for eigenvalue computations"](https://doi.org/10.1016/0377-0427(89)90367-1), *J. Comput. Appl. Math.* 27(1-2), 1989 — the blocked tridiagonalization (LAPACK's `ssytrd` and `slatrd`) the `tridiag` backend runs on the GPU.
- S. Tomov, R. Nath and J. Dongarra, ["Accelerating the reduction to upper Hessenberg, tridiagonal, and bidiagonal forms through hybrid GPU-based computing"](https://doi.org/10.1016/j.parco.2010.06.001), *Parallel Computing* 36(12), 2010 — the hybrid CPU/GPU split (MAGMA) the backend follows, with the panel moved to the GPU.
- C. H. Bischof, B. Lang and X. Sun, ["A framework for symmetric band reduction"](https://doi.org/10.1145/365723.365735), *ACM Trans. Math. Softw.* 26(4), 2000 — reducing a dense matrix to band form, then the band to tridiagonal: the two-stage reduction behind `eigvalsh`'s CPU path.
- A. Haidar, H. Ltaief and J. Dongarra, ["Parallel reduction to condensed forms for symmetric eigenvalue problems using aggregated fine-grained and memory-aware kernels"](https://doi.org/10.1145/2063384.2063394), SC '11, 2011 — the two-stage algorithm as [LAPACK 3.7.0](https://netlib.org/lapack/lapack-3.7.0.html) implements it (`ssyevd_2stage`), which this library calls through Accelerate.
- E. Ringoot, R. Alomairy, V. Churavy and A. Edelman, ["Performant unified GPU kernels for portable singular value computation across hardware and precision"](https://doi.org/10.1145/3754598.3754667), 2025 ([arXiv:2508.06339](https://arxiv.org/abs/2508.06339)), and E. Ringoot, R. Alomairy and A. Edelman, ["Accelerating bidiagonalization of banded matrices through memory-aware bulge-chasing on GPUs"](https://arxiv.org/abs/2510.12705), 2025 — two-stage reductions on GPUs, including Apple's; they prompted measuring the two-stage reduction on Apple Silicon, and the second's GPU-resident design is why the `tridiag` backend keeps its panel on the GPU. Their kernels are not used here.
- NVIDIA, [cuSOLVER `syevjBatched`](https://docs.nvidia.com/cuda/cusolver/index.html#cusolverdn-t-syevjbatch) — Jacobi as the production batched symmetric eigensolver on GPUs.
