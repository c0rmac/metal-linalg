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

**Routing.** Five Metal backends cover the size range (see
[Dispatch](#dispatch)), but Accelerate's LAPACK on the CPU is quick (a single
512×512 in 18 ms on an M1), and since 2.9.0 a batch is spread over every CPU
core, so the public functions run on the GPU only where it was measured
faster, and call LAPACK (Accelerate) on the CPU otherwise: on an M5 Pro for
large batches of matrices up to N = 48 (`batch * N >= 16384`; up to 64 from
1024 matrices), for up to four matrices from N = 1024 on the `tridiag`
backend, and for the eigenvalues alone of one or two matrices from N = 1536
on `tridiag` and from 4096 on `band`, the two-stage reduction. The boundary is
part of the per-device policy (see [Tuning](#tuning));
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

   A column's steps are three dispatches (since 2.12.0; seven before), one
   per point where a whole vector must be done before the next step starts:
   the column's update, which also finishes the previous column's $W$ and
   leaves per-threadgroup partials of the column's norm; the product, whose
   every threadgroup forms the Householder vector from those partials itself,
   with the corrections' dot products in further threadgroups; and the sum of
   the product's tiles with the corrections. Each dispatch, even an empty one,
   costs the GPU about a microsecond, and a dependent one several more while
   the previous drains: on an M5 Pro, outside the product itself, a column
   cost 13 µs at $N = 2048$ and 20 µs at 4096, against the product's 10 and
   38. The update and the corrections run eight threads to a row, so that a
   few thousand rows still fill the GPU. With the matrix copied in on every
   core rather than one, eigenvalues alone are 1.3-1.7x faster at
   $N = 1024$-4096 and eigenvectors 1.2-1.5x. The Householder vector each
   threadgroup forms divides and takes square roots with the fast
   approximations and a Newton step (since 2.15.0): the shader file is built
   for IEEE arithmetic, and a kernel with any IEEE division in it runs all
   of its arithmetic that way, which cost eigenvalues alone 7-11%. Every sum
   over threadgroups is taken in a fixed order, so results do not depend on
   scheduling.
2. **Tridiagonal eigenproblem**: for eigenvectors, LAPACK `sstedc`'s divide
   and conquer on every core but two (`src/divide_conquer.cpp`, since
   2.15.0): the same tree and LAPACK's routines for the deflation, the
   secular equation's roots and the vectors, its leaves, merges and the
   large merges' loops spread over threads. On a 4096 tridiagonal from
   `ssytrd` 50 ms against `sstedc`'s 176 on one core, the eigenvalues bit for
   bit LAPACK's; results do not depend on the number of threads. Eigenvalues
   alone by bisection on the GPU from $N = 512$ (since 2.13.0; see
   [backend 5](#backend-5-eigenvalues-alone-in-two-stages-band)), else
   `ssterf`.
3. **Back-transformation**: `ssytrd`'s reflectors applied to $Z$ 128 at a
   time, each block as $I - V T V^T$, three MPS GEMMs; the next block's $V$
   and $T$ are built on the CPU while the GPU applies this one, $T$ from the
   Gram matrix $V^T V$ (one `ssyrk`, since 2.15.0; `slarft`'s matrix-vector
   products had made the CPU's side the slower).

The split is that of hybrid CPU/GPU libraries such as MAGMA
([Tomov, Nath and Dongarra](https://doi.org/10.1016/j.parco.2010.06.001)),
except that the panel, which they factor on the CPU, stays on the GPU here:
on Apple Silicon the round trip, not the panel's arithmetic, is what costs.

Each matrix is scaled by a power of two first (exact), so magnitudes whose
products over- or underflow float32 work as on the CPU, and only the requested
triangle is read. A batch is pipelined over two workspace slots (since 2.11.0):
while the CPU solves one matrix's tridiagonal problem, the GPU reduces the
next, and while the GPU back-transforms one, the CPU solves the next. On an M5
Pro, per matrix of 2048×2048 with eigenvectors, that is 90 ms alone, 55 ms in
a batch of 4 and 50 ms in a batch of 8 (1.80x; 117, 84 and 79 ms before the
reduction's dispatches were merged in 2.12.0); the GPU then stays ahead of the
CPU path up to batches of 8 at that size, where before 2.11.0 it was ahead only
up to 2. Because the CPU path spreads a batch over every core, the backend still
wins only for a lone matrix or a few, and the policy caps the batch
(`tridiag_max_batch`). With eigenvectors, on an M5 Pro, one $N \times N$
(eigh, then eigvalsh, against the CPU path; 2.15.0 measured side by side with
the CPU path, the median of `sweep_eigh`):

| $N$ | eigh: CPU | tridiag | speedup | eigvalsh: CPU | tridiag | speedup |
|---|---|---|---|---|---|---|
| 1024 | 0.041 s | 0.018 s | 2.30x | 0.018 s | 0.012 s | 1.49x |
| 2048 | 0.243 s | 0.055 s | 4.39x | 0.083 s | 0.036 s | 2.32x |
| 3072 | 0.730 s | 0.133 s | 5.49x | 0.209 s | 0.091 s | 2.30x |
| 4096 | 2.494 s | 0.306 s | 8.15x | 0.483 s | 0.213 s | 2.27x |
| 8192 | 18.29 s | 2.170 s | 8.43x | 2.644 s | 1.656 s | 1.60x |

With eigenvectors the gain grows with $N$, because the CPU's reduction
falls further behind memory bandwidth; for eigenvalues alone the CPU already
uses the two-stage reduction, and the GPU path gains less: 1.5-2.3x since
2.13.0's bisection (1.1-1.6x in 2.12.0, and before that about 1.2x from
$N = 3000$). From $N = 4096$ eigenvalues alone go to the `band` backend
instead (below). Accuracy matches LAPACK's: residual and orthogonality about
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

## Backend 5: eigenvalues alone in two stages (`band`)

For eigenvalues alone the `tridiag` backend's reduction is held to the speed
of memory: a symmetric matrix-vector product a column, reading the whole
trailing matrix each time. The CPU path already avoids that with LAPACK's
two-stage `ssyevd_2stage`, which is why `tridiag` led it by only 1.1-1.6x for
`eigvalsh`. The `band` backend (`src/eigh_band.mm`, new in 2.13.0) reduces in
the same two stages ([Haidar, Ltaief and Dongarra](https://doi.org/10.1145/2063384.2063394)),
the first on the GPU and the second on every CPU core:

1. **To a band** of width $b$ (the policy's `values_band_width`, 16 by
   default) on the GPU (`band_reduce_symmetric` in `src/band_reduce.mm`), $b$
   columns a block: the QR of the panel below the diagonal block,
   $H = I - V T V^T$, its $R$ left in place as the band; then both sides of
   the trailing matrix $A_{22}$ at once, $X = A_{22} V T$ (an MPS product),
   $Y = X - \tfrac12 V (T^T V^T X)$ (two small kernels), and
   $A_{22} \mathrel{-}= V Y^T + Y V^T$ with $[V\ Y]$ and $[Y\ V]^T$ on
   $A_{22}$'s lower 64 x 64 tiles, each off-diagonal tile's transpose written
   over its mirror (`sb_update`, since 2.15.0; MPS has no symmetric rank-$2b$
   update, and the whole matrix's took a third longer). The trailing matrix
   is read twice and written one and a half times a block, where the
   one-stage reduction reads it once a column. The panels are the SVD's
   `band` kernels ([svd.md](svd.md)): up to 128 rows in one simdgroup,
   taller by TSQR with the stacked $R$'s factored as a tree of triangle pairs
   and the Householder vectors rebuilt. The last columns, fewer than $3b$,
   are LAPACK's `ssytrd_sy2sb`.
2. **To tridiagonal** on the CPU (`band_to_tridiagonal` in
   `src/band_chase.cpp`), by Householder bulge chasing, LAPACK's
   `ssytrd_sb2st` kernels: sweep $s$ annihilates column $s$ below the
   subdiagonal and chases the bulge it makes down the band, a block of $b$ at
   a time. Sweeps overlap ([Lang](https://doi.org/10.1137/0914078)): sweep
   $s$ may run its $t$-th step once sweep $s - 1$ has finished its
   $(t + 2)$-th, the blocks they touch being one apart, so the sweeps run
   pipelined on all the cores but two (left to the GPU's host work), each
   thread spinning on the previous sweep's count of finished steps.
   Accelerate's `ssytrd_sb2st` runs on one core: 112 ms for a 4096 band of
   width 16, against 39 ms here.
3. **The eigenvalues** of the tridiagonal by bisection on the GPU
   (`sturm_bisect` in `shaders/Eigh_Tridiag.metal`): a thread an eigenvalue,
   each counting the Sturm sequence's sign changes
   ([Barth, Martin and Wilkinson](https://doi.org/10.1007/BF02162154)) with
   LAPACK's `pivmin` guard ([Demmel, Dhillon and Ren](http://www.emis.de/journals/ETNA/vol.3.1995/pp116-149.dir/pp116-149.html))
   for a fixed number of halvings from the Gershgorin interval, the
   tridiagonal staged in threadgroup memory 1024 entries at a time. LAPACK's
   `ssterf` is sequential, 79 ms at 4096 and 305 at 8192; bisection takes 6
   and 16, accurate to a few float32 ulps of $\|T\|$. Below $N = 512$ `ssterf`
   is the faster and is used. The `tridiag` backend's eigenvalue path uses
   the same bisection since 2.13.0.

A batch is pipelined over two slots, as in `tridiag`: the CPU chases and
solves one matrix while the GPU reduces the next. Each matrix is scaled by a
power of two first, and a non-finite matrix gives NaN eigenvalues and the
`info` flag without being reduced. The band width is narrowed where a
panel would not fit the kernels ($N b \le 131072$: 16 up to $N = 8192$, 8 up
to 16384) and the backend gives way to `tridiag` beyond.

On an M5 Pro, one $N \times N$, eigenvalues alone, against `tridiag` (with
bisection) and the CPU path (2.15.0, measured side by side, the median of
`sweep_eigh`):

| $N$ | CPU | tridiag | band | band / CPU |
|---|---|---|---|---|
| 1024 | 0.018 s | 0.012 s | 0.014 s | 1.28x |
| 2048 | 0.083 s | 0.036 s | 0.039 s | 2.11x |
| 3072 | 0.209 s | 0.091 s | 0.077 s | 2.71x |
| 4096 | 0.483 s | 0.213 s | 0.140 s | 3.44x |
| 8192 | 2.644 s | 1.656 s | 0.726 s | 3.64x |

From about 3072 the band reduction's matrix products beat the one-stage
reduction's bandwidth limit; at 8192 the backend is 2.3x `tridiag`, and the
CPU's own two-stage driver takes 3.6x as long. Below that, a block's panel
factorization, a chain of dependent steps whose latency does not shrink with
$N$, costs more than the products save. At 4096 the chase takes 35 ms,
bisection 6, and the GPU's stage, with the matrix's copy in, the rest. Eigenvalues are
within $6 \times 10^{-6}$ of LAPACK's `ssyevd` relative to the largest at
the sizes tested (`tridiag`'s within $3 \times 10^{-7}$). The
routing sweep fits `values_band_min_n`, from which $N$ eigenvalues alone use
it: 4096 on the M5 Pro, where at 3072 the backend is 7% ahead of `tridiag`,
inside the fit's 0.5% tolerance on the geometric mean, whose tie-break takes
the higher threshold. How the stages were built and measured is in [the
two-stage study](studies/two-stage-apple-m5-pro.md).

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

Where the GPU/CPU rule (below) says CPU, one large matrix or a few still go to
the GPU's LAPACK-style backends:

```
up to tridiag_max_batch matrices (eigvalsh: values_tridiag_max_batch; 0 = any):
  eigvalsh: band     iff  N >= values_band_min_n      (0 = never)
  tridiag            iff  N >= tridiag_min_n          (eigvalsh: values_tridiag_min_n; 0 = never)
otherwise the CPU path
```

**The large-batch clause** (since 2.12.0). The GPU-or-CPU rule is a product,
`N <= gpu_max_n` and `batch * N >= gpu_min_batch_times_n`, and shared with the
CPU (below) the GPU also wins large batches of matrices just above
`gpu_max_n`, which the product cannot take without also taking their small
batches, which the CPU wins. So the rule has a second clause: the GPU also for
N above `gpu_max_n` up to `gpu_big_batch_max_n` in a batch of at least
`gpu_big_batch_min` (0: never; `gpu_max_n = 0` is still never the GPU). It is
fitted together with the product rule (stage 2 of `tuning/tune_eigh.py`), and applies to
eigenvalues alone only while they follow the eigenvectors' rule
(`values_gpu_min_batch = 0`). `EIGH_GPU_BIG_BATCH_MAX_N` and
`EIGH_GPU_BIG_BATCH_MIN` override it.

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

On an Apple M5 Pro (20 GPU cores, 18 CPU cores) with 2.13.0, from the routing
sweep in [`results/apple-m5-pro-20gpu/20261004-06bc11/eigh/`](results/apple-m5-pro-20gpu/20261004-06bc11/eigh/):
min of two randomised passes, on mains, against the CPU path as the library
runs it, a batch spread over all 18 cores. Each cell is the speedup of the
fastest GPU backend over the CPU, with its time and name (`ql+CPU`: `ql`
sharing the batch with the CPU path, see [Dispatch](#dispatch)):

| N | batch 1 | batch 16 | batch 256 | batch 4096 |
|---|---|---|---|---|
| 4 | 0.01x (0.17 ms, ql) | 0.11x (0.16 ms, simd) | 0.49x (0.17 ms, ql) | **1.19x** (0.29 ms, ql) |
| 8 | 0.03x (0.16 ms, tg) | 0.23x (0.17 ms, tg) | 0.67x (0.20 ms, simd) | **1.42x** (0.56 ms, simd) |
| 16 | 0.05x (0.23 ms, tg) | 0.30x (0.24 ms, tg) | 0.86x (0.30 ms, ql) | **1.77x** (1.45 ms, ql+CPU) |
| 32 | 0.11x (0.33 ms, tg) | 0.38x (0.37 ms, tg) | **1.06x** (0.64 ms, ql) | **2.06x** (4.53 ms, ql+CPU) |
| 64 | 0.12x (1.21 ms, tg) | 0.23x (1.26 ms, tg) | 0.89x (2.89 ms, ql+CPU) | **1.68x** (23.5 ms, ql+CPU) |
| 128 | 0.08x (5.85 ms, block) | 0.12x (6.57 ms, block) | 0.24x (42.5 ms, block) | 0.22x (670.4 ms, block) |
| 256 | 0.19x (11.7 ms, block) | 0.19x (20.7 ms, block) | 0.14x (318.2 ms, block) | -- |
| 512 | 0.30x (30.1 ms, block) | 0.12x (132.4 ms, block) | -- | -- |
| 1024 | 0.39x (101.9 ms, block) | 0.11x (1.14 s, block) | -- | -- |

Against a CPU that uses its cores the GPU's region is small: large batches
of matrices up to N = 64, where `ql`, shared with the CPU from about 1024
matrices, is up to 2.2x faster than the CPU alone, and
one large matrix, which the `tridiag` backend takes (backend 3; 3.04x at
N = 1536, 4.39x at 2048, 8.15x at 4096), and for its eigenvalues alone
`tridiag` or, from 4096, `band` (backend 5; 3.44x at 4096). Without `ql` the
GPU would win almost nowhere: at 4096 matrices of 32×32 the whole-matrix
Jacobi kernel takes 16.4 ms and the CPU 9.4 ms. `ql`, alone and sharing the batch with the CPU,
against the best Jacobi kernel and against the CPU:

| N | batch | ql | shared with the CPU | best Jacobi | CPU | Jacobi / ql | CPU / ql | CPU / shared |
|---|---|---|---|---|---|---|---|---|
| 16 | 4096 | 1.64 ms | 1.45 ms | 2.40 ms | 2.57 ms | 1.46x | 1.57x | 1.77x |
| 24 | 4096 | 3.20 ms | 2.66 ms | 7.47 ms | 5.82 ms | 2.34x | 1.82x | 2.19x |
| 32 | 256 | 0.64 ms | 0.79 ms | 1.61 ms | 0.67 ms | 2.53x | 1.06x | 0.85x |
| 32 | 4096 | 5.39 ms | 4.53 ms | 16.4 ms | 9.35 ms | 3.05x | 1.73x | 2.06x |
| 48 | 1024 | 4.86 ms | 3.86 ms | 15.5 ms | 6.11 ms | 3.20x | 1.26x | 1.58x |
| 48 | 4096 | 17.3 ms | 12.0 ms | 63.2 ms | 23.2 ms | 3.65x | 1.34x | 1.93x |
| 64 | 1024 | 11.8 ms | 6.89 ms | 25.8 ms | 10.2 ms | 2.17x | 0.86x | 1.48x |
| 64 | 4096 | 45.8 ms | 23.5 ms | 98.7 ms | 39.6 ms | 2.16x | 0.87x | 1.68x |

For a lone small matrix and small batches `ql` is slower than the
whole-matrix kernel (0.62x at 32×32 alone), whose many threads per matrix
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
| Apple M1 | 8 | — | — | — | — | — | measured before 2.9.0, out of date and no longer used since 2.14.0: estimated like any unmeasured Mac (the old row's study: [`studies/eigh-routing-apple-m1.md`](studies/eigh-routing-apple-m1.md)) |
| Apple M5 Pro | 20 | never | N = 96 | N = 12-64, shared with the CPU from batch 1024 | N <= 48 and batch * N >= 16384, or N = 49-64 in batches of 1024+ (eigvalsh: N <= 48 and batch * N >= 16384) | from N = 1024, batch <= 4 (eigvalsh: from 1536, batch <= 2, and `band` from 4096) | measured — run [`20261004-06bc11`](results/apple-m5-pro-20gpu/20261004-06bc11/eigh/report.md) |
| anything else | — | estimated | estimated | estimated | estimated | estimated | **estimated** from the M5 Pro's timings ([how](tuning.md#macs-nobody-has-measured)) |

The M5 Pro row is the first measured against the CPU path that spreads a
batch over every core (2.9.0). Against it the GPU keeps two regions: large
batches of matrices up to N = 48 (batch × N at least 16384, so 512 matrices of
32×32 or 2048 of 8×8), on the `ql` backend from N = 12, shared with the
CPU path from 1024 matrices (since 2.11.0), and up to four large matrices on
`tridiag` from N = 1024 (since 2.12.0; 1536 and two before). Since 2.12.0
batches of 1024 and more go to the GPU, shared, up to N = 64 (the large-batch
clause), which the product rule cannot reach without also taking small batches
of N = 49-64 that the CPU wins: on the 205 points of the 2.12.0 run the row
scored 1.0094 geometric-mean regret, worst 1.38x, against 1.0157 for the
product rule alone and 1.0112, worst 1.59x, for the row it replaced, and on the
2.13.0 run 1.0095, worst 1.35x. Eigenvalues alone go to
the GPU at the same batches, shared likewise, to `tridiag` from N = 1536 and
to `band` from 4096 (since 2.13.0);
before 2.11.0 they never went to the GPU, the CPU's eigenvalue paths being
faster than the GPU alone everywhere measured except `tridiag` for one matrix
from N = 3072. Before 2.9.0 the
same machine routed batches up to N = 1024 to the GPU, measured against one
CPU core ([study](studies/routing-apple-m5-pro.md)); against every core that
routing is 1.71x slower than the oracle on geometric mean, worst 14x, and
the new row 1.003x. The block crossover, 96, is unchanged; simd mode, which
took N <= 8 in the 2.12.0 run, is off in the 2.13.0 one (1.0186 against
1.0277 with it, on the GPU's choices alone; up to N = 4 is within 0.5%).

The GPU/CPU rule is `N <= gpu_max_n`, `batch * N >= gpu_min_batch_times_n` and
`batch >= gpu_min_batch`; the last is 1 (no minimum) on the M1. Eigenvalues
alone follow the same rule with `values_gpu_max_n`,
`values_gpu_min_batch_times_n` and `values_gpu_min_batch`, fitted on the same
sweep's eigenvalue-only timings (`<backend>_vals`); `values_gpu_min_batch = 0`
means "as for eigenvectors". The sweep reaches N = 2048 for lone matrices and
small batches, so `gpu_max_n` is a measured cap rather than the edge of the
grid.
On any GPU without a table entry `eigh_policy_source()` reports
`estimated:<name> (from Apple M5 Pro, ...)`: the M5 Pro's timings refitted
for that GPU against its CPU ([how](tuning.md#macs-nobody-has-measured)), so an unmeasured device is visible
rather than silent, and routed close to its best. The `EighPolicy` defaults
(the M1's values from before the CPU path used every core, with no `ql`,
`tridiag` or sharing) remain only for a Mac with nothing to estimate from,
reported as `default:untuned-device (<name>)`.

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
| `EIGH_VALUES_BAND_MIN_N` | the band backend for eigenvalues alone from this N, within the same batch cap (0: never) |
| `EIGH_VALUES_BAND_WIDTH` | the band backend's band width as a policy field (`values_band_width`: 8, 16 or 32; 0 is 16) |
| `EIGH_BAND_WIDTH=8` / `16` / `32` | the band backend's band width where the policy leaves it 0 (default 16) |
| `EIGH_QL_MIN_N`, `EIGH_QL_MAX_N` | the ql backend on the GPU for N in this window (`EIGH_QL_MAX_N=0`: never) |
| `METAL_LINALG_CPU_THREADS=<n>` | CPU threads a batch is spread over (default: every core; all three decompositions) |
| `EIGH_DEVICE=gpu` / `cpu` / `tridiag` / `band` | bypass the GPU/CPU boundary; `tridiag` forces that backend, `band` that backend for eigenvalues alone (`tridiag` with eigenvectors) |
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

319 checks: every backend, and both modes of backend 1, across
$N = 1 \ldots 512$ (odd sizes, sizes straddling the 16-block and 32-group
boundaries, several thread and inner-sweep counts; for `ql` every simdgroup
boundary up to its limit, 87, and the switch to a chaser of its own at 33),
the `tridiag` backend to 1100, `band` at each band width from 1×1 to
1100×1100 (either side of the one-simdgroup panel, the TSQR leaf and the
switch to bisection), batched and 4-D inputs, a batch split over
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
- B. Lang, ["A parallel algorithm for reducing symmetric banded matrices to tridiagonal form"](https://doi.org/10.1137/0914078), *SIAM J. Sci. Comput.* 14(6), 1993 — Householder bulge chasing with the sweeps pipelined, the `band` backend's second stage.
- W. Barth, R. S. Martin and J. H. Wilkinson, ["Calculation of the eigenvalues of a symmetric tridiagonal matrix by the method of bisection"](https://doi.org/10.1007/BF02162154), *Numerische Mathematik* 9, 1967 — Sturm-sequence bisection, which the `band` and `tridiag` backends run on the GPU for eigenvalues alone.
- J. W. Demmel, I. Dhillon and H. Ren, ["On the correctness of some bisection-like parallel eigenvalue algorithms in floating point arithmetic"](http://www.emis.de/journals/ETNA/vol.3.1995/pp116-149.dir/pp116-149.html), *Electron. Trans. Numer. Anal.* 3, 1995 — why the counts are monotone in floating point, and the `pivmin` guard (LAPACK's `sstebz`).
- C. H. Bischof, B. Lang and X. Sun, ["A framework for symmetric band reduction"](https://doi.org/10.1145/365723.365735), *ACM Trans. Math. Softw.* 26(4), 2000 — reducing a dense matrix to band form, then the band to tridiagonal: the two-stage reduction behind `eigvalsh`'s CPU path.
- A. Haidar, H. Ltaief and J. Dongarra, ["Parallel reduction to condensed forms for symmetric eigenvalue problems using aggregated fine-grained and memory-aware kernels"](https://doi.org/10.1145/2063384.2063394), SC '11, 2011 — the two-stage algorithm as [LAPACK 3.7.0](https://netlib.org/lapack/lapack-3.7.0.html) implements it (`ssyevd_2stage`), which this library calls through Accelerate, and which the `band` backend runs with its first stage on the GPU.
- E. Ringoot, R. Alomairy, V. Churavy and A. Edelman, ["Performant unified GPU kernels for portable singular value computation across hardware and precision"](https://doi.org/10.1145/3754598.3754667), 2025 ([arXiv:2508.06339](https://arxiv.org/abs/2508.06339)), and E. Ringoot, R. Alomairy and A. Edelman, ["Accelerating bidiagonalization of banded matrices through memory-aware bulge-chasing on GPUs"](https://arxiv.org/abs/2510.12705), 2025 — two-stage reductions on GPUs, including Apple's; they prompted measuring the two-stage reduction on Apple Silicon, and the second's GPU-resident design is why the `tridiag` backend keeps its panel on the GPU. Their kernels are not used here.
- NVIDIA, [cuSOLVER `syevjBatched`](https://docs.nvidia.com/cuda/cusolver/index.html#cusolverdn-t-syevjbatch) — Jacobi as the production batched symmetric eigensolver on GPUs.
