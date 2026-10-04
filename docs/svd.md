# Singular value decomposition (`svd`)

```cpp
#include <metal_linalg/svd.h>

// a: MLX array of shape [M, N] or [..., M, N] (real: float32, or cast to it; complex input throws), any M and N
// Returns the thin SVD, K = min(M, N):
//   U [..., M, K] orthonormal columns, S [..., K] descending, Vt [..., K, N] orthonormal rows
auto [U, S, Vt] = metal_linalg::svd_accelerated(a);
array S2 = metal_linalg::svdvals_accelerated(a);     // singular values only
```

This is the economy form, like this library's QR and numpy's
`full_matrices=False`. `mlx::core::linalg::svd` returns the full-size factors
(`U` is M×M) and as of MLX 0.31 runs only on the CPU. Wide input, rank-deficient
input and magnitudes from 1e-30 to 1e+37 are handled; non-finite input yields
NaN output.

## Algorithm: one-sided Jacobi, one simdgroup per column pair

Hestenes' method works on the columns $g_1 \ldots g_N$ of $G = A$ directly. For
a pair $(p, q)$ it forms

$$\alpha = g_p \cdot g_p, \qquad \beta = g_q \cdot g_q, \qquad \gamma = g_p \cdot g_q$$

and applies the plane rotation that makes the two columns orthogonal, to $G$
and to the same columns of $V$. With $\zeta = (\beta - \alpha) / 2\gamma$ the
rotation is the one the eigensolver uses, $t = \operatorname{sign}(\zeta) /
(|\zeta| + \sqrt{1 + \zeta^2})$. When every pair is orthogonal to working
precision $G = U \Sigma$: the column norms are the singular values and the
normalised columns are $U$. It is two-sided Jacobi on $A^T A$ without ever
forming $A^T A$, which is what keeps the small singular values accurate.

**Why it suits the GPU better than the eigensolver's two-sided method.** A
rotation touches only columns $p$ and $q$. The round-robin ordering makes the
$N/2$ pairs of a round disjoint, so when each pair is owned by one simdgroup
nothing it reads or writes during the round is touched by any other. The
inner products are a `simd_sum` over the 32 lanes, the rotation parameters
are uniform across them, and the same lanes apply the rotation. A round needs
**one** threadgroup barrier, between rounds, where the two-sided method needs
three.

**Null columns.** In a rank-deficient matrix some columns cancel to rounding
noise. Rotating two of them against each other is where one-sided Jacobi
fails to terminate: the rotation amplifies their relative error, which puts
them back out of line with the large columns, whose correction changes their
angles to one another, and so on down to underflow. A column whose norm is
below the rank tolerance times the largest column's is therefore still
orthogonalised against every column that is not null, which is well
conditioned and settles in two passes, but never against another null column.
Such singular values are reported as they are, tiny, and the host gives `U` an
orthonormal completion for them. The tolerance is numpy's default for
`matrix_rank`, `max(M, N) * eps`, with a floor of `64 * eps`.

## Two Jacobi kernels, four GPU backends

As for QR and the eigensolver, different shapes get different shaders. Two
independent choices make four GPU backends:

| choice | option | what runs | for |
|---|---|---|---|
| kernel | whole-matrix (`Svd_Jacobi.metal`) | one threadgroup per matrix, one simdgroup per column pair | small matrices, batched |
| kernel | block (`Svd_BlockJacobi.metal`) | columns in blocks of 16; for each pair of blocks one threadgroup forms the 32×32 Gram matrix, runs the eigensolver's Jacobi on it in threadgroup memory and applies the rotation to $G$ and $V$ as tile products | large matrices |
| preconditioning | none | the kernel on the matrix itself | square and moderately tall input |
| preconditioning | QR (`qr_jacobi`, `qr_block_jacobi`) | this library's QR, then the kernel on the K×K triangular factor, then $U = Q\,U_R$ | long thin input |

**Which kernel.** The whole-matrix kernel gives a matrix one GPU core, so a
single large matrix cannot use the rest of the chip: 512×512 takes half a
second on an M1. The block kernel spreads one matrix over the grid, at the
cost of several dispatches per round, so it overtakes the whole-matrix kernel
from a short side of about 192 for a single matrix (6.7x faster at 512) and
earlier in batches, where the dispatches are shared.

**Whether to precondition.** QR does not pay for square input: it takes the
sweep count from 10 to 9 and costs a QR. For tall input it is structural, since
the rotations then act on K×K instead of L×K, but the path carries a fixed
cost of a few milliseconds, so the matrix has to be wide enough as well as
long: about 2x faster at 1024×64 and 2048×64, 2.7x slower at 1024×16. Design
notes and every measurement so far are in
[`studies/svd-design-notes.md`](studies/svd-design-notes.md).

## A fifth backend for large matrices: bidiagonalization on the GPU (`Svd_Bidiag.metal`)

Jacobi needs a dozen sweeps of $O(k^2 l)$ each, so from a short side of
about a thousand even the block kernel loses to LAPACK, whose `sgesdd` spends
most of its time reducing $A$ to upper bidiagonal form $A = Q B P^T$
(`sgebrd`) and applying $Q$ and $P$ to the singular vectors of $B$ (`sormbr`):
on an M5 Pro, at 4096×4096, 1.8 s and 1.0 s of 3.6 s. The `bidiag` backend
(`src/svd_bidiag.mm`) keeps LAPACK's method and moves those two steps to the
GPU, as the eigensolver's `tridiag` backend does for `ssytrd`:

1. **Reduction**, blocked `sgebrd` in panels of 32 columns, as LAPACK's
   `slabrd` blocks it, every step on the GPU. Per column: the panel's earlier
   reflectors applied to the column and to the row, both Householder vectors
   (`slarfg`, norm scaled by the largest entry), and the two matrix-vector
   products with the trailing matrix, `slabrd`'s $X$ and $Y$, in 64 × 64
   tiles. Per panel: the trailing update $A_{22} \mathrel{-}= V Y^T + X U^T$
   as two MPS GEMMs. The panels' command buffers are queued back to back and
   the host waits once per matrix. The last 33 columns or fewer are reduced
   by LAPACK.

   A column's steps are four dispatches (since 2.12.0; twelve before), as
   the `tridiag` backend's are three: the column's update with the previous
   row's $X$ finished, then $A^T v$ with the column's Householder vector
   formed from norm partials in every threadgroup, then the row's update with
   $Y$, then $A u$ with the row's vector. Before, the row's vector alone took
   9 µs a column at $k = 4096$ (one threadgroup reading a row three times with
   a stride), and the small kernels together 52 µs, against 160 µs for the
   two products. With the matrix also copied in on every core rather than
   one, singular values alone are 1.2-1.8x faster at $k = 1024$-4096, and
   with vectors 1.1-1.3x.
2. **Bidiagonal SVD**: LAPACK `sbdsdc` (divide and conquer) on the CPU, with
   the singular vectors of $B$ or without.
3. **Back-transformation**: the reflectors of $Q$ and of $P$ applied to the
   singular vectors of $B$ 128 at a time, each block as $I - V T V^T$
   (`slarft`), as MPS GEMMs; the next block is built on the CPU while the GPU
   applies this one.

A matrix at least twice as tall as wide (and $k \ge 64$) is first reduced by
this library's QR, the $k \times k$ factor is decomposed as above, and
$U = Q U_R$; a wide matrix goes through its transpose. Each matrix is scaled
by a power of two first (exact). A batch is pipelined over two workspace
slots (since 2.11.0): the CPU solves one matrix's bidiagonal problem while the
GPU reduces the next, and solves the next while the GPU back-transforms this
one. On an M5 Pro, per matrix of 2048×2048 with vectors, 253 ms alone, 168 ms
in a batch of 4 and 155 ms in a batch of 8 (1.63x; 308, 202 and 186 ms before
the reduction's dispatches were merged in 2.12.0), so the GPU stays ahead of
the CPU path up to batches of 4 at that size. The backend is still for large
matrices, not batches of small ones.
`svdvals` (singular values alone) skips step 3 and the vectors of step 2.

On an M5 Pro, one $k \times k$ matrix, against the CPU path (`sgesdd`), from
the routing sweep in [`results/apple-m5-pro-20gpu/20261004-fd9bd8/svd/`](results/apple-m5-pro-20gpu/20261004-fd9bd8/svd/report.md)
(min of two randomised passes):

| $k$ | svd: CPU | bidiag | speedup | svdvals: CPU | bidiag | speedup |
|---|---|---|---|---|---|---|
| 512 | 16.7 ms | 16.5 ms | 1.01x | 7.7 ms | 9.5 ms | 0.81x |
| 1024 | 77.6 ms | 55.3 ms | 1.40x | 35.6 ms | 26.4 ms | 1.35x |
| 1536 | 177 ms | 131 ms | 1.35x | 84.5 ms | 58.9 ms | 1.43x |
| 2048 | 432 ms | 247 ms | 1.75x | 192 ms | 111 ms | 1.73x |
| 3072 | 1.27 s | 0.60 s | 2.10x | 0.69 s | 0.33 s | 2.08x |
| 4096 | 3.42 s | 1.51 s | 2.26x | 1.88 s | 0.81 s | 2.32x |

The gain grows with $k$, as the CPU's reduction falls further behind memory
bandwidth. The M5 Pro uses the backend from $k = 1024$, with vectors or
without, for up to two matrices: it decomposes a batch one matrix after
another (pipelined), and the CPU path spreads one over every core
(`bidiag_max_batch`). Since the reduction's dispatches were merged (2.12.0)
it wins by 1.35-1.75x from 1024 to 2048, where before it was 1.0-1.4x with
vectors and lost below 2048 for singular values alone. Accuracy
matches LAPACK's: at 2048 and 4096, square, tall and wide, reconstruction
and orthogonality about $4 \times 10^{-6}$, singular values within
$1.3 \times 10^{-6}$ of float64 LAPACK's relative to $\sigma_\text{max}$
(about $10^{-5}$ for `svdvals`, as for LAPACK's own singular-values-only
path).

## A sixth backend for batches: `golub_kahan` (`Svd_GolubKahan.metal`)

Against a CPU path that spreads a batch over every core, the Jacobi kernels
lost nearly everywhere (see [Performance](#performance)): one-sided Jacobi does
several times the flops of LAPACK's method and needs a dozen sweeps. The
`golub_kahan` backend (`src/svd_golub_kahan.mm`, new in 2.10.0) is LAPACK's
method in one threadgroup per matrix, the SVD counterpart of the eigensolver's
`ql` ([eigh.md](eigh.md)):

1. **Bidiagonalization**, $A = Q B P^T$ (`sgebd2`): per column a Householder
   reflector from the left, then per row one from the right, the matrix in
   threadgroup memory. The left reflector's column sums are walked by the
   thread owning each column and its update is row-local; the right reflector
   needs only its row, so the thread owning that row forms it as soon as the
   row is updated, and the right update runs on into the next column's sum:
   four barriers per column.
2. **Forming $Q$ and $P$** from the reflectors: $Q$ in place (`sorg2r`), $V =
   P$ in a device-memory workspace.
3. **Implicit bidiagonal QR** (`sbdsqr`'s shifted sweep, its shift the smaller
   singular value of the trailing 2×2), chasing from the top of the block and
   deflating at the bottom; a negligible diagonal entry is set to zero and
   chased out of its block with rotations of its own (Golub and Van Loan
   §8.6.2). The iteration works on the bidiagonal alone, so one thread runs it
   and records each step's rotations, the left ones for $U$ and the right
   ones for $V$, and every thread then applies the recorded list to its own
   row, carrying the value the rotations share in a register. A step costs
   one barrier, and from $k = 33$ a simdgroup of its own computes the next
   step while the rows apply this one.
4. Negative values flip their column of $V$; a rank sort orders them.

A wide matrix is decomposed as its transpose. Each matrix is scaled by a power
of two first, as in the other kernels, and the kernel is built with fast math
and the eigensolver `ql` kernel's Newton-refined division and square root.

**Why $V$ lives in device memory.** The kernel is latency-bound (the QR
iteration is one thread's chain of dependent rotations), so its speed is
decided by how many matrices share a GPU core, and threadgroup memory bounds
that. Kept beside the matrix in threadgroup memory, $V$ halved it; stored in
device memory, transposed so that the threads owning consecutive rows of it
touch consecutive addresses, it made 4096 matrices of 32×32 1.4x faster and
1024 of 48×48 1.5x on an M5 Pro, and raised the size limit from 59 to 83.

**Since 2.11.0**, two changes for the shapes it handled worst. A phase that
sums down columns (the left reflectors, and forming $Q$ and $V$) gives each
column a group of lanes, sized from the threads there are to spare, instead
of one thread: on a tall matrix the columns are few and long, and one thread
per column was a long dependent chain while most threads waited. That made
256×16 1.4x faster and 128×32 1.2x, and squares about 1.1x. And from $k = 40$
for singular values alone, or 60 with vectors, the work is split over two
dispatches: the reduction, then the QR iteration in threadgroups with almost
no threadgroup memory, so that many matrices overlap their iterations (for
singular values alone, where the iteration is most of the work, 1.3x faster
at 48×48 and 1.55x at 64×64; with vectors 9% at 64×64).

**Sizes.** The matrix lives in threadgroup memory, so the backend takes
squares up to 83×83 with 32 KB (`detail::svd_gk_max_k()`), longer matrices
when tall (444×16, 202×32). Inside its window the router runs it on the
matrix itself where it fits (`golub_kahan`) and otherwise after this
library's QR, on the $k \times k$ factor (`qr_golub_kahan`), as the Jacobi
kernels are preconditioned.

On an M5 Pro, $k \times k$ matrices in batches, from the routing sweep in
[`results/apple-m5-pro-20gpu/20261003-c0878c/svd/`](results/apple-m5-pro-20gpu/20261003-c0878c/svd/)
(2.11.0; min of two randomised passes; the CPU path spreads the batch over 18 cores; "shared" is the
batch split between golub_kahan and the CPU path, see below):

| k×k | batch | golub_kahan | shared with the CPU | best Jacobi | CPU | Jacobi / GK | CPU / GK | CPU / shared |
|---|---|---|---|---|---|---|---|---|
| 8 | 4096 | 0.91 ms | 0.99 ms | 1.42 ms | 1.44 ms | 1.57x | **1.59x** | **1.46x** |
| 16 | 1024 | 0.90 ms | 0.90 ms | 1.75 ms | 1.19 ms | 1.95x | **1.32x** | **1.32x** |
| 16 | 4096 | 2.42 ms | 2.30 ms | 5.77 ms | 4.50 ms | 2.38x | **1.86x** | **1.96x** |
| 24 | 4096 | 5.76 ms | 4.53 ms | 14.1 ms | 9.52 ms | 2.46x | **1.65x** | **2.10x** |
| 32 | 256 | 0.95 ms | 0.87 ms | 2.19 ms | 1.09 ms | 2.31x | **1.15x** | **1.25x** |
| 32 | 1024 | 2.64 ms | 2.20 ms | 6.69 ms | 4.07 ms | 2.54x | **1.54x** | **1.85x** |
| 32 | 4096 | 9.33 ms | 8.78 ms | 22.4 ms | 15.4 ms | 2.40x | **1.66x** | **1.76x** |
| 40 | 4096 | 18.5 ms | 14.0 ms | 49.3 ms | 24.0 ms | 2.67x | **1.30x** | **1.71x** |
| 48 | 1024 | 7.16 ms | 5.17 ms | 21.1 ms | 9.16 ms | 2.95x | **1.28x** | **1.77x** |
| 48 | 4096 | 27.9 ms | 18.8 ms | 80.2 ms | 34.0 ms | 2.87x | **1.22x** | **1.81x** |
| 56 | 4096 | 43.6 ms | 24.9 ms | 104.4 ms | 40.1 ms | 2.39x | 0.92x | **1.61x** |
| 64 | 1024 | 17.4 ms | 9.03 ms | 30.4 ms | 14.5 ms | 1.74x | 0.83x | **1.60x** |
| 64 | 4096 | 64.8 ms | 33.7 ms | 113.8 ms | 52.7 ms | 1.76x | 0.81x | **1.56x** |
| 80 | 4096 | 147.9 ms | 65.7 ms | 309.3 ms | 96.5 ms | 2.09x | 0.65x | **1.47x** |

It is 1.6-3x faster than the best Jacobi kernel at every size it takes, and
alone ahead of the CPU for large batches up to 48×48; shared with the CPU, up
to 80×80 (1.5-2.1x). Alone, beyond 48 the CPU pulls away: the QR iteration is one thread's serial chain per matrix, and with the
matrix in threadgroup memory too few matrices share a core to hide it. Tall
matrices are the CPU's whatever the backend, since the CPU path reduces them
by a QR first (256×32, 4096 of them: 0.84x), which the routing's
`gpu_max_l` says.

Accuracy is that of LAPACK's QR iteration: backward stable, singular values
accurate relative to $\sigma_\text{max}$. In `tests/test_svd.cpp`, from 1×1 to
83×83, tall, wide, rank-deficient (the zero-diagonal chase), zero, identity,
graded from 1e+4 to 1e-4 and scaled from 1e-30 to 1e+37: reconstruction and
orthogonality at most $2.4 \times 10^{-6}$, singular values within
$6 \times 10^{-7}$ of LAPACK's relative to $\|A\|_F$. One-sided Jacobi
computes small singular values to high *relative* accuracy, which a QR
iteration does not; where that matters, `gk_max_k = 0` (`SVD_GK_MAX_K=0`)
keeps the Jacobi kernels.

## Routing

With $k = \min(M, N)$ and $l = \max(M, N)$:

```
GPU iff  k <= gpu_max_k,  l <= gpu_max_l,  batch * k >= gpu_min_batch_times_k  and  batch >= gpu_min_batch,
     or  gpu_max_k < k <= gpu_big_batch_max_k,  l <= gpu_max_l  and  batch >= gpu_big_batch_min   (large batches)
     else CPU
         (svdvals: the values_gpu_* constants, without the large-batch clause, unless values_gpu_min_batch = 0)
on the GPU:
  golub_kahan iff  gk_min_k <= k <= gk_max_k  (clipped to svd_gk_max_k()):
                   on the matrix if it fits, else after QR (qr_golub_kahan);
                   from a batch of share_min_batch, shared with the CPU path
  otherwise the Jacobi kernels:
    precondition with QR iff  l >= qr_min_rows,  k >= qr_min_k  and  l >= 2k
    block kernel iff  k >= block_min_k,  or  k >= block_min_k_batched and batch >= block_min_batch
where that says CPU:
  bidiag instead iff  k >= bidiag_min_k  (svdvals: k >= values_bidiag_min_k; 0 = never)
```

The CPU path is a fair one: LAPACK (Accelerate) called directly, `sgesdd`,
preceded by a thin QR (`sgeqrf`, `sorgqr`) when the matrix is at least twice
as tall as wide, so that it too computes thin factors only. Since 2.9.0 a
batch is spread over every CPU core, each core solving whole matrices with
Accelerate's own threading off: on an M5 Pro 11-15x faster than one matrix at
a time for batches of 8×8 to 128×128, 3-10x for larger ones. A lone matrix
keeps Accelerate's threading; `set_cpu_threads()` or `METAL_LINALG_CPU_THREADS`
caps the cores used. The tables below were measured against this CPU path.

As for QR and the eigensolver, the policy is a per-device table, keyed on the
Metal device name and GPU core count:

| GPU | cores | GPU iff | golub_kahan | QR from | block from | else bidiag | status |
|---|---|---|---|---|---|---|---|
| Apple M5 Pro | 20 | k <= 56, l <= 256 and batch * k >= 16384, or 57 <= k <= 80 in batches of 1024+ (svdvals: k, l <= 56 and batch * k >= 16384) | k = 8 .. 80, shared with the CPU from batch 1024 | 512 rows, k >= 32 | k = 192; k = 64 in batches of 64+ | from k = 1024 (svdvals too), batches up to 2 | measured — run [`20261004-fd9bd8`](results/apple-m5-pro-20gpu/20261004-fd9bd8/svd/report.md) |
| anything else | — | k <= 64 and batch * k >= 1024 | never | 512 rows, k >= 64 | k = 192 | never | **untuned default** |

On the M5 Pro large batches of small matrices, up to 56×56 and a long side
of 256, go to `golub_kahan` on the GPU, shared with the CPU from 1024
matrices, and so do batches of 1024 and more up to 80×80 (the large-batch
clause, since 2.12.0); everything else in a batch goes to the CPU, and one
or two large matrices to `bidiag`. Against the best backend at each of
the 291 points measured, the row scores 1.020 geometric-mean regret, worst
1.67x, against 1.028 for the product rule alone (on the previous run, 2.11.0's
row scored 1.027 and 2.10.0's 1.039). The clause is what
takes 64×64 and 80×80 in batches of 1024 and more, which shared win by
1.5-1.6x but which one product rule cannot take without also taking their
small batches, which the CPU wins. The long-side cap
`gpu_max_l` (new in 2.10.0, no cap on a device without it) is what lets the
rule take the square batches the GPU wins without the tall ones it loses:
with a cap on k alone, the fit stopped at k = 24. In 2.9.0, measured against
the same CPU path, the Jacobi kernels won only for large batches of the
smallest matrices; before 2.9.0 the row sent batches up to k = 1024 to the
GPU, measured against one CPU core.

The M1 has no row. The defaults come from measurements on an M1 taken while
the machine was heavily loaded by other jobs, good enough to place the
crossovers roughly and not for a table entry, so `svd_policy_source()`
reports `default:untuned-device` there too. Measuring a Mac is one command,
`python3 tuning/run.py`, which covers all three decompositions; see
[`tuning.md`](tuning.md).

`set_svd_policy()` and the environment variables `SVD_QR_MIN_ROWS`,
`SVD_QR_MIN_K`, `SVD_BLOCK_MIN_K`, `SVD_BLOCK_MIN_K_BATCHED`,
`SVD_BLOCK_MIN_BATCH`, `SVD_GPU_MAX_K`, `SVD_GPU_MIN_BATCH_TIMES_K`,
`SVD_GPU_MIN_BATCH`, `SVD_GPU_MAX_L`, `SVD_VALUES_GPU_MAX_K`,
`SVD_VALUES_GPU_MIN_BATCH_TIMES_K`, `SVD_VALUES_GPU_MIN_BATCH`,
`SVD_VALUES_GPU_MAX_L`, `SVD_BIDIAG_MIN_K`, `SVD_VALUES_BIDIAG_MIN_K`,
`SVD_BIDIAG_MAX_BATCH`, `SVD_VALUES_BIDIAG_MAX_BATCH`, `SVD_GK_MIN_K`,
`SVD_GK_MAX_K`, `SVD_SHARE_MIN_BATCH`, `SVD_GPU_BIG_BATCH_MAX_K`,
`SVD_GPU_BIG_BATCH_MIN` and `SVD_DEVICE=gpu|cpu|bidiag` override it.
`svd_backend(m, n, batch)` and `svdvals_backend(m, n, batch)` say which of the
eight backends a problem gets, with vectors and for singular values alone.

**Singular values alone** (since 2.11.0) have a GPU-or-CPU rule of their own,
the `values_gpu_*` constants: both sides skip the vectors, by different
amounts (the CPU's back-transformation and the GPU's vector updates), so the
boundary is not the same. In 2.10.0 `svdvals` followed the vectors' rule and,
on an M5 Pro, ran batches of 40×40 to 48×48 on the GPU at up to 1.5x the
CPU's time. The rule is fitted on `gk_vals` and `cpu_vals` timings (stage 2b of
`tuning/tune_svd.py`); `values_gpu_min_batch = 0` means "as with vectors".

**Sharing a batch with the CPU** (since 2.11.0). From a batch of
`share_min_batch` (0: never), a batch that goes to `golub_kahan` is solved by
the GPU and the CPU path at once, as the eigensolver's `ql` batches are (see
[eigh.md](eigh.md#dispatch)): the GPU takes chunks from the front, CPU workers
a few matrices at a time from the back, and they meet wherever their speeds
put them. On an M5 Pro, against the faster of the two alone: 1.41x for 4096
matrices of 16×16, 1.45x for 4096 of 40×40, 1.43x for 1024 of 48×48, 1.71x for
1024 of 56×56 and 1.62x for 1024 of 64×64 (with vectors), 1.45x for 1024 of
48×48 (singular values alone). `svd_shares_batch()` and
`svdvals_shares_batch()` report it.

## Accuracy

From `tests/test_svd.cpp`, relative to $\|A\|_F$, Gaussian input:

| shape | $\|A - U\Sigma V^T\|_F$ | $\|U^TU - I\|_F/\sqrt{K}$ | $\|V^TV - I\|_F/\sqrt{K}$ | $\max \lvert \sigma - \sigma_\text{LAPACK} \rvert$ | sweeps |
|---|---|---|---|---|---|
| 8×8 | 3.3e-07 | 1.5e-07 | 3.9e-07 | 1.2e-07 | 5 |
| 64×64 | 1.3e-06 | 1.8e-06 | 1.4e-06 | 1.9e-07 | 9 |
| 256×256 | 3.7e-06 | 8.4e-06 | 4.0e-06 | 3.7e-07 | 12 |
| 512×512 | 6.9e-06 | 1.8e-05 | 7.1e-06 | 3.8e-07 | 13 |
| 2048×64 | 1.3e-06 | 1.2e-05 | 1.3e-06 | 1.7e-07 | 8 |

Rank-deficient input reconstructs as well as full-rank input: over 204 random
instances from rank 1 of 24×24 to rank 50 of 600×130, through all four GPU
backends, the worst reconstruction error was 2.3e-06 and the worst sweep count
12. How null columns are kept from stalling the sweeps is in
[`studies/svd-design-notes.md`](studies/svd-design-notes.md).

## Performance

Apple M5 Pro (20 GPU cores, 18 CPU cores) with 2.11.0, from the routing sweep in
[`results/apple-m5-pro-20gpu/20261003-c0878c/svd/`](results/apple-m5-pro-20gpu/20261003-c0878c/svd/): min of two
randomised passes, on mains, against the CPU path as the library runs it, a
batch spread over all 18 cores. Each cell is the speedup of the fastest GPU
backend over the CPU, with the GPU time and the backend (`whole` and `block`
are the two Jacobi kernels on the matrix itself, `QR+` after preconditioning,
`GK` the golub_kahan backend, `GK+CPU` it sharing the batch with the CPU
path); bold is where the GPU was ahead.

| shape | batch 1 | batch 4 | batch 16 | batch 64 | batch 256 | batch 4096 |
|---|---|---|---|---|---|---|
| 4×4 | 0.02x (0.24 ms, whole) | 0.12x (0.19 ms, GK) | 0.25x (0.15 ms, whole) | 0.32x (0.25 ms, whole) | 0.66x (0.21 ms, GK) | **1.75x** (0.39 ms, GK) |
| 8×8 | 0.03x (0.25 ms, GK) | 0.06x (0.31 ms, whole) | 0.37x (0.18 ms, whole) | 0.57x (0.33 ms, GK+CPU) | 0.79x (0.25 ms, GK) | **1.59x** (0.91 ms, GK) |
| 16×16 | 0.09x (0.23 ms, whole) | 0.17x (0.24 ms, whole) | 0.36x (0.25 ms, whole) | 0.53x (0.34 ms, whole) | **1.09x** (0.39 ms, GK) | **1.96x** (2.30 ms, GK+CPU) |
| 32×32 | 0.09x (0.62 ms, whole) | 0.25x (0.36 ms, whole) | 0.47x (0.40 ms, whole) | 0.55x (0.71 ms, GK) | **1.25x** (0.87 ms, GK+CPU) | **1.76x** (8.78 ms, GK+CPU) |
| 48×48 | 0.12x (0.92 ms, whole) | 0.20x (0.77 ms, whole) | 0.33x (0.87 ms, whole) | 0.58x (1.21 ms, GK) | **1.54x** (1.53 ms, GK+CPU) | **1.81x** (18.8 ms, GK+CPU) |
| 64×64 | 0.19x (1.07 ms, whole) | 0.20x (1.15 ms, whole) | 0.33x (1.18 ms, whole) | 0.53x (1.95 ms, GK) | **1.08x** (3.43 ms, GK+CPU) | **1.56x** (33.7 ms, GK+CPU) |
| 128×128 | 0.21x (4.91 ms, whole) | 0.27x (4.94 ms, whole) | 0.34x (5.05 ms, whole) | 0.42x (14.3 ms, block) | 0.46x (47.6 ms, block) | 0.40x (751.2 ms, block) |
| 256×256 | 0.30x (13.0 ms, block) | 0.30x (14.1 ms, block) | 0.25x (24.6 ms, block) | 0.25x (82.9 ms, block) | 0.23x (339.2 ms, block) | -- |
| 512×512 | 0.52x (32.0 ms, block) | 0.38x (48.4 ms, block) | 0.18x (154.0 ms, block) | 0.16x (628.7 ms, block) | -- | -- |
| 1024×1024 | 0.67x (115.1 ms, block) | 0.32x (290.5 ms, block) | 0.24x (1.22 s, block) | -- | -- | -- |
| 64×8 | 0.06x (0.20 ms, whole) | 0.13x (0.22 ms, whole) | 0.34x (0.24 ms, GK) | 0.67x (0.24 ms, GK) | 0.97x (0.30 ms, GK) | **1.83x** (1.60 ms, GK+CPU) |
| 64×32 | 0.13x (0.53 ms, whole) | 0.19x (0.58 ms, whole) | 0.57x (0.40 ms, whole) | 0.57x (0.83 ms, whole) | **1.47x** (1.02 ms, GK+CPU) | **1.70x** (12.6 ms, GK+CPU) |
| 256×32 | 0.23x (0.51 ms, whole) | 0.31x (0.51 ms, whole) | 0.58x (0.53 ms, whole) | 0.54x (1.34 ms, whole) | 0.77x (3.66 ms, GK+CPU) | **1.06x** (30.3 ms, GK+CPU) |
| 1024×32 | 0.32x (0.69 ms, QR+whole) | 0.44x (0.87 ms, QR+whole) | 0.57x (1.14 ms, whole) | 0.63x (3.48 ms, GK+CPU) | 0.66x (9.36 ms, GK+CPU) | -- |
| 1024×64 | 0.38x (1.55 ms, QR+whole) | 0.51x (2.19 ms, QR+whole) | 0.63x (3.42 ms, QR+whole) | 0.64x (8.74 ms, GK+CPU) | 0.68x (27.5 ms, GK+CPU) | -- |
| 2048×64 | 0.48x (1.89 ms, QR+whole) | 0.56x (3.52 ms, QR+whole) | 0.63x (5.00 ms, QR+whole) | 0.70x (14.0 ms, GK+CPU) | 0.74x (46.9 ms, GK+CPU) | 0.69x (736.7 ms, QR+block) |
| 1024×256 | 0.47x (13.9 ms, QR+block) | 0.47x (15.8 ms, QR+block) | 0.43x (27.9 ms, QR+block) | 0.44x (95.9 ms, QR+block) | 0.41x (390.5 ms, QR+block) | -- |
| 2048×256 | 0.55x (16.2 ms, QR+block) | 0.56x (18.8 ms, QR+block) | 0.54x (36.3 ms, QR+block) | 0.55x (129.3 ms, QR+block) | 0.52x (518.3 ms, QR+block) | -- |

`--` was not measured.

Against a CPU that uses its cores, the GPU wins large batches of small
matrices, with `golub_kahan`, shared with the CPU from about 1024 matrices:
1.6-2x at 4096 matrices from 4×4 to 64×64, and from 256 matrices for 16×16 to
64×32. Lone matrices and small batches stay the CPU's, as do tall shapes past a
long side of 256 and anything from 128×128; one large matrix wins on
`bidiag`. The tables in this document's versions before 2.9.0, with up to 16x
for batches, were against one CPU core.

Against `mlx::core::linalg::svd` as it stands, tall shapes look far better
than this, because MLX computes the full M×M `U`. That is a fair description
of what calling MLX costs and an unfair comparison of algorithms, so the table
uses a CPU path that computes thin factors too. Regenerate with
`./build/benchmark_svd`. The earlier M1 figures, taken on a heavily loaded
machine, are kept in
[`studies/svd-design-notes.md`](studies/svd-design-notes.md) for the shape of
the result only.

## Tests

```sh
cmake --build build --target test_svd
./build/test_svd          # or: ctest --test-dir build
```

329 checks: square, tall and wide shapes around the simdgroup, pair-count and
block boundaries, batches, every simdgroup count, all seven GPU backends and
each branch of the CPU one (the `bidiag` backend from 1×1 to 1100×1060, with
QR first and through the transpose, with vectors and without; `golub_kahan`
from 1×1 to its limit, either side of its chaser simdgroup, directly and
after a QR), rank deficiency repeated over random instances per shape and
backend, structured spectra (graded columns, singular values from 1e+4 to
1e-4, repeated values), magnitudes from 1e-30 to 1e+37, NaN inside a batch,
and the routing policy, including the batch-dependent kernel crossover, the
golub_kahan window, the long-side cap, the rule for singular values alone and
sharing a batch with the CPU, without assuming any device's values.

## References

- M. R. Hestenes, ["Inversion of matrices by biorthogonalization and related results"](https://doi.org/10.1137/0106005), *J. SIAM* 6(1), 1958 — one-sided Jacobi.
- Z. Drmač and K. Veselić, ["New fast and accurate Jacobi SVD algorithm I"](https://doi.org/10.1137/050639193), *SIAM J. Matrix Anal. Appl.* 29(4), 2008 — the modern form, preconditioning with QR, and what LAPACK's `xGESVJ` / `xGEJSV` implement.
- J. Demmel and K. Veselić, ["Jacobi's method is more accurate than QR"](https://epubs.siam.org/doi/10.1137/0613074), 1992.
- R. P. Brent and F. T. Luk, 1985 — the parallel ordering; see the eigensolver's references.
- G. H. Golub and W. Kahan, ["Calculating the singular values and pseudo-inverse of a matrix"](https://doi.org/10.1137/0702016), *J. SIAM Ser. B Numer. Anal.* 2(2), 1965 — reduction to bidiagonal form by Householder reflections, the method of the `bidiag` backend.
- J. Demmel and W. Kahan, ["Accurate singular values of bidiagonal matrices"](https://doi.org/10.1137/0911052), *SIAM J. Sci. Stat. Comput.* 11(5), 1990 — LAPACK's `sbdsqr`, whose shifted QR sweep, shift and deflation the `golub_kahan` backend runs.
- G. H. Golub and C. F. Van Loan, *Matrix Computations*, 4th ed., 2013 — §5.4.8 (bidiagonalization) and §8.6 (the Golub-Kahan SVD step, and zero diagonal entries), the `golub_kahan` backend.
- J. J. Dongarra, S. J. Hammarling and D. C. Sorensen, ["Block reduction of matrices to condensed forms for eigenvalue computations"](https://doi.org/10.1016/0377-0427(89)90367-1), *J. Comput. Appl. Math.* 27(1-2), 1989 — the blocked bidiagonalization (LAPACK's `sgebrd` and `slabrd`) the backend runs on the GPU.
- M. Gu and S. C. Eisenstat, ["A divide-and-conquer algorithm for the bidiagonal SVD"](https://doi.org/10.1137/S0895479892242232), *SIAM J. Matrix Anal. Appl.* 16(1), 1995 — LAPACK's `sbdsdc`, which solves the bidiagonal problem.
- S. Tomov, R. Nath and J. Dongarra, ["Accelerating the reduction to upper Hessenberg, tridiagonal, and bidiagonal forms through hybrid GPU-based computing"](https://doi.org/10.1016/j.parco.2010.06.001), *Parallel Computing* 36(12), 2010 — the hybrid CPU/GPU split (MAGMA) the backend follows, with the panel kept on the GPU.
- E. Ringoot, R. Alomairy, V. Churavy and A. Edelman, ["Performant unified GPU kernels for portable singular value computation across hardware and precision"](https://doi.org/10.1145/3754598.3754667), 2025 ([arXiv:2508.06339](https://arxiv.org/abs/2508.06339)), and E. Ringoot, R. Alomairy and A. Edelman, ["Accelerating bidiagonalization of banded matrices through memory-aware bulge-chasing on GPUs"](https://arxiv.org/abs/2510.12705), 2025 — GPU singular value computation, including on Apple's GPUs; they prompted the `bidiag` backend, whose GPU-resident panel follows their design. Their kernels are not used here.
