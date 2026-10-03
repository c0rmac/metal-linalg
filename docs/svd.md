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
2. **Bidiagonal SVD**: LAPACK `sbdsdc` (divide and conquer) on the CPU, with
   the singular vectors of $B$ or without.
3. **Back-transformation**: the reflectors of $Q$ and of $P$ applied to the
   singular vectors of $B$ 128 at a time, each block as $I - V T V^T$
   (`slarft`), as MPS GEMMs; the next block is built on the CPU while the GPU
   applies this one.

A matrix at least twice as tall as wide (and $k \ge 64$) is first reduced by
this library's QR, the $k \times k$ factor is decomposed as above, and
$U = Q U_R$; a wide matrix goes through its transpose. Each matrix is scaled
by a power of two first (exact), and matrices are decomposed one after
another, so the backend is for large matrices, not batches.
`svdvals` (singular values alone) skips step 3 and the vectors of step 2.

On an M5 Pro, one $k \times k$ matrix, against the CPU path (`sgesdd`), from
the routing sweep in [`results/apple-m5-pro-20gpu/20261003-064803/svd/`](results/apple-m5-pro-20gpu/20261003-064803/svd/)
(min of two randomised passes):

| $k$ | svd: CPU | bidiag | speedup | svdvals: CPU | bidiag | speedup |
|---|---|---|---|---|---|---|
| 512 | 16.8 ms | 25.7 ms | 0.65x | 7.7 ms | 18.8 ms | 0.41x |
| 1024 | 77.4 ms | 76.0 ms | 1.02x | 35.5 ms | 47.2 ms | 0.75x |
| 1536 | 178 ms | 159 ms | 1.12x | 84.6 ms | 95.4 ms | 0.89x |
| 2048 | 437 ms | 306 ms | 1.43x | 196 ms | 169 ms | 1.16x |
| 3072 | 1.30 s | 0.71 s | 1.83x | 0.71 s | 0.43 s | 1.65x |
| 4096 | 3.44 s | 1.76 s | 1.95x | 2.00 s | 1.06 s | 1.88x |

The gain grows with $k$, as the CPU's reduction falls further behind memory
bandwidth. The M5 Pro uses the backend from $k = 1024$ with vectors and from
2048 for singular values alone, for a lone matrix only: it decomposes a batch
one matrix after another, and the CPU path spreads one over every core
(`bidiag_max_batch`). Up to 2048 it wins by a few percent to 1.4x. Accuracy
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

**Sizes.** The matrix lives in threadgroup memory, so the backend takes
squares up to 83×83 with 32 KB (`detail::svd_gk_max_k()`), longer matrices
when tall (444×16, 202×32). Inside its window the router runs it on the
matrix itself where it fits (`golub_kahan`) and otherwise after this
library's QR, on the $k \times k$ factor (`qr_golub_kahan`), as the Jacobi
kernels are preconditioned.

On an M5 Pro, $k \times k$ matrices in batches, from the routing sweep in
[`results/apple-m5-pro-20gpu/20261003-106b6c/svd/`](results/apple-m5-pro-20gpu/20261003-106b6c/svd/)
(min of two randomised passes; the CPU path spreads the batch over 18 cores):

| k×k | batch | golub_kahan | best Jacobi | CPU | Jacobi / GK | CPU / GK |
|---|---|---|---|---|---|---|
| 8 | 4096 | 0.93 ms | 1.41 ms | 1.42 ms | 1.51x | **1.52x** |
| 16 | 1024 | 0.86 ms | 1.76 ms | 1.16 ms | 2.03x | **1.34x** |
| 16 | 4096 | 2.41 ms | 5.76 ms | 4.42 ms | 2.39x | **1.84x** |
| 24 | 4096 | 5.67 ms | 14.1 ms | 9.43 ms | 2.50x | **1.66x** |
| 32 | 256 | 0.96 ms | 2.19 ms | 1.10 ms | 2.30x | **1.15x** |
| 32 | 1024 | 2.57 ms | 6.91 ms | 3.97 ms | 2.69x | **1.55x** |
| 32 | 4096 | 9.30 ms | 22.6 ms | 15.4 ms | 2.43x | **1.65x** |
| 40 | 4096 | 18.0 ms | 49.6 ms | 23.8 ms | 2.75x | **1.32x** |
| 48 | 1024 | 7.23 ms | 21.3 ms | 9.06 ms | 2.94x | **1.25x** |
| 48 | 4096 | 27.6 ms | 79.4 ms | 34.2 ms | 2.88x | **1.24x** |
| 56 | 4096 | 44.6 ms | 104 ms | 40.7 ms | 2.32x | 0.91x |
| 64 | 4096 | 73.0 ms | 114 ms | 53.3 ms | 1.56x | 0.73x |
| 80 | 4096 | 174 ms | 309 ms | 97.7 ms | 1.77x | 0.56x |

It is 1.5-2.9x faster than the best Jacobi kernel at every size it takes, and
ahead of the CPU for large batches up to 48×48. Beyond that the CPU pulls
away: the QR iteration is one thread's serial chain per matrix, and with the
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
GPU iff  k <= gpu_max_k,  l <= gpu_max_l,  batch * k >= gpu_min_batch_times_k  and  batch >= gpu_min_batch,  else CPU
on the GPU:
  golub_kahan iff  gk_min_k <= k <= gk_max_k  (clipped to svd_gk_max_k()):
                   on the matrix if it fits, else after QR (qr_golub_kahan)
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
| Apple M5 Pro | 20 | k <= 48, l <= 64 and batch * k >= 4096 | k = 4 .. 56 | 512 rows, k >= 32 | k = 192; k = 64 in batches of 64+ | from k = 1024 (svdvals 2048), lone matrices | measured — run [`20261003-106b6c`](results/apple-m5-pro-20gpu/20261003-106b6c/svd/report.md) |
| anything else | — | k <= 64 and batch * k >= 1024 | never | 512 rows, k >= 64 | k = 192 | never | **untuned default** |

On the M5 Pro large batches of small matrices, up to 48×48 and a long side
of 64, go to `golub_kahan` on the GPU, everything else in a batch to the CPU,
and one large matrix to `bidiag`. Against the best backend at each of the 295
points measured, the row scores 1.005 geometric-mean regret, worst 1.26x
(1.030 and 2.09x for the 2.9.0 row on the same data). The long-side cap
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
`SVD_GPU_MIN_BATCH`, `SVD_GPU_MAX_L`, `SVD_BIDIAG_MIN_K`, `SVD_VALUES_BIDIAG_MIN_K`,
`SVD_BIDIAG_MAX_BATCH`, `SVD_VALUES_BIDIAG_MAX_BATCH`, `SVD_GK_MIN_K`,
`SVD_GK_MAX_K` and `SVD_DEVICE=gpu|cpu|bidiag` override it.
`svd_backend(m, n, batch)` and `svdvals_backend(m, n, batch)` say which of the
eight backends a problem gets, with vectors and for singular values alone.

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

Apple M5 Pro (20 GPU cores, 18 CPU cores) with 2.10.0, from the routing sweep in
[`results/apple-m5-pro-20gpu/20261003-106b6c/svd/`](results/apple-m5-pro-20gpu/20261003-106b6c/svd/): min of two
randomised passes, on mains, against the CPU path as the library runs it, a
batch spread over all 18 cores. Each cell is the speedup of the fastest GPU
backend over the CPU, with the GPU time and the backend (`whole` and `block`
are the two Jacobi kernels on the matrix itself, `QR+` after preconditioning,
`GK` the golub_kahan backend); bold is where the GPU was ahead.

| shape | batch 1 | batch 4 | batch 16 | batch 64 | batch 256 | batch 4096 |
|---|---|---|---|---|---|---|
| 4×4 | 0.02x (0.18 ms, whole) | 0.10x (0.22 ms, GK) | 0.25x (0.16 ms, whole) | 0.64x (0.20 ms, GK) | 0.67x (0.21 ms, GK) | **1.83x** (0.38 ms, GK) |
| 8×8 | 0.05x (0.17 ms, whole) | 0.10x (0.23 ms, whole) | 0.41x (0.19 ms, whole) | 0.67x (0.22 ms, whole) | 0.81x (0.24 ms, GK) | **1.52x** (0.93 ms, GK) |
| 16×16 | 0.06x (0.34 ms, GK) | 0.18x (0.22 ms, whole) | 0.35x (0.27 ms, whole) | 0.56x (0.33 ms, whole) | **1.05x** (0.39 ms, GK) | **1.84x** (2.41 ms, GK) |
| 32×32 | 0.17x (0.34 ms, whole) | 0.26x (0.35 ms, whole) | 0.37x (0.51 ms, whole) | 0.51x (0.77 ms, GK) | **1.15x** (0.96 ms, GK) | **1.65x** (9.30 ms, GK) |
| 48×48 | 0.15x (0.79 ms, whole) | 0.20x (0.81 ms, whole) | 0.34x (0.87 ms, whole) | 0.56x (1.22 ms, GK) | 0.97x (2.38 ms, GK) | **1.24x** (27.6 ms, GK) |
| 64×64 | 0.19x (1.11 ms, whole) | 0.21x (1.09 ms, whole) | 0.35x (1.13 ms, whole) | 0.53x (1.97 ms, GK) | 0.69x (5.43 ms, GK) | 0.73x (73.0 ms, GK) |
| 128×128 | 0.21x (4.90 ms, whole) | 0.27x (4.93 ms, whole) | 0.34x (5.06 ms, whole) | 0.42x (14.4 ms, block) | 0.45x (48.2 ms, block) | 0.40x (749.6 ms, block) |
| 256×256 | 0.30x (13.0 ms, block) | 0.29x (14.2 ms, block) | 0.25x (24.9 ms, block) | 0.24x (83.1 ms, block) | 0.23x (346.8 ms, block) | -- |
| 512×512 | 0.52x (32.1 ms, block) | 0.38x (48.4 ms, block) | 0.18x (154.2 ms, block) | 0.16x (630.0 ms, block) | -- | -- |
| 1024×1024 | 0.67x (115.3 ms, block) | 0.32x (292.8 ms, block) | 0.25x (1.23 s, block) | -- | -- | -- |
| 64×8 | 0.06x (0.20 ms, whole) | 0.16x (0.19 ms, whole) | 0.35x (0.23 ms, GK) | 0.75x (0.22 ms, whole) | 0.97x (0.30 ms, GK) | **1.48x** (1.93 ms, GK) |
| 64×32 | 0.13x (0.52 ms, QR+whole) | 0.28x (0.38 ms, whole) | 0.49x (0.47 ms, whole) | 0.55x (0.85 ms, whole) | 0.93x (1.63 ms, GK) | **1.30x** (16.2 ms, GK) |
| 256×32 | 0.22x (0.51 ms, whole) | 0.33x (0.48 ms, whole) | 0.62x (0.52 ms, whole) | 0.53x (1.34 ms, whole) | 0.69x (3.92 ms, GK) | 0.84x (38.4 ms, GK) |
| 1024×32 | 0.32x (0.70 ms, QR+whole) | 0.45x (0.85 ms, QR+whole) | 0.51x (1.15 ms, whole) | 0.61x (3.46 ms, QR+whole) | 0.61x (10.1 ms, GK) | 0.61x (135.6 ms, GK) |
| 1024×64 | 0.37x (1.58 ms, QR+whole) | 0.52x (2.15 ms, QR+whole) | 0.62x (3.41 ms, QR+whole) | 0.61x (9.44 ms, GK) | 0.65x (29.2 ms, GK) | 0.61x (431.8 ms, QR+block) |
| 2048×64 | 0.48x (1.88 ms, QR+whole) | 0.56x (3.50 ms, QR+whole) | 0.63x (4.91 ms, QR+whole) | 0.66x (14.9 ms, QR+whole) | 0.66x (53.3 ms, GK) | 0.69x (732.2 ms, QR+block) |
| 1024×256 | 0.46x (14.2 ms, QR+block) | 0.46x (16.1 ms, QR+block) | 0.44x (28.0 ms, QR+block) | 0.44x (96.5 ms, QR+block) | 0.41x (391.6 ms, QR+block) | -- |
| 2048×256 | 0.55x (16.3 ms, QR+block) | 0.55x (19.1 ms, QR+block) | 0.53x (36.6 ms, QR+block) | 0.51x (137.7 ms, QR+block) | 0.52x (515.8 ms, QR+block) | -- |

`--` was not measured.

Against a CPU that uses its cores, the GPU wins large batches of small
matrices, which since 2.10.0 is `golub_kahan`'s: 1.2-1.8x from 4×4 to 48×48
at 4096 matrices, and from 256 matrices for 16×16 and 32×32, where the Jacobi
kernels had won only at 4×4 and 64×8. Lone matrices and small batches stay
the CPU's, as do tall shapes and k from 56 up; one large matrix wins on
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

314 checks: square, tall and wide shapes around the simdgroup, pair-count and
block boundaries, batches, every simdgroup count, all seven GPU backends and
each branch of the CPU one (the `bidiag` backend from 1×1 to 1024×1024, with
QR first and through the transpose, with vectors and without; `golub_kahan`
from 1×1 to its limit, either side of its chaser simdgroup, directly and
after a QR), rank deficiency repeated over random instances per shape and
backend, structured spectra (graded columns, singular values from 1e+4 to
1e-4, repeated values), magnitudes from 1e-30 to 1e+37, NaN inside a batch,
and the routing policy, including the batch-dependent kernel crossover, the
golub_kahan window and the long-side cap, without assuming any device's
values.

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
