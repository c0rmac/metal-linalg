# Singular value decomposition (`svd`)

```cpp
#include <metal_linalg/svd.h>

// a: MLX array of shape [M, N] or [..., M, N] (float32 or auto-cast), any M and N
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

## Two kernels, four GPU backends

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

## Routing

With $k = \min(M, N)$ and $l = \max(M, N)$:

```
GPU iff  k <= gpu_max_k,  batch * k >= gpu_min_batch_times_k  and  batch >= gpu_min_batch,  else CPU
on the GPU:
  precondition with QR iff  l >= qr_min_rows,  k >= qr_min_k  and  l >= 2k
  block kernel iff  k >= block_min_k,  or  k >= block_min_k_batched and batch >= block_min_batch
```

The CPU path is a fair one: MLX's `svd` through Accelerate, preceded by a thin
QR when the matrix is tall, so that it too computes thin factors only.

As for QR and the eigensolver, the policy is a per-device table, keyed on the
Metal device name and GPU core count:

| GPU | cores | QR from | block from | GPU iff | status |
|---|---|---|---|---|---|
| Apple M5 Pro | 20 | 512 rows, k >= 32 | k = 192; k = 64 in batches of 64+ | k <= 1024, batch * k >= 512 and batch >= 4 | measured — see [`studies/routing-apple-m5-pro.md`](studies/routing-apple-m5-pro.md) |
| anything else | — | 512 rows, k >= 64 | k = 192 | k <= 64 and batch * k >= 1024 | **untuned default** |

The M1 has no row. The defaults come from measurements on an M1 taken while
the machine was heavily loaded by other jobs, good enough to place the
crossovers roughly and not for a table entry, so `svd_policy_source()`
reports `default:untuned-device` there too. Tuning is one command, on an idle
machine; see [`tuning.md`](tuning.md):

```sh
cmake --build build --target sweep_svd
python3 tuning/tune_svd.py build/sweep_svd --max-k 1024
```

`set_svd_policy()` and the environment variables `SVD_QR_MIN_ROWS`,
`SVD_QR_MIN_K`, `SVD_BLOCK_MIN_K`, `SVD_BLOCK_MIN_K_BATCHED`,
`SVD_BLOCK_MIN_BATCH`, `SVD_GPU_MAX_K`, `SVD_GPU_MIN_BATCH_TIMES_K`,
`SVD_GPU_MIN_BATCH` and `SVD_DEVICE=gpu|cpu` override it.
`svd_backend(m, n, batch)` says which of the five backends a problem gets.

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

Apple M5 Pro (20 GPU cores, 18 CPU cores), from the routing sweep in
[`results/svd-apple-m5-pro/`](results/svd-apple-m5-pro/): min of two
randomised passes, idle machine on mains. Each cell is the speedup of the
fastest GPU backend over a thin SVD on the CPU, with the GPU time and the
backend that won (`whole` and `block` are the two kernels on the matrix
itself, `QR+` after preconditioning); bold is where the GPU was ahead.

| shape | batch 1 | batch 4 | batch 16 | batch 64 | batch 256 | batch 4096 |
|---|---|---|---|---|---|---|
| 4×4 | 0.10x (0.26 ms, whole) | 0.14x (0.26 ms, whole) | 0.31x (0.20 ms, whole) | 0.59x (0.25 ms, whole) | **2.26x** (0.23 ms, whole) | **6.65x** (1.17 ms, whole) |
| 8×8 | 0.17x (0.20 ms, whole) | 0.21x (0.20 ms, whole) | 0.28x (0.35 ms, whole) | **1.51x** (0.23 ms, whole) | **4.44x** (0.27 ms, whole) | **13.2x** (1.43 ms, whole) |
| 16×16 | 0.13x (0.31 ms, whole) | 0.18x (0.49 ms, whole) | 0.32x (0.84 ms, whole) | **2.87x** (0.34 ms, whole) | **2.75x** (1.36 ms, whole) | **10.5x** (5.70 ms, whole) |
| 32×32 | 0.22x (0.35 ms, whole) | 0.67x (0.39 ms, whole) | 0.77x (1.19 ms, whole) | **3.82x** (0.89 ms, whole) | **6.07x** (2.22 ms, whole) | **9.41x** (22.8 ms, block) |
| 64×64 | 0.21x (1.09 ms, whole) | 0.68x (1.15 ms, whole) | **1.54x** (1.89 ms, whole) | **3.59x** (3.16 ms, whole) | **4.75x** (9.66 ms, block) | **6.46x** (114 ms, block) |
| 128×128 | 0.18x (4.84 ms, whole) | 0.68x (4.90 ms, whole) | **2.61x** (5.04 ms, whole) | **3.56x** (14.8 ms, block) | **4.43x** (47.5 ms, block) | GPU only (737 ms, block) |
| 256×256 | 0.30x (13.5 ms, block) | **1.07x** (14.6 ms, block) | **2.49x** (24.8 ms, block) | **2.99x** (83.4 ms, block) | GPU only (339 ms, block) | -- |
| 512×512 | 0.51x (32.3 ms, block) | **1.37x** (47.9 ms, block) | **1.68x** (155 ms, block) | GPU only (624 ms, block) | -- | -- |
| 1024×1024 | 0.69x (114 ms, block) | GPU only (290 ms, block) | -- | -- | -- | -- |
| 64×8 | 0.26x (0.20 ms, whole) | 0.19x (0.36 ms, whole) | 0.45x (0.38 ms, whole) | **2.33x** (0.23 ms, whole) | **4.39x** (0.45 ms, whole) | **15.8x** (1.98 ms, whole) |
| 256×32 | 0.14x (1.27 ms, whole) | 0.77x (0.73 ms, whole) | **1.57x** (1.28 ms, whole) | **6.05x** (1.28 ms, whole) | **6.03x** (5.16 ms, whole) | **10.4x** (48.9 ms, block) |
| 1024×32 | 0.35x (0.98 ms, whole) | **1.27x** (1.01 ms, whole) | **4.39x** (1.12 ms, whole) | **4.00x** (4.90 ms, QR+whole) | **6.06x** (13.3 ms, block) | **8.87x** (146 ms, block) |
| 1024×64 | 0.39x (2.85 ms, QR+whole) | **1.39x** (3.10 ms, QR+whole) | **4.29x** (4.01 ms, QR+whole) | **6.72x** (10.4 ms, QR+whole) | **8.84x** (32.2 ms, block) | GPU only (491 ms, QR+block) |
| 2048×64 | 0.59x (3.81 ms, QR+whole) | **2.03x** (4.33 ms, QR+whole) | **6.16x** (5.80 ms, QR+whole) | **8.35x** (17.3 ms, QR+whole) | **10.1x** (56.6 ms, block) | GPU only (966 ms, QR+whole) |
| 1024×256 | 0.54x (15.8 ms, QR+block) | **1.75x** (18.6 ms, QR+block) | **4.49x** (29.8 ms, QR+block) | **5.20x** (103 ms, QR+block) | GPU only (408 ms, QR+block) | -- |
| 2048×256 | 0.66x (19.8 ms, QR+block) | **2.29x** (22.7 ms, QR+block) | **5.38x** (38.9 ms, QR+block) | **5.89x** (142 ms, QR+block) | GPU only (555 ms, QR+block) | -- |

"GPU only" is a point where the CPU was not timed because it would take
seconds; `--` was not measured.

The same limit as the eigensolver applies: the GPU wins for batches, and a
single matrix of any size is faster on the CPU, 0.51x at 512×512 and still
0.79x at 4096×4096 (4.7 s against 3.7 s). What the block kernel changed is the
size at which a *batch* pays: with the whole-matrix kernel alone the win
stopped at k = 128; with it 512×512 wins from batch 4. Tall input through the
QR path is the strongest region, 6 to 10x from batch 64. Small batched
matrices reach 13 to 16x, less than the eigensolver's 20x, because one-sided
Jacobi needs 10 sweeps where the two-sided method needs 7.

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

197 checks: square, tall and wide shapes around the simdgroup, pair-count and
block boundaries, batches, every simdgroup count, all four GPU backends and
each branch of the CPU one, rank deficiency repeated over twelve random
instances per shape and backend, structured spectra (graded columns, singular
values from 1e+4 to 1e-4, repeated values), magnitudes from 1e-30 to 1e+37,
NaN inside a batch, and the routing policy, including the batch-dependent
kernel crossover, without assuming any device's values.

## References

- M. R. Hestenes, ["Inversion of matrices by biorthogonalization and related results"](https://doi.org/10.1137/0106005), *J. SIAM* 6(1), 1958 — one-sided Jacobi.
- Z. Drmač and K. Veselić, ["New fast and accurate Jacobi SVD algorithm I"](https://doi.org/10.1137/050639193), *SIAM J. Matrix Anal. Appl.* 29(4), 2008 — the modern form, preconditioning with QR, and what LAPACK's `xGESVJ` / `xGEJSV` implement.
- J. Demmel and K. Veselić, ["Jacobi's method is more accurate than QR"](https://epubs.siam.org/doi/10.1137/0613074), 1992.
- R. P. Brent and F. T. Luk, 1985 — the parallel ordering; see the eigensolver's references.
