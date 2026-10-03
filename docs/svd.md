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

## Routing

With $k = \min(M, N)$ and $l = \max(M, N)$:

```
GPU iff  k <= gpu_max_k,  batch * k >= gpu_min_batch_times_k  and  batch >= gpu_min_batch,  else CPU
on the GPU:
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

| GPU | cores | QR from | block from | GPU iff | else bidiag | status |
|---|---|---|---|---|---|---|
| Apple M5 Pro | 20 | 512 rows, k >= 32 | k = 192; k = 64 in batches of 64+ | k <= 8 and batch * k >= 4096 | from k = 1024 (svdvals 2048), lone matrices | measured — run [`20261003-2d2c19`](results/apple-m5-pro-20gpu/20261003-2d2c19/svd/report.md) |
| anything else | — | 512 rows, k >= 64 | k = 192 | k <= 64 and batch * k >= 1024 | never | **untuned default** |

The M5 Pro row is the first measured against the CPU path that spreads a
batch over every core (2.9.0), and against it the Jacobi kernels win only for
large batches of the smallest matrices (see [Performance](#performance)):
the row sends almost every batch to the CPU, and one large matrix to `bidiag`.
Before 2.9.0 it sent batches up to k = 1024 to the GPU, measured against one
CPU core.

The M1 has no row. The defaults come from measurements on an M1 taken while
the machine was heavily loaded by other jobs, good enough to place the
crossovers roughly and not for a table entry, so `svd_policy_source()`
reports `default:untuned-device` there too. Measuring a Mac is one command,
`python3 tuning/run.py`, which covers all three decompositions; see
[`tuning.md`](tuning.md).

`set_svd_policy()` and the environment variables `SVD_QR_MIN_ROWS`,
`SVD_QR_MIN_K`, `SVD_BLOCK_MIN_K`, `SVD_BLOCK_MIN_K_BATCHED`,
`SVD_BLOCK_MIN_BATCH`, `SVD_GPU_MAX_K`, `SVD_GPU_MIN_BATCH_TIMES_K`,
`SVD_GPU_MIN_BATCH`, `SVD_BIDIAG_MIN_K`, `SVD_VALUES_BIDIAG_MIN_K`,
`SVD_BIDIAG_MAX_BATCH`, `SVD_VALUES_BIDIAG_MAX_BATCH` and
`SVD_DEVICE=gpu|cpu|bidiag` override it. `svd_backend(m, n, batch)` and
`svdvals_backend(m, n, batch)` say which of the six backends a problem gets,
with vectors and for singular values alone.

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

Apple M5 Pro (20 GPU cores, 18 CPU cores) with 2.9.0, from the routing sweep in
[`results/apple-m5-pro-20gpu/20261003-2d2c19/svd/`](results/apple-m5-pro-20gpu/20261003-2d2c19/svd/): min of two
randomised passes, on mains, against the CPU path as the library runs it, a
batch spread over all 18 cores. Each cell is the speedup of the fastest GPU
backend over the CPU, with the GPU time and the backend (`whole` and `block`
are the two kernels on the matrix itself, `QR+` after preconditioning); bold
is where the GPU was ahead.

| shape | batch 1 | batch 4 | batch 16 | batch 64 | batch 256 | batch 4096 |
|---|---|---|---|---|---|---|
| 4×4 | 0.02x (0.18 ms, whole) | 0.10x (0.24 ms, whole) | 0.17x (0.26 ms, whole) | 0.82x (0.17 ms, whole) | 0.64x (0.20 ms, whole) | **1.68x** (0.43 ms, whole) |
| 8×8 | 0.04x (0.20 ms, whole) | 0.13x (0.18 ms, whole) | 0.40x (0.19 ms, whole) | 0.58x (0.21 ms, whole) | 0.73x (0.26 ms, whole) | **1.01x** (1.42 ms, whole) |
| 16×16 | 0.09x (0.22 ms, whole) | 0.15x (0.28 ms, whole) | 0.38x (0.25 ms, whole) | 0.56x (0.32 ms, whole) | 0.63x (0.65 ms, whole) | 0.76x (5.81 ms, whole) |
| 32×32 | 0.19x (0.33 ms, whole) | 0.26x (0.36 ms, whole) | 0.50x (0.39 ms, whole) | 0.46x (0.86 ms, whole) | 0.51x (2.19 ms, whole) | 0.69x (22.5 ms, block) |
| 64×64 | 0.19x (1.07 ms, whole) | 0.22x (1.09 ms, whole) | 0.34x (1.18 ms, whole) | 0.32x (3.24 ms, whole) | 0.42x (9.07 ms, block) | 0.48x (116 ms, block) |
| 128×128 | 0.21x (4.92 ms, whole) | 0.27x (4.95 ms, whole) | 0.34x (5.05 ms, whole) | 0.42x (14.5 ms, block) | 0.45x (48.1 ms, block) | 0.40x (755 ms, block) |
| 256×256 | 0.30x (13.1 ms, block) | 0.30x (14.2 ms, block) | 0.26x (24.7 ms, block) | 0.24x (84.5 ms, block) | 0.22x (354 ms, block) | -- |
| 512×512 | 0.52x (32.3 ms, block) | 0.38x (48.7 ms, block) | 0.18x (159 ms, block) | 0.16x (646 ms, block) | -- | -- |
| 1024×1024 | 0.67x (117 ms, block) | 0.31x (307 ms, block) | 0.25x (1.22 s, block) | -- | -- | -- |
| 64×8 | 0.06x (0.20 ms, whole) | 0.17x (0.18 ms, whole) | 0.43x (0.22 ms, whole) | 0.71x (0.23 ms, whole) | 0.98x (0.31 ms, whole) | **1.42x** (2.02 ms, whole) |
| 256×32 | 0.25x (0.47 ms, whole) | 0.33x (0.49 ms, whole) | 0.63x (0.52 ms, whole) | 0.54x (1.34 ms, whole) | 0.58x (4.77 ms, block) | 0.64x (51.4 ms, block) |
| 1024×32 | 0.32x (0.69 ms, QR+whole) | 0.45x (0.87 ms, QR+whole) | 0.54x (1.15 ms, whole) | 0.44x (4.61 ms, QR+whole) | 0.47x (13.0 ms, block) | 0.53x (161 ms, block) |
| 1024×64 | 0.38x (1.55 ms, QR+whole) | 0.52x (2.14 ms, QR+whole) | 0.50x (3.56 ms, QR+whole) | 0.54x (10.6 ms, QR+whole) | 0.57x (33.2 ms, block) | 0.56x (483 ms, block) |
| 2048×64 | 0.48x (1.92 ms, QR+whole) | 0.57x (3.47 ms, QR+whole) | 0.57x (5.57 ms, QR+whole) | 0.53x (18.2 ms, QR+whole) | 0.59x (59.7 ms, block) | 0.47x (1.09 s, QR+block) |
| 1024×256 | 0.48x (13.6 ms, QR+block) | 0.43x (17.9 ms, QR+block) | 0.40x (30.1 ms, QR+block) | 0.40x (108 ms, QR+block) | 0.38x (432 ms, QR+block) | -- |
| 2048×256 | 0.57x (15.9 ms, QR+block) | 0.47x (22.5 ms, QR+block) | 0.50x (39.9 ms, QR+block) | 0.48x (149 ms, QR+block) | 0.46x (586 ms, QR+block) | -- |

`--` was not measured.

Against a CPU that uses its cores, the Jacobi kernels lose nearly everywhere:
one-sided Jacobi does several times the flops of LAPACK's method, as the
eigensolver's Jacobi kernels do, and needs more sweeps than they do. They win
only for large batches of the smallest matrices (1.4-1.7x at 4096 of 4×4 and
64×8), and one large matrix wins on `bidiag`. The tables in this document's
earlier versions, with up to 16x for batches, were against one CPU core. The
eigensolver's answer, a batched kernel doing LAPACK's method (`ql`, see
[eigh.md](eigh.md)), has an SVD counterpart, Householder bidiagonalization and
implicit bidiagonal QR in one threadgroup, which has not been built.

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

229 checks: square, tall and wide shapes around the simdgroup, pair-count and
block boundaries, batches, every simdgroup count, all five GPU backends and
each branch of the CPU one (the `bidiag` backend from 1×1 to 1024×1024, with
QR first and through the transpose, with vectors and without), rank deficiency repeated over twelve random
instances per shape and backend, structured spectra (graded columns, singular
values from 1e+4 to 1e-4, repeated values), magnitudes from 1e-30 to 1e+37,
NaN inside a batch, and the routing policy, including the batch-dependent
kernel crossover, without assuming any device's values.

## References

- M. R. Hestenes, ["Inversion of matrices by biorthogonalization and related results"](https://doi.org/10.1137/0106005), *J. SIAM* 6(1), 1958 — one-sided Jacobi.
- Z. Drmač and K. Veselić, ["New fast and accurate Jacobi SVD algorithm I"](https://doi.org/10.1137/050639193), *SIAM J. Matrix Anal. Appl.* 29(4), 2008 — the modern form, preconditioning with QR, and what LAPACK's `xGESVJ` / `xGEJSV` implement.
- J. Demmel and K. Veselić, ["Jacobi's method is more accurate than QR"](https://epubs.siam.org/doi/10.1137/0613074), 1992.
- R. P. Brent and F. T. Luk, 1985 — the parallel ordering; see the eigensolver's references.
- G. H. Golub and W. Kahan, ["Calculating the singular values and pseudo-inverse of a matrix"](https://doi.org/10.1137/0702016), *J. SIAM Ser. B Numer. Anal.* 2(2), 1965 — reduction to bidiagonal form by Householder reflections, the method of the `bidiag` backend.
- J. J. Dongarra, S. J. Hammarling and D. C. Sorensen, ["Block reduction of matrices to condensed forms for eigenvalue computations"](https://doi.org/10.1016/0377-0427(89)90367-1), *J. Comput. Appl. Math.* 27(1-2), 1989 — the blocked bidiagonalization (LAPACK's `sgebrd` and `slabrd`) the backend runs on the GPU.
- M. Gu and S. C. Eisenstat, ["A divide-and-conquer algorithm for the bidiagonal SVD"](https://doi.org/10.1137/S0895479892242232), *SIAM J. Matrix Anal. Appl.* 16(1), 1995 — LAPACK's `sbdsdc`, which solves the bidiagonal problem.
- S. Tomov, R. Nath and J. Dongarra, ["Accelerating the reduction to upper Hessenberg, tridiagonal, and bidiagonal forms through hybrid GPU-based computing"](https://doi.org/10.1016/j.parco.2010.06.001), *Parallel Computing* 36(12), 2010 — the hybrid CPU/GPU split (MAGMA) the backend follows, with the panel kept on the GPU.
- E. Ringoot, R. Alomairy, V. Churavy and A. Edelman, ["Performant unified GPU kernels for portable singular value computation across hardware and precision"](https://doi.org/10.1145/3754598.3754667), 2025 ([arXiv:2508.06339](https://arxiv.org/abs/2508.06339)), and E. Ringoot, R. Alomairy and A. Edelman, ["Accelerating bidiagonalization of banded matrices through memory-aware bulge-chasing on GPUs"](https://arxiv.org/abs/2510.12705), 2025 — GPU singular value computation, including on Apple's GPUs; they prompted the `bidiag` backend, whose GPU-resident panel follows their design. Their kernels are not used here.
