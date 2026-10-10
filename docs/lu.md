# LU factorization, solve and inverse (`lu_factor`, `solve`, `inv`)

```cpp
#include <metal_linalg/lu.h>

// a: MLX array [N, N] or [..., N, N] (real: float32, or cast to it; complex input throws)
auto [LU, pivots] = metal_linalg::lu_factor_accelerated(a);   // P a = L U, as mx::linalg::lu_factor
mlx::core::array x = metal_linalg::solve_accelerated(a, b);   // a x = b; b [..., N, K] or [..., N]
mlx::core::array ai = metal_linalg::inv_accelerated(a);       // a^-1

// With LAPACK's info for each matrix: 0, or k where U(k-1, k-1) is exactly zero.
metal_linalg::LuResult r = metal_linalg::lu_factor_ex_accelerated(a);
metal_linalg::SolveResult s = metal_linalg::solve_ex_accelerated(a, b);
```

Since 2.18.0. MLX's own `linalg::lu_factor`, `solve` and `inv` run only on the
CPU. As for the other decompositions, each call is routed by a policy measured
on the Mac it runs on: to LAPACK on every CPU core, or to the GPU path for
large matrices. `lu_backend(n, batch)` reports which (`cpu` or `blocked`). The
same is in the C API (`metal_linalg_lu_factor`, `_solve`, `_inv`), Python
(`metal_linalg.lu_factor`, `solve`, `inv` and their `_ex` forms), PyTorch
(`metal_linalg_torch.lu_factor`, `lu_factor_ex`, `solve`, `solve_ex`, `inv`,
`inv_ex`, with torch's gradients for `solve` and `inv`, and `torch.compile`)
and Swift (`luFactorAccelerated`, `solveAccelerated`, `invAccelerated`).

Conventions are MLX's and LAPACK's: `LU` holds `U` on and above the diagonal
and `L` below it with its unit diagonal implied; the pivots are the row swaps
in order, row `i` swapped with row `pivots[i]` (0-based, uint32; the PyTorch
package returns torch's 1-based int32). A singular matrix (a zero on `U`'s
diagonal) is still factored, as by `sgetrf`; its solve and inverse are all NaN
and its `info` says where, the rest of the batch unaffected. The PyTorch
package's `solve` and `inv` raise `torch.linalg.LinAlgError` instead, as
torch's do.

## The factorization

Gaussian elimination with partial pivoting: $P A = L U$, with $L$ unit lower
triangular, every entry at most 1 in magnitude, and $U$ upper triangular.
$2N^3/3$ flops. The blocked right-looking form takes $b$ columns at a time:
factor the $N \times b$ panel with its pivot search, apply its row swaps to the
other columns, form $U_{12} = L_{11}^{-1} A_{12}$ and update the trailing matrix
$A_{22} \leftarrow A_{22} - L_{21} U_{12}$, a matrix product that is nearly all
of the work. The panel is a chain of column reductions (each pivot is the
largest entry left in its column), which no GPU launch can make cheap.

## The backends

### `cpu`: LAPACK on every core

Accelerate's `sgetrf` (and `sgetrs`, `sgetri`) on a column-major copy whose
leading dimension is padded off a power of two: on macOS 27 an unpadded
leading dimension of 4096 cost `sgetrf` 2.7x (169 ms against 62). A batch's
matrices are spread over every core; one large matrix (above 1024) gets
Accelerate's own threads, and the copies in and out run on every core.

### `blocked`: the GPU and the CPU on one matrix

One matrix at a time, in a column-major workspace in memory the CPU and the GPU
share (Apple Silicon's unified memory: nothing is copied between them), in
panels of 128 columns:

- the CPU factors each panel in place (recursively: halves, their updates as
  `strsm` and `sgemm` on Accelerate's matrix units, leaves of 8 columns to
  `sgetrf`; 10-25% faster than `sgetrf` on the whole panel, the same pivots),
  and makes $L_{11}^{-1}$ (`strtri`);
- the GPU applies the panel's row swaps to the other columns (`lu_gather`:
  the panel's sequential swaps composed into one permutation of the rows they
  touch, so every entry moves at once), forms $U_{12}$ by an MPS product with
  $L_{11}^{-1}$ and updates the trailing matrix by another;
- with a look-ahead of one panel: the CPU brings the next panel's 128 columns
  up to date itself (LAPACK's own step: `slaswp`, `strsm`, one `sgemm`) and
  factors it while the GPU updates everything to the right of it.

The CPU's critical path never waits on a GPU launch; with the next panel's
update on the GPU instead (four launches a panel), one 4096 x 4096 took 20 ms
rather than 18. The matrix moves in and out by transposes on the GPU. Column-
major blocks of the workspace read row-major are their transposes, so the
products are written in transposed form, and nothing is copied to change
layout.

**Solve and inverse.** $A X = B$ is $X = U^{-1} L^{-1} P B$: $P B$ by a gather
on the GPU, then both triangular solves 128 rows at a time on the GPU, each
block's step two MPS products (one with the inverse of the diagonal block,
$L_{11}^{-1}$ kept from the factorization and $U_{11}^{-1}$ made by `strtri`,
one updating the rows still to come); the inverse is the same on $P I$. With
fewer right-hand sides than `gpu_solve_min_rhs`, LAPACK's `sgetrs` solves on
the factorization where it lies instead.

Accuracy: the products with $L_{11}^{-1}$ in place of triangular solves cost
some digits against LAPACK (P A - L U at 4096: 5e-5 of max|A| against 1.6e-5),
less with 128-column panels than 256 (1.7e-4), which is why the panels are 128.

## Dispatch

The GPU iff $N \ge$ `gpu_min_n` (0: never) in a batch of at most
`gpu_max_batch` (0: any), for `lu_factor`, `solve` and `inv` alike; the GPU
path's own triangular solves from `gpu_solve_min_rhs` right-hand sides. On the
M5 Pro (run [`20261010-22fef9`](results/apple-m5-pro-20gpu/20261010-22fef9/lu/report.md)):

```
GPU from N = 1536, any batch; its triangular solves on the GPU from 256 right-hand sides
```

`LU_DEVICE=cpu` or `=gpu` forces it; `LU_GPU_MIN_N`, `LU_GPU_MAX_BATCH` and
`LU_GPU_SOLVE_MIN_RHS` set the policy's fields.

## Performance

Apple M5 Pro (20 GPU cores, 18 CPU cores), 2.18.0, from the routing sweep:
medians in ms, the faster of two passes; the GPU path's solve as routed (one
right-hand side: `sgetrs` on its factorization).

| N | batch | lu_factor CPU | GPU | | inv CPU | GPU | | solve (1 rhs) CPU | GPU | |
|---|---|---|---|---|---|---|---|---|---|---|
| 1536 | 1 | 3.23 | **3.13** | 1.03x | 7.51 | **5.20** | 1.44x | | | |
| 2048 | 1 | 6.73 | **4.93** | 1.37x | 16.9 | **8.60** | 1.97x | **7.09** | 7.32 | 0.97x |
| 2048 | 4 | 26.9 | **19.7** | 1.37x | 68.6 | **34.0** | 2.01x | | | |
| 3072 | 1 | 21.2 | **10.3** | 2.05x | 59.5 | **20.3** | 2.93x | | | |
| 4096 | 1 | 47.9 | **16.0** | 2.98x | 137 | **40.0** | 3.44x | 50.4 | **18.5** | 2.73x |
| 4096 | 2 | 95.3 | **32.1** | 2.97x | 274 | **80.9** | 3.39x | | | |

Up to 1024 the CPU path wins every batch (one 1024 x 1024: 1.33 ms against
1.80; the GPU path takes one matrix at a time). At 8192, `lu_factor` 96 ms
against 427 (4.4x).

Against MLX's own functions (CPU only, at an unpadded leading dimension;
`benchmark_lu`): one 4096 x 4096 `lu_factor` in 16 ms against 130 (8.1x),
`solve` 18 against 277 (15x), `inv` 40 against 145 (3.6x); at 2048, 4.5x,
8x and 2.5x. For batches of small matrices the CPU path is 5-24x MLX's (4096
of 8 x 8: `lu_factor` 0.15 ms against 0.88, `inv` 0.28 against 3.1, `solve`
0.13 against 3.2).

## Tuning

`tuning/tune_lu.py` (run by `tuning/run.py`; `--only lu`, about 3 minutes on
an M5 Pro): `lu_factor` and `inv` on both paths for N from 32 to 4096 at
batches from 1 to 1024 (the GPU path where it takes a batch under a second),
and `solve` from 1024 to 4096 with 1 to 256 right-hand sides on the CPU path
and on the GPU path with each way of solving. The GPU-or-CPU rule is fitted on
the `lu_factor` and `inv` points together; on the M5 Pro 1.0001x
geometric-mean regret, worst 1.01x, over 66 points (always the CPU: 1.21x,
worst 3.44x).

## Tests

```sh
cmake --build build --target test_lu
./build/test_lu
```

491 checks: both backends directly and through the routing, from 1 x 1 to
1100 x 1100 (either side of the 128-column panels), batched: P A = L U with
|L| <= 1 and the pivots valid swaps; `solve` with 1, 3, 16 and 40 right-hand
sides and with a vector `b`; `inv` against the identity and against MLX's own;
a singular matrix in a batch of three (its `info`, NaN results for it alone);
batch axes, empties, 0 x 0, a transposed view, mismatched shapes; the routing
policy and `LU_DEVICE`. `test_core`, `test_c_api` and the Python, PyTorch
(values against torch on the CPU and MPS, `lu_unpack`, `LinAlgError`,
gradients against torch's, `opcheck`, `torch.compile`) and Swift suites cover
their layers.

## References

- G. H. Golub and C. F. Van Loan, *Matrix Computations*, 4th ed., s3.2-3.4
  (Gaussian elimination, pivoting, the block forms).
- LAPACK `sgetrf`, `sgetrf2` (the recursive panel), `sgetrs`, `sgetri`, `slaswp`.
- J. Kurzak and J. Dongarra, "Implementing linear algebra routines on
  multi-core processors with pipelining and a look ahead" (2006); the MAGMA
  hybrid LU (Tomov, Nath, Ltaief, Dongarra 2010): CPU panels, GPU updates,
  look-ahead.
