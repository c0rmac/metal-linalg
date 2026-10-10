# Triangular solve (`solve_triangular`)

```cpp
#include <metal_linalg/triangular.h>

// a: MLX array [N, N] or [..., N, N], triangular; b [..., N, K] or [..., N] with a's batch shape.
// Only a's lower triangle is read (the upper with upper = true); with unit_diagonal its
// diagonal is taken as ones and not read.
mlx::core::array x = metal_linalg::solve_triangular_accelerated(a, b);              // a x = b, a lower
mlx::core::array y = metal_linalg::solve_triangular_accelerated(u, b, /*upper=*/true);
```

Since 2.18.0, as MLX's `linalg::solve_triangular` (CPU only) and
`torch.linalg.solve_triangular` (left-hand solves; `unitriangular` as
`unit_diagonal`). In the C API `metal_linalg_solve_triangular`, Python
`metal_linalg.solve_triangular`, PyTorch `metal_linalg_torch.solve_triangular`
(with torch's gradient), Swift `solveTriangularAccelerated`. With
[`cholesky`](cholesky.md) it solves a positive definite system,
$A = L L^T$: $x = L^{-T} (L^{-1} b)$, two calls.

## The backends

- **`cpu`**: BLAS's `strsm` (Accelerate) on the row-major matrices as they are,
  a batch spread over every core.
- **`blocked`** (the GPU): one matrix at a time, 128 rows of $X$ at a time
  (forward for a lower triangle, backward for an upper): $X_i = A_{ii}^{-1}
  B'_i$, then $B' \leftarrow B' - A_{\text{rest},i} X_i$ for the rows still to
  come, two MPS products a block, reading the caller's matrices in place (the
  row-major blocks of $A$ are what MPS reads). The inverses of $A$'s 128 x 128
  diagonal blocks are made on the CPU (`strtri`, on every core). $N^2 K$
  multiply-adds, nearly all in the second product.

## Dispatch

The GPU iff $N \ge$ `gpu_min_n`, $K \ge$ `gpu_min_rhs` and the batch is at
most `gpu_max_batch` (0: any). On the M5 Pro (run
[`20261010-b7aec3`](results/apple-m5-pro-20gpu/20261010-b7aec3/trsm/report.md)):
from N = 2048 with at least 1024 right-hand sides, in batches of up to 4;
1.0055x geometric-mean regret over 92 points against the faster path at each
(always the CPU: 1.052x, worst 3.99x). `TRSM_DEVICE=cpu` or `=gpu` forces it;
`TRSM_GPU_MIN_N`, `TRSM_GPU_MIN_RHS` and `TRSM_GPU_MAX_BATCH` set the fields.

## Performance

Apple M5 Pro, 2.18.0, one lower triangle, medians in ms (min of two passes):

| N | K | CPU (strsm) | GPU | |
|---|---|---|---|---|
| 2048 | 1024 | 3.37 | **1.81** | 1.86x |
| 3072 | 1024 | 6.47 | **2.85** | 2.27x |
| 4096 | 1024 | 10.3 | **4.35** | 2.37x |
| 4096 | 4096 | 50.0 | **12.5** | 3.99x |
| 4096 | 1 | 3.54 | **2.38** | 1.49x (routed to the CPU: K < 1024) |
| 2048 | 1 | **0.62** | 1.20 | 0.52x |

Against MLX's own (`benchmark_trsm`): one 4096 x 4096 with 4096 right-hand
sides in 12.7 ms against 100 (7.9x); with one, 3.6 against 45 (12.7x).

## Tests

`test_trsm` (204 checks): both backends and the routing, lower and upper, with
and without a unit diagonal (junk in the triangle not read), N from 1 to 1100
and K from 1 to 300, batched, against MLX's own and by residual; a vector b,
batch axes, empties, errors; the policy and `TRSM_DEVICE`. The C, Python,
PyTorch (gradients against torch's, `opcheck`) and Swift suites cover their
layers.
