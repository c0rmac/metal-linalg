# Cholesky factorization (`cholesky`)

```cpp
#include <metal_linalg/cholesky.h>

// a: MLX array [N, N] or [..., N, N], symmetric positive definite (real:
// float32, or cast to it; complex input throws). Only the lower triangle is
// read; with upper = true the upper one, and U = L^T is returned.
mlx::core::array L = metal_linalg::cholesky_accelerated(a);            // a = L L^T
mlx::core::array U = metal_linalg::cholesky_accelerated(a, true);      // a = U^T U

// With info [...] (uint32), as torch.linalg.cholesky_ex: 0, or k where the
// leading minor of order k is not positive definite (LAPACK's spotrf).
auto [L2, info] = metal_linalg::cholesky_ex_accelerated(a);
```

Since 2.18.0. As for the other decompositions, each call is routed by a policy
measured on the Mac it runs on: to LAPACK on every CPU core, or to one of three
GPU paths. `cholesky_backend(n, batch)` reports which (`cpu`, `simd`,
`threadgroup` or `blocked`). The same is in the C API
(`metal_linalg_cholesky`), Python (`metal_linalg.cholesky`, `cholesky_ex`),
PyTorch (`metal_linalg_torch.cholesky`, `cholesky_ex`, with autograd and
`torch.compile`) and Swift (`choleskyAccelerated`).

A matrix that is not positive definite -- a pivot that is not a positive
finite number, which includes a NaN or an infinity in the triangle read --
stops there, as `spotrf` does: its `info` is the pivot's index + 1 and its `L`
is all NaN. The other matrices of the batch are unaffected. The MLX, C and
Swift calls report it this way and do not throw; the PyTorch package's
`cholesky` raises `torch.linalg.LinAlgError` with torch's message, and its
`cholesky_ex` returns `info` (as int32, as torch's).

## The factorization

A symmetric positive definite $A \in \mathbb{R}^{N \times N}$ factors uniquely
as

$$A = L L^T$$

with $L$ lower triangular and its diagonal positive. Column $j$ of $L$ is

$$L_{jj} = \sqrt{A_{jj} - \sum_{k<j} L_{jk}^2}, \qquad
L_{ij} = \Big(A_{ij} - \sum_{k<j} L_{ik} L_{jk}\Big) / L_{jj} \quad (i > j),$$

$N^3/3$ multiply-adds in all. It needs no pivoting: it is backward stable for
any positive definite matrix (Golub & Van Loan, s4.2), which is what makes it
the factorization of choice for covariance matrices, normal equations and
Gaussian processes.

The blocked form takes $b$ columns at a time: factor the $b \times b$ diagonal
block, solve the rows below it against that factor (a triangular solve), and
subtract the panel's product with itself from the trailing matrix (a symmetric
rank-$b$ update). The trailing update is $1 - O(b/N)$ of the work and is a
matrix product, which is what a GPU does well; the diagonal blocks are a
chain of dependent steps, which it does not.

## The backends

### `simd`: a matrix in a simdgroup's registers (N up to 32)

A row a lane, the factorization right-looking a column at a time: column $k$'s
pivot is broadcast from lane $k$ by a shuffle, each lane scales its entry, and
each later column $c$ of the rows below is updated with $L_{ck}$ shuffled from
lane $c$. Both loops are unrolled, so a row is only ever indexed by constants
and stays in registers. Up to 8 x 8 four matrices share a simdgroup, up to 16
two. Each simdgroup's matrices are staged through threadgroup memory both
ways, so that device memory is read and written a whole row of consecutive
floats at a time (a lane reading its own row touched 32 cache lines an
instruction: 16384 matrices of 32 x 32 took 4.1 ms, against 0.9 staged). One
dispatch, from the input to the output.

### `threadgroup`: a matrix a threadgroup

In a workspace padded to multiples of 32: panels of 32 columns, the diagonal
block factored in one simdgroup's registers as above, the rows below it by
forward substitution (a thread a row) and the trailing lower triangle by 8 x 8
simdgroup matrix products. Three dispatches (the input in, the factorization,
the output out).

### `blocked`: the large-matrix path

Panels of 128 columns. Each panel's four 32-column sub-panels are one dispatch
each of `chol_panel`: every threadgroup takes a strip of 64 rows below the
diagonal block, and brings both the 32 x 32 diagonal block and its strip up
to date with the panel's earlier columns (simdgroup products, the operands
staged 32 columns at a time through threadgroup memory), factors the block in
registers -- each threadgroup the same arithmetic, so the same factor, which
saves a dispatch and a round trip through memory -- and solves its strip
against it. The factored block is not written back where other threadgroups
still read it; the first threadgroup keeps it aside and one dispatch at the
end puts every block in place (nothing reads a diagonal block again). The
trailing update after each panel is MPS products of rank 128, in column
blocks of 512 so as to skip most of the upper triangle (one 8192 x 8192: 50
ms against 68 for one product over the whole square). All of a matrix's
dispatches go into one command buffer. A batch runs together, every product
batched.

On an M5 Pro, one 4096 x 4096 takes 13 ms, against 29 for the CPU path and
470 for `MPSMatrixDecompositionCholesky` (which also factors only the first
matrix of a batch).

### `cpu`: LAPACK on every core

Accelerate's `spotrf`, a batch's matrices spread over every core; one large
matrix (above 1024) gets Accelerate's own threads, and larger matrices go one
at a time (`spotrf` calls running at once with Accelerate's threading off
corrupted each other's factors from N ~ 1536 on macOS 27). Always
`spotrf('L')` on a column-major copy of the triangle in a scratch whose
leading dimension is padded off a power of two: on macOS 27, `spotrf('U')`
was 4-5x slower than `'L'` up to N = 64 and 1.3-1.4x from 256 to 3072, and
either at a leading dimension of 1024, 2048, 4096 or 8192 lost 10-45% to
cache-set conflicts (`'L'` at 4096: 44 ms at ld 4096, 24 at 4112). A row-major
matrix read column-major is its transpose, so the lower triangle goes in and
out transposed (NEON 4 x 4 and 8 x 8 blocks, along the destination's rows
where it is the caller's matrix), the upper one row by row. Against copying
the triangle and calling `spotrf('U')` in place, as before: 512 x 512 0.41 ms
to 0.09, 1024 1.2 to 0.43, 2048 5.5 to 2.8.

## Dispatch

GPU or CPU, by the policy for this device (`metal_linalg::cholesky_policy()`):
the GPU iff $\text{gpu\_min\_n} \le N \le \text{gpu\_max\_n}$,
$\text{batch} \cdot N \ge \text{gpu\_min\_batch\_times\_n}$ and
$\text{batch} \ge \text{gpu\_min\_batch}$, or $N \ge \text{gpu\_large\_min\_n}$
in a batch of at most `gpu_large_max_batch` (0: any). On the GPU, `simd` up to
`simd_max_n` (at most 32), `blocked` from `blocked_min_n` in batches of at
most `blocked_max_batch` (0: any), `threadgroup` between.

On the M5 Pro (run [`20261010-e37928`](results/apple-m5-pro-20gpu/20261010-e37928/cholesky/report.md)):

```
GPU iff N >= 1536 and batch * N >= 3072 (two matrices of 1536-2048, one from 3072), or N >= 4096
    simd up to 32, threadgroup to 159, blocked from 160
everything else: the CPU
```

`CHOLESKY_DEVICE=cpu` or `=gpu` forces the first half of the decision, `=simd`,
`=threadgroup` or `=blocked` a kernel where it takes the shape. Each policy
field can be set from the environment: `CHOLESKY_SIMD_MAX_N`,
`CHOLESKY_BLOCKED_MIN_N`, `CHOLESKY_BLOCKED_MAX_BATCH`, `CHOLESKY_GPU_MAX_N`,
`CHOLESKY_GPU_MIN_BATCH_TIMES_N`, `CHOLESKY_GPU_MIN_BATCH`,
`CHOLESKY_GPU_MIN_N`, `CHOLESKY_GPU_LARGE_MIN_N`,
`CHOLESKY_GPU_LARGE_MAX_BATCH`.

## Performance

Apple M5 Pro (20 GPU cores, 18 CPU cores), 2.18.0, from the routing sweep
above: medians in ms, the faster of two passes. The CPU path is this library's
(above); the GPU column is the fastest GPU backend.

| N | batch | CPU | GPU | GPU over CPU | routed |
|---|---|---|---|---|---|
| 1536 | 1 | **1.42** | 2.16 | 0.66x | cpu |
| 1536 | 4 | 6.01 | **3.52** | **1.71x** | blocked |
| 1536 | 16 | 24.8 | **10.4** | **2.39x** | blocked |
| 2048 | 1 | **2.96** | 3.37 | 0.88x | cpu |
| 2048 | 2 | 6.04 | **4.40** | **1.37x** | blocked |
| 2048 | 4 | 11.9 | **6.28** | **1.89x** | blocked |
| 2048 | 16 | 48.1 | **20.1** | **2.39x** | blocked |
| 3072 | 1 | 9.09 | **6.82** | **1.33x** | blocked |
| 3072 | 4 | 39.5 | **15.7** | **2.51x** | blocked |
| 4096 | 1 | 29.4 | **13.1** | **2.25x** | blocked |
| 4096 | 2 | 59.8 | **18.7** | **3.19x** | blocked |
| 4096 | 4 | 124 | **31.9** | **3.89x** | blocked |

Below 1536 the CPU path wins every batch measured on this Mac, from 2 x 2 to
1024 x 1024 and from one matrix to 16384: Accelerate's `spotrf('L')` is
fast on the M5 Pro's cores (a 64 x 64 in about 4 us a core), and a Cholesky
factorization is a chain of $N$ dependent steps, which bounds a matrix a
threadgroup or simdgroup however many run at once. The nearest the GPU comes
is 0.93x (4096 of 8 x 8, `simd`), 0.86x (4096 of 96 x 96, `threadgroup`) and
0.84x (16 of 1024 x 1024, `blocked`). On a Mac whose CPU is weaker against its
GPU, the estimated policies (refitted from these timings with the GPU's
slowed by the two Macs' ratio, [tuning.md](tuning.md#macs-nobody-has-measured))
and the Mac's own measurements decide; the kernels are there for it.

Against MLX's own `mx.linalg.cholesky` (CPU only, one matrix at a time;
`benchmark_cholesky`): 16384 of 32 x 32 in 0.94 ms against 20.4 (22x), 16 of
1024 x 1024 in 3.4 against 10.9 (3.3x), one 512 x 512 in 0.10 against 0.14
(1.4x), one 4096 x 4096 in 13.1 against 30.9 (2.4x).

## Tuning

`tuning/tune_cholesky.py` (run by `tuning/run.py`; `--only cholesky` for this
one, about 3 minutes on an M5 Pro): N from 2 to 4096 at batches from 1 to
16384 up to 2^26 floats, every backend that takes the size timed at every
point in two passes of independently shuffled order, min of passes; a probe
point before and after for drift. The kernel choice is fitted on the GPU's
times alone, then the GPU-or-CPU rule on everything, each the candidate with
the smallest worst case within 0.5% of the best geometric-mean regret. On the
M5 Pro: 1.0001x geometric mean, worst 1.02x, over 154 points (always the CPU:
1.057x, worst 3.89x); fitted on half the points, 1.0002x on the other half.

## Tests

```sh
cmake --build build --target test_cholesky
./build/test_cholesky          # or: ctest --test-dir build
```

1409 checks: every backend directly and through the routing, lower and upper,
from 1 x 1 to 1300 x 1300 (each side of the simd kernel's 8, 16 and 32, the
32-column panels, the 128-column panels and the trailing update's column
blocks), batched, each against LAPACK; $L L^T = A$, the other triangle exactly
zero, a positive diagonal; a matrix not positive definite at its first, a
middle and its last pivot inside a batch of three (exact `info`, all NaN for
it alone); NaN in the triangle read and in the one ignored; an infinite pivot;
scales from $10^{-30}$ to $10^{30}$; transposed and unaligned views, empty
batches and 0 x 0 matrices, extra batch axes; and the routing policy and
`CHOLESKY_DEVICE`. `test_core`, `test_c_api` and the Python, PyTorch (values,
devices, `LinAlgError`, gradients against torch's in float64, `opcheck`,
`torch.compile`) and Swift suites cover their layers.

## References

- G. H. Golub and C. F. Van Loan, *Matrix Computations*, 4th ed., s4.2 (the
  Cholesky factorization and its block form).
- LAPACK `spotrf`, `spotf2`.
- I. Murray, *Differentiation of the Cholesky decomposition*, arXiv:1602.07527
  (2016): the gradient the PyTorch package uses, as torch does.
