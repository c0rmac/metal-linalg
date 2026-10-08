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

**Fast division and square roots** (since 2.15.0). The shader is built with
`-fno-fast-math`, for its non-finite checks, which makes `/`, `sqrt()` and
`rsqrt()` the IEEE sequences, and a kernel with any of them in it compiles
all of its arithmetic in IEEE mode. The rotation and the output use the fast
approximations with Newton steps instead (`j_div`, `j_sqrt`, `j_rsqrt` in
`eigh_jacobi_common.h`): 1.14-1.17x on batches of 16×16 to 48×48 on an M5
Pro, 1.03-1.06x from 128×128. `rsqrt` takes two steps: with one, $c$ came out
a little low every time, $c^2 + s^2 < 1$, and $V$'s orthogonality was 10x
worse at 512×512; with two it is a little better than with the IEEE
sequence.

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
   with vectors 1.1-1.3x. Each threadgroup's reflectors divide and take
   square roots with the fast approximations and a Newton step (since
   2.15.0), not the IEEE sequences the shader file is otherwise built for.
2. **Bidiagonal SVD**: on the CPU, `sbdsdc`'s divide and conquer with the
   singular vectors of $B$, on every core but two (`src/divide_conquer.cpp`,
   since 2.15.0): LAPACK's tree and routines, its leaves, merges and the
   large merges' loops spread over threads, and `slasd2`'s deflation
   rewritten to move the right vectors' rows a column at a time (LAPACK
   moves them a strided row at a time, 130 of a 4096 merge's 190 ms). On a
   4096 bidiagonal from `sgebrd`, 100 ms against `sbdsdc`'s 753 on one core;
   the singular values bit for bit LAPACK's, and results independent of the
   number of threads. For one matrix, whose GPU is idle meanwhile, the top
   merges' products (from $k \sim 2048$) run on the GPU on the merges' memory
   in place, and the vectors are written straight into the GPU's buffers:
   1.046x at 4096. For the singular values alone (since 2.13.0;
   `sbdsdc` before), bisection on the GPU from $k = 1024$ (12 ms at 4096,
   see [`band`](#a-seventh-backend-for-the-singular-values-of-large-matrices-band)),
   below it `sbdsqr`, whose dqds is faster than `sbdsdc` and accurate to every
   singular value of $B$, however small.
3. **Back-transformation**: the reflectors of $Q$ and of $P$ applied to the
   singular vectors of $B$ 128 at a time, each block as $I - V T V^T$, as MPS
   GEMMs; the next block is built on the CPU while the GPU applies this one
   ($T$ from the Gram matrix $V^T V$ since 2.15.0).

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

On an M5 Pro, one $k \times k$ matrix, against the CPU path (`sgesdd`), 2.15.0
measured side by side (the median of `sweep_svd`):

| $k$ | svd: CPU | bidiag | speedup | svdvals: CPU | bidiag | speedup |
|---|---|---|---|---|---|---|
| 512 | 16.7 ms | 13.2 ms | 1.27x | 7.6 ms | 8.9 ms | 0.86x |
| 1024 | 78.5 ms | 34.2 ms | 2.29x | 36.0 ms | 21.5 ms | 1.67x |
| 1536 | 182 ms | 69.1 ms | 2.64x | 86.2 ms | 47.5 ms | 1.81x |
| 2048 | 472 ms | 125 ms | 3.79x | 218 ms | 87.6 ms | 2.49x |
| 3072 | 1.42 s | 0.39 s | 3.67x | 0.77 s | 0.31 s | 2.52x |
| 4096 | 3.53 s | 0.94 s | 3.74x | 2.07 s | 0.71 s | 2.91x |

For singular values alone, from $k = 1536$ the `band` backend (below) is
faster still: 10.5x the CPU at 4096.

The gain grows with $k$, as the CPU's reduction falls further behind memory
bandwidth. The M5 Pro uses the backend from $k = 1024$, with vectors or
without, for up to two matrices: it decomposes a batch one matrix after
another (pipelined), and the CPU path spreads one over every core
(`bidiag_max_batch`). With 2.15.0's parallel divide and conquer it wins by
2.3-3.8x from 1024 to 2048 with vectors (1.4-1.8x in 2.14, after 2.12.0
merged the reduction's dispatches; 1.0-1.4x before), and for singular values
alone, with bisection on the GPU since 2.13.0, by 1.7-2.5x, where before
2.12.0 it lost below 2048. Accuracy
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
[`results/apple-m5-pro-20gpu/20261004-06bc11/svd/`](results/apple-m5-pro-20gpu/20261004-06bc11/svd/)
(2.13.0; min of two randomised passes; the CPU path spreads the batch over 18 cores; "shared" is the
batch split between golub_kahan and the CPU path, see below):

| k×k | batch | golub_kahan | shared with the CPU | best Jacobi | CPU | Jacobi / GK | CPU / GK | CPU / shared |
|---|---|---|---|---|---|---|---|---|
| 8 | 4096 | 0.87 ms | 0.96 ms | 1.41 ms | 1.46 ms | 1.62x | **1.67x** | **1.52x** |
| 16 | 1024 | 0.87 ms | 0.95 ms | 1.77 ms | 1.19 ms | 2.02x | **1.36x** | **1.24x** |
| 16 | 4096 | 2.47 ms | 2.28 ms | 5.81 ms | 4.50 ms | 2.36x | **1.82x** | **1.97x** |
| 24 | 4096 | 5.73 ms | 4.70 ms | 14.1 ms | 9.56 ms | 2.47x | **1.67x** | **2.03x** |
| 32 | 256 | 0.92 ms | 0.89 ms | 2.21 ms | 1.13 ms | 2.39x | **1.22x** | **1.27x** |
| 32 | 1024 | 2.68 ms | 2.28 ms | 7.02 ms | 4.17 ms | 2.62x | **1.56x** | **1.83x** |
| 32 | 4096 | 9.29 ms | 7.31 ms | 22.4 ms | 15.5 ms | 2.41x | **1.66x** | **2.11x** |
| 40 | 4096 | 18.7 ms | 13.7 ms | 50.6 ms | 25.8 ms | 2.71x | **1.38x** | **1.88x** |
| 48 | 1024 | 7.21 ms | 5.20 ms | 21.3 ms | 9.02 ms | 2.96x | **1.25x** | **1.73x** |
| 48 | 4096 | 28.2 ms | 19.4 ms | 82.5 ms | 36.5 ms | 2.93x | **1.29x** | **1.88x** |
| 56 | 4096 | 43.9 ms | 25.1 ms | 105.6 ms | 42.0 ms | 2.41x | 0.96x | **1.67x** |
| 64 | 1024 | 17.4 ms | 9.32 ms | 30.4 ms | 14.4 ms | 1.75x | 0.83x | **1.55x** |
| 64 | 4096 | 64.5 ms | 33.9 ms | 114.0 ms | 53.4 ms | 1.77x | 0.83x | **1.58x** |
| 80 | 4096 | 150.0 ms | 66.2 ms | 308.1 ms | 97.6 ms | 2.05x | 0.65x | **1.48x** |

It is 1.6-3x faster than the best Jacobi kernel at every size it takes, and
alone ahead of the CPU for large batches up to 48×48; shared with the CPU, up
to 80×80 (1.5-2.1x). Alone, beyond 48 the CPU pulls away: the QR iteration is one thread's serial chain per matrix, and with the
matrix in threadgroup memory too few matrices share a core to hide it. Tall
matrices are the CPU's whatever the backend, since the CPU path reduces them
by a QR first (256×32, 4096 of them: 0.87x), which the routing's
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

## A seventh backend for the singular values of large matrices: `band`

For the singular values alone, the `bidiag` backend's reduction is held to the
speed of memory: two matrix-vector products a column, each reading the whole
trailing matrix, 650 of its 740 ms at 4096×4096 on an M5 Pro, already at
about 290 GB/s. The `band` backend (`svd_band` in `src/svd_bidiag.mm`, new in
2.13.0) reduces in two stages, as LAPACK's `ssyevd_2stage` does for symmetric
eigenvalues ([Haidar, Ltaief and Dongarra](https://doi.org/10.1145/2063384.2063394)),
and as the eigensolver's own `band` backend does ([eigh.md](eigh.md)):

1. **To a band** of width $b$ on the GPU (the policy's `values_band_width`,
   16 by default; `band_reduce_general` in `src/band_reduce.mm`), $b$ columns a
   block: the QR of the block's column panel, $H = I - V T V^T$; the QR of
   its row panel (transposed: an LQ), $G = I - U S U^T$; and the rest of the
   matrix updated by both, folded so that it is read three times and written
   once a block, as $C \mathrel{-}= V W + (X - V W U) S U^T$ in one product,
   where the one-stage reduction reads it twice a column. The large products
   are MPS GEMMs, the small ones ($b \times b$ and $b$ wide) two kernels of
   the library's (since 2.15.0). A panel of up to 128 rows is factored in one
   simdgroup, four rows a lane in registers, with no threadgroup barrier at
   all; a taller one by TSQR ([Demmel, Grigori, Hoemmen and Langou](https://doi.org/10.1137/080731992)),
   leaves of up to 128 rows in parallel and their stacked $R$'s in one
   threadgroup as a binary tree of triangle pairs (since 2.15.0), a
   simdgroup a pair, then the Householder vectors rebuilt from TSQR's $Q$ by an LU
   with chosen signs ([Ballard, Demmel, Grigori, Jacquelin, Knight and
   Nguyen](https://doi.org/10.1016/j.jpdc.2015.06.003)), so that the update is the
   same compact $I - V T V^T$. The last columns, fewer than $2b$, are LAPACK's.
2. **To bidiagonal** on the CPU's cores (`band_to_bidiagonal` in
   `src/band_chase.cpp`), by Householder bulge chasing with PLASMA's kernels
   ([Haidar, Kurzak and Luszczek](https://doi.org/10.1145/2503210.2503292)):
   sweep $s$ makes row $s$ bidiagonal, a reflector from the right and one
   from the left a block, each clearing one row or column of the bulge it
   chases down the band. Sweep $s$ may run its $t$-th step once sweep $s - 1$
   has finished its $(t + 2)$-th, so the sweeps run pipelined on all the cores
   but two, which the GPU's host work keeps. LAPACK's `sgbbrd` does it by
   rotations on one core: 196 ms for a 4096 band of width 16 on an M5 Pro,
   against 36 ms here.
3. **The singular values** by bisection on the GPU, from $k = 1024$: those
   of the bidiagonal $B$ are the non-negative eigenvalues of the
   $2k \times 2k$ Golub-Kahan tridiagonal (zero diagonal, $B$'s entries
   interleaved off it), a thread each, by Sturm counts as the eigensolver's
   ([eigh.md](eigh.md#backend-5-eigenvalues-alone-in-two-stages-band)). LAPACK's `sbdsqr` (dqds) is
   sequential, 77 ms at 4096 and 300 at 8192; bisection takes 12 and 31. Below
   1024, `sbdsqr`. The `bidiag` backend's singular-value path uses the same
   bisection since 2.13.0.

On an M5 Pro, one square matrix, singular values alone:

| $k$ | CPU | bidiag | band | band / bidiag |
|---|---|---|---|---|
| 1024 | 0.036 s | 0.022 s | 0.021 s | 1.03x |
| 1536 | 0.086 s | 0.048 s | 0.034 s | 1.41x |
| 2048 | 0.218 s | 0.088 s | 0.055 s | 1.60x |
| 3072 | 0.773 s | 0.307 s | 0.113 s | 2.72x |
| 4096 | 2.070 s | 0.712 s | 0.197 s | 3.61x |
| 8192 | 12.64 s | 6.03 s | 1.10 s | 5.50x |

For one matrix the chase starts with the reduction and trails it down the
band, as each block's rows are finished (since 2.15.0): only about 6% of
the chase can go before the GPU is done, since every sweep runs to the
band's end, 1.05x at 2048-4096, the singular values the same bit for bit.

Against the CPU path that is 2.6x at $k = 1536$, 4.0x at 2048, 6.8x at 3072,
10.5x at 4096 and 11.5x at 8192 (2.15.0, measured side by side, the median
of `sweep_svd`). At $k = 4096$ the chase takes 35 ms, bisection 12, and
the GPU's stage, with the matrix's copy in, the rest. Below 1536 the panels'
latency, a chain of dependent steps per block that does not shrink with $k$,
costs more than the products save; the M5 Pro uses the backend from there
(`values_band_min_k`). Accuracy is that
of `bidiag`: singular values within $1 \times 10^{-5}$ of float32 LAPACK's
relative to $\sigma_\text{max}$. How the panel kernels got from 0.5 ms to
0.15 ms each is in [the two-stage study](studies/two-stage-apple-m5-pro.md).

### With singular vectors (since 2.15.0)

The same two stages serve the SVD with vectors (`svd_band_vectors`, width
16), which then needs both stages' transformations back: with $A = Q_1 B_b
P_1^T$ (the band, $Q_1$ and $P_1$ the GPU stage's block reflectors) and
$B_b = Q_2 B P_2^T$ (the bidiagonal, $Q_2$ and $P_2$ the chase's reflectors),
$U = Q_1 Q_2 U_B$ and $V = P_1 P_2 V_B$. LAPACK's two-stage drivers do not
take vectors; PLASMA's and MAGMA's two-stage eigensolvers do. Here $Q = Q_1
Q_2$ and $P = P_1 P_2$ are formed explicitly on the GPU while the CPU does
its two steps, so that most of the GPU's work hides behind the CPU's:

1. **The band reduction keeps its reflectors.** A block's panels write their
   Householder vectors straight into an aggregated layout as they factor
   (flag 16 of the panel kernels), eight blocks to an aggregate of 128; as
   the GPU completes each eight, the CPU builds the aggregate's $T$ from the
   blocks' own (the block-reflector merge, from the Gram matrix $V^T V$).
2. **$Q_1$ and $P_1$ explicit** (the thin $m \times k$ and $k \times k$): the
   aggregates on the GPU, last first, three MPS products each, on the
   shrinking trailing block, encoded and queued while the GPU still reduces
   the matrix and released once the CPU has applied the last columns'
   LAPACK reflectors. Meanwhile the CPU chases the band to bidiagonal, writing
   each reflector it makes into the layout of step 3's kernel.
3. **$Q \leftarrow Q_1 Q_2$ and $P \leftarrow P_1 P_2$** on the GPU
   (`bd_chase_apply` in `shaders/Svd_Bidiag.metal`), in two chunks of groups
   of sweeps, each released as the chase finishes its groups, so the GPU
   goes on from $Q_1$ and $P_1$ without waiting for the chase's end; then
   while the CPU solves $B = U_B \Sigma V_B^T$ by the divide and conquer.
   The chase's reflectors
   of 16 consecutive sweeps at the same step form a block $I - V T V^T$, $V$
   $32 \times 16$, acting on 32 consecutive rows, which the CPU stores as
   $V$ and $Y = -T^T V^T$ (13 nonzero $8 \times 8$ tiles), so that a block
   is two products, $Z = Y X$ and $X \leftarrow X + V Z$; a group of sweeps runs its
   blocks in order along the matrix, and the next group may follow two tiles
   (32 rows) behind. A threadgroup owns 32 of $Q$'s rows (as columns of
   $Q^T$) and runs four groups at once, a simdgroup each, the 16-row tiles
   handed from one simdgroup to the next through threadgroup memory: a
   pipeline about $k/16$ blocks long a pass, where one group at a time made
   every column strip a chain of $k^2/512$ dependent blocks. At $k = 4096$,
   32,896 blocks a side: about 49 ms a side, against 176 ms for the groups
   one after another.
4. **$U = Q U_B$ and $V^T = V_B^T P^T$**, two MPS products, written row-major;
   $U$ is copied out while the GPU forms $V^T$.

On an M5 Pro, one $k \times k$ at 4096: the band reduction 0-155 ms on the
GPU; $Q_1$ and $P_1$ there for 32 ms, under the chase's 43 on the CPU;
$Q_2$ and $P_2$ for 98 ms, under the divide and conquer's 120 (whose top
products go to the GPU once $Q_2$ and $P_2$ are done); the products 41 ms.
`bidiag`'s one-stage reduction alone takes 700 ms. Square,
against the CPU path and `bidiag` (2.15.0, side by side, the median of
`sweep_svd`; 8192 from separate runs):

| $k$ | CPU | bidiag | band | band / bidiag | band / CPU |
|---|---|---|---|---|---|
| 512 | 16.7 ms | 13.2 ms | 11.4 ms | 1.16x | 1.47x |
| 1024 | 77.8 ms | 34.0 ms | 28.1 ms | 1.21x | 2.76x |
| 1536 | 178 ms | 67.4 ms | 51.7 ms | 1.30x | 3.44x |
| 2048 | 444 ms | 120 ms | 82.8 ms | 1.45x | 5.36x |
| 3072 | 1.33 s | 0.360 s | 0.184 s | 1.95x | 7.21x |
| 4096 | 3.51 s | 0.944 s | 0.402 s | 2.35x | 8.73x |
| 8192 | | 7.89 s | 2.93 s | 2.69x | |

(With the display busy separate runs moved by up to 7%: in alternating
runs `band` took 375 ms at 4096, and 368 since its blocks carry $Y$
instead of $T$.)

Tall, 4096×2048, 1.20x `bidiag` (219 ms against 263); 8192×2048, through
the QR first, 1.09x (390 against 426). A batch of two is where `bidiag`'s
pipeline (one matrix's divide and conquer under the next's reduction) still
wins at 1024 (54 ms against 59); at 2048, `band` 1.24x. The backend takes a
batch one matrix after another, each overlapped within itself, from
`band_min_k` within `bidiag_max_batch`.

Accuracy is LAPACK's: reconstruction and orthogonality about $3 \times
10^{-6}$ at 1024, $6 \times 10^{-6}$ at 4096 and $8 \times 10^{-6}$ at 8192
(`bidiag` $2$, $4$ and $6 \times 10^{-6}$: two stages of float32 reflectors
instead of one), singular values within $3 \times 10^{-7}$ of float64
LAPACK's relative to $\|A\|_F$. Memory: at 8192 the call keeps about 1.2 GB
more than `bidiag` (the chase's blocks, $V$ and $Y$, 440 MB a side, and the
explicit $Q$ and $P$).

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
where that says CPU (up to bidiag_max_batch matrices; svdvals: values_bidiag_max_batch):
  band iff  k >= band_min_k  (svdvals: k >= values_band_min_k; 0 = never)
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
| Apple M5 Pro | 20 | k <= 8, l <= 2048 and batch * k >= 4096, or k <= 80 in batches of 256+ (svdvals: k <= 80, l <= 2048 and batch * k >= 16384) | k = 8 .. 80, shared with the CPU from batch 256 | 256 rows, k >= 16 | k = 192; k = 64 in batches of 64+ | from k = 1024 (svdvals too), batches up to 4 (svdvals 2); `band` from k = 1024 with vectors, from 768 for svdvals | measured — run [`20261007-9f2589`](results/apple-m5-pro-20gpu/20261007-9f2589/svd/report.md) (1.0170 geometric-mean regret against the best backend at each of 295 points, worst 1.75x) |
| anything else | — | estimated | estimated | estimated | estimated | estimated | **estimated** from the M5 Pro's timings ([how](tuning.md#macs-nobody-has-measured)) |

On the M5 Pro large batches of small matrices, up to 56×56 and a long side
of 256, go to `golub_kahan` on the GPU, shared with the CPU from 1024
matrices, and so do batches of 1024 and more up to 80×80 (the large-batch
clause, since 2.12.0); everything else in a batch goes to the CPU, and one
or two large matrices to `bidiag`, their singular values alone from
$k = 1536$ to `band`. Against the best backend at each of the 291 points
measured, the 2.12.0 row scored 1.020 geometric-mean regret, worst 1.67x,
against 1.028 for the product rule alone (on the run before, 2.11.0's row
scored 1.027 and 2.10.0's 1.039); on the 2.13.0 run the row scores 1.0198,
worst 1.58x. The clause is what
takes 64×64 and 80×80 in batches of 1024 and more, which shared win by
1.5-1.6x but which one product rule cannot take without also taking their
small batches, which the CPU wins. The long-side cap
`gpu_max_l` (new in 2.10.0, no cap on a device without it) is what lets the
rule take the square batches the GPU wins without the tall ones it loses:
with a cap on k alone, the fit stopped at k = 24. In 2.9.0, measured against
the same CPU path, the Jacobi kernels won only for large batches of the
smallest matrices; before 2.9.0 the row sent batches up to k = 1024 to the
GPU, measured against one CPU core.

The M1 has no row, so it is estimated from the M5 Pro's timings like any Mac
nobody has measured ([how](tuning.md#macs-nobody-has-measured)): `svd_policy_source()` reports
`estimated:Apple M1 (from Apple M5 Pro, ...)`. The `SvdPolicy` defaults, which
come from measurements on an M1 taken while the machine was heavily loaded by
other jobs, good enough to place the crossovers roughly and not for a table
entry, remain only for a Mac with nothing to estimate from
(`default:untuned-device`). Measuring a Mac is one command,
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
`SVD_GPU_BIG_BATCH_MIN`, `SVD_VALUES_BAND_MIN_K`, `SVD_BAND_MIN_K` and
`SVD_DEVICE=gpu|cpu|bidiag|band` override it (`band` with vectors too since
2.15.0; before, it meant `bidiag` there). `SVD_VALUES_BAND_WIDTH` sets the
`band` backend's band width as a policy field (`values_band_width`, 8, 16 or
32; 0 is 16), and `SVD_BAND_WIDTH=8|16|32` where the policy leaves it 0.
`svd_backend(m, n, batch)` and `svdvals_backend(m, n, batch)` say which of the
nine backends a problem gets, with vectors and for singular values alone.

The whole-matrix Jacobi kernel gives a matrix one threadgroup for its whole
solve: 263 ms at 512×512 on an M5 Pro, 2.7 s at 1024×1024. With the display
busy macOS ends a command buffer whose threadgroup runs for more than about
a quarter of a second ("GPU Hang Error"), so since 2.15.0 a solve the cost
model puts over 40 ms (`SVD_DISPATCH_MS`) is split over dispatches of a few
rounds, each matrix resuming where it stopped (its columns and V are in
device memory already; its scale, the sweep's null and negligible levels,
the sweep, the round and whether a pair has rotated yet are kept): bit for
bit the same result, in the same time. The routing does not send such sizes
to the kernel on the M5 Pro, but forced kernels and estimated policies may.

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
| 8×8 | 2.9e-07 | 3.9e-07 | 3.5e-07 | 7.7e-08 | 5 |
| 64×64 | 1.2e-06 | 1.7e-06 | 1.2e-06 | 2.0e-07 | 9 |
| 256×256 | 2.8e-06 | 8.5e-06 | 2.8e-06 | 2.1e-07 | 12 |
| 512×512 | 4.6e-06 | 1.9e-05 | 4.5e-06 | 3.4e-07 | 13 |
| 2048×64 | 1.1e-06 | 1.2e-05 | 1.1e-06 | 1.0e-07 | 7 |

Rank-deficient input reconstructs as well as full-rank input: over 204 random
instances from rank 1 of 24×24 to rank 50 of 600×130, through all four GPU
backends, the worst reconstruction error was 2.3e-06 and the worst sweep count
12. How null columns are kept from stalling the sweeps is in
[`studies/svd-design-notes.md`](studies/svd-design-notes.md).

## Performance

Apple M5 Pro (20 GPU cores, 18 CPU cores) with 2.13.0, from the routing sweep in
[`results/apple-m5-pro-20gpu/20261004-06bc11/svd/`](results/apple-m5-pro-20gpu/20261004-06bc11/svd/): min of two
randomised passes, on mains, against the CPU path as the library runs it, a
batch spread over all 18 cores. Each cell is the speedup of the fastest GPU
backend over the CPU, with the GPU time and the backend (`whole` and `block`
are the two Jacobi kernels on the matrix itself, `QR+` after preconditioning,
`GK` the golub_kahan backend, `GK+CPU` it sharing the batch with the CPU
path); bold is where the GPU was ahead.

| shape | batch 1 | batch 4 | batch 16 | batch 64 | batch 256 | batch 4096 |
|---|---|---|---|---|---|---|
| 4×4 | 0.02x (0.20 ms, GK) | 0.16x (0.15 ms, whole) | 0.23x (0.18 ms, GK) | 0.49x (0.16 ms, GK+CPU) | 0.49x (0.29 ms, whole) | **1.81x** (0.39 ms, GK) |
| 8×8 | 0.05x (0.19 ms, whole) | 0.10x (0.23 ms, GK) | 0.42x (0.18 ms, whole) | 0.58x (0.24 ms, GK) | 0.81x (0.24 ms, GK) | **1.67x** (0.87 ms, GK) |
| 16×16 | 0.06x (0.32 ms, GK) | 0.19x (0.23 ms, whole) | 0.30x (0.31 ms, whole) | 0.56x (0.33 ms, whole) | **1.05x** (0.39 ms, GK) | **1.97x** (2.28 ms, GK+CPU) |
| 32×32 | 0.17x (0.34 ms, whole) | 0.24x (0.39 ms, whole) | 0.42x (0.44 ms, whole) | 0.49x (0.81 ms, GK) | **1.27x** (0.89 ms, GK+CPU) | **2.11x** (7.31 ms, GK+CPU) |
| 48×48 | 0.15x (0.77 ms, whole) | 0.20x (0.80 ms, whole) | 0.34x (0.87 ms, whole) | 0.57x (1.21 ms, GK) | **1.58x** (1.55 ms, GK+CPU) | **1.88x** (19.4 ms, GK+CPU) |
| 64×64 | 0.19x (1.05 ms, whole) | 0.22x (1.05 ms, whole) | 0.33x (1.19 ms, whole) | 0.53x (1.95 ms, GK) | **1.09x** (3.41 ms, GK+CPU) | **1.58x** (33.9 ms, GK+CPU) |
| 128×128 | 0.21x (4.91 ms, whole) | 0.27x (4.94 ms, whole) | 0.33x (5.07 ms, whole) | 0.42x (14.5 ms, block) | 0.45x (47.9 ms, block) | 0.41x (757.2 ms, block) |
| 256×256 | 0.31x (13.0 ms, block) | 0.29x (14.2 ms, block) | 0.25x (25.0 ms, block) | 0.24x (82.9 ms, block) | 0.23x (348.6 ms, block) | -- |
| 512×512 | 0.53x (32.0 ms, block) | 0.39x (48.7 ms, block) | 0.18x (155.6 ms, block) | 0.16x (645.5 ms, block) | -- | -- |
| 1024×1024 | 0.68x (115.4 ms, block) | 0.30x (310.6 ms, block) | 0.25x (1.25 s, block) | -- | -- | -- |
| 64×8 | 0.06x (0.20 ms, whole) | 0.14x (0.21 ms, GK) | 0.35x (0.23 ms, GK) | 0.64x (0.24 ms, GK) | 0.97x (0.29 ms, GK) | **1.57x** (1.84 ms, GK) |
| 64×32 | 0.19x (0.37 ms, whole) | 0.28x (0.38 ms, whole) | 0.35x (0.63 ms, QR+whole) | 0.56x (0.85 ms, GK) | **1.46x** (1.01 ms, GK+CPU) | **1.90x** (11.7 ms, GK+CPU) |
| 256×32 | 0.20x (0.59 ms, QR+whole) | 0.34x (0.48 ms, whole) | 0.57x (0.51 ms, whole) | 0.54x (1.34 ms, whole) | 0.74x (3.66 ms, GK+CPU) | **1.10x** (30.8 ms, GK+CPU) |
| 1024×32 | 0.33x (0.69 ms, QR+whole) | 0.45x (0.86 ms, QR+whole) | 0.57x (1.15 ms, whole) | 0.59x (3.50 ms, QR+whole) | 0.65x (9.41 ms, GK+CPU) | -- |
| 1024×64 | 0.38x (1.55 ms, QR+whole) | 0.52x (2.13 ms, QR+whole) | 0.51x (3.44 ms, QR+whole) | 0.62x (8.72 ms, GK) | 0.71x (27.1 ms, GK+CPU) | -- |
| 2048×64 | 0.48x (1.89 ms, QR+whole) | 0.57x (3.43 ms, QR+whole) | 0.62x (4.99 ms, QR+whole) | 0.70x (14.1 ms, GK+CPU) | 0.74x (47.5 ms, GK+CPU) | 0.86x (592.7 ms, QR+block) |
| 1024×256 | 0.47x (13.9 ms, QR+block) | 0.48x (16.1 ms, QR+block) | 0.55x (28.8 ms, QR+block) | 0.43x (99.2 ms, QR+block) | 0.40x (399.4 ms, QR+block) | -- |
| 2048×256 | 0.57x (16.1 ms, QR+block) | 0.56x (18.8 ms, QR+block) | 0.54x (36.9 ms, QR+block) | 0.53x (133.3 ms, QR+block) | 0.51x (522.2 ms, QR+block) | -- |

`--` was not measured.

Against a CPU that uses its cores, the GPU wins large batches of small
matrices, with `golub_kahan`, shared with the CPU from about 1024 matrices:
1.6-2.1x at 4096 matrices from 4×4 to 64×64, and from 256 matrices for 16×16 to
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

432 checks: square, tall and wide shapes around the simdgroup, pair-count and
block boundaries, batches, every simdgroup count, all eight GPU backends and
each branch of the CPU one (the `bidiag` backend from 1×1 to 1100×1060, with
QR first and through the transpose, with vectors and without; `band` at each
band width from 1×1 to 1100×1060, either side of the one-simdgroup panel and
the TSQR leaf, rank one, zero, scaled and NaN inputs, and with vectors from
1×1 to 2049×2049, either side of the LAPACK tail alone, a partial aggregate,
the chase's tiles and its two kernels, repeated and clustered values; `golub_kahan`
from 1×1 to its limit, either side of its chaser simdgroup, directly and
after a QR), rank deficiency repeated over random instances per shape and
backend, structured spectra (graded columns, singular values from 1e+4 to
1e-4, repeated values), magnitudes from 1e-30 to 1e+37, NaN inside a batch,
and the routing policy, including the batch-dependent kernel crossover, the
golub_kahan window, the long-side cap, the rule for singular values alone,
sharing a batch with the CPU and the band thresholds, without assuming any
device's values.

## References

- M. R. Hestenes, ["Inversion of matrices by biorthogonalization and related results"](https://doi.org/10.1137/0106005), *J. SIAM* 6(1), 1958 — one-sided Jacobi.
- Z. Drmač and K. Veselić, ["New fast and accurate Jacobi SVD algorithm I"](https://doi.org/10.1137/050639193), *SIAM J. Matrix Anal. Appl.* 29(4), 2008 — the modern form, preconditioning with QR, and what LAPACK's `xGESVJ` / `xGEJSV` implement.
- J. Demmel and K. Veselić, ["Jacobi's method is more accurate than QR"](https://epubs.siam.org/doi/10.1137/0613074), 1992.
- R. P. Brent and F. T. Luk, 1985 — the parallel ordering; see the eigensolver's references.
- G. H. Golub and W. Kahan, ["Calculating the singular values and pseudo-inverse of a matrix"](https://doi.org/10.1137/0702016), *J. SIAM Ser. B Numer. Anal.* 2(2), 1965 — reduction to bidiagonal form by Householder reflections, the method of the `bidiag` backend.
- J. Demmel and W. Kahan, ["Accurate singular values of bidiagonal matrices"](https://doi.org/10.1137/0911052), *SIAM J. Sci. Stat. Comput.* 11(5), 1990 — LAPACK's `sbdsqr`, whose shifted QR sweep, shift and deflation the `golub_kahan` backend runs.
- G. H. Golub and C. F. Van Loan, *Matrix Computations*, 4th ed., 2013 — §5.4.8 (bidiagonalization) and §8.6 (the Golub-Kahan SVD step, and zero diagonal entries), the `golub_kahan` backend.
- J. J. Dongarra, S. J. Hammarling and D. C. Sorensen, ["Block reduction of matrices to condensed forms for eigenvalue computations"](https://doi.org/10.1016/0377-0427(89)90367-1), *J. Comput. Appl. Math.* 27(1-2), 1989 — the blocked bidiagonalization (LAPACK's `sgebrd` and `slabrd`) the backend runs on the GPU.
- A. Haidar, H. Ltaief and J. Dongarra, ["Parallel reduction to condensed forms for symmetric eigenvalue problems using aggregated fine-grained and memory-aware kernels"](https://doi.org/10.1145/2063384.2063394), SC '11, 2011 — the two-stage reduction (LAPACK's `ssytrd_2stage`), which the `band` backend applies to the SVD.
- J. Demmel, L. Grigori, M. Hoemmen and J. Langou, ["Communication-optimal parallel and sequential QR and LU factorizations"](https://doi.org/10.1137/080731992), *SIAM J. Sci. Comput.* 34(1), 2012 — TSQR, the `band` backend's tall panels.
- G. Ballard, J. Demmel, L. Grigori, M. Jacquelin, N. Knight and H. D. Nguyen, ["Reconstructing Householder vectors from tall-skinny QR"](https://doi.org/10.1016/j.jpdc.2015.06.003), *J. Parallel Distrib. Comput.* 85, 2015 — the LU with chosen signs that turns TSQR's $Q$ into compact Householder form.
- A. Haidar, J. Kurzak and P. Luszczek, ["An improved parallel singular value algorithm and its implementation for multicore hardware"](https://doi.org/10.1145/2503210.2503292), SC '13, 2013 — the band-to-bidiagonal bulge chasing by Householder reflectors (PLASMA), the `band` backend's second stage.
- B. Lang, ["A parallel algorithm for reducing symmetric banded matrices to tridiagonal form"](https://doi.org/10.1137/0914078), *SIAM J. Sci. Comput.* 14(6), 1993 — pipelining the sweeps of a bulge chase.
- W. Barth, R. S. Martin and J. H. Wilkinson, ["Calculation of the eigenvalues of a symmetric tridiagonal matrix by the method of bisection"](https://doi.org/10.1007/BF02162154), *Numerische Mathematik* 9, 1967 — Sturm-sequence bisection, applied to the Golub-Kahan tridiagonal for singular values alone.
- M. Gu and S. C. Eisenstat, ["A divide-and-conquer algorithm for the bidiagonal SVD"](https://doi.org/10.1137/S0895479892242232), *SIAM J. Matrix Anal. Appl.* 16(1), 1995 — LAPACK's `sbdsdc`, which solves the bidiagonal problem.
- S. Tomov, R. Nath and J. Dongarra, ["Accelerating the reduction to upper Hessenberg, tridiagonal, and bidiagonal forms through hybrid GPU-based computing"](https://doi.org/10.1016/j.parco.2010.06.001), *Parallel Computing* 36(12), 2010 — the hybrid CPU/GPU split (MAGMA) the backend follows, with the panel kept on the GPU.
- E. Ringoot, R. Alomairy, V. Churavy and A. Edelman, ["Performant unified GPU kernels for portable singular value computation across hardware and precision"](https://doi.org/10.1145/3754598.3754667), 2025 ([arXiv:2508.06339](https://arxiv.org/abs/2508.06339)), and E. Ringoot, R. Alomairy and A. Edelman, ["Accelerating bidiagonalization of banded matrices through memory-aware bulge-chasing on GPUs"](https://arxiv.org/abs/2510.12705), 2025 — GPU singular value computation, including on Apple's GPUs; they prompted the `bidiag` backend, whose GPU-resident panel follows their design. Their kernels are not used here.
