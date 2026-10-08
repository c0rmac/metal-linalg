# eigh and the SVD for batches of mid-size matrices

Status: **analysed, not built** (2026-10-08) as a kernel a matrix; **done
another way for eigh** (2.17.0): the `tridiag_batch` backend reduces a whole
batch together on the GPU, a threadgroup a matrix and panel, and leaves the
tridiagonal problems (the part this page found the GPU cannot do well, the
QL rotations on the vectors) to the CPU's cores, pipelined under the GPU's
stages: 1.3-1.5x the CPU at 256-1024 matrices of 96-256 on an M5 Pro
([eigh.md, backend 6](../eigh.md#backend-6-a-batch-reduced-together-tridiag_batch)).
The SVD's counterpart is built too (2.17.0, `bidiag_batch`, [svd.md](../svd.md#an-eighth-backend-for-batches-of-mid-size-matrices-bidiag_batch)):
1.4-1.65x the CPU at 256-1024 matrices of 128-256. The analysis of a kernel a
matrix below stands: the CPU path is 2-10x ahead of every one-matrix GPU
backend here, and what such a kernel could reach is about level with it; see
[Why not now](#why-not-now).

## Where the time goes

Timed stage by stage (the GPU's reduction, the CPU's solve, the GPU's
back-transformation, under the pipeline), both batch backends are bound by
the reduction from 256, and the reduction by memory: about 170-190 GB/s.
Reading the lower triangle alone (eigh) and the trailing block once a step
(the SVD) took 1.2-1.9x off it. At 128 the CPU's solve is as long as the
GPU's work.

## Tried: the panel's own columns in registers

Each panel step also reads the panel's own columns, $V$ and $W$ (the SVD: $V$
and $X$), three times: for the column's update, for the corrections' dot
products and in the corrections themselves; by estimate as much as the
trailing product at $N = 256$. On eigh's panel (2026-10-08), a thread a row,
its row of $V$ and $W$ kept in registers as the steps formed them, so that
no step read them: the update and the corrections register-local, the dot
products summed across the lanes (five shuffle stages) and then the
simdgroups (a partial-sums array; then compare-and-swap into the result);
and, third, the dot products left in memory. Every variant was 10-17% slower
at 128-512 (256 × 256²: 37.5-38.9 ms against 32.3-33.2). The 64 floats a
thread holds cost occupancy, fewer simdgroups a core to hide memory latency
in a kernel bound by memory, and the reads they saved were mostly cache hits
(the panel's columns are 64 KB a matrix at 256, read again within a step).
The SVD's panel, which already holds a column in registers for its fused
pass, was not tried.

## What

Batches of matrices of about 96 to 512 go to the CPU for eigh and the SVD,
which spreads a batch over every core. QR's batches of these sizes went to
the GPU once it had a blocked kernel of its own
([qr-mid-size-kernel.md](qr-mid-size-kernel.md)); the question is whether
eigh and the SVD could have the same.

## The measurements

M5 Pro, 2.16.0, `sweep_eigh` and `sweep_svd` with MLX's buffer cache on,
median, ms:

| batch x shape | CPU | best GPU backend |
|---|---|---|
| eigh, 1024 x 96^2 | 22.5 | 75 (block Jacobi) |
| eigh, 256 x 128^2 | 10.1 | 41 (block Jacobi) |
| eigh, 1024 x 128^2 | 38.7 | 174 (block Jacobi) |
| eigh, 64 x 256^2 | 12.2 | 74 (block Jacobi) |
| eigh, 256 x 256^2 | 44.5 | 328 (block Jacobi) |
| eigh, 16 x 512^2 | 16.4 | 89 (tridiag) |
| eigvalsh, 1024 x 96^2 | 12.0 | 64 (block Jacobi) |
| eigvalsh, 1024 x 128^2 | 20.7 | 144 (block Jacobi) |
| eigvalsh, 256 x 256^2 | 21.0 | 250 (block Jacobi) |
| eigvalsh, 16 x 512^2 | 7.5 | 58 (band) |
| SVD, 1024 x 96^2 | 42.5 | 84 (block Jacobi) |
| SVD, 1024 x 128^2 | 77.2 | 187 (block Jacobi) |
| SVD, 256 x 256^2 | 76.8 | 359 (block Jacobi) |
| svdvals, 1024 x 128^2 | 52.9 | 1626 (bidiag) |
| svdvals, 256 x 256^2 | 39.1 | 867 (bidiag) |

At 256 x 256^2 the CPU does eigh's roughly 9 N^3 flops a matrix at about
0.9 TFLOP/s: Accelerate runs on the CPU's matrix units, a batch spread over
18 cores.

## Why not now

QR's mid-size kernel works because QR is one-sided: its updates are matrix
products (8 x 8 simdgroup MMA, 2.7 TFLOP/s) and its panels a short chain.
eigh and the SVD are two-sided, and their GPU kernels for small matrices
(`ql`, `golub_kahan`) rely on the whole matrix, and then the eigenvectors,
in threadgroup memory: that ends at N = 87 and k = 83 (32 KB). Beyond it:

- **The QL or implicit-QR phase** applies about N^2 dependent rotations to
  the vectors, each touching two columns. In threadgroup memory a sweep is
  one barrier; with the vectors in device memory it is some 50 MB of traffic
  for one 128 x 128 matrix (about 25 thousand rotations of two 128-float
  columns), 50 GB for a batch of 1024: far past the CPU's 39 ms.
- **The vectors in registers**, a row a thread (128 floats a thread at 128),
  as QR's register kernel keeps its matrix: a rotation's columns are
  indexed at run time, which registers do not allow; unrolling every sweep
  over all N column pairs (as QR's kernels unroll their columns) costs N
  checks a pair, and the tridiagonalization would need the rotate-to-index-0
  scheme of QR's register kernel as well. A rewrite of both kernels, at the
  occupancy of 130 registers a thread.
- **The ideal**: at 1024 x 96^2 eigh is about 7.7 MFLOP a matrix, 7.9 GFLOP
  in all; at the 0.5 TFLOP/s the `ql` kernel reaches on its latency-bound
  phases, 16 ms against the CPU's 22.5 (1.4x), and only where everything
  fits.
- **A two-stage reduction in one threadgroup** (to a band by products, then
  the band chased to tridiagonal in threadgroup memory, then QL) would
  replace the BLAS2 traffic of the tridiagonalization, but leaves the QL
  phase's rotations, the larger part with vectors.

The values alone (eigvalsh, svdvals) need no vectors, so the packed lower
triangle (N (N + 1) / 2 floats) in threadgroup memory would take `ql`'s
values-only path to N = 123; but the CPU's values are faster still (12 ms at
1024 x 96^2), and `ql_vals` reaches about 1.6x the CPU only at 4096 x 32^2.

## If someone picks it up

The narrow window is 88-128 with vectors, at batches of 1024 or more: the
`ql` and `golub_kahan` kernels with the vectors in registers (a row a
thread) and their rotations' column pairs unrolled. Expected at best 1.2-1.5x
the CPU there; measure `ql` at 64 x 64 against the CPU first, since the
extension can only be slower per flop. Effort: 3-5 days.
