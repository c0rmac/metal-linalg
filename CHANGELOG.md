# Changes

## 2.17.0

- **QR modes**, as `numpy.linalg.qr`'s and `torch.linalg.qr`'s, in every
  API: `"reduced"` (the default, as before), `"r"` (R alone: Q is never
  formed) and `"complete"` (a square M x M Q, and R [M, N] with zero rows
  below K). Every backend takes the mode and the routing is unchanged. R
  alone is bit-for-bit the reduced R and, on an M5 Pro, 1.3-1.8x faster for
  batches and large matrices on the GPU (4096 of 32 x 32: 0.51 ms against
  0.90; one 4096 x 4096: 47 ms against 62), about the same for one matrix up
  to 1024 x 1024, and 2.4-2.8x for a lone 128 x 128 or 256 x 256 on the CPU
  (`benchmark_qr --modes`). Calls alternating modes on a shape share the
  blocked QR's workspace rather than rebuilding it (10% a call at one
  1024 x 1024 otherwise). The complete Q's first K columns are the reduced Q's.
  - C++: `qr_accelerated(a, mode)` and `core::qr(a, q, r, QrMode)`; the
    three-argument forms remain (and remain exported).
  - C: `metal_linalg_qr_with_mode(a, batch, rows, cols, mode, q, r)` with
    `METAL_LINALG_QR_REDUCED`, `_R` (`q` may be NULL) and `_COMPLETE`.
  - Python with MLX: `ml.qr(a, mode=...)`; `"r"` returns R alone, as numpy.
  - PyTorch: `mlt.qr(A, mode=...)` now computes `"complete"` for M > N
    (it raised `NotImplementedError`), and `"r"` no longer forms Q unless
    `A` requires grad; the operator is `metal_linalg::qr(Tensor a, str
    mode="reduced")`.
  - Swift: `qrAccelerated(..., mode: .r)` and `.complete`, on `[Float]` and
    on `MLXArray`.
- **eigh for batches of mid-size matrices: the `tridiag_batch` backend**,
  where every GPU backend had lost to the CPU: the `tridiag` method for a
  whole batch at once, the reduction of every matrix by the same dispatches
  (a threadgroup a matrix and panel, `td_panel`), the tridiagonal problems on
  the CPU's cores, the back-transformation as batched products (blocks of 64,
  T built on the GPU), pipelined over chunks so that the GPU's stages run
  under the CPU's. The panel kernel is bound by memory and reads only the
  lower triangle for its symmetric products (32 x 32 tiles, a tile's column
  terms summed across a simdgroup by shuffles): 1.5x the whole-matrix read at
  16 x 1024^2. On an M5 Pro 1.46x the CPU at 1024 x 128^2, 1.36x at
  1024 x 96^2, 1.28x at 256 x 256^2, 1.55x at 16 x 1024^2 with eigenvectors;
  eigenvalues alone 1.2-1.35x at 1024 x 96-128^2. Routed by a window of N and
  batch per device (`tridiag_batch_min_n`, `_max_n`, `_min_batch`, and
  `values_` ones).
- **eigh with eigenvectors in two stages: the `band` backend with
  eigenvectors**, as the SVD's: the band reduction's and the chase's
  reflectors kept and applied to the eigenvectors on the GPU while the CPU
  chases and solves. 1.12x `tridiag` at 3072, 1.36x at 4096, 1.59x at 8192
  (1.31 s against 2.09; 13.9x the CPU). Routed from `band_min_n`.
- **The `ql` backend in registers up to N = 32** (`eigh_ql_simd`): the matrix
  a row a lane, under 1 KB of threadgroup memory each instead of 4 KB, four
  matrices a simdgroup up to N = 8 and two up to 16, their QL iterations side
  by side. 1.1-1.5x with eigenvectors at 17-32, 1.25x at 12-16 and 2x up to 8;
  eigenvalues alone by bisection, a lane an eigenvalue, 1.4-3x at every N.
  `EIGH_QL_SIMD=0` turns it off.
- The eigensolver's policy gains seven fields (`band_min_n` and the
  `tridiag_batch` windows), with their `EIGH_*` variables, in the C API's
  `metal_linalg_eigh_policy` (appended, as before: C code built against an
  older header needs rebuilding before it calls `metal_linalg_eigh_policy_set`),
  Python, PyTorch and Swift; `EIGH_DEVICE=band` now means `band` with
  eigenvectors too (before, `tridiag`), and `EIGH_DEVICE=tridiag_batch`
  forces the new backend. Sweep backends `tridiag_batch`, `tridiag_batch_vals`
  and `band`, and stages 4c and 5 of `tuning/tune_eigh.py`, fit them; the
  eigh kernel version is 8, and the M5 Pro is re-measured. A Mac measured
  before keeps the new fields at 0 (never) until it is measured again.
- **SVD for batches of mid-size matrices: the `bidiag_batch` backend**, the
  SVD's counterpart of `tridiag_batch`: every matrix bidiagonalized by the same
  dispatches (`bd_panel`, a threadgroup a matrix and panel), the bidiagonal
  problems on the CPU's cores, both back-transformations as batched products,
  pipelined over chunks. Each panel step reads the trailing block once, where
  `slabrd` reads it twice (a column in registers gives both its product with
  v and its share of A u): 1.3-1.9x from 256 x 256. A matrix at least twice
  as tall or as wide as k goes through this library's QR first, as on the
  CPU, and only R is bidiagonalized (2.4-2.5x at 256 x 1024x128 and
  128x1024; `SVD_BIDIAG_BATCH_QR=0` turns it off). Singular values alone
  from k = 160 go in two stages, a band on the GPU (blocks of batched
  products, `bb_panel` for the panels), then bidiagonal on the CPU's cores:
  1.5x the direct reduction at 512 and 2-2.3x at 1024, 3.3x the CPU at
  16 x 1024^2 (`SVD_BIDIAG_BATCH_BAND=0` turns it off). On an M5 Pro 1.65x the CPU at 1024 x 128^2, 1.42x at
  256 x 256^2, 1.12x at 64 x 512^2, 1.68-1.77x at 256 x 512-1024x128 with
  vectors; singular values alone 2.3x at 1024 x 128^2. Up to 1024 rows and
  columns, any rows when twice as tall as wide. Routed by a window of k, l and
  batch per device
  (`bidiag_batch_min_k`, `_max_k`, `_min_batch`, `_max_l`, and `values_`
  ones).
- **The `golub_kahan` SVD backend in registers up to 32 x 32**
  (`svd_gk_simd`): the matrix, U and V a row a lane, four matrices a
  simdgroup up to 8 rows and two up to 16, their QR iterations side by side;
  from 17 rows with vectors, a runner simdgroup runs 8 matrices' QR
  iterations while their simdgroups apply the last step (1.1-1.35x;
  `SVD_GK_RUN=0`). 1.6-2.1x with vectors up to 16 x 16, 1.3-1.6x at 17-32;
  singular values alone by bisection on the Golub-Kahan tridiagonal, a lane a
  value, 1.6-2.7x. `SVD_GK_SIMD=0` turns it off.
- The SVD's policy gains eight fields (the `bidiag_batch` windows), with their
  `SVD_*` variables, in the C API's `metal_linalg_svd_policy` (appended:
  rebuild C code before it calls `metal_linalg_svd_policy_set`), Python,
  PyTorch and Swift; `SVD_DEVICE=bidiag_batch` forces the new backend. Sweep
  backends `bidiag_batch` and `bidiag_batch_vals` and stage 4 of
  `tuning/tune_svd.py` fit them; the SVD kernel version is 10, and the M5 Pro
  is re-measured.
- **The blocked QR's panels 8 columns wide for up to 4 matrices of 768 to
  3072 rows** (16 elsewhere): a tall panel's TSQR top is a tree of chains
  whose cost grows as the width squared, a third of the call at 1024 x 1024.
  1.05x at 768-1024^2, 1.1x at 1536-2048^2, 1.06x for 4 of 1024^2
  (`QR_PANEL_WIDTH=8` or `16` forces one).
- **The blocked QR's updates inside an aggregate in one kernel**
  (`qr_agg_apply`) for up to 4 matrices and panels of up to 3072 rows: one
  dispatch where two MPS products took two; 1.16x the call at 1024 x 1024,
  1.06-1.12x at 512-3072 (`QR_AGG_KERNEL=0` keeps MPS). Where it uses the
  input in place, its scan (the scale, and NaN) is the GPU's too: 1.01-1.03x
  (`QR_GPU_SCAN=0`).
- **QR's register kernel packs small matrices**: up to 8 rows four a
  simdgroup, up to 16 two (`QR_SIMD_PACK=0` turns it off): 2.5x at 4 x 4,
  1.7-2.5x at 8 x 8, 1.3-1.7x at 16 x 16 for large batches.
- **The SVD with vectors for batches in two stages**: `bidiag_batch` from
  k = 288 (from 128 for batches of up to the CPU's solve threads) reduces
  every matrix to a band by blocks of batched products, as
  for the singular values alone, keeping both stages' reflectors: the CPU's
  cores chase each band to bidiagonal (keeping the chase's) and solve it
  while the GPU forms Q1 and P1; the GPU applies the chase's to them, a
  dispatch a matrix, and forms U and V^T. On an M5 Pro 1.95x the CPU path for
  one 1024 x 1024 (29.6 ms against 57.7), 2.2x for 4, 2.7x for 16, 1.38x for
  8 of 768; 2.7-3.4x the direct reduction at 1024. `SVD_BIDIAG_BATCH_BAND=0`
  turns it off with the values' two stages. The band blocks' two updates are
  merged into three passes over the trailing matrix and five dispatches
  instead of four and eight (the row panel's kernel applies the left update
  and forms V2 T2): 1.04-1.16x, the singular values alone 3.8x the CPU at
  16 x 1024^2 (was 3.3x).
- **eigh with eigenvectors for small batches in two stages**:
  `tridiag_batch` from N = 64, for batches of up to half as many matrices as
  the CPU's solve has threads (all of them from 640, twice from 896), the
  same for the symmetric case: four dispatches a block (the panel kernel
  forms V T, one kernel `Y = X - V (T^T V^T X) / 2`, the two-sided update one
  rank-32 product). On an M5 Pro 2.6x the one-stage reduction for one
  1024 x 1024 (18 ms; 1.86x the CPU), 1.86x for 4, 1.39x for 8, 1.48x for 24;
  1.3x for 8 of 768, 1.4-2.4x for small batches of 64-96.
  `EIGH_TRIDIAG_BATCH_BAND=0` turns it off. For eigenvalues alone it lost to
  LAPACK's two-stage driver and is not used.
- **The `band` backends with vectors hand a batch of two or more of 384-1024
  to those paths** (`SVD_BAND_BATCH=0`, `EIGH_BAND_BATCH=0` keep it): SVD 2 of
  1024^2 in 36 ms against 55, 4 in 46 against 111; eigh 4 in 33 against 68.
  The tuners now fit `band_min_k`/`band_min_n` and the batch cap together
  (stages 3c and 4c), since `band` takes batches.
- **The CPU path's divide and conquer on every core**: eigh with
  eigenvectors from N = 192 and the SVD with vectors from k = 192, for a
  batch of at most a quarter as many matrices as cores, call LAPACK's
  drivers' steps one by one (`ssytrd`, `sormtr`; `sgebrd`, `sormbr`, the QR
  first for tall and wide matrices as before) with the `tridiag` and `bidiag`
  backends' divide and conquer on the idle cores in between (`sstedc` and
  `sbdsdc` run on one). On an M5 Pro eigh 1.26x `ssyevd` at 256, 1.2x at 1024,
  1.17x at 2048; the SVD 1.23x `sgesdd` at 256, 1.32x at 1024, 1.4x at 2048,
  1.16-1.3x tall or wide; the same values (bit for bit for eigh).
  `EIGH_CPU_DC=0` and `SVD_CPU_DC=0` keep the drivers. The CPU path is every
  boundary's baseline, so both are re-measured.
- **Batch backends' pipelines** in eight chunks for matrices of up to
  128 x 128 (four above): 1.07-1.15x at 1024 x 96-128^2. Their
  back-transformations' T built in blocks (16 x 16 diagonal blocks a thread a
  row, then merged in pairs: five barriers where column by column took 128):
  1.04-1.05x for the SVD, 1.02-1.15x for eigh at 96-384 batches.
- **The measurement takes half the time** (eigh and SVD about 40 minutes
  instead of 75 on an M5 Pro): the Jacobi kernels are timed only up to
  N = 96 and k = 128, where they can win, with a canary beyond that warns if
  they ever do (`--full-grid` times them everywhere); a second pass repeats
  only the points whose choice the first left open (`--full-passes` repeats
  all); a call of 20 ms or more gets one warm-up instead of two. Re-analysed
  on the earlier runs, the routing is the full grid's.
- **Fixed: Q and singular or eigen vectors far from orthogonal for some
  exactly rank-deficient matrices** on the GPU paths built on the band
  reduction's panel kernels: the blocked QR (one matrix from about 384 x 384,
  and batches of large ones), the `band` backends of eigh and the SVD, and
  the SVD's QR-first reductions. A constant matrix of ones, 600 x 64, had a
  Q off by 5e4 (`Q^T Q - I`), one of 1024 x 1024 by 3e5; the SVD's `band`
  at 1024 x 1024 by 3e7; eigh's `band` at 1024 by 0.56. Such a matrix's
  trailing columns become rounding noise, then noise of that, down to
  entries near 1e-19 of the largest, whose squares underflow, some and not
  others: the kernels' plain sums of squares then missed part of the vector
  they normalized, and the reflector was not orthogonal. A rest whose sum of
  squares is below 2^-80 (a norm below 2^-40 of the matrix's largest entry,
  each matrix being scaled into [0.5, 1)) is now taken as zero in every
  kernel that sums squares plainly (the panels, the TSQR's tree, the
  register and threadgroup Householder kernels of QR, eigh and the SVD, and
  the streaming QR, whose threshold was 1e-30). Orthogonality on those
  matrices is now 1e-5 or better, as LAPACK's; tests added for each path.
- `benchmark_qr --modes` times the three modes against each other.
- **README**: the introduction states the measured gains (one large matrix
  1.6-10x against LAPACK on every CPU core, 11.7x at 8192; batches of
  thousands of small matrices 1.6-5.7x; 4.6-34x against `torch.linalg`), and
  its performance tables are re-measured on 2.17.0.

## 2.16.0

- **QR kernels for batches of small and mid-size matrices**
  (`qr_householder`), which are now the `unblocked` backend: LAPACK's
  methods, a matrix to a simdgroup or a threadgroup, as the SVD's
  `golub_kahan` and the eigensolver's `ql` are built. Up to 32 columns and
  128 rows in a simdgroup's registers (`sgeqr2`, `sorg2r`: rows a lane, a
  step's dot products four columns to a `simd_sum`, no barrier); up to 4096
  rows blocked in a threadgroup (`sgeqrf`, `sorgqr`: panels of 16 columns in
  registers, T built on the way, the updates by blocks of 32 columns as 8 x 8
  simdgroup matrix products, Q's start and R folded into the passes, an
  aligned input read directly, and a small batch given up to 16 simdgroups a
  matrix). The input read row-major, scanned and scaled on the GPU, Q and R
  written straight out: no CPU pass. On an M5 Pro, through MLX: 1024 of
  128 x 128 in 4.4 ms (the blocked QR 14.5, the CPU 24.5), 256 of 256 x 256
  in 6.2 (14.5, 17.7), 4096 of 64 x 64 in 3.9 (13.1, 9.0), 4096 of 32 x 32 in
  0.87 (3.5, 2.6), 16 of 2048 x 64 in 1.1 (1.7, 2.8). See
  [qr-small-kernel.md](docs/proposals/qr-small-kernel.md) and
  [qr-mid-size-kernel.md](docs/proposals/qr-mid-size-kernel.md).
- `unblocked`'s own kernel (`QR_Unblocked.metal`), 2.5-6x slower wherever it
  ran, is retired; beyond 4096 rows `unblocked` hands the call to the blocked
  QR.
- **Callers' Metal buffers are no longer wrapped again**: through the MLX
  API the input's and outputs' own buffers are used (`KnownBuffer`), rather
  than new buffers over the same memory, whose pages the first command
  buffer maps at about 1 ms a 64 MB; the C API takes a caller's buffer for a
  call (`metal_linalg_know_buffer`, `metal_linalg_forget_buffer`), and the
  PyTorch package passes its MPS tensors' that way. 10-25% off a call for
  large batches of small matrices, every decomposition
  ([known-buffers.md](docs/proposals/known-buffers.md)).
- **The sweeps and benchmarks keep MLX's buffer cache on**, as an MLX program
  has it (off since 2.0): every call's outputs were fresh pages the GPU maps
  on first use, so the timings counted allocation as much as the
  decomposition (4096 of 64 x 64 QR: 6.4 ms on the GPU with it off, 4.0 on;
  the CPU 10.5 and 9.0). **All three decompositions re-measured** on the M5
  Pro (kernel epochs qr 7, eigh 7, svd 9; run
  [`20261007-9f2589`](docs/results/apple-m5-pro-20gpu/20261007-9f2589/summary.md)).
- **QR's routing**: the kernel crossover is on k = min(M, N) rather than
  rows, split by batch (the new kernels walk a matrix's columns with its
  rows in parallel; the blocked QR repays its dispatches on large matrices
  or small batches), and the GPU-or-CPU rule on w = floor(sqrt(M k)) rather
  than k (a tall narrow batch is the unblocked backend's in parallel by its
  rows). On the M5 Pro the blocked QR takes k >= 80 below 8 matrices, k >=
  768 from 8; the GPU takes w <= 448 with batch * w >= 1448, or sqrt(M k) >=
  512 at any batch. 1.020x geometric-mean regret against the best backend at
  each of 221 shapes (held out 1.030x). The fields keep their names.
- eigh's M5 Pro row: the GPU for N <= 16 with batch * N >= 8192, or N <= 48
  in batches of 256+, batches shared with the CPU from 4096 (1.023x regret);
  the SVD's: `golub_kahan` batches shared from 256, the GPU for k <= 80 in
  batches of 256+, `band` for singular values from k = 768 (1.017x).
- `tune_qr.py`: the kernel crossover fitted on k, on the shapes where a GPU
  kernel beats the CPU (fitted on every shape it sent 1024 of 128 x 128 to
  the blocked QR), its batch split shipped where it survives held-out data;
  the GPU-or-CPU rule fitted on sqrt(M k); the unblocked backend timed on
  the tall large shapes too; ties in the large-matrix clause's batch cap go
  to the GPU; the grid's kernel crossover goes down to 64 (80 x 80 and 96 x
  96) and its mid-size batches up to 384.
- Proposals: eigh and the SVD for batches of mid-size matrices analysed
  ([eigh-svd-mid-size.md](docs/proposals/eigh-svd-mid-size.md)): the CPU path
  is 2-10x ahead there, and a GPU kernel would at best be level with it.
- README: QR in the table of large batches of small matrices; every
  performance table re-measured.

## 2.15.0 (2026-10-07)

The proposals left after 2.13.0 ([docs/proposals/](docs/proposals/README.md)),
and what turned up while doing them; measured in
[the 2.15.0 study](docs/studies/proposals-2-15-apple-m5-pro.md).

- **A blocked QR** (`qr_blocked`), which the grid-parallel backend
  (`streaming_reduced`) hands every call it can: the band reduction's panel
  kernels (TSQR for tall panels) on the row-major matrices in place, panels
  of 16 columns gathered into aggregates of 128 whose T is merged on the
  GPU, the updates and Q's formation as rank-128 MPS products; each matrix
  padded with zero rows and columns to whole panels, so the GPU takes every
  column; a batch at once (every kernel's grid takes the batch, every product
  is one batched MPS product); any height up to 2^22 rows (the TSQR's tree
  of leaves). On an M5 Pro, one matrix against 2.14's GPU path and the CPU:
  1024 6.5 ms (15, 17), 2048 18 (49, 93), 4096 60 (231, 619), 8192 x 512 8.5 (64, 48), 100000 x 32 4.0 (58, 7.8); batches: 16 x
  1024^2 24 ms (51, 45), 4 x 2048^2 37 (99, 143), 1024 x 128^2 18 (52, 25).
  Accuracy LAPACK's or a little better. `QR_BLOCKED=0` keeps the streaming
  kernels.
- **QR's routing re-measured with it** (kernel epochs qr 4, then 5): on the
  M5 Pro the GPU from `sqrt(M k) >= 512` in a batch of up to 64 (was k >=
  1024, batch <= 4) besides large batches of k = 16-256, the kernel
  crossover at 128 rows (was 512), batches shared with the CPU from 1024
  (was 64); 1.013x geometric-mean regret over 207 shapes. The large-matrix
  clause now compares `sqrt(M k)`, rows and `k` both, rather than `k`
  (`gpu_large_min_k` keeps its meaning for square matrices): a rule on `k`
  sent a tall 8192 x 512 to the CPU (48 ms against 8.5); `tune_qr.py`'s grid
  gains tall matrices up to 16384 x 64 and batches of 16 large ones, and its
  fit tries the clause and the window in both orders. Wider panels, taller
  TSQR leaves and a look-ahead on a second queue were tried and gave nothing
  ([docs/proposals/qr-fixed-costs.md](docs/proposals/qr-fixed-costs.md)).
- **The SVD with vectors by the two-stage reduction** (`band` with vectors,
  `svd_band_vectors`, routed from the new `band_min_k`): the band reduction
  keeps its reflectors (written straight into blocks of 128 as the panels
  factor), the bulge chase keeps its own, and Q = Q1 Q2 and P = P1 P2 are
  formed on the GPU while the CPU chases the band and solves the bidiagonal
  problem; then U = Q U_B and V^T = V_B^T P^T, one product each. The
  chase's reflectors are applied by a new kernel, `bd_chase_apply`: the
  blocks of 16 sweeps a step, four groups of sweeps at once a threadgroup,
  each a simdgroup two tiles behind the last, tiles handed down through
  threadgroup memory, each block two products (its V and Y = -T^T V^T,
  built on the CPU): about 49 ms a side at 4096, where the groups one after
  another took 176. The divide and conquer's top products go to the GPU
  once Q2 and P2 are done (the CPU is then the bottleneck). The GPU's work runs back to back: Q1 and P1's queued
  during the band reduction, Q2 and P2's released in two chunks as the chase
  finishes their sweeps (1.04-1.06x over waiting for the chase). On an M5
  Pro, one square matrix against `bidiag`: 1.21x at 1024, 1.45x at 2048,
  2.35x at 4096 (402 ms against 944; 368 in alternating runs), about 2.7x
  at 8192; 8.7x the CPU path at 4096. Accuracy LAPACK's (reconstruction and
  orthogonality 6e-6 at 4096); at 8192 the call keeps about 1.2 GB more than
  `bidiag`. `SVD_DEVICE=band` now means `band` with vectors too (before,
  `bidiag`); `band_min_k` and `SVD_BAND_MIN_K` in the C API, Python, PyTorch
  and Swift; a `band` sweep backend and stage 3c of `tuning/tune_svd.py` fit
  it. Until a Mac's routing is measured with it, `band_min_k` is 0 there
  (never).
- **Long Jacobi solves split over dispatches.** The whole-matrix Jacobi
  kernels (the eigensolver's threadgroup mode, the SVD's Jacobi kernel) give
  a matrix one threadgroup for its whole solve, which with the display busy
  macOS ends after about a quarter of a second ("GPU Hang Error"): on an M5
  Pro from N ~ 400 (eigh) and 512 x 512 (SVD), reached by forced kernels and
  estimated policies. A solve the cost model puts over 40 ms
  (`EIGH_DISPATCH_MS`, `SVD_DISPATCH_MS`) now runs a few rounds a dispatch,
  each matrix resuming where it stopped: the same result bit for bit, in the
  same time, about 15 ms a dispatch where one threadgroup ran 1.7-2.8 s.
- **The band chase under the band reduction**, for eigenvalues or singular
  values alone, one matrix: the chase starts at once on threads of its own
  and trails the GPU, each block's rows copied as it completes, a thread
  keeping several sweeps open. Every sweep runs to the band's end, which the
  GPU finishes last, so only about 6% of the chase can go before it:
  eigvalsh and svdvals on `band` 1.04-1.05x at 2048-4096, the values bit
  for bit the same.
- **The divide and conquer's largest products on the GPU**, for one matrix
  in `tridiag` and `bidiag` (whose GPU is idle meanwhile): products of a
  gigaflop or more, from the top merges of n ~ 2048, as MPS products on the
  merges' page-aligned temporaries in place (`GpuGemm`). eigh with vectors
  1.075x at 4096, the SVD on `bidiag` 1.046x. `bidiag`'s divide and conquer
  writes straight into its Metal buffers instead of vectors copied there.
- **The divide and conquer on every core.** With vectors, `tridiag` and
  `bidiag` solved the tridiagonal or bidiagonal problem with LAPACK's
  `sstedc` and `sbdsdc`, on one core. `src/divide_conquer.cpp` walks the same
  tree with LAPACK's routines, its leaves and small merges one per task and
  its large merges' loops (the secular equation's roots, the corrected z, the
  vectors, the products) spread over the cores but two; `slasd2`'s
  deflation, which moved the right vectors' rows one strided row at a time
  (130 ms of a 4096 merge's 190), is rewritten to move them a column at a
  time. On an M5 Pro, a 4096 problem: `sstedc` 176 ms to 50, `sbdsdc` 753 to
  100. The values are LAPACK's bit for bit, the vectors to the last bits,
  and neither depends on the number of threads. With the block reflectors'
  change below, at 4096 eigh with vectors takes 306 ms (438 in 2.14) and the
  SVD with vectors 943 (1535): 8.2x and 3.7x the CPU path.
- **The band backends' panels**: the TSQR top factors its stacked R's as a
  binary tree of triangle pairs, a simdgroup a pair, and the panel kernels
  no longer run in IEEE mode. `Svd_Bidiag.metal` is built with
  `-fno-fast-math`, and a kernel with any IEEE division or square root in
  it, even an untaken one, compiles all of its arithmetic that way; they now
  use the fast approximations with a Newton step. A 4096 x 16 panel 140 us to
  76. svdvals on `band` 1.13-1.30x faster at 1024-4096, eigvalsh 1.08-1.18x.
- **The band reductions' small products** (a b x b product summed over the
  trailing rows and two b wide) are two kernels a block instead of three MPS
  products: 2-5% at 1024-2048.
- **The symmetric band reduction's trailing update on the lower triangle**
  (`sb_update`), each off-diagonal tile's transpose written over its mirror
  so that MPS still computes X = A22 V T on the whole: 1.5 n^2 of memory a
  block instead of 2 n^2, and faster than MPS's update at every size
  (4096: 462 us against 611). eigvalsh on `band` 1.08x at 4096, 1.17x at
  8192 (872 ms to 744).
- **`values_band_width`** in `EighPolicy` and `SvdPolicy` (0: 16), the C
  API, Python, PyTorch and Swift, and `EIGH_VALUES_BAND_WIDTH` /
  `SVD_VALUES_BAND_WIDTH`: the band backends' width as part of the per-device
  policy. The sweeps time the band at widths 8 and 32 too (`band8_vals`,
  `band32_vals`); stages 3b and 4b choose the width, then fit the threshold.
  On the M5 Pro 16 is the fastest or within 2% everywhere but eigvalsh at
  4096 (32, 7% faster).
- **The band thresholds' fit** compares a threshold with the best on the
  points where they choose differently (within 3% there), not only over all
  the band points, where one clear loss was diluted; and the grids gain
  N = 2560 and 3584 (eigh) and k = 1280 and 1792 (SVD). Re-analysed with it,
  the M5 Pro's eigvalsh goes to `band` from 3072 instead of 4096.
- **The back-transformations' block reflectors**: `tridiag` and `bidiag`
  built each block of 128 reflectors' T with `slarft` on the CPU while the
  GPU applied the previous block, and at 4096 the CPU's side (2.5 ms a block)
  was the slower; T now comes from the Gram matrix V^T V (one `ssyrk`) and
  the copies run on every core. eigh with vectors on `tridiag` 1.1x at 4096.
- **The one-stage reductions and bisection** off IEEE arithmetic too:
  eigvalsh on `tridiag` 1.07-1.11x. And the SVD's Jacobi kernel's rotation
  and output: 1.14-1.17x on batches of 16x16 to 48x48, its `rsqrt` with two
  Newton steps (with one, V's orthogonality was 10x worse; with two, a little
  better than with the IEEE sequence).
- Kernel epochs eigh 6, SVD 8 (whose runs must also time `band` with
  vectors), QR 5: every other Mac's routing is stale until it is measured
  again (`tuning/run.py`). The M5 Pro is re-measured (runs
  [`20261007-246324`](docs/results/apple-m5-pro-20gpu/20261007-246324/summary.md),
  eigh and SVD, and [`20261007-82345e`](docs/results/apple-m5-pro-20gpu/20261007-82345e/summary.md),
  QR): the SVD with vectors on `band` from k = 1024 (`band_min_k`), the
  singular values alone from 768 and the eigenvalues alone from 2048, at
  width 16; and the estimated policies of the Macs nobody has measured are
  refitted from it.
- On an M5 Pro, one matrix against the CPU path: svdvals 9.8x at 4096 and
  11.7x at 8192, eigh 8.3x and 8.9x, eigvalsh 3.6x and 3.8x, the SVD with
  vectors 4.2x at 4096 on `bidiag` and 9.8x on `band`, QR 10.0x (2.14: 8.6x,
  5.6x, 2.9x, 2.3x and 2.7x at 4096); README's tables, and its PyTorch
  comparison, re-measured side by side after the last change.
- Fixes: the sweeps' correctness gate failed every backend from N ~ 6500
  (MLX queued the comparison's GPU work behind the CPU reference, past the
  GPU's watchdog); `cpu_threads() - 2` wrapped around for a thread cap of 1
  or 2, and the band chase then ignored the cap; `sb_update` read up to 63
  rows past its staging buffer (never stored); `tuning/kernels.py` did not
  watch the band files for epoch changes; the divide and conquer, like
  LAPACK's `sbdsdc`, could fail to converge (one in about 30 random
  bidiagonals of 2048 with 900 equal singular values and the rest tiny, a
  test case's, which failed now and then), and the SVD threw: now it is
  solved again in double precision (`dbdsdc`, `dstedc`; 0.3 s at 2048), and
  by QR iteration if that fails too.
- New proposals: the CPU path's divide and conquer, the divide and
  conquer's products on the GPU, the band SVD with vectors overlapped
  further.

## 2.14.0 (2026-10-05)

- **Estimated policies for the Macs nobody has measured**, in place of the
  untuned default (the M1's 2.0-era thresholds, with every newer backend
  off), which on the M5 Pro's own timings is 1.3-1.9x the best routing on
  geometric mean and up to 16x at a point. A measured Mac's timings are
  refitted, by the same analysis that fits a run, as if its GPU were slower
  against its CPU by as much as Geekbench 7's Metal against multi-core scores
  say (and the memory bandwidth too, for the large-matrix backends), with a
  margin of 1.25x in the measured Mac's generation and 1.5x outside it; a Mac
  without published scores is estimated from its core counts, with 2x. On
  Macs simulated with a GPU 1-4x weaker, the estimates are within 0-3% of the
  best routing on geometric mean, where the untuned default was 1.3-3x
  ([the study](docs/studies/estimated-policies.md)). Policy sources say
  `estimated:<chip> (from Apple M5 Pro, ...)`, and `calibration_status()`
  reports `uncalibrated` for them as before; the notice still asks for a
  measurement. `METAL_LINALG_ESTIMATE_AS="<chip>[:<GPU cores>[:<CPU
  cores>]]"` estimates as that Mac, even on a measured one.
  `tuning/chip_specs.py` holds every Mac's scores and bandwidth,
  `tuning/estimate.py` the refit; `tuning/generate_tables.py` writes the
  estimated rows (`src/tuned/*_estimated.inc`, `chips.inc`) with the measured
  ones, so every new measurement refits its neighbours' estimates.
  `tests/test_estimate.cpp`, and the correctness suites run again under an
  estimated M1's routing.
- **Runs from before 2.9.0 are no longer used** (`MIN_EPOCHS` in
  `tuning/kernels.py`): the CPU path then ran on one core, and every
  GPU-or-CPU boundary has moved since. The M1's QR and eigh rows were such;
  the M1 is now estimated like any other unmeasured Mac, and the tables and
  the measurements page mark its runs out of date.
- **A fault in the SVD harness's fit for singular values alone**: stage 2b
  left out the batches shared with the CPU, which on a GPU sharing from small
  batches left the rule unconstrained. Fixed; the M5 Pro's refitted row moves
  `values_gpu_max_l` from 56 to 256, so tall batches of singular values alone
  (1024 and more of 64-256 x 8-32) go to `golub_kahan` shared with the CPU,
  1.15-1.9x faster (svdvals 1.029x the best on geometric mean, from 1.043x).
- **MPS tensors in place** in the PyTorch package: on Apple Silicon PyTorch
  keeps MPS tensors in Metal buffers in shared storage, and the library now
  reads its input there and writes its results into new MPS tensors, instead
  of copying the input to the CPU and the results back. A call waits for the
  work queued on MPS first (`torch.mps.synchronize()`). On an M5 Pro the
  copies made a handful of matrices 9-12x slower (eigh of 16 matrices of
  16×16: 1.2 ms, now 0.10 ms), large batches 1.2-1.8x, and one large matrix a
  few percent. `mlt.mps_in_place()` says whether it applies (a
  torch that keeps MPS tensors in private storage still copies), and
  `METAL_LINALG_TORCH_MPS_COPY=1` forces the copies.
- The C API gains `metal_linalg_buffer_contents(buffer, offset, bytes)`: the
  CPU address of a range of a Metal buffer in shared storage, NULL for one in
  private storage, for a range it does not hold, or for an object that is not
  a buffer. It is what passes a GPU framework's tensor memory to the
  decompositions without a copy ([docs/c-api.md](docs/c-api.md)).
  `tests/test_c_api_metal.mm` runs each decomposition on buffers
  sub-allocated from a shared heap, as PyTorch's allocator lays them out.
- **`MLXArray` in place** in the Swift package's `MetalLinalgMLX`: the input
  is read where MLX keeps it once evaluated (`asData(access:
  .noCopyIfContiguous)`), and the results are written into page-aligned
  memory that each output `MLXArray` then owns
  (`MLXArray(rawPointer:_:dtype:finalizer:)`), instead of copying the input
  into a `[Float]` and the results back. `MetalLinalgMLX` now needs mlx-swift
  0.32.2 or later (was 0.25.0): that initializer leaked what its finalizer
  captures before 0.32.2. The C++ MLX API and the Python MLX package already
  used MLX arrays in place.
- `benchmarks/benchmark_torch.py`: the README's comparison with
  `torch.linalg`, and (`--mps-ab`) MPS tensors in place against copied. The
  comparison in both READMEs re-measured with it (PyTorch 2.13 from
  conda-forge, its CPU LAPACK from Accelerate; it was PyTorch 2.14's).

## 2.13.0 (2026-10-04)

- **Two-stage reductions for eigenvalues alone and singular values alone**:
  a `band` backend in each solver, the reduction in two stages as LAPACK's
  `ssyevd_2stage` does it. A to a band of width 16 on the GPU, a block of
  columns at a time, its work matrix products that read the matrix a few
  times a block where the one-stage `tridiag` and `bidiag` reductions read it
  once or twice a column; then the band to tridiagonal or bidiagonal by
  Householder bulge chasing, its sweeps pipelined over the CPU's cores; then
  the eigenvalues or singular values by bisection on the GPU. A block's panels
  are factored in registers, one simdgroup for up to 128 rows, by TSQR beyond,
  with the Householder vectors rebuilt from TSQR's Q. On an M5 Pro, one
  matrix: `svdvals` 226 ms at 4096 x 4096 against `bidiag`'s 733 and the
  CPU's 1949 (8.6x), 1.15 s at 8192 against 11.95 s (10.4x); `eigvalsh`
  161 ms at 4096 (CPU 467, `tridiag` 209: 2.9x the CPU) and 872 ms at 8192
  (CPU 2588, `tridiag` 1749). The M5 Pro uses them for one or two matrices,
  svdvals from k = 1536 and eigvalsh from N = 4096.
  `values_band_min_n` in the eigensolver's policy and `values_band_min_k` in
  the SVD's (`EIGH_VALUES_BAND_MIN_N`, `SVD_VALUES_BAND_MIN_K`; 0, never, on
  a device without measurements) say from which size they are used;
  `EIGH_DEVICE=band` and `SVD_DEVICE=band` force them, `EIGH_BAND_WIDTH` and
  `SVD_BAND_WIDTH` set the width (8, 16, 32); `eigvalsh_backend()` and
  `svdvals_backend()` report `"band"`. The MLX layer adds
  `detail::eigh_band` and `detail::svd_band`.
- **Bisection on the GPU** for the eigenvalues of a tridiagonal (from
  N = 512) and the singular values of a bidiagonal (from k = 1024), in the
  `tridiag` and `bidiag` backends' value paths as well: 6 ms against
  `ssterf`'s 79 at 4096, 12 against `sbdsqr`'s 77. `eigvalsh` on `tridiag` is
  1.4-1.5x faster at 2048-4096 (2.1-2.2x the CPU), `svdvals` on `bidiag`
  1.1-1.2x.
- `bidiag` solves the bidiagonal for singular values alone below k = 1024
  with `sbdsqr` (dqds) instead of `sbdsdc`: faster, and accurate to every
  singular value of the bidiagonal.
- The C API's `metal_linalg_eigh_policy` gains `values_band_min_n` and
  `metal_linalg_svd_policy` `values_band_min_k`, each at its end; the Python
  and PyTorch packages expose both.
- The tuning harness times `band_vals` for both and fits the thresholds
  (stages 3b and 4b). Kernel versions: eigh 5, SVD 7. The M5 Pro is
  re-measured (run `20261004-06bc11`); its eigh row also turns the Jacobi
  kernel's simd mode off (it took N <= 8).
- [The two-stage study](docs/studies/two-stage-apple-m5-pro.md): where the
  time goes, the panel kernels, the parallel chase and the bisection.

## 2.12.0 (2026-10-04)

- **QR shares batches with the CPU** (`share_min_batch` in the QR policy,
  `QR_SHARE_MIN_BATCH`), as eigh and the SVD do since 2.11.0: from that batch a
  batch that goes to the GPU is solved by the GPU kernel and the CPU path at
  once. On an M5 Pro, against the faster of the two alone: 1.2x for 1024
  matrices of 128×128, 1.5x for 1024 of 192×192, 1.6x for 1024 of 256×256.
  `qr_shares_batch()` reports it, `detail::qr_shared` runs it directly. QR's
  GPU workspaces are kept and grown per shape, rather than kept per batch size
  for good.
- **A large-batch clause in the eigh and SVD routing** (`gpu_big_batch_max_n` /
  `gpu_big_batch_max_k` and `gpu_big_batch_min`, `EIGH_GPU_BIG_BATCH_*`,
  `SVD_GPU_BIG_BATCH_*`): the GPU also for sizes above the product rule's cap,
  up to a cap of its own, in batches of at least a batch of its own. Shared
  with the CPU, the GPU wins large batches of 64×64 and 80×80 (1.5-1.6x), which
  a rule `batch * k >= c` cannot take without also taking their small batches,
  which the CPU wins. `gpu_max_n = 0` / `gpu_max_k = 0` is still never the GPU.
- **Fewer dispatches per column in the large-matrix reductions.** The
  `tridiag` backend's reduction takes three dispatches per column instead of
  seven and the `bidiag` backend's four instead of twelve: each step that needs
  a whole vector done before the next is a kernel boundary, and everything
  else is folded into its neighbours (the Householder vector is formed from
  norm partials in every threadgroup of the product that uses it; the previous
  column is finished inside the next column's update). The row-sized steps run
  eight threads to a row, and the matrix is copied in on every core (a strided
  copy took 55 ms at 4096×4096). On an M5 Pro, one matrix: eigvalsh 1.67x at
  N = 1024, 1.47x at 2048, 1.32x at 4096; eigh 1.49x, 1.34x, 1.24x; svdvals
  1.76x, 1.52x, 1.22x; svd 1.34x, 1.24x, 1.12x. Accuracy is unchanged: the
  values differ from float32 LAPACK's as much as before (3e-6 to 2e-5 of the
  largest), and the eigenvector residual is about 1e-6. Every sum over
  threadgroups is taken in a fixed order.
- The C API's policy structs gain these fields at their ends:
  `metal_linalg_qr_policy` `share_min_batch`; `metal_linalg_eigh_policy`
  `gpu_big_batch_max_n`, `gpu_big_batch_min`; `metal_linalg_svd_policy`
  `gpu_big_batch_max_k`, `gpu_big_batch_min`.
- **M5 Pro re-tuned.** QR re-measured (run `20261004-4d6208`): GPU for
  `k <= 128` and `batch * k >= 40960`, shared from batch 64; against the best
  backend at each of the 185 shapes it scores 1.0013 geometric-mean regret,
  worst 1.17x (2.11.0's row: 1.0081, worst 1.45x). eigh and the SVD
  re-measured after the reductions changed (run `20261004-fd9bd8`; epochs
  eigh 4, SVD 6): `tridiag` from N = 1024 for up to four matrices (was 1536
  and two), from 1536 for eigenvalues alone (was 3072); `bidiag` from
  k = 1024 for singular values alone too (was 2048), for up to two matrices;
  the large-batch clause takes eigh up to N = 64 and the SVD up to 80×80 in
  batches of 1024 and more. eigh scores 1.0094, worst 1.38x (the row it
  replaces: 1.0112, worst 1.59x), the SVD 1.020 (the product rule alone:
  1.028). One matrix against the CPU, at N = 1024 / 4096: eigh 1.61x / 5.44x
  (2.11.0: 1.13x / 4.55x), eigvalsh 1.09x / 1.58x (0.68x / 1.22x), svd
  1.40x / 2.26x (1.02x / 2.03x), svdvals 1.35x / 2.32x (0.74x / 1.91x).
  Through metal-linalg-torch, 1024 QRs of 128×128 take 19 ms, from 24
  (torch's MPS path: 1.21 s), one eigh of 2048×2048 90 ms, from 121, and one
  SVD of 4096×4096 1.56 s, from 1.77.
- The tuning harnesses fit the GPU-or-CPU product rule and the large-batch
  clause together (per cap: the rule, the clause over it, the rule again given
  the clause), rather than the clause over the rule fitted alone, which could
  miss a lower cap plus the clause beating a higher cap without one.

## 2.11.0 (2026-10-04)

- **A batch shared between the GPU and the CPU.** From a batch of
  `share_min_batch` (eigh and SVD policies, `EIGH_SHARE_MIN_BATCH`,
  `SVD_SHARE_MIN_BATCH`), a batch the batched GPU kernels (`ql`,
  `golub_kahan`) get is solved by the GPU and the CPU path at once: the GPU
  takes chunks from the front, cpu_threads() - 2 workers a few matrices at a
  time from the back, and they meet wherever their speeds put them, with no
  split to measure. On an M5 Pro, against the faster of the two alone: SVD
  1.4-1.7x for large batches of 16×16 to 64×64, eigh 1.4-1.5x. Two cores are
  left to the GPU's host work: with a worker on every core, the GPU's chunks
  took 5-7x their time alone. `svd_shares_batch()`, `svdvals_shares_batch()`,
  `eigh_shares_batch()` and `eigvalsh_shares_batch()` report it.
- **svdvals has its own GPU-or-CPU rule** (`values_gpu_max_k`,
  `values_gpu_min_batch_times_k`, `values_gpu_min_batch`, `values_gpu_max_l`;
  `SVD_VALUES_GPU_*`), as eigvalsh has. 2.10.0 applied the vectors' rule to
  singular values alone, and on an M5 Pro ran batches of 40×40 to 48×48 on the
  GPU at up to 1.5x the CPU's time. `svdvals_uses_gpu()` reports it.
- **QR of a wide matrix on the CPU is 10-40x faster**: it is factored by its
  leading square block and one matrix product (the same reflectors and R as
  `sgeqrf` on the whole matrix, which Accelerate ran slowly). One 64×2048 in
  0.09 ms instead of 1.47. This closes the gap 2.10.0 noted, where the GPU was
  2-3x faster for small batches of wide matrices but the routing sent them to
  the CPU: the CPU is now the faster one by far.
- **golub_kahan is faster on tall matrices and for singular values alone.**
  Columns are summed by groups of lanes instead of one thread each (256×16
  1.4x, 128×32 1.2x, squares about 1.1x), and from k = 40 (singular values
  alone) or 60 (with vectors) the QR iteration runs as a second dispatch with
  almost no threadgroup memory (singular values alone 1.3x at 48×48, 1.55x at
  64×64). Its workspace is kept and grown per shape, rather than reallocated
  for every batch size.
- **The large-matrix backends pipeline a batch.** `tridiag` (eigh) and
  `bidiag` (SVD) overlap one matrix's CPU solve with the next one's GPU
  reduction and back-transformation: per matrix of 2048×2048, eigh 1.48x and
  the SVD 1.66x faster in batches of 8, so the GPU stays ahead of the CPU for
  larger batches of large matrices than before.
- **Re-measured on the M5 Pro** (run `20261003-c0878c`, all three
  decompositions). eigh and the SVD share batches with the CPU from 1024
  matrices; eigvalsh now goes to the GPU for large batches (batch × N from
  16384), which it never did before; the SVD's GPU region grew to k <= 56 and
  a long side of 256 (from 48 and 64), its golub_kahan window to k = 8 .. 80,
  and svdvals got its own rule. Against the best backend at each point
  measured, the SVD row scores 1.027 geometric-mean regret (2.10.0's row on the
  same points: 1.039). Large batches against the CPU alone: the SVD of 4096
  matrices of 48×48 1.81x (2.10.0: 1.24x), eigh of 4096 of 32×32 2.07x (1.78x);
  through torch, the SVD of 4096 32×32 matrices in 7.7 ms (2.10.0: 9.2 ms;
  torch on MPS: 27 ms).
- `tuning/run.py` turns the calibration notice off for the tools it runs: a
  Mac whose QR measurements were stale could not be measured (the notice
  broke the JSON it reads).
- The C API's policy structs gain these fields at their ends:
  `metal_linalg_svd_policy` `values_gpu_max_k`, `values_gpu_min_batch_times_k`,
  `values_gpu_min_batch`, `values_gpu_max_l`, `share_min_batch`;
  `metal_linalg_eigh_policy` `share_min_batch`.
- Kernel versions: QR 3 (the wide CPU path), eigh 3 (tridiag pipelining),
  SVD 5 (golub_kahan and bidiag changes); the harnesses time `gk_vals`,
  `gk_share`, `gk_share_vals`, `ql_share` and `ql_share_vals`, and fit
  `share_min_batch` (stage 1c) and the SVD's values rule (stage 2b).

## 2.10.0 (2026-10-03)

- **A new batched SVD kernel, `golub_kahan`.** The SVD counterpart of 2.9.0's
  `ql`: LAPACK's method on the GPU, one threadgroup per matrix with the matrix
  in threadgroup memory (k <= 83; longer when tall): Householder
  bidiagonalization, then implicit bidiagonal QR (`sbdsqr`'s shifted sweep,
  with zero diagonal entries chased out), whose rotations one thread records
  a step at a time and every thread applies to its own row of U or V. V lives
  in a device-memory workspace, which doubled how many matrices share a core.
  On an M5 Pro it is 1.5-2.9x faster than the Jacobi kernels at every size it
  takes and 1.2-1.8x faster than the 18-core CPU for large batches up to
  48×48 (4096 of 32×32: 9.3 ms, Jacobi 22.6 ms, CPU 15.4 ms), the region
  where the M5 Pro now uses the GPU for the SVD; through torch, the SVD of
  4096 32×32 matrices went from 19 ms to 9.2 ms (torch on MPS: 27 ms).
  Backward stable like LAPACK's QR iteration (reconstruction and
  orthogonality about 1e-6); the Jacobi kernels remain where tiny singular
  values are wanted to high relative accuracy. A tall matrix that does not
  fit goes through this library's QR first (`qr_golub_kahan`). The policy
  has a window for it, `gk_min_k` .. `gk_max_k` (`SVD_GK_MIN_K`,
  `SVD_GK_MAX_K`), fitted per device and off on devices without
  measurements. `svd_backend()` reports `"golub_kahan"` or
  `"qr_golub_kahan"`; `detail::svd_golub_kahan` runs it directly, and
  `SvdOptions::Kernel::golub_kahan` on the factor of the QR.
- **Two more routing terms.** SVD: a cap on the long side, `gpu_max_l`
  (`SVD_GPU_MAX_L`): the CPU path reduces a tall matrix by a QR first and
  wins tall shapes at any batch (256×16, 4096 of them: 0.72x on the GPU),
  while large batches of 32×32 are the GPU's; with a cap on k alone the fit
  gave up everything above k = 24. QR: a lower bound on k, `gpu_min_k`
  (`QR_GPU_MIN_K`): the parallel CPU path wins the smallest matrices at every
  batch measured (16384 of 16×16: 3.3 ms, 8.1 on the GPU), which a product
  rule sent to the GPU. No cap and no bound on devices without measurements.
- Re-measured on the M5 Pro: QR (run `20261003-af087d`, now with batches up to
  16384) and the SVD (run `20261003-106b6c`, with `golub_kahan` and k = 24,
  40, 56, 80 added to its grid). Against the best backend at each point
  measured, the SVD row now scores 1.005 geometric-mean regret, worst 1.26x
  (2.9.0's row on the same points: 1.030, worst 2.09x), and QR's 1.034
  (2.9.0's: 1.060). The SVD's kernel version moved (QR's routing changes the timings
  of its QR-preconditioned backends), so earlier SVD runs are stale.
- The C API's policy structs gain these fields at their ends:
  `metal_linalg_svd_policy` `gk_min_k`, `gk_max_k`, `gpu_max_l`;
  `metal_linalg_qr_policy` `gpu_min_k`.
- `benchmark_svd` has a `gk` column; the QR harness measures batches of 4096
  and 16384, and the SVD harness fits the `golub_kahan` window as stage 1b,
  as the eigensolver's fits `ql`'s.

## 2.9.0 (2026-10-03)

- **The CPU path spreads a batch over every core.** QR, eigh and the SVD on
  the CPU solved a batch one matrix at a time, so a batch used one core: they
  now give each core whole matrices, with Accelerate's own threading off
  inside them. On an M5 Pro (18 cores) a batch of small or mid-size matrices
  is 7.5-15x faster than before (eigh, 2048 of 64×64: 284 ms to 19 ms), and
  even two 2048×2048 matrices are 1.55x faster than one after the other with
  Accelerate's threading. A lone matrix is unchanged. `set_cpu_threads(n)`
  (C: `metal_linalg_set_cpu_threads`, Python and torch: `set_cpu_threads`,
  Swift: `cpuThreads`) or `METAL_LINALG_CPU_THREADS=n` caps the cores used,
  for a program that runs several solves at once; `cpu_threads()` reports it.
  Every GPU-or-CPU boundary was measured against the one-core path, so the
  routing has been re-measured (run `20261003-2d2c19`): the kernel versions of
  all three decompositions moved (tuning/kernels.py), and earlier runs,
  including the M1's, are now stale. On the M5 Pro most batches now go to the
  CPU, and calls the old routing sent to the GPU are faster: against 2.8.x,
  eigh 1.5-8x (512×512, batch 16: 129 ms to 16 ms), the SVD 1.65-4.2x, QR up to
  2x, over the shapes in the study below; one QR shape measured, 1000 of
  256×128, is 7% slower, a near-tie the fitted rule gives to the CPU. The
  PyTorch table in the README was re-measured: metal-linalg-torch is now
  ahead of torch's MPS kernels on every row (eigh of 4096 16×16 matrices:
  2.0 ms, torch on MPS 5.3 ms). The QR sweep measures batches of 256 and 1024
  for 64×64 to 256×256, where the boundary now runs.
- **A new batched eigensolver kernel, `ql`.** LAPACK's method on the GPU,
  one threadgroup per matrix with the matrix in threadgroup memory
  (N <= 87): Householder tridiagonalization, then implicit QL, whose
  rotations one thread records a sweep at a time and every thread applies to
  its own row of the eigenvectors, so a sweep costs one barrier rather than
  one per rotation. It does several times fewer flops than the Jacobi
  kernels: on an M5 Pro 2-3.5x faster than the whole-matrix Jacobi kernel from
  N = 24 in batches, and up to 1.8x faster than the 18-core CPU for large
  batches up to N = 48 (4096 of 32×32: 5.4 ms, Jacobi 16.8 ms, CPU 9.6 ms), the
  region where the M5 Pro now uses the GPU. Same accuracy as the other backends
  (residual and orthogonality about 1e-6). The policy has
  a window for it, `ql_min_n` .. `ql_max_n` (`EIGH_QL_MIN_N`,
  `EIGH_QL_MAX_N`), fitted per device; it is off on devices without
  measurements. `eigh_backend()` reports `"ql"`, and `detail::eigh_ql`
  runs it directly.
- **Two routing terms the faster CPU path needs.** QR: a large-matrix clause,
  `gpu_large_min_k` and `gpu_large_max_batch` (`QR_GPU_LARGE_MIN_K`,
  `QR_GPU_LARGE_MAX_BATCH`): the CPU path now wins batches of small and
  mid-size matrices, while one large matrix is still about 2x faster on the
  GPU, and one product rule cannot say both. eigh and the SVD: batch caps for
  the `tridiag` and `bidiag` backends (`tridiag_max_batch`,
  `values_tridiag_max_batch`, `bidiag_max_batch`, `values_bidiag_max_batch`,
  and `EIGH_`/`SVD_`-prefixed variables), which solve a batch one matrix after
  another and lose to the CPU beyond a few matrices. All are fitted per device;
  0 means never (the clause) or any batch (the caps), which is what devices
  without measurements have.
- The C API's policy structs gain these fields at their ends:
  `metal_linalg_eigh_policy` `ql_min_n`, `ql_max_n`, `tridiag_max_batch`,
  `values_tridiag_max_batch`; `metal_linalg_svd_policy` `bidiag_max_batch`,
  `values_bidiag_max_batch`; `metal_linalg_qr_policy` `gpu_large_min_k`,
  `gpu_large_max_batch`.
- `benchmark_eigh` compares every GPU backend, `ql` included, with the
  library's CPU path rather than MLX's one-core `eigh`.
- [docs/studies/performance-headroom-apple-m5-pro.md](docs/studies/performance-headroom-apple-m5-pro.md):
  the measurements behind both changes, and what else was measured and
  dropped.

## 2.8.1 (2026-10-03)

- Release tarballs (what Homebrew downloads) leave out `docs/results/`, every
  submitted measurement run: 0.7 MB rather than 1.3 MB, and no longer growing
  with each run. The routing tables built from them are unchanged, and the
  runs stay in the repository.

## 2.8.0 (2026-10-03)

- **PyTorch support: `pip install metal-linalg-torch`.** A second Python
  package, `metal_linalg_torch`, for `torch` tensors on the CPU or MPS:
  `qr`, `eigh`, `eigvalsh`, `svd` and `svdvals` with the arguments and result
  types of their `torch.linalg` namesakes, results on the input's device. They
  are custom operators (`torch.ops.metal_linalg.*`) with fake
  implementations and torch.linalg's backward formulas, so they train and
  compile (`torch.compile(fullgraph=True)`). The package calls the library's C
  API through ctypes and is compiled against neither torch nor Python: one
  wheel serves every PyTorch from 2.4 and every Python from 3.10, and it
  neither installs nor loads MLX. On an M5 Pro, against `torch.linalg`: QR of
  1024 128×128 matrices 7x faster than torch on the CPU and 45x faster than
  on MPS; one 4096×4096 SVD 2x faster than either. See
  [python-torch/README.md](python-torch/README.md).
- **Fix: wrong eigenvalues from `eigvalsh` on macOS 14** (since 2.5.0, every
  binding). For N >= 128 on the CPU path, eigenvalues alone came from
  Accelerate's `ssyevd_2stage`, which on macOS 14 returns values off by 1.5-7%
  of the largest. The two-stage driver is now used only from macOS 15, and
  only once it has matched `ssyevd` on a fixed matrix (checked once per
  process); otherwise `ssyevd`, as for N < 128. `eigh`, macOS 15 and later,
  and the GPU paths were not affected. The Swift tests, which CI runs on
  macOS 14, now check it.
- C API: `metal_linalg_calibration_message(what)`, the calibration notice as
  a string, for bindings that report it their own way.
- The README's Python section covers both packages, and which to install.

## 2.7.0 (2026-10-03)

- **SVD of large matrices on the GPU: up to 1.95x faster (1.88x for `svdvals`) at 4096×4096 on an M5 Pro.** A new backend,
  `bidiag`, keeps LAPACK's method (Householder bidiagonalization, `sbdsdc`
  divide and conquer, back-transformation) and moves the reduction and the
  back-transformation to the GPU: the panels of `sgebrd` as Metal kernels
  (`Svd_Bidiag.metal`) queued without a round trip per column, the trailing
  updates and the blocked back-transforms as MPS GEMMs. Tall input is reduced
  by QR first, wide input goes through its transpose. Where the routing
  chooses the CPU it is used from `bidiag_min_k` on, and for `svdvals` from
  `values_bidiag_min_k` on, both measured per device (0 = never, the default
  for unmeasured Macs). See [docs/svd.md](docs/svd.md).
- New: `svdvals_backend(m, n, batch)` (C++, C `metal_linalg_svdvals_backend`,
  Python `svdvals_backend`, Swift `svdvalsBackend`); `SvdPolicy` gains
  `bidiag_min_k` and `values_bidiag_min_k`; environment variables
  `SVD_BIDIAG_MIN_K`, `SVD_VALUES_BIDIAG_MIN_K` and `SVD_DEVICE=bidiag`.
- Tuning: the SVD sweep times `bidiag`, and the CPU and `bidiag` again for
  singular values alone, up to 4096×4096 (`run.py` passes `--max-k 4096`);
  `tune_svd.py` fits the two thresholds in a third stage with a held-out
  check. Runs from before 2.7.0 are *incomplete* for the SVD: valid, with
  `bidiag` off until remeasured. The M5 Pro is remeasured.
- README: "Where the GPU wins" shows the large-matrix backends (eigh's
  `tridiag`, the SVD's `bidiag`) next to the batched kernels; a lone large
  matrix is no longer the CPU's at every size.
- A full measurement now takes about an hour and a half (the SVD sweep about
  50 minutes, up from 27).
- The measurements page lists every M5 variant (M5 8/10-core, M5 Pro 16/20,
  M5 Max 32/40, M5 Ultra 64/80).

## 2.6.0 (2026-10-03)

- **Which measurements are current is tracked, per decomposition.** A
  release does not by itself make measurements stale; a change to what a
  decomposition times does, and bumps its kernel version in
  `tuning/kernels.py` (with a history of why). Runs record the versions they
  measured; the tables use runs at the current version, or else the newest
  older ones -- marked stale, still used -- rather than falling back to the
  untuned default. A run from before a newer backend is *incomplete*: valid,
  with that backend off on its Mac until remeasured. The SVD's version is
  now 2: QR, which its QR-preconditioned backends call, routes small problems
  to the CPU since 2.2.3, so the M5 Pro's SVD measurements are stale.
- **[The measurements page](https://c0rmac.github.io/metal-linalg/docs/measurements)**
  (`docs/measurements.md`, generated with the tables): every Apple Silicon
  chip, colour-coded per decomposition (current, incomplete, stale, not
  measured), with run counts, dates and library versions, every submitted
  run, and the kernel-version history.
- **The library says when a Mac is not calibrated.** On the first use of a
  decomposition with no measurements for this Mac, one line on stderr says so
  and how to measure and submit; stale or incomplete calibrations get a
  softer note. Once per decomposition per process; Python raises
  `metal_linalg.CalibrationWarning` at import instead.
  `METAL_LINALG_NO_CALIBRATION_NOTICE=1`, `set_calibration_notices(false)`
  (C++) or `metal_linalg_set_calibration_notices(0)` (C) turns it off. Policy
  sources read `tuned-stale:` / `tuned-incomplete:` for such rows; new
  `calibration_message()` (C++) and `ml.calibration_status()` (Python).
- CI: the *Kernel versions* workflow warns on a pull request that changes a
  decomposition's timed files without bumping its version.
- **The measurement sweeps fit the Mac's memory.** Every shape is skipped if
  its estimated peak (six copies of its arrays) exceeds 35% of physical RAM,
  so an 8 GB Mac no longer attempts SVD shapes of 2 GB per array, which could
  swap or stop the run. On a 48 GB Mac the grids are unchanged.
  `METAL_LINALG_TUNING_MEMORY_GB` overrides the budget; `run.py` prints it.

## 2.5.0 (2026-10-03)

- **eigh with eigenvectors on the GPU for large matrices: up to 6x faster.**
  A new backend, `tridiag`, keeps LAPACK's method (Householder
  tridiagonalization, the tridiagonal eigenproblem, back-transformation) and
  runs its two expensive steps on the GPU: the reduction entirely on the GPU,
  per column as small kernels and per panel as one GEMM, queued so the host
  waits once per matrix; and the back-transformation as blocked GEMMs. The
  tridiagonal eigenproblem stays on LAPACK. On an M5 Pro, one N x N with
  eigenvectors: 1.1x the CPU at N = 1024, 2.0x at 2048, 2.9x at 3072, 4.8x at
  4096, 5.9x at 8192 (18.7 s to 3.2 s). Accuracy as LAPACK's (residual and
  orthogonality ~1e-6), any magnitude (power-of-two scaling), one triangle
  read. See docs/eigh.md, "Backend 3".
- **Routing**: `EighPolicy` gains `tridiag_min_n` and `values_tridiag_min_n`:
  where the policy sends a call to the CPU, the tridiag backend takes it from
  that N (0 = never, which devices without measurements keep).
  `EighBackend::tridiag`, the name `"tridiag"` in the C, Python and Swift
  routing functions, `EIGH_TRIDIAG_MIN_N` / `EIGH_VALUES_TRIDIAG_MIN_N`, and
  `EIGH_DEVICE=tridiag` to force it. The C policy struct gains the two fields
  at its end.
- **Measured on the M5 Pro**: tridiag from N = 1024 with eigenvectors (on the
  shapes where it was timed, 1.039x geomean regret against 1.175x without it;
  held out 1.069x against 1.114x). For eigenvalues alone the CPU's two-stage
  reduction stays ahead up to 2048 and tridiag is off.
- The eigh sweep times the new backend (`tridiag`, `tridiag_vals`) and reaches
  N = 4096 for lone matrices; `tune_eigh.py` fits the thresholds as stage 4,
  on the points where the backend was timed, with a held-out check.

## 2.4.1 (2026-10-03)

- **The committed shaders load on macOS 14 and 15 again.** The metallibs in
  `shaders/prebuilt/`, which Homebrew builds and the Swift package use, had
  been compiled for macOS 26 (AIR v28), which older systems cannot read, so
  the GPU paths of those installs could fail on macOS 14 and 15. Every shader
  is now compiled for macOS 14 (AIR v26) whatever Mac builds it
  (`METAL_LINALG_SHADER_MIN_MACOS`), and the committed copies are rebuilt.
  The PyPI wheels were not affected: their shaders were compiled for
  macOS 14 already.
- **The shaders compile with Xcode 27's Metal compiler**, which rejects an
  address space on a by-value parameter; three QR helpers take their tile by
  `const` reference instead. Building from source with the Metal toolchain
  installed failed before this.
- **The Swift package builds on macOS 14.** It embedded the shaders with
  C23's `#embed`, which the Xcode on macOS 14 does not support, so the
  package did not compile there. `swift/CMetalLinalg/embedded_shaders.c` now
  holds them as plain byte arrays, generated from `shaders/prebuilt/` by
  `cmake/EmbedSwiftShaders.cmake` whenever those are refreshed.
- CI: the wheels are installed and tested on macOS 14 before every PyPI
  upload, and the Swift package's tests (prebuilt shaders, every kernel) run
  on macOS 14 and 15 for every change to the shaders or the library, with a
  check that its copy of the shaders matches `shaders/prebuilt/`.

## 2.4.0 (2026-10-02)

- **`eigvalsh` has its own GPU-or-CPU boundary.** Since 2.3.0 the CPU
  computes eigenvalues alone by the two-stage reduction, up to 3x faster than
  the path eigh's boundary was measured against, so routing `eigvalsh` by
  that boundary sent work to the GPU where the CPU was faster: on an M5 Pro
  up to 3.9x slower (1.061x geomean regret over 194 shapes). `EighPolicy`
  gains `values_gpu_max_n`, `values_gpu_min_batch_times_n` and
  `values_gpu_min_batch` (with `EIGH_VALUES_GPU_*` overrides);
  `values_gpu_min_batch = 0` means "as for eigenvectors", which is what a
  device measured before keeps. New: `eigvalsh_backend()` and
  `eigvalsh_uses_gpu()` (C++), `metal_linalg_eigvalsh_backend()` (C),
  `ml.eigvalsh_backend()` (Python), `eigvalshBackend()` (Swift). The C policy
  struct gains the three fields at its end.
- **The M5 Pro's eigenvalues-alone boundary is measured**: GPU iff
  N <= 256, batch * N >= 2048 and batch >= 32 (1.0075x geomean regret,
  worst 1.28x; held out 1.0068x).
- **The eigh sweep times every backend for eigenvalues alone too**
  (`<backend>_vals`) and reaches N = 2048 for lone matrices and small
  batches, so `gpu_max_n` is a measured cap rather than the edge of the
  grid (on the M5 Pro the CPU is 2.4-2.6x faster at 1536 and 2048). A full
  `tuning/run.py` now takes about an hour; `--only eigh` about 27 minutes.

## 2.3.0 (2026-10-02)

- **Large eigenvalue-only problems are much faster on the CPU.**
  `eigvalsh` from N = 128 uses LAPACK's two-stage reduction
  (`ssyevd_2stage`: dense to band in matrix-matrix products, then band to
  tridiagonal) instead of `ssyevd`, whose reduction is bound by memory
  bandwidth. On an M5 Pro: 1.2x at N = 1024, 1.6x at 2048, 3.8x at 4096,
  5.7x at 8192 (14.8 s to 2.6 s). The matrix is always handed over as its
  lower triangle, on which the two-stage reduction is about 1.5x faster.
  `eigh` with eigenvectors is unchanged: LAPACK's two-stage driver does not
  return them.

## 2.2.3 (2026-10-02)

- **QR uses its CPU path on the M5 Pro.** The M5 Pro's QR row predated QR's
  CPU path, so every QR call went to the GPU: a lone 64×64 took 0.42 ms
  rather than 0.03 ms in LAPACK. A new QR measurement sets the boundary to
  "GPU iff `batch * k >= 512`" (1.08x geomean regret over 173 shapes, 1.67x
  for always the GPU). The M1 row still sends every QR call to the GPU until
  an M1 is remeasured (`python3 tuning/run.py --only qr`).
- `tuning/run.py --only qr|eigh|svd` measures one decomposition.
- QR tuning: submissions with CPU timings pass validation (they were
  rejected); the GPU size limit is never put at the largest size measured;
  lone matrices up to 3072×3072 are measured.
- The Python `qr` docstring no longer says QR always runs on the GPU.

## 2.2.2 (2026-10-02)

- **Complex input raises an error** (`std::invalid_argument` in C++,
  `ValueError` in Python, `MetalLinalgError.invalidArgument` in Swift)
  instead of being cast to float32, which kept only the real parts and
  returned the decomposition of a different matrix: for the Hermitian
  `[[2, i], [-i, 2]]`, `eigvalsh` gave `[2, 2]` instead of `[1, 3]`.

## 2.2.1 (2026-10-02)

- Importing the Python package with a different MLX than it was built
  against raises the `ImportError` that names both versions and the fix,
  instead of the loader's missing-symbol error: the version is now checked
  before the extension is loaded.

## 2.2.0 (2026-10-02)

- **The Python package is on PyPI**: `pip install metal-linalg` installs a
  prebuilt wheel (Apple Silicon, macOS 14+, Python 3.10 to 3.14) and the MLX
  it was built against, which it pins exactly (`mlx==0.32.3`). Every release
  builds, checks and tests the wheels and publishes them
  ([wheels.yml](.github/workflows/wheels.yml)). Building from source no
  longer needs `--no-build-isolation`: the build fetches the pinned MLX and
  nanobind itself.
- The Python build accepts nanobind 3's ABI, which the pip MLX uses from
  0.32.3 (nanobind 3.0.1); it still builds against MLX built with nanobind
  2.x, such as Homebrew's.
- The Python tests check residuals with CPU matmuls: MLX 0.32.3 multiplies
  float32 on the M5's GPU to only about 1e-2, which failed checks of results
  that are accurate to 1e-6.

## 2.1.0 (2026-10-02)

- **QR has a CPU path**, like the eigensolver and the SVD: LAPACK's `sgeqrf`
  and `sorgqr`, for lone and small-batch calls that do not pay for a GPU
  launch (on an M5 Pro, 4 matrices of 64×64 take 0.13 ms on the CPU against
  1.04 ms on the GPU). `QrPolicy` gains the GPU-or-CPU boundary
  (`gpu_max_k`, `gpu_min_batch_times_k`, `gpu_min_batch`), with the
  `QR_GPU_*` and `QR_DEVICE` environment overrides; `QrBackend::cpu`,
  `qr_gpu_backend()`, `qr_uses_gpu()` and `detail::qr_cpu` are new, and the C,
  Python and Swift APIs follow. The QR sweep times the CPU and the tuner fits
  the boundary; a device measured before this keeps sending every QR call to
  the GPU until it is measured again. See [docs/qr.md](docs/qr.md).
- The QR guide documents the backends' sign conventions for R's diagonal,
  which differ, and how to normalise them.

## 2.0.1 (2026-10-01)

- metal-linalg is licensed under the MIT licence ([LICENSE](LICENSE)).

## 2.0.0 (2026-10-01)

The project is renamed from `qr-apple-silicon` to **metal-linalg**, since it
now covers three decompositions, and is packaged as a library.

### New

- **Symmetric eigensolver**: `eigh_accelerated`, `eigvalsh_accelerated`. Cyclic
  Jacobi in two kernels, a whole-matrix kernel (simd and threadgroup modes)
  and a block kernel that spreads one matrix over the GPU, with a per-device
  route to MLX's CPU `eigh` where that is faster. See [docs/eigh.md](docs/eigh.md).
- **Thin SVD**: `svd_accelerated`, `svdvals_accelerated`. One-sided Jacobi in a
  whole-matrix and a block kernel, each optionally after this library's QR for
  tall input, with a per-device route to the CPU. See [docs/svd.md](docs/svd.md).
- **Per-device routing policies** for all three solvers, measured on an Apple
  M1 (QR, eigh) and an Apple M5 Pro (all three), with environment and
  programmatic overrides.
- **Measuring a Mac is one command**, `python3 tuning/run.py`: it checks the
  machine, builds, tests, measures all three decompositions and writes a
  uniquely named submission (`docs/results/<device>/<date>-<random>/`), so any
  number of people with the same Mac can contribute.
  The library's per-device tables (`src/tuned/`) are generated from every
  run submitted for each device, by `tuning/generate_tables.py`, which a
  GitHub Action runs on each results pull request (to validate it and show the
  effect) and after each merge (to apply it). See [docs/tuning.md](docs/tuning.md). For QR
  this replaces the fixed rule of 1.0 with a crossover on the row count alone,
  measured over square, tall, wide and near-square shapes
  ([study](docs/studies/qr-routing-apple-m1.md)).
- **Python package** `metal_linalg` (`python/`, `pyproject.toml`): `qr`, `eigh`,
  `eigvalsh`, `svd`, `svdvals` on `mlx.core` arrays, the routing queries and
  policies. Compiled against the installed MLX and sharing its arrays without
  copying; see [python/README.md](python/README.md).
- **A core without MLX**, on plain float buffers: `<metal_linalg/core.h>`
  (`core::qr`, `core::eigh`, `core::svd` and every backend). The MLX API is
  now a thin layer over it, with the same names and signatures as before.
  `-DMETAL_LINALG_WITH_MLX=OFF` builds the core alone.
- **A C API**, `<metal_linalg/c_api.h>`: the decompositions, routing queries
  and policies behind C types, with status codes and per-thread error
  messages. See [docs/c-api.md](docs/c-api.md).
- **A Swift package**: `MetalLinalg` on `[Float]`, and `MetalLinalgMLX` on
  mlx-swift's `MLXArray`, built on the C API; the shaders reach it through
  C23 `#embed` of `shaders/prebuilt/`. See [docs/swift.md](docs/swift.md).
- **Objective-C**: a guide, [docs/objective-c.md](docs/objective-c.md), and
  `examples/objc_quickstart.mm`; Objective-C can use the MLX API from `.mm`
  files or the C API from anywhere.
- Every measurement run records the exact Mac (e.g. "MacBook Pro (16-inch, M5
  Pro)"), and its power and thermal state after each part of the run; a
  device's combined summary compares the runs by machine, to find outliers.
- [CONTRIBUTING.md](CONTRIBUTING.md): measuring a Mac and sending the results as a
  pull request (with the GitHub CLI, with git alone, or as a zip on an issue).
- [docs/reading-reports.md](docs/reading-reports.md) explains every number in
  a measurement report, with worked examples from the M5 Pro run.
- `<metal_linalg/device.h>`: `device_name()`, `gpu_core_count()`.
- `<metal_linalg/metal_linalg.h>`, which includes everything.
- `qr_backend(m, n, batch)`, like `eigh_backend` and `svd_backend`.
- `sweep_qr --policy`, like the other two sweeps.
- `examples/`: five self-checking programs (a quick start, orthonormal bases
  with QR, PCA with eigh, the nearest orthogonal matrix with the SVD, and the
  routing queries), built with the tests and run by `ctest`.

### Packaging

- The compiled shaders are embedded in the library, so an installed
  `libmetal_linalg` is self-contained: nothing is looked up on disk at run
  time and consumers need no Metal compiler. Without the compiler the build
  uses the metallibs committed under `shaders/prebuilt/`.
- A CMake package: `find_package(MetalLinalg)` and
  `metal_linalg::metal_linalg`. As a subproject (`add_subdirectory`,
  `FetchContent`) it builds static and installs nothing.
- A Homebrew formula, in its own tap, [c0rmac/homebrew-metal-linalg](https://github.com/c0rmac/homebrew-metal-linalg):
  `brew tap c0rmac/metal-linalg`, `brew trust c0rmac/metal-linalg`, then `brew install metal-linalg`.
- **Releases are automatic**: every update to `main` that changes the library
  publishes the next version (a tag, a GitHub release with its source tarball)
  and points the Homebrew formula at it. See CONTRIBUTING.md, "Releases".

### Breaking changes

| 1.x | 2.0 |
|---|---|
| namespace `custom_math` | `metal_linalg` |
| `#include "qr.h"`, `"qr_detail.h"` | `#include <metal_linalg/qr.h>` (the `detail` backends are in it) |
| `DispatchPolicy` | `QrPolicy` |
| `dispatch_policy()`, `dispatch_policy_source()`, `set_dispatch_policy()` | `qr_policy()`, `qr_policy_source()`, `set_qr_policy()` |
| `m_crossover_for_batch(p, batch)` | `qr_backend(m, n, batch)` |
| CMake target `qr_metal` | `metal_linalg::metal_linalg` |

### Changed

- **The CPU paths call LAPACK directly**: `ssyevd` for eigh and `sgesdd` for
  the SVD (Accelerate, after a thin QR for tall input), instead of MLX's CPU
  `eigh` and `svd`. On an M5 Pro the eigensolver's CPU path is as fast as
  before and the SVD's is as fast on square and wide shapes and 5-25% faster
  on tall ones. Non-finite input gives NaN on the CPU too, for that matrix
  alone, as on the GPU. The routing tables were measured against the old
  paths and are due to be remeasured.
- The eigensolver's tuning sweep times the library's own CPU path, as the
  SVD's already did, rather than MLX's.

### Fixes

- **QR was not scale-invariant.** The kernels compare squared column norms
  with an absolute threshold, so entries around 1e-3 lost accuracy, below 1e-5
  the factorisation failed, and above 1e+18 it returned NaN. Every matrix is
  now scaled by an exact power of two on the way in.
- **QR discarded real data for nearly dependent columns**: the reflection
  threshold (1e-7 on a squared norm) treated column tails shorter than 3e-4 of
  the matrix's scale as zero. It is now 1e-30, which only guards the division.
- **The streaming QR backends could write past their buffers.** Their cached
  workspaces were keyed by the padded shape but sized by the exact one, so a
  call whose shape padded like an earlier, smaller one (1000×1000, then
  1024×1024) reused buffers too small for it. They are now keyed by the exact
  shape.
- **Every call leaked Metal objects**: the library was built without ARC and
  without an autorelease pool, so each call kept its command buffers and a
  buffer wrapper alive for the life of the thread, which mattered in long
  loops. It is now built with ARC, and every GPU call drains its own pool.
- **Transposed and other strided views were read as their untransposed
  buffer**: an unevaluated MLX array reports itself contiguous, so contiguity
  is now checked after evaluation.

## 1.0.0

`qr-apple-silicon`: batched QR on Apple GPUs (`custom_math::qr_accelerated`),
with single-threadgroup and grid-parallel Householder backends and a fixed
rule choosing between them.
