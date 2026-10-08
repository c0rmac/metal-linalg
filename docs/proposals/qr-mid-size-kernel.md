# A QR kernel for batches of mid-size matrices

Status: **done** in 2.16.0 (2026-10-08); see [Done](#done-2026-10-08).

## What

Batches of mid-size matrices (about 64 to 512) go to the blocked QR
([qr-blocked.md](qr-blocked.md)), whose work for a batch is the band
reduction's panel kernels and batched MPS products: some forty dependent
dispatches a call at 128 x 128, each small. The SVD and the eigensolver have
a kernel of their own for this range (block Jacobi); QR has none. Build one
as LAPACK's blocked QR runs on one core, in one threadgroup a matrix:

1. Panels of 16 columns factored as the register kernel factors a matrix
   ([qr-small-kernel.md](qr-small-kernel.md)), a thread a row, R rows a
   thread, the dot products of a step `simd_sum`s and one slot a simdgroup:
   two barriers a column. T built on the way (slarft's recurrence, the
   reflector's dot products with the columns left of it in the same sums).
2. Each panel's H = I - V T V^T applied to the columns right of it as 8 x 8
   simdgroup matrix products: W = V^T C, then C -= V (T^T W).
3. Q from [I; 0] by the panels backward, C -= V (T (V^T C)).
4. The matrix and Q in a device workspace; the input read row-major as it
   is, scanned and scaled on the GPU, Q and R written straight out.

The unblocked backend hands it every matrix the register kernel does not
take, and the routing's kernel crossover is re-measured.

## Why: the measurements

M5 Pro, 2.15.0's routing and kernels, through MLX (ms):

| batch x shape | CPU | blocked QR | GPU flops rate |
|---|---|---|---|
| 1024 x 128 x 128 | 25.4 | 17.2 | 0.3 TFLOP/s |
| 256 x 256 x 256 | 18.6 | 17.3 | 0.7 TFLOP/s |
| 64 x 512 x 512 | 23.6 | 18.7 | 1.2 TFLOP/s |
| 16 x 2048 x 64 | 2.9 | 2.1 | |
| 4096 x 80 x 80 | 24.3 | 26.7 | |

The GPU is level with the CPU here, at a tenth of the rate the same
products reach for one large matrix.

## Effort

2-3 days.

## Expected gain

2-4x the CPU on large batches of 96 x 96 to 256 x 256.

## Done (2026-10-08)

**Built as planned** (`qr_householder_wy` in `shaders/QR_Householder.metal`,
`src/qr_householder.mm`), then changed by what the measurements showed. GPU
time (command buffer timestamps) unless a call is given:

| step | 1024 x 128^2 | 256 x 256^2 | 64 x 512^2 |
|---|---|---|---|
| first cut: panels of 16, each applied at once, two accumulators a simdgroup | 4.8 ms | 7.1 | 12.3 |
| two rows a thread (four for n <= 32), not one | 4.4 | 6.4 | 12.2 |
| blocks of 32 columns (T merged from the panels'), two column tiles a simdgroup; threadgroup memory sized to the block | 4.4 | 6.4 | 11.9 |
| Q's [I; 0] and its unwritten zeros made in registers, not read; R written block by block; the input copied while scanned | 4.2 | 6.1 | 11.7 |

- **The bound is memory, not the products.** At 1024 x 128^2, without the
  updates and the panels the kernel still took 1.5 ms: the input copied in,
  R out, the panels' loads and stores, about 384 MB at the GPU's 300 GB/s.
  The updates are 2.1 ms (2.7 TFLOP/s) and the panels 0.45.
- **Blocks wider than 16 hardly helped**: rank-64 updates were no faster
  than rank-32 (64 x 512^2: 20.9 ms a call against 20.4), and a generic
  routine for them first ran 1.4x slower than the hand-written rank-16 one
  until its loops were specialised by width and its diagonal tiles handled
  outside the main loop. Two column tiles a simdgroup (each V tile serving
  two products) gained 10-20% at 512; four, no more.
- **More simdgroups a matrix did not help**: from 2 to 32 (with rows a thread
  to match), within 5% except where they cost rows a thread. A wide matrix
  is the exception: its work is the updates of its many columns, so it gets a
  simdgroup for every 128 columns too, up to 8 (64 of 64 x 1024: 1.3 ms a
  call against 2.7).
- **Threadgroup memory sets residency**: 10.5 KB a threadgroup against 6.5
  cost 5-10% at 128^2; the scratch is sized to the block, larger only for
  tall narrow matrices, whose few column tiles are split by rows.

**It replaced two kernels.** Against the register kernel it wins from 48
columns (4096 of 48 x 48: 2.3 ms against 3.3; 64 x 64: 3.6 against 4.4), so
the register kernel keeps n <= 32 and loses its 64-column instances. The
threadgroup-memory kernel of [qr-small-kernel.md](qr-small-kernel.md) lost
everywhere (1024 of 200 x 30: 1.3 against 2.2; 4096 of 80 x 80: 7.4 against
34) and is gone; so is the unblocked backend's own kernel,
`QR_Unblocked.metal` (2.5-6x slower wherever it ran). The unblocked backend
is now the two Householder kernels, up to 4096 rows, and the blocked QR
beyond.

**Measured** (M5 Pro, through MLX, ms a call, with MLX's buffers reused, see
[known-buffers.md](known-buffers.md), and MLX's buffer cache off, as the sweeps
had it then):

| batch x shape | CPU | blocked QR | this kernel |
|---|---|---|---|
| 1024 x 128 x 128 | 25.1 | 17.8 | 6.4 |
| 256 x 256 x 256 | 18.7 | 17.8 | 8.3 |
| 64 x 512 x 512 | 24.0 | 19.2 | 14.6 |
| 64 x 384 x 384 | 10.6 | 10.8 | 7.6 |
| 4096 x 64 x 64 | 10.8 | 16.1 | 5.9 |
| 16 x 1024 x 64 | 1.45 | 1.34 | 0.67 |
| 64 x 4096 x 64 | 17.3 | 13.5 | 8.8 |
| 256 x 2048 x 32 | 10.0 | 11.7 | 5.2 |
| 256 x 32 x 512 | 1.65 | 1.63 | 1.16 |

Lone matrices and small batches stay with the CPU or the blocked QR (one
threadgroup a matrix is latency-bound: one 512 x 512 takes 4.9 ms, the
blocked QR 2.9, the CPU 3.7); the re-measured routing (kernel epoch qr 6)
finds the crossovers.

**Then, the same night:**

- **Small batches get more simdgroups a matrix.** A batch that would leave
  the GPU with fewer than about 12 simdgroups a core gives each matrix up to
  16, one row a thread: its panels are a latency-bound chain, which more
  threads shorten. One 256 x 256 0.98 ms (1.52 before), one 384 x 384 2.0
  (3.0), 16 of 256 x 256 1.19 (1.74), 32 of them 1.66 (2.09); large batches
  unchanged (from 2 to 32 simdgroups a matrix made no difference there).
- **An aligned input read directly.** A matrix that needs neither padding
  nor scaling is read by block 0 straight from the input, which writes the
  workspace, so the copy pass is gone: 2-4% (1024 of 128^2 4.65 to 4.51 ms).
- With MLX's buffer cache on in the sweeps ([known-buffers.md](known-buffers.md#then-2026-10-08)),
  1024 of 128 x 128 takes 4.5 ms a call, 4096 of 64 x 64 3.9.

**Left:** the fixed passes are most of what remains at 128^2. Keeping a
128 x 128 matrix on chip would save most of them, but at 64 KB it is twice a
threadgroup's memory. A lone matrix of 256-384 is still a little slower
than the CPU (0.98 ms against 0.84 at 256): its panels' two barriers a
column are the chain.
