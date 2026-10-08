# MLX's buffers, not wrapped again

Status: **done** in 2.16.0 (2026-10-08): the MLX API, then the C API and
the PyTorch package; and the remaining per-call cost found to be the sweeps'
own ([Then](#then-2026-10-08)).

## What

The core takes plain pointers (`core.h`), and each GPU backend makes Metal
buffers of the caller's memory with `newBufferWithBytesNoCopy`
(`wrap_host`, `input_buffer` in `src/metal_runtime.mm`). Making the buffer
is quick, but the first command buffer that uses it pays to map its pages
for the GPU: on an M5 Pro some 15 us a MB of `malloc`ed memory, and about
0.7-1.4 ms for 64 MB of memory that is already a Metal buffer, against
0.2-0.6 ms when the buffer object itself is reused. Through the MLX API the
memory always is a Metal buffer, MLX's: its arrays live in them.

So the MLX layer (`src/mlx_api.cpp`) registers its input's and outputs'
buffers for the length of a call (`detail::KnownBuffer`,
`src/known_buffers.h`), and `wrap_host` and `input_buffer` hand out a
registered buffer for memory that starts where it does. A pointer into the
middle of a buffer (an array that is a slice) is wrapped as before.

## Why: the measurements

At 4096 x 64 x 64 the QR's three buffers are 64 MB each; the call took 7.1
ms for 3.7 of GPU time.

## Done (2026-10-08)

M5 Pro, `sweep_qr` through MLX (with MLX's buffer cache off, as the sweeps
had it then), the Householder kernels, ms a call:

| shape | before | after |
|---|---|---|
| 4096 x 32 x 32 | 1.69 | 1.41 |
| 4096 x 64 x 64 | 7.12 | 5.89 |
| 1024 x 128 x 128 | 7.54 | 6.37 |
| 256 x 256 x 256 | 9.50 | 8.33 |
| 16 x 1024 x 64 | 0.88 | 0.67 |

Every backend that wraps its caller's memory gains as much for the same
bytes, the eigensolver's and the SVD's too; their routing, measured before
it, is a little conservative about the GPU until they are re-measured.

## Then (2026-10-08)

**The rest of the gap was the sweeps' own.** At 4096 x 64 x 64 the call was
still 5.9 ms for 3.65 of GPU time, and command-buffer timestamps put 2.4 ms
of it between the commit and the GPU starting, about 12 us for every MB the
call touched. The sweep tools (and the benchmarks) had turned MLX's buffer
cache off since 2.0, so every call's outputs were fresh pages, which the GPU
maps on first use. Through MLX as a program has it, cache on, the same call
takes 3.9 ms, the commit-to-start 0.1. The sweeps and the benchmarks keep
the cache on since; it moved every GPU backend more than the CPU path:

| shape (QR) | CPU, cache off -> on | GPU, cache off -> on |
|---|---|---|
| 4096 x 64 x 64 | 10.5 -> 9.0 ms | 6.4 -> 4.0 |
| 1024 x 128 x 128 | 25.1 -> 24.4 | 7.2 -> 4.6 |
| 4096 x 16 x 16 | 1.32 -> 0.89 | 0.52 -> 0.39 |

so all three decompositions were re-measured (kernel epochs qr 7, eigh 7,
svd 9).

**Residency sets, tried.** Keeping the call's buffers in a residency set of
the library's own (`MTLResidencySet`, macOS 15) cut the commit-to-start to
0.1-0.5 ms but cost 1.7 ms a call to make it resident again, and holding
MLX's buffers in it kept MLX's allocator from recycling them: each call got
fresh memory. With the cache on there is nothing left for it to save.

**The C API and PyTorch**: `metal_linalg_know_buffer` and
`metal_linalg_forget_buffer` register a caller's `MTLBuffer` for memory that
starts where it does, as the MLX layer does for its arrays; the PyTorch
package registers its MPS tensors' buffers for each call.
