# MLX's buffers, not wrapped again

Status: **done** in 2.16.0 (2026-10-08) for the MLX API; what is left is
[below](#left).

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

M5 Pro, `sweep_qr` through MLX, the Householder kernels, ms a call:

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

## Left

- At 4096 x 64 x 64 the call is still 5.9 ms for 3.65 of GPU time: about 2
  ms between, which the GPU timestamps do not see: the command buffer's
  submission, making 192 MB of buffers resident for a queue that has not
  used them, MLX's own allocation of the outputs. A residency set
  (`MTLResidencySet`, macOS 15) kept by the library might take part of it;
  not tried.
- The C API and the PyTorch package pass plain pointers and gain nothing:
  a C entry point taking `MTLBuffer`s (and PyTorch's MPS tensors through it,
  as the `torch-mps-zero-copy` branch does for its own path) would.
