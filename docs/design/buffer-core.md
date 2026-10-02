# Buffer-level core: design and status

**Branch `buffer-core`, worktree `../metal-linalg-buffer-core`. Not merged.
Everything below has been built and tested on an M5 Pro.**

## Why

mlx-swift exposes only MLX's C API (`mlx-c`) to other Swift packages, so a
Swift API cannot be built against MLX's C++ the way the C++ and Python APIs
are. Splitting the library into a core that works on plain float buffers,
with no MLX inside, lets every language share one implementation of the
kernels and the routing:

```
                     core (raw Metal + Accelerate; no MLX)
                     include/metal_linalg/core.h
        ┌──────────────────┬──────────────────────┬────────────────────┐
   MLX C++ API         C API                 Swift package        (later) PyTorch
   qr.h eigh.h svd.h   c_api.h               MLXArray via mlx-c
   (Python uses this)  (any language)
```

## What changes inside

| today (MLX) | in the core |
|---|---|
| `prepare_input`, `prepare_input_scaled` | `core::prepare`: scan, scale and wrap or copy into a Metal buffer, in plain C++ |
| outputs built as `mx::array` copies | written into caller-provided buffers; the MLX layer allocates each `mx::array` first and passes its memory |
| transposes, rescaling, NaN fills | plain loops |
| CPU fallbacks (`mx::linalg::eigh`, `svd`) | Accelerate LAPACK directly (`ssyevd`, or `ssyevd_2stage` for eigenvalues alone from N = 128 since 2.3.0; `sgesdd`, after `sgeqrf`/`sorgqr` for tall input) |
| `matmul(Q, U_R)` in the QR-preconditioned SVD | Metal Performance Shaders, on the GPU |

The public C++ API keeps its names and signatures. Policies, options,
backends and the `info` decoders move to `core.h`, which the MLX headers
include.

## Consequences, measured (M5 Pro)

- **CPU paths.** eigh's `ssyevd` matches MLX's CPU `eigh` within noise at
  every shape tried (1x512 to 4096x8). The SVD's matches MLX's on square and
  wide shapes and is 5-25% faster on tall ones (4096x256: 18.0 ms against
  23.9). A plain `sgesdd` on tall input was 30% *slower*: LAPACK sees the
  row-major matrix transposed, i.e. wide, and Accelerate's LQ routines run at
  about half the speed of its QR ones, so tall input is transposed and
  reduced by QR explicitly. The routing tables are therefore still safe (the
  CPU got no slower anywhere) but slightly conservative for tall SVDs.
- **GPU paths.** Equal or faster at every point tried, for every backend
  (e.g. eigh simd 4096x8: 0.67 ms against 0.88). Large inputs are now
  scanned and rescaled on the CPU instead of by MLX on the GPU; split over
  row blocks, that costs nothing measurable (QR 1x2048x2048: 50.0 ms against
  50.5).
- **Memory.** No more per-call leaks of command buffers and buffer wrappers.

## Status

- [x] `core.h`; MLX headers include it (public C++ API source-compatible: every test, benchmark, sweep and example compiles unchanged)
- [x] buffer utilities in `src/metal_runtime.{h,mm}`: `input_buffer`, `scaled_input`, `scan`, `copy_out`, `transpose_out`, `HostBuffer`, row-block parallelism
- [x] QR, eigh and SVD backends on buffers; CPU paths via LAPACK; MPS for `Q * U_R`
- [x] MLX adapter (`src/mlx_api.cpp`)
- [x] library built with ARC, with an autorelease pool per GPU call
- [x] `METAL_LINALG_WITH_MLX=OFF` builds the core alone (links only system frameworks)
- [x] C API (`c_api.h`, `src/c_api.cpp`), tested from C99 (`tests/test_c_api.c`)
- [x] shaders for SwiftPM: C23 `#embed` of `shaders/prebuilt/*.metallib`, by paths relative to the source (no flags, so the package stays usable as a dependency)
- [x] Swift package: `MetalLinalg` on `[Float]` (6 tests) and `MetalLinalgMLX` on `MLXArray` (4 tests, against mlx-swift 0.32.3), all passing
- [x] all CMake suites pass (`test_qr`, `test_eigh`, `test_svd`, `test_core`, `test_c_api`, six examples; the three MLX suites ten times over)
- [x] Python package built against this branch (MLX 0.32.1, nanobind 2.15.0): 12 tests pass
- [ ] remeasure on the new CPU paths; bump `EPOCH` then (not before: it would discard the existing runs and leave every device untuned)

Also fixed here and on main: the streaming QR workspaces were keyed by padded
shape but sized by exact shape, so a larger shape padding alike (1000x1000
then 1024x1024) wrote past the buffers.
