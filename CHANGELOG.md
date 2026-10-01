# Changes

## 2.0.0 (unreleased)

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
  programmatic overrides, and harnesses to measure a new device
  (`tuning/tune_{qr,eigh,svd}.py`, [docs/tuning.md](docs/tuning.md)). For QR
  this replaces the fixed rule of 1.0 with a crossover on the row count alone,
  measured over square, tall, wide and near-square shapes
  ([study](docs/studies/qr-routing-apple-m1.md)).
- `<metal_linalg/device.h>`: `device_name()`, `gpu_core_count()`.
- `<metal_linalg/metal_linalg.h>`, which includes everything.
- `qr_backend(m, n, batch)`, like `eigh_backend` and `svd_backend`.
- `sweep_qr --policy`, like the other two sweeps.

### Packaging

- The compiled shaders are embedded in the library, so an installed
  `libmetal_linalg` is self-contained: nothing is looked up on disk at run
  time and consumers need no Metal compiler. Without the compiler the build
  uses the metallibs committed under `shaders/prebuilt/`.
- A CMake package: `find_package(MetalLinalg)` and
  `metal_linalg::metal_linalg`. As a subproject (`add_subdirectory`,
  `FetchContent`) it builds static and installs nothing.
- A Homebrew formula, in its own tap: `brew install c0rmac/metal-linalg/metal-linalg`.

### Breaking changes

| 1.x | 2.0 |
|---|---|
| namespace `custom_math` | `metal_linalg` |
| `#include "qr.h"`, `"qr_detail.h"` | `#include <metal_linalg/qr.h>` (the `detail` backends are in it) |
| `DispatchPolicy` | `QrPolicy` |
| `dispatch_policy()`, `dispatch_policy_source()`, `set_dispatch_policy()` | `qr_policy()`, `qr_policy_source()`, `set_qr_policy()` |
| `m_crossover_for_batch(p, batch)` | `qr_backend(m, n, batch)` |
| CMake target `qr_metal` | `metal_linalg::metal_linalg` |

### Fixes

- **QR was not scale-invariant.** The kernels compare squared column norms
  with an absolute threshold, so entries around 1e-3 lost accuracy, below 1e-5
  the factorisation failed, and above 1e+18 it returned NaN. Every matrix is
  now scaled by an exact power of two on the way in.
- **QR discarded real data for nearly dependent columns**: the reflection
  threshold (1e-7 on a squared norm) treated column tails shorter than 3e-4 of
  the matrix's scale as zero. It is now 1e-30, which only guards the division.
- **Transposed and other strided views were read as their untransposed
  buffer**: an unevaluated MLX array reports itself contiguous, so contiguity
  is now checked after evaluation.

## 1.0.0

`qr-apple-silicon`: batched QR on Apple GPUs (`custom_math::qr_accelerated`),
with single-threadgroup and grid-parallel Householder backends and a fixed
rule choosing between them.
