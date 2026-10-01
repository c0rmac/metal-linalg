# Changes

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
