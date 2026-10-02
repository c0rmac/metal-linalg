# Changes

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
