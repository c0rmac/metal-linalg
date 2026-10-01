# metal-linalg

> **Formerly `qr-apple-silicon`.** The project was renamed in version 2.0,
> when it grew from QR to QR, symmetric eigendecomposition and SVD. Links to
> `github.com/c0rmac/qr-apple-silicon` redirect here; to update an existing
> clone, run `git remote set-url origin https://github.com/c0rmac/metal-linalg.git`.
> The changes from 1.x are listed in [CHANGELOG.md](CHANGELOG.md).

QR decomposition, symmetric eigendecomposition and singular value
decomposition for batches of matrices on Apple GPUs, for
[MLX](https://github.com/ml-explore/mlx). A C++ library, installed with
Homebrew or built from source inside your own project.

Each solver has several Metal kernels, one per regime (small matrices in large
batches, large matrices spread over the whole GPU, long thin matrices), and
every call is routed to the fastest of them, or to MLX's CPU path, by a policy
measured on the device it runs on. MLX's own `linalg::eigh` and `linalg::svd`
run only on the CPU.

```cpp
#include <metal_linalg/metal_linalg.h>

auto [Q, R]     = metal_linalg::qr_accelerated(a);     // a: mlx::core::array [..., M, N]
auto [w, V]     = metal_linalg::eigh_accelerated(s);   // s symmetric [..., N, N]
auto [U, S, Vt] = metal_linalg::svd_accelerated(a);    // thin factors
```

## What it provides

| operation | functions | GPU kernels | CPU path | details |
|---|---|---|---|---|
| QR | `qr_accelerated` | Householder in one threadgroup per matrix; grid-parallel blocked Householder | none | [docs/qr.md](docs/qr.md) |
| symmetric eigendecomposition | `eigh_accelerated`, `eigvalsh_accelerated` | whole-matrix Jacobi; block Jacobi | MLX `eigh` | [docs/eigh.md](docs/eigh.md) |
| thin SVD | `svd_accelerated`, `svdvals_accelerated` | whole-matrix one-sided Jacobi; block one-sided Jacobi; either after QR for tall input | MLX `svd`, thin | [docs/svd.md](docs/svd.md) |

Input is any batch shape `[..., M, N]`, any real dtype (computed in float32),
any magnitude from 1e-30 to 1e+37, rank-deficient or not. The eigensolver and
the SVD return NaN for a non-finite matrix rather than raising, leaving the
rest of its batch intact. The QR and SVD factors are the thin ones,
`K = min(M, N)`.

## Where the GPU wins

For batches. On an Apple M5 Pro, the best GPU kernel against the CPU:

| | lone matrix | batch 16 | batch 256 | batch 4096 |
|---|---|---|---|---|
| eigh 16×16 | 0.10x | 0.67x | 6.5x | 15.9x |
| eigh 128×128 | 0.08x | 1.08x | 2.9x | 2.8x |
| eigh 512×512 | 0.30x | 1.10x | — | — |
| SVD 32×32 | 0.22x | 0.77x | 6.1x | 9.4x |
| SVD 2048×64 | 0.59x | 6.2x | 10.1x | GPU only |
| SVD 512×512 | 0.51x | 1.68x | — | — |

A single matrix is faster on the CPU at every size measured, up to 4096×4096,
on every device so far, so the routing keeps it there; batches of up to
1024×1024 go to the GPU. The full tables are in the per-solver docs.

## Installation

```bash
brew install c0rmac/metal-linalg/metal-linalg
```

This installs `libmetal_linalg.dylib`, the headers under
`include/metal_linalg/` and a CMake package. The compiled Metal shaders are
embedded in the library, so nothing is looked up on disk at run time and
nothing needs the Metal compiler. The formula lives in its own tap,
[c0rmac/homebrew-metal-linalg](https://github.com/c0rmac/homebrew-metal-linalg).

From source:

```bash
git clone https://github.com/c0rmac/metal-linalg.git
cd metal-linalg
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/opt/homebrew
cmake --build build -j
cmake --install build          # or --prefix <dir>
```

**Requirements:** an Apple Silicon Mac, MLX (`brew install mlx`), CMake 3.25
or later and a C++20 compiler (Xcode's clang). The Metal shader compiler is
needed only to change a shader: without it the build uses the compiled
shaders committed under `shaders/prebuilt/`. From Xcode 26 it is a separate
download (`xcodebuild -downloadComponent MetalToolchain`); after editing a
shader with it, `cmake --build build --target update_prebuilt_shaders`
refreshes `shaders/prebuilt/`.

Build options: `-DMETAL_LINALG_BUILD_TESTS=OFF` skips the tests, benchmarks
and tuning harnesses; `-DMETAL_LINALG_SHARED=OFF` builds a static library.

## Using it from CMake

Against the installed package:

```cmake
find_package(MetalLinalg REQUIRED)
target_link_libraries(my_app PRIVATE metal_linalg::metal_linalg)
```

Or built from source inside your own project, so that the shaders and routing
tables are those of the checkout. As a subproject the library is static,
folded into your binary, and installs nothing of its own:

```cmake
add_subdirectory(path/to/metal-linalg)          # or:
include(FetchContent)
FetchContent_Declare(metal_linalg
    GIT_REPOSITORY https://github.com/c0rmac/metal-linalg.git GIT_TAG v2.0.0)
FetchContent_MakeAvailable(metal_linalg)

target_link_libraries(my_app PRIVATE metal_linalg::metal_linalg)
```

[isomorphism](https://github.com/c0rmac/isomorphism)'s MLX backend uses it
this way for `qr`, `eigh` and `svd`, or the Homebrew package when configured
with `-DMETAL_LINALG_USE_INSTALLED=ON`.

## API at a glance

| header | contents |
|---|---|
| `<metal_linalg/metal_linalg.h>` | all of the below |
| `<metal_linalg/qr.h>` | `qr_accelerated`; `QrPolicy`, `qr_policy()`, `set_qr_policy()`, `qr_policy_source()`; `qr_backend(m, n, batch)` |
| `<metal_linalg/eigh.h>` | `eigh_accelerated`, `eigvalsh_accelerated`; `EighPolicy`, `eigh_policy()`, `set_eigh_policy()`, `eigh_policy_source()`; `eigh_backend(n, batch)`, `eigh_uses_gpu` |
| `<metal_linalg/svd.h>` | `svd_accelerated`, `svdvals_accelerated`; `SvdPolicy`, `svd_policy()`, `set_svd_policy()`, `svd_policy_source()`; `svd_backend(m, n, batch)`, `svd_uses_gpu` |
| `<metal_linalg/device.h>` | `device_name()`, `gpu_core_count()`: the GPU the policies were resolved for |

Each header's `metal_linalg::detail` namespace has the individual backends,
which always run their kernel, with options (tolerances, sweep bounds, launch
parameters) and a per-matrix `info` word; these are what the tests and tuning
harnesses use. The functions are written for, and tested with, the GPU as the
default MLX device: call `mlx::core::set_default_device(Device::gpu)` first.

## Per-device routing

Which kernel is fastest, and where the GPU overtakes the CPU, depends on the
chip, so each solver's thresholds are a table of measured policies keyed on
the Metal device name and GPU core count:

| device | QR | eigh | SVD |
|---|---|---|---|
| Apple M1, 8 GPU cores | measured | measured | untuned |
| Apple M5 Pro, 20 GPU cores | measured | measured | measured |
| anything else | untuned default | untuned default | untuned default |

`qr_policy_source()`, `eigh_policy_source()` and `svd_policy_source()` say
which applies (`tuned:Apple M5 Pro`, `default:untuned-device (<name>)`,
`env:...` or `user`). The defaults err toward the CPU, so an untuned device
misses GPU wins rather than routing work to a kernel that takes seconds.
Measuring a new device takes about 40 minutes and one command per solver; see
[docs/tuning.md](docs/tuning.md). Every threshold can also be overridden with
an environment variable or `set_*_policy()`.

## Tests, benchmarks and tuning

```sh
ctest --test-dir build --output-on-failure    # test_qr, test_eigh, test_svd
./build/benchmark_qr                          # GPU against the CPU, per solver
./build/benchmark_eigh
./build/benchmark_svd
./build/sweep_svd --policy                    # the device and the policy in effect
python3 tuning/tune_svd.py build/sweep_svd    # measure this device; see docs/tuning.md
```

The tests (75 QR, 135 eigh, 197 SVD checks) cover every backend directly and
through the router, shapes around every kernel boundary, batches, transposed
views, structured and rank-deficient input, magnitudes from 1e-30 to 1e+37,
NaN inside a batch (eigh, SVD), and the routing policies without assuming any
device's values.

## Layout

| path | contents |
|---|---|
| `include/metal_linalg/` | the public headers |
| `src/` | host code: routing policies, the Metal runtime, one driver per backend |
| `shaders/` | the Metal kernels; `prebuilt/` holds their compiled metallibs |
| `tests/` | correctness tests |
| `benchmarks/` | GPU-against-CPU benchmarks |
| `tuning/` | the per-device sweep binaries and analysis scripts |
| `docs/` | per-solver guides, the tuning guide, studies and committed results |

## Documentation

- [QR](docs/qr.md), [symmetric eigensolver](docs/eigh.md),
  [SVD](docs/svd.md): algorithms, kernels, routing, accuracy, performance
- [Tuning on another device](docs/tuning.md)
- Studies: QR routing on an [M1](docs/studies/qr-routing-apple-m1.md),
  eigensolver routing on an [M1](docs/studies/eigh-routing-apple-m1.md), all
  three on an [M5 Pro](docs/studies/routing-apple-m5-pro.md);
  [eigensolver launch parameters](docs/studies/eigh-launch-parameters-apple-m1.md);
  [SVD design notes](docs/studies/svd-design-notes.md)
- Raw timings and generated reports: [docs/results/](docs/results/)
- [Changes](CHANGELOG.md)
