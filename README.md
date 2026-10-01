# metal-linalg

> **Formerly `qr-apple-silicon`.** The project was renamed in version 2.0,
> when it grew from QR to QR, symmetric eigendecomposition and SVD. Links to
> `github.com/c0rmac/qr-apple-silicon` redirect here; to update an existing
> clone, run `git remote set-url origin https://github.com/c0rmac/metal-linalg.git`.
> The changes from 1.x are listed in [CHANGELOG.md](CHANGELOG.md).

QR decomposition, symmetric eigendecomposition and singular value
decomposition for batches of matrices on Apple GPUs, for
[MLX](https://github.com/ml-explore/mlx) and for plain float buffers. A C++
library, installed with Homebrew or built from source inside your own
project, with Python bindings for `mlx.core` arrays, a C API, and a Swift
package (on `[Float]`, and on mlx-swift's `MLXArray`).

Each solver has several Metal kernels, one per regime (small matrices in large
batches, large matrices spread over the whole GPU, long thin matrices), and
every call is routed to the fastest of them, or to LAPACK on the CPU, by a
policy measured on the device it runs on. MLX's own `linalg::eigh` and
`linalg::svd` run only on the CPU.

> **Contributions welcome: measure your Mac.** The routing is only as good as
> the measurements behind it, and every new chip needs its own. So far an M1
> and an M5 Pro have been measured; every other Mac runs a cautious default
> that misses much of what its GPU can do. If you have an Apple Silicon Mac,
> one command measures it (`python3 tuning/run.py`, about 40 minutes of the
> Mac's time) and produces a results folder to send as a pull request. Each
> run improves the library for everyone with that Mac, and runs from several
> people with the same Mac are combined. Contributions are what keep the
> library up to date as Apple ships new chips: [how to contribute](docs/tuning.md).

## Contents

- [What it provides](#what-it-provides)
- [Installation](#installation):
  [with Homebrew](#with-homebrew),
  [from source](#from-source),
  [requirements and build options](#requirements-and-build-options)
- [Quick start](#quick-start)
- [Examples](#examples):
  [orthonormal bases with QR](#orthonormal-bases-with-qr),
  [principal components with eigh](#principal-components-with-eigh),
  [nearest orthogonal matrix with the SVD](#nearest-orthogonal-matrix-with-the-svd),
  [seeing and changing the routing](#seeing-and-changing-the-routing)
- [Using it in a CMake project](#using-it-in-a-cmake-project):
  [against the installed package](#against-the-installed-package),
  [built from source inside your project](#built-from-source-inside-your-project)
- [Using it from Python](#using-it-from-python)
- [Using it from C, Swift and Objective-C](#using-it-from-c-swift-and-objective-c):
  [C](#c), [Swift and Objective-C](#swift-and-objective-c)
- [Where the GPU wins](#where-the-gpu-wins)
- [API at a glance](#api-at-a-glance)
- [Per-device routing](#per-device-routing)
- [Tests, benchmarks and tuning](#tests-benchmarks-and-tuning)
- [Layout](#layout)
- [Documentation](#documentation)

## What it provides

| operation | functions | GPU kernels | CPU path | details |
|---|---|---|---|---|
| QR | `qr_accelerated` | Householder in one threadgroup per matrix; grid-parallel blocked Householder | none | [docs/qr.md](docs/qr.md) |
| symmetric eigendecomposition | `eigh_accelerated`, `eigvalsh_accelerated` | whole-matrix Jacobi; block Jacobi | LAPACK `ssyevd` | [docs/eigh.md](docs/eigh.md) |
| thin SVD | `svd_accelerated`, `svdvals_accelerated` | whole-matrix one-sided Jacobi; block one-sided Jacobi; either after QR for tall input | LAPACK `sgesdd` | [docs/svd.md](docs/svd.md) |

Input is any batch shape `[..., M, N]`, any real dtype (computed in float32),
any magnitude from 1e-30 to 1e+37, rank-deficient or not. The eigensolver and
the SVD return NaN for a non-finite matrix rather than raising, leaving the
rest of its batch intact. The QR and SVD factors are the thin ones,
`K = min(M, N)`.

## Installation

### With Homebrew

```bash
brew install c0rmac/metal-linalg/metal-linalg
```

This installs `libmetal_linalg.dylib`, the headers under
`include/metal_linalg/` and a CMake package. The compiled Metal shaders are
embedded in the library, so nothing is looked up on disk at run time and
nothing needs the Metal compiler. The formula lives in its own tap,
[c0rmac/homebrew-metal-linalg](https://github.com/c0rmac/homebrew-metal-linalg).

### From source

```bash
git clone https://github.com/c0rmac/metal-linalg.git
cd metal-linalg
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/opt/homebrew
cmake --build build -j
cmake --install build          # or --prefix <dir>
```

### Requirements and build options

**Requirements:** an Apple Silicon Mac, MLX (`brew install mlx`), CMake 3.25
or later and a C++20 compiler (Xcode's clang). The Metal shader compiler is
needed only to change a shader: without it the build uses the compiled
shaders committed under `shaders/prebuilt/`. From Xcode 26 it is a separate
download (`xcodebuild -downloadComponent MetalToolchain`); after editing a
shader with it, `cmake --build build --target update_prebuilt_shaders`
refreshes `shaders/prebuilt/`.

Build options: `-DMETAL_LINALG_BUILD_TESTS=OFF` skips the tests, benchmarks
and tuning harnesses; `-DMETAL_LINALG_SHARED=OFF` builds a static library;
`-DMETAL_LINALG_WITH_MLX=OFF` builds the buffer API and the C API alone, with
no MLX anywhere.

## Quick start

A complete program: one batch, all three decompositions. `CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.25)
project(my_app LANGUAGES CXX)

find_package(MetalLinalg REQUIRED)

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE metal_linalg::metal_linalg)
```

`main.cpp`:

```cpp
#include <metal_linalg/metal_linalg.h>
#include <mlx/mlx.h>

#include <cstdio>

namespace mx = mlx::core;
namespace ml = metal_linalg;

int main() {
    mx::set_default_device(mx::Device::gpu);

    // A batch of 1000 random 64 x 32 matrices.
    mx::array A = mx::random::normal({1000, 64, 32});

    // QR: Q [1000, 64, 32] with orthonormal columns, R [1000, 32, 32] upper triangular.
    auto [Q, R] = ml::qr_accelerated(A);

    // Thin SVD: U [1000, 64, 32], S [1000, 32] descending, Vt [1000, 32, 32].
    auto [U, S, Vt] = ml::svd_accelerated(A);

    // Symmetric eigendecomposition of C = A^T A: w [1000, 32] ascending, V [1000, 32, 32].
    mx::array C = mx::matmul(mx::swapaxes(A, -1, -2), A);
    auto [w, V] = ml::eigh_accelerated(C);

    mx::eval({Q, R, U, S, Vt, w, V});

    mx::array qr_err  = mx::max(mx::abs(mx::subtract(mx::matmul(Q, R), A)));
    mx::array svd_err = mx::max(mx::abs(mx::subtract(
        mx::matmul(mx::multiply(U, mx::expand_dims(S, -2)), Vt), A)));
    std::printf("max |QR - A| = %.1e, max |U diag(S) Vt - A| = %.1e\n",
                qr_err.item<float>(), svd_err.item<float>());
}
```

```sh
cmake -S . -B build -DCMAKE_PREFIX_PATH=/opt/homebrew
cmake --build build
./build/my_app      # e.g. max |QR - A| = 5.2e-06, max |U diag(S) Vt - A| = 6.7e-06
```

Every function takes any batch shape `[..., M, N]` and returns MLX arrays.
Unlike most MLX operations the decompositions are not lazy: each call
evaluates its input and runs its kernels before it returns. A few finishing
steps, such as undoing the input scaling, come back as ordinary lazy arrays,
so `eval` the results as usual. The functions are written for, and tested
with, the GPU as the default MLX device.

## Examples

Each of these is a complete program in [`examples/`](examples/), built with
the tests and run by `ctest`, so it stays correct.

### Orthonormal bases with QR

[`orthonormal_bases.cpp`](examples/orthonormal_bases.cpp): 10,000 sets of 4
vectors in R^16, orthonormalised in one call.

```cpp
mx::array vectors = mx::random::normal({10000, 16, 4});
auto [Q, R] = metal_linalg::qr_accelerated(vectors);        // Q [10000, 16, 4], Q^T Q = I
```

### Principal components with eigh

[`pca.cpp`](examples/pca.cpp): the dominant direction of each of 1000 point
clouds, from the eigenvector of its
covariance with the largest eigenvalue. Eigenvalues come back ascending, so
that is the last column.

```cpp
// X: [1000 clouds, 500 points, 8 dims]
mx::array cov = mx::divide(mx::matmul(mx::swapaxes(X, -1, -2), X), mx::array(499.0f));
auto [variance, components] = metal_linalg::eigh_accelerated(cov);
mx::array principal = mx::take(components, 7, -1);          // [1000, 8], one axis per cloud
```

`eigvalsh_accelerated(cov)` returns the eigenvalues alone, for about a third
less work.

### Nearest orthogonal matrix with the SVD

[`nearest_orthogonal.cpp`](examples/nearest_orthogonal.cpp): projecting
100,000 noisy 3×3 matrices back onto the orthogonal group (the orthogonal
Procrustes problem). If M = U S V^T, the nearest orthogonal matrix is U V^T.

```cpp
auto [U, S, Vt] = metal_linalg::svd_accelerated(observed);  // observed: [100000, 3, 3]
mx::array nearest = mx::matmul(U, Vt);
```

`svdvals_accelerated(a)` returns the singular values alone, for about half
the work.

### Seeing and changing the routing

[`routing.cpp`](examples/routing.cpp): which kernel a shape will get on this
machine, and why.

```cpp
std::printf("%s, %u GPU cores, eigh policy from %s\n", metal_linalg::device_name(),
            metal_linalg::gpu_core_count(), metal_linalg::eigh_policy_source());
// Apple M5 Pro, 20 GPU cores, eigh policy from tuned:Apple M5 Pro

metal_linalg::eigh_backend(512, 64);   // EighBackend::block: a batch of large matrices goes to the GPU
metal_linalg::eigh_backend(512, 1);    // EighBackend::cpu: a lone matrix is faster on the CPU

auto p = metal_linalg::eigh_policy();  // replace the measured policy at run time
p.gpu_min_batch = 1;
metal_linalg::set_eigh_policy(p);      // eigh_policy_source() is now "user"
```

The same can be done without recompiling through environment variables, e.g.
`EIGH_GPU_MIN_BATCH=1` or `SVD_DEVICE=gpu`; [docs/tuning.md](docs/tuning.md)
lists them.

## Using it in a CMake project

### Against the installed package

```cmake
find_package(MetalLinalg REQUIRED)
target_link_libraries(my_app PRIVATE metal_linalg::metal_linalg)
```

### Built from source inside your project

Built as part of your own project, the library uses the shaders and routing
tables of the checkout you point it at. As a subproject it is static, folded
into your binary, and installs nothing of its own:

```cmake
add_subdirectory(path/to/metal-linalg)          # or:
include(FetchContent)
FetchContent_Declare(metal_linalg
    GIT_REPOSITORY https://github.com/c0rmac/metal-linalg.git GIT_TAG v2.0.0)
FetchContent_MakeAvailable(metal_linalg)

target_link_libraries(my_app PRIVATE metal_linalg::metal_linalg)
```

## Using it from Python

```python
import mlx.core as mx
import metal_linalg as ml

a = mx.random.normal((1000, 64, 32))
Q, R = ml.qr(a)
U, S, Vt = ml.svd(a)
w, V = ml.eigh(a.swapaxes(-1, -2) @ a)
ml.eigh_backend(512, 1)      # 'cpu': which backend a shape gets on this Mac
```

The package is compiled against the MLX you have installed, so it shares
MLX's arrays without copying:

```bash
pip install mlx scikit-build-core "nanobind==2.15.0"
pip install --no-build-isolation git+https://github.com/c0rmac/metal-linalg.git
```

The nanobind version has to match the one your MLX was built with, and the
build checks it. [python/README.md](python/README.md) has the details,
including Homebrew's Python.

## Using it from C, Swift and Objective-C

### C

Under the MLX API is a core that works on plain float buffers (row-major,
the matrices of a batch one after another) and has no MLX in it:
`<metal_linalg/core.h>` in C++, and the same through a C API,
`<metal_linalg/c_api.h>`:

```c
float w[2 * 64], v[2 * 64 * 64];
metal_linalg_status st = metal_linalg_eigh(a, /*batch*/ 2, /*n*/ 64, /*lower*/ 1, w, v, NULL);
if (st != METAL_LINALG_OK) fprintf(stderr, "%s\n", metal_linalg_last_error());
```

### Swift and Objective-C

The Swift package wraps the C API, on `[Float]` and on `MLXArray`:

```swift
// .package(url: "https://github.com/c0rmac/metal-linalg.git", from: "2.0.0")
import MetalLinalg
let (w, v) = try eighAccelerated(a, batch: 2, n: 64)
```

Objective-C calls either API directly. See [docs/c-api.md](docs/c-api.md),
[docs/swift.md](docs/swift.md) and [docs/objective-c.md](docs/objective-c.md).

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

## API at a glance

| header | contents |
|---|---|
| `<metal_linalg/metal_linalg.h>` | all of the below |
| `<metal_linalg/qr.h>` | `qr_accelerated`; `QrPolicy`, `qr_policy()`, `set_qr_policy()`, `qr_policy_source()`; `qr_backend(m, n, batch)` |
| `<metal_linalg/eigh.h>` | `eigh_accelerated`, `eigvalsh_accelerated`; `EighPolicy`, `eigh_policy()`, `set_eigh_policy()`, `eigh_policy_source()`; `eigh_backend(n, batch)`, `eigh_uses_gpu` |
| `<metal_linalg/svd.h>` | `svd_accelerated`, `svdvals_accelerated`; `SvdPolicy`, `svd_policy()`, `set_svd_policy()`, `svd_policy_source()`; `svd_backend(m, n, batch)`, `svd_uses_gpu` |
| `<metal_linalg/device.h>` | `device_name()`, `gpu_core_count()`: the GPU the policies were resolved for |
| `<metal_linalg/core.h>` | the same on float buffers, without MLX: `core::qr`, `core::eigh`, `core::svd`; the policies, backends and options |
| `<metal_linalg/c_api.h>` | the C API: `metal_linalg_qr`, `_eigh`, `_svd`, the routing queries and policies |

Each header's `metal_linalg::detail` namespace has the individual backends,
which always run their kernel, with options (tolerances, sweep bounds, launch
parameters) and a per-matrix `info` word; these are what the tests and tuning
harnesses use.

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
Every threshold can also be overridden with an environment variable or
`set_*_policy()`.

**Is your Mac missing, or would you like to add to its measurements?**
`python3 tuning/run.py` measures all three decompositions in one command
(about 40 minutes) and saves a uniquely named submission to send as a pull
request. When it is merged, a GitHub Action recombines every run for that kind
of Mac and updates the library's table. See [docs/tuning.md](docs/tuning.md).

## Tests, benchmarks and tuning

```sh
ctest --test-dir build --output-on-failure    # test_qr, test_eigh, test_svd, test_core, test_c_api, the examples
./build/benchmark_qr                          # GPU against the CPU, per solver
./build/benchmark_eigh
./build/benchmark_svd
./build/sweep_svd --policy                    # the device and the policy in effect
python3 tuning/run.py                         # measure this Mac; see docs/tuning.md
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
| `python/` | the Python package: bindings, the `metal_linalg` module, its tests |
| `Package.swift`, `swift/` | the Swift package: the C module map, the embedded shaders, the Swift API and its tests |
| `src/` | host code: routing policies, the Metal runtime, one driver per backend |
| `shaders/` | the Metal kernels; `prebuilt/` holds their compiled metallibs |
| `examples/` | small self-checking programs, one per use case |
| `tests/` | correctness tests |
| `benchmarks/` | GPU-against-CPU benchmarks |
| `tuning/` | measuring a Mac (`run.py`), combining runs (`combine.py`), and the sweeps behind them |
| `docs/` | per-solver guides, the tuning guide, studies, and every submitted run under `results/<device>/<id>/` |

## Documentation

- [QR](docs/qr.md), [symmetric eigensolver](docs/eigh.md),
  [SVD](docs/svd.md): algorithms, kernels, routing, accuracy, performance
- Other languages: [C](docs/c-api.md), [Swift](docs/swift.md),
  [Objective-C](docs/objective-c.md), [Python](python/README.md)
- [Measuring your Mac](docs/tuning.md), [how the measurements work](docs/tuning-details.md),
  and [how to read a measurement report](docs/reading-reports.md)
- Studies: QR routing on an [M1](docs/studies/qr-routing-apple-m1.md),
  eigensolver routing on an [M1](docs/studies/eigh-routing-apple-m1.md), all
  three on an [M5 Pro](docs/studies/routing-apple-m5-pro.md);
  [eigensolver launch parameters](docs/studies/eigh-launch-parameters-apple-m1.md);
  [SVD design notes](docs/studies/svd-design-notes.md)
- Every submitted run, with its raw timings and reports: [docs/results/](docs/results/)
- [Changes](CHANGELOG.md)
