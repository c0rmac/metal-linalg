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
every call is routed to the fastest of them or, for the eigensolver and the
SVD, to LAPACK on the CPU, by a policy measured on the device it runs on. QR
has no CPU path yet: every QR call runs on the GPU. MLX's own `linalg::eigh`
and `linalg::svd` run only on the CPU.

> **Contributions welcome: measure your Mac.** The routing is only as good as
> the measurements behind it, and every new chip needs its own. So far an M1
> and an M5 Pro have been measured; every other Mac runs a cautious default
> that misses much of what its GPU can do. If you have an Apple Silicon Mac,
> one command measures it (`python3 tuning/run.py`, about 40 minutes of the
> Mac's time) and produces a results folder to send as a pull request. Each
> run improves the library for everyone with that Mac, and runs from several
> people with the same Mac are combined. Contributions are what keep the
> library up to date as Apple ships new chips: [how to contribute](CONTRIBUTING.md).

## Contents

- [Overview](#overview)
  - [What it provides](#what-it-provides)
  - [Platforms](#platforms)
  - [Where the GPU wins](#where-the-gpu-wins)
  - [How calls are routed](#how-calls-are-routed)
- [C++](#c)
  - [Install with Homebrew](#install-with-homebrew)
  - [Build from source](#build-from-source)
  - [Use it in a CMake project](#use-it-in-a-cmake-project)
  - [Quick start](#quick-start)
- [Python](#python)
  - [Install](#install)
  - [Quick start](#quick-start-1)
- [Swift](#swift)
  - [Install](#install-1)
  - [Quick start](#quick-start-2)
- [C and Objective-C](#c-and-objective-c)
  - [Install](#install-2)
  - [Quick start](#quick-start-3)
  - [Objective-C](#objective-c)
- [Examples](#examples)
  - [Orthonormal bases with QR](#orthonormal-bases-with-qr)
  - [Principal components with eigh](#principal-components-with-eigh)
  - [Nearest orthogonal matrix with the SVD](#nearest-orthogonal-matrix-with-the-svd)
  - [Seeing and changing the routing](#seeing-and-changing-the-routing)
- [Reference](#reference)
  - [API at a glance](#api-at-a-glance)
  - [Tests and benchmarks](#tests-and-benchmarks)
  - [Repository layout](#repository-layout)
  - [Further documentation](#further-documentation)
- [Contributing](#contributing)
- [License](#license)

## Overview

### What it provides

| operation | functions | GPU kernels | CPU path | details |
|---|---|---|---|---|
| QR | `qr_accelerated` | Householder in one threadgroup per matrix; grid-parallel blocked Householder | none yet: always on the GPU | [docs/qr.md](docs/qr.md) |
| symmetric eigendecomposition | `eigh_accelerated`, `eigvalsh_accelerated` | whole-matrix Jacobi; block Jacobi | LAPACK `ssyevd` | [docs/eigh.md](docs/eigh.md) |
| thin SVD | `svd_accelerated`, `svdvals_accelerated` | whole-matrix one-sided Jacobi; block one-sided Jacobi; either after QR for tall input | LAPACK `sgesdd` | [docs/svd.md](docs/svd.md) |

Input is any batch shape `[..., M, N]`, any real dtype (computed in float32),
any magnitude from 1e-30 to 1e+37, rank-deficient or not. The eigensolver and
the SVD return NaN for a non-finite matrix rather than raising, leaving the
rest of its batch intact. The QR and SVD factors are the thin ones,
`K = min(M, N)`.

### Platforms

The same solvers and routing, from four places, each on an Apple Silicon Mac:

| platform | works on | install | |
|---|---|---|---|
| C++ | MLX arrays (`mlx::core::array`) | Homebrew, or CMake from source | [C++](#c) |
| Python | MLX arrays (`mlx.core.array`) | `pip` | [Python](#python) |
| Swift | `[Float]`, or mlx-swift's `MLXArray` | Swift Package Manager | [Swift](#swift) |
| C and Objective-C | plain float buffers, no MLX | as for C++ | [C and Objective-C](#c-and-objective-c) |

### Where the GPU wins

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

QR is not in this table because it has no CPU path yet: every QR call runs on
the GPU, whatever its size and batch, and the routing only chooses between its
two GPU kernels. So a lone small matrix, which for eigh and SVD goes to the
CPU, still goes to the GPU for QR. A CPU path for QR, routed the same way as
the other two, is planned.

### How calls are routed

Which GPU kernel is fastest and, for eigh and SVD, where the GPU overtakes the
CPU depend on the chip, so each solver's thresholds are a table of measured
policies keyed on the Metal device name and GPU core count. For QR, *measured*
means the crossover between its two GPU kernels; for eigh and SVD it also
means where calls move to the CPU:

<!-- generated by tuning/generate_tables.py from docs/results/; do not edit -->
| device | QR | eigh | SVD |
|---|---|---|---|
| Apple M1, 8 GPU cores | measured | measured | untuned |
| Apple M5 Pro, 20 GPU cores | measured | measured | measured |
| anything else | untuned default | untuned default | untuned default |
<!-- end of generated table -->

`qr_policy_source()`, `eigh_policy_source()` and `svd_policy_source()` say
which applies (`tuned:Apple M5 Pro`, `default:untuned-device (<name>)`,
`env:...` or `user`). The defaults err toward the CPU, so an untuned device
misses GPU wins rather than routing work to a kernel that takes seconds.
Every threshold can also be overridden with an environment variable or
`set_*_policy()`; see [seeing and changing the routing](#seeing-and-changing-the-routing).

## C++

### Install with Homebrew

The formula lives in its own tap,
[c0rmac/homebrew-metal-linalg](https://github.com/c0rmac/homebrew-metal-linalg).
Tap it, trust it, then install:

```bash
brew tap c0rmac/metal-linalg
brew trust c0rmac/metal-linalg        # Homebrew 7 and later ask this of any third-party tap
brew install metal-linalg
```

This builds and installs `libmetal_linalg.dylib`, the headers under
`include/metal_linalg/` and a CMake package. The compiled Metal shaders are
embedded in the library, so nothing is looked up on disk at run time and
nothing needs the Metal compiler. Updating, uninstalling and depending on it
from another formula are covered in
[the tap's README](https://github.com/c0rmac/homebrew-metal-linalg#readme).

### Build from source

You need CMake 3.25 or later, a C++20 compiler (Xcode's clang) and MLX
(`brew install mlx`). The Metal shader compiler is needed only to change a
shader: without it the build uses the compiled shaders committed under
`shaders/prebuilt/`. From Xcode 26 it is a separate download (`xcodebuild
-downloadComponent MetalToolchain`); after editing a shader with it,
`cmake --build build --target update_prebuilt_shaders` refreshes
`shaders/prebuilt/`.

```bash
git clone https://github.com/c0rmac/metal-linalg.git
cd metal-linalg
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/opt/homebrew
cmake --build build -j
cmake --install build          # or --prefix <dir>
```

Build options:

| option | default | effect |
|---|---|---|
| `METAL_LINALG_WITH_MLX` | `ON` | the MLX API; `OFF` builds the buffer API and the C API alone, with no MLX anywhere |
| `METAL_LINALG_SHARED` | `ON` when built on its own, `OFF` as a subproject | a shared or a static library |
| `METAL_LINALG_BUILD_TESTS` | `ON` when built on its own, `OFF` as a subproject | the tests, benchmarks and tuning harnesses |
| `METAL_LINALG_BUILD_EXAMPLES` | that of `METAL_LINALG_BUILD_TESTS` | the examples, run as tests |
| `METAL_LINALG_USE_PREBUILT_SHADERS` | `OFF`, and `ON` without a Metal compiler | the metallibs in `shaders/prebuilt/` instead of compiling the shaders |

### Use it in a CMake project

Against the installed package:

```cmake
find_package(MetalLinalg REQUIRED)
target_link_libraries(my_app PRIVATE metal_linalg::metal_linalg)
```

Or built as part of your own project, so that the library uses the shaders
and routing tables of the checkout you point it at. As a subproject it is
static, folded into your binary, and installs nothing of its own:

```cmake
add_subdirectory(path/to/metal-linalg)          # or:
include(FetchContent)
FetchContent_Declare(metal_linalg
    GIT_REPOSITORY https://github.com/c0rmac/metal-linalg.git GIT_TAG v2.0.0)   # or main, for the latest
FetchContent_MakeAvailable(metal_linalg)

target_link_libraries(my_app PRIVATE metal_linalg::metal_linalg)
```

### Quick start

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

## Python

### Install

You need an Apple Silicon Mac, Python 3.10 or later, Xcode's command line
tools (`xcode-select --install`) and CMake (`brew install cmake`). The
package is compiled on your Mac against the MLX you have installed, so that
it shares MLX's arrays without copying:

```bash
pip install mlx scikit-build-core "nanobind==2.15.0"
pip install --no-build-isolation git+https://github.com/c0rmac/metal-linalg.git
```

The nanobind version has to match the one your MLX was built with (2.15.0 for
MLX 0.32), and the build checks it. After upgrading MLX, reinstall the
package. [python/README.md](python/README.md) has the details, including
Homebrew's Python.

### Quick start

```python
import mlx.core as mx
import metal_linalg as ml

a = mx.random.normal((1000, 64, 32))
Q, R = ml.qr(a)
U, S, Vt = ml.svd(a)
w, V = ml.eigh(a.swapaxes(-1, -2) @ a)
ml.eigh_backend(512, 1)      # 'cpu': which backend a shape gets on this Mac
```

Inputs may be `mx.array`, NumPy arrays or nested lists; outputs are float32
`mx.array`. `ml.eigvalsh` and `ml.svdvals` return the values alone.

## Swift

### Install

The repository is a Swift package with two libraries: `MetalLinalg`, on
`[Float]`, which needs nothing else, and `MetalLinalgMLX`, on mlx-swift's
`MLXArray`. macOS 14 or later, on Apple Silicon. In `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/c0rmac/metal-linalg.git", from: "2.0.0"),
],
targets: [
    .target(name: "MyApp", dependencies: [
        .product(name: "MetalLinalg", package: "metal-linalg"),        // or "MetalLinalgMLX"
    ]),
]
```

or in Xcode, File > Add Package Dependencies, with the repository's URL.

### Quick start

```swift
import MetalLinalg

// Two symmetric 64 x 64 matrices, row-major, one after the other.
let (w, v) = try eighAccelerated(a, batch: 2, n: 64)       // w ascending, v's columns the vectors
```

```swift
import MLX
import MetalLinalgMLX

let a = MLXRandom.normal([1000, 64, 32])
let (q, r) = try qrAccelerated(a)
let (u, s, vt) = try svdAccelerated(a)
```

Errors are thrown as `MetalLinalgError`. [docs/swift.md](docs/swift.md) has
the whole API and how the package is built.

## C and Objective-C

### Install

The C API is part of the library: install it as for [C++](#c), with Homebrew
or from source. To leave MLX out of the library altogether, build from
source with `-DMETAL_LINALG_WITH_MLX=OFF`.

### Quick start

Under the MLX API is a core that works on plain float buffers (row-major,
the matrices of a batch one after another) and has no MLX in it:
`<metal_linalg/core.h>` in C++, and the same through a C API,
`<metal_linalg/c_api.h>`:

```c
#include <metal_linalg/c_api.h>
#include <stdio.h>

int main(void) {
    /* Two symmetric 2 x 2 matrices, row-major, one after the other. */
    const float a[8] = {2, 1, 1, 2,   4, 0, 0, 1};
    float w[4], v[8];
    if (metal_linalg_eigh(a, 2, 2, /*lower*/ 1, w, v, NULL) != METAL_LINALG_OK) {
        fprintf(stderr, "eigh failed: %s\n", metal_linalg_last_error());
        return 1;
    }
    printf("%g %g | %g %g\n", w[0], w[1], w[2], w[3]);   /* 1 3 | 1 4 */
    return 0;
}
```

```bash
cc -std=c99 main.c -I/opt/homebrew/include -L/opt/homebrew/lib -lmetal_linalg -o main
```

The conventions (layout, optional outputs, errors) are in
[docs/c-api.md](docs/c-api.md).

### Objective-C

Objective-C++ (`.mm`) calls the C++ API directly, on MLX arrays; any
Objective-C file can call the C API on its own buffers, with no MLX. See
[docs/objective-c.md](docs/objective-c.md).

## Examples

In C++. Each is a complete program in [`examples/`](examples/), built with the
tests and run by `ctest`, so it stays correct.

### Orthonormal bases with QR

[`orthonormal_bases.cpp`](examples/orthonormal_bases.cpp): 10,000 sets of 4
vectors in R^16, orthonormalised in one call.

```cpp
mx::array vectors = mx::random::normal({10000, 16, 4});
auto [Q, R] = metal_linalg::qr_accelerated(vectors);        // Q [10000, 16, 4], Q^T Q = I
```

### Principal components with eigh

[`pca.cpp`](examples/pca.cpp): the dominant direction of each of 1000 point
clouds, from the eigenvector of its covariance with the largest eigenvalue.
Eigenvalues come back ascending, so that is the last column.

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

## Reference

### API at a glance

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

### Tests and benchmarks

```sh
ctest --test-dir build --output-on-failure    # test_qr, test_eigh, test_svd, test_core, test_c_api, the examples
./build/benchmark_qr                          # GPU against the CPU, per solver
./build/benchmark_eigh
./build/benchmark_svd
./build/sweep_svd --policy                    # the device and the policy in effect
```

The tests (75 QR, 135 eigh and 197 SVD checks through MLX, 186 on the buffer
API and 45 on the C API) cover every backend directly and through the router,
shapes around every kernel boundary, batches, transposed views, structured
and rank-deficient input, magnitudes from 1e-30 to 1e+37, NaN inside a batch
(eigh, SVD), and the routing policies without assuming any device's values.
The Python and Swift packages have their own tests; see their guides.

### Repository layout

| path | contents |
|---|---|
| `include/metal_linalg/` | the public headers |
| `src/` | host code: routing policies, the Metal runtime, one driver per backend, the MLX and C layers |
| `shaders/` | the Metal kernels; `prebuilt/` holds their compiled metallibs |
| `python/` | the Python package: bindings, the `metal_linalg` module, its tests |
| `Package.swift`, `swift/` | the Swift package: the C module map, the embedded shaders, the Swift API and its tests |
| `examples/` | small self-checking programs, one per use case |
| `tests/` | correctness tests |
| `benchmarks/` | GPU-against-CPU benchmarks |
| `tuning/` | measuring a Mac (`run.py`), combining runs (`combine.py`), and the sweeps behind them |
| `docs/` | per-solver guides, the tuning guide, studies, and every submitted run under `results/<device>/<id>/` |

### Further documentation

- [QR](docs/qr.md), [symmetric eigensolver](docs/eigh.md),
  [SVD](docs/svd.md): algorithms, kernels, routing, accuracy, performance
- Other languages: [C](docs/c-api.md), [Swift](docs/swift.md),
  [Objective-C](docs/objective-c.md), [Python](python/README.md)
- [Contributing](CONTRIBUTING.md): measuring your Mac and sending the results;
  [the measurement reference](docs/tuning.md), [how the measurements work](docs/tuning-details.md)
  and [how to read a measurement report](docs/reading-reports.md)
- Studies: QR routing on an [M1](docs/studies/qr-routing-apple-m1.md),
  eigensolver routing on an [M1](docs/studies/eigh-routing-apple-m1.md), all
  three on an [M5 Pro](docs/studies/routing-apple-m5-pro.md);
  [eigensolver launch parameters](docs/studies/eigh-launch-parameters-apple-m1.md);
  [SVD design notes](docs/studies/svd-design-notes.md)
- Every submitted run, with its raw timings and reports: [docs/results/](docs/results/)
- [Changes](CHANGELOG.md)

## Contributing

The most useful contribution is measuring your Mac: one command,
`python3 tuning/run.py`, then a pull request with the results folder.
[CONTRIBUTING.md](CONTRIBUTING.md) walks through both, including how to make
the pull request, and covers bug reports and code changes too.

## License

MIT; see [LICENSE](LICENSE).
