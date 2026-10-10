# metal-linalg

QR decomposition, symmetric eigendecomposition (`eigh`), singular value
decomposition (SVD), Cholesky factorization, LU factorization (with linear
solve and inverse) and triangular solves for batches of matrices on Apple
Silicon GPUs, for
[MLX](https://github.com/ml-explore/mlx), [PyTorch](https://pytorch.org) and
plain float buffers. On an M5 Pro it is **1.6-11x faster than Apple's LAPACK
on all 18 CPU cores for one large matrix** (QR, eigh and the SVD from
1024×1024 to 4096×4096, and 13.9x at 8192; Cholesky, LU and the inverse
1.3-3.4x from 2048-3072 to 4096), **1.3-5.7x faster for batches of thousands
of small matrices**, and **4.9-30x faster than PyTorch's `torch.linalg`** for QR,
eigh and the SVD (1.8-3.9x for Cholesky, LU, solve, inverse and triangular
solves, which PyTorch has MPS kernels for), against
whichever of its CPU and MPS paths is quicker ([where the GPU
wins](#where-the-gpu-wins)). A C++ library, installed with Homebrew or built from
source inside your own project, with Python packages for `mlx.core` arrays
and for `torch` tensors, a C API, and a Swift package (on `[Float]`, and on
mlx-swift's `MLXArray`).

Each solver has several Metal kernels, one per regime (small matrices in large
batches, large matrices spread over the whole GPU, long thin matrices), and
every call is routed to the fastest of them, or to LAPACK on the CPU (a batch
spread over every core), by a policy measured on the device it runs on. MLX's own `linalg::eigh`,
`linalg::svd`, `linalg::cholesky`, `linalg::lu_factor`, `linalg::solve`, `linalg::inv` and
`linalg::solve_triangular`
(`mx.linalg.*` in Python) run only on the CPU.

> **Contributions welcome: measure your Mac.** The routing is only as good as
> the measurements behind it, and every new chip needs its own.
> **[See which Macs are measured](https://c0rmac.github.io/metal-linalg/docs/measurements)**:
> every Apple Silicon chip, colour-coded per decomposition (current, out of
> date, or not measured yet). So far an M5 Pro has current measurements (an
> M1's predate 2.9.0 and are no longer used); every other Mac runs settings
> estimated from them and published benchmarks, which lean toward the CPU and
> miss some of what its GPU can do.
> If you have an Apple Silicon Mac,
> one command measures it (`python3 tuning/run.py`, about 45 minutes of the
> Mac's time on an M5 Pro, longer on smaller chips) and produces a results folder to send as a pull request. Each
> run improves the library for everyone with that Mac, and runs from several
> people with the same Mac are combined. Contributions are what keep the
> library up to date as Apple ships new chips: [how to contribute](CONTRIBUTING.md).

> **Formerly `qr-apple-silicon`.** The project was renamed in version 2.0,
> when it grew from QR to QR, symmetric eigendecomposition and SVD. Links to
> `github.com/c0rmac/qr-apple-silicon` redirect here; to update an existing
> clone, run `git remote set-url origin https://github.com/c0rmac/metal-linalg.git`.
> The changes from 1.x are listed in [CHANGELOG.md](CHANGELOG.md).

## Contents

- [Which Macs are measured](https://c0rmac.github.io/metal-linalg/docs/measurements) (the measurements page)
- [Overview](#overview)
  - [What it provides](#what-it-provides)
  - [Platforms](#platforms)
  - [Where the GPU wins](#where-the-gpu-wins):
    [QR](#qr), [eigh](#symmetric-eigensolver-eigh-eigvalsh), [SVD](#svd-svd-svdvals),
    [Cholesky](#cholesky), [LU, solve and inverse](#lu-solve-and-inverse),
    [triangular solve](#triangular-solve)
  - [How calls are routed](#how-calls-are-routed)
- [C++](#c)
  - [Install with Homebrew](#install-with-homebrew)
  - [Build from source](#build-from-source)
  - [Use it in a CMake project](#use-it-in-a-cmake-project)
  - [Quick start](#quick-start)
- [Python](#python)
  - [MLX or PyTorch](#mlx-or-pytorch)
  - [With MLX](#with-mlx)
  - [With PyTorch](#with-pytorch)
- [Swift](#swift)
  - [Install](#install)
  - [Quick start](#quick-start-1)
- [C and Objective-C](#c-and-objective-c)
  - [Install](#install-1)
  - [Quick start](#quick-start-2)
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

| operation | functions | details |
|---|---|---|
| QR | `qr_accelerated` | [docs/qr.md](docs/qr.md) |
| symmetric eigendecomposition | `eigh_accelerated`, `eigvalsh_accelerated` | [docs/eigh.md](docs/eigh.md) |
| thin SVD | `svd_accelerated`, `svdvals_accelerated` | [docs/svd.md](docs/svd.md) |
| Cholesky (since 2.18.0) | `cholesky_accelerated`, `cholesky_ex_accelerated` | [docs/cholesky.md](docs/cholesky.md) |
| LU, solve, inverse (since 2.18.0) | `lu_factor_accelerated`, `solve_accelerated`, `inv_accelerated` (and `_ex` forms with `info`) | [docs/lu.md](docs/lu.md) |
| triangular solve (since 2.18.0) | `solve_triangular_accelerated` | [docs/trsm.md](docs/trsm.md) |

On the CPU a batch is spread over every core, each solving whole matrices
(`set_cpu_threads()` or `METAL_LINALG_CPU_THREADS` caps it).

Input is any batch shape `[..., M, N]`, any real dtype (computed in float32),
any magnitude from 1e-30 to 1e+37, rank-deficient or not. The eigensolver and
the SVD return NaN for a non-finite matrix rather than raising, leaving the
rest of its batch intact; so does Cholesky for a matrix that is not positive
definite, with LAPACK's `info` from `cholesky_ex_accelerated` (the PyTorch
package raises, as `torch.linalg.cholesky` does). The SVD's factors are the thin ones,
`K = min(M, N)`; QR's too by default, with numpy's and torch's other modes:
R alone (Q never formed: up to 1.8x faster on the GPU, 2.8x on the CPU)
and a square, complete Q.

### Platforms

The same solvers and routing, from five places, each on an Apple Silicon Mac:

| platform | works on | install | |
|---|---|---|---|
| C++ | MLX arrays (`mlx::core::array`) | Homebrew, or CMake from source | [C++](#c) |
| Python, MLX | MLX arrays (`mlx.core.array`) | `pip install metal-linalg` | [With MLX](#with-mlx) |
| Python, PyTorch | `torch` tensors, on the CPU or MPS | `pip install metal-linalg-torch` | [With PyTorch](#with-pytorch) |
| Swift | `[Float]`, or mlx-swift's `MLXArray` | Swift Package Manager | [Swift](#swift) |
| C and Objective-C | plain float buffers, no MLX | as for C++ | [C and Objective-C](#c-and-objective-c) |

### Where the GPU wins

Speedup of the GPU path the router uses over the library's CPU path
(Accelerate on all 18 cores) on an Apple M5 Pro (20 GPU cores); **bold** where
the GPU is faster. Each solver's doc has the full tables and how its backends work.

#### QR

One N×N matrix:

| 1024 | 1536 | 2048 | 3072 | 4096 |
|---|---|---|---|---|
| **3.24x** | **4.07x** | **6.21x** | **7.67x** | **10.4x** |

Batches of large matrices, and tall ones:

| 16 × 1024×1024 | 4 × 2048×2048 | one 8192×512 |
|---|---|---|
| **2.1x** | **4.5x** | **5.7x** |

Batches of small matrices:

| | lone matrix | batch 16 | batch 256 | batch 4096 |
|---|---|---|---|---|
| 16×16 | 0.02x | 0.28x | 0.53x | **1.32x** |
| 32×32 | 0.06x | 0.63x | 0.63x | **3.14x** |
| 64×64 | 0.23x | 0.87x | **1.37x** | **2.42x** |
| 128×128 | 0.82x | **1.32x** | **5.19x** | **5.72x** |
| 256×256 | **1.03x** | **1.59x** | **2.81x** | **2.90x** |

#### Symmetric eigensolver (eigh, eigvalsh)

One N×N matrix:

| | 1024 | 1536 | 2048 | 3072 | 4096 | 8192 |
|---|---|---|---|---|---|---|
| eigh, with eigenvectors | **1.94x** | **2.63x** | **3.92x** | **6.07x** | **11.0x** | **13.9x** |
| eigvalsh, eigenvalues alone | **1.58x** | **1.98x** | **2.25x** | **3.06x** | **3.73x** | **3.7x** |

Batches of small matrices:

| | lone matrix | batch 16 | batch 256 | batch 4096 |
|---|---|---|---|---|
| eigh 24×24 | 0.22x | 0.40x | **1.31x** | **2.44x** |
| eigh 32×32 | 0.31x | 0.42x | **1.41x** | **2.51x** |
| eigh 64×64 | 0.74x | 0.27x | 0.95x | **1.67x** |
| eigvalsh 32×32 | 0.36x | 0.60x | **1.34x** | **3.58x** |

Batches of mid-size matrices:

| | batch 16 | batch 64 | batch 256 | batch 1024 |
|---|---|---|---|---|
| eigh 128×128 | 0.39x | 0.87x | **1.19x** | **1.52x** |
| eigh 256×256 | 0.87x | **1.31x** | **1.57x** | **1.67x** |
| eigh 512×512 | **1.05x** | **1.36x** | **1.40x** | **1.48x** |
| eigvalsh 128×128 | 0.34x | 0.61x | **1.24x** | **1.50x** |
| eigvalsh 256×256 | 0.60x | 0.87x | 0.99x | **1.11x** |

#### SVD (svd, svdvals)

One N×N matrix:

| | 1024 | 1536 | 2048 | 3072 | 4096 | 8192 |
|---|---|---|---|---|---|---|
| SVD, with vectors | **2.09x** | **2.68x** | **3.99x** | **6.48x** | **8.43x** | |
| svdvals, singular values alone | **1.88x** | **2.52x** | **3.69x** | **7.22x** | **10.1x** | **11.8x** |

Batches of small matrices:

| | lone matrix | batch 16 | batch 256 | batch 4096 |
|---|---|---|---|---|
| 16×16 | 0.15x | 0.66x | **1.35x** | **3.92x** |
| 32×32 | 0.43x | 0.57x | **1.76x** | **2.87x** |
| 48×48 | 0.67x | 0.38x | **1.67x** | **2.09x** |
| 64×64 | 0.76x | 0.35x | **1.18x** | **1.68x** |

Batches of mid-size matrices:

| | batch 16 | batch 64 | batch 256 | batch 1024 |
|---|---|---|---|---|
| SVD 128×128 | 0.71x | **1.06x** | **1.63x** | **2.05x** |
| SVD 256×256 | 0.77x | **1.23x** | **1.55x** | **1.55x** |
| SVD 512×512 | **1.01x** | **1.36x** | **1.16x** | **1.12x** |
| svdvals 128×128 | 0.64x | **1.64x** | **2.38x** | **2.53x** |
| svdvals 256×256 | 0.76x | **1.23x** | **1.39x** | **1.56x** |

#### Cholesky

N×N matrices, alone and in batches (below 1536 the CPU path wins every batch):

| | lone matrix | batch 2 | batch 4 | batch 16 |
|---|---|---|---|---|
| 1024×1024 | 0.38x | 0.55x | 0.64x | 0.84x |
| 1536×1536 | 0.66x | **1.11x** | **1.71x** | **2.39x** |
| 2048×2048 | 0.88x | **1.37x** | **1.89x** | **2.39x** |
| 3072×3072 | **1.33x** | **1.89x** | **2.51x** | |
| 4096×4096 | **2.25x** | **3.19x** | **3.89x** | |

Against MLX's own `mx.linalg.cholesky` (CPU only):

| | MLX | metal-linalg | |
|---|---|---|---|
| 16384 × 32×32 | 21.8 ms | 0.96 ms | 22.7x |
| 16 × 1024×1024 | 11.2 ms | 3.59 ms | 3.1x |
| one 4096×4096 | 32.0 ms | 13.4 ms | 2.4x |

#### LU, solve and inverse

N×N matrices:

| | 1024 | 1536 | 2048 | 3072 | 4096 |
|---|---|---|---|---|---|
| lu_factor | 0.74x | **1.03x** | **1.37x** | **2.05x** | **2.98x** |
| lu_factor, batch 2 | 0.49x | **1.03x** | **1.38x** | **2.34x** | **2.96x** |
| inv | **1.01x** | **1.44x** | **1.97x** | **2.93x** | **3.44x** |
| inv, batch 2 | 0.72x | **1.45x** | **2.03x** | **2.98x** | **3.39x** |
| solve, 1 right-hand side | 0.73x | | 0.97x | | **2.73x** |
| solve, 256 right-hand sides | 0.68x | | **1.36x** | | **2.67x** |

Against MLX's own (CPU only):

| | MLX | metal-linalg | |
|---|---|---|---|
| lu_factor, one 4096×4096 | 130 ms | 16.5 ms | 7.9x |
| solve, one 4096×4096, 1 right-hand side | 283 ms | 19.2 ms | 14.7x |
| inv, one 4096×4096 | 144 ms | 40.0 ms | 3.6x |
| inv, 4096 × 8×8 | 3.19 ms | 0.24 ms | 13.1x |

#### Triangular solve

One N×N triangle, K right-hand sides:

| | 1024 | 2048 | 3072 | 4096 |
|---|---|---|---|---|
| K = 1 | 0.29x | 0.52x | **1.04x** | **1.49x** |
| K = 256 | 0.28x | 0.57x | 0.75x | 0.98x |
| K = 1024 | 0.93x | **1.86x** | **2.27x** | **2.37x** |
| K = 4096 | | | | **3.99x** |

Against MLX's own `solve_triangular` (CPU only):

| | MLX | metal-linalg | |
|---|---|---|---|
| 4096×4096, K = 4096 | 108 ms | 13.3 ms | 8.1x |
| 2048×2048, K = 1024 | 8.6 ms | 1.73 ms | 5.0x |
| 4096×4096, K = 1 | 46 ms | 3.8 ms | 12.1x |

Measured with the routing sweeps (median, the faster of two to four passes):
QR, eigh and the SVD with 2.17.0, Cholesky, LU and the triangular solve with
2.18.0 (runs `20261010-e37928`, `-22fef9`, `-b7aec3`); the MLX comparisons
with `benchmark_cholesky`, `benchmark_lu` and `benchmark_trsm`.

### How calls are routed

Which kernel is fastest, and where the GPU overtakes the CPU, depends on the
chip, so each solver's thresholds are a table of measured policies keyed on
the Metal device name and GPU core count:

<!-- generated by tuning/generate_tables.py from docs/results/; do not edit -->
| device | QR | eigh | SVD | Cholesky | LU | triangular solve |
|---|---|---|---|---|---|---|
| Apple M1, 8 GPU cores | estimated (out of date) | estimated (out of date) | estimated | estimated | estimated | estimated |
| Apple M5 Pro, 20 GPU cores | measured | measured | measured | measured | measured | measured |
| anything else | estimated | estimated | estimated | estimated | estimated | estimated |

Every chip, and what is current: [the measurements page](https://c0rmac.github.io/metal-linalg/docs/measurements).
<!-- end of generated table -->

`qr_policy_source()`, `eigh_policy_source()` and `svd_policy_source()` say
which applies (`tuned:Apple M5 Pro`, `estimated:<name> (from Apple M5 Pro, ...)`,
`env:...` or `user`). A Mac nobody has measured gets an estimate: a measured
Mac's timings refitted for a GPU as much weaker against its CPU as Geekbench
and the memory bandwidth say, with a margin, so that it misses some GPU wins
rather than sending work to a backend that loses
([how, and how close it comes](docs/studies/estimated-policies.md)).
`METAL_LINALG_ESTIMATE_AS="Apple M4 Max:40"` shows what another Mac gets.
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

### MLX or PyTorch

There are two Python packages, with the same solvers and routing; install the
one for the arrays you use:

| you use | install | import |
|---|---|---|
| MLX (`mlx.core.array`) | `pip install metal-linalg` | `import metal_linalg as ml` |
| PyTorch (`torch.Tensor`) | `pip install metal-linalg-torch` | `import metal_linalg_torch as mlt` |

They are independent: the PyTorch package neither installs nor loads MLX,
and the MLX package does not need torch. Both can be installed side by side.

### With MLX

```bash
pip install metal-linalg
```

Prebuilt wheels for Apple Silicon Macs on macOS 14 or later, Python 3.10 to
3.14. Each release is built against one MLX release and pins it (`mlx==0.32.3`
at present), since it shares MLX's library and arrays; pip installs that
`mlx` with it. [python/README.md](python/README.md) covers building from
source, and against Homebrew's MLX.

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
`mx.array`. `ml.eigvalsh` and `ml.svdvals` return the values alone;
`ml.cholesky(p)` factors symmetric positive definite matrices, and
`ml.cholesky_ex(p)` says which were not; `ml.lu_factor`, `ml.solve` and
`ml.inv` as `mx.linalg`'s, and `ml.solve_triangular`.

### With PyTorch

```bash
pip install metal-linalg-torch
```

One prebuilt wheel for Apple Silicon Macs on macOS 14 or later, for every
Python from 3.10 and every PyTorch from 2.4: it calls the library's C API and
is compiled against neither, so it pins nothing and upgrading torch never
breaks it.

```python
import torch
import metal_linalg_torch as mlt

a = torch.randn(1000, 64, 32, device="mps")   # or on the CPU
Q, R = mlt.qr(a)                              # like torch.linalg.qr
U, S, Vh = mlt.svd(a)                         # thin factors, like torch.linalg.svd(a, full_matrices=False)
L, V = mlt.eigh(a.mT @ a)                     # like torch.linalg.eigh
C = mlt.cholesky(a.mT @ a + torch.eye(32, device=a.device))   # like torch.linalg.cholesky
X = mlt.solve(a.mT @ a + torch.eye(32, device=a.device), a.mT)  # like torch.linalg.solve; inv, lu_factor too
Y = mlt.solve_triangular(C, a.mT, upper=False)                  # like torch.linalg.solve_triangular
mlt.svd_backend(4096, 4096)                   # 'bidiag': which backend a shape gets on this Mac
```

The functions mirror their `torch.linalg` namesakes (arguments, result types,
the input's device), support autograd with torch's own formulas, and compile
with `torch.compile`: they are the custom operators
`torch.ops.metal_linalg.*`. Computation is in float32. Against `torch.linalg`
on an M5 Pro with PyTorch 2.13 (conda-forge's, its CPU LAPACK from Accelerate) and
metal-linalg 2.17 (each cell the best of ten calls over two to four runs of [`benchmarks/benchmark_torch.py`](benchmarks/benchmark_torch.py);
the same tensors on MPS for torch's MPS path and for this package, which uses them in place):

| | torch, CPU | torch, MPS | metal-linalg-torch |
|---|---|---|---|
| QR, 1024 × 128×128 | 201 ms | 32 ms | 4.2 ms |
| SVD, 256 × 128×64 | 69 ms | 70 ms | 5.1 ms |
| SVD, 4096 × 32×32 | 212 ms | 217 ms | 7.9 ms |
| eigh, 4096 × 16×16 | 33 ms | 35 ms | 1.1 ms |
| eigh, one 2048×2048 | 249 ms | 254 ms | 51 ms |
| SVD, one 4096×4096 | 3.56 s | 3.56 s | 345 ms |
| eigvalsh, one 4096×4096 | 1.77 s | 1.78 s | 124 ms |
| svdvals, one 4096×4096 | 1.94 s | 1.95 s | 191 ms |

It is ahead on every row. Of these calls PyTorch 2.13 runs only QR on the GPU
for MPS tensors, and this is 7x faster there (the QR batch runs on the
library's Householder kernels); its SVD takes as long on MPS
as on the CPU, and eigh, eigvalsh and svdvals have no MPS kernels and go
through its CPU fallback. Against those, 13-30x for the other batches of
small matrices (the SVD of 256 matrices of 128×64 is a QR and then
`golub_kahan` on the GPU; it and the SVD of 4096 of 32×32 run on the GPU and
the CPU at once), 4.9x for eigh of one 2048×2048
and 10x for the SVD of one 4096×4096 with its vectors, and 10-14x for its
eigenvalues or singular values alone (the last three by a two-stage
reduction).

The functions new in 2.18.0, where PyTorch 2.13 has MPS kernels of its own
(`python benchmarks/benchmark_torch.py --new`, best of ten calls, two runs):

| | torch, CPU | torch, MPS | metal-linalg-torch |
|---|---|---|---|
| cholesky, one 4096×4096 | 82 ms | 23 ms | 13 ms |
| cholesky, 4 × 2048×2048 | 60 ms | 13 ms | 6.7 ms |
| cholesky, 4096 × 32×32 | 1.3 ms | 1.3 ms | 0.38 ms |
| lu_factor, one 4096×4096 | 68 ms | 59 ms | 15 ms |
| lu_factor, 4 × 2048×2048 | 31 ms | 101 ms | 17 ms |
| solve, one 4096×4096, 1 rhs | 73 ms | 285 ms | 19 ms |
| solve, one 2048×2048, 512 rhs | 14 ms | 34 ms | 6.9 ms |
| inv, one 4096×4096 | 154 ms | 135 ms | 40 ms |
| inv, 1024 × 64×64 | 4.4 ms | 800 ms | 1.7 ms |
| solve_triangular, 4096×4096, 4096 rhs | 58 ms | 37 ms | 14 ms |

Ahead of torch's faster path on every row, by 1.8-3.9x (the batch of 32×32
Cholesky factorizations is on this library's CPU path, whose `spotrf` calls
are faster than torch's; the rest on the GPU).

[python-torch/README.md](python-torch/README.md) has the details: what differs
from `torch.linalg`, gradients, and MPS tensors.

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

metal_linalg::eigh_backend(32, 4096);  // EighBackend::ql: a large batch of small matrices goes to the GPU
metal_linalg::eigh_backend(512, 64);   // EighBackend::band: a batch of mid-size ones, reduced in two stages
metal_linalg::eigh_backend(2048, 1);   // EighBackend::band: one large matrix, mostly on the GPU

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
| `<metal_linalg/qr.h>` | `qr_accelerated(a, mode)`; `QrPolicy`, `qr_policy()`, `set_qr_policy()`, `qr_policy_source()`; `qr_backend(m, n, batch)` |
| `<metal_linalg/eigh.h>` | `eigh_accelerated`, `eigvalsh_accelerated`; `EighPolicy`, `eigh_policy()`, `set_eigh_policy()`, `eigh_policy_source()`; `eigh_backend(n, batch)`, `eigvalsh_backend(n, batch)`, `eigh_uses_gpu`, `eigvalsh_uses_gpu` |
| `<metal_linalg/svd.h>` | `svd_accelerated`, `svdvals_accelerated`; `SvdPolicy`, `svd_policy()`, `set_svd_policy()`, `svd_policy_source()`; `svd_backend(m, n, batch)`, `svdvals_backend(m, n, batch)`, `svd_uses_gpu`, `svdvals_uses_gpu` |
| `<metal_linalg/triangular.h>` | `solve_triangular_accelerated(a, b, upper, unit_diagonal)`; `TrsmPolicy`, `trsm_policy()`, `set_trsm_policy()`, `trsm_policy_source()`; `trsm_backend(n, k, batch)` |
| `<metal_linalg/lu.h>` | `lu_factor_accelerated(a)`, `solve_accelerated(a, b)`, `inv_accelerated(a)` and `_ex` forms (with `info`); `LuPolicy`, `lu_policy()`, `set_lu_policy()`, `lu_policy_source()`; `lu_backend(n, batch)` |
| `<metal_linalg/cholesky.h>` | `cholesky_accelerated(a, upper)`, `cholesky_ex_accelerated` (with `info`); `CholeskyPolicy`, `cholesky_policy()`, `set_cholesky_policy()`, `cholesky_policy_source()`; `cholesky_backend(n, batch)` |
| `<metal_linalg/device.h>` | `device_name()`, `gpu_core_count()`: the GPU the policies were resolved for; `cpu_threads()`, `set_cpu_threads()`: how many cores the CPU paths spread a batch over |
| `<metal_linalg/core.h>` | the same on float buffers, without MLX: `core::qr`, `core::eigh`, `core::svd`, `core::cholesky`, `core::lu_factor`, `core::solve`, `core::inv`, `core::solve_triangular`; the policies, backends and options |
| `<metal_linalg/c_api.h>` | the C API: `metal_linalg_qr`, `_qr_with_mode`, `_eigh`, `_svd`, `_cholesky`, `_lu_factor`, `_solve`, `_inv`, `_solve_triangular`, the routing queries and policies |

Each header's `metal_linalg::detail` namespace has the individual backends,
which always run their kernel, with options (tolerances, sweep bounds, launch
parameters) and a per-matrix `info` word; these are what the tests and tuning
harnesses use.

### Tests and benchmarks

```sh
ctest --test-dir build --output-on-failure    # test_qr, test_eigh, test_svd, test_cholesky, test_lu, test_trsm, test_core, test_c_api, the examples
./build/benchmark_qr                          # GPU against the CPU, per solver
./build/benchmark_eigh
./build/benchmark_svd
./build/benchmark_cholesky                    # also against MLX's own mx.linalg.cholesky
./build/benchmark_lu                          # lu_factor, solve, inv, also against MLX's own
./build/benchmark_trsm                        # the triangular solve, also against MLX's own
./build/sweep_svd --policy                    # the device and the policy in effect
```

The tests (382 QR, 456 eigh, 611 SVD, 1409 Cholesky, 491 LU and 204
triangular solve checks through MLX, 271 on the buffer API and 186 on the C
API) cover every backend directly and through the router,
shapes around every kernel boundary, batches, transposed views, structured
and rank-deficient input, magnitudes from 1e-30 to 1e+37, NaN inside a batch
(eigh, SVD), and the routing policies without assuming any device's values.
The Python (MLX and PyTorch) and Swift packages have their own tests; see their guides.

### Repository layout

| path | contents |
|---|---|
| `include/metal_linalg/` | the public headers |
| `src/` | host code: routing policies, the Metal runtime, one driver per backend, the MLX and C layers |
| `shaders/` | the Metal kernels; `prebuilt/` holds their compiled metallibs |
| `python/` | the Python package for MLX: bindings, the `metal_linalg` module, its tests |
| `python-torch/` | the Python package for PyTorch: the `metal_linalg_torch` module over the C API, its tests |
| `Package.swift`, `swift/` | the Swift package: the C module map, the embedded shaders, the Swift API and its tests |
| `examples/` | small self-checking programs, one per use case |
| `tests/` | correctness tests |
| `benchmarks/` | GPU-against-CPU benchmarks |
| `tuning/` | measuring a Mac (`run.py`), combining runs (`combine.py`), and the sweeps behind them |
| `docs/` | per-solver guides, the tuning guide, studies, and every submitted run under `results/<device>/<id>/` |

### Further documentation

- [QR](docs/qr.md), [symmetric eigensolver](docs/eigh.md),
  [SVD](docs/svd.md), [Cholesky](docs/cholesky.md), [LU, solve and
  inverse](docs/lu.md), [triangular solve](docs/trsm.md): algorithms, kernels,
  routing, accuracy, performance
- Other languages: [C](docs/c-api.md), [Swift](docs/swift.md),
  [Objective-C](docs/objective-c.md), Python with [MLX](python/README.md) or
  [PyTorch](python-torch/README.md)
- [Contributing](CONTRIBUTING.md): measuring your Mac and sending the results;
  [the measurement reference](docs/tuning.md), [how the measurements work](docs/tuning-details.md)
  and [how to read a measurement report](docs/reading-reports.md)
- Studies: QR routing on an [M1](docs/studies/qr-routing-apple-m1.md),
  eigensolver routing on an [M1](docs/studies/eigh-routing-apple-m1.md), all
  three on an [M5 Pro](docs/studies/routing-apple-m5-pro.md);
  [eigensolver launch parameters](docs/studies/eigh-launch-parameters-apple-m1.md);
  [SVD design notes](docs/studies/svd-design-notes.md);
  [performance headroom on an M5 Pro](docs/studies/performance-headroom-apple-m5-pro.md),
  behind the CPU path's use of every core and the `ql` kernel;
  [the two-stage reduction on an M5 Pro](docs/studies/two-stage-apple-m5-pro.md),
  behind the `band` backends and bisection on the GPU
- Every submitted run, with its raw timings and reports: [docs/results/](docs/results/)
- [Proposals](docs/proposals/README.md): work scoped but not done yet, with
  the measurements behind it
- [Changes](CHANGELOG.md)

## Contributing

The most useful contribution is measuring your Mac: one command,
`python3 tuning/run.py`, then a pull request with the results folder.
[CONTRIBUTING.md](CONTRIBUTING.md) walks through both, including how to make
the pull request, and covers bug reports and code changes too.

## License

MIT; see [LICENSE](LICENSE).
