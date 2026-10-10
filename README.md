# metal-linalg

QR decomposition, symmetric eigendecomposition (`eigh`), singular value
decomposition (SVD), Cholesky factorization, LU factorization (with linear
solve and inverse) and triangular solves for batches of matrices on Apple
Silicon GPUs, for
[MLX](https://github.com/ml-explore/mlx), [PyTorch](https://pytorch.org) and
plain float buffers. On an M5 Pro it is **1.6-11x faster than Apple's LAPACK
on all 18 CPU cores for one large matrix** (1024×1024 to 4096×4096, and
13.9x at 8192), **1.3-5.7x faster for batches of thousands of small
matrices**, and **4.9-30x faster than PyTorch's `torch.linalg`**, against
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
  - [Where the GPU wins](#where-the-gpu-wins)
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

| operation | functions | GPU kernels | CPU path | details |
|---|---|---|---|---|
| QR | `qr_accelerated` | Householder in one simdgroup's registers, or blocked in one threadgroup, per matrix; grid-parallel blocked Householder | LAPACK `sgeqrf`, `sorgqr` | [docs/qr.md](docs/qr.md) |
| symmetric eigendecomposition | `eigh_accelerated`, `eigvalsh_accelerated` | whole-matrix Jacobi; block Jacobi; tridiagonalization and implicit QL in one simdgroup's registers (N <= 32) or one threadgroup per matrix (N <= 87); a batch of mid-size matrices tridiagonalized together, their tridiagonal problems on every CPU core (N <= 1024); Householder tridiagonalization for large N (with LAPACK's tridiagonal solver, or bisection on the GPU for eigenvalues alone); for large N, a two-stage reduction (to a band on the GPU, then to tridiagonal on every CPU core, the eigenvectors' transformations applied on the GPU) | LAPACK `ssyevd`; `ssyevd_2stage` for eigenvalues alone from N = 128 | [docs/eigh.md](docs/eigh.md) |
| thin SVD | `svd_accelerated`, `svdvals_accelerated` | whole-matrix one-sided Jacobi; block one-sided Jacobi; either after QR for tall input; bidiagonalization and implicit QR in one simdgroup's registers (up to 32 x 32) or one threadgroup per matrix (k <= 83); a batch of mid-size matrices bidiagonalized together, their bidiagonal problems on every CPU core (up to 1024 x 1024); Householder bidiagonalization for large k (with LAPACK's bidiagonal solver, or bisection on the GPU for singular values alone); for large k, a two-stage reduction (to a band on the GPU, then to bidiagonal on every CPU core, the singular vectors' transformations applied on the GPU) | LAPACK `sgesdd` | [docs/svd.md](docs/svd.md) |
| Cholesky (since 2.18.0) | `cholesky_accelerated`, `cholesky_ex_accelerated` | in one simdgroup's registers (up to 32 x 32); one threadgroup per matrix; for large N, 32-column sub-panels each brought up to date, factored and solved in one dispatch, the trailing update as MPS products on the lower triangle | LAPACK `spotrf` (lower, padded) | [docs/cholesky.md](docs/cholesky.md) |
| LU, solve, inverse (since 2.18.0) | `lu_factor_accelerated`, `solve_accelerated`, `inv_accelerated` (and `_ex` forms with `info`) | for large N, the GPU and the CPU on one matrix: pivoted panels on the CPU in shared memory with a look-ahead, row swaps and MPS products on the GPU; blocked triangular solves on the GPU | LAPACK `sgetrf`, `sgetrs`, `sgetri` (padded) | [docs/lu.md](docs/lu.md) |
| triangular solve (since 2.18.0) | `solve_triangular_accelerated` | blocked: 128 rows at a time, two MPS products a block with the diagonal blocks' inverses | BLAS `strsm` | [docs/trsm.md](docs/trsm.md) |

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

On an Apple M5 Pro (20 GPU cores), against a CPU path that spreads every call
over all 18 CPU cores, the GPU wins in two places, and the router sends work
there and nowhere else.

**One large matrix: up to 11x, and 13.9x at 8192.** eigh and the SVD keep
LAPACK's method and move its memory-bound reduction (to tridiagonal or
bidiagonal form) and its back-transformation to the GPU; the divide and
conquer between them runs on every CPU core (since 2.15.0). For the eigenvalues
or singular values alone, large matrices are reduced in two stages (since
2.13.0): to a band on the GPU, in blocks whose work is matrix products, then
to tridiagonal or bidiagonal on every CPU core, and the values come from
bisection on the GPU. Since 2.15.0 the SVD with vectors can take the two
stages too, both stages' transformations applied on the GPU while the CPU
chases the band and solves the bidiagonal problem. QR runs on the same
panels and matrix products (since 2.15.0, a batch at once). One N×N matrix
against the CPU:

| | 1024 | 1536 | 2048 | 3072 | 4096 |
|---|---|---|---|---|---|
| svdvals, singular values alone | 1.88x | 2.52x | **3.69x** | **7.22x** | **10.1x** |
| eigh, with eigenvectors | 1.94x | 2.63x | **3.92x** | **6.07x** | **11.0x** |
| eigvalsh, eigenvalues alone | 1.58x | 1.98x | 2.25x | **3.06x** | **3.73x** |
| SVD, with vectors | 2.09x | 2.68x | **3.99x** | **6.48x** | **8.43x** |
| QR | **3.24x** | **4.07x** | **6.21x** | **7.67x** | **10.4x** |
| Cholesky | 0.38x | 0.66x | 0.88x | **1.33x** | **2.25x** |
| LU (`lu_factor`) | 0.74x | 1.03x | **1.37x** | **2.05x** | **2.98x** |
| inverse | 1.01x | **1.44x** | **1.97x** | **2.93x** | **3.44x** |

At 8192, the singular values alone take 1.06 s against the CPU's 12.6 s
(11.8x), eigh 1.27 s against 17.6 s (13.9x), and the eigenvalues alone 0.71 s
against 2.64 s (3.7x, against LAPACK's own two-stage driver). The M5 Pro uses
these backends for one matrix or a few: eigh and the SVD with vectors from
N = 512 (two stages, since 2.17.0 eigh's too: its eigenvectors' transformations
applied on the GPU), svdvals from 768, eigvalsh from 1024, QR from 512; a
few of them in two stages are reduced as one batch. The CPU path, faster
in 2.17.0 with vectors (its divide and conquer on every core), lowered the
SVD's ratios against 2.16.0's while the GPU's times held (`bidiag`, the
one-stage reduction, 3.5x at 4096). QR's GPU path takes batches too: 16 of
1024×1024 2.1x, 4 of 2048×2048 4.5x, and tall matrices, one 8192×512 5.7x.
At 4096, 2.14 had svdvals at 8.62x, eigh 5.62x, eigvalsh 2.91x, the SVD with
vectors 2.32x and QR 2.7x.

Cholesky (since 2.18.0) is $N^3/3$ work, a third of QR's, and the CPU path
(`spotrf` on every core) is quick at it: the router keeps a lone matrix on the
CPU up to 3072, but takes batches to the GPU from 1536, where it leads by
1.7x for 4 of 1536×1536, 1.9x for 4 of 2048×2048, 2.5x for 4 of 3072×3072 and
3.9x for 4 of 4096×4096. Below 1536 the CPU wins every batch on this Mac (the
nearest the GPU kernels come is 0.93x, 4096 matrices of 8×8);
[docs/cholesky.md](docs/cholesky.md) has the measurements. Against MLX's own
`mx.linalg.cholesky`, which factors one matrix at a time, the library is
1.4-2.4x faster for one matrix and 20-24x for 16384 small ones.

LU (since 2.18.0) splits each matrix between the two: the CPU factors each
pivoted panel of 128 columns in memory the GPU shares, and brings the next
panel up to date itself, while the GPU swaps rows and updates the rest by
matrix products; `solve` and `inv` finish with blocked triangular solves on
the GPU. The router takes it to the GPU from 1536 ([docs/lu.md](docs/lu.md)).
Against MLX's own CPU functions, one 4096×4096 `lu_factor` is 8.1x faster,
`solve` 15x and `inv` 3.6x. The triangular solve takes many right-hand sides
to the GPU (from 1024 at N = 2048): 4096×4096 with 4096 in 12.5 ms against
50 for `strsm` on every core (4.0x) and 100 for MLX's ([docs/trsm.md](docs/trsm.md)).

**Large batches of small matrices: up to 5.7x.** LAPACK's own methods, a
matrix to a threadgroup or a simdgroup, carry the GPU's lead: the
eigensolver's `ql` kernel (tridiagonalization and QL), the SVD's
`golub_kahan` (bidiagonalization and implicit QR, 1.6-3x faster than the
Jacobi kernels it replaced), and since 2.16.0 QR's Householder kernels (up to
32 columns in a simdgroup's registers; above that blocked in a threadgroup,
the updates as 8×8 simdgroup matrix products), which take QR's batches from
the CPU from 64×64 at 256 matrices and lead by 5.2-5.7x at 128×128. eigh's
and the SVD's large batches are shared, the GPU and the CPU solving them at
once (from 256 matrices, eigh's from N = 48 and the SVD's from k = 32: below
those, since 2.17.0, the kernels in registers alone are faster), up to 64×64
for eigh and 80×80 for the SVD. The best GPU route against the CPU alone:

| | lone matrix | batch 16 | batch 256 | batch 4096 |
|---|---|---|---|---|
| eigh 24×24 | 0.22x | 0.40x | 1.31x | **2.44x** |
| eigh 32×32 | 0.31x | 0.42x | 1.41x | **2.51x** |
| eigh 64×64 | 0.74x | 0.27x | 0.95x | 1.67x |
| eigvalsh 32×32 | 0.36x | 0.60x | 1.34x | **3.58x** |
| SVD 16×16 | 0.15x | 0.66x | 1.35x | **3.92x** |
| SVD 32×32 | 0.43x | 0.57x | 1.76x | **2.87x** |
| SVD 48×48 | 0.67x | 0.38x | 1.67x | **2.09x** |
| SVD 64×64 | 0.76x | 0.35x | 1.18x | 1.68x |
| QR 16×16 | 0.02x | 0.28x | 0.53x | 1.32x |
| QR 32×32 | 0.06x | 0.63x | 0.63x | **3.14x** |
| QR 64×64 | 0.23x | 0.87x | 1.37x | **2.42x** |
| QR 128×128 | 0.82x | 1.32x | **5.19x** | **5.72x** |
| QR 256×256 | 1.03x | 1.59x | **2.81x** | **2.90x** |

QR 16×16 at 4096 matrices takes 0.27-0.31 ms on the GPU in some runs and
0.68-0.80 ms in others, its clocks' state (2.9x the CPU or 1.25x); the table
has the slower, which all four of its passes measured.

**Batches of mid-size matrices: up to 2.5x** (since 2.17.0). The whole
batch is reduced together on the GPU, a threadgroup a matrix and panel, and
the tridiagonal or bidiagonal problems are solved on every CPU core under
it (`tridiag_batch`, `bidiag_batch`; from 384, two stages); against the CPU
alone:

| | batch 16 | batch 64 | batch 256 | batch 1024 |
|---|---|---|---|---|
| eigh 128×128 | 0.39x | 0.87x | 1.19x | 1.52x |
| eigh 256×256 | 0.87x | 1.31x | 1.57x | 1.67x |
| eigh 512×512 | 1.05x | 1.36x | 1.40x | 1.48x |
| eigvalsh 128×128 | 0.34x | 0.61x | 1.24x | 1.50x |
| eigvalsh 256×256 | 0.60x | 0.87x | 0.99x | 1.11x |
| SVD 128×128 | 0.71x | 1.06x | 1.63x | **2.05x** |
| SVD 256×256 | 0.77x | 1.23x | 1.55x | 1.55x |
| SVD 512×512 | 1.01x | 1.36x | 1.16x | 1.12x |
| svdvals 128×128 | 0.64x | 1.64x | **2.38x** | **2.53x** |
| svdvals 256×256 | 0.76x | 1.23x | 1.39x | 1.56x |

Lone small matrices and small batches stay on the CPU, which is up to 100x faster
there: since 2.9.0 it spreads a batch over every core ([the
performance-headroom study](docs/studies/performance-headroom-apple-m5-pro.md)
has why). The numbers for one large matrix are 2.17.0's, measured side by
side with the CPU path (`sweep_eigh`, `sweep_svd`, `sweep_qr`, median, MLX's
buffer cache on as an MLX program has it; at 8192 the CPU's for the values
alone from 2.15.0, its path unchanged since); the batches' too (each GPU
backend against the CPU path, the best of four passes for small matrices
and two for mid-size ones, sharing from batch 64 as the routing sweeps time
it), and QR's batches of large and tall matrices (the best of two passes). The full tables are in the per-solver docs; how the two-stage reduction
got there is in [the two-stage study](docs/studies/two-stage-apple-m5-pro.md),
and 2.15.0's changes in [its study](docs/studies/proposals-2-15-apple-m5-pro.md).

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
