# metal-linalg for Python

QR, symmetric eigendecomposition and SVD for batches of matrices on Apple
GPUs, on [MLX](https://github.com/ml-explore/mlx) arrays.

```python
import mlx.core as mx
import metal_linalg as ml

a = mx.random.normal((1000, 64, 32))     # a batch of 1000 matrices

Q, R = ml.qr(a)                          # Q (1000, 64, 32), R (1000, 32, 32)
U, S, Vt = ml.svd(a)                     # thin: U (1000, 64, 32), S (1000, 32), Vt (1000, 32, 32)
w, V = ml.eigh(a.swapaxes(-1, -2) @ a)   # w ascending (1000, 32), V (1000, 32, 32)
```

Each call is routed to the fastest Metal kernel for its shape and batch, or to
LAPACK on the CPU, by a policy measured on the Mac it runs on:

```python
ml.device_name()                      # 'Apple M5 Pro'
ml.eigh_policy_source()               # 'tuned:Apple M5 Pro'
ml.eigh_backend(512, 64)              # 'block': 64 matrices of 512x512 go to the GPU
ml.eigh_backend(512, 1)               # 'cpu': one matrix is faster on the CPU
ml.set_eigh_policy(gpu_min_batch=1)   # override the measured policy
```

Inputs may be `mx.array`, NumPy arrays or nested lists; outputs are float32
`mx.array`. `ml.eigvalsh` and `ml.svdvals` return the values alone, for less
work. The kernels and the routing are described in the
[main README](https://github.com/c0rmac/metal-linalg).

## Installing

The package is compiled on your Mac against the MLX you have installed,
because it has to share MLX's own library with the `mlx` package. You need an
Apple Silicon Mac, Xcode's command line tools (`xcode-select --install`),
CMake (`brew install cmake`) and Python 3.10 or later.

```bash
pip install mlx scikit-build-core "nanobind==2.15.0"
pip install --no-build-isolation git+https://github.com/c0rmac/metal-linalg.git
```

`--no-build-isolation` makes the build use the `mlx` and `nanobind` you just
installed, rather than fetching fresh copies that might not match.

**The nanobind version matters.** `mx.array` can only pass between `mlx` and
this package if both were built with the same nanobind internals; for MLX 0.32
that is nanobind 2.15.0 (2.13 and 2.14, and 3.x, are not compatible). The
build checks this before compiling and stops with a message naming the
mismatch, so if a later MLX needs a different nanobind, the message says so.

**After upgrading MLX**, reinstall this package so it is rebuilt against the
new version. Importing it with a different MLX than it was built against
raises an `ImportError` that says so.

**With Homebrew's Python and MLX** (`brew install mlx` includes the `mlx`
Python package), use a virtual environment that can see Homebrew's packages:

```bash
python3 -m venv --system-site-packages ~/.venvs/metal-linalg
source ~/.venvs/metal-linalg/bin/activate
pip install scikit-build-core "nanobind==2.15.0"
pip install --no-build-isolation git+https://github.com/c0rmac/metal-linalg.git
```

From a clone of the repository, `pip install --no-build-isolation .` instead.

## Tests

```bash
python -m unittest discover -s python/tests -v
```
