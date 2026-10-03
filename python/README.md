# metal-linalg for Python

QR, symmetric eigendecomposition and SVD for batches of matrices on Apple
GPUs, on [MLX](https://github.com/ml-explore/mlx) arrays. For PyTorch tensors,
install [`metal-linalg-torch`](https://github.com/c0rmac/metal-linalg/blob/main/python-torch/README.md) instead.

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

```bash
pip install metal-linalg
```

On an Apple Silicon Mac with macOS 14 or later and Python 3.10 to 3.14,
this installs a prebuilt wheel and the `mlx` it was built against; nothing is
compiled.

**MLX is pinned to one release.** The package shares MLX's own library and
passes `mx.array` objects through MLX's internals, so it works only with the
MLX it was built against, and asks pip for exactly that one (`mlx==0.32.3`
for this release). Installing it may therefore move your `mlx` to that
version, and upgrading `mlx` past it on its own breaks the import with an
`ImportError` that says so; upgrade both together, `pip install -U
metal-linalg mlx`, once a release of this package for the newer MLX is out.

### From source

On a Mac with no matching wheel, pip builds the package from source, which
needs Xcode's command line tools (`xcode-select --install`) and fetches CMake
and the rest itself. To build a clone, or the main branch:

```bash
pip install .                                                  # in a clone
pip install git+https://github.com/c0rmac/metal-linalg.git     # main branch
```

### Against another MLX

To build against an MLX other than the pinned one, a source build of MLX or
Homebrew's, turn off pip's build isolation so the build sees your MLX, and
`--no-deps` so pip leaves it alone. The build must also use the nanobind your
MLX was built with: the pip MLX 0.32.3 uses nanobind 3.0.1, Homebrew's MLX
0.32.1 uses 2.15.0. The build checks this before compiling and stops with a
message naming the mismatch.

With Homebrew's Python and MLX (`brew install mlx` includes the `mlx` Python
package), in a virtual environment that can see Homebrew's packages:

```bash
python3 -m venv --system-site-packages ~/.venvs/metal-linalg
source ~/.venvs/metal-linalg/bin/activate
pip install scikit-build-core "nanobind==2.15.0"
pip install --no-build-isolation --no-deps git+https://github.com/c0rmac/metal-linalg.git
```

After upgrading that MLX, reinstall the package the same way.

## Calibration

Routing comes from measurements made on each kind of Mac. Where there are
none for yours, or they are stale or incomplete, importing the package raises
a `metal_linalg.CalibrationWarning` once per decomposition, saying so and how
to measure your Mac (about an hour and a half) and submit the results:

```python
ml.calibration_status()     # {'qr': 'current', 'eigh': 'current', 'svd': 'stale'}
```

`warnings.filterwarnings("ignore", category=ml.CalibrationWarning)`, or
`METAL_LINALG_NO_CALIBRATION_NOTICE=1`, silences it. Which chips are measured:
[the measurements page](https://c0rmac.github.io/metal-linalg/docs/measurements).

## Tests

```bash
python -m unittest discover -s python/tests -v
```
