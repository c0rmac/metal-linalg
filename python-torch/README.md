# metal-linalg for PyTorch

QR, symmetric eigendecomposition and SVD for batches of matrices on Apple
GPUs, on [PyTorch](https://pytorch.org) tensors.

```python
import torch
import metal_linalg_torch as mlt

a = torch.randn(1000, 64, 32, device="mps")   # a batch of 1000 matrices; "cpu" works too

Q, R = mlt.qr(a)                     # Q (1000, 64, 32), R (1000, 32, 32)
U, S, Vh = mlt.svd(a)                # thin: U (1000, 64, 32), S (1000, 32), Vh (1000, 32, 32)
L, V = mlt.eigh(a.mT @ a)            # L ascending (1000, 32), V (1000, 32, 32)
```

The functions take the arguments of their `torch.linalg` namesakes and return
the same result types (`Q, R = ...`, `result.eigenvalues`, ...), on the
device of their input. Each call is routed to the fastest Metal kernel for
its shape and batch, or to LAPACK on the CPU, by a policy measured on the
Mac it runs on:

```python
mlt.device_name()                      # 'Apple M5 Pro'
mlt.eigh_policy_source()               # 'tuned:Apple M5 Pro'
mlt.eigh_backend(32, 4096)             # 'ql': 4096 matrices of 32x32 go to the GPU
mlt.svd_backend(4096, 4096)            # 'bidiag': GPU bidiagonalization, LAPACK's solver
mlt.svdvals_backend(4096, 4096)        # 'band': the two-stage reduction, values alone
mlt.set_eigh_policy(gpu_min_batch=1)   # override the measured policy
```

The kernels, the routing and the measurements behind them are described in
the [main README](https://github.com/c0rmac/metal-linalg).

## Installing

```bash
pip install metal-linalg-torch
```

On an Apple Silicon Mac with macOS 14 or later, Python 3.10 or later and
PyTorch 2.4 or later. This is the PyTorch package; for
[MLX](https://github.com/ml-explore/mlx) arrays install `metal-linalg`
instead. The two are independent: this one does not install or load MLX.

One wheel serves every PyTorch and every Python: the package calls the
library's C API (on plain float buffers) and is not compiled against
either, so upgrading `torch` never breaks it.

To build from source instead (needs Xcode's command line tools):

```bash
pip install ./python-torch                                                      # in a clone
pip install "git+https://github.com/c0rmac/metal-linalg.git#subdirectory=python-torch"
```

## Functions

| function | like | returns |
|---|---|---|
| `qr(A, mode="reduced")` | `torch.linalg.qr` | `(Q, R)`, thin; `mode="r"` gives an empty `Q` |
| `eigh(A, UPLO="L")` | `torch.linalg.eigh` | `(eigenvalues, eigenvectors)`, ascending |
| `eigvalsh(A, UPLO="L")` | `torch.linalg.eigvalsh` | eigenvalues, ascending; less work than `eigh` |
| `svd(A, full_matrices=False)` | `torch.linalg.svd` | `(U, S, Vh)`, thin, `S` descending |
| `svdvals(A)` | `torch.linalg.svdvals` | singular values, descending; about half the work of `svd` |

`A` is `[..., M, N]` with any number of batch dimensions, on `"cpu"` or
`"mps"`. What differs from `torch.linalg`:

- **float32.** The kernels compute in float32. Other real dtypes (float64,
  float16, bfloat16, integers) are converted, and results are float32;
  complex input raises `TypeError`.
- **Thin factors only.** `svd` defaults to `full_matrices=False` (torch's
  default is `True`), and `full_matrices=True` or `qr(mode="complete")` on a
  non-square matrix raises `NotImplementedError`.
- **No exceptions for bad matrices.** A matrix with a NaN or an infinity
  gives NaN results for that matrix alone, leaving the rest of its batch
  intact, where torch raises.

## Autograd and torch.compile

Underneath, the functions are custom operators,
`torch.ops.metal_linalg.{qr, eigh, eigvalsh, svd, svdvals}`, with fake
implementations and the backward formulas `torch.linalg` uses:

```python
A = torch.randn(64, 32, 32, device="mps", requires_grad=True)
S = mlt.svdvals(A)
S.sum().backward()                       # A.grad: the gradient of the nuclear norm

f = torch.compile(lambda x: mlt.eigh(x @ x.mT).eigenvalues.sum())
f(torch.randn(8, 16, 16))
```

As in torch, the gradients of `eigh` and `svd` are defined only for distinct
eigenvalues or singular values (they divide by their differences), and are
not unique for a loss that depends on the signs of the vectors. Second
derivatives are not supported. `eigvalsh` and `svdvals` compute the vectors
as well when their input requires grad, since the gradient needs them.

## Performance

Against `torch.linalg` on an M5 Pro with PyTorch 2.13 (conda-forge's, its CPU
LAPACK from Accelerate) and metal-linalg 2.14 (best of five,
[`benchmarks/benchmark_torch.py`](https://github.com/c0rmac/metal-linalg/blob/main/benchmarks/benchmark_torch.py);
the same tensors on MPS for torch's MPS path and for this package, which uses
them in place, see [MPS tensors](#mps-tensors)):

| | torch, CPU | torch, MPS | metal-linalg-torch |
|---|---|---|---|
| QR, 1024 × 128×128 | 201 ms | 34 ms | 15 ms |
| SVD, 256 × 128×64 | 68 ms | 71 ms | 5.5 ms |
| SVD, 4096 × 32×32 | 224 ms | 227 ms | 7.4 ms |
| eigh, 4096 × 16×16 | 32 ms | 34 ms | 2.1 ms |
| eigh, one 2048×2048 | 248 ms | 261 ms | 89 ms |
| SVD, one 4096×4096 | 3.56 s | 3.58 s | 1.59 s |
| eigvalsh, one 4096×4096 | 1.78 s | 1.79 s | 157 ms |
| svdvals, one 4096×4096 | 2.00 s | 2.00 s | 229 ms |

It is ahead on every row. Of these calls PyTorch 2.13 runs only QR on the GPU
for MPS tensors, and this is 2.3x faster there; its SVD takes as long on MPS
as on the CPU, and eigh, eigvalsh and svdvals have no MPS kernels and go
through its CPU fallback. Against those, 12-31x for the other batches of
small matrices (the SVD of 256 matrices of 128×64 runs on the library's CPU
path, which spreads a batch over every core; the two batches of 4096, and the
QR batch, run on the GPU and the CPU at once), 2.2-2.9x for one large matrix
with its vectors, and 9-11x for its eigenvalues or singular values alone, by
a two-stage reduction. Which
backend a shape gets on your Mac: `mlt.svd_backend(m, n, batch)` and its
siblings.

## MPS tensors

An MPS tensor is used in place. On Apple Silicon PyTorch keeps MPS tensors in
Metal buffers in shared storage, which the CPU can address too: the library
reads its input there and writes its results into new MPS tensors, with no
copy to the CPU and back. Its kernels run on a Metal command queue of its
own, so a call first waits for the work PyTorch has queued on the GPU
(`torch.mps.synchronize()`), which may still be writing the input, and
returns when its results are written. `mlt.mps_in_place()` says whether this
applies; where it does not (a torch that keeps MPS tensors in private
storage), or with `METAL_LINALG_TORCH_MPS_COPY=1`, an MPS tensor is copied to
the CPU and the results back.

Up to 2.13 every MPS call made those copies. They cost most where the
decomposition costs least: 9-12x for a handful of matrices, 1.2-1.8x for
large batches, a few percent for one large matrix (M5 Pro, PyTorch 2.13, best
of five with the two alternating, `benchmarks/benchmark_torch.py --mps-ab`):

| | copied (2.13) | in place |
|---|---|---|
| eigh, 16 × 16×16 | 1.2 ms | 0.10 ms |
| QR, 16 × 48×16 | 0.69 ms | 0.08 ms |
| SVD, 1024 × 32×32 | 4.4 ms | 3.0 ms |
| QR, 16384 × 64×32 | 25 ms | 14 ms |
| eigh, 16384 × 32×32 | 22 ms | 17 ms |
| eigh, one 2048×2048 | 93 ms | 90 ms |
| svdvals, one 4096×4096 | 231 ms | 227 ms |

A CPU tensor is used in place too, and its results stay on the CPU, though
the work still runs on the GPU where that is faster.

Calls are thread-safe; they are serialised, and release the GIL.

## Calibration

Each Mac's routing comes from measurements of that Mac. On one that has
none (or older ones), the import raises a `CalibrationWarning` once per
decomposition, and `mlt.calibration_status()` says where each stands. The
library works either way, with a cautious default; measuring the Mac takes
one command and improves it for everyone with that Mac: see
[which Macs are measured](https://c0rmac.github.io/metal-linalg/docs/measurements)
and [how to contribute](https://github.com/c0rmac/metal-linalg/blob/main/CONTRIBUTING.md).
`METAL_LINALG_NO_CALIBRATION_NOTICE=1` silences the warning.
