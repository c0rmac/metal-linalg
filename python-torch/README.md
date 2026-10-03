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
mlt.eigh_backend(512, 64)              # 'block': 64 matrices of 512x512 go to the GPU
mlt.svd_backend(4096, 4096)            # 'bidiag': GPU bidiagonalization, LAPACK's solver
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

Against `torch.linalg` on an M5 Pro with PyTorch 2.14 (best of five; the
same tensors on MPS for torch's MPS path and for this package, copies
included):

| | torch, CPU | torch, MPS | metal-linalg-torch |
|---|---|---|---|
| QR, 1024 × 128×128 | 160 ms | 1040 ms | 23 ms |
| SVD, 256 × 128×64 | 61 ms | 12 ms | 10 ms |
| SVD, 4096 × 32×32 | 196 ms | 27 ms | 27 ms |
| eigh, 4096 × 16×16 | 28 ms | 5.6 ms | 3.9 ms |
| eigh, one 2048×2048 | 259 ms | 244 ms | 121 ms |
| SVD, one 4096×4096 | 3.58 s | 3.59 s | 1.79 s |

PyTorch's own MPS kernels are close for small matrices in batches; the gains
are in QR and in large matrices, where torch falls back to the CPU. Which
backend a shape gets on your Mac: `mlt.svd_backend(m, n, batch)` and its
siblings.

## MPS tensors

The library runs its own Metal kernels on host memory. An MPS tensor is
copied to the CPU (which waits for the work PyTorch has queued on the GPU)
and the results are copied back: two copies at memory bandwidth around a
decomposition. A CPU tensor is used in place, and its results stay on the
CPU, though the work still runs on the GPU where that is faster.

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
