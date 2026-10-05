"""QR, symmetric eigendecomposition and SVD for batches of matrices on Apple
GPUs, for PyTorch.

    import torch
    import metal_linalg_torch as mlt

    Q, R = mlt.qr(a)                 # a: [..., M, N], on "cpu" or "mps"
    L, V = mlt.eigh(s)               # s symmetric [..., N, N]; L ascending
    U, S, Vh = mlt.svd(a)            # thin factors; S descending

The functions take the arguments of their torch.linalg namesakes and return
the same result types, on the input's device. Each call is routed to the
fastest Metal kernel for its shape and batch, or to LAPACK on the CPU, by a
policy measured on the Mac it runs on. They support autograd (with
torch.linalg's formulas) and torch.compile: underneath they are the custom
operators torch.ops.metal_linalg.*. Computation is in float32: other real
dtypes are converted, results are float32. See
https://github.com/c0rmac/metal-linalg/blob/main/python-torch/README.md.
"""

from collections import namedtuple

import torch

from . import _lib, _ops  # noqa: F401  (_ops registers torch.ops.metal_linalg.*)
from ._build import version as __version__

__all__ = [
    "qr", "eigh", "eigvalsh", "svd", "svdvals",
    "device_name", "gpu_core_count", "cpu_threads", "set_cpu_threads", "mps_in_place",
    "qr_backend", "eigh_backend", "eigvalsh_backend", "svd_backend", "svdvals_backend",
    "qr_policy", "eigh_policy", "svd_policy",
    "set_qr_policy", "set_eigh_policy", "set_svd_policy",
    "qr_policy_source", "eigh_policy_source", "svd_policy_source",
    "calibration_status", "CalibrationWarning",
]

_DEVICES = ("cpu", "mps", "meta")


def _prepare(A, name):
    """A as a float32 tensor [..., M, N] on a device the library serves."""
    if not isinstance(A, torch.Tensor):
        A = torch.as_tensor(A)
    if A.is_complex():
        raise TypeError(f"metal_linalg_torch.{name}: complex input is not supported (got {A.dtype}); "
                        f"the decompositions are real, in float32")
    if A.dim() < 2:
        raise ValueError(f"metal_linalg_torch.{name}: expected a tensor of shape [..., M, N], "
                         f"got shape {tuple(A.shape)}")
    if A.device.type not in _DEVICES:
        raise ValueError(f"metal_linalg_torch.{name}: tensors on {A.device} are not supported; "
                         f"use the CPU or MPS")
    return A if A.dtype == torch.float32 else A.to(torch.float32)


def _uplo(UPLO, name):
    if UPLO not in ("L", "U"):
        raise ValueError(f"metal_linalg_torch.{name}: UPLO must be 'L' or 'U', got {UPLO!r}")
    return UPLO == "L"


def _needs_grad(a):
    return torch.is_grad_enabled() and a.requires_grad


# torch.linalg's result types, for eager calls. Inside torch.compile, plain
# named tuples with the same fields, which every torch from 2.4 traces (2.4
# cannot trace the construction of a torch.return_types).
_TRACED = {"linalg_qr": namedtuple("linalg_qr", ["Q", "R"]),
           "linalg_eigh": namedtuple("linalg_eigh", ["eigenvalues", "eigenvectors"]),
           "linalg_svd": namedtuple("linalg_svd", ["U", "S", "Vh"])}


def _result(kind, values):
    if torch.compiler.is_compiling():
        return _TRACED[kind](*values)
    return getattr(torch.return_types, kind)(values)


# ---------------------------------------------------------------------------
# Decompositions
# ---------------------------------------------------------------------------

def qr(A, mode="reduced"):
    """Thin QR of a batch of matrices, like ``torch.linalg.qr``: ``A = Q @ R``.

    ``A`` is ``[..., M, N]``. Returns ``(Q, R)``: ``Q`` ``[..., M, K]`` with
    orthonormal columns and ``R`` ``[..., K, N]`` upper triangular,
    ``K = min(M, N)``. ``mode`` is ``"reduced"`` (the default) or ``"r"``
    (``Q`` is then an empty tensor, as in torch); ``"complete"`` is the same
    as ``"reduced"`` when ``M <= N`` and is not supported otherwise.
    """
    a = _prepare(A, "qr")
    if mode not in ("reduced", "r", "complete"):
        raise ValueError(f"metal_linalg_torch.qr: mode must be 'reduced', 'r' or 'complete', got {mode!r}")
    if mode == "complete" and a.shape[-2] > a.shape[-1]:
        raise NotImplementedError("metal_linalg_torch.qr: mode='complete' for M > N (a square Q) is not "
                                  "supported; the library computes the thin factors. Use "
                                  "torch.linalg.qr for it")
    Q, R = torch.ops.metal_linalg.qr(a)
    if mode == "r":
        Q = Q.new_empty(0)
    return _result("linalg_qr", (Q, R))


def eigh(A, UPLO="L"):
    """Eigendecomposition of a batch of symmetric matrices, like
    ``torch.linalg.eigh``: ``A = V @ diag(L) @ V.mT``.

    ``A`` is ``[..., N, N]``; only the triangle named by ``UPLO`` (``"L"``
    or ``"U"``) is read. Returns ``(eigenvalues, eigenvectors)``:
    eigenvalues ``[..., N]`` ascending, eigenvectors ``[..., N, N]`` as
    columns. A non-finite matrix yields NaN rather than an error.
    """
    a = _prepare(A, "eigh")
    w, V = torch.ops.metal_linalg.eigh(a, _uplo(UPLO, "eigh"))
    return _result("linalg_eigh", (w, V))


def eigvalsh(A, UPLO="L"):
    """Eigenvalues only of a batch of symmetric matrices, ascending, like
    ``torch.linalg.eigvalsh``; less work than :func:`eigh`. If ``A``
    requires grad, the eigenvectors are computed too, as the gradient needs
    them."""
    a = _prepare(A, "eigvalsh")
    lower = _uplo(UPLO, "eigvalsh")
    if _needs_grad(a):
        return torch.ops.metal_linalg.eigh(a, lower)[0]
    return torch.ops.metal_linalg.eigvalsh(a, lower)


def svd(A, full_matrices=False):
    """Thin SVD of a batch of matrices: ``A = U @ diag(S) @ Vh``.

    ``A`` is ``[..., M, N]``. Returns ``(U, S, Vh)``: ``U`` ``[..., M, K]``,
    ``S`` ``[..., K]`` descending and ``Vh`` ``[..., K, N]``,
    ``K = min(M, N)``. Unlike ``torch.linalg.svd``, whose default is
    ``full_matrices=True``, the factors are the thin ones: pass
    ``full_matrices=True`` only for square matrices, where it is the same.
    A non-finite matrix yields NaN rather than an error.
    """
    a = _prepare(A, "svd")
    if full_matrices and a.shape[-2] != a.shape[-1]:
        raise NotImplementedError("metal_linalg_torch.svd: full_matrices=True for a non-square matrix is "
                                  "not supported; the library computes the thin factors "
                                  "(full_matrices=False). Use torch.linalg.svd for the full ones")
    U, S, Vh = torch.ops.metal_linalg.svd(a)
    return _result("linalg_svd", (U, S, Vh))


def svdvals(A):
    """Singular values only, descending, like ``torch.linalg.svdvals``;
    about half the work of :func:`svd`. If ``A`` requires grad, the singular
    vectors are computed too, as the gradient needs them."""
    a = _prepare(A, "svdvals")
    if _needs_grad(a):
        return torch.ops.metal_linalg.svd(a)[1]
    return torch.ops.metal_linalg.svdvals(a)


# ---------------------------------------------------------------------------
# The device and its routing
# ---------------------------------------------------------------------------

def device_name():
    """The GPU the policies were resolved for, e.g. ``"Apple M5 Pro"``."""
    return _lib.text(_lib.device_name())


def gpu_core_count():
    """Its GPU core count; 0 if it could not be read."""
    return int(_lib.gpu_core_count())


def cpu_threads():
    """How many CPU threads the CPU paths spread a batch over: every core by
    default. A lone matrix keeps Accelerate's own threading."""
    return int(_lib.cpu_threads())


def set_cpu_threads(n):
    """Caps :func:`cpu_threads`, for a program that runs several solves at once
    on threads of its own; 0 restores every core. ``METAL_LINALG_CPU_THREADS``
    sets it from the environment."""
    _lib.set_cpu_threads(int(n))


def mps_in_place():
    """Whether MPS tensors are used in place: the library reads their memory
    and writes its results into MPS tensors, with no copy to the CPU and back.
    True where torch keeps MPS tensors in shared memory, as on Apple Silicon;
    ``METAL_LINALG_TORCH_MPS_COPY=1`` forces the copies."""
    return _ops.mps_in_place()


def qr_backend(m, n, batch=1):
    """Which backend :func:`qr` uses for ``batch`` matrices of ``m x n``:
    ``"cpu"``, ``"unblocked"`` or ``"streaming_reduced"``."""
    return _lib.text(_lib.qr_backend(m, n, batch))


def eigh_backend(n, batch=1):
    """Which backend :func:`eigh` uses: ``"cpu"``, ``"simd"``,
    ``"threadgroup"``, ``"block"``, ``"tridiag"`` or ``"ql"``."""
    return _lib.text(_lib.eigh_backend(n, batch))


def eigvalsh_backend(n, batch=1):
    """Which backend :func:`eigvalsh` uses, under the policy's
    eigenvalues-alone boundary: as :func:`eigh_backend`, or ``"band"`` (the
    two-stage reduction) for large N."""
    return _lib.text(_lib.eigvalsh_backend(n, batch))


def svd_backend(m, n, batch=1):
    """Which backend :func:`svd` uses: ``"cpu"``, ``"jacobi"``,
    ``"block_jacobi"``, ``"qr_jacobi"``, ``"qr_block_jacobi"``, ``"bidiag"``,
    ``"golub_kahan"`` or ``"qr_golub_kahan"``."""
    return _lib.text(_lib.svd_backend(m, n, batch))


def svdvals_backend(m, n, batch=1):
    """Which backend :func:`svdvals` uses: as :func:`svd_backend`, or
    ``"band"`` (the two-stage reduction, singular values alone)."""
    return _lib.text(_lib.svdvals_backend(m, n, batch))


def qr_policy():
    """The QR routing policy in effect, as a dict of its fields."""
    return _lib.get_policy("qr")


def eigh_policy():
    """The eigensolver routing policy in effect, as a dict of its fields."""
    return _lib.get_policy("eigh")


def svd_policy():
    """The SVD routing policy in effect, as a dict of its fields."""
    return _lib.get_policy("svd")


def set_qr_policy(policy=None, **fields):
    """Replaces the QR policy. Pass a dict from :func:`qr_policy`, or just the
    fields to change: ``set_qr_policy(gpu_min_batch=1)``."""
    _lib.set_policy("qr", {**(policy or {}), **fields})


def set_eigh_policy(policy=None, **fields):
    """Replaces the eigensolver policy, e.g. ``set_eigh_policy(gpu_min_batch=1)``."""
    _lib.set_policy("eigh", {**(policy or {}), **fields})


def set_svd_policy(policy=None, **fields):
    """Replaces the SVD policy, e.g. ``set_svd_policy(bidiag_min_k=1024)``."""
    _lib.set_policy("svd", {**(policy or {}), **fields})


def qr_policy_source():
    """Where the QR policy came from: ``"tuned:<device>"``, ``"estimated:<device>
    (from <measured device>, ...)"`` on a Mac nobody has measured,
    ``"default:untuned-device (<device>)"``, ``"env:..."`` or ``"user"``."""
    return _lib.text(_lib.qr_policy_source())


def eigh_policy_source():
    """Where the eigensolver policy came from; see :func:`qr_policy_source`."""
    return _lib.text(_lib.eigh_policy_source())


def svd_policy_source():
    """Where the SVD policy came from; see :func:`qr_policy_source`."""
    return _lib.text(_lib.svd_policy_source())


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

class CalibrationWarning(UserWarning):
    """Raised once per decomposition at import when this Mac's calibration
    is missing, stale or incomplete: the library still works, but measuring
    this Mac and submitting the results would make it faster (see
    :func:`calibration_status`). Silence it with
    ``warnings.filterwarnings("ignore", category=metal_linalg_torch.CalibrationWarning)``
    or ``METAL_LINALG_NO_CALIBRATION_NOTICE=1``."""


def calibration_status():
    """How current this Mac's measurements are, per decomposition:
    ``{"qr": state, "eigh": state, "svd": state}``, each ``"current"``,
    ``"stale"`` (measured on older kernels, still used), ``"incomplete"``
    (from before a newer backend, which stays off) or ``"uncalibrated"``
    (not measured: settings estimated from a measured Mac). See https://c0rmac.github.io/metal-linalg/docs/measurements."""
    out = {}
    for key, source in (("qr", qr_policy_source), ("eigh", eigh_policy_source),
                        ("svd", svd_policy_source)):
        s = source()
        out[key] = ("uncalibrated" if s.startswith(("default:", "estimated:"))
                    else "stale" if s.startswith("tuned-stale:")
                    else "incomplete" if s.startswith("tuned-incomplete:") else "current")
    return out


def _calibration_warnings():
    import os
    import warnings
    _lib.set_calibration_notices(0)   # this package warns instead of printing
    flag = os.environ.get("METAL_LINALG_NO_CALIBRATION_NOTICE", "")
    if flag and flag != "0":
        return
    for source in (qr_policy_source, eigh_policy_source, svd_policy_source):
        source()   # resolves the policy, which records its calibration
    for what in ("QR", "eigh", "SVD"):
        msg = _lib.text(_lib.calibration_message(what.encode()))
        if msg:
            warnings.warn(msg, CalibrationWarning, stacklevel=3)


_calibration_warnings()
