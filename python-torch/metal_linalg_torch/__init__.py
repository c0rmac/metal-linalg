"""QR, symmetric eigendecomposition, SVD, Cholesky and LU (with solve and
inverse) for batches of matrices on Apple GPUs, for PyTorch.

    import torch
    import metal_linalg_torch as mlt

    Q, R = mlt.qr(a)                 # a: [..., M, N], on "cpu" or "mps"
    L, V = mlt.eigh(s)               # s symmetric [..., N, N]; L ascending
    U, S, Vh = mlt.svd(a)            # thin factors; S descending
    L = mlt.cholesky(p)              # p symmetric positive definite; p = L @ L.mT
    X = mlt.solve(A, B)              # A [..., N, N], B [..., N] or [..., N, K]

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
    "qr", "eigh", "eigvalsh", "svd", "svdvals", "cholesky", "cholesky_ex",
    "lu_factor", "lu_factor_ex", "solve", "solve_ex", "inv", "inv_ex", "solve_triangular",
    "device_name", "gpu_core_count", "cpu_threads", "set_cpu_threads", "cpu_only", "set_cpu_only", "CpuOnly",
    "mps_in_place",
    "qr_backend", "eigh_backend", "eigvalsh_backend", "svd_backend", "svdvals_backend", "cholesky_backend", "lu_backend", "trsm_backend",
    "qr_policy", "eigh_policy", "svd_policy", "cholesky_policy", "lu_policy", "trsm_policy",
    "set_qr_policy", "set_eigh_policy", "set_svd_policy", "set_cholesky_policy", "set_lu_policy", "set_trsm_policy",
    "qr_policy_source", "eigh_policy_source", "svd_policy_source", "cholesky_policy_source", "lu_policy_source", "trsm_policy_source",
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
           "linalg_svd": namedtuple("linalg_svd", ["U", "S", "Vh"]),
           "linalg_cholesky_ex": namedtuple("linalg_cholesky_ex", ["L", "info"]),
           "linalg_lu_factor": namedtuple("linalg_lu_factor", ["LU", "pivots"]),
           "linalg_lu_factor_ex": namedtuple("linalg_lu_factor_ex", ["LU", "pivots", "info"]),
           "linalg_solve_ex": namedtuple("linalg_solve_ex", ["result", "info"]),
           "linalg_inv_ex": namedtuple("linalg_inv_ex", ["inverse", "info"])}


def _result(kind, values):
    if torch.compiler.is_compiling():
        return _TRACED[kind](*values)
    return getattr(torch.return_types, kind)(values)


# ---------------------------------------------------------------------------
# Decompositions
# ---------------------------------------------------------------------------

def qr(A, mode="reduced"):
    """QR of a batch of matrices, like ``torch.linalg.qr``: ``A = Q @ R``.

    ``A`` is ``[..., M, N]``. Returns ``(Q, R)``: ``Q`` ``[..., M, K]`` with
    orthonormal columns and ``R`` ``[..., K, N]`` upper triangular,
    ``K = min(M, N)``. ``mode`` as torch's: ``"reduced"`` (the default),
    ``"r"`` (``R`` alone, ``Q`` an empty tensor and never formed:
    up to 2.8x faster) or ``"complete"`` (``Q`` ``[..., M, M]`` square, ``R``
    ``[..., M, N]`` with zero rows below ``K``).

    Unlike torch's, ``mode="r"`` is differentiable: when ``A`` requires
    grad, ``Q`` is computed for the gradient and dropped. As torch's,
    ``mode="complete"`` is not differentiable when ``M > N``.
    """
    a = _prepare(A, "qr")
    if mode not in ("reduced", "r", "complete"):
        raise ValueError(f"metal_linalg_torch.qr: mode must be 'reduced', 'r' or 'complete', got {mode!r}")
    if mode == "r" and _needs_grad(a):
        _, R = torch.ops.metal_linalg.qr(a)
        return _result("linalg_qr", (R.new_empty(0), R))
    Q, R = torch.ops.metal_linalg.qr(a, mode)
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


def _not_positive_definite(info):
    """torch.linalg.cholesky's error for the first matrix that failed."""
    bad = torch.nonzero(info.reshape(-1).cpu())
    if bad.numel() == 0:
        return None
    i = int(bad[0, 0])
    k = int(info.reshape(-1)[i])
    where = f"(Batch element {i}): " if info.dim() else ""
    return getattr(torch.linalg, "LinAlgError", RuntimeError)(
        f"linalg.cholesky: {where}The factorization could not be completed because the input is not "
        f"positive-definite (the leading minor of order {k} is not positive-definite).")


def cholesky(A, upper=False):
    """Cholesky factorization of a batch of symmetric positive definite
    matrices, like ``torch.linalg.cholesky``: ``A = L @ L.mT``.

    ``A`` is ``[..., N, N]``; only its lower triangle is read (the upper one
    with ``upper=True``). Returns ``L`` ``[..., N, N]``, lower triangular
    with a positive diagonal and zeros above it, or ``U = L.mT`` with
    ``upper=True``. Raises ``torch.linalg.LinAlgError``, as torch does, if a
    matrix is not positive definite (or holds a NaN or infinity where it is
    read); :func:`cholesky_ex` reports it instead.
    """
    a = _prepare(A, "cholesky")
    L, info = torch.ops.metal_linalg.cholesky(a, bool(upper))
    if not torch.compiler.is_compiling() and a.device.type != "meta":
        err = _not_positive_definite(info)
        if err is not None:
            raise err
    return L


def cholesky_ex(A, upper=False, check_errors=False):
    """:func:`cholesky` and ``info``, like ``torch.linalg.cholesky_ex``:
    ``(L, info)``, ``info`` ``[...]`` int32, 0 for a matrix factored, else
    ``k`` where its leading minor of order ``k`` is not positive definite.
    That matrix's ``L`` is all NaN (torch's holds a partial factor).
    ``check_errors=True`` raises as :func:`cholesky` does."""
    a = _prepare(A, "cholesky_ex")
    L, info = torch.ops.metal_linalg.cholesky(a, bool(upper))
    if check_errors and not torch.compiler.is_compiling() and a.device.type != "meta":
        err = _not_positive_definite(info)
        if err is not None:
            raise err
    return _result("linalg_cholesky_ex", (L, info))


def _singular(info, what):
    """torch's error for the first singular matrix of a batch, or None."""
    if torch.compiler.is_compiling() or info.device.type == "meta":
        return None
    bad = torch.nonzero(info.reshape(-1).cpu())
    if bad.numel() == 0:
        return None
    i = int(bad[0, 0])
    where = f"(Batch element {i}): " if info.dim() else ""
    msg = {"solve": "The solver failed because the input matrix is singular.",
           "inv": f"The diagonal element {int(info.reshape(-1)[i])} is zero, the inversion could not be completed "
                  f"because the input matrix is singular."}[what]
    return getattr(torch.linalg, "LinAlgError", RuntimeError)(f"linalg.{what}: {where}{msg}")


def _no_pivot(pivot, name):
    if not pivot:
        raise NotImplementedError(f"metal_linalg_torch.{name}: only pivot=True (partial pivoting) is supported")


def lu_factor(A, *, pivot=True):
    """LU factorization with partial pivoting, like ``torch.linalg.lu_factor``:
    ``(LU, pivots)``, ``LU`` ``[..., N, N]`` (``U`` on and above the diagonal,
    ``L`` below it with its unit diagonal implied) and ``pivots`` ``[..., N]``
    int32, 1-based as LAPACK's and torch's. Square matrices only. Not
    differentiable (torch's is)."""
    a = _prepare(A, "lu_factor")
    _no_pivot(pivot, "lu_factor")
    LU, piv, _ = torch.ops.metal_linalg.lu_factor(a)
    return _result("linalg_lu_factor", (LU, piv))


def lu_factor_ex(A, *, pivot=True, check_errors=False):
    """:func:`lu_factor` and ``info``, like ``torch.linalg.lu_factor_ex``: 0,
    or ``k`` where ``U``'s k-th diagonal entry is exactly zero."""
    a = _prepare(A, "lu_factor_ex")
    _no_pivot(pivot, "lu_factor_ex")
    LU, piv, info = torch.ops.metal_linalg.lu_factor(a)
    if check_errors and bool((info != 0).any()):
        raise getattr(torch.linalg, "LinAlgError", RuntimeError)(
            "linalg.lu_factor_ex: U is exactly singular (info " + str(info.reshape(-1).tolist()) + ")")
    return _result("linalg_lu_factor_ex", (LU, piv, info))


def _solve(A, B, left, name):
    a = _prepare(A, name)
    b = B if isinstance(B, torch.Tensor) else torch.as_tensor(B)
    if not left:
        raise NotImplementedError(f"metal_linalg_torch.{name}: only left=True is supported")
    vector = b.dim() == a.dim() - 1
    bm = (b.unsqueeze(-1) if vector else b).to(torch.float32)
    X, info = torch.ops.metal_linalg.solve(a, bm)
    return (X.squeeze(-1) if vector else X), info


def solve(A, B, *, left=True):
    """``X`` with ``A @ X = B``, like ``torch.linalg.solve``: ``A``
    ``[..., N, N]``, ``B`` ``[..., N, K]`` or ``[..., N]`` (when it has one
    dimension fewer than ``A``), the batch shapes equal (no broadcasting).
    Raises ``torch.linalg.LinAlgError`` for a singular matrix, as torch does.
    Differentiable in ``A`` and ``B``."""
    X, info = _solve(A, B, left, "solve")
    err = _singular(info, "solve")
    if err is not None:
        raise err
    return X


def solve_ex(A, B, *, left=True, check_errors=False):
    """:func:`solve` and ``info``, like ``torch.linalg.solve_ex``; a singular
    matrix's result is all NaN."""
    X, info = _solve(A, B, left, "solve_ex")
    if check_errors:
        err = _singular(info, "solve")
        if err is not None:
            raise err
    return _result("linalg_solve_ex", (X, info))


def inv(A):
    """The inverse of a batch of square matrices, like ``torch.linalg.inv``.
    Raises ``torch.linalg.LinAlgError`` for a singular matrix. Differentiable."""
    a = _prepare(A, "inv")
    X, info = torch.ops.metal_linalg.inv(a)
    err = _singular(info, "inv")
    if err is not None:
        raise err
    return X


def solve_triangular(A, B, *, upper, left=True, unitriangular=False):
    """``X`` with ``A @ X = B`` for triangular ``A``, like
    ``torch.linalg.solve_triangular``: ``A`` ``[..., N, N]`` (only the
    triangle ``upper`` names is read; with ``unitriangular`` its diagonal is
    taken as ones), ``B`` ``[..., N, K]`` with ``A``'s batch shape (no
    broadcasting). ``left=False`` is not supported. Differentiable in ``A``
    and ``B``."""
    a = _prepare(A, "solve_triangular")
    if not left:
        raise NotImplementedError("metal_linalg_torch.solve_triangular: only left=True is supported")
    b = B if isinstance(B, torch.Tensor) else torch.as_tensor(B)
    return torch.ops.metal_linalg.solve_triangular(a, b.to(torch.float32), bool(upper), bool(unitriangular))


def inv_ex(A, *, check_errors=False):
    """:func:`inv` and ``info``, like ``torch.linalg.inv_ex``; a singular
    matrix's inverse is all NaN."""
    a = _prepare(A, "inv_ex")
    X, info = torch.ops.metal_linalg.inv(a)
    if check_errors:
        err = _singular(info, "inv")
        if err is not None:
            raise err
    return _result("linalg_inv_ex", (X, info))


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


def cpu_only():
    """Whether this thread's calls are kept on the CPU (:func:`set_cpu_only`)."""
    return bool(_lib.cpu_only())


def set_cpu_only(on):
    """CPU only, for the calling thread: while on, its calls take their CPU
    paths and never the GPU, whatever the routing policies and the
    ``*_DEVICE`` environment variables say, and the ``*_backend`` queries
    answer the same; an MPS tensor's results still come back on MPS. For work
    meant to stay on the CPU. Other threads are not affected (nor is autograd's
    backward pass where torch runs it on a thread of its own); off by default.
    :class:`CpuOnly` turns it on for a ``with`` block."""
    _lib.set_cpu_only(1 if on else 0)


class CpuOnly:
    """``with CpuOnly(): ...`` keeps the block's calls on the CPU
    (:func:`set_cpu_only`), then restores the setting before it."""

    def __init__(self, on=True):
        self._on = bool(on)
        self._previous = None

    def __enter__(self):
        self._previous = cpu_only()
        set_cpu_only(self._on)
        return self

    def __exit__(self, *exc):
        set_cpu_only(self._previous)
        return False


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
    ``"threadgroup"``, ``"block"``, ``"tridiag"``, ``"ql"``, ``"band"`` (the
    two-stage reduction, from the policy's ``band_min_n``) or
    ``"tridiag_batch"`` (a batch of mid-size matrices at once, inside the
    policy's ``tridiag_batch_*`` window)."""
    return _lib.text(_lib.eigh_backend(n, batch))


def eigvalsh_backend(n, batch=1):
    """Which backend :func:`eigvalsh` uses, under the policy's
    eigenvalues-alone boundary: as :func:`eigh_backend`, or ``"band"`` (the
    two-stage reduction) for large N."""
    return _lib.text(_lib.eigvalsh_backend(n, batch))


def svd_backend(m, n, batch=1):
    """Which backend :func:`svd` uses: ``"cpu"``, ``"jacobi"``,
    ``"block_jacobi"``, ``"qr_jacobi"``, ``"qr_block_jacobi"``, ``"bidiag"``,
    ``"band"`` (the two-stage reduction, from the policy's ``band_min_k``),
    ``"golub_kahan"``, ``"qr_golub_kahan"`` or ``"bidiag_batch"`` (a batch of
    mid-size matrices at once, inside the policy's ``bidiag_batch_*``
    window)."""
    return _lib.text(_lib.svd_backend(m, n, batch))


def svdvals_backend(m, n, batch=1):
    """Which backend :func:`svdvals` uses: as :func:`svd_backend`, with
    ``"band"`` from the policy's ``values_band_min_k``."""
    return _lib.text(_lib.svdvals_backend(m, n, batch))


def cholesky_backend(n, batch=1):
    """Which backend :func:`cholesky` uses for ``batch`` matrices of
    ``n x n``: ``"cpu"``, ``"simd"`` (up to 32 x 32), ``"threadgroup"`` or
    ``"blocked"`` (the large-matrix path)."""
    return _lib.text(_lib.cholesky_backend(n, batch))


def lu_backend(n, batch=1):
    """Which backend :func:`lu_factor`, :func:`solve` and :func:`inv` use for
    ``batch`` matrices of ``n x n``: ``"cpu"`` or ``"blocked"`` (the GPU path)."""
    return _lib.text(_lib.lu_backend(n, batch))


def trsm_backend(n, k=1, batch=1):
    """Which backend :func:`solve_triangular` uses for ``batch`` triangles of
    ``n x n`` with ``k`` right-hand sides: ``"cpu"`` or ``"blocked"``."""
    return _lib.text(_lib.trsm_backend(n, k, batch))


def qr_policy():
    """The QR routing policy in effect, as a dict of its fields."""
    return _lib.get_policy("qr")


def eigh_policy():
    """The eigensolver routing policy in effect, as a dict of its fields."""
    return _lib.get_policy("eigh")


def svd_policy():
    """The SVD routing policy in effect, as a dict of its fields."""
    return _lib.get_policy("svd")


def cholesky_policy():
    """The Cholesky routing policy in effect, as a dict of its fields."""
    return _lib.get_policy("cholesky")


def lu_policy():
    """The LU routing policy in effect, as a dict of its fields."""
    return _lib.get_policy("lu")


def trsm_policy():
    """The triangular solve's routing policy in effect, as a dict of its fields."""
    return _lib.get_policy("trsm")


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


def set_cholesky_policy(policy=None, **fields):
    """Replaces the Cholesky policy, e.g. ``set_cholesky_policy(gpu_large_min_n=1024)``."""
    _lib.set_policy("cholesky", {**(policy or {}), **fields})


def set_lu_policy(policy=None, **fields):
    """Replaces the LU policy, e.g. ``set_lu_policy(gpu_min_n=1024)``."""
    _lib.set_policy("lu", {**(policy or {}), **fields})


def set_trsm_policy(policy=None, **fields):
    """Replaces the triangular solve's policy, e.g. ``set_trsm_policy(gpu_min_rhs=64)``."""
    _lib.set_policy("trsm", {**(policy or {}), **fields})


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


def cholesky_policy_source():
    """Where the Cholesky policy came from; see :func:`qr_policy_source`."""
    return _lib.text(_lib.cholesky_policy_source())


def lu_policy_source():
    """Where the LU policy came from; see :func:`qr_policy_source`."""
    return _lib.text(_lib.lu_policy_source())


def trsm_policy_source():
    """Where the triangular solve's policy came from; see :func:`qr_policy_source`."""
    return _lib.text(_lib.trsm_policy_source())


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
    ``{"qr": state, "eigh": state, "svd": state, "cholesky": state, "lu": state, "trsm": state}``, each ``"current"``,
    ``"stale"`` (measured on older kernels, still used), ``"incomplete"``
    (from before a newer backend, which stays off) or ``"uncalibrated"``
    (not measured: settings estimated from a measured Mac). See https://c0rmac.github.io/metal-linalg/docs/measurements."""
    out = {}
    for key, source in (("qr", qr_policy_source), ("eigh", eigh_policy_source),
                        ("svd", svd_policy_source), ("cholesky", cholesky_policy_source), ("lu", lu_policy_source),
                        ("trsm", trsm_policy_source)):
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
    for source in (qr_policy_source, eigh_policy_source, svd_policy_source, cholesky_policy_source,
                   lu_policy_source, trsm_policy_source):
        source()   # resolves the policy, which records its calibration
    for what in ("QR", "eigh", "SVD", "Cholesky", "LU", "triangular solve"):
        msg = _lib.text(_lib.calibration_message(what.encode()))
        if msg:
            warnings.warn(msg, CalibrationWarning, stacklevel=3)


_calibration_warnings()
