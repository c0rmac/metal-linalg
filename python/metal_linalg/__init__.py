"""QR, symmetric eigendecomposition, SVD, Cholesky and LU (with solve and
inverse) for batches of matrices on Apple GPUs, for MLX.

    import mlx.core as mx
    import metal_linalg as ml

    Q, R = ml.qr(a)              # a: [..., M, N]
    w, V = ml.eigh(s)            # s symmetric [..., N, N]; w ascending
    U, S, Vt = ml.svd(a)         # thin factors; S descending
    L = ml.cholesky(p)           # p symmetric positive definite; p = L L^T
    x = ml.solve(a, b)           # a [..., N, N], b [..., N] or [..., N, K]

Each call is routed to the fastest Metal kernel for its shape and batch, or to
MLX's CPU path, by a policy measured on the Mac it runs on. Inputs may be
mx.array or anything mx.array accepts (NumPy arrays, nested lists); outputs
are float32 mx.array. See https://github.com/c0rmac/metal-linalg.
"""

import mlx.core as mx

from ._build import mlx_version as _built_mlx

# Checked before loading the extension: against another MLX it would fail to
# load with a missing-symbol error that names neither version.
if _built_mlx != mx.__version__:
    raise ImportError(
        f"metal_linalg was built against MLX {_built_mlx} but MLX "
        f"{mx.__version__} is installed. Install the matching pair with "
        f"`pip install -U metal-linalg mlx`, or `pip install mlx=={_built_mlx}` "
        f"(see https://github.com/c0rmac/metal-linalg/blob/main/python/README.md).")

from . import _core  # noqa: E402

__version__ = _core.__version__


class CalibrationWarning(UserWarning):
    """Raised once per decomposition at import when this Mac's calibration
    is missing, stale or incomplete: the library still works, but measuring
    this Mac and submitting the results would make it faster (see
    :func:`calibration_status`). Silence it with
    ``warnings.filterwarnings("ignore", category=metal_linalg.CalibrationWarning)``
    or ``METAL_LINALG_NO_CALIBRATION_NOTICE=1``."""


def calibration_status():
    """How current this Mac's measurements are, per decomposition:
    ``{"qr": state, "eigh": state, "svd": state, "cholesky": state, "lu": state, "trsm": state}``, each ``"current"``,
    ``"stale"`` (measured on older kernels, still used), ``"incomplete"``
    (from before a newer backend, which stays off) or ``"uncalibrated"``
    (not measured: settings estimated from a measured Mac). See https://c0rmac.github.io/metal-linalg/docs/measurements."""
    out = {}
    for key, source in (("qr", _core.qr_policy_source), ("eigh", _core.eigh_policy_source),
                        ("svd", _core.svd_policy_source), ("cholesky", _core.cholesky_policy_source),
                        ("lu", _core.lu_policy_source), ("trsm", _core.trsm_policy_source)):
        s = source()
        out[key] = ("uncalibrated" if s.startswith(("default:", "estimated:"))
                    else "stale" if s.startswith("tuned-stale:")
                    else "incomplete" if s.startswith("tuned-incomplete:") else "current")
    return out


def _calibration_warnings():
    import os
    import warnings
    _core.set_calibration_notices(False)   # this package warns instead of printing
    flag = os.environ.get("METAL_LINALG_NO_CALIBRATION_NOTICE", "")
    if flag and flag != "0":
        return
    for source in (_core.qr_policy_source, _core.eigh_policy_source, _core.svd_policy_source,
                   _core.cholesky_policy_source, _core.lu_policy_source, _core.trsm_policy_source):
        source()   # resolves the policy, which records its calibration
    for what in ("QR", "eigh", "SVD", "Cholesky", "LU", "triangular solve"):
        msg = _core.calibration_message(what)
        if msg:
            warnings.warn(msg, CalibrationWarning, stacklevel=3)


_calibration_warnings()

__all__ = [
    "qr", "eigh", "eigvalsh", "svd", "svdvals", "cholesky", "cholesky_ex",
    "lu_factor", "lu_factor_ex", "solve", "solve_ex", "inv", "inv_ex", "solve_triangular",
    "device_name", "gpu_core_count", "cpu_threads", "set_cpu_threads",
    "qr_backend", "eigh_backend", "eigvalsh_backend", "svd_backend", "svdvals_backend", "cholesky_backend", "lu_backend", "trsm_backend",
    "qr_policy", "eigh_policy", "svd_policy", "cholesky_policy", "lu_policy", "trsm_policy",
    "set_qr_policy", "set_eigh_policy", "set_svd_policy", "set_cholesky_policy", "set_lu_policy", "set_trsm_policy",
    "qr_policy_source", "eigh_policy_source", "svd_policy_source", "cholesky_policy_source", "lu_policy_source", "trsm_policy_source",
    "calibration_status", "CalibrationWarning",
]


def _array(a):
    return a if isinstance(a, mx.array) else mx.array(a)


# ---------------------------------------------------------------------------
# Decompositions
# ---------------------------------------------------------------------------

def qr(a, mode="reduced"):
    """QR of a batch of matrices: ``a = Q @ R``.

    ``a`` is ``[..., M, N]``. Returns ``Q`` ``[..., M, K]`` with orthonormal
    columns and ``R`` ``[..., K, N]`` upper triangular, ``K = min(M, N)``.
    Runs on the GPU, or in LAPACK on the CPU for problems too small to pay
    for a GPU launch, as this Mac was measured (see :func:`qr_backend`).

    ``mode`` as :func:`numpy.linalg.qr`'s: ``"reduced"`` (the above),
    ``"r"`` (``R`` alone, returned on its own; ``Q`` is never formed:
    up to 2.8x faster) or ``"complete"`` (``Q`` ``[..., M, M]`` square, ``R``
    ``[..., M, N]`` with zero rows below ``K``).
    """
    if mode not in ("reduced", "r", "complete"):
        raise ValueError(f'mode must be "reduced", "r" or "complete", not {mode!r}')
    q, r = _core.qr(_array(a), mode)
    return r if mode == "r" else (q, r)


def eigh(a, uplo="L"):
    """Eigendecomposition of a batch of symmetric matrices: ``a = V diag(w) V^T``.

    ``a`` is ``[..., N, N]``; only the triangle named by ``uplo`` (``"L"`` or
    ``"U"``) is read. Returns ``w`` ``[..., N]`` in ascending order and ``V``
    ``[..., N, N]`` with the eigenvectors as columns, like ``mx.linalg.eigh``.
    A non-finite matrix yields NaN rather than an error.
    """
    return _core.eigh(_array(a), uplo)


def eigvalsh(a, uplo="L"):
    """Eigenvalues only of a batch of symmetric matrices, ascending; less work
    than :func:`eigh`: about a third less on the GPU, and on the CPU from
    N = 128 a two-stage reduction several times faster at large N (5.7x at
    N = 8192 on an M5 Pro)."""
    return _core.eigvalsh(_array(a), uplo)


def svd(a):
    """Thin SVD of a batch of matrices: ``a = U diag(S) Vt``.

    ``a`` is ``[..., M, N]``. Returns ``U`` ``[..., M, K]``, ``S`` ``[..., K]``
    descending and ``Vt`` ``[..., K, N]``, ``K = min(M, N)``. Unlike
    ``mx.linalg.svd`` the factors are the thin ones. A non-finite matrix
    yields NaN rather than an error.
    """
    return _core.svd(_array(a))


def svdvals(a):
    """Singular values only, descending; about half the work of :func:`svd`."""
    return _core.svdvals(_array(a))


def cholesky(a, upper=False):
    """Cholesky factorization of a batch of symmetric positive definite
    matrices: ``a = L @ L^T``.

    ``a`` is ``[..., N, N]``; only its lower triangle is read (the upper one
    with ``upper=True``). Returns ``L`` ``[..., N, N]``, lower triangular with
    a positive diagonal and zeros above it, or with ``upper=True``
    ``U = L^T`` (``a = U^T @ U``), like ``mx.linalg.cholesky``. A matrix that
    is not positive definite (or holds a NaN or infinity where it is read)
    comes back all NaN rather than raising; :func:`cholesky_ex` says which
    and where. On the GPU for large matrices, else in LAPACK on every CPU
    core, as this Mac was measured (see :func:`cholesky_backend`).
    """
    return _core.cholesky(_array(a), bool(upper))


def cholesky_ex(a, upper=False):
    """:func:`cholesky` and ``info`` ``[...]`` (uint32), as
    ``torch.linalg.cholesky_ex``: 0 for a matrix factored, else ``k`` where
    its leading minor of order ``k`` is not positive definite (LAPACK's
    ``spotrf`` convention); that matrix's ``L`` is all NaN."""
    return _core.cholesky_ex(_array(a), bool(upper))


def lu_factor(a):
    """LU factorization with partial pivoting, ``P a = L U``, like
    ``mx.linalg.lu_factor``: ``(LU, pivots)``, ``LU`` ``[..., N, N]`` holding
    ``U`` on and above the diagonal and ``L`` below it (its unit diagonal
    implied), ``pivots`` ``[..., N]`` uint32, the row swaps in order (row
    ``i`` was swapped with row ``pivots[i]``). Large matrices on the GPU
    (see :func:`lu_backend`)."""
    lu, piv, _ = _core.lu_factor(_array(a))
    return lu, piv


def lu_factor_ex(a):
    """:func:`lu_factor` and ``info`` ``[...]`` (uint32): 0, or ``k`` where
    ``U``'s k-th diagonal entry is exactly zero (``a`` is singular)."""
    return _core.lu_factor(_array(a))


def solve(a, b):
    """``x`` with ``a @ x = b``, like ``mx.linalg.solve``: ``a``
    ``[..., N, N]``, ``b`` ``[..., N, K]`` or ``[..., N]`` with ``a``'s batch
    shape. A singular matrix's ``x`` is all NaN rather than an error."""
    return _core.solve(_array(a), _array(b))[0]


def solve_ex(a, b):
    """:func:`solve` and ``info``, as :func:`lu_factor_ex`'s."""
    return _core.solve(_array(a), _array(b))


def inv(a):
    """The inverse of a batch of square matrices, like ``mx.linalg.inv``. A
    singular matrix's inverse is all NaN rather than an error."""
    return _core.inv(_array(a))[0]


def solve_triangular(a, b, upper=False, unit_diagonal=False):
    """``x`` with ``a @ x = b`` for triangular ``a``, like
    ``mx.linalg.solve_triangular``: ``a`` ``[..., N, N]`` (only its lower
    triangle read, the upper with ``upper``; with ``unit_diagonal`` its
    diagonal taken as ones), ``b`` ``[..., N, K]`` or ``[..., N]`` with
    ``a``'s batch shape. On the GPU for large N and K (see
    :func:`trsm_backend`)."""
    return _core.solve_triangular(_array(a), _array(b), bool(upper), bool(unit_diagonal))


def inv_ex(a):
    """:func:`inv` and ``info``, as :func:`lu_factor_ex`'s."""
    return _core.inv(_array(a))


# ---------------------------------------------------------------------------
# The device and its routing
# ---------------------------------------------------------------------------

def device_name():
    """The GPU the policies were resolved for, e.g. ``"Apple M5 Pro"``."""
    return _core.device_name()


def gpu_core_count():
    """Its GPU core count; 0 if it could not be read."""
    return _core.gpu_core_count()


def cpu_threads():
    """How many CPU threads the CPU paths spread a batch over: every core by
    default. A lone matrix keeps Accelerate's own threading."""
    return _core.cpu_threads()


def set_cpu_threads(n):
    """Caps :func:`cpu_threads`, for a program that runs several solves at once
    on threads of its own; 0 restores every core. ``METAL_LINALG_CPU_THREADS``
    sets it from the environment."""
    _core.set_cpu_threads(int(n))


def qr_backend(m, n, batch=1):
    """Which backend :func:`qr` uses for ``batch`` matrices of ``m x n``:
    ``"cpu"``, ``"unblocked"`` or ``"streaming_reduced"``."""
    return _core.qr_backend(m, n, batch)


def eigh_backend(n, batch=1):
    """Which backend :func:`eigh` uses: ``"cpu"``, ``"simd"``,
    ``"threadgroup"``, ``"block"``, ``"tridiag"``, ``"ql"``, ``"band"`` (the
    two-stage reduction, from the policy's ``band_min_n``) or
    ``"tridiag_batch"`` (a batch of mid-size matrices at once, inside the
    policy's ``tridiag_batch_*`` window)."""
    return _core.eigh_backend(n, batch)


def eigvalsh_backend(n, batch=1):
    """Which backend :func:`eigvalsh` uses; as :func:`eigh_backend`, under
    the eigenvalues-alone boundary of the policy (``values_gpu_*``), and
    ``"band"`` (the two-stage reduction) from ``values_band_min_n``."""
    return _core.eigvalsh_backend(n, batch)


def svd_backend(m, n, batch=1):
    """Which backend :func:`svd` uses: ``"cpu"``, ``"jacobi"``,
    ``"block_jacobi"``, ``"qr_jacobi"``, ``"qr_block_jacobi"``, ``"bidiag"``,
    ``"band"`` (the two-stage reduction, from the policy's ``band_min_k``),
    ``"golub_kahan"``, ``"qr_golub_kahan"`` or ``"bidiag_batch"`` (a batch of
    mid-size matrices at once, inside the policy's ``bidiag_batch_*``
    window)."""
    return _core.svd_backend(m, n, batch)


def svdvals_backend(m, n, batch=1):
    """Which backend :func:`svdvals` uses; as :func:`svd_backend`, with the
    policy's ``values_bidiag_min_k`` for the bidiag backend, and ``"band"``
    (the two-stage reduction) from ``values_band_min_k``."""
    return _core.svdvals_backend(m, n, batch)


def cholesky_backend(n, batch=1):
    """Which backend :func:`cholesky` uses for ``batch`` matrices of
    ``n x n``: ``"cpu"``, ``"simd"`` (up to 32 x 32, a matrix in a
    simdgroup's registers), ``"threadgroup"`` or ``"blocked"`` (the
    large-matrix path)."""
    return _core.cholesky_backend(n, batch)


def lu_backend(n, batch=1):
    """Which backend :func:`lu_factor`, :func:`solve` and :func:`inv` use for
    ``batch`` matrices of ``n x n``: ``"cpu"`` or ``"blocked"`` (the GPU
    path)."""
    return _core.lu_backend(n, batch)


def trsm_backend(n, k=1, batch=1):
    """Which backend :func:`solve_triangular` uses for ``batch`` triangles of
    ``n x n`` with ``k`` right-hand sides: ``"cpu"`` or ``"blocked"``."""
    return _core.trsm_backend(n, k, batch)


def qr_policy():
    """The QR routing policy in effect, as a dict of its fields."""
    return _core.qr_policy()


def eigh_policy():
    """The eigensolver routing policy in effect, as a dict of its fields."""
    return _core.eigh_policy()


def svd_policy():
    """The SVD routing policy in effect, as a dict of its fields."""
    return _core.svd_policy()


def cholesky_policy():
    """The Cholesky routing policy in effect, as a dict of its fields."""
    return _core.cholesky_policy()


def lu_policy():
    """The LU routing policy in effect, as a dict of its fields."""
    return _core.lu_policy()


def trsm_policy():
    """The triangular solve's routing policy in effect, as a dict of its fields."""
    return _core.trsm_policy()


def set_qr_policy(policy=None, **fields):
    """Replaces the QR policy. Pass a dict from :func:`qr_policy`, or just the
    fields to change: ``set_qr_policy(m_crossover_small_batch=320)``."""
    _core.set_qr_policy({**(policy or {}), **fields})


def set_eigh_policy(policy=None, **fields):
    """Replaces the eigensolver policy, e.g. ``set_eigh_policy(gpu_min_batch=1)``."""
    _core.set_eigh_policy({**(policy or {}), **fields})


def set_svd_policy(policy=None, **fields):
    """Replaces the SVD policy, e.g. ``set_svd_policy(gpu_min_batch=1)``."""
    _core.set_svd_policy({**(policy or {}), **fields})


def set_cholesky_policy(policy=None, **fields):
    """Replaces the Cholesky policy, e.g. ``set_cholesky_policy(gpu_large_min_n=1024)``."""
    _core.set_cholesky_policy({**(policy or {}), **fields})


def set_lu_policy(policy=None, **fields):
    """Replaces the LU policy, e.g. ``set_lu_policy(gpu_min_n=1024)``."""
    _core.set_lu_policy({**(policy or {}), **fields})


def set_trsm_policy(policy=None, **fields):
    """Replaces the triangular solve's policy, e.g. ``set_trsm_policy(gpu_min_rhs=64)``."""
    _core.set_trsm_policy({**(policy or {}), **fields})


def qr_policy_source():
    """Where the QR policy came from: ``"tuned:<device>"``, ``"estimated:<device>
    (from <measured device>, ...)"`` on a Mac nobody has measured,
    ``"default:untuned-device (<device>)"``, ``"env:..."`` or ``"user"``."""
    return _core.qr_policy_source()


def eigh_policy_source():
    """Where the eigensolver policy came from; see :func:`qr_policy_source`."""
    return _core.eigh_policy_source()


def svd_policy_source():
    """Where the SVD policy came from; see :func:`qr_policy_source`."""
    return _core.svd_policy_source()


def cholesky_policy_source():
    """Where the Cholesky policy came from; see :func:`qr_policy_source`."""
    return _core.cholesky_policy_source()


def lu_policy_source():
    """Where the LU policy came from; see :func:`qr_policy_source`."""
    return _core.lu_policy_source()


def trsm_policy_source():
    """Where the triangular solve's policy came from; see :func:`qr_policy_source`."""
    return _core.trsm_policy_source()
