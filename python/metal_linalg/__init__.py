"""QR, symmetric eigendecomposition and SVD for batches of matrices on Apple
GPUs, for MLX.

    import mlx.core as mx
    import metal_linalg as ml

    Q, R = ml.qr(a)              # a: [..., M, N]
    w, V = ml.eigh(s)            # s symmetric [..., N, N]; w ascending
    U, S, Vt = ml.svd(a)         # thin factors; S descending

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
    ``{"qr": state, "eigh": state, "svd": state}``, each ``"current"``,
    ``"stale"`` (measured on older kernels, still used), ``"incomplete"``
    (from before a newer backend, which stays off) or ``"uncalibrated"``
    (the untuned default). See https://c0rmac.github.io/metal-linalg/docs/measurements."""
    out = {}
    for key, source in (("qr", _core.qr_policy_source), ("eigh", _core.eigh_policy_source),
                        ("svd", _core.svd_policy_source)):
        s = source()
        out[key] = ("uncalibrated" if s.startswith("default:") else "stale" if s.startswith("tuned-stale:")
                    else "incomplete" if s.startswith("tuned-incomplete:") else "current")
    return out


def _calibration_warnings():
    import os
    import warnings
    _core.set_calibration_notices(False)   # this package warns instead of printing
    flag = os.environ.get("METAL_LINALG_NO_CALIBRATION_NOTICE", "")
    if flag and flag != "0":
        return
    for source in (_core.qr_policy_source, _core.eigh_policy_source, _core.svd_policy_source):
        source()   # resolves the policy, which records its calibration
    for what in ("QR", "eigh", "SVD"):
        msg = _core.calibration_message(what)
        if msg:
            warnings.warn(msg, CalibrationWarning, stacklevel=3)


_calibration_warnings()

__all__ = [
    "qr", "eigh", "eigvalsh", "svd", "svdvals",
    "device_name", "gpu_core_count", "cpu_threads", "set_cpu_threads",
    "qr_backend", "eigh_backend", "eigvalsh_backend", "svd_backend", "svdvals_backend",
    "qr_policy", "eigh_policy", "svd_policy",
    "set_qr_policy", "set_eigh_policy", "set_svd_policy",
    "qr_policy_source", "eigh_policy_source", "svd_policy_source",
    "calibration_status", "CalibrationWarning",
]


def _array(a):
    return a if isinstance(a, mx.array) else mx.array(a)


# ---------------------------------------------------------------------------
# Decompositions
# ---------------------------------------------------------------------------

def qr(a):
    """Thin QR of a batch of matrices: ``a = Q @ R``.

    ``a`` is ``[..., M, N]``. Returns ``Q`` ``[..., M, K]`` with orthonormal
    columns and ``R`` ``[..., K, N]`` upper triangular, ``K = min(M, N)``.
    Runs on the GPU, or in LAPACK on the CPU for problems too small to pay
    for a GPU launch, as this Mac was measured (see :func:`qr_backend`).
    """
    return _core.qr(_array(a))


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
    ``"threadgroup"``, ``"block"``, ``"tridiag"`` or ``"ql"``."""
    return _core.eigh_backend(n, batch)


def eigvalsh_backend(n, batch=1):
    """Which backend :func:`eigvalsh` uses; as :func:`eigh_backend`, under
    the eigenvalues-alone boundary of the policy (``values_gpu_*``), and
    ``"band"`` (the two-stage reduction) from ``values_band_min_n``."""
    return _core.eigvalsh_backend(n, batch)


def svd_backend(m, n, batch=1):
    """Which backend :func:`svd` uses: ``"cpu"``, ``"jacobi"``,
    ``"block_jacobi"``, ``"qr_jacobi"``, ``"qr_block_jacobi"``, ``"bidiag"``,
    ``"golub_kahan"`` or ``"qr_golub_kahan"``."""
    return _core.svd_backend(m, n, batch)


def svdvals_backend(m, n, batch=1):
    """Which backend :func:`svdvals` uses; as :func:`svd_backend`, with the
    policy's ``values_bidiag_min_k`` for the bidiag backend, and ``"band"``
    (the two-stage reduction) from ``values_band_min_k``."""
    return _core.svdvals_backend(m, n, batch)


def qr_policy():
    """The QR routing policy in effect, as a dict of its fields."""
    return _core.qr_policy()


def eigh_policy():
    """The eigensolver routing policy in effect, as a dict of its fields."""
    return _core.eigh_policy()


def svd_policy():
    """The SVD routing policy in effect, as a dict of its fields."""
    return _core.svd_policy()


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


def qr_policy_source():
    """Where the QR policy came from: ``"tuned:<device>"``,
    ``"default:untuned-device (<device>)"``, ``"env:..."`` or ``"user"``."""
    return _core.qr_policy_source()


def eigh_policy_source():
    """Where the eigensolver policy came from; see :func:`qr_policy_source`."""
    return _core.eigh_policy_source()


def svd_policy_source():
    """Where the SVD policy came from; see :func:`qr_policy_source`."""
    return _core.svd_policy_source()
