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

__all__ = [
    "qr", "eigh", "eigvalsh", "svd", "svdvals",
    "device_name", "gpu_core_count",
    "qr_backend", "eigh_backend", "svd_backend",
    "qr_policy", "eigh_policy", "svd_policy",
    "set_qr_policy", "set_eigh_policy", "set_svd_policy",
    "qr_policy_source", "eigh_policy_source", "svd_policy_source",
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
    Always runs on the GPU.
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
    """Eigenvalues only of a batch of symmetric matrices, ascending; about a
    third less work than :func:`eigh`."""
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


def qr_backend(m, n, batch=1):
    """Which backend :func:`qr` uses for ``batch`` matrices of ``m x n``:
    ``"cpu"``, ``"unblocked"`` or ``"streaming_reduced"``."""
    return _core.qr_backend(m, n, batch)


def eigh_backend(n, batch=1):
    """Which backend :func:`eigh` uses: ``"cpu"``, ``"simd"``,
    ``"threadgroup"`` or ``"block"``."""
    return _core.eigh_backend(n, batch)


def svd_backend(m, n, batch=1):
    """Which backend :func:`svd` uses: ``"cpu"``, ``"jacobi"``,
    ``"block_jacobi"``, ``"qr_jacobi"`` or ``"qr_block_jacobi"``."""
    return _core.svd_backend(m, n, batch)


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
