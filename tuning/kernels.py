"""Which measurements are current: kernel versions per decomposition.

A run measures the library as it was. Most releases change nothing a
measurement depends on (docs, packaging, another decomposition, the tables
themselves), so a run stays valid across them. What makes a decomposition's
measurements stale is a change to what it times: its kernels, their launch
parameters, its CPU path, or anything it calls. That is tracked here, per
decomposition, by hand:

  KERNEL_EPOCHS   the current kernel version of each decomposition. Bump one,
                  and add a line to HISTORY saying why, when a change would
                  alter its timings. Every run records the versions it was
                  measured at (tuning/run.py).
  REQUIRED        the backends a run must have timed for its decomposition to
                  be complete. A run that predates a backend is not stale --
                  what it timed still holds -- but incomplete: the new backend
                  stays off on that Mac until it is measured again.
  PATHS           the files whose changes may alter a decomposition's timings.
                  tuning/check_kernel_epochs.py warns on a pull request that
                  changes them without bumping the epoch; a person decides.

How the combiner uses them (tuning/combine.py): runs at the current epoch are
used if there are any; otherwise the newest older runs, marked stale, rather
than nothing (an untuned default is usually worse than slightly old
measurements). The library reports the state at run time
(policy source "tuned-stale:" / "tuned-incomplete:", and a one-line notice),
and docs/measurements.md shows it for every chip.
"""

KERNEL_EPOCHS = {"qr": 1, "eigh": 1, "svd": 2}

# (decomposition, epoch, library version, date, why)
HISTORY = [
    ("qr", 1, "2.0.0", "2026-09-30", "the measurements as 2.0 introduced them"),
    ("eigh", 1, "2.0.0", "2026-09-30", "the measurements as 2.0 introduced them"),
    ("svd", 1, "2.0.0", "2026-09-30", "the measurements as 2.0 introduced them"),
    ("svd", 2, "2.2.3", "2026-10-02",
     "QR, which the QR-preconditioned SVD backends call first, routes small problems to the "
     "CPU since its CPU boundary was measured; those backends' timings have changed"),
]

REQUIRED = {
    "qr": {"unblocked", "reduced", "cpu"},
    "eigh": {"cpu", "simd", "tg", "block", "tridiag",
             "cpu_vals", "simd_vals", "tg_vals", "block_vals", "tridiag_vals"},
    "svd": {"cpu", "jacobi", "block", "qr", "qrblock", "bidiag", "cpu_vals", "bidiag_vals"},
}

# Backends added after a decomposition's first measurements, and when: what
# an incomplete run is missing, in words.
ADDED = {
    "cpu": "the CPU path (2.1.0)",
    "cpu_vals": "the eigenvalue-only paths (2.4.0)",
    "simd_vals": "the eigenvalue-only paths (2.4.0)",
    "tg_vals": "the eigenvalue-only paths (2.4.0)",
    "block_vals": "the eigenvalue-only paths (2.4.0)",
    "tridiag": "the tridiag backend (2.5.0)",
    "tridiag_vals": "the tridiag backend (2.5.0)",
    "bidiag": "the bidiag backend (2.7.0)",
    "bidiag_vals": "the bidiag backend (2.7.0)",
}
# Where a backend name means something else for one decomposition.
ADDED_FOR = {
    "svd": {"cpu_vals": "the singular-value-only paths (2.7.0)"},
}


def added(op, backend):
    """What an incomplete `op` run that never timed `backend` is missing, in words."""
    return ADDED_FOR.get(op, {}).get(backend) or ADDED.get(backend, backend)

_QR = ["shaders/QR_", "src/qr"]
_JACOBI = ["shaders/eigh_jacobi_common.h", "shaders/block_jacobi_common.h"]
PATHS = {
    "qr": _QR,
    "eigh": ["shaders/Eigh_", "src/eigh"] + _JACOBI,
    # The QR-preconditioned SVD backends call QR, routed by its table.
    "svd": ["shaders/Svd_", "src/svd"] + _JACOBI + _QR + ["src/tuned/qr.inc"],
}
# A decomposition's own table is generated from its measurements, not a change to them.
OWN_TABLE = {"qr": "src/tuned/qr.inc", "eigh": "src/tuned/eigh.inc", "svd": "src/tuned/svd.inc"}


def run_epoch(info, op):
    """The kernel version a submission measured `op` at. Runs from before
    per-decomposition versions recorded one global epoch, 1."""
    return int((info.get("epochs") or {}).get(op, info.get("epoch", 1)))


def touches(op, path):
    """Whether a changed file may alter `op`'s timings."""
    if path == OWN_TABLE[op] or path.endswith(".md"):
        return False
    return any(path.startswith(p) for p in PATHS[op])
