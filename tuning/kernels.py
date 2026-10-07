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

  MIN_EPOCHS      the oldest kernel version whose runs are used at all: those
                  from before 2.9.0, when the CPU path ran on one core, are
                  not, since every GPU-or-CPU boundary moved with it. A Mac
                  with only such runs is estimated instead (tuning/estimate.py).

How the combiner uses them (tuning/combine.py): runs at the current epoch are
used if there are any; otherwise the newest older runs from MIN_EPOCHS on,
marked stale, rather than an estimate. The library reports the state at run time
(policy source "tuned-stale:" / "tuned-incomplete:", and a one-line notice),
and docs/measurements.md shows it for every chip.
"""

KERNEL_EPOCHS = {"qr": 6, "eigh": 6, "svd": 8}
# The versions from which the CPU path spreads a batch over every core (2.9.0).
MIN_EPOCHS = {"qr": 2, "eigh": 2, "svd": 3}

# (decomposition, epoch, library version, date, why)
HISTORY = [
    ("qr", 1, "2.0.0", "2026-09-30", "the measurements as 2.0 introduced them"),
    ("eigh", 1, "2.0.0", "2026-09-30", "the measurements as 2.0 introduced them"),
    ("svd", 1, "2.0.0", "2026-09-30", "the measurements as 2.0 introduced them"),
    ("svd", 2, "2.2.3", "2026-10-02",
     "QR, which the QR-preconditioned SVD backends call first, routes small problems to the "
     "CPU since its CPU boundary was measured; those backends' timings have changed"),
    ("qr", 2, "2.9.0", "2026-10-03",
     "the CPU path spreads a batch over every core (7.5-15x faster for batches of small matrices "
     "on an M5 Pro), so every GPU-or-CPU boundary has moved"),
    ("eigh", 2, "2.9.0", "2026-10-03",
     "the CPU path spreads a batch over every core (7.5-15x faster for batches of small matrices "
     "on an M5 Pro), so every GPU-or-CPU boundary has moved"),
    ("svd", 3, "2.9.0", "2026-10-03",
     "the CPU path spreads a batch over every core (7.5-15x faster for batches of small matrices "
     "on an M5 Pro), so every GPU-or-CPU boundary has moved"),
    ("svd", 4, "2.10.0", "2026-10-03",
     "QR keeps the smallest matrices on the CPU at any batch (gpu_min_k), which moves the timings "
     "of the QR-preconditioned backends"),
    ("qr", 3, "2.11.0", "2026-10-03",
     "the CPU path factors a wide matrix by its leading square block and one matrix product, "
     "10-40x faster than sgeqrf on the whole matrix"),
    ("eigh", 3, "2.11.0", "2026-10-03",
     "the tridiag backend pipelines a batch over two slots (CPU solve of one matrix while the GPU "
     "reduces the next), 1.4-1.5x per matrix for batches of 2048 x 2048"),
    ("svd", 5, "2.11.0", "2026-10-03",
     "golub_kahan splits its column sums over lanes (1.2-1.4x on tall matrices) and runs the QR "
     "iteration as a second dispatch from k = 40 (singular values alone) or 60; the bidiag backend "
     "pipelines a batch over two slots (1.5-1.7x per matrix for batches of 2048 x 2048)"),
    ("eigh", 4, "2.12.0", "2026-10-04",
     "the tridiag backend's reduction takes three dispatches per column instead of seven, and "
     "copies the matrix in on every core: 1.3-1.7x for eigenvalues alone, 1.2-1.5x with vectors"),
    ("svd", 6, "2.12.0", "2026-10-04",
     "the bidiag backend's reduction takes four dispatches per column instead of twelve, and "
     "copies the matrix in on every core: 1.2-1.8x for singular values alone, 1.1-1.3x with vectors"),
    ("eigh", 5, "2.13.0", "2026-10-04",
     "the tridiag backend's eigenvalues alone come from bisection on the GPU rather than ssterf from "
     "N = 512: 6 ms against 79 at 4096"),
    ("svd", 7, "2.13.0", "2026-10-04",
     "the bidiag backend's singular values alone come from bisection on the GPU from k = 1024, and "
     "below that from sbdsqr (dqds) rather than sbdsdc: 12 ms against 93 at 4096"),
    ("eigh", 6, "2.15.0", "2026-10-07",
     "the tridiag backend's eigenvectors come from a divide and conquer on every core (sstedc's 176 ms "
     "to 50 at 4096); the band backend's panels are faster (the TSQR top a tree, no IEEE division), "
     "its small products are kernels of their own and its trailing update is on the lower triangle: "
     "eigh with vectors 1.3-1.5x and eigvalsh 1.1-1.2x at 2048-8192"),
    ("svd", 8, "2.15.0", "2026-10-07",
     "the bidiag backend's singular vectors come from a divide and conquer on every core (sbdsdc's "
     "753 ms to 100 at 4096): the SVD with vectors 1.7-1.9x at 2048-4096; the band backend's panels "
     "and small products are faster: svdvals 1.1-1.3x; and the band backend takes singular vectors "
     "too (band_min_k): 2.3x bidiag at 4096"),
    ("qr", 4, "2.15.0", "2026-10-07",
     "the reduced backend hands one matrix, or a few large ones, to the blocked QR (the band "
     "reduction's panels, aggregates of 128 columns, MPS products): 2.1x at 1024, 2.6x at 2048, "
     "3.7x at 4096, 5x on tall 4096 x 1024 and 8192 x 512"),
    ("qr", 5, "2.15.0", "2026-10-07",
     "the blocked QR takes a batch at once and any height (padded to whole panels, batched MPS "
     "products): every shape the reduced backend's streaming kernels took, 1.8-3.5x faster; batches "
     "of 512-2048 now beat the CPU (16 x 1024^2: 25 ms against 48); the large clause counts rows "
     "and k, sqrt(M k), and the grid has tall large shapes"),
    ("qr", 6, "2.16.0", "2026-10-08",
     "the unblocked backend is new Householder kernels (in a simdgroup's registers up to 32 x 128, "
     "else blocked in a threadgroup up to 4096 rows, its updates 8 x 8 simdgroup matrix products; "
     "its own kernel retired): 1024 of 128 x 128 in 6.4 ms against 17.8 for the blocked QR and 25 "
     "on the CPU, 4096 of 32 x 32 in 1.4 against 2.9 on the CPU; MLX's own buffers no longer "
     "wrapped again; the grid's kernel crossover goes down to 64 rows and its mid-size batches up "
     "to 384"),
]

REQUIRED = {
    "qr": {"unblocked", "reduced", "cpu", "share"},
    "eigh": {"cpu", "simd", "tg", "block", "tridiag", "ql", "ql_share",
             "cpu_vals", "simd_vals", "tg_vals", "block_vals", "tridiag_vals", "ql_vals", "ql_share_vals",
             "band_vals", "band8_vals", "band32_vals"},
    "svd": {"cpu", "jacobi", "block", "qr", "qrblock", "bidiag", "band", "gk", "gk_share", "cpu_vals",
            "bidiag_vals", "gk_vals", "gk_share_vals", "band_vals", "band8_vals", "band32_vals"},
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
    "ql": "the ql backend (2.9.0)",
    "ql_vals": "the ql backend (2.9.0)",
    "gk": "the golub_kahan backend (2.10.0)",
    "gk_vals": "the GPU-or-CPU rule for singular values alone (2.11.0)",
    "gk_share": "sharing a batch between the GPU and the CPU (2.11.0)",
    "gk_share_vals": "sharing a batch between the GPU and the CPU (2.11.0)",
    "ql_share": "sharing a batch between the GPU and the CPU (2.11.0)",
    "ql_share_vals": "sharing a batch between the GPU and the CPU (2.11.0)",
    "share": "sharing a batch between the GPU and the CPU (2.12.0)",
    "band_vals": "the band backend, the two-stage reduction for eigenvalues or singular values alone (2.13.0)",
    "band8_vals": "the band backend's width as part of the policy (2.15.0)",
    "band32_vals": "the band backend's width as part of the policy (2.15.0)",
}
# Where a backend name means something else for one decomposition.
ADDED_FOR = {
    "svd": {"cpu_vals": "the singular-value-only paths (2.7.0)",
            "band": "the band backend with singular vectors (2.15.0)"},
}


def added(op, backend):
    """What an incomplete `op` run that never timed `backend` is missing, in words."""
    return ADDED_FOR.get(op, {}).get(backend) or ADDED.get(backend, backend)

# The CPU paths' batch loop (lapack_batches) is in the shared runtime, and
# the two-stage and divide-and-conquer pieces serve both eigh and the SVD.
_SHARED = ["src/metal_runtime", "src/blas_threading"]
_BAND = ["src/band_", "src/bisect", "src/divide_conquer"]
# The blocked QR runs on the band reduction's panel kernels.
_QR = ["shaders/QR_", "src/qr", "src/band_reduce", "shaders/Svd_Bidiag"] + _SHARED
_JACOBI = ["shaders/eigh_jacobi_common.h", "shaders/block_jacobi_common.h"]
PATHS = {
    "qr": _QR,
    "eigh": ["shaders/Eigh_", "src/eigh", "shaders/Svd_Bidiag"] + _JACOBI + _SHARED + _BAND,
    # The QR-preconditioned SVD backends call QR, routed by its table.
    "svd": ["shaders/Svd_", "src/svd"] + _JACOBI + _QR + ["src/tuned/qr.inc"] + _BAND,
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
