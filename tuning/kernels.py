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

KERNEL_EPOCHS = {"qr": 8, "eigh": 8, "svd": 10, "cholesky": 1, "lu": 1, "trsm": 1}
# The versions from which the CPU path spreads a batch over every core (2.9.0).
MIN_EPOCHS = {"qr": 2, "eigh": 2, "svd": 3, "cholesky": 1, "lu": 1, "trsm": 1}

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
    ("qr", 7, "2.16.0", "2026-10-08",
     "the sweeps keep MLX's buffer cache on, as an MLX program has it (off, every call's outputs "
     "were fresh pages the GPU maps at about 12 us a MB: 4096 of 64 x 64 6.4 ms on the GPU against "
     "4.0 with it on); the blocked kernel gives a small batch more simdgroups a matrix (one 384 x 384 "
     "2.0 ms against 3.0) and reads an aligned input directly; the GPU-or-CPU rule is on sqrt(M k)"),
    ("eigh", 7, "2.16.0", "2026-10-08",
     "the sweeps keep MLX's buffer cache on, as an MLX program has it, and MLX's own buffers are no "
     "longer wrapped again: the GPU backends' calls on large batches 10-40% cheaper"),
    ("eigh", 8, "2.17.0", "2026-10-09",
     "the ql backend keeps a matrix of up to 32 in a simdgroup's registers, four or two a simdgroup up to "
     "8 or 16 (1.1-2x with eigenvectors; eigenvalues alone by bisection, a lane an eigenvalue, 1.4-3x), "
     "and two new backends: "
     "tridiag_batch (a batch of mid-size matrices reduced together, its symmetric products from the lower "
     "triangle alone: 1.3-1.5x the CPU at 256-1024 matrices of 96-256, 1.55x at 16 of 1024) and band with "
     "eigenvectors (the two-stage reduction, 1.36x tridiag at 4096); the CPU path with eigenvectors, for a "
     "batch of at most a quarter as many matrices as cores, runs ssyevd's steps with the divide and conquer "
     "on the idle cores from N = 192 (1.17-1.26x); tridiag_batch with eigenvectors in two stages for small "
     "batches from N = 64 (2.6x the one-stage reduction at 1 x 1024^2; four dispatches a band block), band "
     "with eigenvectors hands it batches of two or more (4 x 1024^2 2x), and the batch back-transformations' "
     "T is built in blocks (1.02-1.15x)"),
    ("svd", 9, "2.16.0", "2026-10-08",
     "the sweeps keep MLX's buffer cache on, as an MLX program has it, and MLX's own buffers are no "
     "longer wrapped again: the GPU backends' calls on large batches 10-40% cheaper; the QR-"
     "preconditioned backends run on 2.16.0's QR"),
    ("svd", 10, "2.17.0", "2026-10-09",
     "a new backend, bidiag_batch: a batch of mid-size matrices bidiagonalized together, each panel step "
     "reading the trailing block once, the bidiagonal problems on the CPU's cores, the back-transformations "
     "as batched products, a tall or wide matrix's R after this library's QR (1.4-2x the CPU at "
     "256-1024 matrices of 128-256), singular values alone from k = 160 in two stages (3.3x the CPU at "
     "16 x 1024^2); and golub_kahan in registers up to 32 x 32, four or two matrices a simdgroup up to "
     "8 or 16 rows, from 17 rows a runner simdgroup for the QR iterations (1.3-2.1x with vectors, "
     "singular values alone by bisection 1.6-2.7x); the CPU path with vectors, for a batch of at most a "
     "quarter as many matrices as cores, runs sgesdd's steps with the divide and conquer on the idle cores "
     "from k = 192 (1.2-1.4x); bidiag_batch with vectors in two stages from k = 288 (128 for small batches; "
     "2.7-3.4x the direct reduction at 1024), its band blocks in three passes (1.04-1.16x), band with vectors "
     "hands it batches of two or more (4 x 1024^2 2.4x), and the back-transformations' T in blocks "
     "(1.04-1.05x)"),
    ("qr", 8, "2.17.0", "2026-10-09",
     "the register kernel packs small matrices, four a simdgroup up to 8 rows and two up to 16 (2.5x at "
     "4 x 4, 1.7-2.5x at 8 x 8, 1.3-1.7x at 16 x 16 for large batches); the blocked QR's panels 8 columns "
     "wide for up to 4 matrices of 768-3072 rows (1.05-1.1x), its in-aggregate updates one kernel there "
     "(1.16x at 1024^2), its input scanned on the GPU (1.01-1.03x)"),
    ("cholesky", 1, "2.18.0", "2026-10-10",
     "the measurements as Cholesky introduced them: the CPU path (spotrf('L') on a padded copy), the simd "
     "and threadgroup kernels, and the blocked path (fused sub-panels, MPS products on the lower triangle)"),
    ("lu", 1, "2.18.0", "2026-10-10",
     "the measurements as LU introduced them: the CPU path (sgetrf, sgetrs, sgetri on a padded copy) and the "
     "blocked GPU path (CPU panels with a look-ahead, GPU swaps and products, the GPU's triangular solves)"),
    ("trsm", 1, "2.18.0", "2026-10-10",
     "the measurements as the triangular solve introduced them: strsm on every core, and the blocked GPU path "
     "(diagonal blocks' inverses on the CPU, two MPS products a block)"),
]

REQUIRED = {
    "qr": {"unblocked", "reduced", "cpu", "share"},
    "eigh": {"cpu", "simd", "tg", "block", "tridiag", "ql", "ql_share", "tridiag_batch", "band",
             "cpu_vals", "simd_vals", "tg_vals", "block_vals", "tridiag_vals", "ql_vals", "ql_share_vals",
             "tridiag_batch_vals", "band_vals", "band8_vals", "band32_vals"},
    "svd": {"cpu", "jacobi", "block", "qr", "qrblock", "bidiag", "band", "gk", "gk_share", "bidiag_batch",
            "cpu_vals", "bidiag_vals", "gk_vals", "gk_share_vals", "band_vals", "band8_vals", "band32_vals",
            "bidiag_batch_vals"},
    "cholesky": {"cpu", "simd", "tg", "blocked"},
    "lu": {"cpu", "blocked", "inv_cpu", "inv_blocked", "solve_cpu", "solve_trsm", "solve_getrs"},
    "trsm": {"cpu", "blocked"},
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
    "tridiag_batch": "the tridiag_batch backend (2.17.0)",
    "tridiag_batch_vals": "the tridiag_batch backend (2.17.0)",
    "bidiag_batch": "the bidiag_batch backend (2.17.0)",
    "bidiag_batch_vals": "the bidiag_batch backend (2.17.0)",
}
# Where a backend name means something else for one decomposition.
ADDED_FOR = {
    "svd": {"cpu_vals": "the singular-value-only paths (2.7.0)",
            "band": "the band backend with singular vectors (2.15.0)"},
    "eigh": {"band": "the band backend with eigenvectors (2.17.0)"},
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
    "cholesky": ["shaders/Cholesky", "src/cholesky"] + _SHARED,
    "lu": ["shaders/LU", "src/lu", "src/transpose.h"] + _SHARED,
    "trsm": ["src/trsm"] + _SHARED,
}
# A decomposition's own table is generated from its measurements, not a change to them.
OWN_TABLE = {"qr": "src/tuned/qr.inc", "eigh": "src/tuned/eigh.inc", "svd": "src/tuned/svd.inc",
             "cholesky": "src/tuned/cholesky.inc", "lu": "src/tuned/lu.inc",
             "trsm": "src/tuned/trsm.inc"}


def run_epoch(info, op):
    """The kernel version a submission measured `op` at. Runs from before
    per-decomposition versions recorded one global epoch, 1."""
    return int((info.get("epochs") or {}).get(op, info.get("epoch", 1)))


def touches(op, path):
    """Whether a changed file may alter `op`'s timings."""
    if path == OWN_TABLE[op] or path.endswith(".md"):
        return False
    return any(path.startswith(p) for p in PATHS[op])
