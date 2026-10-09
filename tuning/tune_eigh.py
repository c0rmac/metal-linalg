#!/usr/bin/env python3
"""Measure this GPU's eigensolver routing and write up the result.

    cmake --build build --target sweep_eigh
    python3 tuning/tune_eigh.py build/sweep_eigh

Produces, in --out (default eigh-tune-results/):

    raw.csv        every timed run, so the analysis can be redone without remeasuring
    results.json   the full analysis: plot-ready series, decision surfaces, rule
                   comparison, held-out verdicts, noise floor, device identity
    report.md      a written report generated from results.json

Re-render the report from an existing run without touching the GPU:

    python3 tuning/tune_eigh.py --from eigh-tune-results/results.json

THE DECISION
------------
For a batch of `batch` symmetric N x N matrices, which of four backends should
`eigh_accelerated` use?

    cpu     MLX's eigh on the CPU (Accelerate LAPACK)
    simd    whole-matrix Jacobi, one simdgroup per matrix
    tg      whole-matrix Jacobi, one threadgroup per matrix
    block   block Jacobi, one matrix spread over the grid
    ql      tridiagonalization and implicit QL, one threadgroup per matrix
            (N up to the device's limit, reported by `sweep_eigh --policy`)

eigh.mm encodes the answer as a per-device routing policy (EighPolicy):

    simd_max_n              simd at or below this N, tg above        (GPU backend 1)
    block_min_n             block from this N on                     (GPU backend 2)
    block_min_n_batched,    optional: block also from this N once the
    block_min_batch           batch reaches this                     (GPU backend 2)
    gpu_max_n               GPU only up to this N                    (CPU routing)
    gpu_min_batch_times_n   GPU only if batch * N is at least this   (CPU routing)
    gpu_min_batch           GPU only if the batch is at least this   (CPU routing)
    values_*                the same three for eigenvalues alone     (CPU routing, eigvalsh)
    ql_min_n, ql_max_n      ql instead of the Jacobi split for N in
                            this window; 0, 0 = never                (GPU backend 3)

Every backend is also timed computing eigenvalues alone (backend names
<name>_vals): eigvalsh has its own GPU-or-CPU boundary, because the CPU then
uses a faster method (LAPACK's two-stage reduction from N = 128). Stage 3 fits
that boundary given the same GPU split.

The `ql` backend is fitted as stage 1b: a window of N, over the finished
Jacobi split, against the best GPU backend, and validated on held-out points.
The later stages see it through the GPU choice.

The `tridiag` backend (LAPACK's method with the reduction and the
back-transformation on the GPU) replaces the CPU from a threshold N, for
batches up to a cap: it solves a batch one matrix after another, while the CPU
path spreads one over every core. Stage 4 fits tridiag_min_n and
values_tridiag_min_n with tridiag_max_batch and values_tridiag_max_batch on
top of the finished rule; the
earlier stages are scored without it, since no rule they choose between can
pick it. Stage 4b fits the `band` backend's threshold for eigenvalues alone
(values_band_min_n) and its width, stage 4c its threshold with eigenvectors
(band_min_n) together with the batch cap with eigenvectors, which band shares
with tridiag and, taking batches through tridiag_batch's two stages, may need
wider; both before tridiag.

The `tridiag_batch` backend (the tridiag method for a whole batch at once,
since 2.17.0) replaces the CPU for batches of mid-size matrices: stage 5 fits
its window, N in [tridiag_batch_min_n, tridiag_batch_max_n] from batch
tridiag_batch_min_batch, and the values_ one for eigenvalues alone, on top of
everything else.

and the report ends in a row to paste into kTuned[] in eigh.mm.

RUNNING ON ANOTHER MAC
----------------------
Nothing here is specific to the machine it was written on. The device name,
core count and the policy currently in effect are read from the binary
(`sweep_eigh --policy`), not assumed. A short calibration run scales the cost
model that decides which points are too slow to measure, so a faster GPU is
probed further out, which is where its crossovers will have moved. The
candidate values for each constant are the N and batch values measured, so
`--max-n 1024` extends the search as well as the grid; the report says so if
the GPU is still ahead at the largest N measured.

This script measures every backend on an (N, batch) grid and scores every
combination of those constants by regret -- time of the chosen backend over
time of the best measured one, per point -- then reports the whole flat
region of near-optimal constants, not just the argmin. It also tries two
richer rule shapes (a batch-dependent block crossover, and a per-N CPU
boundary) and keeps them only if they survive held-out validation, which is
how the QR study caught refinements that fit one grid and failed the next.

MEASUREMENT NOTES
-----------------
Run it on an idle machine, on mains power, with Low Power Mode off. The load
average, power source and power mode are recorded at both ends of the sweep,
and a probe point is timed before and after it; a busy machine or a probe that
moved by more than 25% marks the run untrustworthy in the report, as does a
single pass. The CPU backend is the one other jobs slow down most, so a busy
machine biases the routing toward the GPU.

Otherwise as tuning/tune_qr.py: randomised order per pass, median of adaptive
repeats, two independent passes for a noise floor, min-of-repeats across
passes, one (N, batch) point per process. Points where a backend would take
more than a few seconds per call are skipped for that backend; a rule that
picks such a backend there is scored with the cost model's estimate, since a
backend known to take several seconds is not a candidate.
"""

import argparse
import csv
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import submissions as sub   # noqa: E402

BACKENDS = ("cpu", "simd", "tg", "block", "ql")
GPU_BACKENDS = ("simd", "tg", "block", "ql")
JACOBI_BACKENDS = ("simd", "tg", "block")   # stage 1 chooses among these
# The Jacobi backends are timed up to this N (since 2.17.0): above it they lost
# at every point of the M5 Pro's grid, to the CPU or the batch and large-matrix
# backends, and took 70% of a sweep's time (block alone, to N = 4096). They do
# several times the flops of the GPU's tridiag_batch and ql, which are timed
# there on the same GPU, so that holds on any Mac; a canary checks it: the
# Jacobi backends are still timed at N = CANARY_N for CANARY_BATCHES, and a
# report warns if one wins there. --full-grid times them everywhere.
JACOBI_MAX_GRID_N = 96
CANARY_N, CANARY_BATCHES = 256, (1, 64)
FULL_GRID = False

# Every point is timed in the first pass; later passes repeat only the points
# whose choice the first left open (contested): a point where, in each mode,
# the fastest backend is at least CLEAR_RATIO ahead of the next needs no
# repeat, since any rule that picks a loser there pays at least that, however
# noisy the loser's one timing (since 2.17.0; on the M5 Pro half the points,
# and half the second pass's time). --full-passes repeats every point.
CLEAR_RATIO = 1.3
FULL_PASSES = False


def contested(lines, ratio=CLEAR_RATIO):
    """Whether a point's sweep output lines (..., backend, ok, ms, p25, p75,
    reps) leave its choice open: a backend failed, or in either mode (with
    vectors, or values alone: the _vals backends) the fastest is within
    `ratio` of the next."""
    by = {}
    for line in lines:
        f = line.split(",")
        backend, ok, ms = f[-6], f[-5], float(f[-4])
        if ok != "1" or ms <= 0:
            return True
        by.setdefault(backend.endswith(VALS), []).append(ms)
    for v in by.values():
        v.sort()
        if len(v) >= 2 and v[1] < ratio * v[0]:
            return True
    return False


def jacobi_timed(N, b):
    """Whether the Jacobi backends are timed at (N, b): see JACOBI_MAX_GRID_N."""
    return FULL_GRID or N <= JACOBI_MAX_GRID_N or (N == CANARY_N and b in CANARY_BATCHES)
INF = 10 ** 9

N_LIST = [2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512]
B_LIST = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
N_QUICK = [4, 8, 16, 32, 64, 96, 128, 192, 256, 512]
B_QUICK = [1, 4, 16, 64, 256, 1024, 4096]

N_EXTRA = [768, 1024, 1536, 2048, 2560, 3072, 3584, 4096]   # added by --max-n
# Above 1024 only lone matrices and small batches are measured, with a larger
# per-call budget: enough to see whether the GPU is still behind the CPU there,
# so that gpu_max_n is a measured cap rather than the edge of the grid.
LARGE_N = 1024
LARGE_BATCHES = [1, 2, 4]
HUGE_N = 2048          # above this, lone matrices only
TRIDIAG_MIN_GRID_N = 128   # tridiag is timed from this N
BAND_MIN_GRID_N = 512      # and band (eigenvalues alone, and with eigenvectors) from this N
TB_MIN_GRID_N, TB_MAX_GRID_N = 48, 1024   # tridiag_batch is timed for N in this range
TB_MIN_GRID_BATCH = 16                     # in batches from this one
CAP_LARGE_MS = 8000.0

# Candidate values for each constant. A value only changes behaviour when it
# crosses a measured N (or batch * N), so the grids are the measured values
# themselves; configure_grids() fills them in from the data.
SIMD_MAXS  = [0, 4, 8, 12, 16, 24, 32]
BLOCK_MINS = [32, 48, 64, 96, 128, 192, 256, 384, INF]
GPU_MAX_NS = [16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, INF]
MIN_BNS    = [0, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, INF]
MIN_BATCHES = [1, 2, 4, 8, 16, 32]
B_GRID     = list(B_LIST)

# The policy in effect on the device, as the tuple the analysis works with:
#   (simd_max, block_min, block_lo, batch_hi, gpu_max_n, min_bn, min_batch)
# block_lo / batch_hi = INF means no batch-dependent block crossover. Replaced
# at run time by what `sweep_eigh --policy` reports; this is only the fallback
# for re-analysing a raw.csv that has no policy.json beside it.
CURRENT = (8, 96, INF, INF, 64, 1024, 1)
# The eigenvalues-alone boundary in effect, (gpu_max_n, min_bn, min_batch), or
# None where the device has none of its own (it then follows CURRENT's).
CURRENT_VALUES = None
# The tridiag thresholds in effect, (with eigenvectors, eigenvalues alone); 0 = never.
CURRENT_TRIDIAG = (0, 0)
CURRENT_BAND = 0           # values_band_min_n in effect
CURRENT_BAND_VEC = 0       # band_min_n in effect
# The tridiag_batch windows in effect, (min_n, max_n, min_batch) with
# eigenvectors and eigenvalues alone; max_n 0 = never.
CURRENT_TB = ((0, 0, 0), (0, 0, 0))
CURRENT_BAND_WIDTH = 0     # values_band_width in effect (0: 16)
# The band backend's widths, as their raw.csv backends (band_vals is 16 wide).
BAND_WIDTHS = {8: "band8", 16: "band", 32: "band32"}
BAND_WIDTH_TOL = 0.01      # another width replaces 16 only if better by more than this
DISAGREE_TOL = 0.03        # see near_on_disagreement
# Their batch caps, the same way round; 0 = any batch.
CURRENT_TRIDIAG_CAP = (0, 0)
# The ql window in effect, (ql_min_n, ql_max_n); (0, 0) = never.
CURRENT_QL = (0, 0)
# The largest N the ql backend takes on the device (`sweep_eigh --policy`); 0 = not timed.
QL_LIMIT = 0
# The ql window the GPU choice applies: (0, 0) while stage 1 is fitted, the
# chosen one after stage 1b.
QL = (0, 0)
# (share_min_batch, share_min_n): from which batch, and for which N, ql shares
# a batch with the CPU path (ql_share): in effect, and the one gpu_choice
# applies (batch 0 = never; fitted in stage 1c).
CURRENT_SHARE = (0, 0)
SHARE = (0, 0)
# The large-batch clause, (max_n, min_batch): the GPU also for N above
# gpu_max_n up to max_n in a batch of at least min_batch; in effect, and the
# one rule_choice applies ((0, 0) = never; fitted in stage 2 after the product
# rule). It applies with eigenvectors only: a measured eigenvalues-alone rule
# has none, as in the library.
CURRENT_BIG = (0, 0)
BIG = (0, 0)
# ql_share is timed from this batch: below it a batch is too small to share.
SHARE_MIN_GRID_BATCH = 64
NO_LIMIT = 0xFFFFFFFF   # kEighNoLimit
VALS = "_vals"

# Per-call cost model, ms. Only used to skip points that would take several
# seconds per call, and to score a rule that picks a backend at such a point.
# The constants are an M1's, deliberately pessimistic; calibrate() rescales
# them to the device.
CAP_MS = 2500.0
SCALE = {"tg": 1.0, "simd": 1.0, "block": 1.0, "cpu": 1.0, "ql": 1.0}


def configure_grids(times, pruned=False):
    """Candidate values from the measured grid. `pruned`: larger N were
    measured without the batched GPU kernels, so no cap on N is a candidate
    (the data could not tell it from the largest cap)."""
    global SIMD_MAXS, BLOCK_MINS, GPU_MAX_NS, B_GRID
    Ns = sorted({N for _, N in times})
    B_GRID = sorted({b for b, _ in times})
    SIMD_MAXS = [0] + [N for N in Ns if N <= 32]
    BLOCK_MINS = [N for N in Ns if N >= 32] + [INF]
    GPU_MAX_NS = [N for N in Ns if N >= 16] + ([] if pruned else [INF])
    global MIN_BATCHES
    MIN_BATCHES = [1] + [b for b in B_GRID if 1 < b <= 32]


def policy_values(pol):
    """The eigenvalues-alone boundary in a policy, or None if it has none."""
    if not pol.get("values_gpu_min_batch", 0):
        return None
    gm = pol["values_gpu_max_n"]
    return (INF if gm >= NO_LIMIT else gm, pol["values_gpu_min_batch_times_n"], pol["values_gpu_min_batch"])


# The backends stages 1-3 never pick: their rules choose between the GPU's
# batched kernels and the CPU, and these take over from the CPU later.
LATER = ("tridiag", "tridiag_batch", "band")


def without_tridiag(times):
    return {p: {k: v for k, v in tv.items() if k not in LATER} for p, tv in times.items()}


def only(times, keep):
    """times without the later stages' backends but those in `keep`."""
    return {p: {k: v for k, v in tv.items() if k not in LATER or k in keep} for p, tv in times.items()}


def split_values(times):
    """(times with eigenvectors, times for eigenvalues alone), each keyed by the
    plain backend names, band left out (stage 4b scores it, band_values).
    Runs from before the _vals backends give an empty second part."""
    vec, val = {}, {}
    for p, tv in times.items():
        a = {k: v for k, v in tv.items() if not k.endswith(VALS)}
        b = {k[:-len(VALS)]: v for k, v in tv.items() if k.endswith(VALS) and k != "band" + VALS}
        # any GPU backend: above JACOBI_MAX_GRID_N only the later ones are
        # timed (stages 1-3 then keep the points with a batched kernel)
        if any(k in a for k in GPU_BACKENDS + LATER):
            vec[p] = a
        if any(k in b for k in GPU_BACKENDS + LATER) and "cpu" in b:
            val[p] = b
    return vec, val


def band_values(times):
    """For eigenvalues alone, the points where band was timed, as {"cpu",
    "tridiag", "band"}, and "band8", "band32" where the other widths were."""
    out = {}
    for p, tv in times.items():
        if "band" + VALS in tv and "tridiag" + VALS in tv and "cpu" + VALS in tv:
            out[p] = {k: tv[k + VALS] for k in ("cpu", "tridiag") + tuple(BAND_WIDTHS.values()) if k + VALS in tv}
    return out


def band_width_choice(points, tol=BAND_WIDTH_TOL):
    """The band backend's width, from the points where all three widths were
    timed: the lowest geometric mean of each width's time over the best
    width's at each point, and 16, the default, unless another is better by
    more than `tol`. Returns (width, {width: geometric mean}); 16 and {} if no
    point has all three. Shared with tune_svd.py."""
    pts = [tv for tv in points.values() if all(k in tv for k in BAND_WIDTHS.values())]
    if not pts:
        return 16, {}
    score = {}
    for w, k in BAND_WIDTHS.items():
        logs = [math.log(tv[k] / min(tv[kk] for kk in BAND_WIDTHS.values())) for tv in pts]
        score[w] = math.exp(sum(logs) / len(logs))
    best = min(score, key=score.get)
    return (16 if score[16] <= score[best] * (1 + tol) else best), score


def with_band_width(points, width):
    """The points with their "band" time the given width's."""
    key = BAND_WIDTHS[width]
    return {p: dict(tv, band=tv[key]) for p, tv in points.items() if key in tv}


def near_on_disagreement(scores, choice, points, best, tol, tol_local=DISAGREE_TOL):
    """The candidates near `best` (scores: {candidate: (geomean, worst,
    over10)}, choice(candidate, point) -> the time of what it picks): within
    `tol` of best's geometric mean over all the points, as everywhere else in
    the fit, and also within `tol_local` of best's on the points where the
    two choose differently. A threshold differs from another only where they
    disagree, and scored over every point a clear loss there is diluted by
    the many points they share (stages 3b and 4b, where the band region has
    few points; docs/reading-reports.md)."""
    near = {}
    for t, v in scores.items():
        if v[0] > scores[best][0] * (1 + tol):
            continue
        diff = [p for p in points if choice(t, p) != choice(best, p)]
        if diff:
            g = math.exp(sum(math.log(choice(t, p) / choice(best, p)) for p in diff) / len(diff))
            if g > 1 + tol_local:
                continue
        near[t] = v
    return near


def policy_to_tuple(pol):
    batched = pol.get("block_min_batch", 0) > 0
    gm = pol["gpu_max_n"]
    return (pol["simd_max_n"], pol["block_min_n"],
            pol["block_min_n_batched"] if batched else INF,
            pol["block_min_batch"] if batched else INF,
            INF if gm >= NO_LIMIT else gm, pol["gpu_min_batch_times_n"],
            max(1, pol.get("gpu_min_batch", 1)))    # absent: a build without the constant


def _route_fields(gm, mb, mbatch):
    if mb >= INF:                 # never GPU
        gm, mb = 0, 0
    return f"{'kEighNoLimit' if gm >= INF else gm}, {mb}, {mbatch}"


def tuned_row(device, params, values=None, tridiag=(0, 0), ql=(0, 0), tridiag_cap=(0, 0), share=(0, 0), big=(0, 0),
              band=0, band_width=0, band_vec=0, tb=((0, 0, 0), (0, 0, 0))):
    """The line to paste into kTuned[] in eigh.mm. `values` is the
    eigenvalues-alone boundary, or None (written as 0, 0, 0: as for eigenvectors);
    `tridiag` the two tridiag thresholds (0: never), `tridiag_cap` their batch
    caps (0: any batch); `ql` the ql window (0, 0: never)."""
    s, bm, lo, bh, gm, mb, mbatch = params
    if lo >= INF or bh >= INF:
        lo, bh = 0, 0
    bm_s = str(1 << 30) if bm >= INF else str(bm)
    v = _route_fields(*values) if values else "0, 0, 0"
    return (f'{{"{device["name"]}", {device["gpu_cores"]},   {s}, {bm_s}, {lo}, {bh},   '
            f'{_route_fields(gm, mb, mbatch)},   {v},   {tridiag[0]}, {tridiag[1]}, '
            f'{tridiag_cap[0]}, {tridiag_cap[1]},   {ql[0]}, {ql[1]},   {share[0]},   {big[0]}, {big[1]},   '
            f'{band}, {band_width},   {band_vec},   {tb[0][0]}, {tb[0][1]}, {tb[0][2]},   '
            f'{tb[1][0]}, {tb[1][1]}, {tb[1][2]},   {share[1]}}},')


def env_line(params, values=None, tridiag=(0, 0), ql=(0, 0), tridiag_cap=(0, 0), share=(0, 0), big=(0, 0), band=0,
             band_width=0, band_vec=0, tb=((0, 0, 0), (0, 0, 0))):
    s, bm, lo, bh, gm, mb, mbatch = params
    extra = (f" EIGH_TRIDIAG_MIN_N={tridiag[0]} EIGH_VALUES_TRIDIAG_MIN_N={tridiag[1]}"
             f" EIGH_TRIDIAG_MAX_BATCH={tridiag_cap[0]} EIGH_VALUES_TRIDIAG_MAX_BATCH={tridiag_cap[1]}"
             f" EIGH_QL_MIN_N={ql[0]} EIGH_QL_MAX_N={ql[1]} EIGH_SHARE_MIN_BATCH={share[0]} EIGH_SHARE_MIN_N={share[1]}"
             f" EIGH_GPU_BIG_BATCH_MAX_N={big[0]} EIGH_GPU_BIG_BATCH_MIN={big[1]} EIGH_VALUES_BAND_MIN_N={band}"
             f" EIGH_VALUES_BAND_WIDTH={band_width} EIGH_BAND_MIN_N={band_vec}"
             f" EIGH_TRIDIAG_BATCH_MIN_N={tb[0][0]} EIGH_TRIDIAG_BATCH_MAX_N={tb[0][1]}"
             f" EIGH_TRIDIAG_BATCH_MIN_BATCH={tb[0][2]} EIGH_VALUES_TRIDIAG_BATCH_MIN_N={tb[1][0]}"
             f" EIGH_VALUES_TRIDIAG_BATCH_MAX_N={tb[1][1]} EIGH_VALUES_TRIDIAG_BATCH_MIN_BATCH={tb[1][2]}")
    if values:
        vg, vm, vb = values
        if vm >= INF:
            vg, vm = 0, 0
        extra += (f" EIGH_VALUES_GPU_MAX_N={NO_LIMIT if vg >= INF else vg} "
                 f"EIGH_VALUES_GPU_MIN_BATCH_TIMES_N={vm} EIGH_VALUES_GPU_MIN_BATCH={vb}")
    if lo >= INF or bh >= INF:
        lo, bh = 0, 0
    if mb >= INF:
        gm, mb = 0, 0
    return (f"EIGH_SIMD_MAX_N={s} EIGH_BLOCK_MIN_N={(1 << 30) if bm >= INF else bm} "
            f"EIGH_BLOCK_MIN_N_BATCHED={lo} EIGH_BLOCK_MIN_BATCH={bh} "
            f"EIGH_GPU_MAX_N={NO_LIMIT if gm >= INF else gm} EIGH_GPU_MIN_BATCH_TIMES_N={mb} "
            f"EIGH_GPU_MIN_BATCH={mbatch}" + extra)


def est_ms(backend, N, b):
    backend = backend[:-len(VALS)] if backend.endswith(VALS) else backend
    if backend.endswith("_share"):   # the GPU and the CPU at once
        return 0.7 * est_ms(backend[:-len("_share")], N, b)
    n3 = float(N) ** 3
    if backend.startswith("band"):   # the two-stage reduction: below tridiag's, for large N
        return 0.8 * est_ms("tridiag", N, b)
    if backend == "tridiag":    # serial over the batch; launches per column, then O(N^3)
        return b * (0.5 + 0.025 * N + 7e-9 * n3) * SCALE.get(backend, 1.0)
    if backend == "tridiag_batch":   # the batch reduced together, a threadgroup a matrix
        return (0.5 + 0.01 * N + b * (0.003 + 1.5e-8 * n3)) * SCALE.get(backend, 1.0)
    if backend == "ql":         # one threadgroup per matrix, as tg, at about a tenth of the work
        return (0.3 + 1e-6 * n3 * max(1.0, b / 8.0)) * SCALE.get(backend, 1.0)
    if backend in ("tg", "simd"):
        t = 0.3 + 1e-5 * n3 * max(1.0, b / 8.0)
    elif backend == "block":
        t = 15.0 + 5.6e-7 * n3 * b
    else:
        t = b * (0.05 + 2.2e-7 * n3)   # cpu
    return t * SCALE.get(backend, 1.0)


def backends_for(N, b):
    cap = CAP_LARGE_MS if N > LARGE_N else CAP_MS
    ks = []
    if N <= 32 and est_ms("simd", N, b) <= cap:
        ks.append("simd")
    if jacobi_timed(N, b) and est_ms("tg", N, b) <= cap:
        ks.append("tg")
    if N >= 32 and jacobi_timed(N, b) and est_ms("block", N, b) <= cap:
        ks.append("block")
    if N <= QL_LIMIT and est_ms("ql", N, b) <= cap:
        ks.append("ql")
        if b >= SHARE_MIN_GRID_BATCH:   # and sharing the batch with the CPU
            ks.append("ql_share")
    tridiag = N >= TRIDIAG_MIN_GRID_N and est_ms("tridiag", N, b) <= cap
    tb = TB_MIN_GRID_N <= N <= TB_MAX_GRID_N and b >= TB_MIN_GRID_BATCH and est_ms("tridiag_batch", N, b) <= cap
    if not ks and not tridiag and not tb:   # (no GPU backend timed here: nothing to compare the CPU with)
        return []
    if tridiag or tb or est_ms("cpu", N, b) <= cap:   # the reference wherever tridiag(_batch) is timed
        ks.append("cpu")
    if tridiag:
        ks.append("tridiag")
    if tb:
        ks.append("tridiag_batch")
    vals = [k + VALS for k in ks]        # each again for eigenvalues alone
    if "tridiag" in ks and N >= BAND_MIN_GRID_N:
        # and the band backend, the region values_band_min_n decides, at each
        # width; with eigenvectors (band_min_n), 16 wide
        vals += [k + VALS for k in BAND_WIDTHS.values()] + ["band"]
    return ks + vals


def point_grid(quick=False, max_n=512):
    Ns = list(N_QUICK if quick else N_LIST) + [n for n in N_EXTRA if n <= max_n]
    Ns = [n for n in Ns if n <= max(max_n, 2)]
    Bs = B_QUICK if quick else B_LIST
    pts = []
    for N in Ns:
        for b in ([1] if N > HUGE_N else LARGE_BATCHES if N > LARGE_N else Bs):
            if not sub.fits_memory(b * N * N):     # see submissions.MEMORY_FRACTION
                continue
            ks = backends_for(N, b)
            if ks:
                pts.append((b, N, ks))
    return pts


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

def run_one(binary, job, limit, attempts=3):
    """One (batch, N) point, all its backends. Retries on crash so a transient
    failure does not silently become a missing data point."""
    b, N, ks = job
    for _ in range(attempts):
        try:
            r = subprocess.run([binary, str(b), str(N), ",".join(ks)],
                               capture_output=True, text=True, timeout=limit)
            lines = [l for l in r.stdout.splitlines() if l.strip()]
            if r.returncode == 0 and len(lines) == len(ks):
                return lines
        except subprocess.TimeoutExpired:
            break
    return [f"{b},{N},{k},0,0,0,0,0" for k in ks]


def sweep(binary, pts, passes, limit, out_csv):
    runs = sum(len(ks) for _, _, ks in pts)
    eff = passes if FULL_PASSES else 1 + 0.5 * (passes - 1)   # later passes: the contested half, about
    est_total = sum(min(est_ms(k, N, b), CAP_MS) * 6 for b, N, ks in pts for k in ks) / 1000 * eff
    print(f"  {len(pts)} points, {runs} backend timings per pass x {passes} passes; "
          f"rough estimate {est_total/60 + len(pts) * eff * 0.5 / 60:.0f} min",
          file=sys.stderr)
    done = 0
    total = len(pts) * passes
    t0 = time.time()
    jobs = list(pts)
    with open(out_csv, "w") as fh:
        fh.write("pass,batch,N,backend,ok,ms,p25,p75,reps\n")
        for p in range(passes):
            order = list(jobs)
            random.Random(9000 + p).shuffle(order)   # independent shuffle per pass
            again = []
            for job in order:
                lines = run_one(binary, job, limit)
                for line in lines:
                    fh.write(f"{p},{line}\n")
                fh.flush()
                if p == 0 and (FULL_PASSES or contested(lines)):
                    again.append(job)
                done += 1
                if done % 20 == 0 or done == total:
                    el = time.time() - t0
                    eta = el / done * (total - done)
                    print(f"    {done}/{total}  elapsed {el/60:.1f}m  eta {eta/60:.1f}m",
                          file=sys.stderr)
            if p == 0 and passes > 1:
                jobs = again
                total = done + len(jobs) * (passes - 1)
                print(f"  later passes: the {len(jobs)} of {len(pts)} points without a clear winner", file=sys.stderr)


def load(paths):
    """-> (times[(b,N)][backend], repeats, submissions); see submissions.combine."""
    times, repeats, subs = sub.combine(paths, lambda r: (int(r["batch"]), int(r["N"])))
    gpu = GPU_BACKENDS + LATER   # any GPU backend: above JACOBI_MAX_GRID_N only the later ones are timed
    times = {p: v for p, v in times.items() if any(k.replace(VALS, "") in gpu for k in v)}
    return times, repeats, subs

def query_policy(binary):
    """The device and the routing policy the library resolved for it."""
    r = subprocess.run([binary, "--policy"], capture_output=True, text=True, timeout=60)
    if r.returncode != 0 or not r.stdout.strip():
        sys.exit(f"{binary} --policy failed; rebuild sweep_eigh")
    return json.loads(r.stdout.strip().splitlines()[-1])


def machine_state():
    """Load, power source and power mode. Recorded with every run, because a
    sweep on a busy or throttled machine measures the other jobs as much as
    the eigensolver, and the CPU backend suffers most. Process names are
    deliberately not recorded: reports get committed."""
    st = {"load_1m": None, "cpus": os.cpu_count() or 0, "power": "unknown",
          "low_power_mode": None, "busy_processes": None}
    try:
        st["load_1m"] = os.getloadavg()[0]
    except OSError:
        pass
    try:
        out = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True, timeout=10).stdout
        st["power"] = "battery" if "Battery Power" in out else "mains" if "AC Power" in out else "unknown"
        out = subprocess.run(["pmset", "-g"], capture_output=True, text=True, timeout=10).stdout
        # `lowpowermode 1` on older macOS; `powermode 1` (0 automatic, 2 high
        # power) on newer releases, which dropped the former.
        m = re.search(r"^\s*lowpowermode\s+(\d)", out, re.M)
        if m:
            st["low_power_mode"] = int(m.group(1))
        else:
            m = re.search(r"^\s*powermode\s+(\d)", out, re.M)
            if m:
                st["low_power_mode"] = 1 if int(m.group(1)) == 1 else 0
        out = subprocess.run(["ps", "-Ao", "pcpu"], capture_output=True, text=True, timeout=10).stdout
        st["busy_processes"] = sum(1 for l in out.splitlines()[1:] if l.strip() and float(l) >= 50.0)
    except Exception:
        pass
    return st


def state_problems(st):
    """Reasons this machine state makes a sweep untrustworthy, if any."""
    out = []
    if st.get("load_1m") is not None and st.get("cpus"):
        if st["load_1m"] > max(2.0, st["cpus"] / 2.0):
            out.append(f"load average {st['load_1m']:.1f} on {st['cpus']} CPUs"
                       + (f", {st['busy_processes']} other processes above 50% CPU"
                          if st.get("busy_processes") else ""))
    if st.get("low_power_mode") == 1:
        out.append("Low Power Mode is on")
    return out


PROBE = (64, 128, ["tg", "block", "cpu"])   # (batch, N, backends)


def probe(binary, limit):
    """Min of three runs of the probe point, per backend."""
    out = {}
    for _ in range(3):   # min of three: interference is one-sided, as in the sweep
        for line in run_one(binary, PROBE, limit):
            f = line.split(",")
            if f[3] == "1" and float(f[4]) > 0:
                out[f[2]] = min(out.get(f[2], float("inf")), float(f[4]))
    return out


def calibrate(binary, limit):
    """Rescale the cost model to this device from one batched probe point.

    block and cpu scale close to linearly in batch, so their model can be
    trusted in both directions and a faster device is probed further out. The
    whole-matrix kernel is memory-bound once several large matrices run
    together, which the model does not capture, so it is only ever scaled up.
    """
    out = probe(binary, limit)
    for k, ms in out.items():
        ratio = ms / est_ms(k, PROBE[1], PROBE[0])
        lo = 1.0 if k == "tg" else 0.05
        SCALE[k] = min(4.0, max(lo, ratio))
    SCALE["simd"] = SCALE["tg"]
    return {"probe": {"batch": PROBE[0], "N": PROBE[1], "ms": out}, "scale": dict(SCALE)}


def drift(before, after):
    """Did the machine change state during the sweep? Ratio of the probe point
    after the sweep to before it, per backend. A laptop that started cool and
    ended throttled, or picked up a background job, moves these well away from
    1, and the timings on either side of the change are not comparable."""
    ratios = {k: after[k] / before[k] for k in before if k in after and before[k] > 0}
    worst = max((max(r, 1.0 / r) for r in ratios.values()), default=1.0)
    return {"before_ms": before, "after_ms": after, "ratio": ratios,
            "worst": worst, "ok": worst <= 1.25}


def load_sidecar(raw_paths):
    """policy.json written beside raw.csv at sweep time, if there is one."""
    side = os.path.join(os.path.dirname(os.path.abspath(raw_paths[0])), "policy.json")
    if os.path.exists(side):
        return json.load(open(side))
    return None


# ---------------------------------------------------------------------------
# Rules and scoring
# ---------------------------------------------------------------------------
#
# The decision has two stages, mirrored from eigh.mm:
#
#   stage 1  which GPU backend, if the GPU is used:
#            block if N >= block_min or (N >= block_lo and batch >= batch_hi),
#            else simd if N <= simd_max, else tg
#   stage 2  GPU or CPU: GPU iff N <= gpu_max_n, batch * N >= min_bn and batch >= min_batch
#
# Stage 1 is fitted against the best *GPU* backend at each point, as if there
# were no CPU. Fitting it jointly with stage 2 would let the CPU routing hide
# the crossover -- on a GPU where the CPU wins above some N, every block
# crossover above that N scores the same -- and the split would be untuned for
# a forced-GPU call or a bigger GPU. Stage 2 is then fitted given stage 1.

def gpu_choice(split, N, b, ql=None, share=None):
    """The GPU backend: ql inside its window (QL unless `ql` is given),
    ql_share from the batch and N of SHARE (or `share`, (min_batch, min_n);
    batch 0 = never), else the Jacobi split."""
    lo, hi = QL if ql is None else ql
    if hi and lo <= N <= (min(hi, QL_LIMIT) if QL_LIMIT else hi):
        sb, sn = SHARE if share is None else share
        return "ql_share" if sb and b >= sb and N >= sn else "ql"
    simd_max, block_min, block_lo, batch_hi = split
    if N >= block_min or (N >= block_lo and b >= batch_hi):
        return "block"
    return "simd" if N <= simd_max else "tg"


def rule_choice(params, N, b):
    """params = (simd_max, block_min, block_lo, batch_hi, gpu_max_n, min_bn, min_batch)."""
    gpu_max_n, min_bn, min_batch = params[4], params[5], params[6]
    bn, bb = BIG
    if bb and gpu_max_n and gpu_max_n < N <= bn and b >= bb:
        return gpu_choice(params[:4], N, b)   # the large-batch clause, above gpu_max_n
    if N > gpu_max_n or b * N < min_bn or b < min_batch:
        return "cpu"
    return gpu_choice(params[:4], N, b)


def with_big(big, fn):
    """fn() with the large-batch clause `big` in effect."""
    global BIG
    saved, BIG = BIG, tuple(big)
    try:
        return fn()
    finally:
        BIG = saved


def fit_big(params, times, tol):
    """The large-batch clause over the product rule `params`: N up to a
    measured N above gpu_max_n, from a measured batch. (0, 0) unless it improves
    the geomean regret by more than `tol`; inside that, the smallest worst case,
    then the best geomean, then the larger batch. -> ((max_n, min_batch), scores)."""
    ns = sorted({N for (_, N) in times})
    bs = sorted({b for (b, _) in times if b > 1})
    cands = [(0, 0)] + [(n, b) for n in ns if n > params[4] for b in bs]
    scores = {c: with_big(c, lambda c=c: _score3(score_rule(params, times))) for c in cands}
    base = scores[(0, 0)][0]
    best = min(v[0] for v in scores.values())
    if best >= base / (1 + tol):
        return (0, 0), scores
    near = {c: v for c, v in scores.items() if v[0] <= best * (1 + tol)}
    return min(near, key=lambda c: (near[c][1], near[c][0], -c[1], c[0])), scores


def evaluate(choice_fn, times):
    """Regret statistics of a rule over the measured points (oracle = best
    measured backend among those present in `times`)."""
    logs, worst, over10, est_picks = [], 1.0, 0, 0
    tot_chosen, tot_best = 0.0, 0.0
    per_point = {}
    for (b, N), tv in times.items():
        best = min(tv.values())
        k = choice_fn(N, b)
        est = k not in tv
        if not est:
            r = tv[k] / best
        else:
            # The backend the rule picks was not timed here (the cost model
            # skipped it), so its regret is the model's guess: it counts in
            # the geomean, is flagged per point, and is kept out of `worst`.
            r = max(1.0, est_ms(k, N, b) / best)
            est_picks += 1
        per_point[(b, N)] = (k, r, est)
        logs.append(math.log(r))
        if not est:
            worst = max(worst, r)
        over10 += r > 1.10
        tot_chosen += r * best
        tot_best += best
    n = len(logs)
    return {"geomean": math.exp(math.fsum(logs) / n) if n else 1.0, "worst": worst,
            "over10": over10, "total_ratio": tot_chosen / tot_best if tot_best else 1.0,
            "estimated_picks": est_picks, "n": n, "_per_point": per_point}


def _strip(e):
    return {k: v for k, v in e.items() if not k.startswith("_")}


def gpu_only(times, backends=GPU_BACKENDS):
    out = {}
    for p, tv in times.items():
        g = {k: v for k, v in tv.items() if k in backends}
        if g:
            out[p] = g
    return out


def jacobi_only(times):
    return gpu_only(times, JACOBI_BACKENDS)


def score_split(split, gtimes):
    return evaluate(lambda N, b: gpu_choice(split, N, b), gtimes)


def score_rule(params, times):
    return evaluate(lambda N, b: rule_choice(params, N, b), times)


def fit_split(gtimes):
    """Stage 1 grid: (simd_max, block_min), no batch term."""
    out = {}
    for s in SIMD_MAXS:
        for bm in BLOCK_MINS:
            e = score_split((s, bm, INF, INF), gtimes)
            out[(s, bm, INF, INF)] = (e["geomean"], e["worst"], e["over10"])
    return out


def fit_split_batch(base, gtimes):
    """Stage 1 refinement: (block_min, block_lo, batch_hi) refitted jointly, with
    simd_max from the base fit. block_min is refitted rather than inherited:
    without a batch term it settles where the large-batch wins outweigh the
    small-batch losses, and the batch term exists precisely to separate the two."""
    s = base[0]
    out = {}
    for bm in BLOCK_MINS:
        for lo in BLOCK_MINS:
            if lo >= bm:
                continue
            for bh in B_GRID:
                e = score_split((s, bm, lo, bh), gtimes)
                out[(s, bm, lo, bh)] = (e["geomean"], e["worst"], e["over10"])
    return out


def ql_candidates(gtimes):
    """The ql windows worth scoring: (0, 0), never, and every pair of N at
    which ql was timed. A bound only matters where it crosses a measured N."""
    Ns = sorted({N for (b, N), tv in gtimes.items() if "ql" in tv})
    return [(0, 0)] + [(lo, hi) for i, lo in enumerate(Ns) for hi in Ns[i:]]


def fit_ql(split, gtimes):
    """Stage 1b grid: the ql window over the Jacobi split, against the best
    GPU backend, ql included."""
    out = {}
    for w in ql_candidates(gtimes):
        e = evaluate(lambda N, b, w=w: gpu_choice(split, N, b, ql=w), gtimes)
        out[w] = (e["geomean"], e["worst"], e["over10"])
    return out


def fit_routing(split, times):
    """Stage 2 grid: (gpu_max_n, min_bn, min_batch) given the GPU split."""
    out = {}
    for gm in GPU_MAX_NS:
        for mb in MIN_BNS:
            for mbatch in MIN_BATCHES:
                p = split + (gm, mb, mbatch)
                e = score_rule(p, times)
                out[p] = (e["geomean"], e["worst"], e["over10"])
    return out


def near_optimal(scores, tol):
    best = min(g for g, _, _ in scores.values())
    return best, {p: v for p, v in scores.items() if v[0] <= best * (1 + tol)}


def choose(near, current, current_score=None, best=None, tol=0.005):
    """Inside the flat region, keep the policy in effect unless another
    combination improves the worst case by more than 10%; a noise-level
    geomean gain is not worth a constant that moves between runs.

    The policy in effect need not be one of the candidates (its values may lie
    between measured sizes), so its own score can be passed in; it then counts
    as inside the region if it scores within `tol` of the best."""
    best_worst = min(near, key=lambda p: (near[p][1], near[p][0]))
    score = near.get(current, current_score)
    if score is not None:
        inside = current in near or (best is not None and score[0] <= best * (1 + tol))
        if inside and score[1] <= near[best_worst][1] * 1.10:
            return current
    return best_worst


def _score3(e):
    return (e["geomean"], e["worst"], e["over10"])


def fit_with_clause(sc2, gm_at, fit_big, score, with_big, current, current_score, tol):
    """The product rule and the large-batch clause, fitted together: for each
    cap (p[gm_at]) of the stage-2 grid `sc2`, the rule fitted without the
    clause, the clause over it (fit_big(p) -> (max, min_batch)), and the rule
    fitted again given that clause; then `choose` over every combination, keyed
    (params, big), with the policy in effect as `current`. Fitting the rule
    first and the clause over it alone can miss a lower cap plus the clause
    beating a higher cap without one. -> ((params, big), {(params, big): score3})."""
    combos = {}
    for gm in sorted({p[gm_at] for p in sc2}):
        sub = {p: v for p, v in sc2.items() if p[gm_at] == gm}
        p0 = min(sub, key=lambda p: sub[p])
        big = tuple(fit_big(p0))
        if big == (0, 0):
            combos[(p0, big)] = sub[p0]
            continue
        sub2 = with_big(big, lambda: {p: score(p) for p in sub})
        p1 = min(sub2, key=lambda p: sub2[p])
        combos[(p1, big)] = sub2[p1]
    best, near = near_optimal(combos, tol)
    return choose(near, current, current_score, best, tol), combos


def split_points(times, seed=7):
    """Half the batches at every N in each half, so both halves see every N."""
    by_n = defaultdict(list)
    for (b, N) in times:
        by_n[N].append(b)
    rng = random.Random(seed)
    train, test = {}, {}
    for N, bs in by_n.items():
        bs = sorted(bs)
        rng.shuffle(bs)
        for i, b in enumerate(bs):
            (train if i % 2 == 0 else test)[(b, N)] = times[(b, N)]
    return train, test


def bootstrap(base_fn, ref_fn, test, iters=2000, seed=11):
    """Does `ref` beat `base` on the held-out half beyond sampling noise?
    Resamples the test points; reports how often the refinement's geometric
    mean regret is lower and the gain's median and 5th percentile."""
    lb = evaluate(base_fn, test)["_per_point"]
    lr = evaluate(ref_fn, test)["_per_point"]
    keys = list(test.keys())
    n = len(keys)
    dl = [math.log(lb[k][1]) - math.log(lr[k][1]) for k in keys]
    rng = random.Random(seed)
    gains = []
    for _ in range(iters):
        gains.append(math.fsum(dl[rng.randrange(n)] for _ in range(n)) / n)
    gains.sort()
    return {"p_better": sum(g > 0 for g in gains) / iters,
            "gain_median": math.exp(gains[iters // 2]) - 1.0,
            "gain_p05": math.exp(gains[int(0.05 * iters)]) - 1.0}


def justified(boot, base_test, ref_test):
    return boot["p_better"] >= 0.95 and ref_test["worst"] <= base_test["worst"] * 1.02


def refine_cpu_table(params, train, test):
    """Stage 2 refinement: a per-N minimum batch for the GPU, read off the
    training half (smallest batch at which the rule's GPU backend beats the
    CPU), with the product rule as fallback for an N not seen."""
    gm, mb, mbatch = params[4], params[5], params[6]
    split = params[:4]
    table = {}
    by_n = defaultdict(list)
    for (b, N), tv in train.items():
        by_n[N].append((b, tv))
    for N, rows in by_n.items():
        wins = sorted(b for b, tv in rows
                      if gpu_choice(split, N, b) in tv and tv.get("cpu", INF) >= tv[gpu_choice(split, N, b)])
        table[N] = wins[0] if wins else INF

    def ch(N, b):
        thr = table.get(N)
        if thr is None:
            if N > gm or b * N < mb or b < mbatch:
                return "cpu"
        elif b < thr:
            return "cpu"
        return gpu_choice(split, N, b)

    return ch, {str(N): (v if v < INF else None) for N, v in sorted(table.items())}


def noise_floor(repeats):
    """Pass-to-pass ratio per (point, backend), bucketed by runtime."""
    buckets = defaultdict(list)
    allr = []
    for key, v in repeats.items():
        if len(v) < 2 or min(v) <= 0:
            continue
        ratio = max(v) / min(v)
        allr.append(ratio)
        t = min(v)
        lab = ("<1 ms" if t < 1 else "1-3 ms" if t < 3 else "3-10 ms" if t < 10 else
               "10-30 ms" if t < 30 else "30-100 ms" if t < 100 else ">100 ms")
        buckets[lab].append(ratio)

    def q(a, p):
        a = sorted(a)
        return a[min(len(a) - 1, int(p * (len(a) - 1)))] if a else 0.0

    order = ["<1 ms", "1-3 ms", "3-10 ms", "10-30 ms", "30-100 ms", ">100 ms"]
    return {"overall": {"n": len(allr), "median": q(allr, .5), "p90": q(allr, .9),
                        "max": max(allr) if allr else 0},
            "by_runtime": [{"bucket": lab, "n": len(buckets[lab]), "median": q(buckets[lab], .5),
                            "p90": q(buckets[lab], .9), "max": max(buckets[lab])}
                           for lab in order if buckets.get(lab)]}


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def _band(near, idx):
    vals = [p[idx] for p in near]
    return [min(vals), max(vals)]


def _curve(score_fn, base, idx, grid):
    out = []
    for v in grid:
        p = list(base)
        p[idx] = v
        e = score_fn(tuple(p))
        out.append({"value": v if v < INF else None, "geomean": e["geomean"],
                    "worst": e["worst"], "over10": e["over10"]})
    return out


def tridiag_choice(params, th, N, b):
    """`th` is a threshold N, or (threshold, batch cap) with cap 0 for any batch."""
    th, cap = th if isinstance(th, tuple) else (th, 0)
    k = rule_choice(params, N, b)
    return "tridiag" if (k == "cpu" and th and N >= th and (not cap or b <= cap)) else k


def tridiag_points(times):
    """The points where tridiag was timed: the region its threshold decides.
    Scored over every point, the few large matrices it wins on by 2-6x would
    move the geometric mean by less than the tolerance, and the threshold would
    stay wherever it was."""
    return {p: tv for p, tv in times.items() if "tridiag" in tv}


def fit_tridiag(params, times, current, tol):
    """Stage 4: the tridiag threshold and batch cap given the whole rule
    `params`, over the measured N from TRIDIAG_MIN_GRID_N (and 0, never) and
    the measured batches (and 0, any). -> ((threshold, cap), scores). `times`
    are the points where tridiag was timed. Inside the flat region the policy
    in effect stays; otherwise the smallest worst case, then the least use of
    the backend."""
    ths = sorted({N for (_, N) in times if N >= TRIDIAG_MIN_GRID_N})
    caps = [0] + sorted({b for (b, _) in times})
    cands = [(0, 0)] + [(th, cap) for th in ths for cap in caps]
    scores = {c: _score3(evaluate(lambda N, b, c=c: tridiag_choice(params, c, N, b), times)) for c in cands}
    best = min(g for g, _, _ in scores.values())
    near = {c: v for c, v in scores.items() if v[0] <= best * (1 + tol)}
    current = tuple(current)
    if current in near:
        return current, scores
    return min(near, key=lambda c: (near[c][1], -c[0] if c[0] else 0, c[1] if c[1] else INF)), scores


def analyse(times, repeats, device, tol=0.005, drift_info=None, states=None):
    global QL, SHARE, BIG
    QL, SHARE, BIG = (0, 0), (0, 0), (0, 0)    # stage 1 is the Jacobi split alone
    btimes = band_values(times)            # stage 4b's, before the _vals names go
    times_full, vtimes_full = split_values(times)
    times, vtimes = without_tridiag(times_full), without_tridiag(vtimes_full)
    # Stages 1-3 on the points where a batched GPU kernel was timed (the
    # grid times the Jacobi backends to JACOBI_MAX_GRID_N, ql to its limit)
    times = {p: tv for p, tv in times.items() if any(k in tv for k in GPU_BACKENDS)}
    vtimes = {p: tv for p, tv in (vtimes or {}).items() if any(k in tv for k in GPU_BACKENDS)}
    configure_grids(times, pruned=max((N for _, N in times_full), default=0) > max((N for _, N in times), default=0))
    single_pass = not any(len(v) >= 2 for v in repeats.values())
    res = {"device": device, "n_points": len(times), "single_pass": single_pass, "drift": drift_info,
           "grid": {"N": sorted({N for _, N in times}), "batch": sorted({b for b, _ in times})},
           "tolerance": tol, "current": list(CURRENT)}
    gtimes = gpu_only(times)
    train, test = split_points(times)
    # Stage 1 picks among the Jacobi backends only: the ql window is fitted
    # over it in stage 1b, and its own oracle would otherwise shift theirs.
    jtimes = jacobi_only(times)
    gtrain, gtest = jacobi_only(train), jacobi_only(test)
    floor = noise_floor(repeats)

    # ---- stage 1: GPU split, against the best GPU backend ----
    cur_split = CURRENT[:4]
    sc1 = fit_split(jtimes)
    best1, near1 = near_optimal(sc1, tol)
    split = choose(near1, cur_split, _score3(score_split(cur_split, jtimes)), best1, tol)
    s1 = {
        "n_points": len(jtimes),
        "current": _strip(score_split(cur_split, jtimes)),
        "chosen": _strip(score_split(split, jtimes)),
        "chosen_params": list(split),
        "band": {"n_near_optimal": len(near1), "simd_max": _band(near1, 0),
                 "block_min": _band(near1, 1), "current_in_band": cur_split in near1},
        "curves": {"simd_max": _curve(lambda p: score_split(p, jtimes), split, 0, SIMD_MAXS),
                   "block_min": _curve(lambda p: score_split(p, jtimes), split, 1, BLOCK_MINS)},
    }
    # refinement: batch-dependent block crossover, fitted on train, validated on test
    tr_sc1 = fit_split(gtrain)
    _, tr_near1 = near_optimal(tr_sc1, tol)
    tr_split = choose(tr_near1, None)     # the two-constant fit itself, not the policy in effect
    tr_sc1b = fit_split_batch(tr_split, gtrain)
    if tr_sc1b:
        tr_splitb = min(tr_sc1b, key=lambda p: (tr_sc1b[p][0], tr_sc1b[p][1]))
        base_fn = lambda N, b: gpu_choice(tr_split, N, b)
        ref_fn = lambda N, b: gpu_choice(tr_splitb, N, b)
        boot = bootstrap(base_fn, ref_fn, gtest)
        a_test = _strip(evaluate(base_fn, gtest))
        r_test = _strip(evaluate(ref_fn, gtest))
        ok = justified(boot, a_test, r_test)
        s1["holdout"] = {
            "train_points": len(gtrain), "test_points": len(gtest),
            "base": {"params": list(tr_split), "train": _strip(evaluate(base_fn, gtrain)), "test": a_test},
            "batch_block": {"params": {"block_lo": tr_splitb[2], "batch_hi": tr_splitb[3]},
                            "train": _strip(evaluate(ref_fn, gtrain)), "test": r_test,
                            "bootstrap": boot, "justified": ok},
        }
        if ok:
            # Adopt the refinement, refitted on all GPU points; report its own
            # flat region, since the earlier band no longer applies.
            sc1b = fit_split_batch(split, jtimes)
            _, near1b = near_optimal(sc1b, tol)
            split = min(near1b, key=lambda p: (near1b[p][1], near1b[p][0]))
            s1["chosen_params"] = list(split)
            s1["chosen"] = _strip(score_split(split, jtimes))
            s1["band_batched"] = {"n_near_optimal": len(near1b), "block_min": _band(near1b, 1),
                                  "block_lo": _band(near1b, 2), "batch_hi": _band(near1b, 3)}
            s1["curves"]["block_min"] = _curve(lambda p: score_split(p, jtimes), split, 1, BLOCK_MINS)
            s1["curves"]["block_lo"] = _curve(lambda p: score_split(p, jtimes), split, 2,
                                              [v for v in BLOCK_MINS if v < split[1]])
            s1["batch_curve"] = _curve(lambda p: score_split(p, jtimes), split, 3, B_GRID)
    res["stage1"] = s1

    # ---- stage 1b: the ql window over the split ----
    ql = (0, 0)
    have_ql = any("ql" in tv for tv in gtimes.values())
    if have_ql:
        scq = fit_ql(split, gtimes)
        bestq, nearq = near_optimal(scq, tol)
        cur_q = tuple(CURRENT_QL)
        cur_score = _score3(evaluate(lambda N, b: gpu_choice(split, N, b, ql=cur_q), gtimes))
        ql = choose(nearq, cur_q, cur_score, bestq, tol)
        tr_g, te_g = gpu_only(train), gpu_only(test)
        scq_tr = fit_ql(split, tr_g)
        _, nearq_tr = near_optimal(scq_tr, tol)
        ql_tr = choose(nearq_tr, (0, 0))
        fn = lambda N, b: gpu_choice(split, N, b, ql=ql)
        never = lambda N, b: gpu_choice(split, N, b, ql=(0, 0))
        res["stage1b"] = {
            "n_points": len(gtimes),
            "chosen": list(ql),
            "current": list(cur_q),
            "with": _strip(evaluate(fn, gtimes)),
            "without": _strip(evaluate(never, gtimes)),
            "band": {"n_near_optimal": len(nearq),
                     "ql_min_n": [min(p[0] for p in nearq), max(p[0] for p in nearq)],
                     "ql_max_n": [min(p[1] for p in nearq), max(p[1] for p in nearq)]},
            "holdout": {"train_points": len(tr_g), "test_points": len(te_g), "fitted": list(ql_tr),
                        "test": _strip(evaluate(lambda N, b: gpu_choice(split, N, b, ql=ql_tr), te_g)),
                        "without_test": _strip(evaluate(never, te_g))},
            "speedup_vs_jacobi": sorted([[N, b, min(v for k, v in tv.items() if k in JACOBI_BACKENDS) / tv["ql"]]
                                         for (b, N), tv in gtimes.items()
                                         if "ql" in tv and any(k in tv for k in JACOBI_BACKENDS)],
                                        key=lambda x: (x[0], x[1])),
            "limit": QL_LIMIT,
        }
    QL = tuple(ql)
    res["ql_chosen"] = list(ql)
    res["current_ql"] = list(CURRENT_QL)

    # ---- stage 1c: from which batch, and for which N, ql shares the batch with
    # the CPU path, against the best GPU backend, the shared one included, on
    # the points where it was timed and ql is the GPU's choice. Both: since
    # 2.17.0 the ql kernel in registers alone wins the small N that sharing
    # takes at large N, so a batch threshold alone shared nowhere
    share = (0, 0)
    sht = {p: tv for p, tv in gpu_only(times, GPU_BACKENDS + ("ql_share",)).items()
           if "ql_share" in tv and gpu_choice(split, p[1], p[0]) == "ql"}
    if sht:
        cands = [(0, 0)] + [(b, n) for b in sorted({b for (b, _) in sht}) for n in [0] + sorted({N for (_, N) in sht})]
        scs = {c: _score3(evaluate(lambda N, b, c=c: gpu_choice(split, N, b, share=c), sht)) for c in cands}
        bests, nears = near_optimal(scs, tol)
        cur_s = _score3(evaluate(lambda N, b: gpu_choice(split, N, b, share=CURRENT_SHARE), sht))
        share = choose(nears, CURRENT_SHARE, cur_s, bests, tol)
        res["stage1c"] = {
            "n_points": len(sht), "chosen": list(share), "current": list(CURRENT_SHARE),
            "with": _strip(evaluate(lambda N, b: gpu_choice(split, N, b, share=share), sht)),
            "without": _strip(evaluate(lambda N, b: gpu_choice(split, N, b, share=(0, 0)), sht)),
            "curve": [[c[0], c[1], v[0], v[1]] for c, v in sorted(scs.items())],
            "speedup_vs_ql": sorted([[N, b, tv["ql"] / tv["ql_share"]] for (b, N), tv in sht.items()
                                     if "ql" in tv], key=lambda x: (x[0], x[1])),
        }
    SHARE = share
    res["share_chosen"] = list(share)

    # ---- stage 2: CPU routing given the split ----
    cur_route = CURRENT[4:]
    cur_e = with_big(CURRENT_BIG, lambda: score_rule(split + cur_route, times))
    sc2 = fit_routing(split, times)                 # without the clause: the band and curves
    best2, near2 = near_optimal(sc2, tol)
    (params, big), _ = fit_with_clause(sc2, 4, lambda p: fit_big(p, times, tol)[0],
                                       lambda p: _score3(score_rule(p, times)), with_big,
                                       (split + cur_route, tuple(CURRENT_BIG)), _score3(cur_e), tol)
    without_big = score_rule(params, times)
    BIG = big
    s2 = {
        "big": {"chosen": list(big), "current": list(CURRENT_BIG), "without": _strip(without_big),
                "with": _strip(score_rule(params, times))},
        "current": _strip(cur_e),
        "chosen": _strip(score_rule(params, times)),
        "chosen_params": list(params[4:]),
        "band": {"n_near_optimal": len(near2), "gpu_max_n": _band(near2, 4),
                 "min_bn": _band(near2, 5), "min_batch": _band(near2, 6),
                 "current_in_band": (split + cur_route) in near2},
        "curves": {"gpu_max_n": _curve(lambda p: score_rule(p, times), params, 4, GPU_MAX_NS),
                   "min_bn": _curve(lambda p: score_rule(p, times), params, 5, MIN_BNS),
                   "min_batch": _curve(lambda p: score_rule(p, times), params, 6, MIN_BATCHES)},
    }
    tr_sc2 = fit_routing(split, train)
    _, tr_near2 = near_optimal(tr_sc2, tol)
    tr_params = choose(tr_near2, split + cur_route)
    base_fn = lambda N, b: rule_choice(tr_params, N, b)
    ref_fn, table = refine_cpu_table(tr_params, train, test)
    boot = bootstrap(base_fn, ref_fn, test)
    a_test = _strip(evaluate(base_fn, test))
    r_test = _strip(evaluate(ref_fn, test))
    s2["holdout"] = {
        "train_points": len(train), "test_points": len(test),
        "base": {"params": list(tr_params[4:]), "train": _strip(evaluate(base_fn, train)), "test": a_test},
        "cpu_table": {"params": {"min_batch_by_n": table}, "train": _strip(evaluate(ref_fn, train)),
                      "test": r_test, "bootstrap": boot, "justified": justified(boot, a_test, r_test)},
    }
    res["stage2"] = s2

    # ---- stage 3: CPU routing for eigenvalues alone, given the same split ----
    values = None
    if vtimes:
        eigh_boundary = _strip(score_rule(params, vtimes))   # what eigvalsh would do without its own
        BIG_EIGH, BIG = BIG, (0, 0)                          # a rule of its own has no clause
        cur_v = CURRENT_VALUES or cur_route          # none of its own: it follows stage 2's
        sc3 = fit_routing(split, vtimes)
        best3, near3 = near_optimal(sc3, tol)
        vfull = choose(near3, split + cur_v, _score3(score_rule(split + cur_v, vtimes)), best3, tol)
        values = tuple(vfull[4:])
        vtrain, vtest = split_points(vtimes)
        _, tr_near3 = near_optimal(fit_routing(split, vtrain), tol)
        tr_v = choose(tr_near3, split + cur_v)
        res["stage3"] = {
            "n_points": len(vtimes),
            "current": _strip(score_rule(split + cur_v, vtimes)),
            "current_is_eigh_boundary": CURRENT_VALUES is None,
            "eigh_boundary": eigh_boundary,
            "chosen": _strip(score_rule(vfull, vtimes)),
            "chosen_params": list(values),
            "band": {"n_near_optimal": len(near3), "gpu_max_n": _band(near3, 4),
                     "min_bn": _band(near3, 5), "min_batch": _band(near3, 6)},
            "holdout": {"train_points": len(vtrain), "test_points": len(vtest),
                        "params": list(tr_v[4:]),
                        "test": _strip(evaluate(lambda N, b: rule_choice(tr_v, N, b), vtest)),
                        "eigh_boundary_test": _strip(evaluate(lambda N, b: rule_choice(params, N, b), vtest))},
        }
        BIG = BIG_EIGH
    res["values_chosen"] = list(values) if values else None

    # ---- stage 4: the tridiag backend instead of the CPU ----
    tridiag, tridiag_cap = [0, 0], [0, 0]
    have_td = any("tridiag" in tv for tv in times_full.values())
    if have_td:
        s4 = {}
        for which, full, rule_params, cur in (("vectors", times_full, params,
                                                (CURRENT_TRIDIAG[0], CURRENT_TRIDIAG_CAP[0])),
                                               ("values", vtimes_full, (split + values) if values else params,
                                                (CURRENT_TRIDIAG[1], CURRENT_TRIDIAG_CAP[1]))):
            if not full or not any("tridiag" in tv for tv in full.values()):
                continue
            full = tridiag_points(only(full, ("tridiag",)))
            th, scores = fit_tridiag(rule_params, full, cur, tol)
            tr, te = split_points(full)
            th_tr, _ = fit_tridiag(rule_params, tr, cur, tol)
            fn = lambda N, b, t=th: tridiag_choice(rule_params, t, N, b)
            s4[which] = {
                "n_points": len(full),
                "chosen": th[0],
                "cap": th[1],
                "without": _strip(evaluate(lambda N, b: tridiag_choice(rule_params, 0, N, b), full)),
                "with": _strip(evaluate(fn, full)),
                "curve": [[t, v[0], v[1]] for t, v in sorted(scores.items())],
                "holdout": {"train_points": len(tr), "test_points": len(te), "fitted": list(th_tr),
                            "test": _strip(evaluate(lambda N, b: tridiag_choice(rule_params, th_tr, N, b), te)),
                            "without_test": _strip(evaluate(lambda N, b: tridiag_choice(rule_params, 0, N, b), te))},
                "speedup_vs_cpu": sorted([[N, b, tv["cpu"] / tv["tridiag"]] for (b, N), tv in full.items()
                                          if "cpu" in tv and "tridiag" in tv], key=lambda x: (x[0], x[1])),
            }
            tridiag[0 if which == "vectors" else 1] = th[0]
            tridiag_cap[0 if which == "vectors" else 1] = th[1]
        res["stage4"] = s4

    # ---- stage 4b: the band backend (eigenvalues alone) before tridiag, from
    # its own threshold, within the same batch cap
    band, width = 0, 0
    vrule = (split + values) if values else params
    bfull = {p: tv for p, tv in btimes.items() if rule_choice(vrule, p[1], p[0]) == "cpu"}
    if bfull:
        w, wscores = band_width_choice(bfull)
        width = w if wscores else 0   # 0: the widths were not all timed (a run from before 2.15.0)
        bfull = with_band_width(bfull, w)
        def band_choice(t, N, b):
            if tridiag_cap[1] and b > tridiag_cap[1]:
                return "cpu"
            if t and N >= t:
                return "band"
            return "tridiag" if tridiag[1] and N >= tridiag[1] else "cpu"
        cands = [0] + sorted({N for (_, N) in bfull})
        bscores = {t: _score3(evaluate(lambda N, b, t=t: band_choice(t, N, b), bfull)) for t in cands}
        best_t = min(bscores, key=lambda t: bscores[t][0])
        near_b = near_on_disagreement(bscores, lambda t, p: bfull[p][band_choice(t, p[1], p[0])], bfull,
                                      best_t, tol)
        band = CURRENT_BAND if CURRENT_BAND in near_b else min(near_b, key=lambda t: (near_b[t][1], -t if t else 0))
        res["stage4b"] = {
            "n_points": len(bfull), "chosen": band, "current": CURRENT_BAND,
            "width": width, "current_width": CURRENT_BAND_WIDTH,
            "width_scores": [[k, v] for k, v in sorted(wscores.items())],
            "near": sorted(near_b),
            "with": _strip(evaluate(lambda N, b: band_choice(band, N, b), bfull)),
            "without": _strip(evaluate(lambda N, b: band_choice(0, N, b), bfull)),
            "curve": [[t, v[0], v[1]] for t, v in sorted(bscores.items())],
            "speedup_vs_tridiag": sorted([[N, b, tv["tridiag"] / tv["band"]] for (b, N), tv in bfull.items()]),
            "speedup_vs_cpu": sorted([[N, b, tv["cpu"] / tv["band"]] for (b, N), tv in bfull.items()]),
        }
    # What the library's cpu_side routes a call to that the rules give the
    # CPU, given the thresholds fitted so far (eigh.mm): the tridiag_batch
    # window first, then the cap, band and tridiag.
    def cpu_side(N, b, vectors, band_vec=0, tbw=(0, 0, 0)):
        lo, hi, mb = tbw
        if hi and lo <= N <= hi and b >= max(mb, 1):
            return "tridiag_batch"
        i = 0 if vectors else 1
        if tridiag_cap[i] and b > tridiag_cap[i]:
            return "cpu"
        bt = band_vec if vectors else band
        if bt and N >= bt:
            return "band"
        return "tridiag" if tridiag[i] and N >= tridiag[i] else "cpu"

    # ---- stage 4c: the band backend with eigenvectors before tridiag, and the
    # batch cap with it: band takes a batch of two or more of up to 1024 in
    # tridiag_batch's two stages, every matrix at once (since 2.17.0), so the
    # cap that suited tridiag, a matrix at a time, need not suit band. Both
    # fitted over band's points and tridiag's (where band was not timed and is
    # picked, the model's estimate counts, flagged); with band never, stage
    # 4's cap stands.
    band_vec = 0
    vfull_b = {p: {k: tv[k] for k in ("cpu", "tridiag", "band") if k in tv} for p, tv in times_full.items()
               if all(k in tv for k in ("cpu", "tridiag", "band")) and rule_choice(params, p[1], p[0]) == "cpu"}
    if vfull_b:
        pts = {p: {k: v for k, v in tv.items() if k in ("cpu", "tridiag")}
               for p, tv in tridiag_points(only(times_full, ("tridiag",))).items()
               if rule_choice(params, p[1], p[0]) == "cpu" and "cpu" in tv}
        for p, tv in vfull_b.items():
            pts.setdefault(p, {}).update(tv)

        def choice_c(c, N, b):
            t, cap = c
            if cap and b > cap:
                return "cpu"
            if t and N >= t:
                return "band"
            return "tridiag" if tridiag[0] and N >= tridiag[0] else "cpu"
        caps = [0] + sorted({b for (b, _) in pts})
        cands = [(0, tridiag_cap[0])] + [(t, cap) for t in sorted({N for (_, N) in vfull_b}) for cap in caps]
        cscores = {c: _score3(evaluate(lambda N, b, c=c: choice_c(c, N, b), pts)) for c in cands}
        best = min(v[0] for v in cscores.values())
        near_c = {c: v for c, v in cscores.items() if v[0] <= best * (1 + tol)}
        current = (CURRENT_BAND_VEC, CURRENT_TRIDIAG_CAP[0])
        chosen = (current if current in near_c
                  else min(near_c, key=lambda c: (near_c[c][1], -c[0] if c[0] else 0, c[1] if c[1] else INF)))
        band_vec = chosen[0]
        if band_vec:
            tridiag_cap[0] = chosen[1]
        res["stage4c"] = {
            "n_points": len(pts), "band_points": len(vfull_b), "chosen": band_vec, "cap": chosen[1],
            "current": list(current), "near": sorted([list(c) for c in near_c]),
            "with": _strip(evaluate(lambda N, b: choice_c(chosen, N, b), pts)),
            "without": _strip(evaluate(lambda N, b: choice_c((0, tridiag_cap[0]), N, b), pts)),
            "curve": [[list(c), v[0], v[1]] for c, v in sorted(cscores.items())],
            "speedup_vs_tridiag": sorted([[N, b, tv["tridiag"] / tv["band"]] for (b, N), tv in vfull_b.items()]),
            "speedup_vs_cpu": sorted([[N, b, tv["cpu"] / tv["band"]] for (b, N), tv in vfull_b.items()]),
        }
    res["band_vec_chosen"] = band_vec
    res["current_band_vec"] = CURRENT_BAND_VEC

    # ---- stage 5: the tridiag_batch window, with eigenvectors and for
    # eigenvalues alone, over the points where it was timed and the rules
    # give the CPU; against the CPU and whatever else cpu_side would pick
    tb = [(0, 0, 0), (0, 0, 0)]
    s5 = {}
    for i, (which, full, rule_params) in enumerate((("vectors", times_full, params),
                                                    ("values", vtimes_full, (split + values) if values else params))):
        pts = {p: {k: v for k, v in tv.items() if k in ("cpu", "tridiag", "band", "tridiag_batch")}
               for p, tv in (full or {}).items()
               if "tridiag_batch" in tv and "cpu" in tv and rule_choice(rule_params, p[1], p[0]) == "cpu"}
        if not pts:
            continue
        Ns = sorted({N for (_, N) in pts})
        Bs = sorted({b for (b, _) in pts})
        cands = [(0, 0, 0)] + [(lo, hi, mb) for lo in Ns for hi in Ns if hi >= lo for mb in Bs]
        vec = which == "vectors"
        choice_t = lambda c, N, b: cpu_side(N, b, vec, band_vec, c)
        tscores = {c: _score3(evaluate(lambda N, b, c=c: choice_t(c, N, b), pts)) for c in cands}
        best_t, near_t = near_optimal(tscores, tol)
        cur = tuple(CURRENT_TB[i])
        cur_score = _score3(evaluate(lambda N, b: choice_t(cur, N, b), pts))
        if cur in near_t or cur_score[0] <= best_t * (1 + tol):
            chosen = cur
        else:
            # the smallest worst case; then the most conservative window: the
            # largest batch, the narrowest range of N
            chosen = min(near_t, key=lambda c: (near_t[c][1], -c[2], c[1] - c[0]))
        tb[i] = chosen
        tr, te = split_points(pts)
        tr_scores = {c: _score3(evaluate(lambda N, b, c=c: choice_t(c, N, b), tr)) for c in cands}
        _, tr_near = near_optimal(tr_scores, tol)
        fitted = min(tr_near, key=lambda c: (tr_near[c][1], -c[2], c[1] - c[0]))
        s5[which] = {
            "n_points": len(pts), "chosen": list(chosen), "current": list(cur),
            "with": _strip(evaluate(lambda N, b: choice_t(chosen, N, b), pts)),
            "without": _strip(evaluate(lambda N, b: choice_t((0, 0, 0), N, b), pts)),
            "n_near_optimal": len(near_t),
            "holdout": {"train_points": len(tr), "test_points": len(te), "fitted": list(fitted),
                        "test": _strip(evaluate(lambda N, b: choice_t(fitted, N, b), te)),
                        "without_test": _strip(evaluate(lambda N, b: choice_t((0, 0, 0), N, b), te))},
            "speedup_vs_cpu": sorted([[N, b, tv["cpu"] / tv["tridiag_batch"]] for (b, N), tv in pts.items()],
                                     key=lambda x: (x[0], x[1])),
        }
    if s5:
        res["stage5"] = s5
    res["tb_chosen"] = [list(t) for t in tb]
    res["current_tb"] = [list(t) for t in CURRENT_TB]

    res["band_chosen"] = band
    res["band_width_chosen"] = width
    res["current_band"] = CURRENT_BAND
    res["tridiag_chosen"] = tridiag
    res["current_tridiag"] = list(CURRENT_TRIDIAG)
    res["tridiag_cap_chosen"] = tridiag_cap
    res["current_tridiag_cap"] = list(CURRENT_TRIDIAG_CAP)
    res["current_values"] = list(CURRENT_VALUES) if CURRENT_VALUES else None

    # ---- the whole rule ----
    res["chosen"] = list(params)
    res["rules"] = {"current": _strip(with_big(CURRENT_BIG, lambda: score_rule(CURRENT, times))),
                    "chosen": _strip(score_rule(params, times))}

    # ---- surfaces ----
    cells = []
    for (b, N), tv in sorted(times.items()):
        gpu = {k: tv[k] for k in GPU_BACKENDS if k in tv}
        best_gpu = min(gpu, key=gpu.get) if gpu else None
        cells.append({"N": N, "batch": b, "best": min(tv, key=tv.get), "best_gpu": best_gpu,
                      "times": tv, "rule": rule_choice(params, N, b),
                      "rule_gpu": gpu_choice(params[:4], N, b),
                      "gpu_speedup": (tv["cpu"] / gpu[best_gpu]) if ("cpu" in tv and best_gpu) else None})
    res["surface"] = cells

    # ---- warnings ----
    warns = []
    pp = score_rule(params, times)["_per_point"]
    misses = sorted(((r, N, b, k) for (b, N), (k, r, e) in pp.items() if r > 1.25 and not e), reverse=True)
    if misses:
        warns.append("chosen rule loses more than 25% at " +
                     ", ".join(f"N={N} batch={b} ({k}, {r:.2f}x)" for r, N, b, k in misses[:8]) +
                     (" ..." if len(misses) > 8 else ""))
    gpp = score_split(split, gtimes)["_per_point"]
    gm = sorted(((r, N, b, k) for (b, N), (k, r, e) in gpp.items() if r > 1.25 and not e), reverse=True)
    if gm:
        warns.append("GPU split loses more than 25% against the best GPU backend at " +
                     ", ".join(f"N={N} batch={b} ({k}, {r:.2f}x)" for r, N, b, k in gm[:8]) +
                     (" ..." if len(gm) > 8 else ""))
    ests = sorted(((r, N, b, k) for (b, N), (k, r, e) in pp.items() if e), reverse=True)
    if ests:
        warns.append(f"chosen rule picks a backend that was not timed at {len(ests)} points; the cost "
                     f"model's guess counts in the geomean and is kept out of the worst case: " +
                     ", ".join(f"N={N} batch={b} ({k}, est {r:.2f}x)" for r, N, b, k in ests[:8]) +
                     (" ..." if len(ests) > 8 else ""))
    if single_pass:
        warns.insert(0, "single pass: there is no noise floor and no min-of-repeats, so this run is a "
                        "smoke test of the pipeline, not a measurement. Do not paste its row; run "
                        "without --quick")
    busy = []
    for when, st in (states or {}).items():
        for prob in state_problems(st):
            busy.append(f"{prob} at the {when} of the sweep")
    if busy:
        warns.insert(0, "the machine was not idle: " + "; ".join(busy) + ". The CPU backend is slowed "
                        "most by this, which biases the routing toward the GPU; rerun when idle")
    on_battery = any(st.get("power") == "battery" for st in (states or {}).values())
    if on_battery:
        warns.append("run on battery power; macOS may limit performance differently than on mains")
    if drift_info and not drift_info["ok"]:
        warns.insert(0, "the machine changed state during the sweep: the probe point moved by " +
                     ", ".join(f"{k} x{r:.2f}" for k, r in sorted(drift_info["ratio"].items())) +
                     " between the start and the end. Timings from either side are not comparable; "
                     "rerun on mains power with Low Power Mode off and nothing else running")
    max_n = max(N for _, N in times)
    top = [c for c in cells if c["N"] == max_n and c["gpu_speedup"] is not None]
    if params[4] >= max_n or any(c["gpu_speedup"] > 1.0 for c in top):
        warns.append(f"the GPU is still ahead of the CPU at the largest N measured ({max_n}): "
                     f"gpu_max_n is a lower bound, rerun with --max-n 1024 to find the cap")
    if str(device.get("source", "")).startswith("default:untuned-device"):
        warns.append("this device has no entry in kTuned[]: it is running the untuned default; "
                     "paste the row above into src/eigh.mm")
    if (tuple(params) != tuple(CURRENT) or (values and tuple(values) != tuple(CURRENT_VALUES or ()))
            or tuple(res.get("tridiag_chosen", (0, 0))) != tuple(CURRENT_TRIDIAG)
            or tuple(tridiag_cap) != tuple(CURRENT_TRIDIAG_CAP)
            or tuple(ql) != tuple(CURRENT_QL) or share != CURRENT_SHARE or tuple(big) != tuple(CURRENT_BIG)
            or res.get("band_chosen", 0) != CURRENT_BAND or band_vec != CURRENT_BAND_VEC
            or [list(t) for t in tb] != [list(t) for t in CURRENT_TB]):
        warns.append(f"the fitted policy differs from the one in effect ({device.get('source', 'unknown')}): "
                     f"update this device's row in kTuned[] in src/eigh.mm")
    # The canary: a Jacobi backend fastest beyond the grid's Jacobi range
    canary = sorted((N, b, k) for which in (times_full, vtimes_full or {}) for (b, N), tv in which.items()
                    if N > JACOBI_MAX_GRID_N and tv for k in [min(tv, key=tv.get)] if k in JACOBI_BACKENDS)
    if canary:
        warns.append("a Jacobi backend is the fastest beyond the N up to which the sweep times them ("
                     + ", ".join(f"N={N} batch={b} ({k})" for N, b, k in canary[:6])
                     + f"; JACOBI_MAX_GRID_N = {JACOBI_MAX_GRID_N}): rerun the sweep with --full-grid")
    res["warnings"] = warns
    res["noise"] = floor
    res["trustworthy"] = (not single_pass) and (drift_info is None or drift_info["ok"]) and not busy
    res["machine"] = states
    if values:
        vmax = max(N for _, N in vtimes)
        vtop = [(b, N) for (b, N) in vtimes if N == vmax]
        if values[0] >= vmax or any(vtimes[p]["cpu"] > min(v for k, v in vtimes[p].items() if k != "cpu")
                                    for p in vtop):
            warns.append(f"eigenvalues alone: the GPU is still ahead at the largest N measured ({vmax}); "
                         f"values_gpu_max_n is a lower bound")
        misses_v = sorted(((r, N, b, k) for (b, N), (k, r, e) in score_rule(split + values, vtimes)["_per_point"].items()
                           if r > 1.25 and not e), reverse=True)
        if misses_v:
            warns.append("eigenvalues alone: the chosen boundary loses more than 25% at " +
                         ", ".join(f"N={N} batch={b} ({k}, {r:.2f}x)" for r, N, b, k in misses_v[:8]) +
                         (" ..." if len(misses_v) > 8 else ""))
    else:
        warns.append("no eigenvalues-alone timings (a run from before they were measured): the row's "
                      "values_* fields are 0, so eigvalsh follows eigh's boundary")
    if not have_td:
        warns.append("no tridiag timings (a run from before the backend existed): the row's tridiag "
                     "thresholds are 0, so the backend stays off on this device")
    if not have_ql:
        warns.append("no ql timings (a run from before the backend existed): the row's ql window is "
                     "0, 0, so the backend stays off on this device")
    band, band_width = res.get("band_chosen", 0), res.get("band_width_chosen", 0)
    res["tuned_row"] = tuned_row(device, params, values, tridiag, ql, tridiag_cap, share, big, band, band_width,
                                 band_vec, tb)
    res["env_line"] = env_line(params, values, tridiag, ql, tridiag_cap, share, big, band, band_width, band_vec, tb)
    res["big_chosen"] = list(big)
    res["n_candidates"] = {"split": len(SIMD_MAXS) * len(BLOCK_MINS),
                           "routing": len(GPU_MAX_NS) * len(MIN_BNS) * len(MIN_BATCHES)}
    return res


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _fmt_v(v):
    return "none" if v is None or v >= INF else str(v)


def _mermaid(title, xlab, pts):
    xs = [_fmt_v(p["value"]) for p in pts]
    ys = [min(p["geomean"], 3.0) for p in pts]
    ymax = math.ceil(max(ys) * 100) / 100 + 0.01
    return "\n".join([
        "```mermaid", "xychart-beta", f'    title "{title}"',
        f'    x-axis "{xlab}" [' + ", ".join(xs) + "]",
        f'    y-axis "geometric-mean regret" 1.0 --> {ymax:.2f}',
        "    line [" + ", ".join(f"{y:.4f}" for y in ys) + "]",
        "```"])


def _curve_block(L, xlab, pts):
    L.append(_mermaid(f"Regret by {xlab}", xlab, pts))
    L.append("")
    L.append("| " + xlab + " | " + " | ".join(_fmt_v(p["value"]) for p in pts) + " |")
    L.append("|---|" + "---|" * len(pts))
    L.append("| geomean | " + " | ".join(f"{p['geomean']:.4f}" for p in pts) + " |")
    L.append("| worst | " + " | ".join(f"{p['worst']:.2f}x" for p in pts) + " |")
    L.append("")


def _surface(cells, key):
    Ns = sorted({c["N"] for c in cells})
    Bs = sorted({c["batch"] for c in cells})
    g = {(c["N"], c["batch"]): c for c in cells}
    letter = {"cpu": "c", "simd": "s", "tg": "t", "block": "B", "ql": "q", "ql_share": "Q", None: "."}
    lines = ["```", "  N \\ batch " + " ".join(f"{b:>5}" for b in Bs)]
    for N in Ns:
        row = []
        for b in Bs:
            c = g.get((N, b))
            if c is None:
                row.append("    .")
            elif key == "speedup":
                v = c["gpu_speedup"]
                row.append(f"{v:5.2f}" if v is not None else "  gpu")
            else:
                row.append(f"    {letter[c[key]]}")
        lines.append(f"  {N:>9} " + " ".join(row))
    lines.append("```")
    return "\n".join(lines)


def _stats_row(name, e):
    return (f"| {name} | {e['geomean']:.4f} | {e['worst']:.2f}x | {e['over10']} "
            f"| {e['total_ratio']:.3f} | {e['estimated_picks']} |")


def _holdout_row(name, params, r):
    b = r.get("bootstrap")
    verdict = "baseline" if b is None else ("**justified**" if r["justified"] else "rejected")
    boot = "" if b is None else f" (better in {b['p_better']*100:.0f}% of resamples, median gain {b['gain_median']*100:.1f}%)"
    return (f"| {name} | {json.dumps(params)} | {r['train']['geomean']:.4f} | {r['test']['geomean']:.4f} "
            f"| {r['test']['worst']:.2f}x | {verdict}{boot} |")


def _fmt_td(f):
    """A fitted (threshold, cap), or a bare threshold from before the cap existed."""
    th, cap = (f[0], f[1]) if isinstance(f, (list, tuple)) else (f, 0)
    if not th:
        return "never"
    return f"from {th}" + (f", batch <= {cap}" if cap else "")


def write_report(res, path):
    d = res["device"]
    s, bm, lo, bh, gm, mb, mbatch = res["chosen"]
    s1, s2 = res["stage1"], res["stage2"]
    cur = res["rules"]["current"]
    cho = res["rules"]["chosen"]
    L = []
    L.append(f"# Eigensolver routing on {d['name']} ({d['gpu_cores']} GPU cores)")
    L.append("")
    L.append("What every section and number below means: [reading-reports.md](https://github.com/c0rmac/metal-linalg/blob/main/docs/reading-reports.md).")
    if res.get("calibration"):
        c = res["calibration"]["scale"]
        L.append("")
        L.append(f"Cost model scaled to this device from one probe point: block x{c['block']:.2f}, "
                 f"cpu x{c['cpu']:.2f}, whole-matrix x{c['tg']:.2f} (1.00 is an M1).")
    if res.get("machine"):
        bits = []
        for when, st in res["machine"].items():
            if st.get("load_1m") is not None:
                bits.append(f"load {st['load_1m']:.1f}/{st['cpus']} at the {when}")
        st0 = next(iter(res["machine"].values()))
        L.append("")
        L.append("Machine state: " + ", ".join(bits) + f"; power {st0.get('power', 'unknown')}"
                 + ("; Low Power Mode on" if st0.get("low_power_mode") == 1 else "") + ".")
    else:
        L.append("")
        L.append("Machine state was not recorded for this run.")
    if res.get("drift"):
        dr = res["drift"]
        L.append("")
        L.append("Probe point after the sweep relative to before it: " +
                 ", ".join(f"{k} x{r:.2f}" for k, r in sorted(dr["ratio"].items())) +
                 (" (stable)." if dr["ok"] else " (**unstable**, see warnings)."))
    L.append("")
    L.append(f"Generated by `tuning/tune_eigh.py` from {res['n_points']} (N, batch) points, "
             f"N in {res['grid']['N']}, batch in {res['grid']['batch']}, four backends, "
             f"{'one pass' if res.get('single_pass') else 'two or more passes, min-of-repeats'}.")
    L.append("")
    if len(res.get("submissions") or []) > 1:
        L.append(f"Combined from {len(res['submissions'])} submissions ({', '.join(res['submissions'])}): "
          f"each submission's fastest pass, then the median across submissions. The noise "
          f"floor is per submission.")
        L.append("")
    L.append("## Answer")
    L.append("")
    if res.get("trustworthy", True):
        L.append("Row for `kTuned[]` in `src/eigh.mm`:")
    else:
        L.append("**Indicative only; do not paste this row.** See the first warning below.")
    L.append("")
    L.append("```cpp")
    L.append("// device, GPU cores,   simd_max_n, block_min_n, block_min_n_batched, block_min_batch,   "
             "gpu_max_n, gpu_min_batch_times_n, gpu_min_batch,   values_gpu_max_n, "
             "values_gpu_min_batch_times_n, values_gpu_min_batch,   tridiag_min_n, values_tridiag_min_n, "
             "tridiag_max_batch, values_tridiag_max_batch,   ql_min_n, ql_max_n,   share_min_batch,   "
             "gpu_big_batch_max_n, gpu_big_batch_min,   values_band_min_n, values_band_width")
    L.append(res["tuned_row"])
    L.append("```")
    L.append("")
    L.append("To try it without rebuilding:")
    L.append("")
    L.append("```sh")
    L.append(res["env_line"])
    L.append("```")
    L.append("")
    src = d.get("source", "unknown")
    same = tuple(res["current"]) == tuple(res["chosen"]) and (res.get("values_chosen") is None or
            res.get("current_values") == res["values_chosen"]) and \
        res.get("current_tridiag", [0, 0]) == res.get("tridiag_chosen", [0, 0]) and \
        res.get("current_tridiag_cap", [0, 0]) == res.get("tridiag_cap_chosen", [0, 0]) and \
        res.get("current_ql", [0, 0]) == res.get("ql_chosen", [0, 0])
    L.append(f"The policy in effect on this device came from `{src}`. " +
             ("It matches the fitted one." if same else
              "It differs from the fitted one; see the warnings."))
    L.append("")
    L.append(f"Against the best measured backend at every point the whole rule scores "
             f"{cho['geomean']:.4f} geometric-mean regret, worst {cho['worst']:.2f}x, "
             f"{cho['over10']} of {cho['n']} points losing more than 10%, and {cho['total_ratio']:.3f}x "
             f"the oracle's total time. The decision is fitted in two stages, below, because the "
             f"CPU routing would otherwise hide the GPU backend crossover.")
    L.append("")
    if res["warnings"]:
        L.append("## Warnings")
        L.append("")
        for w in res["warnings"]:
            L.append(f"- {w}")
        L.append("")

    # ---- stage 1 ----
    L.append("## Stage 1: which GPU backend")
    L.append("")
    L.append(f"Scored against the best *GPU* backend at each of the {s1['n_points']} points, "
             f"as if there were no CPU: this is the rule a forced-GPU call (`EIGH_DEVICE=gpu`) "
             f"and the `detail` entry points follow, and it is what a GPU with more cores will "
             f"lean on.")
    L.append("")
    L.append("| rule | geomean regret | worst | >10% | total time / oracle | est. picks |")
    L.append("|---|---|---|---|---|---|")
    L.append(_stats_row(f"policy in effect {tuple(_fmt_v(v) for v in res['current'][:4])}", s1["current"]))
    L.append(_stats_row(f"fitted {tuple(_fmt_v(v) for v in s1['chosen_params'])}", s1["chosen"]))
    L.append("")
    b1 = s1["band"]
    L.append(f"{b1['n_near_optimal']} of {res['n_candidates']['split']} (simd_max_n, block_min_n) pairs are "
             f"within {res['tolerance']*100:.1f}% of the best geomean: simd_max_n {b1['simd_max'][0]} .. "
             f"{b1['simd_max'][1]}, block_min_n {_fmt_v(b1['block_min'][0])} .. {_fmt_v(b1['block_min'][1])}.")
    L.append("")
    if "band_batched" in s1:
        bb = s1["band_batched"]
        L.append(f"With the batch term adopted (below), {bb['n_near_optimal']} combinations are within "
                 f"{res['tolerance']*100:.1f}% of the best: block_min_n {_fmt_v(bb['block_min'][0])} .. "
                 f"{_fmt_v(bb['block_min'][1])}, block_min_n_batched {_fmt_v(bb['block_lo'][0])} .. "
                 f"{_fmt_v(bb['block_lo'][1])}, block_min_batch {_fmt_v(bb['batch_hi'][0])} .. "
                 f"{_fmt_v(bb['batch_hi'][1])}. The curves below vary one constant around the chosen combination.")
        L.append("")
    _curve_block(L, "block_min_n", s1["curves"]["block_min"])
    if "block_lo" in s1["curves"]:
        _curve_block(L, "block_min_n_batched", s1["curves"]["block_lo"])
    if "batch_curve" in s1:
        _curve_block(L, "block_min_batch", s1["batch_curve"])
    _curve_block(L, "simd_max_n", s1["curves"]["simd_max"])
    if "holdout" in s1:
        h = s1["holdout"]
        L.append(f"Held-out check of a batch-dependent crossover (block from a lower N once the batch "
                 f"is large enough). Fitted on {h['train_points']} points, scored on the other "
                 f"{h['test_points']}; the verdict is a bootstrap over the test points.")
        L.append("")
        L.append("| rule | fitted on train | train geomean | test geomean | test worst | verdict |")
        L.append("|---|---|---|---|---|---|")
        L.append(_holdout_row("two constants", h["base"]["params"], h["base"]))
        L.append(_holdout_row("batch-dependent block crossover", h["batch_block"]["params"], h["batch_block"]))
        L.append("")
    L.append("Best GPU backend per point (`s` simd, `t` threadgroup, `B` block, `q` ql, `Q` ql shared with the CPU), then what the split "
             "picks, the ql window of stage 1b included:")
    L.append("")
    L.append(_surface(res["surface"], "best_gpu"))
    L.append("")
    L.append(_surface(res["surface"], "rule_gpu"))
    L.append("")

    # ---- stage 1b ----
    s1b = res.get("stage1b")
    if s1b:
        L.append("## Stage 1b: the ql backend")
        L.append("")
        lo_, hi_ = s1b["chosen"]
        L.append(f"Inside a window of N, `ql` (tridiagonalization and implicit QL, one threadgroup per matrix, "
                 f"N <= {s1b['limit']} on this device) instead of the Jacobi backend the split picks, fitted over "
                 f"the {s1b['n_points']} points against the best GPU backend, ql included. Chosen: "
                 + ("never." if not hi_ else f"N = {lo_} .. {hi_}."))
        L.append("")
        L.append("| rule | geomean regret | worst | >10% | total time / oracle | est. picks |")
        L.append("|---|---|---|---|---|---|")
        L.append(_stats_row("without ql (the split alone)", s1b["without"]))
        L.append(_stats_row(f"with ql for N in {tuple(s1b['chosen'])}", s1b["with"]))
        L.append("")
        bq = s1b["band"]
        L.append(f"{bq['n_near_optimal']} windows are within {res['tolerance']*100:.1f}% of the best geomean: "
                 f"ql_min_n {bq['ql_min_n'][0]} .. {bq['ql_min_n'][1]}, ql_max_n {bq['ql_max_n'][0]} .. {bq['ql_max_n'][1]}.")
        L.append("")
        h = s1b["holdout"]
        L.append(f"Held out: fitted on {h['train_points']} points (window {tuple(h['fitted'])}), scored on the other "
                 f"{h['test_points']}: geomean {h['test']['geomean']:.4f}x, worst {h['test']['worst']:.2f}x, against "
                 f"{h['without_test']['geomean']:.4f}x, worst {h['without_test']['worst']:.2f}x without ql.")
        L.append("")
        if s1b["speedup_vs_jacobi"]:
            L.append("ql over the best Jacobi backend, N x batch: " +
                     ", ".join(f"{N}x{b} {r:.2f}x" for N, b, r in s1b["speedup_vs_jacobi"]))
            L.append("")

    s1c = res.get("stage1c")
    if s1c:
        L.append("## Stage 1c: sharing a batch with the CPU")
        L.append("")
        sb_, sn_ = s1c["chosen"] if isinstance(s1c["chosen"], list) else (s1c["chosen"], 0)
        L.append("From a batch and an N on, `ql_share`: ql and the CPU path at once on one batch, the GPU taking "
                 "chunks from the front and the CPU from the back. Fitted against the best GPU backend, the shared "
                 f"one included, on the {s1c['n_points']} points where it was timed and ql is the GPU's choice. "
                 "Chosen: " + (f"from batch {sb_}" + (f" and N = {sn_}." if sn_ else ".") if sb_ else "never."))
        L.append("")
        L.append("| rule | geomean regret | worst | >10% | total time / oracle | est. picks |")
        L.append("|---|---|---|---|---|---|")
        L.append(_stats_row("ql alone", s1c["without"]))
        L.append(_stats_row(f"shared from batch {sb_}" + (f", N >= {sn_}" if sn_ else "") if sb_ else "shared never",
                            s1c["with"]))
        L.append("")
        if s1c["speedup_vs_ql"]:
            L.append("ql_share over ql alone, N x batch: " +
                     ", ".join(f"{N}x{b} {r:.2f}x" for N, b, r in s1c["speedup_vs_ql"]))
            L.append("")

    # ---- stage 2 ----
    L.append("## Stage 2: GPU or CPU")
    L.append("")
    L.append("Given the split above, GPU iff `N <= gpu_max_n`, `batch * N >= gpu_min_batch_times_n` and "
             "`batch >= gpu_min_batch`, scored against the best of all four backends. `worst` is over the "
             "points where the chosen backend was timed; a pick the cost model had to guess is listed in "
             "the warnings instead.")
    L.append("")
    L.append("| rule | geomean regret | worst | >10% | total time / oracle | est. picks |")
    L.append("|---|---|---|---|---|---|")
    L.append(_stats_row("oracle (best per point)", {"geomean": 1.0, "worst": 1.0, "over10": 0,
                                                    "total_ratio": 1.0, "estimated_picks": 0}))
    L.append(_stats_row(f"policy in effect {tuple(_fmt_v(v) for v in res['current'][4:])}", s2["current"]))
    L.append(_stats_row(f"fitted {tuple(_fmt_v(v) for v in s2['chosen_params'])}", s2["chosen"]))
    if s2.get("big"):
        bg = s2["big"]
        L.append("")
        L.append("Large batches: " + (f"the GPU also for N above gpu_max_n up to {bg['chosen'][0]} in a batch of at "
                                      f"least {bg['chosen'][1]}, fitted with the product rule (per cap, the rule, "
                                      "the clause over it and the rule again given the clause, the best kept)"
                                      if bg["chosen"][1] else
                                      "no clause; none beat the product rule alone by more than the tolerance") +
                 f" (product rule alone {bg['without']['geomean']:.4f}, worst {bg['without']['worst']:.2f}x; "
                 f"chosen {bg['with']['geomean']:.4f}, worst {bg['with']['worst']:.2f}x).")
    L.append("")
    b2 = s2["band"]
    L.append(f"{b2['n_near_optimal']} of {res['n_candidates']['routing']} combinations are "
             f"within {res['tolerance']*100:.1f}% of the best geomean: gpu_max_n {_fmt_v(b2['gpu_max_n'][0])} .. "
             f"{_fmt_v(b2['gpu_max_n'][1])}, gpu_min_batch_times_n {_fmt_v(b2['min_bn'][0])} .. {_fmt_v(b2['min_bn'][1])}"
             + (f", gpu_min_batch {_fmt_v(b2['min_batch'][0])} .. {_fmt_v(b2['min_batch'][1])}." if "min_batch" in b2 else "."))
    L.append("")
    _curve_block(L, "gpu_min_batch_times_n", s2["curves"]["min_bn"])
    if "min_batch" in s2["curves"]:
        _curve_block(L, "gpu_min_batch", s2["curves"]["min_batch"])
    _curve_block(L, "gpu_max_n", s2["curves"]["gpu_max_n"])
    h = s2["holdout"]
    L.append(f"Held-out check of a per-N boundary (a lookup table of the smallest batch at which the "
             f"GPU wins, per N) against the product rule. Fitted on {h['train_points']} points, scored "
             f"on the other {h['test_points']}.")
    L.append("")
    L.append("| rule | fitted on train | train geomean | test geomean | test worst | verdict |")
    L.append("|---|---|---|---|---|---|")
    L.append(_holdout_row("product rule", h["base"]["params"], h["base"]))
    L.append(_holdout_row("per-N table", h["cpu_table"]["params"], h["cpu_table"]))
    L.append("")
    L.append("Best backend per point (`c` CPU, `s` simd, `t` threadgroup, `B` block, `q` ql, `Q` ql shared with the CPU, `.` not measured), "
             "what the whole rule picks, and the speedup of the best GPU backend over the CPU:")
    L.append("")
    L.append(_surface(res["surface"], "best"))
    L.append("")
    L.append(_surface(res["surface"], "rule"))
    L.append("")
    L.append(_surface(res["surface"], "speedup"))
    L.append("")

    s3 = res.get("stage3")
    if s3:
        L.append("## Stage 3: GPU or CPU, eigenvalues alone")
        L.append("")
        L.append("The same rule for `eigvalsh`, with its own thresholds (`values_gpu_max_n`, "
                 "`values_gpu_min_batch_times_n`, `values_gpu_min_batch`), fitted on the `_vals` timings "
                 f"of {s3['n_points']} points given the split above. The CPU computes eigenvalues alone by "
                 "LAPACK's two-stage reduction from N = 128, so the boundary need not be eigh's.")
        L.append("")
        L.append("| rule | geomean regret | worst | >10% | total time / oracle | est. picks |")
        L.append("|---|---|---|---|---|---|")
        cur_label = ("eigh's boundary, in effect" if s3["current_is_eigh_boundary"]
                     else "policy in effect")
        L.append(_stats_row(cur_label, s3["current"]))
        if not s3["current_is_eigh_boundary"]:
            L.append(_stats_row("eigh's fitted boundary", s3["eigh_boundary"]))
        L.append(_stats_row(f"fitted {tuple(_fmt_v(v) for v in s3['chosen_params'])}", s3["chosen"]))
        L.append("")
        b3 = s3["band"]
        L.append(f"{b3['n_near_optimal']} combinations are within {res['tolerance']*100:.1f}% of the best "
                 f"geomean: values_gpu_max_n {_fmt_v(b3['gpu_max_n'][0])} .. {_fmt_v(b3['gpu_max_n'][1])}, "
                 f"values_gpu_min_batch_times_n {_fmt_v(b3['min_bn'][0])} .. {_fmt_v(b3['min_bn'][1])}, "
                 f"values_gpu_min_batch {_fmt_v(b3['min_batch'][0])} .. {_fmt_v(b3['min_batch'][1])}.")
        L.append("")
        h3 = s3["holdout"]
        L.append(f"Held out: fitted on {h3['train_points']} points "
                 f"{tuple(_fmt_v(v) for v in h3['params'])}, scored on the other {h3['test_points']}: "
                 f"geomean {h3['test']['geomean']:.4f}x, worst {h3['test']['worst']:.2f}x, against "
                 f"{h3['eigh_boundary_test']['geomean']:.4f}x, worst {h3['eigh_boundary_test']['worst']:.2f}x "
                 f"for eigh's boundary on the same points.")
        L.append("")

    s4 = res.get("stage4")
    if s4:
        L.append("## Stage 4: the tridiag backend instead of the CPU")
        L.append("")
        L.append("Where the rule above chooses the CPU, the `tridiag` backend from a threshold N on "
                 "(0: never), for batches up to a cap (0: any; it solves a batch one matrix after another, "
                 "the CPU path spreads one over every core), fitted over the measured N and batches against "
                 "the best of all backends, tridiag included, on the points where tridiag was timed "
                 "(N >= 128, within the cost cap): the region the threshold decides.")
        L.append("")
        L.append("| | threshold | batch cap | geomean regret | worst | without tridiag: geomean | worst | held out (fitted on half) |")
        L.append("|---|---|---|---|---|---|---|---|")
        for which, e in s4.items():
            h = e["holdout"]
            L.append(f"| {'with eigenvectors' if which == 'vectors' else 'eigenvalues alone'} | "
                     f"{e['chosen'] or 'never'} | {e.get('cap') or 'any'} | {e['with']['geomean']:.4f} | "
                     f"{e['with']['worst']:.2f}x | {e['without']['geomean']:.4f} | {e['without']['worst']:.2f}x | "
                     f"{_fmt_td(h['fitted'])}: {h['test']['geomean']:.4f} vs {h['without_test']['geomean']:.4f} |")
        L.append("")
        for which, e in s4.items():
            if e["speedup_vs_cpu"]:
                L.append(f"tridiag over the CPU ({'with eigenvectors' if which == 'vectors' else 'eigenvalues alone'}), "
                         "N x batch: " + ", ".join(f"{N}x{b} {r:.2f}x" for N, b, r in e["speedup_vs_cpu"]))
                L.append("")

    s4b = res.get("stage4b")
    if s4b:
        L += ["## Stage 4b: the band backend for eigenvalues alone", "",
              "For eigenvalues alone, where the rules above choose the CPU or tridiag, the `band` backend "
              "(the two-stage reduction: A to a band on the GPU in blocks whose work is matrix products, the "
              "band to tridiagonal on the CPU's cores, then bisection on the GPU) from a threshold N on (0: "
              f"never), within tridiag's batch cap, fitted on the {s4b['n_points']} points where it was timed "
              f"(N >= {BAND_MIN_GRID_N}) against the CPU, tridiag and band.", "",
              f"Chosen: {s4b['chosen'] or 'never'} (in effect: {s4b['current'] or 'never'}): "
              f"{s4b['with']['geomean']:.4f} geometric-mean regret, worst {s4b['with']['worst']:.2f}x; without "
              f"band {s4b['without']['geomean']:.4f}, worst {s4b['without']['worst']:.2f}x.", "",
              "Its band's width: " + (f"{s4b['width']} (in effect: {s4b['current_width'] or 16}); geometric mean "
              "of each width's time over the best width's at each point: " +
              ", ".join(f"{w} {g:.3f}" for w, g in s4b["width_scores"]) +
              f". 16, the default, unless another is better by more than {BAND_WIDTH_TOL:.0%}."
              if s4b.get("width_scores") else "16, the default: the other widths were not timed."), "",
              "Thresholds within the fit's tolerance of the best, and within "
              f"{DISAGREE_TOL:.0%} of it on the points where the two choose differently: " +
              (", ".join(str(t or "never") for t in s4b.get("near", [])) or "none") + ".", "",
              "band over tridiag, N x batch: " +
              ", ".join(f"{N}x{b} {r:.2f}x" for N, b, r in s4b["speedup_vs_tridiag"]), "",
              "band over the CPU, N x batch: " +
              ", ".join(f"{N}x{b} {r:.2f}x" for N, b, r in s4b["speedup_vs_cpu"]), ""]
    s4c = res.get("stage4c")
    if s4c:
        L += ["## Stage 4c: the band backend with eigenvectors", "",
              "With eigenvectors, where the rules above choose the CPU or tridiag, the `band` backend (the "
              "two-stage reduction, its reflectors and the chase's applied to the eigenvectors on the GPU while "
              "the CPU chases and solves) from a threshold N on (0: never), within tridiag's batch cap, fitted "
              f"on the {s4c['n_points']} points where it was timed (N >= {BAND_MIN_GRID_N}) against the CPU, "
              "tridiag and band.", "",
              f"Chosen: {s4c['chosen'] or 'never'} (in effect: {s4c['current'] or 'never'}): "
              f"{s4c['with']['geomean']:.4f} geometric-mean regret, worst {s4c['with']['worst']:.2f}x; without "
              f"band {s4c['without']['geomean']:.4f}, worst {s4c['without']['worst']:.2f}x.", "",
              "Thresholds within the fit's tolerance of the best, and within "
              f"{DISAGREE_TOL:.0%} of it on the points where the two choose differently: " +
              (", ".join(str(t or "never") for t in s4c.get("near", [])) or "none") + ".", "",
              "band over tridiag, N x batch: " +
              ", ".join(f"{N}x{b} {r:.2f}x" for N, b, r in s4c["speedup_vs_tridiag"]), "",
              "band over the CPU, N x batch: " +
              ", ".join(f"{N}x{b} {r:.2f}x" for N, b, r in s4c["speedup_vs_cpu"]), ""]
    s5 = res.get("stage5")
    if s5:
        L += ["## Stage 5: the tridiag_batch backend for batches of mid-size matrices", "",
              "Where the rules above choose the CPU, the `tridiag_batch` backend (the tridiag method for a whole "
              "batch at once: the reduction of every matrix by the same dispatches, the tridiagonal problems on "
              "the CPU's cores, the back-transformation as batched products) for N in a window from a batch "
              f"on, fitted on the points where it was timed (N {TB_MIN_GRID_N}-{TB_MAX_GRID_N}, batches from "
              f"{TB_MIN_GRID_BATCH}) against the CPU and whatever else the CPU's side would pick. Inside the "
              "flat region the window in effect stays; otherwise the smallest worst case, then the largest "
              "batch and the narrowest window.", ""]
        for which, t5 in s5.items():
            lo_, hi_, mb_ = t5["chosen"]
            clo, chi, cmb = t5["current"]
            L += [f"**{'With eigenvectors' if which == 'vectors' else 'Eigenvalues alone'}** ({t5['n_points']} "
                  "points): " + (f"N {lo_}-{hi_} from batch {mb_}" if hi_ else "never") +
                  " (in effect: " + (f"N {clo}-{chi} from batch {cmb}" if chi else "never") + f"): "
                  f"{t5['with']['geomean']:.4f} geometric-mean regret, worst {t5['with']['worst']:.2f}x; without "
                  f"it {t5['without']['geomean']:.4f}, worst {t5['without']['worst']:.2f}x. Held out: the window "
                  f"fitted on half the points, {t5['holdout']['fitted']}, scores "
                  f"{t5['holdout']['test']['geomean']:.4f} on the other half, against "
                  f"{t5['holdout']['without_test']['geomean']:.4f} without it.", "",
                  "tridiag_batch over the CPU, N x batch: " +
                  ", ".join(f"{N}x{b} {r:.2f}x" for N, b, r in t5["speedup_vs_cpu"]), ""]
    L.append("## Noise floor")
    L.append("")
    nf = res["noise"]
    L.append(f"Pass-to-pass ratio (max/min of the same measurement across passes), "
             f"{nf['overall']['n']} measurements: median {nf['overall']['median']:.3f}, "
             f"p90 {nf['overall']['p90']:.3f}, max {nf['overall']['max']:.2f}. The held-out "
             f"verdicts use a bootstrap rather than this figure, since a mean over many points "
             f"is far less noisy than one measurement.")
    L.append("")
    L.append("| runtime | n | median | p90 | max |")
    L.append("|---|---|---|---|---|")
    for r in nf["by_runtime"]:
        L.append(f"| {r['bucket']} | {r['n']} | {r['median']:.3f} | {r['p90']:.3f} | {r['max']:.2f} |")
    L.append("")
    with open(path, "w") as fh:
        fh.write("\n".join(L))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global CURRENT, CURRENT_VALUES, CURRENT_TRIDIAG, CURRENT_TRIDIAG_CAP, CURRENT_QL, QL_LIMIT, CURRENT_SHARE, CURRENT_BIG
    global CURRENT_BAND, CURRENT_BAND_WIDTH, CURRENT_BAND_VEC, CURRENT_TB
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("binary", nargs="?", help="path to the built sweep_eigh")
    ap.add_argument("--out", default="eigh-tune-results", help="output directory")
    ap.add_argument("--passes", type=int, default=2, help="independent passes (minimum 2 for a noise floor)")
    ap.add_argument("--limit", type=int, default=240, help="per-point timeout, seconds")
    ap.add_argument("--quick", action="store_true", help="coarser grid, one pass, roughly a third of the time")
    ap.add_argument("--full-grid", action="store_true",
                    help="time the Jacobi backends at every N (default: up to JACOBI_MAX_GRID_N, and the canary)")
    ap.add_argument("--full-passes", action="store_true",
                    help="repeat every point in every pass (default: the later passes repeat the contested ones)")
    ap.add_argument("--max-n", type=int, default=512,
                    help="largest N on the grid (768 and 1024 are added up to this)")
    ap.add_argument("--from", dest="from_json", help="re-render the report from an existing results.json")
    ap.add_argument("--reanalyse", metavar="RAW.CSV", nargs="+", help="re-run the analysis on raw.csv files")
    args = ap.parse_args()
    global FULL_GRID, FULL_PASSES
    FULL_GRID, FULL_PASSES = args.full_grid, args.full_passes

    os.makedirs(args.out, exist_ok=True)
    if args.from_json:
        res = json.load(open(args.from_json))
        write_report(res, os.path.join(args.out, "report.md"))
        print(f"wrote {args.out}/report.md")
        return

    side = None
    if args.reanalyse:
        raw_paths = args.reanalyse
        side = load_sidecar(raw_paths)
        if args.binary:                       # a binary was given too: ask it
            side = dict(side or {}, policy=query_policy(args.binary))
    else:
        if not args.binary:
            ap.error("give the sweep_eigh binary, or --from / --reanalyse")
        policy = query_policy(args.binary)
        QL_LIMIT = policy.get("ql_limit", 0)   # the grid times ql up to it
        print(f"device: {policy['device']} ({policy['gpu_cores']} GPU cores), "
              f"policy from {policy['source']}", file=sys.stderr)
        state0 = machine_state()
        for prob in state_problems(state0):
            print(f"warning: {prob}; this sweep will be marked untrustworthy. "
                  f"Ctrl-C and rerun when the machine is idle.", file=sys.stderr)
        if state0.get("power") == "battery":
            print("warning: on battery power; prefer mains", file=sys.stderr)
        cal = calibrate(args.binary, args.limit)
        print("cost model scale: " + ", ".join(f"{k} x{v:.2f}" for k, v in cal["scale"].items()),
              file=sys.stderr)
        side = {"policy": policy, "calibration": cal, "machine": {"start": state0}}
        json.dump(side, open(os.path.join(args.out, "policy.json"), "w"), indent=1)

        passes = 1 if args.quick else max(2, args.passes)
        raw = os.path.join(args.out, "raw.csv")
        pts = point_grid(args.quick, args.max_n)
        print(f"sweep -> {raw}", file=sys.stderr)
        sweep(args.binary, pts, passes, args.limit, raw)
        raw_paths = [raw]
        side["drift"] = drift(cal["probe"]["ms"], probe(args.binary, args.limit))
        side["machine"]["end"] = machine_state()
        json.dump(side, open(os.path.join(args.out, "policy.json"), "w"), indent=1)

    device = {"name": "unknown", "gpu_cores": 0, "source": "unknown"}
    calibration = None
    if side:
        pol = side.get("policy")
        if pol:
            device = {"name": pol["device"], "gpu_cores": pol["gpu_cores"], "source": pol["source"]}
            CURRENT = policy_to_tuple(pol)
            CURRENT_VALUES = policy_values(pol)
            CURRENT_TRIDIAG = (pol.get("tridiag_min_n", 0), pol.get("values_tridiag_min_n", 0))
            CURRENT_BAND = pol.get("values_band_min_n", 0)
            CURRENT_BAND_WIDTH = pol.get("values_band_width", 0)
            CURRENT_BAND_VEC = pol.get("band_min_n", 0)
            CURRENT_TB = ((pol.get("tridiag_batch_min_n", 0), pol.get("tridiag_batch_max_n", 0),
                           pol.get("tridiag_batch_min_batch", 0)),
                          (pol.get("values_tridiag_batch_min_n", 0), pol.get("values_tridiag_batch_max_n", 0),
                           pol.get("values_tridiag_batch_min_batch", 0)))
            CURRENT_TRIDIAG_CAP = (pol.get("tridiag_max_batch", 0), pol.get("values_tridiag_max_batch", 0))
            CURRENT_QL = (pol.get("ql_min_n", 0), pol.get("ql_max_n", 0))
            CURRENT_SHARE = (pol.get("share_min_batch", 0), pol.get("share_min_n", 0))
            CURRENT_BIG = (pol.get("gpu_big_batch_max_n", 0), pol.get("gpu_big_batch_min", 0))
            QL_LIMIT = pol.get("ql_limit", 0)
        calibration = side.get("calibration")
        if calibration:
            SCALE.update(calibration["scale"])

    times, repeats, subs = load(raw_paths)
    if not times:
        sys.exit("no usable measurements")
    res = analyse(times, repeats, device, drift_info=(side or {}).get("drift"),
                  states=(side or {}).get("machine"))
    res["raw"] = raw_paths
    res["submissions"] = subs
    res["calibration"] = calibration
    res = sub.portable(res)
    with open(os.path.join(args.out, "results.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    write_report(res, os.path.join(args.out, "report.md"))

    s_, bm, lo, bh, gm, mb, mbatch = res["chosen"]
    print(f"\n{res['n_points']} points on {device['name']} ({device['gpu_cores']} GPU cores)")
    print(f"GPU split:   simd_max_n={s_} block_min_n={_fmt_v(bm)} block_min_n_batched={_fmt_v(lo)} "
          f"block_min_batch={_fmt_v(bh)}   ({res['stage1']['chosen']['geomean']:.4f}x vs best GPU backend, "
          f"worst {res['stage1']['chosen']['worst']:.2f}x)")
    print(f"CPU routing: gpu_max_n={_fmt_v(gm)} gpu_min_batch_times_n={_fmt_v(mb)} gpu_min_batch={mbatch}   "
          f"({res['rules']['chosen']['geomean']:.4f}x vs best of all, worst {res['rules']['chosen']['worst']:.2f}x)")
    print(f"in effect ({device['source']}): {res['rules']['current']['geomean']:.4f}x, "
          f"worst {res['rules']['current']['worst']:.2f}x")
    if res.get("values_chosen"):
        vg, vm, vb = res["values_chosen"]
        s3 = res["stage3"]
        print(f"eigenvalues alone: values_gpu_max_n={_fmt_v(vg)} values_gpu_min_batch_times_n={_fmt_v(vm)} "
              f"values_gpu_min_batch={vb}   ({s3['chosen']['geomean']:.4f}x, worst {s3['chosen']['worst']:.2f}x; "
              f"with eigh's boundary {s3['eigh_boundary']['geomean']:.4f}x, worst {s3['eigh_boundary']['worst']:.2f}x)")
    if res.get("stage1b"):
        s1b = res["stage1b"]
        lo_, hi_ = s1b["chosen"]
        print(f"ql window:   " + (f"N={lo_}..{hi_}" if hi_ else "never") +
              f"   ({s1b['with']['geomean']:.4f}x vs best GPU backend, worst {s1b['with']['worst']:.2f}x; "
              f"without it {s1b['without']['geomean']:.4f}x, worst {s1b['without']['worst']:.2f}x)")
    if res.get("big_chosen") and res["big_chosen"][1]:
        print(f"large batches: also the GPU for N above gpu_max_n up to {res['big_chosen'][0]} "
              f"from batch {res['big_chosen'][1]}")
    if res.get("stage1c"):
        c1 = res["stage1c"]
        sb_, sn_ = c1["chosen"] if isinstance(c1["chosen"], list) else (c1["chosen"], 0)
        print(f"share with the CPU: {('from batch ' + str(sb_) + (f', N >= {sn_}' if sn_ else '')) if sb_ else 'never'}   "
              f"({c1['with']['geomean']:.4f}x vs best GPU backend; ql alone {c1['without']['geomean']:.4f}x)")
    for which, s4 in (res.get("stage4") or {}).items():
        print(f"tridiag ({which}): {_fmt_td((s4['chosen'], s4.get('cap', 0)))}   ({s4['with']['geomean']:.4f}x, worst "
              f"{s4['with']['worst']:.2f}x; without it {s4['without']['geomean']:.4f}x, worst {s4['without']['worst']:.2f}x; "
              f"held out {s4['holdout']['test']['geomean']:.4f}x vs {s4['holdout']['without_test']['geomean']:.4f}x)")
    if res.get("stage4b"):
        b4 = res["stage4b"]
        print(f"band (values): {b4['chosen'] or 'never'}, width {b4['width'] or 16}   ({b4['with']['geomean']:.4f}x, worst "
              f"{b4['with']['worst']:.2f}x; without {b4['without']['geomean']:.4f}x)")
    if res.get("stage4c"):
        c4 = res["stage4c"]
        print(f"band (vectors): {c4['chosen'] or 'never'}   ({c4['with']['geomean']:.4f}x, worst "
              f"{c4['with']['worst']:.2f}x; without {c4['without']['geomean']:.4f}x)")
    for which, t5 in (res.get("stage5") or {}).items():
        lo_, hi_, mb_ = t5["chosen"]
        print(f"tridiag_batch ({which}): " + (f"N={lo_}..{hi_} from batch {mb_}" if hi_ else "never") +
              f"   ({t5['with']['geomean']:.4f}x, worst {t5['with']['worst']:.2f}x; without it "
              f"{t5['without']['geomean']:.4f}x, worst {t5['without']['worst']:.2f}x; held out "
              f"{t5['holdout']['test']['geomean']:.4f}x vs {t5['holdout']['without_test']['geomean']:.4f}x)")
    print(f"\nkTuned[] row:  {res['tuned_row']}" +
          ("" if res["trustworthy"] else "     <-- indicative only, do not paste (see first warning)"))
    for w in res["warnings"]:
        print("warning:", w)
    print(f"wrote {args.out}/results.json and report.md")


if __name__ == "__main__":
    main()
