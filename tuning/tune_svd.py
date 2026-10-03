#!/usr/bin/env python3
"""Measure this GPU's SVD routing and write up the result.

    cmake --build build --target sweep_svd
    python3 tuning/tune_svd.py build/sweep_svd

Produces, in --out (default svd-tune-results/):

    raw.csv        every timed run
    results.json   the full analysis, plot-ready
    report.md      a written report generated from results.json
    policy.json    the device, the policy it was running, calibration, machine state

Re-render or re-analyse without touching the GPU:

    python3 tuning/tune_svd.py --from svd-tune-results/results.json
    python3 tuning/tune_svd.py --reanalyse svd-tune-results/raw.csv

THE DECISION
------------
For a batch of `batch` matrices of shape M x N, with k = min(M, N) and
l = max(M, N), which of seven backends should `svd_accelerated` use?

    cpu      thin SVD through MLX (Accelerate LAPACK), QR first when tall
    jacobi   whole-matrix one-sided Jacobi kernel on the matrix itself
    block    block one-sided Jacobi kernel on the matrix itself
    qr       this library's QR, then the whole-matrix kernel on the k x k factor
    qrblock  this library's QR, then the block kernel on the k x k factor
    bidiag   GPU bidiagonalization (LAPACK's sgebrd, panels on the GPU) and
             back-transforms, LAPACK's bidiagonal solve; QR first when tall
    gk       Householder bidiagonalization and implicit bidiagonal QR, one
             threadgroup per matrix (golub_kahan): on the matrix itself where
             it fits in threadgroup memory, else on the k x k factor after
             this library's QR; k up to the device's limit

svd.mm encodes the answer as a per-device routing policy (SvdPolicy):

    qr_min_rows, qr_min_k          precondition with QR iff l >= qr_min_rows,
                                   k >= qr_min_k and l >= 2k
    block_min_k                    block kernel from this k on
    block_min_k_batched,           ... and from this smaller k once the batch
      block_min_batch              reaches block_min_batch (0, 0 = not used)
    gpu_max_k                      GPU only up to this k
    gpu_min_batch_times_k          GPU only if batch * k is at least this
    gpu_min_batch                  GPU only from this batch (1 = no minimum)
    gpu_max_l                      GPU only up to this l = max(M, N) (no cap = never
                                   above it): the CPU's QR-first path wins tall
                                   shapes that a cap on k alone would send to the GPU
    bidiag_min_k,                  where the rule says CPU, bidiag instead
      values_bidiag_min_k          from this k on, with vectors / for
                                   singular values alone (0 = never)
    gk_min_k, gk_max_k             gk instead of the Jacobi backends for k in
                                   this window (0, 0 = never)
    share_min_batch                from this batch, gk shares the batch with the
                                   CPU path (gk_share; 0 = never)
    values_gpu_max_k, ..._min_batch_times_k, ..._min_batch, ..._max_l
                                   the GPU-or-CPU rule again for singular
                                   values alone (values_gpu_min_batch = 0:
                                   as with vectors), fitted on gk_vals and
                                   cpu_vals where gk is timed

The method is that of tuning/tune_eigh.py, whose helpers this imports: fitted
in stages (the Jacobi split against the best Jacobi backend alone; 1b, the gk
window over it against the best GPU backend, gk included, as tune_eigh.py fits
its ql window; then the CPU boundary given both, then bidiag in place of the
CPU on the points where it was timed), every combination scored by regret, the flat region
reported rather than the argmin, refinements (the batch-dependent block
crossover, a per-k CPU boundary) kept only on a held-out bootstrap.

Only shapes with M >= N are measured. A wide matrix is decomposed through its
transpose by every backend, so it costs what the transposed shape costs.

Run it on an idle machine, on mains power, with Low Power Mode off; the run is
marked untrustworthy otherwise (see tuning/tune_eigh.py and docs/tuning.md).
"""

import argparse
import csv
import json
import math
import os
import random
import subprocess
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tune_eigh as te   # noqa: E402  shared machinery

INF = te.INF
NO_LIMIT = te.NO_LIMIT
GPU_BACKENDS = ("jacobi", "block", "qr", "qrblock")
BLOCK_BACKENDS = ("block", "qrblock")
BLOCK_FROM_K = 32        # the block kernel is not timed below this k

# 24 .. 80 between the powers of two: where the gk window ends.
K_SQUARE = [4, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 128, 192, 256]
K_TALL   = [8, 16, 32, 64, 128, 256]
ASPECTS  = [2, 4, 8, 16, 32]
B_LIST   = [1, 4, 16, 64, 256, 1024, 4096]
K_SQUARE_Q, K_TALL_Q, ASPECTS_Q, B_QUICK = [8, 32, 64, 128, 256], [16, 64, 256], [4, 16], [1, 16, 256, 4096]
K_EXTRA  = [384, 512, 768, 1024]   # added up to --max-k
MAX_ROWS = 2048
# Above 1024 the Jacobi kernels are out of reach; only the CPU and the bidiag
# backend are timed, lone matrices and small batches, with a larger per-call
# budget: the region bidiag_min_k decides.
K_LARGE = [1536, 2048, 3072, 4096]   # added up to --max-k
LARGE_K = 1024
LARGE_BATCHES = [1, 2, 4]
HUGE_K = 2048          # above this, lone matrices only
CAP_LARGE_MS = 8000.0
BIDIAG_MIN_GRID_K = 128   # bidiag is timed from this k
VALS = "_vals"            # suffix: the same backend for singular values alone

# Filled from the data by configure_grids().
QR_MIN_ROWS = [128, 256, 512, 1024, 2048, INF]
QR_MIN_KS   = [8, 16, 32, 64, 128, INF]
BLOCK_MINS  = [32, 48, 64, 96, 128, 192, 256, 384, INF]
GPU_MAX_KS  = [8, 16, 32, 64, 128, 256, INF]
MIN_BKS     = [0, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, INF]
MIN_BATCHES = [1, 4, 16]
GPU_MAX_LS  = [16, 32, 64, 128, 256, 512, INF]
B_GRID      = list(B_LIST)

# The policy in effect, as the tuple the analysis works with:
#   (qr_min_rows, qr_min_k, block_min, block_lo, batch_hi, gpu_max_k, min_bk, min_batch, gpu_max_l)
# block_lo / batch_hi = INF means no batch-dependent block crossover. Replaced
# by what `sweep_svd --policy` reports; this is only the fallback for
# re-analysing a raw.csv that has no policy.json beside it.
CURRENT = (512, 64, 192, INF, INF, 64, 1024, 1, INF)
N_SPLIT = 5              # the first five are the GPU split, the rest the CPU routing
# The bidiag thresholds in effect, (with vectors, singular values alone); 0 = never.
CURRENT_BIDIAG = (0, 0)
# Their batch caps, the same way round; 0 = any batch.
CURRENT_BIDIAG_CAP = (0, 0)
# The gk window in effect, (gk_min_k, gk_max_k); (0, 0) = never.
CURRENT_GK = (0, 0)
# The values_gpu_* rule in effect, (max_k, min_bk, min_batch, max_l);
# min_batch 0 = as with vectors.
CURRENT_VALUES = (0, 0, 0, INF)
# The largest k the gk backend takes on the device (`sweep_svd --policy`); 0 = not timed.
GK_LIMIT = 0
# The gk window gpu_choice applies: (0, 0) while stage 1 is fitted, the fitted
# one from stage 1b on.
GK = (0, 0)
# The batch from which gk shares a batch with the CPU path: in effect, and the
# one gpu_choice applies (0 = never; fitted in stage 1c).
CURRENT_SHARE = 0
SHARE = 0
# gk_share is timed from this batch: below it a batch is too small to share.
SHARE_MIN_GRID_BATCH = 64

CAP_MS = 2500.0
SCALE = {"jacobi": 1.0, "block": 1.0, "qr": 1.0, "qrblock": 1.0, "cpu": 1.0, "bidiag": 1.0, "gk": 1.0}
PROBE = (64, 128, 32, ["jacobi", "block", "qr", "qrblock", "cpu", "gk"])   # (batch, M, N, backends)


def est_ms(backend, M, N, b):
    """Per-call cost model, ms; an M1's, pessimistic, rescaled by calibrate()."""
    if backend.endswith(VALS):      # values alone: the reduction without the back-transforms
        return 0.6 * est_ms(backend[:-len(VALS)], M, N, b)
    if backend.endswith("_share"):  # the GPU and the CPU at once
        return 0.7 * est_ms(backend[:-len("_share")], M, N, b)
    if backend == "bidiag":         # serial over the batch: launches per column, then O(M N^2)
        k = min(M, N)
        return b * (1.0 + 0.04 * k + 4e-8 * max(M, N) * k * k) * SCALE.get(backend, 1.0)
    if backend == "gk":             # one threadgroup per matrix, as jacobi, at about a fifth of the work
        return (0.3 + 2e-7 * (M + N) * min(M, N) ** 2 * max(b, 8)) * SCALE.get(backend, 1.0)
    jac = lambda m, n: 0.3 + 1.2e-6 * m * n * n * max(b, 8)
    blk = lambda m, n: 3.0 + 0.06 * n + 5e-7 * m * n * n * b
    qr = 3.0 + 6e-6 * M * N * b
    if backend == "jacobi":
        t = jac(M, N)
    elif backend == "block":
        t = blk(M, N)
    elif backend == "qr":
        t = qr + jac(N, N)
    elif backend == "qrblock":
        t = qr + blk(N, N)
    else:
        t = b * (0.05 + 1.2e-6 * N ** 3 + 2e-7 * M * N * N)
    return t * SCALE.get(backend, 1.0)


def backends_for(M, N, b):
    ks = []
    tall = M >= 2 * N
    k = min(M, N)
    if k <= LARGE_K:
        if est_ms("jacobi", M, N, b) <= CAP_MS:
            ks.append("jacobi")
        if N >= BLOCK_FROM_K and est_ms("block", M, N, b) <= CAP_MS:
            ks.append("block")
        if tall and est_ms("qr", M, N, b) <= CAP_MS:
            ks.append("qr")
        if tall and N >= BLOCK_FROM_K and est_ms("qrblock", M, N, b) <= CAP_MS:
            ks.append("qrblock")
        if k <= GK_LIMIT and est_ms("gk", M, N, b) <= CAP_MS:
            # and both again for singular values alone: the region the
            # values_gpu_* rule decides
            ks += ["gk", "gk" + VALS, "cpu" + VALS]
            if b >= SHARE_MIN_GRID_BATCH:   # and sharing the batch with the CPU
                ks += ["gk_share", "gk_share" + VALS]
    cap = CAP_LARGE_MS if k > LARGE_K else CAP_MS
    bidiag = k >= BIDIAG_MIN_GRID_K and est_ms("bidiag", M, N, b) <= cap
    if not ks and not bidiag:
        return []
    if bidiag or est_ms("cpu", M, N, b) <= CAP_MS:   # the reference wherever bidiag is timed
        ks.append("cpu")
    if bidiag:   # and the two again for singular values alone, the region values_bidiag_min_k decides
        ks += ["bidiag", "bidiag" + VALS] + ([] if "cpu" + VALS in ks else ["cpu" + VALS])
    return ks


def point_grid(quick=False, max_k=512):
    sq = list(K_SQUARE_Q if quick else K_SQUARE) + [k for k in K_EXTRA + K_LARGE if k <= max_k]
    shapes = [(k, k) for k in sq if k <= max(max_k, 4)]
    for k in (K_TALL_Q if quick else K_TALL):
        for a in (ASPECTS_Q if quick else ASPECTS):
            if k * a <= MAX_ROWS:
                shapes.append((k * a, k))
    pts = []
    for M, N in shapes:
        k = min(M, N)
        for b in ([1] if k > HUGE_K else LARGE_BATCHES if k > LARGE_K else B_QUICK if quick else B_LIST):
            if not te.sub.fits_memory(b * (M * N + N * N)):   # see submissions.MEMORY_FRACTION
                continue
            ks = backends_for(M, N, b)
            if ks:
                pts.append((b, M, N, ks))
    return pts


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

def run_one(binary, job, limit, attempts=3):
    b, M, N, ks = job
    for _ in range(attempts):
        try:
            r = subprocess.run([binary, str(b), str(M), str(N), ",".join(ks)],
                               capture_output=True, text=True, timeout=limit)
            lines = [l for l in r.stdout.splitlines() if l.strip()]
            if r.returncode == 0 and len(lines) == len(ks):
                return lines
        except subprocess.TimeoutExpired:
            break
    return [f"{b},{M},{N},{k},0,0,0,0,0" for k in ks]


def sweep(binary, pts, passes, limit, out_csv):
    runs = sum(len(ks) for *_, ks in pts)
    print(f"  {len(pts)} points, {runs} backend timings per pass x {passes} passes", file=sys.stderr)
    done, total, t0 = 0, len(pts) * passes, time.time()
    with open(out_csv, "w") as fh:
        fh.write("pass,batch,M,N,backend,ok,ms,p25,p75,reps\n")
        for p in range(passes):
            order = list(pts)
            random.Random(9000 + p).shuffle(order)
            for job in order:
                for line in run_one(binary, job, limit):
                    fh.write(f"{p},{line}\n")
                fh.flush()
                done += 1
                if done % 20 == 0 or done == total:
                    el = time.time() - t0
                    print(f"    {done}/{total}  elapsed {el/60:.1f}m  eta {el/done*(total-done)/60:.1f}m",
                          file=sys.stderr)


def load(paths):
    """-> (times[(b,M,N)][backend], repeats, submissions); see submissions.combine."""
    times, repeats, subs = te.sub.combine(paths, lambda r: (int(r["batch"]), int(r["M"]), int(r["N"])))
    times = {p: v for p, v in times.items() if any(k in v for k in GPU_BACKENDS + ("bidiag", "gk"))}
    return times, repeats, subs


def split_bidiag(times):
    """-> (times for stages 1-2: the Jacobi backends, gk and the CPU, at points
    with a Jacobi backend; times with vectors, bidiag included; times for
    singular values alone, as {"cpu", "bidiag"}; and for singular values alone
    on the GPU, as {"cpu", "gk"}). Runs from before a backend give empty parts."""
    base, vec, val, gval = {}, {}, {}, {}
    for p, tv in times.items():
        a = {k: v for k, v in tv.items() if k in GPU_BACKENDS or k in ("cpu", "gk", "gk_share")}
        if any(k in a for k in GPU_BACKENDS):
            base[p] = a
        if "bidiag" in tv and "cpu" in tv:
            vec[p] = {k: v for k, v in tv.items() if not k.endswith(VALS)}
        if "bidiag" + VALS in tv and "cpu" + VALS in tv:
            val[p] = {"cpu": tv["cpu" + VALS], "bidiag": tv["bidiag" + VALS]}
        if "gk" + VALS in tv and "cpu" + VALS in tv:
            gval[p] = {"cpu": tv["cpu" + VALS], "gk": tv["gk" + VALS]}
            if "gk_share" + VALS in tv:
                gval[p]["gk_share"] = tv["gk_share" + VALS]
    return base, vec, val, gval

def query_policy(binary):
    r = subprocess.run([binary, "--policy"], capture_output=True, text=True, timeout=60)
    if r.returncode != 0 or not r.stdout.strip():
        sys.exit(f"{binary} --policy failed; rebuild sweep_svd")
    return json.loads(r.stdout.strip().splitlines()[-1])


def probe(binary, limit):
    out = {}
    for _ in range(3):
        for line in run_one(binary, PROBE, limit):
            f = line.split(",")
            if f[4] == "1" and float(f[5]) > 0:
                out[f[3]] = min(out.get(f[3], float("inf")), float(f[5]))
    return out


def calibrate(binary, limit):
    out = probe(binary, limit)
    for k, ms in out.items():
        SCALE[k] = min(4.0, max(0.05, ms / est_ms(k, PROBE[1], PROBE[2], PROBE[0])))
    return {"probe": {"batch": PROBE[0], "M": PROBE[1], "N": PROBE[2], "ms": out}, "scale": dict(SCALE)}


def _cap(v):
    """A threshold as the analysis holds it: anything out of reach is INF."""
    return INF if v >= (1 << 30) else v


def policy_to_tuple(pol):
    gm = pol["gpu_max_k"]
    batched = pol.get("block_min_batch", 0) > 0
    return (_cap(pol["qr_min_rows"]), _cap(pol["qr_min_k"]),
            _cap(pol.get("block_min_k", INF)),       # absent: a build without the block kernel
            _cap(pol["block_min_k_batched"]) if batched else INF,
            _cap(pol["block_min_batch"]) if batched else INF,
            INF if gm >= NO_LIMIT else gm, pol["gpu_min_batch_times_k"],
            max(1, pol.get("gpu_min_batch", 1)),    # absent: a build without the constant
            _cap(pol.get("gpu_max_l", NO_LIMIT)))   # absent: a build from before 2.10.0


def _cxx(params):
    r, a, bm, lo, bh, gm, mb, mbatch, ml = params
    if mb >= INF:
        gm, mb = 0, 0
    if r >= INF or a >= INF:          # the QR-preconditioned backends are never used
        r, a = 1 << 30, 1 << 30
    if lo >= INF or bh >= INF:        # no batch-dependent block crossover
        lo, bh = 0, 0
    if bm >= INF:                     # the block kernel is never used
        bm = 1 << 30
    return (r, a, bm, lo, bh, "kSvdNoLimit" if gm >= INF else gm, mb, mbatch,
            "kSvdNoLimit" if ml >= INF else ml)


def _cxx_values(values):
    vgm, vmb, vmbatch, vml = values
    return (vgm, vmb, vmbatch, "kSvdNoLimit" if vml >= INF else vml)


def tuned_row(device, params, bidiag=(0, 0), bidiag_cap=(0, 0), gk=(0, 0), values=(0, 0, 0, INF), share=0):
    """The line for kTuned[] in svd.mm; `bidiag` the two bidiag thresholds (0: never),
    `bidiag_cap` their batch caps (0: any batch), `gk` the gk window (0, 0: never),
    `values` the values_gpu_* rule (min_batch 0: as with vectors)."""
    r, a, bm, lo, bh, gm, mb, mbatch, ml = _cxx(params)
    vgm, vmb, vmbatch, vml = _cxx_values(values)
    return (f'{{"{device["name"]}", {device["gpu_cores"]},   {r}, {a},   {bm}, {lo}, {bh},   '
            f'{gm}, {mb}, {mbatch}, {ml},   {vgm}, {vmb}, {vmbatch}, {vml},   {bidiag[0]}, {bidiag[1]}, {bidiag_cap[0]}, {bidiag_cap[1]},   '
            f'{gk[0]}, {gk[1]},   {share}}},')


def env_line(params, bidiag=(0, 0), bidiag_cap=(0, 0), gk=(0, 0), values=(0, 0, 0, INF), share=0):
    r, a, bm, lo, bh, gm, mb, mbatch, ml = _cxx(params)
    vgm, vmb, vmbatch, vml = values
    gm = NO_LIMIT if gm == "kSvdNoLimit" else gm
    ml = NO_LIMIT if ml == "kSvdNoLimit" else ml
    return (f"SVD_QR_MIN_ROWS={r} SVD_QR_MIN_K={a} SVD_BLOCK_MIN_K={bm} "
            f"SVD_BLOCK_MIN_K_BATCHED={lo} SVD_BLOCK_MIN_BATCH={bh} "
            f"SVD_GPU_MAX_K={gm} SVD_GPU_MIN_BATCH_TIMES_K={mb} SVD_GPU_MIN_BATCH={mbatch} SVD_GPU_MAX_L={ml} "
            f"SVD_BIDIAG_MIN_K={bidiag[0]} SVD_VALUES_BIDIAG_MIN_K={bidiag[1]} "
            f"SVD_BIDIAG_MAX_BATCH={bidiag_cap[0]} SVD_VALUES_BIDIAG_MAX_BATCH={bidiag_cap[1]} "
            f"SVD_GK_MIN_K={gk[0]} SVD_GK_MAX_K={gk[1]} SVD_SHARE_MIN_BATCH={share} "
            f"SVD_VALUES_GPU_MAX_K={vgm} SVD_VALUES_GPU_MIN_BATCH_TIMES_K={vmb} "
            f"SVD_VALUES_GPU_MIN_BATCH={vmbatch} SVD_VALUES_GPU_MAX_L={NO_LIMIT if vml >= INF else vml}")


# ---------------------------------------------------------------------------
# Rules and scoring
# ---------------------------------------------------------------------------

def gpu_choice(split, M, N, b, gk=None, share=None):
    """The GPU backend: gk inside its window (GK unless `gk` is given; clipped
    to the device's limit), gk_share from the batch SHARE (or `share`; 0 =
    never), else the Jacobi split."""
    rows, min_k, block_min, block_lo, batch_hi = split
    l, k = max(M, N), min(M, N)
    lo, hi = GK if gk is None else gk
    if hi and lo <= k <= (min(hi, GK_LIMIT) if GK_LIMIT else hi):
        sh = SHARE if share is None else share
        return "gk_share" if sh and b >= sh else "gk"
    pre = l >= rows and k >= min_k and l >= 2 * k
    blk = k >= block_min or (k >= block_lo and b >= batch_hi)
    if pre:
        return "qrblock" if blk else "qr"
    return "block" if blk else "jacobi"


def rule_choice(params, M, N, b):
    """params = (qr_min_rows, qr_min_k, block_min, block_lo, batch_hi, gpu_max_k, min_bk, min_batch,
    gpu_max_l)."""
    k = min(M, N)
    if (k > params[N_SPLIT] or b * k < params[N_SPLIT + 1] or b < params[N_SPLIT + 2]
            or max(M, N) > params[N_SPLIT + 3]):
        return "cpu"
    return gpu_choice(params[:N_SPLIT], M, N, b)


def _th_cap(th):
    """A threshold k, or (threshold, batch cap) with cap 0 for any batch."""
    return th if isinstance(th, tuple) else (th, 0)


def bidiag_choice(params, th, M, N, b):
    th, cap = _th_cap(th)
    k = rule_choice(params, M, N, b)
    return "bidiag" if (k == "cpu" and th and min(M, N) >= th and (not cap or b <= cap)) else k


def fit_bidiag(choice, times, current, tol):
    """Stage 3: a bidiag threshold over the measured k from BIDIAG_MIN_GRID_K
    (and 0, never), with a batch cap over the measured batches (and 0, any):
    the backend solves a batch one matrix after another, while the CPU path
    spreads one over every core. Scored on `times`, the points where bidiag
    was timed: over every point, the few large matrices it wins on would move
    the geometric mean by less than the tolerance. `choice((th, cap), M, N, b)`.
    -> ((threshold, cap), scores)."""
    ths = sorted({min(M, N) for (_, M, N) in times if min(M, N) >= BIDIAG_MIN_GRID_K})
    caps = [0] + sorted({b for (b, _, _) in times})
    cands = [(0, 0)] + [(th, cap) for th in ths for cap in caps]
    scores = {c: te._score3(evaluate(lambda M, N, b, c=c: choice(c, M, N, b), times)) for c in cands}
    best = min(g for g, _, _ in scores.values())
    near = {c: v for c, v in scores.items() if v[0] <= best * (1 + tol)}
    current = tuple(current)
    if current in near:
        return current, scores
    return min(near, key=lambda c: (near[c][1], -c[0] if c[0] else 0, c[1] if c[1] else INF)), scores


def evaluate(choice_fn, times):
    logs, worst, over10, est_picks, tc, tb, per = [], 1.0, 0, 0, 0.0, 0.0, {}
    for (b, M, N), tv in times.items():
        best = min(tv.values())
        k = choice_fn(M, N, b)
        est = k not in tv
        if not est:
            r = tv[k] / best
        else:
            # Not timed here (the cost model skipped it): the model's guess
            # counts in the geomean, is flagged, and stays out of `worst`.
            r = max(1.0, est_ms(k, M, N, b) / best)
            est_picks += 1
        per[(b, M, N)] = (k, r, est)
        logs.append(math.log(r))
        if not est:
            worst = max(worst, r)
        over10 += r > 1.10
        tc += r * best
        tb += best
    n = len(logs)
    return {"geomean": math.exp(math.fsum(logs) / n) if n else 1.0, "worst": worst, "over10": over10,
            "total_ratio": tc / tb if tb else 1.0, "estimated_picks": est_picks, "n": n,
            "_per_point": per}


def _strip(e):
    return {k: v for k, v in e.items() if not k.startswith("_")}


def gpu_only(times):
    return {p: g for p, g in ((p, {k: v for k, v in tv.items() if k in GPU_BACKENDS})
                              for p, tv in times.items()) if g}


def score_split(split, gt):
    return evaluate(lambda M, N, b: gpu_choice(split, M, N, b), gt)


def score_rule(params, times):
    return evaluate(lambda M, N, b: rule_choice(params, M, N, b), times)


def fit_split(gt):
    """Stage 1 grid: (qr_min_rows, qr_min_k, block_min), no batch term."""
    out = {}
    for r in QR_MIN_ROWS:
        for a in QR_MIN_KS:
            if (r >= INF) != (a >= INF):
                continue                      # "never" is one candidate, not a row of them
            for bm in BLOCK_MINS:
                p = (r, a, bm, INF, INF)
                e = score_split(p, gt)
                out[p] = (e["geomean"], e["worst"], e["over10"])
    return out


def fit_split_batch(base, gt):
    """Stage 1 refinement: (block_min, block_lo, batch_hi) refitted jointly, with
    the preconditioning constants from the base fit. block_min is refitted
    rather than inherited: without a batch term it settles where the
    large-batch wins outweigh the small-batch losses, and the batch term
    exists to separate the two."""
    out = {}
    for bm in BLOCK_MINS:
        for lo in BLOCK_MINS:
            if lo >= bm:
                continue
            for bh in B_GRID:
                if bh <= 1:
                    continue                  # batch >= 1 always: that is block_min = lo
                p = (base[0], base[1], bm, lo, bh)
                e = score_split(p, gt)
                out[p] = (e["geomean"], e["worst"], e["over10"])
    return out


def gk_candidates(gt):
    """The gk windows worth scoring: (0, 0), never, and every pair of k at
    which gk was timed. A bound only matters where it crosses a measured k."""
    ks = sorted({min(M, N) for (b, M, N), tv in gt.items() if "gk" in tv})
    return [(0, 0)] + [(lo, hi) for i, lo in enumerate(ks) for hi in ks[i:]]


def fit_gk(split, gt):
    """Stage 1b grid: the gk window over the Jacobi split, against the best
    GPU backend, gk included."""
    out = {}
    for w in gk_candidates(gt):
        e = evaluate(lambda M, N, b, w=w: gpu_choice(split, M, N, b, gk=w), gt)
        out[w] = (e["geomean"], e["worst"], e["over10"])
    return out


def gpu_with_gk_share(times):
    return {p: g for p, g in ((p, {k: v for k, v in tv.items() if k in GPU_BACKENDS + ("gk", "gk_share")})
                              for p, tv in times.items()) if g}


def gpu_with_gk(times):
    return {p: g for p, g in ((p, {k: v for k, v in tv.items() if k in GPU_BACKENDS + ("gk",)})
                              for p, tv in times.items()) if g}


def fit_routing(split, times):
    out = {}
    for gm in GPU_MAX_KS:
        for ml in GPU_MAX_LS:
            if ml < gm:
                continue                      # k <= l: the same as gpu_max_k = ml
            for mb in MIN_BKS:
                for mbatch in MIN_BATCHES:
                    p = tuple(split) + (gm, mb, mbatch, ml)
                    e = score_rule(p, times)
                    out[p] = (e["geomean"], e["worst"], e["over10"])
    return out


def values_choice(values, M, N, b, vec_params):
    """The backend svdvals runs: gk on the GPU (the only GPU backend timed for
    singular values alone, so `values` is only scored where gk is the GPU's
    choice), else the CPU; as with vectors while values_gpu_min_batch is 0."""
    vgm, vmb, vmbatch, vml = values
    gpu = "gk_share" if SHARE and b >= SHARE else "gk"
    if not vmbatch:
        return gpu if rule_choice(vec_params, M, N, b) != "cpu" else "cpu"
    k, l = min(M, N), max(M, N)
    return gpu if (k <= vgm and l <= vml and b * k >= vmb and b >= vmbatch) else "cpu"


def fit_values(gval, vec_params, tol):
    """Stage 2b: the values_gpu_* rule, on the points where gk and the CPU were
    timed for singular values alone and gk is the GPU's choice; max_k over the
    k measured there (0: never), as the gk window bounds the backend.
    -> ((max_k, min_bk, min_batch, max_l), scores)."""
    ks = sorted({min(M, N) for (_, M, N) in gval})
    cands = [(0, 0, 1, INF)] + [(gm, mb, mbatch, ml)
                                for gm in ks for ml in GPU_MAX_LS if ml >= gm
                                for mb in MIN_BKS if mb < INF for mbatch in MIN_BATCHES]
    scores = {c: te._score3(evaluate(lambda M, N, b, c=c: values_choice(c, M, N, b, vec_params), gval))
              for c in cands}
    best, near = te.near_optimal(scores, tol)
    return min(near, key=lambda c: (near[c][1], near[c][0], -c[0])), scores


def split_points(times, seed=7):
    """Half the batches of every shape in each half."""
    by_shape = defaultdict(list)
    for (b, M, N) in times:
        by_shape[(M, N)].append(b)
    rng = random.Random(seed)
    train, test = {}, {}
    for (M, N), bs in by_shape.items():
        bs = sorted(bs)
        rng.shuffle(bs)
        for i, b in enumerate(bs):
            (train if i % 2 == 0 else test)[(b, M, N)] = times[(b, M, N)]
    return train, test


def bootstrap(base_fn, ref_fn, test, iters=2000, seed=11):
    lb = evaluate(base_fn, test)["_per_point"]
    lr = evaluate(ref_fn, test)["_per_point"]
    keys = list(test.keys())
    n = len(keys)
    dl = [math.log(lb[k][1]) - math.log(lr[k][1]) for k in keys]
    rng = random.Random(seed)
    gains = sorted(math.fsum(dl[rng.randrange(n)] for _ in range(n)) / n for _ in range(iters))
    return {"p_better": sum(g > 0 for g in gains) / iters,
            "gain_median": math.exp(gains[iters // 2]) - 1.0,
            "gain_p05": math.exp(gains[int(0.05 * iters)]) - 1.0}


def refine_cpu_table(params, train):
    """Per-k minimum batch for the GPU, read off the training half."""
    split, (gm, mb, mbatch, ml) = params[:N_SPLIT], params[N_SPLIT:N_SPLIT + 4]
    by_k = defaultdict(list)
    for (b, M, N), tv in train.items():
        by_k[min(M, N)].append((b, M, N, tv))
    table = {}
    for k, rows in by_k.items():
        wins = sorted(b for b, M, N, tv in rows
                      if gpu_choice(split, M, N, b) in tv
                      and tv.get("cpu", INF) >= tv[gpu_choice(split, M, N, b)])
        table[k] = wins[0] if wins else INF

    def ch(M, N, b):
        if max(M, N) > ml:
            return "cpu"
        thr = table.get(min(M, N))
        if thr is None:
            if min(M, N) > gm or b * min(M, N) < mb or b < mbatch:
                return "cpu"
        elif b < thr:
            return "cpu"
        return gpu_choice(split, M, N, b)

    return ch, {str(k): (v if v < INF else None) for k, v in sorted(table.items())}


def configure_grids(times):
    global QR_MIN_ROWS, QR_MIN_KS, BLOCK_MINS, GPU_MAX_KS, B_GRID
    tall = [(max(M, N), min(M, N)) for _, M, N in times if max(M, N) >= 2 * min(M, N)]
    ks = sorted({min(M, N) for _, M, N in times})
    QR_MIN_ROWS = sorted({l for l, _ in tall}) + [INF]
    QR_MIN_KS = sorted({k for _, k in tall}) + [INF]
    BLOCK_MINS = [k for k in ks if k >= BLOCK_FROM_K] + [INF]
    GPU_MAX_KS = [k for k in ks if k >= 8] + [INF]
    global GPU_MAX_LS
    GPU_MAX_LS = sorted({max(M, N) for _, M, N in times if max(M, N) >= 16}) + [INF]
    B_GRID = sorted({b for b, _, _ in times})
    global MIN_BATCHES
    MIN_BATCHES = [1] + [b for b in B_GRID if 1 < b <= 32]


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyse(times, repeats, device, tol=0.005, drift_info=None, states=None):
    global GK, SHARE
    GK, SHARE = (0, 0), 0    # stage 1 is the Jacobi split alone
    times_all = times
    times, vtimes, valtimes, gvaltimes = split_bidiag(times)
    if not times:
        sys.exit("no Jacobi-backend timings: the GPU split and the CPU boundary cannot be fitted")
    configure_grids(times)
    single_pass = not any(len(v) >= 2 for v in repeats.values())
    shapes = sorted({(M, N) for _, M, N in times}, key=lambda s: (s[1], s[0]))
    res = {"device": device, "n_points": len(times), "single_pass": single_pass,
           "drift": drift_info, "machine": states, "tolerance": tol, "current": list(CURRENT),
           "grid": {"shapes": [list(s) for s in shapes], "batch": sorted({b for b, _, _ in times})}}
    gt = gpu_only(times)
    train, test = split_points(times)
    gtrain, gtest = gpu_only(train), gpu_only(test)
    score1 = lambda p: score_split(p, gt)

    # ---- stage 1: GPU backend, against the best GPU backend
    cur_split = tuple(CURRENT[:N_SPLIT])
    sc1 = fit_split(gt)
    best1, near1 = te.near_optimal(sc1, tol)
    split = te.choose(near1, cur_split, te._score3(score1(cur_split)), best1, tol)
    s1 = {
        "n_points": len(gt), "n_candidates": len(sc1),
        "current": _strip(score1(cur_split)),
        "chosen": _strip(score1(split)), "chosen_params": list(split),
        "band": {"n_near_optimal": len(near1), "qr_min_rows": te._band(near1, 0),
                 "qr_min_k": te._band(near1, 1), "block_min": te._band(near1, 2),
                 "current_in_band": cur_split in near1},
        "curves": {"qr_min_rows": te._curve(score1, split, 0, QR_MIN_ROWS),
                   "qr_min_k": te._curve(score1, split, 1, [k for k in QR_MIN_KS if k < INF]),
                   "block_min": te._curve(score1, split, 2, BLOCK_MINS)},
    }
    # refinement: batch-dependent block crossover, fitted on train, validated on test
    tr_sc1 = fit_split(gtrain) if gtrain else {}
    if tr_sc1 and gtest:
        _, tr_near1 = te.near_optimal(tr_sc1, tol)
        tr_split = te.choose(tr_near1, None)   # the three-constant fit itself, not the policy in effect
        tr_sc1b = fit_split_batch(tr_split, gtrain)
        if tr_sc1b:
            tr_splitb = min(tr_sc1b, key=lambda p: (tr_sc1b[p][0], tr_sc1b[p][1]))
            base_fn = lambda M, N, b: gpu_choice(tr_split, M, N, b)
            ref_fn = lambda M, N, b: gpu_choice(tr_splitb, M, N, b)
            boot = bootstrap(base_fn, ref_fn, gtest)
            a_test, r_test = _strip(evaluate(base_fn, gtest)), _strip(evaluate(ref_fn, gtest))
            ok = te.justified(boot, a_test, r_test)
            s1["holdout"] = {
                "train_points": len(gtrain), "test_points": len(gtest),
                "base": {"params": list(tr_split), "train": _strip(evaluate(base_fn, gtrain)),
                         "test": a_test},
                "batch_block": {"params": {"block_min": tr_splitb[2], "block_lo": tr_splitb[3],
                                           "batch_hi": tr_splitb[4]},
                                "train": _strip(evaluate(ref_fn, gtrain)), "test": r_test,
                                "bootstrap": boot, "justified": ok},
            }
            if ok:
                # Adopt it, refitted on all GPU points; report its own flat
                # region, since the earlier band no longer applies.
                sc1b = fit_split_batch(split, gt)
                _, near1b = te.near_optimal(sc1b, tol)
                split = min(near1b, key=lambda p: (near1b[p][1], near1b[p][0]))
                s1["chosen_params"] = list(split)
                s1["chosen"] = _strip(score1(split))
                s1["band_batched"] = {"n_near_optimal": len(near1b), "block_min": te._band(near1b, 2),
                                      "block_lo": te._band(near1b, 3), "batch_hi": te._band(near1b, 4)}
                s1["curves"]["block_min"] = te._curve(score1, split, 2, BLOCK_MINS)
                s1["curves"]["block_lo"] = te._curve(score1, split, 3,
                                                     [v for v in BLOCK_MINS if v < split[2]])
                s1["curves"]["batch_hi"] = te._curve(score1, split, 4, [b for b in B_GRID if b > 1])
    res["stage1"] = s1

    # ---- stage 1b: the gk window over the split
    gk = (0, 0)
    gkt = gpu_with_gk(times)
    if any("gk" in tv for tv in gkt.values()):
        scg = fit_gk(split, gkt)
        bestg, nearg = te.near_optimal(scg, tol)
        cur_g = tuple(CURRENT_GK)
        cur_score = te._score3(evaluate(lambda M, N, b: gpu_choice(split, M, N, b, gk=cur_g), gkt))
        gk = te.choose(nearg, cur_g, cur_score, bestg, tol)
        tr_g, te_g = gpu_with_gk(train), gpu_with_gk(test)
        _, nearg_tr = te.near_optimal(fit_gk(split, tr_g), tol)
        gk_tr = te.choose(nearg_tr, (0, 0))
        never = lambda M, N, b: gpu_choice(split, M, N, b, gk=(0, 0))
        res["stage1b"] = {
            "n_points": len(gkt), "chosen": list(gk), "current": list(cur_g),
            "with": _strip(evaluate(lambda M, N, b: gpu_choice(split, M, N, b, gk=gk), gkt)),
            "without": _strip(evaluate(never, gkt)),
            "band": {"n_near_optimal": len(nearg),
                     "gk_min_k": [min(p[0] for p in nearg), max(p[0] for p in nearg)],
                     "gk_max_k": [min(p[1] for p in nearg), max(p[1] for p in nearg)]},
            "holdout": {"train_points": len(tr_g), "test_points": len(te_g), "fitted": list(gk_tr),
                        "test": _strip(evaluate(lambda M, N, b: gpu_choice(split, M, N, b, gk=gk_tr), te_g)),
                        "without_test": _strip(evaluate(never, te_g))},
            "speedup_vs_jacobi": sorted([[M, N, b, min(v for k, v in tv.items() if k in GPU_BACKENDS) / tv["gk"]]
                                         for (b, M, N), tv in gkt.items()
                                         if "gk" in tv and any(k in tv for k in GPU_BACKENDS)],
                                        key=lambda x: (x[1], x[0], x[2])),
            "speedup_vs_cpu": sorted([[M, N, b, times[(b, M, N)]["cpu"] / tv["gk"]]
                                      for (b, M, N), tv in gkt.items()
                                      if "gk" in tv and "cpu" in times[(b, M, N)]],
                                     key=lambda x: (x[1], x[0], x[2])),
            "limit": GK_LIMIT,
        }
    GK = tuple(gk)
    res["gk_chosen"] = list(gk)
    res["current_gk"] = list(CURRENT_GK)

    # ---- stage 1c: from which batch gk shares the batch with the CPU path,
    # against the best GPU backend, the shared one included, on the points
    # where it was timed and gk is the GPU's choice
    share = 0
    sht = {p: tv for p, tv in gpu_with_gk_share(times).items()
           if "gk_share" in tv and gpu_choice(split, *p[1:], p[0]) == "gk"}
    if sht:
        cands = [0] + sorted({b for (b, _, _) in sht})
        scs = {c: te._score3(evaluate(lambda M, N, b, c=c: gpu_choice(split, M, N, b, share=c), sht))
               for c in cands}
        bests, nears = te.near_optimal(scs, tol)
        share = te.choose(nears, CURRENT_SHARE,
                          scs.get(CURRENT_SHARE) or te._score3(evaluate(
                              lambda M, N, b: gpu_choice(split, M, N, b, share=CURRENT_SHARE), sht)), bests, tol)
        res["stage1c"] = {
            "n_points": len(sht), "chosen": share, "current": CURRENT_SHARE,
            "with": _strip(evaluate(lambda M, N, b: gpu_choice(split, M, N, b, share=share), sht)),
            "without": _strip(evaluate(lambda M, N, b: gpu_choice(split, M, N, b, share=0), sht)),
            "curve": [[c, v[0], v[1]] for c, v in sorted(scs.items())],
            "speedup_vs_gk": sorted([[M, N, b, tv["gk"] / tv["gk_share"]] for (b, M, N), tv in sht.items()],
                                    key=lambda x: (x[1], x[0], x[2])),
        }
    SHARE = share
    res["share_chosen"] = share
    if share:
        # The window again, with sharing in effect: gk shared with the CPU can
        # lead the Jacobi backends where gk alone did not.
        scg2 = fit_gk(split, gpu_with_gk_share(times))
        bestg2, nearg2 = te.near_optimal(scg2, tol)
        gk2 = te.choose(nearg2, tuple(gk), te._score3(evaluate(lambda M, N, b: gpu_choice(split, M, N, b, gk=gk),
                                                                gpu_with_gk_share(times))), bestg2, tol)
        if tuple(gk2) != tuple(gk):
            res.setdefault("stage1b", {})["refitted_with_share"] = {"before": list(gk), "after": list(gk2)}
            gk = tuple(gk2)
            GK = gk
            res["gk_chosen"] = list(gk)

    # ---- stage 2: CPU boundary
    R0, R1, R2, R3 = N_SPLIT, N_SPLIT + 1, N_SPLIT + 2, N_SPLIT + 3
    cur = tuple(split) + tuple(CURRENT[N_SPLIT:])
    sc2 = fit_routing(split, times)
    best2, near2 = te.near_optimal(sc2, tol)
    params = te.choose(near2, cur, te._score3(score_rule(cur, times)), best2, tol)
    s2 = {
        "n_candidates": len(sc2),
        "current": _strip(score_rule(cur, times)), "chosen": _strip(score_rule(params, times)),
        "chosen_params": list(params[N_SPLIT:]),
        "band": {"n_near_optimal": len(near2), "gpu_max_k": te._band(near2, R0),
                 "min_bk": te._band(near2, R1), "min_batch": te._band(near2, R2),
                 "max_l": te._band(near2, R3),
                 "current_in_band": cur in near2},
        "curves": {"gpu_max_k": te._curve(lambda p: score_rule(p, times), params, R0, GPU_MAX_KS),
                   "min_bk": te._curve(lambda p: score_rule(p, times), params, R1, MIN_BKS),
                   "min_batch": te._curve(lambda p: score_rule(p, times), params, R2, MIN_BATCHES),
                   "gpu_max_l": te._curve(lambda p: score_rule(p, times), params, R3, GPU_MAX_LS)},
    }
    tr_sc2 = fit_routing(split, train)
    tr_best2, tr_near2 = te.near_optimal(tr_sc2, tol)
    tr_params = te.choose(tr_near2, cur, te._score3(score_rule(cur, train)), tr_best2, tol)
    base_fn = lambda M, N, b: rule_choice(tr_params, M, N, b)
    ref_fn, table = refine_cpu_table(tr_params, train)
    a_test, r_test = _strip(evaluate(base_fn, test)), _strip(evaluate(ref_fn, test))
    boot = bootstrap(base_fn, ref_fn, test) if test else {"p_better": 0, "gain_median": 0, "gain_p05": 0}
    s2["holdout"] = {
        "train_points": len(train), "test_points": len(test),
        "base": {"params": list(tr_params[N_SPLIT:]), "train": _strip(evaluate(base_fn, train)),
                 "test": a_test},
        "cpu_table": {"params": {"min_batch_by_k": table}, "train": _strip(evaluate(ref_fn, train)),
                      "test": r_test, "bootstrap": boot, "justified": te.justified(boot, a_test, r_test)},
    }
    res["stage2"] = s2

    # ---- stage 2b: GPU or CPU for singular values alone
    gval = {p: tv for p, tv in gvaltimes.items() if gpu_choice(params[:N_SPLIT], *p[1:], p[0]) == "gk"}
    values = (0, 0, 0, INF)
    if gval:
        values, vscores = fit_values(gval, params, tol)
        tr_v, ts_v = split_points(gval)
        values_tr, _ = fit_values(tr_v, params, tol) if tr_v else (values, None)
        as_vectors = lambda M, N, b: values_choice((0, 0, 0, INF), M, N, b, params)
        res["stage2b"] = {
            "n_points": len(gval), "chosen": [v if v < INF else None for v in values],
            "current": [v if v < INF else None for v in CURRENT_VALUES],
            "with": _strip(evaluate(lambda M, N, b: values_choice(values, M, N, b, params), gval)),
            "as_vectors": _strip(evaluate(as_vectors, gval)),
            "cpu_always": _strip(evaluate(lambda M, N, b: "cpu", gval)),
            "holdout": {"train_points": len(tr_v), "test_points": len(ts_v),
                        "fitted": [v if v < INF else None for v in values_tr],
                        "test": _strip(evaluate(lambda M, N, b: values_choice(values_tr, M, N, b, params), ts_v)),
                        "as_vectors_test": _strip(evaluate(as_vectors, ts_v))},
            "speedup_vs_cpu": sorted([[M, N, b, tv["cpu"] / tv["gk"]] for (b, M, N), tv in gval.items()],
                                     key=lambda x: (x[1], x[0], x[2])),
        }
    res["values_chosen"] = [v if v < INF else None for v in values]

    # ---- stage 3: the bidiag backend instead of the CPU
    bidiag, bidiag_cap = [0, 0], [0, 0]
    s3 = {}
    for which, full, cur, choice in (
            ("vectors", vtimes, (CURRENT_BIDIAG[0], CURRENT_BIDIAG_CAP[0]),
             lambda th, M, N, b: bidiag_choice(params, th, M, N, b)),
            # where svdvals' own GPU rule says CPU, bidiag from its own threshold
            ("values", {p: tv for p, tv in valtimes.items() if values_choice(values, *p[1:], p[0], params) == "cpu"},
             (CURRENT_BIDIAG[1], CURRENT_BIDIAG_CAP[1]),
             lambda th, M, N, b: "bidiag" if (_th_cap(th)[0] and min(M, N) >= _th_cap(th)[0] and
                                              (not _th_cap(th)[1] or b <= _th_cap(th)[1])) else "cpu")):
        if not full:
            continue
        th, scores = fit_bidiag(choice, full, cur, tol)
        tr, ts = split_points(full)
        th_tr, _ = fit_bidiag(choice, tr, cur, tol) if tr else (th, None)
        s3[which] = {
            "n_points": len(full), "chosen": th[0], "cap": th[1],
            "without": _strip(evaluate(lambda M, N, b: choice(0, M, N, b), full)),
            "with": _strip(evaluate(lambda M, N, b, t=th: choice(t, M, N, b), full)),
            "curve": [[t, v[0], v[1]] for t, v in sorted(scores.items())],
            "holdout": {"train_points": len(tr), "test_points": len(ts), "fitted": list(th_tr),
                        "test": _strip(evaluate(lambda M, N, b: choice(th_tr, M, N, b), ts)),
                        "without_test": _strip(evaluate(lambda M, N, b: choice(0, M, N, b), ts))},
            "speedup_vs_cpu": sorted([[M, N, b, tv["cpu"] / tv["bidiag"]] for (b, M, N), tv in full.items()],
                                     key=lambda x: (min(x[0], x[1]), x[0], x[2])),
        }
        bidiag[0 if which == "vectors" else 1] = th[0]
        bidiag_cap[0 if which == "vectors" else 1] = th[1]
    res["stage3"] = s3
    res["bidiag_chosen"] = bidiag
    res["current_bidiag"] = list(CURRENT_BIDIAG)
    res["bidiag_cap_chosen"] = bidiag_cap
    res["current_bidiag_cap"] = list(CURRENT_BIDIAG_CAP)

    res["chosen"] = list(params)
    res["rules"] = {"current": _strip(score_rule(tuple(CURRENT), times)),
                    "chosen": _strip(score_rule(params, times))}

    cells = []
    for (b, M, N), tv in sorted(times.items(), key=lambda kv: (kv[0][2], kv[0][1], kv[0][0])):
        gpu = {k: tv[k] for k in GPU_BACKENDS + ("gk",) if k in tv}
        bg = min(gpu, key=gpu.get) if gpu else None
        cells.append({"M": M, "N": N, "batch": b, "best": min(tv, key=tv.get), "best_gpu": bg,
                      "times": tv, "rule": rule_choice(params, M, N, b),
                      "rule_gpu": gpu_choice(params[:N_SPLIT], M, N, b),
                      "gpu_speedup": (tv["cpu"] / gpu[bg]) if ("cpu" in tv and bg) else None})
    res["surface"] = cells

    # ---- warnings
    warns, busy = [], []
    for when, st in (states or {}).items():
        for prob in te.state_problems(st):
            busy.append(f"{prob} at the {when} of the sweep")
    if busy:
        warns.append("the machine was not idle: " + "; ".join(busy) + ". The CPU backend is slowed most "
                     "by this, which biases the routing toward the GPU; rerun when idle")
    if drift_info and not drift_info["ok"]:
        warns.append("the machine changed state during the sweep: the probe point moved by " +
                     ", ".join(f"{k} x{r:.2f}" for k, r in sorted(drift_info["ratio"].items())) +
                     " between the start and the end; rerun")
    if single_pass:
        warns.append("single pass: no noise floor and no min-of-repeats, so this run is a smoke test "
                     "of the pipeline, not a measurement. Do not paste its row; run without --quick")
    res["trustworthy"] = not (busy or single_pass or (drift_info and not drift_info["ok"]))
    if any(st.get("power") == "battery" for st in (states or {}).values()):
        warns.append("run on battery power; macOS may limit performance differently than on mains")
    pp = score_rule(params, times)["_per_point"]
    misses = sorted(((r, M, N, b, k) for (b, M, N), (k, r, e) in pp.items() if r > 1.25 and not e), reverse=True)
    if misses:
        warns.append("chosen rule loses more than 25% at " +
                     ", ".join(f"{M}x{N} batch={b} ({k}, {r:.2f}x)" for r, M, N, b, k in misses[:8]) +
                     (" ..." if len(misses) > 8 else ""))
    gpp = score_split(params[:N_SPLIT], gt)["_per_point"]
    gmiss = sorted(((r, M, N, b, k) for (b, M, N), (k, r, e) in gpp.items() if r > 1.25 and not e), reverse=True)
    if gmiss:
        warns.append("GPU split loses more than 25% against the best GPU backend at " +
                     ", ".join(f"{M}x{N} batch={b} ({k}, {r:.2f}x)" for r, M, N, b, k in gmiss[:8]) +
                     (" ..." if len(gmiss) > 8 else ""))
    ests = sorted(((r, M, N, b, k) for (b, M, N), (k, r, e) in pp.items() if e), reverse=True)
    if ests:
        warns.append(f"chosen rule picks a backend that was not timed at {len(ests)} points; the cost "
                     f"model's guess counts in the geomean and is kept out of the worst case: " +
                     ", ".join(f"{M}x{N} batch={b} ({k}, est {r:.2f}x)" for r, M, N, b, k in ests[:8]) +
                     (" ..." if len(ests) > 8 else ""))
    max_k = max(min(M, N) for _, M, N in times)
    if not any(k in tv for tv in times.values() for k in BLOCK_BACKENDS):
        warns.append("no block-kernel timings in this data: block_min_k is not fitted, only carried "
                     "over; rerun the sweep with a current sweep_svd")
    elif params[2] >= INF:
        warns.append(f"the block kernel never beat the whole-matrix kernel up to k = {max_k}: "
                     f"rerun with a larger --max-k to find the crossover")
    # Above LARGE_K the Jacobi kernels are not timed by design: there bidiag_min_k decides.
    if max_k < LARGE_K and (params[N_SPLIT] >= max_k or any(c["gpu_speedup"] and c["gpu_speedup"] > 1.0
                                                         for c in cells if min(c["M"], c["N"]) == max_k)):
        warns.append(f"the GPU is still ahead of the CPU at the largest k measured ({max_k}): "
                     f"gpu_max_k is a lower bound, rerun with a larger --max-k to find the cap")
    if str(device.get("source", "")).startswith("default:untuned-device"):
        warns.append("this device has no entry in kTuned[]: it is running the untuned default; "
                     "paste the row above into src/svd.mm")
    if not s3:
        warns.append("no bidiag timings (a run from before the backend existed): the row's bidiag "
                     "thresholds are 0, never; rerun the sweep with a current sweep_svd")
    if "stage1b" not in res:
        warns.append("no gk timings (a run from before the backend existed): the row's gk window is "
                     "0, 0, never; rerun the sweep with a current sweep_svd")
    if (tuple(params) != tuple(CURRENT) or tuple(bidiag) != tuple(CURRENT_BIDIAG)
            or tuple(bidiag_cap) != tuple(CURRENT_BIDIAG_CAP) or tuple(gk) != tuple(CURRENT_GK)
            or tuple(values) != tuple(CURRENT_VALUES) or share != CURRENT_SHARE):
        warns.append(f"the fitted policy differs from the one in effect ({device.get('source', 'unknown')})")
    res["warnings"] = warns
    res["noise"] = te.noise_floor(repeats)
    res["tuned_row"] = tuned_row(device, params, bidiag, bidiag_cap, gk, values, share)
    res["env_line"] = env_line(params, bidiag, bidiag_cap, gk, values, share)
    return res


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _surface(cells, key):
    shapes = []
    for c in cells:
        if (c["M"], c["N"]) not in shapes:
            shapes.append((c["M"], c["N"]))
    Bs = sorted({c["batch"] for c in cells})
    g = {(c["M"], c["N"], c["batch"]): c for c in cells}
    letter = {"cpu": "c", "jacobi": "J", "block": "B", "qr": "j", "qrblock": "b", "gk": "g", "gk_share": "G",
              None: "."}
    lines = ["```", "      M x N  \\ batch " + " ".join(f"{b:>5}" for b in Bs)]
    for M, N in shapes:
        row = []
        for b in Bs:
            c = g.get((M, N, b))
            if c is None:
                row.append("    .")
            elif key == "speedup":
                v = c["gpu_speedup"]
                row.append(f"{v:5.2f}" if v is not None else "  gpu")
            else:
                row.append(f"    {letter[c[key]]}")
        lines.append(f"  {M:>5} x {N:<5}       " + " ".join(row))
    lines.append("```")
    return "\n".join(lines)


def write_report(res, path):
    d, s1, s2 = res["device"], res["stage1"], res["stage2"]
    cho = res["rules"]["chosen"]
    fv = te._fmt_v
    L = [f"# SVD routing on {d['name']} ({d['gpu_cores']} GPU cores)", "",
         "What every section and number below means: [reading-reports.md](https://github.com/c0rmac/metal-linalg/blob/main/docs/reading-reports.md).", ""]
    if res.get("calibration"):
        c = res["calibration"]["scale"]
        L += ["Cost model scaled to this device from one probe point: " +
              ", ".join(f"{k} x{v:.2f}" for k, v in sorted(c.items())) + " (1.00 is an M1).", ""]
    if res.get("machine"):
        st0 = next(iter(res["machine"].values()))
        L += ["Machine state: " + ", ".join(f"load {st['load_1m']:.1f}/{st['cpus']} at the {w}"
                                           for w, st in res["machine"].items()
                                           if st.get("load_1m") is not None) +
              f"; power {st0.get('power', 'unknown')}.", ""]
    else:
        L += ["Machine state was not recorded for this run.", ""]
    if res.get("drift"):
        dr = res["drift"]
        L += ["Probe point after the sweep relative to before it: " +
              ", ".join(f"{k} x{r:.2f}" for k, r in sorted(dr["ratio"].items())) +
              (" (stable)." if dr["ok"] else " (**unstable**)."), ""]
    L += [f"Generated by `tuning/tune_svd.py` from {res['n_points']} (shape, batch) points, "
          f"{len(res['grid']['shapes'])} shapes with M >= N, batch in {res['grid']['batch']}, seven "
          f"backends (and the CPU and bidiag again for singular values alone), {'one pass' if res['single_pass'] else 'two or more passes, min-of-repeats'}.", "",
          ]
    if len(res.get("submissions") or []) > 1:
        L.append(f"Combined from {len(res['submissions'])} submissions ({', '.join(res['submissions'])}): "
          f"each submission's fastest pass, then the median across submissions. The noise "
          f"floor is per submission.")
        L.append("")
    L += ["## Answer", ""]
    L += ["Row for `kTuned[]` in `src/svd.mm`:" if res["trustworthy"]
          else "**Indicative only; do not paste this row.** See the warnings below.", "",
          "```cpp", "// device, GPU cores,   qr_min_rows, qr_min_k,   block_min_k, block_min_k_batched, "
          "block_min_batch,   gpu_max_k, gpu_min_batch_times_k, gpu_min_batch, gpu_max_l,   "
          "values_gpu_max_k, values_gpu_min_batch_times_k, values_gpu_min_batch, values_gpu_max_l,   "
          "bidiag_min_k, values_bidiag_min_k, bidiag_max_batch, values_bidiag_max_batch,   gk_min_k, gk_max_k,   "
          "share_min_batch",
          res["tuned_row"], "```", "", "To try it without rebuilding:", "", "```sh", res["env_line"], "```", "",
          f"The policy in effect on this device came from `{d.get('source', 'unknown')}`. "
          f"Against the best measured backend at every point the fitted rule scores "
          f"{cho['geomean']:.4f} geometric-mean regret, worst {cho['worst']:.2f}x, {cho['over10']} of "
          f"{cho['n']} points losing more than 10%, and {cho['total_ratio']:.3f}x the oracle's total time.", ""]
    if res["warnings"]:
        L += ["## Warnings", ""] + [f"- {w}" for w in res["warnings"]] + [""]

    hdr = ["| rule | geomean regret | worst | >10% | total time / oracle | est. picks |", "|---|---|---|---|---|---|"]
    L += ["## Stage 1: which GPU backend", "",
          f"Scored against the best GPU backend at each of the {s1['n_points']} points, as if there were "
          f"no CPU: this is the rule a forced-GPU call (`SVD_DEVICE=gpu`) follows, and what a GPU with "
          f"more cores will lean on. Two independent choices. Precondition with QR iff the long side "
          f"is at least `qr_min_rows`, the short side at least `qr_min_k`, and the long side at least "
          f"twice the short one. Block kernel iff the short side is at least `block_min_k`, or at "
          f"least `block_min_k_batched` in a batch of `block_min_batch` or more.", ""] + hdr
    L += [te._stats_row(f"policy in effect {tuple(fv(v) for v in res['current'][:N_SPLIT])}", s1["current"]),
          te._stats_row(f"fitted {tuple(fv(v) for v in s1['chosen_params'])}", s1["chosen"]), ""]
    b1 = s1["band"]
    L += [f"Without a batch term, {b1['n_near_optimal']} of {s1['n_candidates']} combinations are within "
          f"{res['tolerance']*100:.1f}% of the best geomean: qr_min_rows {fv(b1['qr_min_rows'][0])} .. "
          f"{fv(b1['qr_min_rows'][1])}, qr_min_k {fv(b1['qr_min_k'][0])} .. {fv(b1['qr_min_k'][1])}, "
          f"block_min_k {fv(b1['block_min'][0])} .. {fv(b1['block_min'][1])}.", ""]
    if "band_batched" in s1:
        bb = s1["band_batched"]
        L += [f"With the batch term adopted (below), {bb['n_near_optimal']} combinations are within "
              f"{res['tolerance']*100:.1f}% of the best: block_min_k {fv(bb['block_min'][0])} .. "
              f"{fv(bb['block_min'][1])}, block_min_k_batched {fv(bb['block_lo'][0])} .. "
              f"{fv(bb['block_lo'][1])}, block_min_batch {fv(bb['batch_hi'][0])} .. "
              f"{fv(bb['batch_hi'][1])}. The curves vary one constant around the chosen combination.", ""]
    te._curve_block(L, "block_min_k", s1["curves"]["block_min"])
    if "block_lo" in s1["curves"]:
        te._curve_block(L, "block_min_k_batched", s1["curves"]["block_lo"])
    if "batch_hi" in s1["curves"]:
        te._curve_block(L, "block_min_batch", s1["curves"]["batch_hi"])
    for name in ("qr_min_rows", "qr_min_k"):
        if s1["curves"][name]:            # empty when no tall shape was measured
            te._curve_block(L, name, s1["curves"][name])
    if "holdout" in s1:
        h1 = s1["holdout"]
        L += [f"Held-out check of a batch-dependent block crossover (block from a smaller k once the "
              f"batch is large enough), fitted on {h1['train_points']} points and scored on the other "
              f"{h1['test_points']}; the verdict is a bootstrap over the test points.", "",
              "| rule | fitted on train | train geomean | test geomean | test worst | verdict |",
              "|---|---|---|---|---|---|",
              te._holdout_row("one block crossover", [fv(v) for v in h1["base"]["params"]], h1["base"]),
              te._holdout_row("batch-dependent block crossover",
                              {k: fv(v) for k, v in h1["batch_block"]["params"].items()},
                              h1["batch_block"]), ""]
    L += ["Best GPU backend per point (`J` whole-matrix kernel, `B` block kernel, `j` and `b` the same "
          "after QR, `g` gk, `G` gk shared with the CPU), then what the rule picks, the gk window included:", "",
          _surface(res["surface"], "best_gpu"), "", _surface(res["surface"], "rule_gpu"), ""]

    s1b = res.get("stage1b")
    if s1b:
        lo_, hi_ = s1b["chosen"]
        L += ["## Stage 1b: the gk backend", "",
              f"Inside a window of k = min(M, N), `gk` (Householder bidiagonalization and implicit QR, one "
              f"threadgroup per matrix, k <= {s1b['limit']} on this device; on the matrix itself where it "
              f"fits, else after a QR) instead of the Jacobi backend the split picks, fitted over the "
              f"{s1b['n_points']} points against the best GPU backend, gk included. Chosen: "
              + ("never." if not hi_ else f"k = {lo_} .. {hi_}."), ""] + hdr
        L += [te._stats_row("without gk (the split alone)", s1b["without"]),
              te._stats_row(f"with gk for k in {tuple(s1b['chosen'])}", s1b["with"]), ""]
        if s1b.get("refitted_with_share"):
            rw = s1b["refitted_with_share"]
            L += [f"Refitted once stage 1c chose to share batches with the CPU (gk shared leads the Jacobi "
                  f"backends where gk alone did not): k = {rw['before'][0]} .. {rw['before'][1]} became "
                  f"k = {rw['after'][0]} .. {rw['after'][1]}, the window in the row.", ""]
        bg = s1b["band"]
        h = s1b["holdout"]
        L += [f"{bg['n_near_optimal']} windows are within {res['tolerance']*100:.1f}% of the best geomean: "
              f"gk_min_k {bg['gk_min_k'][0]} .. {bg['gk_min_k'][1]}, gk_max_k {bg['gk_max_k'][0]} .. "
              f"{bg['gk_max_k'][1]}.", "",
              f"Held out: fitted on {h['train_points']} points (window {tuple(h['fitted'])}), scored on the "
              f"other {h['test_points']}: geomean {h['test']['geomean']:.4f}x, worst {h['test']['worst']:.2f}x, "
              f"against {h['without_test']['geomean']:.4f}x, worst {h['without_test']['worst']:.2f}x without gk.", ""]
        for key, what in (("speedup_vs_jacobi", "the best Jacobi backend"), ("speedup_vs_cpu", "the CPU")):
            if s1b.get(key):
                L += [f"gk over {what}, M x N x batch: " +
                      ", ".join(f"{M}x{N}x{b} {r:.2f}x" for M, N, b, r in s1b[key]), ""]

    s1c = res.get("stage1c")
    if s1c:
        L += ["## Stage 1c: sharing a batch with the CPU", "",
              "From a batch on, `gk_share`: gk and the CPU path at once on one batch, the GPU taking chunks "
              "from the front and the CPU from the back. Fitted against the best GPU backend, the shared one "
              f"included, on the {s1c['n_points']} points where it was timed and gk is the GPU's choice. "
              "Chosen: " + (f"from batch {s1c['chosen']}." if s1c["chosen"] else "never."), ""] + hdr
        L += [te._stats_row("gk alone", s1c["without"]),
              te._stats_row(f"shared from batch {s1c['chosen'] or 'never'}", s1c["with"]), "",
              "gk_share over gk alone, M x N x batch: " +
              ", ".join(f"{M}x{N}x{b} {r:.2f}x" for M, N, b, r in s1c["speedup_vs_gk"]), ""]

    L += ["## Stage 2: GPU or CPU", "",
          "GPU iff `k <= gpu_max_k`, `l <= gpu_max_l` (l = max(M, N)), `batch * k >= gpu_min_batch_times_k` "
          "and `batch >= gpu_min_batch`, "
          "with k = min(M, N), scored against the best of the CPU and the GPU backends (gk included). `worst` is over the points "
          "where the chosen backend was timed; a pick the cost model had to guess is listed in the "
          "warnings instead.", ""] + hdr
    L += [te._stats_row("oracle (best per point)", {"geomean": 1.0, "worst": 1.0, "over10": 0,
                                                    "total_ratio": 1.0, "estimated_picks": 0}),
          te._stats_row(f"policy in effect {tuple(fv(v) for v in res['current'][N_SPLIT:])}", s2["current"]),
          te._stats_row(f"fitted {tuple(fv(v) for v in s2['chosen_params'])}", s2["chosen"]), ""]
    b2 = s2["band"]
    L += [f"{b2['n_near_optimal']} of {s2['n_candidates']} combinations are within "
          f"{res['tolerance']*100:.1f}% of the best geomean: gpu_max_k {fv(b2['gpu_max_k'][0])} .. "
          f"{fv(b2['gpu_max_k'][1])}, gpu_min_batch_times_k {fv(b2['min_bk'][0])} .. {fv(b2['min_bk'][1])}, "
          f"gpu_min_batch {fv(b2['min_batch'][0])} .. {fv(b2['min_batch'][1])}" +
          (f", gpu_max_l {fv(b2['max_l'][0])} .. {fv(b2['max_l'][1])}" if "max_l" in b2 else "") + ".", ""]
    te._curve_block(L, "gpu_min_batch_times_k", s2["curves"]["min_bk"])
    if "gpu_max_l" in s2["curves"]:
        te._curve_block(L, "gpu_max_l", s2["curves"]["gpu_max_l"])
    te._curve_block(L, "gpu_min_batch", s2["curves"]["min_batch"])
    te._curve_block(L, "gpu_max_k", s2["curves"]["gpu_max_k"])
    h = s2["holdout"]
    L += [f"Held-out check of a per-k boundary against the product rule, fitted on {h['train_points']} "
          f"points and scored on the other {h['test_points']}; the verdict is a bootstrap.", "",
          "| rule | fitted on train | train geomean | test geomean | test worst | verdict |",
          "|---|---|---|---|---|---|",
          te._holdout_row("product rule", h["base"]["params"], h["base"]),
          te._holdout_row("per-k table", h["cpu_table"]["params"], h["cpu_table"]), "",
          "Best backend per point (`c` CPU, `J` whole-matrix kernel, `B` block kernel, `j` and `b` the "
          "same after QR, `g` gk, `G` gk shared with the CPU, `.` not measured), what the rule picks, and the speedup of the best GPU "
          "backend over the CPU:", "",
          _surface(res["surface"], "best"), "", _surface(res["surface"], "rule"), "",
          _surface(res["surface"], "speedup"), ""]

    s2b = res.get("stage2b")
    if s2b:
        c = s2b["chosen"]
        L += ["## Stage 2b: GPU or CPU for singular values alone", "",
              "The rule of stage 2 again, with constants of its own, for `svdvals`: both sides skip the vectors, "
              "by different amounts. Fitted on the "
              f"{s2b['n_points']} points where `gk` and the CPU were timed for singular values alone and `gk` "
              "is the GPU's choice. Chosen: " +
              ("never the GPU." if not c[0] else
               f"GPU iff k <= {c[0]}, l <= {fv(c[3])}, batch * k >= {c[1]} and batch >= {c[2]}."), ""] + hdr
        L += [te._stats_row("CPU always", s2b["cpu_always"]),
              te._stats_row("as with vectors (stage 2's rule)", s2b["as_vectors"]),
              te._stats_row(f"fitted {tuple(fv(v) for v in c)}", s2b["with"]), ""]
        h = s2b["holdout"]
        L += [f"Held out: fitted on {h['train_points']} points ({tuple(fv(v) for v in h['fitted'])}), scored on "
              f"the other {h['test_points']}: geomean {h['test']['geomean']:.4f}x, worst {h['test']['worst']:.2f}x, "
              f"against {h['as_vectors_test']['geomean']:.4f}x, worst {h['as_vectors_test']['worst']:.2f}x as with "
              "vectors.", "",
              "gk over the CPU for singular values alone, M x N x batch: " +
              ", ".join(f"{M}x{N}x{b} {r:.2f}x" for M, N, b, r in s2b["speedup_vs_cpu"]), ""]

    s3 = res.get("stage3")
    if s3:
        L += ["## Stage 3: the bidiag backend instead of the CPU", "",
              "Where the rule above chooses the CPU, the `bidiag` backend (GPU bidiagonalization, then "
              "LAPACK's bidiagonal solve) from a threshold k = min(M, N) on (0: never), for batches up to "
              "a cap (0: any; it solves a batch one matrix after another, the CPU path spreads one over "
              "every core), fitted on the "
              "points where bidiag was timed (k >= 128, within the cost cap): the region the threshold "
              "decides. With vectors it is scored against the best of all backends, bidiag included; "
              "for singular values alone (`svdvals`), bidiag against the CPU at the points where the "
              "rule chooses the CPU.", "",
              "| | threshold | batch cap | geomean regret | worst | without bidiag: geomean | worst | held out (fitted on half) |",
              "|---|---|---|---|---|---|---|---|"]
        for which, e in s3.items():
            h = e["holdout"]
            L.append(f"| {'with vectors' if which == 'vectors' else 'singular values alone'} | "
                     f"{e['chosen'] or 'never'} | {e.get('cap') or 'any'} | {e['with']['geomean']:.4f} | "
                     f"{e['with']['worst']:.2f}x | {e['without']['geomean']:.4f} | {e['without']['worst']:.2f}x | "
                     f"{te._fmt_td(h['fitted'])}: {h['test']['geomean']:.4f} vs "
                     f"{h['without_test']['geomean']:.4f} |")
        L.append("")
        for which, e in s3.items():
            if e["speedup_vs_cpu"]:
                L += [f"bidiag over the CPU ({'with vectors' if which == 'vectors' else 'singular values alone'}), "
                      "M x N x batch: " + ", ".join(f"{M}x{N}x{b} {r:.2f}x" for M, N, b, r in e["speedup_vs_cpu"]), ""]

    nf = res["noise"]
    L += ["## Noise floor", "",
          f"Pass-to-pass ratio, {nf['overall']['n']} measurements: median {nf['overall']['median']:.3f}, "
          f"p90 {nf['overall']['p90']:.3f}, max {nf['overall']['max']:.2f}.", "",
          "| runtime | n | median | p90 | max |", "|---|---|---|---|---|"]
    L += [f"| {r['bucket']} | {r['n']} | {r['median']:.3f} | {r['p90']:.3f} | {r['max']:.2f} |"
          for r in nf["by_runtime"]] + [""]
    with open(path, "w") as fh:
        fh.write("\n".join(L))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global CURRENT, CURRENT_BIDIAG, CURRENT_BIDIAG_CAP, CURRENT_GK, GK_LIMIT, CURRENT_VALUES, CURRENT_SHARE
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("binary", nargs="?", help="path to the built sweep_svd")
    ap.add_argument("--out", default="svd-tune-results", help="output directory")
    ap.add_argument("--passes", type=int, default=2, help="independent passes (minimum 2 for a noise floor)")
    ap.add_argument("--limit", type=int, default=240, help="per-point timeout, seconds")
    ap.add_argument("--quick", action="store_true", help="coarser grid, one pass; a smoke test only")
    ap.add_argument("--max-k", type=int, default=512,
                    help="largest square size on the grid (384 .. 4096 are added up to this)")
    ap.add_argument("--from", dest="from_json", help="re-render the report from an existing results.json")
    ap.add_argument("--reanalyse", metavar="RAW.CSV", nargs="+", help="re-run the analysis on raw.csv files")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    if args.from_json:
        write_report(json.load(open(args.from_json)), os.path.join(args.out, "report.md"))
        print(f"wrote {args.out}/report.md")
        return

    side = None
    if args.reanalyse:
        raw_paths = args.reanalyse
        side = te.load_sidecar(raw_paths)
        if args.binary:
            side = dict(side or {}, policy=query_policy(args.binary))
    else:
        if not args.binary:
            ap.error("give the sweep_svd binary, or --from / --reanalyse")
        policy = query_policy(args.binary)
        print(f"device: {policy['device']} ({policy['gpu_cores']} GPU cores), policy from {policy['source']}",
              file=sys.stderr)
        GK_LIMIT = policy.get("gk_limit", 0)   # before the grid, which times gk up to it
        state0 = te.machine_state()
        for prob in te.state_problems(state0):
            print(f"warning: {prob}; this sweep will be marked untrustworthy. "
                  f"Ctrl-C and rerun when the machine is idle.", file=sys.stderr)
        if state0.get("power") == "battery":
            print("warning: on battery power; prefer mains", file=sys.stderr)
        cal = calibrate(args.binary, args.limit)
        print("cost model scale: " + ", ".join(f"{k} x{v:.2f}" for k, v in cal["scale"].items()), file=sys.stderr)
        side = {"policy": policy, "calibration": cal, "machine": {"start": state0}}
        json.dump(side, open(os.path.join(args.out, "policy.json"), "w"), indent=1)
        raw = os.path.join(args.out, "raw.csv")
        print(f"sweep -> {raw}", file=sys.stderr)
        sweep(args.binary, point_grid(args.quick, args.max_k), 1 if args.quick else max(2, args.passes),
              args.limit, raw)
        raw_paths = [raw]
        side["drift"] = te.drift(cal["probe"]["ms"], probe(args.binary, args.limit))
        side["machine"]["end"] = te.machine_state()
        json.dump(side, open(os.path.join(args.out, "policy.json"), "w"), indent=1)

    device = {"name": "unknown", "gpu_cores": 0, "source": "unknown"}
    calibration = None
    if side:
        pol = side.get("policy")
        if pol:
            device = {"name": pol["device"], "gpu_cores": pol["gpu_cores"], "source": pol["source"]}
            CURRENT = policy_to_tuple(pol)
            CURRENT_BIDIAG = (pol.get("bidiag_min_k", 0), pol.get("values_bidiag_min_k", 0))
            CURRENT_BIDIAG_CAP = (pol.get("bidiag_max_batch", 0), pol.get("values_bidiag_max_batch", 0))
            CURRENT_GK = (pol.get("gk_min_k", 0), pol.get("gk_max_k", 0))
            CURRENT_SHARE = pol.get("share_min_batch", 0)
            CURRENT_VALUES = (pol.get("values_gpu_max_k", 0), pol.get("values_gpu_min_batch_times_k", 0),
                              pol.get("values_gpu_min_batch", 0), _cap(pol.get("values_gpu_max_l", NO_LIMIT)))
            GK_LIMIT = pol.get("gk_limit", 0)
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
    res = te.sub.portable(res)
    json.dump(res, open(os.path.join(args.out, "results.json"), "w"), indent=1)
    write_report(res, os.path.join(args.out, "report.md"))

    r, a, bm, lo, bh, gm, mb, mbatch, ml = res["chosen"]
    print(f"\n{res['n_points']} points on {device['name']} ({device['gpu_cores']} GPU cores)")
    print(f"GPU backend: qr_min_rows={te._fmt_v(r)} qr_min_k={te._fmt_v(a)} block_min_k={te._fmt_v(bm)} "
          f"block_min_k_batched={te._fmt_v(lo)} block_min_batch={te._fmt_v(bh)}   "
          f"({res['stage1']['chosen']['geomean']:.4f}x vs best GPU backend, worst {res['stage1']['chosen']['worst']:.2f}x)")
    print(f"CPU routing: gpu_max_k={te._fmt_v(gm)} gpu_max_l={te._fmt_v(ml)} "
          f"gpu_min_batch_times_k={te._fmt_v(mb)} gpu_min_batch={mbatch}   "
          f"({res['rules']['chosen']['geomean']:.4f}x vs best of all, worst {res['rules']['chosen']['worst']:.2f}x)")
    print(f"in effect ({device['source']}): {res['rules']['current']['geomean']:.4f}x, "
          f"worst {res['rules']['current']['worst']:.2f}x")
    if res.get("stage1b"):
        b1 = res["stage1b"]
        lo_, hi_ = res["gk_chosen"]
        print(f"gk window: {f'k = {lo_} .. {hi_}' if hi_ else 'never'}   ({b1['with']['geomean']:.4f}x vs best GPU backend, "
              f"worst {b1['with']['worst']:.2f}x; without {b1['without']['geomean']:.4f}x)")
    if res.get("stage1c"):
        c1 = res["stage1c"]
        print(f"share with the CPU: {'from batch ' + str(c1['chosen']) if c1['chosen'] else 'never'}   "
              f"({c1['with']['geomean']:.4f}x vs best GPU backend; gk alone {c1['without']['geomean']:.4f}x)")
    if res.get("stage2b"):
        b2 = res["stage2b"]
        print(f"svdvals routing: {tuple(te._fmt_v(v) for v in b2['chosen'])}   ({b2['with']['geomean']:.4f}x, worst "
              f"{b2['with']['worst']:.2f}x; as with vectors {b2['as_vectors']['geomean']:.4f}x, worst "
              f"{b2['as_vectors']['worst']:.2f}x)")
    for which, s3 in res.get("stage3", {}).items():
        print(f"bidiag ({which}): {te._fmt_td((s3['chosen'], s3.get('cap', 0)))}   ({s3['with']['geomean']:.4f}x, worst "
              f"{s3['with']['worst']:.2f}x; without {s3['without']['geomean']:.4f}x)")
    print(f"\nkTuned[] row:  {res['tuned_row']}" +
          ("" if res["trustworthy"] else "     <-- indicative only, do not paste (see warnings)"))
    for w in res["warnings"]:
        print("warning:", w)
    print(f"wrote {args.out}/results.json and report.md")


if __name__ == "__main__":
    main()
