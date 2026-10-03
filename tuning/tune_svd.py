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
l = max(M, N), which of six backends should `svd_accelerated` use?

    cpu      thin SVD through MLX (Accelerate LAPACK), QR first when tall
    jacobi   whole-matrix one-sided Jacobi kernel on the matrix itself
    block    block one-sided Jacobi kernel on the matrix itself
    qr       this library's QR, then the whole-matrix kernel on the k x k factor
    qrblock  this library's QR, then the block kernel on the k x k factor
    bidiag   GPU bidiagonalization (LAPACK's sgebrd, panels on the GPU) and
             back-transforms, LAPACK's bidiagonal solve; QR first when tall

svd.mm encodes the answer as a per-device routing policy (SvdPolicy):

    qr_min_rows, qr_min_k          precondition with QR iff l >= qr_min_rows,
                                   k >= qr_min_k and l >= 2k
    block_min_k                    block kernel from this k on
    block_min_k_batched,           ... and from this smaller k once the batch
      block_min_batch              reaches block_min_batch (0, 0 = not used)
    gpu_max_k                      GPU only up to this k
    gpu_min_batch_times_k          GPU only if batch * k is at least this
    gpu_min_batch                  GPU only from this batch (1 = no minimum)
    bidiag_min_k,                  where the rule says CPU, bidiag instead
      values_bidiag_min_k          from this k on, with vectors / for
                                   singular values alone (0 = never)

The method is that of tuning/tune_eigh.py, whose helpers this imports: fitted
in three stages (the GPU backend against the best GPU backend alone, then the
CPU boundary given it, then bidiag in place of the CPU on the points where it
was timed), every combination scored by regret, the flat region
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

K_SQUARE = [4, 8, 16, 32, 48, 64, 96, 128, 192, 256]
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
B_GRID      = list(B_LIST)

# The policy in effect, as the tuple the analysis works with:
#   (qr_min_rows, qr_min_k, block_min, block_lo, batch_hi, gpu_max_k, min_bk, min_batch)
# block_lo / batch_hi = INF means no batch-dependent block crossover. Replaced
# by what `sweep_svd --policy` reports; this is only the fallback for
# re-analysing a raw.csv that has no policy.json beside it.
CURRENT = (512, 64, 192, INF, INF, 64, 1024, 1)
N_SPLIT = 5              # the first five are the GPU split, the rest the CPU routing
# The bidiag thresholds in effect, (with vectors, singular values alone); 0 = never.
CURRENT_BIDIAG = (0, 0)

CAP_MS = 2500.0
SCALE = {"jacobi": 1.0, "block": 1.0, "qr": 1.0, "qrblock": 1.0, "cpu": 1.0, "bidiag": 1.0}
PROBE = (64, 128, 32, ["jacobi", "block", "qr", "qrblock", "cpu"])   # (batch, M, N, backends)


def est_ms(backend, M, N, b):
    """Per-call cost model, ms; an M1's, pessimistic, rescaled by calibrate()."""
    if backend.endswith(VALS):      # values alone: the reduction without the back-transforms
        return 0.6 * est_ms(backend[:-len(VALS)], M, N, b)
    if backend == "bidiag":         # serial over the batch: launches per column, then O(M N^2)
        k = min(M, N)
        return b * (1.0 + 0.04 * k + 4e-8 * max(M, N) * k * k) * SCALE.get(backend, 1.0)
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
    cap = CAP_LARGE_MS if k > LARGE_K else CAP_MS
    bidiag = k >= BIDIAG_MIN_GRID_K and est_ms("bidiag", M, N, b) <= cap
    if not ks and not bidiag:
        return []
    if bidiag or est_ms("cpu", M, N, b) <= CAP_MS:   # the reference wherever bidiag is timed
        ks.append("cpu")
    if bidiag:   # and the two again for singular values alone, the region values_bidiag_min_k decides
        ks += ["bidiag", "cpu" + VALS, "bidiag" + VALS]
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
    times = {p: v for p, v in times.items() if any(k in v for k in GPU_BACKENDS + ("bidiag",))}
    return times, repeats, subs


def split_bidiag(times):
    """-> (times for stages 1-2: the Jacobi backends and the CPU, at points
    with a Jacobi backend; times with vectors, bidiag included; times for
    singular values alone, as {"cpu", "bidiag"}). Runs from before the bidiag
    backend give empty second-stage parts."""
    base, vec, val = {}, {}, {}
    for p, tv in times.items():
        a = {k: v for k, v in tv.items() if k in GPU_BACKENDS or k == "cpu"}
        if any(k in a for k in GPU_BACKENDS):
            base[p] = a
        if "bidiag" in tv and "cpu" in tv:
            vec[p] = {k: v for k, v in tv.items() if not k.endswith(VALS)}
        if "bidiag" + VALS in tv and "cpu" + VALS in tv:
            val[p] = {"cpu": tv["cpu" + VALS], "bidiag": tv["bidiag" + VALS]}
    return base, vec, val

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
            max(1, pol.get("gpu_min_batch", 1)))    # absent: a build without the constant


def _cxx(params):
    r, a, bm, lo, bh, gm, mb, mbatch = params
    if mb >= INF:
        gm, mb = 0, 0
    if r >= INF or a >= INF:          # the QR-preconditioned backends are never used
        r, a = 1 << 30, 1 << 30
    if lo >= INF or bh >= INF:        # no batch-dependent block crossover
        lo, bh = 0, 0
    if bm >= INF:                     # the block kernel is never used
        bm = 1 << 30
    return (r, a, bm, lo, bh, "kSvdNoLimit" if gm >= INF else gm, mb, mbatch)


def tuned_row(device, params, bidiag=(0, 0)):
    """The line for kTuned[] in svd.mm; `bidiag` the two bidiag thresholds (0: never)."""
    r, a, bm, lo, bh, gm, mb, mbatch = _cxx(params)
    return (f'{{"{device["name"]}", {device["gpu_cores"]},   {r}, {a},   {bm}, {lo}, {bh},   '
            f'{gm}, {mb}, {mbatch},   {bidiag[0]}, {bidiag[1]}}},')


def env_line(params, bidiag=(0, 0)):
    r, a, bm, lo, bh, gm, mb, mbatch = _cxx(params)
    gm = NO_LIMIT if gm == "kSvdNoLimit" else gm
    return (f"SVD_QR_MIN_ROWS={r} SVD_QR_MIN_K={a} SVD_BLOCK_MIN_K={bm} "
            f"SVD_BLOCK_MIN_K_BATCHED={lo} SVD_BLOCK_MIN_BATCH={bh} "
            f"SVD_GPU_MAX_K={gm} SVD_GPU_MIN_BATCH_TIMES_K={mb} SVD_GPU_MIN_BATCH={mbatch} "
            f"SVD_BIDIAG_MIN_K={bidiag[0]} SVD_VALUES_BIDIAG_MIN_K={bidiag[1]}")


# ---------------------------------------------------------------------------
# Rules and scoring
# ---------------------------------------------------------------------------

def gpu_choice(split, M, N, b):
    rows, min_k, block_min, block_lo, batch_hi = split
    l, k = max(M, N), min(M, N)
    pre = l >= rows and k >= min_k and l >= 2 * k
    blk = k >= block_min or (k >= block_lo and b >= batch_hi)
    if pre:
        return "qrblock" if blk else "qr"
    return "block" if blk else "jacobi"


def rule_choice(params, M, N, b):
    """params = (qr_min_rows, qr_min_k, block_min, block_lo, batch_hi, gpu_max_k, min_bk, min_batch)."""
    k = min(M, N)
    if k > params[N_SPLIT] or b * k < params[N_SPLIT + 1] or b < params[N_SPLIT + 2]:
        return "cpu"
    return gpu_choice(params[:N_SPLIT], M, N, b)


def bidiag_choice(params, th, M, N, b):
    k = rule_choice(params, M, N, b)
    return "bidiag" if (k == "cpu" and th and min(M, N) >= th) else k


def fit_bidiag(choice, times, current, tol):
    """Stage 3: a bidiag threshold over the measured k from BIDIAG_MIN_GRID_K
    (and 0, never), scored on `times`, the points where bidiag was timed: over
    every point, the few large matrices it wins on would move the geometric
    mean by less than the tolerance. `choice(th, M, N, b)`. -> (chosen, scores)."""
    cands = [0] + sorted({min(M, N) for (_, M, N) in times if min(M, N) >= BIDIAG_MIN_GRID_K})
    scores = {th: te._score3(evaluate(lambda M, N, b, th=th: choice(th, M, N, b), times)) for th in cands}
    best = min(g for g, _, _ in scores.values())
    near = {th: v for th, v in scores.items() if v[0] <= best * (1 + tol)}
    if current in near:
        return current, scores
    return min(near, key=lambda th: (near[th][1], -th if th else 0)), scores


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


def fit_routing(split, times):
    out = {}
    for gm in GPU_MAX_KS:
        for mb in MIN_BKS:
            for mbatch in MIN_BATCHES:
                p = tuple(split) + (gm, mb, mbatch)
                e = score_rule(p, times)
                out[p] = (e["geomean"], e["worst"], e["over10"])
    return out


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
    split, gm, mb, mbatch = params[:N_SPLIT], params[N_SPLIT], params[N_SPLIT + 1], params[N_SPLIT + 2]
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
    B_GRID = sorted({b for b, _, _ in times})
    global MIN_BATCHES
    MIN_BATCHES = [1] + [b for b in B_GRID if 1 < b <= 32]


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyse(times, repeats, device, tol=0.005, drift_info=None, states=None):
    times_all = times
    times, vtimes, valtimes = split_bidiag(times)
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

    # ---- stage 2: CPU boundary
    R0, R1, R2 = N_SPLIT, N_SPLIT + 1, N_SPLIT + 2
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
                 "current_in_band": cur in near2},
        "curves": {"gpu_max_k": te._curve(lambda p: score_rule(p, times), params, R0, GPU_MAX_KS),
                   "min_bk": te._curve(lambda p: score_rule(p, times), params, R1, MIN_BKS),
                   "min_batch": te._curve(lambda p: score_rule(p, times), params, R2, MIN_BATCHES)},
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

    # ---- stage 3: the bidiag backend instead of the CPU
    bidiag = [0, 0]
    s3 = {}
    for which, full, cur, choice in (
            ("vectors", vtimes, CURRENT_BIDIAG[0], lambda th, M, N, b: bidiag_choice(params, th, M, N, b)),
            # svdvals follows the same GPU rule; where that says CPU, bidiag from its own threshold
            ("values", {p: tv for p, tv in valtimes.items() if rule_choice(params, *p[1:], p[0]) == "cpu"},
             CURRENT_BIDIAG[1], lambda th, M, N, b: "bidiag" if th and min(M, N) >= th else "cpu")):
        if not full:
            continue
        th, scores = fit_bidiag(choice, full, cur, tol)
        tr, ts = split_points(full)
        th_tr, _ = fit_bidiag(choice, tr, cur, tol) if tr else (th, None)
        s3[which] = {
            "n_points": len(full), "chosen": th,
            "without": _strip(evaluate(lambda M, N, b: choice(0, M, N, b), full)),
            "with": _strip(evaluate(lambda M, N, b, t=th: choice(t, M, N, b), full)),
            "curve": [[t, v[0], v[1]] for t, v in sorted(scores.items())],
            "holdout": {"train_points": len(tr), "test_points": len(ts), "fitted": th_tr,
                        "test": _strip(evaluate(lambda M, N, b: choice(th_tr, M, N, b), ts)),
                        "without_test": _strip(evaluate(lambda M, N, b: choice(0, M, N, b), ts))},
            "speedup_vs_cpu": sorted([[M, N, b, tv["cpu"] / tv["bidiag"]] for (b, M, N), tv in full.items()],
                                     key=lambda x: (min(x[0], x[1]), x[0], x[2])),
        }
        bidiag[0 if which == "vectors" else 1] = th
    res["stage3"] = s3
    res["bidiag_chosen"] = bidiag
    res["current_bidiag"] = list(CURRENT_BIDIAG)

    res["chosen"] = list(params)
    res["rules"] = {"current": _strip(score_rule(tuple(CURRENT), times)),
                    "chosen": _strip(score_rule(params, times))}

    cells = []
    for (b, M, N), tv in sorted(times.items(), key=lambda kv: (kv[0][2], kv[0][1], kv[0][0])):
        gpu = {k: tv[k] for k in GPU_BACKENDS if k in tv}
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
    if tuple(params) != tuple(CURRENT) or tuple(bidiag) != tuple(CURRENT_BIDIAG):
        warns.append(f"the fitted policy differs from the one in effect ({device.get('source', 'unknown')})")
    res["warnings"] = warns
    res["noise"] = te.noise_floor(repeats)
    res["tuned_row"] = tuned_row(device, params, bidiag)
    res["env_line"] = env_line(params, bidiag)
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
    letter = {"cpu": "c", "jacobi": "J", "block": "B", "qr": "j", "qrblock": "b", None: "."}
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
          f"{len(res['grid']['shapes'])} shapes with M >= N, batch in {res['grid']['batch']}, six "
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
          "block_min_batch,   gpu_max_k, gpu_min_batch_times_k, gpu_min_batch,   "
          "bidiag_min_k, values_bidiag_min_k",
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
          "after QR), then what the rule picks:", "",
          _surface(res["surface"], "best_gpu"), "", _surface(res["surface"], "rule_gpu"), ""]

    L += ["## Stage 2: GPU or CPU", "",
          "GPU iff `k <= gpu_max_k`, `batch * k >= gpu_min_batch_times_k` and `batch >= gpu_min_batch`, "
          "with k = min(M, N), scored against the best of the CPU and the four Jacobi backends. `worst` is over the points "
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
          f"gpu_min_batch {fv(b2['min_batch'][0])} .. {fv(b2['min_batch'][1])}.", ""]
    te._curve_block(L, "gpu_min_batch_times_k", s2["curves"]["min_bk"])
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
          "same after QR, `.` not measured), what the rule picks, and the speedup of the best GPU "
          "backend over the CPU:", "",
          _surface(res["surface"], "best"), "", _surface(res["surface"], "rule"), "",
          _surface(res["surface"], "speedup"), ""]

    s3 = res.get("stage3")
    if s3:
        L += ["## Stage 3: the bidiag backend instead of the CPU", "",
              "Where the rule above chooses the CPU, the `bidiag` backend (GPU bidiagonalization, then "
              "LAPACK's bidiagonal solve) from a threshold k = min(M, N) on (0: never), fitted on the "
              "points where bidiag was timed (k >= 128, within the cost cap): the region the threshold "
              "decides. With vectors it is scored against the best of all backends, bidiag included; "
              "for singular values alone (`svdvals`), bidiag against the CPU at the points where the "
              "rule chooses the CPU.", "",
              "| | threshold | geomean regret | worst | without bidiag: geomean | worst | held out (fitted on half) |",
              "|---|---|---|---|---|---|---|"]
        for which, e in s3.items():
            h = e["holdout"]
            L.append(f"| {'with vectors' if which == 'vectors' else 'singular values alone'} | "
                     f"{e['chosen'] or 'never'} | {e['with']['geomean']:.4f} | {e['with']['worst']:.2f}x | "
                     f"{e['without']['geomean']:.4f} | {e['without']['worst']:.2f}x | "
                     f"from {h['fitted'] or 'never'}: {h['test']['geomean']:.4f} vs "
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
    global CURRENT, CURRENT_BIDIAG
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

    r, a, bm, lo, bh, gm, mb, mbatch = res["chosen"]
    print(f"\n{res['n_points']} points on {device['name']} ({device['gpu_cores']} GPU cores)")
    print(f"GPU backend: qr_min_rows={te._fmt_v(r)} qr_min_k={te._fmt_v(a)} block_min_k={te._fmt_v(bm)} "
          f"block_min_k_batched={te._fmt_v(lo)} block_min_batch={te._fmt_v(bh)}   "
          f"({res['stage1']['chosen']['geomean']:.4f}x vs best GPU backend, worst {res['stage1']['chosen']['worst']:.2f}x)")
    print(f"CPU routing: gpu_max_k={te._fmt_v(gm)} gpu_min_batch_times_k={te._fmt_v(mb)} gpu_min_batch={mbatch}   "
          f"({res['rules']['chosen']['geomean']:.4f}x vs best of all, worst {res['rules']['chosen']['worst']:.2f}x)")
    print(f"in effect ({device['source']}): {res['rules']['current']['geomean']:.4f}x, "
          f"worst {res['rules']['current']['worst']:.2f}x")
    for which, s3 in res.get("stage3", {}).items():
        print(f"bidiag ({which}): from k={s3['chosen'] or 'never'}   ({s3['with']['geomean']:.4f}x, worst "
              f"{s3['with']['worst']:.2f}x; without {s3['without']['geomean']:.4f}x)")
    print(f"\nkTuned[] row:  {res['tuned_row']}" +
          ("" if res["trustworthy"] else "     <-- indicative only, do not paste (see warnings)"))
    for w in res["warnings"]:
        print("warning:", w)
    print(f"wrote {args.out}/results.json and report.md")


if __name__ == "__main__":
    main()
