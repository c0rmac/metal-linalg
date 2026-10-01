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

eigh.mm encodes the answer as a per-device routing policy (EighPolicy):

    simd_max_n              simd at or below this N, tg above        (GPU backend 1)
    block_min_n             block from this N on                     (GPU backend 2)
    block_min_n_batched,    optional: block also from this N once the
    block_min_batch           batch reaches this                     (GPU backend 2)
    gpu_max_n               GPU only up to this N                    (CPU routing)
    gpu_min_batch_times_n   GPU only if batch * N is at least this   (CPU routing)

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

BACKENDS = ("cpu", "simd", "tg", "block")
GPU_BACKENDS = ("simd", "tg", "block")
INF = 10 ** 9

N_LIST = [2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512]
B_LIST = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
N_QUICK = [4, 8, 16, 32, 64, 96, 128, 192, 256, 512]
B_QUICK = [1, 4, 16, 64, 256, 1024, 4096]

N_EXTRA = [768, 1024]   # added by --max-n

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
NO_LIMIT = 0xFFFFFFFF   # kEighNoLimit

# Per-call cost model, ms. Only used to skip points that would take several
# seconds per call, and to score a rule that picks a backend at such a point.
# The constants are an M1's, deliberately pessimistic; calibrate() rescales
# them to the device.
CAP_MS = 2500.0
SCALE = {"tg": 1.0, "simd": 1.0, "block": 1.0, "cpu": 1.0}


def configure_grids(times):
    """Candidate values from the measured grid."""
    global SIMD_MAXS, BLOCK_MINS, GPU_MAX_NS, B_GRID
    Ns = sorted({N for _, N in times})
    B_GRID = sorted({b for b, _ in times})
    SIMD_MAXS = [0] + [N for N in Ns if N <= 32]
    BLOCK_MINS = [N for N in Ns if N >= 32] + [INF]
    GPU_MAX_NS = [N for N in Ns if N >= 16] + [INF]
    global MIN_BATCHES
    MIN_BATCHES = [1] + [b for b in B_GRID if 1 < b <= 32]


def policy_to_tuple(pol):
    batched = pol.get("block_min_batch", 0) > 0
    gm = pol["gpu_max_n"]
    return (pol["simd_max_n"], pol["block_min_n"],
            pol["block_min_n_batched"] if batched else INF,
            pol["block_min_batch"] if batched else INF,
            INF if gm >= NO_LIMIT else gm, pol["gpu_min_batch_times_n"],
            max(1, pol.get("gpu_min_batch", 1)))    # absent: a build without the constant


def tuned_row(device, params):
    """The line to paste into kTuned[] in eigh.mm."""
    s, bm, lo, bh, gm, mb, mbatch = params
    if lo >= INF or bh >= INF:
        lo, bh = 0, 0
    if mb >= INF:                 # never GPU
        gm, mb = 0, 0
    gm_s = "kEighNoLimit" if gm >= INF else str(gm)
    bm_s = str(1 << 30) if bm >= INF else str(bm)
    return (f'{{"{device["name"]}", {device["gpu_cores"]},   {s}, {bm_s}, {lo}, {bh},   '
            f'{gm_s}, {mb}, {mbatch}}},')


def env_line(params):
    s, bm, lo, bh, gm, mb, mbatch = params
    if lo >= INF or bh >= INF:
        lo, bh = 0, 0
    if mb >= INF:
        gm, mb = 0, 0
    return (f"EIGH_SIMD_MAX_N={s} EIGH_BLOCK_MIN_N={(1 << 30) if bm >= INF else bm} "
            f"EIGH_BLOCK_MIN_N_BATCHED={lo} EIGH_BLOCK_MIN_BATCH={bh} "
            f"EIGH_GPU_MAX_N={NO_LIMIT if gm >= INF else gm} EIGH_GPU_MIN_BATCH_TIMES_N={mb} "
            f"EIGH_GPU_MIN_BATCH={mbatch}")


def est_ms(backend, N, b):
    n3 = float(N) ** 3
    if backend in ("tg", "simd"):
        t = 0.3 + 1e-5 * n3 * max(1.0, b / 8.0)
    elif backend == "block":
        t = 15.0 + 5.6e-7 * n3 * b
    else:
        t = b * (0.05 + 2.2e-7 * n3)   # cpu
    return t * SCALE.get(backend, 1.0)


def backends_for(N, b):
    ks = []
    if N <= 32 and est_ms("simd", N, b) <= CAP_MS:
        ks.append("simd")
    if est_ms("tg", N, b) <= CAP_MS:
        ks.append("tg")
    if N >= 32 and est_ms("block", N, b) <= CAP_MS:
        ks.append("block")
    if not ks:
        return []
    if est_ms("cpu", N, b) <= CAP_MS:
        ks.append("cpu")
    return ks


def point_grid(quick=False, max_n=512):
    Ns = list(N_QUICK if quick else N_LIST) + [n for n in N_EXTRA if n <= max_n]
    Ns = [n for n in Ns if n <= max(max_n, 2)]
    Bs = B_QUICK if quick else B_LIST
    pts = []
    for N in Ns:
        for b in Bs:
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
    est_total = sum(min(est_ms(k, N, b), CAP_MS) * 6 for b, N, ks in pts for k in ks) / 1000 * passes
    print(f"  {len(pts)} points, {runs} backend timings per pass x {passes} passes; "
          f"rough estimate {est_total/60 + len(pts) * passes * 0.5 / 60:.0f} min",
          file=sys.stderr)
    done = 0
    total = len(pts) * passes
    t0 = time.time()
    with open(out_csv, "w") as fh:
        fh.write("pass,batch,N,backend,ok,ms,p25,p75,reps\n")
        for p in range(passes):
            order = list(pts)
            random.Random(9000 + p).shuffle(order)   # independent shuffle per pass
            for job in order:
                for line in run_one(binary, job, limit):
                    fh.write(f"{p},{line}\n")
                fh.flush()
                done += 1
                if done % 20 == 0 or done == total:
                    el = time.time() - t0
                    eta = el / done * (total - done)
                    print(f"    {done}/{total}  elapsed {el/60:.1f}m  eta {eta/60:.1f}m",
                          file=sys.stderr)


def load(paths):
    """-> (times[(b,N)][backend], repeats, submissions); see submissions.combine."""
    times, repeats, subs = sub.combine(paths, lambda r: (int(r["batch"]), int(r["N"])))
    times = {p: v for p, v in times.items() if any(k in v for k in GPU_BACKENDS)}
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

def gpu_choice(split, N, b):
    simd_max, block_min, block_lo, batch_hi = split
    if N >= block_min or (N >= block_lo and b >= batch_hi):
        return "block"
    return "simd" if N <= simd_max else "tg"


def rule_choice(params, N, b):
    """params = (simd_max, block_min, block_lo, batch_hi, gpu_max_n, min_bn, min_batch)."""
    gpu_max_n, min_bn, min_batch = params[4], params[5], params[6]
    if N > gpu_max_n or b * N < min_bn or b < min_batch:
        return "cpu"
    return gpu_choice(params[:4], N, b)


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


def gpu_only(times):
    out = {}
    for p, tv in times.items():
        g = {k: v for k, v in tv.items() if k in GPU_BACKENDS}
        if g:
            out[p] = g
    return out


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


def analyse(times, repeats, device, tol=0.005, drift_info=None, states=None):
    configure_grids(times)
    single_pass = not any(len(v) >= 2 for v in repeats.values())
    res = {"device": device, "n_points": len(times), "single_pass": single_pass, "drift": drift_info,
           "grid": {"N": sorted({N for _, N in times}), "batch": sorted({b for b, _ in times})},
           "tolerance": tol, "current": list(CURRENT)}
    gtimes = gpu_only(times)
    train, test = split_points(times)
    gtrain, gtest = gpu_only(train), gpu_only(test)
    floor = noise_floor(repeats)

    # ---- stage 1: GPU split, against the best GPU backend ----
    cur_split = CURRENT[:4]
    sc1 = fit_split(gtimes)
    best1, near1 = near_optimal(sc1, tol)
    split = choose(near1, cur_split, _score3(score_split(cur_split, gtimes)), best1, tol)
    s1 = {
        "n_points": len(gtimes),
        "current": _strip(score_split(cur_split, gtimes)),
        "chosen": _strip(score_split(split, gtimes)),
        "chosen_params": list(split),
        "band": {"n_near_optimal": len(near1), "simd_max": _band(near1, 0),
                 "block_min": _band(near1, 1), "current_in_band": cur_split in near1},
        "curves": {"simd_max": _curve(lambda p: score_split(p, gtimes), split, 0, SIMD_MAXS),
                   "block_min": _curve(lambda p: score_split(p, gtimes), split, 1, BLOCK_MINS)},
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
            sc1b = fit_split_batch(split, gtimes)
            _, near1b = near_optimal(sc1b, tol)
            split = min(near1b, key=lambda p: (near1b[p][1], near1b[p][0]))
            s1["chosen_params"] = list(split)
            s1["chosen"] = _strip(score_split(split, gtimes))
            s1["band_batched"] = {"n_near_optimal": len(near1b), "block_min": _band(near1b, 1),
                                  "block_lo": _band(near1b, 2), "batch_hi": _band(near1b, 3)}
            s1["curves"]["block_min"] = _curve(lambda p: score_split(p, gtimes), split, 1, BLOCK_MINS)
            s1["curves"]["block_lo"] = _curve(lambda p: score_split(p, gtimes), split, 2,
                                              [v for v in BLOCK_MINS if v < split[1]])
            s1["batch_curve"] = _curve(lambda p: score_split(p, gtimes), split, 3, B_GRID)
    res["stage1"] = s1

    # ---- stage 2: CPU routing given the split ----
    cur_route = CURRENT[4:]
    sc2 = fit_routing(split, times)
    best2, near2 = near_optimal(sc2, tol)
    params = choose(near2, split + cur_route, _score3(score_rule(split + cur_route, times)), best2, tol)
    s2 = {
        "current": _strip(score_rule(split + cur_route, times)),
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

    # ---- the whole rule ----
    res["chosen"] = list(params)
    res["rules"] = {"current": _strip(score_rule(CURRENT, times)),
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
    if tuple(params) != tuple(CURRENT):
        warns.append(f"the fitted policy differs from the one in effect ({device.get('source', 'unknown')}): "
                     f"update this device's row in kTuned[] in src/eigh.mm")
    res["warnings"] = warns
    res["noise"] = floor
    res["trustworthy"] = (not single_pass) and (drift_info is None or drift_info["ok"]) and not busy
    res["machine"] = states
    res["tuned_row"] = tuned_row(device, params)
    res["env_line"] = env_line(params)
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
    letter = {"cpu": "c", "simd": "s", "tg": "t", "block": "B", None: "."}
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
             "gpu_max_n, gpu_min_batch_times_n, gpu_min_batch")
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
    same = tuple(res["current"]) == tuple(res["chosen"])
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
    L.append("Best GPU backend per point (`s` simd, `t` threadgroup, `B` block), then what the split picks:")
    L.append("")
    L.append(_surface(res["surface"], "best_gpu"))
    L.append("")
    L.append(_surface(res["surface"], "rule_gpu"))
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
    L.append("Best backend per point (`c` CPU, `s` simd, `t` threadgroup, `B` block, `.` not measured), "
             "what the whole rule picks, and the speedup of the best GPU backend over the CPU:")
    L.append("")
    L.append(_surface(res["surface"], "best"))
    L.append("")
    L.append(_surface(res["surface"], "rule"))
    L.append("")
    L.append(_surface(res["surface"], "speedup"))
    L.append("")

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
    global CURRENT
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("binary", nargs="?", help="path to the built sweep_eigh")
    ap.add_argument("--out", default="eigh-tune-results", help="output directory")
    ap.add_argument("--passes", type=int, default=2, help="independent passes (minimum 2 for a noise floor)")
    ap.add_argument("--limit", type=int, default=240, help="per-point timeout, seconds")
    ap.add_argument("--quick", action="store_true", help="coarser grid, one pass, roughly a third of the time")
    ap.add_argument("--max-n", type=int, default=512,
                    help="largest N on the grid (768 and 1024 are added up to this)")
    ap.add_argument("--from", dest="from_json", help="re-render the report from an existing results.json")
    ap.add_argument("--reanalyse", metavar="RAW.CSV", nargs="+", help="re-run the analysis on raw.csv files")
    args = ap.parse_args()

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
    print(f"\nkTuned[] row:  {res['tuned_row']}" +
          ("" if res["trustworthy"] else "     <-- indicative only, do not paste (see first warning)"))
    for w in res["warnings"]:
        print("warning:", w)
    print(f"wrote {args.out}/results.json and report.md")


if __name__ == "__main__":
    main()
