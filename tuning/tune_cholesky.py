#!/usr/bin/env python3
"""Measure Cholesky's routing on this Mac -- GPU or CPU, and which GPU kernel
-- and write up the result.

    cmake --build build --target sweep_cholesky
    python3 tuning/tune_cholesky.py build/sweep_cholesky

Produces, in --out (default cholesky-tune-results/):

    raw.csv        every timed run, so the analysis can be redone without remeasuring
    policy.json    the device and the policy in effect when it was measured, the
                   machine's state before and after, and the probe point's drift
    results.json   the analysis: the fitted kernel choice and GPU-or-CPU rule, how
                   they score against the best backend at every point, the noise floor
    report.md      a written report generated from results.json

Redo the analysis from existing measurements (several runs merge):

    python3 tuning/tune_cholesky.py --reanalyse cholesky-tune-results/raw.csv --out DIR

THE GRID
--------
Square matrices from 2 to 4096 at batches from 1 to 16384, every one whose
batch * N^2 is at most 2^26 floats (256 MB) and fits the memory budget
(tuning/submissions.py). Every point times the CPU path and each GPU kernel
that takes the size: simd up to 32, the threadgroup kernel up to 1024 (one
2048 x 2048 takes it 60 ms), the blocked path from 160 (up to 128 it runs the
threadgroup kernel itself). Two passes in independently shuffled orders, so
drift over the run does not correlate with size, combined by min-of-passes;
their disagreement is the noise floor. A probe point timed before and after
the sweep catches a machine that changed state in between.

THE FIT
-------
First the GPU kernel (cholesky_gpu_backend): simd up to simd_max_n, blocked
from blocked_min_n in batches of at most blocked_max_batch (0: any), else the
threadgroup kernel, fitted on the GPU's times alone (the routing that sends
most of a fast CPU's work to it still has the kernel matter on a Mac with a
slower CPU, and to CHOLESKY_DEVICE=gpu). Then GPU or CPU (uses_gpu in
src/cholesky.mm): the GPU iff gpu_min_n <= N <= gpu_max_n, batch * N >=
gpu_min_batch_times_n and batch >= gpu_min_batch, or N >= gpu_large_min_n in a
batch of at most gpu_large_max_batch. Each fit takes, of the candidates within
0.5% of the best geometric-mean regret, the one with the smallest worst case.
"""

import argparse
import bisect
import json
import math
import os
import random
import subprocess
import sys
import time
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import submissions as sub   # noqa: E402
import tune_eigh as te      # noqa: E402  machine state checks

GPU_BACKENDS = ("simd", "tg", "blocked")
BACKENDS = GPU_BACKENDS + ("cpu",)
NO_LIMIT = 0xFFFFFFFF
INF = float("inf")

SIZES = (2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 160, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096)
BATCHES = (1, 2, 4, 16, 64, 256, 1024, 4096, 16384)
QUICK_SIZES = (8, 32, 128, 512, 2048)
QUICK_BATCHES = (1, 64, 4096)
MAX_ELEMENTS = 1 << 26     # batch * N^2
SIMD_MAX_N = 32            # the simd kernel's largest matrix
TG_MAX_N = 1024            # the threadgroup kernel timed up to this
BLOCKED_MIN_N = 129        # up to 128 the blocked path is the threadgroup kernel
PROBE = (16, 512, ["cpu", "tg", "blocked"])   # (batch, N, backends): the drift check
TOL = 0.005                # candidates within this of the best geomean are near-ties
MISSING = 100.0            # a rule choosing a backend not timed at a point: its regret there


# ---------------------------------------------------------------------------
# Measuring
# ---------------------------------------------------------------------------

def backends_for(b, N):
    out = ["cpu"]
    if N <= SIMD_MAX_N:
        out.append("simd")
    if N <= TG_MAX_N:
        out.append("tg")
    if N >= BLOCKED_MIN_N:
        out.append("blocked")
    return out


def grid(quick=False, max_n=4096):
    pts = []
    for N in (QUICK_SIZES if quick else SIZES):
        if N > max_n:
            continue
        for b in (QUICK_BATCHES if quick else BATCHES):
            if b * N * N <= MAX_ELEMENTS and sub.fits_memory(b * N * N):
                pts.append((b, N))
    return pts


def run_one(binary, job, limit, attempts=3):
    """One point, every backend of it in one process: CSV lines. Retries on a
    crash, so a transient failure does not silently become a missing point."""
    b, N, backends = job
    for _ in range(attempts):
        try:
            r = subprocess.run([binary, str(b), str(N), ",".join(backends)],
                               capture_output=True, text=True, timeout=limit)
            lines = [l for l in r.stdout.strip().splitlines() if l.count(",") == 7]
            if r.returncode == 0 and len(lines) == len(backends):
                return lines
        except subprocess.TimeoutExpired:
            break
    return [f"{b},{N},{k},0,0,0,0,0" for k in backends]


def sweep(binary, pts, passes, limit, out_csv):
    jobs = [(b, N, backends_for(b, N)) for b, N in pts]
    total = len(jobs) * passes
    print(f"  {len(pts)} points x {passes} passes = {total} processes "
          f"({sum(len(j[2]) for j in jobs) * passes} timed backends)", file=sys.stderr)
    done, t0 = 0, time.time()
    with open(out_csv, "w") as fh:
        fh.write("pass,batch,N,backend,ok,ms,p25,p75,reps\n")
        for p in range(passes):
            order = list(jobs)
            random.Random(7100 + p).shuffle(order)   # independent shuffle per pass
            for job in order:
                for line in run_one(binary, job, limit):
                    fh.write(f"{p},{line}\n")
                fh.flush()
                done += 1
                if done % 20 == 0 or done == total:
                    el = time.time() - t0
                    print(f"    {done}/{total}  elapsed {el / 60:.1f}m  eta {el / done * (total - done) / 60:.1f}m",
                          file=sys.stderr)


def probe(binary, limit):
    """Min of three runs of the probe point, per backend."""
    out = {}
    for _ in range(3):
        for line in run_one(binary, PROBE, limit):
            f = line.split(",")
            if f[3] == "1" and float(f[4]) > 0:
                out[f[2]] = min(out.get(f[2], INF), float(f[4]))
    return out


def load(paths):
    """-> (best[(b, N)][backend], repeats, submissions); see submissions.combine."""
    return sub.combine(paths, lambda r: (int(r["batch"]), int(r["N"])))


def query_policy(binary):
    r = subprocess.run([binary, "--policy"], capture_output=True, text=True, timeout=60)
    if r.returncode != 0 or not r.stdout.strip():
        sys.exit(f"{binary} --policy failed; rebuild sweep_cholesky")
    return json.loads(next(l for l in reversed(r.stdout.strip().splitlines()) if l.startswith("{")))


def load_sidecar(binary, raw_paths):
    """The sweep's policy.json beside its raw.csv: {"policy", "machine",
    "drift"}; without one, the binary's policy alone."""
    if raw_paths:
        path = os.path.join(os.path.dirname(os.path.abspath(raw_paths[0])), "policy.json")
        if os.path.exists(path):
            side = json.load(open(path))
            return side if "policy" in side else {"policy": side}
    if binary:
        return {"policy": query_policy(binary)}
    return {"policy": {"device": "unknown", "gpu_cores": 0, "source": "unknown"}}


# ---------------------------------------------------------------------------
# Rules and scoring
# ---------------------------------------------------------------------------

def gpu_time(t, name, N):
    """A GPU kernel's time at a point; the blocked path up to 128 is the
    threadgroup kernel (cholesky_gpu.mm), timed as that."""
    if name in t:
        return t[name]
    if name == "blocked" and N < BLOCKED_MIN_N:
        return t.get("tg")
    return None


def kernel_rule(k):
    """cholesky_gpu_backend: (simd_max_n, blocked_min_n, blocked_max_batch)."""
    simd, bmin, bcap = k

    def rule(b, N):
        if N <= min(SIMD_MAX_N, simd):
            return "simd"
        if N >= bmin and (not bcap or b <= bcap):
            return "blocked"
        return "tg"
    return rule


def uses_gpu(r, b, N):
    """uses_gpu in src/cholesky.mm: r = (gpu_max_n, gpu_min_batch_times_n,
    gpu_min_batch, gpu_min_n, gpu_large_min_n, gpu_large_max_batch)."""
    gm, mb, mbatch, mk, lk, lcap = r
    if lk and N >= lk and (not lcap or b <= lcap):
        return True
    return mk <= N <= gm and b * N >= mb and b >= max(1, mbatch)


def evaluate(choose, pts, gpu_only=False):
    """Regret of a rule against the best backend timed at each point (the
    best GPU kernel with gpu_only): geomean, worst, total time over best."""
    reg, worst, worst_pt, slow, tot, tbest = [], 1.0, None, 0, 0.0, 0.0
    for (b, N), t in pts.items():
        pool = {k: gpu_time(t, k, N) for k in GPU_BACKENDS} if gpu_only else dict(t)
        pool = {k: v for k, v in pool.items() if v is not None}
        if not pool:
            continue
        best = min(pool.values())
        name = choose(b, N)
        got = gpu_time(t, name, N) if name != "cpu" else t.get("cpu")
        r = got / best if got is not None else MISSING
        reg.append(r)
        tot += got if got is not None else best * MISSING
        tbest += best
        if r > worst:
            worst, worst_pt = r, [b, N]
        slow += r > 1.10
    if not reg:
        return None
    return {"geomean": math.exp(math.fsum(map(math.log, reg)) / len(reg)), "worst": worst,
            "worst_point": worst_pt, "over_10pct": slow, "n": len(reg), "total": tot / tbest}


def stat(e):
    return {"geomean": round(e["geomean"], 4), "worst": round(e["worst"], 3), "worst_point": e["worst_point"],
            "over_10pct": e["over_10pct"], "n": e["n"], "total": round(e["total"], 4)}


def fit_kernel(best):
    """The kernel choice, on the GPU's times alone."""
    pts = {k: v for k, v in best.items() if any(g in v for g in GPU_BACKENDS)}
    sizes = sorted({N for _, N in pts})
    batches = sorted({b for b, _ in pts})
    cands = [(s, bmin, bcap)
             for s in [0] + [N for N in sizes if N <= SIMD_MAX_N]
             for bmin in [N for N in sizes if 48 <= N <= 1536]
             for bcap in [0] + batches]
    scored = {c: evaluate(kernel_rule(c), pts, gpu_only=True) for c in cands}
    g = min(e["geomean"] for e in scored.values())
    near = {c: e for c, e in scored.items() if e["geomean"] <= g * (1 + TOL)}
    # The smallest worst case; then simd the furthest, blocked the latest and
    # its batch cap none (any batch) or the largest: ties are points the
    # candidates route alike (up to 128 the blocked path is the threadgroup
    # kernel, so every threshold up to the first size it differs at ties).
    chosen = min(near, key=lambda c: (near[c]["worst"], near[c]["geomean"], -c[0], -c[1],
                                      -(c[2] if c[2] else INF)))
    return chosen, {"simd_max_n": chosen[0], "blocked_min_n": chosen[1], "blocked_max_batch": chosen[2],
                    "score": stat(scored[chosen]),
                    "simd_never": stat(evaluate(kernel_rule((0,) + chosen[1:]), pts, gpu_only=True)),
                    "blocked_everywhere": stat(evaluate(kernel_rule((chosen[0], 48, 0)), pts, gpu_only=True))}


def fit_routing(best, kernel):
    """GPU or CPU, the kernel given, on every point with a CPU time."""
    krule = kernel_rule(kernel)
    pts = {k: v for k, v in best.items() if "cpu" in v and gpu_time(v, krule(*k), k[1]) is not None}
    if not pts:
        return None
    sizes = sorted({N for _, N in pts})
    bns = sorted({b * N for b, N in pts})
    batches = sorted({b for b, _ in pts})
    # Limits inside the grid only: one at the largest size measured fits as
    # well as none but says nothing beyond it, where the GPU may win. A window
    # spans two measured sizes at least: one around a single size says
    # nothing about those either side of it (on the run of 2026-10-10 it took
    # 16384 matrices of 2 x 2 for the GPU at 1.02x, inside the noise). The
    # lower bound goes up to the largest sizes: the GPU wins batches of large
    # matrices before lone ones (1.1x for 2 of 1536 on an M5 Pro, 0.66x for
    # one), which the window with gpu_min_batch and the large clause together
    # can say.
    by_triple = defaultdict(list)
    for mk in [0] + sizes:
        for gm in [0] + sizes[:-1] + [INF]:
            if gm and (gm < mk or sum(mk <= N <= gm for N in sizes) < 2):
                continue
            for mbatch in [1] + [b for b in batches if 1 < b <= 16]:
                by_triple[(gm, mbatch, mk)] = [0] + bns
    large_cands = [(0, 0)] + [(lk, cap) for lk in sizes for cap in [0] + batches]

    def rule_of(params, large):
        gm, mb, mbatch, mk = params
        r = (gm, mb, mbatch, mk) + tuple(large)
        return lambda b, N: krule(b, N) if uses_gpu(r, b, N) else "cpu"

    def approx(points, large):
        """{candidate: geomean regret}, every candidate at once by prefix sums."""
        lk, lcap = large
        rows = []
        for (b, N), t in points.items():
            bst = min(t.values())
            g = gpu_time(t, krule(b, N), N)
            forced = bool(lk) and N >= lk and (not lcap or b <= lcap)
            rows.append((N, b, b * N, math.log(t["cpu"] / bst), math.log(g / bst), forced))
        n, out = len(rows), {}
        for (gm, mbatch, mk), mbs in by_triple.items():
            fixed, elig = 0.0, []
            for N, b, bn, lc, lg, forced in rows:
                if forced:
                    fixed += lg
                elif N < mk or N > gm or b < mbatch:
                    fixed += lc
                else:
                    elig.append((bn, lc, lg))
            elig.sort()
            keys = [e[0] for e in elig]
            pre = [0.0]
            for e in elig:
                pre.append(pre[-1] + e[1])
            suf = [0.0] * (len(elig) + 1)
            for i in range(len(elig) - 1, -1, -1):
                suf[i] = suf[i + 1] + elig[i][2]
            for mb in mbs:
                i = bisect.bisect_left(keys, mb)
                out[(gm, mb, mbatch, mk)] = math.exp((fixed + pre[i] + suf[i]) / n)
        return out

    def fit_rule(points, large):
        a = approx(points, large)
        cut = min(a.values()) * (1 + TOL) * (1 + 1e-9)
        scored = {c: evaluate(rule_of(c, large), points) for c, v in a.items() if v <= cut}
        g = min(e["geomean"] for e in scored.values())
        near = {c: e for c, e in scored.items() if e["geomean"] <= g * (1 + TOL)}
        # ties: the smallest worst case, then the GPU used least
        return min(near, key=lambda c: (near[c]["worst"], near[c]["geomean"], -c[3], c[0], -c[1], -c[2]))

    def fit_large(points, params):
        scored = {c: evaluate(rule_of(params, c), points) for c in large_cands}
        g = min(e["geomean"] for e in scored.values())
        near = {c: e for c, e in scored.items() if e["geomean"] <= g * (1 + TOL)}
        # ties: the smallest worst case; then the clause's size the largest
        # (used least), its batch cap any or the largest (the GPU's lead on
        # large matrices grows with the batch)
        return min(near, key=lambda c: (near[c]["worst"], near[c]["geomean"], -c[0] if c[0] else 0,
                                        -(c[1] if c[1] else INF)))

    def fit(points):
        # the rule, the clause given it, the rule again; or the clause first
        # (the CPU elsewhere), then the rule: the better of the two
        fits = []
        params = fit_rule(points, (0, 0))
        large = fit_large(points, params)
        if large != (0, 0):
            params = fit_rule(points, large)
        fits.append((params, large))
        large = fit_large(points, (0, 0, 1, 0))
        if large != (0, 0):
            fits.append((fit_rule(points, large), large))
        scored = [(evaluate(rule_of(p, l), points), p, l) for p, l in fits]
        e, params, large = min(scored, key=lambda x: (x[0]["geomean"], x[0]["worst"]))
        return params, large, e

    params, large, e = fit(pts)
    keys = sorted(pts)
    train = {k: pts[k] for i, k in enumerate(keys) if i % 2 == 0}
    test = {k: pts[k] for i, k in enumerate(keys) if i % 2 == 1}
    p_tr, l_tr, _ = fit(train)
    gm, mb, mbatch, mk = params
    lim = lambda v: NO_LIMIT if v >= INF else v
    return {
        "gpu_max_n": lim(gm), "gpu_min_batch_times_n": mb, "gpu_min_batch": mbatch, "gpu_min_n": mk,
        "gpu_large_min_n": large[0], "gpu_large_max_batch": large[1],
        "chosen": stat(e),
        "gpu_always": stat(evaluate(lambda b, N: krule(b, N), pts)),
        "cpu_always": stat(evaluate(lambda b, N: "cpu", pts)),
        "held_out": {"fitted": [lim(p_tr[0]), p_tr[1], p_tr[2], p_tr[3], l_tr[0], l_tr[1]],
                     "routing": stat(evaluate(rule_of(p_tr, l_tr), test)),
                     "cpu_always": stat(evaluate(lambda b, N: "cpu", test))},
    }


def noise_floor(repeats):
    """Pass-to-pass ratio, bucketed by runtime."""
    buckets, allr = defaultdict(list), []
    for (b, N, k, *_), v in repeats.items():
        if len(v) < 2 or min(v) <= 0:
            continue
        r = max(v) / min(v)
        allr.append(r)
        t = min(v)
        lab = ("<1 ms" if t < 1 else "1-3 ms" if t < 3 else "3-10 ms" if t < 10 else
               "10-30 ms" if t < 30 else ">30 ms")
        buckets[lab].append(r)

    def q(a, p):
        a = sorted(a)
        return a[min(len(a) - 1, int(p * (len(a) - 1)))] if a else 0.0
    return {"overall": {"n": len(allr), "median": q(allr, .5), "p90": q(allr, .9), "max": max(allr) if allr else 0},
            "by_runtime": [{"bucket": lab, "n": len(buckets[lab]), "median": q(buckets[lab], .5),
                            "p90": q(buckets[lab], .9)}
                           for lab in ("<1 ms", "1-3 ms", "3-10 ms", "10-30 ms", ">30 ms") if buckets.get(lab)]}


def analyse(best, repeats, device, states=None, drift_info=None, single_pass=False):
    kernel, kfit = fit_kernel(best)
    routing = fit_routing(best, kernel)
    krule = kernel_rule(kernel)
    r = (routing["gpu_max_n"], routing["gpu_min_batch_times_n"], routing["gpu_min_batch"], routing["gpu_min_n"],
         routing["gpu_large_min_n"], routing["gpu_large_max_batch"]) if routing else (0, 0, 1, 0, 0, 0)
    points = []
    for (b, N), t in sorted(best.items(), key=lambda x: (x[0][1], x[0][0])):
        routed = krule(b, N) if uses_gpu(r, b, N) else "cpu"
        points.append({"batch": b, "N": N, "ms": {k: round(v, 4) for k, v in sorted(t.items())},
                       "best": min(t, key=t.get), "routed": routed})
    noise = noise_floor(repeats)
    warns = []
    if single_pass:
        warns.append("single pass: there is no noise floor and no min-of-repeats, so this run is a smoke "
                     "test of the pipeline, not a measurement. Do not paste its row; run without --quick")
    busy = [f"{p} at the {when} of the sweep" for when, st in (states or {}).items() for p in te.state_problems(st)]
    if busy:
        warns.insert(0, "the machine was not idle: " + "; ".join(busy) + ". The CPU backend is slowed most "
                        "by this, which biases the routing toward the GPU; rerun when idle")
    if any(st.get("power") == "battery" for st in (states or {}).values()):
        warns.append("run on battery power; macOS may limit performance differently than on mains")
    if drift_info and not drift_info["ok"]:
        warns.insert(0, "the machine changed state during the sweep: the probe point moved by " +
                     ", ".join(f"{k} x{v:.2f}" for k, v in sorted(drift_info["ratio"].items())) +
                     "; rerun on mains power with Low Power Mode off and nothing else running")
    if not single_pass and noise["overall"]["median"] > 1.10:
        warns.append(f"run-to-run noise {noise['overall']['median']:.2f}x (median) is above 10%")
    trustworthy = not (single_pass or busy or (drift_info and not drift_info["ok"])
                       or noise["overall"]["median"] > 1.10)
    lim = lambda v: "kCholeskyNoLimit" if v >= NO_LIMIT else str(v)
    rt = routing or {"gpu_max_n": 0, "gpu_min_batch_times_n": 0, "gpu_min_batch": 1, "gpu_min_n": 0,
                     "gpu_large_min_n": 0, "gpu_large_max_batch": 0}
    entry = (f'{{"{device.get("device", "unknown")}", {device.get("gpu_cores", 0)},   '
             f'{kernel[0]}, {kernel[1]}, {kernel[2]},   {lim(rt["gpu_max_n"])}, {rt["gpu_min_batch_times_n"]}, '
             f'{rt["gpu_min_batch"]}, {rt["gpu_min_n"]},   {rt["gpu_large_min_n"]}, {rt["gpu_large_max_batch"]}}},')
    return {"device": {"name": device.get("device"), "gpu_cores": device.get("gpu_cores"),
                       "cpu_threads": device.get("cpu_threads"), "source": device.get("source")},
            "kernel": kfit, "routing": routing, "points": points, "noise": noise, "drift": drift_info,
            "states": states, "trustworthy": bool(trustworthy), "warnings": warns, "ktuned_entry": entry}


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_report(res, path):
    d, k, rt = res["device"], res["kernel"], res["routing"]
    L = [f"# Cholesky routing: {d['name']}, {d['gpu_cores']} GPU cores", ""]
    if res.get("submissions"):
        L += [f"From {len(res['submissions'])} run(s): {', '.join(res['submissions'])}.", ""]
    for w in res["warnings"]:
        L.append(f"> **Warning:** {w}")
    if res["warnings"]:
        L.append("")
    L += ["## Answer", "",
          "Row for `kTuned[]` (src/tuned/cholesky.inc, generated):" if res["trustworthy"]
          else "**Indicative only; do not use this row.** See the warnings above.", "",
          "```cpp", "// device, GPU cores,   simd_max_n, blocked_min_n, blocked_max_batch,   gpu_max_n, "
          "gpu_min_batch_times_n, gpu_min_batch, gpu_min_n,   gpu_large_min_n, gpu_large_max_batch",
          res["ktuned_entry"], "```", "",
          "## The GPU kernel", "",
          f"simd up to {k['simd_max_n']}, blocked from {k['blocked_min_n']} "
          f"(batches of {'any size' if not k['blocked_max_batch'] else 'at most ' + str(k['blocked_max_batch'])}), "
          "the threadgroup kernel between. Against the best GPU kernel at every point:", "",
          "| rule | geomean | worst | points over 10% |", "|---|---|---|---|"]
    for name, key in (("fitted", "score"), ("no simd", "simd_never"), ("blocked from 48", "blocked_everywhere")):
        s = k[key]
        L.append(f"| {name} | {s['geomean']:.4f}x | {s['worst']:.2f}x at {s['worst_point']} | {s['over_10pct']} of {s['n']} |")
    L.append("")
    if rt:
        gm = "no limit" if rt["gpu_max_n"] >= NO_LIMIT else rt["gpu_max_n"]
        L += ["## GPU or CPU", "",
              f"The GPU iff {rt['gpu_min_n']} <= N <= {gm}, batch x N >= {rt['gpu_min_batch_times_n']} and "
              f"batch >= {rt['gpu_min_batch']}" +
              (f"; or N >= {rt['gpu_large_min_n']} in a batch of "
               f"{'any size' if not rt['gpu_large_max_batch'] else 'at most ' + str(rt['gpu_large_max_batch'])}"
               if rt["gpu_large_min_n"] else "") + ". Against the best backend at every point:", "",
              "| rule | geomean | worst | points over 10% | total time over best |", "|---|---|---|---|---|"]
        for name, s in (("fitted", rt["chosen"]), ("GPU always", rt["gpu_always"]), ("CPU always", rt["cpu_always"]),
                        ("held out: fitted on half", rt["held_out"]["routing"]),
                        ("held out: CPU always", rt["held_out"]["cpu_always"])):
            L.append(f"| {name} | {s['geomean']:.4f}x | {s['worst']:.2f}x at {s['worst_point']} | "
                     f"{s['over_10pct']} of {s['n']} | {s['total']:.3f}x |")
        L.append("")
    n = res["noise"]["overall"]
    L += ["## Noise", "", f"Pass-to-pass ratio: median {n['median']:.3f}, p90 {n['p90']:.3f}, max {n['max']:.2f} "
          f"over {n['n']} timings.", ""]
    if res.get("drift"):
        L += [f"Probe point {PROBE[1]} x {PROBE[1]}, batch {PROBE[0]}, after the sweep over before: " +
              ", ".join(f"{b} {v:.3f}x" for b, v in sorted(res["drift"]["ratio"].items())) + ".", ""]
    L += ["## Every point", "", "Median ms, min over passes (the fastest in bold); `routed` is the fitted rule's choice.", "",
          "| N | batch | cpu | simd | tg | blocked | best | routed | GPU's best over CPU |", "|---|---|---|---|---|---|---|---|---|"]
    for p in res["points"]:
        ms = p["ms"]
        cells = []
        for b in ("cpu", "simd", "tg", "blocked"):
            v = ms.get(b)
            cells.append("—" if v is None else (f"**{v:.3f}**" if b == p["best"] else f"{v:.3f}"))
        g = [ms[b] for b in GPU_BACKENDS if b in ms]
        sp = f"{ms['cpu'] / min(g):.2f}x" if g and "cpu" in ms else "—"
        L.append(f"| {p['N']} | {p['batch']} | " + " | ".join(cells) + f" | {p['best']} | {p['routed']} | {sp} |")
    open(path, "w").write("\n".join(L) + "\n")


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("binary", nargs="?", help="path to the built sweep_cholesky")
    ap.add_argument("--out", default="cholesky-tune-results", help="output directory")
    ap.add_argument("--passes", type=int, default=2, help="independent passes; 2 is the minimum for a noise floor")
    ap.add_argument("--limit", type=int, default=300, help="per-point timeout, seconds")
    ap.add_argument("--max-n", type=int, default=4096, help="the largest N measured")
    ap.add_argument("--quick", action="store_true", help="a few points, one pass: a smoke test")
    ap.add_argument("--reanalyse", metavar="RAW.CSV", nargs="+",
                    help="redo the analysis from existing raw.csv files (merged) without remeasuring")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    if a.reanalyse:
        best, repeats, subs = load(a.reanalyse)
        if not best:
            sys.exit(f"no usable measurements in {a.reanalyse}")
        side = load_sidecar(a.binary, a.reanalyse)
        res = analyse(best, repeats, side["policy"], side.get("machine"), side.get("drift"))
        res["submissions"] = subs
    else:
        if not a.binary:
            ap.error("need the path to sweep_cholesky (or --reanalyse)")
        passes = 1 if a.quick else a.passes
        if passes < 2 and not a.quick:
            ap.error("--passes must be at least 2: without a repeat there is no noise floor")
        pol = query_policy(a.binary)
        side = {"policy": pol, "machine": {"start": te.machine_state()}}
        json.dump(side, open(os.path.join(a.out, "policy.json"), "w"), indent=1)
        for prob in te.state_problems(side["machine"]["start"]):
            print(f"warning: {prob}; this sweep will be marked untrustworthy", file=sys.stderr)
        before = probe(a.binary, a.limit)
        raw = os.path.join(a.out, "raw.csv")
        sweep(a.binary, grid(a.quick, a.max_n), passes, a.limit, raw)
        side["drift"] = te.drift(before, probe(a.binary, a.limit))
        side["machine"]["end"] = te.machine_state()
        json.dump(side, open(os.path.join(a.out, "policy.json"), "w"), indent=1)
        best, repeats, subs = load(raw)
        if not best:
            sys.exit("no usable measurements: is the binary path right?")
        res = analyse(best, repeats, pol, side["machine"], side["drift"], single_pass=passes < 2)
        res["submissions"] = subs
    res = sub.portable(res)
    json.dump(res, open(os.path.join(a.out, "results.json"), "w"), indent=2)
    write_report(res, os.path.join(a.out, "report.md"))
    k, rt = res["kernel"], res["routing"]
    print(f"  kernel: simd <= {k['simd_max_n']}, blocked >= {k['blocked_min_n']} "
          f"(batch <= {k['blocked_max_batch'] or 'any'})   ({k['score']['geomean']:.4f}x of the best GPU kernel)")
    if rt:
        gm = "no limit" if rt["gpu_max_n"] >= NO_LIMIT else rt["gpu_max_n"]
        print(f"  GPU iff {rt['gpu_min_n']} <= N <= {gm}, batch*N >= {rt['gpu_min_batch_times_n']}, batch >= "
              f"{rt['gpu_min_batch']}" + (f", or N >= {rt['gpu_large_min_n']} in a batch of at most "
                                         f"{rt['gpu_large_max_batch'] or 'any'}" if rt["gpu_large_min_n"] else "") +
              f"   ({rt['chosen']['geomean']:.4f}x vs {rt['cpu_always']['geomean']:.4f}x always CPU, "
              f"worst {rt['chosen']['worst']:.2f}x)")
    print(f"  kTuned entry: {res['ktuned_entry']}" +
          ("" if res["trustworthy"] else "     <-- indicative only (see the warnings)"))
    for w in res["warnings"]:
        print("  warning:", w)
    print(f"  -> {a.out}/report.md, {a.out}/results.json")


if __name__ == "__main__":
    main()
