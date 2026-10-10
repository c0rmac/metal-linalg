#!/usr/bin/env python3
"""Measure LU's routing on this Mac -- lu_factor, solve and inv on the GPU or
the CPU, and how the GPU path solves -- and write up the result.

    cmake --build build --target sweep_lu
    python3 tuning/tune_lu.py build/sweep_lu

Produces, in --out (default lu-tune-results/): raw.csv (every timed run),
policy.json (the device, the policy in effect, the machine's state before and
after, the probe point's drift), results.json (the fit) and report.md.

    python3 tuning/tune_lu.py --reanalyse lu-tune-results/raw.csv --out DIR

THE GRID
--------
lu_factor and inv, on the CPU path and the GPU path (blocked), for N from 32
to 4096 at batches from 1 to 1024, every point whose batch * N^2 is at most
2^25 floats (the GPU path factors one matrix at a time, so it is timed only
where a batch takes it under a second: N >= 128, batches up to 64), and solve
for N from 1024 to 4096 with 1 to 256 right-hand sides: the CPU path, and the
GPU path's factorization solved by its triangular solves on the GPU or by
LAPACK's sgetrs on the same memory. Two passes in independently shuffled
orders, min of passes; a probe point before and after for drift.

THE FIT
-------
The GPU iff N >= gpu_min_n in a batch of at most gpu_max_batch (0: any),
scored on lu_factor's and inv's points together against the faster of the two
paths at each; then gpu_solve_min_rhs on the solve points. Each the candidate
with the smallest worst case within 0.5% of the best geometric-mean regret.
"""

import argparse
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

NO_LIMIT = 0xFFFFFFFF
INF = float("inf")
SIZES = (32, 64, 128, 256, 512, 768, 1024, 1536, 2048, 3072, 4096)
BATCHES = (1, 2, 4, 16, 64, 256, 1024)
QUICK_SIZES = (128, 1024, 2048)
QUICK_BATCHES = (1, 16)
SOLVE_SIZES = (1024, 2048, 4096)
SOLVE_RHS = (1, 4, 16, 64, 256)
MAX_ELEMENTS = 1 << 25
BLOCKED_MIN_N = 128
BLOCKED_MAX_BATCH = 64
PROBE = (1, 2048, 1, ["cpu", "blocked"])   # (batch, N, K, backends)
TOL = 0.005
PAIRS = {"lu": ("cpu", "blocked"), "inv": ("inv_cpu", "inv_blocked")}


# ---------------------------------------------------------------------------
# Measuring
# ---------------------------------------------------------------------------

def jobs(quick=False, max_n=4096):
    out = []
    for N in (QUICK_SIZES if quick else SIZES):
        if N > max_n:
            continue
        for b in (QUICK_BATCHES if quick else BATCHES):
            if b * N * N > MAX_ELEMENTS or not sub.fits_memory(3 * b * N * N):
                continue
            backends = ["cpu", "inv_cpu"]
            if N >= BLOCKED_MIN_N and b <= BLOCKED_MAX_BATCH:
                backends += ["blocked", "inv_blocked"]
            out.append((b, N, 1, backends))
    for N in SOLVE_SIZES:
        if N > max_n or (quick and N != 2048):
            continue
        for k in ((1, 64) if quick else SOLVE_RHS):
            out.append((1, N, k, ["solve_cpu", "solve_trsm", "solve_getrs"]))
    return out


def run_one(binary, job, limit, attempts=3):
    b, N, K, backends = job
    for _ in range(attempts):
        try:
            r = subprocess.run([binary, str(b), str(N), str(K), ",".join(backends)],
                               capture_output=True, text=True, timeout=limit)
            lines = [l for l in r.stdout.strip().splitlines() if l.count(",") == 8]
            if r.returncode == 0 and len(lines) == len(backends):
                return lines
        except subprocess.TimeoutExpired:
            break
    return [f"{b},{N},{K},{k},0,0,0,0,0" for k in backends]


def sweep(binary, todo, passes, limit, out_csv):
    total = len(todo) * passes
    print(f"  {len(todo)} points x {passes} passes = {total} processes", file=sys.stderr)
    done, t0 = 0, time.time()
    with open(out_csv, "w") as fh:
        fh.write("pass,batch,N,K,backend,ok,ms,p25,p75,reps\n")
        for p in range(passes):
            order = list(todo)
            random.Random(8300 + p).shuffle(order)
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
    out = {}
    for _ in range(3):
        for line in run_one(binary, PROBE, limit):
            f = line.split(",")
            if f[4] == "1" and float(f[5]) > 0:
                out[f[3]] = min(out.get(f[3], INF), float(f[5]))
    return out


def load(paths):
    return sub.combine(paths, lambda r: (int(r["batch"]), int(r["N"]), int(r["K"])))


def query_policy(binary):
    r = subprocess.run([binary, "--policy"], capture_output=True, text=True, timeout=60)
    if r.returncode != 0 or not r.stdout.strip():
        sys.exit(f"{binary} --policy failed; rebuild sweep_lu")
    return json.loads(next(l for l in reversed(r.stdout.strip().splitlines()) if l.startswith("{")))


def load_sidecar(binary, raw_paths):
    if raw_paths:
        path = os.path.join(os.path.dirname(os.path.abspath(raw_paths[0])), "policy.json")
        if os.path.exists(path):
            side = json.load(open(path))
            return side if "policy" in side else {"policy": side}
    if binary:
        return {"policy": query_policy(binary)}
    return {"policy": {"device": "unknown", "gpu_cores": 0, "source": "unknown"}}


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def stat(reg, pts):
    if not reg:
        return None
    worst = max(range(len(reg)), key=lambda i: reg[i])
    return {"geomean": round(math.exp(math.fsum(map(math.log, reg)) / len(reg)), 4), "worst": round(reg[worst], 3),
            "worst_point": list(pts[worst]), "over_10pct": sum(r > 1.10 for r in reg), "n": len(reg)}


def routing_regrets(best, mn, cap):
    """Regret of GPU iff N >= mn and (cap == 0 or batch <= cap), on every
    lu_factor and inv point timed on both paths."""
    reg, pts = [], []
    for (b, N, K), t in sorted(best.items()):
        if K != 1:
            continue
        for op, (cpu, gpu) in PAIRS.items():
            if cpu not in t or gpu not in t:
                continue
            on_gpu = bool(mn) and N >= mn and (not cap or b <= cap)
            reg.append(t[gpu if on_gpu else cpu] / min(t[cpu], t[gpu]))
            pts.append((op, b, N))
    return reg, pts


def fit_routing(best):
    sizes = sorted({N for (b, N, K) in best if K == 1})
    batches = sorted({b for (b, N, K) in best if K == 1})
    cands = [(0, 0)] + [(mn, cap) for mn in sizes[:-1] + [sizes[-1]] for cap in [0] + batches]
    scored = {}
    for c in cands:
        reg, pts = routing_regrets(best, *c)
        if reg:
            scored[c] = stat(reg, pts)
    g = min(s["geomean"] for s in scored.values())
    near = {c: s for c, s in scored.items() if s["geomean"] <= g * (1 + TOL)}
    # ties: the smallest worst case; then the GPU from the largest size, any batch or the largest cap
    chosen = min(near, key=lambda c: (near[c]["worst"], near[c]["geomean"], -c[0] if c[0] else 0,
                                      -(c[1] if c[1] else INF)))
    keys = sorted(best)
    half = {k: best[k] for i, k in enumerate(keys) if i % 2 == 0}
    other = {k: best[k] for i, k in enumerate(keys) if i % 2 == 1}
    hs = {}
    for c in cands:
        reg, pts = routing_regrets(half, *c)
        if reg:
            hs[c] = stat(reg, pts)
    hg = min(s["geomean"] for s in hs.values())
    hnear = {c: s for c, s in hs.items() if s["geomean"] <= hg * (1 + TOL)}
    hc = min(hnear, key=lambda c: (hnear[c]["worst"], hnear[c]["geomean"], -c[0] if c[0] else 0,
                                   -(c[1] if c[1] else INF)))
    held = stat(*routing_regrets(other, *hc))
    return {"gpu_min_n": chosen[0], "gpu_max_batch": chosen[1], "chosen": scored[chosen],
            "cpu_always": scored[(0, 0)], "gpu_always": stat(*routing_regrets(best, 1, 0)),
            "held_out": {"fitted": list(hc), "routing": held, "cpu_always": stat(*routing_regrets(other, 0, 0))}}


def fit_solve(best):
    """gpu_solve_min_rhs: the GPU's triangular solves from this many
    right-hand sides, sgetrs below, on the solve points."""
    pts = {k: t for k, t in best.items() if "solve_trsm" in t and "solve_getrs" in t}
    if not pts:
        return {"gpu_solve_min_rhs": 16, "chosen": None}
    ks = sorted({K for (_, _, K) in pts})
    scored = {}
    for th in ks + [NO_LIMIT]:
        reg, where = [], []
        for (b, N, K), t in sorted(pts.items()):
            reg.append(t["solve_trsm" if K >= th else "solve_getrs"] / min(t["solve_trsm"], t["solve_getrs"]))
            where.append((b, N, K))
        scored[th] = stat(reg, where)
    g = min(s["geomean"] for s in scored.values())
    near = {c: s for c, s in scored.items() if s["geomean"] <= g * (1 + TOL)}
    th = min(near, key=lambda c: (near[c]["worst"], near[c]["geomean"], -c))   # ties: sgetrs the longest
    return {"gpu_solve_min_rhs": th, "chosen": scored[th], "always_trsm": scored[ks[0]],
            "always_getrs": scored[NO_LIMIT]}


def noise_floor(repeats):
    allr = [max(v) / min(v) for v in repeats.values() if len(v) >= 2 and min(v) > 0]
    allr.sort()
    q = lambda p: allr[min(len(allr) - 1, int(p * (len(allr) - 1)))] if allr else 0.0
    return {"overall": {"n": len(allr), "median": q(0.5), "p90": q(0.9), "max": allr[-1] if allr else 0}}


def analyse(best, repeats, device, states=None, drift_info=None, single_pass=False):
    routing = fit_routing(best)
    solve = fit_solve(best)
    noise = noise_floor(repeats)
    warns = []
    if single_pass:
        warns.append("single pass: a smoke test of the pipeline, not a measurement; run without --quick")
    busy = [f"{p} at the {when} of the sweep" for when, st in (states or {}).items() for p in te.state_problems(st)]
    if busy:
        warns.insert(0, "the machine was not idle: " + "; ".join(busy) + "; rerun when idle")
    if any(st.get("power") == "battery" for st in (states or {}).values()):
        warns.append("run on battery power; macOS may limit performance differently than on mains")
    if drift_info and not drift_info["ok"]:
        warns.insert(0, "the machine changed state during the sweep: the probe point moved by " +
                     ", ".join(f"{k} x{v:.2f}" for k, v in sorted(drift_info["ratio"].items())))
    if not single_pass and noise["overall"]["median"] > 1.10:
        warns.append(f"run-to-run noise {noise['overall']['median']:.2f}x (median) is above 10%")
    trustworthy = not (single_pass or busy or (drift_info and not drift_info["ok"])
                       or noise["overall"]["median"] > 1.10)
    th = solve["gpu_solve_min_rhs"]
    entry = (f'{{"{device.get("device", "unknown")}", {device.get("gpu_cores", 0)},   '
             f'{routing["gpu_min_n"]}, {routing["gpu_max_batch"]},   '
             f'{"kLuNoLimit" if th >= NO_LIMIT else th}}},')
    points = [{"batch": b, "N": N, "K": K, "ms": {k: round(v, 4) for k, v in sorted(t.items())}}
              for (b, N, K), t in sorted(best.items(), key=lambda x: (x[0][2], x[0][1], x[0][0]))]
    return {"device": {"name": device.get("device"), "gpu_cores": device.get("gpu_cores"),
                       "cpu_threads": device.get("cpu_threads"), "source": device.get("source")},
            "routing": routing, "solve": solve, "points": points, "noise": noise, "drift": drift_info,
            "states": states, "trustworthy": bool(trustworthy), "warnings": warns, "ktuned_entry": entry}


def write_report(res, path):
    d, rt, sv = res["device"], res["routing"], res["solve"]
    L = [f"# LU routing: {d['name']}, {d['gpu_cores']} GPU cores", ""]
    if res.get("submissions"):
        L += [f"From {len(res['submissions'])} run(s): {', '.join(res['submissions'])}.", ""]
    L += [f"> **Warning:** {w}" for w in res["warnings"]] + ([""] if res["warnings"] else [])
    L += ["## Answer", "", "Row for `kTuned[]` (src/tuned/lu.inc, generated):" if res["trustworthy"]
          else "**Indicative only; do not use this row.** See the warnings above.", "",
          "```cpp", "// device, GPU cores,   gpu_min_n, gpu_max_batch,   gpu_solve_min_rhs", res["ktuned_entry"], "```", "",
          "## GPU or CPU", "",
          (f"lu_factor, solve and inv on the GPU from N = {rt['gpu_min_n']}"
           + (f" in batches of at most {rt['gpu_max_batch']}" if rt["gpu_max_batch"] else "") if rt["gpu_min_n"]
           else "Always the CPU") + ". Against the faster path at every lu_factor and inv point:", "",
          "| rule | geomean | worst | points over 10% |", "|---|---|---|---|"]
    for name, s in (("fitted", rt["chosen"]), ("CPU always", rt["cpu_always"]), ("GPU always", rt["gpu_always"]),
                    ("held out: fitted on half", rt["held_out"]["routing"]),
                    ("held out: CPU always", rt["held_out"]["cpu_always"])):
        if s:
            L.append(f"| {name} | {s['geomean']:.4f}x | {s['worst']:.2f}x at {s['worst_point']} | "
                     f"{s['over_10pct']} of {s['n']} |")
    L.append("")
    if sv.get("chosen"):
        L += ["## The GPU path's solve", "",
              f"Its triangular solves on the GPU from {sv['gpu_solve_min_rhs']} right-hand sides, LAPACK's sgetrs "
              f"below: {sv['chosen']['geomean']:.4f}x (always the GPU's {sv['always_trsm']['geomean']:.4f}x, always "
              f"sgetrs {sv['always_getrs']['geomean']:.4f}x).", ""]
    n = res["noise"]["overall"]
    L += ["## Noise", "", f"Pass-to-pass ratio: median {n['median']:.3f}, p90 {n['p90']:.3f}, max {n['max']:.2f} "
          f"over {n['n']} timings.", ""]
    if res.get("drift"):
        L += ["Probe point after the sweep over before: " +
              ", ".join(f"{b} {v:.3f}x" for b, v in sorted(res["drift"]["ratio"].items())) + ".", ""]
    L += ["## Every point", "", "Median ms, min over passes.", "",
          "| N | batch | K | cpu | blocked | inv_cpu | inv_blocked | solve_cpu | solve_trsm | solve_getrs |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for p in res["points"]:
        ms = p["ms"]
        cells = [f"{ms[k]:.3f}" if k in ms else "—"
                 for k in ("cpu", "blocked", "inv_cpu", "inv_blocked", "solve_cpu", "solve_trsm", "solve_getrs")]
        L.append(f"| {p['N']} | {p['batch']} | {p['K']} | " + " | ".join(cells) + " |")
    open(path, "w").write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("binary", nargs="?", help="path to the built sweep_lu")
    ap.add_argument("--out", default="lu-tune-results")
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--limit", type=int, default=300, help="per-point timeout, seconds")
    ap.add_argument("--max-n", type=int, default=4096)
    ap.add_argument("--quick", action="store_true", help="a few points, one pass: a smoke test")
    ap.add_argument("--reanalyse", metavar="RAW.CSV", nargs="+")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.reanalyse:
        best, repeats, subs = load(a.reanalyse)
        if not best:
            sys.exit(f"no usable measurements in {a.reanalyse}")
        side = load_sidecar(a.binary, a.reanalyse)
        res = analyse(best, repeats, side["policy"], side.get("machine"), side.get("drift"))
    else:
        if not a.binary:
            ap.error("need the path to sweep_lu (or --reanalyse)")
        passes = 1 if a.quick else a.passes
        if passes < 2 and not a.quick:
            ap.error("--passes must be at least 2")
        pol = query_policy(a.binary)
        side = {"policy": pol, "machine": {"start": te.machine_state()}}
        json.dump(side, open(os.path.join(a.out, "policy.json"), "w"), indent=1)
        before = probe(a.binary, a.limit)
        raw = os.path.join(a.out, "raw.csv")
        sweep(a.binary, jobs(a.quick, a.max_n), passes, a.limit, raw)
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
    rt, sv = res["routing"], res["solve"]
    print(f"  GPU from N = {rt['gpu_min_n']} (batch <= {rt['gpu_max_batch'] or 'any'}): {rt['chosen']['geomean']:.4f}x "
          f"vs {rt['cpu_always']['geomean']:.4f}x always CPU, worst {rt['chosen']['worst']:.2f}x; "
          f"the GPU's triangular solves from {sv['gpu_solve_min_rhs']} right-hand sides")
    print(f"  kTuned entry: {res['ktuned_entry']}" + ("" if res["trustworthy"] else "     <-- indicative only"))
    for w in res["warnings"]:
        print("  warning:", w)
    print(f"  -> {a.out}/report.md, {a.out}/results.json")


if __name__ == "__main__":
    main()
