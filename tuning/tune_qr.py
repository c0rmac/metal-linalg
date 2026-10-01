#!/usr/bin/env python3
"""Measure this GPU's dispatch crossover band and write up the result.

    cmake --build build --target sweep_qr
    python3 tuning/tune_qr.py build/sweep_qr

Produces, in --out (default qr-tune-results/):

    raw.csv        every timed run, so the analysis can be redone without remeasuring
    results.json   the full analysis: plot-ready series, decision surface, rule
                   comparison, noise floor, band candidate, device identity
    report.md      a written report generated from results.json

Re-render the report from an existing run without touching the GPU:

    python3 tuning/tune_qr.py --from qr-tune-results/results.json

WHY THE GRID LOOKS LIKE THIS
----------------------------
The shape list is deliberately balanced across square, tall, wide and
near-square inputs, in equal-ish measure. This is not fussiness. On an M1 a
square-heavy grid put the crossover at 512 with 1.003x geometric-mean regret --
apparently near-optimal -- and that threshold lost up to 1.95x on tall inputs.
No amount of held-out validation caught it, because the sample never probed the
region where the rule failed. A grid that cannot see a failure mode will report
excellent numbers right up until someone hits it.

MEASUREMENT NOTES
-----------------
* Order is randomised. Under a size-ordered sweep, thermal/DVFS drift over the
  run correlates perfectly with problem size, and downclocking looks exactly
  like "big matrices are slow" -- the thing being measured.
* Timing is the median of repeated calls, not the mean: GPU timings are
  right-skewed and a mean chases outliers.
* Two independent passes give a run-to-run noise floor. Without it there is no
  basis for calling a difference real, and the fit will chase noise.
* Passes are combined with min-of-repeats, since interference is one-sided.
* The reported band is the whole flat region, not the argmin. The argmin of a
  flat curve is noise; the band is the finding.
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

BACKENDS = ("unblocked", "reduced")
REGIONS = ("square", "tall", "wide", "near-square")

# Candidate thresholds. A threshold only changes behaviour when it crosses a
# measured M, so values between two measured M's are equivalent by construction.
THRESHOLDS = [128, 192, 256, 288, 320, 352, 384, 416, 448, 480, 512, 576, 640, 768, 1024]

# The decision-surface panels are a dense cross product, measured explicitly so
# the heatmap has no holes. This is the picture that shows *why* the crossover
# sits where it does, so it is worth the extra shapes.
SURFACE_BATCHES = (1, 16)
SURFACE_M = (128, 256, 384, 512, 640)
SURFACE_N = (32, 64, 128, 256, 512)

MEM_CAP_ELEMS = 40_000_000        # batch*M*N
MEM_CAP_SQUARE = 100_000_000      # batch*max(M,N)^2, bounds the M x M workspace


# ---------------------------------------------------------------------------
# Shape grid
# ---------------------------------------------------------------------------

def region(M, N):
    if M == N:
        return "square"
    if M > N * 2:
        return "tall"
    if N > M * 2:
        return "wide"
    return "near-square"


def shape_grid(full=False):
    square_dims = [64, 128, 192, 256, 320, 384, 448, 512, 640, 768]
    tall = [(256, 16), (256, 64), (384, 32), (384, 64), (512, 32),
            (512, 128), (640, 64), (1024, 64), (1024, 256), (2048, 64)]
    near = [(384, 256), (448, 256), (512, 384), (256, 384), (320, 448)]
    batches_sq = [1, 4, 16, 64]
    batches_rect = [1, 4, 16]
    if full:
        square_dims += [96, 576, 1024]
        tall += [(320, 32), (448, 32), (448, 64), (768, 128)]
        near += [(640, 448), (384, 512)]
        batches_sq = [1, 2, 4, 16, 64, 128]
        batches_rect = [1, 4, 16, 64]

    pts = []
    pts += [(b, d, d) for d in square_dims for b in batches_sq]
    pts += [(b, M, N) for M in SURFACE_M for N in SURFACE_N for b in SURFACE_BATCHES]
    for (M, N) in tall:
        pts += [(b, M, N) for b in batches_rect]
        pts += [(b, N, M) for b in batches_rect]      # the wide transpose
    for (M, N) in near:
        pts += [(b, M, N) for b in batches_rect]

    out = []
    for (b, M, N) in sorted(set(pts)):
        if b * M * N > MEM_CAP_ELEMS:
            continue
        if b * max(M, N) ** 2 > MEM_CAP_SQUARE:
            continue
        out.append((b, M, N))
    return out


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

def run_one(binary, job, limit, attempts=3):
    """One (shape, backend). Retries on crash so a transient failure does not
    silently become a missing data point."""
    b, M, N, backend = job
    for _ in range(attempts):
        try:
            r = subprocess.run([binary, str(b), str(M), str(N), backend],
                               capture_output=True, text=True, timeout=limit)
            line = r.stdout.strip()
            if r.returncode == 0 and line:
                return line
        except subprocess.TimeoutExpired:
            break
    return f"{b},{M},{N},{backend},0,0,0,0,0"


def sweep(binary, pts, passes, limit, out_csv):
    jobs = [(b, M, N, k) for (b, M, N) in pts for k in BACKENDS]
    total = len(jobs) * passes
    print(f"  {len(pts)} shapes x {len(BACKENDS)} backends x {passes} passes "
          f"= {total} timed runs", file=sys.stderr)
    done = 0
    t0 = time.time()
    with open(out_csv, "w") as fh:
        fh.write("pass,batch,M,N,backend,ok,ms,p25,p75,reps\n")
        for p in range(passes):
            order = list(jobs)
            random.Random(9000 + p).shuffle(order)   # independent shuffle per pass
            for job in order:
                fh.write(f"{p},{run_one(binary, job, limit)}\n")
                fh.flush()
                done += 1
                if done % 25 == 0 or done == total:
                    el = time.time() - t0
                    eta = el / done * (total - done)
                    print(f"    {done}/{total}  elapsed {el/60:.1f}m  eta {eta/60:.1f}m",
                          file=sys.stderr)


def load(paths):
    """-> (best[(b,M,N)][backend], repeats, submissions); see submissions.combine.

    Several raw.csv files from one submission merge by min-of-passes, so a
    sweep can be topped up with extra shapes rather than remeasured."""
    best, repeats, subs = sub.combine(paths, lambda r: (int(r["batch"]), int(r["M"]), int(r["N"])))
    best = {k: v for k, v in best.items() if all(x in v for x in BACKENDS)}
    return best, repeats, subs

# ---------------------------------------------------------------------------
# Device identity
# ---------------------------------------------------------------------------

def query_policy(binary):
    """`sweep_qr --policy`: the device and the policy in effect, as the library
    detects them. This is what the sweep writes to policy.json."""
    try:
        r = subprocess.run([binary, "--policy"], capture_output=True, text=True, timeout=60)
        return json.loads(r.stdout.strip().splitlines()[-1])
    except Exception:
        return None


def _from_policy(pol):
    return {"name": pol["device"], "gpu_cores": pol["gpu_cores"],
            "concurrent_matrices": pol.get("concurrent_matrices", 0),
            "source": pol.get("source", "unknown"), "threadgroup_memory": 32768}


def device_info(binary=None, raw_paths=None):
    """The device and the policy in effect: from the binary when there is one,
    else from the policy.json the sweep left beside its raw.csv, else from
    system_profiler."""
    pol = query_policy(binary) if binary else None
    if pol is None and raw_paths:
        side = os.path.join(os.path.dirname(os.path.abspath(raw_paths[0])), "policy.json")
        if os.path.exists(side):
            pol = json.load(open(side))
    if pol is not None:
        return _from_policy(pol)
    info = {"name": "unknown", "gpu_cores": 0, "threadgroup_memory": 0,
            "concurrent_matrices": 0}
    try:
        out = subprocess.run(["system_profiler", "SPDisplaysDataType"],
                             capture_output=True, text=True, timeout=40).stdout
        for line in out.splitlines():
            if "Chipset Model" in line:
                info["name"] = line.split(":", 1)[1].strip()
            elif "Total Number of Cores" in line and not info["gpu_cores"]:
                m = re.search(r"\d+", line)
                if m:
                    info["gpu_cores"] = int(m.group())
    except Exception:
        pass
    # qr_unblocked requests 5120 B of threadgroup memory; with the device's
    # per-threadgroup limit that fixes how many matrices stay resident, which is
    # what decides whether batch count can matter at all.
    info["threadgroup_memory"] = 32768
    if info["gpu_cores"]:
        info["concurrent_matrices"] = (info["threadgroup_memory"] // 5120) * info["gpu_cores"]
    return info


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def evaluate(rule, pts, tie=0.10):
    reg, worst, worst_pt, bad = [], 1.0, None, 0
    th = to = 0.0
    for (b, M, N), t in pts.items():
        best = min(t.values())
        got = t[rule(b, M, N)]
        r = got / best
        reg.append(r)
        th += got
        to += best
        if r > worst:
            worst, worst_pt = r, [b, M, N]
        if r > 1 + tie:
            bad += 1
    if not reg:
        return None
    return {"geomean": math.exp(math.fsum(map(math.log, reg)) / len(reg)),
            "worst": worst, "worst_shape": worst_pt,
            "over_tie": bad, "n": len(reg), "total": th / to}


def flat_rule(t):
    return lambda b, M, N: "reduced" if M >= t else "unblocked"


def two_regime(lo, hi, sat):
    return lambda b, M, N: "reduced" if M >= (hi if b < sat else lo) else "unblocked"


def narrow_n(t1, tn, t2):
    return lambda b, M, N: "reduced" if (M >= t1 or (N <= tn and M >= t2)) else "unblocked"


def noise_floor(repeats):
    """Pass-to-pass ratio, bucketed by runtime. Fast shapes are dominated by
    submission jitter and cannot settle a small difference."""
    buckets = defaultdict(list)
    allr = []
    for (b, M, N, k, *_), v in repeats.items():
        if len(v) < 2 or min(v) <= 0:
            continue
        ratio = max(v) / min(v)
        allr.append(ratio)
        t = min(v)
        lab = ("<1 ms" if t < 1 else "1-3 ms" if t < 3 else "3-10 ms"
               if t < 10 else "10-30 ms" if t < 30 else "30-100 ms" if t < 100 else ">100 ms")
        buckets[lab].append(ratio)

    def q(a, p):
        a = sorted(a)
        return a[min(len(a) - 1, int(p * (len(a) - 1)))] if a else 0.0

    order = ["<1 ms", "1-3 ms", "3-10 ms", "10-30 ms", "30-100 ms", ">100 ms"]
    return {
        "overall": {"n": len(allr), "median": q(allr, .5), "p90": q(allr, .9),
                    "max": max(allr) if allr else 0},
        "by_runtime": [{"bucket": lab, "n": len(buckets[lab]),
                        "median": q(buckets[lab], .5), "p90": q(buckets[lab], .9),
                        "max": max(buckets[lab])}
                       for lab in order if buckets.get(lab)],
    }


def analyse(best, repeats, device):
    by_region = {r: {k: v for k, v in best.items() if region(k[1], k[2]) == r}
                 for r in REGIONS}

    curves = {"thresholds": THRESHOLDS, "pooled": [], "pooled_worst": []}
    for r in REGIONS:
        curves[r] = []
    for t in THRESHOLDS:
        e = evaluate(flat_rule(t), best)
        curves["pooled"].append(round(e["geomean"], 4))
        curves["pooled_worst"].append(round(e["worst"], 3))
        for r in REGIONS:
            er = evaluate(flat_rule(t), by_region[r]) if by_region[r] else None
            curves[r].append(round(er["geomean"], 4) if er else None)

    # The band: every threshold whose regret is within `tol` of the best. The
    # argmin of a flat curve is noise; the band is the finding. Report the
    # middle of the band as the value to ship.
    tol = 0.003
    gbest = min(curves["pooled"])
    band = [t for t, g in zip(THRESHOLDS, curves["pooled"]) if g <= gbest + tol]
    chosen = band[len(band) // 2] if band else THRESHOLDS[len(THRESHOLDS) // 2]

    cost = [{"threshold": t,
             "geomean": g,
             "excess_pct": round((g / gbest - 1) * 100, 2),
             "in_band": t in band}
            for t, g in zip(THRESHOLDS, curves["pooled"])]

    # Decision surface
    surface = {}
    for b in SURFACE_BATCHES:
        cells = [{"M": M, "N": N, "ratio": round(t["reduced"] / t["unblocked"], 3)}
                 for (bb, M, N), t in sorted(best.items())
                 if bb == b and M in SURFACE_M and N in SURFACE_N]
        if cells:
            surface[str(b)] = cells

    # Candidate rules, scored overall and per region
    def orig(b, M, N):
        mx = max(M, N)
        if mx >= 512:
            return "reduced"
        if b >= 16:
            return "unblocked"
        if mx >= 128:
            return "reduced"
        return "unblocked"

    rules = {
        "original 5-rule heuristic": orig,
        "max(M,N) >= 512": lambda b, M, N: "reduced" if max(M, N) >= 512 else "unblocked",
        "K = min(M,N) >= 128": lambda b, M, N: "reduced" if min(M, N) >= 128 else "unblocked",
        f"M >= {chosen}  (chosen)": flat_rule(chosen),
    }
    rule_rows = []
    for lab, h in rules.items():
        e = evaluate(h, best)
        e["label"] = lab
        e["per_region"] = {r: (round(evaluate(h, by_region[r])["geomean"], 4)
                               if by_region[r] else None) for r in REGIONS}
        e["geomean"] = round(e["geomean"], 4)
        e["worst"] = round(e["worst"], 3)
        e["total"] = round(e["total"], 4)
        rule_rows.append(e)

    # Which feature? If M stops being the best feature on some GPU, that is a
    # structural change and matters far more than the threshold moving.
    feats = {"M (rows)": lambda b, M, N: M,
             "max(M, N)": lambda b, M, N: max(M, N),
             "K = min(M, N)": lambda b, M, N: min(M, N)}
    feature_rows = []
    for name, key in feats.items():
        scored = [(evaluate(lambda b, M, N, t=t, key=key:
                            "reduced" if key(b, M, N) >= t else "unblocked", best), t)
                  for t in THRESHOLDS]
        e, t = min(scored, key=lambda x: x[0]["geomean"])
        feature_rows.append({"feature": name, "best_threshold": t,
                             "geomean": round(e["geomean"], 4),
                             "worst": round(e["worst"], 3)})
    feature_rows.sort(key=lambda r: r["geomean"])

    # Refinements that failed on an M1. Re-tested here rather than assumed: a
    # device with many more cores saturates qr_unblocked much later, so the
    # batch split in particular could be justified elsewhere. Fitted on half the
    # points, scored on the other half, so a win has to survive held-out data.
    keys = sorted(best)
    tr = {k: best[k] for i, k in enumerate(keys) if i % 2 == 0}
    te = {k: best[k] for i, k in enumerate(keys) if i % 2 == 1}
    base_te = evaluate(flat_rule(chosen), te)

    def fit(factory, grids):
        bf = None
        for combo in _product(grids):
            e = evaluate(factory(*combo), tr)
            if bf is None or e["geomean"] < bf[0]["geomean"]:
                bf = (e, combo)
        return bf

    refinements = []
    e2, c2 = fit(two_regime, [[256, 288, 320, 352, 384, 416],
                              [352, 384, 416, 448, 480, 512, 576],
                              [2, 4, 8, 16, 32]])
    te2 = evaluate(two_regime(*c2), te)
    refinements.append({
        "name": "batch-dependent split",
        "form": f"M >= ({c2[1]} if batch < {c2[2]} else {c2[0]})",
        "train_geomean": round(e2["geomean"], 4),
        "test_geomean": round(te2["geomean"], 4),
        "test_worst": round(te2["worst"], 3),
        "justified": bool(te2["geomean"] < base_te["geomean"] - 0.002
                          and te2["worst"] <= base_te["worst"] + 0.01),
    })
    e3, c3 = fit(narrow_n, [[352, 384, 416, 448],
                            [16, 32, 64, 96],
                            [128, 192, 224, 256, 288, 320]])
    te3 = evaluate(narrow_n(*c3), te)
    refinements.append({
        "name": "narrow-N special case",
        "form": f"M >= {c3[0]} or (N <= {c3[1]} and M >= {c3[2]})",
        "train_geomean": round(e3["geomean"], 4),
        "test_geomean": round(te3["geomean"], 4),
        "test_worst": round(te3["worst"], 3),
        "justified": bool(te3["geomean"] < base_te["geomean"] - 0.002
                          and te3["worst"] <= base_te["worst"] + 0.01),
    })

    counts = defaultdict(int)
    for k in best:
        counts[region(k[1], k[2])] += 1

    return {
        "device": device,
        "n_points": len(best),
        "coverage": dict(counts),
        "noise": noise_floor(repeats),
        "threshold_curves": curves,
        "band": {"lo": min(band), "hi": max(band), "chosen": chosen,
                 "best_geomean": round(gbest, 4), "tolerance": tol},
        "cost_of_missing": cost,
        "surface": surface,
        "rules": rule_rows,
        "features": feature_rows,
        "refinements": refinements,
        "baseline_held_out": {"geomean": round(base_te["geomean"], 4),
                              "worst": round(base_te["worst"], 3)},
        "ktuned_entry": (f'{{"{device["name"]}", {device["gpu_cores"]}, '
                         f'{chosen}, {chosen}, 16}},'),
    }


def _product(grids):
    if not grids:
        yield ()
        return
    for head in grids[0]:
        for rest in _product(grids[1:]):
            yield (head,) + rest


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _mermaid_curve(curves):
    th = curves["thresholds"]
    series = [("pooled", curves["pooled"])]
    for r in ("tall", "square"):
        if any(v is not None for v in curves.get(r, [])):
            series.append((r + " only", curves[r]))
    vals = [v for _, s in series for v in s if v is not None]
    ymax = math.ceil(max(vals) * 100) / 100 + 0.01
    lines = ["```mermaid", "xychart-beta",
             '    title "Regret by threshold (lower is better)"',
             "    x-axis [" + ", ".join(str(t) for t in th) + "]",
             f'    y-axis "geometric-mean regret" 1.0 --> {ymax:.2f}']
    for name, s in series:
        pretty = ", ".join(f"{v:.4f}" if v is not None else "1.0" for v in s)
        lines.append(f"    line [{pretty}]")
    lines.append("```")
    lines.append("")
    lines.append("Series order: " + ", ".join(n for n, _ in series) + ".")
    return "\n".join(lines)


def _ascii_surface(cells, chosen):
    """Dense M x N panel. Each cell carries the ratio and a glyph, so the shape
    of the surface reads at a glance and the numbers are still there."""
    Ms = sorted({c["M"] for c in cells})
    Ns = sorted({c["N"] for c in cells})
    g = {(c["M"], c["N"]): c["ratio"] for c in cells}

    def glyph(v):
        if v is None:
            return "  . "
        if v < 0.60:
            return "###"      # reduced wins big
        if v < 0.85:
            return "## "
        if v < 0.95:
            return "#  "
        if v <= 1.05:
            return "~  "      # tie
        if v <= 1.30:
            return ".  "
        return "   "          # unblocked wins big

    first = min((m for m in Ms if m >= chosen), default=None)
    out = ["      " + "".join(f"N={n:<8}" for n in Ns)]
    out.append("      " + "-" * (9 * len(Ns)))
    for M in Ms:
        row = f"{M:>5} "
        for N in Ns:
            v = g.get((M, N))
            row += (f"{glyph(v)}{v:<5.2f}" if v is not None else f"{glyph(v)}{'':<5}") + " "
        out.append(row.rstrip() + ("   <- M >= %d" % chosen if M == first else ""))
    out.append("")
    out.append("      ratio = reduced / unblocked.  ### <0.60  ## <0.85  # <0.95")
    out.append("      ~ tie (0.95-1.05)   . <1.30   blank >1.30  (unblocked wins)")
    return "\n".join(out)


def write_report(res, path):
    d = res["device"]
    band, curves = res["band"], res["threshold_curves"]
    L = []
    A = L.append

    A(f"# QR dispatch crossover — {d['name']}")
    A("")
    A("What every section and number below means: [reading-reports.md](https://github.com/c0rmac/metal-linalg/blob/main/docs/reading-reports.md).")
    A("")
    A(f"Generated by `tuning/tune_qr.py`. "
      f"{res['n_points']} shapes, {res['noise']['overall']['n']} shape/backend pairs "
      f"measured twice.")
    A("")
    if len(res.get("submissions") or []) > 1:
        A(f"Combined from {len(res['submissions'])} submissions ({', '.join(res['submissions'])}): "
          f"each submission's fastest pass, then the median across submissions. The noise "
          f"floor is per submission.")
        A("")
    A("| | |")
    A("|---|---|")
    A(f"| Device | {d['name']} |")
    A(f"| GPU cores | {d['gpu_cores'] or 'undetected'} |")
    A(f"| Resident matrices (`qr_unblocked`) | {d['concurrent_matrices'] or 'unknown'} |")
    A(f"| Shapes measured | {res['n_points']} |")
    A(f"| Coverage | " + ", ".join(f"{k} {v}" for k, v in sorted(res['coverage'].items())) + " |")
    A("")

    A("## The band")
    A("")
    A(f"```\nM >= {band['chosen']}  ->  qr_streaming_amx_reduced\notherwise  ->  qr_unblocked\n```")
    A("")
    A(f"The optimum is flat from **{band['lo']} to {band['hi']}** rows "
      f"(every threshold within {band['tolerance']*100:.1f}% of the best, "
      f"{band['best_geomean']:.4f}x). Any value inside that band is equivalent on "
      f"this hardware; **{band['chosen']}** is the middle of it.")
    A("")
    A("Paste into `kTuned[]` in `src/qr.mm`:")
    A("")
    A(f"```c\n    {res['ktuned_entry']}\n```")
    A("")
    A("### Cost of missing the band")
    A("")
    A("| threshold | regret | excess | |")
    A("|---|---|---|---|")
    for c in res["cost_of_missing"]:
        mark = "**in band**" if c["in_band"] else ""
        A(f"| {c['threshold']} | {c['geomean']:.4f}x | +{c['excess_pct']:.2f}% | {mark} |")
    A("")
    A("The penalty is usually asymmetric. Erring low costs little; erring high "
      "degrades specifically on tall inputs. Adding GPU cores makes the "
      "grid-parallel backend relatively stronger and pushes the true crossover "
      "down, so an untuned device is safer low than high.")
    A("")

    A("## Regret by threshold")
    A("")
    A("Regret is how much slower the chosen backend is than the best one measured "
      "at that shape; 1.0000x would be a perfect oracle. The per-region curves "
      "are the point: a grid covering only one aspect ratio can be nearly flat, "
      "and will then pick a threshold out of noise.")
    A("")
    A(_mermaid_curve(curves))
    A("")
    A("| threshold | pooled | " + " | ".join(REGIONS) + " | pooled worst |")
    A("|---|---|" + "---|" * (len(REGIONS) + 1))
    for i, t in enumerate(curves["thresholds"]):
        cells = " | ".join(f"{curves[r][i]:.4f}" if curves[r][i] is not None else "—"
                           for r in REGIONS)
        A(f"| {t} | {curves['pooled'][i]:.4f} | {cells} | {curves['pooled_worst'][i]:.2f}x |")
    A("")

    A("## Which feature decides")
    A("")
    A("`qr_unblocked` gives each matrix a single threadgroup, which sweeps `M` "
      "rows per Householder reflection while `N` parallelises across that "
      "threadgroup's threads. Rows and columns are therefore not "
      "interchangeable, and a rule on `max(M, N)` cannot express the difference.")
    A("")
    A("| feature | best threshold | geomean regret | worst |")
    A("|---|---|---|---|")
    for f in res["features"]:
        A(f"| {f['feature']} | {f['best_threshold']} | {f['geomean']:.4f}x | {f['worst']:.2f}x |")
    A("")
    if res["features"][0]["feature"] != "M (rows)":
        A(f"> **The best feature here is {res['features'][0]['feature']}, not M.** "
          "The dispatcher keys on `M`. If this reproduces, `qr_accelerated` needs "
          "revisiting for this GPU — a change of feature is structural and matters "
          "more than the threshold moving.")
        A("")

    A("## Decision surface")
    A("")
    for b, cells in res["surface"].items():
        A(f"**batch {b}**")
        A("")
        A("```")
        A(_ascii_surface(cells, band["chosen"]))
        A("```")
        A("")

    A("## Candidate rules")
    A("")
    A("Per-region columns matter more than the overall number: an aggregate can "
      "look excellent while one aspect class is badly served.")
    A("")
    A("| rule | geomean | worst | >10% off | " + " | ".join(REGIONS) + " |")
    A("|---|---|---|---|" + "---|" * len(REGIONS))
    for r in res["rules"]:
        per = " | ".join(f"{r['per_region'][x]:.3f}" if r["per_region"][x] is not None
                         else "—" for x in REGIONS)
        A(f"| {r['label']} | {r['geomean']:.4f}x | {r['worst']:.2f}x | "
          f"{r['over_tie']}/{r['n']} | {per} |")
    A("")

    A("## Refinements tested")
    A("")
    A("Both of these were rejected on an 8-core M1, and both are re-tested here "
      "rather than assumed. A GPU with many more cores saturates `qr_unblocked` "
      "much later, so the batch split in particular could be justified elsewhere. "
      "Each is fitted on half the shapes and scored on the other half.")
    A("")
    A(f"Baseline for comparison — `M >= {band['chosen']}` on the held-out half: "
      f"**{res['baseline_held_out']['geomean']:.4f}x** geomean, "
      f"{res['baseline_held_out']['worst']:.2f}x worst.")
    A("")
    A("| refinement | fitted form | train | held-out | held-out worst | verdict |")
    A("|---|---|---|---|---|---|")
    for r in res["refinements"]:
        v = "**adopt**" if r["justified"] else "reject"
        A(f"| {r['name']} | `{r['form']}` | {r['train_geomean']:.4f}x | "
          f"{r['test_geomean']:.4f}x | {r['test_worst']:.2f}x | {v} |")
    A("")
    if any(r["justified"] for r in res["refinements"]):
        A("> At least one refinement cleared held-out validation on this GPU. "
          "`QrPolicy` already carries the batch-split fields "
          "(`m_crossover_small_batch` / `m_crossover_large_batch` / "
          "`batch_threshold`) — set them from the fitted form above.")
    else:
        A("> Neither refinement survived. Ship the plain threshold. A refinement "
          "that looks good on the training half and not on the held-out half is "
          "fitting noise, which is exactly what the split is there to catch.")
    A("")

    A("## Measurement noise")
    A("")
    n = res["noise"]["overall"]
    A(f"Run-to-run ratio over {n['n']} shape/backend pairs measured twice: "
      f"median {n['median']:.3f}x, p90 {n['p90']:.3f}x, max {n['max']:.3f}x.")
    A("")
    A("| runtime | pairs | median | p90 | max |")
    A("|---|---|---|---|---|")
    for r in res["noise"]["by_runtime"]:
        A(f"| {r['bucket']} | {r['n']} | {r['median']:.3f}x | {r['p90']:.3f}x | {r['max']:.3f}x |")
    A("")
    A("Noise concentrates in fast shapes, where submission jitter dominates. A "
      "difference smaller than the p90 figure for its size bucket is not a result.")
    A("")
    A("---")
    A("")
    A("Raw timings in `raw.csv`; full analysis including plot-ready series in "
      "`results.json`. Re-render this report without remeasuring: "
      "`python3 tuning/tune_qr.py --from results.json`.")

    open(path, "w").write("\n".join(L) + "\n")


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("binary", nargs="?", help="path to the built sweep_qr")
    ap.add_argument("--out", default="qr-tune-results", help="output directory")
    ap.add_argument("--passes", type=int, default=2,
                    help="independent passes; 2 is the minimum for a noise floor")
    ap.add_argument("--limit", type=int, default=150, help="per-run timeout, seconds")
    ap.add_argument("--full", action="store_true", help="denser grid, roughly 3x longer")
    ap.add_argument("--from", dest="from_json", help="re-render from an existing results.json")
    ap.add_argument("--reanalyse", metavar="RAW.CSV", nargs="+",
                    help="redo the analysis from existing raw.csv files (several may "
                         "be given; they are merged) without remeasuring")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    if a.from_json:
        res = json.load(open(a.from_json))
        out_md = os.path.join(os.path.dirname(a.from_json) or ".", "report.md")
        write_report(res, out_md)
        print(f"report -> {out_md}")
        return

    if a.reanalyse:
        best, repeats, subs = load(a.reanalyse)
        if not best:
            sys.exit(f"no usable measurements in {a.reanalyse}")
        res = analyse(best, repeats, device_info(a.binary, a.reanalyse))
        res["submissions"] = subs
        with open(os.path.join(a.out, "results.json"), "w") as fh:
            json.dump(res, fh, indent=2)
        write_report(res, os.path.join(a.out, "report.md"))
        b = res["band"]
        print(f"  band {b['lo']}..{b['hi']}   ship M >= {b['chosen']}   "
              f"({b['best_geomean']:.4f}x geomean regret)")
        print(f"  -> {a.out}/report.md, {a.out}/results.json")
        return

    if not a.binary:
        ap.error("need the path to sweep_qr (or --from results.json)")
    if a.passes < 2:
        ap.error("--passes must be at least 2: without a repeat there is no noise floor")

    raw = os.path.join(a.out, "raw.csv")
    pts = shape_grid(a.full)
    counts = defaultdict(int)
    for (_, M, N) in pts:
        counts[region(M, N)] += 1
    print("  grid coverage: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())),
          file=sys.stderr)

    pol = query_policy(a.binary)
    if pol is None:
        sys.exit(f"{a.binary} --policy failed; rebuild sweep_qr")
    with open(os.path.join(a.out, "policy.json"), "w") as fh:
        json.dump(pol, fh, indent=1)
    sweep(a.binary, pts, a.passes, a.limit, raw)
    best, repeats, subs = load(raw)
    if not best:
        sys.exit("no usable measurements — is the binary path right?")

    res = analyse(best, repeats, _from_policy(pol))
    res["submissions"] = subs
    with open(os.path.join(a.out, "results.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    write_report(res, os.path.join(a.out, "report.md"))

    b = res["band"]
    print()
    print(f"  band {b['lo']}..{b['hi']}   ship M >= {b['chosen']}   "
          f"({b['best_geomean']:.4f}x geomean regret)")
    print(f"  kTuned entry:  {res['ktuned_entry']}")
    print(f"  -> {a.out}/report.md, {a.out}/results.json, {a.out}/raw.csv")


if __name__ == "__main__":
    main()
