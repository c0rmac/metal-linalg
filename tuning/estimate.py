#!/usr/bin/env python3
"""Estimated policies for the Macs nobody has measured.

    python3 tuning/estimate.py                # the per-Mac estimates, as a table
    python3 tuning/estimate.py --study        # docs/studies/estimated-policies.md's tables

A measured Mac's routing comes from its own timings. Any other Mac gets an
estimate: the timings of a measured Mac (the anchor) refitted as if its GPU
were s times slower against its CPU, s taken from how the two Macs compare.
The refit is the anchor's own analysis (tune_*.py --reanalyse) on its raw
timings with every GPU backend's time multiplied by s, so an estimated row is
a fitted row like any other, for a GPU that much weaker.

Two slowdowns, since the backends are limited by different things:

    small  the batched kernels (one threadgroup a matrix), and the Jacobi
           block kernels: the GPU's compute against the CPU's
    large  the large-matrix reductions (eigh tridiag and band, SVD bidiag
           and band, QR's streaming kernel), which stream the matrix through
           memory: also the memory bandwidth against the CPU

For a Mac in tuning/chip_specs.py, from Geekbench 7's multi-core and Metal
scores and the memory bandwidth:

    q_small = (metal / cpu) / (anchor's metal / cpu)
    q_large = min(q_small, (bandwidth / cpu) / (anchor's bandwidth / cpu))
    s       = max(1, margin / q),  margin 1.25 within the anchor's generation, else 1.5

and for any other Mac from its core counts alone, q = (GPU cores / CPU cores)
over the anchor's, with margin 2 (the check in the study puts that model
within 2x of the benchmark one for every listed Mac). s is never below 1: a
GPU stronger than the anchor's still gets the anchor's own crossovers, which
leave some of its wins unused but route nothing to a backend that loses.

The library picks the estimated row whose slowdowns are the smallest on the
LADDER at least the Mac's (src/estimate.mm mirrors slowdowns() below); the
rows are refitted for every pair on the ladder, for every anchor, by
generate() here, which tuning/generate_tables.py calls, so the estimates
follow the measurements.
"""

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import chip_specs  # noqa: E402

LADDER = [1.0, 1.15, 1.3, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0]
MARGIN_SAME_GENERATION = 1.25   # benchmarks, a Mac of the anchor's generation
MARGIN_OTHER_GENERATION = 1.5   # benchmarks, another generation
MARGIN_CORE_COUNTS = 2.0        # a Mac with no benchmarks: core counts alone

HARNESS = {"qr": "tune_qr.py", "eigh": "tune_eigh.py", "svd": "tune_svd.py"}

# Which raw.csv backends run on the CPU alone, which are large-matrix GPU
# backends, and which share a batch between a GPU backend and the CPU path
# (shared: the GPU and CPU backends timed alone at the same point). Every
# other backend is a batched GPU kernel.
BACKENDS = {
    "eigh": {"cpu": {"cpu", "cpu_vals"},
             "large": {"tridiag", "tridiag_vals", "band", "band_vals", "band8_vals", "band32_vals",
                       "tridiag_batch", "tridiag_batch_vals"},
             "shared": {"ql_share": ("ql", "cpu"), "ql_share_vals": ("ql_vals", "cpu_vals")}},
    "svd": {"cpu": {"cpu", "cpu_vals"},
            "large": {"bidiag", "bidiag_vals", "band", "band_vals", "band8_vals", "band32_vals",
                      "bidiag_batch", "bidiag_batch_vals"},
            "shared": {"gk_share": ("gk", "cpu"), "gk_share_vals": ("gk_vals", "cpu_vals")}},
    "qr": {"cpu": {"cpu"},
           "large": {"reduced"},
           "shared": {"share": ("unblocked", "cpu")}},
}


# ---------------------------------------------------------------------------
# How a Mac compares with an anchor
# ---------------------------------------------------------------------------

def generation(name):
    """'Apple M5 Pro' -> 'm5', 'Apple A18 Pro' -> 'a18'; None for anything else."""
    parts = chip_specs.canonical(name).split()
    if len(parts) >= 2 and parts[0] == "apple" and parts[1][:1] in ("m", "a") and parts[1][1:].isdigit():
        return parts[1]
    return None


def slowdowns(device, gpu_cores, cpu_cores, anchor):
    """(small, large, basis) for a Mac against an anchor (a chip_specs row):
    how many times slower its GPU is against its CPU than the anchor's,
    margin included, at least 1."""
    _, a_gpu, a_cpu, a_score, a_metal, a_bw, _ = anchor
    if not gpu_cores or not cpu_cores:
        return LADDER[-1], LADDER[-1], "unknown cores"
    spec = chip_specs.find(device, gpu_cores, cpu_cores)
    if spec:
        _, _, _, score, metal, bw, _ = spec
        q_small = (metal / score) / (a_metal / a_score)
        q_large = min(q_small, (bw / score) / (a_bw / a_score))
        same = generation(device) is not None and generation(device) == generation(anchor[0])
        margin, basis = (MARGIN_SAME_GENERATION if same else MARGIN_OTHER_GENERATION), "benchmarks"
    else:
        q_small = q_large = (gpu_cores / cpu_cores) / (a_gpu / a_cpu)
        margin, basis = MARGIN_CORE_COUNTS, "core counts"
    return max(1.0, margin / q_small), max(1.0, margin / q_large), basis


def on_ladder(s):
    """The smallest ladder value at least s (the largest if none is)."""
    return next((v for v in LADDER if v >= s - 1e-9), LADDER[-1])


def choose_anchor(device, gpu_cores, anchors):
    """The anchor (a chip_specs row) for a Mac: one of its generation if there
    is one, then the nearest in GPU cores."""
    gen = generation(device)
    return min(anchors, key=lambda a: (generation(a[0]) != gen,
                                       abs(math.log(max(gpu_cores, 1) / a[1]))))


# ---------------------------------------------------------------------------
# Refitting an anchor's timings for a weaker GPU
# ---------------------------------------------------------------------------

def _point(r):
    return tuple(r[k] for k in ("batch", "M", "N") if k in r)


def inflate(src, dst, op, small, large):
    """src raw.csv to dst with every GPU backend's times multiplied: batched
    kernels by `small`, large-matrix ones by `large`. A shared backend's time
    is rescaled as two workers in parallel, 1/T = 1/T_gpu + 1/T_cpu, keeping
    the share's measured overhead over that ideal."""
    with open(src, newline="") as fh:
        rows = list(csv.DictReader(fh))
    best = defaultdict(dict)
    for r in rows:
        if r["ok"] == "1":
            t = best[_point(r)]
            t[r["backend"]] = min(float(r["ms"]), t.get(r["backend"], math.inf))
    kinds = BACKENDS[op]

    def factor(r):
        b = r["backend"]
        if b in kinds["cpu"]:
            return 1.0
        if b in kinds["shared"]:
            gpu, cpu = kinds["shared"][b]
            t = best[_point(r)]
            if gpu in t and cpu in t:
                both = lambda g, c: 1.0 / (1.0 / g + 1.0 / c)
                return both(small * t[gpu], t[cpu]) / both(t[gpu], t[cpu])
            return small
        return large if b in kinds["large"] else small

    with open(dst, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            f = factor(r)
            for c in ("ms", "p25", "p75"):
                if r.get(c):
                    r[c] = f"{float(r[c]) * f:.6f}"
            w.writerow(r)


def refit(op, raws, small, large, work):
    """The row tune_<op>.py fits from the anchor's raw.csv files, inflated."""
    d = tempfile.mkdtemp(prefix=f"{op}-{small}-{large}-", dir=work)
    paths = []
    for i, raw in enumerate(raws):
        sub = os.path.join(d, f"run{i}")
        os.makedirs(sub)
        inflate(os.path.join(ROOT, raw), os.path.join(sub, "raw.csv"), op, small, large)
        side = os.path.join(ROOT, os.path.dirname(raw), "policy.json")
        if os.path.exists(side):
            shutil.copy(side, sub)
        paths.append(os.path.join(sub, "raw.csv"))
    out = os.path.join(d, "out")
    r = subprocess.run([sys.executable, os.path.join(HERE, HARNESS[op]), "--reanalyse", *paths, "--out", out],
                       capture_output=True, text=True, cwd=ROOT)
    if r.returncode != 0:
        raise RuntimeError(f"{HARNESS[op]} failed refitting at {small}, {large}:\n{r.stdout}{r.stderr}")
    res = json.load(open(os.path.join(out, "results.json")))
    return (res.get("ktuned_entry") or res.get("tuned_row")).rstrip(",").strip()


def generate(anchors):
    """{op: include-file text} of estimated rows, and src/tuned/chips.inc's
    text. anchors: {op: [(chip_specs row, [raw.csv paths relative to the
    repository])]}, the measured Macs with current runs for that op."""
    files = {}
    with tempfile.TemporaryDirectory() as work, ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as pool:
        for op in ("qr", "eigh", "svd"):
            jobs = [(spec, raws, s, l) for spec, raws in anchors.get(op, []) for s in LADDER for l in LADDER]
            rows = list(pool.map(lambda j: refit(op, j[1], j[2], j[3], work), jobs))
            lines = ["// Generated by tuning/generate_tables.py (tuning/estimate.py); do not edit.",
                     "// A measured Mac's timings refitted as if its GPU were the first two",
                     "// numbers / 100 times slower against its CPU (batched kernels, then",
                     "// large-matrix backends), for the Macs nobody has measured; the third",
                     "// is the measured Mac's CPU core count. See docs/tuning.md and",
                     "// docs/studies/estimated-policies.md."]
            for (spec, _, s, l), row in zip(jobs, rows):
                lines.append(f"{{{round(s * 100)}, {round(l * 100)}, {spec[2]}, {row}}},")
            files[op] = "\n".join(lines) + "\n"
    chips = ["// Generated by tuning/generate_tables.py from tuning/chip_specs.py; do not edit.",
             "// name, GPU cores, CPU cores, Geekbench 7 multi-core and Metal, bandwidth GB/s."]
    for name, gpu, cpu, score, metal, bw, _ in chip_specs.SPECS:
        chips.append(f'{{"{name}", {gpu}, {cpu}, {score}, {metal}, {bw}}},')
    files["chips"] = "\n".join(chips) + "\n"
    return files


# ---------------------------------------------------------------------------
# The per-Mac table
# ---------------------------------------------------------------------------

def per_mac(anchor_specs):
    """[(name, gpu, cpu, anchor, small, large, basis)] for every listed Mac."""
    import chips   # noqa: E402  (the measurements page's chip list)
    out, seen = [], set()
    for name, gpu, cpu, *_ in chip_specs.SPECS:
        a = choose_anchor(name, gpu, anchor_specs)
        s, l, basis = slowdowns(name, gpu, cpu, a)
        out.append((name, gpu, cpu, a[0] + f" ({a[1]})", s, l, basis))
        seen.add((name, gpu))
    for _, members in chips.GENERATIONS:
        for name, cores in members:
            for gpu in cores:
                if (name, gpu) not in seen:
                    out.append((name, gpu, None, "", None, None, "no CPU core count listed"))
    return out


# ---------------------------------------------------------------------------
# The study: estimated rows on simulated Macs
# ---------------------------------------------------------------------------
#
# A Mac whose GPU is s times slower against its CPU than the anchor's is
# simulated by the anchor's timings inflated by s, as the refit does. On those
# timings each candidate policy is scored against the best backend timed at
# each point: the untuned default, the anchor's own row, and the estimated
# rows for slowdowns at, above and below the true one.

# The struct fields a row holds, in order (src/<op>.mm, TunedEntry), as policy
# fields of the C API.
ROW_FIELDS = {
    "eigh": ["simd_max_n", "block_min_n", "block_min_n_batched", "block_min_batch", "gpu_max_n",
             "gpu_min_batch_times_n", "gpu_min_batch", "values_gpu_max_n", "values_gpu_min_batch_times_n",
             "values_gpu_min_batch", "tridiag_min_n", "values_tridiag_min_n", "tridiag_max_batch",
             "values_tridiag_max_batch", "ql_min_n", "ql_max_n", "share_min_batch", "gpu_big_batch_max_n",
             "gpu_big_batch_min", "values_band_min_n", "values_band_width", "band_min_n", "tridiag_batch_min_n",
             "tridiag_batch_max_n", "tridiag_batch_min_batch", "values_tridiag_batch_min_n",
             "values_tridiag_batch_max_n", "values_tridiag_batch_min_batch"],
    "svd": ["qr_min_rows", "qr_min_k", "block_min_k", "block_min_k_batched", "block_min_batch", "gpu_max_k",
            "gpu_min_batch_times_k", "gpu_min_batch", "gpu_max_l", "values_gpu_max_k",
            "values_gpu_min_batch_times_k", "values_gpu_min_batch", "values_gpu_max_l", "bidiag_min_k",
            "values_bidiag_min_k", "bidiag_max_batch", "values_bidiag_max_batch", "gk_min_k", "gk_max_k",
            "share_min_batch", "gpu_big_batch_max_k", "gpu_big_batch_min", "values_band_min_k",
            "values_band_width", "band_min_k", "bidiag_batch_min_k", "bidiag_batch_max_k",
            "bidiag_batch_min_batch", "bidiag_batch_max_l", "values_bidiag_batch_min_k",
            "values_bidiag_batch_max_k", "values_bidiag_batch_min_batch", "values_bidiag_batch_max_l"],
    "qr": ["m_crossover_small_batch", "m_crossover_large_batch", "batch_threshold", "gpu_max_k",
           "gpu_min_batch_times_k", "gpu_min_batch", "gpu_min_k", "gpu_large_min_k", "gpu_large_max_batch",
           "share_min_batch"],
}
NO_LIMIT = 0xFFFFFFFF
DEFAULTS = {   # the untuned default (include/metal_linalg/core.h, and qr.mm's crossovers)
    "eigh": dict(simd_max_n=8, block_min_n=96, block_min_n_batched=0, block_min_batch=0, gpu_max_n=64,
                 gpu_min_batch_times_n=1024, gpu_min_batch=1, values_gpu_max_n=0,
                 values_gpu_min_batch_times_n=0, values_gpu_min_batch=0, tridiag_min_n=0,
                 values_tridiag_min_n=0, tridiag_max_batch=0, values_tridiag_max_batch=0, values_band_min_n=0,
                 values_band_width=0, band_min_n=0, tridiag_batch_min_n=0, tridiag_batch_max_n=0,
                 tridiag_batch_min_batch=0, values_tridiag_batch_min_n=0, values_tridiag_batch_max_n=0,
                 values_tridiag_batch_min_batch=0,
                 gpu_big_batch_max_n=0, gpu_big_batch_min=0, ql_min_n=0, ql_max_n=0, share_min_batch=0),
    "svd": dict(qr_min_rows=512, qr_min_k=64, block_min_k=192, block_min_k_batched=0, block_min_batch=0,
                gpu_max_k=64, gpu_min_batch_times_k=1024, gpu_min_batch=1, gpu_max_l=NO_LIMIT,
                gpu_big_batch_max_k=0, gpu_big_batch_min=0, values_gpu_max_k=0, values_gpu_min_batch_times_k=0,
                values_gpu_min_batch=0, values_gpu_max_l=NO_LIMIT, bidiag_min_k=0, values_bidiag_min_k=0,
                bidiag_max_batch=0, values_bidiag_max_batch=0, values_band_min_k=0, values_band_width=0,
                band_min_k=0, gk_min_k=0, gk_max_k=0, bidiag_batch_min_k=0, bidiag_batch_max_k=0,
                bidiag_batch_min_batch=0, bidiag_batch_max_l=NO_LIMIT, values_bidiag_batch_min_k=0,
                values_bidiag_batch_max_k=0, values_bidiag_batch_min_batch=0, values_bidiag_batch_max_l=NO_LIMIT,
                share_min_batch=0),
    "qr": dict(m_crossover_small_batch=384, m_crossover_large_batch=384, batch_threshold=16,
               gpu_max_k=NO_LIMIT, gpu_min_batch_times_k=1024, gpu_min_batch=1, gpu_min_k=0,
               gpu_large_min_k=0, gpu_large_max_batch=0, share_min_batch=0),
}


def row_policy(op, row):
    """A generated row ('{"Apple M5 Pro", 20,   0, 96, ...}') as C API policy fields."""
    values = [NO_LIMIT if v.strip().endswith("NoLimit") else int(v) for v in row.strip("{} ").split(",")[2:]]
    p = dict(zip(ROW_FIELDS[op], values))
    if op == "qr" and p["gpu_min_batch"] == 0:   # measured before the CPU path (qr.mm's apply)
        p.update(gpu_max_k=NO_LIMIT, gpu_min_batch_times_k=0, gpu_min_batch=1, gpu_min_k=0)
    return p


def _library():
    import importlib.util
    os.environ.setdefault("METAL_LINALG_NO_CALIBRATION_NOTICE", "1")
    os.environ.setdefault("METAL_LINALG_TORCH_LIBRARY",
                          os.path.join(ROOT, "build-torch", "libmetal_linalg.dylib"))
    spec = importlib.util.spec_from_file_location(
        "_lib", os.path.join(ROOT, "python-torch", "metal_linalg_torch", "_lib.py"))
    lib = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lib)
    return lib


def _times(raws):
    """{(M, N, batch): {backend: ms}}: the fastest pass of each run, then the
    median over runs, as tuning/combine.py combines them."""
    per_run = []
    for raw in raws:
        t = defaultdict(dict)
        with open(os.path.join(ROOT, raw), newline="") as fh:
            for r in csv.DictReader(fh):
                if r["ok"] != "1":
                    continue
                key = (int(r.get("M", r["N"])), int(r["N"]), int(r["batch"]))
                t[key][r["backend"]] = min(float(r["ms"]), t[key].get(r["backend"], math.inf))
        per_run.append(t)
    out = defaultdict(dict)
    for key in set().union(*per_run):
        for b in set().union(*(t.get(key, {}) for t in per_run)):
            v = sorted(t[key][b] for t in per_run if b in t.get(key, {}))
            out[key][b] = v[len(v) // 2] if len(v) % 2 else 0.5 * (v[len(v) // 2 - 1] + v[len(v) // 2])
    return out


def _inflated(op, times, small, large):
    kinds = BACKENDS[op]
    both = lambda g, c: 1.0 / (1.0 / g + 1.0 / c)
    out = {}
    for key, t in times.items():
        u = {}
        for b, ms in t.items():
            if b in kinds["cpu"]:
                u[b] = ms
            elif b in kinds["shared"]:
                g, c = kinds["shared"][b]
                u[b] = ms * (both(small * t[g], t[c]) / both(t[g], t[c]) if g in t and c in t else small)
            else:
                u[b] = ms * (large if b in kinds["large"] else small)
        out[key] = u
    return out


BAND_COLUMN = {8: "band8", 32: "band32"}   # the band backend's raw.csv columns by width (16: "band")


def _column(lib, op, values, m, n, batch):
    """The raw.csv backend the policy in effect routes a call to."""
    pol = lib.get_policy(op)
    if op == "eigh":
        name = lib.text((lib.eigvalsh_backend if values else lib.eigh_backend)(n, batch))
        col = {"cpu": "cpu", "simd": "simd", "threadgroup": "tg", "block": "block", "tridiag": "tridiag",
               "ql": "ql_share" if pol["share_min_batch"] and batch >= pol["share_min_batch"] else "ql",
               # with eigenvectors the band is 16 wide
               "band": BAND_COLUMN.get(pol.get("values_band_width", 0), "band") if values else "band",
               "tridiag_batch": "tridiag_batch"}[name]
        return col + "_vals" if values else col
    if op == "svd":
        name = lib.text((lib.svdvals_backend if values else lib.svd_backend)(m, n, batch))
        share = pol["share_min_batch"] and batch >= pol["share_min_batch"]
        col = {"cpu": "cpu", "jacobi": "jacobi", "block_jacobi": "block", "qr_jacobi": "qr",
               "qr_block_jacobi": "qrblock", "bidiag": "bidiag",
               # with vectors the band is 16 wide
               "band": BAND_COLUMN.get(pol.get("values_band_width", 0), "band") if values else "band",
               "golub_kahan": "gk_share" if share else "gk", "qr_golub_kahan": "qr_gk",
               "bidiag_batch": "bidiag_batch"}[name]
        return col + "_vals" if values and col in ("cpu", "gk", "gk_share", "bidiag", "band", "band8", "band32",
                                                   "bidiag_batch") \
            else col
    name = lib.text(lib.qr_backend(m, n, batch))
    if name != "cpu" and pol["share_min_batch"] and batch >= pol["share_min_batch"]:
        return "share"
    return {"cpu": "cpu", "unblocked": "unblocked", "streaming_reduced": "reduced"}[name]


def _score(lib, op, values, policy, times):
    """(geometric-mean regret, worst, share of points over 1.25x, points) of
    a policy against the best backend timed at each point."""
    saved = lib.get_policy(op)
    lib.set_policy(op, policy)
    try:
        regrets = []
        for (m, n, b), t in times.items():
            col = _column(lib, op, values, m, n, b)
            pool = {k: v for k, v in t.items() if k.endswith("_vals") == values or op == "qr"
                    or (values and k in ("jacobi", "block", "qr", "qrblock", "simd", "tg", "ql"))}
            if col not in t or not pool:
                continue
            regrets.append(t[col] / min(pool.values()))
    finally:
        lib.set_policy(op, saved)
    g = math.exp(sum(map(math.log, regrets)) / len(regrets))
    return g, max(regrets), sum(r > 1.25 for r in regrets) / len(regrets), len(regrets)


def study(anchor_name="Apple M5 Pro", anchor_gpu=20):
    import combine
    import generate_tables
    lib = _library()
    per_device = {d: combine.combine_device(os.path.join(ROOT, generate_tables.RESULTS, d))
                  for d in generate_tables.devices()}
    anchors = generate_tables.anchors({d: r for d, r in per_device.items() if r})
    rows = {}
    for op in ("qr", "eigh", "svd"):
        text = open(os.path.join(ROOT, "src", "tuned", f"{op}_estimated.inc")).read()
        for line in text.splitlines():
            if line.startswith("{") and f'"{anchor_name}", {anchor_gpu},' in line:
                s, l, _, rest = line.split(",", 3)
                rows[(op, int(s.strip("{ ")), int(l))] = rest.strip().rstrip(",").rstrip("}").strip() + "}"
    print("## Simulated Macs\n")
    print("Each Mac is the anchor's timings with every GPU backend s times slower (the batched kernels "
          "and the large-matrix backends alike). Geometric-mean time over the best backend timed at "
          "each point, worst point, and the share of points more than 1.25x off.\n")
    cases = [("untuned default", None), ("anchor's own row", 1.0), ("estimated, s exact", "exact"),
             ("estimated, s x 1.25 (the margin)", 1.25), ("estimated, s / 1.5 (benchmarks 1.5x off)", 1 / 1.5)]
    for op, values, title in (("eigh", False, "eigh"), ("eigh", True, "eigvalsh"), ("svd", False, "SVD"),
                              ("svd", True, "svdvals"), ("qr", False, "QR")):
        spec, raws = next((a, r) for a, r in anchors[op] if a[0] == anchor_name and a[1] == anchor_gpu)
        base = _times(raws)
        print(f"### {title}\n")
        print("| true s | " + " | ".join(c for c, _ in cases) + " |")
        print("|---|" + "---|" * len(cases))
        for s_true in (1.0, 1.3, 1.75, 2.5, 4.0):
            t = _inflated(op, base, s_true, s_true)
            cells = []
            for _, how in cases:
                if how is None:
                    pol = DEFAULTS[op]
                else:
                    s_est = 1.0 if how == 1.0 else on_ladder(s_true if how == "exact" else max(1.0, s_true * how))
                    pol = row_policy(op, rows[(op, round(s_est * 100), round(s_est * 100))])
                g, worst, over, _ = _score(lib, op, values, pol, t)
                cells.append(f"{g:.2f}x, worst {worst:.1f}x, {over:.0%}")
            print(f"| {s_true} | " + " | ".join(cells) + " |")
        print()
    print("## Core counts against benchmarks\n")
    print("For every listed Mac, the GPU-against-CPU ratio the core-count fallback gives over the one "
          "the benchmarks give (above 1: the fallback overrates the GPU).\n")
    a = chip_specs.find(anchor_name, anchor_gpu)
    ratios = []
    print("| Mac | GPU cores | CPU cores | from benchmarks | from core counts | ratio |")
    print("|---|---|---|---|---|---|")
    for name, gpu, cpu, score, metal, bw, _ in chip_specs.SPECS:
        qb = (metal / score) / (a[4] / a[3])
        qc = (gpu / cpu) / (a[1] / a[2])
        ratios.append(qc / qb)
        print(f"| {name} | {gpu} | {cpu} | {qb:.2f} | {qc:.2f} | {qc / qb:.2f} |")
    print(f"\nRatio from {min(ratios):.2f} to {max(ratios):.2f}.\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--anchor", default="Apple M5 Pro:20", help="name:GPU cores of the measured Mac")
    ap.add_argument("--study", action="store_true", help="the study's tables (needs build-torch/ built)")
    args = ap.parse_args()
    name, gpu = args.anchor.rsplit(":", 1)
    if args.study:
        study(name, int(gpu))
        return
    anchor = chip_specs.find(name, int(gpu))
    print("| Mac | GPU cores | CPU cores | basis | slowdown, batched | slowdown, large | ladder |")
    print("|---|---|---|---|---|---|---|")
    for n, g, c, _, s, l, basis in per_mac([anchor]):
        if s is None:
            continue
        print(f"| {n} | {g} | {c} | {basis} | {s:.2f} | {l:.2f} | {on_ladder(s)}, {on_ladder(l)} |")


if __name__ == "__main__":
    main()
