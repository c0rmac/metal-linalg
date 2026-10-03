"""What a submission is, and how several combine. Shared by the harnesses.

A submission is one run of tuning/run.py on one Mac: a directory

    docs/results/<device>/<id>/
        submission.json      device spec, versions, conditions, the fitted rows
        qr/ eigh/ svd/       each harness's raw.csv, results.json, report.md

where <device> is the chip and GPU core count (apple-m5-pro-20gpu) and <id>
is the UTC date and six random hex digits (20260930-3fa9c1), so any number of
people with the same Mac can contribute without colliding.

Combining: within a submission the passes are reduced to their minimum, as
interference only ever slows a run down. Across submissions, the median of
those minima: machines of one spec differ a little, and the median keeps one
unusual machine from moving the result. The noise floor stays per submission,
so it measures run-to-run noise rather than the spread between machines.
"""

import csv
import os
import re
import secrets
import statistics
import subprocess
import time
from collections import defaultdict


# The single measurement epoch of runs made before 2.6.0, which all recorded 1.
# Which measurements are current is now tracked per decomposition, in
# tuning/kernels.py.
EPOCH = 1


# Memory. A sweep must not allocate more than the Mac it runs on can hold: the
# smallest Apple Silicon Macs have 8 GB, and a point that swaps measures the
# disk, or stops the run. A point is measured only if its estimated peak --
# FOOTPRINT_COPIES copies of its arrays (input, outputs, workspaces, the
# correctness check), in float32 -- fits in MEMORY_FRACTION of physical RAM:
# about 2.8 GB on an 8 GB Mac, 17 GB on a 48 GB one.
# METAL_LINALG_TUNING_MEMORY_GB sets the budget instead.
MEMORY_FRACTION = 0.35
FOOTPRINT_COPIES = 6


def physical_memory_bytes():
    try:
        out = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout
        return int(out.strip())
    except (OSError, ValueError):
        return 0


def memory_budget_bytes():
    env = os.environ.get("METAL_LINALG_TUNING_MEMORY_GB")
    if env:
        return float(env) * 2 ** 30
    mem = physical_memory_bytes()
    return MEMORY_FRACTION * mem if mem else 2.8 * 2 ** 30   # unknown: as an 8 GB Mac


def fits_memory(elements):
    """Whether a point whose largest arrays hold `elements` floats in all fits
    the memory budget."""
    return elements * 4 * FOOTPRINT_COPIES <= memory_budget_bytes()


def submission_of(raw_path):
    """The submission a raw.csv belongs to: its submission directory when it
    sits in one, else the file itself, so stray files each count once."""
    d = os.path.dirname(os.path.dirname(os.path.abspath(raw_path)))
    if os.path.exists(os.path.join(d, "submission.json")):
        return d
    return os.path.abspath(raw_path)


def combine(paths, point_of):
    """Every usable row of the raw.csv files in `paths`, combined.

    `point_of(row)` gives the tuple identifying a measurement point. Returns
    (best, repeats, submissions): best[point][backend] in ms, the median over
    submissions of each one's fastest pass; repeats[point + (backend, i)], the
    passes of submission i, for the noise floor; and the submissions used.
    With one submission this is plain min-of-passes."""
    if isinstance(paths, str):
        paths = [paths]
    per = defaultdict(lambda: defaultdict(list))
    subs = []
    for path in paths:
        sub = submission_of(path)
        if sub not in subs:
            subs.append(sub)
        i = subs.index(sub)
        for r in csv.DictReader(open(path)):
            if r.get("ok") != "1":
                continue
            ms = float(r["ms"])
            if ms > 0:
                per[(point_of(r), r["backend"])][i].append(ms)
    best, repeats = defaultdict(dict), {}
    for (point, backend), by_sub in per.items():
        best[point][backend] = statistics.median(min(v) for v in by_sub.values())
        for i, v in by_sub.items():
            repeats[point + (backend, i)] = v
    return best, repeats, [os.path.basename(s) if os.path.isdir(s) else s for s in subs]


def device_slug(name, gpu_cores):
    """'Apple M5 Pro', 20 -> 'apple-m5-pro-20gpu'."""
    s = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "unknown-device"
    return f"{s}-{gpu_cores}gpu"


def new_id():
    """UTC date and 24 random bits: unique per run, and sorts by date."""
    return time.strftime("%Y%m%d", time.gmtime()) + "-" + secrets.token_hex(3)


def portable(x):
    """`x` with every float rounded to 12 significant digits, for writing.

    The statistics go through math.log and math.exp, which macOS's and
    glibc's maths libraries may round differently in the last bit, so the
    same runs analysed here and by the GitHub Action (Linux) would differ in
    the 16th digit and the Action would commit the difference. Twelve digits
    is far below anything the measurements resolve.
    """
    if isinstance(x, float):
        return float(f"{x:.12g}")
    if isinstance(x, dict):
        return {k: portable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(portable(v) for v in x)
    return x
