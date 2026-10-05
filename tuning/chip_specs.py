"""What is published about each Apple Silicon Mac, for the estimated policies.

A Mac nobody has measured gets its routing estimated from one that has been
(tuning/estimate.py, docs/studies/estimated-policies.md): from how fast its
GPU is against its CPU, relative to the measured Mac's. These are the numbers
that ratio is taken from, per (chip, GPU cores, CPU cores):

    cpu        Geekbench 7 multi-core score
    metal      Geekbench 7 Metal (GPU compute) score
    bandwidth  unified-memory bandwidth, GB/s, from Apple's specifications

The Geekbench scores are the medians of the per-Mac averages in Primate Labs'
Mac Benchmark Chart (https://browser.geekbench.com/mac-benchmarks, read
2026-10-05), across the Macs that ship the configuration. Where the chart has
no entry for a configuration, the score is derived and `derived` says how:
from the same chip with more GPU cores, scaled by core count, or for the M5
Ultra from the GPU chart (https://browser.geekbench.com/metal-benchmarks) and
the Ultra-to-Max multi-core ratio of M1 to M3 (1.6).

A Mac missing here is estimated from its core counts instead (estimate.mm).
Add new Macs as they appear in the chart; tuning/generate_tables.py writes
this table into src/tuned/chips.inc.
"""

# (name as MTLDevice.name, GPU cores, CPU cores, cpu, metal, bandwidth, derived)
SPECS = [
    ("Apple M1", 7, 8, 8244, 26940, 68.25, ""),
    ("Apple M1", 8, 8, 8376, 28220, 68.25, ""),
    ("Apple M1 Pro", 14, 8, 11186, 61045, 200, ""),
    ("Apple M1 Pro", 16, 10, 13918, 65411, 200, ""),
    ("Apple M1 Max", 24, 10, 14220, 80797, 400, "metal: 32-core x 24/32"),
    ("Apple M1 Max", 32, 10, 14220, 107729, 400, ""),
    ("Apple M1 Ultra", 48, 20, 23590, 110949, 800, "metal: 64-core x 48/64"),
    ("Apple M1 Ultra", 64, 20, 23590, 147932, 800, ""),
    ("Apple M2", 8, 8, 9675, 34592, 100, "metal: 10-core x 8/10"),
    ("Apple M2", 10, 8, 9667, 43240, 100, ""),
    ("Apple M2 Pro", 16, 10, 13769, 74117, 200, ""),
    ("Apple M2 Pro", 19, 12, 16608, 82790, 200, ""),
    ("Apple M2 Max", 30, 12, 17034, 112733, 400, "metal: 38-core x 30/38"),
    ("Apple M2 Max", 38, 12, 17191, 142795, 400, ""),
    ("Apple M2 Ultra", 60, 24, 28252, 174681, 800, "metal: 76-core x 60/76"),
    ("Apple M2 Ultra", 76, 24, 28252, 221263, 800, ""),
    ("Apple M3", 8, 8, 11708, 43077, 100, ""),
    ("Apple M3", 10, 8, 11639, 47897, 100, ""),
    ("Apple M3 Pro", 14, 11, 15285, 70153, 150, ""),
    ("Apple M3 Pro", 18, 12, 16846, 76900, 150, ""),
    ("Apple M3 Max", 30, 14, 22116, 131837, 300, ""),
    ("Apple M3 Max", 40, 16, 24463, 164453, 400, ""),
    ("Apple M3 Ultra", 60, 28, 35125, 225674, 819, ""),
    ("Apple M3 Ultra", 80, 32, 39168, 249179, 819, ""),
    ("Apple M4", 8, 8, 13437, 47416, 120, ""),
    ("Apple M4", 8, 10, 15080, 47609, 120, ""),
    ("Apple M4", 10, 10, 15175, 53134, 120, ""),
    ("Apple M4 Pro", 16, 12, 21730, 102286, 273, ""),
    ("Apple M4 Pro", 20, 14, 24770, 114846, 273, ""),
    ("Apple M4 Max", 32, 14, 25370, 169387, 410, ""),
    ("Apple M4 Max", 40, 16, 28723, 204645, 546, ""),
    ("Apple M5", 8, 10, 17054, 65773, 153, ""),
    ("Apple M5", 10, 10, 17172, 71194, 153, ""),
    ("Apple M5 Pro", 16, 15, 29568, 124778, 307, ""),
    ("Apple M5 Pro", 20, 18, 33545, 135808, 307, ""),
    ("Apple M5 Max", 32, 18, 34809, 194121, 460, ""),
    ("Apple M5 Max", 40, 18, 34809, 239533, 614, ""),
    ("Apple M5 Ultra", 64, 36, 55694, 277600, 920,
     "cpu: M5 Max x 1.6; metal: GPU chart x 64/80; bandwidth: 2 x the 32-core M5 Max"),
    ("Apple M5 Ultra", 80, 36, 55694, 347000, 1228, "cpu: M5 Max x 1.6; metal: GPU chart"),
    ("Apple A18 Pro", 5, 6, 7285, 31147, 60, ""),     # MacBook Neo
]


def find(name, gpu_cores, cpu_cores=None):
    """The entry for that Mac: the exact CPU core count if one is given and
    listed, else the first with that chip and GPU core count; None if none."""
    rows = [s for s in SPECS if s[0] == name and s[1] == gpu_cores]
    exact = [s for s in rows if s[2] == cpu_cores]
    return (exact or rows or [None])[0]
