"""The Apple Silicon Macs the measurements page lists, measured or not.

One entry per chip, as Metal names it (MTLDevice.name), with the GPU core
counts it ships in: the library tunes per (name, cores), since the binned
variants of a chip have different crossovers. A chip that has been measured
but is missing here is added to the page from its results, so nothing
submitted is hidden; adding it here only makes the page list it before anyone
has measured it. Add new chips as they appear.
"""

GENERATIONS = [
    ("M1", [("Apple M1", [7, 8]), ("Apple M1 Pro", [14, 16]),
            ("Apple M1 Max", [24, 32]), ("Apple M1 Ultra", [48, 64])]),
    ("M2", [("Apple M2", [8, 10]), ("Apple M2 Pro", [16, 19]),
            ("Apple M2 Max", [30, 38]), ("Apple M2 Ultra", [60, 76])]),
    ("M3", [("Apple M3", [8, 10]), ("Apple M3 Pro", [14, 18]),
            ("Apple M3 Max", [30, 40]), ("Apple M3 Ultra", [60, 80])]),
    ("M4", [("Apple M4", [8, 10]), ("Apple M4 Pro", [16, 20]),
            ("Apple M4 Max", [32, 40])]),
    ("M5", [("Apple M5", [10])]),
]


def generation_of(name):
    """'Apple M5 Pro' -> 'M5'; None if it does not look like an M-series chip."""
    parts = name.split()
    if len(parts) >= 2 and parts[0] == "Apple" and parts[1].startswith("M") and parts[1][1:].isdigit():
        return parts[1]
    return None
