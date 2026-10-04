#!/usr/bin/env python3
"""Combine every run submitted for one device into its rows for kTuned[].

    python3 tuning/combine.py docs/results/apple-m5-pro-20gpu

For each decomposition it uses every submission whose measurements of it are
trustworthy and at the current kernel version (tuning/kernels.py) -- or, if
none is, the newest older ones, marked stale -- skips the rest (smoke tests,
interrupted runs, untrustworthy results, superseded kernel versions) and says why,
re-runs the analysis over all of them together, and writes
docs/results/<device>/combined/: one report per decomposition and a
summary.md. No GPU is needed. How runs combine is described in
tuning/submissions.py.

tuning/generate_tables.py does this for every device and writes the rows into
the library; this script is the same step for one device, to look at.
"""

import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import submissions as sub   # noqa: E402
import kernels             # noqa: E402

OPS = [("qr", "tune_qr.py", "src/qr.mm"), ("eigh", "tune_eigh.py", "src/eigh.mm"),
       ("svd", "tune_svd.py", "src/svd.mm")]


# The fields of each row after the device name and core count, as in kTuned[].
FIELDS = {
    "qr": ["m_crossover_small_batch", "m_crossover_large_batch", "batch_threshold",
           "gpu_max_k", "gpu_min_batch_times_k", "gpu_min_batch", "gpu_min_k", "gpu_large_min_k",
           "gpu_large_max_batch", "share_min_batch"],
    "eigh": ["simd_max_n", "block_min_n", "block_min_n_batched", "block_min_batch",
             "gpu_max_n", "gpu_min_batch_times_n", "gpu_min_batch",
             "values_gpu_max_n", "values_gpu_min_batch_times_n", "values_gpu_min_batch",
             "tridiag_min_n", "values_tridiag_min_n", "tridiag_max_batch", "values_tridiag_max_batch",
             "ql_min_n", "ql_max_n", "share_min_batch", "gpu_big_batch_max_n", "gpu_big_batch_min",
             "values_band_min_n"],
    "svd": ["qr_min_rows", "qr_min_k", "block_min_k", "block_min_k_batched", "block_min_batch",
            "gpu_max_k", "gpu_min_batch_times_k", "gpu_min_batch", "gpu_max_l",
            "values_gpu_max_k", "values_gpu_min_batch_times_k", "values_gpu_min_batch", "values_gpu_max_l", "bidiag_min_k", "values_bidiag_min_k",
            "bidiag_max_batch", "values_bidiag_max_batch", "gk_min_k", "gk_max_k", "share_min_batch",
            "gpu_big_batch_max_k", "gpu_big_batch_min", "values_band_min_k"],
}


def row_values(row):
    """'{"Apple M5 Pro", 20,   0, 96, ...}' -> ['0', '96', ...]."""
    body = row.strip().strip("{},").split('"')[-1]
    return [v.strip() for v in body.split(",") if v.strip()][1:]


def row_difference(op, own, combined):
    """How one run's own row differs from the combined one, in words."""
    if not own or not combined:
        return "—"
    a, b = row_values(own), row_values(combined)
    diffs = [f"{f} {x} (combined {y})" for f, x, y in zip(FIELDS[op], a, b) if x != y]
    return "same" if not diffs else "; ".join(diffs)


def describe_conditions(cond):
    """One line on the power and thermal state over a run."""
    if not cond:
        return "not recorded"
    states = [c for c in cond.values() if isinstance(c, dict)]
    power = sorted({c.get("power") for c in states if c.get("power")})
    modes = sorted({c.get("power_mode") for c in states if c.get("power_mode")})
    watts = sorted({c.get("charger_watts") for c in states if c.get("charger_watts")})
    warnings = sorted({line for c in states for line in (c.get("thermal") or [])
                       if "No " not in line and "has been recorded" not in line})
    loads = [c["load_1m"] for c in states if c.get("load_1m") is not None]
    parts = ["/".join(power) or "power unknown"]
    if watts:
        parts.append(f"{'/'.join(str(w) for w in watts)} W charger")
    if modes:
        parts.append("power mode " + "/".join(modes))
    if loads:
        parts.append(f"load {min(loads):.1f}-{max(loads):.1f}")
    parts.append("thermal warnings: " + "; ".join(warnings) if warnings else "no thermal warnings")
    return ", ".join(parts)


def runs_table(subs, ops):
    """Every run with its machine, and whether its own rows agree with the
    combined ones: where to look first when the runs for one chip disagree."""
    L = ["| run | machine | memory | macOS | conditions | " + " | ".join(o for o, _, _ in OPS) + " |",
         "|---|---|---|---|---|" + "---|" * len(OPS)]
    for name, info in subs:
        m = info.get("machine") or {}
        machine = f"{m.get('product_name') or 'unknown'} ({m.get('model_identifier') or '?'})"
        cells = []
        for op, _, _ in OPS:
            own = (info.get("results", {}).get(op) or {}).get("row")
            used = name in ops[op]["used"]
            cells.append(row_difference(op, own, ops[op]["row"]) if used else ("not used" if own else "—"))
        L.append(f"| {name} | {machine} | {info.get('memory_gb') or '?'} GB | {info.get('macos') or '?'} | "
                 f"{describe_conditions(info.get('conditions'))} | " + " | ".join(cells) + " |")
    return "\n".join(L) + "\n"


def load_submissions(device_dir):
    """[(id, submission.json)] for every submission of one device."""
    out = []
    for name in sorted(os.listdir(device_dir)):
        path = os.path.join(device_dir, name, "submission.json")
        if name != "combined" and os.path.exists(path):
            out.append((name, json.load(open(path))))
    return out


def backends_measured(raw):
    """The backends a raw.csv timed successfully."""
    import csv
    return {r["backend"] for r in csv.DictReader(open(raw)) if r.get("ok") == "1"}


def eligible(subs, op, device_dir):
    """(used, skipped, status) for one decomposition: used as [(id, raw.csv)],
    skipped as ["id (reason)"], and status:

        {"state": "current" | "incomplete" | "stale" | "none",
         "epoch": the kernel version the used runs measured,
         "current_epoch": kernels.KERNEL_EPOCHS[op],
         "missing": backends the used runs never timed (incomplete)}

    Runs at the current kernel version are used if there are any; otherwise
    the newest older ones, rather than none: slightly old measurements usually
    beat the untuned default. See tuning/kernels.py."""
    current = kernels.KERNEL_EPOCHS[op]
    usable, skipped = [], []
    for name, info in subs:
        r = info.get("results", {}).get(op)
        raw = os.path.join(device_dir, name, op, "raw.csv")
        if not r or not os.path.exists(raw):
            continue
        if info.get("quick"):
            skipped.append(f"{name} (smoke test)")
        elif info.get("status") == "running":
            skipped.append(f"{name} (interrupted)")
        elif not r.get("trustworthy"):
            skipped.append(f"{name} ({r.get('why') or 'not trustworthy'})")
        else:
            usable.append((name, raw, kernels.run_epoch(info, op)))
    status = {"state": "none", "epoch": None, "current_epoch": current, "missing": []}
    if not usable:
        return [], skipped, status
    at_current = [u for u in usable if u[2] == current]
    if at_current:
        chosen, epoch = at_current, current
    else:
        epoch = max(e for _, _, e in usable if e < current) if any(e < current for _, _, e in usable) \
            else max(e for _, _, e in usable)
        chosen = [u for u in usable if u[2] == epoch]
    for name, _, e in usable:
        if e != epoch:
            skipped.append(f"{name} (measured at kernel version {e}; "
                           f"{'superseded' if e < epoch else 'newer than this checkout'})")
    timed = set().union(*(backends_measured(raw) for _, raw, _ in chosen))
    missing = sorted(kernels.REQUIRED[op] - timed)
    status.update(epoch=epoch, missing=missing,
                  state="stale" if epoch != current else ("incomplete" if missing else "current"))
    return [(n, raw) for n, raw, _ in chosen], skipped, status


def combine_device(device_dir):
    """Combines one device's runs. Returns {"device", "subs", "ops": {op: {"row",
    "used", "skipped"}}} and writes <device_dir>/combined/. Paths handed to the
    harnesses are relative to the repository, so the output does not depend
    on where the repository is."""
    device_dir = os.path.relpath(os.path.abspath(device_dir), ROOT)
    subs = load_submissions(os.path.join(ROOT, device_dir))
    if not subs:
        return None
    out_root = os.path.join(device_dir, "combined")
    result = {"device": subs[0][1]["device"], "subs": [n for n, _ in subs], "ops": {},
              "sub_info": {n: {"date": i.get("date"), "library_version": i.get("library_version"),
                               "macos": i.get("macos"),
                               "machine": (i.get("machine") or {}).get("product_name"),
                               "memory_gb": i.get("memory_gb"),
                               "measured": sorted((i.get("results") or {}).keys()),
                               "epochs": {op: kernels.run_epoch(i, op) for op, _, _ in OPS}}
                           for n, i in subs}}
    for op, harness, target in OPS:
        used, skipped, status = eligible(subs, op, device_dir)
        entry = {"row": None, "used": [n for n, _ in used], "skipped": skipped, "target": target,
                 "status": status}
        if used:
            out = os.path.join(out_root, op)
            cmd = [sys.executable, os.path.join("tuning", harness), "--reanalyse",
                   *[raw for _, raw in used], "--out", out]
            r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
            if r.returncode != 0:
                raise RuntimeError(f"{harness} failed for {device_dir}:\n{r.stdout}{r.stderr}")
            res = json.load(open(os.path.join(ROOT, out, "results.json")))
            entry["row"] = (res.get("ktuned_entry") or res.get("tuned_row")).rstrip(",").strip()
        result["ops"][op] = entry

    lines = []
    for op, e in result["ops"].items():
        skip = f"; skipped {', '.join(e['skipped'])}" if e["skipped"] else ""
        if e["row"]:
            n = len(e["used"])
            st = e["status"]
            state = {"current": "current", "incomplete": "incomplete, never timed " + ", ".join(st["missing"]),
                     "stale": f"stale, measured at kernel version {st['epoch']} (current {st['current_epoch']})"
                     }[st["state"]]
            lines.append(f"- **{op}** ({state}) from {n} run{'s' if n > 1 else ''} ({', '.join(e['used'])}){skip}: "
                         f"`{e['row']}` in `{e['target']}` ([report]({op}/report.md))")
        else:
            lines.append(f"- **{op}**: no usable runs{skip}")
    d = result["device"]
    result["runs_table"] = runs_table(subs, result["ops"])
    os.makedirs(os.path.join(ROOT, out_root), exist_ok=True)
    with open(os.path.join(ROOT, out_root, "summary.md"), "w") as fh:
        fh.write(f"# {d['name']}, {d['gpu_cores']} GPU cores — combined\n\n"
                 f"Generated by `tuning/generate_tables.py` from {len(subs)} "
                 f"run{'s' if len(subs) > 1 else ''}; do not edit.\n\n" + "\n".join(lines) + "\n\n"
                 "## Runs\n\n"
                 "Each run's machine, and how the settings it fitted on its own differ from the "
                 "combined ones. A machine whose runs consistently disagree with the others for "
                 "the same chip is the first thing to check.\n\n" + result["runs_table"])
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("device_dir", help="docs/results/<device>")
    args = ap.parse_args()
    res = combine_device(args.device_dir)
    if res is None:
        sys.exit(f"no submissions in {args.device_dir}")
    d = res["device"]
    print(f"{d['name']}, {d['gpu_cores']} GPU cores: {len(res['subs'])} runs")
    for op, e in res["ops"].items():
        print(f"  {e['target']:12s} {e['row'] or '(no usable runs)'}")
        for s in e["skipped"]:
            print(f"      skipped {s}")


if __name__ == "__main__":
    main()
