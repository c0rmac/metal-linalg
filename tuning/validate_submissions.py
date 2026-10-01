#!/usr/bin/env python3
"""Check that every submission under docs/results/ is well-formed.

    python3 tuning/validate_submissions.py

Run by the GitHub Action on every pull request that adds results, before
anything is computed from them. It only reads files. Checks, per submission:
the folder is docs/results/<device>/<id>/ with <device> matching the chip and
GPU core count in submission.json and <id> of the form 20260930-27b6c2; only
the files tuning/run.py writes are present, none of them oversized; every
decomposition it reports has a raw.csv with the expected columns and sane
values; and it is a complete, full run rather than a smoke test.
"""

import csv
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import submissions as sub   # noqa: E402

RESULTS = os.path.join(ROOT, "docs", "results")
OPS = {"qr": (["pass", "batch", "M", "N", "backend", "ok", "ms", "p25", "p75", "reps"],
              {"unblocked", "reduced", "complete"}),
       "eigh": (["pass", "batch", "N", "backend", "ok", "ms", "p25", "p75", "reps"],
                {"cpu", "simd", "tg", "block"}),
       "svd": (["pass", "batch", "M", "N", "backend", "ok", "ms", "p25", "p75", "reps"],
               {"cpu", "jacobi", "block", "qr", "qrblock"})}
TOP_FILES = {"submission.json", "summary.md", "qr.log", "eigh.log", "svd.log"}
OP_FILES = {"raw.csv", "results.json", "report.md", "policy.json"}
ID = re.compile(r"^\d{8}-[0-9a-f]{6}$")
MAX_FILE = 5 * 2**20


def check_raw(full, op, errors):
    header, backends = OPS[op]
    path = os.path.relpath(full, ROOT)
    with open(full, newline="") as fh:
        rows = csv.reader(fh)
        first = next(rows, None)
        if first != header:
            errors.append(f"{path}: columns {first}, expected {header}")
            return
        n = 0
        for i, r in enumerate(rows, start=2):
            if len(r) != len(header):
                errors.append(f"{path}:{i}: {len(r)} fields, expected {len(header)}")
                return
            row = dict(zip(header, r))
            try:
                ints = [int(row[k]) for k in ("pass", "batch", "N", "ok", "reps") + (("M",) if "M" in row else ())]
                ms = float(row["ms"])
            except ValueError:
                errors.append(f"{path}:{i}: a value is not a number")
                return
            if row["backend"] not in backends:
                errors.append(f"{path}:{i}: unknown backend {row['backend']!r}")
                return
            if min(ints) < 0 or not (0 <= ms < 1e7):
                errors.append(f"{path}:{i}: value out of range")
                return
            n += 1
        if n == 0:
            errors.append(f"{path}: no measurements")


def check_submission(device, sid, errors):
    d = os.path.join(RESULTS, device, sid)
    rel = os.path.relpath(d, ROOT)
    try:
        info = json.load(open(os.path.join(d, "submission.json")))
    except (OSError, ValueError) as e:
        errors.append(f"{rel}/submission.json: {e}")
        return
    if not (ID.match(sid) or sid == "legacy") or info.get("id") != sid:
        errors.append(f"{rel}: the folder must be named after the submission id (like 20260930-27b6c2)")
    dev = info.get("device") or {}
    if not isinstance(dev.get("name"), str) or not isinstance(dev.get("gpu_cores"), int):
        errors.append(f"{rel}: submission.json lacks the device name and GPU core count")
    elif sub.device_slug(dev["name"], dev["gpu_cores"]) != device:
        errors.append(f"{rel}: device folder should be {sub.device_slug(dev['name'], dev['gpu_cores'])}")
    if sid != "legacy" and not (info.get("machine") or {}).get("model_identifier"):
        errors.append(f"{rel}: submission.json lacks the machine model; run it again with the "
                      f"current tuning/run.py")
    if info.get("quick"):
        errors.append(f"{rel}: a smoke test (--quick) is not a submission")
    if info.get("status") not in (None, "complete"):
        errors.append(f"{rel}: the run did not finish (status {info.get('status')!r})")
    if not isinstance(info.get("results"), dict) or not info["results"]:
        errors.append(f"{rel}: submission.json has no results")
        return

    for name in os.listdir(d):
        path = os.path.join(d, name)
        if os.path.isdir(path):
            if name not in OPS:
                errors.append(f"{rel}/{name}: unexpected folder")
                continue
            for f in os.listdir(path):
                if f not in OP_FILES:
                    errors.append(f"{rel}/{name}/{f}: unexpected file")
                elif os.path.getsize(os.path.join(path, f)) > MAX_FILE:
                    errors.append(f"{rel}/{name}/{f}: larger than 5 MB")
        elif name not in TOP_FILES:
            errors.append(f"{rel}/{name}: unexpected file")
        elif os.path.getsize(path) > MAX_FILE:
            errors.append(f"{rel}/{name}: larger than 5 MB")

    for op in info["results"]:
        if op not in OPS:
            errors.append(f"{rel}: unknown decomposition {op!r} in submission.json")
            continue
        raw = os.path.join(d, op, "raw.csv")
        if not os.path.exists(raw):
            errors.append(f"{rel}/{op}/raw.csv is missing")
        else:
            check_raw(raw, op, errors)


def main():
    errors, count = [], 0
    for device in sorted(os.listdir(RESULTS)):
        ddir = os.path.join(RESULTS, device)
        if not os.path.isdir(ddir):
            errors.append(f"docs/results/{device}: unexpected file")
            continue
        for sid in sorted(os.listdir(ddir)):
            if sid == "combined":
                continue
            if not os.path.isdir(os.path.join(ddir, sid)):
                errors.append(f"docs/results/{device}/{sid}: unexpected file")
                continue
            check_submission(device, sid, errors)
            count += 1
    for e in errors:
        print(f"error: {e}")
    print(f"{count} submissions checked, {len(errors)} problems")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
