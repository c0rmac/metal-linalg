#!/usr/bin/env python3
"""Measure this Mac for metal-linalg: every decomposition, one command.

    python3 tuning/run.py              # about 50 minutes on an M5 Pro; leave the Mac alone
    python3 tuning/run.py --quick      # a 15-minute smoke test, not a submission
    python3 tuning/run.py --only qr    # one decomposition (qr ~5 min, eigh ~12, svd ~22, cholesky ~3, lu ~5, trsm ~3)

Checks that the Mac is fit to measure, builds the tools, runs the correctness
tests, then the QR, eigensolver, SVD, Cholesky, LU and triangular solve sweeps in turn,
and writes everything to one new submission:

    docs/results/<device>/<id>/     e.g. docs/results/apple-m5-pro-20gpu/20260930-27b6c2/

<id> is the date and a random suffix, so any number of people with the same
Mac can submit without colliding. What it records: the chip, its core counts
and memory, the macOS and MLX versions, the load average and power source,
and the timings. Nothing that identifies you or the machine.

Requires: Apple Silicon, `brew install mlx cmake`, Python 3.8 or later.
See docs/tuning.md.
"""

import argparse
import base64
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import submissions as sub   # noqa: E402
import kernels             # noqa: E402
import tune_eigh as te      # noqa: E402  machine state checks

# (decomposition, harness, sweep binary, options, options for --quick)
SWEEPS = [
    ("qr",   "tune_qr.py",   "sweep_qr",   [], []),
    ("eigh", "tune_eigh.py", "sweep_eigh", ["--max-n", "4096"], ["--quick"]),
    ("svd",  "tune_svd.py",  "sweep_svd",  ["--max-k", "4096"], ["--quick"]),
    ("cholesky", "tune_cholesky.py", "sweep_cholesky", ["--max-n", "4096"], ["--quick"]),
    ("lu",   "tune_lu.py",   "sweep_lu",   ["--max-n", "4096"], ["--quick"]),
    ("trsm", "tune_trsm.py", "sweep_trsm", ["--max-n", "4096"], ["--quick"]),
]
TESTS = ["test_qr", "test_eigh", "test_svd", "test_cholesky", "test_lu", "test_trsm"]
REPO_URL = "https://github.com/c0rmac/metal-linalg"


def library_version():
    """The version in CMakeLists.txt's project(), which every release sets."""
    import re
    m = re.search(r"project\(\s*\w+\s+VERSION\s+(\d+\.\d+\.\d+)", open(os.path.join(ROOT, "CMakeLists.txt")).read())
    return m.group(1) if m else None


def fail(msg):
    print(f"\n{msg}", file=sys.stderr)
    sys.exit(1)


def say(msg=""):
    print(msg, flush=True)


def capture(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def find_tools():
    """cmake and the Homebrew prefix MLX lives under."""
    cmake = shutil.which("cmake") or next(
        (p for p in ("/opt/homebrew/bin/cmake", "/usr/local/bin/cmake") if os.path.exists(p)), None)
    if not cmake:
        fail("CMake is not installed. Install it with:  brew install cmake")
    prefix = "/opt/homebrew"
    if shutil.which("brew"):
        rc, out = capture(["brew", "--prefix"])
        if rc == 0 and out.strip():
            prefix = out.strip().splitlines()[-1]
    if not os.path.isdir(os.path.join(prefix, "share", "cmake", "MLX")):
        fail("MLX is not installed. Install it with:  brew install mlx")
    return cmake, prefix


def check_machine(anyway):
    st = te.machine_state()
    problems = te.state_problems(st)
    if st.get("power") == "battery":
        problems.append("running on battery")
    if problems and not anyway:
        fail("This Mac is not ready to measure: " + "; ".join(problems) + ".\n"
             "Plug into power, turn Low Power Mode off, quit other apps, and run this again.\n"
             "(--anyway runs regardless, but the results will be marked untrustworthy.)")
    return st


def build(cmake, prefix, build_dir):
    say("Building the measurement tools ...")
    steps = [[cmake, "-S", ROOT, "-B", build_dir, "-DCMAKE_BUILD_TYPE=Release",
              f"-DCMAKE_PREFIX_PATH={prefix}", "-DMETAL_LINALG_BUILD_EXAMPLES=OFF"],
             [cmake, "--build", build_dir, "-j", "--target"] + [s[2] for s in SWEEPS] + TESTS]
    for cmd in steps:
        rc, out = capture(cmd)
        if rc != 0:
            fail("The build failed:\n" + "\n".join(out.splitlines()[-25:]))


def run_tests(build_dir):
    say("Checking correctness first ...")
    for t in TESTS:
        rc, out = capture([os.path.join(build_dir, t)])
        last = [l for l in out.splitlines() if "checks" in l]
        if rc != 0:
            log = os.path.join(build_dir, f"{t}.log")
            open(log, "w").write(out)
            fail(f"{t} failed on this Mac, so its timings would mean nothing. Please open an issue at\n"
                 f"{REPO_URL}/issues with the output in {log}.")
        say(f"  {t}: {last[-1].strip() if last else 'passed'}")


def spec(build_dir):
    """The Mac, as the library sees it plus a few system facts."""
    rc, out = capture([os.path.join(build_dir, "sweep_qr"), "--policy"])
    if rc != 0:
        fail("Could not read the device from sweep_qr:\n" + out)
    # The JSON line, wherever a notice on stderr put it.
    pol = json.loads(next(l for l in reversed(out.strip().splitlines()) if l.startswith("{")))

    def sysctl(name):
        rc, v = capture(["sysctl", "-n", name])
        return int(v) if rc == 0 and v.strip().isdigit() else None

    levels = [sysctl(f"hw.perflevel{i}.physicalcpu") for i in range(3)]
    mem = sysctl("hw.memsize")
    rc, macos = capture(["sw_vers", "-productVersion"])
    mlx = None
    if shutil.which("brew"):
        rc2, v = capture(["brew", "list", "--versions", "mlx"])
        mlx = v.split()[-1] if rc2 == 0 and v.split() else None
    rc3, commit = capture(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"])
    rc4, dirty = capture(["git", "-C", ROOT, "status", "--porcelain", "--untracked-files=no"])
    return {
        "device": {"name": pol["device"], "gpu_cores": pol["gpu_cores"],
                   "slug": sub.device_slug(pol["device"], pol["gpu_cores"])},
        "machine": machine(),
        "cpu": {"cores": sysctl("hw.ncpu"), "per_level": [n for n in levels if n]},
        "memory_gb": round(mem / 2**30) if mem else None,
        "macos": macos.strip() if rc == 0 else None,
        "macos_build": capture(["sw_vers", "-buildVersion"])[1].strip() or None,
        "mlx": mlx,
        "metal_linalg": (commit.strip() + ("+changes" if dirty.strip() else "")) if rc3 == 0 else None,
    }


def system_profiler(kind):
    rc, out = capture(["system_profiler", kind, "-json"])
    try:
        return json.loads(out).get(kind, []) if rc == 0 else []
    except ValueError:
        return []


def machine():
    """The exact Mac, e.g. "MacBook Pro (16-inch, M5 Pro)", Mac17,8: one chip
    ships in machines that cool it very differently, and the chip name alone
    cannot tell them apart. Only model-level facts; never serial numbers."""
    rc, ident = capture(["sysctl", "-n", "hw.model"])
    rc2, tree = capture(["ioreg", "-arc", "IOPlatformDevice", "-k", "product-name"])
    product = None
    if rc2 == 0:
        m = re.search(r"<key>product-name</key>\s*<data>\s*([A-Za-z0-9+/=\s]+?)\s*</data>", tree)
        if m:
            product = base64.b64decode(m.group(1)).decode("utf-8", "replace").strip("\0 ")
    display = None
    for gpu in system_profiler("SPDisplaysDataType"):
        for d in gpu.get("spdisplays_ndrvs", []):
            if d.get("spdisplays_connection_type") == "spdisplays_internal":
                display = d.get("_spdisplays_pixels")
    return {"product_name": product, "model_identifier": ident.strip() if rc == 0 else None,
            "built_in_display": display}


def conditions():
    """The machine state (load, power, Low Power Mode) plus what bears on
    throttling: the charger, the battery, macOS's power mode and its thermal
    and performance warnings."""
    st = te.machine_state()
    rc, pm = capture(["pmset", "-g"])
    m = re.search(r"^\s*powermode\s+(\d)", pm, re.M)
    st["power_mode"] = {"0": "automatic", "1": "low power", "2": "high power"}.get(m.group(1)) if m else None
    rc, therm = capture(["pmset", "-g", "therm"])
    st["thermal"] = [l.strip() for l in therm.splitlines() if l.strip()] if rc == 0 else None
    for e in system_profiler("SPPowerDataType"):
        if "sppower_ac_charger_watts" in e:
            st["charger_watts"] = int(e["sppower_ac_charger_watts"])
            st["charger"] = e.get("sppower_ac_charger_name")
        if "sppower_battery_charge_info" in e:
            st["battery_percent"] = e["sppower_battery_charge_info"].get("sppower_battery_state_of_charge")
    return st


def run_sweep(op, harness, binary, options, out_dir, log_path):
    """Runs one harness, showing its progress and keeping a log."""
    cmd = [sys.executable, os.path.join(HERE, harness), binary, "--out", out_dir] + options
    with open(log_path, "w") as log:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in p.stdout:
            log.write(line)
            if line.strip():
                print(f"  [{op}] {line.rstrip()}", flush=True)
        return p.wait()


def result_of(op, out_dir):
    path = os.path.join(out_dir, "results.json")
    if not os.path.exists(path):
        return {"trustworthy": False, "row": None, "why": "no results"}
    r = json.load(open(path))
    row = r.get("ktuned_entry") or r.get("tuned_row")
    drift = (r.get("drift") or {}).get("ratio")
    if op == "qr":   # judged by its noise floor
        ok = r["noise"]["overall"]["median"] <= 1.10
        why = None if ok else "run-to-run noise above 10%"
    else:
        ok = bool(r.get("trustworthy"))
        why = None if ok else "; ".join(r.get("warnings", [])[:1])
    return {"trustworthy": ok, "row": row, "why": why, "probe_drift": drift}


def write_summary(path, info):
    d = info["device"]
    mach = info.get("machine") or {}
    L = [f"# {d['name']}, {d['gpu_cores']} GPU cores — submission {info['id']}", "",
         f"{mach.get('product_name') or 'unknown model'} ({mach.get('model_identifier')}), "
         f"{info['cpu']['cores']} CPU cores, {info['memory_gb']} GB, macOS {info['macos']}, "
         f"MLX {info['mlx']}, metal-linalg {info['metal_linalg']}. "
         f"{'Smoke test (--quick), not a submission. ' if info['quick'] else ''}"
         f"Measured {info['date']}.", "",
         "| decomposition | trustworthy | row for `kTuned[]` | minutes | report |",
         "|---|---|---|---|---|"]
    for op, r in info["results"].items():
        L.append(f"| {op} | {'yes' if r['trustworthy'] else 'no: ' + (r.get('why') or '')} | "
                 f"`{r['row']}` | {info['minutes'].get(op, '')} | [{op}/report.md]({op}/report.md) |")
    open(path, "w").write("\n".join(L) + "\n")


def main():
    # The tools this starts would each say the Mac's calibration is out of
    # date, which is why it is being measured.
    os.environ["METAL_LINALG_NO_CALIBRATION_NOTICE"] = "1"
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true",
                    help="a short smoke test of the whole pipeline; not for submitting")
    ap.add_argument("--only", metavar="OPS",
                    help="measure only these decompositions, comma-separated (qr, eigh, svd, cholesky, lu, trsm): to "
                         "remeasure one after its routing or kernels change; the others keep the "
                         "settings fitted from earlier runs")
    ap.add_argument("--anyway", action="store_true",
                    help="measure even if the Mac is busy or on battery (results marked untrustworthy)")
    ap.add_argument("--full-grid", action="store_true",
                    help="eigh and SVD: time the Jacobi backends at every size, not only where they can win (a "
                         "report says when that is worth doing)")
    ap.add_argument("--full-passes", action="store_true",
                    help="eigh and SVD: repeat every point in every pass, not only those without a clear winner")
    args = ap.parse_args()
    known = [op for op, *_ in SWEEPS]
    only = known if not args.only else [op.strip() for op in args.only.split(",") if op.strip()]
    if not only or any(op not in known for op in only):
        fail(f"--only takes a comma-separated list of {', '.join(known)}; got {args.only!r}.")
    sweeps = [s for s in SWEEPS if s[0] in only]

    if sys.platform != "darwin" or platform.machine() != "arm64":
        fail("metal-linalg measures Apple Silicon Macs; this is not one.")
    cmake, prefix = find_tools()
    check_machine(args.anyway)

    build_dir = os.path.join(ROOT, "build-tuning")
    build(cmake, prefix, build_dir)
    run_tests(build_dir)

    info = spec(build_dir)
    info["id"] = sub.new_id()
    info["date"] = time.strftime("%Y-%m-%d", time.gmtime())
    info["quick"] = args.quick
    info["epoch"] = sub.EPOCH
    info["epochs"] = dict(kernels.KERNEL_EPOCHS)   # what each decomposition is measured at
    info["library_version"] = library_version()
    slug = info["device"]["slug"]
    base = os.path.join(build_dir, "quick") if args.quick else os.path.join(ROOT, "docs", "results")
    out = os.path.join(base, slug, info["id"])
    os.makedirs(out)
    info = {"id": info.pop("id"), "date": info.pop("date"), **info}
    info.update({"status": "running", "conditions": {"start": conditions()}, "results": {}, "minutes": {}})
    json.dump(info, open(os.path.join(out, "submission.json"), "w"), indent=1)

    minutes = {"qr": (2, 5), "eigh": (8, 12), "svd": (8, 22), "cholesky": (1, 3), "lu": (1, 5), "trsm": (1, 3)}   # an M5 Pro's, 2.18.0
    total = str(sum(minutes[op][0 if args.quick else 1] for op in only))
    info["memory_budget_gb"] = round(sub.memory_budget_bytes() / 2 ** 30, 1)
    say(f"\nMemory: shapes are capped to {info['memory_budget_gb']} GB at peak "
        f"({int(sub.MEMORY_FRACTION * 100)}% of this Mac's RAM), so nothing swaps.")
    say(f"\nMeasuring {info['device']['name']} ({info['device']['gpu_cores']} GPU cores): about {total} "
        f"minutes. Leave the Mac alone until it finishes.\nWriting to {os.path.relpath(out, ROOT)}/\n")
    for op, harness, binary, options, quick_options in sweeps:
        say(f"--- {op} ---")
        t0 = time.time()
        extra = ([] if op in ("qr", "cholesky", "lu", "trsm") else
                 (["--full-grid"] if args.full_grid else []) + (["--full-passes"] if args.full_passes else []))
        rc = run_sweep(op, harness, os.path.join(build_dir, binary),
                       (quick_options if args.quick else options) + extra,
                       os.path.join(out, op), os.path.join(out, f"{op}.log"))
        info["minutes"][op] = round((time.time() - t0) / 60, 1)
        info["results"][op] = result_of(op, os.path.join(out, op))
        info["conditions"][f"after_{op}"] = conditions()
        if rc != 0:
            info["results"][op].update(trustworthy=False, why=f"the harness exited with status {rc}")
        json.dump(info, open(os.path.join(out, "submission.json"), "w"), indent=1)

    info["status"] = "complete"
    json.dump(info, open(os.path.join(out, "submission.json"), "w"), indent=1)
    write_summary(os.path.join(out, "summary.md"), info)

    rel = os.path.relpath(out, ROOT)
    say("\n" + "=" * 72)
    for op, r in info["results"].items():
        say(f"{op:5s} {'ok ' if r['trustworthy'] else 'NOT trustworthy: ' + (r.get('why') or '')}"
            f"  {r['row'] or ''}")
    say(f"\nEverything is in {rel}/ (summary.md first).")
    if args.quick:
        say("This was a smoke test. Run without --quick to measure for real.")
        return
    if not all(r["trustworthy"] for r in info["results"].values()):
        say("Some measurements are not trustworthy (see above); they will not be used. "
            "Please run again with the Mac idle and on power.")
        return
    d = info["device"]
    say(f"""
Thank you! To contribute these results, open a pull request that adds the folder:

  gh repo fork --remote                     # once; needs `brew install gh` and `gh auth login`
  git switch -c results/{slug}-{info['id']}
  git add {rel}
  git commit -m "Results: {d['name']}, {d['gpu_cores']} GPU cores ({info['id']})"
  git push -u origin HEAD
  gh pr create --fill --repo {REPO_URL.split("github.com/")[1]}

CONTRIBUTING.md shows the same with git alone. Or zip {rel} and attach it to
a new issue at {REPO_URL}/issues/new

Once it is merged, the library's settings for this Mac are recomputed from every
run submitted for it, and updated automatically.""")


if __name__ == "__main__":
    main()
