#!/usr/bin/env python3
"""Checks the built metal-linalg-torch wheel before it is published:

    python3 python-torch/check_wheel.py dist-torch/*.whl

The wheel is tagged py3-none (it holds no Python extension, only a C library
loaded with ctypes), so it must contain exactly one libmetal_linalg.dylib, for
arm64, linking nothing but system libraries (in particular no MLX and no
torch), built for no newer macOS than its tag says, plus the Python sources.
"""

import re
import subprocess
import sys
import tempfile
import zipfile

EXPECTED = {"__init__.py", "_build.py", "_lib.py", "_ops.py", "libmetal_linalg.dylib"}


def otool(*args):
    return subprocess.run(["otool", *args], capture_output=True, text=True, check=True).stdout


def check(wheel):
    errors = []
    tag = re.search(r"-py3-none-macosx_(\d+)_(\d+)_(\w+)\.whl$", wheel)
    if not tag or tag.group(3) != "arm64":
        return ["not a py3-none arm64 macOS wheel tag"]
    with tempfile.TemporaryDirectory() as tmp, zipfile.ZipFile(wheel) as z:
        files = {n.split("/", 1)[1] for n in z.namelist() if n.startswith("metal_linalg_torch/")}
        if files != EXPECTED:
            errors.append(f"package files are {sorted(files)}, expected {sorted(EXPECTED)}")
        if "libmetal_linalg.dylib" not in files:
            return errors
        lib = z.extract("metal_linalg_torch/libmetal_linalg.dylib", tmp)
        for dep in [line.split()[0] for line in otool("-L", lib).splitlines()[1:]]:
            if not (dep.startswith(("/System/Library/", "/usr/lib/")) or dep == "@rpath/libmetal_linalg.dylib"):
                errors.append(f"links {dep}")
        load = otool("-l", lib)
        minos = re.search(r"cmd LC_BUILD_VERSION\n(?:.*\n)*?\s+minos (\d+)\.(\d+)", load)
        if not minos or (int(minos.group(1)), int(minos.group(2))) > (int(tag.group(1)), int(tag.group(2))):
            errors.append(f"built for macOS {minos and minos.group(1)} but tagged {tag.group(1)}.{tag.group(2)}")
        archs = subprocess.run(["lipo", "-archs", lib], capture_output=True, text=True).stdout.split()
        if archs != ["arm64"]:
            errors.append(f"architectures {archs}")
        syms = subprocess.run(["nm", "-gU", lib], capture_output=True, text=True).stdout
        for sym in ("_metal_linalg_qr", "_metal_linalg_eigh", "_metal_linalg_svd", "_metal_linalg_cholesky",
                    "_metal_linalg_lu_factor", "_metal_linalg_solve", "_metal_linalg_inv",
                    "_metal_linalg_solve_triangular",
                    "_metal_linalg_calibration_message",
                    "_metal_linalg_buffer_contents"):
            if sym not in syms.split():
                errors.append(f"does not export {sym}")
    return errors


failed = False
for wheel in sys.argv[1:]:
    errors = check(wheel)
    print(f"{'ok ' if not errors else 'BAD'} {wheel}")
    for e in errors:
        print(f"      {e}")
    failed |= bool(errors)
sys.exit(1 if failed else 0)
