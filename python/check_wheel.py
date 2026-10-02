#!/usr/bin/env python3
"""Checks built wheels before they are published:

    python3 python/check_wheel.py dist/*.whl

The usual repair step (delocate) cannot be used: it would copy libmlx.dylib
into the wheel, and a second MLX in the process breaks mx.array exchange with
the `mlx` package. So instead this checks that the extension is arm64, links
nothing but system libraries and the libmlx of the `mlx` package beside it,
finds that one through @loader_path alone (no path of the build machine), and
targets no newer macOS than its tag says.
"""

import re
import subprocess
import sys
import tempfile
import zipfile


def otool(*args):
    return subprocess.run(["otool", *args], capture_output=True, text=True, check=True).stdout


def check(wheel):
    errors = []
    tag = re.search(r"macosx_(\d+)_(\d+)_(\w+)\.whl$", wheel)
    if not tag or tag.group(3) != "arm64":
        return [f"not an arm64 macOS wheel tag"]
    with tempfile.TemporaryDirectory() as tmp, zipfile.ZipFile(wheel) as z:
        sos = [n for n in z.namelist() if n.endswith(".so")]
        if len(sos) != 1:
            return [f"expected one extension module, found {sos}"]
        so = z.extract(sos[0], tmp)
        libs = [line.split()[0] for line in otool("-L", so).splitlines()[1:]]
        for lib in libs:
            if not (lib.startswith(("/System/Library/", "/usr/lib/")) or lib == "@rpath/libmlx.dylib"):
                errors.append(f"links {lib}")
        if "@rpath/libmlx.dylib" not in libs:
            errors.append("does not link libmlx.dylib")
        load = otool("-l", so)
        rpaths = re.findall(r"cmd LC_RPATH\n\s+cmdsize \d+\n\s+path (\S+)", load)
        if rpaths != ["@loader_path/../mlx/lib"]:
            errors.append(f"rpaths are {rpaths}, not just @loader_path/../mlx/lib")
        minos = re.search(r"cmd LC_BUILD_VERSION\n(?:.*\n)*?\s+minos (\d+)\.(\d+)", load)
        if not minos or (int(minos.group(1)), int(minos.group(2))) > (int(tag.group(1)), int(tag.group(2))):
            errors.append(f"built for macOS {minos and minos.group(1)} but tagged {tag.group(1)}.{tag.group(2)}")
        archs = subprocess.run(["lipo", "-archs", so], capture_output=True, text=True).stdout.split()
        if archs != ["arm64"]:
            errors.append(f"architectures {archs}")
    return errors


failed = False
for wheel in sys.argv[1:]:
    errors = check(wheel)
    print(f"{'ok ' if not errors else 'BAD'} {wheel}")
    for e in errors:
        print(f"      {e}")
    failed |= bool(errors)
sys.exit(1 if failed else 0)
