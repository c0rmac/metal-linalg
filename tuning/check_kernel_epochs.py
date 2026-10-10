#!/usr/bin/env python3
"""Warn when a change may make measurements stale without saying so.

    python3 tuning/check_kernel_epochs.py <base-ref>     # e.g. origin/main

For each decomposition, if the files that can alter its timings changed since
<base-ref> (kernels.PATHS) but its kernel version (kernels.KERNEL_EPOCHS) did
not, prints a GitHub warning annotation and a line for the job summary. It
never fails: whether a change alters timings -- a refactor or a comment does
not -- is for a person to decide. If it does, bump the version in
tuning/kernels.py and add a line to HISTORY; the affected Macs then show as
stale (still used) until they are measured again.
"""

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import kernels   # noqa: E402

NAMES = {"qr": "QR", "eigh": "eigh", "svd": "SVD", "cholesky": "Cholesky"}


def git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout


def epochs_at(ref):
    """KERNEL_EPOCHS as tuning/kernels.py had it at `ref` (all 1 before it existed)."""
    try:
        src = git("show", f"{ref}:tuning/kernels.py")
    except subprocess.CalledProcessError:
        return {op: 1 for op in kernels.KERNEL_EPOCHS}
    scope = {}
    exec(compile(src, "kernels.py", "exec"), scope)
    return scope["KERNEL_EPOCHS"]


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    base = sys.argv[1]
    changed = [p for p in git("diff", "--name-only", f"{base}...HEAD").splitlines() if p]
    before = epochs_at(base)
    lines = []
    for op, name in NAMES.items():
        hits = [p for p in changed if kernels.touches(op, p)]
        if not hits:
            continue
        if op not in before:   # a decomposition the base does not have
            lines.append(f"- {name}: new, at kernel version {kernels.KERNEL_EPOCHS[op]}")
            continue
        bumped = kernels.KERNEL_EPOCHS[op] != before.get(op, 1)
        if bumped:
            lines.append(f"- {name}: kernel version {before.get(op, 1)} -> {kernels.KERNEL_EPOCHS[op]} "
                         f"(changed: {', '.join(hits[:6])}{' ...' if len(hits) > 6 else ''})")
            continue
        msg = (f"{name}'s kernels or what they call changed ({', '.join(hits[:6])}"
               f"{' ...' if len(hits) > 6 else ''}) but its kernel version in tuning/kernels.py did not. "
               f"If this changes {name}'s timings, bump KERNEL_EPOCHS['{op}'] and add a HISTORY line, "
               f"so measured Macs show as stale until remeasured; if not (a refactor, a comment), ignore this.")
        print(f"::warning title={name} measurements may be stale::{msg}")
        lines.append(f"- **{name}: kernel version not bumped.** {msg}")
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    text = "## Kernel versions\n\n" + ("\n".join(lines) if lines else
                                         "No change to what any decomposition's measurements time.") + "\n"
    print(text)
    if summary:
        open(summary, "a").write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
