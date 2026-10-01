#!/usr/bin/env python3
"""Release automation, run by .github/workflows/release.yml.

    release.py plan                    -> skip=, version=, bump= (for $GITHUB_OUTPUT)
    release.py set-version VERSION     write VERSION into the version files
    release.py notes VERSION           CHANGELOG.md's section for VERSION, if any
    release.py tap VERSION TAP_REPO    point the tap's formulas at the release

Configuration, from the environment:

    RELEASE_VERSION_FILES  files that carry the version (default CMakeLists.txt);
                           CMakeLists.txt's project() is the one read
    RELEASE_IGNORE         paths whose changes alone release nothing
                           ("docs/" a directory, "*.md" a suffix)
    RELEASE_DEFER          on a push, paths whose change is released by a later
                           run instead (one triggered by the workflow that
                           processes them)
    HOMEBREW_TAP_TOKEN     a token that can push to TAP_REPO; without it the
                           tap is left alone and the job summary says what to
                           change by hand

Needs Python 3.8+ and git; nothing else.
"""

import hashlib
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.request

SEMVER = re.compile(r"\d+\.\d+\.\d+")
PROJECT_VERSION = re.compile(r"(project\(\s*\w+\s+VERSION\s+)(\d+\.\d+\.\d+)")


def git(*args, cwd=None):
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True,
                          cwd=cwd).stdout.strip()


def parse(v):
    return tuple(int(x) for x in v.split("."))


def released():
    """Every vX.Y.Z tag, oldest first."""
    tags = git("tag", "--list", "v*").split()
    return sorted((t[1:] for t in tags if SEMVER.fullmatch(t[1:])), key=parse)


def file_version():
    m = PROJECT_VERSION.search(open("CMakeLists.txt").read())
    if not m:
        sys.exit("release.py: no project(... VERSION x.y.z) in CMakeLists.txt")
    return m.group(2)


def matches(path, patterns):
    for p in patterns:
        if p.endswith("/") and path.startswith(p):
            return True
        if p.startswith("*") and path.endswith(p[1:]):
            return True
        if path == p:
            return True
    return False


def summary(text):
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if path:
        with open(path, "a") as fh:
            fh.write(text + "\n")
    print(text, file=sys.stderr)


def plan():
    ignore = os.environ.get("RELEASE_IGNORE", "").split()
    defer = os.environ.get("RELEASE_DEFER", "").split()

    def skip(why):
        summary(f"No release: {why}.")
        print("skip=true")

    if os.environ.get("GITHUB_EVENT_NAME") == "push" and defer:
        before, after = os.environ.get("PUSH_BEFORE", ""), os.environ.get("PUSH_AFTER", "HEAD")
        if before and before.strip("0"):
            changed = git("diff", "--name-only", before, after).splitlines()
            if any(matches(f, defer) for f in changed):
                return skip("this push changes " + " or ".join(defer) +
                            ", so it is released when the workflow that processes those has run")

    tags = released()
    last = tags[-1] if tags else None
    if last:
        changed = [f for f in git("diff", "--name-only", f"v{last}", "HEAD").splitlines()
                   if not matches(f, ignore)]
        if not changed:
            return skip(f"nothing but {' '.join(ignore)} has changed since v{last}")

    current = file_version()
    if last is None or parse(current) > parse(last):
        version = current                 # a version set by hand, not yet released
    else:
        major, minor, patch = parse(last)
        version = f"{major}.{minor}.{patch + 1}"
    print("skip=false")
    print(f"version={version}")
    print(f"bump={'true' if version != current else 'false'}")


def set_version(version):
    files = os.environ.get("RELEASE_VERSION_FILES", "CMakeLists.txt").split()
    for path in files:
        s = open(path).read()
        if path.endswith("CMakeLists.txt"):
            s, n = PROJECT_VERSION.subn(lambda m: m.group(1) + version, s, count=1)
        elif path.endswith("pyproject.toml"):
            s, n = re.subn(r'(?m)^(version\s*=\s*")[^"]*(")', lambda m: m.group(1) + version + m.group(2),
                           s, count=1)
        else:
            sys.exit(f"release.py: don't know how to set the version in {path}")
        if n != 1:
            sys.exit(f"release.py: no version found in {path}")
        open(path, "w").write(s)


def notes(version):
    """The CHANGELOG.md section headed with this version, without its heading."""
    if not os.path.exists("CHANGELOG.md"):
        return
    lines, out, inside = open("CHANGELOG.md").read().splitlines(), [], False
    for line in lines:
        if line.startswith("## "):
            if inside:
                break
            inside = re.match(rf"## v?{re.escape(version)}\b", line) is not None
            continue
        if inside:
            out.append(line)
    text = "\n".join(out).strip()
    if text:
        print(text)


def tarball_sha256(url):
    for attempt in range(6):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return hashlib.sha256(r.read()).hexdigest()
        except Exception as e:  # GitHub makes the archive on first request
            if attempt == 5:
                raise
            print(f"fetching {url}: {e}; retrying", file=sys.stderr)
            time.sleep(10)


def update_formula(text, repo, url, sha):
    """`text` with its stable source pointed at url/sha, or None if it is not this repo's."""
    if f"github.com/{repo}" not in text:
        return None
    stable = re.compile(r'(?m)^(\s*)url "https://github\.com/' + re.escape(repo) +
                        r'/archive/refs/tags/[^"]+"\s*\n\s*sha256 "[0-9a-f]{64}"')
    if stable.search(text):
        return stable.sub(lambda m: f'{m.group(1)}url "{url}"\n{m.group(1)}sha256 "{sha}"', text, count=1)
    head = re.compile(r'(?m)^(\s*)head "https://github\.com/' + re.escape(repo) + r'\.git"')
    if head.search(text):    # head-only until now: add the stable source above it
        return head.sub(lambda m: f'{m.group(1)}url "{url}"\n{m.group(1)}sha256 "{sha}"\n{m.group(0)}',
                        text, count=1)
    return None


def tap(version, tap_repo):
    repo = os.environ["GITHUB_REPOSITORY"]
    url = f"https://github.com/{repo}/archive/refs/tags/v{version}.tar.gz"
    sha = tarball_sha256(url)
    token = os.environ.get("HOMEBREW_TAP_TOKEN", "")
    if not token:
        summary(f"""Released v{version}. The Homebrew formulas in `{tap_repo}` were **not** updated:
this repository has no `HOMEBREW_TAP_TOKEN` secret (see CONTRIBUTING.md, "Releases").
To update them by hand, set in each formula that builds this repository:

```ruby
url "{url}"
sha256 "{sha}"
```""")
        print("::warning::HOMEBREW_TAP_TOKEN is not set; the tap was not updated (see the job summary)")
        return

    work = tempfile.mkdtemp()
    git("clone", "--depth", "1", f"https://x-access-token:{token}@github.com/{tap_repo}.git", work)
    changed = []
    formula_dir = os.path.join(work, "Formula")
    for name in sorted(os.listdir(formula_dir)):
        path = os.path.join(formula_dir, name)
        if not name.endswith(".rb"):
            continue
        old = open(path).read()
        new = update_formula(old, repo, url, sha)
        if new is not None and new != old:
            open(path, "w").write(new)
            changed.append(name)
    if not changed:
        summary(f"Released v{version}; no formula in `{tap_repo}` needed changing.")
        return
    git("config", "user.name", "github-actions[bot]", cwd=work)
    git("config", "user.email", "41898282+github-actions[bot]@users.noreply.github.com", cwd=work)
    git("add", "Formula", cwd=work)
    git("commit", "-m", f"{repo.split('/')[1]} v{version}", cwd=work)
    git("push", "origin", "HEAD", cwd=work)
    summary(f"Released v{version} and pointed {', '.join(changed)} in `{tap_repo}` at it "
            f"(sha256 `{sha}`).")


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "plan":
        plan()
    elif cmd == "set-version" and len(args) == 1:
        set_version(args[0])
    elif cmd == "notes" and len(args) == 1:
        notes(args[0])
    elif cmd == "tap" and len(args) == 2:
        tap(*args)
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
