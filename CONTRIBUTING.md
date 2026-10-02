# Contributing

The most useful contribution needs no code: **measuring your Mac**. metal-linalg
picks the fastest kernel for each problem, or the CPU, from measurements made
on each kind of Mac, and every new chip needs its own. A Mac nobody has
measured runs a cautious default that misses much of what its GPU can do.
One command measures it, and a pull request sends the results.

Any number of people with the same Mac can contribute: every run is saved
under its own ID, and the runs for a Mac are combined.

- [Measure your Mac](#measure-your-mac)
  - [1. Get the tools and the code](#1-get-the-tools-and-the-code)
  - [2. Run the measurement](#2-run-the-measurement)
  - [3. Send the results as a pull request](#3-send-the-results-as-a-pull-request)
  - [What happens next](#what-happens-next)
- [Other contributions](#other-contributions)
- [Releases](#releases)

## Measure your Mac

### 1. Get the tools and the code

Once:

```bash
xcode-select --install          # Apple's compilers, if you don't have Xcode
brew install mlx cmake gh       # gh, GitHub's command-line tool, sends the results
git clone https://github.com/c0rmac/metal-linalg.git
cd metal-linalg
```

You need an Apple Silicon Mac and, to send the results as a pull request, a
GitHub account. (Without one, see the last option in step 3.)

### 2. Run the measurement

Plug the Mac in, quit other apps, then:

```bash
python3 tuning/run.py
```

That is the whole measurement. It takes about 40 minutes:

- it checks the Mac is ready (on power, not busy, Low Power Mode off), and
  stops with a message saying what to change if not;
- it builds the library and its measuring tools, and runs the correctness
  tests;
- it measures QR, the eigensolver and the SVD, one after another.

Leave the Mac alone until it says it has finished. The results are in a new
folder, `docs/results/<your Mac>/<ID>/`, for example
`docs/results/apple-m5-pro-20gpu/20260930-27b6c2/`; its `summary.md` says what
was found. The last thing the command prints is the commands of step 3 with
your folder's names filled in, ready to copy.

What the folder records is listed in
[docs/tuning.md](docs/tuning.md#what-is-recorded): nothing that identifies you
or your particular Mac. If the command stops with a problem,
[the same page](docs/tuning.md#if-something-goes-wrong) says what to do.

### 3. Send the results as a pull request

You cannot push to this repository directly, so the results go in as a
**pull request**: you put the folder on a branch of your own copy of the
repository (a *fork*), and ask for that branch to be merged. Pick one of the
three ways below.

**With the GitHub CLI** (`gh`, installed in step 1). Sign in once with
`gh auth login`, then, from the `metal-linalg` folder:

```bash
gh repo fork --remote                           # once: your fork becomes `origin`, this repository `upstream`
git switch -c results/<your-mac>-<id>           # a branch for this run
git add docs/results/<your-mac>/<id>
git commit -m "Results: <chip>, <n> GPU cores (<id>)"
git push -u origin HEAD                         # to your fork
gh pr create --fill --repo c0rmac/metal-linalg  # the pull request
```

**With git alone.** Open [the repository](https://github.com/c0rmac/metal-linalg)
on GitHub and click **Fork**. Then, with your GitHub username in the address:

```bash
git remote add fork https://github.com/<your-username>/metal-linalg.git
git switch -c results/<your-mac>-<id>
git add docs/results/<your-mac>/<id>
git commit -m "Results: <chip>, <n> GPU cores (<id>)"
git push -u fork HEAD
```

The push prints a link ("Create a pull request for ..."). Open it, check
that the base repository is `c0rmac/metal-linalg` and the base branch `main`,
and click **Create pull request**. (If git asks for a password, GitHub wants a
[personal access token](https://github.com/settings/tokens), not your
account password; `gh auth login` sets this up for you.)

**Without git.** Zip the `docs/results/<your-mac>/<id>/` folder and attach it
to a [new issue](https://github.com/c0rmac/metal-linalg/issues/new), and a
maintainer will add it.

Add only your results folder: the command writes nothing else that belongs
in the repository (its build goes to `build-tuning/`, which git ignores).

### What happens next

1. **An automatic check runs on the pull request.** It validates your folder
   and shows, in the check's summary, how the library's settings for your Mac
   would change. If it fails, its log says why.
2. **A maintainer reviews and merges it.**
3. **The library updates itself.** After the merge, a GitHub Action
   recomputes the settings for your kind of Mac from every run submitted for
   it, yours included, and commits them. Everyone with that Mac gets them
   with the next update of the library.

If your Mac is already measured, your run still helps: the settings come
from the median of the runs' timings, so each run makes them less sensitive
to one machine's quirks, and runs on different models (a 14-inch and a 16-inch MacBook Pro
cool the same chip differently) show whether they need different settings.

## Other contributions

**Bug reports.** Please open an [issue](https://github.com/c0rmac/metal-linalg/issues)
with what you ran, what happened and what you expected, your Mac and macOS
version, and the output of `./build/sweep_svd --policy` (the device and the
routing in effect).

**Code.** Fork the repository, make your change on a branch, and open a pull
request as in step 3. Before sending it:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/opt/homebrew
cmake --build build -j
ctest --test-dir build --output-on-failure      # every test must pass
```

A change to a shader needs the Metal shader compiler (`xcodebuild
-downloadComponent MetalToolchain`), and `cmake --build build --target
update_prebuilt_shaders` to refresh the compiled copies in
`shaders/prebuilt/`, which the Swift package and builds without the compiler
use. Changing what a kernel does changes the measurements behind the routing;
say so in the pull request, and the maintainers will arrange remeasuring.

## Releases

Releases are automatic. Every update to `main` that changes the library
(anything beyond `docs/`, Markdown files and `.github/`) makes the
[Release](.github/workflows/release.yml) workflow publish the next version:

1. **The version.** The one in `CMakeLists.txt`'s `project()` is released as
   it is if it has no tag yet; otherwise the patch number goes up by one, and
   the workflow commits the new version (`CMakeLists.txt`, `pyproject.toml`)
   to `main`. To start a minor or major version, set it in both files by hand.
2. **The release.** A `vX.Y.Z` tag and a GitHub release, with that version's
   `CHANGELOG.md` section as notes when there is one. GitHub attaches the
   source tarball, which is what Homebrew builds.
3. **Homebrew.** The formula in
   [c0rmac/homebrew-metal-linalg](https://github.com/c0rmac/homebrew-metal-linalg)
   is pointed at the new tarball and its checksum.
4. **PyPI.** [wheels.yml](.github/workflows/wheels.yml) builds the sdist
   and a wheel per Python (3.10 to 3.14) at the new tag, checks and tests
   them, and they are attached to the GitHub release and published to
   [PyPI](https://pypi.org/project/metal-linalg/).

A pull request that adds measurements is released once the Tuned policies
workflow has turned them into tables.

Step 3 needs a token that can push to the tap repository, kept in this
repository as the secret `HOMEBREW_TAP_TOKEN`. To set it up once:

1. On GitHub, **Settings > Developer settings > Personal access tokens >
   Fine-grained tokens > Generate new token**. Give it access to *only* the
   tap repositories (`homebrew-metal-linalg`, and `homebrew-isomorphism` if
   isomorphism uses the same token), with the permission **Contents: Read and
   write**.
2. Store it in this repository (it prompts for the token, so it never
   appears in your shell history):

   ```bash
   gh secret set HOMEBREW_TAP_TOKEN --repo c0rmac/metal-linalg
   ```

Without the secret, releases are still made, and each run's summary gives the
two formula lines to change by hand. A token expires: when it does, generate a
new one and set the secret again.

Step 4 uses PyPI's trusted publishing, so no PyPI token is stored anywhere.
To set it up once, on [pypi.org](https://pypi.org/manage/account/publishing/),
add a (pending) trusted publisher for the project `metal-linalg`: owner
`c0rmac`, repository `metal-linalg`, workflow `release.yml`, environment
`pypi`. If an upload fails, rerun just that step for the release with
**Actions > Release > Run workflow**, `pypi_version` set to its version.

### Following a new MLX

The Python package works only with the MLX release it was built against, so
`pyproject.toml` pins it (the two `MLX_PIN` lines), together with the
nanobind that MLX was built with. When MLX releases, change both `mlx==`
pins; if the build then stops on a nanobind mismatch, it names the internals
version MLX uses, and `nanobind==` goes to the release with that version
(MLX's `CMakeLists.txt` names the nanobind tag it fetches). The release that
follows publishes wheels for the new MLX.
