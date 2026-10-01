# Measuring your Mac

metal-linalg chooses the fastest kernel for each problem, or the CPU, from
measurements made on each kind of Mac. If yours isn't in the table below, or
you'd like to add to its measurements, it takes one command and about 40
minutes of the Mac's time. Any number of people with the same Mac can
contribute: each run is saved under its own ID, and the runs are combined.

| Mac | GPU cores | QR | eigh | SVD | runs |
|---|---|---|---|---|---|
| Apple M1 | 8 | measured | measured | — | 1 |
| Apple M5 Pro | 20 | measured | measured | measured | 1 |

## 1. Install the tools (once)

```bash
xcode-select --install          # Apple's compilers, if you don't have Xcode
brew install mlx cmake
git clone https://github.com/c0rmac/metal-linalg.git
cd metal-linalg
```

## 2. Measure

Plug the Mac in, quit other apps, then:

```bash
python3 tuning/run.py
```

It checks the Mac is ready (on power, not busy, Low Power Mode off) and stops
with a message if not. Then it builds the tools, runs the correctness tests,
and measures QR, the eigensolver and the SVD, one after another. Leave the Mac
alone until it says it has finished.

## 3. Send the results

The results are in a new folder, `docs/results/<your Mac>/<ID>/`, and the
command finishes by printing what to type to send them as a pull request. If
you'd rather not use git, zip that folder and attach it to a
[new issue](https://github.com/c0rmac/metal-linalg/issues/new).

That's all. On the pull request, an automatic check validates your folder and
shows how the library's settings for your Mac would change. Once it is
merged, the settings are recomputed from every run for that Mac, yours
included, and the library is updated automatically.

## What is recorded

The exact model of Mac (for example "MacBook Pro (16-inch, M5 Pro)", since the
same chip runs at different speeds in machines that cool it differently), its
chip, CPU and GPU core counts, memory and built-in display, the macOS and MLX
versions, and, at the start and after each part of the run, the load average,
the power source and charger, the power mode and any thermal warnings. And the
timings. Nothing that identifies you or your particular Mac: no names,
hostnames or serial numbers.

## If something goes wrong

| what you see | what to do |
|---|---|
| "This Mac is not ready to measure" | plug in, turn Low Power Mode off, quit other apps, and run it again |
| "CMake is not installed" / "MLX is not installed" | `brew install cmake mlx` |
| a correctness test failed | please open an issue with the log file it names; don't send timings |
| some results "NOT trustworthy" | something interfered (another app, a thermal change); run it again with the Mac idle. Untrustworthy results are never used |
| you stopped it part-way | delete the incomplete `docs/results/<your Mac>/<ID>/` folder and run it again |

`python3 tuning/run.py --quick` is a 15-minute smoke test of the whole
pipeline. Its results are written to `build-tuning/quick/` and are not for
sending.

---

**Maintainers:** a GitHub Action (`.github/workflows/tuned-policies.yml`)
checks each results pull request, shows the effect on the tables, and after
the merge regenerates `src/tuned/` from every run and commits it. Nothing
needs doing by hand. How the measurements, the combining and the Action work
is in [`tuning-details.md`](tuning-details.md); what every number in a report
means is in [`reading-reports.md`](reading-reports.md).
