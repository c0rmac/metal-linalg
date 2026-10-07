# Long single-threadgroup kernels and the GPU's watchdog

Status: **done** in 2.15.0 (2026-10-07); see [Done](#done-2026-10-07).

## What

The whole-matrix Jacobi kernels give a matrix one threadgroup for its whole
solve. On 2026-10-07, while the display was busy (apps redrawing), macOS
ended such command buffers after about a quarter of a second with
`kIOGPUCommandBufferCallbackErrorHang` ("GPU Hang Error"): eigh in
threadgroup mode from N = 448 (384 took 204 ms and passed), and the SVD's
Jacobi kernel at 512 x 512. The same calls had passed that morning with the
machine idle, and the 2.14.1 build failed the same way, so it is the
system's, not a regression. The library threw, as it does for a GPU error;
`test_eigh` and `test_svd` stopped there.

The routing does not send those sizes to those kernels on the M5 Pro (block
Jacobi from N = 96; the SVD's Jacobi kernel up to k = 56), but forced modes
(`EighOptions::Mode::threadgroup`, `EIGH_DEVICE=gpu`), estimated policies on
slower Macs, and the test suites do.

## Plan

1. Measure each single-threadgroup kernel's time at the largest N its
   routing window or a forced mode allows (threadgroup mode up to its
   threadgroup-memory limit, the SVD's Jacobi kernel, `ql`, `golub_kahan`),
   idle and with the display busy.
2. Where one can run longer than about 100 ms, split it: the sweeps of a
   Jacobi solve as separate dispatches with the matrix in device memory
   between them (the kernels already loop over sweeps), or cap the forced
   mode's N where splitting does not fit the kernel.
3. Tests: run the large single-threadgroup cases in their own command
   buffers per sweep, or skip them above the cap with a message, so that a
   busy display cannot fail the suite.

## Effort

About a day, more if a kernel has to be restructured to keep its state
between dispatches.

## Expected gain

None in speed: calls that now fail on a busy Mac would work. Errors of this
kind in the routing sweeps would otherwise read as backends that cannot run
(the gate marks them failed), which would bias a run measured while the Mac
is in use.

## Where to start

`src/eigh.mm` (`eigh_jacobi`, whose error message already mentions the
watchdog), `src/svd.mm` and the Jacobi kernels' dispatch; `tests/test_eigh.cpp`
and `tests/test_svd.cpp`'s size lists.

## Done (2026-10-07)

**Measured** (M5 Pro, one matrix, the kernel's own time):

| single-threadgroup kernel | 256 | 384 | 448 | 512 | 768 | 1024 |
|---|---|---|---|---|---|---|
| eigh, threadgroup mode | 55 ms | 203 | 328 | 493 | 1708 | |
| the SVD's Jacobi kernel | 34 | 113 | | 263 | 1031 | 2827 |

`ql` and `golub_kahan` are bounded by threadgroup memory (k up to 80 or so)
and run in milliseconds; the block Jacobi kernels spread a matrix over many
threadgroups and dispatches; `bd_chase_apply` (the band SVD with vectors)
gives a threadgroup about 20 ms at 4096 and, by its n^2 per strip, about 80
at 8192. So the two whole-matrix Jacobi kernels were the ones to fix.

**Split, not capped.** Both keep the matrix (W and V; G and V) in device
memory already, with only the round's rotation parameters in threadgroup
memory, so a solve can stop at any round boundary and resume. With
`round_budget` > 0 a dispatch runs at most that many rounds of the
tournament, then saves where it stopped in a 24-byte `JacobiState` a matrix
(`eigh_jacobi_common.h`: the scale, the eigensolver's ||A||_F^2 or the SVD's
null and negligible levels, the sweep, the round, and whether a pair has
rotated yet this sweep) and returns; the next dispatch resumes, a finished
matrix returning at once. The host (`run_split_jacobi` in
`metal_runtime.mm`) gives a solve the cost model puts over 40 ms
(`EIGH_DISPATCH_MS`, `SVD_DISPATCH_MS`) enough rounds for about that much a
dispatch, and command buffers of about 750 ms, until every matrix is done.
The routed sizes (under 40 ms by the model) are not split at all.

Bit for bit the same as the whole solve, at every size tried (eigh 64 to
768, the SVD 64 to 1024, batches, tall), and in the same time (1.00x at
768 and 1024: about 100 dispatches of 15 ms for eigh at 768, 370 of 7 ms for
the SVD at 1024). The test suites pass as they are and with every solve
forced into dispatches of a round or two (`EIGH_DISPATCH_MS=0.05`,
`SVD_DISPATCH_MS=0.05`), the rank-deficient, NaN and scaled cases included.

Not reproduced again: on the afternoon this was done the display was busy
with full-screen video, which did not trip the watchdog (a 2.8 s threadgroup
ran through); the earlier failures came with apps redrawing. The fix bounds
the longest threadgroup run at about 15 ms instead of 1.7-2.8 s.

