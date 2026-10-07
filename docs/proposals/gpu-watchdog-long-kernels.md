# Long single-threadgroup kernels and the GPU's watchdog

Status: proposal, not started (2026-10-07).

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
