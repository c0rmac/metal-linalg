# A one-triangle trailing update for the symmetric band reduction

Status: proposal, not started (2026-10-04).

## What

The eigensolver's `band` backend keeps the whole trailing matrix, both
triangles, and updates all of it each block, because MPS has matrix products
but no symmetric ones (no SYMM, no SYR2K). Store and update one triangle
instead, with kernels of our own, so that each block moves about half the
memory.

## Why: the measurements

Per block of $b$ columns, with $A_{22}$ the $n_1 \times n_1$ trailing matrix
(`band_reduce_symmetric` in `src/band_reduce.mm`):

- $X = A_{22} (V T)$: reads $A_{22}$ once;
- $A_{22} \mathrel{-}= [V\ Y][Y\ V]^T$: reads and writes it once;

so $3 n_1^2$ floats a block, about $4n^3/b$ bytes in all: 17 GB at
$n = 4096$, $b = 16$, 59 ms at 290 GB/s. Profiled on the M5 Pro (see
[the index](README.md) for how), these two products take 73 ms of eigvalsh's
161 ms at 4096; most of the rest is the panels (about 42 ms profiled), the
chase (35) and bisection (6). At 8192 (872 ms in all) the products are most of the time.

They are bound by memory, not arithmetic: the update does $4b = 64$ flops per
element read and written (8 bytes), about 8 flops a byte, where the M5 Pro's
balance point is several times that. So a kernel of our own only has to move
the data at full bandwidth, not to match MPS's arithmetic.

## Plan

1. **The update, lower triangle only**: a kernel that walks the lower
   triangle's tiles and applies $A_{ij} \mathrel{-}= [V\ Y]_i [Y\ V]_j^T$, reading
   the $2b$-wide rows of $[V\ Y]$ and $[Y\ V]$ for its tile (they are small and
   stay in cache). Target: within 10-15% of the bandwidth MPS reaches on the
   full update. Saves a third of the traffic ($3 n_1^2 \to 2 n_1^2$).
2. **$X$ from the lower triangle**: $X = L W + L^T W - D W$ ($L$ the lower
   triangle with the diagonal, $W = V T$). Reading each tile once and using it
   twice (for $X_i$ and $X_j$) needs the contributions to $X_j$ from other
   threadgroups: atomic adds on $X$ ($n_1 \times b$ floats; Metal has float
   atomics on device memory), or per-tile-column partials and a second pass.
   Reading the triangle twice (once by tile rows, once by tile columns) is
   simpler and costs what reading the full matrix costs now, so it only
   matters with step 1 done. With both: $1.5 n_1^2$, half of today's.
3. Drop the mirroring: `eigh_band.mm` copies the given triangle in and
   mirrors it on the CPU (`mirror_lower`), which would no longer be needed.
4. Keep MPS for the general (SVD) reduction, which has no symmetry to use.
5. Tests (the `band` section of `tests/test_eigh.cpp`), A/B timings at 2048,
   4096 and 8192, the eigh re-measure (epoch up) and the docs.

A cheaper first try for step 1, without a kernel: the lower triangle updated
in column strips by MPS (one product per strip, from the strip's diagonal
down). It saves the same traffic, but each strip is a product to encode (25-30
µs of CPU each), and at a few hundred blocks with several strips each the
encoding could outrun the GPU; measure before committing to it.

## Effort

2-6 days: 1-2 for the lower-triangle update kernel at full bandwidth, 2-3 for
an $X$ that reads the triangle once, a day for integration, tests, the
re-measure and docs. The strip version of step 1, if its encoding keeps up,
is about a day.

## Expected gain

An estimate: step 1 alone saves about a third of the 73 ms at 4096 (~24 ms:
161 ms to ~137, 1.18x); with step 2, about half (~35 ms: to ~126, 1.28x). At
8192, where the products are most of the 872 ms, the saving should be several
hundred ms (perhaps 872 to 600-650). Measure the products' share at 8192
before quoting it.

## Risks and open questions

- Getting a hand-written kernel to full bandwidth; tile shape and the
  diagonal tiles (half-wasted) need tuning.
- The panel kernel reads the next block's columns below the diagonal block:
  the lower triangle, so it is unaffected.
- Float atomics make $X$'s summation order nondeterministic; the library has
  so far kept every sum in a fixed order (see the `tridiag` backend's
  reduction), so prefer the two-pass partials if results must stay
  reproducible run to run.

## Where to start

- `src/band_reduce.mm`: `band_reduce_symmetric`'s per-block products.
- `src/eigh_band.mm`: the copy-in and `mirror_lower`.
- A new kernel file, or `shaders/Eigh_Tridiag.metal`, whose `tridiag`
  reduction already reads only the lower triangle in 64 × 64 tiles for its
  matrix-vector product: the same tile walk, with a block of vectors instead
  of one.
