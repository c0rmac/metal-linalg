# The two-stage reduction with singular vectors

Status: proposal, not started (2026-10-04). Larger than the others; do the
[parallel divide and conquer](parallel-divide-and-conquer.md) first.

## What

Let the SVD with vectors use the `band` backend's two-stage reduction (dense
to band on the GPU, band to bidiagonal on the CPU) instead of `bidiag`'s
one-stage one, which means back-transforming the bidiagonal's singular
vectors through both stages: $U = Q_1 Q_2 U_B$ and
$V^T = V_B^T P_2^T P_1^T$, with $Q_1, P_1$ the GPU stage's block reflectors and
$Q_2, P_2$ the bulge chase's. LAPACK's two-stage drivers do not offer vectors
either (`ssyevd_2stage` takes `JOBZ = 'N'` only); the task-based libraries'
two-stage eigensolvers (PLASMA, MAGMA) do.

The SVD only: for eigh the one-stage reduction reads the matrix once a column
(the SVD's twice), and the second back-transformation would cost about what
the two-stage reduction saves (below).

## Why: the measurements

One 4096 × 4096 on the M5 Pro (run `20261004-06bc11` and separate timings):

| SVD with vectors, `bidiag` | |
|---|---|
| total | 1535 ms |
| GPU reduction (svdvals on `bidiag`, 733 ms, less 12 of bisection) | about 720 ms |
| `sbdsdc` on one core | 718 ms |
| back-transformation (the rest) | about 100 ms |

The `band` backend's reduction of the same matrix: about 180 ms on the GPU
and 35 ms of chase.

## Plan

1. **Keep the GPU stage's reflectors.** The panels now write $R$ (or $L$) in
   place and their Householder vectors to a workspace that the next block
   overwrites. Write the vectors into the matrix below (column panels) and
   right of (row panels) the band, as LAPACK's `sgebrd` stores its
   reflectors, and keep each block's $T$ and $S$ ($b \times b$).
2. **Keep the chase's reflectors.** `band_to_bidiagonal` computes, per sweep
   and step, a left and a right reflector of length at most $b$, and discards
   them. Store $(v, \tau)$ for each: about $n^2 / 2b$ reflectors a side, $n^2$
   floats in all a side.
3. **Apply $Q_2$ and $P_2$** to $U_B$ and $V_B^T$ ($n \times n$ each) on the
   GPU. About $2n^3$ flops a side (275 GFLOP for both at 4096), but in pieces
   of $b$ rows, ordered: reflectors from the same sweep chain, and those of
   consecutive sweeps overlap. The way to make this fast is to group them
   into blocks that can be applied as small matrix products, as the papers
   below do for multicore CPUs; how best to group them for a GPU is the open
   part. Applying them one at a time, a thread a column, would take seconds.
4. **Apply $Q_1$ and $P_1$** as `bidiag` applies `sgebrd`'s reflectors today
   (128 at a time, `slarft`, three MPS GEMMs), generalised from an offset of
   one (the bidiagonal's superdiagonal) to an offset of $b$.
5. Route: `svd_band` with vectors, a policy threshold (`band_min_k`), a sweep
   backend (`band`), tests (orthogonality and reconstruction, as `bidiag`'s).

## Effort

1-2 weeks. Steps 1, 2 and 4 are a day or two each; step 3, the grouped
application of the chase's reflectors on the GPU, is most of it and the
riskiest.

## Expected gain

An estimate. With step 3 at a realistic 2-4 TFLOP/s, about 70-140 ms for both
sides, and step 4 about what `bidiag`'s back-transformation costs now:

| one 4096 × 4096 SVD with vectors | now | two-stage |
|---|---|---|
| reduction | ~720 | ~215 (180 + 35) |
| bidiagonal solver | 718 (or ~175 with parallel D&C) | the same |
| back-transformation | ~100 | ~200-240 |
| total | 1535 (or ~1000) | ~1130 (or ~600) |

So about 1.35x on today's code, about 1.65x once the divide and conquer is
parallel; the two-stage reduction's gain is masked until the solver is fast.

For eigh: the reduction would go from about 200 ms to about 155, while $Q_2$'s
application adds 35-70 ms: nothing to gain.

## Risks and open questions

- Step 3's speed is the whole case; prototype it first, on the chase's
  reflectors from a real band, before building the rest.
- Memory: the chase's reflectors take as much as a matrix a side.
- Batches: the two-slot pipeline would need the stored reflectors per slot.
- Accuracy should match `bidiag`'s (both stages are orthogonal
  transformations), but check reconstruction at 8192, where errors from two
  stages of float32 reflectors add.

## Where to start

- `src/band_reduce.mm` and the panel kernels in `shaders/Svd_Bidiag.metal`
  (flag 8 already writes a second copy of $V$; another flag could write it
  into the matrix).
- `src/band_chase.cpp`: the `Sweep::step` reflectors.
- `src/svd_bidiag.mm`: `bidiag_impl`'s back-transformation (step 3 of its
  header comment).

## References

- A. Haidar, J. Kurzak and P. Luszczek, ["An improved parallel singular value algorithm and its implementation for multicore hardware"](https://doi.org/10.1145/2503210.2503292), SC '13, 2013 — the two-stage SVD with vectors on multicore CPUs; read first for how it back-transforms through the chase.
- A. Haidar, H. Ltaief and J. Dongarra, ["Parallel reduction to condensed forms for symmetric eigenvalue problems using aggregated fine-grained and memory-aware kernels"](https://doi.org/10.1145/2063384.2063394), SC '11, 2011 — the symmetric counterpart.
