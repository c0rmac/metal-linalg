# The two-stage reduction with singular vectors

Status: **done** in 2.15.0 (2026-10-07): the `band` backend takes singular
vectors (`svd_band_vectors`, routed by `band_min_k`); see
[svd.md](../svd.md#with-singular-vectors-since-2150). On an M5 Pro, one
square matrix: 1.21x `bidiag` at 1024, 1.45x at 2048, 2.35x at 4096 (402 ms
against 944), about 2.7x at 8192; 8.7x the CPU path at 4096 (with the
overlap of [band-vectors-overlap.md](band-vectors-overlap.md#done-2026-10-07)). It was first
prototyped and parked the same day, with Q2 at 170 ms a side ([Prototype
](#prototype-2026-10-07)); what made it pay is below ([Done](#done-2026-10-07)).
The sources of both prototypes are in [two-stage-vectors/](two-stage-vectors/).

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

## Prototype (2026-10-07)

What was built, in [two-stage-vectors/](two-stage-vectors/) (`build.sh`):

- **The chase's reflectors, recorded** (`chase_record.cpp`, `band_chase.cpp`
  storing each sweep's left and right reflectors, n^2 floats a side). Their
  product reproduces the chase: Q2^T A P2 equals the bidiagonal to 4e-6
  (`check_q2`). Recording costs nothing measurable (36.7 ms against 35 for
  the chase at 4096).
- **An order that blocks them.** Sweep s's j-th left reflector acts on rows
  s + 1 + j nb ... s + (j + 1) nb. Reflectors of one sweep are disjoint;
  sweep s' > s's reflector must come first (in reverse order, as U = Q2 U_B
  is applied) where they overlap, which is for j' = j when s' - s < nb and
  j' = j - 1 when s' - s < 2 nb. So with groups of ib consecutive sweeps:
  groups from the last to the first, in a group the steps j in increasing
  order, each step's ib reflectors a block I - V T V^T, V (nb + ib - 1) x ib.
  `check_q2` confirms it gives the same Q2 as one reflector at a time, bit
  for bit, for ib = 8, 16 and 32. The blocks do 4.5 n^3 flops a side at nb
  = ib = 16 (the zeros in V's staircase cost half again over the 2 n^3
  minimum).
- **The GPU kernel** (`q2.metal`): a threadgroup a strip of 32 columns of
  U_B, the 32,896 blocks (n = 4096) one after another, each V T V^T applied
  with simdgroup products. Correct to a few 1e-6 of |U_B| against one
  reflector at a time on the CPU.

Measured, n = 4096, a side:

| version | time |
|---|---|
| a threadgroup a 32-column strip, the blocks in sequence | 170 ms |
| the same, one threadgroup only | 83 ms (2.5 us a block) |
| a sliding window: the 15 rows a block shares with the next kept in threadgroup memory | 190 ms |
| 64-column strips (256, 512 or 1024 threads) | 188-212 ms |
| 32-column strips with 64 or 128 threads | 217-341 ms |
| a simdgroup a group, groups staggered a block apart, U_B in registers | 788 ms (spilled) |

and at 2048, 35 ms; 1024, 14 ms. The single threadgroup's 83 ms is the
chain of 32,896 dependent blocks at 2.5 us each (five barriers and their
loads); beyond about 40 threadgroups they run in waves, and 128 strips need
3-4 of them.

**What would make it pay**: several blocks in flight per threadgroup. The
dependencies allow it: block (G - 1, j) needs only (G, j) and (G, j - 1), so
consecutive groups can trail each other by one block, a wavefront. The
staggered version did that with a simdgroup a group but spilled registers;
the same schedule with the window in threadgroup memory and the simdgroups
of a step sharing the products (as the 170 ms kernel does), or larger groups
(ib = 32 halves the chain and adds a third of the flops), are the next
things to try. Q1 and P1 (step 4) and the reflector storage in the GPU stage
(step 1) were not started.

## Done (2026-10-07)

Two changes to the plan made it pay.

**Q2 applied as a pipeline of groups.** The prototype's kernel gave each
32-column strip of U_B the 32,896 blocks one after another; one strip alone
took 83 ms (2.5 us a block), and the 128 strips ran in waves. But block
(G - 1, p) needs only (G, p) and (G, p + 1): the next group can follow two
tiles behind. In `bd_chase_apply` (shaders/Svd_Bidiag.metal; prototyped as
`q2w.metal`) a threadgroup runs four groups at once, a simdgroup each, each
keeping its block's two 16-row tiles in registers and handing its lower one
to the next simdgroup through threadgroup memory; the first loads from and the
last stores to device memory. A pass is about k / 16 + 2K steps of K
blocks. At 4096: 60-64 ms a side (2048: 8.4 ms; 1024: 3.5), from 176.
Tried after and not kept: the next tile prefetched into registers (61 to
60 ms), device-memory fences only at the steps that need them (no change),
less threadgroup memory a threadgroup for more of them at once (55-62 ms
across the variants), and the two sides' kernels run concurrently (177 ms
against 132: their strips competing for the caches). With the arithmetic
removed the kernel still takes 43 ms: what is left is the steps' latency,
not the products.

**Q = Q1 Q2 and P = P1 P2 formed explicitly, under the CPU's work.** Q2 and
P2 are applied from the right to the explicit Q1 and P1 (the kernel's `down`
direction on Q^T), while the CPU runs the divide and conquer; Q1 and P1 are
formed from the band reduction's aggregated reflectors while the CPU chases
the band. Then U = Q U_B and V^T = V_B^T P^T are one product each. At 4096:

| step | GPU | CPU |
|---|---|---|
| the band reduction, keeping its reflectors | 158 ms | the aggregates' T |
| Q1, P1 explicit / the chase | 33 | 40 |
| Q Q2, P P2 / the divide and conquer | 133 | 113 |
| U, V^T, and the output | 43 | |

about 400 ms against the estimate's ~600 for this order and 740 for the
plan's (Q2 applied to U_B after the divide and conquer). The memory the plan
worried about: the chase writes its reflectors straight into the kernel's
blocks (no second copy), and the panels theirs into the aggregates; at 8192
the call keeps about 1.1 GB more than `bidiag`.

**What is left.** The divide and conquer is now on the critical path with
the chase and the band reduction; the GPU waits about 20 ms for Q2 and P2 at
4096. Applying Q2's blocks as the chase completes each group (a shared event
a few groups at a time) would hide them under the chase as well, and leave
the GPU free for the divide and conquer's top products
([its proposal](divide-and-conquer-gpu-products.md)). A batch of two at 1024
is still `bidiag`'s (its two-slot pipeline); the routing measures where.

