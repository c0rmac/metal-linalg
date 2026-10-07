# A faster TSQR top kernel

Status: **done in 2.15.0** (2026-10-07). Step 1, timing the phases, found
the top's QR of the stacked R's at 57 of 91 us and its last b x b step at 24
(not nothing, as guessed). The stacked R's are now factored as a binary tree
of triangle pairs (LAPACK's `stpqrt2`), a simdgroup a pair and a lane a
column, E carried down the tree, and the LU and inverses in registers and
shuffles. Doing that uncovered the larger cause: the file is built with
`-fno-fast-math`, and a kernel with any IEEE division or square root in it
(an untaken one was enough) compiles all its arithmetic in IEEE mode. The
panel kernels now divide and take square roots with the fast approximations
and a Newton step. A 4096 x 16 panel: top 91.6 to 42.9 us, leaf 32.6 to 20.8,
rebuild 15.6 to 12.0. svdvals on `band` 1.30x at 1024, 1.24x at 2048, 1.13x
at 4096; eigvalsh 1.18x, 1.15x, 1.08x. The dispatch merge (step 3) was not
tried. See [the study](../studies/proposals-2-15-apple-m5-pro.md), section 2;
the same IEEE finding is followed up in [ieee-mode-audit.md](ieee-mode-audit.md).

What follows is the proposal as written on 2026-10-04.

## What

Make `bd_tsqr_top` (in `shaders/Svd_Bidiag.metal`), the middle kernel of a
tall panel's TSQR in the `band` backends' GPU stage, several times faster. It
factors the leaves' stacked $R$'s and turns the result into the panel's
compact Householder form ($T$, and what `bd_tsqr_rebuild` needs). Today it
takes about 95 µs a panel, the largest single cost of a panel.

## Why: the measurements

The `band` backends' GPU stage on the M5 Pro, every kernel and matrix product
timed in a command buffer of its own (which adds about 10 µs to each; see
[the index](README.md)):

| GPU time | svdvals 2048 | svdvals 4096 | eigvalsh 2048 | eigvalsh 4096 |
|---|---|---|---|---|
| TSQR top | 21.9 ms (239 × 91.6 µs) | 49.0 ms (495 × 98.9 µs) | 10.9 ms (119 × 91.3 µs) | 24.0 ms (247 × 97.0 µs) |
| TSQR leaves | 11.1 ms (46 µs each) | 24.1 ms (49 µs) | 5.1 ms (43 µs) | 10.8 ms (44 µs) |
| TSQR rebuild | 5.8 ms (24 µs) | 13.3 ms (27 µs) | 2.7 ms (23 µs) | 6.4 ms (26 µs) |
| one-simdgroup panels (up to 128 rows) | 0.7 ms (44 µs) | 0.7 ms (46 µs) | 0.3 ms (42 µs) | 0.3 ms (42 µs) |
| large matrix products | 9.9 ms | 96.5 ms | 7.1 ms | 73.1 ms |
| small matrix products | 4.2 ms | 21.0 ms | 3.6 ms | 8.3 ms |
| total (profiled) | 53.6 ms | 204.5 ms | 29.7 ms | 122.8 ms |

against, for the whole call, svdvals 70 and 226 ms and eigvalsh 47 and 161
ms. The top takes about the same time at 2048 as at 4096, with 16 leaves (256
stacked rows at $b = 16$) or 32 (512 rows): its time is not in the size of
the stacked matrix but in its fixed, sequential steps.

## What the kernel does, in order

1. `qr_rows`: the QR of the stacked $R$'s, a thread a row, the row rotated in
   registers; per column, two threadgroup barriers (the norm's partials, then
   the dot products') and simdgroup reductions. 16 columns, so 32 barriers
   over up to 1024 threads, then `form_t` for $T$.
2. $M = T V(0{:}b, :)^T$, and $E = I - V M$ for every stacked row (to the
   scratch, for the rebuild): loops over $b \times b$ entries in threadgroup
   memory, a barrier each.
3. $Q_1$, the panel $Q$'s top $b$ rows, from $E_0$ and leaf 0's $V$ and $T$:
   two more $b \times b$ products, with an inner $b$-long loop each.
4. Simdgroup 0 only: the LU of $Q_1 - S$ with chosen signs ($b$ dependent
   steps), $L_1^{-1}$ and $U^{-1}$ (a lane a column, $b^2$-long chains through
   threadgroup memory), $T_H = -U S L_1^{-T}$, the writes.

Moving step 4 from the whole threadgroup to simdgroup 0 (done in 2.13.0)
changed nothing measurable, so the time is probably in steps 1-3, but that is
a guess: measure first.

## Plan

1. **Time the phases.** A scratch harness that dispatches `bd_tsqr_top` alone
   on scratch data, 32 leaves at $b = 16$, with variants that return after
   each phase (a debug flag in `PanelParams`, or `#if` blocks); results are
   wrong, the timings are what counts. 2-3 hours.
2. **Rewrite the slow phases.** Candidates, in the order the timings will
   probably point:
   - Steps 2-4 in one simdgroup, a row a lane, the $b \times b$ matrices in
     registers and columns broadcast by `simd_shuffle` instead of threadgroup
     memory and barriers ($b \le 32$ fits a simdgroup).
   - Step 1 as a tree instead of one threadgroup-wide QR: pairs (or groups of
     up to 8) of $R$'s factored by one simdgroup each, with no threadgroup
     barrier inside, a barrier between levels (5 levels for 32 leaves). This
     is the "two-level TSQR" idea; the catch is that the rebuild needs $E$,
     the top's $Q$ applied down the tree, which then takes one level's worth
     of work per level. Only worth it if step 1 dominates.
   - Use the structure: at column $j$ only $j + 1$ rows of each stacked $R$
     are nonzero, so most threads do nothing useful in the early columns.
3. **Optionally drop a dispatch.** Let the last leaf threadgroup to finish
   run the top (a device atomic counter, the usual "last block" pattern), so
   that leaf, top and rebuild are two dependent dispatches rather than three:
   a few µs each, 2-5 ms at 4096.
4. Tests, the A/B timing (`bench band|eigband 2048|4096`, best of 5), then the
   re-measure (eigh and SVD epochs up; about 80 minutes unattended) and the
   docs, the two-stage study's section 4 included.

## Effort

About a day of work, plus the re-measure. The panel kernels are fiddly: rows
in registers have to be indexed by compile-time constants (the `UNROLL`
macros), or the compiler moves them to the stack; and fully unrolled
variants have taken the Metal compiler over ten minutes to build.

## Expected gain

If the top drops from about 95 µs to about 30:

| | now | then |
|---|---|---|
| svdvals, 2048 | 70 ms | ~55 ms (1.28x) |
| svdvals, 4096 | 226 ms | ~194 ms (1.16x) |
| eigvalsh, 2048 | 47 ms | ~39 ms (ties `tridiag`'s 39) |
| eigvalsh, 4096 | 161 ms | ~145 ms (1.11x) |

At 2048 the panels are a larger share, so the `band` thresholds (eigvalsh
from 4096, svdvals from 1536 on the M5 Pro) may move down. The leaves (about
45 µs) and the rebuild (about 25 µs) are the next targets in the same
kernels.

## Risks and open questions

- The profile's 10 µs a dispatch overstates every small kernel; confirm each
  gain with the A/B timing of the whole call, not with the profile.
- Correctness: `tests/test_svd.cpp` and `tests/test_eigh.cpp` run `band` at
  every width from 1×1 to 1100×1100, either side of the one-simdgroup panel
  and the TSQR leaf; rank-deficient panels must keep working (the LU with
  chosen signs is what makes those stable).

## Where to start

- `shaders/Svd_Bidiag.metal`: `bd_tsqr_top`, `qr_rows`, `form_t`.
- `src/band_reduce.mm`: `panel()`, which sizes the top's threadgroup
  (`leaves * cols` threads, rounded to 32).
- [The two-stage study](../studies/two-stage-apple-m5-pro.md), section 4, for
  how the panel kernels got to where they are and what did not work.
