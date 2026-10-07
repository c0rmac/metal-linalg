# Fusing the band reduction's small products

Status: **done in 2.15.0** (2026-10-07). Each block's three small MPS
products are two kernels: `bd_small_partial` (partials of the b x b product
over 256 rows a threadgroup, staged in threadgroup memory) and `bd_sy_apply`
or `bd_ge_apply` (the partials summed in order, then the b-wide work in one
pass). A first version, a thread an entry over its rows straight from device
memory, was slower than MPS: latency, not work. eigvalsh on `band` 2-5%
faster at 1024-2048, svdvals 1-3% at 2048, nothing measurable at 4096: as
estimated. MPS's b x b product summed over n rows took 10-20 us whatever n
was. See [the study](../studies/proposals-2-15-apple-m5-pro.md), section 3.

What follows is the proposal as written on 2026-10-04.

## What

Each block of the `band` backends' GPU stage makes, besides its two or three
large matrix products, three small ones, each an MPS GEMM with its own encode
and dispatch. Replace each block's three with one or two kernels of our own.

The three, per block (`src/band_reduce.mm`):

| general (SVD) | symmetric (eigh) |
|---|---|
| $WU$ ($b \times b$, summed over $n_1$) | $Z = V^T X$ ($b \times b$, summed over $n_1$) |
| $X^T \mathrel{-}= (WU)^T V_{low}^T$ ($b \times m_2$) | $M = T^T Z / 2$ ($b \times b$) |
| $Y^T = S^T X^T$ ($b \times m_2$) | $Y = X - V M$ ($n_1 \times b$) |

## Why: the measurements

Profiled on the M5 Pro (each product in a command buffer of its own, which
adds about 10 µs to each; see [the index](README.md)):

| | svdvals 2048 | svdvals 4096 | eigvalsh 2048 | eigvalsh 4096 |
|---|---|---|---|---|
| small products | 4.2 ms (387 × 10.9 µs) | 21.0 ms (771 × 27.2 µs) | 3.6 ms (380 × 9.5 µs) | 8.3 ms (764 × 10.8 µs) |
| whole call | 70 ms | 226 ms | 47 ms | 161 ms |

Much of each figure is the profile's own overhead; in the real run (one
command buffer a block) what remains is a dependent dispatch's gap, a few µs,
plus the work, which for the general case's two $b \times m_2$ products is a
pass over $m_2$ rows each.

## Plan

- Symmetric: $Z$ needs a sum over $n_1$ rows, so a grid-wide reduction, then
  $M$ and $Y$: two dispatches (partial sums per threadgroup; then each
  threadgroup finishes $Z$ and $M$ from the partials itself, $b \times b$ is
  tiny, and updates its rows of $Y$) instead of three GEMMs.
- General: $WU$ the same way (partials, then finished per threadgroup), and
  the two $b \times m_2$ products fused into one pass over $X^T$'s columns:
  two dispatches instead of three, and one pass over $m_2$ instead of two.
- Tests as for the band backends; A/B timing of the whole call.

## Effort

About half a day, more if the partial-sum pattern has to be written from
scratch (the `tridiag` reduction's kernels already sum per-threadgroup
partials in a fixed order; reuse that).

## Expected gain

A dispatch and some memory traffic per block: an estimate of 5-10 ms for
svdvals at 4096 (2-4%) and less for eigvalsh; relatively more at 2048 for
svdvals.

## Risks

- MPS's small products are well tuned; a hand-written $b \times m_2$ pass has
  to read $X^T$ at full bandwidth to win.
- Keep the sums in a fixed order, as elsewhere in the library, so results do
  not change from run to run.

## Where to start

`src/band_reduce.mm`, the per-block products in `band_reduce_general` and
`band_reduce_symmetric`; kernels beside the panel kernels in
`shaders/Svd_Bidiag.metal`.
