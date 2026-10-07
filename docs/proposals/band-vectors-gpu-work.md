# Less GPU work for the band SVD with vectors

Status: proposal, not started (2026-10-07).

## What

The `band` backend with singular vectors is bound by the GPU: at 4096 on an
M5 Pro its GPU works 363 ms of the call's 363, without a gap since 2.15.0
([band-vectors-overlap.md](band-vectors-overlap.md#done-2026-10-07)), while
the CPU is idle through the band reduction and after the divide and
conquer. Either the GPU does less, or the CPU takes some of it:

| GPU work at 4096 | ms |
|---|---|
| the band reduction | 154 |
| Q1 and P1 explicit | 32 |
| Q Q2 and P P2 (`bd_chase_apply`) | 133 |
| U = Q U_B, V^T = V_B^T P^T | 40 |

## Plan

1. **P1 on the CPU, under the reduction.** P1 = G_0 G_1 ... from the row
   panels' reflectors, accumulated forward (P <- P G_a) as each aggregate of
   eight blocks completes: about 137 GFLOP on the CPU's matrix units (about
   2 TFLOP/s) over the reduction's 150 ms, where the GPU's backward
   accumulation takes about 16 ms. The forward order costs twice the flops
   (no zeros to skip) but runs on otherwise idle cores.
2. **`bd_chase_apply`'s step latency.** With its products removed the kernel
   still took 43 of its 61 ms a side: the steps' handoffs and barriers.
   Prefetching the next tile, fewer device fences and less threadgroup memory
   did not help (two-stage-vectors.md); what was not tried: two blocks a
   step per simdgroup (half the steps, twice the registers), and the tiles
   handed down in registers by simdgroup shuffles where a simdgroup owns two
   groups.
3. Measure each against the timeline (the command buffers' GPU times), since
   only GPU time saved shortens the call.

## Effort

About 2 days for the first, open-ended for the second.

## Expected gain

About 16 ms (1.05x) from the first; from the second, up to about 40 ms
(1.1x) if the kernel's steps cost half what they do.

## Where to start

`queue_q1_p1` and `build_aggregate` in `src/svd_bidiag.mm`, and `bd_chase_apply`
(`shaders/Svd_Bidiag.metal`).
