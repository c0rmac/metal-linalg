# Less GPU work for the band SVD with vectors

Status: **done** in 2.15.0 (2026-10-07), the kernel's part: its blocks
carry Y = -T^T V^T instead of T, and Q2 and P2 took 98 ms instead of 135
at 4096; with the CPU then the bottleneck, P1 on the CPU would not pay.
See [Done](#done-2026-10-07). The kernel's step latency, measured again in
2.17.0: not what bounds it ([below](#the-step-latency-measured-2026-10-09)).

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

## Done (2026-10-07)

**The kernel's blocks carry Y = -T^T V^T.** Each block applied
X <- (I - V T V^T)^T X as W = V^T X, W <- T^T W, X <- X - V W: three
dependent products a step, 60 tile products a block. The CPU now builds
Y = -T^T V^T for each block (from the chase's reflectors, row by row:
Y_i = -tau_i (V_i + sum_{m < i} (V_m . V_i) Y_m, each reflector spanning 16
of the block's 32 rows), and the kernel takes Z = Y X, X <- X + V Z: two
products a step, 52 tile products. The block keeps V's six nonzero 8 x 8
tiles and Y's seven (Y's eighth is zero), 13 tiles, packed in the order the
kernel loads them. One handoff buffer a simdgroup (two, to save a barrier a
step, ran no faster in twice the threadgroup memory). `bd_chase_apply`'s
GPU time for both sides, M5 Pro:

| k | before | after |
|---|---|---|
| 1024 | 2.5 ms | 1.9 ms |
| 2048 | 23.6 ms | 13.6 ms |
| 4096 | 133 ms | 98 ms |

The narrow variant (16 columns a threadgroup, eight groups at once) is the
faster up to 1024 columns, the two equal at 2048, the wide one (32
columns, four groups) faster above: 36 against 46 ms at 3072, 98 against
113 at 4096.

**Then the CPU is the bottleneck.** At 4096 the GPU's Q2 and P2 (98 ms) now
end before the divide and conquer (about 120 ms on the CPU), so the call's
path is the band reduction (GPU), the chase (CPU, Q1 and P1 under it), the
divide and conquer (CPU), the products (GPU). Two consequences:

1. The divide and conquer's top products go to the GPU
   ([divide-and-conquer-gpu-products.md](divide-and-conquer-gpu-products.md#done-2026-10-07))
   here too, but only once Q2 and P2's last command buffer has completed
   (`MpsGemm`'s `after`): queued behind them, the first product had waited
   53 ms, longer than the CPU takes for it.
2. P1 on the CPU (step 1 of the plan) would take GPU work out from under
   the chase, where the GPU is not the bottleneck (Q1 and P1's 32 ms under
   the chase's 43): not built. Measured in 2.17.0 by leaving P1's GPU work
   out (the wrong answer, the right timing): 1536 47.9 to 47.3 ms, 2048
   76.9 to 77.2, 3072 x 2048 89.1 to 87.1, 4096 347.7 to 345.0, at most
   1.02x before the CPU's own cost of P1, on the CPU that is the bottleneck.

Found on the way: `MpsGemm` wrapped every one of a solve's page-aligned
temporaries as a Metal buffer when it was allocated, tens of microseconds
each, though at 1024 no product reaches the GPU; now only when a product
first needs it.

**Measured** (alternating builds, M5 Pro, one matrix): 1.04-1.05x at 4096
(383 to 368 ms), 1.04x at 2048 (85.5 to 81.8), 1.03x at 3000 x 2048;
768-1536 within 2% either way. Memory: 13 tiles a block instead of 12,
about 440 MB a side at 8192.

**What is left** is the CPU's: the chase (43 ms at 4096) and the divide and
conquer (about 120 ms, its top products now on the GPU). The step latency
of `bd_chase_apply` (two blocks a step, register handoffs by simdgroup
shuffles) no longer shortens the call while the CPU is the bottleneck.

## The step latency, measured (2026-10-09)

2.17.0 runs `bd_chase_apply` in more places (eigh's `band` with vectors, the
batch backends' two stages), so the step-latency idea was measured rather
than left on the judgement above. M5 Pro, min of three interleaved runs:

| call | as now | without `bd_chase_apply` | twice the simdgroups a threadgroup |
|---|---|---|---|
| SVD 2048 | 79.8 ms | 78.8 (1.01x) | 1.007x |
| SVD 4096 | 352.2 | 313.5 (1.12x) | 1.003x |
| eigh 2048 | 50.8 | 49.4 (1.03x) | 0.989x |
| eigh 4096 | 206.6 | 191.6 (1.08x) | 1.007x |
| SVD 8 x 1024^2 | 64.5 | 58.6 (1.10x) | 0.996x |
| SVD 16 x 1024^2 | 114.2 | 108.3 (1.05x) | 0.977x |
| eigh 8 x 1024^2 | 45.4 | 38.8 (1.17x) | 0.982x |
| eigh 16 x 1024^2 | 68.1 | 62.4 (1.09x) | 0.961x |
| eigh 16 x 512^2 | 15.5 | 14.2 (1.09x) | 0.973x |

The kernel without its work is the gains' upper bound. Twice the
simdgroups (8 a threadgroup wide, 16 narrow) covers twice the groups a pass
and so halves the steps, as two blocks a step a simdgroup would: the calls
did not move, and the kernel's own GPU time at 4096 went from 98 to 102 ms
for the SVD's two sides (eigh's 48 to 49); only the narrow variant at 2048
was faster (14.0 to 11.7 ms), off the call's critical path. At 4096 the
kernel is bound by its products, not its steps: 13 tile products a block
for each of 32 columns, about 400 GFLOP a side, with some 25 simdgroups a
core resident to hide the steps' latency. Neither untried variant (two
blocks a step, the handoffs by shuffles) changes the products, so neither
was built.
