# The SVD's batched band blocks in three passes

Status: **done** in 2.17.0 (2026-10-09), in a different form from the plan's; see [Outcome](#outcome).

## What

`bidiag_batch`'s band reduction (`encode_band` in `src/svd_bidiag_batch.mm`)
applies each block's two panels one after the other: the column panel's
$Q^T$ to the columns right of it ($Z = S V_1$, then $S \mathrel{-}= Z T_1
V_1^T$: a read and a read-write of the trailing matrix), then the row
panel's $Q$ to the rows below ($Y = V_2^T S_2$, then $S_2 \mathrel{-}= V_2
T_2^T Y$: another read and read-write). As LAPACK's `slabrd` does for the
one-stage reduction, the two can be merged: $Z = A^T V_1$ over the block's
rows (one read); the row panel updated from $Z$ alone ($b$ rows) and
factored; $P = (A_{22} - V_1 W^T) V_2$ from $A_{22} V_2$ (one read) and small
products; then $A_{22} \mathrel{-}= [V_1\ \ P T_2] [W\ \ V_2]^T$, one rank-$2b$
product (one read-write). Three passes over the trailing matrix a block
instead of four.

## Why: the measurements

The reduction is bound by memory; on an M5 Pro it is 8 ms of one
1024 x 1024's 29 with vectors in two stages, 12-17 ms a chunk of 4-5 at
1024. For one matrix and a batch's first chunk it is on the critical path;
later chunks' reductions run under the CPU's solves.

## Expected gain

A quarter of the reduction where it is exposed: about 2 ms of one 1024 x
1024's 29 (7%), a few % for batches. The eigensolver's counterpart already
makes one rank-32 product of its update (1.1x).

## Effort

Half a day with tests: the small products' order and the row panel's update
are where it can go wrong.

## Outcome

Built as planned first: three passes, but the small products ($W = Z T_1$,
the row panel's update, $W^T V_2$, the correction of $P$, $P T_2$) as MPS
products of their own, ten dispatches a block against the old eight. It lost
almost everywhere (0.91-0.96x; 1.12x only at 16 x 1024^2 for the singular
values alone): a dispatch's fixed cost outweighed the pass it saved.

Built again with the small work folded away: the row panel's kernel
(`bb_panel`, flag `merge`) forms $W = Z T_1$ a row at a time, updates its
rows before factoring them, and forms $V_2 T_2$ at the end (no reductions:
$T_1$, $V_{1t}$ and $T_2$ in threadgroup memory); it writes $W^T$ into 16
spare rows below the matrix, so that one product gives both $A_{22} V_2 T_2$
and $W^T V_2 T_2$; one small product corrects $Q = A_{22} V_2 T_2 - V_{1b}
W^T V_2 T_2$; and $A_{22} \mathrel{-}= [W\ V_2][V_{1b}\ Q]^T$. Three passes
and five dispatches against four and eight. Against the old blocks,
interleaved, the minimum of four rounds each, M5 Pro:

| | old | merged |
|---|---|---|
| singular values alone, 1 x 1024^2 | 26.7 ms | 25.0 |
| singular values alone, 16 x 1024^2 | 64.6 | 55.9 |
| singular values alone, 64 x 512^2 | 38.0 | 33.5 |
| singular values alone, 256 x 256^2 | 30.3 | 27.3 |
| with vectors, 1 x 1024^2 | 30.7 | 28.5 |
| with vectors, 32 x 512^2 | 45.1 | 42.0 |
| with vectors, 16 x 384^2 | 15.4 | 14.2 |

With the reduction faster, two stages with vectors now pay from k = 288 at
any batch (1.10-1.25x the direct reduction at 288-320) and from 128 for
batches of up to the CPU's solve threads (1.05-1.8x at 2-16 matrices of
128-256; level or behind from 32, where the CPU's chases bound it), against
384 before.

