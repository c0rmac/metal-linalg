# The SVD's batched band blocks in three passes

Status: proposal, not started (2026-10-09).

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
