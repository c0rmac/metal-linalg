# The blocked QR's fixed costs and panels

Status: **done** in 2.15.0 (2026-10-07) as far as it went: the CPU's round
trip removed; wider panels, taller leaves and a look-ahead tried and
rejected; what is left below.

## What

For one matrix of 256 to 1024 the blocked QR ([qr-blocked.md](qr-blocked.md))
spends its time in latency rather than arithmetic: at 256 x 256 its forward
pass is about 75 dependent dispatches, 1.6 ms of GPU time for 32 MFLOP. And
between its two passes the CPU waited for the GPU, factored the last
columns by LAPACK, and released Q's formation, 0.2-0.35 ms. Where the time
goes, and what removes it.

## Why: the measurements

M5 Pro, one matrix, after the batch was added. Each part's cost from runs
without it (results wrong, timings indicative):

| | 1024 x 1024 | 4096 x 4096 |
|---|---|---|
| the call | 6.6 ms | 63 ms |
| updates inside aggregates (2 MPS products a panel) | 0.8 | 7.5 |
| the TSQR leaves | 1.4 | 4.9 |
| the TSQR top (a tree of pairs, then b x b work in one simdgroup) | 1.8 | 8.9 |
| the TSQR rebuild | 0.9 | 2.2 |

The top costs 33-35 us a panel whatever the height: its serial b x b work
(an LU, two triangular inverses, by shuffles) and a level of the tree a
barrier.

## Plan

1. Pad the matrix with zero rows and columns to whole panels with twice
   their width in rows, so that the GPU takes every column: no LAPACK tail,
   and Q's formation queued behind an event the CPU signals once Q's start
   is written, without waiting for the forward pass.
2. Commit the first command buffer after the first panel.
3. Try fewer, larger panel steps: 32-column panels, leaves of 256 rows
   (8 rows a lane, a shorter tree, and one kernel for panels up to 256
   rows).

## Effort

A day.

## Done (2026-10-07)

1 and 2: one 256 x 256 2.5 to 1.5 ms, 300 x 1000 2.6 to 1.8, 8192 x 512 9.7
to 8.4; 512-4096 within 2%. (With the batch at once, see
[qr-blocked-batched.md](qr-blocked-batched.md).)

3, rejected: 32-column panels took 1.4-1.5x 16's time at 512-2048 (4.4 ms
against 3.0 at 512, 25 against 18 at 2048) and the same at 4096; leaves of
256 rows were slower from 512 on (7.2 against 6.5 ms at 1024, 65 against 63
at 4096), faster only at 256 (1.48 against 1.69), and a rank-one 300 x 200
lost its orthogonality (1.4e-2) in the 256-row single-simdgroup panel. The
panels on a second queue beside the trailing update: no gain
([qr-look-ahead.md](qr-look-ahead.md)).

**Left**, each a few percent: the leaves and the top as one dispatch (the
last leaf's threadgroup to finish does the top: a dispatch's gap a panel,
some 3-5 us of about 75); the updates inside aggregates as kernels of their
own (about 6% of the call, if they halve); the input's scan on the GPU
(0.2-0.6 ms before the GPU starts). The CPU stays the faster for one matrix
up to about 384 x 384 and for batches of up to 64 matrices of 128-256.
