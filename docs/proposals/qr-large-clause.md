# QR's large-matrix clause by rows and k

Status: **done** in 2.15.0 (2026-10-07); see [Done](#done-2026-10-07).

## What

QR's routing sends a call to the GPU anyway for `k = min(M, N)` from
`gpu_large_min_k` (in a batch of at most `gpu_large_max_batch`). A tall
matrix's cost grows with its rows too, which `k` ignores. Compare a size that
counts them, in the library and in `tuning/tune_qr.py` alike, and give the
grid the shapes that tell: tall matrices large by their rows, which it
lacked (its tallest was 2048 x 64).

## Why: the measurements

M5 Pro, after the re-measure of 2026-10-07 (`gpu_large_min_k` 768, by `k`):

| shape | GPU (blocked) | CPU | routed to |
|---|---|---|---|
| 8192 x 512 | 8.5 ms | 48.4 ms | CPU |
| 4096 x 1024 | 11.4 ms | 66.3 ms | GPU |
| 40000 x 256 | 18.4 ms | 78.9 ms | CPU |
| 100000 x 32 | 4.0 ms | 7.8 ms | CPU |
| 1024 x 256 | 2.2 ms | 2.6 ms | CPU |

## Plan

1. `qr_uses_gpu` (`src/qr.mm`): the clause on a size that counts rows.
2. `tune_qr.py`: the same comparison in `routed`, its candidates the grid's
   sizes; the grid's tall shapes 2048 x 512 to 16384 x 64 at batches 1 and
   4 (the reduced kernel and the CPU only: the other kernel's workspace is
   M x M, and the crossover sends them to the reduced one anyway), their
   wide transposes, and batches of 16 large square matrices. And fit the
   clause and the window in both orders, keeping the better: with the
   blocked QR the GPU wins two regions (large batches of small matrices,
   shared with the CPU; and anything from about k = 384), and the window
   fitted first took the second, leaving no clause worth adding (1.0508x
   against 1.0193x).
3. Re-measure (kernel epoch qr 5).

## Effort

Half a day.

## Expected gain

5x on tall matrices the clause misses, such as 8192 x 512.

## Done (2026-10-07)

Built as planned. Which size: fitted on the new run (207 shapes, its tall
ones included), geometric-mean regret against the best backend at each
shape, and held out (fitted on half, scored on the other):

| the clause's size | fitted | held out |
|---|---|---|
| `cbrt(max(M, N) k^2)`, the work's | 1.0225x | 1.0314x |
| `cbrt(M k^2)` | 1.0171x | 1.0274x |
| `sqrt(max(M, N) k)` | 1.0175x | 1.0340x |
| `sqrt(M k)` | **1.0133x** | **1.0274x** |

`sqrt(M k)`: `k` for a square matrix, and for a wide one too, which the CPU
path factors by its leading square block and one product, cheaper than its
work says; more for a tall one, which LAPACK takes dearly. The M5 Pro's row:
GPU iff `16 <= k <= 256` and `batch * k >= 20480`, or `sqrt(M k) >= 512` in
a batch of up to 64 ([qr.md](../qr.md#tuning)). Its worst shape is 4 of 2048
x 64 (1.0 ms on the GPU, 1.7 on the CPU, `sqrt(M k)` = 362).
`gpu_large_min_k` keeps its name and its meaning for square matrices; the
loader had dropped the tall shapes (timed without the M x M kernel), which
it now keeps.
