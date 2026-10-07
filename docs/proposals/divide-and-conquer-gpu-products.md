# The divide and conquer's top products on the GPU

Status: **done** in 2.15.0 (2026-10-07); see [Done](#done-2026-10-07).

## What

`src/divide_conquer.cpp` runs the large merges' loops on every core; what is
left of them is the products of the halves' vectors with the merge's (two
`sgemm`s a merge), which the CPU's matrix units run at about 2 TFLOP/s
however the work is split. In the `tridiag` and `bidiag` backends the vectors
live in Metal buffers in shared storage (`ws.Z`, `ws.U`, `ws.VT`): run the
largest merges' products as MPS products instead.

## Why: the measurements

The top merge of a 4096 problem on an M5 Pro, 16 threads:

| | whole solve | top merge | its products |
|---|---|---|---|
| tridiagonal (`sstedc`'s) | 50 ms | 28.5 ms | 22.8 ms |
| bidiagonal (`sbdsdc`'s) | 100 ms | 59 ms (with the deflation's copies) | 48 ms |

The products at the top are about 47 GFLOP (tridiagonal) and 90 (bidiagonal,
U's and VT's); MPS's float32 product runs at 7-8 TFLOP/s on this GPU. Splitting
them over the CPU's threads or leaving them to Accelerate's own threading
made no difference (within 10%).

## Plan

1. In `merge_symmetric` and `merge_bidiagonal`, above a size (say 2048), hand
   the products to a callback that the backends supply: Q2 and S copied (or
   allocated) in page-aligned memory wrapped as Metal buffers
   (`wrap_host`), one command buffer, MPS products into Q's columns.
2. Only where the GPU is idle: for one matrix. In a batch the solve of one
   matrix overlaps the next one's reduction on the GPU, and these products
   would compete with it; keep the CPU's there, or measure.
3. Tests: the `tridiag` and `bidiag` sections at 2048-4096, and the
   bit-for-bit checks of the values against LAPACK (the products only touch
   the vectors).

## Effort

About two days.

## Expected gain

An estimate: the top merges' products from about 23 and 48 ms to 6 and 12
at 4096, so eigh with vectors about 1.05x and the SVD with vectors about
1.04x for one matrix. Small; worth doing if the two-stage SVD with vectors
([two-stage-vectors.md](two-stage-vectors.md)) is picked up, where the
divide and conquer becomes a larger share.

## Where to start

`gemm` and `secular_symmetric` / `secular_bidiagonal` in
`src/divide_conquer.cpp`; the workspaces in `src/eigh_tridiag.mm` and
`src/svd_bidiag.mm`.

## Done (2026-10-07)

`divide_conquer.h`'s `GpuGemm`: a solve hands any product of 1 GFLOP or
more (the top merges', from n ~ 2048) to it, which runs it as an MPS product
and waits; the merges' temporaries (Q2 and S, U2, VT2 and Q) are page-aligned
allocations (`Pages`) it wraps in place as they are made, and the caller adds
its output buffers (`MpsGemm` in `metal_runtime.mm`). The hook is per call
and per thread (the large merges run on the calling thread, the small ones,
which stay on the CPU, on the workers). The `bidiag` backend's divide and
conquer now writes straight into its Metal buffers U and VT (U's first K
rows) instead of vectors copied there after.

Only for one matrix: in a batch the GPU reduces the next matrix meanwhile.
And not on the `band` backend with vectors, whose GPU is the bottleneck
([band-vectors-overlap.md](band-vectors-overlap.md#done-2026-10-07)).

M5 Pro, one matrix, against the build before, alternating runs: eigh with
vectors (`tridiag`) 1.075x at 4096 (278 ms to 259), 1.02x at 2048; the SVD
with vectors (`bidiag`) 1.046x at 4096 (871 to 832), 1.02x at 2048. Tests at
2048 (and 3000 x 2048, and deflation-heavy 2100 x 2048) check the GPU's
path: residual and orthogonality about 3e-6, as before.

