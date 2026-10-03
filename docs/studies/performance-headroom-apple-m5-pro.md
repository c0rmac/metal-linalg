# Performance headroom on an Apple M5 Pro

Could the eigensolver, the SVD and QR be made faster than they are, by more
elaborate kernels or by using the CPU and the GPU at the same time? This study
measures where the time goes today and tests each idea against measured
ceilings. The experiments and their raw output are in
[`performance-headroom/`](performance-headroom/), built by
`performance-headroom/build.sh`.

**What came of it (2.9.0).** Items 1 and 3 of the table below were built:
the CPU paths now spread a batch over every core, and the tridiagonal-QL
kernel is a fourth eigensolver backend, `ql` ([eigh.md](../eigh.md)). The
routing was re-measured with both (run
[`20261003-2d2c19`](../results/apple-m5-pro-20gpu/20261003-2d2c19/summary.md)),
which also needed three new routing terms, because the faster CPU changed the
shape of the boundaries: QR got a clause for large matrices (the GPU still wins
one 2048×2048 by 1.9x while losing every mid-size batch), and the serial
`tridiag` and `bidiag` backends got batch caps. Over the shapes of section 2,
routed calls are now 1.5-8x faster for eigh, 1.65-4.2x for the SVD and up to
2x for QR than with 2.8's routing, one QR shape 7% slower. The re-tune also
showed the SVD's Jacobi kernels losing to the CPU almost everywhere, as the
eigensolver's did before `ql`; the SVD counterpart of `ql` (section 4) was
the open item.

**And in 2.10.0** it was built: `golub_kahan`, Householder bidiagonalization
and implicit bidiagonal QR in one threadgroup per matrix ([svd.md](../svd.md)),
1.5-2.9x faster than the Jacobi kernels and 1.2-1.8x faster than the 18-core
CPU for large batches up to 48×48 (run
[`20261003-106b6c`](../results/apple-m5-pro-20gpu/20261003-106b6c/summary.md)).
It, too, needed a routing term: the CPU's QR-first path wins tall matrices
however small k is, so the SVD's GPU-or-CPU rule caps the long side
(`gpu_max_l`), as QR's now has a lower bound on k (`gpu_min_k`) for the
smallest matrices, which the parallel CPU wins at any batch. Two things
learned while building the `ql` kernel correct this study:

- *The register-resident layout suggested in section 4 would not have helped.*
  A matrix's state is about $N^2$ floats wherever it lives, and Apple's GPUs
  draw registers and threadgroup memory from the same on-chip storage, so
  moving the rows into registers does not put more matrices on a core. What
  did help at N = 48 to 87 was running the QL iteration on a simdgroup of its
  own, overlapped with the rows applying the previous sweep (9-13%), and
  sizing threadgroup memory for the N of the call.
- *The compiler's math mode was worth 1.4x.* The production kernel was first
  1.4x slower than the prototype, which turned out to be one IEEE-mode
  (`precise::`) division anywhere in the kernel: the compiler then built the
  whole kernel without fast-math optimisations. The shipped kernel uses fast
  math throughout, with a Newton step where accuracy matters.

The sections below are the study as written before either change.

Machine: MacBook Pro (16-inch, M5 Pro, Mac17,8), 20 GPU cores, 18 CPU cores
(6 super, 12 performance), 48 GB, macOS 27.0.1, on mains, 2026-10-03. A desktop
session was running (load average about 2.7 before the runs), so treat the
numbers as indicative rather than tuning-grade. Each time is the minimum of at
least 7 runs after 250 ms of warm-up.

## The answer

Yes, and by a lot in places. The largest single gain, though, is on the CPU
side of the routing, which every GPU-against-CPU comparison depends on.

1. **The CPU path runs a batch on one core.** `eigh_cpu`, `svd_cpu` and
   `qr_cpu` call LAPACK on one matrix after another. Spreading the batch over
   the 18 cores makes them **7.5 to 15x faster** for small and mid-size
   matrices (3 to 7x for the largest batched shapes). Against that CPU, the GPU
   kernels the M5 Pro policy picks today lose or tie in 19 of the 22 batched
   shapes measured. Parallelising the CPU path and re-measuring the routing
   would make the eigh and SVD calls in that grid from 32×32 up **1.7 to 8.5x
   faster than today**, and QR's up to 2x, without a new kernel.
2. **The CPU and the GPU at the same time pays across a batch, not within a
   matrix.** Giving the GPU part of a batch and the cores the rest was
   **1.03 to 1.7x faster** than the better of the two alone wherever their
   speeds are within about 3x. Unified memory means nothing is copied: both
   read the same input and write disjoint slices of the output. Within one
   matrix, the steps form a dependency chain. The parts that could overlap are
   either matrix products, where CPU and GPU together reach only 1.13 to 1.38x
   of the GPU, or bandwidth-bound, where both share the same 280 GB/s.
3. **More elaborate kernels: one clear case, measured.** A prototype batched
   eigensolver that uses LAPACK's method on the GPU (tridiagonalization, then
   implicit QL) instead of Jacobi is correct to the same 1e-6. With its compact
   layout, at N = 24 to 32 and batches of 2048 or more, it is **3 to 4x faster
   than today's GPU kernel and 2.1 to 2.3x faster than the 18-core CPU**. At no
   N up to 32 is it slower than either. At N = 48 to 64 it is 2.2 to 3.4x
   faster than today's kernel but still 0.66 to 0.86x of the CPU. The profile
   shows why, and what would fix it.
4. **Large single matrices have moderate headroom.** In the `tridiag` and
   `bidiag` backends the GPU reduction runs at 1.4 to 3.4x its memory-bandwidth
   floor. LAPACK's divide and conquer, 27 to 43% of the time, runs on **one
   CPU core** while 17 cores and the GPU wait. Batches of large matrices could
   pipeline the two (estimated 1.4 to 1.7x).
5. **What does not help, measured:** the M5's per-core neural accelerators for
   FP32 work (strict FP32 runs at the same 7.5 TFLOP/s as MPS), FP32
   emulation from FP16 products, sharing one matrix's products between CPU
   and GPU (under 3% overall), LAPACK's MRRR solver (`sstemr`) instead of
   divide and conquer, and bisection on all cores instead of `ssterf`.

Ranked by value for effort:

| # | change | gain | evidence | effort |
|---|---|---|---|---|
| 1 | CPU path parallel over the batch, then re-measure the routing | eigh 1.7-8.5x, SVD 1.7-4.3x, QR up to 2x over today on batched shapes from 32×32; 7.5-15x on batched shapes already on the CPU | measured | small |
| 2 | split a batch between CPU and GPU when their speeds are close | 1.03-1.7x over the better device | measured (coarse split search) | moderate |
| 3 | batched tridiagonal-QL eigensolver kernel, then its SVD counterpart | at N = 24-32, 3-4x over today's GPU kernel and up to 2.3x over the 18-core CPU (1.1-1.8x at N = 8-16); N = 48-64 needs a register-resident layout | prototype measured at N ≤ 64; both built (2.9.0, 2.10.0) | substantial |
| 4a | pipeline batches of large matrices (CPU solve of one, GPU reduction of the next) | 1.4-1.7x for batches | estimated from measured steps | small-moderate |
| 4b | fewer dispatches per column in the GPU reductions | 1.25-1.6x at N = 2048-4096, and a lower `tridiag`/`bidiag` crossover | estimated | moderate |
| 4c | parallel divide and conquer (Accelerate exports `slaed*`, `slasd*`) | 1.3-1.45x at N = 4096 | estimated | high |
| 4d | two-stage reduction with the first stage on the GPU | 1.4-1.6x for `eigvalsh` at large N; `svdvals` by analogy | estimated from measured CPU steps (symmetric case only) | high |

## 1. Ceilings

What this machine can do at most (`exp_gemm`):

| | throughput | error |
|---|---|---|
| GPU read bandwidth | 282 GB/s | |
| CPU read bandwidth, 18 threads | 230 GB/s | |
| MPS FP32 GEMM, 4096³ | 7.08 TFLOP/s | 7.0e-08 |
| Metal 4 `matmul2d`, FP32, strict | 7.50-7.70 TFLOP/s | 7-10e-08 |
| `matmul2d`, FP32, `relaxed_precision` | 11.6 TFLOP/s | 3.3e-05 |
| `matmul2d`, FP16 inputs, FP32 accumulate | 22.9 TFLOP/s | 1.4e-05 |
| `matmul2d`, BF16 inputs | 22.8 TFLOP/s | 1.2e-04 |
| Accelerate `cblas_sgemm` on the CPU | 2.53 TFLOP/s | 8.8e-08 |
| MPS on the GPU and `sgemm` on the CPU at once | 9.74 TFLOP/s combined (7.37 + 2.36) | |

The error is the largest of 256 sampled entries of
$|c_{ij} - \hat c_{ij}| / (\|a_i\| \|b_j\|)$, against float64.

The neural accelerators speed up only reduced-precision inputs. Strict FP32
`matmul2d` runs at MPS's speed. Emulating FP32 from three FP16 or BF16
products (hi·hi + hi·lo + lo·hi) would net about 7.6 TFLOP/s, no better than
FP32 itself. FP16 would also need per-block scaling for its exponent range.
So for decompositions accurate to FP32 the tensor units offer nothing on this
chip. They would matter only for a GEMM-bound solver run at reduced precision,
or followed by FP32 refinement, and none of the current backends is GEMM-bound
(section 5).

## 2. The CPU path runs a batch on one core

`exp_batch` runs each shape four ways: the library's CPU path as shipped
(one matrix at a time); the same LAPACK calls with the batch split over the
18 cores (`dispatch_apply`, each worker single-threaded via
`BLASSetThreading`); the GPU backend the M5 Pro policy picks (`exp_routes`
confirms that it picks the GPU for every shape here); and the batch split
between the GPU and the cores at once. All times are in ms.

| solver | shape | batch | today (GPU) | CPU, 1 core | CPU, 18 cores | CPU + GPU | best vs today |
|---|---|---|---|---|---|---|---|
| eigh | 8×8 | 4096 | 0.50 | 9.01 | 0.64 | **0.47** | 1.06x |
| eigh | 16×16 | 4096 | 2.17 | 32.97 | 2.16 | **1.28** | 1.70x |
| eigh | 32×32 | 256 | 1.50 | 8.19 | 0.57 | **0.52** | 2.9x |
| eigh | 32×32 | 4096 | 16.07 | 130.92 | 9.42 | **6.43** | 2.5x |
| eigh | 64×64 | 256 | 9.83 | 35.39 | 2.46 | **2.20** | 4.5x |
| eigh | 64×64 | 2048 | 75.90 | 283.93 | 18.72 | **16.34** | 4.6x |
| eigh | 128×128 | 256 | 42.05 | 122.59 | 9.82 | **9.79** | 4.3x |
| eigh | 256×256 | 64 | 72.61 | 123.44 | **12.08** | 15.71 | 6.0x |
| eigh | 512×512 | 16 | 129.46 | 132.80 | **15.31** | 29.63 | 8.5x |
| svd | 8×8 | 4096 | 1.23 | 18.42 | 1.26 | **0.94** | 1.31x |
| svd | 32×32 | 4096 | 24.82 | 210.33 | 14.98 | **10.64** | 2.3x |
| svd | 64×64 | 256 | 8.45 | 45.67 | 3.52 | **2.87** | 2.9x |
| svd | 128×128 | 64 | 13.87 | 63.73 | 5.73 | **4.99** | 2.8x |
| svd | 256×256 | 16 | 24.25 | 53.79 | **5.60** | 13.15 | 4.3x |
| svd | 512×512 | 4 | 48.55 | 62.58 | **18.14** | 33.42 | 2.7x |
| svd | 1024×64 | 64 | 10.30 | 33.36 | 4.91 | **4.58** | 2.2x |
| svd | 2048×256 | 16 | 38.72 | 129.85 | **18.91** | 22.52 | 2.0x |
| qr | 16×16 | 10000 | 3.80 | 21.33 | 2.04 | **1.83** | 2.1x |
| qr | 64×64 | 1000 | 4.46 | 26.98 | 2.26 | **2.19** | 2.0x |
| qr | 256×128 | 1000 | 35.30 | 294.89 | 37.40 | **22.90** | 1.54x |
| qr | 512×512 | 32 | 16.86 | 93.79 | 12.43 | **10.03** | 1.68x |
| qr | 1024×512 | 16 | 18.08 | 91.58 | 15.40 | **12.09** | 1.50x |

Two consequences:

- **Every GPU-against-CPU ratio for a batch, in the docs and in the tuned
  policies, compares the GPU with one CPU core.** (A lone large matrix does get
  Accelerate's own threading.) The routing sweeps time the shipped CPU path, so the
  M5 Pro rows send batched work to the GPU that 18 cores do up to 8.5x faster.
  Against the parallel CPU, today's GPU kernels are ahead at only three shapes
  (eigh 8×8, by 1.29x; svd 8×8, 1.03x; QR 256×128, 1.06x). The "up to 20x for
  batches of small matrices" in `eigh.md` is against one core.
- **Callers already routed to the CPU get the same 7.5-15x** whenever they pass
  a batch, for example a batch of 16 matrices of 8×8, which the M5 Pro policy
  keeps on the CPU.

The change is small: chunk the batch loop over `dispatch_apply` with
per-worker scratch, and set `BLASSetThreading` to single-threaded in each
worker on macOS 15 and later. The routing then has to be re-measured, since
every GPU/CPU boundary moves. Two caveats. A caller that already runs its own
threads wants a way to cap the cores used. And on battery the GPU may be the
better choice for energy even where it is slower. The M1, with 4 performance
and 4 efficiency cores, would gain less (unmeasured).

## 3. The CPU and the GPU at the same time

**Across a batch.** The CPU + GPU column runs the GPU on the first *g*
matrices from its own thread while the cores take the rest, with *g* the
best of five fractions around the throughput-balanced split. It wins wherever
the two devices are within about 3x of each other, by 1.03 to 1.70x over the
better one. It gets 60-92% of the throughput the two would have with no
interference (for example eigh 32×32 ×4096: 6.43 ms measured, 5.94 ms ideal):
73-92% for the longer eigh and SVD calls, 68-79% for QR, which moves more
data per flop, and as little as 60% for sub-millisecond calls, where starting
the threads costs. Some of the shortfall is the coarse split search rather
than contention.
Unified memory is what makes this cheap: the GPU reads the caller's buffer in
place and writes its slice of the output, and the CPU workers write theirs.
A production version needs the split fraction per device and shape class,
which the tuning harness could fit from the two timings it already takes.

**Within one large matrix.** The large-matrix backends (section 5) run
reduction on the GPU, then the tridiagonal or bidiagonal solve on the CPU, then
the back-transformation on the GPU. Each step needs the previous one's result.
What could overlap:

- The reduction is bandwidth-bound and the CPU shares the same memory: no gain.
- The matrix products of the back-transformation: CPU and GPU together reach
  1.13-1.38x of the GPU alone (9.74 against 7.08-7.70 TFLOP/s at 4096), on
  steps that take 3-10% of the time. At most 2-3% overall.
- The CPU's divide and conquer runs on one core while the GPU waits: the one
  real opportunity, section 5.

**Across a batch of large matrices** the chain can be pipelined. The CPU
solves matrix *b* while the GPU reduces matrix *b + 1*, and the steady state
becomes the slower of the two devices instead of their sum. From the measured
steps below, that gives eigh at N = 4096 about 346 ms per matrix instead of 521
(1.5x), the SVD at k = 4096 about 1.04 s instead of 1.78 (1.7x), and eigh at
N = 8192 1.36x.

## 4. A better kernel for batched small and mid-size matrices

The GPU loses to the parallel CPU because of the algorithm, not the hardware.
Cyclic Jacobi does several times the flops of LAPACK's method
(tridiagonalization, then implicit QL or divide and conquer, then the
back-transformation): by a rough count about $9N^3$ per sweep over 7 sweeps
at N = 64, against $5$-$10N^3$ in all. `eigh.md` gives the reason it was not used: on a GPU
the QL phase "applies roughly N² dependent Givens rotations, each of which
would be a barrier". That holds if the rotations are applied as they are
made. It stops holding if one thread runs the QL iteration on $(d, e)$, which
never reads $Z$, and records a sweep's rotations, and then each thread applies
the whole recorded sequence to **its own row** of $Z$. Rows are independent,
so a sweep costs two barriers instead of one per rotation.

The prototype (`tdql.metal`) is one threadgroup per matrix, N ≤ 64, with
thread *i* owning row *i*. Its steps are a Householder tridiagonalization
(`ssytd2`), Q formed in place (`sorg2r`'s backward accumulation), implicit QL
with the recorded sweeps, then a rank sort. Over 64 matrices per shape it is as
accurate as the existing kernels: residual and orthogonality ≤ 1.5e-6 relative
to $\|A\|_F$, eigenvalues within 7.5e-7 of LAPACK, every matrix converged. It
does not yet scale magnitudes, handle NaN input or write the `info` word.

| N | batch | prototype | today's GPU kernel | CPU, 18 cores | vs today | vs 18 cores |
|---|---|---|---|---|---|---|
| 8 | 4096 | 0.46 | 0.50 | 0.64 | 1.09x | 1.39x |
| 16 | 4096 | 1.22 | 2.15 | 2.16 | 1.76x | 1.77x |
| 24 | 4096 | 2.40 | 7.10 | 5.53 | 2.96x | 2.30x |
| 32 | 256 | 0.53 | 1.49 | 0.57 | 2.81x | 1.08x |
| 32 | 2048 | 2.13 | 8.26 | 4.41 | 3.88x | 2.07x |
| 32 | 4096 | 4.04 | 16.26 | 9.18 | 4.02x | 2.27x |
| 48 | 2048 | 13.11 | 29.26 | 11.33 | 2.23x | 0.86x |
| 64 | 256 | 3.57 | 9.97 | 2.34 | 2.79x | 0.66x |
| 64 | 2048 | 22.26 | 75.54 | 18.54 | 3.39x | 0.83x |

The CPU column is the best of all runs of that shape, so the right-hand
ratio is the conservative one. N ≤ 32 uses the compact layout (4.7 KB of
threadgroup memory per matrix), N > 32 the full one (17.7 KB).

**What limits it** (`exp_stage`, batch 2048):

| N | layout | tridiagonalize | form Q | QL, no Z updates | QL with Z updates (whole kernel) |
|---|---|---|---|---|---|
| 64 | 17.7 KB | 4.2 | +3.4 | +10.6 | +4.2 = 22.3 |
| 32 | 17.7 KB | 1.7 | +0.6 | | 5.9 |
| 32 | 4.7 KB | 0.33 | +0.27 | +0.99 | +0.40 = 1.99 |

The QL iteration is two-thirds of the time, and most of that is the single
thread's chain of dependent operations. Shortening the chain (one `rsqrt` in
place of a square root and two divides) changed it by only 5%. Cutting threadgroup
memory from 17.7 to 4.7 KB per matrix made the same N = 32 kernel **3x
faster**, with the tridiagonalization alone 5x faster. The cost is latency,
so it falls with the number of matrices resident on a core, and threadgroup
memory sets that number.

**Next step for N = 48-128:** keep each thread's row of $A$, and later of $Z$,
in registers, with only the broadcast vectors in threadgroup memory (about
1.5 KB). The tridiagonalization's products are row-local with broadcast
operands, and the QL updates are row-local, so the only cross-thread work is
the column dots when forming Q (`simd_sum`). The N ≤ 32 result suggests a
similar 2-3x, which would put N = 64 at about 2x the 18-core CPU. That is an
estimate, not a measurement. Above about 128 the registers run out and the
block Jacobi backend, or the CPU, stays the answer for now.

**The SVD counterpart** follows the same pattern: Householder
bidiagonalization, then implicit-shift bidiagonal QR (`sbdsqr`) with each
thread owning a row of U and of V. It was built in 2.10.0 (`golub_kahan`,
see the note at the top); keeping V in device memory rather than beside the
matrix in threadgroup memory was what made it pay. QR already uses the
efficient algorithm on the GPU, and its gains are those of sections 2 and 3.

## 5. Large single matrices

Where one large `eigh` with eigenvectors spends its time (`exp_large`). The
reduction is the `eigvalsh` time less `ssterf`. The back-transformation is the
remainder. The floor is reading the trailing lower triangle once per column,
$N^3/6$ floats, at 282 GB/s.

| N | tridiag backend | GPU reduction | floor | × floor | `sstedc`, CPU | back-transform | CPU path (`ssyevd`) |
|---|---|---|---|---|---|---|---|
| 2048 | 115 | 69 | 20 | 3.4x | 42 | 4 | 234 |
| 4096 | 521 | 308 (59%) | 162 | 1.9x | 175 (34%) | 38 (7%) | 2457 |
| 8192 | 3056 | 2034 (67%) | 1300 | 1.56x | 812 (27%) | 210 (7%) | 18.7 s (`eigh.md`) |

And one large SVD with vectors (`exp_svdlarge`). The floor is twice the
symmetric one, since both $A v$ and $A^T u$ are read each column; `sbdsdc` is
timed on the matrix's own bidiagonal.

| k | bidiag backend | GPU reduction | floor | × floor | `sbdsdc`, CPU | back-transform | CPU path (`sgesdd`) |
|---|---|---|---|---|---|---|---|
| 2048 | 315 | 148 | 81 | 1.8x | 134 (43%) | 33 | 456 |
| 4096 | 1783 | 937 (53%) | 650 | 1.44x | 742 (42%) | 104 (6%) | 3612 |

What each opportunity is worth:

- **Divide and conquer runs on one core.** `sstedc` takes 175 ms at
  N = 4096 with Accelerate's threading on and 187 ms with it off
  (`exp_stedc_threads`). `sbdsdc` is in the same position. That is 27-43% of
  the time with 17 cores and the GPU idle. The independent subproblems of the
  recursion, the secular-equation roots (independent per eigenvalue) and the
  merge GEMMs could all run in parallel. Accelerate exports the building
  blocks (`slaed0`-`slaed9`, `slasd0`-`slasd8`), so a parallel driver is
  possible without reimplementing LAPACK. At 4x on this step, eigh at
  N = 4096 would go from 521 to about 390 ms and the SVD from 1.78 to about
  1.23 s. The effort is high.
- **The reduction sits above its floor by about 3.4-5 µs per dispatch** at
  N = 2048-4096 (seven dispatches per column, so 14,300 and 28,700 of them).
  Fusing the per-column kernels into about three per column, and keeping the
  symmetric product at full bandwidth, could bring it to about 1.2x the floor:
  eigh 4096 521 → ~410 ms, eigh 2048 115 → ~70 ms. That would also lower the
  size from which `tridiag` and `bidiag` beat the CPU (1024 to 2048 today). At
  8192 the gap is 12.8 µs per dispatch, so there the product kernel itself
  falls behind and needs looking at.
- **Two-stage reduction.** LAPACK reduces the dense matrix to a band (all
  matrix products, which suit the GPU), then the band to tridiagonal on the
  CPU. At N = 4096, `ssytrd_sb2st` on the CPU takes 118 ms at bandwidth 16,
  130 at 32 and 327 at 64; the one-stage GPU reduction takes 308. With the
  first stage on the GPU (91.6 GFLOP, an estimated 40-80 ms at bandwidth 16),
  the reduction would take about 160-200 ms, or less if the two stages were
  pipelined. That suits `eigvalsh` (est. 1.4-1.6x at large N), and `svdvals`
  by analogy, though the bidiagonal case was not measured.
  With vectors, the second stage's reflectors must also be applied, which
  takes back most of the gain.
- **Pipelining batches of large matrices**: section 3.

Two ideas measured and dropped:

- **MRRR instead of divide and conquer.** Accelerate's `sstemr` is 6.8-9.5x
  slower than `sstedc` (1.51 s against 0.18 s at N = 4096) and 40-90x less
  orthogonal in FP32 (1e-4 against 3e-6) (`exp_mrrr`).
- **Bisection on 18 cores instead of `ssterf`** for eigenvalues alone. It
  takes all 18 cores to match one core of `ssterf` (75.6 against 79.3 ms at
  4096), and it is less accurate (1.3e-5) (`exp_bisect`). `ssterf` is 13-23%
  of `eigvalsh`'s time, so a GPU bisection would be worth at most that.

## Method and limits

- One machine. The M1 and other chips are unmeasured. Their balance of CPU
  cores to GPU cores differs, which is why each change would go through the
  per-device tuning, as the routing does now.
- The CPU + GPU split searched five fractions per shape, so its gains are a
  lower bound.
- `exp_batch` was run twice. The first run warmed up with one call, which left
  the GPU at low clocks for sub-millisecond shapes (eigh 8×8 ×4096 read
  1.76 ms instead of 0.50). All the tables use the second run, with 250 ms of
  warm-up. Both runs are in [`results.txt`](performance-headroom/results.txt).
- The prototype kernel is a measurement device, not a backend. It omits
  scaling, NaN handling and the `info` word, and has been checked on Gaussian
  input only.
- Estimates are marked as such. Everything else is in `results.txt`, produced
  by the programs beside it:

```sh
docs/studies/performance-headroom/build.sh build    # library build dir; binaries go to $TMPDIR
cd ${TMPDIR:-/tmp}/metal-linalg-headroom
./exp_batch 18 && ./exp_gemm && ./exp_large && ./exp_svdlarge && ./exp_tdql
./exp_stage tdql64.metallib 64 && ./exp_stage tdql32.metallib 32
./exp_mrrr && ./exp_bisect && ./exp_stedc_threads && ./exp_routes
```
