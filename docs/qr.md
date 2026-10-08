# QR decomposition (`qr`)

```cpp
#include <metal_linalg/qr.h>

// a: MLX array of shape [M, N] or [..., M, N] (real: float32, or cast to it; complex input throws)
// Returns: {Q, R} where Q is [..., M, K] and R is [..., K, N], K = min(M, N)
auto [Q, R] = metal_linalg::qr_accelerated(a);

// mode, as numpy's and torch's (since 2.17.0): "reduced" (the above), "r"
// (R alone, Q an empty array and never formed) or "complete" (Q [..., M, M])
auto [_, R_alone] = metal_linalg::qr_accelerated(a, "r");
auto [Q_square, R_full] = metal_linalg::qr_accelerated(a, "complete");
```

As for the eigensolver and the SVD, each call is routed by a policy measured on
the Mac it runs on: to LAPACK on the CPU when the batch is too small to pay for
a GPU launch, otherwise to one of two GPU kernels. `qr_backend(m, n, batch)`
reports which (`cpu`, `unblocked` or `streaming_reduced`). For example:

```cpp
#include <metal_linalg/qr.h>
#include <mlx/mlx.h>

using namespace mlx::core;

std::vector<float> data = {
     1,  2,  3,  4,
     5,  6,  7,  8,
     9, 10, 11, 12,
    13, 14, 15, 16
};
array A(data.begin(), {4, 4}, float32);

set_default_device(Device::gpu);
auto [Q, R] = metal_linalg::qr_accelerated(A);
eval({Q, R});

// Q: [4, 4] orthogonal, R: [4, 4] upper triangular, A = Q R
```

## The QR Decomposition explained

Given a matrix $A \in \mathbb{R}^{M \times N}$, the QR decomposition factors it as:

$$A = QR$$

where $K = \min(M, N)$.

**Q** $\in \mathbb{R}^{M \times K}$ is a matrix with orthonormal columns. That is, for any two columns $q_i$ and $q_j$:

$$q_i^T q_j = \delta_{ij} = \begin{cases} 1 & \text{if } i = j \\ 0 & \text{if } i \neq j \end{cases}$$

which can be stated compactly as $Q^T Q = I_K$. The columns of $Q$ form an orthonormal basis for the column space of $A$.

**R** $\in \mathbb{R}^{K \times N}$ is upper triangular. Every entry strictly below the main diagonal is zero:

$$R_{ij} = 0 \quad \text{for all } i > j$$

This is the *thin* (or *reduced*) QR decomposition. The full decomposition extends $Q$ to a square $M \times M$ orthogonal matrix, but the thin form is sufficient to reconstruct $A$ and is more compact when $M > N$.

Since 2.17.0 every API takes a `mode`, as `numpy.linalg.qr` and `torch.linalg.qr` do:

| mode | Q | R | |
|---|---|---|---|
| `"reduced"` (the default) | $M \times K$ | $K \times N$ | the thin factors, as above |
| `"r"` | not formed | $K \times N$ | the same R, bit for bit; 1.3-1.7x faster on the GPU, 2.3-2.6x on the CPU ([below](#modes)) |
| `"complete"` | $M \times M$, orthogonal | $M \times N$, zero below row $K$ | Q's first $K$ columns are the thin Q |

In C++ and in Python with MLX `"r"` follows numpy: Python returns R alone, C++
an empty Q with it. The PyTorch package follows torch (an empty Q tensor), the
C API takes `METAL_LINALG_QR_R` with `q` NULL, and Swift a `QrMode`.

For example, given:

```
A = [[ 1,  2,  3 ],
     [ 4,  5,  6 ],
     [ 7,  8,  9 ]]
```

the thin QR decomposition yields:

```
Q = [[-0.123,  0.904,  0.408 ],        R = [[-8.124, -9.601, -11.078 ],
     [-0.492,  0.301, -0.816 ],              [  0.0,   0.905,   1.809 ],
     [-0.862, -0.301,  0.408 ]]              [  0.0,   0.0,     0.0   ]]
```

One can verify $QR = A$ and $Q^T Q = I_3$.

The decomposition is fundamental to solving linear least-squares problems, performing Gram-Schmidt orthogonalisation, and as the core step in the QR algorithm for computing eigenvalues.

### References

- G. H. Golub and C. F. Van Loan, [*Matrix Computations*](https://jhupbooks.press.jhu.edu/title/matrix-computations), 4th ed. Johns Hopkins University Press, 2013. §5.1 — Householder reflections and QR factorisation.
- R. Schreiber and C. Van Loan, ["A storage-efficient WY representation for products of Householder transformations"](https://epubs.siam.org/doi/10.1137/0910005), *SIAM Journal on Scientific and Statistical Computing*, vol. 10, no. 1, pp. 53–57, 1989 — the Compact WY representation used for block updates.

### The Householder Reflection

Both shaders build $Q$ and $R$ by successively applying **Householder reflections**. A Householder reflector is an orthogonal matrix of the form:

$$H = I - \tau v v^T, \quad \tau \in \mathbb{R}, \quad v \in \mathbb{R}^M$$

chosen so that $H x = \mu e_k$ — i.e. it zeros out every entry of a column vector $x$ below position $k$, leaving a single scalar $\mu$ on the diagonal. The sign of $\mu$ is chosen to avoid catastrophic cancellation:

$$\mu = -\text{sign}(\alpha)\|x\|_2$$

where $\alpha = x_k$ is the pivot element. The reflector vector $v$ is then:

$$v_k = 1, \quad v_i = \frac{x_i}{\alpha - \mu} \text{ for } i > k, \quad \tau = \frac{\mu - \alpha}{\mu}$$

Applying $K = \min(M, N)$ reflectors in sequence drives $A$ to upper triangular form:

$$H_K \cdots H_2 H_1 A = R \implies A = H_1 H_2 \cdots H_K R = QR$$

Since each $H_i$ is orthogonal, their product $Q = H_1 H_2 \cdots H_K$ is also orthogonal. Rather than forming this product one reflector at a time, both shaders use the **Compact WY representation** to batch the updates.

### The Compact WY Representation

For a block of $b$ consecutive Householder reflectors, the product can be written as:

$$H_1 H_2 \cdots H_b = I - Y T Y^T$$

where $Y \in \mathbb{R}^{M \times b}$ has the $b$ reflector vectors as its columns, and $T \in \mathbb{R}^{b \times b}$ is an upper triangular matrix constructed recursively:

$$T_{jj} = \tau_j, \quad T_{ij} = -\tau_j \sum_{m=i}^{j-1} T_{im} (y_m^T y_j) \quad \text{for } i < j$$

This lets a full block update be expressed as a pair of matrix multiplications:

$$A \leftarrow A - Y \bigl( T^T (Y^T A) \bigr)$$

which maps directly onto the AMX matrix coprocessor's 8×8 `simdgroup_matrix` tiles.

### Algorithm: `qr_streaming_amx` (multi-kernel, block size $b = 32$)

For large matrices ($M$ or $N \geq 512$), a single-kernel dispatch causes Q-accumulation to bottleneck on a single shader multiprocessor. The streaming variant splits the computation across **four separate kernel dispatches** per block, allowing the GPU scheduler to assign the trailing update across all available cores in parallel.

The block size is widened to $b = 32$ to match the SIMD group width, maximising AMX tile utilisation and reducing the number of host-side dispatch iterations.

**Kernel 0 — Preprocess.** The input is transposed from row-major to column-major and padded with identity blocks. $Q$ is initialised to $I_{M \times M}$.

**Kernel 1 — Panel factorisation.** Dispatched with 1 threadgroup per matrix. For each column $k$ in the current block, 1024 threads cooperatively compute $\tau_k$ and the normalised reflector vector, then apply it to the remaining $b - k - 1$ panel columns. The $\tau$ values and diagonal elements of $R$ are written to global memory for use by subsequent kernels.

**Kernel 2 — T-matrix construction.** Dispatched with 1 threadgroup per matrix. Reads the reflector columns from $A$ and $\tau$ from global memory and builds $T \in \mathbb{R}^{32 \times 32}$ in threadgroup memory using the recursive Compact WY formula. The completed $T$ is written to global memory negated (i.e. $-T$ is stored), so that Kernel 3 can use `simdgroup_multiply_accumulate` (which adds) rather than needing a subtract path.

**Kernel 3 — Grid-parallel trailing update.** Dispatched with one threadgroup per 32-column tile of the trailing submatrix. Each threadgroup independently computes:

$$\text{Phase 1:} \quad Z^T = A_\text{trail}^T \cdot Y$$

$$\text{Phase 2:} \quad Z_\text{final}^T = Z^T \cdot T$$

$$\text{Phase 3:} \quad A_\text{trail} \leftarrow A_\text{trail} + Y \cdot Z_\text{final}^T$$

$Y$ and $T$ are loaded into threadgroup memory (L1 cache) once per tile and reused across all AMX sweeps. The same kernel is reused for Q-accumulation by setting a flag that redirects the target pointer from $A$ to $Q$.

**Kernel 4 — Haar fix.** Ensures the output $Q$ is a uniform sample from the Haar measure on $O(M)$ and that $R$ has non-negative diagonal. For each column $k$ where $R_{kk} < 0$, the signs of column $k$ in both $Q$ and $R$ are flipped. If the resulting $\det(Q) < 0$, the final column is negated to enforce $\det(Q) = +1$, placing $Q$ in $SO(M)$.

### Algorithm: `qr_blocked` (since 2.15.0, block size $b = 16$, aggregates of 128)

The streaming kernels above factor each 32-column panel in one threadgroup
and stream every panel's trailing update through threadgroups a tile at a
time: on an M5 Pro one 4096×4096 ran at 0.8 TFLOP/s. The blocked QR does the
same Householder QR with the two-stage reduction's machinery
([svd.md](svd.md)), its work as matrix products:

1. **Panels of 16 columns**, factored by the band reduction's panel kernels:
   a panel of up to 128 rows in one simdgroup, rows in registers; a taller
   one by TSQR (leaves of 128 rows a simdgroup each, their stacked
   triangles' QR a tree of pairs, the Householder vectors rebuilt from TSQR's
   Q), so that every panel gives the compact $H = I - V T V^T$.
2. **Aggregates of 128 columns.** Inside one, each panel's $H$ is applied to
   the aggregate's columns right of it, $W = (VT)^T C$ and $C \mathrel{-}= V W$;
   the aggregate's $T_a$ is merged from its panels' $T$'s and the Gram matrix
   $Y^T Y$ (a small kernel), and $I - Y T_a Y^T$ is applied to every column
   right of the aggregate as three MPS products, $Z = Y^T C$,
   $W = T_a^T Z$, $C \mathrel{-}= Y W$: rank-128 updates.
3. **Q** from $[I; 0]$ by the aggregates backwards, three products each.

Each matrix is padded with zero rows and columns to whole panels with twice
their width in rows (the panel kernels' need), which changes neither R nor
Q, so the GPU takes every column. Each is factored row-major in place (a
panel's rows are read contiguously), so neither the input nor Q is
transposed. A batch is one pass of kernels: every kernel takes the matrix as
its grid's z, every product is one batched MPS product. The forward pass is
committed a panel, then an aggregate, at a time, so that the GPU starts
while the CPU encodes the rest; Q's formation is queued at once behind an
event the CPU signals after writing Q's start. Up to $2^{22}$ rows (the TSQR
tree's $2^{15}$ leaves of 128); taller matrices go to the streaming kernels.

On an M5 Pro, one 4096×4096 takes 62 ms against the streaming kernels' 231
and the CPU's 634: the forward pass 42 ms (its panels about 25, latency
rather than arithmetic), Q's formation 16 (6 TFLOP/s). Accuracy is LAPACK's
or a little better (reconstruction and orthogonality 2.1e-6 at 4096 against
2.4e-6). See [the proposal](proposals/qr-blocked.md#done-2026-10-07) for
the measurements behind each choice.

### Algorithm: `qr_householder` (since 2.16.0, small and mid-size matrices)

The `unblocked` backend's first kernel gave each matrix a threadgroup and
worked column by column through device memory, a barrier per phase of every
column, one thread building T, the matrix padded to 32 rows and Q to a full
square; around it the CPU scanned and copied the input and copied Q and R
back. Since 2.16.0 the backend is two kernels built as the SVD's
`golub_kahan` and the eigensolver's `ql` are, a matrix to a simdgroup or a
threadgroup, LAPACK's methods; the first kernel is gone.

**In registers, a simdgroup a matrix** (`qr_householder_simd`), up to 32
columns and 128 rows: LAPACK's `sgeqr2` (a Householder reflector a column,
applied to the columns right of it) and `sorg2r` (Q accumulated in place
from the reflectors, backward). Each lane holds its rows (row $s \cdot 32 +$
lane), as the band reduction's panels do. A column's norm is one
`simd_sum`; the dot products of the columns right of it are `simd_sum`s too,
four columns to one (on a `float4`); every update is the lane's own FMAs.
No barrier, no threadgroup memory, four matrices to a threadgroup. A lane's
row is only ever indexed by constants (the loops are expanded by the
preprocessor), so the current column is kept at index 0 by rotating the
row, left a step while factoring and right a step while Q is formed.

**Blocked, a threadgroup a matrix** (`qr_householder_wy`), up to 4096 rows:
LAPACK's `sgeqrf` and `sorgqr` as they run on one core.

1. Panels of 16 columns, R rows a thread (two; four for $n \le 32$), factored
   as above, two barriers a column (the norm, then the dot products with the
   panel's other columns: those right of the column update the panel, those
   left of it give T's new column, slarft's recurrence).
2. Blocks of 32 columns: each panel's $H = I - V T V^T$ applied to the rest
   of its block, the block's T merged from its panels' (from $V^T V$), and
   the block's $H^T$ applied to every column right of it,
   $W = V^T C$, $C \mathrel{-}= V (T^T W)$, as 8×8 `simdgroup_matrix`
   products, two column tiles a simdgroup (each tile of V serving two
   products); the block's rows of R written once it is done.
3. Q from $[I; 0]$ by the blocks backward, $C \mathrel{-}= V (T (V^T C))$, the
   identity and the zeros of Q not yet written made in registers rather than
   read, so that Q is written once (into the caller's Q where it has the
   workspace's shape).

The matrix, padded with zero rows and columns to multiples of 8 (and to K
rounded up to 16), is in a device workspace; one that needs neither padding
nor scaling is read by the first block straight from the input. A simdgroup
for every 64 rows (128 with four rows a thread), and for every 128 columns
of a wide matrix, up to 8; and for a small batch, which would leave the GPU
fewer than about 12 simdgroups a core, up to 16 a matrix, one row a thread
(its panels are a latency-bound chain: one 384×384 in 2.0 ms against 3.0).

Both read the caller's input row-major as it is, find each matrix's scale
and non-finite entries on the GPU, and write Q and R straight out. On an M5
Pro, through MLX: 1024 of 128×128 in 6.4 ms (the blocked QR 17.8, the CPU
25), 256 of 256×256 in 8.3 (17.8, 18.7), 4096 of 64×64 in 5.9 (16.1, 10.8),
4096 of 32×32 in 1.4 (4.2, 2.9). The blocked kernel's time at 128×128 is
half memory traffic (the input in, R out, the panels' loads and stores),
the updates running at about 2.7 TFLOP/s. Tried and slower: a column a lane
in the register kernel (2.2x at 32×32, 6x at 64×64: `simd_sum` is cheap on
this GPU); register instances of 64 columns (the blocked kernel 1.2-1.4x
faster there); the whole matrix in threadgroup memory, a thread a row (1.7x
slower than the blocked kernel at 200×30, 4.6x at 80×80); blocks of 64
columns (no faster than 32). See
[qr-small-kernel.md](proposals/qr-small-kernel.md) and
[qr-mid-size-kernel.md](proposals/qr-mid-size-kernel.md).

## How it works

Two Metal backends and a CPU path handle different regimes, with a dispatcher that selects between them at runtime (a third Metal backend is retained but unused):

**CPU (`qr_cpu`)** — LAPACK's `sgeqrf` and `sorgqr` (Accelerate), after transposing each matrix into the column-major layout LAPACK reads. A batch is spread over every CPU core (since 2.9.0), each core solving whole matrices with Accelerate's own threading off: on an M5 Pro that is 10-12x faster than one matrix at a time for batches of 16×16 to 64×64, and 6-8x for 512×512 and larger. A lone matrix keeps Accelerate's threading. `set_cpu_threads()` or `METAL_LINALG_CPU_THREADS` caps the cores used, for a program that runs several solves at once. A wide matrix (M < N) is factored by its leading M×M block, $A_1 = Q R_1$, and $R_2 = Q^T A_2$ by one matrix product: the same reflectors and the same R as `sgeqrf` on the whole matrix, which Accelerate ran 10-40x slower (since 2.11.0; on an M5 Pro one 64×2048 in 0.09 ms against 1.47, 16 of them in 0.33 ms against 4.2). Before, the GPU was 2-3x faster than this path for small batches of wide matrices, which a rule on k alone sent to the CPU.

**`qr_unblocked`** — The GPU path for small and mid-size matrices, a matrix to a simdgroup or a threadgroup: since 2.16.0 the Householder kernels (`qr_householder`, above), up to 4096 rows; beyond them the blocked QR. (Its own kernel, a threadgroup a matrix walking device memory, was retired in 2.16.0.)

**`qr_streaming_amx_reduced`** — The GPU path for large matrices. Since 2.15.0 it hands every call it can to the blocked QR (`qr_blocked`, above; `QR_BLOCKED=0` turns that off), which beat its own kernels at every shape and batch measured (1.8-3.5x on an M5 Pro). Its own kernels, kept for matrices taller than $2^{22}$ rows: multi-pass panel factorisation with grid-parallel trailing matrix updates, column panels of width 32, the T-matrix for each WY representation, then a grid of threadgroups for the trailing update, Q accumulated at its economic width of `K = min(M, N)` columns by a backward pass.

**`qr_streaming_amx_complete`** — The same panel factorisation, but accumulating the full `M x M` orthogonal factor inside the forward loop and slicing Q down to `K` columns at the end. Measured to be within noise of the reduced backend, so it is no longer dispatched to.

### Modes

Every backend takes the mode, routed exactly as `"reduced"` is. For `"r"` none
of them forms Q: the Householder kernels stop after the factorisation, the
blocked QR skips the backward accumulation, and the CPU path skips `sorgqr`
(except for a wide matrix, whose $R_2 = Q^T A_2$ needs Q). For `"complete"`
with $M > N$, the backward accumulation starts from the $M \times M$ identity
instead of its first $K$ columns, with the same reflectors: Q's first $K$
columns are the reduced Q's, and R gains zero rows. The register kernel holds
at most 32 columns of Q, so `qr_householder` gives a complete Q of a tall
matrix to its blocked kernel; the grid-parallel kernels kept for more than
$2^{22}$ rows accumulate $K$ columns alone, and refuse `"complete"` there.

The modes against each other on an M5 Pro (2.17.0, the MLX API, median of
the routed call, `./build/benchmark_qr --modes`):

| batch × shape | backend | reduced | R alone | complete | R alone, faster by |
|---|---|---|---|---|---|
| 4096 × 32×32 | `unblocked` | 0.84 ms | 0.50 ms | 0.83 ms | 1.70x |
| 4096 × 64×64 | `unblocked` | 3.80 ms | 2.69 ms | 3.81 ms | 1.41x |
| 1024 × 128×128 | `unblocked` | 4.45 ms | 3.06 ms | 4.40 ms | 1.46x |
| 256 × 256×256 | `unblocked` | 6.24 ms | 4.11 ms | 6.32 ms | 1.52x |
| 16 × 1024×1024 | `streaming_reduced` | 21.0 ms | 15.0 ms | 21.1 ms | 1.39x |
| one 4096×4096 | `streaming_reduced` | 60.6 ms | 45.1 ms | 60.7 ms | 1.34x |
| one 8192×512 | `streaming_reduced` | 8.60 ms | 7.49 ms | 27.8 ms | 1.15x |
| one 512×512 | `streaming_reduced` | 2.86 ms | 2.73 ms | 2.85 ms | 1.05x |
| one 128×128 | `cpu` | 0.23 ms | 0.09 ms | 0.23 ms | 2.59x |
| one 256×256 | `cpu` | 0.78 ms | 0.34 ms | 0.80 ms | 2.31x |
| one 64×2048 | `cpu` | 0.09 ms | 0.09 ms | 0.09 ms | none (wide: $R_2$ needs Q) |

For square and wide matrices the complete Q is the reduced one. For tall ones
it costs what its size does: $M^2$ floats a matrix written, and $M/K$ times
the accumulation's work (one 8192×512: 3.2x the reduced call, for a Q 16x
the size).

### Magnitude and nearly dependent columns

Every matrix is scaled by a power of two on the way in, so that its largest
entry lies in [0.5, 1), and R is scaled back on the way out. A power of two is
exact, so this costs no accuracy. It is there because the kernels compare
*squared* column norms with an absolute threshold. Unscaled, and with that
threshold at its former 1e-7, every backend lost accuracy from entries around
1e-3 (relative error 4e-3 at 64×64), failed outright below 1e-5, returned NaN
above 1e+18, and discarded any column tail shorter than 3e-4 of the matrix's
scale, which is real data whenever columns are nearly dependent. All four are
covered by regression tests now (`[ magnitude ]` and
`[ nearly dependent columns ]` in `tests/test_qr.cpp`). The blocked
Householder kernel scales only a matrix whose largest entry is beyond
2^20 or 2^-20 (its sums of squares are plain, without a threshold, and need
no more), which saves it a pass over the matrix.

### Dispatch logic

Two decisions, as for the eigensolver and the SVD. First GPU or CPU, with
`k = min(M, N)` and `w = floor(sqrt(M k))` (k for a square or wide matrix,
more for a tall one):

```
GPU iff  gpu_min_k <= w <= gpu_max_k,  batch * w >= gpu_min_batch_times_k  and  batch >= gpu_min_batch,
     or  sqrt(M k) >= gpu_large_min_k  and  batch <= gpu_large_max_batch       (large matrices)
else CPU
on the GPU, from a batch of share_min_batch: the batch shared with the CPU path
```

**Sharing a batch with the CPU** (since 2.12.0). From a batch of
`share_min_batch` (0: never), a batch that goes to the GPU is solved by the
GPU kernel and the CPU path at once, as the eigensolver's and the SVD's
batched kernels are (see [eigh.md](eigh.md#dispatch)): the GPU takes chunks
from the front of the batch, CPU workers a few matrices at a time from the
back, and they meet wherever their speeds put them. QR's GPU region is where
the two are closest (on an M5 Pro, 1024 matrices of 128×128: 17.7 ms on the
GPU, 24.2 on the CPU, 14.6 shared), and sharing also takes shapes neither wins
alone by much (1024 of 256×256: 80 ms on the GPU, 69 on the CPU, 43 shared).
`qr_shares_batch(m, n, batch)` reports it; `QR_SHARE_MIN_BATCH` overrides it.
The threshold is fitted by `tuning/tune_qr.py` before the GPU-or-CPU boundary,
which is then fitted with sharing in effect.

**The size is `w`, rows and k both** (since 2.16.0; `k` before, and the fields
keep their names). The unblocked backend's kernels take a tall, narrow batch in
parallel by its rows, which a rule on `k` cannot see: on the M5 Pro's run of
2026-10-08 it sent 16 of 1024×64 to the CPU at 1.9x the GPU's time, and
refitted on `w` the row's regret went from 1.037x to 1.024x.

The GPU needs enough work to pay for a launch, so lone and small-batch calls go
to LAPACK. Since 2.9.0 the CPU path also spreads a batch over every core,
which beats the GPU kernels for batches of small and mid-size matrices too,
while one large matrix, which Accelerate threads only weakly, is still faster
on the GPU (on an M5 Pro 1.9x at 2048×2048, 2.2x at 3072×3072 in 2.14; with
2.15.0's blocked QR 5.1x and 7.1x, and batches of 1024×1024 and larger too:
16 of them 1.9x). The product
rule cannot say both, hence the large-matrix clause; `gpu_large_max_batch = 0`
means any batch, and `gpu_large_min_k = 0` turns the clause off. Since 2.15.0
the clause compares `sqrt(M k)`, rows and `k` both, rather than `k`: `k` for
a square or wide matrix, more for a tall one. One 8192×512 takes 8.5 ms on
the GPU's blocked QR and 48 on the CPU, where a rule on `k = 512` sent it to
the CPU; a wide one stays the CPU's longer, its path factoring the leading
square block and the rest by one product. Fitted on the same measurements,
`sqrt(M k)` scored 1.0133x geometric-mean regret (1.0274x held out), the
work's own size `cbrt(max(M, N) k^2)` 1.0225x (1.0314x).

The lower bound `gpu_min_k` (2.10.0; 0 = none) keeps the smallest matrices on
the CPU however large the batch. On an M5 Pro the CPU wins every square batch
of 8×8 to 64×64 measured, up to 16384 matrices (16384 of 16×16: 3.3 ms
against 8.1 on the GPU), while the GPU wins large batches of 128×128 (1024 of
them: 23.7 ms against 24.4), so a product rule alone sent the large batches
of small matrices to the GPU. It was measured at `128 <= k <= 192` with
`batch * k >= 40960` in 2.10.0; since 2.12.0, with a batch shared between the
GPU and the CPU from 64 matrices, the shared GPU route beats the CPU alone for
large batches of small matrices too (1024 of 64×64: 2.6 ms against 3.1;
16384 of 16×16: 2.9 ms against 3.5), and
the measured row is `k <= 128` with `batch * k >= 40960`, no lower bound. Then,
on the GPU, which kernel:

```
k >= m_crossover  ->  qr_streaming_amx_reduced      (on an M5 Pro: 80 for batches below 8, 768 from 8)
otherwise         ->  qr_unblocked
```

The crossover is on `k = min(M, N)` since 2.16.0, and on `M` before. The
unblocked backend's kernels give a matrix a simdgroup or a threadgroup whose
threads hold its rows: its depth is its `k` columns, a panel step a column,
while its rows run in parallel. The blocked QR spreads a matrix over the
whole GPU at several dispatches a panel, which a large matrix repays, or a
small batch (one threadgroup a matrix leaves most of the GPU idle: one
4096×256 takes 13 ms on the unblocked backend, 2.6 on the blocked QR). On
the shapes the GPU takes in the M5 Pro's runs of 2026-10-08, `k` fitted at
1.040x regret and `M` at 1.129x (run `8633ac`); split by batch (the fields
`m_crossover_small_batch`, `m_crossover_large_batch`, `batch_threshold`), at
1.008x held out against 1.032x for one threshold (run `9f2589`). The split's field names
are historical.

Before 2.16.0 the `unblocked` backend's kernel swept a matrix's rows a
column at a time, `M` was its depth, and a tall matrix and its transpose
wanted opposite backends despite sharing both `max(M, N)` and `K = min(M, N)`
(measured on an M1, batch 16, with that kernel):

| shape | `K` | `qr_unblocked` | `qr_streaming_amx_reduced` | winner |
|---|---|---|---|---|
| 2048 x 64 | 64 | 93.06 ms | 9.40 ms | reduced, **9.9x** |
| 64 x 2048 | 64 | 3.95 ms | 5.79 ms | unblocked, **1.5x** |
| 512 x 32 | 32 | 3.83 ms | 1.92 ms | reduced, **2.0x** |
| 32 x 512 | 32 | 1.45 ms | 1.70 ms | unblocked, **1.2x** |

Full measurement study, including why two earlier cross-validated answers were
wrong: [`studies/qr-routing-apple-m1.md`](studies/qr-routing-apple-m1.md).

On the M1, with that kernel, batch did not enter the rule, and neither did
`N`. Both were tried. A batch-dependent threshold and a narrow-`N` special
case each scored well on the grid they were fitted to and then failed on
held-out data -- the narrow-`N` term went from 1.007x on the training half to
a *worse* worst case (1.48x vs 1.25x) on the held-out half. A cost model built
from the actual thread count (`32 * min(ceil(N_pad/8), 32)`) did worse still,
at 1.100x. The batch split is tested on every run, and adopted where it
survives held-out data (the M5 Pro, 2.16.0).

`N` does affect the true crossover -- the first `qr_unblocked` kernel's
threadgroup width scaled with `N` and only saturated past `N ~ 256`, so thin
matrices favoured the grid-parallel backend from a lower `M` -- but no rule
keyed on `N` beat a plain threshold once it was validated honestly.

`qr_streaming_amx_complete` is not dispatched to. It is within noise of
`qr_streaming_amx_reduced` everywhere it was measured (best margin 5.6% against a
7-12% noise floor) while allocating the full `M x M` Q, so it is kept and tested
but unused.

### Signs

A QR factorisation is unique only up to the signs of R's diagonal (with the
matching columns of Q), and the backends do not all choose the same ones. The
CPU path, `qr_unblocked` and the blocked QR keep the Householder reflections'
signs (the CPU path and `qr_unblocked` LAPACK's convention exactly, slarfg's
beta = -sign(alpha) times the norm), so about
half of R's diagonal is negative; the streaming kernels
(`qr_streaming_amx_reduced` where it does not hand the call to the blocked
QR: taller than $2^{22}$ rows, or `QR_BLOCKED=0`) make the diagonal
non-negative and, for square input, flip the last column of Q so that
det(Q) = +1. Every result satisfies `Q R = A`; a caller that needs one
convention, for example to sample Haar-distributed rotations, normalises it:
flip column `i` of Q and row `i` of R wherever `R[i][i] < 0`.

### Tuning

**The crossover is hardware-specific.** `384` was measured on an 8-core Apple
M1 over 421 shapes; on a 20-core M5 Pro it was `512`, then `128` once the
grid-parallel backend handed its calls to the blocked QR (2.15.0), and since
2.16.0's kernels it is on `k`: 80 for batches below 8, 768 from 8. None is
a universal constant. The library ships a table of measured values rather than a formula,
because the crossover depends on both core count and per-core throughput and the
two push in opposite directions across GPU generations: more cores favour the
grid-parallel backend, a faster core favours the single-threadgroup one, and on
the M5 Pro the second effect won.

| GPU | cores | `m_crossover` | GPU or CPU | status |
|---|---|---|---|---|
| Apple M1 | 8 | — | — | measured before 2.9.0, out of date and no longer used since 2.14.0: estimated like any unmeasured Mac (the old row's study: [`studies/qr-routing-apple-m1.md`](studies/qr-routing-apple-m1.md)) |
| Apple M5 Pro | 20 | k: 80 below batch 8, 768 from 8 | GPU iff `w <= 448` and `batch * w >= 1448` (w = floor(sqrt(M k))), or `sqrt(M k) >= 512`; no batch shared with the CPU | measured — run [`20261007-9f2589`](results/apple-m5-pro-20gpu/20261007-9f2589/qr/report.md) |
| anything else | — | estimated | estimated | **estimated** from the M5 Pro's timings ([how](tuning.md#macs-nobody-has-measured)) |

The GPU-or-CPU boundary is measured by every run made since QR had a CPU path;
a device's row sends every call to the GPU until such a run has been submitted
for it (`python3 tuning/run.py --only qr` measures QR alone in about 5
minutes). Against the best backend at each of the 221 shapes of its run
(2.16.0, MLX's buffer cache on), the M5 Pro row scores 1.0200 geometric-mean
regret, worst 1.85x, 1.0302 held out; always the CPU would score 1.442,
always the GPU 1.70. The CPU is the fastest at 104 of the 221 shapes: lone
matrices up to about 384 and small batches. An earlier run of the same day
(`20261007-8633ac`, the cache off), first analysed with the crossover on `M`
and fitted on every shape (the CPU's included), scored 1.091 and sent 1024 of
128×128 to the blocked QR (17.8 ms, against 6.4 on the unblocked backend):
the crossover is now fitted where a GPU kernel beats the CPU. With the
blocked QR (2.15.0) the row scored 1.0133 on its run's 207 shapes, and the
2.12 row 1.0013 on its 185. The M5 Pro row of 2.9.0 was the first measured against the CPU path that
spreads a batch over every core (2.9.0), and against it the CPU was fastest at
151 of the 178 shapes measured. The GPU keeps one or a few large matrices
(1.3x at 1536×1536, 1.9x at 2048×2048, 2.2x at 3072×3072, alone) and large
batches up to `k = 128` (1024 of 128×128: 21 ms against 24 ms on the CPU,
through `metal-linalg-torch` with its copies). The rule scores 1.035x against
the best backend at every shape on geometric mean. Its worst case, 2.43x, is a
batch of wide matrices (16 of 64×2048) that the GPU's single-threadgroup kernel
does well on, which a rule on `k` cannot single out; and since the product
`batch * k` also sends very large batches of tiny matrices to the GPU, 10000
of 16×16 take 3.7 ms there against 2.1 ms on the CPU, a shape the grid does not
reach (its batches stop at 1024). Before 2.9.0 this Mac sent every
batch with `batch * k >= 512` to the GPU, measured against one CPU core.

On any GPU without a table entry `qr_policy_source()` reports
`estimated:<name> (from Apple M5 Pro, ...)`: the M5 Pro's timings refitted
for that GPU against its CPU ([how](tuning.md#macs-nobody-has-measured)), so an unmeasured device is visible
rather than silent. The defaults in the row above it remain only for a Mac
with nothing to estimate from, reported as `default:untuned-device (<name>)`.

**To measure another Mac**, run `python3 tuning/run.py`, which measures all
three decompositions in one go (`--only qr` for QR alone); see
[`tuning.md`](tuning.md). Its QR part
re-tests the refinements that failed on an M1 rather than assuming they fail
everywhere (a batch-dependent split could be justified on a GPU with far more
cores), and warns if `M` is no longer the best feature on that hardware, which
would be a structural change rather than a moved threshold. The committed runs
are under [`results/`](results/), one folder per device.

To override the policy without rebuilding, set `QR_M_CROSSOVER` (the kernel
crossover), `QR_GPU_MAX_K`, `QR_GPU_MIN_K`, `QR_GPU_MIN_BATCH_TIMES_K`, `QR_GPU_MIN_BATCH`,
`QR_GPU_LARGE_MIN_K` and `QR_GPU_LARGE_MAX_BATCH` (the GPU-or-CPU boundary),
`QR_SHARE_MIN_BATCH` (sharing a batch with the CPU),
or `QR_DEVICE=gpu` or `cpu` to force one side; or call `set_qr_policy()`:

```cpp
auto p = metal_linalg::qr_policy();
p.m_crossover_small_batch = p.m_crossover_large_batch = 320;
p.gpu_min_batch_times_k = 0;          // every call on the GPU
p.gpu_min_k = 0;
metal_linalg::set_qr_policy(p);
```

`./build/probe_occupancy` reports the pipelines' threadgroup limits (the
Householder kernels' and the streaming kernels'). The policy's
`concurrent_matrices` is the residency of the `unblocked` backend's first
kernel (threadgroups of 5 KB: 6 per core, 120 on a 20-core M5 Pro), kept as
it was for comparison with earlier runs.

### Tests

```sh
cmake --build build --target test_qr
./build/test_qr          # or: ctest --test-dir build
```

`tests/test_qr.cpp` checks each backend, the CPU path included, directly as well as through the dispatcher and the routing policy, verifying output shapes, reconstruction (`Q*R == A`), orthogonality (`Q^T*Q == I`) and upper-triangularity of `R`, across input magnitudes from 1e-30 to 1e+37 and for nearly dependent columns. Its modes section runs every backend in `"r"` and `"complete"` over tall, square and wide shapes: R alone equal to the reduced R, the complete Q orthogonal with the reduced Q as its first columns, and R's rows below $K$ zero.


## Benchmark

```sh
cmake --build build --target benchmark_qr
./build/benchmark_qr
```

It times the dispatched GPU path against MLX's CPU `qr` over two classes of
shape:

| class | shapes (M × N) | batch sizes |
|---|---|---|
| small | 8×8, 16×16, 32×32, 64×64, 128×64, 256×128, 512×256 | 10, 50, 100, 500, 1000, 5000, 10000, 15000 |
| large | 512×512, 1024×512, 5000×5000 | 1, 8, 16, 32 |

and reports both times, the speedup and the reconstruction error
`||QR - A||_F` of each, as the mean of 5 timed runs after 2 warmups. The
largest shapes stop at a per-shape batch limit, the `max_batch` field in
`SMALL_CONFIGS` and `LARGE_CONFIGS` at the top of
`benchmarks/benchmark_qr.cpp`, shown as `—`; 0 removes the limit.

A table from an earlier version of the backends, on an M1, is kept in
[`studies/qr-benchmark-apple-m1.md`](studies/qr-benchmark-apple-m1.md).
