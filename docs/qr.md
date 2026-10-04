# QR decomposition (`qr`)

```cpp
#include <metal_linalg/qr.h>

// a: MLX array of shape [M, N] or [..., M, N] (real: float32, or cast to it; complex input throws)
// Returns: {Q, R} where Q is [..., M, K] and R is [..., K, N], K = min(M, N)
auto [Q, R] = metal_linalg::qr_accelerated(a);
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

### Algorithm: `qr_unblocked` (single-kernel, block size $b = 16$)

This kernel processes the entire matrix in one GPU dispatch. It operates on matrices stored in **column-major** format to align with AMX load/store strides, and pads dimensions to multiples of 32 (rows) and 16 (columns) to eliminate in-kernel boundary branching.

For each block of $b = 16$ columns starting at column $s$:

**Step 1 — Panel factorisation.** For each column $k = 0, \ldots, b-1$ within the block:

1. All 1024 threads cooperatively compute $\|x_\text{tail}\|^2$ via a two-phase threadgroup reduction (intra-SIMD via `simd_sum`, then inter-SIMD via shared memory).
2. Thread 0 computes $\mu$, $\tau$, and the scale $1/(\alpha - \mu)$ and broadcasts them through threadgroup memory.
3. Each thread normalises its portion of the tail: $A[r, k] \leftarrow A[r, k] / (\alpha - \mu)$ for $r > k$.
4. The reflector is applied to the remaining $b - k - 1$ columns of the panel: for each $j > k$, $a_j \leftarrow a_j - \tau (v_k^T a_j) v_k$, with the dot product accumulated via threadgroup reduction.

**Step 2 — Form T.** The $b \times b$ upper triangular matrix $T$ is built in threadgroup memory using the recursive formula above.

**Step 3 — Trailing matrix update.** Each SIMD group owns a set of 8-column tiles of the trailing submatrix $A[{:}, s+b{:}]$. Using AMX 8×8 tiles, it computes:

$$Z = Y^T A_\text{trail}, \quad Z \leftarrow T Z, \quad A_\text{trail} \leftarrow A_\text{trail} - Y Z$$

**Step 4 — Q accumulation.** The same WY update is applied to $Q$ (initialised to $I$):

$$Q \leftarrow Q - Y \bigl( T (Y^T Q) \bigr)$$

**Step 5 — Restore diagonal.** The stored $\mu$ values are written back to the diagonal of $A$ (overwriting the temporary $v_k = 1$ sentinel placed there during factorisation).

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

## How it works

Two Metal backends and a CPU path handle different regimes, with a dispatcher that selects between them at runtime (a third Metal backend is retained but unused):

**CPU (`qr_cpu`)** — LAPACK's `sgeqrf` and `sorgqr` (Accelerate), after transposing each matrix into the column-major layout LAPACK reads. A batch is spread over every CPU core (since 2.9.0), each core solving whole matrices with Accelerate's own threading off: on an M5 Pro that is 10-12x faster than one matrix at a time for batches of 16×16 to 64×64, and 6-8x for 512×512 and larger. A lone matrix keeps Accelerate's threading. `set_cpu_threads()` or `METAL_LINALG_CPU_THREADS` caps the cores used, for a program that runs several solves at once. A wide matrix (M < N) is factored by its leading M×M block, $A_1 = Q R_1$, and $R_2 = Q^T A_2$ by one matrix product: the same reflectors and the same R as `sgeqrf` on the whole matrix, which Accelerate ran 10-40x slower (since 2.11.0; on an M5 Pro one 64×2048 in 0.09 ms against 1.47, 16 of them in 0.33 ms against 4.2). Before, the GPU was 2-3x faster than this path for small batches of wide matrices, which a rule on k alone sent to the CPU.

**`qr_unblocked`** — Standard Householder QR in a single kernel dispatch. Used for smaller matrices where the overhead of multi-pass streaming is not worth it.

**`qr_streaming_amx_reduced`** — Multi-pass panel factorisation with grid-parallel trailing matrix updates, designed to saturate the GPU for large matrices. Factorises column panels of width 32, computes the T-matrix for each WY representation, then launches a grid of threadgroups for the trailing update. Q is accumulated directly at its economic width of `K = min(M, N)` columns via a backward pass.

**`qr_streaming_amx_complete`** — The same panel factorisation, but accumulating the full `M x M` orthogonal factor inside the forward loop and slicing Q down to `K` columns at the end. Measured to be within noise of the reduced backend, so it is no longer dispatched to.

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
`[ nearly dependent columns ]` in `tests/test_qr.cpp`).

### Dispatch logic

Two decisions, as for the eigensolver and the SVD. First GPU or CPU, with
`k = min(M, N)`:

```
GPU iff  gpu_min_k <= k <= gpu_max_k,  batch * k >= gpu_min_batch_times_k  and  batch >= gpu_min_batch,
     or  k >= gpu_large_min_k  and  batch <= gpu_large_max_batch       (large matrices)
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

The GPU needs enough work to pay for a launch, so lone and small-batch calls go
to LAPACK. Since 2.9.0 the CPU path also spreads a batch over every core,
which beats the GPU kernels for batches of small and mid-size matrices too,
while one large matrix, which Accelerate threads only weakly, is still faster
on the GPU (on an M5 Pro 1.9x at 2048×2048, 2.2x at 3072×3072). The product
rule cannot say both, hence the large-matrix clause; `gpu_large_max_batch = 0`
means any batch, and `gpu_large_min_k = 0` turns the clause off.

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
M >= m_crossover  ->  qr_streaming_amx_reduced      (384 on an M1, 512 on an M5 Pro)
otherwise         ->  qr_unblocked
```

The crossover is on `M` alone, and rows are **not** interchangeable with columns.
`qr_unblocked` gives each matrix a single threadgroup, which must sweep `M` rows
for every Householder reflection, so `M` is its serial depth; `N` parallelises
across the threadgroup's threads. `qr_streaming_amx_reduced` spreads each matrix
over a grid instead, paying roughly three kernel launches per 32-column panel.

That makes a tall matrix and its transpose want opposite backends despite sharing
both `max(M, N)` and `K = min(M, N)` (measured on an M1, batch 16):

| shape | `K` | `qr_unblocked` | `qr_streaming_amx_reduced` | winner |
|---|---|---|---|---|
| 2048 x 64 | 64 | 93.06 ms | 9.40 ms | reduced, **9.9x** |
| 64 x 2048 | 64 | 3.95 ms | 5.79 ms | unblocked, **1.5x** |
| 512 x 32 | 32 | 3.83 ms | 1.92 ms | reduced, **2.0x** |
| 32 x 512 | 32 | 1.45 ms | 1.70 ms | unblocked, **1.2x** |

Full measurement study, including why two earlier cross-validated answers were
wrong: [`studies/qr-routing-apple-m1.md`](studies/qr-routing-apple-m1.md).

Batch does not enter the rule, and neither does `N`. Both were tried. A
batch-dependent threshold and a narrow-`N` special case each scored well on the
grid they were fitted to and then failed on held-out data -- the narrow-`N` term
went from 1.007x on the training half to a *worse* worst case (1.48x vs 1.25x) on
the held-out half. A cost model built from the actual thread count
(`32 * min(ceil(N_pad/8), 32)`) did worse still, at 1.100x.

`N` does affect the true crossover -- `qr_unblocked`'s threadgroup width scales
with `N` and only saturates past `N ~ 256`, so thin matrices favour the
grid-parallel backend from a lower `M` -- but no rule keyed on `N` beat a plain
threshold once it was validated honestly.

`qr_streaming_amx_complete` is not dispatched to. It is within noise of
`qr_streaming_amx_reduced` everywhere it was measured (best margin 5.6% against a
7-12% noise floor) while allocating the full `M x M` Q, so it is kept and tested
but unused.

### Signs

A QR factorisation is unique only up to the signs of R's diagonal (with the
matching columns of Q), and the backends do not all choose the same ones. The
CPU path and `qr_unblocked` use LAPACK's Householder convention, so about half
of R's diagonal is negative; `qr_streaming_amx_reduced` makes the diagonal
non-negative and, for square input, flips the last column of Q so that
det(Q) = +1. Every result satisfies `Q R = A`; a caller that needs one
convention, for example to sample Haar-distributed rotations, normalises it:
flip column `i` of Q and row `i` of R wherever `R[i][i] < 0`.

### Tuning

**The crossover is hardware-specific.** `384` was measured on an 8-core Apple
M1 over 421 shapes; on a 20-core M5 Pro it is `512`. Neither is a universal
constant. The library ships a table of measured values rather than a formula,
because the crossover depends on both core count and per-core throughput and the
two push in opposite directions across GPU generations: more cores favour the
grid-parallel backend, a faster core favours the single-threadgroup one, and on
the M5 Pro the second effect won.

| GPU | cores | `m_crossover` | GPU or CPU | status |
|---|---|---|---|---|
| Apple M1 | 8 | 384 | always the GPU (measured before the CPU path) | measured before 2.9.0 — see [`studies/qr-routing-apple-m1.md`](studies/qr-routing-apple-m1.md) |
| Apple M5 Pro | 20 | 512 | GPU iff `k <= 128` and `batch * k >= 40960`, or `k >= 1024` and `batch <= 4`; shared with the CPU from batch 64 | measured — run [`20261004-4d6208`](results/apple-m5-pro-20gpu/20261004-4d6208/qr/report.md) |
| anything else | — | 384 | GPU iff `batch * k >= 1024` | **untuned default** |

The GPU-or-CPU boundary is measured by every run made since QR had a CPU path;
a device's row sends every call to the GPU until such a run has been submitted
for it (`python3 tuning/run.py --only qr` measures QR alone in about 4
minutes). Against the best backend at each of the 185 shapes of its run, the
M5 Pro row scores 1.0013 geometric-mean regret, worst 1.17x (2.11.0's row on
the same data: 1.0081, worst 1.45x). The M5 Pro row of 2.9.0 was the first measured against the CPU path that
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

`qr_policy_source()` reports `default:untuned-device (<name>)` on any GPU
without a table entry, so an untuned device is visible rather than silent.

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

`./build/probe_occupancy` reports the threadgroup-memory limits that set how
many matrices `qr_unblocked` keeps resident (6 per core: 48 on an 8-core M1,
120 on a 20-core M5 Pro), which governs whether batch count can matter at all.

### Tests

```sh
cmake --build build --target test_qr
./build/test_qr          # or: ctest --test-dir build
```

`tests/test_qr.cpp` checks each backend, the CPU path included, directly as well as through the dispatcher and the routing policy, verifying output shapes, reconstruction (`Q*R == A`), orthogonality (`Q^T*Q == I`) and upper-triangularity of `R`, across input magnitudes from 1e-30 to 1e+37 and for nearly dependent columns.


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
