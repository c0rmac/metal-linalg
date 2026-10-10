/* metal-linalg's C API: the decompositions on float buffers, for any language
 * that can call C (Swift, Rust, Python's ctypes, ...). It is the buffer core
 * of core.h behind C types; see there for the contracts in full.
 *
 * Matrices are float32, row-major and contiguous: a batch is its matrices one
 * after another. The caller allocates every output. Calls return when their
 * results are written. A NaN or an infinity in a matrix gives NaN output for
 * that matrix alone.
 */
#ifndef METAL_LINALG_C_API_H
#define METAL_LINALG_C_API_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum metal_linalg_status {
    METAL_LINALG_OK               = 0,
    METAL_LINALG_INVALID_ARGUMENT = 1,  /* a null pointer or a shape the backend cannot take */
    METAL_LINALG_RUNTIME_ERROR    = 2,  /* a GPU failure, or a finite matrix that did not converge */
    METAL_LINALG_OUT_OF_MEMORY    = 3
} metal_linalg_status;

/* The message of the most recent failure on this thread, or "" if there has
 * been none. Valid until the next failure on this thread. */
const char* metal_linalg_last_error(void);

/* ---------------------------------------------------------------------------
 * Decompositions
 * ------------------------------------------------------------------------- */

/* A = Q R with K = min(rows, cols): q [batch, rows, K] with orthonormal
 * columns, r [batch, K, cols] upper triangular. */
metal_linalg_status metal_linalg_qr(const float* a, uint32_t batch, uint32_t rows, uint32_t cols,
                                    float* q, float* r);

/* The same with a mode, as numpy.linalg.qr's and torch.linalg.qr's:
 * METAL_LINALG_QR_REDUCED as above; METAL_LINALG_QR_R, R alone, r
 * [batch, K, cols], q unused (NULL is fine) and Q never formed;
 * METAL_LINALG_QR_COMPLETE, q [batch, rows, rows] square, r
 * [batch, rows, cols] with zero rows below K. */
typedef enum metal_linalg_qr_mode {
    METAL_LINALG_QR_REDUCED  = 0,
    METAL_LINALG_QR_R        = 1,
    METAL_LINALG_QR_COMPLETE = 2,
} metal_linalg_qr_mode;
metal_linalg_status metal_linalg_qr_with_mode(const float* a, uint32_t batch, uint32_t rows, uint32_t cols,
                                              metal_linalg_qr_mode mode, float* q, float* r);

/* A = V diag(w) V^T for symmetric n x n matrices, reading only the lower
 * triangle if `lower` is nonzero, else the upper. w [batch, n] ascending;
 * v [batch, n, n] with the eigenvectors as columns, or NULL for the
 * eigenvalues alone; info [batch] (see below) or NULL. */
metal_linalg_status metal_linalg_eigh(const float* a, uint32_t batch, uint32_t n, int lower,
                                      float* w, float* v, uint32_t* info);

/* Thin SVD A = U diag(s) Vt with K = min(rows, cols): u [batch, rows, K],
 * s [batch, K] descending, vt [batch, K, cols]; u and vt both NULL for the
 * singular values alone; info [batch] or NULL. */
metal_linalg_status metal_linalg_svd(const float* a, uint32_t batch, uint32_t rows, uint32_t cols,
                                     float* u, float* s, float* vt, uint32_t* info);

/* A = L L^T for symmetric positive definite n x n matrices (since 2.18.0),
 * reading the lower triangle, or with `upper` nonzero the upper one and
 * writing U = L^T: l [batch, n, n], the other triangle zero. info [batch] or
 * NULL: 0, or k where the leading minor of order k is not positive definite
 * (as LAPACK's spotrf), and then that matrix's l is all NaN. */
metal_linalg_status metal_linalg_cholesky(const float* a, uint32_t batch, uint32_t n, int upper,
                                          float* l, uint32_t* info);

/* P A = L U with partial pivoting for n x n matrices (since 2.18.0), as
 * LAPACK's sgetrf: lu [batch, n, n] holds U on and above the diagonal and L
 * below it (unit diagonal implied); pivots [batch, n] the row swaps in order,
 * 0-based. info [batch] or NULL: 0, or k where U(k-1, k-1) is exactly zero
 * (the matrix is singular; the factorization completes). */
metal_linalg_status metal_linalg_lu_factor(const float* a, uint32_t batch, uint32_t n, float* lu,
                                           uint32_t* pivots, uint32_t* info);

/* A X = B: b and x [batch, n, nrhs]; info as metal_linalg_lu_factor's, a
 * singular matrix's x all NaN (since 2.18.0). */
metal_linalg_status metal_linalg_solve(const float* a, uint32_t batch, uint32_t n, const float* b,
                                       uint32_t nrhs, float* x, uint32_t* info);

/* A^-1: x [batch, n, n]; info as metal_linalg_lu_factor's, a singular
 * matrix's x all NaN (since 2.18.0). */
metal_linalg_status metal_linalg_inv(const float* a, uint32_t batch, uint32_t n, float* x, uint32_t* info);

/* A X = B with A triangular (since 2.18.0): the lower triangle of a read (the
 * upper with `upper` nonzero; with `unit` nonzero the diagonal taken as ones
 * and not read); b and x [batch, n, nrhs]. */
metal_linalg_status metal_linalg_solve_triangular(const float* a, uint32_t batch, uint32_t n, const float* b,
                                                  uint32_t nrhs, int upper, int unit, float* x);

/* An `info` word: bits 0-15 the sweeps taken, bit 16 converged, bit 17 the
 * matrix held a NaN or an infinity, bit 18 (SVD) rank-deficient. The CPU
 * paths report a converged matrix as one sweep. */
#define METAL_LINALG_INFO_SWEEPS(info)         ((info) & 0xFFFFu)
#define METAL_LINALG_INFO_CONVERGED(info)      (((info) >> 16) & 1u)
#define METAL_LINALG_INFO_NONFINITE(info)      (((info) >> 17) & 1u)
#define METAL_LINALG_INFO_RANK_DEFICIENT(info) (((info) >> 18) & 1u)

/* ---------------------------------------------------------------------------
 * Metal buffers
 * ------------------------------------------------------------------------- */

/* The CPU address of `bytes` bytes at byte `offset` in a Metal buffer, so
 * that the functions above can read their input from, and write their outputs
 * to, memory a GPU framework owns (a PyTorch MPS tensor's, say) without a
 * copy. `buffer` is an Objective-C object as a pointer, an id<MTLBuffer> for
 * an address, or NULL; a pointer that is not a heap block (a buffer's
 * contents, say) also gives NULL. NULL is returned unless the object is a
 * buffer in shared storage (which the CPU can read and write) holding the
 * whole range. The caller makes sure no GPU work on the buffer
 * is pending first (torch.mps.synchronize(), say): the decompositions read
 * and write it from their own command queue and from the CPU. */
void* metal_linalg_buffer_contents(const void* buffer, uint64_t offset, uint64_t bytes);

/* Tells the decompositions that `contents` is where the Metal buffer
 * `buffer` (an id<MTLBuffer> as a pointer, in shared storage) starts, until
 * metal_linalg_forget_buffer with the same two: their GPU backends then use
 * that buffer for memory starting there, rather than wrapping the memory in
 * a new buffer, whose pages the first command buffer using it has to map
 * (about 1 ms for 64 MB on an M5 Pro). The caller keeps the buffer alive
 * meanwhile. Returns 1 if it was registered, 0 if not (the same checks as
 * metal_linalg_buffer_contents, and its contents must be `contents`), when
 * there is nothing to forget. */
int metal_linalg_know_buffer(const void* contents, const void* buffer);
void metal_linalg_forget_buffer(const void* contents, const void* buffer);

/* ---------------------------------------------------------------------------
 * Device and routing
 * ------------------------------------------------------------------------- */

/* The default Metal device's name, e.g. "Apple M5 Pro" ("" if none), and its
 * GPU core count (0 if unknown). */
const char* metal_linalg_device_name(void);

/* Calibration notices on stderr (see set_calibration_notices in device.h):
 * on by default; 0 turns them off. METAL_LINALG_NO_CALIBRATION_NOTICE=1 also does. */
void metal_linalg_set_calibration_notices(int enabled);
/* The notice for "QR", "eigh", "SVD", "Cholesky", "LU" or "triangular solve" on this Mac, or "" if
 * its calibration is current: for a binding that reports it its own way
 * (Python warns). Valid until the next call on this thread. */
const char* metal_linalg_calibration_message(const char* decomposition);
uint32_t    metal_linalg_gpu_core_count(void);

/* CPU threads the CPU paths spread a batch over (see set_cpu_threads in
 * device.h): every core by default; 0 restores that, and
 * METAL_LINALG_CPU_THREADS=n sets it from the environment. */
void     metal_linalg_set_cpu_threads(uint32_t n);
uint32_t metal_linalg_cpu_threads(void);

/* CPU only, for the calling thread (since 2.19.0; see set_cpu_only in
 * device.h): while non-zero, this thread's calls take their CPU paths and
 * never the GPU, whatever the policies and the *_DEVICE environment variables
 * say, and the *_backend queries below answer the same. Other threads are not
 * affected; off by default. */
void metal_linalg_set_cpu_only(int on);
int  metal_linalg_cpu_only(void);

/* The backend a call of that shape uses under the policy in effect, by name:
 *   QR    "cpu", "unblocked", "streaming_reduced"
 *   eigh  "cpu", "simd", "threadgroup", "block", "tridiag", "ql", "band", "tridiag_batch"
 *   SVD   "cpu", "jacobi", "block_jacobi", "qr_jacobi", "qr_block_jacobi", "bidiag",
 *         "golub_kahan", "qr_golub_kahan", "band", "bidiag_batch"
 *   Cholesky "cpu", "simd", "threadgroup", "blocked"
 *   LU    "cpu", "blocked" (lu_factor, solve and inv alike)
 *   triangular solve "cpu", "blocked"
 * The strings are static. */
const char* metal_linalg_qr_backend(uint32_t rows, uint32_t cols, uint32_t batch);
const char* metal_linalg_eigh_backend(uint32_t n, uint32_t batch);
const char* metal_linalg_eigvalsh_backend(uint32_t n, uint32_t batch);   /* eigenvalues alone */
const char* metal_linalg_svd_backend(uint32_t rows, uint32_t cols, uint32_t batch);
const char* metal_linalg_svdvals_backend(uint32_t rows, uint32_t cols, uint32_t batch);   /* values alone */
const char* metal_linalg_cholesky_backend(uint32_t n, uint32_t batch);
const char* metal_linalg_lu_backend(uint32_t n, uint32_t batch);
const char* metal_linalg_trsm_backend(uint32_t n, uint32_t nrhs, uint32_t batch);

/* The routing policies, field for field as in core.h, which says what each
 * field does. A policy is resolved on first use from the tuned table, then
 * the environment; setting one replaces both. `gpu_cores` and
 * `concurrent_matrices` are informational and ignored when setting. */

#define METAL_LINALG_NO_LIMIT 0xFFFFFFFFu   /* gpu_max_n / gpu_max_k: no cap */

typedef struct metal_linalg_qr_policy {
    uint32_t m_crossover_small_batch;
    uint32_t m_crossover_large_batch;
    uint32_t batch_threshold;
    uint32_t gpu_max_k;
    uint32_t gpu_min_batch_times_k;
    uint32_t gpu_min_batch;
    uint32_t gpu_cores;
    uint32_t concurrent_matrices;
    uint32_t gpu_large_min_k;       /* the GPU also from this sqrt(M k) (0: never), */
    uint32_t gpu_large_max_batch;   /* for batches up to this (0: any) */
    uint32_t gpu_min_k;             /* the rule above only from this k (0: no lower bound) */
    uint32_t share_min_batch;       /* a GPU batch shared with the CPU from this batch (0: never) */
} metal_linalg_qr_policy;

typedef struct metal_linalg_eigh_policy {
    uint32_t simd_max_n;
    uint32_t block_min_n;
    uint32_t block_min_n_batched;
    uint32_t block_min_batch;
    uint32_t gpu_max_n;
    uint32_t gpu_min_batch_times_n;
    uint32_t gpu_min_batch;
    uint32_t gpu_cores;
    uint32_t values_gpu_max_n;               /* eigenvalues alone; values_gpu_min_batch = 0: */
    uint32_t values_gpu_min_batch_times_n;   /* as for eigenvectors */
    uint32_t values_gpu_min_batch;
    uint32_t tridiag_min_n;          /* the tridiag backend instead of the CPU from this N; */
    uint32_t values_tridiag_min_n;   /* 0: never */
    uint32_t ql_min_n;               /* the ql backend on the GPU for N in [ql_min_n, ql_max_n]; */
    uint32_t ql_max_n;               /* ql_max_n = 0: never */
    uint32_t tridiag_max_batch;          /* tridiag only for batches up to this; */
    uint32_t values_tridiag_max_batch;   /* 0: any batch */
    uint32_t share_min_batch;        /* a ql batch shared with the CPU from this batch (0: never) */
    uint32_t gpu_big_batch_max_n;    /* the GPU also for N up to this in a batch of at least */
    uint32_t gpu_big_batch_min;      /* gpu_big_batch_min (0: never) */
    uint32_t values_band_min_n;      /* eigenvalues alone: the band backend from this N (0: never) */
    uint32_t values_band_width;      /* ... its band's width, 8, 16 or 32 (0: 16) */
    uint32_t band_min_n;             /* with eigenvectors: the band backend from this N (0: never) */
    uint32_t tridiag_batch_min_n;    /* the tridiag_batch backend instead of the CPU for N in */
    uint32_t tridiag_batch_max_n;    /* [min_n, max_n] (max_n 0: never) in a batch of at least */
    uint32_t tridiag_batch_min_batch;            /* min_batch; since 2.17.0 */
    uint32_t values_tridiag_batch_min_n;         /* the same for eigenvalues alone */
    uint32_t values_tridiag_batch_max_n;
    uint32_t values_tridiag_batch_min_batch;
    uint32_t share_min_n;            /* ... share_min_batch only for N of at least this (0: any N); 2.17.0 */
} metal_linalg_eigh_policy;

typedef struct metal_linalg_svd_policy {
    uint32_t qr_min_rows;
    uint32_t qr_min_k;
    uint32_t block_min_k;
    uint32_t block_min_k_batched;
    uint32_t block_min_batch;
    uint32_t gpu_max_k;
    uint32_t gpu_min_batch_times_k;
    uint32_t gpu_min_batch;
    uint32_t gpu_cores;
    uint32_t bidiag_min_k;          /* the bidiag backend instead of the CPU from this k; */
    uint32_t values_bidiag_min_k;   /* 0: never */
    uint32_t bidiag_max_batch;          /* bidiag only for batches up to this; */
    uint32_t values_bidiag_max_batch;   /* 0: any batch */
    uint32_t gk_min_k;              /* the golub_kahan backend on the GPU for k in [gk_min_k, gk_max_k]; */
    uint32_t gk_max_k;              /* gk_max_k = 0: never */
    uint32_t gpu_max_l;             /* the GPU only up to this max(M, N) (0xFFFFFFFF: no cap) */
    uint32_t values_gpu_max_k;              /* the GPU-or-CPU rule for singular values alone; */
    uint32_t values_gpu_min_batch_times_k;  /* values_gpu_min_batch = 0: as with vectors */
    uint32_t values_gpu_min_batch;
    uint32_t values_gpu_max_l;
    uint32_t share_min_batch;       /* a golub_kahan batch shared with the CPU from this batch (0: never) */
    uint32_t gpu_big_batch_max_k;   /* the GPU also for k up to this in a batch of at least */
    uint32_t gpu_big_batch_min;     /* gpu_big_batch_min (0: never) */
    uint32_t values_band_min_k;     /* singular values alone: the band backend from this k (0: never) */
    uint32_t values_band_width;     /* ... its band's width, 8, 16 or 32 (0: 16) */
    uint32_t band_min_k;            /* with vectors: the band backend from this k (0: never) */
    uint32_t bidiag_batch_min_k;    /* the bidiag_batch backend instead of the CPU for k in */
    uint32_t bidiag_batch_max_k;    /* [min_k, max_k] (max_k 0: never) and max(M, N) up to */
    uint32_t bidiag_batch_min_batch;        /* max_l in a batch of at least min_batch; */
    uint32_t bidiag_batch_max_l;            /* since 2.17.0 */
    uint32_t values_bidiag_batch_min_k;     /* the same for singular values alone */
    uint32_t values_bidiag_batch_max_k;
    uint32_t values_bidiag_batch_min_batch;
    uint32_t values_bidiag_batch_max_l;
    uint32_t share_min_k;           /* ... share_min_batch only for k of at least this (0: any k); 2.17.0 */
} metal_linalg_svd_policy;

typedef struct metal_linalg_cholesky_policy {   /* since 2.18.0 */
    uint32_t simd_max_n;             /* the GPU kernels: simd up to this n (at most 32), */
    uint32_t blocked_min_n;          /* blocked from this n for batches up to */
    uint32_t blocked_max_batch;      /* this (0: any), else threadgroup */
    uint32_t gpu_max_n;              /* GPU or CPU: the GPU iff gpu_min_n <= n <= gpu_max_n, */
    uint32_t gpu_min_batch_times_n;  /* batch * n >= this and */
    uint32_t gpu_min_batch;          /* batch >= this; */
    uint32_t gpu_min_n;
    uint32_t gpu_large_min_n;        /* or n >= this (0: never) in a batch of at most */
    uint32_t gpu_large_max_batch;    /* this (0: any) */
    uint32_t gpu_cores;              /* informational */
} metal_linalg_cholesky_policy;

typedef struct metal_linalg_lu_policy {   /* since 2.18.0 */
    uint32_t gpu_min_n;           /* the GPU iff n >= this (0: never) */
    uint32_t gpu_max_batch;       /* in a batch of at most this (0: any) */
    uint32_t gpu_solve_min_rhs;   /* solve: the GPU's triangular solves from this many right-hand sides */
    uint32_t gpu_cores;           /* informational */
} metal_linalg_lu_policy;

typedef struct metal_linalg_trsm_policy {   /* since 2.18.0 */
    uint32_t gpu_min_n;       /* the GPU iff n >= this (0: never), */
    uint32_t gpu_min_rhs;     /* nrhs >= this */
    uint32_t gpu_max_batch;   /* and batch <= this (0: any) */
    uint32_t gpu_cores;       /* informational */
} metal_linalg_trsm_policy;

metal_linalg_qr_policy       metal_linalg_qr_policy_get(void);
metal_linalg_eigh_policy     metal_linalg_eigh_policy_get(void);
metal_linalg_svd_policy      metal_linalg_svd_policy_get(void);
metal_linalg_cholesky_policy metal_linalg_cholesky_policy_get(void);
metal_linalg_lu_policy       metal_linalg_lu_policy_get(void);
metal_linalg_trsm_policy     metal_linalg_trsm_policy_get(void);

void metal_linalg_qr_policy_set(const metal_linalg_qr_policy* p);
void metal_linalg_eigh_policy_set(const metal_linalg_eigh_policy* p);
void metal_linalg_svd_policy_set(const metal_linalg_svd_policy* p);
void metal_linalg_cholesky_policy_set(const metal_linalg_cholesky_policy* p);
void metal_linalg_lu_policy_set(const metal_linalg_lu_policy* p);
void metal_linalg_trsm_policy_set(const metal_linalg_trsm_policy* p);

/* Where the policy in effect came from: "tuned:<device>", "estimated:<device>
 * (from <measured device>, ...)" on a Mac nobody has measured, "env:<variables>",
 * "user" or "default:untuned-device (<device>)". Valid until that policy is
 * next set. */
const char* metal_linalg_qr_policy_source(void);
const char* metal_linalg_eigh_policy_source(void);
const char* metal_linalg_svd_policy_source(void);
const char* metal_linalg_cholesky_policy_source(void);
const char* metal_linalg_lu_policy_source(void);
const char* metal_linalg_trsm_policy_source(void);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* METAL_LINALG_C_API_H */
