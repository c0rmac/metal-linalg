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

/* An `info` word: bits 0-15 the sweeps taken, bit 16 converged, bit 17 the
 * matrix held a NaN or an infinity, bit 18 (SVD) rank-deficient. The CPU
 * paths report a converged matrix as one sweep. */
#define METAL_LINALG_INFO_SWEEPS(info)         ((info) & 0xFFFFu)
#define METAL_LINALG_INFO_CONVERGED(info)      (((info) >> 16) & 1u)
#define METAL_LINALG_INFO_NONFINITE(info)      (((info) >> 17) & 1u)
#define METAL_LINALG_INFO_RANK_DEFICIENT(info) (((info) >> 18) & 1u)

/* ---------------------------------------------------------------------------
 * Device and routing
 * ------------------------------------------------------------------------- */

/* The default Metal device's name, e.g. "Apple M5 Pro" ("" if none), and its
 * GPU core count (0 if unknown). */
const char* metal_linalg_device_name(void);

/* Calibration notices on stderr (see set_calibration_notices in device.h):
 * on by default; 0 turns them off. METAL_LINALG_NO_CALIBRATION_NOTICE=1 also does. */
void metal_linalg_set_calibration_notices(int enabled);
/* The notice for decomposition "QR", "eigh" or "SVD" on this Mac, or "" if
 * its calibration is current: for a binding that reports it its own way
 * (Python warns). Valid until the next call on this thread. */
const char* metal_linalg_calibration_message(const char* decomposition);
uint32_t    metal_linalg_gpu_core_count(void);

/* The backend a call of that shape uses under the policy in effect, by name:
 *   QR    "cpu", "unblocked", "streaming_reduced"
 *   eigh  "cpu", "simd", "threadgroup", "block", "tridiag"
 *   SVD   "cpu", "jacobi", "block_jacobi", "qr_jacobi", "qr_block_jacobi", "bidiag"
 * The strings are static. */
const char* metal_linalg_qr_backend(uint32_t rows, uint32_t cols, uint32_t batch);
const char* metal_linalg_eigh_backend(uint32_t n, uint32_t batch);
const char* metal_linalg_eigvalsh_backend(uint32_t n, uint32_t batch);   /* eigenvalues alone */
const char* metal_linalg_svd_backend(uint32_t rows, uint32_t cols, uint32_t batch);
const char* metal_linalg_svdvals_backend(uint32_t rows, uint32_t cols, uint32_t batch);   /* values alone */

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
} metal_linalg_svd_policy;

metal_linalg_qr_policy   metal_linalg_qr_policy_get(void);
metal_linalg_eigh_policy metal_linalg_eigh_policy_get(void);
metal_linalg_svd_policy  metal_linalg_svd_policy_get(void);

void metal_linalg_qr_policy_set(const metal_linalg_qr_policy* p);
void metal_linalg_eigh_policy_set(const metal_linalg_eigh_policy* p);
void metal_linalg_svd_policy_set(const metal_linalg_svd_policy* p);

/* Where the policy in effect came from: "tuned:<device>", "env:<variables>",
 * "user" or "default:untuned-device (<device>)". Valid until that policy is
 * next set. */
const char* metal_linalg_qr_policy_source(void);
const char* metal_linalg_eigh_policy_source(void);
const char* metal_linalg_svd_policy_source(void);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* METAL_LINALG_C_API_H */
