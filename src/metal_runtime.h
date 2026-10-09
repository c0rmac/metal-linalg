#pragma once

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include <metal_linalg/core.h>
#include "divide_conquer.h"

#include <dispatch/dispatch.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace metal_linalg::detail {

using core::Matrices;

// Rounds `v` up to the next multiple of `multiple`.
uint pad_up(uint v, uint multiple);

// Floats in a batch of matrices.
inline size_t element_count(const Matrices& a) { return (size_t)a.batch * a.rows * a.cols; }

// -----------------------------------------------------------------------------
// Inputs
// -----------------------------------------------------------------------------

// The matrices as a Metal buffer the kernels can read. Memory that starts on
// a page boundary (an MLX array's, or a large malloc) is wrapped in place;
// anything else is copied once. With `transpose`, always a copy, holding each
// matrix transposed: [batch, cols, rows].
id<MTLBuffer> input_buffer(id<MTLDevice> device, const Matrices& a, bool transpose = false);

// Which entries of each matrix scan() reads. The eigensolvers read one
// triangle, and junk in the other must not set the scale.
enum class Part { all, lower, upper };

// Largest magnitude and finiteness of each matrix, on the CPU.
void scan(const Matrices& a, Part part, float* amax, char* finite);

// As input_buffer, with every matrix scaled by a power of two so that its
// largest entry lies in [0.5, 1).
//
// The Householder kernels compare squared column norms with an absolute
// threshold, and a squared norm under- or overflows float32 well inside the
// range of a float. Unscaled, that made QR lose accuracy from entries around
// 1e-3 (relative error 4e-3 at 64 x 64), fail outright below 1e-5, and return
// NaN above 1e+18. A power of two is exact, so scaling costs no accuracy: Q is
// unchanged and R is recovered by multiplying with `unscale`.
//
// A matrix that is all zero or contains a non-finite entry is left unscaled.
struct ScaledInput {
    id<MTLBuffer>      buffer;
    std::vector<float> unscale;    // [batch]: multiply R (or S) by this
    bool               scaled;     // false if every factor is 1, and unscale can be skipped
    std::vector<char>  nonfinite;  // [batch]: 1 where the matrix holds a NaN or an infinity
};
ScaledInput scaled_input(id<MTLDevice> device, const Matrices& a, bool transpose = false);

// -----------------------------------------------------------------------------
// Outputs and host work
// -----------------------------------------------------------------------------

// Page-aligned host memory, for intermediates the GPU reads in place.
class HostBuffer {
public:
    explicit HostBuffer(size_t floats);
    ~HostBuffer();
    HostBuffer(const HostBuffer&) = delete;
    HostBuffer& operator=(const HostBuffer&) = delete;
    float* data() { return data_; }
private:
    float* data_ = nullptr;
};

// `batch` matrices of `per` floats from src to dst, matrix b multiplied by
// factor[b]. With no factors, a plain copy.
void copy_out(const float* src, float* dst, uint32_t batch, size_t per,
              const std::vector<float>* factor = nullptr);

// Each rows x cols matrix of src, transposed into dst as cols x rows.
void transpose_out(const float* src, float* dst, uint32_t batch, uint32_t rows, uint32_t cols);

// dst[j * ld_dst + i] = scale * src[i * ld_src + j] for i < rows, j < cols:
// one row-major matrix into column-major storage, in cache-sized blocks on
// every core.
void transpose_scaled(const float* src, size_t ld_src, float* dst, size_t ld_dst, uint32_t rows, uint32_t cols,
                      float scale);

// Floats of host work worth a thread of its own.
constexpr size_t kGrain = size_t(1) << 16;

// Runs f(t) for t in [0, tasks), on the CPU's cores when there is more than one.
template <class F>
void parallel_for(size_t tasks, const F& f) {
    if (tasks <= 1) {
        if (tasks == 1) f(0);
        return;
    }
    dispatch_apply_f(tasks, DISPATCH_APPLY_AUTO, (void*)&f,
                     [](void* ctx, size_t t) { (*static_cast<const F*>(ctx))(t); });
}

// Runs f(b) for every matrix b, in parallel across groups of matrices large
// enough to be worth a thread.
template <class F>
void for_each_matrix(uint32_t batch, size_t per, const F& f) {
    const size_t group = std::max<size_t>(1, kGrain / std::max<size_t>(per, 1));
    parallel_for((batch + group - 1) / group, [&](size_t t) {
        const size_t end = std::min<size_t>(batch, (t + 1) * group);
        for (size_t b = t * group; b < end; ++b) f((uint32_t)b);
    });
}

// Runs f(b, r0, r1) over rows [r0, r1) of every matrix b, covering each
// matrix once: whole matrices, grouped, when they are small, and row blocks
// of a matrix in parallel when it is large, so a lone big matrix still uses
// every core.
template <class F>
void for_each_rows(uint32_t batch, uint32_t rows, uint32_t cols, const F& f) {
    const size_t per = (size_t)rows * cols;
    const size_t blocks = std::min<size_t>(rows, std::max<size_t>(1, per / kGrain));
    if (blocks <= 1) {
        for_each_matrix(batch, per, [&](uint32_t b) { f(b, 0u, rows); });
        return;
    }
    const uint32_t step = (uint32_t)((rows + blocks - 1) / blocks);
    const size_t per_matrix = (rows + step - 1) / step;
    parallel_for((size_t)batch * per_matrix, [&](size_t t) {
        const uint32_t b  = (uint32_t)(t / per_matrix);
        const uint32_t r0 = (uint32_t)(t % per_matrix) * step;
        f(b, r0, std::min(rows, r0 + step));
    });
}

// The CPU paths' batch loop: f(b0, b1) solves matrices [b0, b1) with LAPACK,
// and is called on chunks of [0, batch) spread over up to cpu_threads()
// threads, each with Accelerate's own threading off. `per` is the floats in
// one matrix. One matrix, or cpu_threads() == 1, runs f(0, batch) on the
// calling thread with Accelerate's threading as the caller has it.
//
// Splitting the batch was faster at every shape measured on an M5 Pro, from
// 2 matrices of 2048 x 2048 (1.55x) to 4096 of 8 x 8 (14x): Accelerate gains
// little from its own threads on one matrix of these sizes, so whole matrices
// per core use the machine better. On macOS 14, where Accelerate's threading
// cannot be switched off per thread (and in a build with an SDK older than
// macOS 15's, which does not declare the switch), only matrices too small for
// it to thread are split, so the two never compete for the cores.
//
// An exception from f is rethrown on the calling thread once every chunk has
// stopped; chunks not yet started are skipped.
void lapack_batches(uint32_t batch, size_t per, const std::function<void(uint32_t, uint32_t)>& f);
// The same on at most `threads` threads (beside the GPU's host work, say).
void lapack_batches(uint32_t batch, size_t per, unsigned threads, const std::function<void(uint32_t, uint32_t)>& f);
// The threads one matrix of lapack_batches(batch, per, f) can spread its own
// work over (the CPU paths' divide and conquer): the cores its threads leave
// idle, shared out. 1 on share_batch's workers and from a batch of
// cpu_threads() up.
unsigned lapack_threads_per_matrix(uint32_t batch, size_t per);
// The fewest of those threads with which the CPU paths run that divide and
// conquer themselves rather than call LAPACK's drivers. On an M5 Pro (6 + 12
// cores), eigh of 1024 x 1024: 4 matrices with 4 threads each 1.2x faster
// than ssyevd, 6 with 3 1.1x slower, 8 with 2 1.25x slower.
constexpr unsigned kCpuDcMinThreads = 4;

// A batch on the GPU and the CPU at once. gpu(b0, count) solves matrices
// [b0, b0 + count) with a GPU backend, cpu(b0, count) with the CPU path. The
// CPU side is cpu_threads() workers, each taking cpu_chunk matrices at a time
// from the back of the batch and solving them on its own thread with
// Accelerate's threading off (lapack_batches runs inline on them). The GPU
// takes from the front on the calling thread: a quarter of the batch (at least
// gpu_chunk) first, then, from the two rates measured so far, the share of
// what is left that it would finish as the CPU finishes the rest. Each side
// does what its speed earns, with no split to measure in advance.
//
// Where the two are close in speed, the batch takes about the time of the
// harmonic sum: on an M5 Pro the SVD of 4096 matrices of 40 x 40 in 10.2 ms,
// against 16.5 on the GPU alone and 24.2 on the CPU. An exception on either
// side stops both, and the first is rethrown here.
void share_batch(uint32_t batch, uint32_t gpu_chunk, uint32_t cpu_chunk,
                 const std::function<void(uint32_t, uint32_t)>& gpu,
                 const std::function<void(uint32_t, uint32_t)>& cpu);

// -----------------------------------------------------------------------------
// Metal
// -----------------------------------------------------------------------------

// The library is built with ARC; Metal still autoreleases command buffers and
// encoders. One of these at the top of every GPU entry point drains them when
// the call returns, also by an exception, so a caller without a pool of its
// own (a C++ or Python program) does not accumulate them. These two functions
// are what @autoreleasepool compiles to.
extern "C" void* objc_autoreleasePoolPush(void);
extern "C" void  objc_autoreleasePoolPop(void* token);

class AutoreleasePool {
public:
    AutoreleasePool() : token_(objc_autoreleasePoolPush()) {}
    ~AutoreleasePool() { objc_autoreleasePoolPop(token_); }
    AutoreleasePool(const AutoreleasePool&) = delete;
    AutoreleasePool& operator=(const AutoreleasePool&) = delete;
private:
    void* token_;
};

// `floats` floats of page-aligned host memory (a HostBuffer, or an
// allocation known to be aligned) as a Metal buffer, without copying.
id<MTLBuffer> wrap_host(id<MTLDevice> device, float* data, size_t floats);

struct EmbeddedShader;

// Device, command queue and shader library for one embedded metallib
// (src/shaders.h). One instance per shader, shared by every backend that
// uses that library.
//
// This deliberately lives in a single translation unit. Each backend used to
// carry its own copy of this scaffolding under the same namespace; the inline
// members then collided at link time, one definition won for the whole binary,
// and `qr_streaming_amx_complete` silently ran the *reduced* metallib.
struct MetalRuntime {
    id<MTLDevice>       device;
    id<MTLCommandQueue> queue;
    id<MTLLibrary>      library;

    // `tag` only decorates error messages, e.g. "qr_unblocked".
    static MetalRuntime& shared(const EmbeddedShader& shader, const char* tag);
};

// Compiles `name` from `library` into a pipeline state. Pass `constants` for
// functions declared with function constants, or nil for plain kernels.
id<MTLComputePipelineState> make_pipeline(id<MTLDevice> device,
                                          id<MTLLibrary> library,
                                          NSString* name,
                                          MTLFunctionConstantValues* constants);

// The upper triangular T of the compact WY form H(0) ... H(kb-1) = I - V T V^T
// of kb Householder reflectors, as LAPACK's slarft("F", "C"): V (m x kb,
// column-major, ld m) unit lower trapezoidal with its zeros and ones
// explicit, T column-major (ld ldt). From the Gram matrix V^T V, one ssyrk on
// the CPU's matrix units, rather than slarft's matrix-vector products: on an
// M5 Pro 0.1 ms against 1.1 at 4096 x 128, where the back-transformations
// built a block's V and T on the CPU while the GPU applied the last, and the
// CPU's side was the slower.
void compact_wy_t(uint32_t m, uint32_t kb, const float* V, const float* tau, float* T, uint32_t ldt);

// The divide and conquer's large products (divide_conquer.h) as MPS
// products on `queue`, a command buffer each, waited for: for a solve whose
// GPU would otherwise be idle, or with `after`, only once that command buffer
// has completed (before, the CPU's: on the same queue they would wait for
// it). Operands in the buffers added with add_buffer (the solve's output) or
// in the page-aligned memory the solve adds; anything else is left to the
// CPU.
class MpsGemm final : public GpuGemm {
public:
    MpsGemm(id<MTLDevice> device, id<MTLCommandQueue> queue, id<MTLCommandBuffer> after = nil)
        : device_(device), queue_(queue), after_(after) {}
    void add_buffer(id<MTLBuffer> buffer);
    void add(const float* base, size_t floats) override;
    void remove(const float* base) override;
    bool gemm(long m, long n, long k, const float* A, long lda, const float* B, long ldb, float* C, long ldc,
              bool accumulate) override;
private:
    struct Region {
        const char*   base;
        size_t        bytes;
        id<MTLBuffer> buffer;   // a host region's wrapped when a product first needs it
    };
    // The region holding `rows` x `cols` floats from p (column-major, ld),
    // and p's offset in it; nil if none does.
    id<MTLBuffer> find(const float* p, long rows, long cols, long ld, size_t& offset);
    id<MTLDevice>        device_;
    id<MTLCommandQueue>  queue_;
    id<MTLCommandBuffer> after_;
    std::vector<Region>  regions_;
};

// The whole-matrix Jacobi kernels (Eigh_Jacobi, Svd_Jacobi) give a matrix one
// threadgroup for its whole solve. With the display busy, macOS ends a command
// buffer whose threadgroup runs for more than about a quarter of a second
// ("GPU Hang Error"), which on an M5 Pro the eigensolver's threadgroup mode
// reaches from N ~ 400 and the SVD's kernel from 512 x 512. A longer solve is
// split over dispatches of a few rounds each, every matrix resuming where it
// stopped (JacobiState in eigh_jacobi_common.h, 24 bytes a matrix).
constexpr size_t kJacobiStateBytes = 24;
// Rounds per dispatch for a solve the cost model puts at `solve_core_ms` on
// one core, of `rounds` per sweep: 0 (the whole solve in one dispatch) when it
// is under the dispatch target (40 ms, or the environment variable `env`),
// else enough rounds for about that much.
uint32_t jacobi_round_budget(double solve_core_ms, uint32_t rounds, const char* env);
// Runs a split solve: command buffers of `per_buffer` dispatches, each
// encoded by `encode`, until the `count` matrices' states (in `state`, from
// `offset` bytes) are all done, or `max_dispatches`. Throws on a GPU error,
// naming `what`.
void run_split_jacobi(id<MTLCommandQueue> queue, id<MTLBuffer> state, size_t offset, uint32_t count,
                      uint32_t per_buffer, uint32_t max_dispatches,
                      const std::function<void(id<MTLComputeCommandEncoder>)>& encode, const std::string& what);

// Threads for CPU work that runs beside the GPU's (the band chase, the divide
// and conquer): cpu_threads() less the two cores the GPU's host work keeps
// (encoding the next matrix's work in a batch, waiting on the GPU), and at
// least one.
inline unsigned cpu_threads_beside_gpu() {
    const unsigned t = metal_linalg::cpu_threads();
    return t > 2 ? t - 2 : 1u;
}

// The first stage of the two-stage reductions, on the GPU (band_reduce.mm).
// A band width the panel kernels have, 8, 16 or 32: `want` rounded up, or
// with want = 0 the environment variable `env`, else 16.
uint32_t band_width(uint32_t want, const char* env);
// The GPU's blocks of b columns in the general reduction of n columns.
uint32_t band_blocks(uint32_t n, uint32_t b);
// The widest band of at most b the panel kernels take for a matrix of `rows`
// rows (rows b <= 128 * 1024), or 0 if none.
uint32_t band_fit(uint32_t rows, uint32_t b);
// The general reduction's reflectors, kept for the singular vectors, A =
// Q1 B P1^T with Q1 = H_0 H_1 ... and P1 = G_0 G_1 ...: GPU block k's column
// panel H_k = I - V T V^T (V on rows k b .. m - 1) and row panel
// G_k = I - U S U^T (U on columns (k + 1) b .. n - 1). The caller provides
// the buffers and the layout, for band_blocks(n, b) blocks: V and U are
// written row-major at qoff[k] and poff[k] in qv and pv, ld qld[k] and
// pld[k]; T and S row-major (ld 32) at k 1024 in qt and pt. The columns from
// `tail` on are LAPACK's (`steps`): a step's column panel's reflectors stay
// below A's diagonal (sgeqrf's, taus tq); its row panel's, sgelqf's, are
// copied to lq (bk x nr, ld bk; taus tp) before the band is cleared of them.
struct BandKeep;

// A band reduction's progress: its GPU blocks' command buffers (`done[k]`
// for block k, columns k b .. k b + b - 1), and `while_gpu`, if set, run
// once all blocks are queued, before the wait for the last: per-block work
// as they complete. General: block k finishes the band's rows k b .. k b +
// b - 1; symmetric: its lower band's columns k b .. k b + b - 1.
struct BandWatch {
    std::vector<id<MTLCommandBuffer>> done;
    std::function<void(BandWatch&)> while_gpu;
    virtual ~BandWatch() = default;
};

struct BandKeep : BandWatch {
    id<MTLBuffer> qv, pv, qt, pt;
    std::vector<size_t> qoff, poff;
    std::vector<uint32_t> qld, pld;
    uint32_t tail = 0;
    struct Step {
        uint32_t k, bk, nr;
        std::vector<float> tq, tp, lq;
    };
    std::vector<Step> steps;
    // The symmetric reduction's (since 2.17.0): Q1 = H_0 H_1 ... H_tail, GPU
    // block k's H_k = I - V T V^T, V on rows (k + 1) b .. n - 1, written as
    // qv, qoff, qld and qt say (pv, pt unused); the trailing block from
    // `tail` on is LAPACK's ssytrd_sy2sb's, its reflectors below the band in
    // A (kd wide panels, sy2sb_tau their taus, its order n - tail).
    uint32_t sy2sb_kd = 0;
    std::vector<float> sy2sb_tau;
};

// A (m x n column-major, m >= n, in shared storage) to an upper band of width
// b, A = Q B P^T: the band in A's upper band, the rest of A scratch (with
// `keep`, Q's and P's reflectors as above; `watch`, if not keep itself, its
// progress). False if m is too tall for the panel kernels (m b > 128 * 1024).
// The general reduction's last columns from column k on, with LAPACK on the
// CPU (A column-major m x n, ld lda): the same block steps, so that A ends an
// upper band of width b; the batch backend's tail (svd_bidiag_batch.mm).
void band_general_tail(float* A, uint32_t m, uint32_t n, uint32_t lda, uint32_t b, uint32_t k);
// The same keeping its reflectors as band_reduce_general's tail keeps them:
// keep->tail = k and keep->steps (the QR's reflectors stay below the band in
// A, the LQ's are copied out); only those two fields are touched.
void band_general_tail(float* A, uint32_t m, uint32_t n, uint32_t lda, uint32_t b, uint32_t k, BandKeep* keep);

// eigh with eigenvectors for a batch in two stages (svd_bidiag_batch.mm), the
// tridiag_batch backend's from kEighBandMinN (eigh_tridiag.mm): n up to 1024.
void eigh_band_batch(const core::Matrices& a, bool lower, float* w, float* v, uint32_t* info);
// The smallest k (or N) from which bidiag_batch and tridiag_batch reduce in
// two stages with vectors; the band backends with vectors hand them a batch
// of two or more from there (svd_bidiag.mm).
constexpr uint32_t kBatchBandVectorsMinK = 384;

bool band_reduce_general(id<MTLBuffer> A, uint32_t m, uint32_t n, uint32_t lda, uint32_t b,
                         BandKeep* keep = nullptr, BandWatch* watch = nullptr);
// A symmetric A (n x n, both triangles, in shared storage) to a band of width
// b, Q^T A Q: the band in A's lower band. Likewise false if n is too large.
bool band_reduce_symmetric(id<MTLBuffer> A, uint32_t n, uint32_t lda, uint32_t b, BandWatch* watch = nullptr,
                           BandKeep* keep = nullptr);
// The GPU's blocks in the symmetric reduction of order n.
uint32_t band_blocks_symmetric(uint32_t n, uint32_t b);

// The blocked QR (qr_blocked.mm) on the band reduction's panel kernels.
// A (m x n, row-major, ld lda, in place) <- H_j^T A for its column panels'
// H_j = I - V_j T_j V_j^T, b columns each, while (j + 1) b <= min(m, n) and
// m - j b >= 2 b: R in A's upper triangle there, the rest of those columns
// scratch. The panels are applied within aggregates of 128 columns, then
// each aggregate's I - Y Ta Y^T to the columns right of it. Kept: each
// block's V (row-major, at k (ldv + 1) in v, k = j b, zeros above it in v),
// T (ld 32, at j 1024 in t), and each aggregate's Ta (ld 128, 128^2 apart in
// ta). vt (the panels' V T, m x 32), g (128 x 128), z and z2 (128 x ldw,
// ldw >= max(n, K)) and sc (the TSQR's) scratch. A batch of `batch`
// matrices at once, each buffer's matrices its stride (in floats) apart.
struct QrStore {
    id<MTLBuffer> v, vt, t, ta, g, z, z2, sc;
    uint32_t ldv = 0, ldw = 0, batch = 1;
    size_t sa = 0, sv = 0, svt = 0, st = 0, sta = 0, sg = 0, sz = 0, ssc = 0;
};
// The TSQR scratch a panel of m rows needs, in floats.
size_t qr_scratch_floats(uint32_t m);
// The panel width for m rows (the TSQR's 128 * 1024 / b rows at most), 0 if
// none fits.
uint32_t qr_block_width(uint32_t m);
uint32_t qr_blocks_count(uint32_t m, uint32_t n, uint32_t b);
// Encodes the blocks, starting in cb, committing it and the next ones as it
// goes (into `committed`; cb is left the last, uncommitted); returns the
// columns done.
uint32_t qr_blocks(id<MTLCommandBuffer> __strong& cb, id<MTLBuffer> A, uint32_t m, uint32_t n, uint32_t lda,
                   uint32_t b, const QrStore& st, std::vector<id<MTLCommandBuffer>>& committed);
// Q (m x K row-major, ld ldq) <- H_0 ... H_{blocks - 1} Q, by aggregates,
// aggregate a on Q's rows and columns from its first (the rest of those rows
// zero in a Q formed from the identity). V may have more rows than Q, zero
// past Q's (a factored matrix padded with zero rows): they are left out.
void qr_blocks_apply(id<MTLCommandBuffer> cb, id<MTLBuffer> Q, uint32_t m, uint32_t K, uint32_t ldq, uint32_t b,
                     uint32_t blocks, const QrStore& st);
// Each R (k x n row-major, sr apart, k n if 0) = up[i] times A's upper
// triangle (ld lda, sa apart), zeros below.
void qr_r_out(id<MTLCommandBuffer> cb, id<MTLBuffer> A, id<MTLBuffer> R, uint32_t k, uint32_t n, uint32_t lda,
              size_t sa, uint32_t batch, id<MTLBuffer> up, size_t sr = 0);
// Each dst (mp x np, row-major, sd apart) = scale[i] src (m x n, m n
// apart), zeros around it.
void qr_scale_copy(id<MTLCommandBuffer> cb, id<MTLBuffer> src, id<MTLBuffer> dst, uint32_t m, uint32_t n,
                   uint32_t mp, uint32_t np, size_t sd, uint32_t batch, id<MTLBuffer> scale);
// Each src matrix's (per floats, per apart) largest |x| as an IEEE bit
// pattern into bits (zeroed by the caller; at least 0x7f800000 for a NaN or
// infinity), then its scales into down and up (as qr_blocked.mm's host scan
// would set them; 1 for a zero or non-finite matrix).
void qr_scan_scales(id<MTLCommandBuffer> cb, id<MTLBuffer> src, id<MTLBuffer> bits, id<MTLBuffer> down,
                    id<MTLBuffer> up, size_t per, uint32_t batch);
id<MTLCommandQueue> qr_queue();
id<MTLDevice> qr_device();

// By bisection on the GPU (bisect.mm): the eigenvalues of the symmetric
// tridiagonal (d, n; e, n - 1) into w, ascending, or the singular values of
// the upper bidiagonal (d, e) into s, descending. False, and nothing done,
// below the order from which the GPU is the faster: LAPACK's ssterf or
// sbdsqr then.
bool tridiagonal_eigenvalues(uint32_t n, const float* d, const float* e, float* w);
bool bidiagonal_singular_values(uint32_t n, const float* d, const float* e, float* s);

} // namespace metal_linalg::detail
