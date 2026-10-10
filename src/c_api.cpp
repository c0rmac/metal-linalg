// The C API (c_api.h) over the buffer core (core.h): argument checks, and
// exceptions turned into status codes with the message kept per thread.
#include <metal_linalg/c_api.h>
#include <metal_linalg/core.h>

#include <algorithm>
#include <new>
#include <stdexcept>
#include <string>

using namespace metal_linalg;

namespace {

thread_local std::string g_last_error;

template <class F>
metal_linalg_status guarded(F&& f) {
    try {
        f();
        return METAL_LINALG_OK;
    } catch (const std::invalid_argument& e) {
        g_last_error = e.what();
        return METAL_LINALG_INVALID_ARGUMENT;
    } catch (const std::bad_alloc&) {
        g_last_error = "out of memory";
        return METAL_LINALG_OUT_OF_MEMORY;
    } catch (const std::exception& e) {
        g_last_error = e.what();
        return METAL_LINALG_RUNTIME_ERROR;
    } catch (...) {
        g_last_error = "unknown error";
        return METAL_LINALG_RUNTIME_ERROR;
    }
}

void require(bool ok, const char* what) {
    if (!ok) throw std::invalid_argument(what);
}

// Inputs and required outputs may be null only when they hold nothing.
bool present(const void* p, uint64_t count) { return p != nullptr || count == 0; }

const char* name(QrBackend b) {
    switch (b) {
        case QrBackend::streaming_reduced: return "streaming_reduced";
        case QrBackend::cpu:               return "cpu";
        default:                           return "unblocked";
    }
}

const char* name(EighBackend b) {
    switch (b) {
        case EighBackend::simd:        return "simd";
        case EighBackend::threadgroup: return "threadgroup";
        case EighBackend::block:       return "block";
        case EighBackend::tridiag:     return "tridiag";
        case EighBackend::ql:          return "ql";
        case EighBackend::band:        return "band";
        case EighBackend::tridiag_batch: return "tridiag_batch";
        default:                       return "cpu";
    }
}

const char* name(SvdBackend b) {
    switch (b) {
        case SvdBackend::jacobi:          return "jacobi";
        case SvdBackend::block_jacobi:    return "block_jacobi";
        case SvdBackend::qr_jacobi:       return "qr_jacobi";
        case SvdBackend::qr_block_jacobi: return "qr_block_jacobi";
        case SvdBackend::bidiag:          return "bidiag";
        case SvdBackend::band:            return "band";
        case SvdBackend::golub_kahan:     return "golub_kahan";
        case SvdBackend::qr_golub_kahan:  return "qr_golub_kahan";
        case SvdBackend::bidiag_batch:    return "bidiag_batch";
        default:                          return "cpu";
    }
}

const char* name(LuBackend b) { return b == LuBackend::blocked ? "blocked" : "cpu"; }

const char* name(CholeskyBackend b) {
    switch (b) {
        case CholeskyBackend::simd:        return "simd";
        case CholeskyBackend::threadgroup: return "threadgroup";
        case CholeskyBackend::blocked:     return "blocked";
        case CholeskyBackend::cpu:         return "cpu";
    }
    return "cpu";
}

} // namespace

extern "C" {

const char* metal_linalg_last_error(void) { return g_last_error.c_str(); }

metal_linalg_status metal_linalg_qr(const float* a, uint32_t batch, uint32_t rows, uint32_t cols,
                                    float* q, float* r) {
    return guarded([&] {
        const uint64_t k = std::min(rows, cols);
        require(present(a, (uint64_t)batch * rows * cols), "[qr] a is NULL");
        require(present(q, (uint64_t)batch * rows * k), "[qr] q is NULL");
        require(present(r, (uint64_t)batch * k * cols), "[qr] r is NULL");
        core::qr({a, batch, rows, cols}, q, r);
    });
}

metal_linalg_status metal_linalg_qr_with_mode(const float* a, uint32_t batch, uint32_t rows, uint32_t cols,
                                              metal_linalg_qr_mode mode, float* q, float* r) {
    return guarded([&] {
        require(mode == METAL_LINALG_QR_REDUCED || mode == METAL_LINALG_QR_R || mode == METAL_LINALG_QR_COMPLETE,
                "[qr] mode must be METAL_LINALG_QR_REDUCED, _R or _COMPLETE");
        const core::QrMode m = mode == METAL_LINALG_QR_R          ? core::QrMode::r
                               : mode == METAL_LINALG_QR_COMPLETE ? core::QrMode::complete
                                                                  : core::QrMode::reduced;
        const uint64_t qc = core::qr_q_cols(m, rows, cols), rr = core::qr_r_rows(m, rows, cols);
        require(present(a, (uint64_t)batch * rows * cols), "[qr] a is NULL");
        if (qc) require(present(q, (uint64_t)batch * rows * qc), "[qr] q is NULL");
        require(present(r, (uint64_t)batch * rr * cols), "[qr] r is NULL");
        core::qr({a, batch, rows, cols}, qc ? q : nullptr, r, m);
    });
}

metal_linalg_status metal_linalg_eigh(const float* a, uint32_t batch, uint32_t n, int lower,
                                      float* w, float* v, uint32_t* info) {
    return guarded([&] {
        require(present(a, (uint64_t)batch * n * n), "[eigh] a is NULL");
        require(present(w, (uint64_t)batch * n), "[eigh] w is NULL");
        core::eigh({a, batch, n, n}, lower != 0, w, v, info);
    });
}

metal_linalg_status metal_linalg_cholesky(const float* a, uint32_t batch, uint32_t n, int upper,
                                          float* l, uint32_t* info) {
    return guarded([&] {
        require(present(a, (uint64_t)batch * n * n), "[cholesky] a is NULL");
        require(present(l, (uint64_t)batch * n * n), "[cholesky] l is NULL");
        core::cholesky({a, batch, n, n}, upper != 0, l, info);
    });
}

metal_linalg_status metal_linalg_lu_factor(const float* a, uint32_t batch, uint32_t n, float* lu,
                                           uint32_t* pivots, uint32_t* info) {
    return guarded([&] {
        require(present(a, (uint64_t)batch * n * n), "[lu_factor] a is NULL");
        require(present(lu, (uint64_t)batch * n * n), "[lu_factor] lu is NULL");
        require(present(pivots, (uint64_t)batch * n), "[lu_factor] pivots is NULL");
        core::lu_factor({a, batch, n, n}, lu, pivots, info);
    });
}

metal_linalg_status metal_linalg_solve(const float* a, uint32_t batch, uint32_t n, const float* b,
                                       uint32_t nrhs, float* x, uint32_t* info) {
    return guarded([&] {
        require(present(a, (uint64_t)batch * n * n), "[solve] a is NULL");
        require(present(b, (uint64_t)batch * n * nrhs), "[solve] b is NULL");
        require(present(x, (uint64_t)batch * n * nrhs), "[solve] x is NULL");
        core::solve({a, batch, n, n}, b, nrhs, x, info);
    });
}

metal_linalg_status metal_linalg_inv(const float* a, uint32_t batch, uint32_t n, float* x, uint32_t* info) {
    return guarded([&] {
        require(present(a, (uint64_t)batch * n * n), "[inv] a is NULL");
        require(present(x, (uint64_t)batch * n * n), "[inv] x is NULL");
        core::inv({a, batch, n, n}, x, info);
    });
}

metal_linalg_status metal_linalg_svd(const float* a, uint32_t batch, uint32_t rows, uint32_t cols,
                                     float* u, float* s, float* vt, uint32_t* info) {
    return guarded([&] {
        require(present(a, (uint64_t)batch * rows * cols), "[svd] a is NULL");
        require(present(s, (uint64_t)batch * std::min(rows, cols)), "[svd] s is NULL");
        require((u == nullptr) == (vt == nullptr), "[svd] u and vt must both be given or both be NULL");
        core::svd({a, batch, rows, cols}, u, s, vt, info);
    });
}

const char* metal_linalg_device_name(void) { return device_name(); }
void metal_linalg_set_calibration_notices(int enabled) { set_calibration_notices(enabled != 0); }
const char* metal_linalg_calibration_message(const char* decomposition) {
    thread_local std::string msg;
    msg = decomposition ? calibration_message(decomposition) : std::string();
    return msg.c_str();
}
uint32_t    metal_linalg_gpu_core_count(void) { return gpu_core_count(); }
void        metal_linalg_set_cpu_threads(uint32_t n) { set_cpu_threads(n); }
uint32_t    metal_linalg_cpu_threads(void) { return cpu_threads(); }

const char* metal_linalg_qr_backend(uint32_t rows, uint32_t cols, uint32_t batch) {
    return name(qr_backend(rows, cols, batch));
}
const char* metal_linalg_eigh_backend(uint32_t n, uint32_t batch) {
    return name(eigh_backend(n, batch));
}
const char* metal_linalg_eigvalsh_backend(uint32_t n, uint32_t batch) {
    return name(eigvalsh_backend(n, batch));
}
const char* metal_linalg_svd_backend(uint32_t rows, uint32_t cols, uint32_t batch) {
    return name(svd_backend(rows, cols, batch));
}
const char* metal_linalg_svdvals_backend(uint32_t rows, uint32_t cols, uint32_t batch) {
    return name(svdvals_backend(rows, cols, batch));
}
const char* metal_linalg_cholesky_backend(uint32_t n, uint32_t batch) {
    return name(cholesky_backend(n, batch));
}
const char* metal_linalg_lu_backend(uint32_t n, uint32_t batch) { return name(lu_backend(n, batch)); }

metal_linalg_lu_policy metal_linalg_lu_policy_get(void) {
    const LuPolicy p = lu_policy();
    return {p.gpu_min_n, p.gpu_max_batch, p.gpu_solve_min_rhs, p.gpu_cores};
}

void metal_linalg_lu_policy_set(const metal_linalg_lu_policy* c) {
    if (!c) return;
    LuPolicy p = lu_policy();
    p.gpu_min_n         = c->gpu_min_n;
    p.gpu_max_batch     = c->gpu_max_batch;
    p.gpu_solve_min_rhs = c->gpu_solve_min_rhs;
    set_lu_policy(p);
}

metal_linalg_cholesky_policy metal_linalg_cholesky_policy_get(void) {
    const CholeskyPolicy p = cholesky_policy();
    return {p.simd_max_n, p.blocked_min_n, p.blocked_max_batch, p.gpu_max_n, p.gpu_min_batch_times_n,
            p.gpu_min_batch, p.gpu_min_n, p.gpu_large_min_n, p.gpu_large_max_batch, p.gpu_cores};
}

void metal_linalg_cholesky_policy_set(const metal_linalg_cholesky_policy* c) {
    if (!c) return;
    CholeskyPolicy p = cholesky_policy();
    p.simd_max_n            = c->simd_max_n;
    p.blocked_min_n         = c->blocked_min_n;
    p.blocked_max_batch     = c->blocked_max_batch;
    p.gpu_max_n             = c->gpu_max_n;
    p.gpu_min_batch_times_n = c->gpu_min_batch_times_n;
    p.gpu_min_batch         = c->gpu_min_batch;
    p.gpu_min_n             = c->gpu_min_n;
    p.gpu_large_min_n       = c->gpu_large_min_n;
    p.gpu_large_max_batch   = c->gpu_large_max_batch;
    set_cholesky_policy(p);
}

metal_linalg_qr_policy metal_linalg_qr_policy_get(void) {
    const QrPolicy p = qr_policy();
    return {p.m_crossover_small_batch, p.m_crossover_large_batch, p.batch_threshold,
            p.gpu_max_k, p.gpu_min_batch_times_k, p.gpu_min_batch,
            p.gpu_cores, p.concurrent_matrices, p.gpu_large_min_k, p.gpu_large_max_batch, p.gpu_min_k,
            p.share_min_batch};
}

metal_linalg_eigh_policy metal_linalg_eigh_policy_get(void) {
    const EighPolicy p = eigh_policy();
    return {p.simd_max_n, p.block_min_n, p.block_min_n_batched, p.block_min_batch,
            p.gpu_max_n, p.gpu_min_batch_times_n, p.gpu_min_batch, p.gpu_cores,
            p.values_gpu_max_n, p.values_gpu_min_batch_times_n, p.values_gpu_min_batch,
            p.tridiag_min_n, p.values_tridiag_min_n, p.ql_min_n, p.ql_max_n,
            p.tridiag_max_batch, p.values_tridiag_max_batch, p.share_min_batch,
            p.gpu_big_batch_max_n, p.gpu_big_batch_min, p.values_band_min_n, p.values_band_width,
            p.band_min_n, p.tridiag_batch_min_n, p.tridiag_batch_max_n, p.tridiag_batch_min_batch,
            p.values_tridiag_batch_min_n, p.values_tridiag_batch_max_n, p.values_tridiag_batch_min_batch,
            p.share_min_n};
}

metal_linalg_svd_policy metal_linalg_svd_policy_get(void) {
    const SvdPolicy p = svd_policy();
    return {p.qr_min_rows, p.qr_min_k, p.block_min_k, p.block_min_k_batched, p.block_min_batch,
            p.gpu_max_k, p.gpu_min_batch_times_k, p.gpu_min_batch, p.gpu_cores,
            p.bidiag_min_k, p.values_bidiag_min_k, p.bidiag_max_batch, p.values_bidiag_max_batch,
            p.gk_min_k, p.gk_max_k, p.gpu_max_l,
            p.values_gpu_max_k, p.values_gpu_min_batch_times_k, p.values_gpu_min_batch, p.values_gpu_max_l,
            p.share_min_batch, p.gpu_big_batch_max_k, p.gpu_big_batch_min, p.values_band_min_k, p.values_band_width,
            p.band_min_k, p.bidiag_batch_min_k, p.bidiag_batch_max_k, p.bidiag_batch_min_batch, p.bidiag_batch_max_l,
            p.values_bidiag_batch_min_k, p.values_bidiag_batch_max_k, p.values_bidiag_batch_min_batch,
            p.values_bidiag_batch_max_l, p.share_min_k};
}

// The informational fields keep the detected values.
void metal_linalg_qr_policy_set(const metal_linalg_qr_policy* c) {
    if (!c) return;
    QrPolicy p = qr_policy();
    p.m_crossover_small_batch = c->m_crossover_small_batch;
    p.m_crossover_large_batch = c->m_crossover_large_batch;
    p.batch_threshold         = c->batch_threshold;
    p.gpu_max_k               = c->gpu_max_k;
    p.gpu_min_batch_times_k   = c->gpu_min_batch_times_k;
    p.gpu_min_batch           = c->gpu_min_batch;
    p.gpu_large_min_k         = c->gpu_large_min_k;
    p.gpu_large_max_batch     = c->gpu_large_max_batch;
    p.gpu_min_k               = c->gpu_min_k;
    p.share_min_batch         = c->share_min_batch;
    set_qr_policy(p);
}

void metal_linalg_eigh_policy_set(const metal_linalg_eigh_policy* c) {
    if (!c) return;
    EighPolicy p = eigh_policy();
    p.simd_max_n            = c->simd_max_n;
    p.block_min_n           = c->block_min_n;
    p.block_min_n_batched   = c->block_min_n_batched;
    p.block_min_batch       = c->block_min_batch;
    p.gpu_max_n             = c->gpu_max_n;
    p.gpu_min_batch_times_n = c->gpu_min_batch_times_n;
    p.gpu_min_batch         = c->gpu_min_batch;
    p.values_gpu_max_n             = c->values_gpu_max_n;
    p.values_gpu_min_batch_times_n = c->values_gpu_min_batch_times_n;
    p.values_gpu_min_batch         = c->values_gpu_min_batch;
    p.tridiag_min_n                = c->tridiag_min_n;
    p.values_tridiag_min_n         = c->values_tridiag_min_n;
    p.ql_min_n                     = c->ql_min_n;
    p.ql_max_n                     = c->ql_max_n;
    p.tridiag_max_batch            = c->tridiag_max_batch;
    p.values_tridiag_max_batch     = c->values_tridiag_max_batch;
    p.share_min_batch              = c->share_min_batch;
    p.gpu_big_batch_max_n          = c->gpu_big_batch_max_n;
    p.gpu_big_batch_min            = c->gpu_big_batch_min;
    p.values_band_min_n            = c->values_band_min_n;
    p.values_band_width            = c->values_band_width;
    p.band_min_n                     = c->band_min_n;
    p.tridiag_batch_min_n            = c->tridiag_batch_min_n;
    p.tridiag_batch_max_n            = c->tridiag_batch_max_n;
    p.tridiag_batch_min_batch        = c->tridiag_batch_min_batch;
    p.values_tridiag_batch_min_n     = c->values_tridiag_batch_min_n;
    p.values_tridiag_batch_max_n     = c->values_tridiag_batch_max_n;
    p.values_tridiag_batch_min_batch = c->values_tridiag_batch_min_batch;
    p.share_min_n                    = c->share_min_n;
    set_eigh_policy(p);
}

void metal_linalg_svd_policy_set(const metal_linalg_svd_policy* c) {
    if (!c) return;
    SvdPolicy p = svd_policy();
    p.qr_min_rows           = c->qr_min_rows;
    p.qr_min_k              = c->qr_min_k;
    p.block_min_k           = c->block_min_k;
    p.block_min_k_batched   = c->block_min_k_batched;
    p.block_min_batch       = c->block_min_batch;
    p.gpu_max_k             = c->gpu_max_k;
    p.gpu_min_batch_times_k = c->gpu_min_batch_times_k;
    p.gpu_min_batch         = c->gpu_min_batch;
    p.bidiag_min_k          = c->bidiag_min_k;
    p.values_bidiag_min_k   = c->values_bidiag_min_k;
    p.bidiag_max_batch        = c->bidiag_max_batch;
    p.values_bidiag_max_batch = c->values_bidiag_max_batch;
    p.gk_min_k                = c->gk_min_k;
    p.gk_max_k                = c->gk_max_k;
    p.gpu_max_l               = c->gpu_max_l;
    p.values_gpu_max_k             = c->values_gpu_max_k;
    p.values_gpu_min_batch_times_k = c->values_gpu_min_batch_times_k;
    p.values_gpu_min_batch         = c->values_gpu_min_batch;
    p.values_gpu_max_l             = c->values_gpu_max_l;
    p.share_min_batch              = c->share_min_batch;
    p.gpu_big_batch_max_k          = c->gpu_big_batch_max_k;
    p.gpu_big_batch_min            = c->gpu_big_batch_min;
    p.values_band_min_k            = c->values_band_min_k;
    p.values_band_width            = c->values_band_width;
    p.band_min_k                   = c->band_min_k;
    p.bidiag_batch_min_k            = c->bidiag_batch_min_k;
    p.bidiag_batch_max_k            = c->bidiag_batch_max_k;
    p.bidiag_batch_min_batch        = c->bidiag_batch_min_batch;
    p.bidiag_batch_max_l            = c->bidiag_batch_max_l;
    p.values_bidiag_batch_min_k     = c->values_bidiag_batch_min_k;
    p.values_bidiag_batch_max_k     = c->values_bidiag_batch_max_k;
    p.values_bidiag_batch_min_batch = c->values_bidiag_batch_min_batch;
    p.values_bidiag_batch_max_l     = c->values_bidiag_batch_max_l;
    p.share_min_k                   = c->share_min_k;
    set_svd_policy(p);
}

const char* metal_linalg_qr_policy_source(void)   { return qr_policy_source(); }
const char* metal_linalg_eigh_policy_source(void) { return eigh_policy_source(); }
const char* metal_linalg_svd_policy_source(void)  { return svd_policy_source(); }
const char* metal_linalg_cholesky_policy_source(void) { return cholesky_policy_source(); }
const char* metal_linalg_lu_policy_source(void) { return lu_policy_source(); }

} // extern "C"
