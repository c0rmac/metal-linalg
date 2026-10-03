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
        default:                          return "cpu";
    }
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

metal_linalg_status metal_linalg_eigh(const float* a, uint32_t batch, uint32_t n, int lower,
                                      float* w, float* v, uint32_t* info) {
    return guarded([&] {
        require(present(a, (uint64_t)batch * n * n), "[eigh] a is NULL");
        require(present(w, (uint64_t)batch * n), "[eigh] w is NULL");
        core::eigh({a, batch, n, n}, lower != 0, w, v, info);
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
uint32_t    metal_linalg_gpu_core_count(void) { return gpu_core_count(); }

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

metal_linalg_qr_policy metal_linalg_qr_policy_get(void) {
    const QrPolicy p = qr_policy();
    return {p.m_crossover_small_batch, p.m_crossover_large_batch, p.batch_threshold,
            p.gpu_max_k, p.gpu_min_batch_times_k, p.gpu_min_batch,
            p.gpu_cores, p.concurrent_matrices};
}

metal_linalg_eigh_policy metal_linalg_eigh_policy_get(void) {
    const EighPolicy p = eigh_policy();
    return {p.simd_max_n, p.block_min_n, p.block_min_n_batched, p.block_min_batch,
            p.gpu_max_n, p.gpu_min_batch_times_n, p.gpu_min_batch, p.gpu_cores,
            p.values_gpu_max_n, p.values_gpu_min_batch_times_n, p.values_gpu_min_batch,
            p.tridiag_min_n, p.values_tridiag_min_n};
}

metal_linalg_svd_policy metal_linalg_svd_policy_get(void) {
    const SvdPolicy p = svd_policy();
    return {p.qr_min_rows, p.qr_min_k, p.block_min_k, p.block_min_k_batched, p.block_min_batch,
            p.gpu_max_k, p.gpu_min_batch_times_k, p.gpu_min_batch, p.gpu_cores,
            p.bidiag_min_k, p.values_bidiag_min_k};
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
    set_svd_policy(p);
}

const char* metal_linalg_qr_policy_source(void)   { return qr_policy_source(); }
const char* metal_linalg_eigh_policy_source(void) { return eigh_policy_source(); }
const char* metal_linalg_svd_policy_source(void)  { return svd_policy_source(); }

} // extern "C"
