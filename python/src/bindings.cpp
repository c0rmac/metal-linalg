// Python bindings: metal_linalg._core. The public module is python/metal_linalg,
// which wraps these with argument conversion and documentation.
//
// Arrays cross as mlx.core.array without a copy: this module is built in
// nanobind's "mlx" domain, with the nanobind ABI mlx.core was built with, so
// both modules share one registry of types (see python/CMakeLists.txt).

#include <nanobind/nanobind.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>

#include <mlx/mlx.h>

#include <metal_linalg/metal_linalg.h>

#include <stdexcept>
#include <string>

namespace nb = nanobind;
using namespace nb::literals;
namespace ml = metal_linalg;

namespace {

const char* name(ml::QrBackend b) {
    switch (b) {
        case ml::QrBackend::unblocked: return "unblocked";
        case ml::QrBackend::cpu:       return "cpu";
        default:                       return "streaming_reduced";
    }
}
const char* name(ml::EighBackend b) {
    switch (b) {
        case ml::EighBackend::cpu:         return "cpu";
        case ml::EighBackend::simd:        return "simd";
        case ml::EighBackend::threadgroup: return "threadgroup";
        case ml::EighBackend::tridiag:     return "tridiag";
        case ml::EighBackend::ql:          return "ql";
        case ml::EighBackend::band:        return "band";
        case ml::EighBackend::tridiag_batch: return "tridiag_batch";
        default:                           return "block";
    }
}
const char* name(ml::SvdBackend b) {
    switch (b) {
        case ml::SvdBackend::cpu:          return "cpu";
        case ml::SvdBackend::jacobi:       return "jacobi";
        case ml::SvdBackend::block_jacobi: return "block_jacobi";
        case ml::SvdBackend::qr_jacobi:    return "qr_jacobi";
        case ml::SvdBackend::bidiag:       return "bidiag";
        case ml::SvdBackend::band:         return "band";
        case ml::SvdBackend::golub_kahan:  return "golub_kahan";
        case ml::SvdBackend::qr_golub_kahan: return "qr_golub_kahan";
        case ml::SvdBackend::bidiag_batch: return "bidiag_batch";
        default:                           return "qr_block_jacobi";
    }
}
const char* name(ml::LuBackend b) { return b == ml::LuBackend::blocked ? "blocked" : "cpu"; }
const char* name(ml::TrsmBackend b) { return b == ml::TrsmBackend::blocked ? "blocked" : "cpu"; }
const char* name(ml::CholeskyBackend b) {
    switch (b) {
        case ml::CholeskyBackend::simd:        return "simd";
        case ml::CholeskyBackend::threadgroup: return "threadgroup";
        case ml::CholeskyBackend::blocked:     return "blocked";
        default:                               return "cpu";
    }
}

// Policies cross as dicts of their fields. FIELDS lists them once, for both
// directions; an unknown key on the way in is an error, not silently ignored.
#define QR_FIELDS(X) X(m_crossover_small_batch) X(m_crossover_large_batch) X(batch_threshold) \
                     X(gpu_max_k) X(gpu_min_batch_times_k) X(gpu_min_batch)                   \
                     X(gpu_cores) X(concurrent_matrices) X(gpu_large_min_k) X(gpu_large_max_batch) \
                     X(gpu_min_k) X(share_min_batch)
#define EIGH_FIELDS(X) X(simd_max_n) X(block_min_n) X(block_min_n_batched) X(block_min_batch) \
                       X(gpu_max_n) X(gpu_min_batch_times_n) X(gpu_min_batch) X(gpu_cores)      \
                       X(values_gpu_max_n) X(values_gpu_min_batch_times_n) X(values_gpu_min_batch) \
                       X(tridiag_min_n) X(values_tridiag_min_n) X(ql_min_n) X(ql_max_n) X(share_min_batch) \
                       X(gpu_big_batch_max_n) X(gpu_big_batch_min) X(values_band_min_n) X(values_band_width) \
                       X(tridiag_max_batch) X(values_tridiag_max_batch) X(band_min_n) \
                       X(tridiag_batch_min_n) X(tridiag_batch_max_n) X(tridiag_batch_min_batch) \
                       X(values_tridiag_batch_min_n) X(values_tridiag_batch_max_n) X(values_tridiag_batch_min_batch) \
                       X(share_min_n)
#define SVD_FIELDS(X) X(qr_min_rows) X(qr_min_k) X(block_min_k) X(block_min_k_batched)       \
                      X(block_min_batch) X(gpu_max_k) X(gpu_min_batch_times_k) X(gpu_min_batch) \
                      X(gpu_cores) X(bidiag_min_k) X(values_bidiag_min_k)  \
                      X(bidiag_max_batch) X(values_bidiag_max_batch) X(gk_min_k) X(gk_max_k) X(gpu_max_l) \
                      X(values_gpu_max_k) X(values_gpu_min_batch_times_k) X(values_gpu_min_batch) X(values_gpu_max_l) \
                      X(share_min_batch) X(gpu_big_batch_max_k) X(gpu_big_batch_min) X(values_band_min_k) X(values_band_width) \
                      X(band_min_k) X(bidiag_batch_min_k) X(bidiag_batch_max_k) X(bidiag_batch_min_batch) \
                      X(bidiag_batch_max_l) X(values_bidiag_batch_min_k) X(values_bidiag_batch_max_k) \
                      X(values_bidiag_batch_min_batch) X(values_bidiag_batch_max_l) X(share_min_k)
#define CHOLESKY_FIELDS(X) X(simd_max_n) X(blocked_min_n) X(blocked_max_batch) X(gpu_max_n) \
                           X(gpu_min_batch_times_n) X(gpu_min_batch) X(gpu_min_n) X(gpu_large_min_n) \
                           X(gpu_large_max_batch) X(gpu_cores)
#define LU_FIELDS(X) X(gpu_min_n) X(gpu_max_batch) X(gpu_solve_min_rhs) X(gpu_cores)
#define TRSM_FIELDS(X) X(gpu_min_n) X(gpu_min_rhs) X(gpu_max_batch) X(gpu_cores)

#define TO_DICT(f) d[#f] = p.f;
#define FROM_DICT(f) if (key == #f) { p.f = nb::cast<unsigned>(value); return; }

template <typename P, typename Fill>
P from_dict(P p, const nb::dict& d, Fill fill) {
    for (auto [k, v] : d) fill(p, nb::cast<std::string>(k), v);
    return p;
}

} // namespace

NB_MODULE(_core, m) {
    // Registers mlx.core.array with the shared registry before anything here uses it.
    nb::module_::import_("mlx.core");

    m.attr("__version__") = METAL_LINALG_VERSION;
    m.attr("mlx_version") = METAL_LINALG_MLX_VERSION;

    m.def("qr", [](const mlx::core::array& a, const std::string& mode) { return ml::qr_accelerated(a, mode); },
          "a"_a, "mode"_a = "reduced");
    m.def("eigh", &ml::eigh_accelerated, "a"_a, "uplo"_a = "L");
    m.def("eigvalsh", &ml::eigvalsh_accelerated, "a"_a, "uplo"_a = "L");
    m.def("svd", &ml::svd_accelerated, "a"_a);
    m.def("svdvals", &ml::svdvals_accelerated, "a"_a);
    m.def("cholesky", &ml::cholesky_accelerated, "a"_a, "upper"_a = false);
    m.def("cholesky_ex", [](const mlx::core::array& a, bool upper) {
        ml::CholeskyResult r = ml::cholesky_ex_accelerated(a, upper);
        return std::make_pair(r.l, r.info);
    }, "a"_a, "upper"_a = false);
    m.def("lu_factor", [](const mlx::core::array& a) {
        ml::LuResult r = ml::lu_factor_ex_accelerated(a);
        return std::make_tuple(r.lu, r.pivots, r.info);
    }, "a"_a);
    m.def("solve", [](const mlx::core::array& a, const mlx::core::array& b) {
        ml::SolveResult r = ml::solve_ex_accelerated(a, b);
        return std::make_pair(r.x, r.info);
    }, "a"_a, "b"_a);
    m.def("solve_triangular", &ml::solve_triangular_accelerated, "a"_a, "b"_a, "upper"_a = false,
          "unit_diagonal"_a = false);
    m.def("inv", [](const mlx::core::array& a) {
        ml::SolveResult r = ml::inv_ex_accelerated(a);
        return std::make_pair(r.x, r.info);
    }, "a"_a);

    m.def("device_name", [] { return std::string(ml::device_name()); });
    m.def("gpu_core_count", &ml::gpu_core_count);
    m.def("set_cpu_threads", &ml::set_cpu_threads, "n"_a);
    m.def("cpu_threads", &ml::cpu_threads);
    m.def("set_calibration_notices", &ml::set_calibration_notices, "enabled"_a);
    m.def("calibration_message", [](const std::string& what) { return ml::calibration_message(what.c_str()); },
          "decomposition"_a);

    m.def("qr_backend", [](unsigned rows, unsigned cols, unsigned batch) {
        return name(ml::qr_backend(rows, cols, batch)); }, "m"_a, "n"_a, "batch"_a = 1);
    m.def("eigh_backend", [](unsigned n, unsigned batch) {
        return name(ml::eigh_backend(n, batch)); }, "n"_a, "batch"_a = 1);
    m.def("eigvalsh_backend", [](unsigned n, unsigned batch) {
        return name(ml::eigvalsh_backend(n, batch)); }, "n"_a, "batch"_a = 1);
    m.def("svd_backend", [](unsigned rows, unsigned cols, unsigned batch) {
        return name(ml::svd_backend(rows, cols, batch)); }, "m"_a, "n"_a, "batch"_a = 1);
    m.def("svdvals_backend", [](unsigned rows, unsigned cols, unsigned batch) {
        return name(ml::svdvals_backend(rows, cols, batch)); }, "m"_a, "n"_a, "batch"_a = 1);
    m.def("cholesky_backend", [](unsigned n, unsigned batch) {
        return name(ml::cholesky_backend(n, batch)); }, "n"_a, "batch"_a = 1);
    m.def("lu_backend", [](unsigned n, unsigned batch) {
        return name(ml::lu_backend(n, batch)); }, "n"_a, "batch"_a = 1);
    m.def("trsm_backend", [](unsigned n, unsigned k, unsigned batch) {
        return name(ml::trsm_backend(n, k, batch)); }, "n"_a, "k"_a = 1, "batch"_a = 1);

    m.def("qr_policy_source", [] { return std::string(ml::qr_policy_source()); });
    m.def("eigh_policy_source", [] { return std::string(ml::eigh_policy_source()); });
    m.def("svd_policy_source", [] { return std::string(ml::svd_policy_source()); });
    m.def("cholesky_policy_source", [] { return std::string(ml::cholesky_policy_source()); });
    m.def("lu_policy_source", [] { return std::string(ml::lu_policy_source()); });
    m.def("trsm_policy_source", [] { return std::string(ml::trsm_policy_source()); });

    m.def("qr_policy", [] { auto p = ml::qr_policy(); nb::dict d; QR_FIELDS(TO_DICT) return d; });
    m.def("eigh_policy", [] { auto p = ml::eigh_policy(); nb::dict d; EIGH_FIELDS(TO_DICT) return d; });
    m.def("svd_policy", [] { auto p = ml::svd_policy(); nb::dict d; SVD_FIELDS(TO_DICT) return d; });
    m.def("cholesky_policy", [] { auto p = ml::cholesky_policy(); nb::dict d; CHOLESKY_FIELDS(TO_DICT) return d; });
    m.def("lu_policy", [] { auto p = ml::lu_policy(); nb::dict d; LU_FIELDS(TO_DICT) return d; });
    m.def("trsm_policy", [] { auto p = ml::trsm_policy(); nb::dict d; TRSM_FIELDS(TO_DICT) return d; });

    m.def("set_qr_policy", [](const nb::dict& d) {
        ml::set_qr_policy(from_dict(ml::qr_policy(), d, [](ml::QrPolicy& p, const std::string& key, nb::handle value) {
            QR_FIELDS(FROM_DICT)
            throw nb::key_error(("unknown QR policy field: " + key).c_str());
        }));
    }, "policy"_a);
    m.def("set_eigh_policy", [](const nb::dict& d) {
        ml::set_eigh_policy(from_dict(ml::eigh_policy(), d, [](ml::EighPolicy& p, const std::string& key, nb::handle value) {
            EIGH_FIELDS(FROM_DICT)
            throw nb::key_error(("unknown eigh policy field: " + key).c_str());
        }));
    }, "policy"_a);
    m.def("set_svd_policy", [](const nb::dict& d) {
        ml::set_svd_policy(from_dict(ml::svd_policy(), d, [](ml::SvdPolicy& p, const std::string& key, nb::handle value) {
            SVD_FIELDS(FROM_DICT)
            throw nb::key_error(("unknown SVD policy field: " + key).c_str());
        }));
    }, "policy"_a);
    m.def("set_cholesky_policy", [](const nb::dict& d) {
        ml::set_cholesky_policy(from_dict(ml::cholesky_policy(), d,
                                          [](ml::CholeskyPolicy& p, const std::string& key, nb::handle value) {
            CHOLESKY_FIELDS(FROM_DICT)
            throw nb::key_error(("unknown Cholesky policy field: " + key).c_str());
        }));
    }, "policy"_a);
    m.def("set_lu_policy", [](const nb::dict& d) {
        ml::set_lu_policy(from_dict(ml::lu_policy(), d, [](ml::LuPolicy& p, const std::string& key, nb::handle value) {
            LU_FIELDS(FROM_DICT)
            throw nb::key_error(("unknown LU policy field: " + key).c_str());
        }));
    }, "policy"_a);
    m.def("set_trsm_policy", [](const nb::dict& d) {
        ml::set_trsm_policy(from_dict(ml::trsm_policy(), d, [](ml::TrsmPolicy& p, const std::string& key, nb::handle value) {
            TRSM_FIELDS(FROM_DICT)
            throw nb::key_error(("unknown triangular solve policy field: " + key).c_str());
        }));
    }, "policy"_a);
}
