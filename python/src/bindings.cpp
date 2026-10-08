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
        default:                           return "qr_block_jacobi";
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
                       X(tridiag_max_batch) X(values_tridiag_max_batch)
#define SVD_FIELDS(X) X(qr_min_rows) X(qr_min_k) X(block_min_k) X(block_min_k_batched)       \
                      X(block_min_batch) X(gpu_max_k) X(gpu_min_batch_times_k) X(gpu_min_batch) \
                      X(gpu_cores) X(bidiag_min_k) X(values_bidiag_min_k)  \
                      X(bidiag_max_batch) X(values_bidiag_max_batch) X(gk_min_k) X(gk_max_k) X(gpu_max_l) \
                      X(values_gpu_max_k) X(values_gpu_min_batch_times_k) X(values_gpu_min_batch) X(values_gpu_max_l) \
                      X(share_min_batch) X(gpu_big_batch_max_k) X(gpu_big_batch_min) X(values_band_min_k) X(values_band_width) \
                      X(band_min_k)

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

    m.def("qr_policy_source", [] { return std::string(ml::qr_policy_source()); });
    m.def("eigh_policy_source", [] { return std::string(ml::eigh_policy_source()); });
    m.def("svd_policy_source", [] { return std::string(ml::svd_policy_source()); });

    m.def("qr_policy", [] { auto p = ml::qr_policy(); nb::dict d; QR_FIELDS(TO_DICT) return d; });
    m.def("eigh_policy", [] { auto p = ml::eigh_policy(); nb::dict d; EIGH_FIELDS(TO_DICT) return d; });
    m.def("svd_policy", [] { auto p = ml::svd_policy(); nb::dict d; SVD_FIELDS(TO_DICT) return d; });

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
}
