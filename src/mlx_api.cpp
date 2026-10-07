// The MLX API (qr.h, eigh.h, svd.h) over the buffer core (core.h). Each call
// makes its input an evaluated, contiguous float32 array, allocates its
// outputs as MLX arrays, and has the core write into their memory.
#include <metal_linalg/eigh.h>
#include <metal_linalg/qr.h>
#include <metal_linalg/svd.h>

#include <mlx/mlx.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace mx = mlx::core;

namespace metal_linalg {
namespace {

// An input as the core sees it. `array` owns the memory `matrices` points at.
struct Input {
    mx::array      array;
    mx::Shape      batch_shape;
    core::Matrices matrices;
};

Input prepare(const mx::array& a, const char* who) {
    if (a.ndim() < 2) {
        throw std::invalid_argument(std::string("[") + who + "] Input must be at least a 2D matrix.");
    }
    // Real input only: casting a complex array to float32 would keep the real
    // parts and return the decomposition of a different matrix.
    if (mx::issubdtype(a.dtype(), mx::complexfloating)) {
        throw std::invalid_argument(std::string("[") + who + "] Complex input is not supported: "
                                    "the decompositions are real (float32).");
    }
    // The evaluation has to come first. An unevaluated MLX array reports
    // itself as row-contiguous (its flags default to true), so checking flags
    // on a lazy transposed view and then evaluating hands back the
    // *un-transposed* buffer.
    mx::array x = mx::astype(a, mx::float32);
    mx::eval({x});
    if (!x.flags().row_contiguous) {
        x = mx::contiguous(x);
        mx::eval({x});
    }

    const mx::Shape& s = x.shape();
    uint64_t batch = 1;
    for (size_t i = 0; i + 2 < s.size(); ++i) batch *= (uint64_t)s[i];
    if (batch > std::numeric_limits<uint32_t>::max()) {
        throw std::invalid_argument(std::string("[") + who + "] The batch has more than 2^32 matrices.");
    }

    Input in{x, mx::Shape(s.begin(), s.end() - 2), {}};
    in.matrices.data  = x.size() ? x.data<float>() : nullptr;
    in.matrices.batch = (uint32_t)batch;
    in.matrices.rows  = (uint32_t)s[s.size() - 2];
    in.matrices.cols  = (uint32_t)s[s.size() - 1];
    return in;
}

// The batch shape followed by `tail`.
mx::Shape shape_of(const mx::Shape& batch_shape, std::initializer_list<uint32_t> tail) {
    mx::Shape s = batch_shape;
    for (uint32_t d : tail) s.push_back((int)d);
    return s;
}

// An array whose memory the core writes. Empty arrays are never written.
mx::array output(const mx::Shape& shape, mx::Dtype dtype = mx::float32) {
    size_t n = 1;
    for (int d : shape) n *= (size_t)d;
    if (n == 0) return mx::zeros(shape, dtype);
    return mx::array(mx::allocator::malloc(n * dtype.size()), shape, dtype);
}

mx::array nothing() { return mx::zeros(mx::Shape{0}, mx::float32); }

template <class T>
T* memory(mx::array& x) { return x.size() ? x.data<T>() : nullptr; }

bool parse_uplo(const std::string& uplo, const char* who) {
    if (uplo == "L" || uplo == "l") return true;
    if (uplo == "U" || uplo == "u") return false;
    throw std::invalid_argument(std::string("[") + who + "] uplo must be \"L\" or \"U\".");
}

// --- QR -----------------------------------------------------------------------

template <class Fn>
std::pair<mx::array, mx::array> run_qr(const mx::array& a, const char* who, Fn fn) {
    Input in = prepare(a, who);
    const uint32_t M = in.matrices.rows, N = in.matrices.cols, K = std::min(M, N);
    mx::array q = output(shape_of(in.batch_shape, {M, K}));
    mx::array r = output(shape_of(in.batch_shape, {K, N}));
    fn(in.matrices, memory<float>(q), memory<float>(r));
    return {q, r};
}

// --- eigh ---------------------------------------------------------------------

template <class Fn>
EighResult run_eigh(const mx::array& a, bool vectors, const char* who, Fn fn) {
    Input in = prepare(a, who);
    const uint32_t n = in.matrices.cols;
    if (in.matrices.rows != n) {
        throw std::invalid_argument(std::string("[") + who + "] Input matrices must be square.");
    }
    mx::array w    = output(shape_of(in.batch_shape, {n}));
    mx::array v    = vectors ? output(shape_of(in.batch_shape, {n, n})) : nothing();
    mx::array info = output(in.batch_shape, mx::uint32);
    fn(in.matrices, memory<float>(w), vectors ? memory<float>(v) : nullptr, memory<uint32_t>(info));
    return {w, v, info};
}

// --- SVD ----------------------------------------------------------------------

template <class Fn>
SvdResult run_svd(const mx::array& a, bool uv, const char* who, Fn fn) {
    Input in = prepare(a, who);
    const uint32_t M = in.matrices.rows, N = in.matrices.cols, K = std::min(M, N);
    mx::array u    = uv ? output(shape_of(in.batch_shape, {M, K})) : nothing();
    mx::array s    = output(shape_of(in.batch_shape, {K}));
    mx::array vt   = uv ? output(shape_of(in.batch_shape, {K, N})) : nothing();
    mx::array info = output(in.batch_shape, mx::uint32);
    fn(in.matrices, uv ? memory<float>(u) : nullptr, memory<float>(s),
       uv ? memory<float>(vt) : nullptr, memory<uint32_t>(info));
    return {u, s, vt, info};
}

} // namespace

// =============================================================================
// Public API
// =============================================================================

std::pair<mx::array, mx::array> qr_accelerated(const mx::array& a) {
    return run_qr(a, "qr", core::qr);
}

std::pair<mx::array, mx::array> eigh_accelerated(const mx::array& a, const std::string& uplo) {
    const bool lower = parse_uplo(uplo, "eigh");
    EighResult r = run_eigh(a, true, "eigh", [&](const core::Matrices& m, float* w, float* v, uint32_t* info) {
        core::eigh(m, lower, w, v, info);
    });
    return {r.eigenvalues, r.eigenvectors};
}

mx::array eigvalsh_accelerated(const mx::array& a, const std::string& uplo) {
    const bool lower = parse_uplo(uplo, "eigvalsh");
    return run_eigh(a, false, "eigvalsh", [&](const core::Matrices& m, float* w, float* v, uint32_t* info) {
        core::eigh(m, lower, w, v, info);
    }).eigenvalues;
}

std::tuple<mx::array, mx::array, mx::array> svd_accelerated(const mx::array& a) {
    SvdResult r = run_svd(a, true, "svd", core::svd);
    return {r.U, r.S, r.Vt};
}

mx::array svdvals_accelerated(const mx::array& a) {
    return run_svd(a, false, "svdvals", core::svd).S;
}

// =============================================================================
// Each backend on its own
// =============================================================================

namespace detail {

std::pair<mx::array, mx::array> qr_unblocked(const mx::array& a) {
    return run_qr(a, "qr_unblocked", core::detail::qr_unblocked);
}

std::pair<mx::array, mx::array> qr_streaming_amx_reduced(const mx::array& a) {
    return run_qr(a, "qr_streaming_amx_reduced", core::detail::qr_streaming_amx_reduced);
}

std::pair<mx::array, mx::array> qr_streaming_amx_complete(const mx::array& a) {
    return run_qr(a, "qr_streaming_amx_complete", core::detail::qr_streaming_amx_complete);
}

std::pair<mx::array, mx::array> qr_blocked(const mx::array& a) {
    return run_qr(a, "qr_blocked", core::detail::qr_blocked);
}

std::pair<mx::array, mx::array> qr_cpu(const mx::array& a) {
    return run_qr(a, "qr_cpu", core::detail::qr_cpu);
}

std::pair<mx::array, mx::array> qr_shared(const mx::array& a) {
    return run_qr(a, "qr", core::detail::qr_shared);
}

EighResult eigh_jacobi(const mx::array& a, bool compute_vectors, bool lower, const EighOptions& opt) {
    return run_eigh(a, compute_vectors, "eigh", [&](const core::Matrices& m, float* w, float* v, uint32_t* info) {
        core::detail::eigh_jacobi(m, lower, opt, w, v, info);
    });
}

EighResult eigh_block_jacobi(const mx::array& a, bool compute_vectors, bool lower, const EighOptions& opt) {
    return run_eigh(a, compute_vectors, "eigh_block", [&](const core::Matrices& m, float* w, float* v, uint32_t* info) {
        core::detail::eigh_block_jacobi(m, lower, opt, w, v, info);
    });
}

EighResult eigh_tridiag(const mx::array& a, bool compute_vectors, bool lower) {
    return run_eigh(a, compute_vectors, "eigh", [&](const core::Matrices& m, float* w, float* v, uint32_t* info) {
        core::detail::eigh_tridiag(m, lower, w, v, info);
    });
}

EighResult eigh_band(const mx::array& a, bool lower, uint32_t width) {
    return run_eigh(a, false, "eigvalsh", [&](const core::Matrices& m, float* w, float*, uint32_t* info) {
        core::detail::eigh_band(m, lower, w, info, width);
    });
}

EighResult eigh_ql(const mx::array& a, bool compute_vectors, bool lower) {
    return run_eigh(a, compute_vectors, "eigh", [&](const core::Matrices& m, float* w, float* v, uint32_t* info) {
        core::detail::eigh_ql(m, lower, w, v, info);
    });
}

EighResult eigh_ql_shared(const mx::array& a, bool compute_vectors, bool lower) {
    return run_eigh(a, compute_vectors, "eigh", [&](const core::Matrices& m, float* w, float* v, uint32_t* info) {
        core::detail::eigh_ql_shared(m, lower, w, v, info);
    });
}

EighResult eigh_cpu(const mx::array& a, bool compute_vectors, bool lower) {
    return run_eigh(a, compute_vectors, "eigh", [&](const core::Matrices& m, float* w, float* v, uint32_t* info) {
        core::detail::eigh_cpu(m, lower, w, v, info);
    });
}

SvdResult svd_jacobi(const mx::array& a, bool compute_uv, const SvdOptions& opt) {
    return run_svd(a, compute_uv, "svd", [&](const core::Matrices& m, float* u, float* s, float* vt, uint32_t* info) {
        core::detail::svd_jacobi(m, opt, u, s, vt, info);
    });
}

SvdResult svd_block_jacobi(const mx::array& a, bool compute_uv, const SvdOptions& opt) {
    return run_svd(a, compute_uv, "svd_block", [&](const core::Matrices& m, float* u, float* s, float* vt, uint32_t* info) {
        core::detail::svd_block_jacobi(m, opt, u, s, vt, info);
    });
}

SvdResult svd_qr_jacobi(const mx::array& a, bool compute_uv, const SvdOptions& opt) {
    return run_svd(a, compute_uv, "svd", [&](const core::Matrices& m, float* u, float* s, float* vt, uint32_t* info) {
        core::detail::svd_qr_jacobi(m, opt, u, s, vt, info);
    });
}

SvdResult svd_bidiag(const mx::array& a, bool compute_uv) {
    return run_svd(a, compute_uv, "svd", core::detail::svd_bidiag);
}

SvdResult svd_band(const mx::array& a, uint32_t width) {
    return run_svd(a, false, "svd", [width](const core::Matrices& m, float*, float* s, float*, uint32_t* info) {
        core::detail::svd_band(m, s, info, width);
    });
}

SvdResult svd_band_vectors(const mx::array& a) {
    return run_svd(a, true, "svd", core::detail::svd_band_vectors);
}

SvdResult svd_golub_kahan(const mx::array& a, bool compute_uv) {
    return run_svd(a, compute_uv, "svd", core::detail::svd_golub_kahan);
}

SvdResult svd_golub_kahan_shared(const mx::array& a, bool compute_uv) {
    return run_svd(a, compute_uv, "svd", core::detail::svd_golub_kahan_shared);
}

SvdResult svd_cpu(const mx::array& a, bool compute_uv) {
    return run_svd(a, compute_uv, "svd", core::detail::svd_cpu);
}

} // namespace detail

} // namespace metal_linalg
