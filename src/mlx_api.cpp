// The MLX API (qr.h, eigh.h, svd.h, cholesky.h) over the buffer core
// (core.h). Each call makes its input an evaluated, contiguous float32 array,
// allocates its outputs as MLX arrays, and has the core write into their
// memory.
#include <metal_linalg/cholesky.h>
#include <metal_linalg/lu.h>
#include <metal_linalg/triangular.h>
#include <metal_linalg/eigh.h>
#include <metal_linalg/qr.h>
#include <metal_linalg/svd.h>

#include <mlx/mlx.h>

#include "known_buffers.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

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

// The Metal buffer behind an array's memory, for detail::KnownBuffer; nullptr
// without Metal (the buffer is then not one) or for an empty array.
void* metal_buffer(mx::array& x) {
    static const bool metal = mx::metal::is_available();
    return metal && x.size() ? x.buffer().ptr() : nullptr;
}

// The input's and the outputs' buffers, known to the core for the call.
class Known {
public:
    explicit Known(Input& in) { add(in.array); }
    Known& operator()(mx::array& x) {
        add(x);
        return *this;
    }

private:
    std::vector<std::unique_ptr<detail::KnownBuffer>> known_;
    void add(mx::array& x) {
        if (x.size()) known_.push_back(std::make_unique<detail::KnownBuffer>(x.data<char>(), metal_buffer(x)));
    }
};

bool parse_uplo(const std::string& uplo, const char* who) {
    if (uplo == "L" || uplo == "l") return true;
    if (uplo == "U" || uplo == "u") return false;
    throw std::invalid_argument(std::string("[") + who + "] uplo must be \"L\" or \"U\".");
}

// --- QR -----------------------------------------------------------------------

core::QrMode parse_qr_mode(const std::string& mode, const char* who) {
    if (mode == "reduced") return core::QrMode::reduced;
    if (mode == "r") return core::QrMode::r;
    if (mode == "complete") return core::QrMode::complete;
    throw std::invalid_argument(std::string("[") + who + "] mode must be \"reduced\", \"r\" or \"complete\".");
}

// Q [..., M, K] (M x M complete, an empty array for R alone) and R [..., K, N]
// (M x N complete).
using QrFn = void (*)(const core::Matrices&, float*, float*, core::QrMode);

std::pair<mx::array, mx::array> run_qr(const mx::array& a, const char* who, QrFn fn,
                                       core::QrMode mode = core::QrMode::reduced) {
    Input in = prepare(a, who);
    const uint32_t M = in.matrices.rows, N = in.matrices.cols;
    const uint32_t QC = core::qr_q_cols(mode, M, N), RR = core::qr_r_rows(mode, M, N);
    // R alone: Q an empty array, materialized (a lazy one would make the
    // caller's eval schedule a fill of nothing: some 30 us a call)
    mx::array q = mode == core::QrMode::r ? mx::array(std::initializer_list<float>{}, mx::Shape{0})
                                          : output(shape_of(in.batch_shape, {M, QC}));
    mx::array r = output(shape_of(in.batch_shape, {RR, N}));
    Known known(in);
    known(q)(r);
    fn(in.matrices, memory<float>(q), memory<float>(r), mode);
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
    Known known(in);
    known(w)(v)(info);
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
    Known known(in);
    known(u)(s)(vt)(info);
    fn(in.matrices, uv ? memory<float>(u) : nullptr, memory<float>(s),
       uv ? memory<float>(vt) : nullptr, memory<uint32_t>(info));
    return {u, s, vt, info};
}

// --- Cholesky -----------------------------------------------------------------

using CholeskyFn = void (*)(const core::Matrices&, bool, float*, uint32_t*);

CholeskyResult run_cholesky(const mx::array& a, bool upper, const char* who, CholeskyFn fn) {
    Input in = prepare(a, who);
    const uint32_t n = in.matrices.cols;
    if (in.matrices.rows != n) {
        throw std::invalid_argument(std::string("[") + who + "] Input matrices must be square.");
    }
    mx::array l    = output(shape_of(in.batch_shape, {n, n}));
    mx::array info = output(in.batch_shape, mx::uint32);
    Known known(in);
    known(l)(info);
    fn(in.matrices, upper, memory<float>(l), memory<uint32_t>(info));
    return {l, info};
}

// --- LU, solve, inverse -------------------------------------------------------

void require_square(const Input& in, const char* who) {
    if (in.matrices.rows != in.matrices.cols)
        throw std::invalid_argument(std::string("[") + who + "] Input matrices must be square.");
}

using LuFn = void (*)(const core::Matrices&, float*, uint32_t*, uint32_t*);
using SolveFn = void (*)(const core::Matrices&, const float*, uint32_t, float*, uint32_t*);
using InvFn = void (*)(const core::Matrices&, float*, uint32_t*);

LuResult run_lu(const mx::array& a, LuFn fn) {
    Input in = prepare(a, "lu_factor");
    require_square(in, "lu_factor");
    const uint32_t n = in.matrices.cols;
    mx::array lu     = output(shape_of(in.batch_shape, {n, n}));
    mx::array pivots = output(shape_of(in.batch_shape, {n}), mx::uint32);
    mx::array info   = output(in.batch_shape, mx::uint32);
    if (in.matrices.batch && n) fn(in.matrices, memory<float>(lu), memory<uint32_t>(pivots), memory<uint32_t>(info));
    return {lu, pivots, info};
}

// b [..., N, K] or [..., N] against a [..., N, N], the batch shapes equal.
SolveResult run_solve(const mx::array& a, const mx::array& b, SolveFn fn) {
    Input in = prepare(a, "solve");
    require_square(in, "solve");
    const uint32_t n = in.matrices.cols;
    const bool vector = b.ndim() == a.ndim() - 1;
    if (!vector && b.ndim() != a.ndim())
        throw std::invalid_argument("[solve] b must be [..., N] or [..., N, K] with a's batch shape.");
    const mx::array bm = vector ? mx::expand_dims(b, -1) : b;
    Input rhs = prepare(bm, "solve");
    if (rhs.batch_shape != in.batch_shape || rhs.matrices.rows != n)
        throw std::invalid_argument("[solve] b must be [..., N] or [..., N, K] with a's batch shape.");
    const uint32_t k = rhs.matrices.cols;
    mx::array x    = output(shape_of(in.batch_shape, {n, k}));
    mx::array info = output(in.batch_shape, mx::uint32);
    if (in.matrices.batch && n)
        fn(in.matrices, rhs.matrices.data, k, memory<float>(x), memory<uint32_t>(info));
    return {vector ? mx::squeeze(x, -1) : x, info};
}

using TrsmFn = void (*)(const core::Matrices&, const float*, uint32_t, bool, bool, float*);

mx::array run_trsm(const mx::array& a, const mx::array& b, bool upper, bool unit, TrsmFn fn) {
    Input in = prepare(a, "solve_triangular");
    require_square(in, "solve_triangular");
    const uint32_t n = in.matrices.cols;
    const bool vector = b.ndim() == a.ndim() - 1;
    if (!vector && b.ndim() != a.ndim())
        throw std::invalid_argument("[solve_triangular] b must be [..., N] or [..., N, K] with a's batch shape.");
    const mx::array bm = vector ? mx::expand_dims(b, -1) : b;
    Input rhs = prepare(bm, "solve_triangular");
    if (rhs.batch_shape != in.batch_shape || rhs.matrices.rows != n)
        throw std::invalid_argument("[solve_triangular] b must be [..., N] or [..., N, K] with a's batch shape.");
    const uint32_t k = rhs.matrices.cols;
    mx::array x = output(shape_of(in.batch_shape, {n, k}));
    Known known(in);
    known(rhs.array)(x);
    if (in.matrices.batch && n && k) fn(in.matrices, rhs.matrices.data, k, upper, unit, memory<float>(x));
    return vector ? mx::squeeze(x, -1) : x;
}

SolveResult run_inv(const mx::array& a, InvFn fn) {
    Input in = prepare(a, "inv");
    require_square(in, "inv");
    const uint32_t n = in.matrices.cols;
    mx::array x    = output(shape_of(in.batch_shape, {n, n}));
    mx::array info = output(in.batch_shape, mx::uint32);
    if (in.matrices.batch && n) fn(in.matrices, memory<float>(x), memory<uint32_t>(info));
    return {x, info};
}

} // namespace

// =============================================================================
// Public API
// =============================================================================

mx::array cholesky_accelerated(const mx::array& a, bool upper) {
    return run_cholesky(a, upper, "cholesky", core::cholesky).l;
}

CholeskyResult cholesky_ex_accelerated(const mx::array& a, bool upper) {
    return run_cholesky(a, upper, "cholesky", core::cholesky);
}

namespace detail {
CholeskyResult cholesky_simd(const mx::array& a, bool upper) {
    return run_cholesky(a, upper, "cholesky", core::detail::cholesky_simd);
}
CholeskyResult cholesky_threadgroup(const mx::array& a, bool upper) {
    return run_cholesky(a, upper, "cholesky", core::detail::cholesky_threadgroup);
}
CholeskyResult cholesky_blocked(const mx::array& a, bool upper) {
    return run_cholesky(a, upper, "cholesky", core::detail::cholesky_blocked);
}
CholeskyResult cholesky_cpu(const mx::array& a, bool upper) {
    return run_cholesky(a, upper, "cholesky", core::detail::cholesky_cpu);
}
} // namespace detail

std::pair<mx::array, mx::array> lu_factor_accelerated(const mx::array& a) {
    LuResult r = run_lu(a, core::lu_factor);
    return {r.lu, r.pivots};
}
LuResult lu_factor_ex_accelerated(const mx::array& a) { return run_lu(a, core::lu_factor); }
mx::array solve_accelerated(const mx::array& a, const mx::array& b) { return run_solve(a, b, core::solve).x; }
SolveResult solve_ex_accelerated(const mx::array& a, const mx::array& b) { return run_solve(a, b, core::solve); }
mx::array inv_accelerated(const mx::array& a) { return run_inv(a, core::inv).x; }
SolveResult inv_ex_accelerated(const mx::array& a) { return run_inv(a, core::inv); }

mx::array solve_triangular_accelerated(const mx::array& a, const mx::array& b, bool upper, bool unit_diagonal) {
    return run_trsm(a, b, upper, unit_diagonal, core::solve_triangular);
}

namespace detail {
LuResult lu_factor_cpu(const mx::array& a) { return run_lu(a, core::detail::lu_factor_cpu); }
LuResult lu_factor_blocked(const mx::array& a) { return run_lu(a, core::detail::lu_factor_blocked); }
SolveResult solve_cpu(const mx::array& a, const mx::array& b) { return run_solve(a, b, core::detail::solve_cpu); }
SolveResult solve_blocked(const mx::array& a, const mx::array& b) {
    return run_solve(a, b, core::detail::solve_blocked);
}
SolveResult inv_cpu(const mx::array& a) { return run_inv(a, core::detail::inv_cpu); }
SolveResult inv_blocked(const mx::array& a) { return run_inv(a, core::detail::inv_blocked); }
mx::array solve_triangular_cpu(const mx::array& a, const mx::array& b, bool upper, bool unit_diagonal) {
    return run_trsm(a, b, upper, unit_diagonal, core::detail::solve_triangular_cpu);
}
mx::array solve_triangular_blocked(const mx::array& a, const mx::array& b, bool upper, bool unit_diagonal) {
    return run_trsm(a, b, upper, unit_diagonal, core::detail::solve_triangular_blocked);
}
} // namespace detail

std::pair<mx::array, mx::array> qr_accelerated(const mx::array& a, const std::string& mode) {
    return run_qr(a, "qr", core::qr, parse_qr_mode(mode, "qr"));
}

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
    return run_qr(a, "qr_streaming_amx_complete",
                  [](const core::Matrices& m, float* q, float* r, core::QrMode) {
                      core::detail::qr_streaming_amx_complete(m, q, r);
                  });
}

std::pair<mx::array, mx::array> qr_householder(const mx::array& a) {
    return run_qr(a, "qr_householder", core::detail::qr_householder);
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

EighResult eigh_tridiag_batch(const mx::array& a, bool compute_vectors, bool lower) {
    return run_eigh(a, compute_vectors, "eigh", [&](const core::Matrices& m, float* w, float* v, uint32_t* info) {
        core::detail::eigh_tridiag_batch(m, lower, w, v, info);
    });
}

EighResult eigh_band_vectors(const mx::array& a, bool lower) {
    return run_eigh(a, true, "eigh", [&](const core::Matrices& m, float* w, float* v, uint32_t* info) {
        core::detail::eigh_band_vectors(m, lower, w, v, info);
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

SvdResult svd_bidiag_batch(const mx::array& a, bool compute_uv) {
    return run_svd(a, compute_uv, "svd", core::detail::svd_bidiag_batch);
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
