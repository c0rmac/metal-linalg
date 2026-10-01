// The three decompositions on one batch, each checked by reconstruction.
#include <metal_linalg/metal_linalg.h>
#include <mlx/mlx.h>

#include <cstdio>

namespace mx = mlx::core;
namespace ml = metal_linalg;

// Largest absolute entry of x, as a float.
float max_abs(const mx::array& x) { return mx::max(mx::abs(x)).item<float>(); }

int main() {
    mx::set_default_device(mx::Device::gpu);
    mx::random::seed(0);

    // A batch of 1000 random 64 x 32 matrices.
    mx::array A = mx::random::normal({1000, 64, 32});

    // QR: Q [1000, 64, 32] with orthonormal columns, R [1000, 32, 32] upper triangular.
    auto [Q, R] = ml::qr_accelerated(A);

    // Thin SVD: U [1000, 64, 32], S [1000, 32] descending, Vt [1000, 32, 32].
    auto [U, S, Vt] = ml::svd_accelerated(A);

    // Symmetric eigendecomposition of C = A^T A: w [1000, 32] ascending, V [1000, 32, 32].
    mx::array C = mx::matmul(mx::swapaxes(A, -1, -2), A);
    auto [w, V] = ml::eigh_accelerated(C);

    mx::eval({Q, R, U, S, Vt, w, V});

    const float qr_err  = max_abs(mx::subtract(mx::matmul(Q, R), A));
    const float svd_err = max_abs(mx::subtract(mx::matmul(mx::multiply(U, mx::expand_dims(S, -2)), Vt), A));
    const float eig_err = max_abs(mx::subtract(mx::matmul(C, V), mx::multiply(V, mx::expand_dims(w, -2))))
                        / max_abs(C);
    // The eigenvalues of A^T A are the squared singular values of A.
    const float agree   = max_abs(mx::subtract(w, mx::sort(mx::square(S), -1))) / max_abs(w);

    std::printf("max |QR - A|                   %.1e\n", qr_err);
    std::printf("max |U diag(S) Vt - A|         %.1e\n", svd_err);
    std::printf("max |C V - V diag(w)| / |C|    %.1e\n", eig_err);
    std::printf("eigenvalues vs S^2, relative   %.1e\n", agree);
    return (qr_err < 1e-4f && svd_err < 1e-4f && eig_err < 1e-5f && agree < 1e-5f) ? 0 : 1;
}
