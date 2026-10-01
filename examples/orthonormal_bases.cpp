// QR: orthonormal bases for 10,000 subspaces at once. Each column of Q spans
// the same space as the first columns of the input, and Q^T Q = I.
#include <metal_linalg/qr.h>
#include <mlx/mlx.h>

#include <cstdio>

namespace mx = mlx::core;

int main() {
    mx::set_default_device(mx::Device::gpu);
    mx::random::seed(1);

    // 10,000 sets of 4 vectors in R^16, as the columns of 16 x 4 matrices.
    mx::array vectors = mx::random::normal({10000, 16, 4});

    auto [Q, R] = metal_linalg::qr_accelerated(vectors);   // Q [10000, 16, 4]
    mx::eval({Q, R});

    // Q^T Q should be the 4 x 4 identity for every one of the 10,000 bases.
    mx::array gram = mx::matmul(mx::swapaxes(Q, -1, -2), Q);
    const float err = mx::max(mx::abs(mx::subtract(gram, mx::eye(4)))).item<float>();

    std::printf("%d bases, max |Q^T Q - I| = %.1e\n", Q.shape(0), err);
    return err < 1e-5f ? 0 : 1;
}
