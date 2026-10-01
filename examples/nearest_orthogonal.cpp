// SVD: the nearest orthogonal matrix to each of 100,000 noisy 3 x 3 matrices
// (the orthogonal Procrustes problem). If M = U S V^T, it is U V^T.
#include <metal_linalg/qr.h>
#include <metal_linalg/svd.h>
#include <mlx/mlx.h>

#include <cstdio>

namespace mx = mlx::core;
namespace ml = metal_linalg;

float max_abs(const mx::array& x) { return mx::max(mx::abs(x)).item<float>(); }

int main() {
    mx::set_default_device(mx::Device::gpu);
    mx::random::seed(3);

    // Ground truth: random orthogonal matrices; observed: the same plus noise.
    auto [truth, unused] = ml::qr_accelerated(mx::random::normal({100000, 3, 3}));
    mx::array observed = mx::add(truth, mx::random::normal({100000, 3, 3}, 0.0f, 0.01f));

    auto [U, S, Vt] = ml::svd_accelerated(observed);
    mx::array nearest = mx::matmul(U, Vt);
    mx::eval({nearest});

    const float ortho = max_abs(mx::subtract(mx::matmul(mx::swapaxes(nearest, -1, -2), nearest), mx::eye(3)));
    const float noise = max_abs(mx::subtract(observed, truth));
    const float error = max_abs(mx::subtract(nearest, truth));

    std::printf("max |R^T R - I| = %.1e; distance to the truth %.4f, down from %.4f before projecting\n",
                ortho, error, noise);
    return (ortho < 1e-5f && error < noise) ? 0 : 1;
}
