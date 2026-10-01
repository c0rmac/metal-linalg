// Symmetric eigensolver: principal component analysis of 1000 point clouds
// at once. Each cloud is stretched along one hidden direction; the
// eigenvector of the largest eigenvalue of its covariance recovers it.
#include <metal_linalg/eigh.h>
#include <metal_linalg/qr.h>
#include <mlx/mlx.h>

#include <cstdio>
#include <vector>

namespace mx = mlx::core;
namespace ml = metal_linalg;

int main() {
    mx::set_default_device(mx::Device::gpu);
    mx::random::seed(2);

    const int clouds = 1000, points = 500, dims = 8;

    // Hidden axes: a random orthonormal basis per cloud (the Q of a QR).
    auto [axes, unused] = ml::qr_accelerated(mx::random::normal({clouds, dims, dims}));

    // Points with standard deviation 5 along the first axis and 1 along the others.
    std::vector<float> sd(dims, 1.0f);
    sd[0] = 5.0f;
    mx::array stretch(sd.begin(), {dims});
    mx::array local  = mx::multiply(mx::random::normal({clouds, points, dims}), stretch);
    mx::array X      = mx::matmul(local, mx::swapaxes(axes, -1, -2));   // [clouds, points, dims]

    // Covariance of each cloud, then its eigendecomposition. Eigenvalues come
    // back ascending, so the principal axis is the last eigenvector.
    mx::array cov = mx::divide(mx::matmul(mx::swapaxes(X, -1, -2), X), mx::array(float(points - 1)));
    auto [variance, components] = ml::eigh_accelerated(cov);
    mx::array principal = mx::take(components, dims - 1, -1);           // [clouds, dims]
    mx::array hidden    = mx::take(axes, 0, -1);                         // [clouds, dims]

    // An eigenvector's sign is arbitrary, so compare by |cosine|.
    mx::array cosine = mx::abs(mx::sum(mx::multiply(principal, hidden), -1));
    const float worst = mx::min(cosine).item<float>();
    const float top   = mx::mean(mx::take(variance, dims - 1, -1)).item<float>();

    std::printf("%d clouds: mean largest variance %.2f (expected about 25), worst |cos| to the hidden axis %.4f\n",
                clouds, top, worst);
    return worst > 0.99f ? 0 : 1;
}
