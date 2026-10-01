// Objective-C++: the library's C++ API called from a .mm file, with the
// matrices coming from, and going back to, an app's own float buffers.
// See docs/objective-c.md.
#import <Foundation/Foundation.h>

#include <metal_linalg/metal_linalg.h>
#include <mlx/mlx.h>

#include <vector>

namespace mx = mlx::core;
namespace ml = metal_linalg;

int main() {
    @autoreleasepool {
        mx::set_default_device(mx::Device::gpu);

        // 256 symmetric 8 x 8 matrices in an app's own buffer, row-major.
        const int batch = 256, n = 8;
        std::vector<float> data((size_t)batch * n * n);
        for (int b = 0; b < batch; ++b)
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < n; ++j)
                    data[((size_t)b * n + i) * n + j] = (i == j) ? 2.0f + b % 5 : 1.0f / (1 + i + j);

        mx::array A(data.begin(), {batch, n, n}, mx::float32);   // copies the buffer
        auto [w, V] = ml::eigh_accelerated(A);                   // w ascending, V's columns the vectors

        // Back to plain memory: evaluate, make contiguous, copy out.
        mx::array values = mx::contiguous(w);
        mx::eval({values, V});
        std::vector<float> eigenvalues(values.data<float>(), values.data<float>() + values.size());

        NSLog(@"%s routes %d matrices of %dx%d to the %s backend", ml::device_name(), batch, n, n,
              ml::eigh_backend(n, batch) == ml::EighBackend::cpu ? "CPU" : "GPU");
        NSLog(@"matrix 0: smallest eigenvalue %.4f, largest %.4f", eigenvalues[0], eigenvalues[n - 1]);

        // Check A V = V diag(w).
        mx::array residual = mx::subtract(mx::matmul(A, V), mx::multiply(V, mx::expand_dims(w, -2)));
        const float err = mx::max(mx::abs(residual)).item<float>();
        NSLog(@"max |A V - V diag(w)| = %.1e", err);
        return err < 1e-4f ? 0 : 1;
    }
}
