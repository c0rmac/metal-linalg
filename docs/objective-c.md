# Using metal-linalg from Objective-C

Objective-C++ (a `.mm` file) calls the library's C++ API directly, on MLX
arrays as below; the library is itself written in Objective-C++. Code that
holds its matrices in its own memory can skip MLX and use the C API instead,
from `.m` or `.mm` files: see [Without MLX](#without-mlx). A complete program is [`examples/objc_quickstart.mm`](../examples/objc_quickstart.mm),
built and run with the other examples by `ctest`.

```objc
#import <Foundation/Foundation.h>
#include <metal_linalg/metal_linalg.h>
#include <mlx/mlx.h>

namespace mx = mlx::core;

// floats: an app's own buffer of `batch` symmetric n x n matrices, row-major
mx::array A(floats.begin(), {batch, n, n}, mx::float32);     // copies the buffer
auto [w, V] = metal_linalg::eigh_accelerated(A);             // w ascending

mx::array values = mx::contiguous(w);
mx::eval({values, V});
const float* eigenvalues = values.data<float>();             // batch x n
```

Arrays are MLX's (`mlx::core::array`): build one from a buffer as above, and
read results with `eval`, then `data<float>()` on a contiguous array. The
functions are written for, and tested with, the GPU as the default MLX device.

## Building

**With CMake**, as for C++ (see the [README](../README.md#in-a-cmake-project)),
with Objective-C++ enabled:

```cmake
project(my_app LANGUAGES CXX OBJCXX)
find_package(MetalLinalg REQUIRED)
add_executable(my_app main.mm)
target_link_libraries(my_app PRIVATE metal_linalg::metal_linalg "-framework Foundation")
```

**In Xcode**, with metal-linalg and MLX installed by Homebrew, in the target's
build settings:

| setting | value |
|---|---|
| C++ Language Dialect | C++20 |
| Header Search Paths | `/opt/homebrew/include` |
| Library Search Paths | `/opt/homebrew/lib` |
| Other Linker Flags | `-lmetal_linalg -lmlx` |
| Runpath Search Paths | `/opt/homebrew/lib` |

and name the files that call the library `.mm`. An app that ships to other
Macs embeds `libmetal_linalg.dylib` and `libmlx.dylib` in its bundle
(Frameworks, Libraries and Embedded Content) and sets the runpath to
`@executable_path/../Frameworks` instead.

## Without MLX

An app that has its matrices in its own memory need not involve MLX at all:
the C API (plain C, so usable from `.m` files too) and the C++ buffer API
`<metal_linalg/core.h>` take float buffers directly, with no copies in and
out of MLX arrays. Build with `-DMETAL_LINALG_WITH_MLX=OFF` to leave MLX out
of the library as well.

```objc
#import <Foundation/Foundation.h>
#include <metal_linalg/c_api.h>

// `matrices`: `batch` symmetric n x n matrices, row-major, one after another.
NSData* eigenvalues(NSData* matrices, uint32_t batch, uint32_t n, NSError** error) {
    NSMutableData* w = [NSMutableData dataWithLength:(NSUInteger)batch * n * sizeof(float)];
    metal_linalg_status st = metal_linalg_eigh(matrices.bytes, batch, n, 1, w.mutableBytes, NULL, NULL);
    if (st != METAL_LINALG_OK) {
        if (error) *error = [NSError errorWithDomain:@"metal_linalg" code:st
                                            userInfo:@{NSLocalizedDescriptionKey: @(metal_linalg_last_error())}];
        return nil;
    }
    return w;
}
```

See [docs/c-api.md](c-api.md) for the conventions.

## From Swift

Use the Swift package, on `[Float]` or on mlx-swift's `MLXArray`:
[docs/swift.md](swift.md).
