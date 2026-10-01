# Using metal-linalg from Objective-C

The library's API is C++, and Objective-C++ (a `.mm` file) calls C++
directly: the library is itself written in Objective-C++. Nothing else is
needed. A complete program is [`examples/objc_quickstart.mm`](../examples/objc_quickstart.mm),
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

**With CMake**, as for C++ (see the [README](../README.md#using-it-in-a-cmake-project)),
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

## From Swift

Swift apps built on [mlx-swift](https://github.com/ml-explore/mlx-swift) use
its own copy of MLX, so its `MLXArray` cannot be handed to the Homebrew build
above; a Swift package that does that is in preparation. A Swift app that
does not use mlx-swift can call metal-linalg through a small Objective-C++
class that takes and returns plain buffers, exposed to Swift with a bridging
header:

```objc
// LinalgBridge.h, imported by the Swift bridging header
#import <Foundation/Foundation.h>
@interface LinalgBridge : NSObject
/// Eigenvalues of `batch` symmetric n x n matrices, row-major; returns batch * n values.
+ (NSData *)eigenvaluesOf:(NSData *)matrices batch:(NSInteger)batch n:(NSInteger)n;
@end
```

```objc
// LinalgBridge.mm
#import "LinalgBridge.h"
#include <metal_linalg/metal_linalg.h>
#include <mlx/mlx.h>
namespace mx = mlx::core;

@implementation LinalgBridge
+ (NSData *)eigenvaluesOf:(NSData *)matrices batch:(NSInteger)batch n:(NSInteger)n {
    const float* in = static_cast<const float*>(matrices.bytes);
    mx::array a(in, {(int)batch, (int)n, (int)n}, mx::float32);
    mx::array w = mx::contiguous(metal_linalg::eigvalsh_accelerated(a));
    mx::eval({w});
    return [NSData dataWithBytes:w.data<float>() length:w.nbytes()];
}
@end
```
