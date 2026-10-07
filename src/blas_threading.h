#pragma once
// Accelerate's own threading, switched off per thread where macOS allows it
// (15 and later, built with an SDK that declares the switch). Used by the
// CPU paths that spread work over the cores themselves (lapack_batches,
// share_batch, the divide and conquer), so that Accelerate's threads and
// theirs do not compete for the same cores.

// BLASSetThreading, from the macOS 15 SDK on. Built with an older SDK, the
// library cannot switch Accelerate's threading off.
#if __has_include(<vecLib/thread_api.h>)
#include <vecLib/thread_api.h>
#define METAL_LINALG_HAVE_BLAS_THREADING 1
#else
#define METAL_LINALG_HAVE_BLAS_THREADING 0
#endif

namespace metal_linalg::detail {

// Accelerate's BLAS and LAPACK single-threaded on this thread while in scope.
class SingleThreadedBlas {
public:
    SingleThreadedBlas() {
#if METAL_LINALG_HAVE_BLAS_THREADING
        if (__builtin_available(macOS 15.0, *)) {
            old_ = (int)BLASGetThreading();
            BLASSetThreading(BLAS_THREADING_SINGLE_THREADED);
        }
#endif
    }
    ~SingleThreadedBlas() {
#if METAL_LINALG_HAVE_BLAS_THREADING
        if (__builtin_available(macOS 15.0, *)) {
            if (old_ >= 0) BLASSetThreading((BLAS_THREADING)old_);
        }
#endif
    }
    SingleThreadedBlas(const SingleThreadedBlas&) = delete;
    SingleThreadedBlas& operator=(const SingleThreadedBlas&) = delete;

private:
    int old_ = -1;
};

// Whether SingleThreadedBlas has an effect here.
inline bool can_single_thread_blas() {
#if METAL_LINALG_HAVE_BLAS_THREADING
    if (__builtin_available(macOS 15.0, *)) return true;
#endif
    return false;
}

} // namespace metal_linalg::detail
