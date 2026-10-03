#include "metal_runtime.h"
#include "shaders.h"

#include <metal_linalg/device.h>

#include <Accelerate/Accelerate.h>
// BLASSetThreading, from the macOS 15 SDK on. Built with an older SDK, the
// library cannot switch Accelerate's threading off and splits a batch only
// where Accelerate would not thread anyway (see lapack_batches).
#if __has_include(<vecLib/thread_api.h>)
#include <vecLib/thread_api.h>
#define METAL_LINALG_HAVE_BLAS_THREADING 1
#else
#define METAL_LINALG_HAVE_BLAS_THREADING 0
#endif

#include <atomic>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <map>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>
#include <unistd.h>

namespace metal_linalg::detail {

uint pad_up(uint v, uint multiple) {
    return ((v + multiple - 1) / multiple) * multiple;
}

namespace {

size_t page_size() {
    static const size_t page = (size_t)getpagesize();
    return page;
}

size_t page_round(size_t bytes) {
    const size_t page = page_size();
    return std::max(page, (bytes + page - 1) / page * page);
}

id<MTLBuffer> new_shared_buffer(id<MTLDevice> device, size_t bytes) {
    id<MTLBuffer> b = [device newBufferWithLength:std::max<size_t>(bytes, 16)
                                          options:MTLResourceStorageModeShared];
    if (!b) throw std::runtime_error("Could not allocate a Metal buffer of " + std::to_string(bytes) + " bytes.");
    return b;
}

} // namespace

id<MTLBuffer> wrap_host(id<MTLDevice> device, float* data, size_t floats) {
    id<MTLBuffer> b = [device newBufferWithBytesNoCopy:data
                                                length:page_round(floats * sizeof(float))
                                               options:MTLResourceStorageModeShared
                                           deallocator:nil];
    if (!b) throw std::runtime_error("Could not wrap host memory as a Metal buffer.");
    return b;
}

id<MTLBuffer> input_buffer(id<MTLDevice> device, const Matrices& a, bool transpose) {
    const size_t bytes = element_count(a) * sizeof(float);
    if (transpose) {
        id<MTLBuffer> b = new_shared_buffer(device, bytes);
        transpose_out(a.data, static_cast<float*>([b contents]), a.batch, a.rows, a.cols);
        return b;
    }
    // The length is rounded up to whole pages, as newBufferWithBytesNoCopy
    // requires: a page-aligned start means the last page is mapped too, and
    // the kernels never read past the matrices.
    if (reinterpret_cast<uintptr_t>(a.data) % page_size() == 0) {
        id<MTLBuffer> b = [device newBufferWithBytesNoCopy:(void*)a.data
                                                    length:page_round(bytes)
                                                   options:MTLResourceStorageModeShared
                                               deallocator:nil];
        if (b) return b;
    }
    id<MTLBuffer> b = new_shared_buffer(device, bytes);
    std::memcpy([b contents], a.data, bytes);
    return b;
}

void scan(const Matrices& a, Part part, float* amax, char* finite) {
    const size_t per = (size_t)a.rows * a.cols;
    std::fill(amax, amax + a.batch, 0.0f);
    std::fill(finite, finite + a.batch, 1);
    std::mutex merge;   // only taken when a matrix is split into row blocks
    const bool split = per >= 2 * kGrain;
    for_each_rows(a.batch, a.rows, a.cols, [&](uint32_t b, uint32_t r0, uint32_t r1) {
        const float* d = a.data + b * per;
        float m = 0.0f;
        unsigned ok = 1;
        for (uint32_t i = r0; i < r1; ++i) {
            uint32_t j0 = 0, j1 = a.cols;
            if (part == Part::lower) j1 = std::min(a.cols, i + 1);
            if (part == Part::upper) j0 = std::min(a.cols, i);
            const float* row = d + (size_t)i * a.cols;
            for (uint32_t j = j0; j < j1; ++j) {
                const float v = std::fabs(row[j]);
                ok &= (unsigned)(v <= FLT_MAX);   // false for NaN and for infinity
                m = v > m ? v : m;
            }
        }
        if (!split) {
            amax[b]   = m;
            finite[b] = ok ? 1 : 0;
            return;
        }
        std::lock_guard<std::mutex> lock(merge);
        amax[b]   = std::max(amax[b], m);
        finite[b] = finite[b] && ok;
    });
}

ScaledInput scaled_input(id<MTLDevice> device, const Matrices& a, bool transpose) {
    const uint32_t batch = a.batch;
    std::vector<float> amax(batch, 0.0f);
    std::vector<char>  finite(batch, 1);
    scan(a, Part::all, amax.data(), finite.data());

    ScaledInput out;
    out.unscale.assign(batch, 1.0f);
    out.nonfinite.resize(batch);
    out.scaled = false;
    std::vector<float> down(batch, 1.0f);
    for (uint32_t b = 0; b < batch; ++b) {
        out.nonfinite[b] = !finite[b];
        if (!finite[b] || !(amax[b] > 0.0f)) continue;
        int e = 0;
        std::frexp(amax[b], &e);
        if (e == 0) continue;   // already in [0.5, 1)
        down[b]        = std::ldexp(1.0f, -e);
        out.unscale[b] = std::ldexp(1.0f, e);
        out.scaled     = true;
    }
    if (!out.scaled) {
        out.buffer = input_buffer(device, a, transpose);
        return out;
    }

    // One pass: scale (and transpose) into a fresh buffer.
    const size_t per = (size_t)a.rows * a.cols;
    out.buffer = new_shared_buffer(device, element_count(a) * sizeof(float));
    float* dst = static_cast<float*>([out.buffer contents]);
    if (transpose) {
        for_each_matrix(batch, per, [&](uint32_t b) {
            float* d = dst + b * per;
            vDSP_mtrans(a.data + b * per, 1, d, 1, a.cols, a.rows);
            vDSP_vsmul(d, 1, &down[b], d, 1, per);
        });
    } else {
        for_each_rows(batch, a.rows, a.cols, [&](uint32_t b, uint32_t r0, uint32_t r1) {
            const size_t off = b * per + (size_t)r0 * a.cols;
            vDSP_vsmul(a.data + off, 1, &down[b], dst + off, 1, (size_t)(r1 - r0) * a.cols);
        });
    }
    return out;
}

HostBuffer::HostBuffer(size_t floats) {
    void* p = nullptr;
    if (posix_memalign(&p, page_size(), page_round(floats * sizeof(float))) != 0) {
        throw std::bad_alloc();
    }
    data_ = static_cast<float*>(p);
}

HostBuffer::~HostBuffer() { std::free(data_); }

void copy_out(const float* src, float* dst, uint32_t batch, size_t per,
              const std::vector<float>* factor) {
    if (!dst) return;
    if (!factor) {
        std::memcpy(dst, src, (size_t)batch * per * sizeof(float));
        return;
    }
    for_each_matrix(batch, per, [&](uint32_t b) {
        vDSP_vsmul(src + b * per, 1, &(*factor)[b], dst + b * per, 1, per);
    });
}

void transpose_out(const float* src, float* dst, uint32_t batch, uint32_t rows, uint32_t cols) {
    if (!dst) return;
    const size_t per = (size_t)rows * cols;
    for_each_matrix(batch, per, [&](uint32_t b) {
        vDSP_mtrans(src + b * per, 1, dst + b * per, 1, cols, rows);
    });
}

namespace {

// Chunks per thread in lapack_batches: more than one, so that threads on the
// faster cores take more of the batch than those on the slower ones.
constexpr uint32_t kChunksPerThread = 4;

// Without per-thread control of Accelerate's threading (macOS 14), the largest
// matrix lapack_batches splits a batch of: well under the sizes Accelerate
// threads one call across cores.
constexpr size_t kUnthreadedMaxFloats = 256 * 256;

// Accelerate's BLAS and LAPACK single-threaded on this thread while in scope,
// where macOS supports choosing (15 and later, and an SDK that declares it).
class SingleThreadedBlas {
public:
    SingleThreadedBlas() {
#if METAL_LINALG_HAVE_BLAS_THREADING
        if (@available(macOS 15.0, *)) {
            old_ = (int)BLASGetThreading();
            BLASSetThreading(BLAS_THREADING_SINGLE_THREADED);
        }
#endif
    }
    ~SingleThreadedBlas() {
#if METAL_LINALG_HAVE_BLAS_THREADING
        if (@available(macOS 15.0, *)) {
            if (old_ >= 0) BLASSetThreading((BLAS_THREADING)old_);
        }
#endif
    }
    SingleThreadedBlas(const SingleThreadedBlas&) = delete;
    SingleThreadedBlas& operator=(const SingleThreadedBlas&) = delete;
private:
    int old_ = -1;
};

bool can_single_thread_blas() {
#if METAL_LINALG_HAVE_BLAS_THREADING
    if (@available(macOS 15.0, *)) return true;
#endif
    return false;
}

} // namespace

void lapack_batches(uint32_t batch, size_t per, const std::function<void(uint32_t, uint32_t)>& f) {
    const uint32_t threads = std::min<uint32_t>(cpu_threads(), batch);
    if (threads <= 1 || (!can_single_thread_blas() && per > kUnthreadedMaxFloats)) {
        f(0, batch);
        return;
    }
    const uint32_t chunks = std::min<uint32_t>(batch, threads * kChunksPerThread);
    std::atomic<uint32_t> next{0};
    std::atomic<bool>     failed{false};
    std::mutex            m;
    std::exception_ptr    error;
    parallel_for(threads, [&](size_t) {
        SingleThreadedBlas single;
        for (uint32_t c; !failed && (c = next.fetch_add(1)) < chunks;) {
            const uint32_t b0 = (uint32_t)((uint64_t)batch * c / chunks);
            const uint32_t b1 = (uint32_t)((uint64_t)batch * (c + 1) / chunks);
            try {
                f(b0, b1);
            } catch (...) {
                std::lock_guard<std::mutex> lock(m);
                if (!error) error = std::current_exception();
                failed = true;
            }
        }
    });
    if (error) std::rethrow_exception(error);
}

MetalRuntime& MetalRuntime::shared(const EmbeddedShader& shader, const char* tag) {
    if (shader.bytes == nullptr || shader.len == 0) {
        throw std::runtime_error(std::string("[") + tag + "] Embedded shader '" +
                                 (shader.name ? shader.name : "") + "' is empty.");
    }

    static std::map<std::string, MetalRuntime> runtimes;

    auto [it, inserted] = runtimes.try_emplace(shader.name);
    if (!inserted) {
        return it->second;
    }

    // Erase the half-built entry on failure so a later call can retry.
    MetalRuntime& rt = it->second;

    rt.device = MTLCreateSystemDefaultDevice();
    if (!rt.device) {
        runtimes.erase(it);
        throw std::runtime_error(std::string("[") + tag + "] No Metal device available.");
    }
    rt.queue = [rt.device newCommandQueue];

    NSError* err = nil;
    dispatch_data_t data = dispatch_data_create(shader.bytes, shader.len, nullptr,
                                                DISPATCH_DATA_DESTRUCTOR_DEFAULT);
    rt.library = [rt.device newLibraryWithData:data error:&err];
    if (!rt.library) {
        runtimes.erase(it);
        throw std::runtime_error(std::string("[") + tag + "] Cannot load embedded shader '" +
                                 shader.name + "': " +
                                 (err ? err.localizedDescription.UTF8String : "unknown error"));
    }

    return rt;
}

id<MTLComputePipelineState> make_pipeline(id<MTLDevice> device,
                                          id<MTLLibrary> library,
                                          NSString* name,
                                          MTLFunctionConstantValues* constants) {
    const std::string fn_name = name.UTF8String;

    NSError* err = nil;
    id<MTLFunction> fn = constants
        ? [library newFunctionWithName:name constantValues:constants error:&err]
        : [library newFunctionWithName:name];
    if (!fn) {
        throw std::runtime_error("Cannot specialise Metal function '" + fn_name + "': " +
                                 (err ? err.localizedDescription.UTF8String
                                      : "not found in library"));
    }

    id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:fn error:&err];
    if (!pso) {
        throw std::runtime_error("Cannot create pipeline for '" + fn_name + "': " +
                                 (err ? err.localizedDescription.UTF8String : "unknown error"));
    }
    return pso;
}

} // namespace metal_linalg::detail
