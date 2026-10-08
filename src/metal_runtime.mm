#include "metal_runtime.h"
#include "known_buffers.h"
#include "shaders.h"

#include <metal_linalg/device.h>

#include <Accelerate/Accelerate.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
// Built with an SDK older than macOS 15's, the library cannot switch
// Accelerate's threading off and splits a batch only where Accelerate would
// not thread anyway (see lapack_batches).
#include "blas_threading.h"

#include <atomic>
#include <chrono>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <map>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <thread>
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

// The buffers KnownBuffer registers, any thread's (a batch's GPU share may
// run on a thread of its own).
struct Known {
    const void* data;
    void*       buffer;
};
std::mutex& known_mutex() {
    static std::mutex m;
    return m;
}
std::vector<Known>& known() {
    static std::vector<Known> k;
    return k;
}

// A registered buffer that starts at `data` and holds `bytes`, else nil.
id<MTLBuffer> known_buffer(const void* data, size_t bytes) {
    std::lock_guard<std::mutex> lock(known_mutex());
    for (const Known& k : known()) {
        if (k.data != data) continue;
        id<MTLBuffer> b = (__bridge id<MTLBuffer>)k.buffer;
        if (b.contents == data && b.length >= bytes) return b;
    }
    return nil;
}

} // namespace

void know_buffer(const void* data, void* buffer) {
    if (!data || !buffer) return;
    std::lock_guard<std::mutex> lock(known_mutex());
    known().push_back({data, buffer});
}

void forget_buffer(const void* data, void* buffer) {
    if (!data || !buffer) return;
    std::lock_guard<std::mutex> lock(known_mutex());
    std::vector<Known>& k = known();
    for (size_t i = k.size(); i-- > 0;)
        if (k[i].data == data && k[i].buffer == buffer) {
            k.erase(k.begin() + (long)i);
            break;
        }
}

KnownBuffer::KnownBuffer(const void* data, void* buffer) : data_(data), buffer_(buffer) { know_buffer(data_, buffer_); }

KnownBuffer::~KnownBuffer() { forget_buffer(data_, buffer_); }

id<MTLBuffer> wrap_host(id<MTLDevice> device, float* data, size_t floats) {
    if (id<MTLBuffer> b = known_buffer(data, floats * sizeof(float))) return b;
    id<MTLBuffer> b = [device newBufferWithBytesNoCopy:data
                                                length:page_round(floats * sizeof(float))
                                               options:MTLResourceStorageModeShared
                                           deallocator:nil];
    if (!b) throw std::runtime_error("Could not wrap host memory as a Metal buffer.");
    return b;
}

id<MTLBuffer> input_buffer(id<MTLDevice> device, const Matrices& a, bool transpose) {
    const size_t bytes = element_count(a) * sizeof(float);
    if (!transpose)
        if (id<MTLBuffer> b = known_buffer(a.data, bytes)) return b;
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

void transpose_scaled(const float* src, size_t ld_src, float* dst, size_t ld_dst, uint32_t rows, uint32_t cols,
                      float scale) {
    constexpr uint32_t kBlock = 32;
    const uint32_t col_blocks = (cols + kBlock - 1) / kBlock;
    // A strip of columns per task: each reads kBlock floats of a row at a time
    // and writes whole columns.
    parallel_for(rows * (size_t)cols >= kGrain ? col_blocks : 1, [&](size_t t) {
        const uint32_t j0 = rows * (size_t)cols >= kGrain ? (uint32_t)t * kBlock : 0;
        const uint32_t j1 = rows * (size_t)cols >= kGrain ? std::min(cols, j0 + kBlock) : cols;
        for (uint32_t i0 = 0; i0 < rows; i0 += kBlock) {
            const uint32_t i1 = std::min(rows, i0 + kBlock);
            for (uint32_t j = j0; j < j1; ++j) {
                float* d = dst + (size_t)j * ld_dst;
                for (uint32_t i = i0; i < i1; ++i) d[i] = scale * src[(size_t)i * ld_src + j];
            }
        }
    });
}

void MpsGemm::add_buffer(id<MTLBuffer> buffer) {
    regions_.push_back({static_cast<const char*>(buffer.contents), (size_t)buffer.length, buffer});
}

// Wrapping costs tens of microseconds, and most of a solve's temporaries
// never reach the GPU's products: only when one does.
void MpsGemm::add(const float* base, size_t floats) {
    regions_.push_back({reinterpret_cast<const char*>(base), floats * sizeof(float), nil});
}

void MpsGemm::remove(const float* base) {
    const char* b = reinterpret_cast<const char*>(base);
    regions_.erase(std::remove_if(regions_.begin(), regions_.end(), [&](const Region& r) { return r.base == b; }),
                   regions_.end());
}

id<MTLBuffer> MpsGemm::find(const float* p, long rows, long cols, long ld, size_t& offset) {
    const char* a = reinterpret_cast<const char*>(p);
    const size_t span = ((size_t)(cols - 1) * ld + rows) * sizeof(float);
    for (Region& r : regions_)
        if (a >= r.base && a + span <= r.base + r.bytes) {
            offset = (size_t)(a - r.base);
            if (!r.buffer) r.buffer = wrap_host(device_, reinterpret_cast<float*>(const_cast<char*>(r.base)), r.bytes / sizeof(float));
            return r.buffer;
        }
    return nil;
}

bool MpsGemm::gemm(long m, long n, long k, const float* A, long lda, const float* B, long ldb, float* C, long ldc,
                   bool accumulate) {
    if (after_ && after_.status < MTLCommandBufferStatusCompleted) return false;   // the GPU still busy
    size_t oa = 0, ob = 0, oc = 0;
    id<MTLBuffer> ba = find(A, m, k, lda, oa), bb = find(B, k, n, ldb, ob), bc = find(C, m, n, ldc, oc);
    if (!ba || !bb || !bc) return false;
    // On the row-major views of the column-major operands: C^T = B^T A^T.
    auto view = [](id<MTLBuffer> b, size_t off, long rows, long cols, long ld) {
        MPSMatrixDescriptor* d = [MPSMatrixDescriptor matrixDescriptorWithRows:(NSUInteger)rows
                                                                       columns:(NSUInteger)cols
                                                                      rowBytes:(NSUInteger)ld * sizeof(float)
                                                                      dataType:MPSDataTypeFloat32];
        return [[MPSMatrix alloc] initWithBuffer:b offset:off descriptor:d];
    };
    @autoreleasepool {
        id<MTLCommandBuffer> cb = [queue_ commandBuffer];
        MPSMatrixMultiplication* g = [[MPSMatrixMultiplication alloc] initWithDevice:device_ transposeLeft:NO
                                     transposeRight:NO resultRows:(NSUInteger)n resultColumns:(NSUInteger)m
                                     interiorColumns:(NSUInteger)k alpha:1.0 beta:accumulate ? 1.0 : 0.0];
        [g encodeToCommandBuffer:cb leftMatrix:view(bb, ob, n, k, ldb) rightMatrix:view(ba, oa, k, m, lda)
                    resultMatrix:view(bc, oc, n, m, ldc)];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.error)   // C may be partly written: not one for the CPU to redo
            throw std::runtime_error(std::string("divide and conquer: GPU error in a product: ") +
                                     cb.error.localizedDescription.UTF8String);
    }
    return true;
}

uint32_t jacobi_round_budget(double solve_core_ms, uint32_t rounds, const char* env) {
    constexpr double kSweeps = 8.0;   // a typical solve's, for the cost of a round
    double target = 40.0;
    if (const char* s = env ? std::getenv(env) : nullptr) {   // fractions too (the tests split to a round)
        const double v = std::strtod(s, nullptr);
        if (v > 0.0) target = v;
    }
    if (rounds == 0 || solve_core_ms <= target) return 0;
    const double round_ms = solve_core_ms / (kSweeps * rounds);
    return (uint32_t)std::max(1.0, std::floor(target / round_ms));
}

void run_split_jacobi(id<MTLCommandQueue> queue, id<MTLBuffer> state, size_t offset, uint32_t count,
                      uint32_t per_buffer, uint32_t max_dispatches,
                      const std::function<void(id<MTLComputeCommandEncoder>)>& encode, const std::string& what) {
    constexpr uint32_t kDone = 2;   // kJacobiDone
    auto* st = static_cast<unsigned char*>(state.contents) + offset;
    std::memset(st, 0, (size_t)count * kJacobiStateBytes);
    for (uint32_t sent = 0; sent < max_dispatches;) {
        id<MTLCommandBuffer> cmd = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        for (uint32_t d = 0; d < per_buffer && sent < max_dispatches; ++d, ++sent) encode(enc);
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        if (cmd.error)
            throw std::runtime_error(what + ": GPU kernel error: " + cmd.error.localizedDescription.UTF8String);
        bool all = true;
        for (uint32_t b = 0; b < count && all; ++b) {
            uint32_t flags;
            std::memcpy(&flags, st + (size_t)b * kJacobiStateBytes + 20, 4);
            all = (flags & kDone) != 0;
        }
        if (all) return;
    }
}

void compact_wy_t(uint32_t m, uint32_t kb, const float* V, const float* tau, float* T, uint32_t ldt) {
    std::vector<float> G((size_t)kb * kb);
    cblas_ssyrk(CblasColMajor, CblasUpper, CblasTrans, (int)kb, (int)m, 1.0f, V, (int)m, 0.0f, G.data(), (int)kb);
    for (uint32_t j = 0; j < kb; ++j) {   // T(0:j, j) = -tau_j T(0:j, 0:j) G(0:j, j)
        for (uint32_t i = 0; i < j; ++i) {
            float s = 0.0f;
            for (uint32_t k = i; k < j; ++k) s += T[i + (size_t)k * ldt] * G[k + (size_t)j * kb];
            T[i + (size_t)j * ldt] = -tau[j] * s;
        }
        T[j + (size_t)j * ldt] = tau[j];
    }
}

namespace {

// Chunks per thread in lapack_batches: more than one, so that threads on the
// faster cores take more of the batch than those on the slower ones.
constexpr uint32_t kChunksPerThread = 4;

// Without per-thread control of Accelerate's threading (macOS 14), the largest
// matrix lapack_batches splits a batch of: well under the sizes Accelerate
// threads one call across cores.
constexpr size_t kUnthreadedMaxFloats = 256 * 256;

// Set on share_batch's CPU workers, each of which is already one of the
// cores' worth: lapack_batches then solves its matrices on the calling thread.
thread_local bool tls_share_worker = false;

} // namespace

void lapack_batches(uint32_t batch, size_t per, const std::function<void(uint32_t, uint32_t)>& f) {
    if (tls_share_worker) {
        f(0, batch);   // single-threaded already (share_batch)
        return;
    }
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

void share_batch(uint32_t batch, uint32_t gpu_chunk, uint32_t cpu_chunk,
                 const std::function<void(uint32_t, uint32_t)>& gpu,
                 const std::function<void(uint32_t, uint32_t)>& cpu) {
    using clock = std::chrono::steady_clock;
    gpu_chunk = std::max(gpu_chunk, 1u);
    cpu_chunk = std::max(cpu_chunk, 1u);
    std::mutex            m;
    uint32_t              front = 0, back = batch;   // [front, back) is unclaimed
    std::atomic<uint32_t> cpu_done{0};
    std::atomic<bool>     failed{false};
    std::exception_ptr    error;
    auto fail = [&] {
        std::lock_guard<std::mutex> lock(m);
        if (!error) error = std::current_exception();
        failed = true;
    };

    // The CPU: one worker per thread, each a core's worth, solving a few
    // matrices at a time with Accelerate's threading off; two threads fewer
    // than cpu_threads(), because the GPU's side needs a core for its own host
    // work (the input scan and copies, waiting on the GPU). With a worker on
    // every core, the GPU's chunks took 5-7x their time alone on an M5 Pro, and
    // sharing was slower than the GPU alone; with two cores left, 1.3-1.5x
    // faster.
    const uint32_t threads = std::max(1u, cpu_threads());
    const uint32_t workers = can_single_thread_blas() ? std::max(1u, threads > 2 ? threads - 2 : 1u) : 1u;
    std::vector<std::thread> pool;
    pool.reserve(workers);
    for (uint32_t w = 0; w < workers; ++w) {
        pool.emplace_back([&] {
            SingleThreadedBlas single;
            tls_share_worker = can_single_thread_blas();
            try {
                while (!failed) {
                    uint32_t b0, count;
                    {
                        std::lock_guard<std::mutex> lock(m);
                        if (back <= front) break;
                        count = std::min(cpu_chunk, back - front);
                        back -= count;
                        b0 = back;
                    }
                    cpu(b0, count);
                    cpu_done += count;
                }
            } catch (...) {
                fail();
            }
            tls_share_worker = false;
        });
    }

    // The GPU, on this thread: a quarter of the batch first; then, from the
    // two rates measured so far, the share of what is left that it would
    // finish as the CPU finishes the rest, until nothing is left.
    const auto start = clock::now();
    double gpu_seconds = 0.0;
    uint32_t gpu_done = 0;
    try {
        while (!failed) {
            uint32_t b0, count;
            {
                std::lock_guard<std::mutex> lock(m);
                if (back <= front) break;
                const uint32_t left = back - front;
                if (gpu_done == 0) {
                    count = std::max(left / 4, std::min(left, gpu_chunk));
                } else {
                    const double elapsed = std::chrono::duration<double>(clock::now() - start).count();
                    const double rg = gpu_done / std::max(gpu_seconds, 1e-9);
                    const double rc = cpu_done.load() / std::max(elapsed, 1e-9);
                    count = (uint32_t)std::ceil(left * rg / (rg + rc));
                    count = std::min(left, std::max(count, std::min(left, gpu_chunk / 4 + 1)));
                }
                b0 = front;
                front += count;
            }
            const auto t0 = clock::now();
            gpu(b0, count);
            gpu_seconds += std::chrono::duration<double>(clock::now() - t0).count();
            gpu_done += count;
        }
    } catch (...) {
        fail();
    }
    for (std::thread& t : pool) t.join();
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
