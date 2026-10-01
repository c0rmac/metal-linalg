#include "metal_runtime.h"
#include "shaders.h"

#include <cmath>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>
#include <unistd.h>

namespace metal_linalg::detail {

uint pad_up(uint v, uint multiple) {
    return ((v + multiple - 1) / multiple) * multiple;
}

mlx::core::array prepare_input(const mlx::core::array& a) {
    using namespace mlx::core;

    array out = astype(a, float32);
    eval({out});   // flags are only meaningful once the array is materialised
    if (!out.flags().row_contiguous) {
        out = contiguous(out);
        eval({out});
    }
    // A contiguous view into a larger array (one slice of a batch, say) can
    // start mid-page, which newBufferWithBytesNoCopy rejects. Copy it once.
    if (reinterpret_cast<uintptr_t>(out.data<float>()) % (uintptr_t)getpagesize() != 0) {
        out = copy(out);
        eval({out});
    }
    return out;
}

ScaledInput prepare_input_scaled(const mlx::core::array& a) {
    using namespace mlx::core;

    array x = prepare_input(a);
    const Shape& shape = x.shape();
    if (x.ndim() < 2 || x.size() == 0) return {x, array(1.0f), false, {}};

    const size_t per = (size_t)shape[shape.size() - 2] * shape[shape.size() - 1];
    const size_t batch = x.size() / per;

    // Largest magnitude per matrix. Small inputs are scanned in place, which
    // is cheaper than a GPU launch and a readback; large ones are reduced on
    // the GPU.
    std::vector<float> amax(batch, 0.0f);
    std::vector<char>  finite(batch, 1);
    if (x.size() <= (1u << 20)) {
        const float* d = x.data<float>();
        for (size_t b = 0; b < batch; ++b) {
            float m = 0.0f;
            bool ok = true;
            for (size_t i = 0; i < per; ++i) {
                const float v = std::fabs(d[b * per + i]);
                ok = ok && std::isfinite(v);
                m = v > m ? v : m;
            }
            amax[b] = m;
            finite[b] = ok;
        }
    } else {
        array mx  = reshape(max(abs(x), std::vector<int>{-2, -1}), {-1});
        array bad = reshape(any(logical_or(isnan(x), isinf(x)), std::vector<int>{-2, -1}), {-1});
        eval({mx, bad});
        for (size_t b = 0; b < batch; ++b) {
            amax[b] = mx.data<float>()[b];
            finite[b] = !bad.data<bool>()[b];
        }
    }

    std::vector<float> down(batch, 1.0f), up(batch, 1.0f);
    bool any_scaled = false;
    for (size_t b = 0; b < batch; ++b) {
        if (!finite[b] || !(amax[b] > 0.0f)) continue;
        int e = 0;
        std::frexp(amax[b], &e);
        if (e == 0) continue;   // already in [0.5, 1)
        down[b] = std::ldexp(1.0f, -e);
        up[b]   = std::ldexp(1.0f, e);
        any_scaled = true;
    }
    std::vector<char> nonfinite(batch);
    for (size_t b = 0; b < batch; ++b) nonfinite[b] = !finite[b];
    if (!any_scaled) return {x, array(1.0f), false, nonfinite};

    Shape fshape(shape.begin(), shape.end() - 2);
    fshape.push_back(1);
    fshape.push_back(1);
    array scaled = multiply(x, array(down.begin(), fshape, float32));
    return {prepare_input(scaled), array(up.begin(), fshape, float32), true, nonfinite};
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
