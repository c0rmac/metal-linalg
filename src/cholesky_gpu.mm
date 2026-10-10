// Cholesky on the GPU (shaders/Cholesky.metal): three kernels, by size.
//
//   - simd: up to 32 x 32, a matrix in a simdgroup's registers (several to a
//     simdgroup below 16), read from the input and written to the output
//     directly, one dispatch;
//   - threadgroup: a matrix to a threadgroup, in a zero-padded workspace,
//     panels of 32 columns (load, factor, store: three dispatches);
//   - blocked: one or a few large matrices across the whole GPU, panels of
//     128 columns: four 32-column sub-panels each, every one brought up to
//     date with the ones before it, factored and solved for the rows below
//     it in one dispatch of chol_panel (a 32 x 32 diagonal block and a strip
//     of rows a threadgroup), the trailing matrix updated by MPS products of
//     rank 128, in column blocks so as to skip most of the upper triangle.
//
// Every path reports a matrix that is not positive definite in `info` (the
// failing pivot's index + 1, as spotrf) and writes it out as NaN.

#include <metal_linalg/core.h>
#include <metal_linalg/device.h>
#include "metal_runtime.h"
#include "shaders.h"

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unistd.h>

using metal_linalg::core::Matrices;
using metal_linalg::detail::AutoreleasePool;
using metal_linalg::detail::MetalRuntime;
using metal_linalg::detail::copy_out;
using metal_linalg::detail::input_buffer;
using metal_linalg::detail::make_pipeline;

namespace metal_linalg {
namespace {

// Must match the structs of the same names in Cholesky.metal.
struct CsParams { uint32_t n, batch, upper; };
struct CwParams { uint32_t n, np, upper, batch; };
struct CfParams { uint32_t n, ld, sw, off, base, np; };
struct CpParams { uint32_t ld, sw, j0, j, jb, rows, nblk; };
struct CdParams { uint32_t ld, sw, nblk, from; };

constexpr uint32_t kPanel = 128;       // the blocked path's panel (64 and 256 were slower to n = 4096)
// The trailing update in column blocks this wide from twice it: its lower
// triangle and the blocks' diagonals, a little over half the full square's
// products (one 8192 x 8192: 50 ms against 68 in one product; blocks of 256
// or 1024 no better).
constexpr uint32_t kTrailingBlock = 512;
constexpr uint32_t kStrip = 64;        // rows a chol_panel threadgroup solves (kPanelStrip)
constexpr uint32_t kSimdPerGroup = 4;  // simdgroups a threadgroup of chol_simd
constexpr size_t kWorkFloats = size_t(1) << 26;   // the workspace a chunk may take (256 MB)

uint32_t round_up(uint32_t x, uint32_t to) { return (x + to - 1) / to * to; }

struct Cache {
    MetalRuntime& rt = MetalRuntime::shared(METAL_LINALG_SHADER(Cholesky), "cholesky");
    id<MTLComputePipelineState> simd[3] = {nil, nil, nil};
    id<MTLComputePipelineState> load = nil, store = nil, factor = nil, panel = nil, put = nil;
    id<MTLBuffer> work = nil, out = nil, fail = nil, diag = nil;
    size_t work_floats = 0, out_floats = 0, fail_count = 0, diag_floats = 0;

    id<MTLComputePipelineState> pipeline(id<MTLComputePipelineState> __strong& p, NSString* name) {
        if (!p) p = make_pipeline(rt.device, rt.library, name, nil);
        return p;
    }
    id<MTLComputePipelineState> simd_pipeline(uint32_t b) {
        const int i = b == 8 ? 0 : b == 16 ? 1 : 2;
        return pipeline(simd[i], [NSString stringWithFormat:@"chol_simd_%u_%u", b, b]);
    }
    id<MTLBuffer> grow(id<MTLBuffer> __strong& b, size_t& have, size_t want, MTLResourceOptions opt) {
        if (!b || have < want) {
            b = [rt.device newBufferWithLength:std::max<size_t>(16, want * sizeof(float)) options:opt];
            if (!b) throw std::runtime_error("[cholesky] could not allocate a GPU workspace");
            have = want;
        }
        return b;
    }
    id<MTLBuffer> workspace(size_t floats) { return grow(work, work_floats, floats, MTLResourceStorageModePrivate); }
    id<MTLBuffer> output(size_t floats) { return grow(out, out_floats, floats, MTLResourceStorageModeShared); }
    id<MTLBuffer> fails(size_t count) { return grow(fail, fail_count, count, MTLResourceStorageModeShared); }
    id<MTLBuffer> diagonal(size_t floats) { return grow(diag, diag_floats, floats, MTLResourceStorageModePrivate); }

    static Cache& shared() {
        static Cache c;
        return c;
    }
};

void check(id<MTLCommandBuffer> cmd, const char* what, uint32_t n, uint32_t batch) {
    [cmd waitUntilCompleted];
    if (cmd.error)
        throw std::runtime_error(std::string("[cholesky] ") + what + ": GPU error: " +
                                 cmd.error.localizedDescription.UTF8String + " (n = " + std::to_string(n) + ", " +
                                 std::to_string(batch) + " matrices).");
}

// The caller's output as a buffer where it is page-aligned, else a workspace
// copied out afterwards.
struct Output {
    id<MTLBuffer> buffer;
    bool direct;
};
Output output_buffer(Cache& c, float* l, size_t floats) {
    if (reinterpret_cast<uintptr_t>(l) % (size_t)getpagesize() == 0)
        return {metal_linalg::detail::wrap_host(c.rt.device, l, floats), true};
    return {c.output(floats), false};
}

void finish(Cache& c, const Output& o, float* l, uint32_t* info, uint32_t batch, uint32_t n) {
    if (!o.direct) copy_out(static_cast<const float*>([o.buffer contents]), l, batch, (size_t)n * n);
    if (info) std::copy_n(static_cast<const uint32_t*>([c.fail contents]), batch, info);
}

// The workspace path's encoders: the input into the padded workspace, and
// the factor out of it.
void encode_load(Cache& c, id<MTLCommandBuffer> cmd, id<MTLBuffer> src, size_t src_off, id<MTLBuffer> w,
                 size_t fail_off, const CwParams& p) {
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:c.pipeline(c.load, @"chol_load")];
    [enc setBuffer:src offset:src_off atIndex:0];
    [enc setBuffer:w offset:0 atIndex:1];
    [enc setBuffer:c.fail offset:fail_off atIndex:2];
    [enc setBytes:&p length:sizeof p atIndex:3];
    [enc dispatchThreads:MTLSizeMake(p.np, p.np, p.batch) threadsPerThreadgroup:MTLSizeMake(32, 8, 1)];
    [enc endEncoding];
}
void encode_store(Cache& c, id<MTLCommandBuffer> cmd, id<MTLBuffer> w, id<MTLBuffer> dst, size_t dst_off,
                  size_t fail_off, const CwParams& p) {
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:c.pipeline(c.store, @"chol_store")];
    [enc setBuffer:w offset:0 atIndex:0];
    [enc setBuffer:dst offset:dst_off atIndex:1];
    [enc setBuffer:c.fail offset:fail_off atIndex:2];
    [enc setBytes:&p length:sizeof p atIndex:3];
    [enc dispatchThreads:MTLSizeMake(p.n, p.n, p.batch) threadsPerThreadgroup:MTLSizeMake(32, 8, 1)];
    [enc endEncoding];
}

// chol_factor's threadgroup: a thread a row of the panel's rows below, and
// the trailing tiles over its simdgroups.
uint32_t factor_threads(Cache& c, uint32_t np) {
    const uint32_t most = (uint32_t)c.pipeline(c.factor, @"chol_factor").maxTotalThreadsPerThreadgroup;
    return std::min<uint32_t>(most, np >= 128 ? 256 : 128);
}
void encode_factor(Cache& c, id<MTLCommandBuffer> cmd, id<MTLBuffer> w, size_t fail_off, const CfParams& p,
                   uint32_t batch) {
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:c.pipeline(c.factor, @"chol_factor")];
    [enc setBuffer:w offset:0 atIndex:0];
    [enc setBuffer:c.fail offset:fail_off atIndex:1];
    [enc setBytes:&p length:sizeof p atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake(batch, 1, 1) threadsPerThreadgroup:MTLSizeMake(factor_threads(c, p.np), 1, 1)];
    [enc endEncoding];
}

MPSMatrix* mps(id<MTLBuffer> b, size_t off, uint32_t rows, uint32_t cols, uint32_t ld, uint32_t count,
               size_t stride) {
    if (count <= 1) {
        MPSMatrixDescriptor* d = [MPSMatrixDescriptor matrixDescriptorWithRows:rows columns:cols
                                                                      rowBytes:(size_t)ld * 4
                                                                      dataType:MPSDataTypeFloat32];
        return [[MPSMatrix alloc] initWithBuffer:b offset:off * 4 descriptor:d];
    }
    // MPS's batched products step from one matrix to the next by rows x
    // rowBytes (see band_reduce.mm), so the descriptor is a whole stride tall
    // and the product's own sizes say what it reads.
    MPSMatrixDescriptor* d = [MPSMatrixDescriptor matrixDescriptorWithRows:(uint32_t)(stride / ld) columns:cols
                                                                  matrices:count rowBytes:(size_t)ld * 4
                                                               matrixBytes:stride * 4 dataType:MPSDataTypeFloat32];
    return [[MPSMatrix alloc] initWithBuffer:b offset:off * 4 descriptor:d];
}

// C -= A B^T, m x n over k, the kernels kept by shape (making one costs more
// than encoding it; MPS kernels are not thread-safe, so a thread's own).
void gemm_sub(id<MTLDevice> dev, id<MTLCommandBuffer> cmd, MPSMatrix* A, MPSMatrix* B, MPSMatrix* C,
              uint32_t m, uint32_t n, uint32_t k, uint32_t batch) {
    using Key = std::tuple<uint32_t, uint32_t, uint32_t>;
    thread_local std::map<Key, MPSMatrixMultiplication*> kernels;
    auto it = kernels.find(Key{m, n, k});
    if (it == kernels.end()) {
        if (kernels.size() >= 4096) kernels.clear();
        MPSMatrixMultiplication* g = [[MPSMatrixMultiplication alloc] initWithDevice:dev transposeLeft:NO
            transposeRight:YES resultRows:m resultColumns:n interiorColumns:k alpha:-1.0 beta:1.0];
        it = kernels.insert_or_assign(Key{m, n, k}, g).first;
    }
    it->second.batchStart = 0;
    it->second.batchSize = batch;
    [it->second encodeToCommandBuffer:cmd leftMatrix:A rightMatrix:B resultMatrix:C];
}

// Matrices a chunk of the workspace paths takes: the workspace at most
// kWorkFloats, and a whole number of them.
uint32_t chunk_of(uint32_t batch, size_t per) {
    return (uint32_t)std::clamp<size_t>(kWorkFloats / std::max<size_t>(per, 1), 1, batch);
}

void require_square(const Matrices& a, const char* what) {
    if (a.rows != a.cols)
        throw std::invalid_argument(std::string("[cholesky] ") + what + ": the matrices must be square, got " +
                                    std::to_string(a.rows) + "x" + std::to_string(a.cols) + ".");
}

} // namespace

namespace core::detail {

void cholesky_simd(const Matrices& a, bool upper, float* l, uint32_t* info) {
    require_square(a, "simd");
    const uint32_t n = a.cols, batch = a.batch;
    if (n == 0 || batch == 0) return;
    if (n > 32) throw std::invalid_argument("[cholesky] simd: n = " + std::to_string(n) + " is more than 32.");
    AutoreleasePool pool;
    Cache& c = Cache::shared();
    const uint32_t b = n <= 8 ? 8 : n <= 16 ? 16 : 32;
    const size_t per = (size_t)n * n;
    const Output o = output_buffer(c, l, per * batch);
    id<MTLBuffer> src = input_buffer(c.rt.device, a);
    c.fails(batch);
    id<MTLCommandBuffer> cmd = [c.rt.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:c.simd_pipeline(b)];
    [enc setBuffer:src offset:0 atIndex:0];
    [enc setBuffer:o.buffer offset:0 atIndex:1];
    [enc setBuffer:c.fail offset:0 atIndex:2];
    const CsParams p{n, batch, upper ? 1u : 0u};
    [enc setBytes:&p length:sizeof p atIndex:3];
    const uint32_t per_tg = kSimdPerGroup * (32 / b);
    [enc setThreadgroupMemoryLength:kSimdPerGroup * 32 * 33 * sizeof(float) atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake((batch + per_tg - 1) / per_tg, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(32 * kSimdPerGroup, 1, 1)];
    [enc endEncoding];
    [cmd commit];
    check(cmd, "simd", n, batch);
    finish(c, o, l, info, batch, n);
}

void cholesky_threadgroup(const Matrices& a, bool upper, float* l, uint32_t* info) {
    require_square(a, "threadgroup");
    const uint32_t n = a.cols, batch = a.batch;
    if (n == 0 || batch == 0) return;
    AutoreleasePool pool;
    Cache& c = Cache::shared();
    const uint32_t np = round_up(n, 32);
    const size_t per = (size_t)n * n, wper = (size_t)np * np;
    const uint32_t chunk = chunk_of(batch, wper);
    const Output o = output_buffer(c, l, per * batch);
    id<MTLBuffer> src = input_buffer(c.rt.device, a);
    id<MTLBuffer> w = c.workspace(wper * chunk);
    c.fails(batch);
    for (uint32_t b0 = 0; b0 < batch; b0 += chunk) {
        const uint32_t bc = std::min(chunk, batch - b0);
        const CwParams wp{n, np, upper ? 1u : 0u, bc};
        id<MTLCommandBuffer> cmd = [c.rt.queue commandBuffer];
        encode_load(c, cmd, src, b0 * per * 4, w, b0 * 4, wp);
        encode_factor(c, cmd, w, b0 * 4, CfParams{n, np, (uint32_t)wper, 0, 0, np}, bc);
        encode_store(c, cmd, w, o.buffer, b0 * per * 4, b0 * 4, wp);
        [cmd commit];
        check(cmd, "threadgroup", n, bc);
    }
    finish(c, o, l, info, batch, n);
}

void cholesky_blocked(const Matrices& a, bool upper, float* l, uint32_t* info) {
    require_square(a, "blocked");
    const uint32_t n = a.cols, batch = a.batch;
    if (n == 0 || batch == 0) return;
    AutoreleasePool pool;
    Cache& c = Cache::shared();
    id<MTLDevice> dev = c.rt.device;
    const uint32_t np = round_up(n, 32);
    const size_t per = (size_t)n * n, wper = (size_t)np * np;
    // MPS's batched views read a stride's slack past the last matrix
    const uint32_t chunk = chunk_of(batch, wper);
    const Output o = output_buffer(c, l, per * batch);
    id<MTLBuffer> src = input_buffer(dev, a);
    id<MTLBuffer> w = c.workspace(wper * (chunk + 1));
    id<MTLBuffer> d = n > kPanel ? c.diagonal((size_t)chunk * np * 32) : nil;
    c.fails(batch);
    for (uint32_t b0 = 0; b0 < batch; b0 += chunk) {
        const uint32_t bc = std::min(chunk, batch - b0);
        const size_t fo = b0 * 4;
        const CwParams wp{n, np, upper ? 1u : 0u, bc};
        id<MTLCommandBuffer> cmd = [c.rt.queue commandBuffer];
        encode_load(c, cmd, src, b0 * per * 4, w, fo, wp);
        if (n <= kPanel) {
            // one panel: the threadgroup kernel on the whole matrix
            encode_factor(c, cmd, w, fo, CfParams{n, np, (uint32_t)wper, 0, 0, np}, bc);
        } else {
            const uint32_t nblk = np / 32;
            for (uint32_t j0 = 0; j0 < n; j0 += kPanel) {
                const uint32_t j1 = std::min(n, j0 + kPanel);
                // the panel's sub-panels, one serial encoder: each dispatch
                // sees the one before it
                id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
                [enc setComputePipelineState:c.pipeline(c.panel, @"chol_panel")];
                [enc setBuffer:w offset:0 atIndex:0];
                [enc setBuffer:c.fail offset:fo atIndex:1];
                [enc setBuffer:d offset:0 atIndex:2];
                for (uint32_t j = j0; j < j1; j += 32) {
                    const uint32_t rows = np - j - 32;
                    const CpParams pp{np, (uint32_t)wper, j0, j, std::min(32u, n - j), rows, nblk};
                    [enc setBytes:&pp length:sizeof pp atIndex:3];
                    [enc dispatchThreadgroups:MTLSizeMake(std::max(1u, (rows + kStrip - 1) / kStrip), bc, 1)
                        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                }
                [enc endEncoding];
                if (j1 < n) {   // A(j1:, j1:) -= L(j1:, j0:j1) L(j1:, j0:j1)^T, on and below the diagonal
                    const uint32_t m = n - j1, k = j1 - j0;
                    const uint32_t cw = m >= 2 * kTrailingBlock ? kTrailingBlock : m;
                    for (uint32_t c0 = 0; c0 < m; c0 += cw) {
                        const uint32_t wc = std::min(cw, m - c0);
                        gemm_sub(dev, cmd, mps(w, (size_t)(j1 + c0) * np + j0, m - c0, k, np, bc, wper),
                                 mps(w, (size_t)(j1 + c0) * np + j0, wc, k, np, bc, wper),
                                 mps(w, (size_t)(j1 + c0) * np + j1 + c0, m - c0, wc, np, bc, wper), m - c0, wc, k, bc);
                    }
                }
            }
            // the diagonal blocks into place
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:c.pipeline(c.put, @"chol_put")];
            [enc setBuffer:w offset:0 atIndex:0];
            [enc setBuffer:c.fail offset:fo atIndex:1];
            [enc setBuffer:d offset:0 atIndex:2];
            const CdParams dp{np, (uint32_t)wper, nblk, 0};
            [enc setBytes:&dp length:sizeof dp atIndex:3];
            [enc dispatchThreads:MTLSizeMake(1024, nblk, bc) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc endEncoding];
        }
        encode_store(c, cmd, w, o.buffer, b0 * per * 4, fo, wp);
        [cmd commit];
        check(cmd, "blocked", n, bc);
    }
    finish(c, o, l, info, batch, n);
}

} // namespace core::detail
} // namespace metal_linalg
