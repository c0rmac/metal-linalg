// LU with partial pivoting on the GPU, for large matrices (LuBackend::blocked),
// and the solve and the inverse built on it.
//
// One matrix at a time, in a column-major workspace in shared memory, with
// its leading dimension padded off a power of two (as LAPACK's in
// lu_cpu.mm). Panels of 128 columns: each is factored on the CPU (recursively,
// panel_lu) in place -- the pivot search is a chain of reductions down a
// column, which the CPU does in a fraction of what the GPU's launches would
// cost -- while the GPU, in the same memory, swaps the other columns' rows
// (lu_gather, shaders/LU.metal), forms U12 = L11^-1 A12 (an MPS product with
// the inverse of the panel's unit lower triangle, which the CPU makes with
// strtri) and updates the trailing matrix, A22 -= L21 U12 (an MPS product).
// With a look-ahead of one panel: the CPU brings the next panel's columns up
// to date itself (LAPACK's step, on Accelerate's matrix units) and factors it
// while the GPU updates the rest. The matrix moves in and out by transposes on
// the GPU. On an M5 Pro one 4096 x 4096 in 18 ms and one 8192 x 8192 in 96,
// against 62 and 427 for sgetrf on every core (169 at 4096 at an unpadded
// leading dimension).
//
// The solve, A X = B: B with its rows permuted, then L^-1 and U^-1 a block of
// 128 rows at a time on the GPU, each block's step two MPS products -- one
// with the inverse of the diagonal block (L11^-1 kept from the factorization,
// U11^-1 made by strtri), one updating the rows still to come -- and the
// inverse the same on the identity. With only a few right-hand sides, LAPACK's
// sgetrs on the factorization in the same memory instead (policy
// gpu_solve_min_rhs).
//
// Row-major views: a column-major block of the workspace read row-major is its
// transpose, so the products are written in transposed form (C^T = B^T A^T)
// or with MPS's transpose flags, and nothing is copied to change layout.
#ifndef ACCELERATE_NEW_LAPACK
#define ACCELERATE_NEW_LAPACK   // LAPACK's current interface; before any Accelerate header
#endif
#include <metal_linalg/core.h>
#include "metal_runtime.h"
#include "shaders.h"
#include "transpose.h"

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <Accelerate/Accelerate.h>

#include <algorithm>
#include <cstring>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

using metal_linalg::core::Matrices;
using metal_linalg::detail::AutoreleasePool;
using metal_linalg::detail::MetalRuntime;
using metal_linalg::detail::copy_out;
using metal_linalg::detail::input_buffer;
using metal_linalg::detail::make_pipeline;
using metal_linalg::detail::padded_ld;
using metal_linalg::detail::transpose;

namespace metal_linalg::core::detail {
namespace {

// Must match the structs of the same names in LU.metal.
struct GatherParams { uint32_t ld, count, c0, c1; };
struct MoveParams { uint32_t rows, cols, ldi, ldo; };

constexpr uint32_t kPanel = 128;   // 64: 1.4x slower at 4096; 256: its L11^-1 3x less accurate

struct Cache {
    MetalRuntime& rt = MetalRuntime::shared(METAL_LINALG_SHADER(LU), "lu");
    id<MTLComputePipelineState> gather = nil, move = nil, permute = nil, identity = nil;
    id<MTLBuffer> w = nil, linv = nil, uinv = nil, swaps[2] = {nil, nil}, temp = nil, y = nil, out = nil,
                  perm = nil;
    size_t w_floats = 0, linv_floats = 0, uinv_floats = 0, swap_count[2] = {0, 0}, temp_floats = 0,
           y_floats = 0, out_floats = 0, perm_count = 0;

    id<MTLBuffer> grow(id<MTLBuffer> __strong& b, size_t& have, size_t want, MTLResourceOptions opt) {
        if (!b || have < want) {
            b = [rt.device newBufferWithLength:std::max<size_t>(16, want * sizeof(float)) options:opt];
            if (!b) throw std::runtime_error("[lu] could not allocate a GPU workspace");
            have = want;
        }
        return b;
    }
    id<MTLComputePipelineState> pipeline(id<MTLComputePipelineState> __strong& p, NSString* name) {
        if (!p) p = make_pipeline(rt.device, rt.library, name, nil);
        return p;
    }
    id<MTLComputePipelineState> gather_pipeline() { return pipeline(gather, @"lu_gather"); }
    static Cache& shared() {
        static Cache c;
        return c;
    }
};

// MPS kernels kept by shape (making one costs more than encoding it; MPS
// kernels are not thread-safe, so a thread's own).
MPSMatrixMultiplication* gemm(id<MTLDevice> dev, uint32_t m, uint32_t n, uint32_t k, bool ta, float alpha,
                              float beta) {
    using Key = std::tuple<uint32_t, uint32_t, uint32_t, bool, float, float>;
    thread_local std::map<Key, MPSMatrixMultiplication*> kernels;
    const Key key{m, n, k, ta, alpha, beta};
    auto it = kernels.find(key);
    if (it == kernels.end()) {
        if (kernels.size() >= 4096) kernels.clear();
        MPSMatrixMultiplication* g = [[MPSMatrixMultiplication alloc] initWithDevice:dev transposeLeft:ta
            transposeRight:NO resultRows:m resultColumns:n interiorColumns:k alpha:alpha beta:beta];
        it = kernels.emplace(key, g).first;
    }
    return it->second;
}
MPSMatrixCopy* copier(id<MTLDevice> dev, uint32_t rows, uint32_t cols) {
    using Key = std::pair<uint32_t, uint32_t>;
    thread_local std::map<Key, MPSMatrixCopy*> kernels;
    auto it = kernels.find(Key{rows, cols});
    if (it == kernels.end()) {
        if (kernels.size() >= 4096) kernels.clear();
        it = kernels.emplace(Key{rows, cols}, [[MPSMatrixCopy alloc] initWithDevice:dev copyRows:rows copyColumns:cols
                                                              sourcesAreTransposed:NO destinationsAreTransposed:NO]).first;
    }
    return it->second;
}

// rows x cols floats at `off` of b, row-major with row stride ld.
MPSMatrix* view(id<MTLBuffer> b, size_t off, uint32_t rows, uint32_t cols, uint32_t ld) {
    MPSMatrixDescriptor* d = [MPSMatrixDescriptor matrixDescriptorWithRows:rows columns:cols rowBytes:(size_t)ld * 4
                                                                  dataType:MPSDataTypeFloat32];
    return [[MPSMatrix alloc] initWithBuffer:b offset:off * 4 descriptor:d];
}

void wait(id<MTLCommandBuffer> cmd, uint32_t n) {
    [cmd waitUntilCompleted];
    if (cmd.error)
        throw std::runtime_error(std::string("[lu] GPU error: ") + cmd.error.localizedDescription.UTF8String +
                                 " (n = " + std::to_string(n) + ").");
}

// LU of the m x n column-major panel at A (leading dimension lda), with
// partial pivoting, recursively: the left half, its swaps on the right half,
// the right half's top rows by a triangular solve and the rest by one product
// (on Accelerate's matrix units), then the right half, and its swaps back on
// the left. Leaves of 8 columns to sgetrf. 10-25% faster than sgetrf on the
// whole panel from 2048 rows (M5 Pro), the same pivots. ipiv 1-based,
// relative to the panel; the return sgetrf's info.
__LAPACK_int panel_lu(uint32_t m, uint32_t n, float* A, uint32_t lda, __LAPACK_int* ipiv) {
    if (n <= 8 || m < 1024) {
        __LAPACK_int lm = (__LAPACK_int)m, ln = (__LAPACK_int)n, ll = (__LAPACK_int)lda, info = 0;
        sgetrf_(&lm, &ln, A, &ll, ipiv, &info);
        if (info < 0) throw std::runtime_error("[lu] sgetrf rejected argument " + std::to_string(-info));
        return info;
    }
    const uint32_t n1 = n / 2, n2 = n - n1;
    __LAPACK_int info = panel_lu(m, n1, A, lda, ipiv);
    __LAPACK_int ln2 = (__LAPACK_int)n2, ll = (__LAPACK_int)lda, k1 = 1, k2 = (__LAPACK_int)n1, inc = 1;
    slaswp_(&ln2, A + (size_t)n1 * lda, &ll, &k1, &k2, ipiv, &inc);
    cblas_strsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasUnit, (__LAPACK_int)n1, ln2, 1.0f, A, ll,
                A + (size_t)n1 * lda, ll);
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, (__LAPACK_int)(m - n1), ln2, (__LAPACK_int)n1, -1.0f,
                A + n1, ll, A + (size_t)n1 * lda, ll, 1.0f, A + (size_t)n1 * lda + n1, ll);
    const __LAPACK_int info2 = panel_lu(m - n1, n2, A + (size_t)n1 * lda + n1, lda, ipiv + n1);
    for (uint32_t i = n1; i < n; ++i) ipiv[i] += (__LAPACK_int)n1;
    __LAPACK_int ln1 = (__LAPACK_int)n1, kk1 = (__LAPACK_int)n1 + 1, kk2 = (__LAPACK_int)n;
    slaswp_(&ln1, A, &ll, &kk1, &kk2, ipiv, &inc);
    return info ? info : (info2 ? info2 + (__LAPACK_int)n1 : 0);
}

// One matrix's factorization in the workspace: c.w holds L and U
// column-major (leading dimension ld), c.linv every panel's L11^-1 (column-
// major, kPanel x kPanel apart, leading dimension kPanel).
struct Factor {
    uint32_t n = 0, ld = 0, info = 0;
    std::vector<__LAPACK_int> ipiv;   // sgetrf's, 1-based
};

// out[c][r] = in[r][c] for in rows x cols (row strides ldi, ldo), on the GPU.
void encode_transpose(Cache& c, id<MTLCommandBuffer> cmd, id<MTLBuffer> in, size_t in_off, id<MTLBuffer> out,
                      size_t out_off, uint32_t rows, uint32_t cols, uint32_t ldi, uint32_t ldo) {
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:c.pipeline(c.move, @"lu_transpose")];
    [enc setBuffer:in offset:in_off * 4 atIndex:0];
    [enc setBuffer:out offset:out_off * 4 atIndex:1];
    const MoveParams mp{rows, cols, ldi, ldo};
    [enc setBytes:&mp length:sizeof mp atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake((cols + 31) / 32, (rows + 31) / 32, 1) threadsPerThreadgroup:MTLSizeMake(32, 8, 1)];
    [enc endEncoding];
}

// The matrix at `off` floats into `src` (n x n, row-major), factored.
void factor(Cache& c, id<MTLBuffer> src, size_t off, uint32_t n, Factor& f) {
    id<MTLDevice> dev = c.rt.device;
    const uint32_t ld = padded_ld(n), nblocks = (n + kPanel - 1) / kPanel;
    f.n = n;
    f.ld = ld;
    f.info = 0;
    f.ipiv.assign(n, 0);
    c.grow(c.w, c.w_floats, (size_t)n * ld, MTLResourceStorageModeShared);
    c.grow(c.linv, c.linv_floats, (size_t)nblocks * kPanel * kPanel, MTLResourceStorageModeShared);
    for (int s = 0; s < 2; ++s) c.grow(c.swaps[s], c.swap_count[s], 4 * kPanel, MTLResourceStorageModeShared);
    c.grow(c.temp, c.temp_floats, (size_t)kPanel * n, MTLResourceStorageModePrivate);
    float* W = static_cast<float*>([c.w contents]);
    {   // row-major in, column-major
        id<MTLCommandBuffer> cmd = [c.rt.queue commandBuffer];
        encode_transpose(c, cmd, src, off, c.w, 0, n, n, n, ld);
        [cmd commit];
        wait(cmd, n);
    }
    uint32_t count[2] = {0, 0};

    // The panel at j on the CPU: sgetrf, its swaps as a gather list, L11^-1.
    auto panel = [&](uint32_t j, int s) {
        const uint32_t jb = std::min(kPanel, n - j);
        const __LAPACK_int err = panel_lu(n - j, jb, W + (size_t)j * ld + j, ld, f.ipiv.data() + j);
        if (err > 0 && f.info == 0) f.info = j + (uint32_t)err;
        // the sequential swaps composed: row r ends up with row src[r]'s content
        std::map<uint32_t, uint32_t> src;
        for (uint32_t k = j; k < j + jb; ++k) {
            f.ipiv[k] += (__LAPACK_int)j;   // relative to the panel -> global, still 1-based
            const uint32_t r = (uint32_t)f.ipiv[k] - 1;
            if (r == k) continue;
            const auto a_ = src.find(k), b_ = src.find(r);
            const uint32_t ka = a_ == src.end() ? k : a_->second, rb = b_ == src.end() ? r : b_->second;
            src[k] = rb;
            src[r] = ka;
        }
        uint32_t* e = static_cast<uint32_t*>([c.swaps[s] contents]);
        uint32_t cnt = 0;
        for (const auto& [dst, from] : src)
            if (dst != from) { e[2 * cnt] = dst; e[2 * cnt + 1] = from; ++cnt; }
        count[s] = cnt;
        // L11^-1, unit lower, column-major (leading dimension kPanel)
        float* I = static_cast<float*>([c.linv contents]) + (size_t)(j / kPanel) * kPanel * kPanel;
        for (uint32_t col = 0; col < jb; ++col)
            for (uint32_t i = 0; i < jb; ++i)
                I[(size_t)col * kPanel + i] = i == col ? 1.0f : i > col ? W[(size_t)(j + col) * ld + j + i] : 0.0f;
        char lo = 'L', unit = 'U';
        __LAPACK_int ljb = (__LAPACK_int)jb, lk = (__LAPACK_int)kPanel, e2 = 0;
        strtri_(&lo, &unit, &ljb, I, &lk, &e2);
    };

    // Panel j's swaps on columns [c0, c1) (and on the columns left of it
    // with `left`), then U12 and the trailing update on [c0, c1).
    auto encode = [&](id<MTLCommandBuffer> cmd, uint32_t j, int s, uint32_t c0, uint32_t c1, bool left) {
        const uint32_t jb = std::min(kPanel, n - j);
        if (count[s]) {
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:c.gather_pipeline()];
            [enc setBuffer:c.w offset:0 atIndex:0];
            [enc setBuffer:c.swaps[s] offset:0 atIndex:1];
            GatherParams gp{ld, count[s], c0, c1};
            if (c1 > c0) {
                [enc setBytes:&gp length:sizeof gp atIndex:2];
                [enc dispatchThreadgroups:MTLSizeMake(c1 - c0, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            }
            if (left && j > 0) {
                gp.c0 = 0;
                gp.c1 = j;
                [enc setBytes:&gp length:sizeof gp atIndex:2];
                [enc dispatchThreadgroups:MTLSizeMake(j, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            }
            [enc endEncoding];
        }
        const uint32_t w = c1 - c0, r = n - j - jb;
        if (w == 0 || r == 0) return;
        // U12 = L11^-1 A12, as U12^T = A12^T L11^-T: into temp, then back
        MPSMatrix* A12 = view(c.w, (size_t)c0 * ld + j, w, jb, ld);
        MPSMatrix* Li = view(c.linv, (size_t)(j / kPanel) * kPanel * kPanel, jb, jb, kPanel);
        MPSMatrix* T = view(c.temp, 0, w, jb, jb);
        [gemm(dev, w, jb, jb, false, 1.0f, 0.0f) encodeToCommandBuffer:cmd leftMatrix:A12 rightMatrix:Li resultMatrix:T];
        MPSMatrixCopyDescriptor* cd = [MPSMatrixCopyDescriptor descriptorWithSourceMatrix:T destinationMatrix:A12
                                                                                 offsets:(MPSMatrixCopyOffsets){0, 0, 0, 0}];
        [copier(dev, w, jb) encodeToCommandBuffer:cmd copyDescriptor:cd];
        // A22 -= L21 U12, as A22^T -= U12^T L21^T
        MPSMatrix* A22 = view(c.w, (size_t)c0 * ld + j + jb, w, r, ld);
        MPSMatrix* L21 = view(c.w, (size_t)j * ld + j + jb, jb, r, ld);
        [gemm(dev, w, r, jb, false, -1.0f, 1.0f) encodeToCommandBuffer:cmd leftMatrix:A12 rightMatrix:L21 resultMatrix:A22];
    };

    // The next panel's columns brought up to date with panel j on the CPU,
    // as LAPACK's right-looking step: its row swaps, U12 by a triangular
    // solve, and its trailing part by one product (Accelerate's, on its
    // matrix units: 0.1 ms at 4096). Done on the GPU it was four launches
    // the CPU waited on, every panel: 20 ms a 4096 x 4096 against 14.
    auto ahead = [&](uint32_t j, uint32_t c0, uint32_t c1) {
        const uint32_t jb = std::min(kPanel, n - j), w = c1 - c0, m = n - j - jb;
        __LAPACK_int lw = (__LAPACK_int)w, lld = (__LAPACK_int)ld, k1 = (__LAPACK_int)j + 1,
                     k2 = (__LAPACK_int)(j + jb), inc = 1;
        slaswp_(&lw, W + (size_t)c0 * ld, &lld, &k1, &k2, f.ipiv.data(), &inc);
        cblas_strsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasUnit, (__LAPACK_int)jb, lw, 1.0f,
                    W + (size_t)j * ld + j, lld, W + (size_t)c0 * ld + j, lld);
        if (m)
            cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, (__LAPACK_int)m, lw, (__LAPACK_int)jb, -1.0f,
                        W + (size_t)j * ld + j + jb, lld, W + (size_t)c0 * ld + j, lld, 1.0f,
                        W + (size_t)c0 * ld + j + jb, lld);
    };

    // Each step: the GPU takes panel j's update of everything right of the
    // next panel (and its swaps left of j), while the CPU waits only for the
    // previous step's update, brings the next panel up to date itself and
    // factors it.
    panel(0, 0);
    id<MTLCommandBuffer> previous = nil;
    for (uint32_t j = 0, s = 0; j < n; j += kPanel, s ^= 1) {
        const uint32_t jb = std::min(kPanel, n - j), next = j + jb;
        if (next >= n) {   // the last panel: its swaps on the columns left of it
            id<MTLCommandBuffer> cmd = [c.rt.queue commandBuffer];
            encode(cmd, j, (int)s, n, n, true);
            [cmd commit];
            if (previous) wait(previous, n);
            previous = cmd;
            break;
        }
        const uint32_t nn = std::min(kPanel, n - next);
        id<MTLCommandBuffer> rest = [c.rt.queue commandBuffer];
        encode(rest, j, (int)s, next + nn, n, true);
        [rest commit];
        if (previous) wait(previous, n);   // the next panel's columns have every earlier panel's update
        ahead(j, next, next + nn);
        panel(next, (int)s ^ 1);
        previous = rest;
    }
    if (previous) wait(previous, n);
}

// U11^-1 of every diagonal block into c.uinv (as c.linv), for the solve;
// false if U is singular (then there is no solution).
bool invert_u(Cache& c, const Factor& f) {
    if (f.info) return false;
    const uint32_t n = f.n, ld = f.ld, nblocks = (n + kPanel - 1) / kPanel;
    c.grow(c.uinv, c.uinv_floats, (size_t)nblocks * kPanel * kPanel, MTLResourceStorageModeShared);
    const float* W = static_cast<const float*>([c.w contents]);
    float* U = static_cast<float*>([c.uinv contents]);
    metal_linalg::detail::parallel_for(nblocks, [&](size_t blk) {
        const uint32_t j = (uint32_t)blk * kPanel, jb = std::min(kPanel, n - j);
        float* I = U + blk * kPanel * kPanel;
        for (uint32_t col = 0; col < jb; ++col)
            for (uint32_t i = 0; i < jb; ++i) I[(size_t)col * kPanel + i] = i <= col ? W[(size_t)(j + col) * ld + j + i] : 0.0f;
        char up = 'U', diag = 'N';
        __LAPACK_int ljb = (__LAPACK_int)jb, lk = (__LAPACK_int)kPanel, e = 0;
        strtri_(&up, &diag, &ljb, I, &lk, &e);
    });
    return true;
}

// X = U^-1 L^-1 Y on the GPU, Y (n x k, row-major) holding P B: X's blocks
// forward through L, then backward through U, every step two MPS products.
void solve_gpu(Cache& c, const Factor& f, uint32_t k, id<MTLBuffer> x, size_t x_off) {
    id<MTLDevice> dev = c.rt.device;
    const uint32_t n = f.n, ld = f.ld;
    c.grow(c.temp, c.temp_floats, (size_t)kPanel * k, MTLResourceStorageModePrivate);
    id<MTLCommandBuffer> cmd = [c.rt.queue commandBuffer];
    for (uint32_t i0 = 0; i0 < n; i0 += kPanel) {   // Z = L^-1 Y, into X
        const uint32_t ib = std::min(kPanel, n - i0), m = n - i0 - ib;
        MPSMatrix* Li = view(c.linv, (size_t)(i0 / kPanel) * kPanel * kPanel, ib, ib, kPanel);   // (L11^-1)^T
        MPSMatrix* Yi = view(c.y, (size_t)i0 * k, ib, k, k);
        MPSMatrix* Xi = view(x, x_off + (size_t)i0 * k, ib, k, k);
        [gemm(dev, ib, k, ib, true, 1.0f, 0.0f) encodeToCommandBuffer:cmd leftMatrix:Li rightMatrix:Yi resultMatrix:Xi];
        if (m) {   // Y below -= L21 Z_i
            MPSMatrix* L21 = view(c.w, (size_t)i0 * ld + i0 + ib, ib, m, ld);   // L21^T
            MPSMatrix* Yb = view(c.y, (size_t)(i0 + ib) * k, m, k, k);
            [gemm(dev, m, k, ib, true, -1.0f, 1.0f) encodeToCommandBuffer:cmd leftMatrix:L21 rightMatrix:Xi resultMatrix:Yb];
        }
    }
    const uint32_t last = (n - 1) / kPanel * kPanel;
    for (int64_t i0 = last; i0 >= 0; i0 -= kPanel) {   // X = U^-1 Z, from the bottom
        const uint32_t ib = std::min<uint32_t>(kPanel, n - (uint32_t)i0);
        MPSMatrix* Ui = view(c.uinv, (size_t)(i0 / kPanel) * kPanel * kPanel, ib, ib, kPanel);   // (U11^-1)^T
        MPSMatrix* Xi = view(x, x_off + (size_t)i0 * k, ib, k, k);
        MPSMatrix* T = view(c.temp, 0, ib, k, k);
        [gemm(dev, ib, k, ib, true, 1.0f, 0.0f) encodeToCommandBuffer:cmd leftMatrix:Ui rightMatrix:Xi resultMatrix:T];
        MPSMatrixCopyDescriptor* cd = [MPSMatrixCopyDescriptor descriptorWithSourceMatrix:T destinationMatrix:Xi
                                                                                 offsets:(MPSMatrixCopyOffsets){0, 0, 0, 0}];
        [copier(dev, ib, k) encodeToCommandBuffer:cmd copyDescriptor:cd];
        if (i0) {   // X above -= U12 X_i
            MPSMatrix* U12 = view(c.w, (size_t)i0 * ld, ib, (uint32_t)i0, ld);   // U12^T
            MPSMatrix* Xa = view(x, x_off, (uint32_t)i0, k, k);
            [gemm(dev, (uint32_t)i0, k, ib, true, -1.0f, 1.0f) encodeToCommandBuffer:cmd leftMatrix:U12 rightMatrix:Xi
                                                                      resultMatrix:Xa];
        }
    }
    [cmd commit];
    wait(cmd, n);
}

// The permutation the row swaps make: row i of P A is row perm[i] of A.
std::vector<uint32_t> permutation(const Factor& f) {
    std::vector<uint32_t> perm(f.n);
    for (uint32_t i = 0; i < f.n; ++i) perm[i] = i;
    for (uint32_t k = 0; k < f.n; ++k) std::swap(perm[k], perm[(uint32_t)f.ipiv[k] - 1]);
    return perm;
}

void require_square(const Matrices& a, const char* what) {
    if (a.rows != a.cols)
        throw std::invalid_argument(std::string("[") + what + "] blocked: the matrices must be square, got " +
                                    std::to_string(a.rows) + "x" + std::to_string(a.cols) + ".");
}

} // namespace

// The caller's output as a buffer where it is page-aligned (an MLX array's,
// known to the library, or a large allocation), else a workspace copied out.
struct Output {
    id<MTLBuffer> buffer;
    bool direct;
};
Output output_buffer(Cache& c, float* out, size_t floats) {
    if (reinterpret_cast<uintptr_t>(out) % (size_t)getpagesize() == 0)
        return {metal_linalg::detail::wrap_host(c.rt.device, out, floats), true};
    return {c.grow(c.out, c.out_floats, floats, MTLResourceStorageModeShared), false};
}

// P (perm) into c.perm, for the permutation kernels.
id<MTLBuffer> upload_perm(Cache& c, const Factor& f) {
    const std::vector<uint32_t> perm = permutation(f);
    c.grow(c.perm, c.perm_count, f.n, MTLResourceStorageModeShared);
    std::copy(perm.begin(), perm.end(), static_cast<uint32_t*>([c.perm contents]));
    return c.perm;
}

void lu_factor_blocked(const Matrices& a, float* lu, uint32_t* pivots, uint32_t* info) {
    require_square(a, "lu_factor");
    const uint32_t n = a.cols, batch = a.batch;
    if (n == 0 || batch == 0) return;
    AutoreleasePool pool;
    Cache& c = Cache::shared();
    Factor f;
    const size_t per = (size_t)n * n;
    id<MTLBuffer> src = input_buffer(c.rt.device, a);
    const Output o = output_buffer(c, lu, per * batch);
    for (uint32_t b = 0; b < batch; ++b) {
        factor(c, src, b * per, n, f);
        id<MTLCommandBuffer> cmd = [c.rt.queue commandBuffer];   // the factors out, row-major
        encode_transpose(c, cmd, c.w, 0, o.buffer, b * per, n, n, f.ld, n);
        [cmd commit];
        wait(cmd, n);
        for (uint32_t k = 0; k < n; ++k) pivots[(size_t)b * n + k] = (uint32_t)(f.ipiv[k] - 1);
        if (info) info[b] = f.info;
    }
    if (!o.direct) copy_out(static_cast<const float*>([o.buffer contents]), lu, batch, per);
}

void solve_blocked(const Matrices& a, const float* bm, uint32_t nrhs, float* x, uint32_t* info) {
    require_square(a, "solve");
    const uint32_t n = a.cols, batch = a.batch;
    if (n == 0 || batch == 0 || nrhs == 0) {
        if (info) std::fill(info, info + batch, 0u);
        return;
    }
    AutoreleasePool pool;
    Cache& c = Cache::shared();
    Factor f;
    const size_t per = (size_t)n * n, bper = (size_t)n * nrhs;
    const bool on_gpu = nrhs >= std::max(1u, lu_policy().gpu_solve_min_rhs);
    id<MTLBuffer> src = input_buffer(c.rt.device, a);
    id<MTLBuffer> rhs = input_buffer(c.rt.device, Matrices{bm, batch, n, nrhs});
    const Output o = output_buffer(c, x, bper * batch);
    float* X = static_cast<float*>([o.buffer contents]);
    std::vector<uint32_t> singular;
    for (uint32_t b = 0; b < batch; ++b) {
        factor(c, src, b * per, n, f);
        if (info) info[b] = f.info;
        if (f.info) {
            singular.push_back(b);
            continue;
        }
        if (on_gpu) {   // Y = P B on the GPU, then the blocked triangular solves into X
            c.grow(c.y, c.y_floats, bper, MTLResourceStorageModePrivate);
            id<MTLBuffer> perm = upload_perm(c, f);
            invert_u(c, f);
            id<MTLCommandBuffer> cmd = [c.rt.queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:c.pipeline(c.permute, @"lu_permute_rows")];
            [enc setBuffer:rhs offset:b * bper * 4 atIndex:0];
            [enc setBuffer:c.y offset:0 atIndex:1];
            [enc setBuffer:perm offset:0 atIndex:2];
            const MoveParams mp{n, nrhs, nrhs, nrhs};
            [enc setBytes:&mp length:sizeof mp atIndex:3];
            [enc dispatchThreads:MTLSizeMake(nrhs, n, 1) threadsPerThreadgroup:MTLSizeMake(std::min(nrhs, 256u), 1, 1)];
            [enc endEncoding];
            [cmd commit];
            solve_gpu(c, f, nrhs, o.buffer, b * bper);
        } else {
            // sgetrs on the factorization where it lies, B column-major
            const uint32_t ld = f.ld;
            const float* in = bm + b * bper;
            float* out = X + b * bper;
            std::vector<float> B((size_t)ld * nrhs);
            if (nrhs == 1) std::memcpy(B.data(), in, n * sizeof(float));
            else transpose(in, nrhs, B.data(), ld, n, nrhs);
            char trans = 'N';
            __LAPACK_int ln = (__LAPACK_int)n, lr = (__LAPACK_int)nrhs, lld = (__LAPACK_int)ld, e = 0;
            sgetrs_(&trans, &ln, &lr, static_cast<float*>([c.w contents]), &lld, f.ipiv.data(), B.data(), &lld, &e);
            if (nrhs == 1) std::memcpy(out, B.data(), n * sizeof(float));
            else transpose(B.data(), ld, out, nrhs, nrhs, n);
        }
    }
    for (uint32_t b : singular) std::fill(X + b * bper, X + (b + 1) * bper, std::numeric_limits<float>::quiet_NaN());
    if (!o.direct) copy_out(X, x, batch, bper);
}

void inv_blocked(const Matrices& a, float* x, uint32_t* info) {
    require_square(a, "inv");
    const uint32_t n = a.cols, batch = a.batch;
    if (n == 0 || batch == 0) return;
    AutoreleasePool pool;
    Cache& c = Cache::shared();
    Factor f;
    const size_t per = (size_t)n * n;
    id<MTLBuffer> src = input_buffer(c.rt.device, a);
    const Output o = output_buffer(c, x, per * batch);
    float* X = static_cast<float*>([o.buffer contents]);
    std::vector<uint32_t> singular;
    for (uint32_t b = 0; b < batch; ++b) {
        factor(c, src, b * per, n, f);
        if (info) info[b] = f.info;
        if (f.info) {
            singular.push_back(b);
            continue;
        }
        c.grow(c.y, c.y_floats, per, MTLResourceStorageModePrivate);
        id<MTLBuffer> perm = upload_perm(c, f);
        invert_u(c, f);
        id<MTLCommandBuffer> cmd = [c.rt.queue commandBuffer];   // Y = P I
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:c.pipeline(c.identity, @"lu_permuted_identity")];
        [enc setBuffer:c.y offset:0 atIndex:0];
        [enc setBuffer:perm offset:0 atIndex:1];
        const MoveParams mp{n, n, n, n};
        [enc setBytes:&mp length:sizeof mp atIndex:2];
        [enc dispatchThreads:MTLSizeMake(n, n, 1) threadsPerThreadgroup:MTLSizeMake(std::min(n, 256u), 1, 1)];
        [enc endEncoding];
        [cmd commit];
        solve_gpu(c, f, n, o.buffer, b * per);
    }
    for (uint32_t b : singular) std::fill(X + b * per, X + (b + 1) * per, std::numeric_limits<float>::quiet_NaN());
    if (!o.direct) copy_out(X, x, batch, per);
}

} // namespace metal_linalg::core::detail
