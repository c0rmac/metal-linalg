// The first stage of the two-stage reductions, on the GPU: a matrix to a band
// of width b by blocks of b columns, the work matrix products. The SVD's
// `band` backend reduces a general matrix to an upper band (svd_bidiag.mm),
// the eigensolver's a symmetric one to a lower band (eigh_band.mm); the band
// then goes to bidiagonal or tridiagonal on the CPU (band_chase.cpp).
//
// A block's panels (b columns, or b rows transposed) are factored by the
// kernels in shaders/Svd_Bidiag.metal: in one simdgroup, rows in registers,
// for up to kLeafRows rows; taller ones by TSQR, leaves of up to kLeafRows
// rows in parallel, their stacked R's in one threadgroup, and the Householder
// vectors rebuilt from TSQR's Q, so that a panel's transformation is always
// the compact H = I - V T V^T. The products are MPS GEMMs on the row-major
// views of the column-major matrices. The GPU takes blocks while enough
// columns remain; LAPACK the last few.
//
// For the SVD with vectors (BandKeep) the general reduction also keeps its
// blocks' reflectors: the panels write V and U a second time into the
// caller's layout (flag 16), T and S into its buffers, and the LAPACK tail's
// reflectors are kept before the band is cleared of them.
//
// Why: the one-stage reductions (tridiag, bidiag) read the trailing matrix
// once or twice a column, and on an M5 Pro they were at about 290 GB/s; these
// read it three (symmetric) or four (general) times a block of b columns.

#ifndef ACCELERATE_NEW_LAPACK
#define ACCELERATE_NEW_LAPACK
#endif
#include <Accelerate/Accelerate.h>

#include "metal_runtime.h"
#include "shaders.h"

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <algorithm>
#include <cstdlib>
#include <map>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace metal_linalg::detail {
namespace {

using L = __LAPACK_int;

// Must match PanelParams and SmallParams in Svd_Bidiag.metal.
struct PanelParams { uint32_t rows, cols, rs, cs, ldv, flags, shift, ldt, leaf, ldw, dup, ldk, sa, sv, svt, st, ss; };
struct SmallParams { uint32_t n, b, ldw, a0, b0, rb, ldc, m, per; };
struct SyParams    { uint32_t n, lda, ldw, b; };

constexpr uint32_t kPartialRows = 256;   // rows a partial of the small b x b products
constexpr uint32_t kApplyPer    = 64;    // rows or columns a threadgroup of bd_*_apply

constexpr uint32_t kBandMax  = 32;          // the panel kernels' widest instance
constexpr uint32_t kLeafRows = 128;         // must match 32 * PANEL_R: a short panel's or a TSQR leaf's rows, at most
constexpr uint32_t kLw       = 3 * kBandMax;   // the [V W/Y ...] buffer's ld
constexpr uint32_t kQrAgg    = 128;         // the blocked QR's aggregate of panels, columns

struct Panels { id<MTLComputePipelineState> panel, leaf, top, rebuild; };
struct Small { id<MTLComputePipelineState> partial, sy, ge, sbupdate; };

// Buffers for one (m, n), only the latest kept: [W^T U] (general) or
// [V Y V] (symmetric), n x kLw; [V_low^T; Y^T], 2b x ldr; V and V T, m x 32;
// X^T, 32 x ldr; the panels' T and S; the small products' partials; the
// TSQR's scratch.
struct Buffers {
    uint32_t      m = 0, n = 0, ldr = 0;
    id<MTLBuffer> bl, br, bv, bvt, bxt, bpart, bt, bs, bsc;
};

struct State {
    MetalRuntime& rt = MetalRuntime::shared(METAL_LINALG_SHADER(Svd_Bidiag), "svd_bidiag");
    Panels panels[3];
    Small  small;
    bool   have = false;
    id<MTLComputePipelineState> scale_copy, merge_t, r_out, agg_apply[2];
    Buffers buf;

    const Panels& kernels(uint32_t b) {
        if (!have) {
            const char* sizes[3] = {"8", "16", "32"};
            for (int i = 0; i < 3; ++i) {
                auto mk = [&](const char* k) {
                    return make_pipeline(rt.device, rt.library, [NSString stringWithFormat:@"%s_%s", k, sizes[i]], nil);
                };
                panels[i] = {mk("bd_panel_qr"), mk("bd_tsqr_leaf"), mk("bd_tsqr_top"), mk("bd_tsqr_rebuild")};
            }
            auto mk1 = [&](NSString* k) { return make_pipeline(rt.device, rt.library, k, nil); };
            small = {mk1(@"bd_small_partial"), mk1(@"bd_sy_apply"), mk1(@"bd_ge_apply"), mk1(@"sb_update")};
            have = true;
        }
        return panels[b <= 8 ? 0 : b <= 16 ? 1 : 2];
    }

    Buffers& buffers(uint32_t m, uint32_t n) {
        if (buf.bl && buf.m == m && buf.n == n) return buf;
        auto priv = [&](size_t f) {
            return [rt.device newBufferWithLength:std::max<size_t>(f, 4) * 4 options:MTLResourceStorageModePrivate];
        };
        Buffers w;
        w.m = m;
        w.n = n;
        // sb_update works in tiles of 64 that may run past the trailing
        // matrix's last row and column, and stages [V Y V]'s rows that far:
        // 64 rows of slack.
        w.ldr = (m + 3) / 4 * 4;
        w.bl  = priv(((size_t)std::max(m, n) + 64) * kLw);
        w.br  = priv((size_t)2 * kBandMax * w.ldr);
        w.bv  = priv((size_t)m * kBandMax);
        w.bvt = priv((size_t)m * kBandMax);
        w.bxt = priv((size_t)kBandMax * w.ldr);
        w.bpart = priv(((size_t)std::max(m, n) / kPartialRows + 1) * 32 * 32);
        w.bt  = priv(kBandMax * kBandMax);
        w.bs  = priv(kBandMax * kBandMax);
        // bd_tsqr_*'s layout: V (m x 32), then per leaf (at least 32 rows each)
        // T, R and E (32 x 32 each), then L1 and U^{-1}, then the top's tree
        // nodes (33 x 32 each, fewer than the leaves).
        w.bsc = priv((size_t)m * 32 + 3 * ((size_t)m / 32 + 2) * 32 * 32 + 2 * 32 * 32 +
                     ((size_t)m / 32 + 2) * 33 * 32);
        return buf = w;
    }

    static State& shared() {
        static State s;
        return s;
    }
};

MPSMatrix* mps(id<MTLBuffer> b, size_t off, uint32_t rows, uint32_t cols, uint32_t ld) {
    MPSMatrixDescriptor* d = [MPSMatrixDescriptor matrixDescriptorWithRows:rows columns:cols
                                                                  rowBytes:(size_t)ld * 4 dataType:MPSDataTypeFloat32];
    return [[MPSMatrix alloc] initWithBuffer:b offset:off * 4 descriptor:d];
}

// `count` matrices, `stride` floats apart. MPS's batched products step from
// one left or result matrix to the next by rows x rowBytes, whatever
// matrixBytes says (macOS 27), so the descriptor is a whole stride tall
// (stride a multiple of ld): the product's own sizes say what it reads. The
// buffer needs a stride of slack past the batch's last matrix.
MPSMatrix* mps(id<MTLBuffer> b, size_t off, uint32_t rows, uint32_t cols, uint32_t ld, uint32_t count,
               size_t stride) {
    if (count <= 1) return mps(b, off, rows, cols, ld);
    if (stride % ld != 0 || stride / ld < rows) throw std::logic_error("[qr] a batched view's stride");
    rows = (uint32_t)(stride / ld);
    MPSMatrixDescriptor* d = [MPSMatrixDescriptor matrixDescriptorWithRows:rows columns:cols matrices:count
                                                                  rowBytes:(size_t)ld * 4 matrixBytes:stride * 4
                                                                  dataType:MPSDataTypeFloat32];
    return [[MPSMatrix alloc] initWithBuffer:b offset:off * 4 descriptor:d];
}

// The products' kernels by shape, reused: making one costs more than
// encoding it, and a reduction encodes hundreds to thousands, of shapes that
// recur from call to call. A thread's own (MPS kernels are not thread-safe),
// cleared when it grows large.
void gemm(id<MTLDevice> dev, id<MTLCommandBuffer> cb, MPSMatrix* A, bool ta, MPSMatrix* B, bool tb, MPSMatrix* C,
          uint32_t m, uint32_t n, uint32_t k, double alpha, double beta, uint32_t batch = 1) {
    using Key = std::tuple<uint32_t, uint32_t, uint32_t, int, double, double>;
    thread_local std::map<Key, MPSMatrixMultiplication*> kernels;
    const Key key{m, n, k, (ta ? 1 : 0) | (tb ? 2 : 0), alpha, beta};
    auto it = kernels.find(key);
    if (it == kernels.end()) {
        if (kernels.size() >= 8192) kernels.clear();
        MPSMatrixMultiplication* g = [[MPSMatrixMultiplication alloc] initWithDevice:dev transposeLeft:ta
            transposeRight:tb resultRows:m resultColumns:n interiorColumns:k alpha:alpha beta:beta];
        it = kernels.insert_or_assign(key, g).first;
    }
    it->second.batchStart = 0;
    it->second.batchSize = batch;
    [it->second encodeToCommandBuffer:cb leftMatrix:A rightMatrix:B resultMatrix:C];
}

// The small products' first kernel: partials of A^T B over the n rows of
// the kLw-wide buffer W, A and B its b columns from a0 and b0.
void small_partials(const Small& sk, const Buffers& w, id<MTLComputeCommandEncoder> enc, uint32_t n, uint32_t b,
                    uint32_t a0, uint32_t b0) {
    const SmallParams q{n, b, kLw, a0, b0, kPartialRows, 0, 0, 0};
    [enc setComputePipelineState:sk.partial];
    [enc setBuffer:w.bl offset:0 atIndex:0];
    [enc setBuffer:w.bpart offset:0 atIndex:1];
    [enc setBytes:&q length:sizeof q atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake((n + kPartialRows - 1) / kPartialRows, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

// The QR of panel P (at A's offset `off`): R in place; H = I - V T V^T as V
// into vout (at voff), T into tout (at toff), and V T, V^T, a second V as
// pp.flags ask; with vk, V into it too (at vkoff, ld ldk).
// In one simdgroup if it has at most kLeafRows rows, else by TSQR: leaves of
// at most kLeafRows rows (a simdgroup each), their stacked R's in one
// threadgroup (a tree of pairs, a simdgroup a pair), then V rebuilt (a thread
// a row).
void panel(const Panels& pk, const Buffers& w, id<MTLCommandBuffer> cb, id<MTLBuffer> A, size_t off, PanelParams pp,
           id<MTLBuffer> vout, size_t voff, id<MTLBuffer> tout, size_t toff = 0, id<MTLBuffer> vk = nil,
           size_t vkoff = 0, uint32_t ldk = 0, id<MTLBuffer> vt = nil, size_t vtoff = 0, uint32_t batch = 1,
           id<MTLBuffer> scratch = nil) {
    const uint32_t leaves = (pp.rows + kLeafRows - 1) / kLeafRows;
    pp.leaf = (pp.rows + leaves - 1) / leaves;
    pp.ldw = kLw;
    pp.ldk = ldk;
    if (vk) pp.flags |= 16u;
    else vk = w.bv;   // bound, unused
    if (!vt) vt = w.bvt;   // V T's, ld 32
    if (!scratch) scratch = w.bsc;   // the TSQR's
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    if (leaves == 1) {
        [enc setComputePipelineState:pk.panel];
        [enc setBuffer:A offset:off * 4 atIndex:0];
        [enc setBuffer:vout offset:voff * 4 atIndex:1];
        [enc setBuffer:vt offset:vtoff * 4 atIndex:2];
        [enc setBuffer:w.br offset:0 atIndex:3];
        [enc setBuffer:tout offset:toff * 4 atIndex:4];
        [enc setBytes:&pp length:sizeof pp atIndex:5];
        [enc setBuffer:w.bl offset:0 atIndex:6];
        [enc setBuffer:w.bv offset:0 atIndex:7];
        [enc setBuffer:vk offset:vkoff * 4 atIndex:8];
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, batch) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    } else {
        [enc setComputePipelineState:pk.leaf];
        [enc setBuffer:A offset:off * 4 atIndex:0];
        [enc setBuffer:scratch offset:0 atIndex:1];
        [enc setBytes:&pp length:sizeof pp atIndex:2];
        [enc setBuffer:w.bl offset:0 atIndex:3];
        [enc setBuffer:w.bv offset:0 atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(leaves, 1, batch) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        // The top: a simdgroup for each pair of leaves' R's, as many as the
        // pipeline takes
        const uint32_t sgs = std::min<uint32_t>(std::max(1u, leaves / 2),
                                                (uint32_t)pk.top.maxTotalThreadsPerThreadgroup / 32);
        [enc setComputePipelineState:pk.top];
        [enc setBuffer:A offset:off * 4 atIndex:0];
        [enc setBuffer:scratch offset:0 atIndex:1];
        [enc setBuffer:tout offset:toff * 4 atIndex:2];
        [enc setBytes:&pp length:sizeof pp atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, batch) threadsPerThreadgroup:MTLSizeMake(32 * sgs, 1, 1)];
        [enc setComputePipelineState:pk.rebuild];
        [enc setBuffer:scratch offset:0 atIndex:0];
        [enc setBuffer:vout offset:voff * 4 atIndex:1];
        [enc setBuffer:vt offset:vtoff * 4 atIndex:2];
        [enc setBuffer:w.br offset:0 atIndex:3];
        [enc setBuffer:tout offset:toff * 4 atIndex:4];
        [enc setBytes:&pp length:sizeof pp atIndex:5];
        [enc setBuffer:vk offset:vkoff * 4 atIndex:6];
        [enc dispatchThreadgroups:MTLSizeMake(leaves, 1, batch)
            threadsPerThreadgroup:MTLSizeMake((pp.leaf + 31) / 32 * 32, 1, 1)];
    }
    [enc endEncoding];
}

void finish(id<MTLCommandBuffer> last) {
    if (!last) return;
    [last waitUntilCompleted];
    if (last.error)
        throw std::runtime_error(std::string("[band] GPU error: ") + last.error.localizedDescription.UTF8String);
}

// The general reduction's last columns, and a matrix too narrow for the GPU's
// blocks: the same block steps with LAPACK on A's shared storage. Each step:
// QR of the column panel, Q^T applied to the columns right of it, LQ of the
// row panel, applied to the rows below it. Where a block is narrower than b,
// the row panel's reflectors would lie inside the band, so they are cleared
// (with `keep`, copied there first).
void general_tail(float* A, uint32_t m, uint32_t n, uint32_t lda, uint32_t b, uint32_t k, BandKeep* keep) {
    std::vector<float> tau(kBandMax), work((size_t)std::max(m, n) * 64 + 64);
    L lw = (L)work.size(), info = 0, LDA = lda;
    if (keep) {
        keep->tail = k;
        keep->steps.clear();
    }
    for (; k < n;) {
        const uint32_t bk = std::min(b, n - k), nr = n - k - bk;
        L mr = m - k, BK = bk, NR = nr;
        float* Akk = A + (size_t)k * lda + k;
        sgeqrf_(&mr, &BK, Akk, &LDA, tau.data(), work.data(), &lw, &info);
        if (keep) keep->steps.push_back({k, bk, nr, std::vector<float>(tau.begin(), tau.begin() + bk), {}, {}});
        if (nr == 0) break;
        sormqr_("L", "T", &mr, &NR, &BK, Akk, &LDA, tau.data(), Akk + (size_t)bk * lda, &LDA, work.data(), &lw, &info);
        float* Pr = Akk + (size_t)bk * lda;
        L kr = std::min(bk, nr), mb = mr - bk;
        sgelqf_(&BK, &NR, Pr, &LDA, tau.data(), work.data(), &lw, &info);
        if (mb > 0)
            sormlq_("R", "T", &mb, &NR, &kr, Pr, &LDA, tau.data(), Pr + bk, &LDA, work.data(), &lw, &info);
        if (keep) {
            BandKeep::Step& st = keep->steps.back();
            st.tp.assign(tau.begin(), tau.begin() + kr);
            st.lq.resize((size_t)bk * nr);
            for (uint32_t c = 0; c < nr; ++c)
                for (uint32_t r = 0; r < bk; ++r) st.lq[r + (size_t)c * bk] = Pr[r + (size_t)c * lda];
        }
        for (uint32_t r = 0; r < bk; ++r)
            for (uint32_t c = r + 1; c < nr; ++c) Pr[r + (size_t)c * lda] = 0.0f;
        k += bk;
    }
}

} // namespace

void band_general_tail(float* A, uint32_t m, uint32_t n, uint32_t lda, uint32_t b, uint32_t k) {
    general_tail(A, m, n, lda, b, k, nullptr);
}

void band_general_tail(float* A, uint32_t m, uint32_t n, uint32_t lda, uint32_t b, uint32_t k, BandKeep* keep) {
    general_tail(A, m, n, lda, b, k, keep);
}

// 16: 32 took 1.4-1.5x its time at 512-2048 (its panels), the same at 4096.
// The TSQR's top is a tree of up to 2^15 leaves of kLeafRows rows.
uint32_t qr_block_width(uint32_t m) {
    return m <= ((size_t)kLeafRows << 15) ? 16u : 0u;
}

size_t qr_scratch_floats(uint32_t m) {
    return (size_t)m * 32 + 3 * ((size_t)m / 32 + 2) * 32 * 32 + 2 * 32 * 32 + ((size_t)m / 32 + 2) * 33 * 32;
}

uint32_t qr_blocks_count(uint32_t m, uint32_t n, uint32_t b) {
    const uint32_t K = std::min(m, n);
    uint32_t j = 0;
    while ((j + 1) * b <= K && m - j * b >= 2 * b) ++j;
    return j;
}

// An aggregate's T from its panels' (bd_merge_t), into st.ta's slot a, a
// threadgroup a matrix.
void merge_t(State& s, id<MTLCommandBuffer> cb, const QrStore& st, uint32_t j0, uint32_t np, uint32_t b,
             uint32_t a) {
    if (!s.merge_t) s.merge_t = make_pipeline(s.rt.device, s.rt.library, @"bd_merge_t", nil);
    const uint32_t p[5] = {np, b, (uint32_t)st.sg, (uint32_t)st.st, (uint32_t)st.sta};
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:s.merge_t];
    [enc setBuffer:st.g offset:0 atIndex:0];
    [enc setBuffer:st.t offset:(size_t)j0 * 1024 * 4 atIndex:1];
    [enc setBuffer:st.ta offset:(size_t)a * kQrAgg * kQrAgg * 4 atIndex:2];
    [enc setBytes:p length:sizeof p atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake(1, 1, st.batch) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
    [enc endEncoding];
}

uint32_t qr_blocks(id<MTLCommandBuffer> __strong& cb, id<MTLBuffer> A, uint32_t m, uint32_t n, uint32_t lda,
                   uint32_t b, const QrStore& st, std::vector<id<MTLCommandBuffer>>& committed) {
    State& s = State::shared();
    const Panels& pk = s.kernels(b);
    Buffers& w = s.buffers(m, n);
    id<MTLDevice> dev = s.rt.device;
    const uint32_t blocks = qr_blocks_count(m, n, b), per = kQrAgg / b, B = st.batch;
    // A's, V's and the scratch's views, the batch's matrices their strides apart
    auto Av = [&](size_t off, uint32_t r, uint32_t c) { return mps(A, off, r, c, lda, B, st.sa); };
    auto Vv = [&](size_t off, uint32_t r, uint32_t c) { return mps(st.v, off, r, c, st.ldv, B, st.sv); };
    auto Zv = [&](id<MTLBuffer> z, uint32_t r, uint32_t c) { return mps(z, 0, r, c, st.ldw, B, st.sz); };
    // Y = [V_j0 ...] an aggregate's: V's zeros above each panel's rows
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    [blit fillBuffer:st.v range:NSMakeRange(0, st.v.length) value:0];
    [blit endEncoding];
    for (uint32_t j0 = 0, a = 0; j0 < blocks; j0 += per, ++a) {
        const uint32_t j1 = std::min(blocks, j0 + per), k0 = j0 * b, wa = (j1 - j0) * b, ma = m - k0;
        for (uint32_t j = j0; j < j1; ++j) {
            // A command buffer after the first panel and then every
            // aggregate, so that the GPU starts while the CPU encodes the
            // rest (at 4096, 1500 kernels and products: several ms)
            if (j == 1 || (j == j0 && a > 0)) {
                [cb commit];
                committed.push_back(cb);
                cb = [s.rt.queue commandBuffer];
            }
            const uint32_t k = j * b, m1 = m - k, nr = k0 + wa - k - b;
            const size_t akk = (size_t)k * lda + k;
            // R in place, rows read contiguously (rs = lda); V and T kept, V T
            // for the aggregate's columns right of the panel: C <- H^T C as
            // W = (V T)^T C, C -= V W
            PanelParams pp{m1, b, lda, 1, st.ldv, 1u, 0, 0, 0, 0, 0};
            pp.sa = (uint32_t)st.sa;
            pp.sv = (uint32_t)st.sv;
            pp.svt = (uint32_t)st.svt;
            pp.st = (uint32_t)st.st;
            pp.ss = (uint32_t)st.ssc;
            panel(pk, w, cb, A, akk, pp, st.v, (size_t)k * st.ldv + k, st.t, (size_t)j * 1024, nil, 0, 0, st.vt, 0,
                  B, st.sc);
            // C <- H^T C inside the aggregate: for up to 4 matrices and panels
            // of up to 3072 rows, qr_agg_apply, one dispatch (on an M5 Pro
            // 1.16x the call at 1024 x 1024, 1.06-1.12x at 512-3072, 1.09x
            // for 4 of 1024^2); else two MPS products, which spread taller
            // panels and larger batches better (the kernel 0.93-0.97x at
            // 4096-8192 rows, level at 8 matrices). QR_AGG_KERNEL=0: always MPS.
            static const bool no_kernel = [] {
                const char* e = std::getenv("QR_AGG_KERNEL");
                return e && std::string(e) == "0";
            }();
            const bool own = (b == 8 || b == 16) && B <= 4 && m1 <= 3072 && !no_kernel;
            if (nr > 0 && !own) {
                gemm(dev, cb, mps(st.vt, 0, m1, b, 32, B, st.svt), true, Av(akk + b, m1, nr), false,
                     Zv(st.z, b, nr), b, nr, m1, 1, 0, B);
                gemm(dev, cb, Vv((size_t)k * st.ldv + k, m1, b), false, Zv(st.z, b, nr), false, Av(akk + b, m1, nr),
                     m1, nr, b, -1, 1, B);
            } else if (nr > 0) {
                const int ai = b == 8 ? 0 : 1;
                if (!s.agg_apply[ai])
                    s.agg_apply[ai] = make_pipeline(s.rt.device, s.rt.library, b == 8 ? @"qr_agg_apply_8" : @"qr_agg_apply_16", nil);
                id<MTLComputePipelineState> ps = s.agg_apply[ai];
                const uint32_t ap[7] = {m1, nr, lda, st.ldv, (uint32_t)st.sa, (uint32_t)st.sv, (uint32_t)st.svt};
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:ps];
                [enc setBuffer:A offset:(akk + b) * 4 atIndex:0];
                [enc setBuffer:st.v offset:((size_t)k * st.ldv + k) * 4 atIndex:1];
                [enc setBuffer:st.vt offset:0 atIndex:2];
                [enc setBytes:ap length:sizeof ap atIndex:3];
                // threads: 8 a row group, up to 128 row groups (fewer for short panels)
                const uint32_t groups = std::clamp<uint32_t>((m1 + 7) / 8, 4u, 128u);
                [enc dispatchThreadgroups:MTLSizeMake((nr + 7) / 8, 1, B)
                    threadsPerThreadgroup:MTLSizeMake((groups * 8 + 31) / 32 * 32, 1, 1)];
                [enc endEncoding];
            }
        }
        // The aggregate's T, from G = Y^T Y
        MPSMatrix* Y = Vv((size_t)k0 * st.ldv + k0, ma, wa);
        gemm(dev, cb, Y, true, Y, false, mps(st.g, 0, wa, wa, kQrAgg, B, st.sg), wa, wa, ma, 1, 0, B);
        merge_t(s, cb, st, j0, j1 - j0, b, a);
        // The columns right of the aggregate: Z = Y^T C, W = Ta^T Z, C -= Y W
        const uint32_t n1 = n - k0 - wa;
        if (n1 > 0) {
            const size_t ac = (size_t)k0 * lda + k0 + wa;
            gemm(dev, cb, Y, true, Av(ac, ma, n1), false, Zv(st.z, wa, n1), wa, n1, ma, 1, 0, B);
            gemm(dev, cb, mps(st.ta, (size_t)a * kQrAgg * kQrAgg, wa, wa, kQrAgg, B, st.sta), true, Zv(st.z, wa, n1),
                 false, Zv(st.z2, wa, n1), wa, n1, wa, 1, 0, B);
            gemm(dev, cb, Y, false, Zv(st.z2, wa, n1), false, Av(ac, ma, n1), ma, n1, wa, -1, 1, B);
        }
    }
    return blocks * b;
}

void qr_blocks_apply(id<MTLCommandBuffer> cb, id<MTLBuffer> Q, uint32_t m, uint32_t K, uint32_t ldq, uint32_t b,
                     uint32_t blocks, const QrStore& st) {
    id<MTLDevice> dev = State::shared().rt.device;
    const uint32_t per = kQrAgg / b, aggs = (blocks + per - 1) / per, B = st.batch;
    const size_t sq = (size_t)m * ldq;   // Q's matrices, m x ldq each
    for (uint32_t a = aggs; a-- > 0;) {
        // Q(k0:, k0:) <- (I - Y Ta Y^T) Q(k0:, k0:): X = Y^T Q, X <- Ta X, Q -= Y X
        const uint32_t j0 = a * per, j1 = std::min(blocks, j0 + per), k0 = j0 * b, wa = (j1 - j0) * b,
                       ma = m - k0, nc = K - k0;
        const size_t qkk = (size_t)k0 * ldq + k0;
        MPSMatrix* Y = mps(st.v, (size_t)k0 * st.ldv + k0, ma, wa, st.ldv, B, st.sv);
        MPSMatrix* Qv = mps(Q, qkk, ma, nc, ldq, B, sq);
        gemm(dev, cb, Y, true, Qv, false, mps(st.z, 0, wa, nc, st.ldw, B, st.sz), wa, nc, ma, 1, 0, B);
        gemm(dev, cb, mps(st.ta, (size_t)a * kQrAgg * kQrAgg, wa, wa, kQrAgg, B, st.sta), false,
             mps(st.z, 0, wa, nc, st.ldw, B, st.sz), false, mps(st.z2, 0, wa, nc, st.ldw, B, st.sz), wa, nc, wa, 1, 0,
             B);
        gemm(dev, cb, Y, false, mps(st.z2, 0, wa, nc, st.ldw, B, st.sz), false, Qv, ma, nc, wa, -1, 1, B);
    }
}

void qr_scale_copy(id<MTLCommandBuffer> cb, id<MTLBuffer> src, id<MTLBuffer> dst, uint32_t m, uint32_t n,
                   uint32_t mp, uint32_t np, size_t sd, uint32_t batch, id<MTLBuffer> scale) {
    State& s = State::shared();
    if (!s.scale_copy) s.scale_copy = make_pipeline(s.rt.device, s.rt.library, @"bd_scale_copy", nil);
    const uint32_t p[5] = {m, n, mp, np, (uint32_t)sd};
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:s.scale_copy];
    [enc setBuffer:src offset:0 atIndex:0];
    [enc setBuffer:dst offset:0 atIndex:1];
    [enc setBytes:p length:sizeof p atIndex:2];
    [enc setBuffer:scale offset:0 atIndex:3];
    [enc dispatchThreads:MTLSizeMake(np, mp, batch) threadsPerThreadgroup:MTLSizeMake(32, 8, 1)];
    [enc endEncoding];
}

void qr_r_out(id<MTLCommandBuffer> cb, id<MTLBuffer> A, id<MTLBuffer> R, uint32_t k, uint32_t n, uint32_t lda,
              size_t sa, uint32_t batch, id<MTLBuffer> up, size_t sr) {
    State& s = State::shared();
    if (!s.r_out) s.r_out = make_pipeline(s.rt.device, s.rt.library, @"bd_qr_r", nil);
    const uint32_t p[5] = {k, n, lda, (uint32_t)sa, sr ? (uint32_t)sr : k * n};
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:s.r_out];
    [enc setBuffer:A offset:0 atIndex:0];
    [enc setBuffer:R offset:0 atIndex:1];
    [enc setBytes:p length:sizeof p atIndex:2];
    [enc setBuffer:up offset:0 atIndex:3];
    [enc dispatchThreads:MTLSizeMake(n, k, batch) threadsPerThreadgroup:MTLSizeMake(32, 8, 1)];
    [enc endEncoding];
}

id<MTLCommandQueue> qr_queue() { return State::shared().rt.queue; }
id<MTLDevice> qr_device() { return State::shared().rt.device; }

uint32_t band_blocks(uint32_t n, uint32_t b) { return n >= 2 * b ? (n - 2 * b) / b + 1 : 0; }

uint32_t band_fit(uint32_t rows, uint32_t b) {
    for (; b >= 8; b /= 2)
        if ((size_t)rows * b <= (size_t)kLeafRows * 1024) return b;
    return 0;
}

uint32_t band_width(uint32_t want, const char* env) {
    if (want == 0 && env)
        if (const char* s = std::getenv(env)) want = (uint32_t)std::max(0L, std::strtol(s, nullptr, 10));
    if (want == 0) want = 16;
    return want <= 8 ? 8u : want <= 16 ? 16u : kBandMax;
}

bool band_reduce_general(id<MTLBuffer> Abuf, uint32_t m, uint32_t n, uint32_t lda, uint32_t b, BandKeep* keep,
                         BandWatch* watch) {
    // A TSQR's stacked R's are one thread a row in one threadgroup: at most
    // 1024 / b leaves.
    if ((size_t)m * b > (size_t)kLeafRows * 1024) return false;
    State& st = State::shared();
    const Panels& pk = st.kernels(b);
    Buffers& w = st.buffers(m, n);
    id<MTLDevice> dev = st.rt.device;
    const uint32_t ldr = w.ldr;
    if (keep) {
        if (keep->qoff.size() < band_blocks(n, b) || keep->poff.size() < band_blocks(n, b))
            throw std::logic_error("[band] BandKeep's layout is short of blocks");
        watch = keep;
    }
    if (watch) watch->done.assign(band_blocks(n, b), nil);
    // Per block, with C the columns right of the column panel and C_low its
    // rows below the row panel: the column panel's QR (H = I - V T V^T),
    // W = T^T V^T C, the row panel C(0:b, :) - V(0:b, :) W, its LQ
    // (G = I - U S U^T), X = C_low U, and then
    // C_low -= V_low W + (X - V_low W U) S U^T as one product: C read three
    // times and written once a block. (sb_update's tiles over all of C, which
    // has no symmetry to save on, did no better than MPS's product.)
    id<MTLCommandBuffer> last = nil;
    uint32_t k = 0;
    for (; k + 2 * b <= n; k += b) {
        const uint32_t m1 = m - k, n1 = n - k - b, m2 = m1 - b;
        const size_t akk = (size_t)k * lda + k, akb = (size_t)(k + b) * lda + k;
        id<MTLCommandBuffer> cb = [st.rt.queue commandBufferWithUnretainedReferences];
        const uint32_t bi = k / b;
        // The column panel: R in place; V, V T, and V_low^T into [V_low^T; Y^T].
        if (keep)
            panel(pk, w, cb, Abuf, akk, PanelParams{m1, b, 1, lda, kBandMax, 3u, b, ldr, 0, 0, 0}, w.bv, 0, keep->qt,
                  (size_t)bi * 1024, keep->qv, keep->qoff[bi], keep->qld[bi]);
        else
            panel(pk, w, cb, Abuf, akk, PanelParams{m1, b, 1, lda, kBandMax, 3u, b, ldr, 0, 0, 0}, w.bv, 0, w.bt);
        // W^T = C^T (V T), into [W^T U]
        gemm(dev, cb, mps(Abuf, akb, n1, m1, lda), false, mps(w.bvt, 0, m1, b, kBandMax), false,
             mps(w.bl, 0, n1, b, kLw), n1, b, m1, 1, 0);
        // The row panel, transposed, C(0:b, :)^T - W^T V(0:b, :)^T as it is
        // loaded (flag 4), and its QR, the row panel's LQ: L in place; U into
        // [W^T U], S.
        id<MTLBuffer> sbuf = keep ? keep->pt : w.bs;
        const size_t soff = keep ? (size_t)bi * 1024 : 0;
        panel(pk, w, cb, Abuf, akb, PanelParams{n1, b, lda, 1, kLw, 4u, 0, 0, 0, 0, 0}, w.bl, b, sbuf, soff,
              keep ? keep->pv : nil, keep ? keep->poff[bi] : 0, keep ? keep->pld[bi] : 0);
        // X^T = U^T C_low^T
        gemm(dev, cb, mps(w.bl, b, n1, b, kLw), true, mps(Abuf, akb + b, n1, m2, lda), false,
             mps(w.bxt, 0, b, m2, ldr), b, m2, n1, 1, 0);
        // W U, then Y^T = S^T (X^T - (W U)^T V_low^T) into [V_low^T; Y^T]
        {
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            small_partials(st.small, w, enc, n1, b, 0, b);
            const SmallParams q{n1, b, kLw, 0, b, kPartialRows, ldr, m2, kApplyPer};
            [enc setComputePipelineState:st.small.ge];
            [enc setBuffer:w.bxt offset:0 atIndex:0];
            [enc setBuffer:w.br offset:0 atIndex:1];
            [enc setBuffer:w.bpart offset:0 atIndex:2];
            [enc setBuffer:sbuf offset:soff * 4 atIndex:3];
            [enc setBytes:&q length:sizeof q atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake((m2 + kApplyPer - 1) / kApplyPer, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc endEncoding];
        }
        // C_low^T -= [W^T U] [V_low^T; Y^T]
        gemm(dev, cb, mps(w.bl, 0, n1, 2 * b, kLw), false, mps(w.br, 0, 2 * b, m2, ldr), false,
             mps(Abuf, akb + b, n1, m2, lda), n1, m2, 2 * b, -1, 1);
        [cb commit];
        if (watch) watch->done[bi] = cb;
        last = cb;
    }
    if (watch && watch->while_gpu) {
        try {
            watch->while_gpu(*watch);
        } catch (...) {
            if (last) [last waitUntilCompleted];
            throw;
        }
    }
    finish(last);
    general_tail(static_cast<float*>(Abuf.contents), m, n, lda, b, k, keep);
    return true;
}

uint32_t band_blocks_symmetric(uint32_t n, uint32_t b) { return n >= 3 * b ? (n - 3 * b) / b + 1 : 0; }

bool band_reduce_symmetric(id<MTLBuffer> Abuf, uint32_t n, uint32_t lda, uint32_t b, BandWatch* watch,
                           BandKeep* keep) {
    if ((size_t)n * b > (size_t)kLeafRows * 1024) return false;
    if (keep) {
        if (keep->qoff.size() < band_blocks_symmetric(n, b)) throw std::logic_error("[band] BandKeep's layout");
        if (!watch) watch = keep;
    }
    if (watch) watch->done.assign(band_blocks_symmetric(n, b), nil);
    State& st = State::shared();
    const Panels& pk = st.kernels(b);
    Buffers& w = st.buffers(n, n);
    id<MTLDevice> dev = st.rt.device;
    // Per block: the panel below the diagonal block, A(k+b:, k:k+b), its QR
    // (H = I - V T V^T, R in place: the band); then both sides of the trailing
    // matrix A22 = A(k+b:, k+b:), kept in full: X = A22 V T (MPS),
    // Y = X - V (T^T V^T X) / 2 (bd_small_partial, bd_sy_apply),
    // A22 -= V Y^T + Y V^T with [V Y V] (its first two and last two b
    // columns) on the lower triangle, mirrored over the upper (sb_update).
    // A22 read twice and written once and a half a block.
    id<MTLCommandBuffer> last = nil;
    uint32_t k = 0;
    for (; k + 3 * b <= n; k += b) {
        const uint32_t n1 = n - k - b, nb = (n1 + 63) / 64;
        const size_t akp = (size_t)k * lda + k + b, a22 = (size_t)(k + b) * lda + k + b;
        id<MTLCommandBuffer> cb = [st.rt.queue commandBufferWithUnretainedReferences];
        // The panel: R in place; V into [V Y V] twice (flag 8), V T; with
        // keep, V and T kept there too.
        const uint32_t bi = k / b;
        id<MTLBuffer> tbuf = keep ? keep->qt : w.bt;
        const size_t toff = keep ? (size_t)bi * 1024 : 0;
        if (keep)
            panel(pk, w, cb, Abuf, akp, PanelParams{n1, b, 1, lda, kLw, 1u | 8u, 0, 0, 0, 0, 2 * b}, w.bl, 0, tbuf,
                  toff, keep->qv, keep->qoff[bi], keep->qld[bi]);
        else
            panel(pk, w, cb, Abuf, akp, PanelParams{n1, b, 1, lda, kLw, 1u | 8u, 0, 0, 0, 0, 2 * b}, w.bl, 0, w.bt);
        // X = A22 (V T), into Y's place
        gemm(dev, cb, mps(Abuf, a22, n1, n1, lda), false, mps(w.bvt, 0, n1, b, kBandMax), false,
             mps(w.bl, b, n1, b, kLw), n1, b, n1, 1, 0);
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        // Z = V^T X, M = T^T Z / 2, Y = X - V M
        small_partials(st.small, w, enc, n1, b, 0, b);
        const SmallParams q{n1, b, kLw, 0, b, kPartialRows, 0, 0, kApplyPer};
        [enc setComputePipelineState:st.small.sy];
        [enc setBuffer:w.bl offset:0 atIndex:0];
        [enc setBuffer:w.bpart offset:0 atIndex:1];
        [enc setBuffer:tbuf offset:toff * 4 atIndex:2];
        [enc setBytes:&q length:sizeof q atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake((n1 + kApplyPer - 1) / kApplyPer, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        // A22 -= [V Y] [Y V]^T, its lower tiles and their mirrors
        const SyParams sp{n1, lda, kLw, b};
        [enc setComputePipelineState:st.small.sbupdate];
        [enc setBuffer:Abuf offset:a22 * 4 atIndex:0];
        [enc setBuffer:w.bl offset:0 atIndex:1];
        [enc setBytes:&sp length:sizeof sp atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake((size_t)nb * (nb + 1) / 2, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
        [cb commit];
        if (watch) watch->done[k / b] = cb;
        last = cb;
    }
    if (watch && watch->while_gpu) {
        try {
            watch->while_gpu(*watch);
        } catch (...) {
            if (last) [last waitUntilCompleted];
            throw;
        }
    }
    finish(last);
    // The trailing block A(k:, k:), fewer than 3b columns: LAPACK's
    // ssytrd_sy2sb, its band copied back into A's lower band.
    const uint32_t nt = n - k;
    if (keep) {
        keep->tail = k;
        keep->sy2sb_kd = 0;
        keep->sy2sb_tau.clear();
    }
    if (nt > 1) {
        float* A = static_cast<float*>(Abuf.contents);
        L N = nt, KD = std::min(b, nt - 1), LDA = lda, LDAB = KD + 1, lw = -1, info = 0;
        std::vector<float> ab((size_t)LDAB * nt), tau(nt);
        float q = 0.0f;
        ssytrd_sy2sb_("L", &N, &KD, A + (size_t)k * lda + k, &LDA, ab.data(), &LDAB, tau.data(), &q, &lw, &info);
        std::vector<float> work(std::max<L>(1, (L)q));
        lw = (L)work.size();
        ssytrd_sy2sb_("L", &N, &KD, A + (size_t)k * lda + k, &LDA, ab.data(), &LDAB, tau.data(), work.data(), &lw,
                      &info);
        if (info != 0) throw std::runtime_error("[band] LAPACK ssytrd_sy2sb failed, info " + std::to_string((long long)info));
        if (keep) {
            keep->sy2sb_kd = (uint32_t)KD;
            keep->sy2sb_tau = tau;
        }
        for (uint32_t c = 0; c < nt; ++c)
            for (uint32_t r = c; r < nt && r <= c + (uint32_t)KD; ++r)
                A[(size_t)(k + c) * lda + k + r] = ab[(size_t)c * LDAB + r - c];
    }
    return true;
}

} // namespace metal_linalg::detail
