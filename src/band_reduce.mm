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
#include <utility>
#include <vector>

namespace metal_linalg::detail {
namespace {

using L = __LAPACK_int;

// Must match PanelParams and SmallParams in Svd_Bidiag.metal.
struct PanelParams { uint32_t rows, cols, rs, cs, ldv, flags, shift, ldt, leaf, ldw, dup; };
struct SmallParams { uint32_t n, b, ldw, a0, b0, rb, ldc, m, per; };
struct SyParams    { uint32_t n, lda, ldw, b; };

constexpr uint32_t kPartialRows = 256;   // rows a partial of the small b x b products
constexpr uint32_t kApplyPer    = 64;    // rows or columns a threadgroup of bd_*_apply

constexpr uint32_t kBandMax  = 32;          // the panel kernels' widest instance
constexpr uint32_t kLeafRows = 128;         // must match 32 * PANEL_R: a short panel's or a TSQR leaf's rows, at most
constexpr uint32_t kLw       = 3 * kBandMax;   // the [V W/Y ...] buffer's ld

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

void gemm(id<MTLDevice> dev, id<MTLCommandBuffer> cb, MPSMatrix* A, bool ta, MPSMatrix* B, bool tb, MPSMatrix* C,
          uint32_t m, uint32_t n, uint32_t k, double alpha, double beta) {
    MPSMatrixMultiplication* g = [[MPSMatrixMultiplication alloc] initWithDevice:dev transposeLeft:ta
        transposeRight:tb resultRows:m resultColumns:n interiorColumns:k alpha:alpha beta:beta];
    [g encodeToCommandBuffer:cb leftMatrix:A rightMatrix:B resultMatrix:C];
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
// into vout (at voff), T into tout, and V T, V^T, a second V as pp.flags ask.
// In one simdgroup if it has at most kLeafRows rows, else by TSQR: leaves of
// at most kLeafRows rows (a simdgroup each), their stacked R's in one
// threadgroup (a tree of pairs, a simdgroup a pair), then V rebuilt (a thread
// a row).
void panel(const Panels& pk, const Buffers& w, id<MTLCommandBuffer> cb, id<MTLBuffer> A, size_t off, PanelParams pp,
           id<MTLBuffer> vout, size_t voff, id<MTLBuffer> tout) {
    const uint32_t leaves = (pp.rows + kLeafRows - 1) / kLeafRows;
    pp.leaf = (pp.rows + leaves - 1) / leaves;
    pp.ldw = kLw;
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    if (leaves == 1) {
        [enc setComputePipelineState:pk.panel];
        [enc setBuffer:A offset:off * 4 atIndex:0];
        [enc setBuffer:vout offset:voff * 4 atIndex:1];
        [enc setBuffer:w.bvt offset:0 atIndex:2];
        [enc setBuffer:w.br offset:0 atIndex:3];
        [enc setBuffer:tout offset:0 atIndex:4];
        [enc setBytes:&pp length:sizeof pp atIndex:5];
        [enc setBuffer:w.bl offset:0 atIndex:6];
        [enc setBuffer:w.bv offset:0 atIndex:7];
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    } else {
        [enc setComputePipelineState:pk.leaf];
        [enc setBuffer:A offset:off * 4 atIndex:0];
        [enc setBuffer:w.bsc offset:0 atIndex:1];
        [enc setBytes:&pp length:sizeof pp atIndex:2];
        [enc setBuffer:w.bl offset:0 atIndex:3];
        [enc setBuffer:w.bv offset:0 atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(leaves, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        // The top: a simdgroup for each pair of leaves' R's, as many as the
        // pipeline takes
        const uint32_t sgs = std::min<uint32_t>(std::max(1u, leaves / 2),
                                                (uint32_t)pk.top.maxTotalThreadsPerThreadgroup / 32);
        [enc setComputePipelineState:pk.top];
        [enc setBuffer:A offset:off * 4 atIndex:0];
        [enc setBuffer:w.bsc offset:0 atIndex:1];
        [enc setBuffer:tout offset:0 atIndex:2];
        [enc setBytes:&pp length:sizeof pp atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(32 * sgs, 1, 1)];
        [enc setComputePipelineState:pk.rebuild];
        [enc setBuffer:w.bsc offset:0 atIndex:0];
        [enc setBuffer:vout offset:voff * 4 atIndex:1];
        [enc setBuffer:w.bvt offset:0 atIndex:2];
        [enc setBuffer:w.br offset:0 atIndex:3];
        [enc setBuffer:tout offset:0 atIndex:4];
        [enc setBytes:&pp length:sizeof pp atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake(leaves, 1, 1)
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
// the row panel's reflectors would lie inside the band, so they are cleared.
void general_tail(float* A, uint32_t m, uint32_t n, uint32_t lda, uint32_t b, uint32_t k) {
    std::vector<float> tau(kBandMax), work((size_t)std::max(m, n) * 64 + 64);
    L lw = (L)work.size(), info = 0, LDA = lda;
    for (; k < n;) {
        const uint32_t bk = std::min(b, n - k), nr = n - k - bk;
        L mr = m - k, BK = bk, NR = nr;
        float* Akk = A + (size_t)k * lda + k;
        sgeqrf_(&mr, &BK, Akk, &LDA, tau.data(), work.data(), &lw, &info);
        if (nr == 0) break;
        sormqr_("L", "T", &mr, &NR, &BK, Akk, &LDA, tau.data(), Akk + (size_t)bk * lda, &LDA, work.data(), &lw, &info);
        float* Pr = Akk + (size_t)bk * lda;
        L kr = std::min(bk, nr), mb = mr - bk;
        sgelqf_(&BK, &NR, Pr, &LDA, tau.data(), work.data(), &lw, &info);
        if (mb > 0)
            sormlq_("R", "T", &mb, &NR, &kr, Pr, &LDA, tau.data(), Pr + bk, &LDA, work.data(), &lw, &info);
        for (uint32_t r = 0; r < bk; ++r)
            for (uint32_t c = r + 1; c < nr; ++c) Pr[r + (size_t)c * lda] = 0.0f;
        k += bk;
    }
}

} // namespace

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

bool band_reduce_general(id<MTLBuffer> Abuf, uint32_t m, uint32_t n, uint32_t lda, uint32_t b) {
    // A TSQR's stacked R's are one thread a row in one threadgroup: at most
    // 1024 / b leaves.
    if ((size_t)m * b > (size_t)kLeafRows * 1024) return false;
    State& st = State::shared();
    const Panels& pk = st.kernels(b);
    Buffers& w = st.buffers(m, n);
    id<MTLDevice> dev = st.rt.device;
    const uint32_t ldr = w.ldr;
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
        // The column panel: R in place; V, V T, and V_low^T into [V_low^T; Y^T].
        panel(pk, w, cb, Abuf, akk, PanelParams{m1, b, 1, lda, kBandMax, 3u, b, ldr, 0, 0, 0}, w.bv, 0, w.bt);
        // W^T = C^T (V T), into [W^T U]
        gemm(dev, cb, mps(Abuf, akb, n1, m1, lda), false, mps(w.bvt, 0, m1, b, kBandMax), false,
             mps(w.bl, 0, n1, b, kLw), n1, b, m1, 1, 0);
        // The row panel, transposed, C(0:b, :)^T - W^T V(0:b, :)^T as it is
        // loaded (flag 4), and its QR, the row panel's LQ: L in place; U into
        // [W^T U], S.
        panel(pk, w, cb, Abuf, akb, PanelParams{n1, b, lda, 1, kLw, 4u, 0, 0, 0, 0, 0}, w.bl, b, w.bs);
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
            [enc setBuffer:w.bs offset:0 atIndex:3];
            [enc setBytes:&q length:sizeof q atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake((m2 + kApplyPer - 1) / kApplyPer, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc endEncoding];
        }
        // C_low^T -= [W^T U] [V_low^T; Y^T]
        gemm(dev, cb, mps(w.bl, 0, n1, 2 * b, kLw), false, mps(w.br, 0, 2 * b, m2, ldr), false,
             mps(Abuf, akb + b, n1, m2, lda), n1, m2, 2 * b, -1, 1);
        [cb commit];
        last = cb;
    }
    finish(last);
    general_tail(static_cast<float*>(Abuf.contents), m, n, lda, b, k);
    return true;
}

bool band_reduce_symmetric(id<MTLBuffer> Abuf, uint32_t n, uint32_t lda, uint32_t b) {
    if ((size_t)n * b > (size_t)kLeafRows * 1024) return false;
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
        // The panel: R in place; V into [V Y V] twice (flag 8), V T.
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
        [enc setBuffer:w.bt offset:0 atIndex:2];
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
        last = cb;
    }
    finish(last);
    // The trailing block A(k:, k:), fewer than 3b columns: LAPACK's
    // ssytrd_sy2sb, its band copied back into A's lower band.
    const uint32_t nt = n - k;
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
        for (uint32_t c = 0; c < nt; ++c)
            for (uint32_t r = c; r < nt && r <= c + (uint32_t)KD; ++r)
                A[(size_t)(k + c) * lda + k + r] = ab[(size_t)c * LDAB + r - c];
    }
    return true;
}

} // namespace metal_linalg::detail
