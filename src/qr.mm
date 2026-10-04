#import <Metal/Metal.h>

#include "calibration.h"
#include "metal_runtime.h"
#include <metal_linalg/core.h>
#include <metal_linalg/device.h>

#include <algorithm>
#include <cstdlib>
#include <string>

namespace metal_linalg {
namespace {

// qr_unblocked asks for this much threadgroup memory (see qr_unblocked.mm).
// Together with the device's per-threadgroup limit it sets how many matrices
// the backend can keep resident at once.
constexpr unsigned kUnblockedThreadgroupBytes = 5120;

// -----------------------------------------------------------------------------
// Tuned crossovers
// -----------------------------------------------------------------------------
// One entry per GPU that has actually been measured. Extrapolating between them
// is not safe: the crossover goes as sqrt(L*R/32) * sqrt(C/(C-1)) for square
// inputs, where C is core count, L kernel-launch overhead and R per-core
// throughput. The C term is weak -- 8 to 80 cores moves it about 6%, inside the
// flat optimum -- but R rises across GPU generations and pushes the other way,
// so the net is not predictable without measuring. Hence a table, not a formula.
struct TunedEntry {
    const char* device_name;   // exact MTLDevice.name
    unsigned    gpu_cores;     // guards against same-name parts with different core counts
    unsigned    m_small_batch;
    unsigned    m_large_batch;
    unsigned    batch_threshold;
    // GPU or CPU. A row from before QR had a CPU path has none of these, so
    // they read as zero; gpu_min_batch = 0 is never a measured value, and
    // marks such a row as "always the GPU", which is what was measured.
    unsigned    gpu_max_k;
    unsigned    gpu_min_batch_times_k;
    unsigned    gpu_min_batch;
    unsigned    gpu_min_k;             // 0 = no lower bound, which rows from before it leave
    // Large matrices on the GPU from this k, for batches up to the cap (0:
    // any); 0, 0 = never, which rows from before the clause leave.
    unsigned    gpu_large_min_k;
    unsigned    gpu_large_max_batch;
    unsigned    share_min_batch;   // 0 = never, which rows from before 2.12.0 leave
    unsigned    calibration;   // kCalibration* (calibration.h); rows without it are current
};

// The rows are generated from every run submitted for a device (docs/results/)
// by tuning/generate_tables.py, which a GitHub Action reruns after each
// merge; see docs/tuning.md. Why the measured values are what they are is in
// docs/studies/. The last row keeps the array non-empty and matches nothing.
constexpr TunedEntry kTuned[] = {
#include "tuned/qr.inc"
    {"", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
};

struct ResolvedPolicy {
    QrPolicy    policy;
    std::string source;
};

// A non-negative integer from the environment; zero is a valid value
// (QR_GPU_MAX_K=0 means never the GPU).
bool env_value(const char* name, unsigned& out) {
    const char* s = std::getenv(name);
    if (!s || !*s) return false;
    char* end = nullptr;
    const long long v = std::strtoll(s, &end, 10);
    if (end == s || v < 0) return false;
    out = v > 0xFFFFFFFFll ? kQrNoLimit : (unsigned)v;
    return true;
}

ResolvedPolicy resolve() {
    ResolvedPolicy r;

    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        const std::string name = device_name();

        r.policy.gpu_cores = gpu_core_count();

        // Resident qr_unblocked threadgroups, i.e. matrices in flight. This one
        // *is* derived from the device rather than assumed.
        if (dev && r.policy.gpu_cores) {
            const unsigned per_core =
                (unsigned)(dev.maxThreadgroupMemoryLength / kUnblockedThreadgroupBytes);
            r.policy.concurrent_matrices = per_core * r.policy.gpu_cores;
        }

        for (const auto& e : kTuned) {
            if (e.device_name[0] != '\0' && name == e.device_name &&
                r.policy.gpu_cores == e.gpu_cores) {
                r.policy.m_crossover_small_batch = e.m_small_batch;
                r.policy.m_crossover_large_batch = e.m_large_batch;
                r.policy.batch_threshold         = e.batch_threshold;
                if (e.gpu_min_batch == 0) {          // measured before the CPU path
                    r.policy.gpu_max_k             = kQrNoLimit;
                    r.policy.gpu_min_batch_times_k = 0;
                    r.policy.gpu_min_batch         = 1;
                } else {
                    r.policy.gpu_max_k             = e.gpu_max_k;
                    r.policy.gpu_min_batch_times_k = e.gpu_min_batch_times_k;
                    r.policy.gpu_min_batch         = e.gpu_min_batch;
                    r.policy.gpu_min_k             = e.gpu_min_k;
                }
                r.policy.gpu_large_min_k     = e.gpu_large_min_k;
                r.policy.gpu_large_max_batch = e.gpu_large_max_batch;
                r.policy.share_min_batch     = e.share_min_batch;
                r.source = detail::tuned_source_prefix(e.calibration) + name;
                detail::calibration_notice("QR", e.calibration);
                break;
            }
        }
        if (r.source.empty()) {
            // No measurements for this GPU, so bias to the safe side rather than
            // reusing a tuned entry verbatim.
            //
            // The penalty is asymmetric: on the measured M1, 320 costs 0.6% and
            // 256 costs 2.3%, while 448 costs 0.6% and 512 costs 2.6% -- and the
            // high side degrades on exactly the tall shapes that are easiest to
            // under-sample. Adding GPU cores makes the grid-parallel backend
            // relatively stronger and pushes the true crossover down, so on an
            // unknown -- and probably larger -- device, erring low is safer.
            r.policy.m_crossover_small_batch = 384;
            r.policy.m_crossover_large_batch = 384;
            r.policy.batch_threshold         = 16;
            // GPU or CPU: the QrPolicy defaults.
            r.source = "default:untuned-device" + (name.empty() ? "" : " (" + name + ")");
            detail::calibration_notice("QR", kUncalibrated);
        }
    }

    // Environment overrides, for retuning without a rebuild. QR_M_CROSSOVER
    // collapses both regimes to one flat threshold, which is what someone
    // bisecting a crossover by hand actually wants.
    std::string env;
    unsigned crossover = 0;
    if (env_value("QR_M_CROSSOVER", crossover) && crossover > 0) {
        r.policy.m_crossover_small_batch = crossover;
        r.policy.m_crossover_large_batch = crossover;
        env = "QR_M_CROSSOVER";
    }
    auto over = [&](const char* name, unsigned& field) {
        if (env_value(name, field)) env += (env.empty() ? "" : ",") + std::string(name);
    };
    over("QR_GPU_MAX_K",             r.policy.gpu_max_k);
    over("QR_GPU_MIN_BATCH_TIMES_K", r.policy.gpu_min_batch_times_k);
    over("QR_GPU_MIN_BATCH",         r.policy.gpu_min_batch);
    over("QR_GPU_MIN_K",             r.policy.gpu_min_k);
    over("QR_GPU_LARGE_MIN_K",       r.policy.gpu_large_min_k);
    over("QR_GPU_LARGE_MAX_BATCH",   r.policy.gpu_large_max_batch);
    over("QR_SHARE_MIN_BATCH",       r.policy.share_min_batch);
    if (!env.empty()) r.source = "env:" + env;
    return r;
}

ResolvedPolicy& state() {
    static ResolvedPolicy s = resolve();
    return s;
}

} // namespace

QrPolicy    qr_policy()        { return state().policy; }
const char* qr_policy_source() { return state().source.c_str(); }

void set_qr_policy(const QrPolicy& p) {
    state().policy = p;
    state().source = "user";
}

// The crossover is on M alone, and rows are not interchangeable with
// columns. qr_unblocked gives each matrix a single threadgroup, which must
// sweep M rows for every Householder reflection: M is its serial depth,
// while N parallelises across the threadgroup's threads.
// qr_streaming_amx_reduced spreads each matrix over a grid instead, paying
// roughly three kernel launches per 32-column panel.
//
// So a 2048x64 and a 64x2048 want opposite backends despite sharing both
// max(M, N) and K = min(M, N) -- 9.9x for reduced on the former, 1.5x for
// unblocked on the latter. A rule keyed on max(M, N) cannot express that.
//
// Batch does not enter on any measured device. It appeared to help by 0.7%
// on a square-heavy grid; sampling tall shapes properly reversed that, and a
// plain threshold now wins on held-out data (1.011x vs 1.014x geometric-mean
// regret, and 1.25x vs 1.54x worst case). The machinery is kept because the
// split may be justified on other hardware -- see tuning/tune_qr.py, which
// only emits it when it clears the noise floor.
QrBackend qr_gpu_backend(unsigned m, unsigned n, unsigned batch) {
    (void)n;
    const QrPolicy& p = state().policy;
    const unsigned crossover = batch < p.batch_threshold ? p.m_crossover_small_batch
                                                         : p.m_crossover_large_batch;
    return m >= crossover ? QrBackend::streaming_reduced : QrBackend::unblocked;
}

// GPU or CPU, as for eigh and the SVD: the GPU needs enough work to pay for a
// launch, and a lone or small-batch call is quicker in LAPACK. Large matrices
// in small batches have a clause of their own (see QrPolicy).
bool qr_uses_gpu(unsigned m, unsigned n, unsigned batch) {
    if (const char* e = std::getenv("QR_DEVICE")) {
        const std::string s = e;
        if (s == "gpu") return true;
        if (s == "cpu") return false;
    }
    const QrPolicy& p = state().policy;
    const unsigned k = std::min(m, n);
    if (p.gpu_large_min_k && k >= p.gpu_large_min_k &&
        (p.gpu_large_max_batch == 0 || batch <= p.gpu_large_max_batch)) {
        return true;
    }
    return k >= p.gpu_min_k && k <= p.gpu_max_k &&
           (unsigned long long)batch * k >= p.gpu_min_batch_times_k && batch >= p.gpu_min_batch;
}

QrBackend qr_backend(unsigned m, unsigned n, unsigned batch) {
    return qr_uses_gpu(m, n, batch) ? qr_gpu_backend(m, n, batch) : QrBackend::cpu;
}

bool qr_shares_batch(unsigned m, unsigned n, unsigned batch) {
    const unsigned from = state().policy.share_min_batch;
    return from != 0 && batch >= from && qr_backend(m, n, batch) != QrBackend::cpu;
}

void core::detail::qr_shared(const Matrices& a, float* q, float* r) {
    const uint32_t M = a.rows, N = a.cols, K = std::min(M, N);
    if (K == 0 || a.batch == 0) return;
    const QrBackend gpu = qr_gpu_backend(M, N, a.batch);
    auto sub = [&](uint32_t b0, uint32_t count) { return Matrices{a.data + (size_t)b0 * M * N, count, M, N}; };
    // The smallest GPU chunk worth a dispatch: eight matrices per core. The
    // CPU's chunks are a few matrices per worker (share_batch).
    metal_linalg::detail::share_batch(
        a.batch, 8 * std::max(1u, gpu_core_count()),
        std::clamp(a.batch / (16 * std::max(1u, cpu_threads())), 1u, 16u),
        [&](uint32_t b0, uint32_t count) {
            if (gpu == QrBackend::streaming_reduced)
                qr_streaming_amx_reduced(sub(b0, count), q + (size_t)b0 * M * K, r + (size_t)b0 * K * N);
            else
                qr_unblocked(sub(b0, count), q + (size_t)b0 * M * K, r + (size_t)b0 * K * N);
        },
        [&](uint32_t b0, uint32_t count) {
            qr_cpu(sub(b0, count), q + (size_t)b0 * M * K, r + (size_t)b0 * K * N);
        });
}

void core::qr(const Matrices& a, float* q, float* r) {
    if (qr_shares_batch(a.rows, a.cols, a.batch)) {
        core::detail::qr_shared(a, q, r);
        return;
    }
    switch (qr_backend(a.rows, a.cols, a.batch)) {
        case QrBackend::cpu:               core::detail::qr_cpu(a, q, r); break;
        case QrBackend::streaming_reduced: core::detail::qr_streaming_amx_reduced(a, q, r); break;
        default:                           core::detail::qr_unblocked(a, q, r); break;
    }
}

} // namespace metal_linalg
