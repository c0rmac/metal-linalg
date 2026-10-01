#import <Metal/Metal.h>

#include <metal_linalg/device.h>
#include <metal_linalg/qr.h>

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
};

// The rows are generated from every run submitted for a device (docs/results/)
// by tuning/generate_tables.py, which a GitHub Action reruns after each
// merge; see docs/tuning.md. Why the measured values are what they are is in
// docs/studies/. The last row keeps the array non-empty and matches nothing.
constexpr TunedEntry kTuned[] = {
#include "tuned/qr.inc"
    {"", 0, 0, 0, 0},
};

struct ResolvedPolicy {
    QrPolicy    policy;
    std::string source;
};

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
                r.source = "tuned:" + name;
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
            r.source = "default:untuned-device" + (name.empty() ? "" : " (" + name + ")");
        }
    }

    // Environment override, for retuning without a rebuild.
    // Environment override collapses both regimes to one flat threshold, which
    // is what someone bisecting a crossover by hand actually wants.
    if (const char* env = std::getenv("QR_M_CROSSOVER")) {
        const long v = std::strtol(env, nullptr, 10);
        if (v > 0) {
            r.policy.m_crossover_small_batch = (unsigned)v;
            r.policy.m_crossover_large_batch = (unsigned)v;
            r.source = "env:QR_M_CROSSOVER";
        }
    }
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
QrBackend qr_backend(unsigned m, unsigned n, unsigned batch) {
    (void)n;
    const QrPolicy& p = state().policy;
    const unsigned crossover = batch < p.batch_threshold ? p.m_crossover_small_batch
                                                         : p.m_crossover_large_batch;
    return m >= crossover ? QrBackend::streaming_reduced : QrBackend::unblocked;
}

std::pair<mlx::core::array, mlx::core::array>
qr_accelerated(const mlx::core::array& a) {
    const auto& shape = a.shape();
    const auto M = static_cast<unsigned>(shape[shape.size() - 2]);
    const auto N = static_cast<unsigned>(shape[shape.size() - 1]);

    unsigned batch = 1;
    for (size_t i = 0; i + 2 < shape.size(); ++i) {
        batch *= static_cast<unsigned>(shape[i]);
    }

    if (qr_backend(M, N, batch) == QrBackend::streaming_reduced) {
        return detail::qr_streaming_amx_reduced(a);
    }
    return detail::qr_unblocked(a);
}

} // namespace metal_linalg
