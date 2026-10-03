#pragma once

#include <string>

namespace metal_linalg {

    // The GPU this library runs on, detected once on first use. Every routing
    // policy (qr_policy, eigh_policy, svd_policy) is looked up by both values,
    // so a part sold under the same name with fewer cores does not pick up a
    // policy measured on the full one.

    // MTLDevice.name of the default Metal device, e.g. "Apple M5 Pro"; empty
    // if there is no Metal device.
    const char* device_name();

    // Its GPU core count, from the IORegistry; 0 if it could not be read.
    unsigned gpu_core_count();

    // CPU threads. The CPU paths (LAPACK through Accelerate) spread a batch
    // over up to this many threads, each solving whole matrices with
    // Accelerate's own threading off; a lone matrix keeps Accelerate's
    // threading. The default is every core the system reports. A caller that
    // runs several solves at once on its own threads can cap it with
    // set_cpu_threads(n) or the environment variable
    // METAL_LINALG_CPU_THREADS=n; 0 restores the default, and 1 solves a
    // batch one matrix after another. The tuned GPU-or-CPU boundaries were
    // measured with every core, so a cap moves the real boundary toward the
    // GPU without moving the policy.
    void     set_cpu_threads(unsigned n);
    unsigned cpu_threads();

    // Calibration notices. When a decomposition's routing policy is first
    // resolved on a Mac without measurements for it, or with measurements
    // that are stale (taken on older kernels) or incomplete (from before a
    // newer backend), the library prints one line to stderr saying so and
    // how to measure the Mac -- at most once per decomposition per process.
    // On by default; set_calibration_notices(false), or the environment
    // variable METAL_LINALG_NO_CALIBRATION_NOTICE=1, turns them off. The
    // policy sources (qr_policy_source() etc.) report the same state:
    // "default:untuned-device", "tuned-stale:<device>", "tuned-incomplete:<device>".
    void set_calibration_notices(bool enabled);
    bool calibration_notices();

    // The notice for one decomposition ("QR", "eigh" or "SVD") as it stands,
    // whether or not it was printed: empty if its calibration is current or
    // its policy has not been resolved yet (any *_policy_source() call does).
    std::string calibration_message(const char* decomposition);

} // namespace metal_linalg
