// How current a device's measurements are, and the notice the library prints
// when they are missing or not current. The states are tuning/kernels.py's:
// tuning/generate_tables.py marks a tuned row whose runs are stale or
// incomplete with a last field, and docs/measurements.md shows the same for
// every chip.
#pragma once

#include <string>

namespace metal_linalg {

// The last field of a kTuned[] row (src/tuned/*.inc); rows without it read 0.
constexpr unsigned kCalibrationCurrent    = 0;   // measured at the current kernels
constexpr unsigned kCalibrationStale      = 1;   // measured on older kernels; still used
constexpr unsigned kCalibrationIncomplete = 2;   // a newer backend never timed; it stays off
constexpr unsigned kUncalibrated          = 3;   // no row: the untuned default

namespace detail {

// "tuned:", "tuned-stale:" or "tuned-incomplete:", the policy-source prefix of
// a tuned row in that state.
const char* tuned_source_prefix(unsigned calibration);

// The one-line notice for decomposition `what` ("QR", "eigh", "SVD") in that
// state on this Mac, or "" for a current calibration.
std::string calibration_notice_text(const char* what, unsigned calibration,
                                    const std::string& device, unsigned gpu_cores);

// Prints that notice to stderr, at most once per decomposition per process,
// unless notices are off (set_calibration_notices, or
// METAL_LINALG_NO_CALIBRATION_NOTICE=1) or there is no Metal device.
void calibration_notice(const char* what, unsigned calibration);

} // namespace detail
} // namespace metal_linalg
