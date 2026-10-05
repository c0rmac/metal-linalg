// The routing of a Mac nobody has measured: a measured Mac's (an anchor's)
// timings refitted as if its GPU were `small` times slower against its CPU
// for the batched kernels and `large` times for the large-matrix backends,
// those two taken from how the Macs compare in published benchmarks
// (tuning/chip_specs.py) or else in core counts. tuning/estimate.py fits the
// rows for every pair on the ladder and holds the reasoning; this mirrors its
// slowdowns(), on_ladder() and choose_anchor(). See docs/tuning.md and
// docs/studies/estimated-policies.md.
#pragma once

#include <string>
#include <vector>

namespace metal_linalg::detail {

// The Mac being estimated.
struct EstimateTarget {
    std::string device;      // MTLDevice.name
    unsigned    gpu_cores;   // 0 if unknown
    unsigned    cpu_cores;   // physical; 0 if unknown
};

// This Mac, or the one METAL_LINALG_ESTIMATE_AS="<name>[:<GPU cores>[:<CPU
// cores>]]" names (cores it leaves out are this Mac's, or for a listed chip
// the CPU cores listed with its GPU cores). `forced` is set when the variable
// is: the policies then estimate even a measured Mac, to preview what another
// Mac gets or to test the estimates.
EstimateTarget estimate_target(bool* forced);

// A measured Mac with estimated rows.
struct Anchor {
    std::string device;
    unsigned    gpu_cores;
    unsigned    cpu_cores;
};

struct Estimate {
    size_t      anchor;       // index into the anchors given
    unsigned    small_x100;   // ladder values x 100
    unsigned    large_x100;
    std::string source;       // "estimated:<device> (from <anchor>: ...)"
};

// The estimate for `target` from the anchors that have rows for one
// decomposition (those tuning/chip_specs.py does not list are passed over).
// False if there are none.
bool estimate(const EstimateTarget& target, const std::vector<Anchor>& anchors, Estimate& out);

// Picks the estimated row for `target` out of a decomposition's table, whose
// entries carry small_x100, large_x100, anchor_cpu_cores and a TunedEntry
// `row` (the anchor's name and GPU cores, and the fields); null if the
// table has no row for the estimate.
template <class Entry, size_t N>
const Entry* estimated_row(const Entry (&table)[N], const EstimateTarget& target, std::string& source) {
    std::vector<Anchor> anchors;
    for (const Entry& e : table) {
        if (e.row.device_name[0] == '\0') continue;
        bool seen = false;
        for (const Anchor& a : anchors) seen = seen || (a.device == e.row.device_name && a.gpu_cores == e.row.gpu_cores);
        if (!seen) anchors.push_back({e.row.device_name, e.row.gpu_cores, e.anchor_cpu_cores});
    }
    Estimate est;
    if (!estimate(target, anchors, est)) return nullptr;
    const Anchor& a = anchors[est.anchor];
    for (const Entry& e : table) {
        if (a.device == e.row.device_name && a.gpu_cores == e.row.gpu_cores &&
            e.small_x100 == est.small_x100 && e.large_x100 == est.large_x100) {
            source = est.source;
            return &e;
        }
    }
    return nullptr;
}

} // namespace metal_linalg::detail
