// Tests of the estimated policies (src/estimate.h): the slowdowns a Mac gets
// against a measured one, which measured Mac it is estimated from, the
// METAL_LINALG_ESTIMATE_AS override, and the policies resolving to an
// estimate under it. The values expected are tuning/estimate.py's for the
// same Macs (python3 tuning/estimate.py prints them), so the two agree.
#include <metal_linalg/core.h>
#include <metal_linalg/device.h>

#include "estimate.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace metal_linalg;
using detail::Anchor;
using detail::Estimate;
using detail::EstimateTarget;

static int g_checks = 0, g_failures = 0;

static void check(bool ok, const std::string& what, const std::string& detail = "") {
    ++g_checks;
    if (!ok) {
        ++g_failures;
        std::printf("  FAIL  %s%s%s\n", what.c_str(), detail.empty() ? "" : ": ", detail.c_str());
    } else {
        std::printf("  ok    %s\n", what.c_str());
    }
}

static bool starts(const std::string& s, const std::string& prefix) { return s.rfind(prefix, 0) == 0; }

static void expect(const std::vector<Anchor>& anchors, const EstimateTarget& t, size_t anchor,
                   unsigned small, unsigned large, const char* basis) {
    Estimate e;
    const bool ok = detail::estimate(t, anchors, e);
    const std::string what = t.device + " " + std::to_string(t.gpu_cores) + "/" + std::to_string(t.cpu_cores);
    check(ok && e.anchor == anchor && e.small_x100 == small && e.large_x100 == large &&
              e.source.find(std::string(basis) + ")") != std::string::npos &&
              starts(e.source, "estimated:" + t.device + " (from ") &&
              e.source.find("(from " + anchors[anchor].device) != std::string::npos,
          what + ": x" + std::to_string(small) + "/x" + std::to_string(large) + " from " + anchors[anchor].device,
          ok ? e.source : "no estimate");
}

int main() {
    // First, before any policy is resolved: under the override every policy
    // is estimated, whatever this Mac's measurements.
    setenv("METAL_LINALG_ESTIMATE_AS", "Apple M1:8:8", 1);
    setenv("METAL_LINALG_NO_CALIBRATION_NOTICE", "1", 1);
    std::printf("\nEstimated policies, on %s (%u GPU cores)\n[ policies under METAL_LINALG_ESTIMATE_AS ]\n",
                device_name(), gpu_core_count());
    const bool has_device = device_name()[0] != '\0';
    for (const char* src : {qr_policy_source(), eigh_policy_source(), svd_policy_source(), cholesky_policy_source()}) {
        check(!has_device || starts(src, "estimated:Apple M1 (from "), "policy source", src);
    }
    if (has_device) {
        // A weaker GPU than any measured one never takes small batches the
        // untuned default sent to the GPU: 16 matrices of 64 x 64.
        check(qr_backend(64, 64, 16) == QrBackend::cpu, "QR, 16 of 64x64, on the CPU");
        check(eigh_backend(64, 16) == EighBackend::cpu, "eigh, 16 of 64x64, on the CPU");
    }

    std::printf("[ slowdowns against the M5 Pro ]\n");
    const std::vector<Anchor> m5 = {{"Apple M5 Pro", 20, 18}};
    expect(m5, {"Apple M5 Max", 40, 18}, 0, 100, 100, "benchmarks");   // a stronger GPU: the anchor's own
    expect(m5, {"Apple M1", 8, 8}, 0, 200, 200, "benchmarks");
    expect(m5, {"Apple M4 Pro", 20, 14}, 0, 150, 150, "benchmarks");
    expect(m5, {"Apple A18 Pro", 5, 6}, 0, 150, 175, "benchmarks");     // bandwidth: the large backends slower
    expect(m5, {"Apple M3", 10, 8}, 0, 150, 175, "benchmarks");
    expect(m5, {"Apple M3 GPU", 10, 8}, 0, 150, 175, "benchmarks");     // iPadOS's form of the name
    expect(m5, {"  apple   m3 ", 10, 8}, 0, 150, 175, "benchmarks");     // case and spacing
    expect(m5, {"Apple M9", 30, 14}, 0, 115, 115, "core counts");       // not listed: core counts
    expect(m5, {"Apple M9", 0, 14}, 0, 400, 400, "unknown cores");      // GPU cores unknown: the most cautious

    std::printf("[ which measured Mac ]\n");
    const std::vector<Anchor> two = {{"Apple M5 Pro", 20, 18}, {"Apple M1", 8, 8}};
    expect(two, {"Apple M1 Pro", 16, 10}, 1, 100, 100, "benchmarks");   // its generation first
    expect(two, {"Apple M5 Max", 40, 18}, 0, 100, 100, "benchmarks");
    expect(two, {"Apple M4", 10, 10}, 1, 150, 175, "benchmarks");       // else the nearest in GPU cores
    {
        Estimate e;
        check(!detail::estimate({"Apple M1", 8, 8}, {{"Apple Z1", 8, 8}}, e), "no listed anchor, no estimate");
    }

    std::printf("[ METAL_LINALG_ESTIMATE_AS ]\n");
    bool forced = false;
    setenv("METAL_LINALG_ESTIMATE_AS", "Apple M4 Max", 1);
    EstimateTarget t = detail::estimate_target(&forced);
    check(forced && t.device == "Apple M4 Max" && t.gpu_cores == 32 && t.cpu_cores == 14,
          "a chip alone: its first listing", t.device + " " + std::to_string(t.gpu_cores) + "/" + std::to_string(t.cpu_cores));
    setenv("METAL_LINALG_ESTIMATE_AS", "Apple M4 Max:40", 1);
    t = detail::estimate_target(&forced);
    check(t.gpu_cores == 40 && t.cpu_cores == 16, "GPU cores: the CPU cores listed with them");
    setenv("METAL_LINALG_ESTIMATE_AS", "Apple M4 Max:40:12", 1);
    t = detail::estimate_target(&forced);
    check(t.gpu_cores == 40 && t.cpu_cores == 12, "all three given");
    unsetenv("METAL_LINALG_ESTIMATE_AS");
    t = detail::estimate_target(&forced);
    check(!forced && t.device == device_name() && t.gpu_cores == gpu_core_count(), "unset: this Mac, not forced");

    std::printf("\n%d checks, %d failed\n", g_checks, g_failures);
    return g_failures ? 1 : 0;
}
