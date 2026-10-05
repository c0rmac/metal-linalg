// The estimate for a Mac nobody has measured (estimate.h); the arithmetic is
// tuning/estimate.py's, line for line, so that the row the library picks is
// the one docs/studies/estimated-policies.md lists for that Mac.
#include "estimate.h"

#include <metal_linalg/device.h>

#include <sys/sysctl.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iterator>
#include <limits>

namespace metal_linalg::detail {
namespace {

struct ChipSpec {
    const char* name;
    unsigned    gpu_cores;
    unsigned    cpu_cores;
    double      cpu;         // Geekbench 7 multi-core
    double      metal;       // Geekbench 7 Metal
    double      bandwidth;   // GB/s
};

// tuning/chip_specs.py; the last row matches nothing.
constexpr ChipSpec kChips[] = {
#include "tuned/chips.inc"
    {"", 0, 0, 0, 0, 0},
};

// tuning/estimate.py's LADDER and margins.
constexpr unsigned kLadder[] = {100, 115, 130, 150, 175, 200, 250, 300, 400};
constexpr double   kMarginSameGeneration  = 1.25;
constexpr double   kMarginOtherGeneration = 1.5;
constexpr double   kMarginCoreCounts      = 2.0;

// The listed entry for that Mac: the exact CPU core count if listed, else the
// first with that chip and GPU core count.
const ChipSpec* find_spec(const std::string& name, unsigned gpu_cores, unsigned cpu_cores) {
    const ChipSpec* first = nullptr;
    for (const ChipSpec& c : kChips) {
        if (c.name[0] == '\0' || name != c.name || c.gpu_cores != gpu_cores) continue;
        if (c.cpu_cores == cpu_cores) return &c;
        if (!first) first = &c;
    }
    return first;
}

// "Apple M5 Pro" -> "M5", "Apple A18 Pro" -> "A18"; "" for anything else.
std::string generation(const std::string& name) {
    if (name.rfind("Apple ", 0) != 0) return "";
    const size_t end = name.find(' ', 6);
    const std::string g = name.substr(6, end == std::string::npos ? std::string::npos : end - 6);
    if (g.size() < 2 || (g[0] != 'M' && g[0] != 'A')) return "";
    for (size_t i = 1; i < g.size(); ++i)
        if (!std::isdigit((unsigned char)g[i])) return "";
    return g;
}

// The smallest ladder value at least s, x 100 (the largest if none is).
unsigned on_ladder(double s) {
    for (unsigned v : kLadder)
        if (v / 100.0 >= s - 1e-9) return v;
    return kLadder[std::size(kLadder) - 1];
}

unsigned physical_cpus() {
    int n = 0;
    size_t len = sizeof n;
    return sysctlbyname("hw.physicalcpu", &n, &len, nullptr, 0) == 0 && n > 0 ? (unsigned)n : 0;
}

bool parse_unsigned(const std::string& s, unsigned& out) {
    if (s.empty()) return false;
    char* end = nullptr;
    const unsigned long v = std::strtoul(s.c_str(), &end, 10);
    if (*end != '\0' || v > std::numeric_limits<unsigned>::max()) return false;
    out = (unsigned)v;
    return true;
}

std::string times(unsigned x100) {
    char b[32];
    std::snprintf(b, sizeof b, "x%g", x100 / 100.0);
    return b;
}

} // namespace

EstimateTarget estimate_target(bool* forced) {
    EstimateTarget t{device_name(), gpu_core_count(), physical_cpus()};
    const char* as = std::getenv("METAL_LINALG_ESTIMATE_AS");
    if (forced) *forced = as && *as;
    if (!as || !*as) return t;

    std::vector<std::string> parts;
    for (std::string s = as;;) {
        const size_t c = s.find(':');
        parts.push_back(s.substr(0, c));
        if (c == std::string::npos) break;
        s = s.substr(c + 1);
    }
    if (!parts[0].empty() && parts[0] != t.device) {
        t.device = parts[0];
        // Cores not given: those listed for the chip, else this Mac's.
        for (const ChipSpec& c : kChips) {
            if (c.name[0] != '\0' && t.device == c.name) {
                t.gpu_cores = c.gpu_cores;
                t.cpu_cores = c.cpu_cores;
                break;
            }
        }
    }
    unsigned v = 0;
    if (parts.size() > 1 && parse_unsigned(parts[1], v)) {
        t.gpu_cores = v;
        if (const ChipSpec* s = find_spec(t.device, v, 0)) t.cpu_cores = s->cpu_cores;
    }
    if (parts.size() > 2 && parse_unsigned(parts[2], v)) t.cpu_cores = v;
    return t;
}

bool estimate(const EstimateTarget& t, const std::vector<Anchor>& anchors, Estimate& out) {
    // The anchor: one of the target's generation if there is one, then the
    // nearest in GPU cores; the first of equals.
    const std::string gen = generation(t.device);
    const ChipSpec* a = nullptr;
    size_t index = 0;
    bool a_other = true;
    double a_dist = 0.0;
    for (size_t i = 0; i < anchors.size(); ++i) {
        const ChipSpec* s = find_spec(anchors[i].device, anchors[i].gpu_cores, anchors[i].cpu_cores);
        if (!s) continue;
        const bool other = generation(s->name) != gen;
        const double dist = std::fabs(std::log((double)std::max(t.gpu_cores, 1u) / s->gpu_cores));
        if (!a || other < a_other || (other == a_other && dist < a_dist)) {
            a = s;
            index = i;
            a_other = other;
            a_dist = dist;
        }
    }
    if (!a) return false;

    double small, large;
    const char* basis;
    if (!t.gpu_cores || !t.cpu_cores) {
        small = large = kLadder[std::size(kLadder) - 1] / 100.0;
        basis = "unknown cores";
    } else if (const ChipSpec* s = find_spec(t.device, t.gpu_cores, t.cpu_cores)) {
        const double q_small = (s->metal / s->cpu) / (a->metal / a->cpu);
        const double q_large = std::min(q_small, (s->bandwidth / s->cpu) / (a->bandwidth / a->cpu));
        const bool same = !gen.empty() && gen == generation(a->name);
        const double margin = same ? kMarginSameGeneration : kMarginOtherGeneration;
        small = std::max(1.0, margin / q_small);
        large = std::max(1.0, margin / q_large);
        basis = "benchmarks";
    } else {
        const double q = ((double)t.gpu_cores / t.cpu_cores) / ((double)a->gpu_cores / a->cpu_cores);
        small = large = std::max(1.0, kMarginCoreCounts / q);
        basis = "core counts";
    }
    out.anchor     = index;
    out.small_x100 = on_ladder(small);
    out.large_x100 = on_ladder(large);
    out.source = "estimated:" + t.device + " (from " + a->name + ", " + std::to_string(a->gpu_cores) +
                 " GPU cores; GPU " + times(out.small_x100) + " batched, " + times(out.large_x100) +
                 " large; " + basis + ")";
    return true;
}

} // namespace metal_linalg::detail
