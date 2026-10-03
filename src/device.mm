#import <Metal/Metal.h>
#import <IOKit/IOKitLib.h>

#include <metal_linalg/device.h>
#include "calibration.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <thread>

namespace metal_linalg {
namespace {

// The core count is not in the Metal API; the GPU's IORegistry entry has it.
unsigned read_gpu_core_count() {
    unsigned cores = 0;
    io_iterator_t it = 0;
    if (IOServiceGetMatchingServices(kIOMainPortDefault,
                                     IOServiceMatching("AGXAccelerator"),
                                     &it) != KERN_SUCCESS) {
        return 0;
    }
    io_object_t svc;
    while ((svc = IOIteratorNext(it))) {
        CFTypeRef v = IORegistryEntryCreateCFProperty(
            svc, CFSTR("gpu-core-count"), kCFAllocatorDefault, 0);
        if (v) {
            if (CFGetTypeID(v) == CFNumberGetTypeID()) {
                int n = 0;
                CFNumberGetValue((CFNumberRef)v, kCFNumberIntType, &n);
                if (n > 0) cores = (unsigned)n;
            }
            CFRelease(v);
        }
        IOObjectRelease(svc);
        if (cores) break;
    }
    IOObjectRelease(it);
    return cores;
}

struct Device {
    std::string name;
    unsigned    cores = 0;
};

const Device& device() {
    static const Device d = [] {
        Device r;
        @autoreleasepool {
            id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
            if (dev) r.name = dev.name.UTF8String;
        }
        r.cores = read_gpu_core_count();
        return r;
    }();
    return d;
}

} // namespace

const char* device_name()    { return device().name.c_str(); }
unsigned    gpu_core_count() { return device().cores; }

namespace {
std::atomic<unsigned> g_cpu_threads{0};   // 0: the default

unsigned default_cpu_threads() {
    static const unsigned n = [] {
        if (const char* s = std::getenv("METAL_LINALG_CPU_THREADS")) {
            const long v = std::strtol(s, nullptr, 10);
            if (v > 0) return (unsigned)v;
        }
        return std::max(1u, std::thread::hardware_concurrency());
    }();
    return n;
}
} // namespace

void set_cpu_threads(unsigned n) { g_cpu_threads = n; }

unsigned cpu_threads() {
    const unsigned n = g_cpu_threads;
    return n ? n : default_cpu_threads();
}

namespace {
std::atomic<bool> g_notices{true};
constexpr const char* kContribute = "https://github.com/c0rmac/metal-linalg/blob/main/CONTRIBUTING.md";
} // namespace

void set_calibration_notices(bool enabled) { g_notices = enabled; }
bool calibration_notices()                 { return g_notices; }

namespace detail {

const char* tuned_source_prefix(unsigned calibration) {
    switch (calibration) {
        case kCalibrationStale:      return "tuned-stale:";
        case kCalibrationIncomplete: return "tuned-incomplete:";
        default:                     return "tuned:";
    }
}

std::string calibration_notice_text(const char* what, unsigned calibration,
                                    const std::string& dev, unsigned cores) {
    const std::string mac = dev + (cores ? ", " + std::to_string(cores) + " GPU cores" : "");
    const std::string hide = " (METAL_LINALG_NO_CALIBRATION_NOTICE=1 hides this)";
    switch (calibration) {
        case kUncalibrated:
            return std::string("metal-linalg: ") + what + " is not calibrated for this Mac (" + mac +
                   "); it is using untuned defaults, which may leave GPU speedups unused. Measuring "
                   "this Mac takes about an hour and a half, and submitting the results improves the library "
                   "for everyone with this chip: " + kContribute + hide;
        case kCalibrationStale:
            return std::string("metal-linalg: ") + what + "'s calibration for this Mac (" + mac +
                   ") was measured on older kernels; it still applies, but measuring this Mac again "
                   "would bring it up to date: " + kContribute + hide;
        case kCalibrationIncomplete:
            return std::string("metal-linalg: ") + what + "'s calibration for this Mac (" + mac +
                   ") predates newer optimisations, which stay off until this Mac is measured "
                   "again: " + kContribute + hide;
        default:
            return "";
    }
}

namespace {
std::mutex g_state_mutex;
std::map<std::string, unsigned> g_states;   // decomposition -> its calibration, once resolved
} // namespace

void calibration_notice(const char* what, unsigned calibration) {
    {
        std::lock_guard<std::mutex> lock(g_state_mutex);
        g_states[what] = calibration;
    }
    if (calibration == kCalibrationCurrent || !g_notices) return;
    if (const char* e = std::getenv("METAL_LINALG_NO_CALIBRATION_NOTICE")) {
        if (*e && std::string(e) != "0") return;
    }
    if (device().name.empty()) return;   // no Metal device: nothing to calibrate
    static std::mutex m;
    static std::set<std::string> shown;
    {
        std::lock_guard<std::mutex> lock(m);
        if (!shown.insert(what).second) return;
    }
    const std::string text = calibration_notice_text(what, calibration, device().name, device().cores);
    std::fprintf(stderr, "%s\n", text.c_str());
}

} // namespace detail

std::string calibration_message(const char* decomposition) {
    unsigned state = kCalibrationCurrent;
    {
        std::lock_guard<std::mutex> lock(detail::g_state_mutex);
        auto it = detail::g_states.find(decomposition);
        if (it == detail::g_states.end() || device().name.empty()) return "";
        state = it->second;
    }
    return detail::calibration_notice_text(decomposition, state, device().name, device().cores);
}

} // namespace metal_linalg
