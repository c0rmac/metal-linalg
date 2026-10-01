#import <Metal/Metal.h>
#import <IOKit/IOKitLib.h>

#include <metal_linalg/device.h>

#include <string>

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

} // namespace metal_linalg
