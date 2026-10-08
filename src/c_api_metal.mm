// The C API's functions that touch Metal objects (c_api.h): a Metal buffer's
// CPU address, so that a GPU framework's tensor memory can be passed to the
// decompositions in place, and that buffer made known to them for a call.
#import <Metal/Metal.h>

#include <metal_linalg/c_api.h>
#include "known_buffers.h"

#include <malloc/malloc.h>

extern "C" void* metal_linalg_buffer_contents(const void* buffer, uint64_t offset, uint64_t bytes) {
    // An Objective-C object is a malloc block. This turns away a pointer that
    // is not one (a buffer's contents passed for the buffer, say) before it is
    // messaged; another object that is not a buffer is turned away below.
    if (buffer == nullptr || malloc_size(buffer) == 0) return nullptr;
    id object = (__bridge id)buffer;
    if (![object conformsToProtocol:@protocol(MTLBuffer)]) return nullptr;
    id<MTLBuffer> b = (id<MTLBuffer>)object;
    if (b.storageMode != MTLStorageModeShared) return nullptr;
    const uint64_t length = b.length;
    if (offset > length || bytes > length - offset) return nullptr;
    char* base = static_cast<char*>(b.contents);
    return base ? base + offset : nullptr;
}

extern "C" int metal_linalg_know_buffer(const void* contents, const void* buffer) {
    if (!contents || metal_linalg_buffer_contents(buffer, 0, 0) != contents) return 0;
    metal_linalg::detail::know_buffer(contents, const_cast<void*>(buffer));
    return 1;
}

extern "C" void metal_linalg_forget_buffer(const void* contents, const void* buffer) {
    metal_linalg::detail::forget_buffer(contents, const_cast<void*>(buffer));
}
