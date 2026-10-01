#pragma once

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

} // namespace metal_linalg
