// Which kernel a problem gets, where that decision came from, and how to
// change it. Every call is routed by a policy measured for the device it
// runs on; the queries below say what the next call would do, and running it
// is not needed to find out.
#include <metal_linalg/metal_linalg.h>
#include <mlx/mlx.h>

#include <cstdio>

namespace ml = metal_linalg;

const char* name(ml::QrBackend b) {
    return b == ml::QrBackend::unblocked ? "GPU, one threadgroup per matrix" : "GPU, grid-parallel";
}
const char* name(ml::EighBackend b) {
    switch (b) {
        case ml::EighBackend::cpu:         return "CPU (MLX / Accelerate)";
        case ml::EighBackend::simd:        return "GPU, whole-matrix kernel, one simdgroup";
        case ml::EighBackend::threadgroup: return "GPU, whole-matrix kernel, one threadgroup";
        default:                           return "GPU, block Jacobi";
    }
}
const char* name(ml::SvdBackend b) {
    switch (b) {
        case ml::SvdBackend::cpu:             return "CPU (MLX / Accelerate)";
        case ml::SvdBackend::jacobi:          return "GPU, whole-matrix kernel";
        case ml::SvdBackend::block_jacobi:    return "GPU, block kernel";
        case ml::SvdBackend::qr_jacobi:       return "GPU, QR then whole-matrix kernel";
        default:                              return "GPU, QR then block kernel";
    }
}

int main() {
    std::printf("device: %s, %u GPU cores\n", ml::device_name(), ml::gpu_core_count());
    std::printf("policies: qr %s | eigh %s | svd %s\n\n",
                ml::qr_policy_source(), ml::eigh_policy_source(), ml::svd_policy_source());

    std::printf("qr   64 x 64,      batch 1000   -> %s\n", name(ml::qr_backend(64, 64, 1000)));
    std::printf("qr   2048 x 256,   batch 4      -> %s\n", name(ml::qr_backend(2048, 256, 4)));
    std::printf("eigh 32 x 32,      batch 4096   -> %s\n", name(ml::eigh_backend(32, 4096)));
    std::printf("eigh 512 x 512,    batch 64     -> %s\n", name(ml::eigh_backend(512, 64)));
    std::printf("eigh 512 x 512,    batch 1      -> %s\n", name(ml::eigh_backend(512, 1)));
    std::printf("svd  1024 x 64,    batch 64     -> %s\n", name(ml::svd_backend(1024, 64, 64)));
    std::printf("svd  256 x 256,    batch 16     -> %s\n", name(ml::svd_backend(256, 256, 16)));

    // A policy can be replaced at run time (or through environment variables
    // such as EIGH_GPU_MIN_BATCH=1; see docs/tuning.md). Here: allow single
    // matrices on the GPU, then restore the measured policy.
    const ml::EighPolicy measured = ml::eigh_policy();
    ml::EighPolicy p = measured;
    p.gpu_min_batch = 1;
    p.gpu_min_batch_times_n = 0;
    p.gpu_max_n = ml::kEighNoLimit;
    ml::set_eigh_policy(p);
    std::printf("\nwith a lone matrix allowed on the GPU (source: %s):\n", ml::eigh_policy_source());
    std::printf("eigh 512 x 512,    batch 1      -> %s\n", name(ml::eigh_backend(512, 1)));
    const bool overridden = ml::eigh_backend(512, 1) != ml::EighBackend::cpu;
    ml::set_eigh_policy(measured);
    return overridden ? 0 : 1;
}
