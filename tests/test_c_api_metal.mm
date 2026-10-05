// Tests of metal_linalg_buffer_contents (c_api.h): which Metal buffers it
// gives an address for, and the decompositions reading their input from, and
// writing their outputs to, buffer memory in place, sub-allocated from a heap
// as PyTorch's MPS allocator does it. Each shape is run at a batch the CPU
// takes and at one the GPU takes on a measured device.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <metal_linalg/c_api.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

static int g_failures = 0;
static int g_checks   = 0;

#define CHECK(cond, ...)                                    \
    do {                                                    \
        ++g_checks;                                         \
        if (!(cond)) {                                      \
            ++g_failures;                                   \
            printf("  FAIL  ");                             \
            printf(__VA_ARGS__);                            \
            printf("\n");                                   \
        }                                                   \
    } while (0)

namespace {

constexpr float kTol = 2e-5f;

// Buffers carved out of one shared heap, like MPS tensors.
struct Arena {
    id<MTLHeap> heap;
    NSMutableArray<id<MTLBuffer>>* held = [NSMutableArray array];

    Arena(id<MTLDevice> device, size_t bytes) {
        MTLHeapDescriptor* d = [MTLHeapDescriptor new];
        d.storageMode = MTLStorageModeShared;
        d.size = bytes;
        heap = [device newHeapWithDescriptor:d];
    }
    // The CPU address of a fresh buffer of `floats` floats from the heap.
    float* floats(size_t floats) {
        id<MTLBuffer> b = [heap newBufferWithLength:std::max<size_t>(floats, 1) * sizeof(float)
                                            options:MTLResourceStorageModeShared];
        if (!b) return nullptr;
        [held addObject:b];
        return static_cast<float*>(metal_linalg_buffer_contents((__bridge const void*)b, 0, floats * sizeof(float)));
    }
};

std::vector<float> random_matrices(size_t count, uint32_t seed, bool symmetric, uint32_t n) {
    std::mt19937 gen(seed);
    std::normal_distribution<float> dist;
    std::vector<float> a(count);
    for (float& x : a) x = dist(gen);
    if (symmetric)
        for (size_t b = 0; b < count / ((size_t)n * n); ++b) {
            float* m = a.data() + b * n * n;
            for (uint32_t i = 0; i < n; ++i)
                for (uint32_t j = 0; j < i; ++j) m[j * n + i] = m[i * n + j];
        }
    return a;
}

// max over the batch of ||A - X diag(d) Y|| / ||A||, with X rows x k, Y k x
// cols, and d all ones when null.
double reconstruction(const float* a, const float* x, const float* d, const float* y,
                      uint32_t batch, uint32_t rows, uint32_t cols, uint32_t k) {
    double worst = 0.0;
    for (uint32_t b = 0; b < batch; ++b) {
        const float* A = a + (size_t)b * rows * cols;
        const float* X = x + (size_t)b * rows * k;
        const float* Y = y + (size_t)b * k * cols;
        double num = 0.0, den = 0.0;
        for (uint32_t i = 0; i < rows; ++i)
            for (uint32_t j = 0; j < cols; ++j) {
                double s = 0.0;
                for (uint32_t t = 0; t < k; ++t)
                    s += (double)X[i * k + t] * (d ? d[(size_t)b * k + t] : 1.0f) * Y[t * cols + j];
                num += (s - A[i * cols + j]) * (s - A[i * cols + j]);
                den += (double)A[i * cols + j] * A[i * cols + j];
            }
        worst = std::max(worst, std::sqrt(num / den));
    }
    return worst;
}

// max over the batch of ||Q^T Q - I|| for Q rows x k.
double orthogonality(const float* q, uint32_t batch, uint32_t rows, uint32_t k) {
    double worst = 0.0;
    for (uint32_t b = 0; b < batch; ++b) {
        const float* Q = q + (size_t)b * rows * k;
        double e = 0.0;
        for (uint32_t i = 0; i < k; ++i)
            for (uint32_t j = 0; j < k; ++j) {
                double s = 0.0;
                for (uint32_t t = 0; t < rows; ++t) s += (double)Q[t * k + i] * Q[t * k + j];
                s -= (i == j);
                e += s * s;
            }
        worst = std::max(worst, std::sqrt(e));
    }
    return worst;
}

double max_difference(const float* x, const float* y, size_t count) {
    double worst = 0.0;
    for (size_t i = 0; i < count; ++i) worst = std::max(worst, (double)std::fabs(x[i] - y[i]));
    return worst;
}

// Vt (k x cols per matrix) as V (cols x k), so it has orthonormal columns.
std::vector<float> transposed(const float* vt, uint32_t batch, uint32_t k, uint32_t cols) {
    std::vector<float> v((size_t)batch * k * cols);
    for (uint32_t b = 0; b < batch; ++b)
        for (uint32_t i = 0; i < k; ++i)
            for (uint32_t j = 0; j < cols; ++j)
                v[(size_t)b * k * cols + j * k + i] = vt[(size_t)b * k * cols + i * cols + j];
    return v;
}

void test_addresses(id<MTLDevice> device) {
    printf("buffer addresses\n");
    CHECK(metal_linalg_buffer_contents(nullptr, 0, 0) == nullptr, "NULL buffer gave an address");

    id<MTLBuffer> shared = [device newBufferWithLength:4096 options:MTLResourceStorageModeShared];
    const void* s = (__bridge const void*)shared;
    char* base = static_cast<char*>(shared.contents);
    CHECK(metal_linalg_buffer_contents(s, 0, 4096) == base, "shared buffer: wrong address");
    CHECK(metal_linalg_buffer_contents(s, 1024, 3072) == base + 1024, "shared buffer at an offset: wrong address");
    CHECK(metal_linalg_buffer_contents(s, 4096, 0) == base + 4096, "empty range at the end: wrong address");
    CHECK(metal_linalg_buffer_contents(s, 1024, 3073) == nullptr, "range past the end gave an address");
    CHECK(metal_linalg_buffer_contents(s, 4097, 0) == nullptr, "offset past the end gave an address");
    CHECK(metal_linalg_buffer_contents(s, 8, UINT64_MAX) == nullptr, "overflowing range gave an address");
    CHECK(metal_linalg_buffer_contents(base, 0, 16) == nullptr, "a buffer's contents taken for the buffer");

    id<MTLBuffer> priv = [device newBufferWithLength:4096 options:MTLResourceStorageModePrivate];
    CHECK(metal_linalg_buffer_contents((__bridge const void*)priv, 0, 16) == nullptr,
          "private buffer gave an address");
    NSObject* other = [NSObject new];
    CHECK(metal_linalg_buffer_contents((__bridge const void*)other, 0, 0) == nullptr,
          "an object that is not a buffer gave an address");
}

void test_qr(Arena& arena, uint32_t batch, uint32_t m, uint32_t n) {
    const uint32_t k = std::min(m, n);
    const size_t na = (size_t)batch * m * n, nq = (size_t)batch * m * k, nr = (size_t)batch * k * n;
    const std::vector<float> host = random_matrices(na, 1, false, 0);
    float* a = arena.floats(na);
    float* q = arena.floats(nq);
    float* r = arena.floats(nr);
    CHECK(a && q && r, "qr %ux%u x%u: no buffer memory", m, n, batch);
    if (!(a && q && r)) return;
    std::copy(host.begin(), host.end(), a);
    std::vector<float> q_ref(nq), r_ref(nr);
    CHECK(metal_linalg_qr(host.data(), batch, m, n, q_ref.data(), r_ref.data()) == METAL_LINALG_OK,
          "qr on host memory: %s", metal_linalg_last_error());
    CHECK(metal_linalg_qr(a, batch, m, n, q, r) == METAL_LINALG_OK, "qr in buffers: %s", metal_linalg_last_error());
    const double rec = reconstruction(host.data(), q, nullptr, r, batch, m, n, k);
    const double orth = orthogonality(q, batch, m, k);
    // Diagonal of R up to sign: the same factorization whichever side solved a matrix.
    double diag = 0.0;
    for (uint32_t b = 0; b < batch; ++b)
        for (uint32_t i = 0; i < k; ++i)
            diag = std::max(diag, (double)std::fabs(std::fabs(r[(size_t)b * k * n + i * n + i]) -
                                                    std::fabs(r_ref[(size_t)b * k * n + i * n + i])));
    printf("  qr %ux%u x%-5u [%s]: reconstruction %.1e, orthogonality %.1e, |R_ii| vs host %.1e\n", m, n, batch,
           metal_linalg_qr_backend(m, n, batch), rec, orth, diag);
    CHECK(rec < kTol && orth < 10 * kTol && diag < 1e-4, "qr %ux%u x%u in buffers is off", m, n, batch);
}

void test_eigh(Arena& arena, uint32_t batch, uint32_t n, bool vectors) {
    const size_t na = (size_t)batch * n * n, nw = (size_t)batch * n;
    const std::vector<float> host = random_matrices(na, 2, true, n);
    float* a = arena.floats(na);
    float* w = arena.floats(nw);
    float* v = vectors ? arena.floats(na) : nullptr;
    CHECK(a && w && (v || !vectors), "eigh %u x%u: no buffer memory", n, batch);
    if (!(a && w && (v || !vectors))) return;
    std::copy(host.begin(), host.end(), a);
    std::vector<float> w_ref(nw);
    CHECK(metal_linalg_eigh(host.data(), batch, n, 1, w_ref.data(), nullptr, nullptr) == METAL_LINALG_OK,
          "eigh on host memory: %s", metal_linalg_last_error());
    CHECK(metal_linalg_eigh(a, batch, n, 1, w, v, nullptr) == METAL_LINALG_OK,
          "eigh in buffers: %s", metal_linalg_last_error());
    const double values = max_difference(w, w_ref.data(), nw);
    double rec = 0.0, orth = 0.0;
    if (vectors) {
        const std::vector<float> vt = transposed(v, batch, n, n);
        rec = reconstruction(host.data(), v, w, vt.data(), batch, n, n, n);
        orth = orthogonality(v, batch, n, n);
    }
    printf("  %s %u x%-5u [%s]: eigenvalues vs host %.1e, reconstruction %.1e, orthogonality %.1e\n",
           vectors ? "eigh    " : "eigvalsh", n, batch,
           vectors ? metal_linalg_eigh_backend(n, batch) : metal_linalg_eigvalsh_backend(n, batch), values, rec, orth);
    CHECK(values < 1e-4 && rec < 10 * kTol && orth < 10 * kTol, "eigh %u x%u in buffers is off", n, batch);
}

void test_svd(Arena& arena, uint32_t batch, uint32_t m, uint32_t n, bool vectors) {
    const uint32_t k = std::min(m, n);
    const size_t na = (size_t)batch * m * n, ns = (size_t)batch * k;
    const std::vector<float> host = random_matrices(na, 3, false, 0);
    float* a = arena.floats(na);
    float* s = arena.floats(ns);
    float* u = vectors ? arena.floats((size_t)batch * m * k) : nullptr;
    float* vt = vectors ? arena.floats((size_t)batch * k * n) : nullptr;
    CHECK(a && s && (!vectors || (u && vt)), "svd %ux%u x%u: no buffer memory", m, n, batch);
    if (!(a && s && (!vectors || (u && vt)))) return;
    std::copy(host.begin(), host.end(), a);
    std::vector<float> s_ref(ns);
    CHECK(metal_linalg_svd(host.data(), batch, m, n, nullptr, s_ref.data(), nullptr, nullptr) == METAL_LINALG_OK,
          "svd on host memory: %s", metal_linalg_last_error());
    CHECK(metal_linalg_svd(a, batch, m, n, u, s, vt, nullptr) == METAL_LINALG_OK,
          "svd in buffers: %s", metal_linalg_last_error());
    const double values = max_difference(s, s_ref.data(), ns);
    double rec = 0.0, orth = 0.0;
    if (vectors) {
        rec = reconstruction(host.data(), u, s, vt, batch, m, n, k);
        const std::vector<float> v = transposed(vt, batch, k, n);
        orth = std::max(orthogonality(u, batch, m, k), orthogonality(v.data(), batch, n, k));
    }
    printf("  %s %ux%u x%-5u [%s]: singular values vs host %.1e, reconstruction %.1e, orthogonality %.1e\n",
           vectors ? "svd    " : "svdvals", m, n, batch,
           vectors ? metal_linalg_svd_backend(m, n, batch) : metal_linalg_svdvals_backend(m, n, batch), values, rec,
           orth);
    CHECK(values < 1e-4 && rec < 10 * kTol && orth < 10 * kTol, "svd %ux%u x%u in buffers is off", m, n, batch);
}

} // namespace

int main() {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            printf("No Metal device; skipped.\n");
            return 0;
        }
        printf("\nC API on Metal buffers, %s (%u GPU cores)\n", metal_linalg_device_name(),
               metal_linalg_gpu_core_count());
        test_addresses(device);

        printf("decompositions in heap buffers\n");
        Arena arena(device, (size_t)512 << 20);
        for (uint32_t batch : {16u, 4096u}) {
            test_qr(arena, batch, 64, 32);
            test_qr(arena, batch, 16, 48);
            test_eigh(arena, batch, 32, true);
            test_eigh(arena, batch, 32, false);
            test_svd(arena, batch, 48, 16, true);
            test_svd(arena, batch, 32, 32, false);
        }

        printf("\n%d of %d checks failed\n", g_failures, g_checks);
    }
    return g_failures ? 1 : 0;
}
