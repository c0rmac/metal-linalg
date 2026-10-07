// MLX's arrays live in Metal buffers already. While a KnownBuffer is alive,
// wrap_host and input_buffer (metal_runtime.mm) hand out its buffer for
// memory that starts where the buffer does, rather than wrapping the memory
// in a new one: a new buffer over the same pages costs the first command
// buffer that uses it some 10-20 us a MB to map (about 1 ms for 64 MB on an
// M5 Pro), which for a large batch of small matrices -- the input, Q and R
// each wrapped -- was up to two fifths of a call. Plain C++, for the MLX
// layer (mlx_api.cpp).
#pragma once

namespace metal_linalg::detail {

class KnownBuffer {
public:
    // `buffer` an MTLBuffer (unretained: the caller keeps it alive), nullptr
    // for none; `data` where the caller's memory starts.
    KnownBuffer(const void* data, void* buffer);
    ~KnownBuffer();
    KnownBuffer(const KnownBuffer&) = delete;
    KnownBuffer& operator=(const KnownBuffer&) = delete;

private:
    const void* data_;
    void*       buffer_;
};

} // namespace metal_linalg::detail
