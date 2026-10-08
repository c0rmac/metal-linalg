# The C API

`<metal_linalg/c_api.h>` is the library on plain float buffers, behind C
types, for any language that can call C: C itself, Swift (the Swift package
is built on it), Rust, Zig, Python's `ctypes`. It needs no MLX; it is the
same core the MLX API is built on (`<metal_linalg/core.h>` in C++).

```c
#include <metal_linalg/c_api.h>
#include <stdio.h>

int main(void) {
    /* Two symmetric 2 x 2 matrices, row-major, one after the other. */
    const float a[8] = {2, 1, 1, 2,   4, 0, 0, 1};
    float w[4], v[8];
    uint32_t info[2];
    metal_linalg_status st = metal_linalg_eigh(a, 2, 2, /*lower*/ 1, w, v, info);
    if (st != METAL_LINALG_OK) {
        fprintf(stderr, "eigh failed: %s\n", metal_linalg_last_error());
        return 1;
    }
    printf("%g %g | %g %g\n", w[0], w[1], w[2], w[3]);   /* 1 3 | 1 4 */
    return 0;
}
```

```bash
cc -std=c99 main.c -I/opt/homebrew/include -L/opt/homebrew/lib -lmetal_linalg -o main
```

## Conventions

- **Layout.** Matrices are float32, row-major and contiguous; a batch is its
  matrices one after another, `batch * rows * cols` floats.
- **Outputs.** The caller allocates them, at the sizes in the header. Memory
  that starts on a page boundary is read by the GPU in place; anything else is
  copied once.
- **Optional outputs.** `v` for eigh, and `u` with `vt` for the SVD, may be
  `NULL` for the values alone, which is cheaper; so may every `info`.
- **Errors.** Every decomposition returns a `metal_linalg_status`:
  `METAL_LINALG_INVALID_ARGUMENT` for a `NULL` input or a shape a backend
  cannot take, `METAL_LINALG_RUNTIME_ERROR` for a GPU failure or a finite
  matrix that did not converge (which the Jacobi methods do not do in
  practice). `metal_linalg_last_error()` has the message, per thread.
- **Non-finite input** is not an error: a matrix holding a NaN or an infinity
  gives NaN results, and the rest of its batch is unaffected. Its `info`
  word has `METAL_LINALG_INFO_NONFINITE` set.
- **Threads.** As for the C++ API, call it from one thread at a time.

## Functions

| function | computes |
|---|---|
| `metal_linalg_qr(a, batch, rows, cols, q, r)` | A = QR, `q` [batch, rows, K], `r` [batch, K, cols], K = min(rows, cols) |
| `metal_linalg_eigh(a, batch, n, lower, w, v, info)` | A = V diag(w) Vᵀ, `w` ascending; one triangle read |
| `metal_linalg_svd(a, batch, rows, cols, u, s, vt, info)` | thin A = U diag(s) Vt, `s` descending |
| `metal_linalg_device_name()`, `metal_linalg_gpu_core_count()` | the GPU the routing was resolved for |
| `metal_linalg_cpu_threads()`, `metal_linalg_set_cpu_threads(n)` | how many cores the CPU paths spread a batch over: every core by default, `0` restores that |
| `metal_linalg_qr_backend(rows, cols, batch)`, `_eigh_backend(n, batch)`, `_eigvalsh_backend(n, batch)`, `_svd_backend(rows, cols, batch)`, `_svdvals_backend(rows, cols, batch)` | the backend a call of that shape uses, by name |
| `metal_linalg_{qr,eigh,svd}_policy_get()`, `_set(&p)`, `_source()` | the routing policies; see [tuning](tuning.md) |
| `metal_linalg_set_calibration_notices(enabled)`, `metal_linalg_calibration_message(what)` | the notice printed when this Mac's measurements are missing or not current: off, or as a string to report another way |
| `metal_linalg_buffer_contents(buffer, offset, bytes)` | the CPU address of a range of a Metal buffer in shared storage, to pass a GPU framework's tensor memory in place (below) |
| `metal_linalg_know_buffer(contents, buffer)`, `metal_linalg_forget_buffer(contents, buffer)` | for the calls between them, the GPU backends use `buffer` for memory starting at `contents` rather than wrapping it in a new buffer (below) |

Each call is routed exactly as in C++: to the fastest Metal kernel for its
shape and batch, or to LAPACK on the CPU, by the policy measured for this Mac.

## Memory a GPU framework owns

A tensor of a GPU framework (PyTorch's MPS, say) lives in a Metal buffer. On
Apple Silicon that buffer is usually in shared storage, which the CPU can
read and write, so the decompositions can take their input from it and write
their outputs into it with no copy: `metal_linalg_buffer_contents` turns the
buffer (an `id<MTLBuffer>` passed as a pointer) and a byte range into a CPU
address, and returns `NULL` for a buffer in private storage, for a range the
buffer does not hold, and for an object that is not a buffer. The
decompositions run on their own command queue, so the framework's queued work
must be finished first (`torch.mps.synchronize()`); they return when their
results are written. This is how the PyTorch package passes MPS tensors.

```c
/* buf: an id<MTLBuffer> holding the input at byte offset off; w_buf the output. */
float* a = metal_linalg_buffer_contents(buf, off, (uint64_t)batch * n * n * sizeof(float));
float* w = metal_linalg_buffer_contents(w_buf, 0, (uint64_t)batch * n * sizeof(float));
if (a && w) metal_linalg_eigh(a, batch, n, 1, w, NULL, NULL);
else        /* private storage: copy to host memory and back instead */;
```

Each call wraps the memory it is given in a Metal buffer of its own, and
the first command buffer using that buffer maps its pages for the GPU (about
1 ms for 64 MB on an M5 Pro, a third of a large batch's QR). Where the
memory is a buffer's from its start, register the buffer itself for the
call instead; the GPU backends then use it. `metal_linalg_know_buffer`
returns 1 if it took the buffer (in shared storage, its contents at
`contents`), 0 if not, when there is nothing to forget:

```c
int ka = metal_linalg_know_buffer(a, buf);      /* buf's contents start at a */
int kw = metal_linalg_know_buffer(w, w_buf);
metal_linalg_eigh(a, batch, n, 1, w, NULL, NULL);
if (ka) metal_linalg_forget_buffer(a, buf);
if (kw) metal_linalg_forget_buffer(w, w_buf);
```

The PyTorch package does this for its MPS tensors.

