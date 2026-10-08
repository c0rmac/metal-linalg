"""The library's C API (include/metal_linalg/c_api.h) through ctypes.

The shared library ships inside this package. It holds the buffer core alone:
Metal, Metal Performance Shaders and Accelerate, no MLX and no torch, so this
package is built once for every torch and every Python. Calls into it release
the GIL (ctypes does), and are serialised by one lock, since the core's
per-shape GPU workspaces are not shared safely between threads.
"""

import ctypes
import os
import threading

_HERE = os.path.dirname(os.path.abspath(__file__))
_PATH = os.environ.get("METAL_LINALG_TORCH_LIBRARY") or os.path.join(_HERE, "libmetal_linalg.dylib")

try:
    lib = ctypes.CDLL(_PATH)
except OSError as e:   # pragma: no cover - a broken install
    raise ImportError(f"metal_linalg_torch: could not load {_PATH}: {e}. The package needs an "
                      f"Apple Silicon Mac with macOS 14 or later; reinstall it with "
                      f"`pip install --force-reinstall metal-linalg-torch`.") from e

lock = threading.Lock()

_u32, _f32p, _u32p, _cstr = ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p

OK, INVALID_ARGUMENT, RUNTIME_ERROR, OUT_OF_MEMORY = 0, 1, 2, 3
NO_LIMIT = 0xFFFFFFFF


def _fn(name, restype, *argtypes):
    f = getattr(lib, name)
    f.restype = restype
    f.argtypes = list(argtypes)
    return f


last_error = _fn("metal_linalg_last_error", _cstr)
_qr = _fn("metal_linalg_qr_with_mode", ctypes.c_int, _f32p, _u32, _u32, _u32, ctypes.c_int, _f32p, _f32p)
_eigh = _fn("metal_linalg_eigh", ctypes.c_int, _f32p, _u32, _u32, ctypes.c_int, _f32p, _f32p, _u32p)
_svd = _fn("metal_linalg_svd", ctypes.c_int, _f32p, _u32, _u32, _u32, _f32p, _f32p, _f32p, _u32p)
buffer_contents = _fn("metal_linalg_buffer_contents", ctypes.c_void_p,
                      ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64)
know_buffer = _fn("metal_linalg_know_buffer", ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p)
forget_buffer = _fn("metal_linalg_forget_buffer", None, ctypes.c_void_p, ctypes.c_void_p)

device_name = _fn("metal_linalg_device_name", _cstr)
gpu_core_count = _fn("metal_linalg_gpu_core_count", _u32)
cpu_threads = _fn("metal_linalg_cpu_threads", _u32)
set_cpu_threads = _fn("metal_linalg_set_cpu_threads", None, _u32)
set_calibration_notices = _fn("metal_linalg_set_calibration_notices", None, ctypes.c_int)
calibration_message = _fn("metal_linalg_calibration_message", _cstr, _cstr)

qr_backend = _fn("metal_linalg_qr_backend", _cstr, _u32, _u32, _u32)
eigh_backend = _fn("metal_linalg_eigh_backend", _cstr, _u32, _u32)
eigvalsh_backend = _fn("metal_linalg_eigvalsh_backend", _cstr, _u32, _u32)
svd_backend = _fn("metal_linalg_svd_backend", _cstr, _u32, _u32, _u32)
svdvals_backend = _fn("metal_linalg_svdvals_backend", _cstr, _u32, _u32, _u32)

qr_policy_source = _fn("metal_linalg_qr_policy_source", _cstr)
eigh_policy_source = _fn("metal_linalg_eigh_policy_source", _cstr)
svd_policy_source = _fn("metal_linalg_svd_policy_source", _cstr)


def _struct(name, fields):
    return type(name, (ctypes.Structure,), {"_fields_": [(f, _u32) for f in fields]})


# Field for field as in c_api.h, which says what each does.
QR_FIELDS = ("m_crossover_small_batch", "m_crossover_large_batch", "batch_threshold",
             "gpu_max_k", "gpu_min_batch_times_k", "gpu_min_batch", "gpu_cores", "concurrent_matrices",
             "gpu_large_min_k", "gpu_large_max_batch", "gpu_min_k", "share_min_batch")
EIGH_FIELDS = ("simd_max_n", "block_min_n", "block_min_n_batched", "block_min_batch",
               "gpu_max_n", "gpu_min_batch_times_n", "gpu_min_batch", "gpu_cores",
               "values_gpu_max_n", "values_gpu_min_batch_times_n", "values_gpu_min_batch",
               "tridiag_min_n", "values_tridiag_min_n", "ql_min_n", "ql_max_n",
               "tridiag_max_batch", "values_tridiag_max_batch", "share_min_batch",
               "gpu_big_batch_max_n", "gpu_big_batch_min", "values_band_min_n", "values_band_width")
SVD_FIELDS = ("qr_min_rows", "qr_min_k", "block_min_k", "block_min_k_batched", "block_min_batch",
              "gpu_max_k", "gpu_min_batch_times_k", "gpu_min_batch", "gpu_cores",
              "bidiag_min_k", "values_bidiag_min_k", "bidiag_max_batch", "values_bidiag_max_batch",
              "gk_min_k", "gk_max_k", "gpu_max_l",
              "values_gpu_max_k", "values_gpu_min_batch_times_k", "values_gpu_min_batch", "values_gpu_max_l",
              "share_min_batch", "gpu_big_batch_max_k", "gpu_big_batch_min", "values_band_min_k",
              "values_band_width", "band_min_k")
# Read back but ignored when set.
INFORMATIONAL = {"gpu_cores", "concurrent_matrices"}

POLICIES = {}
for _what, _fields in (("qr", QR_FIELDS), ("eigh", EIGH_FIELDS), ("svd", SVD_FIELDS)):
    _S = _struct(f"{_what}_policy", _fields)
    POLICIES[_what] = (_S, _fields, _fn(f"metal_linalg_{_what}_policy_get", _S),
                       _fn(f"metal_linalg_{_what}_policy_set", None, ctypes.POINTER(_S)))


def get_policy(what):
    _, fields, get, _ = POLICIES[what]
    with lock:
        p = get()
    return {f: getattr(p, f) for f in fields}


def set_policy(what, values):
    S, fields, get, set_ = POLICIES[what]
    unknown = set(values) - set(fields)
    if unknown:
        raise TypeError(f"unknown {what} policy field(s): {', '.join(sorted(unknown))}; "
                        f"the fields are {', '.join(fields)}")
    with lock:
        p = get()
        for f, v in values.items():
            if f not in INFORMATIONAL:
                setattr(p, f, int(v))
        set_(ctypes.byref(p))


def check(status, what):
    if status == OK:
        return
    msg = (last_error() or b"").decode() or "unknown error"
    if status == INVALID_ARGUMENT:
        raise ValueError(f"metal_linalg_torch.{what}: {msg}")
    if status == OUT_OF_MEMORY:
        raise MemoryError(f"metal_linalg_torch.{what}: {msg}")
    raise RuntimeError(f"metal_linalg_torch.{what}: {msg}")


QR_MODES = {"reduced": 0, "r": 1, "complete": 2}   # metal_linalg_qr_with_mode's


def qr(a, batch, rows, cols, q, r, mode="reduced"):
    with lock:
        check(_qr(a, batch, rows, cols, QR_MODES[mode], q, r), "qr")


def eigh(a, batch, n, lower, w, v):
    with lock:
        check(_eigh(a, batch, n, 1 if lower else 0, w, v, None), "eigh")


def svd(a, batch, rows, cols, u, s, vt):
    with lock:
        check(_svd(a, batch, rows, cols, u, s, vt, None), "svd")


def text(s):
    return (s or b"").decode()
