"""Prints the nanobind ABI tag mlx.core was built with, e.g. v21_system_libcpp_abi1.

An extension can exchange mx.array objects with MLX only if it is built with a
nanobind that produces the same tag under the "mlx" domain; python/CMakeLists.txt
compares the two before building.
"""
import builtins
import ctypes

import mlx.core  # noqa: F401  registers MLX's nanobind internals

PREFIX, SUFFIX = "__nb_internals_", "_mlx__"


def keys():
    # nanobind 2.x keeps its internals in the interpreter-state dict, 1.x in builtins.
    get = ctypes.pythonapi.PyInterpreterState_Get
    get.restype = ctypes.c_void_p
    getdict = ctypes.pythonapi.PyInterpreterState_GetDict
    getdict.restype = ctypes.py_object
    getdict.argtypes = [ctypes.c_void_p]
    return list(getdict(get()).keys()) + list(vars(builtins))


tags = [str(k)[len(PREFIX):-len(SUFFIX)] for k in keys()
        if str(k).startswith(PREFIX) and str(k).endswith(SUFFIX)]
print(tags[0] if tags else "")
