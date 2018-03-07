from __future__ import absolute_import

import os
import ctypes
from pathlib import Path


def _load_lib():
    lib_root = Path(__file__).parents[2]
    lib_path = os.path.join(lib_root, 'cuda/build/lib')

    path_to_so_file = os.path.join(lib_path, 'libc_runtime_api.so')
    lib = ctypes.CDLL(path_to_so_file, ctypes.RTLD_GLOBAL)
    return lib


try:
    _LIB = _load_lib()
except:
    pass


def check_call(ret):
    assert ret == 0


def c_array(ctype, values):
    return (ctype * len(values))(*values)
