# warp_bitonic_sort — TVM FFI binding

This is the TVM FFI-based Python binding for the warp-level bitonic merge sort kernel.
It replaces the pybind11/torch-extension approach in `../py/` with the standalone
[apache-tvm-ffi](https://github.com/apache/tvm-ffi) PackedFunc mechanism.

## Key differences from `../py/` (pybind11 + torch extension)

| | `py/` (pybind11) | `tvmffi/` (TVM FFI) |
|---|---|---|
| Binding mechanism | pybind11 `PYBIND11_MODULE` | `TVM_FFI_DLL_EXPORT_TYPED_FUNC` |
| Tensor type | `at::Tensor` (PyTorch-only) | `tvm::ffi::Tensor` / `DLTensor*` (DLPack — framework-agnostic) |
| Build system | `setup.py` + torch `BuildExtension` | `Makefile` + hipcc/g++ |
| Python FFI dep | pybind11 + PyTorch | `pip install apache-tvm-ffi` (lightweight, ~2MB) |
| Installation | `pip install -e .` (egg) | `make` → load `.so` at runtime |

## Prerequisites

- **apache-tvm-ffi**: `pip install apache-tvm-ffi`
- **ROCm** (hipcc)
- **composable_kernel** headers
- **PyTorch** (for GPU memory allocation and testing)

## Build

```bash
# Inside docker (see ~/launch_docker.sh)
pip install apache-tvm-ffi

cd /raid0/carhuang/repo/gcnasm/warp_sort_bitonic/tvmffi
make
```

## Run test

```bash
python test/test.py
```

Expected output:
```
==================================================
Test warp_bitonic_sort (TVM FFI + PyTorch DLPack)
==================================================
  L=  2  descending  PASS
  L=  2  ascending   PASS
  ...
  L=256  descending  PASS
  L=256  ascending   PASS

All tests PASSED!
```

## Usage from Python

```python
import sys
sys.path.insert(0, "/raid0/carhuang/repo/gcnasm/warp_sort_bitonic/tvmffi")

import torch
import warp_bitonic_sort

x = torch.randn(64, device="cuda", dtype=torch.float)
y = warp_bitonic_sort.warp_bitonic_sort(x, is_descending=True)
print(y)
```

## File structure

```
tvmffi/
├── Makefile                                # build script (auto-detects tvm_ffi paths from pip)
├── README.md
├── csrc/
│   ├── tvm_api.cc                          # TVM FFI export (replaces torch_api.cpp)
│   ├── warp_bitonic_sort.hpp -> ../py      # symlink: pure C host API header
│   └── warp_bitonic_sort.hip -> ../py      # symlink: kernel source
├── warp_bitonic_sort/
│   └── __init__.py                         # Python wrapper (uses tvm_ffi.load_module)
└── test/
    └── test.py                             # test script
```

## How it works

1. **C++ side** (`csrc/tvm_api.cc`): Uses `TVM_FFI_DLL_EXPORT_TYPED_FUNC` to export
   `warp_bitonic_sort_ffi` as a C ABI symbol `__tvm_ffi_warp_bitonic_sort`. The function
   takes `tvm::ffi::Tensor` args which are DLPack-compatible.

2. **Build**: `make` compiles the HIP kernel with hipcc, the FFI wrapper with g++,
   and links against `libtvm_ffi.so` (from the pip package). No torch BuildExtension needed.

3. **Python side**: `tvm_ffi.load_module("lib.so")` loads the `.so` and discovers all
   exported symbols. `mod["warp_bitonic_sort"]` returns a callable. PyTorch tensors
   are converted to `tvm_ffi.Tensor` via `tvm_ffi.from_dlpack()` (zero-copy DLPack).
