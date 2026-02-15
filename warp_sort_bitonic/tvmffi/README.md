# warp_bitonic_sort — TVM FFI binding

TVM FFI-based Python binding for the warp-level bitonic merge sort kernel.
Replaces the pybind11/torch-extension approach in `../py/` with the standalone
[apache-tvm-ffi](https://github.com/apache/tvm-ffi) PackedFunc mechanism.

The build system uses Python-driven Ninja JIT compilation with caching (no Makefile).

## Key differences from `../py/` (pybind11 + torch extension)

| | `py/` (pybind11) | `tvmffi/` (TVM FFI) |
|---|---|---|
| Binding mechanism | pybind11 `PYBIND11_MODULE` | `TVM_FFI_DLL_EXPORT_TYPED_FUNC` |
| Tensor type | `at::Tensor` (PyTorch-only) | `tvm::ffi::Tensor` / `DLTensor*` (DLPack) |
| Build system | `setup.py` + torch `BuildExtension` | Python-driven Ninja JIT |
| Python FFI dep | pybind11 + PyTorch | `pip install apache-tvm-ffi` (~2MB) |
| Installation | `pip install -e .` | Zero-install: JIT compiles on first `import` |
| Caching | egg in `build/` | `~/.cache/warp_bitonic_sort/jit/` |

## Prerequisites

- **apache-tvm-ffi**: `pip install apache-tvm-ffi`
- **ROCm** (hipcc)
- **ninja**: `pip install ninja` or system package
- **composable_kernel** headers
- **PyTorch** (for GPU memory allocation)

## Usage

```bash
# Inside docker (see ~/launch_docker.sh)
pip install apache-tvm-ffi

# First import triggers JIT compilation (cached for subsequent runs)
cd /raid0/carhuang/repo/gcnasm/warp_sort_bitonic/tvmffi
python test/test.py
```

```python
import sys
sys.path.insert(0, "/raid0/carhuang/repo/gcnasm/warp_sort_bitonic/tvmffi")

import torch
import warp_bitonic_sort

x = torch.randn(64, device="cuda", dtype=torch.float)
y = warp_bitonic_sort.warp_bitonic_sort(x, is_descending=True)
```

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `ROCM_PATH` | `/opt/rocm` | ROCm installation path |
| `CK_DIR` | `/raid0/carhuang/repo/composable_kernel` | composable_kernel headers |
| `GPU_ARCH` | `native` | Target GPU architecture |
| `WARP_SORT_CACHE_DIR` | `~/.cache/warp_bitonic_sort` | JIT cache directory |
| `WARP_SORT_JIT_VERBOSE` | `0` | Set to `1` for verbose build output |
| `MAX_JOBS` | (auto) | Max parallel ninja jobs |

## File structure

```
tvmffi/
├── README.md
├── csrc/
│   ├── tvm_api.cc                      # TVM FFI export
│   ├── warp_bitonic_sort.hpp -> ../py  # symlink: C host API header
│   └── warp_bitonic_sort.hip -> ../py  # symlink: HIP kernel source
├── warp_bitonic_sort/
│   ├── __init__.py                     # Public API (lazy JIT on first call)
│   └── jit.py                          # Ninja build generation + JitSpec
└── test/
    └── test.py
```

## How it works (JIT compilation)

1. **First `import warp_bitonic_sort`** — the module is loaded but nothing is compiled yet.

2. **First call to `warp_bitonic_sort()`** — triggers `gen_sort_module().build_and_load()`:
   - `jit.py` generates a `build.ninja` file under `~/.cache/warp_bitonic_sort/jit/`
   - Runs `ninja` to compile the HIP kernel with `hipcc` and the FFI wrapper with `g++`
   - Links into a `.so` against `libtvm_ffi.so` (from pip)
   - Loads the `.so` via `tvm_ffi.load_module()` and caches the result

3. **Subsequent calls** — the cached `.so` is found and loaded directly (no recompilation).

4. **C++ side** (`csrc/tvm_api.cc`): Uses `TVM_FFI_DLL_EXPORT_TYPED_FUNC` to export the
   function. PyTorch tensors are passed via DLPack zero-copy.
