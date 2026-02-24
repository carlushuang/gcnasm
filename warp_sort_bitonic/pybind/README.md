# warp_bitonic_sort — pure pybind11 binding

Uses [pybind11](https://github.com/pybind/pybind11) directly to bind the HIP kernel to Python.
No torch `CUDAExtension`, no `BuildExtension`, no setuptools — just pybind11 headers, hipcc, g++, and Ninja.

The C++ binding (`pybind_api.cpp`) accepts raw GPU device pointers as `int64`.
The Python wrapper (`warp_bitonic_sort/__init__.py`) handles `torch.Tensor` allocation and `data_ptr()` extraction.

## Dependencies

```bash
pip install pybind11 torch  # pybind11 for headers, torch for tensor allocation at runtime
```

## Build & Run

Compilation happens automatically on first import (JIT via Ninja).
Set `OPUS_DIR` if opus headers are not at the default path.

```bash
export OPUS_DIR=/raid0/carhuang/repo/aiter/csrc/include  # or wherever opus headers live
cd pybind
python test/test.py
```

The compiled `.so` is cached at `~/.cache/warp_bitonic_sort_pybind/`.
To force a rebuild, delete that directory.

## How it works

1. `warp_bitonic_sort/jit.py` generates a Ninja build file
2. Ninja compiles:
   - `csrc/warp_bitonic_sort.hip` with `hipcc` (the GPU kernel — identical to torch/ and tvmffi/)
   - `csrc/pybind_api.cpp` with `g++` (the pybind11 binding — lightweight headers)
3. Links into `_C<ext_suffix>.so` (a standard Python extension module)
4. `warp_bitonic_sort/__init__.py` loads the module via `importlib` and provides the Python API
