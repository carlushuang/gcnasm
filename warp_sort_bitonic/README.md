## bitonic merge sort
this example implement warp level bitonic merge sort. The `med3` instruction, combined with `DPP`/`SHFL` makes it ultra fast and instruction saving to implement sorting on AMD GPU.

![image](res/bm-sort.png)

## build/run
```
sh rebuild.sh # build
./build/warp_sort.exe
```
will have output like:
```
[WARP SORT BITONIC]____________________________________________
[origin-2]2.986 3.929
[sorted-2]3.929 2.986
--------------------------------------------------------------- ordered[>][y]
[origin-4]2.857 8.086 3.743 6.629
[sorted-4]8.086 6.629 3.743 2.857
--------------------------------------------------------------- ordered[>][y]
[origin-8]8.571 1.943 1.114 2.157 1.971 7.057 2.586 4.486
[sorted-8]8.571 7.057 4.486 2.586 2.157 1.971 1.943 1.114
--------------------------------------------------------------- ordered[>][y]
[origin-16]6.271 8.200 5.786 1.657 3.086 7.714 7.500 6.057 3.971 5.957 0.743 7.114 8.314 5.557 6.857 4.657
[sorted-16]8.314 8.200 7.714 7.500 7.114 6.857 6.271 6.057 5.957 5.786 5.557 4.657 3.971 3.086 1.657 0.743
--------------------------------------------------------------- ordered[>][y]
[origin-32]8.271 5.500 5.229 2.557 1.471 5.414 5.629 1.457 3.800 6.729 3.600 2.214 1.657 2.629 6.700 4.371 ...
[sorted-32]8.471 8.271 8.057 7.843 7.029 7.000 6.729 6.714 6.700 6.029 5.629 5.500 5.414 5.229 5.029 4.943 ...
--------------------------------------------------------------- ordered[>][y]
[origin-64]3.686 8.371 5.714 5.143 1.657 2.757 6.586 5.457 5.943 6.643 7.657 7.586 0.686 2.229 3.386 7.957 ...
[sorted-64]8.529 8.371 7.957 7.657 7.586 7.257 7.229 7.071 7.043 6.957 6.929 6.643 6.586 6.529 6.457 6.443 ...
--------------------------------------------------------------- ordered[>][y]
[origin-2]4.614 1.800
[sorted-2]1.800 4.614
--------------------------------------------------------------- ordered[<][y]
[origin-4]6.400 5.943 5.071 3.757
[sorted-4]3.757 5.071 5.943 6.400
--------------------------------------------------------------- ordered[<][y]
[origin-8]4.614 6.400 1.686 2.300 1.314 5.371 4.843 0.643
[sorted-8]0.643 1.314 1.686 2.300 4.614 4.843 5.371 6.400
--------------------------------------------------------------- ordered[<][y]
[origin-16]4.529 8.200 1.129 5.729 1.643 8.543 6.800 8.157 5.729 1.114 5.557 8.271 7.557 5.543 7.229 0.543
[sorted-16]0.543 1.114 1.129 1.643 4.529 5.543 5.557 5.729 5.729 6.800 7.229 7.557 8.157 8.200 8.271 8.543
--------------------------------------------------------------- ordered[<][y]
[origin-32]7.900 8.286 7.357 2.186 5.643 3.843 2.386 1.686 6.686 4.057 0.429 4.443 5.871 5.271 5.086 1.829 ...
[sorted-32]0.429 1.014 1.343 1.686 1.829 2.186 2.214 2.329 2.386 2.629 2.657 2.986 3.000 3.843 4.000 4.057 ...
--------------------------------------------------------------- ordered[<][y]
[origin-64]7.871 3.086 0.957 4.943 3.371 3.329 3.071 6.500 7.386 8.514 2.371 4.671 1.657 7.443 2.943 3.000 ...
[sorted-64]0.014 0.357 0.586 0.671 0.700 0.886 0.886 0.957 0.971 1.014 1.071 1.100 1.100 1.300 1.343 1.386 ...
--------------------------------------------------------------- ordered[<][y]

```

this example rely on [opus](https://github.com/ROCm/aiter/) (`opus.hpp`), please modify `OPUS_DIR` inside `rebuild.sh` before build

## python bindings

There are three ways to call this kernel from python:

### 1. `torch/` — torch extension + pybind11
Uses `CUDAExtension` from PyTorch with pybind11 to bind the kernel.
Requires PyTorch at build time and runtime. See [`torch/README.md`](torch/README.md).

```bash
cd torch && python3 setup.py develop
```

### 2. `pybind/` — pure pybind11 + Ninja JIT
Uses [pybind11](https://github.com/pybind/pybind11) directly with a custom Ninja JIT build.
No torch `BuildExtension` — only pybind11 headers, hipcc, and g++.
See [`pybind/README.md`](pybind/README.md).

```bash
pip install pybind11
cd pybind && python test/test.py
```

### 3. `tvmffi/` — TVM FFI binding
Uses [apache-tvm-ffi](https://github.com/apache/tvm-ffi) (`pip install apache-tvm-ffi`) to bind the kernel via the DLPack protocol. The C++ side uses `TVM_FFI_DLL_EXPORT_TYPED_FUNC` — no pybind11, no torch `BuildExtension`. See [`tvmffi/README.md`](tvmffi/README.md).

```bash
pip install apache-tvm-ffi
cd tvmffi && python test/test.py
```

### 4. `pyhip/` — ctypes, no C++ binding at all
Uses Python's built-in `ctypes.CDLL` to load the `.hip` kernel compiled into a plain `.so`. **No C++ binding file** — no pybind11, no torch extension, no TVM. The `extern "C"` host launcher is called directly from Python.

```bash
cd pyhip && python test/test.py
```

### 5. `pyhip_v2/` — Python-side `hipLaunchKernel`, no `<<<>>>` in C++
Same as pyhip but the C++ code has **zero `<<<>>>`** syntax. The `.so` exports a function-pointer table; Python reads it and calls `hipLaunchKernel` from `libamdhip64.so` directly via ctypes.

```bash
cd pyhip_v2 && python test/test.py
```

### 6. `pyhip_v3/` — `--genco` device-only, `hipModuleLaunchKernel`
Compiled with `hipcc --genco` to produce a `.hsaco` code object (device binary only — zero host code generated). Python loads it with `hipModuleLoad` and launches kernels with `hipModuleLaunchKernel` (HIP driver API). The `hip_minimal.h` has **zero host declarations** — no `dim3`, no `hipLaunchKernel`. `extern "C" __global__` wrappers give kernels predictable symbol names.

```bash
cd pyhip_v3 && python test/test.py
```

## Compile-Time Benchmark

Benchmarked on ROCm 7.1.1 (hipcc 7.1, AMD clang 20.0), PyTorch 2.9.1+rocm7.1.1, pybind11, apache-tvm-ffi, Docker image `rocm/atom:nightly_202601190317`. 3 runs each, clean builds.

### End-to-End Results

| Approach | Compile Time | Speedup vs torch |
|---|---|---|
| **torch/** (`setup.py build_ext`) | **21.1s** | 1.0x |
| **pybind/** (pybind11 + Ninja JIT) | **4.2s** | 5.0x |
| **tvmffi/** (TVM FFI + Ninja JIT) | **3.5s** | 6.0x |
| **pyhip/** (hipcc -shared, `<<<>>>` in C++) | **420ms** | 50.3x |
| **pyhip_v2/** (hipcc -shared, Python `hipLaunchKernel`) | **410ms** | 51.6x |
| **pyhip_v3/** (hipcc --genco, `hipModuleLaunchKernel`) | **346ms** | **61.1x** |

### Kernel-Only Compile Breakdown

| Kernel variant | Compile | Preprocessed lines |
|---|---|---|
| torch/pybind/tvmffi (`hip_runtime.h`, all original headers) | 1,729ms | 190,582 |
| pyhip (`hip_minimal.h` + compiler builtins) | 406ms | 11,480 |
| pyhip_v2 (`hip_minimal.h`, no `<<<>>>`) | 381ms | 11,467 |
| pyhip_v3 (`--genco`, device-only) | 347ms | 17,431 |

## Deep Dive: How pyhip → pyhip_v3 Achieves 61x Faster Compile

This section documents the systematic journey from a 21s torch build to a 346ms device-only compile — a 61x improvement — by progressively eliminating every layer of unnecessary overhead.

### Step 1: Eliminate the C++ binding layer (torch → pyhip)

The first three approaches (torch, pybind, tvmffi) all require a **separate C++ binding file** compiled alongside the kernel. This binding file pulls in heavy framework headers:

| Binding | Compiler | Headers | Binding compile time |
|---|---|---|---|
| torch (`torch_api.cpp`) | hipcc | torch, pybind11, ATen, c10 | ~8.2s |
| pybind (`pybind_api.cpp`) | g++ | pybind11 | ~4.0s |
| tvmffi (`tvm_api.cc`) | g++ | tvm_ffi | ~1.1s |
| pyhip (none) | — | — | **0s** |

**pyhip eliminates binding compilation entirely** by marking the host launcher `extern "C"` and calling it via `ctypes.CDLL`. No pybind11, no torch extension, no TVM — just `hipcc -shared` and Python's built-in FFI.

### Step 2: Replace `hip_runtime.h` with minimal declarations + compiler builtins

The standard `<hip/hip_runtime.h>` pulls in massive header trees. pyhip replaces it with a ~60-line `hip_minimal.h` and direct AMDGCN compiler builtins:

| Symbol | Standard HIP | pyhip replacement |
|---|---|---|
| `threadIdx.x` | `<hip/hip_runtime.h>` | `__builtin_amdgcn_workitem_id_x()` |
| `__syncthreads()` | `<hip/hip_runtime.h>` | `__builtin_amdgcn_s_barrier()` |
| `warpSize` | `<hip/hip_runtime.h>` | `__builtin_amdgcn_wavefrontsize()` |
| `__shfl()` | `<hip/hip_runtime.h>` | Custom impl via `__builtin_amdgcn_ds_bpermute()` |
| `dim3`, `<<<>>>` | `<hip/hip_runtime.h>` | Minimal struct + 3 function declarations |

This reduces preprocessed lines from **190K → 11K** (17x) and kernel compile time from **1.7s → 0.4s**.

### Step 3: Suppress implicit heavy includes with `-D__HIPCC_RTC__`

Even with `hip_minimal.h`, hipcc's implicit `__clang_hip_runtime_wrapper.h` pulls in C++ standard library headers (`<cmath>`, `<cstdlib>`, etc.). The `-D__HIPCC_RTC__` flag tells this wrapper to skip those includes (it's the flag used by HIP's runtime compilation path). This requires providing `#define INFINITY __builtin_huge_valf()` since `<cmath>` is no longer included.

### Step 4: Guard device code with `__HIP_DEVICE_COMPILE__`

hipcc compiles `.hip` files in **two passes** — once for the host (x86_64) and once for the device (AMDGPU). The heavy `opus.hpp` template library is only needed on the device side. Wrapping it in `#ifdef __HIP_DEVICE_COMPILE__` with an empty kernel stub for the host pass avoids parsing it twice.

### Step 5: Move kernel launch from C++ to Python (pyhip_v2)

pyhip still uses `<<<>>>` in C++ to launch kernels. pyhip_v2 eliminates this:

- C++ exports a table of kernel function pointers (one per template instantiation)
- Python reads the table via ctypes and calls `hipLaunchKernel()` from `libamdhip64.so` directly

**Caveat**: hipcc still generates host stubs for `__global__` functions internally, so `dim3`/`hipLaunchKernel` declarations are still needed in `hip_minimal.h` — even though user code never writes `<<<>>>`.

### Step 6: Device-only compilation with `--genco` (pyhip_v3)

pyhip_v3 eliminates the host pass entirely:

- `hipcc --genco` compiles **only device code** into a `.hsaco` code object — no host stubs, no linker, no `.so`
- `extern "C" __global__` wrapper functions give each template instantiation a predictable symbol name
- Python loads the `.hsaco` with `hipModuleLoad` and launches with `hipModuleLaunchKernel` (HIP driver API)
- `hip_minimal.h` is truly minimal (35 lines) — **zero host declarations** (no `dim3`, no `hipLaunchKernel`, no `__hipPushCallConfiguration`)

### Summary: progressive optimization

| Step | Change | Compile time | Speedup |
|---|---|---|---|
| torch | baseline (setup.py + hipcc + ninja) | 21.1s | 1.0x |
| pybind | replace torch binding with pybind11 | 4.2s | 5.0x |
| tvmffi | replace pybind11 with tvm_ffi | 3.5s | 6.0x |
| pyhip | eliminate binding entirely (ctypes) + `hip_minimal.h` + builtins + `__HIPCC_RTC__` + device guard | 420ms | 50.3x |
| pyhip_v2 | move `<<<>>>` to Python (`hipLaunchKernel`) | 410ms | 51.6x |
| pyhip_v3 | `--genco` device-only compile (`hipModuleLaunchKernel`) | **346ms** | **61.1x** |

### Key Takeaways

1. **The C++ binding layer is the biggest cost**. Replacing torch's `CUDAExtension` with direct ctypes eliminates both binding compilation and Python/setuptools overhead — a 50x improvement over torch.

2. **Header bloat dominates kernel compile time**. Standard `<hip/hip_runtime.h>` expands to 190K preprocessed lines. A 60-line `hip_minimal.h` with compiler builtins achieves the same functionality at 11K lines — cutting kernel compile from 1.7s to 0.4s.

3. **`-D__HIPCC_RTC__` is a critical flag**. It prevents `__clang_hip_runtime_wrapper.h` from pulling in heavy C++ standard library headers that the kernel never uses.

4. **`--genco` eliminates the host pass**. For kernels launched from Python, there's no need for host stubs at all. Device-only compilation saves ~60ms (347ms vs 406ms) and removes the need for any host-side HIP declarations in C++.

5. **The irreducible floor is device codegen**. At 346ms, the remaining time is spent on AMDGPU backend code generation for 16 kernel instantiations and `opus.hpp` template expansion. This is the true minimum for this kernel's complexity.
