# pyhip/ — Launch HIP kernel from Python with no C++ binding

This example launches the warp bitonic sort kernel **directly from Python**
using `ctypes.CDLL` — no pybind11, no torch extension, no TVM FFI, and
**no C++ binding file** at all. The `.hip` kernel is compiled into a plain
`.so` and loaded via Python's built-in `ctypes`.

## Quick start

```bash
cd pyhip && python test/test.py
```

## How it works

1. **JIT compile** the `.hip` file into a shared library:
   ```
   hipcc -shared -fPIC --offload-arch=native -O3 warp_bitonic_sort.hip -o libwarp_bitonic_sort.so
   ```
2. **Load** via `ctypes.CDLL("libwarp_bitonic_sort.so")`
3. **Declare** the C signature with `ctypes` argtypes/restype
4. **Call** directly: `lib.warp_bitonic_sort_kernel(x.data_ptr(), y.data_ptr(), n, desc)`

The kernel's `extern "C"` host function receives raw device pointers —
`torch.Tensor.data_ptr()` provides exactly that.

## Why this works

The `.hip` kernel file contains a C-linkage host launcher:

```cpp
extern "C"
void warp_bitonic_sort_kernel(void* i_ptr, void* o_ptr, int num_element, int is_descending);
```

This function handles grid/block dispatch and calls `hipLaunchKernelGGL`
internally. From Python's perspective, it's just a C function in a `.so`
that takes a pointer + two ints — `ctypes` handles this natively.

## Alternative approaches (no C++ binding)

Below is a summary of **all viable ways** to launch a HIP kernel from
Python without writing any C++ binding code:

### Option A: ctypes + pre-compiled .so (this example)

**How**: Compile `.hip` → `.so` with `hipcc -shared`. Load with `ctypes.CDLL`.
Call `extern "C"` host function directly.

| Pros | Cons |
|---|---|
| Zero dependencies beyond hipcc + Python stdlib | Still need hipcc to compile |
| No binding headers (pybind11/torch/tvm) at all | Must declare C signatures in Python |
| Fastest compile (no binding .cpp to compile) | Only works for `extern "C"` functions |
| Works with any Python (no version constraints) | Torch only needed for tensor allocation |

**Best for**: Pre-compiled kernels with a clean C host API.

### Option B: hip-python + HIPRTC (runtime compilation)

**How**: `pip install hip-python` (from ROCm). Use `hip.hiprtc` to compile
kernel source strings at runtime, then `hip.hip.hipModuleLaunchKernel`
to launch.

```python
from hip import hip, hiprtc

# Compile kernel source string
prog = hiprtc.hiprtcCreateProgram(kernel_src, "kernel.hip", [], [])
hiprtc.hiprtcCompileProgram(prog, ["--offload-arch=gfx942"])
code = hiprtc.hiprtcGetCode(prog)

# Load and launch
module = hip.hipModuleLoadData(code)
func = hip.hipModuleGetFunction(module, "my_kernel")
hip.hipModuleLaunchKernel(func, gridDim, 1, 1, blockDim, 1, 1, 0, None, args)
```

| Pros | Cons |
|---|---|
| Pure Python, no compilation step | hip-python requires ROCm SDK to build/install |
| JIT compile at runtime | Kernel must be self-contained (no #include of complex headers) |
| Can generate kernels dynamically | HIPRTC has limited C++ support (no templates across TUs) |

**Best for**: Simple kernels written as strings. Not practical for this
kernel (depends on `opus.hpp` templates).

### Option C: ctypes wrapping libhiprtc.so + libamdhip64.so directly

**How**: Same as Option B but without `hip-python`. Use `ctypes` to call
`hiprtcCreateProgram`, `hiprtcCompileProgram`, `hipModuleLoadData`,
`hipModuleLaunchKernel` etc. directly from `libhiprtc.so`/`libamdhip64.so`.

| Pros | Cons |
|---|---|
| No pip install needed (just ROCm libs) | Tedious: must declare dozens of HIP API signatures in ctypes |
| Full control over compilation | Same HIPRTC limitations as Option B |
| Educational | Fragile across ROCm versions |

**Best for**: Environments where `hip-python` can't be installed but ROCm
libs are available.

### Option D: cffi (C Foreign Function Interface)

**How**: Same as Option A but using `cffi` instead of `ctypes`. Declare the
C prototype in `cffi.FFI().cdef(...)`, load the `.so` with `ffi.dlopen()`.

```python
import cffi
ffi = cffi.FFI()
ffi.cdef("void warp_bitonic_sort_kernel(void* i, void* o, int n, int desc);")
lib = ffi.dlopen("./libwarp_bitonic_sort.so")
lib.warp_bitonic_sort_kernel(ffi.cast("void*", ptr), ...)
```

| Pros | Cons |
|---|---|
| Slightly cleaner API than ctypes | Requires `pip install cffi` |
| Supports ABI mode (no compiler needed) | Essentially same as ctypes for this use case |

**Best for**: Projects already using cffi.

### Recommendation

| Kernel type | Recommended approach |
|---|---|
| Complex C++ kernel with templates / headers | **Option A** (ctypes + pre-compiled .so) |
| Simple kernel expressible as a string | **Option B** (hip-python + HIPRTC) |
| No extra pip packages allowed | **Option A** or **Option C** |

For this warp_bitonic_sort kernel (uses `opus.hpp` templates, DPP
intrinsics, `constexpr if`), **Option A is the only practical choice** —
HIPRTC cannot handle the template-heavy opus headers.
