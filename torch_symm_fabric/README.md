# torch_symm_fabric -- cross-GPU transfer over HIP fabric handles, driven from PyTorch

A symmetric memory window built directly on the **HIP virtual memory management API**
with `hipMemHandleTypeFabric`, wrapped zero-copy as **torch tensors**, and used to move
data between GPUs.

Each rank owns one GPU and:

1. creates one physical allocation via `hipMemCreate` with
   `prop.requestedHandleTypes = hipMemHandleTypeFabric`, reserves VA and maps it
   (`hipMemAddressReserve` / `hipMemMap` / `hipMemSetAccess`);
2. exports it as a **64-byte POD fabric handle** (`hipMemExportToShareableHandle`);
3. aliases the mapping as a `torch.Tensor` with **no copy**, via
   `__cuda_array_interface__`, so ordinary torch ops write the exact bytes peers see;
4. all-gathers the handles over `torch.distributed` (gloo -- it is 64 bytes per rank)
   and imports every peer's window (`hipMemImportFromShareableHandle`);
5. reads and writes peer windows with a plain `uint4` copy kernel.

After step 4 a peer's buffer is just a device pointer, so any kernel can load and store
through it. That is the same shape `torch.distributed._symmetric_memory` gives you, but
built on fabric handles, which -- unlike file-descriptor handles -- are position-independent
PODs you can put on a wire.

## Why not just use `torch.distributed._symmetric_memory`?

Because on ROCm it cannot give you a fabric-exportable buffer. In
`c10/cuda/PeerToPeerAccess.cpp`, `get_fabric_access()` is wrapped in
`#if !defined(USE_ROCM) ...  #else return false; #endif`, so a ROCm build always
allocates symmetric memory with the POSIX-fd handle type.

The example prints this at startup rather than asserting it, by retaining the allocation
handle behind a real `symm_mem.empty()` tensor and trying both export types:

```
[probe] torch symm_mem buffer @ 0x73127a800000: vmm-backed=ok  export-fabric=FAIL(hip 1)  export-posix-fd=ok
[probe] -> torch cannot export this buffer over fabric on ROCm; this example allocates its own fabric-capable window instead
```

So the buffer *is* HIP-VMM backed -- `hipMemRetainAllocationHandle` on `t.data_ptr()`
succeeds -- it just was not created with the fabric handle type. Hence step 1 above:
allocate the window ourselves, then hand it to torch instead of the other way round.

## Build and run

```bash
./build.sh                 # arch autodetected; GPU_ARCH=gfx950 ./build.sh to override
./run.sh                   # one rank per visible GPU
./run.sh --window-mib 512 --sizes 1,16,64,256,512
NPROC=2 ./run.sh           # fewer ranks
```

`run.sh` builds on demand and launches through `torchrun`. Useful flags:

| flag | meaning |
|------|---------|
| `--window-mib N` | symmetric window per rank (default 256) |
| `--verify-mib N` | bytes checked in the correctness phase (default 16) |
| `--sizes a,b,c` | bandwidth sweep sizes in MiB |
| `--warmup` / `--loop` | timing iterations (default 5 / 20) |
| `--no-probe` | skip the torch symm_mem probe |
| `--verbose` | correctness lines from every rank, not just rank 0 |

## Results -- 4x gfx1250, all-to-all XGMI, ROCm 7.15

Correctness: every rank reads every rank's window and checks a per-rank sentinel, so a
wrong mapping shows up as "read peer 2 got rank 3's pattern" rather than as garbage.

```
      size    local GB/s     read GB/s    write GB/s   (aggregate)
      ----    ----------     ---------    ----------
       1MB        1166.1         734.2         731.7
      16MB       11641.9        3386.7        3470.8
      64MB       22471.5        4321.6        4724.4
     256MB       24519.3        5069.0        5553.0
```

Aggregate over 4 ring pairs (rank -> rank+1), all pairs concurrent. `local` is the same
copy kernel staying inside one GPU, as a reference ceiling -- so per pair at 256 MB the
fabric mapping sustains ~1.27 TB/s read and ~1.39 TB/s write against a ~6.1 TB/s local
copy. Small sizes are launch-latency bound, not link bound.

## Environment gotchas

These cost real debugging time; all three are checked or worked around by the scripts.

**Fabric export needs ROCm >= 7.15.** On an older runtime `hipMemCreate` with the fabric
handle type *succeeds* and `hipMemExportToShareableHandle` then fails with
`hipErrorInvalidValue`. `main.py` probes this up front and reports the failing step.

**The HIP runtime torch loads is often not `/opt/rocm`.** A pip `rocm-sdk` install puts a
second copy under `_rocm_sdk_core`, and torch loads that one even when `ldd` points at
`/opt/rocm`. Check what is actually mapped:

```bash
python3 -c "import torch,re;torch.cuda.init();print(sorted(set(re.findall(r'\S*libamdhip64\S*',open('/proc/self/maps').read()))))"
```

Build against whatever that prints. This is also why `fabric_symm.hip` never calls
`hipDeviceGetAttribute` for capability checks: `hipDeviceAttribute_t` is a long sequential
enum whose values shift between releases (the fabric attribute is 95 in 7.15 and does not
exist in 7.12), so querying it across a version skew silently asks the wrong question. The
VMM structs and the enums this example does use carry explicit values and are ABI-stable.
Fabric support is probed functionally instead: allocate, export, import, release.

**The rocm-sdk wheels omit the `libamdhip64.so` dev symlink**, shipping only
`libamdhip64.so.7`, and `hipcc --hip-link` puts that missing absolute path straight on the
link line. `build.sh` detects this, compiles to an object, and links the shared library
itself with `-l:libamdhip64.so.7`.

One more, in the code rather than the environment: a HIP call that fails leaves
`hipGetLastError()` armed, and torch checks it after its next kernel launch. Both probes
here deliberately provoke failures, so they clear the error before returning -- otherwise
an unrelated `torch.arange` a few lines later dies with "CUDA error: invalid argument".

## Files

| file | role |
|------|------|
| `fabric_symm.hip` | HIP VMM/fabric calls, capability probe, `uint4` copy kernel, timing loop; plain C ABI |
| `hip_fabric.py` | ctypes bindings, `FabricWindow`, and the zero-copy `tensor_from_ptr` wrapper |
| `main.py` | rank driver: probe, allocate, exchange, verify, benchmark |
| `build.sh` | builds `libfabric_symm.so` |
| `run.sh` | builds if needed, launches one rank per GPU under `torchrun` |

## Notes

Handle exchange goes over `torch.distributed` here because it is already available, but
nothing about the window depends on it -- fabric handles are 64 opaque bytes, so a socket,
MPI, or a key-value store works equally well. That is the property that makes this approach
extend past one node, whereas file-descriptor handles need `SCM_RIGHTS` or `pidfd_getfd`
and stop at the host boundary.
