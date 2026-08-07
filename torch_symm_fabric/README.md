# torch_symm_fabric -- fabric export for pure-torch symmetric memory

The buffer is a plain `torch.distributed._symmetric_memory` tensor: allocated with
`symm_mem.empty()`, filled and compared with ordinary torch ops. This example adds only
the piece torch lacks on ROCm -- exporting that buffer as a **HIP fabric handle** so peers
can map it and read it directly.

```python
buf = symm_mem.empty(numel, dtype=torch.int32, device=device)   # pure torch
buf[:n] = sentinel(n, rank, device)                             # pure torch

win = hf.SymmFabricWindow(buf, dev)                             # export over fabric
dist.all_gather_object(descriptors, win.descriptor())           # 64 bytes per rank
win.import_peers(descriptors, rank)

peer = win.get_buffer(1, (n,), torch.int32)                     # torch tensor on rank 1's memory
assert torch.equal(peer, sentinel(n, 1, device))                # the compare *is* the transfer
```

`get_buffer(peer, sizes, dtype)` mirrors torch's own rendezvous handle, so the buffer
reads the same either way -- it just reaches peers over fabric instead of POSIX fds.

## The one bit that is missing

`torch.distributed._symmetric_memory` allocates through `hipMemCreate`, and on ROCm it
always asks for the POSIX-fd handle type: `c10::cuda::get_fabric_access()` in
`c10/cuda/PeerToPeerAccess.cpp` sits inside `#if !defined(USE_ROCM)` and the ROCm build
compiles to a bare `return false`. The handle types a VMM allocation supports are frozen
at `hipMemCreate` time, so the buffer can never be exported over fabric afterwards --
`hipMemRetainAllocationHandle` on `t.data_ptr()` succeeds, the fabric export then fails
with `hipErrorInvalidValue`.

`fabric_shim.cpp` is an `LD_PRELOAD` interposer on `hipMemCreate` that sets
`requestedHandleTypes` to `hipMemHandleTypeFabric`. Nothing else changes: torch still
allocates, still owns the mapping, still hands you a normal symm_mem tensor.

**This is a swap, not an addition.** `requestedHandleTypes` is documented as a bitmask,
but ROCm rejects the combined `fd|fabric` mask with `hipErrorNotSupported`, so the two are
mutually exclusive:

| | torch's `symm_mem.rendezvous()` | fabric export |
|---|---|---|
| stock torch | works (POSIX fd) | fails, `hipErrorInvalidValue` |
| with `libfabric_shim.so` | fails | works |

So with the shim you use *this* exchange instead of torch's -- which is the point, since
an fd handle needs `SCM_RIGHTS`/`pidfd_getfd` and stops at the host boundary, while a
fabric handle is 64 opaque position-independent bytes you can put on any wire. If your
workload calls `symm_mem.rendezvous()`, do not preload the shim.

`main.py` reports which door is open before doing anything else:

```
[probe] torch symm_mem buffer @ 0x70fc1f400000: vmm-backed=ok  export-fabric=ok  export-posix-fd=FAIL(hip 1)
[probe] fabric-exportable (libfabric_shim.so is active)
```

and without the shim it says so and points at the fix rather than failing obscurely.

## Build and run

```bash
./build.sh                 # arch autodetected; GPU_ARCH=gfx950 ./build.sh to override
./run.sh                   # one rank per visible GPU, shim preloaded
./run.sh --buffer-mib 512 --sizes 1,16,64,256,512
NPROC=2 ./run.sh
FABRIC_SHIM_VERBOSE=1 ./run.sh    # log every allocation the shim upgrades
```

`run.sh` builds on demand, exports `LD_PRELOAD=./libfabric_shim.so` (inherited by the
torchrun workers, which is where it must take effect), and launches. Flags:

| flag | meaning |
|------|---------|
| `--buffer-mib N` | `symm_mem.empty` tensor per rank (default 256) |
| `--verify-mib N` | bytes checked in the correctness phase (default 16) |
| `--sizes a,b,c` | bandwidth sweep sizes in MiB |
| `--warmup` / `--loop` | timing iterations (default 5 / 20) |
| `--verbose` | correctness lines from every rank, not just rank 0 |

## Results -- 4x gfx1250, all-to-all XGMI, ROCm 7.15

Correctness: every rank reads every rank's buffer through torch and checks a per-rank
sentinel, so a wrong mapping surfaces as "peer 2 gave rank 3's pattern" rather than as
garbage.

```
      size    local GB/s     read GB/s    write GB/s   (aggregate)
      ----    ----------     ---------    ----------
       1MB        1160.7         740.4         755.1
      16MB       11575.2        3479.0        3420.3
      64MB       22177.0        4429.7        4678.2
     256MB       23482.0        5044.6        5535.6
```

Aggregate over 4 ring pairs, all concurrent. `local` is the same `uint4` copy kernel
staying inside one GPU, as a reference ceiling -- so per pair at 256 MB the fabric mapping
sustains ~1.26 TB/s read and ~1.38 TB/s write against a ~5.9 TB/s local copy. Small sizes
are launch-latency bound, not link bound.

## Environment gotchas

These cost real debugging time; all are checked or worked around by the scripts.

**Fabric export needs ROCm >= 7.15.** On an older runtime `hipMemCreate` with the fabric
handle type *succeeds* and `hipMemExportToShareableHandle` then fails with
`hipErrorInvalidValue`. `main.py` probes this up front and names the failing step.

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
link line. `build.sh` compiles to an object and links the shared library itself with
`-l:libamdhip64.so.7`.

One more, in the code rather than the environment: a HIP call that fails leaves
`hipGetLastError()` armed, and torch checks it after its next kernel launch. The probes
here deliberately provoke failures, so they clear the error before returning -- otherwise
an unrelated `torch.arange` a few lines later dies with "CUDA error: invalid argument".

## Files

| file | role |
|------|------|
| `fabric_shim.cpp` | `LD_PRELOAD` interposer on `hipMemCreate`; makes torch's allocation fabric-capable |
| `fabric_symm.hip` | export/import of a torch-owned buffer, capability probe, `uint4` copy kernel, timing loop; plain C ABI |
| `hip_fabric.py` | ctypes bindings, `SymmFabricWindow`, and the zero-copy torch view onto peer memory |
| `main.py` | rank driver: allocate in torch, probe, exchange, verify, benchmark |
| `build.sh` | builds `libfabric_symm.so` and `libfabric_shim.so` |
| `run.sh` | builds if needed, preloads the shim, launches one rank per GPU under `torchrun` |

## Notes

The shim exists because the handle type is chosen inside torch. The upstream fix is for
`get_fabric_access()` to have a ROCm implementation, at which point torch would negotiate
fabric itself and `fabric_shim.cpp` could be deleted; everything else here would keep
working unchanged.

The transfer kernels are a plain `uint4` copy. A pipelined `tensor_load_to_lds` read path
would likely raise the read column on gfx1250 and is left as a follow-up.
