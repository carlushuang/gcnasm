# torch_symm_fabric -- fabric export for pure-torch symmetric memory

The buffer is a plain `torch.distributed._symmetric_memory` tensor: allocated with
`symm_mem.empty()`, filled and compared with ordinary torch ops. This example adds only
the piece torch lacks on ROCm -- making that buffer exportable as a **HIP fabric handle**
so peers can map it and read it directly.

```python
buf = symm_mem.empty(numel, dtype=torch.int32, device=device)   # pure torch
hf.rebind_to_fabric(buf)                                        # this buffer, in place
buf[:n] = sentinel(n, rank, device)                             # pure torch

win = hf.SymmFabricWindow(buf, dev)                             # export over fabric
dist.all_gather_object(descriptors, win.descriptor())           # 64 bytes per rank
win.import_peers(descriptors, rank)

peer = win.get_buffer(1, (n,), torch.int32)                     # torch tensor on rank 1's memory
assert torch.equal(peer, sentinel(n, 1, device))                # the compare *is* the transfer
```

`get_buffer(peer, sizes, dtype)` mirrors torch's own rendezvous handle, so the buffer
reads the same either way -- it just reaches peers over fabric instead of POSIX fds.

## Why the rebind

`symm_mem` allocates through `hipMemCreate`, and on ROCm it always asks for the POSIX-fd
handle type: `c10::cuda::get_fabric_access()` in `c10/cuda/PeerToPeerAccess.cpp` sits
inside `#if !defined(USE_ROCM)` and the ROCm build compiles to a bare `return false`.

The set of shareable handle types is frozen at `hipMemCreate` and **there is no API to
change it afterwards** -- `hipMemGetAllocationPropertiesFromHandle` only reads it back,
and the export call validates against what creation recorded. So the buffer cannot be
exported over fabric as allocated: `hipMemRetainAllocationHandle` on `t.data_ptr()`
succeeds, the fabric export then fails with `hipErrorInvalidValue`.

But VMM separates the virtual address from its physical backing, so instead of changing
the flag you can swap the backing. `fs_rebind_fabric` allocates fabric-capable memory,
unmaps torch's range, and maps the new allocation at the exact same address:

```
hipMemGetAddressRange(ptr)  ->  base, size      # the allocation behind the tensor
hipMemCreate(fabric, size)  ->  h
hipMemUnmap(base, size)                          # briefly unbacked
hipMemMap(base, size, 0, h) ; hipMemSetAccess
hipMemRelease(h)                                 # the mapping holds its own reference
```

torch's pointer never changes and its tensor keeps working; the allocation simply became
exportable. `main.py` shows both sides of it:

```
[probe] torch symm_mem buffer @ 0x74cc9b400000
[probe]   as allocated : vmm-backed=ok  export-fabric=FAIL(hip 1)  export-posix-fd=ok
[probe]   after rebind : vmm-backed=ok  export-fabric=ok           export-posix-fd=FAIL(hip 1)
```

Nothing here intercepts or replaces a HIP entry point. An earlier version of this example
used an `LD_PRELOAD` shim on `hipMemCreate` to set the handle type at creation, which
avoided the memory cost below but overrode `hipMemCreate` for *every* caller in the
process -- including buffers that have nothing to do with fabric. The rebind is opt-in per
buffer and visible at the call site, which is worth paying for.

### What it costs

**Twice the physical memory, until torch frees the tensor.** torch still holds a reference
to the original allocation, which is now unmapped but not released, so a 256 MB buffer
occupies 512 MB:

```
free before alloc:      442030 MiB
free after 256MB alloc: 441620 MiB   (used 410)
free after rebind:      441362 MiB   (extra 258)
```

**The previous contents are discarded**, so rebind immediately after allocating and before
filling. `main.py` does the sentinel fill after the rebind for exactly this reason.

**Do not use torch's own rendezvous on a rebound buffer.** `symm_mem.rendezvous()` would
still export the fd of the original allocation, so it may *succeed* and hand peers the
stale, orphaned pages rather than the memory the tensor now points at. Use this exchange
instead -- which is the point, since an fd handle needs `SCM_RIGHTS`/`pidfd_getfd` and
stops at the host boundary, while a fabric handle is 64 opaque position-independent bytes
you can put on any wire.

## Build and run

```bash
./build.sh                 # arch autodetected; GPU_ARCH=gfx950 ./build.sh to override
./run.sh                   # one rank per visible GPU
./run.sh --buffer-mib 512 --sizes 1,16,64,256,512
NPROC=2 ./run.sh
```

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
       1MB        1109.4         738.9         744.9
      16MB       11749.6        3474.8        3406.9
      64MB       22097.5        4460.3        4677.1
     256MB       25118.6        5064.3        5506.8
```

Aggregate over 4 ring pairs, all concurrent. `local` is the same `uint4` copy kernel
staying inside one GPU, as a reference ceiling -- so per pair at 256 MB the fabric mapping
sustains ~1.27 TB/s read and ~1.38 TB/s write against a ~6.3 TB/s local copy. Small sizes
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
| `fabric_symm.hip` | rebind, export/import of a torch-owned buffer, capability probe, `uint4` copy kernel, timing loop; plain C ABI |
| `hip_fabric.py` | ctypes bindings, `rebind_to_fabric`, `SymmFabricWindow`, and the zero-copy torch view onto peer memory |
| `main.py` | rank driver: allocate in torch, rebind, probe, exchange, verify, benchmark |
| `build.sh` | builds `libfabric_symm.so` |
| `run.sh` | builds if needed, launches one rank per GPU under `torchrun` |

## Notes

The rebind exists because torch picks the handle type at allocation. The upstream fix is
for `get_fabric_access()` to have a ROCm implementation, at which point torch would
negotiate fabric itself, the rebind and its 2x memory cost would go away, and everything
else here would keep working unchanged.

The transfer kernels are a plain `uint4` copy. A pipelined `tensor_load_to_lds` read path
would likely raise the read column on gfx1250 and is left as a follow-up.
