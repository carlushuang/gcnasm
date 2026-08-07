# torch_symm_fabric -- fabric export for torch symmetric memory, three ways

Cross-GPU transfer over **HIP fabric handles**, driven from PyTorch. The interesting part
is not the transfer -- it is getting a fabric-capable buffer in the first place, which
torch cannot give you on ROCm. This example implements all three ways to do it and
benchmarks them side by side.

## The problem

`torch.distributed._symmetric_memory` allocates through `hipMemCreate`, and on ROCm it
always asks for the POSIX-fd handle type: `c10::cuda::get_fabric_access()` in
`c10/cuda/PeerToPeerAccess.cpp` sits inside `#if !defined(USE_ROCM)` and the ROCm build
compiles to a bare `return false`.

The set of shareable handle types is frozen at `hipMemCreate` and **there is no API to
change it afterwards** -- `hipMemGetAllocationPropertiesFromHandle` only reads it back,
and the export validates against what creation recorded. So a symm_mem buffer cannot be
exported over fabric as allocated: `hipMemRetainAllocationHandle` on `t.data_ptr()`
succeeds, the fabric export then fails with `hipErrorInvalidValue`.

ROCm also rejects the combined `fd|fabric` mask with `hipErrorNotSupported`, so the two
handle types are mutually exclusive. Whatever you do, you get one or the other.

## The three methods

```bash
./run.sh --method own       # we allocate the fabric window; torch gets a tensor view
./run.sh --method shim      # LD_PRELOAD makes torch's hipMemCreate ask for fabric
./run.sh --method rebind    # torch allocates, we remap that VA onto fabric backing
./bench_methods.sh          # all three, side by side
```

Everything after the buffer exists is shared by all three: export a 64-byte POD fabric
handle, all-gather it, import every peer, then read and write peer memory.

```python
buf = <one of the three methods>                                # fabric-capable int32 buffer
buf[:n] = sentinel(n, rank, device)                             # pure torch

win = hf.SymmFabricWindow(buf, dev)                             # export over fabric
dist.all_gather_object(descriptors, win.descriptor())           # 64 bytes per rank
win.import_peers(descriptors, rank)

peer = win.get_buffer(1, (n,), torch.int32)                     # torch tensor on rank 1's memory
assert torch.equal(peer, sentinel(n, 1, device))                # the compare *is* the transfer
```

`get_buffer(peer, sizes, dtype)` mirrors torch's own rendezvous handle, so a peer buffer
reads the same either way -- it just arrives over fabric instead of POSIX fds.

## Measured -- 4x gfx1250, all-to-all XGMI, ROCm 7.15

`./bench_methods.sh --buffer-mib 256 --sizes 1,16,64,256`, aggregate over 4 ring pairs:

```
method     setup_ms    phys_MiB   local_GB/s    read_GB/s   write_GB/s   status
-------- ---------- ----------- ------------ ------------ ------------ --------
own             0.3         256        23301         5076         5571       OK
shim            0.3         258        23912         5065         5507       OK
rebind          0.3         516        23627         5064         5524       OK
```

**Bandwidth is identical within noise.** All three end up with the same kind of fabric
mapping, so the method costs nothing at run time -- it only differs in what it costs to
set up and what it constrains. Per ring pair at 256 MB that is ~1.27 TB/s read and
~1.38 TB/s write, against a ~5.9 TB/s device-local copy with the same kernel.

**Physical memory is where they separate.** `own` is exactly 1x. `shim` is 1x plus
torch's 2 MiB of signal pad and granularity rounding. `rebind` is **2.02x**, because torch
still holds a reference to the original allocation, now unmapped but not released.

Setup time does not separate them: VMM allocation is lazy, so even the rebind's extra
create/unmap/map is under a millisecond at this size.

## Pros and cons

| | 1. own | 2. shim | 3. rebind |
|---|---|---|---|
| Buffer is a real `symm_mem` tensor | no | yes | yes |
| Physical memory | **1.00x** | **1.01x** | 2.02x |
| Blast radius | none | **process-wide** | one buffer, opt-in |
| Needs `LD_PRELOAD` | no | **yes** | no |
| Opt-in visible at the call site | n/a | no | yes |
| Ordering constraint | none | none | **must precede first write** |
| If `rendezvous()` is called anyway | n/a | fails loudly | **silently wrong memory** |
| Prototypes the upstream fix | no | yes | no |

### 1. own -- allocate it ourselves

Touches no torch internals, so nothing upstream can break it, and it is exactly 1x memory.
Full control of the allocation: granularity, alignment, and `hipMemSetAccess` for several
devices at once. Works on any torch version and does not even require `symm_mem` to exist.

The buffer is not a symmetric-memory tensor, so everything torch layers on top is gone:
`rendezvous()`, signal pads, multicast, `torch.ops.symm_mem.*`, async-TP fusion. The
tensor arrives through `__cuda_array_interface__`, so it lives outside torch's caching
allocator and you own its lifetime and free ordering. It also duplicates machinery torch
already has, which is a long-term divergence cost.

### 2. shim -- interpose `hipMemCreate`

1x memory, the allocation is fabric-native from birth, and the buffer is a genuine torch
symm_mem tensor with torch's bookkeeping intact. Best failure mode of the three: there is
only ever one allocation, so `rendezvous()` fails loudly rather than returning wrong data.
And it is precisely what an upstream `get_fabric_access()` would do, which makes it the
honest way to measure what the upstream fix would buy.

But it overrides `hipMemCreate` for every caller in the process. The
`requestedHandleTypes != 0` guard spares the caching allocator and `expandable_segments`
-- in a real run only the two symm_mem buffers are touched, which
`FABRIC_SHIM_VERBOSE=1` will show you -- yet any other component that legitimately wants
an fd handle would silently lose it. `LD_PRELOAD` is deployment friction, it is sensitive
to link details (static linking, `-Bsymbolic`, a `dlopen`'d HIP would bypass it), and
nothing at the call site reveals that allocation semantics changed.

### 3. rebind -- swap the physical backing

No interposition, nothing global, opt-in per buffer and explicit at the call site. The
tensor stays a real symm_mem tensor at an unchanged pointer. Plain library call, so it
composes with any launcher or notebook, and it only uses public HIP APIs.

The cost is 2x physical memory for the buffer's lifetime. The previous contents are also
discarded, so it must run immediately after allocation -- `main.py` fills the sentinel
after the rebind for exactly this reason. And it has the worst failure mode:
`symm_mem.rendezvous()` would still export the fd of the *orphaned* allocation, so it can
succeed while handing peers stale pages. There is a brief window where the VA is unbacked,
so it is not safe against concurrent access, and it leans on an internal assumption -- one
VMM allocation per symm tensor, remappable -- that holds today but is not contractual.

### What none of them fix

All three lose torch's `rendezvous()`, and with it the collectives built on it
(`torch.ops.symm_mem.*`, async-TP). That is not an implementation accident: as long as
torch's rendezvous is fd-based and ROCm refuses `fd|fabric`, fabric and torch's collectives
are mutually exclusive. Getting both requires torch to negotiate fabric *and* run its
rendezvous over fabric handles -- the upstream change. Worth knowing before building on
symm_mem collectives over a fabric.

## HIP APIs used

Only the step that produces a fabric-capable local buffer differs:

| 1. own | 2. shim | 3. rebind |
|---|---|---|
| `hipMemGetAllocationGranularity` | *(torch allocates)* | `hipMemGetAddressRange` |
| `hipMemCreate` (fabric) | `hipMemCreate` **interposed**, rewrites `prop.requestedHandleTypes` | `hipGetDevice` |
| `hipMemAddressReserve` | `hipMemGetAddressRange` | `hipMemCreate` (fabric) |
| `hipMemMap` | `hipMemRetainAllocationHandle` | `hipMemUnmap` (torch's range) |
| `hipMemSetAccess` | | `hipMemMap` (same VA) |
| `hipMemUnmap` (teardown) | | `hipMemSetAccess` |
| `hipMemAddressFree` (teardown) | | `hipMemRelease` (ours, at once) |
| `hipMemRelease` (teardown) | | `hipMemRetainAllocationHandle` |

Shared by all three -- export, import a peer, tear a peer down:

| Export | Import | Teardown |
|---|---|---|
| `hipMemExportToShareableHandle` (`hipMemHandleTypeFabric`) | `hipMemImportFromShareableHandle` | `hipMemUnmap` |
| | `hipMemAddressReserve` | `hipMemAddressFree` |
| | `hipMemMap` | `hipMemRelease` |
| | `hipMemSetAccess` | |

Plus `hipSetDevice`, `hipGetDeviceCount`, `hipMemGetInfo`, `hipGetLastError`,
`hipGetErrorString`, `hipDeviceSynchronize`, and the functional capability probe
(`granularity -> create -> export -> import -> release`).

Notes: method 2 is the only one needing a non-HIP dependency (`dlsym(RTLD_NEXT, ...)`
from `libdl`) and the only one that writes a field in a `hipMemAllocationProp` it did not
construct -- but it is also the smallest HIP surface, a single entry point. Method 3 is
the only one that calls `hipMemUnmap` on memory it does not own; that is both its trick
and its risk. `hipMemGetAddressRange` is required by methods 2 and 3 and is the one call
here outside the documented allocate/map/export flow -- it is how you recover
`(base, size)` from a pointer torch handed you.

Types used throughout -- `hipMemGenericAllocationHandle_t`, `hipMemFabricHandle_t` (64 B),
`hipMemAllocationProp`, `hipMemAccessDesc`, `hipMemLocation` -- are ABI-stable across the
7.12/7.15 skew checked here, unlike `hipDeviceAttribute_t`.

## Build and run

```bash
./build.sh                 # arch autodetected; GPU_ARCH=gfx950 ./build.sh to override
./run.sh --method rebind
./bench_methods.sh --buffer-mib 512 --sizes 64,512
NPROC=2 ./run.sh --method own
FABRIC_SHIM_VERBOSE=1 ./run.sh --method shim   # log every allocation the shim upgrades
```

| flag | meaning |
|------|---------|
| `--method own\|shim\|rebind` | how to obtain the fabric buffer (default `rebind`) |
| `--buffer-mib N` | buffer per rank (default 256) |
| `--verify-mib N` | bytes checked in the correctness phase (default 16) |
| `--sizes a,b,c` | bandwidth sweep sizes in MiB |
| `--warmup` / `--loop` | timing iterations (default 5 / 20) |
| `--verbose` | correctness lines from every rank, not just rank 0 |

`run.sh` sets `LD_PRELOAD` itself when you ask for `--method shim`, and leaves the
environment clean otherwise.

Correctness runs before the sweep in every method: each rank reads every rank's buffer
through torch and checks a per-rank sentinel, so a wrong mapping surfaces as "peer 2 gave
rank 3's pattern" rather than as garbage.

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
exist in 7.12), so querying it across a version skew silently asks the wrong question.
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
| `fabric_symm.hip` | all three buffer paths, export/import, capability probe, `uint4` copy kernel, timing loop; plain C ABI |
| `fabric_shim.cpp` | `LD_PRELOAD` interposer on `hipMemCreate` (method `shim` only) |
| `hip_fabric.py` | ctypes bindings, `OwnFabricBuffer`, `rebind_to_fabric`, `SymmFabricWindow` |
| `main.py` | rank driver: build the buffer by `--method`, probe, exchange, verify, benchmark |
| `build.sh` | builds `libfabric_symm.so` and `libfabric_shim.so` |
| `run.sh` | builds if needed, preloads the shim when asked, launches under `torchrun` |
| `bench_methods.sh` | runs all three methods and collates the comparison table |

## Which to use

For a benchmark or an experiment, `rebind` -- smallest blast radius, and 2x memory is
irrelevant at these sizes. To argue for the upstream fix, `shim`, because it is the
faithful prototype and shows the numbers a real `get_fabric_access()` would deliver. In
production, if you do not actually need symm_mem semantics, `own` is cheapest and has no
caveats attached, since you are not fighting an allocator that wants a different handle
type.

Once `get_fabric_access()` has a ROCm implementation, torch negotiates fabric itself, both
`shim` and `rebind` become unnecessary, and the shared export/import path keeps working
unchanged.

## Notes

The transfer kernels are a plain `uint4` copy. A pipelined `tensor_load_to_lds` read path
would likely raise the read column on gfx1250 and is left as a follow-up.
