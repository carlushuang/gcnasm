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
And it is the closest prototype of what upstream would do, which makes it the honest way
to measure what the upstream fix would buy -- see the section below for why that fix is
three changes rather than one.

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

## The flat symmetric heap

Peer pointers are laid out as one contiguous span, so `peer(r) = flat_base + r*stride`:

```
[flat] heap 0x6fe733600000 .. 0x6fe773e00000 (1.01GB), stride 258MB, slot offset 0
[flat] peers ['0x6fe733600000', '0x6fe743800000', '0x6fe753a00000', '0x6fe763c00000']
[flat] stride uniform across all 4 ranks: True -> peer(r) = base + r*stride
[flat] self slot 0x6fe733600000 aliases torch buf 0x6fe782c00000 (distinct VA, same memory) OK
```

**This has nothing to do with fabric.** It is local address-space bookkeeping --
`hipMemAddressReserve` one span, `hipMemMap` each allocation at a computed offset -- and
works identically with POSIX fds, or with no shareable handle at all. It is listed here
because you have to build it deliberately: neither a per-peer `hipMemAddressReserve` nor
torch's own rendezvous gives you a uniform stride. Torch's, measured on 4 ranks with a
32 MiB buffer, puts peers 36 MiB apart but your own buffer 262 GB away from them:

```
rank 0: 0x71839dc00000   rank 1: 0x714375c00000   rank 2: 0x714373800000   rank 3: 0x714371400000
strides: ['-0x4028000000', '-0x2400000', '-0x2400000']   -> NOT uniform
```

Every rank is mapped into the span, including this one: our own allocation gets a **second
alias** inside the heap, so `peer(my_rank)` is an ordinary slot and the stride holds across
all ranks. The tensor's original pointer keeps working -- same physical memory, two VAs,
verified by writing through one and reading through the other.

The payoff is device-side. `fs_gather_flat` reads every rank with one base pointer and a
stride, no N-entry pointer array in kernarg, and can address a rank computed at run time:

```cpp
const uint4* src = (const uint4*)(flat_base + (size_t)r * stride + offset);
```

```
[rank 0] gather_flat: all 4 ranks via base+r*stride, 64MB OK
```

Constraints are mild: `stride` is `max(alloc_size)` rounded to the 2 MiB granularity, and
every rank must place its tensor at the same offset within its allocation (checked at
import; it is 0 in every case measured).

## Cross-process, and the road to cross-node

`main.py` runs under torchrun, so its ranks share a parent and a rendezvous. `xproc.py`
deliberately does not: two processes are started independently and meet over a TCP socket,
which is all a fabric handle needs.

```bash
python3 xproc.py serve   --gpu 0 --port 55600 --method rebind
python3 xproc.py connect --gpu 1 --port 55600 --method rebind --host <server-ip>
```

Verified on 4x gfx1250, with `own` and with `rebind`, and -- the stronger case -- between
**two separate containers**, each `pid 1` in its own namespace, sharing no launcher, no fd
table and no PID namespace:

```
[serve]   pid 1 gpu 0 method=rebind: buf 0x771ee3400000, exported 64-byte fabric handle 07145b3e...
[connect] pid 1 gpu 1 method=rebind: connected, imported peer window -> 0x766bda600000
[connect] read server's buffer: OK
[serve]   our buffer now holds the client's pattern: OK
```

Both directions are checked: the client reads the server's sentinel, writes its own pattern
back through the mapping, and the server verifies it.

This is exactly why fabric matters rather than POSIX fds. An fd is a number in one
process's descriptor table -- to hand it to another process you need `SCM_RIGHTS` over a
unix socket or `pidfd_getfd`, and neither crosses a machine boundary. A fabric handle is 64
opaque position-independent bytes you can put on any wire, which is why `xproc.py` needs
nothing more than `struct.pack` and `sendall`.

For real cross-node, only `--host` changes. Note the caveat: this test proves the software
shape on one host. Two arbitrary nodes can only map each other if their GPUs are actually
part of the same fabric domain -- the handle being portable does not by itself create a
path between them.

## The CUDA control case

[`nvidia/`](nvidia/) runs the same experiment on CUDA in **pure python** -- no C++ at all.
It is worth reading next to this one, because the conclusion is not "NVIDIA has fabric and
AMD does not". On an 8x H20 node torch also falls back to POSIX fds, but for a structural
reason that matters: `get_fabric_access()` is *real code* there, so it probes and decides
at run time, and on a fabric-capable node (IMEX configured) the identical script gets
fabric with no changes. On ROCm the function is compiled out, so no node can.

The other difference is what the fallback still buys you: on CUDA `symm_mem.rendezvous()`
plus `hdl.get_buffer(peer)` gives peer tensors in pure python, which is exactly what all
three methods here have to give up.

## Files

| file | role |
|------|------|
| `nvidia/` | the CUDA control case, pure python -- see [nvidia/README.md](nvidia/README.md) |
| `fabric_symm.hip` | all three buffer paths, export/import, capability probe, `uint4` copy kernel, timing loop; plain C ABI |
| `fabric_shim.cpp` | `LD_PRELOAD` interposer on `hipMemCreate` (method `shim` only) |
| `hip_fabric.py` | ctypes bindings, `OwnFabricBuffer`, `rebind_to_fabric`, `SymmFabricWindow` |
| `main.py` | rank driver: build the buffer by `--method`, probe, exchange, verify, benchmark |
| `xproc.py` | two independently-started processes swapping handles over TCP -- the cross-node shape |
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

## What the upstream fix actually is

Not one change but three, all in `CUDASymmetricMemory.cu`. It is worth being precise,
because `get_fabric_access()` alone is *not* sufficient -- the ROCm branches hardcode the
handle type independently of it:

```cpp
// alloc(), CUDA branch: chooses, then honours the choice
bool has_fabric_support = at::cuda::get_fabric_access(device_idx);
handle_type_ = has_fabric_support ? FABRIC_HANDLE : POSIX_FD;

// alloc(), ROCm branch: hardcoded, get_fabric_access() never consulted
#elif defined(USE_ROCM)
  handle_type_ = Expandable_Segments_Handle_Type::POSIX_FD;

// rendezvous export, ROCm branch: hardcoded, ignores the use_fabric_handle template arg
#elif defined(USE_ROCM)
  C10_CUDA_CHECK(hipMemExportToShareableHandle(
      &block_handle, block->alloc_ref->handle,
      hipMemHandleTypePosixFileDescriptor, 0));
```

So: (1) give `c10::cuda::get_fabric_access()` a ROCm implementation, (2) make the ROCm
alloc branch honour it instead of force-assigning `POSIX_FD`, and (3) make the ROCm export
honour `use_fabric_handle`.

The hard part is already written and platform-generic. `make_peer_alloc_info` is templated
on `use_fabric_handle`, and in the fabric instantiation the `IpcChannel` type collapses to
a dummy `int` -- the 64-byte handle rides the ordinary metadata path instead:

```cpp
using IpcChannelType = std::conditional_t<use_fabric_handle, int, IpcChannel>;

if constexpr (!use_fabric_handle) {
    recv_handle = ipc_channel.broadcast_fds(rank, 0, pids, exported);   // SCM_RIGHTS
} else if (use_pg) {
    recv_handle = pg_broadcast(group, dev, 0, exported);                // PG allgather
} else {
    gathered = storeExchange.all_gather(store, rank, world_size, exported);  // TCPStore
}
```

`RendezvousRequest` already carries `clique_id` and `hostname`, `validate_nvlink_fabric_support()`
already rejects groups spanning different NVLink domains, and -- importantly --
`validate_rendezvous_requests()` is already written for multi-host groups:

```cpp
// For NVL72 systems, multiple hosts can be within a single nvlink domain.
// Use (hostname, device_idx) pair to uniquely identify each allocation.
```

Hostname disambiguates (host, device) pairs rather than rejecting cross-host ranks. (The
"participants are not on the same host" abort belongs to `IntraNodeComm`, a different and
older component.)

## What using it would look like

Nothing in user code changes. There is no fabric-specific API, and that is the point -- the
handle type is chosen inside the allocator, so the same script runs on one node or eight:

```python
import torch, torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

# the PG backend does not decide the handle type -- the allocator does, via
# get_fabric_access(). gloo rendezvouses fine; a device backend only lets torch
# route the metadata exchange through the PG allgather instead of TCPStore,
# which matters at large rank counts. nccl/rccl is what a real job wants anyway.
dist.init_process_group("nccl")            # ranks may span hosts
torch.cuda.set_device(local_rank)

buf = symm_mem.empty(N, dtype=torch.bfloat16, device="cuda")
hdl = symm_mem.rendezvous(buf, dist.group.WORLD.group_name)

peer = hdl.get_buffer(r, (N,), torch.bfloat16)   # peer memory, whichever host it is on
hdl.barrier()                                     # device-side sync via the signal pad

out = torch.ops.symm_mem.one_shot_all_reduce(buf, "sum", group_name)
```

Under the hood the only differences are `handle_type_ == FABRIC_HANDLE`, the templated
rendezvous taking the `true` branch, 64-byte handles travelling through the Store or the
PG allgather rather than a Unix socket, and the clique check replacing the implicit
same-host assumption. Every one of the 34 `torch.ops.symm_mem::*` ops then works across
nodes with no call-site change.

That is the target this example is a stand-in for: `shim` and `rebind` become unnecessary,
`own` remains useful only if you do not want symm_mem semantics at all, and the flat
symmetric heap stays worth building because torch still does not give you a uniform stride.

## Notes

The transfer kernels are a plain `uint4` copy. A pipelined `tensor_load_to_lds` read path
would likely raise the read column on gfx1250 and is left as a follow-up.
