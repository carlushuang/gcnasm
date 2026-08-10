# torch_backend/ -- register a fabric SymmetricMemory backend with torch

Everything else in this directory works *around* torch. This plugs *into* it.
`c10d::symmetric_memory::register_allocator()` is a documented extension point, so a
backend implementing `SymmetricMemoryAllocator` + `SymmetricMemory` gets driven by torch's
own entry points -- and, the payoff, **torch's collectives then run on fabric memory**.

```python
import fabric_backend                    # the import registers "FABRIC"
symm_mem.set_backend("FABRIC")

buf = symm_mem.empty(N, dtype=torch.float32, device="cuda")   # -> FabricAllocator::alloc
hdl = symm_mem.rendezvous(buf, group_name)                    # -> FabricAllocator::rendezvous
peer = hdl.get_buffer(3, (N,), torch.float32)
torch.ops.symm_mem.one_shot_all_reduce(buf, "sum", group_name)
```

## It works, including the collectives

4x gfx1250, ROCm 7.15, torch 2.11:

```
backend: FABRIC
rendezvous: world_size=4 rank=0 buffer_size=16MB
flat layout: base=0x7e662a400000 stride=18MB
buffer_ptrs: ['0x7e662a400000', '0x7e662b600000', '0x7e662c800000', '0x7e662da00000']
uniform stride: True
  get_buffer(0..3): OK
SUCCESS

RESULT one_shot_all_reduce:  OK got=10.0 expect=10.0
RESULT two_shot_all_reduce_: OK got=10.0 expect=10.0
```

That is the thing none of the other three methods can do. `shim` and `rebind` produce
fabric buffers that torch's own rendezvous rejects, so `torch.ops.symm_mem::*` is off the
table for them. Here the ops dispatch through our `SymmetricMemory` object and simply
work -- **without waiting for the upstream ROCm fix**, because we bypass
`CUDASymmetricMemoryAllocator` rather than repairing it.

## What you have to implement

| interface | count | notes |
|---|---|---|
| `SymmetricMemoryAllocator` | 7 pure virtuals | `alloc`, `free`, `get_alloc_size`, `rendezvous`, `has_multicast_support`, `supported_device_type`, `name` |
| `SymmetricMemory` | 14 pure virtuals | buffer/signal-pad pointers (host and device), sizes, rank/world/device, `barrier`, `put_signal`, `wait_signal`, multicast |

Note `rendezvous` is *yours*. Registering a backend does not let you reuse torch's
rendezvous -- torch calls yours. That is why the pointer-lookup experiment in the parent
README fails: the allocator owns both halves by design.

Peers are laid out in one flat span, so `get_buffer_ptrs()[r] == flat_base + r*stride` and
`flat_layout(tensor)` exposes the pair for kernels that would rather do arithmetic than
walk `buffer_ptrs_dev`.

## Build and run

```bash
python3 setup.py build_ext --inplace
TORCH_SYMM_MEM_DISABLE_MULTICAST=1 torchrun --nnodes=1 --nproc_per_node=4 demo_backend.py
```

## Gaps and gotchas

**`enable_symm_mem_for_group()` is required, despite being deprecated.** torch warns "there
is no need to call this function anymore", which is true for its own backend but not for a
third-party one: our `rendezvous` calls `get_group_info(group_name)`, and without the
registration that throws `no group info associated with the group name 0`. Passing a
`ProcessGroup` instead of a name does not help -- torch resolves it to the same name and
still looks it up. A backend could avoid this by resolving the process group itself
instead of using `get_group_info`.

**`put_signal` / `wait_signal` are not implemented**, and `barrier` is host-side
(`hipDeviceSynchronize` plus a store round trip) rather than spinning on the signal pad.
Any op that needs real device-side signalling will fail or serialise. The signal pad is
allocated (9216 B, matching torch's, so layouts stay comparable) but unused.

**No multicast.** `has_multicast_support()` returns false, so the `multimem_*` ops are out.

**Built against torch's internal headers.** `SymmetricMemory.hpp` is `TORCH_API` with no
stability guarantee -- expect to revisit this on a torch upgrade. That is the standing cost
of this route versus the ctypes module one directory up, which only uses public HIP.

## When to prefer this over `fabric_mem.py`

Use `fabric_mem.py` if you want peer pointers for your own kernels and nothing else: 250
lines of Python, no C++, nothing to break when torch updates.

Use this backend if you want torch's symm_mem ops on fabric memory. It is a genuine third
option next to "wait for upstream" and "work around torch" -- more code, but entirely in
your control and landing in your repo rather than upstream's queue.

## Files

| file | role |
|------|------|
| `fabric_backend.hip` | the allocator and SymmetricMemory implementation, plus registration |
| `setup.py` | builds it as a torch extension |
| `demo_backend.py` | drives it purely through `symm_mem.*` |
