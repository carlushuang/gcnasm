# nvidia/ -- what CUDA torch does today, in pure python

The parent example needs a C++ helper because torch on ROCm cannot hand you a
fabric-exportable `symm_mem` buffer. This folder is the control case: the same experiment
on CUDA, in **pure python/torch**. `ctypes` appears only in the diagnostic; the data path
is `symm_mem.empty()`, `symm_mem.rendezvous()`, `hdl.get_buffer()` and `tensor.copy_()`.

```bash
./run.sh                                   # one rank per visible GPU, nothing to build
./run.sh --buffer-mib 512 --sizes 64,512
```

## Measured -- 8x H20, single node, NVLink (NV18 full mesh), driver 570.172.08, torch 2.9.1+cu128

```
[probe] torch symm_mem buffer @ 0xa02000000
[probe]   created with requestedHandleTypes=0x1 (POSIX_FD)
[probe]   export-fabric=FAIL 1(CUDA_ERROR_INVALID_VALUE)  export-posix_fd=ok
[probe]   -> torch fell back to POSIX-FD because its fabric gate failed:
[probe]      cuMemCreate(FABRIC) failed: 800(CUDA_ERROR_NOT_PERMITTED)
[rendezvous] ok, world_size=8, 645.6 ms
[rank 0] torch read peer 1..7: 16MB OK

      size    local GB/s     read GB/s    write GB/s   (aggregate)
       1MB        1073.8        1327.5        1541.8
      16MB       13400.2        2738.0        2876.4
      64MB       10111.3        2927.8        3067.3
     256MB       11218.7        2939.4        3124.7
SUCCESS
```

Aggregate over 8 ring pairs, so ~367 GB/s read and ~391 GB/s write per pair against an
NVLink peak of 18 x 26.562 = ~478 GB/s per direction, i.e. 77-82% of line rate through
nothing but `tensor.copy_()`.

## The three things this shows

**1. torch on CUDA does not use fabric here either -- but for a completely different
reason.** On ROCm `get_fabric_access()` is compiled out (`#if !defined(USE_ROCM)`), so no
configuration can ever produce a fabric buffer. On CUDA the function is real: it queries
NVML fabric state and then runs an actual allocate/export/import round trip. On this box
that round trip fails at `cuMemCreate(FABRIC)` with `CUDA_ERROR_NOT_PERMITTED`, because
fabric handles need the IMEX daemon and `nvidia-imex` is not installed here (this is a
single NVLink node, not an MNNVL cluster). torch sees that, falls back to POSIX fds, and
carries on.

The distinction matters: on a fabric-capable node -- GB200 NVL72 with IMEX configured --
the *identical* script would print `requestedHandleTypes=0x8 (FABRIC)` and change nothing
else. On ROCm no node can do that today.

**2. The POSIX-fd fallback is fully functional, in pure python.** `symm_mem.rendezvous()`
succeeds and `hdl.get_buffer(peer, sizes, dtype)` returns a real tensor on the peer's
memory. That is exactly what the ROCm side loses: with the parent example's `shim` or
`rebind` methods, torch's fd-based rendezvous is either broken or dangerous, so the C++
helper has to reimplement the exchange. Here there is no helper at all.

**3. torch picks the handle type; you do not.** There is no public API to request fabric,
which is why the parent example resorts to an `LD_PRELOAD` shim or a VA rebind. On CUDA
you simply do not need one -- the negotiation already happens and is correct.

## H20 gotcha: multicast

`symm_mem.rendezvous()` raises `RuntimeError: CUDA driver error: invalid argument` on this
hardware unless multicast is switched off:

```bash
export TORCH_SYMM_MEM_DISABLE_MULTICAST=1
```

`run.sh` sets it by default. H20 does not support the NVLS multicast object that torch
tries to create during rendezvous, and torch errors out instead of skipping it. Verified
with both `gloo` and `nccl` process groups: fails with multicast enabled, works with it
disabled.

## Comparing to the ROCm numbers

Do not read the bandwidth columns across the two folders as a vendor comparison. Different
interconnect (NVLink vs XGMI), different silicon, and different copy paths -- the ROCm side
uses a hand-written `uint4` kernel, this side uses `tensor.copy_()`. What is comparable is
the *mechanism*, summarised below.

| | ROCm (parent folder) | CUDA (here) |
|---|---|---|
| `get_fabric_access()` | compiled out, always false | real; NVML gate + create/export/import probe |
| handle type torch picks | POSIX-fd, always | POSIX-fd here; fabric where IMEX is configured |
| can you ask for fabric? | no public API -- needs shim or rebind | no public API, but torch already negotiates it |
| `symm_mem.rendezvous()` | works, but not on a fabric buffer | works |
| peer buffer in pure python | no, needs the C++ helper | yes, `hdl.get_buffer()` |
| lines of C++ required | ~440 | 0 |

## Files

| file | role |
|------|------|
| `symm_mem_cuda.py` | probe, rendezvous, cross-GPU verify, bandwidth -- pure python/torch |
| `run.sh` | sets `TORCH_SYMM_MEM_DISABLE_MULTICAST=1`, launches one rank per GPU |
