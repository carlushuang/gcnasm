# bandwidth_memread -- GPU Memory Bandwidth Microbenchmark

A portable (ROCm / CUDA) microbenchmark for measuring **GPU global memory
bandwidth** with two kernels:

| Kernel | Mode | What it measures |
|--------|------|------------------|
| `memread_kernel` | **Read-only** `[ro]` | Peak read bandwidth (accumulate into registers, never store) |
| `memcpy_kernel` | **Read+Write** `[rw]` | Sustainable read+write bandwidth (load then store) |

Both kernels use **float4 (128-bit) vectorized** accesses, an **8x unroll**
factor, and are launched as **persistent kernels** (one workgroup per CU) to
saturate the memory subsystem.

## Key Features

- **Non-temporal loads/stores** (`__builtin_nontemporal_load/store` on HIP)
  bypass L2 cache for streaming workloads -- controlled by compile-time
  `USE_NT_LOAD` / `USE_NT_STORE` flags (both **on** by default for ROCm).
- **Persistent launch**: grid size = `num_CUs * occupancy`, each workgroup
  processes `issues_per_block` elements via a loop with unrolled inner body.
- **Configurable via environment variables**:
  - `BANDWIDTH_TEST_CASE=N` -- override CU count (0 = auto-detect, -1 = 304)
  - `BANDWIDTH_TEST_LIST=2` -- use alternative size list
- **Sweep mode** (default): runs 11 buffer sizes from ~78 KB to ~1.7 GB,
  first read-only then read+write.
- **Single-size mode**: `./bandwidth_kernel.exe <dwords>` to test one specific
  buffer size.

## Build

```bash
# ROCm (default target: native GPU, auto-detected)
make rocm

# CUDA
make cuda
```

The Makefile hipifies `bandwidth_kernel.cu` via `hipify-perl` and compiles with
`hipcc --offload-arch=native`.

## Run

```bash
# Default sweep (11 sizes, ro then rw)
./bandwidth_kernel.exe

# Single buffer size (in dwords, i.e. number of float32 elements)
./bandwidth_kernel.exe 268369920    # ~1 GB

# Use bench.sh for a predefined size sequence
bash bench.sh          # case 0: 78KB .. 3.67GB
bash bench.sh 1        # case 1: fine-grained 50KB .. 6MB sweep
```

## Example Output (MI308X, 80 CUs, gfx942)

```
cu:80, nt_load:1, nt_store:1 (0)
---------------------------------------------
  78.12KB([ro]) -> 0.0030ms, 26.557(GB/s)
   1.53MB([ro]) -> 0.0030ms, 535.330(GB/s)
  63.75MB([ro]) -> 0.0203ms, 3295.656(GB/s)
 148.75MB([ro]) -> 0.0413ms, 3777.035(GB/s)
 256.25MB([ro]) -> 0.0589ms, 4559.552(GB/s)
 320.00MB([ro]) -> 0.0737ms, 4552.372(GB/s)
 511.88MB([ro]) -> 0.1209ms, 4439.997(GB/s)
1023.75MB([ro]) -> 0.2478ms, 4332.844(GB/s)
   1.74GB([ro]) -> 0.4344ms, 4306.987(GB/s)
---------------------------------------------
  78.12KB([rw]) -> 0.0030ms, 52.631(GB/s)
   1.53MB([rw]) -> 0.0030ms, 1076.423(GB/s)
  63.75MB([rw]) -> 0.0441ms, 3032.462(GB/s)
 148.75MB([rw]) -> 0.0929ms, 3358.111(GB/s)
 256.25MB([rw]) -> 0.1633ms, 3290.869(GB/s)
 320.00MB([rw]) -> 0.2237ms, 3000.572(GB/s)
 511.88MB([rw]) -> 0.3216ms, 3337.489(GB/s)
1023.75MB([rw]) -> 0.6438ms, 3335.069(GB/s)
   1.74GB([rw]) -> 1.1347ms, 3297.922(GB/s)
```

**Peak observed**: ~4.56 TB/s read-only, ~3.36 TB/s read+write (MI308X).

## How It Works

### Kernel Design

```
Grid:   num_CUs * occupancy workgroups
Block:  1024 threads

Each thread processes:
    for i in 0..iters:
        for j in 0..UNROLL(8):    // fully unrolled
            v += nt_load(src[base + i*UNROLL*1024 + j*1024 + tid])  // float4

    if (v == magic) *dst = v;     // dead store -- prevents compiler from
                                  // optimizing away the reads
```

- **float4** loads issue 128-bit (16-byte) transactions, maximizing bus
  utilization.
- **8x unroll** gives the compiler enough independent loads to hide memory
  latency via instruction-level parallelism.
- **Non-temporal loads** (`__builtin_nontemporal_load`) tell the hardware to
  bypass L2 on the way in, avoiding cache pollution for streaming patterns.
- The **dead-store trick** (`if (v == magic) *dst = v`) prevents the compiler
  from eliding the loads while never actually writing.

### memcpy_kernel

Same structure but loads into a register array (`T tmp[UNROLL]`) and then
stores all elements with `__builtin_nontemporal_store`.  Reports bandwidth as
`bytes_read + bytes_written` (2x the buffer size).

## Files

| File | Description |
|------|-------------|
| `bandwidth_kernel.cu` | Portable CUDA/HIP source (hipified at build time) |
| `Makefile` | Build rules for ROCm and CUDA |
| `bench.sh` | Convenience script for predefined size sweeps |
