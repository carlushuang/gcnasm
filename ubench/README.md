# AMDGPU Memory Micro-Benchmarks

Handwritten GCN assembly micro-benchmarks for measuring memory subsystem
latency and throughput on AMD Instinct MI308X (gfx942). Latency kernels use
**pointer chasing** (Sattolo single-cycle permutation); throughput kernels use
**unrolled independent operations** with all 64 wavefront lanes active.
All use `s_memrealtime` timing with loop-overhead subtraction.

## Quick Start

```bash
# Inside a ROCm docker container:
cd ubench && bash run.sh

# Or launch docker automatically:
bash run_docker.sh
```

## Directory Layout

```
ubench/
├── README.md
├── build.sh              # Top-level build (assembles + compiles everything)
├── run.sh                # Build + run all tests
├── run_docker.sh         # Launch ROCm docker and run
├── common/               # Shared assembly kernels
│   ├── nop_loop.s        #   Loop-overhead baseline (LDS-style & global-style)
│   └── global_load_latency.s  #   Pointer-chase kernel for global memory
├── mem_latency/          # Test 1: Overall memory hierarchy latency
│   ├── lds_latency.s     #   LDS pointer-chase kernel
│   └── mem_latency.cpp   #   Host: LDS / L1 / L2 / HBM latency
├── lds_detailed/         # Test 2: LDS per-instruction latency
│   ├── lds_detailed.s    #   6 kernels: ds_read/write × b32/b64/b128
│   └── lds_detailed.cpp  #   Host: measure each instruction variant
├── cacheline_stride/     # Test 3: Cache line size analysis
│   └── cacheline_stride.cpp  # Host: L1/L2 latency vs access stride
└── lds_throughput/       # Test 4: LDS throughput (dwords/cycle)
    ├── lds_throughput.s  #   7 kernels: read/write × b32/b64/b128 + NOP
    └── lds_throughput.cpp #  Host: measure throughput per instruction
```

## Benchmarks

### 1. `mem_latency` — Overall Memory Hierarchy

Measures pointer-chase latency at each level of the memory hierarchy by
varying the working-set size:

| Memory Level | Working Set | Technique |
|---|---|---|
| LDS | 1 KB | `ds_read_b32` chase in Local Data Share |
| L1 Cache | 16 KB | `global_load_dword` chase (fits in 32 KB L1) |
| L2 Cache | 256 KB | `global_load_dword` chase (exceeds L1, fits in L2) |
| HBM | 512 MB | `global_load_dword` chase (exceeds 256 MB L2) |

### 2. `lds_detailed` — LDS Per-Instruction Latency

Measures latency for each LDS instruction width:

- **Reads:** `ds_read_b32`, `ds_read_b64`, `ds_read_b128` (pointer chase)
- **Writes:** `ds_write_b32`, `ds_write_b64`, `ds_write_b128` (write + wait)

### 3. `cacheline_stride` — Cache Line Size Analysis

Measures L1/L2 load latency while sweeping the pointer-chase stride from
4 B to 256 B. A jump in latency at stride = N would indicate the cache
line size is N bytes. (Reuses `global_load_latency.hsaco`.)

### 4. `lds_throughput` — LDS Throughput (Dwords/Cycle)

Measures sustained LDS throughput for each instruction width under
bank-conflict-free conditions:

- **All 64 lanes** of the wavefront issue LDS operations simultaneously
- **Unrolled independent operations** (32 for b32, 16 for b64, 8 for b128)
  with no data dependency between them, saturating the LDS pipeline
- **Bank-conflict-free** addressing: lane *i* accesses offset *i* × stride

## Measurement Methodology

1. **Pointer chasing** — Sattolo's algorithm generates a random single-cycle
   permutation. Each array element stores the byte offset of the next element.
   The kernel serially follows the chain: `addr = mem[addr]`. This defeats
   hardware prefetchers and forces one cache miss per access (for working sets
   exceeding the target cache level).

2. **Timing** — `s_memrealtime` reads a 64-bit GPU reference clock (~100 MHz
   on gfx942). The host calibrates the tick frequency empirically using HIP
   event timing, then converts to nanoseconds and shader cycles.

3. **Loop-overhead subtraction** — Matched NOP loops (same VALU/SALU
   instruction mix, no memory access) measure the loop control overhead.
   This is subtracted from raw measurements to isolate memory latency.

4. **Single lane** (latency tests) — Only lane 0 of a single wavefront
   (block=64, grid=1) executes. This eliminates contention and bank conflicts.

5. **Full wavefront** (throughput tests) — All 64 lanes issue independent
   LDS operations with bank-conflict-free addressing. Unrolled instructions
   saturate the LDS pipeline.

## Results (MI308X, gfx942 @ 1420 MHz)

### Overall Memory Latency

```
  Memory Level     |       ns | Shader cycles | Working Set
  -----------------+----------+---------------+------------
  LDS              |       22 |            31 | 1 KB
  L1 Cache         |       70 |           100 | 16 KB
  L2 Cache         |      205 |           291 | 256 KB
  Global (HBM)     |      438 |           622 | 512 MB
```

### LDS Instruction Latency

```
  Instruction      |       ns | Shader cycles
  -----------------+----------+--------------
  ds_read_b32      |       23 |           32
  ds_read_b64      |       31 |           44
  ds_read_b128     |       34 |           48
  ds_write_b32     |       28 |           39
  ds_write_b64     |       34 |           48
  ds_write_b128    |       42 |           60
```

### Cache Line Stride (L1, 16 KB)

```
  Stride  | Shader cycles
  --------+--------------
     4 B  |           99
     8 B  |          104
    16 B  |          103
    32 B  |          103
    64 B  |          103
   128 B  |           99
   256 B  |          102
```

L1 latency is flat across all strides — the 16 KB working set fits entirely
in L1, so every access hits regardless of stride.

### Cache Line Stride (L2, 256 KB)

```
  Stride  | Shader cycles
  --------+--------------
     4 B  |          316
     8 B  |          313
    16 B  |          309
    32 B  |          316
    64 B  |          323
   128 B  |          325
   256 B  |          252
```

L2 latency is flat (309–325 cycles) for strides 4–128 B. The drop at 256 B
is due to the reduced node count (1024) fitting more of the active set in
L1. The pointer-chase pattern touches a new random cache line each iteration,
so spatial locality within a cache line is not exercised.

### LDS Throughput (single wavefront, bank-conflict-free)

```
  Instruction      | Dwords/cycle | Bytes/cycle | Cycles/inst
  -----------------+--------------+-------------+------------
  ds_read_b32      |          9.7 |        38.9 |        6.6
  ds_read_b64      |         11.0 |        43.8 |       11.7
  ds_read_b128     |         11.1 |        44.5 |       23.0
  ds_write_b32     |          6.9 |        27.5 |        9.3
  ds_write_b64     |          8.6 |        34.4 |       14.9
  ds_write_b128    |          8.2 |        32.8 |       31.3
```

Single-wavefront throughput is ~10–11 dwords/cycle for reads and ~7–9
dwords/cycle for writes. The LDS has 32 banks × 4 B = 128 B/cycle peak
bandwidth, but a single wavefront cannot fully saturate the pipeline due
to LDS access latency (~32 shader cycles). Multiple concurrent wavefronts
are needed to reach peak bandwidth.

## Build Requirements

- ROCm with `/opt/rocm/llvm/bin/clang++` and `/opt/rocm/bin/hipcc`
- Target: `gfx942` (MI300 series)
- Docker image used: `rocm/atom:nightly_202601190317`
