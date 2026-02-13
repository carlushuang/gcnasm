# Vector Add -- Hand-Written GCN Assembly for gfx942 (MI300)

A minimal but complete example of a **hand-written AMDGPU assembly kernel** that
performs `C[i] = A[i] + B[i]` on float32 arrays, using the
**`buffer_load_dword ... offen lds`** instruction to load data directly from
global memory into LDS, bypassing VGPRs entirely.

The kernel demonstrates two key GPU programming techniques:
- **Persistent kernel**: grid size = number of CUs (detected at runtime); each
  workgroup processes multiple chunks via a grid-stride loop.
- **Double LDS buffering**: two LDS buffers are used in ping-pong fashion so
  that the next iteration's data is prefetched while the current iteration is
  being computed.

## Files

| File | Description |
|---|---|
| `vector_add_kernel.s` | GCN assembly kernel (gfx942, CDNA3) |
| `main.cpp` | HIP host code -- loads `.hsaco`, launches kernel, verifies results |
| `build.sh` | Assembles the `.s` and compiles the host into a single executable |

## Build & Run

Must be run **inside a ROCm docker container** with gfx942 hardware available:

```bash
# Launch docker (adjust image as needed)
docker run -it --privileged --network=host \
    --device=/dev/kfd --device=/dev/dri --group-add video \
    -v /home/$USER:/dockerx -v /mnt/raid0:/raid0 \
    rocm/atom:nightly_202601190317

# Inside docker
cd /raid0/<path>/gcnasm/vector_add_asm
bash build.sh
./vector_add_asm.exe
```

Build steps performed by `build.sh`:

1. **Assemble** `.s` to `.hsaco` code object:
   `clang++ -x assembler -target amdgcn--amdhsa -mcpu=gfx942 vector_add_kernel.s -o vector_add_kernel.hsaco`
2. **Compile** host code:
   `hipcc main.cpp -o vector_add_asm.exe`

## Data Flow (single iteration)

```
Global Memory (A,B)
        |
        |  buffer_load_dword ... offen lds   (VGPR bypassed)
        v
       LDS
        |
        |  ds_read_b32
        v
      VGPRs
        |
        |  v_add_f32
        v
      VGPRs
        |
        |  global_store_dword
        v
Global Memory (C)
```

## Double LDS Buffer Pipeline (steady state)

```
Time ─────────────────────────────────────────────────────────►

Iteration i                          Iteration i+1
├───────────────────────────────────┤├──────────────────────────────────────┤

  ┌─ prefetch A[i+1],B[i+1] ─────────┐
  │  into ALTERNATE LDS buffer        │  (buffer_load...lds, async)
  │                                   │
  │  ┌─ compute from CURRENT buffer ──┤
  │  │  ds_read A[i], B[i]           │
  │  │  v_add_f32                    │
  │  │  global_store C[i]            │
  │  └───────────────────────────────┘
  │                                   │
  └── s_waitcnt vmcnt(0) ────────────┘
                                       swap buffers (buf0 ↔ buf1)
                                       ┌─ prefetch A[i+2],B[i+2] ──────────┐
                                       │  into ALTERNATE LDS buffer         │
                                       │                                    │
                                       │  ┌─ compute from CURRENT buffer ───┤
                                       │  │  ds_read A[i+1], B[i+1]        │
                                       │  │  v_add_f32                      │
                                       │  │  global_store C[i+1]           │
                                       │  └────────────────────────────────┘
                                       └── s_waitcnt vmcnt(0) ─────────────┘
```

The key insight: the `buffer_load ... lds` for iteration i+1 is issued
**before** the `ds_read` for iteration i. Since the buffer load goes to a
different LDS region than the one being read, the two operations run in parallel,
hiding global memory latency behind useful compute.

---

## Important Points & Lessons Learned

### 1. gfx942 global instruction addressing uses a single VGPR offset

On gfx942 (CDNA3), when a scalar base address (`saddr`) is provided,
`global_load_dword` / `global_store_dword` take a **single VGPR** as the 32-bit
signed byte offset -- not a VGPR pair like older GFX9 targets (gfx900/gfx906):

```asm
; gfx942 -- single VGPR offset
global_load_dword  v_dst, v_offset, s[base:base+1]

; gfx900 -- VGPR pair (only lower half used as offset)
global_load_dword  v_dst, v[offset:offset+1], s[base:base+1]
```

### 2. Code object v5 metadata is required

gfx942 with ROCm 6.x+ requires **`amdhsa.version: [1, 2]`** (code object v5).
Using the older `[1, 0]` (v3) format causes `invalid HSA metadata` assembler
errors.  Key differences from v3:

- `.value_type` and `.is_const` fields are no longer valid in kernel arg metadata.
- Fields like `.reqd_workgroup_size` are optional.

### 3. `buffer_load_dword ... lds` -- the assembler does not expose it

The LLVM MC assembler (as of ROCm 7.1 / clang 20) **does not accept** the `lds`
text modifier on `buffer_load_dword` for any GFX9/CDNA target.  The instruction
encoding exists in hardware and the *disassembler* understands it, but the
*assembler* rejects it.

**Workaround**: emit the 64-bit MUBUF encoding manually via `.long`, setting
**bit 16** (the LDS bit).  A clean macro makes this readable:

```asm
.macro buffer_load_dword_offen_lds vdata, vaddr, srsrc_base
    ; DWORD 0: MUBUF major opcode (0x38), OP=0x14 (buffer_load_dword),
    ;          LDS=1 (bit 16), OFFEN=1 (bit 12)
    .long 0xE0511000
    ; DWORD 1: SOFFSET=0x80 (literal 0), SRSRC, VDATA, VADDR
    .long (0x80 << 24) | ((\srsrc_base / 4) << 16) | (\vdata << 8) | \vaddr
.endm
```

The encoding `0xE0511000` was derived by assembling a regular
`buffer_load_dword ... offen` (which gives `0xE0501000`) and flipping bit 16.
The disassembler confirms: `buffer_load_dword v2, s[16:19], 0 offen lds`.

An alternative instruction that **is** accepted by the assembler is
`global_load_lds_dword`, but it uses the global (flat) memory path rather than
the buffer path.

### 4. `buffer_load ... lds` -- how LDS addressing works

When the LDS bit is set, loaded data bypasses VGPRs and goes directly to LDS:

- **M0 register** holds the per-wave LDS byte base address.
- Each lane writes to LDS at address **`M0 + lane_id * sizeof(element)`**.
- M0 must be set (via `s_mov_b32 m0, ...`) **before** each `buffer_load ... lds`.
- Use `v_readfirstlane_b32` to extract the wave's first lane threadIdx into an
  SGPR, then shift left by 2 (multiply by `sizeof(float)`) to get M0.

```asm
v_readfirstlane_b32 s_tmp, v_threadIdx    ; wave_id * 64
s_lshl_b32         s_tmp, s_tmp, 2        ; * sizeof(float)
s_mov_b32          m0, s_tmp              ; set LDS write base
buffer_load_dword_offen_lds ...           ; data -> LDS[M0 + lane*4]
```

### 5. Buffer Resource Descriptor (SRD) construction for gfx942

Buffer instructions require a 128-bit SRD in 4 consecutive SGPRs:

| Word | Value | Meaning |
|------|-------|---------|
| 0 | `base_address[31:0]` | Low 32 bits of pointer |
| 1 | `base_address[47:32]` | High 16 bits; upper bits zero for stride=0 |
| 2 | `0xFFFFFFFF` | `num_records` = max (no bounds check) |
| 3 | `0x00020000` | gfx942 config: `DATA_FORMAT=32` |

Word 3 value `0x00020000` matches the
[opus library](https://github.com/ROCm/aiter)`s `buffer_default_config()` for
gfx942/gfx90a.  For `buffer_load_dword` (unformatted), only `DATA_FORMAT`
matters; `DST_SEL` and `NUM_FORMAT` are ignored.

Word 1 must mask off bits [31:16] to zero the stride and swizzle fields:
```asm
s_and_b32 s[srd+1], s[ptr_hi], 0xFFFF
```

### 6. `.amdhsa_accum_offset` is mandatory on CDNA

gfx90a / gfx940 / gfx942 have a **unified VGPR/AGPR register file**.  The
`.amdhsa_accum_offset` directive tells the allocator where accumulator registers
(AGPRs) begin.  If no AGPRs are used, set it equal to `.amdhsa_next_free_vgpr`:

```asm
.amdhsa_next_free_vgpr 8
.amdhsa_accum_offset   8      ; no AGPRs used
```

Omitting this directive causes an assembler error on CDNA targets.

### 7. SGPR ordering: user SGPRs first, then system SGPRs

The AMDHSA calling convention places registers in this order:

| Register | Source | Enabled by |
|----------|--------|------------|
| `s[0:1]` | kernarg segment pointer | `.amdhsa_user_sgpr_kernarg_segment_ptr 1` |
| `s2` | workgroup_id_x | `.amdhsa_system_sgpr_workgroup_id_x 1` |
| `v0` | workitem_id_x | `.amdhsa_system_vgpr_workitem_id 0` |

If you also enable `dispatch_ptr` (2 SGPRs), it occupies `s[0:1]` and pushes
`kernarg_ptr` to `s[2:3]` and `workgroup_id_x` to `s4`.

### 8. Kernel arguments and host-side struct packing

The kernel argument layout in the `.s` file must **exactly** match the
`__attribute__((packed))` struct on the host side.  Every pointer is 8 bytes,
every `int`/`uint32_t` is 4 bytes, and total size must match
`.kernarg_segment_size` in the metadata.  An alignment mismatch silently
produces wrong results.

```cpp
struct __attribute__((packed)) {
    float*   A;        // offset 0
    float*   B;        // offset 8
    float*   C;        // offset 16
    uint32_t N;        // offset 24  -- number of elements
    uint32_t stride;   // offset 28  -- num_CUs * 256 (grid-stride step)
} args;
```

### 9. Host loads `.hsaco` at runtime via `hipModule*` APIs

Since the kernel is a standalone code object (not compiled into the host
binary), it is loaded at runtime:

```cpp
hipModuleLoad(&module, "vector_add_kernel.hsaco");
hipModuleGetFunction(&kernel_func, module, "vector_add_kernel");
hipModuleLaunchKernel(kernel_func, gdx,1,1, bdx,1,1, 0, 0, NULL, (void**)&config);
```

The `config` array uses `HIP_LAUNCH_PARAM_BUFFER_POINTER` /
`HIP_LAUNCH_PARAM_BUFFER_SIZE` / `HIP_LAUNCH_PARAM_END` to pass the raw
argument buffer.

### 10. Synchronization: `vmcnt` vs `lgkmcnt`

Two independent wait counters control memory ordering:

| Counter | Tracks | Wait instruction |
|---------|--------|------------------|
| `vmcnt` | Global/buffer memory ops (loads, stores, **buffer_load...lds**) | `s_waitcnt vmcnt(0)` |
| `lgkmcnt` | LDS and scalar memory (kernarg loads, `ds_read`, `ds_write`) | `s_waitcnt lgkmcnt(0)` |

For the async-to-LDS pattern, the sequence is:

```asm
buffer_load_dword ... lds       ; global -> LDS  (increments vmcnt)
s_waitcnt vmcnt(0)              ; ensure LDS is written

ds_read_b32 ...                 ; LDS -> VGPR   (increments lgkmcnt)
s_waitcnt lgkmcnt(0)            ; ensure VGPRs are ready
```

No `s_barrier` is needed because each wave only reads its own LDS region.

### 11. Persistent kernel -- grid = number of CUs

Instead of launching one workgroup per chunk of 256 elements, the host launches
exactly **`num_CUs`** workgroups (one per Compute Unit, detected at runtime via
`hipGetDeviceProperties`).  Each workgroup then processes all its elements via a
**grid-stride loop**:

```
idx = global_id                        // initial element index
stride = num_CUs * 256                 // passed as kernel arg
for (; idx < N; idx += stride) {
    C[idx] = A[idx] + B[idx]
}
```

Benefits:
- Avoids kernel launch overhead for large N (one launch covers everything).
- Every CU is occupied for the entire kernel duration, maximizing utilization.
- Works correctly for any N, including N < total threads (extra lanes are masked).

On the host side this is straightforward:

```cpp
int num_cu = props.multiProcessorCount;  // e.g. 304 on MI300X
int bdx = 256;
int gdx = num_cu;                        // persistent: 1 workgroup per CU
uint32_t stride = gdx * bdx;
```

### 12. Double LDS buffering -- overlapping load with compute

The kernel uses **two LDS buffers** in a ping-pong arrangement so that the
async `buffer_load ... lds` for the **next** iteration overlaps with
the `ds_read` + `v_add_f32` + `global_store` of the **current** iteration.

#### LDS layout (4096 bytes total)

```
Byte offset    Contents
──────────────────────────────────────
[   0, 1024)   Buffer 0 -- A values   (256 threads x 4 bytes)
[1024, 2048)   Buffer 0 -- B values
[2048, 3072)   Buffer 1 -- A values
[3072, 4096)   Buffer 1 -- B values
```

Each buffer holds one workgroup's worth of A and B values (256 floats each =
1024 bytes).

#### Four M0 values (pre-computed, loop-invariant)

The `buffer_load ... lds` instruction writes to LDS at address
`M0 + lane_id * 4`.  Since there are 4 distinct LDS regions, four M0 values
are pre-computed once in SGPRs before the loop:

```asm
s_m0_buf0_a = wave_lds_base + 0       ; buffer 0, A region
s_m0_buf0_b = wave_lds_base + 1024    ; buffer 0, B region
s_m0_buf1_a = wave_lds_base + 2048    ; buffer 1, A region
s_m0_buf1_b = wave_lds_base + 3072    ; buffer 1, B region
```

Where `wave_lds_base = first_lane_threadIdx * 4`.

#### Unrolled ping-pong loop structure

The main loop is **unrolled by 2**: `L_process_buf0` and `L_process_buf1`.
Each half reads from one buffer and prefetches into the other, using fixed
`ds_read` offsets and `m0` values -- no dynamic buffer swapping needed:

```
PROLOGUE:
    load first batch -> buf0
    wait

L_process_buf0:                         L_process_buf1:
    prefetch next -> buf1                   prefetch next -> buf0
    ds_read from buf0 (offset 0, 1024)      ds_read from buf1 (offset 2048, 3072)
    v_add_f32 + global_store                v_add_f32 + global_store
    advance idx                             advance idx
    if done → exit                          if done → exit
    wait for buf1 prefetch                  wait for buf0 prefetch
    ──► L_process_buf1                      ──► L_process_buf0
```

#### Exec mask management for partial-wave prefetch

Within each half-iteration, some lanes may have valid `next_idx` while others
do not (they've exhausted their elements).  The kernel handles this by:

1. **Saving** the current exec mask: `s_mov_b64 s[s_cur_exec], exec`
2. **Narrowing** exec to only lanes where `next_idx < N` for the prefetch
3. **Skipping** the prefetch entirely if all lanes are done (`s_cbranch_execz`)
4. **Restoring** exec from the saved mask for the compute/store phase

This ensures that:
- Only lanes with valid next data issue `buffer_load ... lds` (no out-of-bounds
  reads).
- All lanes with valid *current* data participate in the compute and store.
- After the advance (`v_mov idx = next_idx`), a final bounds check determines
  whether to continue or exit the loop.
