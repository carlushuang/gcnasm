# Vector Add -- Hand-Written GCN Assembly for gfx942 (MI300)

A minimal but complete example of a **hand-written AMDGPU assembly kernel** that
performs `C[i] = A[i] + B[i]` on float32 arrays, using the
**`buffer_load_dword ... offen lds`** instruction to load data directly from
global memory into LDS, bypassing VGPRs entirely.

The kernel demonstrates several GPU programming techniques:
- **Persistent kernel**: grid size = number of CUs (detected at runtime); each
  workgroup processes multiple chunks via a grid-stride loop.
- **Double LDS buffering** with deep pipeline fill: both buffers are loaded in
  the prologue; each loop half reads one buffer while prefetching it for 2
  iterations ahead.
- **OOB-based control flow**: SRDs use `num_records = N * 4` so that
  out-of-bounds buffer loads return 0 and stores are silently dropped,
  eliminating all exec mask manipulation.
- **`vmcnt(3)` accounting**: carefully chosen wait count that drains exactly the
  previous half-iteration's 2 prefetch loads + 1 store, keeping the current
  half's operations fully in flight.

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
        |  buffer_load_dword ... offen lds   (VGPR bypassed, OOB → 0)
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
        |  buffer_store_dword ... offen      (OOB → silently dropped)
        v
Global Memory (C)
```

## Deep-Pipeline Double LDS Buffer (steady state)

```
PROLOGUE (fill both buffers):
    buffer_load A[iter0] -> buf0         vmcnt = 1
    buffer_load B[iter0] -> buf0         vmcnt = 2
    buffer_load A[iter1] -> buf1         vmcnt = 3
    buffer_load B[iter1] -> buf1         vmcnt = 4
    s_waitcnt vmcnt(2)                   buf0 ready, buf1 in flight

MAIN LOOP (each half-iteration, steady state entry vmcnt = 3):

    ┌── ds_read A, B from current buffer
    │   s_waitcnt lgkmcnt(0)
    │
    │   buffer_load A[idx+2*stride] ──┐   prefetch into SAME buffer
    │   buffer_load B[idx+2*stride] ──┘   (we already read it above)
    │
    │   v_add_f32 A, B
    │   buffer_store C[idx]               (OOB → dropped)
    │
    │   idx += stride
    │   s_cbranch_vccz L_done             (exit when all lanes done)
    └── s_waitcnt vmcnt(3)                drain prev half's loads + store

    swap to other buffer, repeat
```

Key design points:
- **Prologue fills both buffers** so the loop body never stalls on the first
  read.  `vmcnt(2)` after 4 loads drains the oldest 2 (buf0), leaving buf1 in
  flight.
- **Prefetch goes into the same buffer** just read (not the alternate), loading
  data for `idx + 2*stride` (2 half-iterations ahead).
- **`vmcnt(3)`** in the loop drains exactly the previous half's 2 prefetch
  loads + 1 store, keeping the current half's 2 prefetches + store in flight.
  See section 13 for the detailed FIFO trace.

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
| 2 | `N * 4` | `num_records` in bytes (enables OOB for idx >= N) |
| 3 | `0x00020000` | gfx942 config: `DATA_FORMAT=32`, `TYPE=0` (raw) |

Setting `num_records = N * sizeof(float)` rather than `0xFFFFFFFF` enables the
OOB (out-of-bounds) behavior used for branchless control flow (see section 12).

Word 3 value `0x00020000` matches the
[opus library](https://github.com/ROCm/aiter)'s `buffer_default_config()` for
gfx942/gfx90a.  `TYPE=0` (raw buffer) means OOB loads return 0 and OOB stores
are silently dropped.

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
| `vmcnt` | Global/buffer memory ops (loads, stores, **buffer_load...lds**, **buffer_store**) | `s_waitcnt vmcnt(N)` |
| `lgkmcnt` | LDS and scalar memory (kernarg loads, `ds_read`, `ds_write`) | `s_waitcnt lgkmcnt(0)` |

For the async-to-LDS pattern, the basic sequence is:

```asm
buffer_load_dword ... lds       ; global -> LDS  (increments vmcnt)
s_waitcnt vmcnt(0)              ; ensure LDS is written

ds_read_b32 ...                 ; LDS -> VGPR   (increments lgkmcnt)
s_waitcnt lgkmcnt(0)            ; ensure VGPRs are ready
```

No `s_barrier` is needed because each wave only reads its own LDS region.

**Important**: `vmcnt` is a FIFO -- `s_waitcnt vmcnt(N)` means "wait until at
most N operations remain outstanding."  Both `buffer_load ... lds` and
`buffer_store` push onto the same FIFO, so stores must be accounted for when
choosing the wait value.  See section 13 for the detailed `vmcnt(3)` analysis.

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

### 12. OOB-based control flow -- no exec mask needed

Traditional GPU kernels use exec mask manipulation (`s_and_saveexec_b64`,
`s_cbranch_execz`, etc.) to prevent out-of-bounds memory accesses.  This kernel
takes a simpler approach by exploiting the **hardware OOB behavior** of buffer
instructions:

| Buffer operation | OOB behavior (`TYPE=0`, raw) |
|-----------------|-------------------------------|
| `buffer_load ... offen lds` | Returns **0** to LDS (harmless) |
| `buffer_store ... offen` | **Silently dropped** (no side effects) |

By setting `num_records = N * 4` in every SRD (A, B, and C), any lane whose
byte offset `>= N * 4` automatically gets this safe behavior.  The kernel
**never touches the exec mask** -- all 64 lanes always execute every
instruction.

```asm
; SRD construction -- the key line
s_lshl_b32 s[s_res_a+2], s[s_n], 2    ; num_records = N * sizeof(float)
```

Loop termination uses a simple scalar comparison:
```asm
v_cmp_gt_u32 vcc, s[s_n], v[v_idx]    ; any lane still in-bounds?
s_cbranch_vccz L_done                  ; if none → exit
```

This eliminates all `s_and_b64 exec`, `s_or_b64 exec`, `s_mov_b64 exec` and
`s_cbranch_execz` instructions, producing cleaner and shorter code.

### 13. Double LDS buffering with deep pipeline fill

The kernel uses **two LDS buffers** in a ping-pong arrangement with a deep
prologue that fills **both** buffers before the loop begins.

#### LDS layout (4096 bytes total)

```
Byte offset    Contents
──────────────────────────────────────
[   0, 1024)   Buffer 0 -- A values   (256 threads x 4 bytes)
[1024, 2048)   Buffer 0 -- B values
[2048, 3072)   Buffer 1 -- A values
[3072, 4096)   Buffer 1 -- B values
```

#### Four M0 values (pre-computed, loop-invariant)

```asm
s_m0_buf0_a = wave_lds_base + 0       ; buffer 0, A region
s_m0_buf0_b = wave_lds_base + 1024    ; buffer 0, B region
s_m0_buf1_a = wave_lds_base + 2048    ; buffer 1, A region
s_m0_buf1_b = wave_lds_base + 3072    ; buffer 1, B region
```

#### Prologue -- fill both buffers

```asm
; iter 0 → buf0
buffer_load A[idx]     -> buf0     ; vmcnt 1
buffer_load B[idx]     -> buf0     ; vmcnt 2
; iter 1 → buf1
buffer_load A[idx+s]   -> buf1     ; vmcnt 3
buffer_load B[idx+s]   -> buf1     ; vmcnt 4
s_waitcnt vmcnt(2)                 ; drain oldest 2 → buf0 is ready
```

#### Unrolled loop with prefetch-to-same-buffer

Each half reads from its buffer, then prefetches **into that same buffer**
for 2 half-iterations ahead (`idx + 2*stride`):

```
L_process_buf0:                         L_process_buf1:
    ds_read A, B from buf0                  ds_read A, B from buf1
    lgkmcnt(0)                              lgkmcnt(0)
    prefetch A[idx+2s] -> buf0              prefetch A[idx+2s] -> buf1
    prefetch B[idx+2s] -> buf0              prefetch B[idx+2s] -> buf1
    v_add_f32                               v_add_f32
    buffer_store C[idx]                     buffer_store C[idx]
    advance idx                             advance idx
    if no lanes valid → L_done              if no lanes valid → L_done
    s_waitcnt vmcnt(3)                      s_waitcnt vmcnt(3)
    ──► L_process_buf1                      ──► L_process_buf0
```

### 14. `vmcnt(3)` -- the critical synchronization accounting

Choosing the right `vmcnt(N)` value is arguably the most subtle part of a
double-buffered GPU kernel.  Getting it wrong either causes correctness bugs
(too loose) or kills performance (too tight).

#### Why `vmcnt(3)` and not `vmcnt(0)` or `vmcnt(2)`?

The `vmcnt` FIFO tracks all outstanding `buffer_load` and `buffer_store`
operations.  `s_waitcnt vmcnt(N)` means "wait until at most N entries remain
in the FIFO."

Consider the steady-state at the top of `L_process_buf0`, just after the
`vmcnt(3)` at the end of `L_process_buf1`:

```
vmcnt FIFO (oldest → newest):
  [already drained by vmcnt(3)]          ← previous buf0's 2 loads + 1 store
  entry 3: prefetch A -> buf1 (from L_process_buf1 we just left)
  entry 2: prefetch B -> buf1 (from L_process_buf1 we just left)
  entry 1: buffer_store C     (from L_process_buf1 we just left)
  ─── vmcnt = 3, exactly what we specified ───
```

At this point buf0's data was loaded **2 half-iterations ago** -- it is
guaranteed ready because `vmcnt(3)` drained everything older than the 3
entries from the half-iteration we just completed.

If we used:
- **`vmcnt(0)`**: Everything drained -- safe but **kills pipelining**.  No
  overlap between memory ops and compute.
- **`vmcnt(2)`**: Would only keep 2 entries (the 2 prefetches), meaning the
  store is also drained.  This is slightly tighter than needed.
- **`vmcnt(3)`**: Keeps all 3 entries from the *previous* half in flight while
  guaranteeing the *current* buffer is ready.  **Optimal overlap.**

#### Special cases

| Location | Wait value | Reason |
|----------|-----------|--------|
| After prologue | `vmcnt(2)` | 4 loads issued; drain oldest 2 (buf0 ready), buf1's 2 loads stay in flight |
| Inside loop | `vmcnt(3)` | Drain prev half's 2 loads + 1 store; keep current half's 2 prefetches + 1 store |
| `L_done` (epilogue) | `vmcnt(0)` | Drain everything before final stores complete |
