# Vector Add -- Hand-Written GCN Assembly for gfx942 (MI300)

A minimal but complete example of a **hand-written AMDGPU assembly kernel** that
performs `C[i] = A[i] + B[i]` on float32 arrays, using the
**`buffer_load_dword ... offen lds`** instruction to load data directly from
global memory into LDS, bypassing VGPRs entirely.

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

## Data Flow

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
    uint32_t N;        // offset 24
    uint32_t __pad0;   // offset 28  (pad to 32 bytes)
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
