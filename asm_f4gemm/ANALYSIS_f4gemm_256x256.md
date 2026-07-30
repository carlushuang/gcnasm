# Reverse-engineering walkthrough: `f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256`

Source: `rebuilt/f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256.s` (3892 lines; instructions end at
line 3403, kernel descriptor at 3406–3443, metadata at 3445–3892). All line numbers below refer to
that file. Cross-checked against `README.md` (same directory) and, for MFMA scale semantics, the
CDNA4 ISA / [Matrix Core Programming on CDNA3/CDNA4](https://salykova.github.io/matrix-cores-cdna).

Launch geometry (from `.amdhsa_*`, lines 3406–3443, and metadata, 3874–3880): 256 threads
(4 × wave64), 512 VGPR (`next_free_vgpr 512`, `accum_offset 256` → 256 arch + 256 acc VGPRs),
98 SGPRs (`next_free_sgpr 98`), 160 KB static LDS (`group_segment_fixed_size 163840`),
`kernarg_segment_size 384`, occupancy 1 workgroup/CU (both VGPR count and LDS force it).

Tile: `D[256×256] (bf16) = alpha · A[256×K] (mxfp4) · B[256×K]^T (mxfp4)`, K-step = 256 elements.

## Instruction census (whole file)

| count | instruction | role |
|---|---|---|
| 512 | `v_mfma_scale_f32_16x16x128_f8f6f4` | all in the mainloop (2 loop bodies × 256) |
| 144 | `ds_read_b128` | A-fragment reads from LDS (16/thread/kstep × 9 static sites) |
| 36 | `ds_read_b32` | scale-A reads from LDS (8/thread/kstep + prologue) |
| 0 | `ds_write*` | **none — LDS is written exclusively by direct-to-LDS buffer loads** |
| 48 | `buffer_load_dwordx4 … lds` | A: global → LDS direct |
| 40 | `buffer_load_dwordx4 … offen` | B: global → VGPR (never touches LDS) |
| 12 | `buffer_load_dword … lds` | ScaleA: global → LDS direct |
| 10 | `buffer_load_dword … offen` | ScaleB: global → VGPR |
| 512 / 512 | `v_accvgpr_read/write_b32` | 256 writes = zero-init; 512 reads = epilogue (256 × 2 paths) |
| 513 | `v_mul_f32_e32` | 512 = acc × alpha in epilogue (2 paths), 1 in prologue div |
| 256 | `v_cvt_pk_bf16_f32` | f32 → bf16 pairs (128 × 2 epilogue paths) |
| 128 | `v_permlane16_swap_b32_e32` | epilogue lane redistribution (64 × 2 paths) |
| 128 | `buffer_atomic_pk_add_bf16` | splitK epilogue (32 groups × 4 dwords) |
| 32 | `buffer_store_dwordx4` | normal epilogue (16 ptrs × 2 column halves) |
| 12 | `s_waitcnt`, 10 | `s_barrier` | pipeline sync (see §4) |

Labels: entry line 9; swizzle loop `label_003D` (48); `label_0042` (54); division path `label_0048`
(61); `label_0068` (89); splitK pointer fixups `label_0090`/`009A`/`00F4`/`010B`/`0124`;
mainloop A `label_041A` (654); mainloop B `label_0971`/`0972` (1112/1114); loop tail
`label_0EC7` (1570); normal epilogue `label_14AC` (2536); end `label_19CC` (3401).

---

## 1. Prologue (lines 9–258)

### 1.1 Kernarg loads (lines 10–28)

`line 10: s_and_b32 s1, s1, 0xffff` — masks the high half of the kernarg-segment pointer to 16 bits
(aiter quirk; harmless for kernarg segments below 4 GB, but worth knowing if the runtime ever
places kernargs high). Then 17 loads from `s[0:1]`:

| line | offset | dest | field | README offset |
|---|---|---|---|---|
| 12 | `0x00` | `s[4:5]` | ptr_D | 0x00 ✓ |
| 13 | `0x10` | `s[8:9]` | ptr_C | 0x10 ✓ |
| 14 | `0x20` | `s[12:13]` | ptr_A | 0x20 ✓ |
| 15 | `0x30` | `s[16:17]` | ptr_B | 0x30 ✓ |
| 16 | `0x40` | `s38` | alpha | 0x40 ✓ |
| 17 | `0x50` | `s39` | beta | 0x50 ✓ |
| 18 | `0x80` | `s40` | stride_C0 | 0x80 ✓ |
| 19 | `0xa0` | `s41` | stride_A0 | 0xa0 ✓ |
| 20 | `0xc0` | `s42` | stride_B0 | 0xc0 ✓ |
| 21 | `0xe0` | `s43` | M | 0xe0 ✓ |
| 22 | `0xf0` | `s44` | N | 0xf0 ✓ |
| 23 | `0x100` | `s45` | K | 0x100 ✓ |
| 24 | `0x110` | `s[20:21]` | ptr_ScaleA | 0x110 ✓ |
| 25 | `0x120` | `s[24:25]` | ptr_ScaleB | 0x120 ✓ |
| 26 | `0x130` | `s36` | stride_ScaleA0 | 0x130 ✓ |
| 27 | `0x150` | `s37` | stride_ScaleB0 | 0x150 ✓ |
| 28 | `0x170` | `s57` | log2_k_split | 0x170 |

Matches the README "read" column exactly: `0x60 stride_D0`, `0x70 stride_D1`, `0x90 stride_C1`,
`0xb0 stride_A1`, `0xd0 stride_B1`, `0x140`, `0x160` are never loaded. One README nuance to
correct: it marks `0x170 log2_k_split` as "splitK kernels only", but **this kernel does load it and
branch on it** — the 256×256 tile carries the full splitK path (§6).

Two dead-on-arrival loads, confirmed by grepping every use:

- **beta (`s39`)**: loaded at line 17, never referenced again. There is no `beta·C` term anywhere.
- **ptr_C (`s[8:9]`)**: a descriptor `s[8:11]` is built (lines 92/96/100/104: `s10 = -16`,
  `s11 = 0x20000`, `s9` masked + `0x40000`) but never appears in any memory instruction. `s10`
  keeps its placeholder `-16` forever (contrast: `s14`, `s18`, `s6` are all recomputed).

### 1.2 Thread decomposition (lines 29–38)

`v0` = flat thread id. Lines 29–33 compute `v1 = tid>>10`, `v2 = (tid>>10)>>10` — both always 0
for 256 threads (template artifact). The live part: `v3 = tid>>6` = **wave id** (line 34), broadcast
to `s49` via `v_readfirstlane` (38); `v0 = tid & 63` = **lane** (35). `s2/s3` (workgroup_id_x/y)
saved to `s46/s47` (36–37); `s4` (workgroup_id_z) saved to `s56` (line 11) **before** line 12
clobbers `s4` with ptr_D — `s56` is the splitK chunk index.

### 1.3 Grid swizzle (lines 40–90)

Implements the README's "flatten `wg_y*gdx + wg_x`, re-swizzle into groups of 32 N-tiles":

- `s54 = (N + 255) >> 8` = gdx (40–41); `s48 = wg_y·gdx + wg_x` = linear tile id (42–43).
- `s52 = ceil(M/256) << 5` = gdy·32 (44–46). Loop `label_003D` (48–53): while `s48 ≥ 32·gdy`:
  `s48 -= 32·gdy; s46 += 32` → `s46` = 32 × panel index, `s48` = index within panel.
- Panel full (≥ 32 N-tiles): `s47 = s48 >> 5` (M-tile), `s52 = s48 & 31` (N-tile within panel) (58–59).
- Edge panel (`label_0048`, 61–88): same division by `s54` (N-tiles in panel) done with a
  float-rcp u32 division sequence (`v_rcp_iflag_f32` + Newton fixup, 62–85) because the divisor is
  runtime-variable.
- `s46 += s52` (line 90): **`s46` = global N-tile index, `s47` = M-tile index.** Verified against
  the D-address math (§6) and the A/B load bases (§2).

### 1.4 Buffer descriptors (lines 91–106)

Five 128-bit buffer descriptors share a template: `dword3 = 0x20000` (95–98), high pointer bits
masked to 16 + `0x40000` (99–106, standard aiter "48-bit address" fixup). `dword2` (num_records)
starts as `-16` (91–94) and is recomputed per buffer:

- A: `s41 >>= 1` (line 118 — README: strides passed as fp4-elements × 2, kernel shifts back to
  bytes); `s14 = s41·M` (119–120).
- B: `s42 >>= 1` (129); `s18 = s42·N` (130–131).
- ScaleA: `s22 = round_up(M,32)·s36` (132–136).
- ScaleB: `s26 = N·s37` (137–138).
- D: computed later at 604–613 (see §6). C: never recomputed (dead).

Note the A/B num_records use the raw M/N (not padded), so out-of-row loads are clamped by the
descriptor — this is why the driver over-allocates A/B rows to 256.

### 1.5 splitK preamble (lines 107–127, 197–202, 216–223, 239–244)

Skipped when `s57 == 0`. With split:
`s58 = round_up(K >> log2_k_split, 256)` = per-chunk K (109–112); this WG's chunk offset is
`s58·s56` (chunk index from wg_z); `s45 = min(K − s58·s56, s58)` = this WG's K extent (113–116).
Then each descriptor is slid to its chunk: A `s12 += s58·s56/2` bytes, `s14` shrunk (123–127);
ScaleA `s20 += s58·s56` (199–202 — 1 byte per K-element because the shuffled scale layout stores a
256-K slab as one 256-B block, see §2.3); B `s16 += 16·(s58·s56/2)` (218–223 — 16 B per packed byte
because the 16×16 preshuffle makes a 32-B K-chunk span 512 B); ScaleB `s24 += s58·s56` (241–244).

### 1.6 Per-thread address precompute (lines 145–257)

All VGPR addresses are computed once here; the mainloop never recomputes an address, it only bumps
the four scalar base pointers `s12` (A), `s16` (B), `s20` (ScaleA), `s24` (ScaleB) by the per-kstep
increments `s61 = 0x80`, `s62 = 0x800`, `s63 = 0x100`, `s64 = 0x100` (set at 254–257).

- **A global** `v212…v219` (145–174): `v212 = s41·row + (lane&7)·16` with
  `row = (lane>>5)·16 + (((lane>>3)&3)>>1)·4 + ((lane>>3)&1) + (s49>>1)·8 + (s49&1)·2 + s47·256`
  (the two wave terms at 158–165). `v213…v219 = v212 + k·(32·s41)` — 8 pointers, 32 rows apart.
  The odd row interleave ({0,1,4,5} per lane + {0,2,8,10} per wave) makes the 4 waves cover 16
  consecutive rows jointly; each load instruction covers 8 rows × 128 B of the current K-slab.
- **A LDS write base** `s59 = 0x1000 + 0x420·s49` (175–176), consumed via `m0`.
- **A LDS read** `v220` (177–195): `= 0x1000 + 0x420·s49 + 0x420·(2·((lane&15)>>3) + ((lane&3)>>1))
  + 0x100·((lane&7)>>2) + 0x80·(lane&1) + 16·(lane>>4)`; `v221 = v220 + 0x8400` (196) = second
  buffer. **Scale LDS read** `v224 = 4·lane` (214–215).
- **ScaleA global** `v222 = 4·lane + (s47·256 + s49·32)·s36`, `v223 = v222 + 128·s36` (204–211).
- **B global** `v225 = 16·lane + (s46·256 + s49·64)·s42`, `v226…v228 = +16·s42` steps,
  `v229…v232 = v225…v228 + 0x400` (225–238; `0x400` = 2 preshuffle blocks = K+128 elements).
- **ScaleB global** `v233 = 4·lane + (s46·256 + s49·64)·s37`, `v234 = v233 + 32·s37` (246–253).

`s60 = s49·0x100` (212–213) is the wave's ScaleA LDS write base (via `m0`).

---

## 2. Global memory traffic

Everything is `buffer_load` against the five descriptors; no `global_load`, no `flat_load`.
Totals per kstep per workgroup: A 32 KB + ScaleA 2 KB (into LDS), B 32 KB + ScaleB 0.5 KB
(into VGPRs).

### 2.1 A — `buffer_load_dwordx4 … lds` (48 sites: 16 prologue + 32 loop)

Per kstep: 8 loads/thread, each 16 B/lane → 8 × 4 waves × 64 × 16 B = 32 KB = 256 rows × 128 B
(= 256 fp4) ✓. The `lds` suffix is the gfx950 **direct-to-LDS** form: the VGPR operand (`v212…`)
is the global address; the LDS address is `m0` (set by the immediately preceding `s_add_u32 m0`)
plus the hardware's implicit per-lane placement. No register ever receives the data and — key
structural fact — **the kernel contains zero `ds_write` instructions**.

`m0` targets (prologue lines 258–559; same pattern in-loop): buffer 0 at
`s59 + {0, 0x1080, 0x2100, 0x3180, 0x4200, 0x5280, 0x6300, 0x7380}`, buffer 1 at the same
`+ 0x8400`. Capacity check: each load instruction = 4 waves × 1 KB = 4 KB; block stride `0x1080`
= 4 wave-chunks of `0x420` = `0x400` data + `0x20` (32 B) pad. So the A buffer is 8 blocks of
(4 × 1 KB + 32 B) = `0x8400` total (32 KB data + 1 KB pad), and `0x420·s49` in `s59` places each
wave's 1 KB chunk inside the block. The per-lane placement inside a wave's `0x400` chunk is
implicit in the direct-to-LDS semantics — from the dword scale loads (§2.3) the natural reading is
`m0 + elemsize·lane`; the read-side formula `v220` (§3) is emitted to match whatever the write side
does, so **treat the `s59`/`v220` formula pair as an indivisible contract** when editing.

### 2.2 B — `buffer_load_dwordx4 … offen` (40 sites: 8 prologue + 32 loop)

Per kstep: 8 loads/thread → 8 × 16 B × 256 = 32 KB ✓, landed in `v[136:167]` (kstep j) /
`v[168:199]` (kstep j+1), consumed **directly by MFMA src0 — B never transits LDS**. This is what
the 16×16 `BpreShuffle` buys: aiter's `shuffle_weight` pre-permutes B so that a thread's plain
16-B chunk is already an MFMA operand fragment. `v225…v228` cover the wave's 4 n-fragments (rows
`ntile·256 + wave·64 + {0,16,32,48}`), `v229…v232` the same rows at K+128 (`+0x400`); hence
`v[136:139]` = n-frag 0 K-half 0, `v[152:155]` = n-frag 0 K-half 1, etc.

### 2.3 Scales — `buffer_load_dword` (12 `lds` + 10 plain)

ScaleA (E8M0, shuffled+padded per README): 4 loads/thread/kstep into LDS (`v222` at `m0 = s60`,
`v223` at `m0 = s60 + 0x400`; second kstep at `+0x800`/`+0xc00`). Each load = 4 B/lane = 256 B/wave
→ 4 waves fill 1 KB; two loads cover the 256 rows × 8 k-groups = 2 KB/kstep. The shuffled layout
(README: `view(sm/32,2,16,sn/8,2,4).permute(0,3,5,2,4,1)`) packs a 32-row × 8-k-group slab as a
contiguous 256-B block — which is why the per-kstep pointer advance is `s63 = 0x100` and the
splitK offset arithmetic at line 199 works without a divide.

ScaleB: 2 plain `buffer_load_dword` → `v208/v209` (kstep j) / `v210/v211` (kstep j+1), 4 B/lane
each = the wave's 64 N-rows × 8 k-groups split across 2 loads × 4 waves. Also straight to VGPR.

### 2.4 Load issue counts

- Prologue (258–559): kstep 0 (8 A-lds, 2 ScaleA-lds, 8 B, 2 ScaleB) **and** kstep 1 (8 A-lds at
  `+0x8400`, 2 ScaleA-lds at `+0x800`) = 30 VMEM ops. B/ScaleB for kstep 1 are loaded inside the
  first loop half instead.
- Each mainloop half (§4) issues 20 VMEM ops: 8 B + 2 ScaleB (to VGPR) + 8 A-lds + 2 ScaleA-lds.

---

## 3. LDS

### 3.1 Map (total used `0x11800` = 71,680 B of 163,840 allocated; ~92 KB dead)

```
0x00000 – 0x01000   ScaleA staging: 2 ksteps × 2 KB, double-buffered
                    [kstep even: 0x000–0x7FF, odd: 0x800–0xFFF]
0x01000 – 0x09400   A buffer 0: 8 blocks × 0x1080 (0x1000 data + 0x80 pad)
0x09400 – 0x11800   A buffer 1 (v221 = v220 + 0x8400)
```

B and ScaleB have no LDS footprint. The 160 KB static allocation is a family-wide constant
(`group_segment_fixed_size 163840`); with 512 VGPRs the kernel is occupancy-1 anyway, so the
waste costs nothing.

### 3.2 Write pattern

No `ds_write` at all. Writes are the direct-to-LDS loads of §2.1/2.3, addressed by `m0`:
A at `s59 + block·0x1080 + buffer·0x8400`, ScaleA at `s60 + {0,0x400} + parity·0x800`. Bank-conflict
avoidance is **padding-based, not XOR-based**: each wave's 1 KB chunk is followed by a 32 B pad
(the `0x420` wave stride), shifting successive waves' chunks across bank phases.

### 3.3 Read pattern feeding MFMA

- A fragments: `ds_read_b128`, 16 quads/thread/kstep. In the prologue (584–599) and identically at
  each loop site: offsets `{0, 64, 512, 576} + {0, 0x1080, 0x2100, 0x3180}` for m-frags 0–7
  (`v[8:39]` = K-half 0, `v[40:71]` = K-half 1) and the same `+0x4200` (offsets 16896…30144,
  lines 663–742 and peers) for m-frags 8–15 (`v[72:135]`). Within a `0x1080` block (2 m-frags):
  `+0`/`+64` = even m-frag K-half 0/1, `+512`/`+576` = odd m-frag. The per-lane scatter `v220`
  (§1.6) plus the 32 B pads is what keeps 64 lanes × 16 B conflict-free.
- ScaleA: `ds_read_b32 v200…v203` (first 64-MFMA group) and `v204…v207` (second group) at
  `v224 = 4·lane` + `{0,256,512,768}` (group 1) / `+1024` (group 2) / `+2048`,`+3072` for the odd
  kstep buffer. 8 dwords/thread/kstep = 2 KB ✓. Each wave reads the **whole** 2 KB scale slab
  (no wave term in `v224`) — scales are shared, A data layout per wave is not.

`lds_direct` / `lds_param` are not used; all LDS traffic is explicit `ds_read` + the implicit
write side of `buffer_load … lds`.

---

## 4. Mainloop structure

### 4.1 Two (nearly) identical loop bodies

`line 652–653: s_cmp_lt_i32 s49, 2; s_cbranch_scc0 label_0971` — waves 0/1 run the loop at
`label_041A` (654–1111), waves 2/3 the one at `label_0972` (1114–1569). The bodies are
instruction-for-identical except (a) self-branch targets, (b) minor scheduling order, and
(c) **body B never executes `s_sub_u32 s22, s22, s63`** — the ScaleA descriptor's num_records is
not decremented for waves 2/3 (body A does it twice: lines 870/876-ish and the grep-confirmed
count 2 vs 0). Benign because SGPRs are per-wave and A rows are over-allocated, but it's an
asymmetry to preserve deliberately or fix consciously — it's the kind of thing a cleanup edit
would "fix" into a regression if the bounds ever mattered.

Why two bodies at all is not recoverable from the asm (they are semantically equivalent); most
likely the source had per-wave-pair scheduling freedom that ended up identical.

### 4.2 Trip count and exit structure

`s50` = K consumed, `s51` = K extent (645; possibly chunk-clamped, §1.5). Each body = **2 ksteps =
256 MFMAs**, with `s_addk_i32 s50, 0x100` + exit test after each kstep half
(877–882, 1105–1111 for body A; 1336–1341, 1563–1569 for body B). Mid-body exits go straight to
`label_0EC7`, so **K = 256 executes exactly one half** (128 MFMAs) and bails — no separate tail
loop. Both halves also re-evaluate prefetch guards: `s62/s64` (B/ScaleB increments) zeroed when
`s50 + 0x200 ≥ s51` (720–724), `s61/s63` (A/ScaleA) when `s50 + 0x300 ≥ s51` (848–858); the setup
does the same for tiny K (646–651). Zeroing the increment makes the dead prefetch re-load the last
valid slab — in-bounds, results discarded.

### 4.3 Pipeline (2-kstep double buffering, 3 things in flight)

Prologue prefetches kstep 0 (A+ScaleA→LDS, B+ScaleB→VGPR) and kstep 1 (A+ScaleA→LDS buf1), then
reads kstep-0 group-1 fragments from LDS (584–603). Each loop half then does, for kstep j
(buffer parity j&1):

1. `s_waitcnt vmcnt(10) lgkmcnt(0)` + `s_barrier` (655/657, 883/885):
   waits until only the 10 most recent VMEM ops are in flight = the B/ScaleB VGPR loads for
   **this** half have landed; `lgkmcnt(0)` drains all prior LDS reads so the A buffer about to be
   overwritten is fully consumed, and the barrier publishes the other waves' direct-to-LDS writes.
2. 64 MFMAs (m-frags 0–7, both K-halves) interleaved with: 8 B + 2 ScaleB loads for kstep j+1,
   and `ds_read` of group-2 fragments for the **current** kstep.
3. `s_waitcnt vmcnt(15) lgkmcnt(0)` + `s_barrier` (763/765, 991/993): paces the pipe — retires the
   A-lds loads of the previous kstep and 5 of the in-flight B loads; the `lgkmcnt(0)` again guards
   LDS reuse.
4. 64 MFMAs (m-frags 8–15) interleaved with: 8 A-lds + 2 ScaleA-lds loads for kstep j+2 into the
   buffer being freed, and `ds_read` of kstep j+1's group-1 fragments (`v[8:71]`, scales `v200–203`)
   from the other LDS buffer.

So the steady state has: current-kstep MFMA operands in registers, next-kstep B in flight to VGPR,
kstep+2 A in flight to LDS. Prologue `s_waitcnt vmcnt(25)` (582) is the same arithmetic with
30 prologue VMEM ops: retires exactly the 5 oldest (A kstep-0 blocks 0–3 + first ScaleA load)
needed by the reads at 584–603.

`label_0EC7` (1570–1574): `s_waitcnt vmcnt(0) lgkmcnt(0); s_barrier` drains everything, then
selects the epilogue on `s57`.

### 4.4 Pipeline timeline

All region boundaries below were re-derived from the raw issue order of body A (lines 654–1111);
nothing here is idealized. MFMA groups are counted per half: **g1 = mfma#1–32 (m-frags 0–7,
K-half 0), g2 = mfma#33–64 (m 0–7, K-half 1), g3 = mfma#65–96 (m 8–15, K-half 0), g4 = mfma#97–128
(m 8–15, K-half 1)** — i.e. the "2 groups of 64" of §4.3 split at the K-half boundary.

#### 4.4.1 Issue-order timeline of one loop half (body A; line numbers = .s file)

```
--- half 1 (lines 654-882): kstep j (even); MFMA operands from LDS buf0;
--- B(j) live in v[136:167]+v208/209 --------------------------------------------
654  label_041A:
655  W1 waitcnt vmcnt(10) lgkmcnt(0)   <- retires B(j) VGPR loads (ledger in 4.4.2);
                                        drains last half's ds_reads into v[8:71]
656  g1 mfma #1
657     barrier + 2 nop                <- publishes kstep-(j+1) A LDS writes x-wave
660  g1 mfma #2..32  (m-frags 0-7, K-half 0)
        per 4-mfma slot: 1x buffer_load_dwordx4 B(j+1) -> v[168:199]  (661..703)
                         1x ds_read_b128 A(j) m8..11 (buf0) -> v[72:119] (663..705)
                                        <- B: ~126 mfma before 1st use (next g1);
                                           ds: operands for g3/g4
707  g2 mfma #33..64 (m-frags 0-7, K-half 1)
        buffer_load_dword ScaleB(j+1) -> v210,v211              (709, 715)
        ds_read_b128 A(j) m12..15 (buf0) -> v[88:135]           (711..742)
        ds_read_b32  ScaleA(j) m8..15  -> v204..v207            (746..758)
        scalar: guard s62/s64 (720..732); bump B ptr s16/s18 (736..744);
                bump ScaleB ptr s24/s26 (748..756)
763  W2 waitcnt vmcnt(15) lgkmcnt(0)   <- retires A(j+1) loads #1-5 (buf1 blocks
                                        0-3 + scale slot 1); drains all buf0 reads
764  g3 mfma #65
765     barrier + 2 nop                <- buf0 now free; A(j+1) blk 0-3 visible x-wave
768  g3 mfma #66..96 (m-frags 8-15, K-half 0)
        per slot: s_add m0; buffer_load_dwordx4 lds A(j+2) -> buf0
                  v212..v218 (768..828); ScaleA(j+2) v222 (804, m0@800)
        ds_read_b128 A(j+1) g1/g2 frags (buf1) -> v[8:67]         (770..826)
                                        <- A: ~2 ksteps before use; feeds next g1/g2
829  g4 mfma #97..128 (m-frags 8-15, K-half 1)
        A(j+2) tail: v219 (836, m0@832); ScaleA(j+2) v223 (844, m0@840)
        ds_read_b128 v[68:71] (830)
        ds_read_b32 ScaleA(j+1) m0-7 -> v200..v203                (834..846)
        scalar: guard s61/s63 (848..857); bump A ptr s12/s14 (860..866);
                bump ScaleA ptr s20/s22 (869..875, body A only);
                s50 += 0x100 + exit test + cbranch 0EC7 (877..882)

--- half 2 (lines 883-1111): kstep j+1 (odd); MFMA operands from LDS buf1;
--- B(j+1) live in v[168:199]+v210/211 ------------------------------------------
883  W3 waitcnt vmcnt(10) lgkmcnt(0)   <- retires A(j+1) #6-10 + B(j+1) loads
884  g1 mfma #1
885     barrier + 2 nop
888  g1 mfma #2..32
        per 4-mfma slot: 1x buffer_load_dwordx4 B(j+2) -> v[136:167]  (889..931)
                         1x ds_read_b128 A(j+1) m8..11 (buf1) -> v[72:119] (891..933)
935  g2 mfma #33..64
        buffer_load_dword ScaleB(j+2) -> v208,v209                (937, 943)
        ds_read_b128 A(j+1) m12..15 (buf1) -> v[88:135]           (939..970)
        ds_read_b32  ScaleA(j+1) m8..15 -> v204..v207             (974..986)
        scalar: guard s62/s64 (948..960); bump B (964..972); bump ScaleB (976..984)
991  W4 waitcnt vmcnt(15) lgkmcnt(0)   <- retires A(j+2) loads #1-5; drains buf1 reads
992  g3 mfma #65
993     barrier + 2 nop                <- buf1 now free for A(j+3)
996  g3 mfma #66..96
        per slot: s_add m0; buffer_load_dwordx4 lds A(j+3) -> buf1
                  (m0 = s59+0x8400+..) v212..v218 (996..1056); ScaleA(j+3) v222 (1032)
        ds_read_b128 A(j+2) g1/g2 frags (buf0) -> v[8:67]         (998..1054)
1057 g4 mfma #97..128
        A(j+3) tail: v219 (1064, m0@1060); ScaleA(j+3) v223 (1072, m0@1068)
        ds_read_b128 v[68:71] (1058)
        ds_read_b32 ScaleA(j+2) m0-7 -> v200..v203                (1062..1074)
        scalar: guard s61/s63 (1076..1085); bump A (1088..1094);
                bump ScaleA (1097..1103); s50 += 0x100 + exit (1105..1110)
1111 s_branch label_041A               <- next W1 retires B(j+2) + A(j+2) stragglers
```

Half 2 is the exact mirror of half 1: B register banks swap (`v[136:167]`↔`v[168:199]`,
`v208/209`↔`v210/211`), LDS buffer roles swap (buf0↔buf1, A-lds `m0` bases shift by `0x8400`),
scale ds_read offsets shift by +2048, and the guards/bumps/exit repeat with the same structure.

Register time-multiplexing worth noting before editing: `v[8:71]` carries kstep j's m-frags 0–7
during g1/g2 and is **re-read with kstep j+1's m-frags 0–7 during g3/g4** — safe only because
g1/g2's last consumer (mfma#64, line 762) precedes the first overwrite (line 770) with W2's
`lgkmcnt(0)` between them. Same for `v[72:135]` (kstep j m 8–15 read during g1/g2, kstep j+1
m 8–15 read during the next half's g1/g2).

#### 4.4.2 Steady-state in-flight chart

Load bundles are issued in units of 10 VMEM ops (8 data + 2 scale), so every `vmcnt` count is a
multiple-of-5 boundary. The outstanding-VMEM ledger (steady state, kstep j, half 1):

```
              issue site            retires at
B(j)   ->VGPR half 2 of kstep j-1   W1 of half 1 (kstep j)        [~1 kstep ahead]
A(j+1) ->LDS  half 2 of kstep j-1   #1-5: W2 of half 1; #6-10: W3 of half 2
B(j+1) ->VGPR g1/g2 of half 1       W3 of half 2                  [~1 kstep ahead]
A(j+2) ->LDS  g3/g4 of half 1       #1-5: W2 of half 2 (kstep j+1);
                                    #6-10: W1 of next iteration   [~2 ksteps ahead]

outstanding ledger (half 1 of kstep j):
  at W1 (655):  B(j)[10] + A(j+1)[10]  = 20 -> vmcnt(10) retires B(j)       -> 10 left
  after g1/g2:  + B(j+1)[10]           = 20 -> vmcnt(15) retires A(j+1)#1-5 -> 15 left
  after g3/g4:  + A(j+2)[10]           = 25
  at W3 (883):                         = 25 -> vmcnt(10) retires A(j+1)#6-10
                                               + B(j+1)                     -> 10 left
  ...and symmetrically for half 2. (Iteration 0 differs only in that the
  prologue's vmcnt(25)/vmcnt(10) pair at 582/655 plays the same role for
  ksteps 0/1.)
```

Residency snapshot at a point inside g3 of half 1 (kstep j):

```
MFMA pipe      : kstep j   - B(j) v[136:167]+v208/209, A(j) m8-15 v[72:135]
VGPR, incoming : kstep j+1 - B(j+1) v[168:199]+v210/211 in flight (issued g1/g2)
LDS buf(j&1)   : kstep j   - fully resident, being drained; A(j+2) overwriting NOW
LDS buf(1-j&1) : kstep j+1 - blocks 0-3 + scale1 complete (retired at W2),
                             blocks 4-7 + scale2 in flight (retire at W3);
                             g1/g2 fragments being copied out to v[8:71] now
global -> LDS  : kstep j+2 - A(j+2)+ScaleA(j+2) in flight (issued g3/g4)
```

So: **A's global path is 2 ksteps deep, B's is 1 kstep deep, and the LDS read path is 1–2 MFMA
groups deep.** The waitcnts quantize retirement to exactly the bundle boundaries; nothing waits
for "everything" except the loop tail (`vmcnt(0)` at 1571).

#### 4.4.3 Wave 0/1 (body A, `label_041A`) vs wave 2/3 (body B, `label_0972`)

Same 256-MFMA skeleton, same barrier/waitcnt positions relative to MFMA index (waits at mfma#0/#64
of each half: body A lines 655/763/883/991, body B lines 1115/1223/1342/1450 — note body B's four
waits are *not* at 883/991, it has its own four), same operand sequence. Two real differences:

1. **Slot order swap in every 4-MFMA memory slot**, systematic across all 32 slots of both
   halves (verified by diffing the non-MFMA instruction positions between the bodies):

```
one 4-mfma memory slot, each body:
 mfma idx | body A (waves 0/1)        | body B (waves 2/3)
 ---------+---------------------------+---------------------------
 #2       | buffer_load B -> VGPR     | ds_read A (LDS)
 #3       | ds_read A (LDS)           | buffer_load B -> VGPR
 #6       | buffer_load ...           | ds_read ...
 #7       | ds_read ...               | buffer_load ...
  ...     | (load leads every slot)   | (read leads every slot)

g3/g4 triple slot:
 body A:  s_add m0 ; ds_read ; buffer_load-lds
 body B:  ds_read ; s_add m0 ; ds_read ; buffer_load-lds  <- reads lead by 1 slot
```

   Interpretation (not provable from the .s): since the `s_barrier`s force all 4 waves through
   each W-point together, the two wave pairs run the same barrier interval concurrently, and the
   swapped slot order makes wave-pair A's VMEM bursts coincide with wave-pair B's LDS bursts and
   vice versa — staggering pressure on the two memory pipes instead of doubling it. This is the
   most plausible reason the source duplicated the loop at all.

2. **The `s22` asymmetry** (already in §4.1): body A decrements ScaleA num_records twice per body
   (lines 875 and the second-half peer at 1103), body B never. Body B is consequently 2
   instructions shorter per body (199 vs 201 non-MFMA ops).

Lockstep mechanics: `s_barrier` is workgroup-scope, so the two bodies' schedules interleave at
barrier granularity — within a barrier interval each pair free-runs (drifting by the few cycles
of their instruction-count and memory-latency differences), and every ~64 MFMAs they re-converge:

```
waves 0/1: W1 ==g1/g2== W2 ==g3/g4== W3 ==g1/g2== W4 ==g3/g4== W1 ...
waves 2/3: W1 ==g1/g2== W2 ==g3/g4== W3 ==g1/g2== W4 ==g3/g4== W1 ...
           |<- free-run; A-pair hits VMEM while B-pair hits LDS ->|
```

Nothing in either body may legally depend on finer inter-wave-pair timing than that; the
cooperative LDS scheme (all waves write disjoint chunks of the same A/ScaleA slabs, all waves read
the whole slab) is safe precisely because the only cross-wave consumers sit behind a barrier whose
preceding `vmcnt`/`lgkmcnt` has retired the writes.

#### 4.4.4 Latency-hiding budget (order-of-magnitude estimates, not measured)

Using the CDNA4 ISA timing for `v_mfma_scale_f32_16x16x128_f8f6f4` with both operands FP4 =
**16 cycles** (the "16 or 32" table entry; 32 only if an operand is FP8). Per wave (the 4 waves
have independent matrix pipes and proceed in parallel between barriers):

```
1 group = 32 mfma ~ 512 cyc;  1 half = 128 mfma ~ 2050 cyc ~ 1 us @ ~2 GHz

B path (1 kstep deep) : issue g1/g2, awaited at W3 ~64-96 mfma later
                        ~1000-1500 cyc (~0.5-0.7 us) of cover.
                        L2 hits trivial; DRAM covered only because loads stream
                        (the wait is for the OLDEST bundle, issued earliest).
A path (2 ksteps deep): first 5 loads awaited ~1 half after issue (~1000-1450 cyc),
                        rest ~1.5 halves (~2000+ cyc ~ 1 us), first mfma use yet
                        another half later. Margin that absorbs DRAM spikes.
LDS ds_read path      : read 1-2 groups (~512-1000 cyc) before use vs ~30-40 cyc
                        LDS latency - massively over-provisioned; the lgkmcnt(0)s
                        are structural (buffer-recycle) guards, not timing waits.
net                   : loop is matrix-pipe issue-bound by design; the waitcnts
                        should ~never stall in steady state.
```

Any edit that adds or removes VMEM ops changes the outstanding ledger in 4.4.2, and the
`vmcnt(10/15)` constants must be re-derived the same way.

---

## 5. Compute: the MFMA sequence

### 5.1 Role swap (the single most important structural fact)

`v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[136:139], v[8:11], a[0:3], v208, v200 …` (line 656):

- **src0 (`v[136:139]`) = B data** (loaded from the ptr_B descriptor), **scale_a (`v208`) = ScaleB**.
- **src1 (`v[8:11]`) = A data** (from LDS), **scale_b (`v200`) = ScaleA**.

The instruction computes with operands swapped: its "A" role holds the weight matrix B, its "B"
role holds the activations A. Consequently each 16×16 accumulator fragment spans 16 rows of N × 16
columns of M, and the wave tile is **256 (M) × 64 (N)** — all 4 waves read the *same* A slab from
LDS (no wave term in the A ds_read offsets beyond the chunk scatter) but each loads its own 64-row
slice of B (`s49·64` in `v225`). This is also why B can bypass LDS: with the preshuffle, B needs no
cross-wave reuse staging, while A — reused by all 4 waves — is staged once through LDS.

### 5.2 Decomposition

Per wave: 64 fragments of 16×16 f32 = 256 accVGPRs (`a[0:255]`), laid out as
**fragment k = m-frag (k mod 8) + 8 · n-frag (k div 8)**, m ∈ [0,16), n ∈ [0,4) — verified by
matching mainloop operand order (§5.3) against epilogue store pairing (§6.2).

Per kstep (K = 256 elements = 2 × the instruction's K=128): **128 MFMAs** — each accumulator hit
twice, once per K-half. Per loop body: 256 MFMAs (2 ksteps) × 2 bodies = 512 static MFMAs total ✓
matches the census. The four 64-MFMA groups per body: (m 0–7, K-half 0), (m 0–7, K-half 1),
(m 8–15, K-half 0), (m 8–15, K-half 1).

### 5.3 Operand feeding and the op_sel / cbsz / blgp code

Within group 1 (656–762): B quads `v[136:139]`→acc `a[0:31]` (n0), `v[140:143]`→`a[32:63]` (n1),
`v[144:147]`→`a[64:95]` (n2), `v[148:151]`→`a[96:127]` (n3); A quads `v[8:39]` cycle m-frags 0–7.
K-half 1 (707–762) repeats with B quads `v[152:167]` and A quads `v[40:71]`, same accumulators,
with `op_sel_hi:[1,1,0]`. Group 2 (765–881) reuses the **same** B quads against `v[72:135]` for
m-frags 8–15 into `a[128:255]`.

Every MFMA carries `cbsz:4 blgp:4` (the standard gfx950 MXFP4 broadcast setting) and a position
code in `op_sel` / `op_sel_hi`. The pattern is fully systematic (verified across all 512):

- `op_sel[1]` = m-frag parity, `op_sel[0]` = n-frag parity, `op_sel_hi[1]` = `op_sel_hi[0]` = K-half.

These bits select which byte of the packed scale VGPR applies: each scale register holds 4 E8M0
bytes per lane and one MFMA consumes one byte per lane (16 rows × 4 k-groups = 64 lanes for
16x16x128; per the CDNA4 ISA the scale is applied per 32-element K-group). Byte index
= parity + 2·K-half:

- ScaleA: `v200` = (m-frags 0,1) × (K-half 0,1), `v201` = m 2,3, `v202` = m 4,5, `v203` = m 6,7;
  `v204–207` the same for m 8–15.
- ScaleB: `v208` = (n 0,1) × (K-half 0,1), `v209` = n 2,3; `v210/v211` the next kstep's pair.

Caveat: the byte-selection reading is an inference from the systematic pattern + ISA operand
rules, not something the disassembly states outright; if you change the MFMA↔scale assignment,
verify against the ISA's op_sel table. (Related: the 508-byte round-trip delta noted in the README
is bits 13/14 of the MFMA first dword — the src2/accumulator op_sel bits, which are don't-care.)

### 5.4 Zero-init

Lines 260–581: 256 × `v_accvgpr_write_b32 aN, 0` interleaved between the prologue's async loads —
free zeroing of all accumulators hidden under the load latency.

---

## 6. Epilogue

Two fully-unrolled paths, selected at 1573–1574 by `s57 == 0`. **Neither touches C/beta** — both
compute `D = alpha · acc` only (the 512 `v_mul_f32` are all `v8 = s38 · acc`).

### 6.1 D addressing (both paths)

Descriptor `s[4:7]` on ptr_D, prepared at 604–613:

- `s40 = stride_C0 << 1` (line 604 — elements → bf16 bytes). **`stride_C0` (0x80) is used for D;
  `stride_D0` (0x60) is never loaded — README claim confirmed.**
- `s4/s5 += s47·256·s40` (605–610): descriptor base advanced to this workgroup's M-tile row.
- `s6 = (M − s47·256)·s40` (611–613): num_records = remaining rows × row stride — the M-edge is
  bounds-checked by the buffer descriptor, there are no explicit compare/mask instructions.

Per-thread column base `v235…v250` (614–643): `v235 = (lane&15)·s40 + (lane>>5)·16 +
((lane>>4)&1)·32 + (s46·256 + s49·64)·2`; `v236…v250 = v235 + k·(16·s40)` = 16 pointers, 16 rows
apart → the wave's full 256 rows. So a store covers rows `mtile·256 + (lane&15) + 16·k` and columns
`ntile·256 + wave·64 + {0,8,16,24}[lane group] + {0,32}`, each lane writing 8 contiguous bf16
(16 B). Wave w stores the tile's columns [w·64, w·64+64) — matching its 256×64 compute quadrant.
There is no N-edge masking; N edge safety comes from the padded allocation (README §D).

### 6.2 Normal path (`label_14AC`, 2536–3399)

32 store groups (16 row-pointers × 2 column-halves). Group (m, h): read `a[4(m+16h) : +3]` and
`a[4(m+16h)+32 : +3]` (= n-frags 2h, 2h+1 of m-frag m — confirms the §5.2 acc layout), scale by
alpha, `v_cvt_pk_bf16_f32` into 4 packed regs, two `v_permlane16_swap_b32` pairs
(`v16↔v18`, `v17↔v19`, each followed by `s_nop 1`) to exchange packed pairs across 16-lane halves —
this converts the MFMA fragment layout (each lane holding 4 columns of one row in each of two
fragments) into 8-consecutive-column runs — then one `buffer_store_dwordx4 v[16:19], vXXX, s[4:7]`.
Address advanced `+64` B (32 columns) for the second half. 32 stores × 16 B × 256 threads = 128 KB
= 256×256 bf16 ✓.

### 6.3 splitK path (`label_0EC7` + 1575–2534)

Identical read/scale/pack/swap sequence, but instead of stores: 4 ×
`buffer_atomic_pk_add_bf16 vN, vXXX, s[4:7], 0 offen offset:{0,4,8,12}` per group (no dwordx4
atomic exists) = 128 atomics, same addresses. This confirms the README: **the splitK bf16-atomic
epilogue is compiled into this kernel even for splitK=1 launches** — it's dead code only because
`s57 == 0` skips it. It also means any edit that shifts register allocation must keep both
epilogues consistent.

---

## 7. Notable / a modifier's checklist

1. **B never touches LDS.** The whole B path is `buffer_load_dwordx4` → MFMA src0, valid only
   because `shuffle_weight` pre-arranged the fragments. If you change the tile's N decomposition,
   B's per-thread addresses (`v225…v232`, lines 225–238) and the preshuffle must change together.
2. **Direct-to-LDS is the only LDS writer.** `m0` must be set immediately before each
   `buffer_load … lds` (the `s_add_u32 m0` / load pairs are interleaved with accvgpr zeroing and
   MFMAs — easy to break when re-scheduling). The write-side placement is hardware-implicit; the
   read side (`v220` formula, lines 175–195; `v224`) must match it. Don't "simplify" one without
   the other.
3. **waitcnt arithmetic is exact, not defensive.** `vmcnt(25)`/`vmcnt(10)`/`vmcnt(15)` count
   precisely the outstanding loads in this schedule (30 prologue ops; 20 per half). Inserting or
   removing any load changes the required counts, and `lgkmcnt(0)` at the four loop waitcnts is
   what makes LDS buffer reuse safe. The `s_barrier` after each waitcnt publishes the cooperative
   LDS writes across waves (all 4 waves read all of A/ScaleA).
4. **512 VGPRs, occupancy 1.** Every VGPR is spoken for: B operands double-buffered in
   `v[136:199]`, A fragments rolling through `v[8:135]`, scales `v[200:211]`, addresses
   `v[212:250]`. The epilogue reuses `v8+` freely because the loop is done. Any new live value in
   the mainloop means spilling something by hand — there is no allocator to save you.
5. **Dead kernargs to not trip over**: beta (`s39`) and the entire C descriptor are loaded/built
   but unused; `stride_D0` is dead by design (D uses `stride_C0`); `tid>>10` computations
   (lines 29–33) are dead. Conversely `log2_k_split` (`s57`) is live even here, and the splitK
   epilogue it gates shares all addressing with the normal path.
6. **Wave-pair asymmetry**: waves 0/1 and 2/3 run textually separate but equivalent loops; body B
   omits the `s22` (ScaleA num_records) decrement. Keep them in sync when editing — and note the
   A/ScaleA bounds tracking only matters because the driver over-allocates operands to 256-row
   multiples.
7. **Line 10** masks the kernarg pointer to 48 bits; descriptor dword3 uses the `0x40000` high-bits
   convention. If you port address math, keep the same convention or the buffer descriptors break.
8. **Loop bookkeeping is scalar and interleaved**: pointer bumps (`s12/s16/s20/s24 += s61…s64`) and
   the `s50`/`s51` exit tests are scheduled *between* MFMAs. The guards (zeroing `s61…s64` near
   K-end) make the prefetch re-read the last slab rather than run out of bounds — preserve that
   trick or replace it with explicit predication.
9. **s_nop placement**: the `s_nop 1` after each `v_permlane16_swap_b32` and around
   `v_readfirstlane` are pipeline hazards (216 `s_nop` total); they are required as-is.
10. Round-trip reality (README): the only bit-level diff after reassembly is the don't-care
    accumulator op_sel bits of the MFMA first dword; everything else is exact, so line-level
    references in this document stay valid after `clang -x assembler` rebuilds.
