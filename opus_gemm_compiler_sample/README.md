# BF16 GEMM (gfx1250, 4-wave compute pipeline)

A hand-tuned BF16 GEMM for AMD RDNA-class gfx1250, built on the
[`opus`](https://github.com/ROCm/aiter) layout/TDM layer. Computes
`C = A · Bᵀ` (A row-major `[M, K]`, B row-major `[N, K]`) in batched form,
using `wmma_16x16x32_bf16` and the **TDM** (Tensor Data Mover) DMA engine.

Unlike the gfx950 [`bf16_gemm`](../opus_gemm/bf16_gemm) siblings — which move data with
`buffer_load` + `ds_write` — this kernel uses gfx1250's TDM to copy global
tiles straight into LDS, with cluster-launch multicast so one DMA fans out to
four workgroups. bf16 results are stored directly: no fp32 workspace, no
split-K, no reduce kernel.

## Files

| File | Role |
|---|---|
| [gemm_defs.h](gemm_defs.h) | Shared `opus_gemm_kargs` + user-facing `opus_gemm_traits` (block tile, ring depth, dtypes, LDS/pad geometry). No `opus` dependency. |
| [gemm_a16w16_4wave_compute_kernel_template.hpp](gemm_a16w16_4wave_compute_kernel_template.hpp) | Kernel template (device-only): derived `kernel_traits`, smem read layouts, the pipeline. |
| [gemm_a16w16_4wave_compute_kernel.cc](gemm_a16w16_4wave_compute_kernel.cc) | Stub TU: host-pass empty body + device-pass explicit instantiation. |
| [gemm_host.cc](gemm_host.cc) | Host driver: random init, CPU reference, validate + benchmark. |
| [Makefile](Makefile) | Build (the kernel TU uses `-D__HIPCC_RTC__` on the host pass to skip the HIP runtime wrapper). |
| [dump_asm.sh](dump_asm.sh) | Runs `llc` on each variant's post-optimisation bitcode and prints the comparison table below. |
| [wm.py](wm.py) | Counts where the WMMA operands actually landed, per variant. |
| [isa/gemm_pin.gfx1250.s](isa/gemm_pin.gfx1250.s) | Committed reference assembly for the default (`pin`) variant. |

## Pipeline

The tile is `B_M × B_N × B_K = 128 × 256 × 128` with a 3-deep LDS ring
(306 KB, so 1 workgroup per CU).

**Symmetric 4 waves.** There is no producer/consumer split — every wave both
issues TDM and runs WMMA. The 4 waves split M (32 rows each) and each
accumulates its own `32 × 256` output strip across the whole K loop. N is
carved into 4 sub-tiles ("msb") of `32 × 64`, giving 32 accumulator slices of
8 VGPR that stay live for the entire kernel (256 fp32 VGPR per lane).

**Half-row TDM (`TDM_LOAD_PART = 2`).** All 4 waves issue one TDM per K step,
each covering half the rows: w0/w1 take the upper/lower half of A, w2/w3 of B.
Halving the tile (A 32→16 KB, B 64→32 KB) halves the per-instruction DMA
latency.

**One folded window.** A and B share a single `opus::tdm<>` instantiation;
`tile_dim1` is the only field that differs and it is patched into the
descriptor at issue time. Two window types would force an `if (isA)` around
every issue, and that branch ends the scheduling region — the descriptor SALU
would land in a basic block containing no WMMA to hide behind.

**Split build/fire.** The descriptor is assembled (`tdm_build`) several WMMAs
before the `tensor_load_to_lds` fires. SALU results do not forward into memory
instructions, so issuing the DMA next to its descriptor setup costs a full SGPR
write-back; the intervening WMMAs cover it for free.

**Register pinning.** The 32 accumulator slices are placed at v256–511 and the
B double-buffer at v512–639 (two banks of 64, one per prefetch buffer), each
buffer split into eight per-WMMA-tile variables rather than one wide vector.
Per-tile access keeps the WMMA reading its B operand in place; pinning a wide
vector and `shufflevector`-ing out of it defeats the pin, because the shuffle
result is a fresh unpinned value. Both banks sit in one 256-VGPR window, so the
WMMA src0 field never needs an `S_SET_VGPR_MSB` switch between sub-tiles.

The request rides on each written value via `__builtin_amdgcn_pin_vgpr`.
Building with `VARIANT=nopin` drops it and changes nothing else, which is what
the comparison below measures.

**Unrolled by ring depth.** The K loop is unrolled by `P = 3` so the LDS slot
index is a literal in every copy. With a runtime slot index, `s * slot_elems`
is a VALU feeding a ds_read address, which the compiler guards with
`s_wait_alu depctr_va_vdst(0)` — and that drains the entire in-flight WMMA
pipe. With the index constant, the multiply folds into the ds_read `offset:`
immediates, removing both the VALU and most of those waits from the hot loop.
The step body lives in one `one_step` lambda, but its two slot indices arrive as
`number<>` and all five call sites pass constants. Letting anything hold them as
a value — an ordinary `int` parameter, or a `static_for` over the slots — turns
them back into registers, and the remainder alone costs 54 extra `va_vdst(0)`
drains that way, 8% on a K where the remainder is 2 of 5 steps.

**Remainder dispatched, not looped.** The at most `P - 1` leftover steps are
spelled out under `if (n_rem >= k)` rather than run in a loop. A loop there
costs the accumulators their pinned tuples: the main loop's values stay live
across it, since a remainder that runs zero times leaves them as the result, so
the two ranges overlap and only one can hold the tuple. Dispatching keeps each
accumulator one live range — pinned placement goes from 444 of 640 WMMA to all
of them, and the copies that reunited the two halves in the epilogue drop from
607 `v_mov` to 126, at 745 VGPRs instead of 848 and 129 fewer instructions
overall. It does not move the clock (those copies all ran once, outside the K
loop); it is what keeps the pinning intact.

**Two-ahead TDM.** The prologue primes all 3 slots unconditionally. Once
`origin0` passes K the descriptor's `tensor_dim0` saturates to 0, so over-issued
trailing steps are zero-extent DMAs that touch no memory and only bump
`tensorcnt`. That makes `loaded == P + g` hold for every shape, which is what
lets the hot loop use an unconditional `s_wait_tensorcnt(1)` and issue the next
TDM with no guard.

**Scheduling.** `sched_group_barrier` groups interleave 1 WMMA : 2 ds_read : 1
SALU inside each tile; a `sched_barrier(0)` at each tile boundary stops the
scheduler sinking a prefetch down to its consumer in the next tile. The last
tile carries the barrier handshake, the next slot's front-load and the next
K step's TDM, all branch-free.

**Grouped epilogue.** C goes out through LDS: each N sub-tile's accumulators are
cast to bf16, rotated across the banks, `ds_store`d into a ring slot that is dead
once the last load DMA retired, and one TDM per wave writes whole `B_N` rows to
global — so the ragged tile needs no guard. The write-back is scheduled the same
way as the K loop, as [VALU][ds_store] per sub-tile: left interleaved, every
`ds_store` inherits the VALU→LDS operand hazard. What remains is 10
`depctr_va_vdst(0)` drains, one per store burst, and both ways of buying them
more distance measured worse — −0.4 to −1.6% fragmenting the bursts, −0.1 to
−2.1% hoisting the casts. The sub-tiles that are final before the last K step
ends are staged inside it, under the WMMAs that follow them.

## Requirements

The **kernel TU** needs a patched LLVM: `__builtin_amdgcn_pin_vgpr`,
`amdgpu_num_vgpr` and gfx1250's v256–v1023 register file are not in stock ROCm.
Build one from the
[`pin_reg_soft_hint`](https://github.com/demonsan/llvm-project/tree/pin_reg_soft_hint)
branch and point `DEVICE_HIPCC` at it; the host TU is ordinary HIP and keeps
using stock ROCm (which also supplies the OpenMP the CPU reference needs).

The pin is a **hint**, not a hard pre-colouring: it is attached to the value and
the allocator honours it when nothing else forces its hand. A request the
allocator cannot satisfy costs correctness nothing — it simply lands elsewhere,
which is what the `wm.py` counts below are for.

Two flags also matter and stock ROCm rejects the second:

```
-mllvm -enable-post-misched=1
-mllvm -amdgpu-expert-scheduling-mode
```

The `opus.hpp` in `OPUS_INCLUDE_DIR` must carry the stateful `opus::tdm<>`
window (the `struct tdm` refactor with `build_desc()` / `set_wg_mask()` and the
inline-asm issue path). The sibling `../../aiter` checkout has it.

## Build & run

```sh
PIN=/path/to/pin-llvm/build/bin       # a pin_reg_soft_hint build
make -j DEVICE_HIPCC="env HIP_CLANG_PATH=$PIN /opt/rocm/bin/hipcc"
./build_pin/gemm_a16w16_4wave_compute.exe
./build_pin/gemm_a16w16_4wave_compute.exe -m 4096 -n 4096 -k 4096 -b 1
```

`HIP_CLANG_PATH` is how hipcc is pointed at another clang, so the flags in the
Makefile stay the same for both toolchains.

Flags: `-m`, `-n`, `-k`, `-b`, `-i` (benchmark iterations; each takes the next
argv). Override `OPUS_INCLUDE_DIR`, `HIPCC`, `DEVICE_HIPCC`, `VARIANT` or `ARCH`
(default `gfx1250`) on the make line. `make clean` removes the build directories
and `asm/`, leaving the committed `isa/`.

The driver validates against an OpenMP CPU reference and prints TFlops. A value
counts as bad only when both the absolute (> 0.5) and relative (> 5%) deviation
are out of tolerance — bf16 output is coarse in absolute terms at large K.

## The comparison

Two builds of one kernel. `VARIANT=pin` (the default) asks for the register plan
with `__builtin_amdgcn_pin_vgpr(v, n)` on each written value; `VARIANT=nopin`
drops those calls and is otherwise the same source — same tile, same ring depth,
same `sched_group_barrier` groups, same 306 KB of LDS, same one workgroup per CU.
Nothing but the placement request differs, so the difference is attributable.

```sh
make variants DEVICE_HIPCC="env HIP_CLANG_PATH=$PIN /opt/rocm/bin/hipcc"
LLC=$PIN/llc ./dump_asm.sh          # both asm into asm/, plus the table
python3 wm.py asm/*.s               # where the WMMA operands actually landed
```

`dump_asm.sh` re-runs `llc` on the bitcode the linker wrapper hands to codegen,
because HIP links the device side through LTO and so never writes a device `.s`
during a normal build.

## Result (gfx1250, `pin_reg_soft_hint` clang), reproducible

Both builds compile the same 768 WMMAs and 864 `ds_read`s, so they differ only in
placement:

| build | asm lines | `s_wait_alu` | `va_vdst(0)` | VGPR | spill |
|---|--:|--:|--:|--:|--:|
| `pin` | 6463 | 288 | **21** | 745 | 0 |
| `nopin` | 6569 | 276 | **160** | 714 | 0 |

`va_vdst(0)` is the number that matters: it drains the whole in-flight VALU pipe
before the dependent instruction, at a floor of ~24 cycles every time one is
reached, and these sit inside the K loop, where the trip count multiplies them.
Pinning removes 139 of them and pays 31 VGPRs, which cost no occupancy — the
306 KB of LDS already caps the kernel at one workgroup per CU.

`wm.py` on the same assembly, counting the 768 WMMAs:

| build | dst in v256–511 | src0 in v512–767 | distinct src0 starts |
|---|--:|--:|--:|
| `pin` | 768 / 768 | 768 / 768 | 16 |
| `nopin` | 0 / 768 | 112 / 768 | 51 |

Every WMMA lands on the tuple it asked for: 32 accumulator slices at v256–v504,
16 B sub-tiles at v512–v632. Without the request the accumulators go to v0–v248
and B scatters across 54 starts, and that scatter is what the drains pay for —
with the two prefetch buffers overlapped, tile `t+1`'s load lands on registers a
tile-`t` WMMA has issued but not yet read, a WAR the hardware can only resolve by
draining.

### Throughput

Both builds validate against the CPU reference with identical `max_abs` and
`max_rel` on every shape, so the two columns are the same arithmetic. Five runs
each, alternating between the builds so clock drift cannot favour one; the
median is reported because two shapes are bimodal. TFlops on AMD gfx1250
(256 CU), `batch = 1`, 300 timed iterations.

| shape | `pin` | `nopin` | gain |
|---|--:|--:|--:|
| 2048³ | 1211 |  874 | **+38.5%** |
| 4096³ | 2121 | 1723 | **+23.1%** |
| 8192³ | 2293 | 1820 | **+26.0%** |
| 4096 x 4096 x 8192 | 2282 | 1805 | **+26.5%** |
| 256 x 5120 x 2880 |  393 |  285 | **+37.7%** |

The worst `pin` run of a shape still beats the best `nopin` run of that shape
everywhere; the narrowest margins are 4096x4096x8192 (1885 vs 1809) and 4096³
(1831 vs 1725). The spread is one-sided: `nopin` repeats to well inside a percent
on every shape, while `pin` occasionally drops a run — the variance comes from
the request not always being met, not from the baseline. The gain is largest on
the smaller shapes, where the K loop carrying those drains is a larger share of
the whole.

## Shape constraints

- **`N` must be a multiple of `B_N` (256).** The epilogue bounds the C store by
  the remaining bytes in the matrix, not per row, so an N tail spills into the
  next row and corrupts it. `M` and `K` are unconstrained (`M` is covered by the
  byte bound, and the TDM zero-fills out-of-range `K`). This matches the aiter
  pipeline this kernel was ported from.
- The launch grid is rounded up to whole 4×4 clusters, because the runtime
  rejects a cluster launch whose grid is not a multiple of the cluster dims. An
  edge cluster therefore carries workgroups with no tile of their own. They fail
  the tile bound check and leave immediately after the cluster-scope (`-3`)
  barrier that precedes the first multicast, having touched neither LDS nor the
  TDM, instead of running the whole K loop on zero-extent DMAs. They cannot leave
  before that barrier: `-3` counts one arrival per workgroup, so skipping it would
  hang every peer of the cluster. Arriving and then leaving is enough because it
  is the only `-3` in the kernel — every later barrier is workgroup-scope (`-1`)
  and all 4 waves of a workgroup exit together.
  The survivors still name the departed in their multicast masks, which neither
  hangs nor writes into the dead LDS — GL1 returns only to the waves that made a
  request — such a request merges with fewer peers than the mask claims and waits
  out the timeout, so an edge cluster gains less than the skipped K loop suggests.
  Measured on shapes with surplus workgroups: +1.4% (¼ surplus) to +7.3% (½
  surplus); fully populated shapes are unchanged.
