# opus-ir specification, schema 1

opus-ir is a typed, hardware-explicit schedule IR for AMD GPUs. Its *shape* follows Cake IR (typed resources + a small fixed op vocabulary + verifier with stable diagnostics + canonical JSON identity). Its *vocabulary* is opus.hpp (ROCm/aiter `csrc/include/opus/opus.hpp`): every resource and op lowers to exactly one opus construct, and every hardware fact the verifier checks is transcribed from an opus arch guard. Reference opus revision: ROCm/aiter `146a4a2e4` (2026-10-14).

## 1. Design invariants

Inherited from Cake:

1. Performance-relevant hardware decisions are explicit: instruction shape, wave grid, vector width, LDS stages and padding, wait counts, barrier placement.
2. Mechanical metadata is derived: LDS offsets, buffer resources, per-lane offsets, `make_tiled_mma` template arguments.
3. One canonical spelling per construct; IR is immutable and exactly JSON-serializable; SHA-256 of canonical JSON is the identity.
4. Exact targets. A schedule for `gfx950` never silently lowers for `gfx942`.
5. Missing analysis or lowering coverage is reported, never treated as proof.

Added for opus:

6. **Vocabulary closure.** If opus has no API for it, the IR has no op for it. New ops arrive by adding the opus API first (kernels belong in aiter/opus, not here).
7. **The layout is opus's layout.** Cake deliberately has no first-class layout. opus does: a strided descriptor with an x/p/y distribution view. The IR adopts exactly that and nothing more: no composition, complement, or swizzle algebra.
8. **Synchronization is counter-based.** AMD has no mbarrier/transaction-bytes. Memory completion is a per-wave in-order counter (`vmcnt`/`lgkmcnt` on gfx9; `loadcnt`/`storecnt`/`dscnt`/`kmcnt`/`asynccnt`/`tensorcnt` on gfx1250), and visibility across waves needs `s_barrier` or a named barrier. The IR makes `wait(counter, n)` explicit, counted in instructions as the hardware does, and the verifier checks it.

### Divergences from Cake

| Concern | Cake (NVIDIA) | opus-ir (AMD / opus) |
|---|---|---|
| Layout | not first-class; raw offsets + swizzle tags | first-class opus `layout`: shape, x-stride, p/y dims, p-coord; mma-derived operand layouts |
| On-chip memory | SMEM + TMEM | LDS only (`smem`); accumulators are VGPR/AGPR register tiles |
| Async copy | `cp.async`, TMA | `gmem::async_load` (CDNA `buffer_load…lds`, gfx1250 `global_load_async_to_lds`), gfx1250 `tdm` |
| Completion | mbarrier phase + tx bytes | wait counters (`wait(counter, outstanding)`) |
| Cross-warp handoff | mbarrier arrive/wait | `sync_threads` (s_barrier), gfx1250 named barriers, `workgroup_barrier` semaphore across WGs |
| Matrix core | mma.sync / wgmma / tcgen05 | mfma (gfx942/950), wmma (gfx12, gfx1250), incl. MX-scaled |
| Tail handling | predication | buffer-resource OOB (`make_gmem(ptr, size)`) |
| Warp | 32 lanes | wave of 64 (gfx9) or 32 (gfx12, gfx1250), from the target |

## 2. Types

`DType` uses opus names; the C++ type is `opus::<name>_t`: `fp32 fp16 bf16 fp8 bf8 fp4 e8m0 i64 u64 i32 u32 i16 u16 i8 u8 int4 uint4`. `fp4`, `int4`, `uint4` are packed two per byte (opus `packs`) and are never byte-addressable scalars. `fp8`/`bf8` are FNUZ on gfx942 and OCP elsewhere (`numeric_limits<fp8_t>`); the verifier emits `TARGET008` info on gfx942.

`Symbol(name, minimum, maximum, multiple_of)` is a runtime `int` kernel argument. Bounds are a correctness contract used to prove LDS capacity, trip counts, and divisibility. `TensorType(dtype, shape)` has `int | Symbol` dims.

### Index expressions

`Expr` is integer arithmetic (`add sub mul div mod cdiv min max`) over constants, symbols, loop variables, and hardware ids. Hardware ids map 1:1 to opus wrappers: `lane` (`lane_id()`), `wave`, `thread_x/y/z` (`thread_id_x()`…), `block_x/y/z` (`block_id_x()`…), and gfx1250 `cluster_x/y/z`, `cluster_wg_x/y/z`. `lane` and `thread_*` are lane-divergent. Everything else is wave-uniform.

JSON: an int is a constant, `{"sym": "K"}`, `{"var": "k"}`, `{"hw": "lane"}`, and `["add", lhs, rhs]`.

## 3. Resources

All resource names share one namespace.

| Resource | Fields | Lowers to |
|---|---|---|
| `Param` | name, `TensorType`, access (`read`/`write`/`read_write`), `oob`, alignment | kernel pointer arg + `auto g_X = opus::make_gmem(ptr[, bytes])`; `oob=true` passes the size for hardware range checking |
| `Lds` | name, dtype, per-stage shape, stages, `pad` (elements appended per row), offset, alignment, `alias_of` | `opus::make_smem(reinterpret_cast<T*>(lds_base + offset + stage*stage_bytes))` |
| `Reg` | name, dtype, per-lane `elems`, `zero_init` | `opus::vector_t<T, elems>` |
| `Mma` | name, kind (`mfma`/`wmma`), a/b/c dtypes, `wave`=(W_M,W_N,W_K), `expand`=(E_M,E_N,E_K), `tile`=(T_M,T_N,T_K), `swap_ab`, `scaled` | `make_mfma<a,b,c>(W…)`, or `make_tiled_mma<a,b,c>(seq<E…>, seq<T…>, seq<W…>[, mfma_adaptor_swap_ab{}])` |
| `Layout` | name, flat static `shape`, `dims` (groups of `p`/`y` per x-dim), `x_stride` (one `Expr` per x-dim), `p_coord` (one `Expr` per `p`) | all-`y` → `make_layout(shape, stride)`; otherwise `make_layout(shape, unfold_x_stride(dim, shape, x_stride), unfold_p_coord(dim, p_coord))` |
| `MmaLayout` | name, mma, operand `a`/`b`/`c`, `x_stride` (2-D) or `packed`, optional `p_coord` | `partition_layout_{a,b,c}(mma, x_stride[, p_coord])` / `partition_layout_{a,b,c}_packed(mma, p_coord)` |
| `Tdm` | name, param, lds, `tile`, `pad_interval`/`pad_amount`, `multicast` (`none`/`x`/`y`), `gather_index_size`, `cache_policy` | `opus::tdm<T, seq<tile…>, padding<…>, multicast/peers_along_*, gather<…>, cache<…>>` + `make_tdm(lds_base, ptr, layout)` |
| `Role` | name, waves, kind | `if (wave in waves) { … }` around the role's ops |
| `Pipeline` | name, stages | none; it documents a loop's ring depth and is checked against `Lds.stages` |
| `NamedBarrier` | name, member roles | gfx1250 `DECLARE_NAMED_BARRIERS` slot (at most 15) + `s_barrier_init_ptr(bar, members)` / `s_barrier_join_ptr` |
| `Semaphore` | name, `read_write u32` param | `opus::workgroup_barrier(ptr)` |

LDS planning (`passes/memory.py`): explicit offsets are honored, and the rest are packed in declaration order at their alignment. `alias_of` reuses the backing offset. A stage occupies `prod(shape[:-1]) * (shape[-1] + pad)` elements.

### Layout semantics

`offset = Σ index[i] * stride[i]` over the flattened shape. Within an x-dim group, strides unfold row-major from that x-dim's stride (`unfold_x_stride`). `p` dims take their index from `p_coord` (`unfold_p_coord`), and `y` dims are the per-lane register space. `elems_per_thread = prod(y extents)`. A layout op issues `elems_per_thread / vec` instructions (opus `layout_load_traits::r_elem`), vectorized along the innermost `y` dim, which `vec` must divide. Layouts are loop-invariant: iteration enters only through an op's scalar `offset` and `stage`.

For an `MmaLayout`, per-lane elements come from the adaptor: A=`W_M·W_K/warp·E_M·E_K`, B=`W_N·W_K/warp·E_N·E_K`, C=`W_M·W_N/warp·E_M·E_N`.

## 4. Operations

Common fields: `opcode`, `inputs`, `outputs`, `role` (null = every wave), `attrs` (canonical JSON), `body` (loops only). `offset` attributes are scalar element offsets that lower to the opus `s_os` argument (an SGPR), so they must be wave-uniform. `stage` attributes select the LDS ring slot and must evaluate to `[0, stages)` from loop variables and constants.

| Opcode | Signature | opus lowering | Counter (gfx9 / gfx1250) |
|---|---|---|---|
| `gmem_load` | param → reg; layout, vec, offset | `r = g.load<vec>(u, offset)` | vm / load |
| `gmem_store` | reg → param; layout, vec, offset | `g.store<vec>(r, u, offset)` | vm / store |
| `gmem_atomic_add` | reg → param; layout, vec, offset | `g.atomic_add<vec>(r, u, offset)` | vm / store |
| `gmem_async_load` | param → lds; layout_g, layout_s, vec, offset, stage | `g.async_load<vec>(lds_stage_ptr, u_g, u_s, offset)` | vm / async |
| `smem_load` | lds → reg; layout, vec, stage | `r = s.load<vec>(u)` | lgkm / ds |
| `smem_tr_load` | lds → reg; layout, vec, stage | `r = s.tr_load<vec>(u)` | lgkm / ds |
| `smem_store` | reg → lds; layout, vec, stage | `s.store<vec>(r, u)` | lgkm / ds |
| `tdm_load` | tdm; origin, stage | position window at `origin`; `opus::async_load(window)` | — / tensor |
| `tdm_store` | tdm; origin, stage | `opus::async_store(window)` | — / tensor |
| `mma` | (mma, a, b[, c][, scale_a, scale_b]) → c; accumulate, scaled | `c = mma(a, b, c)` / `mma(a, b)` / `mma(a, b, c, sa, sb)` | — |
| `cast` | reg → reg | `opus::cast<dst_t>(src)` | — |
| `elementwise` | regs → reg; kind ∈ add sub mul max min copy scale fill, scalar | element loop over `vector_t` (`opus::max/min` for max/min) | — |
| `wait` | counter, outstanding | gfx9 `s_waitcnt_vmcnt/lgkmcnt(number<n>)`; gfx1250 `s_wait_{load,store,ds,km,async,tensor}cnt(number<n>)` | — |
| `sync_threads` | — | `opus::sync_threads()` (gfx1250: `sync_workgroup()`) | — |
| `sync_cluster` | — | `opus::sync_cluster()` | — |
| `barrier_arrive` / `barrier_wait` | named barrier | opus gap: opus wraps only `s_barrier_init_ptr`/`s_barrier_join_ptr` (15 slots from `DECLARE_NAMED_BARRIERS`); signal/wait need opus wrappers over `__builtin_amdgcn_s_barrier_signal_var` / `s_barrier_wait` before this lowers | — |
| `sem_wait` | semaphore; cmp `eq`/`lt`, value, slot | `workgroup_barrier::wait_eq / wait_lt(value, slot)` | — |
| `sem_inc` | → semaphore; slot | `workgroup_barrier::inc(slot)` | — |
| `loop` | var, trip, unroll, pipeline; body | `for (int var = 0; var < trip; ++var)` with `#pragma unroll unroll` | — |

`outstanding` counts instructions, not ops. Example: `g.async_load<8>` over a 4×(8-wide) y-space issues 4 `buffer_load…lds` per lane, so waiting for one A+B stage while the next is in flight is `wait(vm, 8)`.

## 5. Synchronization model

Per wave, each counter is an in-order FIFO of issued instructions. `wait(c, n)` blocks the executing waves until at most `n` remain. An LDS stage write is *complete* for a writer wave once its instructions have drained. A `ds_write` is also ordered for the writer's own later `ds_read`, but other waves still need a wait before the barrier. A completed write becomes *visible* to other waves only through a handoff after completion:

- `sync_threads`: all waves; it also retires all prior LDS reads (ends the WAR window). It must be executed by every wave, so it is illegal inside a role (`SCHED003`).
- named barrier: `barrier_arrive` by role R snapshots R's completed writes and R's reads; a later `barrier_wait` by role R2 makes those writes visible to R2 and retires those reads.

Note that `opus::sync_threads()` is a bare `s_barrier` with no fence. That's why an explicit `wait` before it is required and checked.

Target-specific facts the verifier enforces:

- CDNA `buffer_load…lds` writes LDS at `M0 + inst_offset + lane·bytes`. Only a wave-uniform base survives, so the LDS layout of `gmem_async_load` must be lane-contiguous (`ASYNC001`). gfx1250 async loads are exempt.
- gfx1250 TDM: at most 3 tensor ops in flight per wave. Exceeding that stalls but does not fail (`SYNC006` warning).
- TDM padding applies only in the load direction (`TDM002` on a padded `tdm_store`). Multicast requires a cluster launch (`TDM003`).

## 6. Targets

| Target | Wave | LDS | Matrix core (opus dispatch rows) | Async → LDS bytes/lane | Counters |
|---|---|---|---|---|---|
| gfx942 | 64 | 64 KiB | mfma f16/bf16 32x32x8, 16x16x16; fp8/bf8 32x32x16, 16x16x32; f32 32x32x2, 16x16x4; chained f16/bf16 32x32x16, 16x16x32 | 1, 2, 4 | vm ≤ 63, lgkm ≤ 15 |
| gfx950 | 64 | 160 KiB | gfx942 rows + native f16/bf16 32x32x16, 16x16x32 + scaled f8f6f4 32x32x64, 16x16x128 (fp8·fp8, fp8·fp4, fp4·fp4); `tr_load` | 1, 2, 4, 12, 16 | vm ≤ 63, lgkm ≤ 15 |
| gfx1250 | 32 | 320 KiB | wmma f16/bf16 16x16x32 (c = f32 or same), f32 16x16x4, fp8/bf8 16x16x{64,128} (c = f32/f16), scaled 16x16x128, fp4 32x16x128; `tr_load`, TDM, clusters, named barriers | 1, 2, 4, 8, 16 | load, store, ds, async, tensor ≤ 63; km ≤ 31 |
| gfx1200, gfx1201 | 32 | 64 KiB | wmma f16/bf16 16x16x16 (c = f32 or same), fp8/bf8 16x16x16, chained 16x16x32 | synchronous fallback | none in opus |

On gfx12, opus's `s_waitcnt` uses the gfx9 encoding even though RDNA4 has split counters, so the IR rejects `wait` there (`TARGET006`). This looks like an opus gap worth fixing upstream. wave64 gfx12 is not modeled.

## 7. Diagnostics

`IR001` bad kernel name · `IR002` duplicate/undeclared name · `IR003` undeclared/inconsistent symbol · `IR004` no params · `IR005` empty body · `IR006` unknown opcode/kind · `IR007` operand kinds/arity · `IR008` body on non-loop.
`SCHED001` block size · `SCHED002` role waves · `SCHED003` block-wide sync inside a role · `SCHED005` grid uses device ids · `SCHED006` loop variable scope · `SCHED007` stage not computable/out of range · `SCHED008` LDS stages ≠ pipeline stages (warning) · `SCHED009` divergent scalar offset.
`MEM001` extents/stages · `MEM002` offset alignment · `MEM003` overlap · `MEM004` LDS capacity · `MEM005` bad alias.
`LAYOUT001` dims/shape · `LAYOUT002` p-coord count · `LAYOUT003` x-stride count · `LAYOUT004` p-coord lane-invariant (warning) · `LAYOUT005` spans more threads than the block · `LAYOUT006` bad mma ref · `LAYOUT007` mma layout stride form.
`TYPE001` alignment · `TYPE002` dtype mismatch · `TYPE003` illegal vector width · `TYPE004` layout elems ≠ register elems · `TYPE005` mma/cast/elementwise operand type · `TYPE006` packed row granularity · `TYPE007` semaphore param.
`TARGET000` unknown target · `TARGET001` mma not in opus table (notes list legal shapes) · `TARGET002` synchronous async_load fallback (warning) · `TARGET003` tr_load · `TARGET004` tdm · `TARGET005` named barrier unsupported or more than 15 · `TARGET006` counter · `TARGET007` atomic dtype · `TARGET008` fp8 FNUZ (info) · `TARGET009` cluster · `TARGET010` chained instruction (info).
`ACCESS001` write to read-only · `ACCESS002` read from write-only.
`MMA001` mma waves ≠ executing waves · `MMA002` bad expand/tile/wave split · `MMA003` scaled mismatch · `MMA004` accumulator in ≠ out.
`ASYNC001` CDNA async LDS layout not lane-contiguous.
`TDM001` tile/stage/rank/origin · `TDM002` padding · `TDM003` multicast without cluster.
`SYNC001` wait count out of range · `SYNC002` read of own in-flight async write · `SYNC003` read without handoff · `SYNC004` write-after-read · `SYNC005` non-member barrier use · `SYNC006` TDM ring overrun (warning).
`DATA001` register read before write · `DATA002` LDS stage read before any write (warning).

The hazard simulator runs the program in order for three trip-count policies per loop (minimum, minimum+1, ring depth+2). It treats every multi-wave LDS access as touching the whole stage, so it is conservative: hand-partitioned LDS (each wave touching only its own rows) can be reported as a hazard. It ignores named-barrier phase/parity, and role interleavings other than program order are covered only through the arrive/wait handoff rules.

## 8. Kernel ABI

`extern "C" __global__ __launch_bounds__(block_size) void <name>(params…, symbols…)`: pointer params in declaration order, then symbols as `int` in declaration order. The grid is evaluated on the host from `grid` expressions. A gfx1250 cluster launch passes `cluster` dims.

## 9. Not yet modeled

opus APIs not in schema 1: `load_if`/`store_if`/`async_load_if` predicates, `mov_dpp`/`upd_dpp`/`shfl`/`med3` cross-lane ops, `mdiv`, the `aux` cache-policy argument on gmem ops, buffer atomic fmin/fmax/cmpswap, scaled-mfma `scale_op_sel`/`cbsz`/`abid`/`blgp`, TDM `move()`/descriptor reuse and gather index lists, `layout_cached`/`layout_linear` selection (`cached_vec`), and wave64 gfx12. Each of these needs a builder entry, verifier rule, round-trip test, and an entry in this document before it is accepted (same contributor rule as cake-ir).
