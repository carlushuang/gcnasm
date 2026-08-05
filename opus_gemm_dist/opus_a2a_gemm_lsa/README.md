# opus_a2a_gemm_lsa

Correctness-first prototype for a single-launch 8-rank A2A + GEMM fused kernel.

Target logical shape per rank:

- local input shard: `[2048, 1024]` bf16
- replicated weight: `[8192, 8192]` bf16, stored as `[N, K]`
- final output: `[2048, 8192]` bf16
- split-K partitions: 8, one per rank input shard

Kernel layout:

- workgroups `0..15`: LSA rotate put of local input shard into peer receive windows
- remaining workgroups: quad-subtile GEMM tasks over `(m_tile, n_tile, k_part)`

Build inside the MORI/ROCm container:

```bash
cd /shared/amdgpu/home/jiahao_zhou_qle/blyu/opus_a2a_gemm_lsa
make
```

Run:

```bash
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MORI_SOCKET_IFNAME=enp193s0f0np0
export LD_LIBRARY_PATH=/shared/amdgpu/home/jiahao_zhou_qle/blyu/mori/python/mori:${LD_LIBRARY_PATH:-}
mpirun --allow-run-as-root -n 8 ./build/a2a_gemm_lsa.exe
```

Useful debug flags:

```bash
./build/a2a_gemm_lsa.exe -m 256 -n 256 --k-shard 128
./build/a2a_gemm_lsa.exe --mode 2   # RCCL + pack + GEMM baseline
```

## CCO-SDMA serial and parallel pipeline

Mode 4 uses MORI CCO SDMA to transfer A shards into two receive slots and then
reuses the existing Mode 2 compute-from-receive kernel. The fused LSA path
(mode 0) remains the default and fallback.

MORI must be built with `BUILD_CCO_SDMA=ON`. Enable SDMA at runtime and select
the schedule explicitly:

```bash
export MORI_ENABLE_SDMA=1
export MORI_SDMA_NUM_CHANNELS=1

# SDMA A2A e, then GEMM e; no cross-epoch overlap.
mpirun --allow-run-as-root -n 8 ./build/a2a_gemm_lsa.exe \
  --comm-backend sdma --input-mode generic_a2a --comm-schedule serial \
  --warmup 10 --iters 30

# While GEMM reads receive slot e, SDMA fills slot e+1.
mpirun --allow-run-as-root -n 8 ./build/a2a_gemm_lsa.exe \
  --comm-backend sdma --input-mode generic_a2a --comm-schedule parallel \
  --warmup 10 --iters 30

# Start GEMM after post+self-copy; consume each remote K shard when its SDMA
# ready counter arrives while quiet/notify runs on the communication stream.
mpirun --allow-run-as-root -n 8 ./build/a2a_gemm_lsa.exe \
  --comm-backend sdma --input-mode generic_a2a --comm-schedule intra \
  --warmup 10 --iters 30

# Literal single-kernel experiment: block 0 issues/quiet SDMA while every
# workgroup computes K shards and polls per-source ready epochs.
mpirun --allow-run-as-root -n 8 ./build/a2a_gemm_lsa.exe \
  --comm-backend sdma --input-mode generic_a2a --comm-schedule fused \
  --warmup 10 --iters 30
```

`--comm-schedule auto` is the default and selects parallel. `--mode 4` remains
an equivalent shorthand for profiling scripts; modes 0 through 3 retain their
existing behavior.

`serial` uses one non-blocking HIP stream. SDMA post, the rank-local D2D copy,
quiet/notify, ready polling, and Mode 2 GEMM are enqueued in order without
cross-stream event waits. `fused` uses that same one-stream host setup, but
overlaps SDMA and GEMM inside its single kernel. Only parallel and intra create
separate communication and compute streams. At M=6144, 8-rank generic A2A, a
same-session run changed serial E2E from 0.8067 to 0.7847 ms (2.7% lower);
parallel remained within run-to-run variation.

`intra` is an experimental same-epoch stream-fused schedule. It records a
`stage_ready[slot]` event after SDMA post and the self D2D copy, then launches
a compile-time Mode 3 GEMM. The GEMM computes the local K shard first and uses
system-scope acquire polling before each remote shard. The communication stream
quiet/notifies peer queues concurrently with that GEMM. CCO window and DevComm
state remain outside the compute kernel.

`fused` is a literal single-kernel experiment derived from the external
`Opus_a2a_gemm_cco_sdma` design. Block 0 posts all peer PUTs at kernel entry,
computes its tile, and quiet/notifies one destination after each K shard.
Other workgroups compute the local shard first and acquire the same uint64
ready epochs before remote shards. The implementation supports 4/8 ranks,
broadcast/generic inputs, and dynamic M shapes. The retained fast path uses the
normal hardware workgroup scheduler for all tile counts; a cooperative
persistent variant was correct but slower for large M.

Each send and receive window contains two slots. In parallel/intra mode, a
`comm_ready[slot]` event makes the completed SDMA slot visible to the compute
stream, and a `recv_free[slot]` event prevents the communication stream from
overwriting a slot still consumed by GEMM. Serial mode relies on in-stream
ordering instead. Each remote peer has one SDMA queue. The post kernel submits
one no-signal PUT per remote peer, and the quiet/notify kernel drains that peer
queue through MORI's rptr/wptr state before incrementing the destination's
64-bit ready counter. No caller-owned expected-signal state or PUT-tail local
atomic is required by the current MORI API. The host also clears MORI's benign
`hipErrorPeerAccessAlreadyEnabled` status after DevComm creation before the
first kernel launch.
The destination polling kernel acquires every remote source counter for the
slot's target epoch before recording `comm_ready`.
The rank-local shard is copied into the same receive layout with an asynchronous
device-to-device copy.
The two send slots contain different A values and alternate by epoch; validation
checks the corresponding final C values and every per-slot, per-source remote
ready counter, so stale-slot reuse is observable rather than masked by identical
inputs.

The reported `comm_ms` and `compute_ms` are isolated per-epoch measurements.
They now use one common warmed baseline before any schedule-specific workload,
so serial/fused/parallel/intra measure the same comm and Mode 2 compute paths
from equivalent stream, slot, and counter state.
`pipeline_total_ms` is the end-to-end average across the requested epochs; for
parallel mode it includes fill and drain, so use multiple warm iterations and
at least 30 measured iterations for steady-state comparisons.

Split LSA and single-stream SDMA serial additionally report strict
`critical_*` fields. Four phase-boundary HIP events are recorded per measured
epoch; rank 0 gathers every rank's averages, selects the rank with the largest
E2E, and reports that same rank's communication, compute, and
`barrier_idle_ms`. Consequently
`critical_comm_ms + critical_compute_ms + barrier_idle_ms` equals
`critical_e2e_ms` by construction. SDMA fused uses the same four-event
instrumentation around self-copy and the fused kernel for a like-for-like
strict E2E comparison. These instrumented numbers include event overhead and
therefore supplement rather than replace the lower-overhead headline timings.
Literal fused now also honors `--strict-timing 0`: one start/stop pair encloses
self-copy plus the fused kernel for every measured epoch. Five-run M=1024/2048
tests found that strict timing added 16.9-19.6 us, while low-overhead E2E
matched the historical report values within 0.5%.

Parallel and intra additionally report an actual cross-stream timeline.
Communication and compute streams wait on one common start event; a timing
stream joins their final done events. Per-epoch start/end events provide
`critical_comm_ms`, `critical_compute_ms`, and `overlap_ms`; the report also
emits `exposed_comm_ms = comm - overlap` and
`fill_drain_idle_ms = E2E - (comm + compute - overlap)`. Intra compute time
includes Mode-3's in-kernel remote-ready stalls. Event insertion can materially
perturb small shapes, especially 8-rank parallel, so this path is intended for
decomposition rather than headline ranking.

For low-perturbation overlap analysis, pass `--strict-timing 0` and use
rocprofv3 kernel trace. The 2026-07-31 report profiles every rank concurrently
with `warmup=3,iters=10`, selects each rank's final ten measured epochs, and
chooses the rank with the largest trace timeline. Communication spans
post-to-wait-ready for serial/parallel and post-to-quiet for intra; compute spans
the corresponding Mode-2/Mode-3 GEMM kernel. Interval intersections yield
overlap, with `exposed_comm = comm - overlap` and
`E2E = comm + compute - overlap + fill/drain idle`. These traces are under
`build/system_trace_overlap_20260731/`.

On 8 ranks with generic A2A, `N=K=8192`, `K_SHARD=1024`,
`--warmup 10 --iters 30`, the latest-MORI three-run medians were:

- `M=2048`: fused LSA 0.3108 ms, split LSA 0.4196 ms, RCCL 0.4291 ms,
  SDMA serial 0.2783 ms, SDMA parallel 0.2194 ms.
- `M=6144`: fused LSA 0.9061 ms, split LSA 1.0102 ms, RCCL 1.0287 ms,
  SDMA serial 0.7780 ms, SDMA parallel 0.6124 ms.
- `M=16384`: fused LSA 2.2905 ms, split LSA 2.4332 ms, RCCL 2.6409 ms,
  SDMA serial 2.0211 ms, SDMA parallel 1.5927 ms.

Every displayed mode is the median of three process runs. Across all seven
tested M values, SDMA parallel reduces latency by 27.0% to 33.1% versus fused
LSA and by 19.6% to 22.1% versus SDMA serial. Mode 4 is retained as an
opt-in backend; mode 0 remains the default because SDMA requires an SDMA-enabled
MORI build and runtime configuration. Broadcast and generic inputs both pass
serial and parallel correctness on 4 and 8 ranks.

The matching isolated SDMA breakdown medians were:

- `M=2048`: 0.0925 ms communication and 0.2084 ms compute.
- `M=6144`: 0.2329 ms communication and 0.6078 ms compute.
- `M=16384`: 0.5753 ms communication and 1.5250 ms compute.

The parallel steady state approaches the compute duration because the next
epoch's SDMA transfer is shorter than the current epoch's GEMM.

The experimental intra schedule passed 4/8-rank broadcast and generic A2A
correctness, including odd/even epoch alternation and pipeline-output checks.
For 8-rank generic A2A, `N=K=8192`, `warmup=10,iters=30`:

- `M=1024`: parallel 0.1705 ms, intra 0.1715 ms.
- `M=2048`: 0.2194 ms vs 0.2271 ms.
- `M=4096`: 0.4102 ms vs 0.4279 ms.
- `M=6144`: 0.6124 ms vs 0.6289 ms.
- `M=8192`: 0.8010 ms vs 0.8297 ms.
- `M=12288`: 1.1959 ms vs 1.2256 ms.
- `M=16384`: 1.5927 ms vs 1.6387 ms.

Intra is 0.6% to 4.3% slower than parallel in steady state, averaging 2.9%,
because Mode 3 adds per-shard readiness synchronization while cross-epoch
parallel already hides the full communication phase. For warmed
`M=2048,broadcast,iters=1`, three-run medians were 0.4627 ms serial,
0.4967 ms parallel, and 0.3930 ms intra. Intra improves the local host-pipeline
single-epoch result but remains slower than the external single-kernel fused
reference (0.2676 ms), so it stays opt-in and `auto` remains parallel.

The literal fused schedule also passed 4/8-rank broadcast/generic correctness,
dynamic M, and 1000-epoch stress. Static launch and the fused-only relaxed-ready
protocol reduced the generic full E2E at
`M=1024/2048/4096/6144/8192/12288/16384` to
0.2067/0.2709/0.5216/0.7727/1.0228/1.5308/2.0447 ms.

The dynamic-M remote fused results were
0.2049/0.2635/0.5388/0.7750/1.0686/1.5363/2.0491 ms. The two literal kernels
are now within roughly 5% on every shape: local is faster at five shapes and
remote is faster at M=1024/2048. Both remain substantially slower than the
current cross-epoch parallel schedule, so literal fused stays experimental and
`auto` remains parallel.

Additional fused A/B tests found:

- Static normal launch versus cooperative persistent improved M=4096/8192/16384
  E2E from 0.6301/1.2693/2.4209 to 0.6059/1.1143/2.1926 ms and passed a
  1000-epoch M=4096 stress run.
- Using the recv window as the broadcast SDMA source regressed the M=2048
  kernel median from about 0.3395 to 0.3467 ms, so the separate send window was
  retained.
- C-store modes 0/1/2 produced M=2048 kernel medians of
  0.3444/0.3413/0.3408 ms respectively; the existing mode 2 remains best.
- Replacing the fused ready `fetch_add` with a release store showed no stable
  benefit and was reverted.
- Keeping the monotonic counter but using relaxed system-scope producer and
  consumer atomics reduced the M=2048 fused kernel from roughly 0.34 to
  0.27–0.28 ms. This is safe in the tested CCO path because `quietQueue`
  completes the SDMA transfer before publishing the counter; 4/8-rank generic
  tests and 1000-epoch M=2048/M=4096 stress runs passed.

The clean gfx950 resource build reports:

- `a2a_sdma_post_kernel`: 30 SGPR, 32 VGPR, 0 LDS, no SGPR/VGPR spill,
  8 waves/SIMD.
- `a2a_sdma_quiet_notify_kernel`: 14 SGPR, 10 VGPR, 0 LDS, no spill,
  8 waves/SIMD.
- `a2a_sdma_wait_ready_kernel`: 18 SGPR, 4 VGPR, 0 LDS, no spill,
  8 waves/SIMD.
- reused Mode 2 compute kernel: 76 SGPR, 210 VGPR, 135168 bytes LDS,
  no spill, 2 waves/SIMD.
- intra Mode 3 compute kernel: 84 SGPR, 212 VGPR, 135168 bytes LDS,
  no spill, 2 waves/SIMD.
- literal fused non-persistent kernel: 91 SGPR, 214 VGPR, 135168 bytes LDS,
  no spill, 2 waves/SIMD.
- literal fused persistent A/B variant:
  103 SGPR, 222 VGPR, 135172 bytes LDS, no spill, 2 waves/SIMD.

The symbolized rank-0 timeline is in
`build/traces/sdma_parallel/rank0_results.pftrace`, with its CSV companion in
`rank0_kernel_trace.csv`. It shows the communication kernels on stream 1 and
the Mode 2 GEMM on stream 2. In a steady-state epoch, the next post,
rank-local copy, and quiet/notify interval lies inside the previous GEMM
interval, confirming cross-epoch overlap. The decoded advanced thread trace is
under `build/traces/sdma_parallel_att/` in its generated `ui_output_*`
directory.

The single-stream serial capture is in
`build/traces/sdma_serial_single_stream/rank0_results.pftrace`. Its kernel CSV
shows post, quiet/notify, ready polling, and Mode 2 GEMM all on stream 1.
The corresponding fused capture is in
`build/traces/sdma_fused_single_stream_M2048/rank0_results.pftrace`; the fused
kernel and its isolated comm/compute measurements also all use stream 1.

The intra traces are in
`build/traces/sdma_intra_M2048_single/rank0_results.pftrace` and
`build/traces/sdma_intra_M4096_system/rank0_results.pftrace`. They show Mode 3
GEMM beginning before quiet/notify finishes on the communication stream; the
later remote K-shards are released by the per-source uint64 counters.

The literal fused system and ATT captures are under
`build/traces/sdma_literal_fused_M2048/` and
`build/traces/sdma_literal_fused_M2048_att/`. The timed fused path contains one
GEMM dispatch; block 0 performs SDMA packet submission and
per-shard quiet/notify inside that dispatch. The ATT capture can report cutoff
waves because only one target CU/shader engine is traced, but the symbolized
kernel timeline and correctness result are complete.

The 8-rank `M=16384` captures use the symbolized
`build_thread_trace_m16384` binary and are under
`build_thread_trace_m16384/traces/`:

- `serial_system/` and `fused_system/` contain rank-0 kernel/ HIP API CSV,
  JSON, and Perfetto traces.
- `serial_att_compute/`, `serial_att_post/`, `serial_att_quiet/`, and
  `serial_att_wait/` contain decoded `ui_output_*` ATT projects for each
  serial phase.
- `fused_att_compute/` traces a normal compute CU (`SE0/CU7`), while
  `fused_att_block0/` traces the communication-owning block 0 on `SE2/CU5`.

ATT serialization strongly perturbs reported latency, so those runs are used
only for wave/ISA attribution. Separate normal warmup=10/iters=30 serial and
fused runs both passed correctness.

Persistent compute scheduling is available for fused mode 0:

```bash
# Static one-WG-per-tile baseline (default).
mpirun --allow-run-as-root -n 8 ./build/a2a_gemm_lsa.exe \
  --persistent 0 --warmup 3 --iters 20

# Persistent compute workers. With --compute-wgs 0 (the default), the host uses
# min(tile_count, CU_count - comm_wgs). Override it to sweep worker counts.
mpirun --allow-run-as-root -n 8 ./build/a2a_gemm_lsa.exe \
  --persistent 1 --compute-wgs 252 --warmup 3 --iters 20
```

The persistent-scheduler measurements below used the original
`COMM_PEER_ORDER=0` communication order, to isolate compute scheduling.
For the default `M=2048, N=8192, K=8192` shape there are exactly 256 output
tiles. On a 256-CU gfx950, repeated A/B tests showed no persistent scheduling
speedup: the 252-worker persistent median was about 0.839 ms versus 0.812 ms for
the static baseline. The default therefore remains `--persistent 0`.

A larger `M=6144, N=8192, K=8192` case creates 768 compute tiles, i.e. three
tiles per CU on a 256-CU gfx950:

```bash
mpirun --allow-run-as-root -n 8 ./build/a2a_gemm_lsa.exe \
  -m 6144 -n 8192 --persistent 1 --compute-wgs 248 --warmup 3 --iters 20
```

With the final no-spill scheduler, five alternating A/B runs measured a
2.215 ms median for the 768-CTA static baseline and 2.204 ms for the 248-worker
persistent variant, a small improvement of about 0.5%. Persistent scheduling is
therefore kept as an experiment rather than the default.

Increasing the tile count did not make persistent scheduling more favorable.
Three repeated runs per configuration (`--warmup 3 --iters 20`) gave these
median times for the static baseline versus 252 persistent workers:

- `M=8192`, 1024 tiles, 4 CTA/CU: 2.8839 ms vs 2.8937 ms (persistent 0.34% slower).
- `M=12288`, 1536 tiles, 6 CTA/CU: 4.2415 ms vs 4.2583 ms (persistent 0.40% slower).
- `M=16384`, 2048 tiles, 8 CTA/CU: 5.6097 ms vs 5.6379 ms (persistent 0.50% slower).
- `M=24576`, 3072 tiles, 12 CTA/CU: 8.3997 ms vs 8.4561 ms (persistent 0.67% slower).

The hardware workgroup scheduler already balances these uniform GEMM tiles
well. The persistent counter atomic and workgroup barrier repeat after every
tile, so their accumulated overhead grows slightly with larger tile counts.

## Communication peer ordering

The retained communication order phase-shifts each communication WG's peer
sequence. Concurrent copy WGs therefore write to different peers instead of
all bursting stores to one peer before moving to the next. The historical
`COMM_PEER_ORDER=0/1` build switch has been removed after the A/B below.

Five alternating 8-rank runs of the default shape with static compute scheduling
measured 0.8170 ms for the legacy order and 0.7736 ms for the staggered order,
about 5.6% lower latency. With 252 persistent compute workers the medians were
0.8241 ms and 0.7768 ms, about a 6.1% improvement from peer staggering.

The effect grows with communication volume: at `M=6144, N=8192, K=8192`, three
static-scheduler runs measured 2.209 ms for the legacy order and 1.860 ms for
the staggered order, about 18.8% higher throughput. Both 4-rank and 8-rank
correctness tests pass.

## Broadcast and generic all-to-all inputs

The fused kernel supports two source layouts:

```bash
# Existing behavior: one [M,K_SHARD] source shard is copied to every peer.
./build/a2a_gemm_lsa.exe --input-mode broadcast

# Generic all-to-all: local_a is [rank_count,M,K_SHARD], with one distinct
# source chunk for each destination.
./build/a2a_gemm_lsa.exe --input-mode generic_a2a
```

In generic mode, communication WG stores to destination `d` read from
`local_a[d]`. The receive layout remains `[source_rank,M,K_SHARD]`, so the GEMM
path is unchanged. Mode 2 also sends `local_a[peer]` through RCCL, matching
`dist.all_to_all_single` chunk semantics. Compute-only mode 1 remains
broadcast-only.

For 8 ranks with `M=6144, K_SHARD=1024`, one BF16 chunk is 12 MiB. Broadcast
mode allocates 12 MiB of local A, while generic mode allocates 96 MiB, an
additional 84 MiB per rank. Remote xGMI bytes are unchanged.

Using `M=6144, N=K=8192`, `--warmup 10 --iters 30`:

- fused broadcast, static compute: 0.8635 ms.
- fused generic, static compute: 0.8686 ms median over five runs.
- fused generic, 240 persistent workers: 0.9915 ms.
- generic Mode 2 RCCL+pack+GEMM: 1.0243 ms.
- `gistfile1.py --side a2a_gemm`: 0.980 ms p50.

The final generic fused kernel is about 11.4% faster than gist and 17.9% faster
than Mode 2 on this shape. Broadcast and generic fused performance are now
within 1%.

## Optimization ladder

The final communication and compute configuration is now fixed in the source:

```text
contiguous communication WGs spread across XCCs
phase-shifted peers across communication WGs
OPUS buffer copy with streaming/cache policy
one ready counter per source shard
rank-rotated fixed K-shard order
pair-coalesced ds_bpermute C store
pre-store workgroup barrier retained
comm_wgs=16 by default
```

For 8-rank generic `M=6144,N=K=8192`:

- legacy placement, 4 comm WGs: 1.9405 ms median.
- XCC-spread placement, 16 comm WGs: 1.0303 ms median.
- buffer/cpol copy mode: 0.8939 ms median.
- pair-coalesced C store: 0.8709 ms median.
- final repeated result: 0.8686 ms median.

Per-M-tile ready was correct but regressed from 0.896 ms to 1.558 ms because of
per-tile fences, barriers, and atomics. Ready-aware K ordering made that path a
further 3.9% slower. Removing the pre-store barrier was also about 8% slower.
The seven historical switches for peer order, WG placement, copy mode,
per-tile ready, ready-aware K, C-store mode, and pre-store barrier have been
removed. Their winning behavior is directly encoded in the kernel. Only
`RECORD_WG_HW` remains as an optional diagnostic build switch.

The 2026-08-05 cleanup produced identical gfx950 resource reports for every
kernel instance: Mode 0 static/persistent remain 95/106 SGPR and 212/224 VGPR,
Mode 1 remains 56/218, Mode 2 remains 76/210, Mode 3 remains 84/212, and
Mode 4 static/persistent remain 91/103 SGPR and 214/222 VGPR. All GEMM modes
retain zero spill and two waves/SIMD; the LSA communication kernel remains
75 SGPR / 53 VGPR with eight waves/SIMD.

Three-run low-overhead medians before and after cleanup
(`warmup=10,iters=30`) were:

- `M=2048`: SDMA serial 0.2791 -> 0.2789 ms, parallel
  0.2191 -> 0.2175 ms, intra 0.2279 -> 0.2257 ms, and literal fused
  0.2898 -> 0.2882 ms.
- `M=8192`: fused LSA 1.1529 -> 1.1540 ms, SDMA serial
  1.0223 -> 1.0217 ms, parallel 0.8052 -> 0.7999 ms, intra
  0.8257 -> 0.8348 ms, and literal fused 1.0393 -> 1.0376 ms.

An additional five-run alternating generic fused-LSA A/B at `M=2048`
measured 0.3249 ms before and 0.3278 ms after cleanup. All changes are within
1.1%. Four-rank modes 0/1/2/3, 4-rank SDMA serial/parallel/intra/fused, and
the matching 8-rank fused-LSA and SDMA schedules all passed correctness.

## Multi-shape baseline comparison

Using 8-rank generic A2A, `N=K=8192`, `K_SHARD=1024`,
`--warmup 10 --iters 30`, the strict baseline is XCC0-only placement with
4 communication WGs, pointer copy, and direct C store. The final configuration
uses 16 XCC-spread communication WGs, buffer/cpol copy, and pair-coalesced
C store:

- `M=2048`: 0.8262 ms baseline vs 0.3125 ms final (2.64x).
- `M=4096`: 1.3987 ms vs 0.6235 ms (2.24x).
- `M=6144`: 1.9479 ms vs 0.8731 ms (2.23x).
- `M=8192`: 2.5098 ms vs 1.1462 ms (2.19x).
- `M=12288`: 3.4919 ms vs 1.8101 ms (1.93x).
- `M=16384`: 4.5261 ms vs 2.2858 ms (1.98x).

Across these shapes the final kernel reduces latency by 48% to 62%, with no
SGPR/VGPR spill and unchanged 2 waves/SIMD occupancy.

## PyTorch/RCCL gist comparison

Running `gistfile1.py --side a2a_gemm` with 8 ranks, `N=K=8192`,
`--warmup 10 --iters 30`, and comparing against the final fused generic kernel:

- `M=2048`: gist 0.3530 ms vs fused 0.3125 ms (1.13x, 11.5% lower latency).
- `M=4096`: gist 0.6730 ms vs fused 0.6235 ms (1.08x, 7.4% lower).
- `M=6144`: gist 0.9810 ms vs fused 0.8731 ms (1.12x, 11.0% lower).
- `M=8192`: gist 1.2920 ms vs fused 1.1462 ms (1.13x, 11.3% lower).
- `M=12288`: gist 1.9190 ms vs fused 1.8101 ms (1.06x, 5.7% lower).
- `M=16384`: gist 2.5500 ms vs fused 2.2858 ms (1.12x, 10.4% lower).

The fused kernel is faster on every tested shape, with a 6% to 13% speedup.

## Mode 2 non-fused comparison

Mode 2 runs RCCL all-to-all, `pack_a_shards_kernel`, then the non-persistent
OPUS GEMM. With the same 8-rank generic shapes and `--warmup 10 --iters 30`:

- `M=2048`: Mode 2 0.4371 ms vs fused 0.3125 ms (1.40x, 28.5% lower latency).
- `M=4096`: 0.6918 ms vs 0.6235 ms (1.11x, 9.9% lower).
- `M=6144`: 0.9969 ms vs 0.8731 ms (1.14x, 12.4% lower).
- `M=8192`: 1.3141 ms vs 1.1462 ms (1.15x, 12.8% lower).
- `M=12288`: 1.9403 ms vs 1.8101 ms (1.07x, 6.7% lower).
- `M=16384`: 2.6769 ms vs 2.2858 ms (1.17x, 14.6% lower).

The fused kernel is faster on every tested shape, with a 1.07x to 1.40x
speedup over the in-project non-fused baseline. RCCL communicator creation is
performed after CCO window registration; the reverse order caused ROCr VMM
handle conflicts for the larger receive windows.

## Split LSA communication and compute timing

Mode 3 splits the fused path into an LSA comm-only kernel followed by a
compute-from-recv kernel, with an explicit device synchronization and CCO
barrier between them:

```bash
./build/a2a_gemm_lsa.exe --mode 3 --input-mode generic_a2a
```

With 8 ranks, `N=K=8192`, `--warmup 10 --iters 30`:

- `M=2048`: comm 0.1352 ms, compute 0.1981 ms, split total 0.4253 ms, fused 0.3267 ms.
- `M=6144`: comm 0.3473 ms, compute 0.5567 ms, split total 1.0060 ms, fused 0.8845 ms.
- `M=16384`: comm 0.8510 ms, compute 1.4882 ms, split total 2.4274 ms, fused 2.3144 ms.

Fused reduces split latency by 23.2%, 12.1%, and 4.7% respectively. The
`comm_ms + compute_ms - fused_ms` overlap upper bound is only 0.0066, 0.0195,
and 0.0248 ms (about 1% to 2% of the serial device work). Most of the measured
fused advantage comes from removing the extra kernel transition, device
synchronization, and cross-rank host barrier. This fixed overhead matters most
for the smaller M shape, while its fraction decreases as GEMM work grows.
