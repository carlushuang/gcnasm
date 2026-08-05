# opus_gemm_a2a_lsa

Standalone experiment for quad-subtile BF16 GEMM with direct LSA or
double-buffered SDMA all-to-all output.

This directory is extracted from `opus_dist_gemm` and keeps only the direct
quad GEMM epilogue path:

- each rank computes full `M x N` GEMM,
- shard columns `[rank * shard_n, (rank + 1) * shard_n)` are written directly
  into the destination rank's LSA buffer from the GEMM C store path,
- non-scattered tail columns land in a local buffer,
- validation samples each received shard against deterministic per-source-rank
  inputs.

The direct LSA backend remains the default. The SDMA backend first writes one
contiguous local slab per destination, then overlaps the bulk PUT for epoch
`t` with GEMM for independent epoch `t+1`.

## Build

Inside the ROCm/MORI container:

```bash
cd /shared/amdgpu/home/jiahao_zhou_qle/blyu/mori
python3 -m pip install -q .
SP=$(python3 -c 'import mori, os; print(os.path.dirname(mori.__file__))')

cd ../opus_gemm_a2a_lsa
make MORI_LIB_DIR=$SP
```

Build the persistent and non-persistent tile schedulers into separate
directories so Make does not reuse objects compiled with the other mode:

```bash
make BUILD=build_persistent PERSISTENT=1 MORI_LIB_DIR=$SP
make BUILD=build_nonpersistent PERSISTENT=0 MORI_LIB_DIR=$SP
```

## Run

Use four visible GPUs, preferably idle ones:

```bash
export HIP_VISIBLE_DEVICES=4,5,6,7
export MORI_SOCKET_IFNAME=enp193s0f0np0
export LD_LIBRARY_PATH=$SP:${LD_LIBRARY_PATH:-}

mpirun --allow-run-as-root -n 4 ./build/quad_lsa_direct.exe \
  --output-mode direct \
  -m 2048 -n 18432 -k 8192 --shard-n 2560 --warmup 3 --iters 20
```

`--output-mode local` is the isolated GEMM + local compact-store measurement.
`--output-mode split-lsa` is a serial two-kernel baseline: the same
local-staging GEMM writes `[dst,M,shard_n]`, then a 16-byte vectorized LSA copy
kernel writes each slab into `peer_recv[dst][source_rank]`.

```bash
mpirun --allow-run-as-root -n 4 ./build/quad_lsa_direct.exe \
  --output-mode split-lsa \
  -m 2048 -n 18432 -k 8192 --shard-n 2560 --warmup 10 --iters 30
```

At M=2048, three-run max-rank medians were 0.6147 ms Direct LSA,
0.7184 ms Split LSA, and 0.6855 ms standard SDMA serial. The split path is
16.9% slower than fused Direct LSA and 4.8% slower than SDMA serial, but
provides an isolated LSA communication baseline. The copy kernel uses
29 SGPR, 10 VGPR, no LDS/spill, and averages about 0.200 ms in the rank-0
system trace:
`build/traces/split_lsa_M2048/rank0_results.pftrace`.
A correctness-checked seven-shape sweep (`warmup=10,iters=30`) measured
max-rank latencies of 0.4463/0.7203/1.3029/1.8657/2.6315/3.7169/5.1608 ms
for M=1024/2048/4096/6144/8192/12288/16384.

## 8-rank Direct LSA scheduling

At 8 ranks, the Direct kernel uses a separate compile-time instance for
M>=8192. It processes two M tiles before rotating destination peer, rotates
the peer order by source rank, and inlines the uniform tile decode. Smaller
8-rank shapes and all 4-rank runs retain the original kernel.

Five-run baseline and optimized max-rank medians
(`warmup=10,iters=50`) were:

- `M=8192`: 2.3015 -> 1.9018 ms (17.4% lower), Split LSA 2.5070 ms.
- `M=12288`: 3.6243 -> 2.7721 ms (23.5% lower), Split LSA 3.5583 ms.
- `M=16384`: 5.0546 -> 3.6959 ms (26.9% lower), Split LSA 4.8596 ms.

The 4-rank M=2048/4096/8192 regression panel changed from
0.6128/1.0551/1.9937 ms to 0.6116/1.0526/1.9971 ms; the worst change was a
0.2% regression. The original Direct instance uses 226 VGPR and six SGPR
spills, while the 8-rank striped instance uses 226 VGPR and 16 SGPR spills,
and retains two waves/SIMD. Optimized rank-0 traces are under
`build/traces/direct_ab_8rank/M{8192,16384}/inline/`.

Rejected experiments are retained as results: source-rank rotation without
M striping was <=1%; outlined stripe decode removed spills but was 12-15%
slower than inline; and rank-aware stagger was mixed below 1%.

The removed store-pipeline and C-store branches had these results:

- Store-pipeline mode 0 was the original post-compute store with a tail
  barrier. Mode 1 moved the first two half-tile stores into the final compute
  sequence. Neither beat mode 3, which stores all four half-tiles together and
  drops the unnecessary tail barrier. The old per-shape mode 0/1 numbers were
  not retained.
- Store-pipeline mode 2 prefetched the next persistent tile before storing the
  current C tile, but raised the optimized Direct kernel to 13 SGPR spills and
  failed the resource gate.
- C-store mode 0 was the original MFMA-layout store. Mode 1 staged C through
  LDS for coalesced writes but regressed by roughly 1-3%. Mode 2's wave-local
  `ds_bpermute` pair-coalesced store remained the winner.

The implementation now directly contains the retained mode-3 pipeline and
mode-2 C store; the historical compile-time branches have been removed.
The optimized peer round-robin, chunk-contiguous, and Direct stripe tile
mapping is also unconditional; the old `TILE_ORDER=0` baseline switch has been
removed.

The original `16 phases x 4 delay` store stagger was re-tested before removal.
Five-run max-rank medians with stagger on versus off
(`warmup=10,iters=30`) were:

- 4-rank `M=2048,shard_n=2560`: 0.6412 vs 0.6273 ms (off 2.2% faster).
- 4-rank `M=8192,shard_n=2560`: 2.0092 vs 2.0116 ms (off 0.1% slower).
- 8-rank `M=8192,shard_n=2304`: 1.9231 vs 1.9152 ms (off 0.4% faster).
- 8-rank `M=16384,shard_n=2304`: 3.6983 vs 3.7041 ms (off 0.2% slower).

Stagger had no stable benefit and materially regressed the smallest shape, so
the delay loop and both build switches were removed. This also reduced Direct
SGPR spills from 6 to 4 and striped Direct spills from 16 to 14 without
changing occupancy.

The runtime override `--fused-lsa-stripe auto|0|1` isolates this scheduling
effect without rebuilding. A 2026-07-31 three-run ablation
(`warmup=10,iters=30`) found:

- At 8-rank full-N (`shard_n=2304`), stripe reduced latency by
  0.8/1.5/4.9/12.9/17.3/23.8/27.0% for
  M=1024/2048/4096/6144/8192/12288/16384. Forced stripe made Direct/Fused LSA
  faster than Fused SDMA on all seven shapes.
- At 4-rank full-N (`shard_n=4608`), stripe gains were
  4.6/0.0/-0.8/4.0/6.7/12.6/15.5%. Direct/Fused LSA won through M=4096,
  while Fused SDMA won for M>=6144.
- At the original 4-rank partial-N configuration (`shard_n=2560`), stripe was
  weaker and Fused SDMA remained faster on all seven shapes.

This confirms that both the 8-rank stripe specialization and the mismatched
4/8-rank communication range affected the earlier mode ordering. With full-N,
4 ranks send to 3 peers at 4608 columns each while 8 ranks send to 7 peers at
2304 columns each; the wider fanout is also consistent with better distribution
of rank-rotated direct stores.

For the SDMA pipeline, set the transport variables before MORI initialization:

```bash
export HIP_VISIBLE_DEVICES=0,1,2,3
export MORI_ENABLE_SDMA=1
export MORI_SDMA_NUM_CHANNELS=1

mpirun --allow-run-as-root -n 4 ./build/quad_lsa_direct.exe \
  --output-mode sdma \
  --comm-schedule parallel \
  -m 2048 -n 18432 -k 8192 --shard-n 2560 --warmup 5 --iters 100
```

Both SDMA modes accept `--comm-schedule serial|parallel|auto`. `auto` preserves
the historical defaults: standard SDMA is parallel across epochs, while
chunk-SDMA is serial. Parallel chunk-SDMA alternates the two staging slots so
epoch `t` communication can overlap epoch `t+1` compute.

Serial schedules use one HIP stream: memset, GEMM, SDMA post or
in-kernel chunk PUTs, self-copy, and quiet/notify are submitted in order without
cross-stream events.

At `M=2048,warmup=10,iters=30`, three-run max-rank medians improved from
0.7125 to 0.6861 ms for standard SDMA serial and from 0.6149 to 0.5942 ms for
Chunk-SDMA serial (3.7% and 3.4%). Symbolized traces are under
`build/traces/sdma_serial_single_stream_M2048/` and
`build/traces/chunk_serial_single_stream_M2048/`; all timed dispatches use the
same stream.

An experimental standard-SDMA post path can split each peer's completed
`[M, shard_n]` slab into multiple M-chunk PUTs without changing the GEMM:

```bash
# 0 is the default one-bulk-PUT path; auto uses up to eight M tiles per PUT.
./build/quad_lsa_direct.exe --output-mode sdma --comm-schedule serial \
  --sdma-post-m-tiles auto
```

This path uses one posting lane per peer and serially submits disjoint chunks
to queue 0 before the existing quiet/notify kernel. The chunked post kernel
uses 47 SGPR and 42 VGPR, with no spill and eight waves/SIMD, versus
29 SGPR and 32 VGPR for the bulk post kernel. Three-run 4/8-rank sweeps found
no end-to-end benefit: auto chunking averaged 0.37%/0.34% higher latency, and
one-M-tile PUTs regressed representative large shapes by roughly 3.5%–5.1%.
The default therefore remains one bulk PUT per remote peer.

Direct/Fused LSA, Split LSA, Split SDMA, Split SDMA v2, and Fused SDMA serial
now also report strict `critical_*` phase fields. Four HIP events bracket
compute and communication in every measured epoch; rank 0 gathers all ranks,
selects the largest-E2E rank, and reports that rank's compute, communication,
and `barrier_idle_residual_ms`. The residual is calculated as
`critical_e2e_ms - critical_compute_ms - critical_comm_ms`. Report tables use
the complete record from the three-run median-E2E run instead of mixing
independently reduced max-rank phases or rocprof traces. Event instrumentation
raises absolute latency, so these values are for decomposition rather than the
low-overhead headline comparison.

Pass `--strict-timing 0` to replace the four per-epoch phase events with one
start/stop pair around the full measured loop. On the 2026-08-05 exclusive-GPU
M=1024 retest, strict timing added 17.4-18.7 us for 4-rank bulk/auto/chunk SDMA
and 9.5-17.8 us for the corresponding 8-rank paths. The low-overhead medians
matched the report values within 0.5%; use this mode for headline E2E ranking
and strict timing only for phase decomposition.

A removed direct self-store experiment routed `dst == my_rank` C stores
directly into the receive layout to eliminate the post-GEMM self-shard D2D
copy. With the single-stream serial path,
`M=2048,warmup=10,iters=30` three-run median changes versus the normal copy
path ranged from a 0.2% regression to a 1.1% improvement. M=4096/8192 sweeps
were also mixed (roughly -0.7% to +1.3%) with no benefit that consistently grew
with copy size. At the time of the experiment, the local-staging GEMM SGPR
spill count increased from 6 to 11 and chunk-SDMA from 68 to 76. The code
branch and its `SDMA_DIRECT_SELF_STORE` build switch have been removed; this
record is retained for reference.

The 2026-08-05 branch cleanup fixed the retained store pipeline, C-store, and
tile-order implementations directly in the kernel; removed the
direct-self-store argument; and removed store stagger after the A/B above. A
clean gfx950 resource build reports 106 SGPR / 226 VGPR for all four GEMM
instances: Direct uses 4 SGPR spills, striped Direct uses 14, local staging uses
6, and chunk-SDMA uses 48. All retain zero VGPR spill, zero scratch,
135172-byte LDS, and two waves/SIMD.

Three-run max-rank medians before and after cleanup
(`warmup=10,iters=30`) were:

- 4-rank `M=2048,shard_n=2560`: Direct 0.6402 -> 0.6448 ms,
  standard SDMA serial 0.6994 -> 0.6996 ms, and chunk-SDMA serial
  0.5776 -> 0.5782 ms.
- 8-rank `M=8192,shard_n=2304`: Direct 1.9339 -> 1.9298 ms,
  standard SDMA serial 2.4789 -> 2.4758 ms, and chunk-SDMA serial
  1.9397 -> 1.9396 ms.
- 8-rank `M=16384,shard_n=2304`: Direct 3.7178 -> 3.7222 ms,
  standard SDMA serial 4.7399 -> 4.7519 ms, and chunk-SDMA serial
  3.7677 -> 3.7618 ms.

The largest change was a 0.72% regression at the smallest Direct shape; every
other result changed by at most 0.25%. Local, split-LSA, standard/chunk SDMA
serial, and standard/chunk SDMA parallel correctness checks all passed.

The current MORI SDMA setup assumes local rank `r` is bound to visible device
ordinal `r`; the SDMA results below were collected on physical GPUs 0–3.
`02_gda_put.cpp` is an IBGDA/RDMA example, not the SDMA API used here.

Build and run the isolated 3-peer, 10 MiB-per-peer benchmark with:

```bash
cmake -S /workspace/mori -B /workspace/mori/build -DBUILD_CCO_SDMA=ON
cmake --build /workspace/mori/build --target mori_cco

make sdma_bench
mpirun --allow-run-as-root -n 4 ./build/sdma_a2a_bench.exe \
  --bytes-per-peer 10485760 --warmup 5 --iters 100
```

The SDMA path uses registered CCO windows, `ccoSdma::put`, and a quiet/notify
kernel; it does not maintain a manual IPC peer-pointer table.

## SDMA isolation and pipeline results

For `M=2048, N=18432, K=8192, shard_n=2560`, three alternating
`--warmup 5 --iters 100` runs produced these median max-rank values:

- Direct LSA, `Tlsa`: 0.6136 ms.
- GEMM plus local compact store, `Tlocal`: 0.5033 ms.
- Three concurrent 10 MiB CCO SDMA PUTs, `Tsdma`: 0.1834 ms, or about
  159.8 GiB/s per rank, including queue quiet and ready notification.

Thus `max(Tlocal,Tsdma)=0.5033 ms`, an 18.0% predicted reduction from direct
LSA, passed the 3% integration gate. Direct and local GEMM variants both use
106 SGPRs, 226 VGPRs, six SGPR spills, two waves/SIMD, and no VGPR spills.

The integrated path uses two uncached staging buffers, separate compute and
communication streams, an SDMA completion signal, and only waits before a
staging slot is reused. Alternating three-run medians were:

- `M=1024`: direct 0.4004 ms, CCO SDMA 0.3371 ms.
- `M=2048`: direct 0.6167 ms, CCO SDMA 0.5204 ms (15.6% lower).
- `M=4096`: direct 1.0586 ms, CCO SDMA 0.9032 ms.

All runs passed receive-layout and tail correctness. SDMA validation alternates
two distinct A inputs by epoch so stale or out-of-order results cannot pass.
The reported latency includes pipeline fill/drain amortized over 100 epochs;
very short runs can be slower than direct LSA because that fixed cost is not
hidden.

## Experimental chunk-fused SDMA

`--output-mode chunk-sdma` completes one M chunk across all of a peer's N
tiles, then submits one CCO SDMA PUT while later chunks continue computing.
The tile order rotates peers by source rank to avoid synchronized incast.
The default groups up to eight 256-row tiles per PUT (four for M=1024);
`--chunk-m-tiles` can still override it.

```bash
mpirun --allow-run-as-root -n 8 ./build/quad_lsa_direct.exe \
  --output-mode chunk-sdma \
  --comm-schedule serial \
  -m 2048 -n 18432 -k 8192 --shard-n 2304 --warmup 10 --iters 50
```

At 8 ranks, a five-run `chunk_m_tiles=1/8` sweep (`warmup=10,iters=50`)
reduced max-rank latency by 5.2%/3.8%/6.0% at M=2048/4096/8192.
Chunk-aware ordering then made each chunk ready earlier. The final three-run
max-rank medians versus Standard SDMA serial were:

- `M=1024`: 0.4264 vs 0.4136 ms (Chunk 3.1% slower).
- `M=2048`: 0.7034 vs 0.6829 ms (Chunk 3.0% slower).
- `M=4096`: 1.1002 vs 1.2058 ms (Chunk 8.8% faster).
- `M=6144`: 1.5037 vs 1.7433 ms (Chunk 13.7% faster).
- `M=8192`: 1.9290 vs 2.4566 ms (Chunk 21.5% faster).
- `M=12288`: 2.8674 vs 3.4557 ms (Chunk 17.0% faster).
- `M=16384`: 3.7699 vs 4.7017 ms (Chunk 19.8% faster).

The 4-rank regression panel also improved: M=2048/4096/8192 changed from
0.5936/0.9999/1.8570 ms with one-tile PUTs to
0.5684/0.9353/1.7450 ms with the optimized default. The final chunk kernel
uses 106 SGPR, 226 VGPR, 48 SGPR spills, no VGPR spills, and two waves/SIMD
with the current MORI CCO headers.
At 8-rank M=16384, rank-0 traces show the fused kernel falling from 4.551 to
3.886 ms and quiet/notify from 1.086 to 0.129 ms. Optimized traces are under
`build/traces/serial_breakdown_8rank/M{2048,16384}/chunk-sdma_optimized/`.

Three rejected codegen/synchronization experiments are retained as results:
removing the post-submit barrier deadlocked and raised spills to 78; inlining
the CCO submit raised spills to 99; replacing CCO rank lookups with kernel
arguments raised spills to 72 without a measurable latency gain.

The current MORI SDMA API drains queues from rptr/wptr state and no longer
requires caller-owned expected-signal counters. GEMM->A2A therefore uses
no-signal PUTs and relies on `quietQueue` plus its explicit ready window.
Compared with the previous MORI build, the post kernel changed from
50 SGPR/29 VGPR to 29 SGPR/32 VGPR, quiet/notify from 18 to 14 SGPR, and the
chunk kernel from 69 to 48 SGPR spills. Three-run M=2048/M=8192 integrated
latencies stayed within about 1%, while isolated 4-rank 10 MiB and 8-rank
9 MiB transfers measured 0.1834/0.1682 ms.

An experimental fused-quiet implementation let the last remote chunk submitter
quiet all peer queues and publish ready counters inside the GEMM kernel,
removing the standalone quiet/notify dispatch. At
`M=2048,warmup=10,iters=30`, three-run max-rank medians regressed from
0.5952 to 0.6059 ms in the current single-stream serial mode and from
0.5550 to 0.6143 ms in parallel mode (1.8% and 10.7%). The chunk kernel SGPR
spill count increased from 68 to 71. The quiet
duration is mostly SDMA completion latency, and moving it into the GEMM adds
resource pressure without enough remaining compute to hide it. The experiment
was removed; these results are retained for reference.

## Persistent tail-balance sweep

With `M=2048`, `K=8192`, four ranks, and 256 CUs, varying N changes the
remainder after full 256-CTA scheduling batches. A stable run with
`--warmup 5 --iters 100` measured:

- 512 tiles, remainder 0: 0.5207 ms non-persistent vs 0.5198 ms persistent (+0.17%).
- 520 tiles, remainder 8: 0.6187 ms vs 0.5626 ms (+9.97%).
- 544 tiles, remainder 32: 0.6264 ms vs 0.5778 ms (+8.41%).
- 576 tiles, remainder 64: 0.6294 ms vs 0.6051 ms (+4.02%).
- 640 tiles, remainder 128: 0.6417 ms vs 0.6391 ms (+0.41%).
- 704 tiles, remainder 192: 0.6675 ms vs 0.6624 ms (+0.77%).
- 760 tiles, remainder 248: 0.6953 ms vs 0.6827 ms (+1.85%).
- 768 tiles, remainder 0: 0.7010 ms vs 0.6912 ms (+1.42%).

The strongest persistent benefit occurs just after an exact scheduling batch:
only a small subset of CUs would receive an additional static CTA, creating a
long tail. Dynamic tile assignment lets the first available CUs consume those
remaining tiles. As the remainder approaches a full 256-CTA batch, the static
work becomes more evenly distributed and the advantage mostly disappears.
