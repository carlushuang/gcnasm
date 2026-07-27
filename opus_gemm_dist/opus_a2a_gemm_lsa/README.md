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
```

`--comm-schedule auto` is the default and selects parallel. `--mode 4` remains
an equivalent shorthand for profiling scripts; modes 0 through 3 retain their
existing behavior.

Each send and receive window contains two slots. A `comm_ready[slot]` event
makes the completed SDMA slot visible to the compute stream, and a
`recv_free[slot]` event prevents the communication stream from overwriting a
slot still consumed by GEMM. Each remote peer has one SDMA queue. The post
kernel submits one PUT per remote peer, and the quiet/notify kernel waits for
that peer queue before incrementing the destination's 64-bit ready counter.
The destination polling kernel acquires every remote source counter for the
slot's target epoch before recording `comm_ready`.
The rank-local shard is copied into the same receive layout with an asynchronous
device-to-device copy.
The two send slots contain different A values and alternate by epoch; validation
checks the corresponding final C values and every per-slot, per-source remote
ready counter, so stale-slot reuse is observable rather than masked by identical
inputs.

The reported `comm_ms` and `compute_ms` are isolated per-epoch measurements.
`pipeline_total_ms` is the end-to-end average across the requested epochs; for
parallel mode it includes fill and drain, so use multiple warm iterations and
at least 30 measured iterations for steady-state comparisons.

On 8 ranks with generic A2A, `N=K=8192`, `K_SHARD=1024`,
`--warmup 10 --iters 30`, the end-to-end results were:

- `M=2048`: fused LSA 0.3375 ms, split LSA 0.4618 ms, RCCL 0.4496 ms,
  SDMA serial 0.3014 ms, SDMA parallel 0.2389 ms.
- `M=6144`: fused LSA 0.8901 ms, split LSA 1.0070 ms, RCCL 1.0393 ms,
  SDMA serial 0.8028 ms, SDMA parallel 0.6183 ms.
- `M=16384`: fused LSA 2.2953 ms, split LSA 2.4239 ms, RCCL 2.7045 ms,
  SDMA serial 2.0471 ms, SDMA parallel 1.5465 ms.

The fused-LSA values are medians of three runs. SDMA parallel reduces latency
by 29.2% to 32.6% versus fused LSA and by 20.7% to 24.5% versus SDMA serial,
passing the retention gate on all three shapes. Mode 4 is retained as an
opt-in backend; mode 0 remains the default because SDMA requires an SDMA-enabled
MORI build and runtime configuration. Broadcast and generic inputs both pass
serial and parallel correctness on 4 and 8 ranks.

The isolated SDMA breakdown for the same three generic shapes was:

- `M=2048`: 0.0909 ms communication and 0.1998 ms compute.
- `M=6144`: 0.2359 ms communication and 0.6018 ms compute.
- `M=16384`: 0.5715 ms communication and 1.5201 ms compute.

The parallel steady state approaches the compute duration because the next
epoch's SDMA transfer is shorter than the current epoch's GEMM.

The clean gfx950 resource build reports:

- `a2a_sdma_post_kernel`: 50 SGPR, 29 VGPR, 0 LDS, no SGPR/VGPR spill,
  8 waves/SIMD.
- `a2a_sdma_quiet_notify_kernel`: 18 SGPR, 10 VGPR, 0 LDS, no spill,
  8 waves/SIMD.
- `a2a_sdma_wait_ready_kernel`: 18 SGPR, 4 VGPR, 0 LDS, no spill,
  8 waves/SIMD.
- reused Mode 2 compute kernel: 76 SGPR, 210 VGPR, 135168 bytes LDS,
  no spill, 2 waves/SIMD.

The symbolized rank-0 timeline is in
`build/traces/sdma_parallel/rank0_results.pftrace`, with its CSV companion in
`rank0_kernel_trace.csv`. It shows the communication kernels on stream 1 and
the Mode 2 GEMM on stream 2. In a steady-state epoch, the next post,
rank-local copy, and quiet/notify interval lies inside the previous GEMM
interval, confirming cross-epoch overlap. The decoded advanced thread trace is
under `build/traces/sdma_parallel_att/` in its generated `ui_output_*`
directory.

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

`COMM_PEER_ORDER=1` (the default) phase-shifts each communication WG's peer
sequence. Concurrent copy WGs therefore write to different peers
instead of all bursting stores to one peer before moving to the next. Build the
legacy order for comparison with:

```bash
make BUILD=build_peer_order0 COMM_PEER_ORDER=0
make BUILD=build_peer_order1 COMM_PEER_ORDER=1
```

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

The final default communication configuration is:

```text
COMM_WG_PLACEMENT=1   # contiguous bx, spread across XCCs
COMM_PEER_ORDER=1     # phase-shift peers across communication WGs
COMM_COPY_MODE=2      # OPUS buffer copy with streaming/cache policy
comm_wgs=16           # two communication WGs per XCC on gfx950
A2A_C_STORE_MODE=2    # pair-coalesced ds_bpermute C store
TILE_READY=0
READY_AWARE_K=0
PRESTORE_BARRIER=1
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
These experiments remain available as compile-time switches but are disabled
by default.

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
