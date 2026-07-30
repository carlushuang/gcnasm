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
slower than inline; rank-aware stagger was mixed below 1%;
`STORE_PIPELINE=2` used 13 spills; and `C_STORE_MODE=1` regressed by roughly
1-3%.

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

An optional direct self-store experiment removes the post-GEMM self-shard D2D
copy by routing `dst == my_rank` C stores directly into the receive layout:

```bash
make BUILD=build_self_direct SDMA_DIRECT_SELF_STORE=1
```

It is disabled by default. With the current single-stream serial path, the
`M=2048,warmup=10,iters=30` three-run median changes versus the normal copy
path ranged from a 0.2% regression to a 1.1% improvement. M=4096/8192 sweeps
were also mixed (roughly -0.7% to +1.3%) with no benefit that consistently grew
with copy size. The local-staging GEMM SGPR spill count increased from 6 to 11,
and chunk-SDMA from 68 to 76, so the experiment remains opt-in.

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
