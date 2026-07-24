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
For the SDMA pipeline, set the transport variables before MORI initialization:

```bash
export HIP_VISIBLE_DEVICES=0,1,2,3
export MORI_ENABLE_SDMA=1
export MORI_SDMA_NUM_CHANNELS=1

mpirun --allow-run-as-root -n 4 ./build/quad_lsa_direct.exe \
  --output-mode sdma \
  -m 2048 -n 18432 -k 8192 --shard-n 2560 --warmup 5 --iters 100
```

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
- Three concurrent 10 MiB CCO SDMA PUTs, `Tsdma`: 0.1851 ms, or about
  158.2 GiB/s per rank, including sender and receiver completion signals.

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

`--output-mode chunk-sdma` submits one 1.25 MiB CCO SDMA PUT after the ten
`256x256` output tiles for a `(destination, M-tile)` chunk are locally stored.
It targets single-round latency and remains experimental:

```bash
mpirun --allow-run-as-root -n 4 ./build/quad_lsa_direct.exe \
  --output-mode chunk-sdma \
  -m 2048 -n 18432 -k 8192 --shard-n 2560 --warmup 0 --iters 1
```

Five-process-run single-round max-rank medians were:

- `M=1024`: direct 1.1972 ms, post SDMA 1.1661 ms, chunk SDMA 1.0430 ms.
- `M=2048`: direct 1.6418 ms, post SDMA 1.3942 ms, chunk SDMA 1.4760 ms.
- `M=4096`: direct 2.1058 ms, post SDMA 1.9473 ms, chunk SDMA 1.7362 ms.

The mode improves single-round latency for `M=1024/4096`, but it should not be
used for steady state: at `M=2048,warmup=5,iters=100` it measured about
0.609 ms versus 0.519 ms for the normal double-buffered SDMA path. Inlining
chunk submission also raises the chunk kernel to 61 SGPR spills, so `sdma`
remains the recommended mode.

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
