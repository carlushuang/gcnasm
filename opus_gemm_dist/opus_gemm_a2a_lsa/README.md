# opus_gemm_a2a_lsa

Standalone experiment for quad-subtile BF16 GEMM with direct LSA all-to-all output.

This directory is extracted from `opus_dist_gemm` and keeps only the direct
quad GEMM epilogue path:

- each rank computes full `M x N` GEMM,
- shard columns `[rank * shard_n, (rank + 1) * shard_n)` are written directly
  into the destination rank's LSA buffer from the GEMM C store path,
- non-scattered tail columns land in a local buffer,
- validation samples each received shard against deterministic per-source-rank
  inputs.

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
  -m 2048 -n 18432 -k 8192 --shard-n 2560 --warmup 3 --iters 20
```

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
