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

## Run

Use four visible GPUs, preferably idle ones:

```bash
export HIP_VISIBLE_DEVICES=4,5,6,7
export MORI_SOCKET_IFNAME=enp193s0f0np0
export LD_LIBRARY_PATH=$SP:${LD_LIBRARY_PATH:-}

mpirun --allow-run-as-root -n 4 ./build/quad_lsa_direct.exe \
  -m 2048 -n 18432 -k 8192 --shard-n 2560 --warmup 3 --iters 20
```
