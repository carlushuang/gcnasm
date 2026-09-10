# MXFP8 GEMM：原始 row-major scale

AMD gfx950 / MI355X 上的批量 `C = A × Bᵀ`，输出 FP32。当前方案为 `refill_split_gather`：一次 GEMM kernel，直接读取原始 row-major scale，不需要 host 重排或额外全局 scale workspace。

完整优化过程、scale 路径、性能口径、正确性及失败实验统一保存在 [OPTIMIZATION_LOG.md](OPTIMIZATION_LOG.md)。

2026-09-10 10:37 UTC，在 MI355X / gfx950 的 HIP2（PCI `0000:65:00.0`）上复测：8192³、b=1、w=200、i=100、8-wave，FP32 **0.3826 ms / 2.87361 PFLOPS**。这是一次运行中100次 launch 的平均值，原始输出及源码/二进制哈希见 [同轮对照结果](../mxfp8_gemm_16x16x128_blockscale_bpreshuffle/results/gpu2_20260910_103740/results.json)。

## 文件

| 文件 | 用途 |
|---|---|
| `gemm_a8w8_mxfp8_scale_kernel_template.hpp` | 正式 GEMM 实现和 scale 加载、打包、LDS 发布路径 |
| `gemm_a8w8_mxfp8_scale_kernel.cc` | 每 workgroup 输出 4/1 个 tile 的两个编译实例；一次 launch 只选择其一 |
| `gemm_a8w8_mxfp8_scale_common.h` | 参数结构、tile 配置和编译常量 |
| `gemm_a8w8_mxfp8_scale_host.cc` | 初始化、原始 scale 上传、launch、计时和可选 CPU 参考验证 |
| `Makefile` | 编译、链接产物检查、性能测试和验证 |

## 输入约定

- A：FP8 E4M3，`[batch, M, K]`；B：FP8 E4M3，`[batch, N, K]`。
- SFA：E8M0 字节，`[batch, M, K/32]`；SFB：`[batch, N, K/32]`，均为 row-major。
- 每连续 32 个 K 元素共用一个 scale。C 为 FP32，`[batch, M, N]`。
- 随附 launcher 要求 M/N 为 256 的整数倍，K 为 128 的整数倍，维度和 batch 为正。

## 编译与检查

依赖 ROCm、Opus headers 和支持该 kernel 展开的 clang 23。Makefile 默认使用本机 `/root/workspace/llvm-src/build/bin/clang++`，可用 `HIPCC` 或 `CLANG23_ROOT` 覆盖。

```bash
make -j2 OPUS_INCLUDE_DIR=/root/workspace/aiter/csrc/include
make check regs
```

检查从最终 exe 提取设备镜像，核对 640 条 MFMA、零 `s_setprio` 并输出寄存器/LDS 信息。检查不会改写 exe。预期 persistent/single 为 VGPR 253/238、SGPR 104/75、LDS 139264 字节，spill 和 scratch 为零。

## 运行

```bash
HIP_VISIBLE_DEVICES=2 OMP_NUM_THREADS=32 ./build/gemm_a8w8_mxfp8_scale.exe \
  -m 8192 -n 8192 -k 8192 -b 1 -w 200 -i 100 -v 0
```

`HIP_VISIBLE_DEVICES` 使用 HIP 编号。本机已核对 SMI5 对应 HIP7，SMI2 对应 HIP2。上面的编号是示例，按目标卡选择。

也可使用 `make benchmark GPU=2`。`-v 1` 开启 CPU 参考检查；大矩阵的 CPU 检查会额外耗时。命令行支持 `-m/--m`、`-n/--n`、`-k/--k`、`-b/--b`、`-w/--warmup`、`-i/--iterations`、`-v/--verify`，以及 `--flag=value` 写法。性能计时只包含 GEMM，不含输入生成、上传和 CPU 验证。

当前 standalone 验证器在数值失败时只打印结果，仍返回0，且误差比较未显式拒绝 NaN；不能只用 `make verify` 的退出码判断正确性。优化记录中的历史强化验证使用了单独的测试 host，其 `/tmp` 证据属于本地归档。

host/device repack、timeline 和输入诊断实验入口已移除。`make clean` 删除构建产物；崩溃转储、profiler 临时文件和重复脚本不包含在整理后的代码目录中。
