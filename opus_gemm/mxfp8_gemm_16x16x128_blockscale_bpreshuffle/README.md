# gfx950 MXFP8 blockscale bpreshuffle

基于 `../mxfp8_gemm_16x16x128_scale_rowmajor` 的独立适配版本。原目录没有修改。

本目录实现 **A 保持 row-major、B 使用 AITER 标准 `(16,16)` preshuffle、外部 scale 保持 1×128 / 128×128、GEMM 内广播到硬件 K32 分组**。计算仍是单个 GEMM kernel，无外部 scale 展开、scale 转换 kernel 或额外全局 scale workspace。

已完成独立程序和 Python 接口的 GPU 数值验证，包括完整 8192³。本目录提供可直接调用的 eager Python 接口，**尚未注册到 AITER 自带 dispatcher**。当前采用 `scale_publish_fused`：在 **SMI2/HIP2、PCI 0000:65:00.0** 上，8192³、b=1、w=200、i=100，最后三轮交错复测中位数为 BF16 **0.3683 ms / 2.98501 P**、FP32 **0.3774 ms / 2.91306 P**；尚未稳定达到 3P。完整优化过程及 AITER 接入说明见 [OPTIMIZATION_LOG.md](OPTIMIZATION_LOG.md)。未采用的候选和原始测试记录保存在 `/tmp`。

2026-09-10 10:37 UTC 的发布前单轮复测：同一张 GPU2、相同尺寸及 b/w/i，FP32 **0.3774 ms / 2.91313 P**，BF16 **0.3704 ms / 2.96827 P**；同轮 rowmajor FP32 为 **0.3826 ms / 2.87361 P**。这组结果与上面的多轮中位数分别记录，原始数据见 [results.json](results/gpu2_20260910_103740/results.json)。

## 输入契约

计算 `C[M,N] = dequant(A[M,K]) @ dequant(B[N,K]).T`。

| 输入 | 数据类型 | 量化粒度 | 物理布局 |
|---|---|---|---|
| A | FP8 E4M3FN | — | 连续 row-major `[M,K]`，不重排 |
| B | FP8 E4M3FN 字节 | — | `aiter.ops.shuffle.shuffle_weight(B, layout=(16,16))`，元数据仍为 `[N,K]` |
| A_scale / SFA | E8M0 字节 | 1×128 | 逻辑 `[M,K/128]`，物理连续 `[K/128,M]`，列主序 |
| B_scale / SFB | E8M0 字节 | 128×128 | 连续 row-major `[N/128,K/128]` |
| C | BF16 或 FP32 | — | 连续 row-major `[M,N]`，内部 FP32 累加 |

“分组粒度”和“物理布局”是两个不同要求。SFA 必须已经具有 blockscale bpreshuffle 使用的列主序字节顺序。Python 接口接受两种元数据形式：

- 真正的列主序 `[M,K/128]`，stride 为 `(1,M)`。
- AITER 的约定：元数据为连续 `[M,K/128]`，底层字节实际按 `[K/128,M]` 存放。

第二种形式无法仅从 tensor 的 shape/stride 判断正确性，调用方必须保证其字节含义。普通行主序 `[M,K/128]` 的 scale 不能直接冒充这一格式。接口内部不调用 `transpose/contiguous/to` 修改输入；应由上游量化器直接生成所需布局。

**scale 必须已经是 E8M0。** 任意 FP32 blockscale 不是 E8M0 的另一种存储方式，不能仅靠字节打包获得相同数学结果。Python 接口会拒绝 FP32 scale。本实现不调用 `fp8_legacy_to_mxfp8`，也不重新量化 A/B。

原生 launcher 和独立程序支持 dense batch；Python 接口当前为 2-D GEMM。限制：gfx950，M/N 为 256 的倍数，K 为 128 的倍数；每个输入/输出分配的总字节数不超过有符号 32 位寻址范围。当前没有 M/N/K padding 或 FP16 输出分支。

## scale 加载和消费

保持原有路径：

```text
紧凑 E8M0 SFA/SFB
    → gsf：每个 K128 读取 compact scale byte
    → ssf：A 直接写 raw byte；B 复制同一 byte 并写 packed dword
    → rsfa：TR8 LDS 读取形成四个 M repeat 的 packed dword
    → rsfb：沿用原 LDS packed dword 读取
    → scaled MFMA：继续用 op_sel 选择 packed byte
```

对一个 256×256×128 tile：

1. 四个 A producer wave 合计读取 256 个不同的 A scale。每个 wave 覆盖 4 个 M repeat，每个 repeat 16 行；生产者 lane `call*16+r` 读取 `row + wave_m*16 + call*64 + r` 对应的 scale。
2. A 直接用 `ds_write_b8` 写入 `stage*1024 + wave_m*64 + lane`。RSFA 用 `ds_read_b64_tr_b8`，地址为 `stage*1024 + wave_m*64 + 8*(lane&15)`；返回的低 32 bit 为 `[s(row0), s(row0+64), s(row0+128), s(row0+192)]`，高 32 bit 丢弃。四个 K32 lane 组得到相同的 word。因此一个 K128 scale 在四个硬件 K32 子块上重复使用，A 路径已无 wave shuffle。
3. B 的 N256 tile 只有两个不同的 N128 block scale。每个 half 对应的字节 `s` 打包为 `uint32(s) * 0x01010101u`，供该 half 的所有 N repeat 和 K32 组使用。当前 B producer lane 会重复读相同地址；“两个”指不同的 scale 字节数。
4. B 的 `ssf` 保留 identity 映射，读取两个 B half 的 `ds_read2st64_b32` 也保留。A 的 `rsfa` 地址必须有 `8*(lane&15)`，不能给所有 lane 同一地址；已经用独立 GPU probe 验证实际 TR8 行为。
5. **`op_sel` 保留。A 的四个 byte 对应四个 M repeat，并非四个 K32 分组；K32 广播由四组 lane 使用同一个 TR8 地址模式实现。** B 的四个 byte 相同，沿用现有 selector 也能得到正确结果。

LDS 中保留原来的两阶段分配：SFA 2 KiB、SFB 2 KiB。每个 A stage 实际仅写入 256 bytes，TR8 未使用的高 dword 读取仍在该 stage 分配范围内。它属于单个 workgroup 的共享内存；没有新增全局 scale buffer。

## 相对 rowmajor 的改动

主要改动如下：

- 外部 scale 参数设为 `(GROUP_M,GROUP_N,GROUP_K)=(1,128,128)`，另外保留硬件 `MFMA_SCALE_GROUP_K=32`。不能只把旧的 `GROUP_K=32` 改为 128，否则内部 lane 布局也会被错误改变。
- 重写 B 的全局读取及 LDS consumer 地址：一个 wave 连续读取 1024 bytes packed B，覆盖 8 条 128B cache line。最初只改 global 地址但沿用旧 lane 分工，会覆盖 32 条 line；正式版本已修正。SB 写入方式及分配不变，但 LDS 中 B 的逻辑排列与 RB 地址一起改变。A 的数据加载路径保留。
- 用每 K128 一次的 compact scale producer 替换原 row-major 1×32 的 q0/q1 gather/refill 队列，重写 SFA/SFB 地址和 `ssf` 映射；RSFA 改用 TR8 直接形成打包结果，RSFB 和 `op_sel` 保留。
- 下一轮 scale load 提前到主循环开头；在前 20 条 MFMA 后统一等待 VMEM 完成，再由 A producer 写 raw byte、B producer 打包后写 dword。删除原来前 16 条 MFMA 后的独立 prepare/等待分支，保留发布前的 LGKM 等待和 workgroup barrier。
- 保留 256×256×128 tile、8 wave、1/4 输出 tile 分支和 persistent handoff。增加 BF16 输出；FP32 输出用于调试和验证。
- 独立程序直接生成 compact scale 的目标布局，并用原始 B 计算 CPU reference。新增 C ABI launcher 和 Python adapter，便于直接接收 AITER 风格输入。
- BF16 输出使用默认 store policy，FP32 保留 `nt`。两者仍是 FP32 累加；BF16 多一次输出转换。单独切换 BF16 的 store 提示就测到了明显收益，不能把初版的 BF16 性能差距理解为 BF16 计算吞吐更低。

## 代码入口和构建

- `gemm_a8w8_mxfp8_scale_kernel_template.hpp`：B 全局读取、scale 路径、MFMA 和输出。
- `gemm_a8w8_mxfp8_scale_common.h`：参数 ABI、输入分组和硬件分组定义。
- `gemm_a8w8_mxfp8_scale_kernel.cc`：FP32/BF16 × 1/4 tile 的四个实例。
- `gemm_a8w8_mxfp8_scale_host.cc`：输入生成、B preshuffle、独立 CPU reference 和计时。
- `gemm_a8w8_blockscale_bpreshuffle_launch.cc`：无分配的单 GEMM C ABI launcher。
- `blockscale_bpreshuffle.py`：eager PyTorch adapter，支持 `out=` 和当前 stream。
- `test_blockscale_bpreshuffle.py`：使用 AITER 实际 `shuffle_weight` 的 GPU 集成验证。

```bash
make -j3
make check regs
make verify GPU=2
HIP_VISIBLE_DEVICES=2 python test_blockscale_bpreshuffle.py
HIP_VISIBLE_DEVICES=2 python test_blockscale_bpreshuffle.py --large
```

默认使用本机 clang 23 和 `/root/workspace/aiter/csrc/include`。可通过 `CLANG23_ROOT`、`HIPCC`、`OPUS_INCLUDE_DIR`、`ROCM_PATH` 覆盖。旧编译器可能不能充分展开 K 循环，`make check` 会检查 MFMA 数量。

产物：`build/gemm_a8w8_blockscale_bpreshuffle.exe` 和 `build/libblockscale_bpreshuffle.so`。

独立程序的 `--tiles 0|1|4` 分别表示自动选择、单输出 tile、最多四个相邻 M tile；`--dtype bf16|fp32` 默认 BF16。短验证例子：

```bash
HIP_VISIBLE_DEVICES=2 OMP_NUM_THREADS=16 ./build/gemm_a8w8_blockscale_bpreshuffle.exe \
    -m 1280 -n 512 -k 1152 -b 2 -v 1 -w 0 -i 1 --dtype fp32 --tiles 4
```

以后测 8192³ 时可以使用 `make benchmark GPU=2`，固定 `b=1,w=200,i=100`。计时只覆盖 GEMM，B preshuffle 在计时之外。

## Python 调用及 AITER 接入位置

将本目录加入 Python 搜索路径后，调用方式为：

```python
import torch
from aiter.ops.shuffle import shuffle_weight
from blockscale_bpreshuffle import gemm_a8w8_blockscale_bpreshuffle

# A: FP8 [M,K]；W: 原始 FP8 [N,K]
# A_scale/B_scale 已由上游量化器按上面的 E8M0 布局生成。
# B weight 通常在模型加载时预排好，推理时重复使用。
B_packed = shuffle_weight(W, layout=(16, 16))
C = gemm_a8w8_blockscale_bpreshuffle(
    A, B_packed, A_scale, B_scale, dtype=torch.bfloat16, out=None)
```

适配器只分配必要的输出 tensor；传入 `out` 时不分配输出。输出不能与任何输入的存储区间重叠。输入不作量化、重排、展开或类型转换。C launcher 按输入尺寸和输出类型选一个 kernel，使用调用方的 HIP stream。

AITER 上游 `gemm_a8w8_blockscale_bpreshuffle` 的 gfx950 普通路径会把 E8M0 scale 转到 FP32 再进入常规 backend。将本实现正式接入时，应在该转换之前加入 gfx950 + E8M0 + 支持尺寸/BF16 的分支，并转到本 kernel；其他输入继续原分派。仅放置本目录不会让 AITER 自动调用它。本次没有修改 AITER 工作树或注册全局 monkey patch。

上游流程核对基于提交 `12620102e523c4688e9ecd466499501086ad48c0`，源码快照在 `/tmp/aiter_bpreshuffle_review_12620102e523`。本机 Python 集成测试实际导入的是 `/sgl-workspace/aiter/aiter/ops/shuffle.py`，其默认 `(16,16)` FP8 排列相同。

## 2026-09-10 验证记录

最初验证 GPU：AMD Instinct MI355X / gfx950，HIP7，即 SMI5 / PCI `0000:95:00.0`。后续优化、正式 `scale_publish_fused` 的完整 Python 8192³ 验证及复测改用 **HIP2 / SMI2 / PCI `0000:65:00.0`**。

独立程序以下 6 种配置，分别验证 BF16 和 FP32，共 12 组全部通过；每个 batch 的所有输出均与基于 raw A/B 和 compact scales 的 double CPU reference 比较：

| M | N | K | batch | output tiles/WG |
|---:|---:|---:|---:|---:|
| 256 | 256 | 128 | 1 | 1 |
| 256 | 512 | 256 | 2 | 1 |
| 256 | 512 | 1152 | 1 | 1 |
| 1024 | 256 | 128 | 1 | 4 |
| 1280 | 512 | 256 | 2 | 4 |
| 1280 | 512 | 1152 | 1 | 4 |

输入 scale 使用多个不同 exponent，并随行/K/batch 变化；输出预填 NaN 检查漏写。FP32 校验使用 `1e-4 + 5e-5*sum(abs(term))` 的绝对误差界，BF16 对参考及边界做 RNE rounding；非有限结果直接失败。

Python 集成另验证 5 个配置的 BF16/FP32、真实 AITER B shuffle、两种 SFA 元数据形式、uint8/E8M0 dtype、uint8/FP8 packed B、`out=`、默认输出分配及非默认 stream。选择可精确表示的输入，全部输出与独立 CPU reference **逐元素完全相等**；也检查拒绝 FP32 scale 以及与输入重叠的 `out`。采用版本另通过完整 8192³ 的 GPU reference：先独立展开、反量化输入再用 FP32 GEMM 计算参考；选用可精确累加的二进制分数输入，BF16/FP32 各 67,108,864 个输出全部完全相等。

链接后的机器码每个实例 320 条 scaled MFMA，合计 1280，无 `s_setprio`。资源如下：

| 实例 | VGPR | SGPR | SGPR spill | VGPR spill | scratch bytes | LDS bytes |
|---|---:|---:|---:|---:|---:|---:|
| FP32 / 4 tiles | 250 | 103 | 0 | 0 | 0 | 139264 |
| FP32 / 1 tile | 230 | 70 | 0 | 0 | 0 | 139264 |
| BF16 / 4 tiles | 250 | 103 | 0 | 0 | 0 | 139264 |
| BF16 / 1 tile | 230 | 71 | 0 | 0 | 0 | 139264 |

当前四个实例的 SGPR/VGPR spill 和 scratch 均为零；此前 TR8 版本 4-tile 实例的 2 个 SGPR spill 已消除。四个实例共 40 条 TR8、零 `ds_bpermute`；B scale 仍为 `ds_read2st64_b32`。VMEM、TR8 消费及 LDS 发布等待均已审计。

早期在 SMI5/HIP7、8192³、b1/w200/i100 的原版复测：host scale 预重排 FP32 0.3837 ms / 2.865 P；原 rowmajor FP32 0.3996 ms / 2.752 P。初版 compact bpreshuffle FP32 约0.5013 ms / 2.193 P、BF16约0.5977 ms / 1.840 P。其后依次修正 B 聚合、scale 调度、BF16 store policy，再采用 TR8。原版与新版本量化分组不同；计时均只含 GEMM，不包括 host scale 重排或 B weight 预排。

达到 3P 要求单次 GEMM 不高于 0.366504 ms；当前采用版本尚未达到。候选与交错复测原始输出位于下面的证据目录。

完整构建日志、数值测试输出、B 排列 CPU 检查、ISA、metadata 和原目录文件哈希保存在 `/tmp/mxfp8_blockscale_bpreshuffle_20260910/`。
