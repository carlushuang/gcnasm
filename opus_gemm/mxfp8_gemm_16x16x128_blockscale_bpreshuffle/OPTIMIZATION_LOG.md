# gfx950 blockscale bpreshuffle 优化与接入记录

更新日期：2026-09-10。当前正式方案为 `scale_publish_fused`，源代码清理版对应候选 `next_fused_clean`。

本目录是独立适配工程：`../mxfp8_gemm_16x16x128_scale_rowmajor` 和原 host scale 预排目录 `../mxfp8_gemm_16x16x128_scale` 保持不变。

## 发布前 8-wave 复测（2026-09-10 10:37 UTC）

在 HIP2 / SMI2 / PCI `0000:65:00.0` 顺序运行两个正式目录的现有二进制。M=N=K=8192，b=1、w=200、i=100，OMP_NUM_THREADS=16；各配置只运行一组，以下为组内100次 launch 的平均值，不是多轮中位数。两版均为512线程、wave64，即8-wave；本尺寸自动选择每 workgroup 最多4个输出 tile。

| 版本 | 输出 C | 平均耗时 ms | PFLOPS |
|---|---|---:|---:|
| rowmajor / refill_split_gather | FP32 | 0.3826 | 2.87361 |
| blockscale bpreshuffle / scale_publish_fused | FP32 | 0.3774 | 2.91313 |
| blockscale bpreshuffle / scale_publish_fused | BF16 | 0.3704 | 2.96827 |

本次 BF16 尚未达到3P。计时仅含 GEMM；B 预排、输入准备和上传不计入。rowmajor 使用原生1×32 scale，bpreshuffle 使用1×128 / 128×128 scale，表格用于比较各自输入契约下的吞吐。

本次原始数据随仓库保存于 [results.json](results/gpu2_20260910_103740/results.json)，包含完整命令、返回码、源码及可执行文件 SHA256；同目录保存三个程序的输出。提交前仅清理了 rowmajor 模板7行末尾的空白和该程序日志的末尾空行，未改计算逻辑；记录保留计时源码哈希，并额外保存 `published_kernel_sha256` 标识提交源码。bpreshuffle 源码及两版可执行文件均与计时记录一致。

下文保留此前优化阶段的多轮统计及本地历史归档路径；那些 `/tmp` 归档不随本次源码提交提供。

## 目标与当前结果

目标：8192×8192×8192，batch=1，warmup=200，iterations=100，单 GEMM kernel，无 host scale 重排、无额外全局 scale workspace，达到 3 PFLOPS。

所有后续性能测试固定 **HIP_VISIBLE_DEVICES=2**，对应 **SMI2 / HIP2 / PCI 0000:65:00.0，MI355X / gfx950**。之前 SMI5 对应 HIP7，不能拿两个 GPU 的绝对数值混合判断优化收益。

最后三轮交错复测（`gpu2/role_comparison.json`，正式版本与最后一个控制简化候选比较）：

| 版本 | 输出 C | 中位耗时 ms | 中位 PFLOPS |
|---|---|---:|---:|
| 正式 scale_publish_fused | BF16 | 0.3683 | 2.98501 |
| 正式 scale_publish_fused | FP32 | 0.3774 | 2.91306 |
| 删除 store role 的空 asm | BF16 | 0.3687 | 2.98191 |
| 删除 store role 的空 asm | FP32 | 0.3782 | 2.90722 |

最后候选虽减少10条静态指令，三轮中位数没有收益，因此保留正式版本。多组复测中，正式 BF16 约2.98–2.99P，尚未稳定3P。

此前三轮候选交错复测（`gpu2/finalists_comparison.json`）：

| 版本 | 输出 C | 中位耗时 ms | 中位 PFLOPS |
|---|---|---:|---:|
| 正式 scale_publish_fused | BF16 | 0.3676 | 2.99117 |
| 正式 scale_publish_fused | FP32 | 0.3772 | 2.91499 |
| 进一步放宽第二处 LDS wait | BF16 | 0.3680 | 2.98797 |
| 进一步放宽第二处 LDS wait | FP32 | 0.3782 | 2.90714 |
| 改为 N8 分组的 workgroup 顺序 | BF16 | 0.3692 | 2.97846 |
| 改为 N8 分组的 workgroup 顺序 | FP32 | 0.3778 | 2.91003 |

3P 对应不高于 **0.366503876 ms**。上述采用版本的中位数尚未达到 3P；单次最快曾测到 0.3669 ms / 2.99702P，不能用四舍五入宣称已经稳定达标。FP32/BF16 是输出类型，输入均为 FP8，MFMA 均用 FP32 累加。

采用前的 GPU2 三轮对照（`gpu2/initial_comparison.json`）为：host scale 预排 FP32 0.3662 ms / 3.00228P；rowmajor FP32 0.3824 ms / 2.87549P；此前 blockscale TR8 版 FP32 0.3829 ms / 2.87147P、BF16 0.3743 ms / 2.93768P。这组与本轮后段测量时间不同。严格同时间段的新旧直接对照见 `gpu2/fused_comparison.json`：BF16 中位数 0.3720→0.3701 ms，FP32 0.3832→0.3771 ms。

计时仅覆盖 GEMM。B weight 预排在计时之外；host 预排参考的 scale 预排也在计时之外。原 host/rowmajor 的 1×32 scale 与当前 blockscale 输入的量化粒度不同，因此它们是吞吐参考，并非相同量化数据的数值等价比较。

## 当前完整流程

```text
上游量化器
  ├─ A：FP8 E4M3FN [M,K]，row-major
  ├─ A_scale：E8M0，1×128；物理 [K/128,M]
  ├─ W：FP8 E4M3FN [N,K]
  └─ B_scale：E8M0，128×128；物理 [N/128,K/128]
                         │
模型加载/weight 准备阶段：shuffle_weight(W, layout=(16,16))
                         ↓
                  B_packed [N,K]
                         │
A + B_packed + A_scale + B_scale + 输出 C / 当前 stream
                         ↓
  Python eager adapter → C ABI launcher → 一个 gfx950 GEMM kernel
                         ↓
                  BF16 / FP32 C[M,N]
```

本 kernel 不调用 `fp8_legacy_to_mxfp8`，不重新量化 A/B，也不生成一个全局 `[M,K/32]` / `[N,K/32]` scale tensor。1×128 / 128×128 的 E8M0 scale 在 MFMA 消费处广播，保持输入量化语义。

| 输入 | 类型 | 物理布局 / 地址 |
|---|---|---|
| A | E4M3FN FP8 | 连续 `[M,K]`，`A[m*K+k]` |
| B | E4M3FN FP8 字节 | AITER `(16,16)` preshuffle；tensor shape 仍 `[N,K]` |
| A_scale | E8M0 byte | 逻辑 `[M,K/128]`，物理 `[K/128,M]`；`SFA[k_block*M+m]` |
| B_scale | E8M0 byte | `[N/128,K/128]` 连续；`SFB[(n/128)*(K/128)+k_block]` |
| C | BF16 / FP32 | 连续 `[M,N]`；FP32 累加后写回 |

SFA 接受真正列主序 tensor（shape `[M,K/128]`、stride `(1,M)`），也接受 AITER 的元数据约定：shape/stride 看似连续 `[M,K/128]`，但底层字节已经由量化器按 `[K/128,M]` 输出。**普通行主序 scale 字节不能直接冒充这一格式。** 当前调用接口不会替输入做 transpose/contiguous/to；需要上游直接输出正确布局。

标准 FP8 B weight 排列等价于：

```python
B_packed = (
    W.view(N // 16, 16, K // 32, 2, 16)
     .permute(0, 2, 3, 1, 4).contiguous().view(N, K)
)
```

只有 weight 使用这个 preshuffle；B_scale 保留原始紧凑块布局。

## scale 在 kernel 中如何加载

保留 `GSF → SSF → RSF → scaled MFMA(op_sel)`，在 256×256×128 tile 中使用 8 个 wave。

1. **GSF：全局读取紧凑 scale。** Wave0–3 读取 A scale，wave4–7 读取 B scale。每个 K128 tile，A 有 256 个不同的行 scale；B 只有两个不同的 N128 scale。B producer 当前仍有重复 lane 地址读取，并不是只有两条全局 load 指令。
2. **A 的 SSF：直接写 raw byte。** A producer wave 的 lane=`call*16+r` 读取逻辑行 `tile_row + wave_m*16 + call*64 + r`，然后用 `ds_write_b8` 写到 `stage*1024 + wave_m*64 + lane`。生产者不做四次 shuffle 打包。
3. **A 的 RSF：TR8 形成 packed dword。** `ds_read_b64_tr_b8` 的 lane 地址为 `stage*1024 + wave_m*64 + 8*(lane&15)`。低 32 位返回四个 M repeat 的字节 `[s(m),s(m+64),s(m+128),s(m+192)]`，高 32 位不用。实际 TR8 lane 规则已用独立 GPU probe 验证，不能换成所有 lane 相同的地址。
4. **B 的 SSF/RSF：复制字节并读取两个 N half。** 对 N128 的 scale byte `s`，写入 `uint32(s)*0x01010101u`。RSFB 使用 `ds_read2st64_b32`，读取相隔 512 bytes 的两个 N128 half。
5. **op_sel：继续使用。** A 的 selector 选当前 M repeat 的 scale byte；四组 K32 lane 得到同一个 packed word，因此一个 K128 scale 作用于四个 K32 子组。A 的四个 packed byte 对应四个 M repeat，并非四个 K32 的独立 scale。B packed word 的四个 byte 相同，对应 N128 half 的所有 K32 和 N repeat。

对一个外部 K128 scale `s`，硬件语义是：

```text
K[0:32] → s，K[32:64] → s，K[64:96] → s，K[96:128] → s
```

广播改变的是硬件消费方式，不是把输入重新量化成四个独立 K32 block。

SFA、SFB 各分配两个 1024-byte LDS stage，总 scale LDS 4KiB；A 每 stage 实际写 256 bytes。它是 workgroup 内部 LDS，无额外全局 scale workspace。包括 A/B 数据，整个 kernel 每 WG 的 LDS 分配为 139264 bytes。

## 已采用的优化

- **B 数据合并读取。** 初版仅把原 B 地址改成 preshuffle 地址，却保留旧 lane 分工，一个 wave 会碰到 32 条 128B cache line。现在每个 producer wave 连续读取 1024 bytes packed B，覆盖 8 条 line；同步重写 RB 的 LDS consumer 地址。SB 传输框架保留，A 的 GA/SA/RA 数据路径保留。
- **紧凑 scale producer。** 分开定义外部 GROUP_K=128 与硬件 MFMA_SCALE_GROUP_K=32，不再沿用 1×32 rowmajor scale 的 gather/refill 队列。
- **TR8 A scale 消费。** A 用 raw byte LDS 写入及一条 TR8 读取构成 packed word，去掉 A 的 wave shuffle 链。B 继续 packed word + op_sel。
- **提前 load、延后等待、合并发布。** 主循环开头读取 t+1 的 scale；前 20 条 MFMA 执行后统一 `vmcnt(0)`，A 写 raw byte、B 乘 `0x01010101` 后写 dword，再完成 LDS 等待和 barrier。删除前 16 条 MFMA 后的独立 prepare、A 侧旧 shuffle drain 及分开的条件路径。之后在剩余 12 条 MFMA 期间预取 t+2 的 B。
- **保留 output handoff 与 1/4-tile 分支。** 每 WG 最多计算四个相邻 M tile，以复用 B cache 并摊薄启动/收尾；小工作量走单 tile。
- **BF16 store policy。** BF16 使用默认 store policy；FP32 继续 `nt`。两种输出都是 FP32 MFMA 累加，BF16 仅在写回时 RNE 转换。默认缓存策略对两种类型的效果不同；FP32 改默认 store 本轮反而变慢。
- **清理控制与无用源码。** 合并发布后 persistent VGPR251→250、SGPR106→103，SGPR spill2→0；所有实例无 VGPR spill、无 scratch。删去恒等 pack/prepare lambda 及无用中间变量，清理前后完整 device ELF 逐字节相同。

## 不采用的方向与硬件证据

候选只保存在证据目录，没有叠加到正式代码。包括：128×128/4-wave（BF16 约2.07P）；B scale 直接进寄存器；稀疏 B scale LDS 写入；把两个 B scale 用不同 op_sel 共用一个 word；A scale dword/单 wave producer；A scale 单 wave direct-to-LDS + linear image；放宽 LDS wait；前后移动主循环发布点；改变 loop unroll、C accumulator pin 粒度、workgroup 顺序、输出 cache policy，以及 store role 条件/空 asm。

其中 pin8 产生 VGPR spill，单 wave A direct-to-LDS 产生额外 SGPR spill（6个），BF16 0.3935 ms，未采用；删除内存操作不保证机器码调度或寄存器分配更好。每个正确但较慢的候选都保留原始结果，没有用某一次偶然最快的值替换正式方案。

GPU2 的硬件计数采集仍用 b1/w200/i100，仅采集第201–203次 kernel；性能数字不取 profiler 计时。结果在 `gpu2/profile_fused/summary.json`：

| counter | 当前 fused FP32 | host scale 预排 FP32 |
|---|---:|---:|
| SQ_INSTS_MFMA | 16777216 | 16777216 |
| SQ_INSTS_VMEM_RD | 4718592 | 4325376 |
| SQ_INSTS_LDS_STORE | 524288 | 0 |
| SQ_LDS_BANK_CONFLICT | 0 | 3145728 |
| SQ_INSTS_SALU | 13061120 | 10972160 |
| SQ_INSTS_BRANCH | 2357248 | 1883904 |
| SQ_WAIT_INST_LDS，中位计数 | 24494714 | 29161943 |

这些是硬件 counter 值，不等于源码操作数、字节数或互斥耗时占比。当前 scale 字节更少、bank conflict 为零，但紧凑 scale 的逐 K 加载和发布仍增加指令/控制成本；不能仅按 scale 数据量估算性能，也不能把几个等待 counter 直接相加解释总时间。

## 数值、机器码验证与代码位置

当前正式代码通过 GPU2 的 Python 集成检查：真实 AITER `shuffle_weight`、两种 SFA 元数据形式、uint8/E8M0 scale、BF16/FP32、当前 stream、`out=`、默认输出分配、输入/输出重叠拒绝和 FP32 scale 拒绝。包括完整 8192³：BF16/FP32 各 67108864 个输出与独立反量化参考逐元素完全相等（使用可精确累加的二进制分数输入）。此外有随机 scale/exponent 的 double CPU reference、batch2、K128 和跨 output-tile / K-tail 检查。

`make all check regs` 通过；四个实例各320条 scaled MFMA，共1280，40条TR8，零 `s_setprio`。完整控制流审核覆盖84个scale store的VMEM依赖、80次scale LDS消费等待、42次barrier以及40次TR8的full EXEC。主循环36个展开/rolled publication点都在20条MFMA后执行统一VMEM等待。

| 实例 | VGPR | SGPR | SGPR/VGPR spill | scratch | LDS |
|---|---:|---:|---|---:|---:|
| FP32 / 4 tiles | 250 | 103 | 0 / 0 | 0 | 139264 |
| FP32 / 1 tile | 230 | 70 | 0 / 0 | 0 | 139264 |
| BF16 / 4 tiles | 250 | 103 | 0 / 0 | 0 | 139264 |
| BF16 / 1 tile | 230 | 71 | 0 / 0 | 0 | 139264 |

模板：`gemm_a8w8_mxfp8_scale_kernel_template.hpp`。清理版源码 SHA256：`0fe6a06df9e1686b01a49997f830b9e85c4e39ebf2c674c6ae8dc49071f61b12`。正式 exe 提取的 device ELF SHA256：`9d3699c7d78876961e97a4fc7da1a8af79c34ffb3613268d43c98ab63b8cdff4`，与已审核、已计时候选完全一致。

其它入口：`gemm_a8w8_mxfp8_scale_common.h`（ABI/traits）、`gemm_a8w8_mxfp8_scale_kernel.cc`（四个实例）、`gemm_a8w8_blockscale_bpreshuffle_launch.cc`（单次 launch）、`blockscale_bpreshuffle.py`（eager接口）。构建和调用例子见 README。

## 能否接入 AITER blockscale_bpreshuffle

**能作为 gfx950 + E8M0 + 支持尺寸的 backend 接入；当前已验证输入兼容及直接 eager 调用，尚未注册到 AITER 的 dispatcher。** 接入方式是在 CPU/Python/C++ 分派入口选择本 GEMM，并发射一个 GPU kernel。

核对的 AITER 源码版本为 `12620102e523c4688e9ecd466499501086ad48c0`，快照 `/tmp/aiter_bpreshuffle_review_12620102e523`：

- 入口：`aiter/ops/gemm_op_a8w8.py:904`，`gemm_a8w8_blockscale_bpreshuffle`。
- 在同文件1008–1011行，普通分派会把 E8M0 x_scale/w_scale 转成 FP32。应在这一步之前增加 gfx950 E8M0 分支。
- Weight 预排：`aiter/ops/shuffle.py:137` 的 `shuffle_weight`，由调用方在 GEMM 外完成。实际 Python 集成测试调用的是本机 AITER 的同名函数。

建议分派契约如下（伪代码，`native_backend` 代表后续原生注册，不是当前已经存在的 AITER API）：

```python
# 放在 AITER 的 E8M0 -> FP32 转换之前；Y 已按 out= 约定取得。
if (gfx == "gfx950" and A_B_are_e4m3fn
        and A_scale_is_e8m0 and B_scale_is_e8m0
        and dtype == bf16 and supported_dense_layouts
        and M % 256 == 0 and N % 256 == 0 and K % 128 == 0):
    return native_backend(A, B_packed, A_scale, B_scale, out=Y)
# 其余输入继续走 AITER 原分派。
```

工程接入剩余工作：把模板/实例加入 AITER build，增加 native PyTorch binding/算子注册，并加入上面的分派分支或调优配置。保留 AITER 的 out/current stream/fake op 约定，再验证 torch.compile/图捕获和模型实际 shape。当前 ctypes eager wrapper 已验证 out/stream；不能把它等同于已完成 AITER native backend 注册或已验证模型图捕获。

兼容边界：

- scale **必须本来就是 E8M0**。任意 FP32 1×128/128×128 scale 不可按 E8M0 字节直接解释；广播也不能精确替代任意 FP32 scale。若 DSV4 上游仍输出 FP32 scale，本版不能直接接收，需要另行明确其量化/转换方案。
- 当前 A 不变，B 使用标准 `(16,16)` weight preshuffle，SFA 必须是上述列主序字节。无单独 scale shuffle，无 `fp8_legacy_to_mxfp8`。
- M/N 为256倍数、K为128倍数；每个分配低于2GiB。Python为2-D，C ABI和独立程序支持dense batch。小 M decode、不对齐形状、FP16输出走其它 backend。
- AITER 当前这个公开入口主要接受 BF16/FP16 输出，因此本 backend 先接 BF16 分支；本地 FP32 分支用于性能分析和数值验证。

全部实验、原始计时、profiling、构建、CPU/GPU校验与原目录hash：`/tmp/mxfp8_blockscale_bpreshuffle_20260910/`。本轮新增证据集中在 `gpu2/`。
