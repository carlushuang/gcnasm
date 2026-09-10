# MXFP8 raw row-major scale GEMM 优化记录（整合至 2026-09-10）

当前采用 `refill_split_gather`：一次 GEMM launch 直接消费原始 row-major SFA/SFB，在 VGPR 中整理 scale，再经线程块 LDS 发布给 scaled MFMA。有效的 2026-09-09 空闲卡 25 轮确认中，当前版 **0.3996 ms / 约 2.7515 PFLOPS**，上一版 **0.4035 ms**，host 预重排参考 **0.3830 ms / 约 2.8708 PFLOPS**。当前版相对上一版的配对耗时变化中位数为 **−0.9174%**，25/25 轮领先；相对预重排 GEMM-only 仍有 **4.3092%** 的配对耗时差距。

本文整合初始 row-major 路径、refill-before-barrier、low2-linear-bounded、最终 split-refill 及目录清理的证据。2026-09-10 整理阶段没有新增 GPU 数值验证；此前同日直接复测的数百毫秒结果属于异常状态单次记录，见后文。历史文档原件保存在 [清理前备份](/tmp/mxfp8_cleanup_20260910_aae6hw9_/before)，各项性能证据仍按原实验阶段解释。

实现边界与输入约定如下。

| 项目 | 当前约定 |
|---|---|
| 计算 | 每 batch 执行 `C = A × Bᵀ`；A/B 为 FP8 E4M3，C 为 FP32 |
| A / B / C | row-major `[batch,M,K]` / `[batch,N,K]` / `[batch,M,N]` |
| SFA / SFB | E8M0 字节，row-major `[batch,M,K/32]` / `[batch,N,K/32]`；每行每 32 个 K 元素一个 scale |
| scale stride | `stride_sfa = stride_sfb = K/32`；batch stride 分别为 `M*K/32`、`N*K/32` |
| 形状边界 | 正整数 M/N/K/batch；M、N 为 256 的倍数，K 为 128 的倍数；这里的 K 尾部指队列/展开尾部，并非任意 K 的掩码支持 |
| GEMM tile / 指令 | workgroup tile `256×256×128`，`V_MFMA_SCALE_F32_16X16X128_F8F6F4`，scale 在 MFMA 内参与计算 |
| 调度 | 8 waves / 512 threads；persistent 沿 M 固定 B 处理 4 个输出 tile，较小 grid 选 single 1 tile/WG |
| dispatch | `ceil(num_tiles_m/4)*num_tiles_n >= CU数` 时选 persistent，否则 single；persistent 最后 WG 可不足 4 个输出 |
| 资源约束 | 无 host scale 重排、无额外 global/device scale workspace、无额外 scale kernel；允许使用 VGPR 和既有线程块 LDS |

优化路线按发生顺序如下。各行的基线不同，不能将百分比相加，也不能将不同卡或不同负载阶段的绝对耗时直接相减。

| 阶段 | 采用的变化与结论 | 同阶段证据 |
|---|---|---|
| 初始 row-major 化 | 从 host consumer-major scale 图像改为 kernel 内 gather、4-dword VGPR queue、4×4 lane/byte 转置、LDS 发布；保留原 A/B tiling、scaled MFMA 和 persistent handoff | [早期实验归档](/tmp/pack/OPTIMIZATION_LOG.md)；这些早期最小值/消融只作为探索记录，不能替代后来的配对确认 |
| 准备与发布分离 | 尽早做转置、延迟 LDS store 到 publication barrier 前；通过 `u_gsf/u_ssf` 表达 global/LDS layout；MFMA/VALU 分组中保留 SALU 与 LDS-read 调度槽 | [备份 README](/tmp/mxfp8_cleanup_20260910_aae6hw9_/before/README.md) 保存早期 0.4398→0.4272→0.4183 ms 及 min-of-11 分组实验；卡状态/口径不同，不累计为最终收益 |
| 首轮广泛筛选 | 没有候选超过相同二进制副本的波动；确认 `u32x4_t queue[2]` 会被 LLVM Promote Alloca 转为每 WG 额外 16 KiB LDS，SSA vector 可避免这一代价 | [首轮报告](/tmp/mxfp8_opt_20260909_jxsdi9d4/REPORT.md)：25 轮最佳候选仅 −0.024%，副本自身 −0.032%；未据此替换正式版 |
| `refill_before_barrier` | 四 tile 队列的 future scale load 移到旧 packed 值写出之后、发布 barrier 之前；明确建立 future-load-last，并严格匹配 `vmcnt(1)` 与真实 refill 条件 | [25 轮摘要](/tmp/mxfp8_gap_20260909_xkjwy__8/confirm_refill25/summary.json)：1.2636→1.2543 ms，配对 −0.7277%，25/25 胜出；HIP5 / 物理 SMI4 的占用状态 |
| `low2_linear_bounded` | 相邻两 lane 连续 gather、8-dword SSA queue、按 consumer lane 连续排列的 scale LDS、有限 descriptor extent；按八 tile refill，future allowance 改为 2 | [第一次 25 轮](/tmp/mxfp8_scale_path_20260909_7fza4n_r/confirm_low2_25/summary.json) / [第二次 25 轮](/tmp/mxfp8_scale_path_20260909_7fza4n_r/confirm_low2_repeat25/summary.json)：配对 −0.0478% / −0.0796%；仍为 HIP5/SMI4，占用状态的小幅收益 |
| `refill_split_gather` | 将两个 raw row span 改为具名 q0/q1，分别在旧值最后一次消费后立即补读，保留全部地址、字节映射和边界语义 | [最终 25 轮](/tmp/mxfp8_idle_opt_20260909_8stdiezw/confirm_split25/summary.json)：0.4035→0.3996 ms，配对 −0.9174%，25/25 胜出；空闲 HIP7/SMI5 |

早期记录中的“GPU 5”指 HIP 编号 5，实际是 **SMI4 / PCI0000:85:00.0**。用户指定的物理 **SMI5 / PCI0000:95:00.0 对应 HIP7**；最终确认使用后者。映射见 [device_identity.json](/tmp/mxfp8_idle_opt_20260909_8stdiezw/device_identity.json)。占用卡约 1.25 ms、空闲卡约 0.4 ms 的结果各有自己的对照，不能据其变化声称数倍优化。

low2 与 linear LDS 分别解决两项已测到的成本。两 lane 合读同一行连续 32 bytes，降低分散 gather 请求；LDS packed word 从旧 `wave*256 + row*16 + kg*4` 改为 `wave*256 + (kg*16+row)*4`，与 consumer lane 顺序一致。SFB half1 仍比 half0 高 512 bytes，保留 `ds_read2st64_b32` 和 MFMA byte selector。正式 bounded 版的 [profile_production](/tmp/mxfp8_scale_path_20260909_7fza4n_r/profile_production/summary.json) 复核了这些效果：`SQ_LDS_BANK_CONFLICT` 4,194,304→0、TCP→TCC read requests 41,943,040→37,748,736（−10%），MFMA/VMEM/LDS 指令事件数量不因这两项变化而减少。硬件事件不是实际 HBM 字节或可直接扣除的 kernel 时间；冲突归零没有换来等比例提速，profiler 时间不进入正常性能表。

当前 scale 数据路径可独立理解为以下四步。

1. 四个 wave-N=0 waves 生产 SFA，另四个 wave-N=1 waves 生产 SFB；每 wave 仅选择一个 descriptor。相邻 lane 读取一行的相邻 16-byte spans，共同覆盖八个 K128 tile。每 lane 两条 `buffer_load_dwordx4` 写入 `v_scale_q0/v_scale_q1`，共八个 raw dword；q0/q1 是该 producer 负责的两个 row call，**不是 A/B 两个张量**。
2. lane 的 row/span 坐标放在 `voffset`，输出 tile、producer wave、八 tile group 基址放在 uniform `soffset`。每个 K128 对应每逻辑行四个 scale 字节。准备 tile k 时，`word=k&3`、`span=(k>>2)&1`；两个 quad DPP broadcast、两次 byte perm 和一次 `permlane32_swap` 将四个 row call 拼成 MFMA packed dword。
3. producer 坐标为 `row=(lane>>1)&15`、`kg=(lane&1)|((lane>>5)<<1)`，写入 `kg*16+row` 对应的 LDS slot；consumer 按自身 lane 连续读取。producer lane 与 consumer lane 不能直接混同。最终 packed 值保存在独立 VGPR，发布 store 仍紧靠 barrier。
4. 令 `loops=K/128`，主循环处理 `tile+1<loops` 的当前 tile t，同时准备 t+1。q0 最后一个旧 dword broadcast 完成后立即 refill q0；q1 与已保存的 q0 broadcast 完成 pack 后再 refill q1。链接后的 persistent/single、unrolled/remainder 四条路径均有 **q0 load → 6 MFMA → q1 load → 12 MFMA → publication wait**。请求数、scale 字节数、LDS 布局、MFMA selector 均保持不变。

两条 refill 与 relaxed publication wait 必须共用同一个严格条件：

```cpp
const bool refill = ((tile & 7) == 6 && tile + 2 < loops);
// q0/q1 的旧值分别保存到独立 broadcast/packed VGPR 后：
if (refill) q0 = load_scale_row(0, output_tile, tile + 2);
// 保留已审计的计算与调度间隔。
if (refill) q1 = load_scale_row(1, output_tile, tile + 2);
store_next_scale();
if (refill) vmcnt(2); else vmcnt(0);
lgkmcnt(0);
s_barrier();
```

这是说明顺序的伪代码；实现分散于 MFMA 分组之间，不能将其压成相邻语句重排。旧组的最后一个 tile 是 t+1，新组始于 t+2；q0[3] 先保存，q1[3] 再参与最终 packed 值，因此即便 refill 立即完成，也不会覆盖仍需消费的旧值。所有当前发布所需的 A copies 均早于 future q0/q1，两条 future load 之间及第二条至发布等待之间没有其他 VMEM；`vmcnt(2)` 留下的只能是下一组 scale。没有 refill 的 K 尾部必须 `vmcnt(0)`，否则最年轻的请求可能是当前所需 A copy。随后 `lgkmcnt(0)` 与 barrier 完成跨 wave LDS 发布和复用。

下一组开始消费之前保留编译器生成的等待：已审计 full-loop 路径使用 `vmcnt(4)`，允许的是四条更年轻的 A copies，并不允许旧 scale 未完成。K=128 仅走原 prologue/epilogue；loops=8、16…时最后一次潜在 refill 被严格上界禁止，loops=9、17…时只 refill 确实存在的新组。源码中的 scheduler barrier 约束编译调度，不能代替硬件 wait；gfx950 permlane 输入的 VALU hazard 也不能靠直接删 NOP 解决。

persistent handoff 条件保持 `output_tile+1 < OUTPUT_TILES_PER_WG && block_m+1 < num_tiles_m`。先将当前最后一个 K tile 的 scale 从 LDS 读入独立 MFMA 操作数，再补读下一输出的 q0/q1/A/B。链接代码两个 `vmcnt(16)` 位置前，**最后十六条 VMEM 均为 C stores**，所需的下一输出 loads 更早；因此最多留下 C stores，随后 LGKM wait/barrier 才发布下一输出。主循环 barrier 之后的 cold-B(t+2) 写回已释放 stage，不能提前到旧 stage 的消费者读取完成之前。

尾部 descriptor 指针与 extent 在同一个 uniform 分支构造：

```text
SFA row_offset = first_block_m * B_M * stride_sfa
SFB row_offset = block_n * B_N * stride_sfb
ptr = tensor_batch_base + row_offset
extent = stride_sf_batch - row_offset
```

未消费的 queue spans 可能读到逻辑行尾之外；descriptor 将访问限制在当前 batch 的剩余分配内。gfx950 微探针已验证 `buffer_load_dwordx4` 按每个 DWORD 的 `voffset+soffset` 检查边界，越界 DWORD 归零、合法 DWORD 保留；这是本目标证据，不泛化为所有 AMD 目标的行为。无界 descriptor 的速度不能代表正式 bounded 实现。详细边界和映射证据见 [low2 历史备份](/tmp/mxfp8_cleanup_20260910_aae6hw9_/before/OPTIMIZATION_LOG_LOW2_LINEAR.md)；最终请求顺序见 [WAIT_AUDIT](/tmp/mxfp8_idle_opt_20260909_8stdiezw/candidates/refill_split_gather/WAIT_AUDIT.md) 与 [独立 linked/state 审计](/tmp/mxfp8_idle_opt_20260909_8stdiezw/refill_split_independent_review.md)。

最终有效性能测试统一为 `8192³, batch=1, warmup=200, iterations=100, verify=0`，HIP7/SMI5，grid `(256,1,1)`、4 output tiles/WG。环境保存为 `HIP_FORCE_DEV_KERNARG=1`、`HSA_NO_SCRATCH_RECLAIM=1`、`HIP_VISIBLE_DEVICES=7`、`OMP_NUM_THREADS=32`。同一任务的 GPU 程序串行，每轮随机交错旧版、相同旧二进制副本、新版和预打包参考，未混入 profiler 时间。

| 25 轮版本 | 时间中位数 ms | 中位时间换算 PFLOPS | 配对耗时变化 vs 旧版 | 配对耗时差距 vs prepacked |
|---|---:|---:|---:|---:|
| 旧版 `low2_linear_bounded` | 0.4035 | 2.7249 | 0 | +5.3278% |
| 相同旧二进制副本 | 0.4033 | 2.7263 | −0.0248% | +5.2989% |
| 当前 `refill_split_gather` | **0.3996** | **2.7515** | **−0.9174%** | **+4.3092%** |
| host 预重排参考，GEMM-only | 0.3830 | 2.8708 | −5.0583% | 0 |

配对百分数是 `median_r[100*(candidate_r/reference_r-1)]`。它不等于两个时间中位数之比：当前/旧版的中位数比为耗时 **−0.9665%**、吞吐 **+0.9760%**；当前/prepacked 为耗时 **+4.3342%**、吞吐缺口 **4.1542%**。逐轮配对的 prepacked 吞吐缺口中位数则为 **4.1312%**。`PFLOPS=2*M*N*K/(time_ms*10^12)`，表内由四舍五入后的时间换算；原程序打印的吞吐使用更高内部时间精度。历史 benchmark 的 `speedup_vs_rowmajor_pct` 实际按两个最小耗时之比计算，不能当作中位提升。

当前版在确认阶段范围 0.3992–0.4000 ms，相对旧版的配对 P25/P75 为 −1.0159% / −0.8924%，25/25 轮也胜过相同旧二进制副本。加上同卡四个筛选阶段，共 53/53 轮胜过两份旧版；这是开发阶段的描述性汇总，主要采用证据仍为独立 25 轮确认。随后正式路径 `production_smoke5` 中位 0.3997 ms，5/5 轮胜过旧版，单独作为短复核，不并入 53 轮。完整数据与口径复算见 [RESULTS_REVIEW](/tmp/mxfp8_idle_opt_20260909_8stdiezw/RESULTS_REVIEW.md)、[原始 25 轮](/tmp/mxfp8_idle_opt_20260909_8stdiezw/confirm_split25/raw.jsonl)。

预打包版在 host 构建 consumer-major SFA/SFB 后上传，计时只包住 warmup 后的 GEMM launch 循环；**不含 host packing 与上传成本**，因此上述差距不是端到端流程比较。2026-09-10 清理前，用户原版目录与冻结 prepacked 的 exe、五个源文件和 Makefile 均逐字节一致。历史 3 P 原始数据来自 **HIP2 / 物理 GPU2 / PCI0000:65:00.0**；同一历史记录中 HIP7/SMI5 约 0.3835 ms/2.867 P，与本轮 prepacked 约 2.871 P 一致。不能据跨卡数字宣称源码退化或已恢复 3 P，详见 [历史 3P 核对](/tmp/mxfp8_recheck_3p_20260909_3vbfzovt/REPORT.md)。

2026-09-10 的新直接复测在 HIP2/物理 GPU2，仍用 8192³、b1/w200/i100，各执行一次，原始输出为：

| 版本 | 原始 avg_time，ms | 原始 TFLOPS |
|---|---:|---:|
| host 预重排 | **351.2771** | **3.13** |
| 当前 row-major | **339.2691** | **3.24** |

两进程均正常退出，但绝对耗时严重偏离此前亚毫秒水平，内部变慢原因及两版受干扰程度未定位。这组**异常状态单次结果不作为正常性能提升或反超证据**；没有把 ms 除以 1000，也没有将 TFLOPS 改称 PFLOPS。此前 60 秒超时来自脚本进程上限，不是 GEMM 返回错误；没有完成可替代 9 月 9 日确认结论的新多轮正常对照。诊断 `w0/i1` 和暂停调试进程的输出也不进入正式比较，见 [直接复测报告](/tmp/mxfp8_retest_20260910_wql6svfo/REPORT.md) 与 [原始命令/输出](/tmp/mxfp8_retest_20260910_wql6svfo/direct_results.json)。

资源与正确性结论对应 9 月 9 日采用的冻结 GEMM，清理后重编的机器码对应关系在文末另记。

| 资源 | 上一版 persistent / single | 当前 split persistent / single |
|---|---:|---:|
| VGPR | 252 / 240 | **253 / 238** |
| SGPR | 104 / 76 | **104 / 75** |
| LDS bytes | 139264 / 139264 | **139264 / 139264** |
| VGPR/SGPR spill、private scratch | 0 | **0** |
| 链接后 MFMA 静态数 | 每实例 320，合计 640 | 每实例 **320**，合计 **640** |
| `s_setprio` | 0 | **0** |

编译环境为 clang 23（`/root/workspace/llvm-src/build/bin/clang++`）与 Opus include `/root/workspace/aiter/csrc/include`；历史 `make check regs` 通过。MFMA 数用于检查展开是否退化，不能只看源代码循环。最终数值验证共有 **31/31 个成功 case**：完整 8192³、2 个 quick branch case、28 个 K 尾部/batch/branch case。

- 完整 `8192×8192×8192, batch=1`：CPU reference `errors=0/67108864`、`ALL BATCHES VALID`。
- quick：single `256×512×8192, batch=2`；forced persistent `1536×768×1152, batch=1`，均通过。
- K 为 `128,256,384,512,640,768,896,1024,1920,2048,2176,8064,8192,8320`，各运行 single `256×256` 和 forced persistent `1280×256`，均 batch=2；后者覆盖不足四输出的尾 WG。
- 测试 host 显式拒绝 NaN/Inf，并用 `-fno-finite-math-only` 保留检查。强制分支仅用于验证 host；性能使用正常 host。证据为 [quick](/tmp/mxfp8_idle_opt_20260909_8stdiezw/verification/quick_results.json)、[tails](/tmp/mxfp8_idle_opt_20260909_8stdiezw/verification/tail_results.json)、[full](/tmp/mxfp8_idle_opt_20260909_8stdiezw/verification/full_results.json)。
- 独立状态模型按 refill 立即完成/覆盖检查 1,052 配置、377,610 次 tile 消费、45,720 次成对 refill、1,578 次输出交接；另有 523,776 个迭代位置的 guard 审计。CPU 映射、linked waits 与 GPU 数值检查共同构成证据。

未采用方向压缩如下。百分数仅属于所列阶段、相对其当时 row-major 基线；负数表示更快。失败说明针对这些具体实现，不作为永久排除整类方案的证明。

| 方向 | 结果与取舍 |
|---|---|
| 缩短队列 / 移至 LDS | 早期逐 tile DWORD 读取节省少量 VGPR 却增加 issue；LDS queue 引入读取等待。最终同卡 `refill_low2_group4` +0.3222%；保留八 tile VGPR queue |
| 数组宽队列、过度展开、强压 VGPR | 数组可隐式增加 16 KiB LDS；若干展开/cap 候选产生大量 spill 或尾部数值失败。spill=0 不能证明 LDS 未增加 |
| 更宽 gather、TR8/byte scatter、consumer 合并读 | low4 请求更少但旧阶段约慢 0.56%；正确的 TR8 scatter 约 +1.88%，b128/b96 consumer 约 +1.14%/+1.41%。TR8 已能正确映射，问题不是“不可实现” |
| 缓存 B scales、四 wave/大 chunk | 缓存与更大 chunk 增加 LDS、地址分支或寄存器成本；chunk16 超 LDS 上限。修复正确性或降低 spill 后仍未超过各阶段基线 |
| 直接 A/B scale | 最终 broad/pipeline 分别 +10.6399% / +17.3654%；替换部分逻辑读取约 2×/4×，consumer 增加转置且仍有 vmcnt(0)。均组合 B-prefetch2，不能当作单因素隔离实验；逻辑倍率不等于 HBM 流量倍率，B 的 SGPR lane spill 也不是 scratch |
| B 预取 / producer wave | 最终 prefetch2/3 为 +0.4462%/+0.3717%，wave-N=0 为 +1.3882%；6/8 预取有 4/12 VGPR spills，未进入该轮 GPU 筛选 |
| publication barrier 位置 | 完整发布块从第 20 条 MFMA 后移到第 16/24 条后，最终 +0.4216%/+1.7596%；LDS 生命周期正确仍不保证更快 |
| 队列提前转置与 split 组合 | queue8/pair-ahead 为 −0.5700%/−0.4214%，弱于同轮 split；half-publish +0.5454%，queue8-earlyq0 +0.6943%；统一地址与 split 组合 −0.7690%，也未超过 split |
| cache hint / swizzle / workgroup 遍历 | NT 曾明显退化；SC0/SC1、bounded cache swizzle、部分 N 分组与基线持平；最终 group-N=1 +6.0238%。不采用不满足 bounded 约束的 swizzle 数字 |
| 编译器/映射历史缺陷 | clang 23 曾错误处理 ext_vector 元素直接 bit_cast，先复制到标量再 cast 才正确；旧 phase3 vmcnt(1)“只留下 future B”的解释与重链接 ISA 不符，已由严格 future-scale-last 规则取代 |

较完整的未采用候选记录位于 [首轮报告](/tmp/mxfp8_opt_20260909_jxsdi9d4/REPORT.md)、[refill 阶段报告](/tmp/mxfp8_gap_20260909_xkjwy__8/REPORT.md)、[scale 路径报告](/tmp/mxfp8_scale_path_20260909_7fza4n_r/REPORT.md)、[最终统计复核](/tmp/mxfp8_idle_opt_20260909_8stdiezw/RESULTS_REVIEW.md)。这些阶段对调度/寄存器的敏感性也说明：不能从“指令减少”直接推导提速，采纳结论要有数值和同轮性能证据。

目录已清理为仅保留四个代码文件（common、kernel 头文件、kernel TU、host）、Makefile、简短 README、本记录、`.gitignore` 和重编后的 `build/` 产物；删除不在正常路径上的 repack 实现、timeline 与 `MXFP8_*` 诊断入口，将分散历史文档合并于此。移除这些入口不代表新增算法收益。清理前源码、文档和主要构建产物及其散列保存在 [before](/tmp/mxfp8_cleanup_20260910_aae6hw9_/before) 和 [before_manifest.json](/tmp/mxfp8_cleanup_20260910_aae6hw9_/before_manifest.json)。

清理前身份如下；清理会改变 host 和可执行封装，不能要求新完整 exe 保持同一 SHA，也不能未经核对便将旧验证自动归给新代码。

```text
上一版 row-major exe: b6fad79f9abc6f977566250eb5c4140047cec4a42799df307bc248d3e2535fdd
已验证 split exe:     b9555030e90a931b34ee76211f7894183d7d762b5512376e3d303cc4ce694eac
冻结 prepacked exe:   51aaac1d05fde7a472fc995df92c87f80cd58dfcc710fa073bc8c8b63c6625b7
清理前 kernel 源码:    544fb9249c1a04e3d25abcb32cb610f4ec6e34fd9c2477aa5b3b2ed728820901
```

本轮清理验收已完成（2026-09-10）。

- 正式代码位置：`/root/workspace/gcnasm_new/gcnasm/opus_gemm/mxfp8_gemm_16x16x128_scale_rowmajor`。四个代码文件为 `gemm_a8w8_mxfp8_scale_common.h`、`gemm_a8w8_mxfp8_scale_kernel_template.hpp`、`gemm_a8w8_mxfp8_scale_kernel.cc`、`gemm_a8w8_mxfp8_scale_host.cc`；另保留 `Makefile`、`README.md`、本记录与 `.gitignore`。`build/` 仅保留可执行文件。
- host 从 701 行精简到 393 行；删除 host/device repack、packed scale 临时分配、timeline、7 个输入诊断环境入口及无用 include/helper。原始 scale 直接上传，设备只分配 A/B/C/SFA/SFB；两个 GEMM 编译实例每次选择一个 launch。
- common 删除 7 个无引用常量，保留模板参数位置和类型名；kernel 仅修正过期注释。全部 asm、调度、wait、q0/q1 生命周期和指令逻辑均保留。原有 basic CLI 及长参数/等号写法保留；已移除的 CLI 参数明确报错退出。
- 删除独立 repack 头文件、重复 bench/rebuild 脚本、3 份分散历史/审计文档、约 1.35 GB core/profiler 临时文件以及构建 `.o`。历史文档已合并，源码和文档原件备份在上述 `before/`；崩溃转储未复制。
- `make -j2 OPUS_INCLUDE_DIR=/root/workspace/aiter/csrc/include` 和 `make -j2 check regs` 均返回 0。Makefile 的提取检查已改为独立临时目录，不再改写 exe，也支持 check/regs 并行。
- 清理前后，从最终 exe 提取的完整 GEMM 设备镜像**逐字节相同**，SHA256 为 `4e2b4f17cbc5972054fd8560cc579f7f20fb82ac27b844fe4c2771ced049505f`。两个实例依然为 VGPR253/238、SGPR104/75、LDS139264、spill/scratch0、640 MFMA、`s_setprio=0`。设备镜像一致，因此原有 linked wait/handoff 审计覆盖同一段机器码。
- 正式 exe SHA256：`6a51259e607d10a31d7f1e4e7502b1cc01b40be613d620a333c053ab8ff7aa9e`。host 被精简，所以完整 exe 与历史 b955 镜像的外层封装不同；GEMM 设备镜像保持相同。
- 对照备份，7 个 host 核心函数体（初始化、CPU参考、校验、选择和计时）及 kargs/grid/验证调用块均保持一致。以 GPU 不可见环境检查移除参数报错和基本参数解析，均通过；本轮**没有启动 GPU kernel，没有重跑性能或数值验证**。历史 31 项数值验证明确保留其原运行日期。
- 证据：[构建检查](/tmp/mxfp8_cleanup_20260910_aae6hw9_/build_check.log)、[镜像和源码散列](/tmp/mxfp8_cleanup_20260910_aae6hw9_/build_validation.json)、[host 路径核对](/tmp/mxfp8_cleanup_20260910_aae6hw9_/host_contract_check.json)、[CLI 检查](/tmp/mxfp8_cleanup_20260910_aae6hw9_/cli_check.json)、[删除清单](/tmp/mxfp8_cleanup_20260910_aae6hw9_/removed_files.json)。

使用方法见同目录 [README.md](README.md)。本记录是整理后唯一的优化过程文档；清理不会被计为新增性能收益。

后续目录清理（2026-09-10）：已删除 sibling `mxfp8_gemm_16x16x128_scale_ldsring/`。该目录为早期 `u32x4_t v_scale_queue[2]` 八 tile 数组队列实验，旧转置/补读调度，不含当前 low2-linear-bounded 与 split-refill 路径。其现有 linked 镜像 LDS 为155648字节，比当前139264字节多16KiB；没有发现当前工程对该目录的依赖。为保留实验追溯，源文件和原构建产物已完整归档并逐文件校验到 [历史归档](/tmp/mxfp8_ldsring_retire_20260910_29vyhd1j/archive)，资源/散列检查见 [inspection.json](/tmp/mxfp8_ldsring_retire_20260910_29vyhd1j/inspection.json)。这次只删除历史副本，当前四个源码文件和可执行文件均未改动。
