# nps_locality: memory locality on MI355X (NPS2 partitions vs. address-aware NPS1)

These probes answer three questions for MI355X (gfx950): which locality domain (IOD) a workgroup runs on, which IOD a given address lives on, and what it costs to cross between the two. The motivation is DeepGEMM's locality-aware Mega MoE on B200 ([deepseek-ai/DeepGEMM#462](https://github.com/deepseek-ai/DeepGEMM/pull/462)): it keeps each die's SMs on die-local weights and reports about 100 W lower power and 5–10% higher clocks under heavy load. The question here is whether the same idea works on MI355X.

All numbers come from one 8×MI355X box (ROCm 7.2.0, amdgpu 6.16.13), measured on 2026-11-03.

## TL;DR

- **Two IODs.** MI355X has two IODs, each carrying 4 XCDs and 4 HBM3E stacks. NPS2 gives per-IOD memory, but only together with DPX, QPX or CPX, which split the GPU into two or more HIP devices. SPX supports NPS1 only.
- **IOD locality in SPX + NPS1.** A single kernel can still get it:
  - The IOD of a workgroup is `XCC_ID >> 2` (XCCs 0–3 on one IOD, 4–7 on the other).
  - The IOD of an address is physical address bit 13 XOR a per-2 MiB-page phase bit, so memory alternates between the IODs every 8 KiB.
  - The phase cannot be predicted from the virtual address. Probe it once per allocation (0.41 s for 16 GiB).
- **Power.** Streaming reads in one full-GPU kernel: reading only own-IOD granules loses about 2% bandwidth and saves about 80 W (−8%) compared with naive interleaved reads. Anti-local reads are capped at about 4 TB/s by the IOD–IOD link, which carries about 2 TB/s per direction.
- **Latency.** A cold load takes about 325 ns from the local IOD and about 420 ns across.

## Hardware IDs available inside a kernel

| Register | Read with | Fields used | Meaning on MI355X |
|---|---|---|---|
| `HW_REG_XCC_ID` (hwreg 20) | `s_getreg_b32 s, hwreg(HW_REG_XCC_ID)` | `[3:0]` | XCC index within the current partition: 0–7 in SPX, 0–3 in DPX (logical, not physical) |
| `HW_REG_HW_ID` (hwreg 4) | `s_getreg_b32 s, hwreg(HW_REG_HW_ID)` | CU `[11:8]`, SH `[12]`, SE `[14:13]`, queue `[26:24]`, ME `[31:30]` | CU location inside the XCC; there is no partition field |
| HIP `__smid()` | — | `(xcc << 6) \| (se << 4) \| cu` | It drops SH, but was still unique on this part (128 CUs per DPX partition) |

- **Dispatch order.** Workgroups are dispatched round-robin over XCCs starting at XCC 0 (4096/4096 in SPX; `wg i -> XCC i % 4` in DPX). Read `XCC_ID` rather than assuming `blockIdx % n`; [a third-party report](https://github.com/GolfOscarr/fleet-amd-task/pull/5) saw a non-zero start offset under virtualization.
- **What `XCC_ID` tells you per mode.** In DPX every partition reports XCC IDs 0–3, so `XCC_ID` cannot tell the two NPS2 domains apart; the domain is a property of the HIP device you launched on. In SPX, `XCC_ID >> 2` is the IOD (see the correlation results below).

## Partition modes

`sudo amd-smi partition -a` on MI355X:

| Compute | XCCs per partition | Memory modes allowed |
|---|---|---|
| SPX | 8 | NPS1 |
| DPX | 4 | NPS1, NPS2 |
| QPX | 2 | NPS1, NPS2 |
| CPX | 1 | NPS1, NPS2 |

Switching modes:

```sh
# all GPUs must be idle: no KFD processes and nothing (containers included) holding /dev/kfd or /dev/dri
ls /sys/class/kfd/kfd/proc/
# SPX + NPS1 -> DPX + NPS2
sudo rocm-smi --setcomputepartition DPX
sudo rocm-smi -d 0 --setmemorypartition NPS2 --autorespond y   # hive-wide: all 8 GPUs switch, amdgpu reloads (~1 min)
# back to SPX + NPS1
sudo rocm-smi -d 0 --setmemorypartition NPS1 --autorespond y   # the reload brings SPX back by itself
```

Partitions of one GPU show up as consecutive HIP devices with the same PCI BDF (devices `2k` and `2k+1` in DPX).

## DPX + NPS2: one HIP device per NPS domain

`dpx_partition_probe 0 1 2` and `dpx_concurrent_bw 0 1`. Devices 0 and 1 are the two partitions of one GPU, and device 2 is another GPU. Latency is a pointer chase over cold 128 B lines with vector loads; bandwidth is a streaming read by one partition's 128 CUs.

| Memory on | Latency (median) | Read bandwidth |
|---|---|---|
| same partition (local NPS domain) | 322–325 ns | 2.85–2.90 TB/s |
| sibling partition (other NPS domain, same GPU) | 409–411 ns | 1.89–2.02 TB/s |
| another GPU over XGMI | 984–986 ns | 66 GB/s |

| Both partitions streaming at once | Aggregate |
|---|---|
| local + local | 5.80 TB/s |
| cross + cross (both directions) | 3.85 TB/s |
| partition 0 local + partition 1 reading partition 0's memory | 3.06 TB/s (one domain's HBM is the ceiling) |

On cross-domain reads, XCCs 1 and 3 are consistently about 18 ns faster than XCCs 0 and 2.

## SPX + NPS1: address-aware locality inside one kernel

### Method

`spx_interleave_probe` and `spx_granule_map` time a single cold `global_load_dword` between two `s_memtime` reads, in exact inline asm. Only one XCC works per launch (the others exit after reading `XCC_ID`), and a 2 GiB memset flushes L2 and the Infinity Cache between launches.

### Which XCCs share an IOD

The per-line latency vectors (1 MiB at 128 B resolution) correlate +0.62 to +0.76 within {0,1,2,3} and within {4,5,6,7}, and −0.52 to −0.74 across the two groups.

| | Median latency |
|---|---|
| line homed on the local IOD | 324–328 ns |
| line homed on the other IOD | 408–436 ns (odd XCCs ≈ 410, even ≈ 430) |

### Which IOD an address lives on

- **8 KiB granules.** At 128 B resolution, the home IOD flips exactly every 8 KiB.
- **GF(2) fit.** Over 256 MiB of a fresh 1 GiB allocation, the fit gives `home = addr[13] ^ addr[21]`, with 16 violations in 72,128 confident samples.
- **The phase is not a virtual-address bit.** In a 16 GiB allocation, all 8192 pages of 2 MiB still fit `home = addr[13] ^ phase(page)`: 99.1% of pages fit with at most 2 of 256 granules off, which matches the measurement noise. But the page phase agrees with every virtual-address bit from 21 to 33 only 50% of the time. It depends on physical placement, so it must be probed: one bit per 2 MiB, and mapping 16 GiB from all 8 XCCs takes 0.41 s.
- **Second GPU.** The same rule holds on a second GPU (device 5): all 4096 pages fit, 98.2% with at most 2 errors.

### What locality buys: streaming reads, one SPX kernel, 256 CUs

`spx_granule_map` with exact per-granule lists:

| Mode | dev 0 (16 GiB) | dev 5 (8 GiB) |
|---|---|---|
| naive: every XCC reads every granule | 6.07 TB/s, 967–970 W | 6.03 TB/s, 943–947 W |
| local: XCC 0–3 read IOD-0 granules, XCC 4–7 read IOD-1 granules | 5.95 TB/s, 887–889 W | 5.89 TB/s, 866–868 W |
| anti-local: lists swapped | 3.95 TB/s, 835 W | 3.92 TB/s, 814 W |

`spx_page_phase_bw` uses a per-2 MiB phase table instead of per-granule lists, which is the form a kernel would use:

| Mode | dev 0 (16 GiB) | Efficiency |
|---|---|---|
| naive | 6.18 TB/s, 970–973 W | 6.37 GB/J |
| local | 6.02 TB/s, 885–887 W | 6.80 GB/J |
| anti-local | 3.98 TB/s, 831–833 W | 4.79 GB/J |
| naive grid-stride (the kernel from the DPX test) | 5.86 TB/s, 953–954 W | 6.14 GB/J |

- **Power and efficiency.** Local placement saves about 80 W (−8%) for about 2% less bandwidth, a 7% gain in energy per byte.
- **Link limit.** Anti-local tops out near 4 TB/s, the same per-direction limit as cross + cross in DPX.
- **Where the power goes.** Naive interleaving pushes about 3 TB/s across the IOD–IOD link; that is where the 80 W goes. A rough estimate is about 25 pJ per byte crossed.
- **Clock stays put.** sclk stayed at about 2.39 GHz in every mode: these runs are not power-capped, so the saving shows up as watts, not speed.

## Recipe for a locality-aware kernel in SPX + NPS1

1. **Mode.** Stay in SPX + NPS1: one kernel on all 256 CUs. NPS2 gives the same locality but splits the GPU into separate devices.
2. **Workgroup domain.** Domain of a workgroup = `XCC_ID >> 2`. Read the register.
3. **Probe the phases.** At weight-load time, probe the phase of each 2 MiB page (see `spx_granule_map`; fit all 256 granules of a page to absorb noise).
4. **Place each half on its IOD.** Put each IOD's half of the weights (split along N, as in DeepGEMM) into the 8 KiB granules homed on that IOD. Local granule `k` of domain `d` lives at `base + (k >> 7) * 2 MiB + ((k & 127) * 2 + (d ^ phase[k >> 7])) * 8 KiB`.
5. **Tile layout.** Pack tiles in units of 8 KiB, for example 64 rows × 128 B of FP8 per granule.
6. **Scheduling.** Use per-IOD task queues (XCC 0–3 vs XCC 4–7), as in DeepGEMM's per-die queues. Activations stay half-remote, since both IODs need all of them under an N split.

## Programs

| File | Mode | What it does |
|---|---|---|
| `dpx_partition_probe.cpp` | DPX + NPS2 | Per-workgroup `HW_REG_XCC_ID` / `HW_REG_HW_ID` on every device; latency and bandwidth from one partition to local, sibling and other-GPU memory |
| `dpx_concurrent_bw.cpp` | DPX + NPS2 | Both partitions of one GPU streaming at once: local + local and cross + cross |
| `spx_interleave_probe.cpp`, `analyze_interleave.py` | SPX + NPS1 | Per-line cold latency from each XCC (1 MiB at 128 B, 256 MiB at 4 KiB), then XCC grouping, home pattern and GF(2) fit |
| `spx_granule_map.cpp` | SPX + NPS1 | Home IOD of every 8 KiB granule from all 8 XCCs, per-page fit, then naive / local / anti-local streaming with board power from hwmon |
| `spx_page_phase_bw.cpp` | SPX + NPS1 | The same PoC with a per-2 MiB phase table, plus the plain grid-stride kernel for comparison. Its "4 granules do not alternate" counter is strict and flags a page for a single noisy granule; the 256-granule fit in `spx_granule_map` shows every page alternates |

```sh
sh rebuild.sh                                   # ARCH=gfx950 by default
./build/spx_interleave_probe 0 && python3 analyze_interleave.py <cycles_per_ns printed by the probe>
./build/spx_granule_map 0 16 6                  # device, GiB, seconds per mode
```

The raw outputs of the runs above are in `results/`.

## Caveats

- **Streaming is not a GEMM.** L2 and Infinity Cache reuse will change how much traffic crosses the IOD–IOD link in a real kernel.
- **Watts, not speed.** None of these runs hit the power cap. Whether the saved watts turn into clock and throughput needs a power-capped compute-bound kernel.
- **The phase is physical.** Re-probe after reallocating, or keep weights in long-lived allocations.
- **Noise.** About 1% of single-granule measurements have a margin under 31 ns. The per-page fit absorbs that; raw per-granule labels do not.
