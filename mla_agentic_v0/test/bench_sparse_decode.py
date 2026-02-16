"""
Benchmark for sparse_attn_decode — matching FlashMLA MODEL1 test_flash_mla() metrics.

Reports: TFlops, GB/s, time (us), compute/memory ratio.
Uses CUDA events for timing (no triton or kernelkit dependency).

Usage:
    python test/bench_sparse_decode.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math
import time
import torch
import flash_mla
from flash_mla.quant import (
    quantize_kv_cache,
    abs_indices_to_flat_indices,
    BYTES_PER_TOKEN,
)

D_QK = 512
D_V  = 512


# ═══════════════════════════════════════════════════════════════════════════════
# Benchmark infrastructure
# ═══════════════════════════════════════════════════════════════════════════════

def cuda_bench(fn, warmup=10, rep=50):
    """Time a CUDA function using events. Returns median time in seconds."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    timings = []
    for _ in range(rep):
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        timings.append(start.elapsed_time(end))  # ms

    timings.sort()
    n = len(timings)
    median_ms = timings[n // 2]
    return median_ms / 1000.0  # return seconds


def count_flops_and_mem(b, s_q, h_q, topk):
    """
    Compute FLOPs and memory volume matching FlashMLA's formula.

    FLOPs   = 2 * h_q * (b * s_q * topk) * (d_qk + d_v)
    Mem vol = Q_bytes + KV_bytes + O_bytes
      Q  = 2 * b * s_q * h_q * d_qk          (bf16)
      KV = b * s_q * topk * BYTES_PER_TOKEN   (packed FP8, 584 bytes/token)
      O  = 2 * b * s_q * h_q * d_v            (bf16)
    """
    num_attended = b * s_q * topk
    flop = 2 * h_q * num_attended * (D_QK + D_V)
    mem_vol = (
        2 * b * s_q * h_q * D_QK           # Q (bf16)
        + num_attended * BYTES_PER_TOKEN    # KV (packed)
        + 2 * b * s_q * h_q * D_V          # O (bf16)
    )
    return flop, mem_vol


# ═══════════════════════════════════════════════════════════════════════════════
# Data generation
# ═══════════════════════════════════════════════════════════════════════════════

def gen_benchmark_data(b, h_q, s_q, topk, block_size, device):
    """Generate Q, packed KV, and flat indices for benchmarking."""
    s_kv = max(topk * 2, 4096)
    num_blocks_per_seq = (s_kv + block_size - 1) // block_size
    total_blocks = b * num_blocks_per_seq

    block_table = torch.randperm(total_blocks, device=device, dtype=torch.int32).view(b, num_blocks_per_seq)
    kv_bf16 = torch.randn(total_blocks, block_size, 1, D_QK, device=device, dtype=torch.bfloat16).clamp_(-1, 1)
    kv_packed = quantize_kv_cache(kv_bf16)

    abs_indices = torch.stack([
        torch.randperm(s_kv, device=device)[:topk]
        for _ in range(b * s_q)
    ]).view(b, s_q, topk).to(torch.int32)
    flat_indices = abs_indices_to_flat_indices(abs_indices, block_table, block_size)

    q = torch.randn(b, s_q, h_q, D_QK, device=device, dtype=torch.bfloat16).clamp_(-1, 1)
    sm_scale = D_QK ** (-0.5)
    return q, kv_packed, flat_indices, sm_scale


# ═══════════════════════════════════════════════════════════════════════════════
# Benchmark configs matching FlashMLA MODEL1
# ═══════════════════════════════════════════════════════════════════════════════

def get_benchmark_configs():
    """
    Returns list of (label, b, h_q, s_q, topk, block_size).

    FlashMLA MODEL1 CONFIGs use two KV scopes (main + extra):
      CONFIG1: h_q=64,  topk=128+512=640
      CONFIG2: h_q=128, topk=128+1024=1152
    We merge them into a single topk for our single-scope kernel.
    Batch sizes from FlashMLA: [2, 64, 74, 128, 148, 256].
    """
    configs = []

    # --- MODEL1 CONFIG1: h_q=64, topk=128+512=640, s_q=2 ---
    for b in [2, 64, 74, 128, 148, 256]:
        configs.append(("CFG1", b, 64, 2, 640, 64))

    # --- MODEL1 CONFIG2: h_q=128, topk=128+1024=1152, s_q=2 ---
    for b in [2, 64, 74, 128, 148, 256]:
        configs.append(("CFG2", b, 128, 2, 1152, 64))

    # --- Peak perf: h_q=64, topk=16384 ---
    configs.append(("PEAK", 148, 64, 2, 16384, 64))
    # --- Peak perf: h_q=128, topk=16384 ---
    configs.append(("PEAK", 148, 128, 2, 16384, 64))

    return configs


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    device = torch.device("cuda:0")
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device(device)
    torch.cuda.set_device(device)

    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}")
    print(f"BYTES_PER_TOKEN: {BYTES_PER_TOKEN}")
    print()

    configs = get_benchmark_configs()

    # Header
    print(f"{'Config':<6} {'Bsz':>4} {'h_q':>4} {'s_q':>3} {'topk':>6} "
          f"{'C/M':>5} {'TFlops':>7} {'GB/s':>6} {'us':>8}")
    print("-" * 70)

    results = []

    for label, b, h_q, s_q, topk, block_size in configs:
        torch.cuda.empty_cache()

        try:
            q, kv_packed, flat_indices, sm_scale = gen_benchmark_data(
                b, h_q, s_q, topk, block_size, device
            )
        except Exception as e:
            print(f"{label:<6} {b:>4} {h_q:>4} {s_q:>3} {topk:>6}   OOM: {e}")
            continue

        def run():
            return flash_mla.sparse_attn_decode(
                q, kv_packed, flat_indices, sm_scale=sm_scale
            )

        try:
            time_s = cuda_bench(run, warmup=5, rep=30)
        except Exception as e:
            print(f"{label:<6} {b:>4} {h_q:>4} {s_q:>3} {topk:>6}   ERR: {e}")
            continue

        flop, mem_vol = count_flops_and_mem(b, s_q, h_q, topk)
        cm_ratio = flop / mem_vol
        tflops   = flop / time_s / 1e12
        gbps     = mem_vol / time_s / 1e9
        time_us  = time_s * 1e6

        print(f"{label:<6} {b:>4} {h_q:>4} {s_q:>3} {topk:>6} "
              f"{cm_ratio:>5.1f} {tflops:>7.1f} {gbps:>6.0f} {time_us:>8.1f}")

        results.append({
            "label": label, "b": b, "h_q": h_q, "s_q": s_q, "topk": topk,
            "cm_ratio": cm_ratio, "tflops": tflops, "gbps": gbps, "time_us": time_us,
        })

        # Short cooldown between configs
        time.sleep(0.2)

    # Summary
    if results:
        import numpy as np
        valid_tflops = [r["tflops"] for r in results if r["tflops"] > 0.1]
        if valid_tflops:
            geomean = float(np.exp(np.mean(np.log(valid_tflops))))
            print(f"\nTFlops geomean: {geomean:.1f}")

    # H20 reference numbers (FlashMLA MODEL1 sparse decode, same configs)
    # Obtained from NVIDIA H20 with FlashMLA (flash_mla_with_kvcache, fp8 sparse)
    # topk values are combined: CFG1 = 128+512=640, CFG2 = 128+1024=1152
    h20_results = {
        ("CFG1",   2,  64): {"tflops":   8.7, "gbps":   52, "time_us":   38.7},
        ("CFG1",  64,  64): {"tflops":  77.0, "gbps":  459, "time_us":  139.4},
        ("CFG1",  74,  64): {"tflops":  88.9, "gbps":  530, "time_us":  139.6},
        ("CFG1", 128,  64): {"tflops":  92.1, "gbps":  549, "time_us":  233.2},
        ("CFG1", 148,  64): {"tflops":  98.2, "gbps":  585, "time_us":  252.9},
        ("CFG1", 256,  64): {"tflops": 101.3, "gbps":  604, "time_us":  423.9},
        ("CFG2",   2, 128): {"tflops":  24.7, "gbps":   76, "time_us":   48.9},
        ("CFG2",  64, 128): {"tflops": 105.9, "gbps":  325, "time_us":  364.9},
        ("CFG2",  74, 128): {"tflops": 110.4, "gbps":  338, "time_us":  404.8},
        ("CFG2", 128, 128): {"tflops": 112.3, "gbps":  344, "time_us":  688.3},
        ("CFG2", 148, 128): {"tflops": 114.1, "gbps":  350, "time_us":  783.5},
        ("CFG2", 256, 128): {"tflops": 117.1, "gbps":  359, "time_us": 1319.8},
        ("PEAK", 148,  64): {"tflops": 127.2, "gbps":  567, "time_us": 4996.0},
        ("PEAK", 148, 128): {"tflops": 127.4, "gbps":  288, "time_us": 9978.7},
    }

    # Comparison table
    print("\n" + "=" * 95)
    print("COMPARISON: MI308X (gcnasm/mla) vs H20 (FlashMLA)")
    print("=" * 95)
    print(f"{'Config':<6} {'Bsz':>4} {'h_q':>4}  "
          f"{'MI308 TF':>8} {'H20 TF':>8} {'ratio':>6}  "
          f"{'MI308 GB/s':>10} {'H20 GB/s':>9} {'ratio':>6}  "
          f"{'MI308 us':>8} {'H20 us':>8}")
    print("-" * 95)

    for r in results:
        key = (r["label"], r["b"], r["h_q"])
        h20 = h20_results.get(key)
        if h20:
            tf_ratio = r["tflops"] / h20["tflops"] if h20["tflops"] > 0 else 0
            gb_ratio = r["gbps"] / h20["gbps"] if h20["gbps"] > 0 else 0
            print(f"{r['label']:<6} {r['b']:>4} {r['h_q']:>4}  "
                  f"{r['tflops']:>8.1f} {h20['tflops']:>8.1f} {tf_ratio:>5.1f}x  "
                  f"{r['gbps']:>10.0f} {h20['gbps']:>9.0f} {gb_ratio:>5.1f}x  "
                  f"{r['time_us']:>8.1f} {h20['time_us']:>8.1f}")
        else:
            print(f"{r['label']:<6} {r['b']:>4} {r['h_q']:>4}  "
                  f"{r['tflops']:>8.1f} {'N/A':>8} {'':>6}  "
                  f"{r['gbps']:>10.0f} {'N/A':>9} {'':>6}  "
                  f"{r['time_us']:>8.1f} {'N/A':>8}")


if __name__ == "__main__":
    main()
