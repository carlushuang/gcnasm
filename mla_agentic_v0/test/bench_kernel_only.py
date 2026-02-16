"""Profile MFMA kernel time in isolation (no Python overhead)."""
import torch
import sys
sys.path.insert(0, "/mnt/raid0/carhuang/repo/gcnasm/mla")
import tvm_ffi
from flash_mla import _get_sparse_decode_func, _get_sparse_decode_splitk_func, _compute_num_splits
from flash_mla.quant import BYTES_PER_TOKEN

def bench_kernel(fn, warmup=5, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters  # ms

def main():
    device = "cuda"
    d_qk = 512
    s_q = 2

    # Force JIT compile
    print("Compiling kernels...")
    _get_sparse_decode_func()
    print("Done.\n")

    configs = [
        ("CFG1", 74, 64, 640),
        ("CFG1", 148, 64, 640),
        ("CFG2", 74, 128, 1152),
        ("PEAK", 148, 64, 16384),
    ]

    print(f"{'Config':<6} {'Bsz':>4} {'h_q':>4} {'topk':>6} {'splits':>6} "
          f"{'kernel_us':>10} {'TFlops':>8}")
    print("-" * 60)

    for label, b, h_q, topk in configs:
        # Pre-allocate all tensors
        total_tokens = b * s_q * topk + 1000
        q = torch.randn(b, s_q, h_q, d_qk, device=device, dtype=torch.bfloat16)
        kv = torch.randint(0, 256, (total_tokens, BYTES_PER_TOKEN), device=device, dtype=torch.uint8)
        indices = torch.randint(0, total_tokens, (b, s_q, topk), device=device, dtype=torch.int32)
        out = torch.empty(b, s_q, h_q, d_qk, device=device, dtype=torch.bfloat16)
        lse = torch.empty(b, s_q, h_q, device=device, dtype=torch.float32)
        dummy_tl = torch.empty(1, dtype=torch.int32, device=device)

        q_tvm = tvm_ffi.from_dlpack(q)
        kv_tvm = tvm_ffi.from_dlpack(kv.flatten())
        idx_tvm = tvm_ffi.from_dlpack(indices)
        o_tvm = tvm_ffi.from_dlpack(out)
        lse_tvm = tvm_ffi.from_dlpack(lse)
        tl_tvm = tvm_ffi.from_dlpack(dummy_tl)

        num_splits = _compute_num_splits(topk, h_q, b, s_q)
        sm_scale = d_qk ** (-0.5)

        if num_splits <= 1:
            def kernel_fn():
                _get_sparse_decode_func()(
                    q_tvm, kv_tvm, idx_tvm, o_tvm, lse_tvm,
                    h_q, topk, sm_scale, 0, tl_tvm
                )
        else:
            bsq = b * s_q
            o_partial = torch.empty(num_splits, bsq, h_q, d_qk, device=device, dtype=torch.float32)
            lse_partial = torch.empty(num_splits, bsq, h_q, device=device, dtype=torch.float32)
            op_tvm = tvm_ffi.from_dlpack(o_partial)
            lp_tvm = tvm_ffi.from_dlpack(lse_partial)
            def kernel_fn():
                _get_sparse_decode_splitk_func()(
                    q_tvm, kv_tvm, idx_tvm, op_tvm, lp_tvm, o_tvm, lse_tvm,
                    h_q, topk, num_splits, sm_scale, 0, tl_tvm
                )

        ms = bench_kernel(kernel_fn)
        us = ms * 1000

        flop = 2 * h_q * (b * s_q * topk) * (d_qk + d_qk)
        tflops = flop / (ms / 1000) / 1e12

        print(f"{label:<6} {b:>4} {h_q:>4} {topk:>6} {num_splits:>6} "
              f"{us:>10.1f} {tflops:>8.1f}")

if __name__ == "__main__":
    main()
