"""Microbenchmark: random vs sequential HBM access bandwidth on MI308X."""
import torch
import time

def bench(fn, warmup=5, iters=20):
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
    return start.elapsed_time(end) / iters / 1000.0  # seconds

def main():
    device = "cuda"
    print(f"GPU: {torch.cuda.get_device_name()}")
    print()

    # Simulate KV cache: large buffer with 584-byte tokens
    num_tokens_total = 1_000_000  # 1M tokens in cache
    token_bytes = 584
    kv_buf = torch.randint(0, 256, (num_tokens_total, token_bytes), dtype=torch.uint8, device=device)

    print(f"KV cache: {num_tokens_total} tokens x {token_bytes} bytes = {num_tokens_total * token_bytes / 1e9:.2f} GB")
    print()

    for num_access in [640, 1152, 8192, 16384, 65536, 262144]:
        indices = torch.randint(0, num_tokens_total, (num_access,), dtype=torch.int64, device=device)

        # Random gather
        def random_gather():
            return kv_buf[indices]

        t = bench(random_gather)
        bytes_read = num_access * token_bytes
        gbps = bytes_read / t / 1e9
        print(f"Random gather {num_access:>7} tokens: {t*1e6:>8.1f} us, {gbps:>7.1f} GB/s, {bytes_read/1e6:.1f} MB")

    print()

    # Sequential read for comparison
    for mb in [1, 10, 50, 200]:
        n = mb * 1024 * 1024
        buf = torch.randint(0, 256, (n,), dtype=torch.uint8, device=device)
        def seq_read():
            return buf.sum()
        t = bench(seq_read)
        gbps = n / t / 1e9
        print(f"Sequential read {mb:>4} MB: {t*1e6:>8.1f} us, {gbps:>7.1f} GB/s")

if __name__ == "__main__":
    main()
