"""GPU integration check using aiter's real weight shuffle and a CPU reference.

Run: HIP_VISIBLE_DEVICES=2 python test_blockscale_bpreshuffle.py
Only the test builds reference scales; the GEMM adapter never transforms inputs.
"""
import argparse
import inspect

import torch
from aiter.ops.shuffle import shuffle_weight

from blockscale_bpreshuffle import gemm_a8w8_blockscale_bpreshuffle


def check_case(m, n, k, tiles, metadata_layout, scale_dtype, packed_bytes):
    # Small dyadic values make every product and FP32 accumulation exact, so
    # this integration test can require equality, including BF16 rounding.
    a_cpu = (torch.randint(-4, 5, (m, k)).float() / 2).to(torch.float8_e4m3fn)
    b_cpu = (torch.randint(-4, 5, (n, k)).float() / 2).to(torch.float8_e4m3fn)
    sa_physical = torch.randint(126, 129, (k // 128, m), dtype=torch.uint8)
    sb_cpu = torch.randint(126, 129, (n // 128, k // 128), dtype=torch.uint8)
    sa_values = torch.pow(2.0, sa_physical.T.double() - 127).repeat_interleave(128, 1)
    sb_values = (torch.pow(2.0, sb_cpu.double() - 127)
                 .repeat_interleave(128, 0).repeat_interleave(128, 1))
    ref = ((a_cpu.double() * sa_values) @ (b_cpu.double() * sb_values).T).float()

    # Use a non-default stream to exercise the ctypes launcher's stream ABI.
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        a = a_cpu.cuda()
        b = shuffle_weight(b_cpu.cuda(), layout=(16, 16))
        if packed_bytes:
            b = b.view(torch.uint8)
        sa_storage = sa_physical.cuda()
        sa = sa_storage.view(m, k // 128) if metadata_layout else sa_storage.T
        sb = sb_cpu.cuda()
        sa, sb = sa.view(scale_dtype), sb.view(scale_dtype)
        for dtype in (torch.float32, torch.bfloat16):
            out = torch.full((m, n), float('nan'), dtype=dtype, device=a.device)
            result = gemm_a8w8_blockscale_bpreshuffle(
                a, b, sa, sb, dtype=dtype, out=out, tiles=tiles)
            assert result is out, 'out= must preserve the caller-owned tensor'
            actual = result.cpu()
            torch.testing.assert_close(actual, ref.to(dtype), rtol=0, atol=0)
            print(f'PASS {m}x{n}x{k} tiles={tiles} dtype={dtype} '
                  f'scale={scale_dtype} metadata_layout={metadata_layout} '
                  f'packed_bytes={packed_bytes}', flush=True)

        # Allocation path and dtype contract are part of the eager interface.
        allocated = gemm_a8w8_blockscale_bpreshuffle(a, b, sa, sb, tiles=tiles)
        torch.testing.assert_close(allocated.cpu(), ref.to(torch.bfloat16), rtol=0, atol=0)
        try:
            gemm_a8w8_blockscale_bpreshuffle(a, b, sa.float(), sb.float())
        except ValueError as error:
            assert 'E8M0' in str(error)
        else:
            raise AssertionError('FP32 scales must not be silently interpreted as E8M0')
        if k == 2 * n:
            try:
                gemm_a8w8_blockscale_bpreshuffle(a, b, sa, sb, out=a.view(torch.bfloat16))
            except ValueError as error:
                assert 'overlap' in str(error)
            else:
                raise AssertionError('out must not overwrite A used by other workgroups')
    stream.synchronize()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--large', action='store_true', help='also verify all outputs at 8192 cubed')
    args = parser.parse_args()
    torch.set_num_threads(16)
    torch.manual_seed(20260910)
    print('shuffle_weight source:', inspect.getfile(shuffle_weight), flush=True)
    print('GPU:', torch.cuda.get_device_name(), flush=True)
    e8m0 = getattr(torch, 'float8_e8m0fnu', torch.uint8)
    check_case(256, 256, 128, 1, False, torch.uint8, False)
    check_case(512, 512, 256, 1, True, e8m0, True)
    check_case(1280, 512, 1152, 4, False, e8m0, False)
    check_case(512, 256, 256, 0, True, torch.uint8, False)
    check_case(256, 512, 1024, 1, False, torch.uint8, False)
    if args.large:
        check_large()
    print('ALL PYTHON INTEGRATION CHECKS PASSED', flush=True)


def check_large():
    """Full benchmark shape, independent dequantized FP32 GPU GEMM reference.

    The dyadic products and even their absolute sum fit exactly in FP32. This
    permits exact comparison without a prohibitively slow CPU reference.
    """
    m = n = k = 8192
    device = torch.device('cuda')
    a = (torch.randint(-4, 5, (m, k), device=device).float() / 2).to(torch.float8_e4m3fn)
    raw_b = (torch.randint(-4, 5, (n, k), device=device).float() / 2).to(torch.float8_e4m3fn)
    sa_physical = torch.randint(126, 129, (k // 128, m), dtype=torch.uint8, device=device)
    sb = torch.randint(126, 129, (n // 128, k // 128), dtype=torch.uint8, device=device)
    sa = sa_physical.T
    a_dequant = a.float() * torch.exp2(sa.float() - 127).repeat_interleave(128, 1)
    b_dequant = raw_b.float() * (torch.exp2(sb.float() - 127)
                                .repeat_interleave(128, 0).repeat_interleave(128, 1))
    ref = a_dequant @ b_dequant.T
    del a_dequant, b_dequant
    b = shuffle_weight(raw_b, layout=(16, 16))
    for dtype in (torch.float32, torch.bfloat16):
        out = torch.full((m, n), float('nan'), dtype=dtype, device=device)
        result = gemm_a8w8_blockscale_bpreshuffle(a, b, sa, sb, dtype=dtype, out=out)
        torch.testing.assert_close(result, ref.to(dtype), rtol=0, atol=0)
        print(f'PASS full 8192x8192x8192 {dtype}: all {m*n} outputs exactly equal', flush=True)


if __name__ == '__main__':
    main()
