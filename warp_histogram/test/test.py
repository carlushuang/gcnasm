import warp_histogram
import torch
import aiter
from aiter.test_common import perftest, benchmark

def warp_histogram_torch(x : torch.tensor, buckets : int):
    # TODO: https://github.com/pytorch/pytorch/issues/134570
    y = torch.empty((x.size(0), buckets), dtype = torch.int32)
    for i in range(x.size(0)):
        yy = torch.histogram(x[i, :], bins=buckets, range=(0, buckets))
        y[i, :] = yy[0]
    return y

L = [64, 128, 192, 256, 8192, 16384, 32768, 65536]
buckets = [64, 256]

@perftest()
def run_(x_, bucket_):
    return warp_histogram.warp_histogram(x_, bucket_)

# L = [64]
# buckets = 256
R = 4096
@benchmark()
def test():
    for i in range(len(L)):
        for b in range(len(buckets)):
            current_l = L[i]
            bucket = buckets[b]
            rows = current_l // 2 if R == -1 else R
            x = torch.randint(0, bucket, (rows, current_l)).to(dtype=torch.int32)  # .to(dtype=torch.float)

            y_d = warp_histogram.warp_histogram(x.to(device='cuda'), bucket)
            y_h = warp_histogram_torch(x.to(dtype=torch.float), bucket)

            x_d = x.to(device='cuda')
            _, us = run_(x_d, bucket)
            # print(x)
            # print(y_d)
            # print(y_h)
            is_same = torch.equal(y_d.to(device='cpu'), y_h.to(dtype=torch.int32))
            print(f"bucket:{bucket:<3}, R:{rows:<5}, L:{current_l:<5}, result:{is_same}, us:{us}")

test()