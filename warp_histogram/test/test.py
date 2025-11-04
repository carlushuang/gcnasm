import warp_histogram
import torch

def warp_histogram_torch(x : torch.tensor, buckets : int):
    # TODO: https://github.com/pytorch/pytorch/issues/134570
    y = torch.histogram(x, bins=buckets, range=(0, buckets))
    return y[0]

L = [64, 128, 192, 256]
buckets = [64, 256]

# L = [64]
# buckets = 256

for i in range(len(L)):
    for b in range(len(buckets)):
        current_l = L[i]
        bucket = buckets[b]
        x = torch.randint(0, bucket, (current_l,)).to(dtype=torch.int32)  # .to(dtype=torch.float)

        y_d = warp_histogram.warp_histogram(x.to(device='cuda'), bucket)
        y_h = warp_histogram_torch(x.to(dtype=torch.float), bucket)

        # print(x)
        # print(y_d)
        # print(y_h)
        is_same = torch.equal(y_d.to(device='cpu'), y_h.to(dtype=torch.int32))
        print(f"bucket:{bucket}, L:{current_l}, result:{is_same}")
