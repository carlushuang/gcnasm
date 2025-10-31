import warp_histogram
import torch

def warp_histogram_torch(x : torch.tensor):
    # TODO: https://github.com/pytorch/pytorch/issues/134570
    y = torch.histogram(x, bins=64, range=(0, 64))
    return y[0]

# L = [64, 128, 192]
L = [64]
# torch.manual_seed(8)

for i in range(len(L)):
    current_l = L[i]
    x = torch.randint(0, 64, (current_l,)).to(dtype=torch.int32)  # .to(dtype=torch.float)

    y_d = warp_histogram.warp_histogram(x.to(device='cuda'))
    y_h = warp_histogram_torch(x.to(dtype=torch.float))

    print(x)
    print(y_d)
    print(y_h)
    is_same = torch.equal(y_d.to(device='cpu'), y_h.to(dtype=torch.int32))
    print(f"result:{is_same}")

