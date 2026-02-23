import warp_bitonic_sort
import torch

L = [2, 4, 8, 16, 32, 64, 128, 256]

for i in range(len(L)):
    current_l = L[i]
    x = torch.randn([current_l], device = 'cuda', dtype = torch.float)
    y = warp_bitonic_sort.warp_bitonic_sort(x, True)
    print(x)
    print(y)

