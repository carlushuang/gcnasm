"""
Test script for warp_bitonic_sort via pure pybind11 binding.

Run from the pybind/ directory:
    python test/test.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import warp_bitonic_sort

print("=" * 50)
print("Test warp_bitonic_sort (pybind11 + PyTorch)")
print("=" * 50)

L = [2, 4, 8, 16, 32, 64, 128, 256]

all_pass = True
for current_l in L:
    x = torch.randn([current_l], device="cuda", dtype=torch.float)

    # descending
    y = warp_bitonic_sort.warp_bitonic_sort(x, is_descending=True)
    expected, _ = torch.sort(x, descending=True)

    if torch.allclose(y, expected):
        print(f"  L={current_l:3d}  descending  PASS")
    else:
        print(f"  L={current_l:3d}  descending  FAIL")
        print(f"    input:    {x}")
        print(f"    got:      {y}")
        print(f"    expected: {expected}")
        all_pass = False

    # ascending
    y = warp_bitonic_sort.warp_bitonic_sort(x, is_descending=False)
    expected, _ = torch.sort(x, descending=False)

    if torch.allclose(y, expected):
        print(f"  L={current_l:3d}  ascending   PASS")
    else:
        print(f"  L={current_l:3d}  ascending   FAIL")
        print(f"    input:    {x}")
        print(f"    got:      {y}")
        print(f"    expected: {expected}")
        all_pass = False

print()
if all_pass:
    print("All tests PASSED!")
else:
    print("Some tests FAILED!")
    sys.exit(1)
