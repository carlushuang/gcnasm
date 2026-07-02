# SPDX-License-Identifier: MIT
"""FlyDSL tiled MMA GEMM with register pinning: A/B inputs -> AGPR, C -> VGPR.

Modeled on FlyDSL examples/03-tiledMma.py. The only additions are the two
`pin_agpr` calls on the A/B fragments (see pin.py). The accumulator fragment is
left in VGPR (its natural file), yielding the mixed  v_mfma v[C], a[A], a[B]
form once compiled with a pin-patched LLVM.

Run inside an aiter/FlyDSL container whose FlyDSL is built against the
pin-patched LLVM (see README.md):

    python gemm_pin.py
"""

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx

from pin import pin_agpr  # pin_vgpr also available

block_m = 64
block_n = 64
block_k = 8

# AGPR base registers for the A and B fragments (chosen to not overlap).
PIN_A_AGPR = 0
PIN_B_AGPR = 64


@flyc.kernel
def gemm_kernel(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor):
    tid = fx.thread_idx.x
    bid = fx.block_idx.x

    A = fx.rocdl.make_buffer_tensor(A)
    B = fx.rocdl.make_buffer_tensor(B)
    C = fx.rocdl.make_buffer_tensor(C)

    bA = fx.slice(fx.zipped_divide(A, (block_m, block_k)), (None, bid))
    bB = fx.slice(fx.zipped_divide(B, (block_n, block_k)), (None, bid))
    bC = fx.slice(fx.zipped_divide(C, (block_m, block_n)), (None, bid))

    mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 4, fx.Float32))
    tiled_mma = fx.make_tiled_mma(mma_atom, fx.make_layout((2, 2, 1), (1, 2, 0)))
    thr_mma = tiled_mma.thr_slice(tid)

    copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
    tiled_copy_A = fx.make_tiled_copy_A(copy_atom, tiled_mma)
    tiled_copy_B = fx.make_tiled_copy_B(copy_atom, tiled_mma)
    tiled_copy_C = fx.make_tiled_copy_C(copy_atom, tiled_mma)

    thr_copy_A = tiled_copy_A.get_slice(tid)
    thr_copy_B = tiled_copy_B.get_slice(tid)
    thr_copy_C = tiled_copy_C.get_slice(tid)

    copy_src_A = thr_copy_A.partition_S(bA)
    copy_src_B = thr_copy_B.partition_S(bB)
    copy_dst_C = thr_copy_C.partition_S(bC)

    frag_A = thr_mma.make_fragment_A(bA)
    frag_B = thr_mma.make_fragment_B(bB)
    frag_C = thr_mma.make_fragment_C(bC)  # accumulator: stays in VGPR

    copy_frag_A = thr_copy_A.retile(frag_A)
    copy_frag_B = thr_copy_B.retile(frag_B)
    copy_frag_C = thr_copy_C.retile(frag_C)

    fx.copy(copy_atom, copy_src_A, copy_frag_A, pred=None)
    fx.copy(copy_atom, copy_src_B, copy_frag_B, pred=None)

    # --- register pins ---------------------------------------------------
    # Pin the A/B input fragments to the AGPR file. The load feeding each
    # fragment is then folded to an AGPR-born load and the MFMA reads a[A]/a[B];
    # frag_C is left in VGPR so the accumulator stays v[C].
    #
    # NOTE: frag_A/frag_B are register-backed tensors. pin_agpr() pins the
    # fragment's underlying SSA value; if a fragment lowers to several register
    # values, pin each (loop over frag.__extract_to_ir_values__()) at
    # PIN_*_AGPR + offset. This hook is the expected in-container iteration point
    # (see README "Known open point").
    frag_A = pin_agpr(frag_A, PIN_A_AGPR)
    frag_B = pin_agpr(frag_B, PIN_B_AGPR)
    # ---------------------------------------------------------------------

    frag_C.fill(0)
    fx.gemm(mma_atom, frag_C, frag_A, frag_B, frag_C)

    fx.copy(copy_atom, copy_frag_C, copy_dst_C, pred=None)


@flyc.jit
def tiledMma(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor,
             stream: fx.Stream = fx.Stream(None)):
    gemm_kernel(A, B, C).launch(grid=(1, 1, 1), block=(256, 1, 1), stream=stream)


if __name__ == "__main__":
    M, N, K = block_m, block_n, block_k
    A = torch.randn(M, K, dtype=torch.float32).cuda()
    B = torch.randn(N, K, dtype=torch.float32).cuda()
    C = torch.zeros(M, N, dtype=torch.float32).cuda()

    tiledMma(A, B, C, stream=torch.cuda.Stream())
    torch.cuda.synchronize()

    expected = A @ B.T
    ok = torch.allclose(C, expected, atol=1e-5, rtol=1e-5)
    print("Result correct:", ok)
    if not ok:
        print("Max diff:", (C - expected).abs().max().item())
