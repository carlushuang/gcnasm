# SPDX-License-Identifier: MIT
"""Running FlyDSL MFMA kernel with register pinning (low-level rocdl.mfma path).

High-level tiled_mma fragments are register-space fly.memrefs (not LLVM SSA
values at trace time), so they can't feed llvm.call_intrinsic. rocdl.mfma takes
LLVM vectors, which pin_agpr can pin. This loads A/B (4 f16/lane) into register
tensors, materializes them as vectors (memref_load_vec), pins the vectors to
AGPR, runs one 16x16x16 f16 MFMA, and stores C.

Correctness is checked by comparing the pinned kernel to the identical unpinned
kernel (pin is value-identity), which is robust without hand-deriving the MFMA
lane layout.

    python gemm_pin_rocdl.py
"""

import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr.typing import T

from pin import pin_agpr

LANES = 64  # one wave -> one 16x16x16 MFMA


def build(do_pin):
    @flyc.kernel
    def mfma_kernel(A: fx.Tensor, B: fx.Tensor, Cout: fx.Tensor):
        tid = fx.thread_idx.x
        bA = fx.rocdl.make_buffer_tensor(A)
        bB = fx.rocdl.make_buffer_tensor(B)

        # each lane owns 4 contiguous f16 of A and B
        tA = fx.slice(fx.logical_divide(bA, fx.make_layout(4, 1)), (None, tid))
        tB = fx.slice(fx.logical_divide(bB, fx.make_layout(4, 1)), (None, tid))
        tC = fx.slice(fx.logical_divide(Cout, fx.make_layout(4, 1)), (None, tid))

        copyB = fx.make_copy_atom(fx.rocdl.BufferCopy64b(), fx.Float16)
        copyC = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Float32)

        rA = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Float16)
        rB = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Float16)
        rC = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Float32)
        fx.copy_atom_call(copyB, tA, rA)
        fx.copy_atom_call(copyB, tB, rB)

        a = fx.arith.unwrap(fx.memref_load_vec(rA))   # vector<4xf16>
        b = fx.arith.unwrap(fx.memref_load_vec(rB))
        if do_pin:  # plain Python toggle at trace time
            a = pin_agpr(a, 0)
            b = pin_agpr(b, 8)

        czero = fx.arith.unwrap(fx.arith.constant_vector(0.0, T.vec(4, T.f32)))
        d = fx.rocdl.mfma_f32_16x16x16f16(T.vec(4, T.f32), [a, b, czero, 0, 0, 0])
        fx.memref_store_vec(d, rC)
        fx.copy_atom_call(copyC, rC, tC)

    @flyc.jit
    def run(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor,
            stream: fx.Stream = fx.Stream(None)):
        mfma_kernel(A, B, C).launch(grid=(1, 1, 1), block=(LANES, 1, 1), stream=stream)

    return run


if __name__ == "__main__":
    A = torch.randn(LANES * 4, dtype=torch.float16).cuda()
    B = torch.randn(LANES * 4, dtype=torch.float16).cuda()
    Cp = torch.zeros(LANES * 4, dtype=torch.float32).cuda()
    Cb = torch.zeros(LANES * 4, dtype=torch.float32).cuda()

    build(do_pin=True)(A, B, Cp, stream=torch.cuda.current_stream())
    build(do_pin=False)(A, B, Cb, stream=torch.cuda.current_stream())
    torch.cuda.synchronize()

    maxdiff = (Cp - Cb).abs().max().item()
    print("pinned vs baseline max diff:", maxdiff)
    print("Result correct:", maxdiff == 0.0)
