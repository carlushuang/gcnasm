import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from pin import pin_agpr

@flyc.kernel
def k(A: fx.Tensor, Cout: fx.Tensor):
    tid = fx.thread_idx.x
    bA = fx.rocdl.make_buffer_tensor(A)
    tA = fx.slice(fx.logical_divide(bA, fx.make_layout(4, 1)), (None, tid))
    tC = fx.slice(fx.logical_divide(Cout, fx.make_layout(4, 1)), (None, tid))
    cp = fx.make_copy_atom(fx.rocdl.BufferCopy64b(), fx.Float16)
    rA = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Float16)
    rC = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Float16)
    fx.copy_atom_call(cp, tA, rA)
    a = fx.arith.unwrap(fx.memref_load_vec(rA))
    print("PROBE type(a)=", type(a))
    a2 = pin_agpr(a, 0)
    print("PROBE a is a2:", a is a2, "type(a2)=", type(a2))
    fx.memref_store_vec(a2, rC)
    fx.copy_atom_call(cp, rC, tC)

@flyc.jit
def run(A: fx.Tensor, C: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
    k(A, C).launch(grid=(1,1,1), block=(64,1,1), stream=stream)

A = torch.randn(256, dtype=torch.float16).cuda()
C = torch.zeros(256, dtype=torch.float16).cuda()
run(A, C, stream=torch.cuda.current_stream()); torch.cuda.synchronize()
print("store diff:", (C.float()-A.float()).abs().max().item())
