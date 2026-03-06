#!/usr/bin/env python3
"""
Build, compile, launch, and verify an MLIR elementwise-add kernel on AMD GPU.

  python3 test_add_kernel.py [chip]     # default: gfx942

Pipeline:
  Python (mlir.ir)  →  MLIR (llvm+rocdl)  →  LLVM IR  →  .hsaco
                                                             ↓
                       torch (reference)  ←  HIP launch  ←──┘
"""

import ctypes
import math
import os
import shutil
import struct
import subprocess
import sys
import tempfile

import torch

from mlir.ir import (
    Attribute,
    Context,
    DenseI32ArrayAttr,
    F32Type,
    FunctionType,
    InsertionPoint,
    IntegerAttr,
    IntegerType,
    Location,
    Module,
    Type,
    TypeAttr,
    UnitAttr,
)
from mlir.dialects import llvm, rocdl
from mlir.passmanager import PassManager

BLOCK_SIZE = 256

# ───────────────────────────────────────────────────────────────────────────
# 1. Build the kernel IR (bare-pointer ABI for easy HIP launch)
# ───────────────────────────────────────────────────────────────────────────

def build_vector_add_module() -> Module:
    """
    Builds:
        void vector_add(float *A, float *B, float *C, int64_t N)
    using MLIR's llvm + rocdl dialects.
    """
    ctx = Context()
    with ctx, Location.unknown():
        module = Module.create()
        ptr = llvm.PointerType.get()
        i32 = IntegerType.get_signless(32)
        i64 = IntegerType.get_signless(64)
        f32 = F32Type.get()
        llvm_ft = Type.parse("!llvm.func<void (ptr, ptr, ptr, i64)>")
        no_of = Attribute.parse("#llvm.overflow<none>")
        gep_dyn = DenseI32ArrayAttr.get([-2147483648])

        with InsertionPoint(module.body):
            fn = llvm.LLVMFuncOp("vector_add", TypeAttr.get(llvm_ft))
            fn.operation.attributes["rocdl.kernel"] = UnitAttr.get()

            entry = fn.body.blocks.append(ptr, ptr, ptr, i64)
            compute_bb = fn.body.blocks.append()
            exit_bb = fn.body.blocks.append()

            with InsertionPoint(entry):
                A, B, C, N = entry.arguments
                tid = rocdl.workitem_id_x(i32)
                bid = rocdl.workgroup_id_x(i32)
                bs = llvm.ConstantOp(
                    i32, IntegerAttr.get(i32, BLOCK_SIZE)
                ).result
                off = llvm.MulOp(bid, bs, no_of).result
                gid32 = llvm.AddOp(off, tid, no_of).result
                gid = llvm.SExtOp(i64, gid32).result
                cond = llvm.ICmpOp(
                    llvm.ICmpPredicate.slt, gid, N
                ).result
                llvm.CondBrOp(cond, [], [], compute_bb, exit_bb)

            with InsertionPoint(compute_bb):
                ap = llvm.GEPOp(ptr, A, [gid], gep_dyn, f32).result
                bp = llvm.GEPOp(ptr, B, [gid], gep_dyn, f32).result
                cp = llvm.GEPOp(ptr, C, [gid], gep_dyn, f32).result
                a_val = llvm.LoadOp(f32, ap).result
                b_val = llvm.LoadOp(f32, bp).result
                s = llvm.FAddOp(a_val, b_val).result
                llvm.StoreOp(s, cp)
                llvm.BrOp([], exit_bb)

            with InsertionPoint(exit_bb):
                llvm.ReturnOp()

        assert module.operation.verify()
        return module


# ───────────────────────────────────────────────────────────────────────────
# 2. Compile MLIR → LLVM IR → .hsaco
# ───────────────────────────────────────────────────────────────────────────

MLIR_TRANSLATE = shutil.which("mlir-translate-20") or shutil.which("mlir-translate")
ROCM_CLANG = "/opt/rocm/llvm/bin/clang"


def compile_to_hsaco(module: Module, chip: str, out_path: str) -> str:
    mlir_text = str(module)

    llvm_ir = subprocess.run(
        [MLIR_TRANSLATE, "--mlir-to-llvmir"],
        input=mlir_text, capture_output=True, text=True, check=True,
    ).stdout

    subprocess.run(
        [
            ROCM_CLANG,
            "-x", "ir", "-",
            "-target", "amdgcn-amd-amdhsa",
            f"-mcpu={chip}",
            "-O3",
            "-o", out_path,
        ],
        input=llvm_ir, capture_output=True, text=True, check=True,
    )
    return out_path


# ───────────────────────────────────────────────────────────────────────────
# 3. HIP runtime helpers (ctypes wrappers)
# ───────────────────────────────────────────────────────────────────────────

_hip = ctypes.CDLL("libamdhip64.so")


def _check(err):
    if err != 0:
        raise RuntimeError(f"HIP error {err}")


def hip_module_load(path: str):
    mod = ctypes.c_void_p()
    _check(_hip.hipModuleLoad(ctypes.byref(mod), path.encode()))
    return mod


def hip_get_function(mod, name: str):
    func = ctypes.c_void_p()
    _check(_hip.hipModuleGetFunction(ctypes.byref(func), mod, name.encode()))
    return func


def hip_launch_kernel(func, grid, block, args, shared_mem=0, stream=None):
    """
    args: list of ctypes objects (c_void_p for pointers, c_int64 for ints, …)
    """
    arg_ptrs = (ctypes.c_void_p * len(args))()
    for i, a in enumerate(args):
        arg_ptrs[i] = ctypes.cast(ctypes.pointer(a), ctypes.c_void_p)

    _check(
        _hip.hipModuleLaunchKernel(
            func,
            grid[0], grid[1], grid[2],
            block[0], block[1], block[2],
            shared_mem,
            stream,
            arg_ptrs,
            None,
        )
    )


def hip_device_synchronize():
    _check(_hip.hipDeviceSynchronize())


# ───────────────────────────────────────────────────────────────────────────
# 4. Test driver
# ───────────────────────────────────────────────────────────────────────────

def main():
    chip = sys.argv[1] if len(sys.argv) > 1 else "gfx942"
    N = 1024 * 1024

    # ── build & compile ──────────────────────────────────────────────────
    print(f"[1/4] Building MLIR module (bare-pointer, llvm+rocdl) …")
    module = build_vector_add_module()
    print(module)

    hsaco_path = os.path.join(tempfile.gettempdir(), "vector_add.hsaco")
    print(f"[2/4] Compiling to {hsaco_path}  (target {chip}) …")
    compile_to_hsaco(module, chip, hsaco_path)
    print(f"      {os.path.getsize(hsaco_path)} bytes")

    # ── prepare data ─────────────────────────────────────────────────────
    print(f"[3/4] Preparing data (N={N}) …")
    A = torch.randn(N, dtype=torch.float32, device="cuda")
    B = torch.randn(N, dtype=torch.float32, device="cuda")
    C = torch.zeros(N, dtype=torch.float32, device="cuda")
    ref = A + B

    # ── launch ───────────────────────────────────────────────────────────
    print(f"[4/4] Launching kernel …")
    hip_mod = hip_module_load(hsaco_path)
    func = hip_get_function(hip_mod, "vector_add")

    grid = (math.ceil(N / BLOCK_SIZE), 1, 1)
    block = (BLOCK_SIZE, 1, 1)

    args = [
        ctypes.c_void_p(A.data_ptr()),
        ctypes.c_void_p(B.data_ptr()),
        ctypes.c_void_p(C.data_ptr()),
        ctypes.c_int64(N),
    ]
    hip_launch_kernel(func, grid, block, args)
    hip_device_synchronize()

    # ── verify ───────────────────────────────────────────────────────────
    max_err = (C - ref).abs().max().item()
    print(f"\n      max |C_kernel - C_torch| = {max_err}")
    if max_err < 1e-5:
        print("      PASSED ✓")
    else:
        print("      FAILED ✗")
        sys.exit(1)


if __name__ == "__main__":
    main()
