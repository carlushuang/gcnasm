module attributes {llvm.target_triple = "amdgcn-amd-amdhsa"} {
  llvm.func @pin_mfma2(%pa: !llvm.ptr<1>, %pb: !llvm.ptr<1>, %pc: !llvm.ptr<1>)
      attributes {rocdl.kernel} {
    %tid = rocdl.workitem.id.x : i32
    // per-lane VMEM loads: pa[tid], pb[tid] (each a <4xf16> fragment)
    %pai = llvm.getelementptr %pa[%tid] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, vector<4xf16>
    %pbi = llvm.getelementptr %pb[%tid] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, vector<4xf16>
    %pci = llvm.getelementptr %pc[%tid] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, vector<4xf32>
    %a = llvm.load %pai : !llvm.ptr<1> -> vector<4xf16>
    %b = llvm.load %pbi : !llvm.ptr<1> -> vector<4xf16>
    %c0 = llvm.mlir.constant(dense<0.0> : vector<4xf32>) : vector<4xf32>
    %r0 = llvm.mlir.constant(0 : i32) : i32
    %r8 = llvm.mlir.constant(8 : i32) : i32
    %ai = llvm.bitcast %a : vector<4xf16> to vector<2xi32>
    %bi = llvm.bitcast %b : vector<4xf16> to vector<2xi32>
    %ap = llvm.call_intrinsic "llvm.amdgcn.pin.agpr"(%ai, %r0) : (vector<2xi32>, i32) -> vector<2xi32>
    %bp = llvm.call_intrinsic "llvm.amdgcn.pin.agpr"(%bi, %r8) : (vector<2xi32>, i32) -> vector<2xi32>
    %af = llvm.bitcast %ap : vector<2xi32> to vector<4xf16>
    %bf = llvm.bitcast %bp : vector<2xi32> to vector<4xf16>
    %d = rocdl.mfma.f32.16x16x16f16 %af, %bf, %c0, 0, 0, 0 :
        (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
    llvm.store %d, %pci : vector<4xf32>, !llvm.ptr<1>
    llvm.return
  }
}
