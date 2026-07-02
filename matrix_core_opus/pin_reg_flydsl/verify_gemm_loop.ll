; A K-loop with a loop-carried VGPR accumulator (4 tiles) and A/B inputs pinned
; to AGPR each iteration -- the backend shape of a larger tiled GEMM.
declare i32 @llvm.amdgcn.workitem.id.x()
declare <2 x i32> @llvm.amdgcn.pin.agpr.v2i32(<2 x i32>, i32 immarg)
declare <4 x float> @llvm.amdgcn.mfma.f32.16x16x16f16(<4 x half>, <4 x half>, <4 x float>, i32 immarg, i32 immarg, i32 immarg)

define amdgpu_kernel void @gemm_loop(ptr addrspace(1) %pa, ptr addrspace(1) %pb, ptr addrspace(1) %pc, i32 %nk) {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  br label %loop
loop:
  %k  = phi i32 [ 0, %entry ], [ %kn, %loop ]
  %c0 = phi <4 x float> [ zeroinitializer, %entry ], [ %d0, %loop ]
  %c1 = phi <4 x float> [ zeroinitializer, %entry ], [ %d1, %loop ]
  %c2 = phi <4 x float> [ zeroinitializer, %entry ], [ %d2, %loop ]
  %c3 = phi <4 x float> [ zeroinitializer, %entry ], [ %d3, %loop ]
  %off = add i32 %k, %tid
  %ga  = getelementptr <4 x half>, ptr addrspace(1) %pa, i32 %off
  %a   = load <4 x half>, ptr addrspace(1) %ga
  %ai  = bitcast <4 x half> %a to <2 x i32>
  %ap  = call <2 x i32> @llvm.amdgcn.pin.agpr.v2i32(<2 x i32> %ai, i32 0)
  %af  = bitcast <2 x i32> %ap to <4 x half>
  %o0 = add i32 %off, 0
  %o1 = add i32 %off, 64
  %o2 = add i32 %off, 128
  %o3 = add i32 %off, 192
  %gb0 = getelementptr <4 x half>, ptr addrspace(1) %pb, i32 %o0
  %gb1 = getelementptr <4 x half>, ptr addrspace(1) %pb, i32 %o1
  %gb2 = getelementptr <4 x half>, ptr addrspace(1) %pb, i32 %o2
  %gb3 = getelementptr <4 x half>, ptr addrspace(1) %pb, i32 %o3
  %b0 = load <4 x half>, ptr addrspace(1) %gb0
  %b1 = load <4 x half>, ptr addrspace(1) %gb1
  %b2 = load <4 x half>, ptr addrspace(1) %gb2
  %b3 = load <4 x half>, ptr addrspace(1) %gb3
  %b0i = bitcast <4 x half> %b0 to <2 x i32>
  %b1i = bitcast <4 x half> %b1 to <2 x i32>
  %b2i = bitcast <4 x half> %b2 to <2 x i32>
  %b3i = bitcast <4 x half> %b3 to <2 x i32>
  %b0p = call <2 x i32> @llvm.amdgcn.pin.agpr.v2i32(<2 x i32> %b0i, i32 64)
  %b1p = call <2 x i32> @llvm.amdgcn.pin.agpr.v2i32(<2 x i32> %b1i, i32 66)
  %b2p = call <2 x i32> @llvm.amdgcn.pin.agpr.v2i32(<2 x i32> %b2i, i32 68)
  %b3p = call <2 x i32> @llvm.amdgcn.pin.agpr.v2i32(<2 x i32> %b3i, i32 70)
  %b0f = bitcast <2 x i32> %b0p to <4 x half>
  %b1f = bitcast <2 x i32> %b1p to <4 x half>
  %b2f = bitcast <2 x i32> %b2p to <4 x half>
  %b3f = bitcast <2 x i32> %b3p to <4 x half>
  %d0 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16f16(<4 x half> %af, <4 x half> %b0f, <4 x float> %c0, i32 0, i32 0, i32 0)
  %d1 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16f16(<4 x half> %af, <4 x half> %b1f, <4 x float> %c1, i32 0, i32 0, i32 0)
  %d2 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16f16(<4 x half> %af, <4 x half> %b2f, <4 x float> %c2, i32 0, i32 0, i32 0)
  %d3 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16f16(<4 x half> %af, <4 x half> %b3f, <4 x float> %c3, i32 0, i32 0, i32 0)
  %kn = add i32 %k, 1
  %done = icmp sge i32 %kn, %nk
  br i1 %done, label %exit, label %loop
exit:
  %r0 = phi <4 x float> [ %d0, %loop ]
  %r1 = phi <4 x float> [ %d1, %loop ]
  %r2 = phi <4 x float> [ %d2, %loop ]
  %r3 = phi <4 x float> [ %d3, %loop ]
  %pc1 = getelementptr <4 x float>, ptr addrspace(1) %pc, i32 1
  %pc2 = getelementptr <4 x float>, ptr addrspace(1) %pc, i32 2
  %pc3 = getelementptr <4 x float>, ptr addrspace(1) %pc, i32 3
  store <4 x float> %r0, ptr addrspace(1) %pc
  store <4 x float> %r1, ptr addrspace(1) %pc1
  store <4 x float> %r2, ptr addrspace(1) %pc2
  store <4 x float> %r3, ptr addrspace(1) %pc3
  ret void
}
