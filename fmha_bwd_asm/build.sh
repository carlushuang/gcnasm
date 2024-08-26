ARCH=gfx942
/opt/rocm/llvm/bin/clang++ -x assembler -target amdgcn--amdhsa --offload-arch=$ARCH shaders/bwd_a32.s -o kernel.co
/opt/rocm/bin/hipcc --offload-arch=$ARCH  bwd.cpp -o bwd.exe


# run
./bwd.exe b=1 h=16 s=16384 d=128 dump_result=0 init_pattern=0


###run mask test#####
###Kernel: bwd_a32_m1k0.s , command_line: mask=1
###Kernel: bwd_a32_m1k1.s , command_line: mask=1 mask_kb=1

####run in mi308####
###use noCoex version###
