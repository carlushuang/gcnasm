ARCH=gfx942
/opt/rocm/llvm/bin/clang++ -x assembler -target amdgcn--amdhsa --offload-arch=$ARCH shaders/bwd_a32.s -o kernel.co
/opt/rocm/bin/hipcc --offload-arch=$ARCH  bwd.cpp -o bwd.exe


# run
./bwd.exe b=1 h=16 s=16384 d=128 dump_result=0 init_pattern=0

###commandline option####
#atm=0: a16,default; --corresponding kernel name with a16
#atm=1: a32; --kernel name with a32
#subk=128: 32x128tile,default;
#subk=192: 16x192tile; --kernel name with 16x192
#mask=0: non-causal,default;
#mask=1: causal; --kernel name with m1k1

###run mask test#####
###Kernel: bwd_a32_m1k0.s , command_line: mask=1
###Kernel: bwd_a32_m1k1.s , command_line: mask=1 mask_kb=1

####run in mi308####
###use noCoex version###
