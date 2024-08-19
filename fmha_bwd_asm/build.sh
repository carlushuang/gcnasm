ARCH=gfx942
/opt/rocm/llvm/bin/clang++ -x assembler -target amdgcn--amdhsa --offload-arch=$ARCH shaders/bwd_a16.s -o kernel.co
/opt/rocm/bin/hipcc --offload-arch=$ARCH  bwd.cpp -o bwd.exe


# run
./bwd.exe b=1 h=16 s=16384 d=128 dump_result=0 init_pattern=0
