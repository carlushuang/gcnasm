ARCH=gfx942
/opt/rocm/llvm/bin/clang++ -x assembler -target amdgcn--amdhsa --offload-arch=$ARCH  bwd.s -o kernel.co
/opt/rocm/bin/hipcc --offload-arch=$ARCH  bwd.cpp -o bwd.exe


# run
./bwd.exe
