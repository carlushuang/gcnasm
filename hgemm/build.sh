##HIP version build##

#/opt/rocm/hip/bin/hipcc -x hip --cuda-gpu-arch=gfx908  --cuda-device-only -c -O3 hgemm128x128.hip.cc -o hgemm128x128.hsaco
#/opt/rocm/hip/bin/hipcc -std=c++14 -O3 hgemm.cc -o HIP_hgemm.exe
#./HIP_hgemm.exe

##ASM version build##
#HSACO=kernel_asm.co
#rm -rf $HSACO ./ASM_hgemm.exe
#/opt/rocm/llvm/bin/clang++ -x assembler -target amdgcn--amdhsa -mcpu=gfx908  hgemm128x128.s -o $HSACO
#/opt/rocm/hip/bin/hipcc hgemm.cc -mcpu=gfx908 -o ASM_hgemm.exe
#for i in $(seq 1 1);
#do ./ASM_hgemm.exe ;
#done

##ASM-MFMA version build##
HSACO=kernel_asm.co
rm -rf $HSACO ./ASM_hgemm.exe
/opt/rocm/llvm/bin/clang++ -x assembler -target amdgcn--amdhsa -mcpu=gfx908  hgemm128x128.MAI.s -o $HSACO
/opt/rocm/hip/bin/hipcc hgemm.cc -mcpu=gfx908 -o ASM_hgemm.exe
for i in $(seq 1 10);
do ./ASM_hgemm.exe ;
done