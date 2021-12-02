/opt/rocm/hip/bin/hipcc -x hip --cuda-gpu-arch=gfx908 -save-temps --cuda-device-only -c -O3 hgemm128x128.hip.cc -o hgemm128x128.hsaco
/opt/rocm/hip/bin/hipcc -std=c++14 -O3 hgemm.cc -o host.exe

./host.exe
