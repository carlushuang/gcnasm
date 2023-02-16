cuda:
	nvcc -std=c++14 -O3 memcpy_kernel.cu -o memcpy_kernel.exe

rocm:
	rm -rf memcpy_kernel.cu.hip.cc memcpy_kernel.exe
	/opt/rocm/bin/hipify-perl memcpy_kernel.cu > memcpy_kernel.cu.hip.cc
	/opt/rocm/bin/hipcc memcpy_kernel.cu.hip.cc -std=c++14 -O3 --amdgpu-target=gfx90a -o memcpy_kernel.exe
