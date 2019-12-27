cuda:
	nvcc memcpy_kernel.cu -o memcpy_kernel.cuda

rocm:
	rm -rf memcpy_kernel.cu.hip.cc memcpy_kernel.rocm
	/opt/rocm/bin/hipify-perl memcpy_kernel.cu > memcpy_kernel.cu.hip.cc
	/opt/rocm/bin/hipcc memcpy_kernel.cu.hip.cc -std=c++11 --amdgpu-target=gfx906 -o memcpy_kernel.rocm