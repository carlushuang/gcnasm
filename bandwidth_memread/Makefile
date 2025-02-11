cuda:
	nvcc -std=c++17 -Xptxas -dlcm=cg -DUSE_NT_LOAD=0 -O3 bandwidth_kernel.cu -o bandwidth_kernel.exe
	# nvcc -std=c++17 -O3 bandwidth_kernel.cu -o bandwidth_kernel.exe

rocm:
	rm -rf bandwidth_kernel.cu.hip.cc bandwidth_kernel.exe
	/opt/rocm/bin/hipify-perl bandwidth_kernel.cu > bandwidth_kernel.cu.hip.cc
	/opt/rocm/bin/hipcc bandwidth_kernel.cu.hip.cc -std=c++17 -O3 --offload-arch=gfx942 -o bandwidth_kernel.exe
