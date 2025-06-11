cuda:
	nvcc -std=c++17 -Xptxas -dlcm=cg -DUSE_NT_LOAD=0 -O3 bandwidth_kernel.cu -o bandwidth_kernel.exe
	# nvcc -std=c++17 -O3 bandwidth_kernel.cu -o bandwidth_kernel.exe

KSRC=kernel.s
KOUT=kernel.co
rocm:
	rm -rf bandwidth_kernel.exe $(KOBJ) $(KOUT)	
	/opt/rocm/llvm/bin/clang++ -x assembler -target amdgcn--amdhsa -mcpu=gfx942  $(KSRC) -o $(KOUT)
	#/opt/rocm/bin/hipify-perl bandwidth_kernel.cu > bandwidth_kernel.cu.hip.cc
	/opt/rocm/bin/hipcc bandwidth_kernel.cu.hip.cc -v --save-temps -Wno-gnu-line-marker -std=c++17 -O3 --offload-arch=gfx942 -o bandwidth_kernel.exe
