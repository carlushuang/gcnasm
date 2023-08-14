from __future__ import print_function
import subprocess
import os
import shutil
import sys

k_HSACO = "kernel.co"
k_HSAKN = "kernel_func"
k_WS = "build"
k_CPP_SRC = "co_exec.cpp"
k_CPP_TARGET = "co_exec.exe"
k_ASM_SRC = "kernel.s"
k_ASM_TARGET = k_HSACO
k_ARCH = "gfx90a"
USE_HIP_CLANG = True

class cpp_src_t:
    def get_cxxflags(self):
        if USE_HIP_CLANG:
            return ' -mcpu={} '.format(k_ARCH)
        else:
            return '`/opt/rocm/bin/hipconfig --cpp_config` -Wall -O2  -std=c++11 '
    def get_ldflags(self):
        if USE_HIP_CLANG:
            return ''
        else:
            return " -L/opt/rocm/hcc/lib -L/opt/rocm/lib -L/opt/rocm/lib64" \
                " -Wl,-rpath=/opt/rocm/hcc/lib:/opt/rocm/lib -ldl -lm -lpthread -lhc_am " \
                " -Wl,--whole-archive -lmcwamp -lhip_hcc -lhsa-runtime64 -lhsakmt -Wl,--no-whole-archive"
    def compile(self, src, target, working_dir):
        def do_compile():
            if USE_HIP_CLANG:
                cmd = "/opt/rocm/hip/bin/hipcc "
            else:
                cmd = "g++" + " "
            cmd += self.get_cxxflags() + " "
            cmd += src + " "
            cmd += self.get_ldflags() + " "
            cmd += "-o {}".format(target)
            proc = subprocess.Popen(cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,shell=True)
            (out, _) = proc.communicate()
            if proc.returncode != 0:
                print(cmd)
                msg = "Compilation error:\n"
                msg += str(out)
                raise RuntimeError(msg)
        save_dir = os.getcwd()
        os.chdir(working_dir)
        if os.path.exists(target):
            os.remove(target)
        do_compile()
        os.chdir(save_dir)

    def get_src(self):
        src = '''\
#include <stdio.h>
#include <hip/hip_runtime.h>
#include <random>
#include <iostream>

#define HIP_CALL(call) do{{             \\
    hipError_t err = call;              \\
    if(err != hipSuccess){{             \\
        printf("[hiperror](%d) fail to call %s",(int)err,#call);    \\
        exit(0);                        \\
    }}                                  \\
}} while(0)

#define HSACO "{hsaco}"
#define HSA_KERNEL "{hsakn}"

int main(int argc, char ** argv){{
    hipModule_t module;
    hipFunction_t kernel_func;
    hipEvent_t evt_00, evt_11;
    HIP_CALL(hipSetDevice(0));

    HIP_CALL(hipModuleLoad(&module, HSACO));
    HIP_CALL(hipModuleGetFunction(&kernel_func, module, HSA_KERNEL));

    int num_cu;
    {{
        hipDeviceProp_t dev_prop;
        hipDevice_t dev;
        HIP_CALL(hipGetDevice( &dev ));
        HIP_CALL(hipGetDeviceProperties( &dev_prop, dev ));
        num_cu = dev_prop.multiProcessorCount;
    }}

    int total_loop=100;
    int warm_ups = 5;
    int i;
    int inst_blocks = 1024*16;
    int inst_loop = 256;
    int bdx = 256;
    int gdx = num_cu;

    struct {{
        int * dummy_ptr;
        int inst_blocks;
    }} args;
    size_t arg_size = sizeof(args);
    args.inst_blocks = inst_blocks;
    void* config[] = {{HIP_LAUNCH_PARAM_BUFFER_POINTER, &args, HIP_LAUNCH_PARAM_BUFFER_SIZE,
                    &arg_size, HIP_LAUNCH_PARAM_END}};

    HIP_CALL(hipModuleLaunchKernel(kernel_func, gdx,1,1, bdx,1,1,  0, 0, NULL, (void**)&config ));

}}
'''.format(hsaco=k_HSACO, hsakn=k_HSAKN)
        return src
    def write(self,f):
        f.write(self.get_src())

class asm_src_t:
    def __init__(self,arch):
        self.arch = arch
        self.arch_str = ','.join([arch[3],arch[4],arch[5]])
    def get_asmflags():
        return ""
    def compile(self, src, target, working_dir):
        def do_compile():
            if USE_HIP_CLANG:
                cmd = "/opt/rocm/llvm/bin/clang++" + " "
                cmd += "-x assembler -target amdgcn--amdhsa -mcpu={} ".format(self.arch) + " "
            else:
                cmd = "/opt/rocm/hcc/bin/clang" + " "
                cmd += "-x assembler -target amdgcn--amdhsa -mcpu={} ".format(self.arch) + " "
            cmd += src + " "
            cmd += "-o {}".format(target)
            proc = subprocess.Popen(cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,shell=True)
            (out, _) = proc.communicate()
            if proc.returncode != 0:
                print(cmd)
                msg = "Compilation error:\n"
                msg += str(out)
                raise RuntimeError(msg)
        save_dir = os.getcwd()
        os.chdir(working_dir)
        if os.path.exists(target):
            os.remove(target)
        do_compile()
        os.chdir(save_dir)
    def disassemble(self, hsaco, output, working_dir):
        def do_disassembly():
            if not os.path.exists(hsaco):
                print("not exist {}, fail to disassembly".format(hsaco))
                return
            if USE_HIP_CLANG:
                cmd = "/opt/rocm/llvm/bin/llvm-objdump" + " "
                cmd += "--disassemble --mcpu={}".format(self.arch) + " "
            else:
                cmd = "/opt/rocm/hcc/bin/llvm-objdump" + " "
                cmd += "-disassemble -mcpu={}".format(self.arch) + " "
            cmd += hsaco + " "
            cmd += "> {}".format(output)
            proc = subprocess.Popen(cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,shell=True)
            (out, _) = proc.communicate()
            if proc.returncode != 0:
                print(cmd)
                msg = "Compilation error:\n"
                msg += str(out)
                raise RuntimeError(msg)
        save_dir = os.getcwd()
        os.chdir(working_dir)
        do_disassembly()
        os.chdir(save_dir)

    def get_src(self):
        #if USE_HIP_CLANG:
        asm_src='''\
.text
.global kernel_func
.p2align 8
.type kernel_func,@function

.set k_bdx,     256     ; should be 256 in bdx
.set k_end,     12
.set v_end,     255     ; hard code to this to let occupancy to be 1.  65536 / 256 = 256
.set s_blocks,  12
.set s_end,     31
.set inst_loop, 32

kernel_func:
    s_load_dword        s[s_blocks], s[0:1], 8
    s_waitcnt           lgkmcnt(0)
L_kernel_start:
    s_sub_u32 s[s_blocks], s[s_blocks], 1
    .itr = 0
    .rept inst_loop
        v_exp_f32 v67, v67
        v_mfma_f32_16x16x16f16 a[0:3]  , v[0:1], v[24:25] a[0:3]
        v_exp_f32 v64, v64
        v_mfma_f32_16x16x16f16 a[4:7]  , v[2:3], v[24:25] a[4:7]
        v_exp_f32 v65, v65
        v_mfma_f32_16x16x16f16 a[8:11] , v[4:5], v[24:25] a[8:11]
        v_exp_f32 v66, v66
        v_mfma_f32_16x16x16f16 a[12:15], v[6:7], v[24:25] a[12:15]
        

        v_mfma_f32_16x16x16f16 a[16:19], v[0:1], v[26:27] a[16:19]
        v_mfma_f32_16x16x16f16 a[20:23], v[2:3], v[26:27] a[20:23]
        v_mfma_f32_16x16x16f16 a[24:27], v[4:5], v[26:27] a[24:27]
        v_mfma_f32_16x16x16f16 a[28:31], v[6:7], v[26:27] a[28:31]

    .endr
    s_cmp_gt_u32 s[s_blocks], 0
    s_cbranch_scc1 L_kernel_start

    s_endpgm

.rodata
.p2align 6
.amdhsa_kernel kernel_func
    .amdhsa_group_segment_fixed_size 0
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr 256
    .amdhsa_next_free_sgpr 32
    .amdhsa_ieee_mode 0
    .amdhsa_dx10_clamp 0
    .amdhsa_accum_offset 256
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.version: [ 1, 0 ]
amdhsa.kernels:
  - .name: kernel_func
    .symbol: kernel_func.kd
    .sgpr_count: 32
    .vgpr_count: 256
    .kernarg_segment_align: 4
    .kernarg_segment_size: 12
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .wavefront_size: 64
    .reqd_workgroup_size : [256, 1, 1]
    .max_flat_workgroup_size: 256
    .args:
    - { .size: 8, .offset:   0, .value_kind: global_buffer, .address_space: global}
    - { .size: 4, .offset:   8, .value_kind: by_value}
...
.end_amdgpu_metadata

'''
        return asm_src
    def write(self,f):
        f.write(self.get_src())
def run():
    def prepare_cpp():
        cpp_src = cpp_src_t()
        with open(os.path.join(k_WS, k_CPP_SRC), "w") as f:
            cpp_src.write(f)
        cpp_src.compile(k_CPP_SRC, k_CPP_TARGET, k_WS)
    def prepare_asm(arch):
        asm_src = asm_src_t(arch)
        src_path = os.path.join(k_WS, k_ASM_SRC)
        target_path = os.path.join(k_WS, k_ASM_TARGET)
        if os.path.exists(src_path):
            os.remove(src_path)
        if os.path.exists(target_path):
            os.remove(target_path)

        with open(src_path, "w") as f:
            asm_src.write(f)
        asm_src.compile(k_ASM_SRC,k_ASM_TARGET,k_WS)
        asm_src.disassemble(k_ASM_TARGET, k_ASM_TARGET+".dump.s", k_WS)

    def run_bench():
        def do_run():
            if not os.path.exists(k_CPP_TARGET):
                print("not exist {}, fail to run".format(k_CPP_TARGET))
                return
            if not os.path.exists(k_HSACO):
                print("not exist {}, fail to run".format(k_HSACO))
                return
            cmd = "./{}".format(k_CPP_TARGET)
            proc = subprocess.Popen(cmd,
                        stdout=sys.stdout,
                        stderr=sys.stdout,shell=True)
            (out, _) = proc.communicate()
            if proc.returncode != 0:
                print(cmd)
                msg = "Compilation error:\n"
                msg += str(out)
                raise RuntimeError(msg)
        save_dir = os.getcwd()
        os.chdir(k_WS)
        do_run()
        os.chdir(save_dir)

    shutil.rmtree(k_WS,True)
    os.mkdir(k_WS)
    prepare_cpp()

    prepare_asm(k_ARCH)

    run_bench()

def check_hip_clang():
    # return True/False
    return os.path.exists('/opt/rocm/llvm/bin/clang++')

class test_suite:
    def __init__(self):
        pass
    def __del__(self):
        pass

USE_HIP_CLANG = check_hip_clang()
run()
