from __future__ import print_function
import subprocess
import os
import shutil
import sys

k_HSACO = "kernel.co"
k_HSAKN = "kernel_func"
k_WS = "build"
k_CPP_SRC = "bench.cpp"
k_CPP_TARGET = "bench.exe"
k_ASM_SRC = "kernel.s"
k_ASM_TARGET = k_HSACO
k_ARCH = "gfx942"
k_INST_LOOP = [256, 512, 768, 1024]
USE_HIP_CLANG = True

class cpp_src_t:
    def get_cxxflags(self):
        if USE_HIP_CLANG:
            return ' --offload-arch={} '.format(k_ARCH)
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
                cmd = "/opt/rocm/bin/hipcc "
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

#define HIP_CALL(call) do{{  \\
    hipError_t err = call;  \\
    if(err != hipSuccess){{  \\
        printf("[hiperror](%d) fail to call %s",(int)err,#call);    \\
        exit(0);            \\
    }}                      \\
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

    for(i=0;i<warm_ups;i++)
        HIP_CALL(hipModuleLaunchKernel(kernel_func, gdx,1,1, bdx,1,1,  0, 0, NULL, (void**)&config ));

    hipEventCreate(&evt_00);
    hipEventCreate(&evt_11);

    hipCtxSynchronize();
    hipEventRecord(evt_00, NULL);
    for(i=0;i<total_loop;i++)
        HIP_CALL(hipModuleLaunchKernel(kernel_func, gdx,1,1, bdx,1,1,  0, 0, NULL, (void**)&config ));

    float elapsed_ms;
    hipEventRecord(evt_11, NULL);
    hipEventSynchronize(evt_11);
    hipCtxSynchronize();
    hipEventElapsedTime(&elapsed_ms, evt_00, evt_11);
    hipEventDestroy(evt_00);
    hipEventDestroy(evt_11);

    float time_per_loop = elapsed_ms/total_loop;
    float tips = (double)inst_loop*inst_blocks*num_cu*bdx/time_per_loop/1e9;

    //printf("CU:%d, inst:%s, TIPS:%.3f(2x:%.3f, 4x:%.3f), cost:%fms per loop\\n", num_cu, argv[1], tips, 2*tips, 4*tips, time_per_loop);
    printf("%d\\t%s\\t%.3f\\t%.3f\\t%.3f\\t%.3fms\\n", num_cu, argv[1], tips, 2*tips, 4*tips, time_per_loop);
}}
'''.format(hsaco=k_HSACO, hsakn=k_HSAKN)
        return src
    def write(self,f):
        f.write(self.get_src())

class asm_src_t:
    def __init__(self,arch,bench_inst):
        self.arch = arch
        self.bench_inst = bench_inst
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
                #cmd += "-x assembler -target amdgcn--amdhsa -mcpu={} -mno-code-object-v3".format(self.arch) + " "
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
        if True:
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
.set inst_loop, 256

kernel_func:
    s_load_dword        s[s_blocks], s[0:1], 8
    s_waitcnt           lgkmcnt(0)
L_kernel_start:
    s_sub_u32 s[s_blocks], s[s_blocks], 1
    .itr = 0
    .rept inst_loop
        {bench_inst}
        .itr = .itr+4
        .if .itr > (v_end-4+1)
            .itr = 0
        .endif
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
    - {{ .name: dummy_ptr,   .size: 8, .offset:   0, .value_kind: global_buffer, .value_type: f32, .address_space: global, .is_const: false}}
    - {{ .name: inst_blocks, .size: 4, .offset:   8, .value_kind: by_value, .value_type: i32}}
...
.end_amdgpu_metadata

'''.format(bench_inst=self.bench_inst)
        else:
            asm_src='''\
.hsa_code_object_version 2,0
.hsa_code_object_isa {arch_str}, "AMD", "AMDGPU"

.text
.p2align 8
.amdgpu_hsa_kernel kernel_func

.set k_bdx,     256     ; should be 256 in bdx
.set k_end,     12
.set v_end,     255     ; hard code to this to let occupancy to be 1.  65536 / 256 = 256
.set s_blocks,  12
.set s_end,     31
.set inst_loop, 256

kernel_func:
    .amd_kernel_code_t
        enable_sgpr_kernarg_segment_ptr     = 1
        user_sgpr_count                     = 2
        enable_sgpr_workgroup_id_x = 1
        enable_sgpr_workgroup_id_y = 1
        enable_vgpr_workitem_id             = 0
        is_ptr64                            = 1
        float_mode                          = 2
        wavefront_sgpr_count                = s_end+1+2*3    ; VCC, FLAT_SCRATCH and XNACK must be counted
        workitem_vgpr_count                 = v_end+1
        granulated_workitem_vgpr_count      = v_end/4  ; (workitem_vgpr_count-1)/4
        granulated_wavefront_sgpr_count     = (s_end+2*3)/8     ; (wavefront_sgpr_count-1)/8
        kernarg_segment_byte_size           = k_end
        workgroup_group_segment_byte_size   = 0
    .end_amd_kernel_code_t

    s_load_dword        s[s_blocks], s[0:1], 8
    s_waitcnt           lgkmcnt(0)
L_kernel_start:
    s_sub_u32 s[s_blocks], s[s_blocks], 1
    .itr = 0
    .rept inst_loop
        {bench_inst}
        .itr = .itr+4
        .if .itr > (v_end-4+1)
            .itr = 0
        .endif
    .endr
    s_cmp_gt_u32 s[s_blocks], 0
    s_cbranch_scc1 L_kernel_start

    s_endpgm
'''.format(arch_str=self.arch_str, bench_inst=self.bench_inst)
        return asm_src
    def write(self,f):
        f.write(self.get_src())

bench_inst_dict = [
    ("v_add_co_u32",     "v[.itr], vcc, v[.itr+1], v[.itr+2]"),
    ("v_addc_co_u32",    "v[.itr], vcc, v[.itr+1], v[.itr+2], vcc"),
    ("v_or_b32",         "v[.itr], v[.itr+1], v[.itr+2]"),
    ("v_lshl_or_b32",    "v[.itr], v[.itr+1], v[.itr+2], v[.itr+3]"),
    ("v_lshlrev_b64",    "v[.itr:.itr+1], 8, v[.itr+2:.itr+3]"),
    ("v_mul_lo_u32",     "v[.itr], v[.itr+1], v[.itr+2]"),
    ("v_mul_hi_u32",     "v[.itr], v[.itr+1], v[.itr+2]"),
    ("v_mad_u32_u24",    "v[.itr], v[.itr+1], v[.itr+2], v[.itr+3]"),
    ("v_mad_u64_u32",    "v[.itr+0:.itr+1], vcc, v[.itr+2], v[.itr+3], v[.itr+0:.itr+1]"),
    ("v_mul_i32_i24",    "v[.itr], v[.itr+1], v[.itr+2]"),
    ("v_add_lshl_u32",    "v[.itr], v[.itr+1], v[.itr+2], v[.itr+3]"),
    ("v_sad_u8",        "v[.itr], v[.itr+1], v[.itr+2], v[.itr+3]"),
    ("v_sad_u16",        "v[.itr], v[.itr+1], v[.itr+2], v[.itr+3]"),
    ("v_sad_u32",        "v[.itr], v[.itr+1], v[.itr+2], v[.itr+3]"),

    ("v_dot2_f32_f16",  "v[.itr], v[.itr+1], v[.itr+2], v[.itr+3]"),
    ("v_dot4_i32_i8",   "v[.itr], v[.itr+1], v[.itr+2], v[.itr+3]"),
    ("v_pk_fma_f16",    "v[.itr], v[.itr+1], v[.itr+2], v[.itr+3]"),
    ("v_swap_b32",       "v[.itr], v[.itr+1]"),

    ("v_fmac_f32",      "v[.itr], v[.itr+1], v[.itr+2]"),
    ("v_mac_f32",       "v[.itr], v[.itr+1], v[.itr+2]"),
    ("v_mad_f32",       "v[.itr], v[.itr+1], v[.itr+2], v[.itr+3]"),
    ("v_mac_f16",       "v[.itr], v[.itr+1], v[.itr+2]"),
    ("v_pk_mul_f16",    "v[.itr], v[.itr+1], v[.itr+2], op_sel_hi:[1,1]"),
    ("v_pk_mul_f16",    "v[.itr], v[.itr+1], v[.itr+2]"),
    ("v_sin_f32",       "v[.itr], v[.itr+1]"),
    ("v_cos_f16",       "v[.itr], v[.itr+1]"),
    ("v_exp_f16",       "v[.itr], v[.itr+1]"),
    ("v_sqrt_f32",       "v[.itr], v[.itr+1]")
]

benched_inst_dict = dict()

def bench():
    def prepare_cpp():
        cpp_src = cpp_src_t()
        with open(os.path.join(k_WS, k_CPP_SRC), "w") as f:
            cpp_src.write(f)
        cpp_src.compile(k_CPP_SRC, k_CPP_TARGET, k_WS)
    def prepare_asm(arch, bench_inst):
        inst = bench_inst.split(' ')[0]
        if inst in benched_inst_dict:
            cnt = benched_inst_dict[inst]
            cnt = cnt+1
            inst = inst +'_{}'.format(cnt)
            benched_inst_dict[inst] = cnt
        else:
            benched_inst_dict[inst] = 0
        asm_src = asm_src_t(arch, bench_inst)
        src_path = os.path.join(k_WS, k_ASM_SRC)
        target_path = os.path.join(k_WS, k_ASM_TARGET)
        if os.path.exists(src_path):
            os.remove(src_path)
        if os.path.exists(target_path):
            os.remove(target_path)

        with open(src_path, "w") as f:
            asm_src.write(f)
        asm_src.compile(k_ASM_SRC,k_ASM_TARGET,k_WS)
        asm_src.disassemble(k_ASM_TARGET, k_ASM_TARGET+".dump.{}.s".format(inst), k_WS)

    def run_bench(bench_inst):
        def do_run():
            if not os.path.exists(k_CPP_TARGET):
                print("not exist {}, fail to run".format(k_CPP_TARGET))
                return
            if not os.path.exists(k_HSACO):
                print("not exist {}, fail to run".format(k_HSACO))
                return
            inst = bench_inst.split(' ')[0]
            cmd = "./{} {}".format(k_CPP_TARGET, inst)
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
    print("CU\tinstruction\ttips\t2x\t4x\tper_loop")
    for item in bench_inst_dict:
        bench_inst = item[0] + " " + item[1]
        prepare_asm(k_ARCH, bench_inst)
        run_bench(bench_inst)

def check_hip_clang():
    # return True/False
    return os.path.exists('/opt/rocm/llvm/bin/clang++')

class test_suite:
    def __init__(self):
        pass
    def __del__(self):
        pass


#gen_cpp()
#gen_asm("9,0,6","v_fmac_f32 v[.itr], v[.itr+1], v[.itr+2]")
USE_HIP_CLANG = check_hip_clang()
bench()
