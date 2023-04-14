#ifdef __NVCC__
using fp32 =  float;
using fp32x2 = float2;
using fp32x4 = float4;
#else
using fp32 = __attribute__((__ext_vector_type__(1))) float;
using fp32x2 = __attribute__((__ext_vector_type__(2))) float;
using fp32x4 = __attribute__((__ext_vector_type__(4))) float;
#endif

#include <string>
#include <dlfcn.h>
#include <stdio.h>
#include <array>
#include <unistd.h>
#include <fstream>
#include <iostream>

#ifndef PWD
#define PWD ""
#endif

#ifndef TMP
#define TMP "tmp/"
#endif

#ifndef MODULE_FILE
#define MODULE_FILE "memcpy_kernel.hip.cc"
#endif

#ifndef MODULE_SO
#define MODULE_SO "memcpy_kernel.so"
#endif


typedef void (*memtpy_kernel_t)(void* dst, const void* src, uint32_t dwords);

struct memcpy_module_t{
    memcpy_module_t() : handle(nullptr)
    {
        emit();
        compile();
        load();
    }

    ~memcpy_module_t() {
        if(handle)
            dlclose(handle);
    }

    std::string get_src_path() const
    {
        return std::string(TMP) + std::string("/") + std::string(MODULE_FILE);
    }

    std::string get_so_path() const
    {
        return std::string(TMP) + std::string("/") + std::string(MODULE_SO);
    }

    bool file_exists(const std::string path) const
    {
        return access(path.c_str(), 0) == 0;
    }
    
    void emit() {
        static const std::string kernel_code =
"#include <hip/hip_runtime.h>\n"
"#include <stdint.h>\n"
"#ifdef __NVCC__\n"
"using fp32 =  float;\n"
"using fp32x2 = float2;\n"
"using fp32x4 = float4;\n"
"#else\n"
"using fp32 = __attribute__((__ext_vector_type__(1))) float;\n"
"using fp32x2 = __attribute__((__ext_vector_type__(2))) float;\n"
"using fp32x4 = __attribute__((__ext_vector_type__(4))) float;\n"
"#endif\n"
"#define BLOCK_SIZE 256\n"
"template<typename T>\n"
"__global__\n"
"void memcpy_kernel(T* __restrict__ dst, const T* __restrict__ src, uint32_t n){\n"
"    int idx = (blockIdx.x * BLOCK_SIZE + threadIdx.x);\n"
"    if(idx < n)\n"
"        dst[idx] = src[idx];\n"
"}\n"
"\n"
"template<typename T>\n"
"struct memcpy_with_type\n"
"{\n"
"    static void run(void * B, const void * A, int dwords){\n"
"        int bx = BLOCK_SIZE;\n"
"        int pixels = dwords * 4 / sizeof(T);\n"
"        int gx = (pixels + BLOCK_SIZE - 1) / BLOCK_SIZE;\n"
"        memcpy_kernel<T><<<gx, bx>>>(reinterpret_cast<T*>(B), reinterpret_cast<const T*>(A), pixels);\n"
"    }\n"
"};\n"
"\n"
"extern \"C\"\n"
"void memcpy_fp32(void* dst, const void* src, uint32_t dwords)\n"
"{\n"
"    memcpy_with_type<fp32>::run(dst, src, dwords);\n"
"}\n"
"\n"
"extern \"C\"\n"
"void memcpy_fp32x2(void* dst, const void* src, uint32_t dwords)\n"
"{\n"
"    memcpy_with_type<fp32x2>::run(dst, src, dwords);\n"
"}\n"
"\n"
"extern \"C\"\n"
"void memcpy_fp32x4(void* dst, const void* src, uint32_t dwords)\n"
"{\n"
"    memcpy_with_type<fp32x4>::run(dst, src, dwords);\n"
"}\n";

        if(file_exists(get_src_path())){
            return ;
        }
        std::ofstream out(get_src_path());
        if(!out) {
            printf("error while open file %s\n", get_src_path().c_str());
            exit(0);
        }
        out << kernel_code;
        out.close();
        printf("emitted kernel => %s\n", get_src_path().c_str());
        fflush(stdout);
    }

    void compile()
    {
        if(file_exists(get_so_path())){
            // printf("skip compile due to so %s already there\n", get_so_path().c_str());
            return ;
        }
        std::string cmd = std::string("/opt/rocm/bin/hipcc -std=c++17 --amdgpu-target=gfx90a -fPIC -shared -x hip ") +
                                get_src_path() +
                                std::string(" -o ") + get_so_path() +
                                std::string(" 2>&1 ");  // redirect stderr to stdout
        FILE * h = popen(cmd.c_str(), "r");
        if(!h){
            printf("Not able to open cmd\n");
            exit(0);
        }
        std::array<char, 128> buffer;
        std::string result;
        while (fgets(buffer.data(), 128, h) != nullptr) {
            result += buffer.data();
        }
        int err = pclose(h);
        if(err != 0){
            printf("=== error while compile ===\n");
            fflush(stdout);
            printf("%s\n", result.c_str());
            exit(0);
        }
        printf("compile success so => %s\n", get_so_path().c_str());
        fflush(stdout);
    }

    void load()
    {
        handle = dlopen(get_so_path().c_str(), RTLD_LAZY);
        if(!handle){
            printf("unable to open so %s\n", get_so_path().c_str());
            exit(0);
        }
#define _LOAD_SYM(f_, name_)                                                \
            do {                                                            \
                f_ = reinterpret_cast<decltype(f_)>(dlsym(handle, name_));  \
                if(!f_){                                                    \
                    printf("unable to find function %s\n", name_);          \
                    exit(0);                                                \
                }                                                           \
            }while(0)

        _LOAD_SYM(memcpy_fp32, "memcpy_fp32");
        _LOAD_SYM(memcpy_fp32x2, "memcpy_fp32x2");
        _LOAD_SYM(memcpy_fp32x4, "memcpy_fp32x4");
#undef _LOAD_SYM

    }

    void * handle;
    memtpy_kernel_t memcpy_fp32;
    memtpy_kernel_t memcpy_fp32x2;
    memtpy_kernel_t memcpy_fp32x4;
};
