#include <iostream>
#include <string>
#include <map>
#include <assert.h>
#include <stdio.h>
#include <hip/hip_runtime.h>

#include "config.h"

#define HIP_CALL(call) do{  \
    hipError_t err = call;  \
    if(err != hipSuccess){  \
        printf("[hiperror](%d) fail to call %s",(int)err,#call);    \
        exit(0);            \
    }                       \
} while(0)

#define HSACO "memcpy_2d_example_gfx1030.hsaco"
#define HSA_KERNEL "memcpy_2d_example_gfx1030"

#define ABS(x) ((x) > 0 ? (x) : -1 * (x))

template<typename T>
void rand_vec(Matrix_2d<T> &matrix)
{
    static std::random_device rd;   // seed
    static std::mt19937 mt(rd());
    static std::uniform_real_distribution<T> dist(-10.0, 10.0);

    for (size_t i = 0; i < matrix.rows; ++i) 
        for (size_t j = 0; j < matrix.cols + matrix.padding; ++j) {
            if (j >= matrix.cols)
                continue;
            int offset = i * (matrix.cols + matrix.padding) + j;
            matrix.data[offset] = dist(mt);
        }
}

template <typename T>
static inline bool valid_vector(const Matrix_2d<T> &host_in, const Matrix_2d<T> &host_out, double nrms = 1e-6)
{
    double s0 = 0.0;
    double s1 = 0.0;
    int pp_err = 0;

    int rows = host_in.rows;
    int cols = host_in.cols;
    int padding = host_in.padding;

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols + padding; ++j) {
            if (j >= cols)
                continue;
            double ri = (double)host_in.data[i * (cols + padding) + j];
            double pi = (double)host_out.data[i * (cols + padding) + j];
            double d = ri - pi;
            double dd = d * d;
            double rr = 2.0 * ri * ri;
            s0 += dd;
            s1 += rr;
            double delta = ABS(ri - pi) / ri;

            if(delta > 3e-5) {
                if(pp_err < 100)
                    printf("diff at %4d, ref:%lf, pred:%lf(0x%08x), d:%lf\n", i, ri, pi,((uint32_t *)host_out.data)[i], delta);
                pp_err++;
            }            
        }
    }
    //printf("nrms:%lf, s0:%lf, s1:%lf\n",sqrt(s0/s1),s0,s1);
    return (sqrt (s0 / s1) < nrms) && (pp_err==0);
}

int main(int argc, char **argv)
{
    if (argc <= 1) {
        std::cout << "no input file! please enter params.config" << std::endl;
        return -1;
    }
    // parse params.config
    Config config;
    config.parseConfigFile(argv[1]);

    // add key-value to myStruct
    assert(config.m_contents.find("rows") != config.m_contents.end() && "[error!] failed to parse rows!");
    assert(config.m_contents.find("cols") != config.m_contents.end() && "[error!] failed to parse cols!");
    assert(config.m_contents.find("padding") != config.m_contents.end() && "[error!] failed to parse padding!");

    int rows = std::stoi(config.m_contents["rows"]);
    int cols = std::stoi(config.m_contents["cols"]);
    int padding = std::stoi(config.m_contents["padding"]);  

    // judge cols legality
    assert(cols % (256*8) == 0 && "[!]Only supports cols which is multiples of 2048(2K)");

    Matrix_2d<float> matrix_host_in(rows, cols, padding, 1);
    Matrix_2d<float> matrix_host_out(rows, cols, padding, 1);
    Matrix_2d<float> matrix_dev_in(rows, cols, padding, 0);
    Matrix_2d<float> matrix_dev_out(rows, cols, padding, 0);    

    // kernel preparation
    hipModule_t module;
    hipFunction_t kernel_func;
    hipEvent_t evt_00, evt_11;    
    HIP_CALL(hipSetDevice(0));

    HIP_CALL(hipModuleLoad(&module, HSACO));
    HIP_CALL(hipModuleGetFunction(&kernel_func, module, HSA_KERNEL));    
      
    int num_cu;
    int gcn_arch;
    {
        hipDeviceProp_t dev_prop;
        hipDevice_t dev;
        HIP_CALL(hipGetDevice(&dev));
        HIP_CALL(hipGetDeviceProperties(&dev_prop, dev));
        num_cu = dev_prop.multiProcessorCount;
        gcn_arch = dev_prop.gcnArch;
        if (gcn_arch >= 1000)
            num_cu *= 2;
    }

    int total_loop = 4;
    int warm_ups = 2;

    // initial blockDim, gridDim 
    int bdx = 256;
    int gdx = matrix_host_in.cols / (8 * bdx);
    HIP_CALL(hipMalloc(&matrix_dev_in.data,  sizeof(float) * matrix_dev_in.length));
    HIP_CALL(hipMalloc(&matrix_dev_out.data, sizeof(float) * matrix_dev_out.length));    

    // initial host in data
    rand_vec(matrix_host_in); 

    // memcpy data from host to device
    HIP_CALL(hipMemcpy(matrix_dev_in.data, matrix_host_in.data, sizeof(float) * matrix_dev_in.length, hipMemcpyHostToDevice)); 
    printf("memcpy, input: %p, output: %p, floats: %d\n",matrix_dev_in.data, matrix_dev_out.data, matrix_dev_in.length);

    struct __attribute__((packed)) 
    {
        float  *input;
        float  *output;
        int     rows;
        int     gdx;
        int     bdx;
        int     padding;
    } args;

    size_t arg_size = sizeof(args);
    args.input = matrix_dev_in.data;
    args.output = matrix_dev_out.data;
    args.rows = rows;
    args.gdx = gdx;
    args.bdx = bdx;
    args.padding = padding;    

    void* config_kernel[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args, HIP_LAUNCH_PARAM_BUFFER_SIZE, &arg_size, HIP_LAUNCH_PARAM_END}; 

    // warm up kernel 
    for (int i = 0; i < warm_ups; i++)
        HIP_CALL(hipModuleLaunchKernel(kernel_func, gdx, 1, 1, bdx, 1, 1, 0, 0, NULL, (void**)&config_kernel));    

    hipEventCreate(&evt_00);
    hipEventCreate(&evt_11);   
    hipDeviceSynchronize();
    hipEventRecord(evt_00, NULL);

    // launch kernel
    for(int i = 0; i < total_loop; i++)
        HIP_CALL(hipModuleLaunchKernel(kernel_func, gdx, 1, 1, bdx, 1, 1, 0, 0, NULL, (void**)&config_kernel));   

    float elapsed_ms;      
    hipEventRecord(evt_11, NULL);
    hipEventSynchronize(evt_11);
    hipDeviceSynchronize();
    hipEventElapsedTime(&elapsed_ms, evt_00, evt_11);
    hipEventDestroy(evt_00);
    hipEventDestroy(evt_11);      

    HIP_CALL(hipMemcpy(matrix_host_out.data, matrix_dev_out.data, sizeof(float) * matrix_host_out.length, hipMemcpyDeviceToHost));

    // verification
    bool is_valid = valid_vector(matrix_host_in, matrix_host_out);
    if(!is_valid) 
        printf("Data not valid, please check\n");
    else
        printf("Data is valid :)\n");

    // evaluation
    float time_per_loop_ms = elapsed_ms / total_loop; 
    float gbps = (matrix_host_out.rows * matrix_host_out.cols) * 2 * sizeof(float) / time_per_loop_ms / 1000 / 1000;
    
    std::cout << "---- MEMCPY 2D EXAMPLE EVALUATION ----" << std::endl;
    std::cout << "  rows: " << matrix_host_in.rows << '\t' << "cols: " << matrix_host_in.cols << '\t' << "padding: " << matrix_host_in.padding << std::endl;
    std::cout << "  gdx: " << gdx << '\t' << "bdx " << bdx << std::endl;
    std::cout << "  gbps: " << gbps << std::endl;
    std::cout << "----      FINISH EVALUATION       ----" << std::endl;

    hipFree(matrix_dev_in.data);
    hipFree(matrix_dev_out.data);

    return 0;
}