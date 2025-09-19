#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <random>
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <numeric>

#define RAND_INT 0

#define MAX(x, y) ((x) > (y) ? (x) : (y))
#define HIP_CALL(call) do{  \
    hipError_t err = call;  \
    if(err != hipSuccess){  \
        printf("[hiperror](%d) fail to call %s",(int)err,#call);    \
        exit(0);            \
    }                       \
} while(0)

#define ABS(x) ((x) > 0 ? (x) : -(x))

using uint32x4_t = uint32_t __attribute__((ext_vector_type(4)));
using uint32x8_t = uint32_t __attribute__((ext_vector_type(8)));

/*

// hip.amdgcn.bc - device routine
  HW_ID Register bit structure for RDNA2 & RDNA3
  WAVE_ID     4:0     Wave id within the SIMD.
  SIMD_ID     9:8     SIMD_ID within the WGP: [0] = row, [1] = column.
  WGP_ID      13:10   Physical WGP ID.
  SA_ID       16      Shader Array ID
  SE_ID       20:18   Shader Engine the wave is assigned to for gfx11
  SE_ID       19:18   Shader Engine the wave is assigned to for gfx10
  DP_RATE     31:29   Number of double-precision float units per SIMD

  HW_ID Register bit structure for GCN and CDNA
  WAVE_ID     3:0     Wave buffer slot number. 0-9.
  SIMD_ID     5:4     SIMD which the wave is assigned to within the CU.
  PIPE_ID     7:6     Pipeline from which the wave was dispatched.
  CU_ID       11:8    Compute Unit the wave is assigned to.
  SH_ID       12      Shader Array (within an SE) the wave is assigned to.
  SE_ID       15:13   Shader Engine the wave is assigned to for gfx908, gfx90a, gfx940-942
              14:13   Shader Engine the wave is assigned to for Vega.
  TG_ID       19:16   Thread-group ID
  VM_ID       23:20   Virtual Memory ID
  QUEUE_ID    26:24   Queue from which this wave was dispatched.
  STATE_ID    29:27   State ID (graphics only, not compute).
  ME_ID       31:30   Micro-engine ID.

  XCC_ID Register bit structure for gfx940/950
  XCC_ID      3:0     XCC the wave is assigned to.

#if (defined (__GFX10__) || defined (__GFX11__))
  #define HW_ID               23
#else
  #define HW_ID               4
#endif

hwreg_id = immed.u16[5 : 0];
offset   = immed.u16[10 : 6];
size     = immed.u16[15 : 11].u32 + 1U;

*/
#define _GETREG_IMMED(sz_,os_,reg_) (((sz_) << 11) | ((os_) << 6) | (reg_))

#if (defined (__GFX10__) || defined (__GFX11__))
  #define _HWREG_HW_ID    23
#else
  #define _HWREG_HW_ID    4
#endif

#if (defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__) || defined(__gfx950__))
  #define _HWREG_XCC_ID                   20
#endif

#if (defined(__gfx908__) || defined(__gfx90a__) || \
     defined(__GFX11__))
  #define _HW_ID_SE_ID_SIZE    3
#else //4 SEs/XCC for gfx940-942
  #define _HW_ID_SE_ID_SIZE    2
#endif


#define _BM(bits) ((1 << (bits)) - 1)

// TODO: only support gfx9
struct hwreg {
    __host__ hwreg() {}
    __device__ hwreg(){
        hw_id_ = __builtin_amdgcn_s_getreg(_GETREG_IMMED(31, 0, _HWREG_HW_ID));    // we cache all hw id reg
#if (defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__) || defined(__gfx950__))
        xcc_id_ = __builtin_amdgcn_s_getreg(_GETREG_IMMED(3, 0, _HWREG_XCC_ID));    // we cache all xcc reg
#endif
    }
    __device__ unsigned cu_id() { return (hw_id_ >> 8) & _BM(4); }
    __device__ unsigned sh_id() { return (hw_id_ >> 12) & _BM(1); }
    __device__ unsigned se_id() { return (hw_id_ >> 13) & _BM(_HW_ID_SE_ID_SIZE); }
    __device__ unsigned wave_id() { return (hw_id_ >> 0) & _BM(4); }
#if (defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__) || defined(__gfx950__))
    __device__ unsigned xcc_id() { return xcc_id_; }
#endif
    unsigned hw_id_;
#if (defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__) || defined(__gfx950__))
    unsigned xcc_id_;
#endif
};

#define BLOCK_SIZE 256
// assume this is 1d grid size
__global__ void
dump_hwreg(const void * input, void * output, int /**/)
{
    uint32_t offset = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    uint32_t input_data = *(reinterpret_cast<const uint32_t*>(input) + offset);

    auto r = hwreg{};

    uint32x8_t output_data;
    output_data[0] = blockIdx.x;
    output_data[1] = threadIdx.x;
    output_data[2] = input_data;
    output_data[3] = r.cu_id();
    output_data[4] = r.sh_id();
    output_data[5] = r.se_id();
    output_data[6] = r.wave_id();
#if (defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__) || defined(__gfx950__))
    output_data[7] = r.xcc_id();
#endif    
    *(reinterpret_cast<uint32x8_t*>(output) + offset) = output_data;
}

int main(int argc, char ** argv)
{
    int num_cu = [](){
        hipDeviceProp_t dev_prop;
        hipDevice_t dev;
        HIP_CALL(hipGetDevice( &dev ));
        HIP_CALL(hipGetDeviceProperties( &dev_prop, dev ));
        return dev_prop.multiProcessorCount;
    }();

    int grids = num_cu;
    if (argc >= 2) {
        grids = atoi(argv[1]);
    }


    //fp32 on host
    uint32_t * host_i = (uint32_t*)malloc(grids*BLOCK_SIZE*sizeof(uint32_t));
    uint32x8_t * host_o = (uint32x8_t*)malloc(grids*BLOCK_SIZE*sizeof(uint32x8_t));
    void * dev_i;
    void * dev_o;

    HIP_CALL(hipMalloc(&dev_i, grids*BLOCK_SIZE*sizeof(uint32_t)));
    HIP_CALL(hipMalloc(&dev_o, grids*BLOCK_SIZE*sizeof(uint32x8_t)));
    //fp16 cpy to device
    HIP_CALL(hipMemcpy(dev_i, host_i, grids*BLOCK_SIZE*sizeof(uint32_t), hipMemcpyHostToDevice));

    dump_hwreg<<<grids, BLOCK_SIZE>>>(dev_i, dev_o, grids);

    HIP_CALL(hipMemcpy(host_o, dev_o, grids*BLOCK_SIZE*sizeof(uint32x8_t), hipMemcpyDeviceToHost));

    for(auto i = 0; i < grids*BLOCK_SIZE; i++) {
        if (i % BLOCK_SIZE != 0) continue;
        auto o = host_o[i];

        uint32_t blk_id = o[0];
        uint32_t cu_id  =o[3];
        uint32_t sh_id = o[4];
        uint32_t se_id = o[5];
        uint32_t wave_id = o[6];
        uint32_t xcc_id = o[7]; // host not valid

        printf("block:%4d, cu_id:%2u, sh_id:%2u, se_id:%2u, wave_id:%2u", blk_id, cu_id, sh_id, se_id, wave_id);
        printf(", xcc_id:%2u", xcc_id);

        printf("\n");
    }
}
