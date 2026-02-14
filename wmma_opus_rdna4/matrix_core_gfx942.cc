#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <random>
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <numeric>
#define HALF
#ifdef HALF
#include "half.hpp"
#endif

#include "opus/opus.hpp"

#define LOCAL_SCRATCH 0
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

using fp32_t = float;
using fp16_t = _Float16;
using float16 = half_float::half; // cpu type

using fp16x2_t = fp16_t __attribute__((ext_vector_type(2)));
using fp16x4_t = fp16_t __attribute__((ext_vector_type(4)));
using fp16x8_t = fp16_t __attribute__((ext_vector_type(8)));
using fp16x16_t = fp16_t __attribute__((ext_vector_type(16)));
using fp32x2_t = fp32_t __attribute__((ext_vector_type(2)));
using fp32x4_t = fp32_t __attribute__((ext_vector_type(4)));
using fp32x16_t = fp32_t __attribute__((ext_vector_type(16)));

using int32x4_t = int32_t __attribute__((ext_vector_type(4)));
#define BUFFER_LOAD_DWORD3 0x00020000   // This is valid for 
struct buffer_resource {
    const void * ptr;
    uint32_t range;
    uint32_t config;
};
__device__ int32x4_t make_buffer_resource(const void * ptr, uint32_t size = 0xffffffff)
{
    buffer_resource res {ptr, size, BUFFER_LOAD_DWORD3};
    return __builtin_bit_cast(int32x4_t, res);
}

inline __host__ __device__ 
static constexpr int32_t GetSmemSize(){
    return sizeof(fp16_t) * (128 * 32) + sizeof(fp16_t) * (128 * 32);
}

inline __host__ __device__
static constexpr int32_t GetSmemSizeA(){
    return sizeof(fp16_t) * (128 * 32);
}

// split 4VGPR into 2VGPR x 2
// Note: To use same vgpr, we define a union for register repack.
union half8Packer{
    fp16x8_t vec8;
    fp16x4_t vec4[2];
    fp16x2_t vec2[4];
    fp16_t vec[8];
};

union half4Packer{
    fp16x4_t vec4;
    fp16x2_t vec2[2];
    fp16_t vec[4];
};

union Float4Packer{
    fp32x4_t vec4;
    fp32x2_t vec2[2];
    fp32_t vec[4];
};


// 128x128x32
template<int32_t Block_M, //128
         int32_t Block_N, //128
         int32_t Block_K, //32
         int32_t BLOCKSIZE
         >
__global__ void 
__launch_bounds__(256,1)
mfma_kernel_test2(const void* __restrict__ ptr_a,
                   const void* __restrict__ ptr_b,
                   void* __restrict__ ptr_c,
                   int m,
                   int n,
                   int k,
                   int stride_a, // stride in unit of pixel
                   int stride_b,
                   int stride_c)
{
    using opus::operator""_I;
    //buffer rsrc define
    auto res_a = make_buffer_resource(ptr_a);
    auto res_b = make_buffer_resource(ptr_b);
    auto res_c = make_buffer_resource(ptr_c);
    
    //grid layout
    auto shape_a = opus::make_tuple(m, k);
    auto stride_a_ = opus::make_tuple(stride_a, 1);
    auto shape_b = opus::make_tuple(n, k);
    auto stride_b_ = opus::make_tuple(stride_b, 1);
    auto shape_c = opus::make_tuple(m,n);
    auto stride_c_ = opus::make_tuple(stride_c, 1);

    auto win_a = opus::make_layout(shape_a, stride_a_);
    auto win_b = opus::make_layout(shape_b, stride_b_);
    auto win_c = opus::make_layout(shape_c, stride_c_);
    
    //workgroup coordinate
    auto wg_coord_a = opus::make_tuple( blockIdx.x * Block_M, 0_I);
    auto wg_coord_b = opus::make_tuple( blockIdx.y * Block_N, 0_I);
    auto wg_coord_c = opus::make_tuple( blockIdx.x * Block_M, blockIdx.y * Block_N);

    int32_t wg_offset_a = win_a(wg_coord_a);
    int32_t wg_offset_b = win_b(wg_coord_b);
    int32_t wg_offset_c = win_c(wg_coord_c);


    //blocklevel global read layout
    int lane_id = threadIdx.x % opus::get_warp_size();
    int wave_id = threadIdx.x / opus::get_warp_size();
    constexpr int32_t warpNum = BLOCKSIZE / opus::get_warp_size();
    
    constexpr int32_t AKGldPack = 16 / sizeof(fp16_t); // 8_I
    constexpr int32_t AKGldLane = Block_K / AKGldPack; // 4_I
    constexpr int32_t AMGldLane = opus::get_warp_size() / AKGldLane; //16_I
    constexpr int32_t AMRepeat  = Block_M / (AMGldLane * warpNum); // 128 / (16x4) = 2


    //(MRepeat, MLane, KLane, KPack) BlockM = (MRepeat) x MWaves x Mlane ; BlockK = KLane x KPack
    auto block_shape_a = opus::make_tuple(opus::number<AMRepeat>{}, opus::number<warpNum>{}, opus::number<AMGldLane>{}, opus::number<AKGldLane>{}, opus::number<AKGldPack>{});
    auto block_stride_a = opus::make_tuple(AMGldLane * warpNum * stride_a, AMGldLane * stride_a, stride_a, AKGldPack, 1_I);
    //Note: if you need multiple data access instrutions to be issued, set the corresponding coord dim to opus::_,
    //opus will calculate num_issues automatically.
    auto block_coord_a = opus::make_tuple(opus::_,
                                          wave_id,
                                          lane_id / AKGldLane,
                                          lane_id % AKGldLane,
                                          opus::_);
    //cached_vec means vector size of single issue
    auto block_gmem_a = opus::make_layout<AKGldPack>(block_shape_a, block_stride_a, block_coord_a); // 2 offsets
    
    //GMEM B
    constexpr int32_t BKGldPack = 16 / sizeof(fp16_t); //8_I
    constexpr int32_t BKGldLane = Block_K / BKGldPack; //4_I
    constexpr int32_t BNGldLane = opus::get_warp_size() / BKGldLane; //16_I
    constexpr int32_t BNRepeat  = Block_N / (BNGldLane * warpNum); // 128/(16x4)=2

    auto block_shape_b = opus::make_tuple(opus::number<BNRepeat>{}, opus::number<warpNum>{}, opus::number<BNGldLane>{}, opus::number<BKGldLane>{}, opus::number<BKGldPack>{});
    auto block_stride_b = opus::make_tuple(BNGldLane * warpNum * stride_b, BNGldLane * stride_b, stride_b, BKGldPack, 1_I);
    auto block_coord_b = opus::make_tuple(opus::_,
                                    wave_id,
                                    lane_id / BKGldLane,
                                    lane_id % BKGldLane,
                                    opus::_);
    auto block_gmem_b = opus::make_layout<BKGldPack>(block_shape_b, block_stride_b, block_coord_b); // 2 offsets
    
    //smem store A/B
    //base offset： A 0 B=smemA
    __shared__ char Smem[GetSmemSize()];
    uintptr_t smembase = reinterpret_cast<uintptr_t>(Smem);
    constexpr int32_t sld_b_base = GetSmemSizeA();
    
    //Note: 
    //Just provide every thread first access offset.Others can use immed offset.
    //If immed offset is over DS Inst bits limmited, use this to provide another vgpr offset
    //FIXME:rename smem to sst
    //A SST
    constexpr int32_t AKSstPack = 16 / sizeof(fp16_t); //8
    constexpr int32_t AKSstLane = Block_K / AKSstPack;  //4
    constexpr int32_t AMSstLane = opus::get_warp_size() / AKSstLane; //16
    constexpr int32_t AMSstRepeat  = Block_M / (AMSstLane * warpNum);//2

    auto block_sst_shape_a  = opus::make_tuple(opus::number<AMSstRepeat>{},
                                                opus::number<warpNum>{}, 
                                                opus::number<AMSstLane>{},
                                                opus::number<AKSstLane>{},
                                                opus::number<AKSstPack>{});
    auto block_sst_stride_a = opus::make_tuple(Block_K * warpNum, Block_K * AMSstLane, Block_K, AKSstPack, 1_I);
    auto block_sst_coord_a  = opus::make_tuple(0_I, wave_id, lane_id / AKSstLane, lane_id % AKSstLane, 0_I);
    //cached_vec = 0 usage
    auto block_sst_win_a     = opus::make_layout<0>(block_sst_shape_a, block_sst_stride_a);
    int32_t sst_a_os = block_sst_win_a(block_sst_coord_a) * sizeof(fp16_t);

    //B SST
    constexpr int32_t BKSstPack = 16 / sizeof(fp16_t); //8
    constexpr int32_t BKSstLane = Block_K / BKSstPack;  //4
    constexpr int32_t BNSstLane = opus::get_warp_size() / BKSstLane; //16
    constexpr int32_t BNSstRepeat  = Block_N / (BNSstLane * warpNum);//2

    auto block_sst_shape_b  = opus::make_tuple(opus::number<BNSstRepeat>{},
                                                opus::number<warpNum>{},
                                                opus::number<BNSstLane>{},
                                                opus::number<BKSstLane>{},
                                                opus::number<BKSstPack>{});
    auto block_sst_stride_b = opus::make_tuple(Block_K * warpNum, Block_K * BNSstLane, Block_K, BKSstPack, 1_I);
    auto block_sst_coord_b  = opus::make_tuple(0_I, wave_id, lane_id / BKSstLane, lane_id % BKSstLane, 0_I);
    //cached_vec = 0 usage
    auto block_sst_win_b     = opus::make_layout<0>(block_sst_shape_b, block_sst_stride_b);
    int32_t sst_b_os = block_sst_win_b(block_sst_coord_b)*sizeof(fp16_t);
    constexpr int32_t single_issue_offset = 16 * BLOCKSIZE;

    //smem load
    //A [MRepeat, KRepeat, MWaves, KThreads, MThreads, KPack]
    constexpr int32_t AKSldPack = 16 / sizeof(fp16_t);
    constexpr int32_t AKSldLane = 4_I; // refer to mfma layout
    constexpr int32_t AMSldLane = 16_I; // refer to mfma layout
    static_assert(AKSldLane * AMSldLane == opus::get_warp_size(), "Error warpsize.Please Check MKLanes!\n");
    constexpr int32_t AKSldRepeat  = Block_K / (AKSldPack * AKSldLane); //32/(8x4) = 1
    constexpr int32_t AMSldRepeat  = Block_M / (AMSldLane * warpNum); // 128/(16x4) = 2


    auto block_sld_shape_a  = opus::make_tuple(opus::number<AMSldRepeat>{}, //2
                                               opus::number<warpNum>{}, //4
                                               opus::number<AKSldRepeat>{}, //1
                                               opus::number<AKSldLane>{}, //4
                                               opus::number<AMSldLane>{}, //16
                                               opus::number<AKSldPack>{});//8
    auto block_sld_stride_a = opus::make_tuple(AMSldLane * Block_K * warpNum,
                                               AMSldLane * Block_K,
                                               AKSldPack * AKSldLane,
                                               AKSldPack,
                                               Block_K,
                                               1_I);
    auto block_sld_coord_a  = opus::make_tuple(0_I, wave_id, 0_I, lane_id / AMSldLane, lane_id % AMSldLane, 0_I);
    auto block_sld_win_a = opus::make_layout<0>(block_sld_shape_a, block_sld_stride_a);
    
    //B [NRepeat, KRepeat, NWaves, NThreads, NInterleave, KThreads , KPack]
    // [NRepeat, KRepeat, NWaves, NInterleave, KThreads, NThreads , KPack]
    constexpr int32_t BSldKPack = 16 / sizeof(fp16_t);
    constexpr int32_t BSldKLane = 4_I;
    constexpr int32_t BSldNLane = 16_I;
    constexpr int32_t BSldKRepeat     = Block_K / (BSldKPack * BSldKLane); // 1
    constexpr int32_t BSldNRepeat     = Block_N / BSldNLane; // 8

    auto block_sld_shape_b  = opus::make_tuple(opus::number<BSldNRepeat>{}, 
                                               opus::number<BSldKRepeat>{}, 
                                               opus::number<BSldKLane>{}, 
                                               opus::number<BSldNLane>{}, 
                                               opus::number<BSldKPack>{});
    auto block_sld_stride_b = opus::make_tuple(Block_K * BSldNLane,
                                               BSldKPack * BSldKLane,
                                               BSldKPack,
                                               Block_K,
                                               1_I);
    auto block_sld_coord_b = opus::make_tuple(0_I, 0_I, lane_id /16, lane_id % 16, 0_I);
    auto block_sld_win_b = opus::make_layout<0>(block_sld_shape_b, block_sld_stride_b);

    //layout_linear return linear_offset
    int32_t a_sld_os = block_sld_win_a(block_sld_coord_a) * sizeof(fp16_t);
    int32_t b_sld_os = block_sld_win_b(block_sld_coord_b) * sizeof(fp16_t);

    //C Store trans Layout
    constexpr int32_t CGstNPack = 4_I;
    constexpr int32_t CGstNLane = 4_I;
    constexpr int32_t CGstMLane = 16_I;
    constexpr int32_t CGstNRepeat = Block_N / (CGstNPack * CGstNLane);
    constexpr int32_t CGstMRepeat = Block_M / (CGstMLane * warpNum);

    auto block_gmem_gst_shape_c = opus::make_tuple(opus::number<CGstMRepeat>{},
                                                   opus::number<warpNum>{},
                                                   opus::number<CGstNRepeat>{},
                                                   opus::number<CGstNLane>{},
                                                   opus::number<CGstMLane>{},
                                                   opus::number<CGstNPack>{});
    auto block_gmem_gst_stride_c = opus::make_tuple(stride_c * CGstMLane * warpNum,
                                                    stride_c * CGstMLane,
                                                    CGstNLane * CGstNPack,
                                                    CGstNPack,
                                                    stride_c,
                                                    1_I);
    auto block_gmem_gst_coord_c = opus::make_tuple(opus::_, wave_id, 0_I,  lane_id / 16, lane_id % 16, opus::_);
    auto block_gmem_gst_layout_c = opus::make_layout<CGstNPack>(block_gmem_gst_shape_c, block_gmem_gst_stride_c, block_gmem_gst_coord_c);





    //Pipeline Part
    // gmem load a
    // Note: if different buffer insts have same base but different offsets and offset can be immed.
    // You don't need to make it in cached_layout. Cached_layout is used for variable offset, and immed offset
    // can be set mannually.
    //VGPR store A/B
    opus::array<fp16x8_t, 2> gmem_a_reg;
    opus::array<fp16x8_t, 2> gmem_b_reg;
    asm volatile(
        "buffer_load_dwordx4 %[v_gmem_a0], %[v_gmem_a_os0], %[res_a], 0 offen\n\t"
        "buffer_load_dwordx4 %[v_gmem_a1], %[v_gmem_a_os1], %[res_a], 0 offen\n\t"
        :[v_gmem_a0]"+v"(gmem_a_reg[0]),
         [v_gmem_a1]"+v"(gmem_a_reg[1])
        :[v_gmem_a_os0]"v"(static_cast<int>((block_gmem_a.offsets[0] + wg_offset_a) * sizeof(fp16_t))),
         [v_gmem_a_os1]"v"(static_cast<int>((block_gmem_a.offsets[1] + wg_offset_a) * sizeof(fp16_t))),
         [res_a]"s"(res_a)
        :"memory"
    );
    // __builtin_amdgcn_sched_barrier(0);
    // asm volatile("" 
    //             : 
    //             :"v"(static_cast<int>((block_gmem_a.offsets[0] + wg_offset_a) * sizeof(fp16_t))),
    //              "v"(static_cast<int>((block_gmem_a.offsets[1] + wg_offset_a) * sizeof(fp16_t))),
    //              "v"(static_cast<int>((block_gmem_a.offsets[2] + wg_offset_a) * sizeof(fp16_t))),
    //              "v"(static_cast<int>((block_gmem_a.offsets[3] + wg_offset_a) * sizeof(fp16_t)))
    //             : "memory");
    asm volatile("s_waitcnt vmcnt(0)");

    // gmem load b
    //use + to avoid reuse dst reg
    asm volatile(
        "buffer_load_dwordx4 %[v_gmem_b0], %[v_gmem_b_os0], %[res_b], 0 offen\n\t"
        "buffer_load_dwordx4 %[v_gmem_b1], %[v_gmem_b_os1], %[res_b], 0 offen\n\t"
        :[v_gmem_b0]"+v"(gmem_b_reg[0]),
         [v_gmem_b1]"+v"(gmem_b_reg[1])
        :[v_gmem_b_os0]"v"(static_cast<int>((block_gmem_b.offsets[0] + wg_offset_b) * sizeof(fp16_t))),
         [res_b]"s"(res_b),
         [v_gmem_b_os1]"v"(static_cast<int>((block_gmem_b.offsets[1] + wg_offset_b) * sizeof(fp16_t)))
        :"memory"
    );
    asm volatile("s_waitcnt vmcnt(0)");

    //smem store
    //store a
    asm volatile(
        "ds_write_b128 %[v_sst_a], %[v_gmem_a0] offset:0 \n\t"
        "ds_write_b128 %[v_sst_a], %[v_gmem_a1] offset:%[single_issue_offset0]\n\t"
        "s_waitcnt lgkmcnt(0)\n\t"
        "s_barrier\n\t"
        : 
        : [v_sst_a]"v"(static_cast<int>(sst_a_os + smembase)),
          [v_gmem_a0]"v"(gmem_a_reg[0]),
          [v_gmem_a1]"v"(gmem_a_reg[1]),
          [single_issue_offset0]"n"(single_issue_offset)
        : "memory"
    );
        // __builtin_amdgcn_sched_barrier(0);
    // asm volatile("" 
    //             : 
    //             :"v"(static_cast<int>(sst_a_os + smembase))
    //             : "memory");

    //store b
    asm volatile(
        "ds_write_b128 %[v_sst_b], %[v_gmem_b0] offset:%[single_issue_offset0]\n\t"
        "ds_write_b128 %[v_sst_b], %[v_gmem_b1] offset:%[single_issue_offset1]\n\t"
        :
        :[v_sst_b]"v"(static_cast<int>(sst_b_os + GetSmemSizeA())),
         [v_gmem_b0]"v"(gmem_b_reg[0]),
         [sst_b_base]"n"(sld_b_base),
         [single_issue_offset0]"n"(0 * single_issue_offset),
         [v_gmem_b1]"v"(gmem_b_reg[1]),
         [single_issue_offset1]"n"(1 * single_issue_offset)
        :"memory"
    );

    asm volatile(
        "s_waitcnt lgkmcnt(0)\n\t"
        "s_barrier\n\t"
    );
    // __builtin_amdgcn_sched_barrier(0);

    //smem load A/B 
    //Note use fp16x8_t to load (with ds_read_B128) but bit_cast to fp16x4_t for mfma usage
    //First, reuse the gmem vgpr for sld. Just for register allocation test. You can allocate another part registers for sld.
    opus::array<fp16x8_t, 2> sld_a_reg;
    asm volatile(
        "ds_read_b128 %[v_sld_a0], %[v_sld_a_os] offset:0   \n\t"
        "ds_read_b128 %[v_sld_a1], %[v_sld_a_os] offset:4096 \n\t"
        :[v_sld_a0]"=v"(sld_a_reg[0]),
         [v_sld_a1]"=v"(sld_a_reg[1])
        :[v_sld_a_os]"v"(a_sld_os)
        :"memory"
    );
    asm volatile(
        "s_waitcnt lgkmcnt(0)\n"
        "s_barrier"
    );
    // __builtin_amdgcn_sched_barrier(0);
    opus::array<fp16x8_t, 8> sld_b_reg;
    asm volatile(
        "ds_read_b128 %[v_sld_b0], %[v_sld_b_os] offset:%[sld_b_base] + 0\n\t"
        "ds_read_b128 %[v_sld_b1], %[v_sld_b_os] offset:%[sld_b_base] + 1024\n\t"
        "ds_read_b128 %[v_sld_b2], %[v_sld_b_os] offset:%[sld_b_base] + 2048\n\t"
        "ds_read_b128 %[v_sld_b3], %[v_sld_b_os] offset:%[sld_b_base] + 3072\n\t"
        "ds_read_b128 %[v_sld_b4], %[v_sld_b_os] offset:%[sld_b_base] + 4096\n\t"
        "ds_read_b128 %[v_sld_b5], %[v_sld_b_os] offset:%[sld_b_base] + 5120\n\t"
        "ds_read_b128 %[v_sld_b6], %[v_sld_b_os] offset:%[sld_b_base] + 6144\n\t"
        "ds_read_b128 %[v_sld_b7], %[v_sld_b_os] offset:%[sld_b_base] + 7168\n\t"
        :[v_sld_b0]"=v"(sld_b_reg[0]),
         [v_sld_b1]"=v"(sld_b_reg[1]),
         [v_sld_b2]"=v"(sld_b_reg[2]),
         [v_sld_b3]"=v"(sld_b_reg[3]),
         [v_sld_b4]"=v"(sld_b_reg[4]),
         [v_sld_b5]"=v"(sld_b_reg[5]),
         [v_sld_b6]"=v"(sld_b_reg[6]),
         [v_sld_b7]"=v"(sld_b_reg[7])
        :[v_sld_b_os]"v"(static_cast<int>(b_sld_os)),
         [sld_b_base]"n"(sld_b_base)
        :"memory"
    );
    // 虚拟使用，告诉编译器 b_sld_os 在 asm 后还会被使用 否则b_sld_os 会在上面的内嵌汇编中被覆盖
    asm volatile("" : : "v"(b_sld_os) : "memory");
        // __builtin_amdgcn_sched_barrier(0);

    asm volatile(
        "s_waitcnt lgkmcnt(0)\n\t"
        "s_barrier\n\t"
    );
    // __builtin_amdgcn_sched_barrier(0);

    //repack register
    opus::array<fp16x4_t, 4> sld_a;
    for (int i = 0; i < 2; i++) {
        half8Packer converter;
        converter.vec8 = sld_a_reg[i];
        sld_a[i * 2] = converter.vec4[0];
        sld_a[i * 2 + 1] = converter.vec4[1];
    }

    opus::array<fp16x4_t, 16> sld_b;
    for (int i = 0; i < 8; i++) {
        half8Packer converter;
        converter.vec8 = sld_b_reg[i];
        sld_b[i * 2] = converter.vec4[0];
        sld_b[i * 2 + 1] = converter.vec4[1];
    }

    
    //MMA Matrix C reg
    opus::array<fp32x4_t, 16> v_c;
    v_c.fill({0.0, 0.0, 0.0, 0.0});
    // __builtin_amdgcn_sched_barrier(0);
    //MMA Part
    #pragma unroll
    for(int m=0; m < AMSldRepeat; m++){
        #pragma unroll
        for(int n=0; n < BSldNRepeat; n++){
            asm volatile(
                "v_mfma_f32_16x16x16_f16 %[v_c], %[v_sld_b0], %[v_sld_a0], %[v_c]\n\t"
                "v_mfma_f32_16x16x16_f16 %[v_c], %[v_sld_b1], %[v_sld_a1], %[v_c]\n\t"
                :[v_c]"+v"(v_c[m * 8 + n])
                :[v_sld_a0]"v"(sld_a[m * 2]),
                [v_sld_a1]"v"(sld_a[m * 2 + 1]),
                [v_sld_b0]"v"(sld_b[n * 2]),
                [v_sld_b1]"v"(sld_b[n * 2 + 1])
            );
        }
    }


    //cvt to fp16
    
    opus::array<fp16x2_t, 32> v_c_out;
    opus::array<fp32_t, 64> v_c_tmp;
    for (int i = 0; i < 32; i++) {
        Float4Packer converter;
        converter.vec4 = v_c[i];
        v_c_tmp[i * 4] = converter.vec[0];
        v_c_tmp[i * 4 + 1] = converter.vec[1];
        v_c_tmp[i * 4 + 2] = converter.vec[2];
        v_c_tmp[i * 4 + 3] = converter.vec[3];
    }
    // __builtin_amdgcn_sched_barrier(0);

    #pragma unroll
    for (int i=0; i < AMSldRepeat * BSldNRepeat ;i++){
        asm volatile(
            "v_cvt_f16_f32_sdwa %[v_c_out0], %[v_c0] dst_sel:WORD_0\n\t"
            "v_cvt_f16_f32_sdwa %[v_c_out0], %[v_c1] dst_sel:WORD_1\n\t"
            "v_cvt_f16_f32_sdwa %[v_c_out1], %[v_c2] dst_sel:WORD_0\n\t"
            "v_cvt_f16_f32_sdwa %[v_c_out1], %[v_c3] dst_sel:WORD_1\n\t"
        :[v_c_out0]"+v"(v_c_out[i * 2]),
        [v_c_out1]"+v"(v_c_out[i * 2 + 1])
        :[v_c0]"v"(v_c_tmp[i*4]),
        [v_c1]"v"(v_c_tmp[i*4+1]),
        [v_c2]"v"(v_c_tmp[i*4+2]),
        [v_c3]"v"(v_c_tmp[i*4+3])
        );
    }

    opus::array<fp16x4_t, 16> v_c_store;
    for (int i = 0; i < 16; i++) {
        half4Packer converter;
        converter.vec2[0] = v_c_out[i*2];
        converter.vec2[1] = v_c_out[i*2+1];
        v_c_store[i] = converter.vec4;
    }
    
    //store C to global
    #pragma unroll
    for(int m=0; m < AMSldRepeat; m++){
        for(int n=0; n < BSldNRepeat; n++){
            asm volatile(
                "buffer_store_dwordx2 %[v_c_out], %[v_gmem_c_os0], %[s_res_c], 0 offen offset:%[immed_os]\n\t"
                :
                :[v_c_out]"v"(v_c_store[m * 8 + n]),
                 [v_gmem_c_os0]"v"(static_cast<int>((block_gmem_gst_layout_c.offsets[m] + wg_offset_c) * sizeof(fp16_t))),
                 [s_res_c]"s"(res_c),
                 [immed_os]"n"(n * 32)
            );
        }
    }
    // __builtin_amdgcn_sched_barrier(0);
}




#ifdef RAND_INT
#define PER_PIXEL_CHECK
#endif

static inline bool valid_vector( const float* ref, const float16* pred, int n, double nrms = 1e-3 )
{    
    double s0=0.0;
    double s1=0.0;
#ifdef PER_PIXEL_CHECK
    int pp_err = 0;
#endif
    int i_start = 0, i_end=n;
    
    for( int i=i_start; i<i_end; ++i ){
        double ri=(double)ref[i];
        double pi=(double)pred[i];
        double d=ri-pi;
        double dd=d*d;
        double rr=2.0*ri*ri;
        s0+=dd;
        s1+=rr;
        
#ifdef PER_PIXEL_CHECK
        double delta = ABS(ri-pi)/ri;
        if(delta>1e-3){
            if(pp_err<1024)
                printf("diff at %4d, ref:%lf, pred:%lf(0x%04x), d:%lf\n",i,ri,pi,((uint16_t*)pred)[i],delta);
            pp_err++;
        }
#endif
    }
    // int i_num = i_end - i_start;
    // printf("pp_crr:%d, pp_err:%d, crr_ratio:%.3f, nrms:%lf, s0:%lf, s1:%lf\n",i_num-pp_err, pp_err, (float)(i_num-pp_err)/(float)i_num, sqrt(s0/s1),s0,s1);

    return (sqrt(s0/s1)<nrms)
#ifdef PER_PIXEL_CHECK
        && (pp_err==0)
#endif
    ;
}

void rand_vector_2d(float* v, int row, int col, int ld, float min_v = 0, float max_v = 1){
    int r,c;
    static int flag = 0;
    if(!flag){ srand(time(NULL)); flag = 1; }
    for(r=0;r<row;r++){
        for(c=0;c<col;c++){
            float tmp = float(std::rand()) / float(RAND_MAX);
            v[r*ld+c] = static_cast<float>(min_v + tmp * (max_v - min_v));
            // v[r*ld+c] =   ((float)(r*ld+c)) / (row/2 * col/2) - 5;
        }
    }
}

void rand_vector_2d_int(float* v, int row, int col, int ld){
    int r,c;
    static int flag = 0;
    if(!flag){ srand(time(NULL)); flag = 1; }
    for(r=0;r<row;r++){
        for(c=0;c<col;c++){
            v[r*ld+c] = ((float)(rand() % 10)) - 5;
        }
    }
}

void gemm_rcr(
    const float*  __restrict__ ptr_a,
    const float*  __restrict__ ptr_b,
    float*  ptr_c,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc)
{
    for(auto i_m = 0 ; i_m < m; i_m++) {
        for(auto i_n = 0; i_n < n; i_n++) {
            float acc = 0;
            for(auto i_k = 0; i_k < k; i_k++) {
                acc += ptr_a[i_m * lda + i_k] * ptr_b[i_n * ldb + i_k];
            }
            ptr_c[i_m * ldc + i_n] = acc;
        }
    }
}


int main(int argc, char ** argv)
{
    int m = 128;
    int n = 128;
    int k = 32;

    int lda = k;
    int ldb = k;
    int ldc = n;

    float *host_a, *host_b, *host_c;
    float16 *fp16_a, *fp16_b, *fp16_c, *dev_a, *dev_b, *dev_c;

    //fp32 on host
    host_a = (float*)malloc(lda*m*sizeof(float));
    host_b = (float*)malloc(ldb*n*sizeof(float));
    host_c = (float*)malloc(ldc*m*sizeof(float));

#ifdef RAND_INT
    rand_vector_2d_int(host_a, m, k, lda);
    rand_vector_2d_int(host_b, n, k, ldb);
#else
    rand_vector_2d(host_a, m, k, lda, 0.0, 1.0);
    rand_vector_2d(host_b, n, k, ldb, -0.5, 0.5);
#endif

    //fp16 on host
    fp16_a = (float16*)malloc(lda*m*sizeof(float16));
    fp16_b = (float16*)malloc(ldb*n*sizeof(float16));
    fp16_c = (float16*)malloc(ldc*m*sizeof(float16));
    //convert fp32 a and b into fp16 on host
    for(int i=0; i<lda*m; i++)fp16_a[i]=__float2half_rn(host_a[i]);
    for(int i=0; i<ldb*n; i++)fp16_b[i]=__float2half_rn(host_b[i]);

    HIP_CALL(hipMalloc(&dev_a, lda*m*sizeof(float16)));
    HIP_CALL(hipMalloc(&dev_b, ldb*n*sizeof(float16)));
    HIP_CALL(hipMalloc(&dev_c, ldc*m*sizeof(float16)));
    //fp16 cpy to device
    HIP_CALL(hipMemcpy(dev_a, fp16_a, lda*m*sizeof(float16), hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(dev_b, fp16_b, ldb*n*sizeof(float16), hipMemcpyHostToDevice));

    printf("m:%d,n:%d,k:%d,lda:%d,ldb:%d,ldc:%d\n",  m, n, k, lda, ldb, ldc); fflush(stdout);
    gemm_rcr(host_a, host_b, host_c, m,n,k,lda,ldb,ldc);
    {
    int32_t wg_num_m = m / 128; //gridx
    int32_t wg_num_n = n / 128; //gridy
        mfma_kernel_test2<128, 128, 32, 256>
            <<<dim3(wg_num_m,wg_num_n,1),dim3(256,1,1),0,0>>>(dev_a, dev_b, dev_c, m, n, k, lda, ldb, ldc);
            HIP_CALL(hipMemcpy(fp16_c, dev_c, ldc*m*sizeof(float16), hipMemcpyDeviceToHost));
        bool res = valid_vector( host_c, fp16_c, m*n, 1e-3);
        printf("[128x128x32, standard], %s",res?"valid":"fail");fflush(stdout);
        printf("\n"); fflush(stdout);
    }


    free(host_a);
    free(host_b);
    free(host_c);
    free(fp16_a);
    free(fp16_b);
    free(fp16_c);
    
    HIP_CALL(hipFree(dev_a));
    HIP_CALL(hipFree(dev_b));
    HIP_CALL(hipFree(dev_c));

    // block_run();

    // int cc[5] = {1,2,3,4,5};
    // auto xx = cc;
    // printf("%d, %d %d\n", xx[2], xx[4], std::is_trivially_copyable_v<decltype(cc)> ? 1 : 0);
}