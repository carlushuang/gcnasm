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
#include "../half.hpp"
#endif

#include <opus/opus.hpp>

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
// A: M*K, B: N*K, C:M*N, use 32x32x8 fp16
/*
* V0/V1/   is 32bit register holding A/B matrix data, each register contains 2 fp16 pixel along gemm-k
* a0/a1... is 32bit register holding C matrix data in fp32 (this instruction use fp32 as acc)
* L0, L1.. is lane id with in a single wave, here we only have lane 0~63 (wave64)
* each thread need 2 registers for A, 2 regs for B, 16 regs for C

                                 L0 L1 L2 L3 L4 L5 L6 L7 L8 L9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
                       Matrix B   __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __
                                 |v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0| k0  L0~31
                                 |__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__| k1
                                 |v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1| k2
                                _|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|_k3
                                 |v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0| k4  L32~63
                                 |__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__| k5
                                 |v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1| k6
                                _|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|_k7
     Matrix A
     L0~31       L32~63           Matrix C
     k0 k1 k2 k3 k4 k5 k6 k7      L0 L1 L2 L3 L4 L5 L6 L7 L8 L9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
     _____ _____|_____ _____      __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __
L0  |v0   |v1   |v0   |v1   |    |a0|a0|a0|a0|a0|a0|a0|a0|a0|a0|a0|a0|a0|a0|a0|a0|a0|a0|a0|a0|a0|a0|a0|a0|a0|a0|a0|a0|a0|a0|a0|a0| L0~31
L1  |v0   |v1   |v0   |v1   |    |a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|
L2  |v0   |v1   |v0   |v1   |    |a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|
L3  |v0   |v1   |v0   |v1   |   _|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|_
L4  |v0   |v1   |v0   |v1   |    |a0|a0|a0|a0|a0|a0|a0|a0|a0|a0|a0|a0|a0|a0|a0|a0|a0|a0|a0|a0|a0|a0|a0|a0|a0|a0|a0|a0|a0|a0|a0|a0| L32~63
L5  |v0   |v1   |v0   |v1   |    |a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|
L6  |v0   |v1   |v0   |v1   |    |a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|
L7  |v0   |v1   |v0   |v1   |   _|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|_
L8  |v0   |v1   |v0   |v1   |    |a4|a4|a4|a4|a4|a4|a4|a4|a4|a4|a4|a4|a4|a4|a4|a4|a4|a4|a4|a4|a4|a4|a4|a4|a4|a4|a4|a4|a4|a4|a4|a4| L0~31
L9  |v0   |v1   |v0   |v1   |    |a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|
L10 |v0   |v1   |v0   |v1   |    |a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|
L11 |v0   |v1   |v0   |v1   |   _|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|_
L12 |v0   |v1   |v0   |v1   |    |a4|a4|a4|a4|a4|a4|a4|a4|a4|a4|a4|a4|a4|a4|a4|a4|a4|a4|a4|a4|a4|a4|a4|a4|a4|a4|a4|a4|a4|a4|a4|a4| L32~63
L13 |v0   |v1   |v0   |v1   |    |a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|
L14 |v0   |v1   |v0   |v1   |    |a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|
L15 |v0   |v1   |v0   |v1   |   _|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|_
L16 |v0   |v1   |v0   |v1   |    |a8|a8|a8|a8|a8|a8|a8|a8|a8|a8|a8|a8|a8|a8|a8|a8|a8|a8|a8|a8|a8|a8|a8|a8|a8|a8|a8|a8|a8|a8|a8|a8| L0~31
L17 |v0   |v1   |v0   |v1   |    |a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|
L18 |v0   |v1   |v0   |v1   |    |10|10|10|10|10|10|10|10|10|10|10|10|10|10|10|10|10|10|10|10|10|10|10|10|10|10|10|10|10|10|10|10|
L19 |v0   |v1   |v0   |v1   |   _|11|11|11|11|11|11|11|11|11|11|11|11|11|11|11|11|11|11|11|11|11|11|11|11|11|11|11|11|11|11|11|11|_
L20 |v0   |v1   |v0   |v1   |    |a8|a8|a8|a8|a8|a8|a8|a8|a8|a8|a8|a8|a8|a8|a8|a8|a8|a8|a8|a8|a8|a8|a8|a8|a8|a8|a8|a8|a8|a8|a8|a8| L32~63
L21 |v0   |v1   |v0   |v1   |    |a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|
L22 |v0   |v1   |v0   |v1   |    |10|10|10|10|10|10|10|10|10|10|10|10|10|10|10|10|10|10|10|10|10|10|10|10|10|10|10|10|10|10|10|10|
L23 |v0   |v1   |v0   |v1   |   _|11|11|11|11|11|11|11|11|11|11|11|11|11|11|11|11|11|11|11|11|11|11|11|11|11|11|11|11|11|11|11|11|_
L24 |v0   |v1   |v0   |v1   |    |12|12|12|12|12|12|12|12|12|12|12|12|12|12|12|12|12|12|12|12|12|12|12|12|12|12|12|12|12|12|12|12| L0~31
L25 |v0   |v1   |v0   |v1   |    |13|13|13|13|13|13|13|13|13|13|13|13|13|13|13|13|13|13|13|13|13|13|13|13|13|13|13|13|13|13|13|13|
L26 |v0   |v1   |v0   |v1   |    |14|14|14|14|14|14|14|14|14|14|14|14|14|14|14|14|14|14|14|14|14|14|14|14|14|14|14|14|14|14|14|14|
L27 |v0   |v1   |v0   |v1   |   _|15|15|15|15|15|15|15|15|15|15|15|15|15|15|15|15|15|15|15|15|15|15|15|15|15|15|15|15|15|15|15|15|_
L28 |v0   |v1   |v0   |v1   |    |12|12|12|12|12|12|12|12|12|12|12|12|12|12|12|12|12|12|12|12|12|12|12|12|12|12|12|12|12|12|12|12| L32~63
L29 |v0   |v1   |v0   |v1   |    |13|13|13|13|13|13|13|13|13|13|13|13|13|13|13|13|13|13|13|13|13|13|13|13|13|13|13|13|13|13|13|13|
L30 |v0   |v1   |v0   |v1   |    |14|14|14|14|14|14|14|14|14|14|14|14|14|14|14|14|14|14|14|14|14|14|14|14|14|14|14|14|14|14|14|14|
L31 |v0___|v1___|v0___|v1___|   _|15|15|15|15|15|15|15|15|15|15|15|15|15|15|15|15|15|15|15|15|15|15|15|15|15|15|15|15|15|15|15|15|_
                |
*/
using pin_v192 = float __attribute__((ext_vector_type(192)));
using pin_v16  = float __attribute__((ext_vector_type(16)));
template<class V> __device__ __forceinline__ void pin_vc(V& vv){
  pin_v192 v = __builtin_bit_cast(pin_v192, vv);
  { pin_v16 c;
    c[0]=v[0];
    c[1]=v[1];
    c[2]=v[2];
    c[3]=v[3];
    c[4]=v[4];
    c[5]=v[5];
    c[6]=v[6];
    c[7]=v[7];
    c[8]=v[8];
    c[9]=v[9];
    c[10]=v[10];
    c[11]=v[11];
    c[12]=v[12];
    c[13]=v[13];
    c[14]=v[14];
    c[15]=v[15];
    c=__builtin_amdgcn_pin_vgpr_v16f32(c,0);
    v[0]=c[0];
    v[1]=c[1];
    v[2]=c[2];
    v[3]=c[3];
    v[4]=c[4];
    v[5]=c[5];
    v[6]=c[6];
    v[7]=c[7];
    v[8]=c[8];
    v[9]=c[9];
    v[10]=c[10];
    v[11]=c[11];
    v[12]=c[12];
    v[13]=c[13];
    v[14]=c[14];
    v[15]=c[15];
  }
  { pin_v16 c;
    c[0]=v[16];
    c[1]=v[17];
    c[2]=v[18];
    c[3]=v[19];
    c[4]=v[20];
    c[5]=v[21];
    c[6]=v[22];
    c[7]=v[23];
    c[8]=v[24];
    c[9]=v[25];
    c[10]=v[26];
    c[11]=v[27];
    c[12]=v[28];
    c[13]=v[29];
    c[14]=v[30];
    c[15]=v[31];
    c=__builtin_amdgcn_pin_vgpr_v16f32(c,16);
    v[16]=c[0];
    v[17]=c[1];
    v[18]=c[2];
    v[19]=c[3];
    v[20]=c[4];
    v[21]=c[5];
    v[22]=c[6];
    v[23]=c[7];
    v[24]=c[8];
    v[25]=c[9];
    v[26]=c[10];
    v[27]=c[11];
    v[28]=c[12];
    v[29]=c[13];
    v[30]=c[14];
    v[31]=c[15];
  }
  { pin_v16 c;
    c[0]=v[32];
    c[1]=v[33];
    c[2]=v[34];
    c[3]=v[35];
    c[4]=v[36];
    c[5]=v[37];
    c[6]=v[38];
    c[7]=v[39];
    c[8]=v[40];
    c[9]=v[41];
    c[10]=v[42];
    c[11]=v[43];
    c[12]=v[44];
    c[13]=v[45];
    c[14]=v[46];
    c[15]=v[47];
    c=__builtin_amdgcn_pin_vgpr_v16f32(c,32);
    v[32]=c[0];
    v[33]=c[1];
    v[34]=c[2];
    v[35]=c[3];
    v[36]=c[4];
    v[37]=c[5];
    v[38]=c[6];
    v[39]=c[7];
    v[40]=c[8];
    v[41]=c[9];
    v[42]=c[10];
    v[43]=c[11];
    v[44]=c[12];
    v[45]=c[13];
    v[46]=c[14];
    v[47]=c[15];
  }
  { pin_v16 c;
    c[0]=v[48];
    c[1]=v[49];
    c[2]=v[50];
    c[3]=v[51];
    c[4]=v[52];
    c[5]=v[53];
    c[6]=v[54];
    c[7]=v[55];
    c[8]=v[56];
    c[9]=v[57];
    c[10]=v[58];
    c[11]=v[59];
    c[12]=v[60];
    c[13]=v[61];
    c[14]=v[62];
    c[15]=v[63];
    c=__builtin_amdgcn_pin_vgpr_v16f32(c,48);
    v[48]=c[0];
    v[49]=c[1];
    v[50]=c[2];
    v[51]=c[3];
    v[52]=c[4];
    v[53]=c[5];
    v[54]=c[6];
    v[55]=c[7];
    v[56]=c[8];
    v[57]=c[9];
    v[58]=c[10];
    v[59]=c[11];
    v[60]=c[12];
    v[61]=c[13];
    v[62]=c[14];
    v[63]=c[15];
  }
  { pin_v16 c;
    c[0]=v[64];
    c[1]=v[65];
    c[2]=v[66];
    c[3]=v[67];
    c[4]=v[68];
    c[5]=v[69];
    c[6]=v[70];
    c[7]=v[71];
    c[8]=v[72];
    c[9]=v[73];
    c[10]=v[74];
    c[11]=v[75];
    c[12]=v[76];
    c[13]=v[77];
    c[14]=v[78];
    c[15]=v[79];
    c=__builtin_amdgcn_pin_vgpr_v16f32(c,64);
    v[64]=c[0];
    v[65]=c[1];
    v[66]=c[2];
    v[67]=c[3];
    v[68]=c[4];
    v[69]=c[5];
    v[70]=c[6];
    v[71]=c[7];
    v[72]=c[8];
    v[73]=c[9];
    v[74]=c[10];
    v[75]=c[11];
    v[76]=c[12];
    v[77]=c[13];
    v[78]=c[14];
    v[79]=c[15];
  }
  { pin_v16 c;
    c[0]=v[80];
    c[1]=v[81];
    c[2]=v[82];
    c[3]=v[83];
    c[4]=v[84];
    c[5]=v[85];
    c[6]=v[86];
    c[7]=v[87];
    c[8]=v[88];
    c[9]=v[89];
    c[10]=v[90];
    c[11]=v[91];
    c[12]=v[92];
    c[13]=v[93];
    c[14]=v[94];
    c[15]=v[95];
    c=__builtin_amdgcn_pin_vgpr_v16f32(c,80);
    v[80]=c[0];
    v[81]=c[1];
    v[82]=c[2];
    v[83]=c[3];
    v[84]=c[4];
    v[85]=c[5];
    v[86]=c[6];
    v[87]=c[7];
    v[88]=c[8];
    v[89]=c[9];
    v[90]=c[10];
    v[91]=c[11];
    v[92]=c[12];
    v[93]=c[13];
    v[94]=c[14];
    v[95]=c[15];
  }
  { pin_v16 c;
    c[0]=v[96];
    c[1]=v[97];
    c[2]=v[98];
    c[3]=v[99];
    c[4]=v[100];
    c[5]=v[101];
    c[6]=v[102];
    c[7]=v[103];
    c[8]=v[104];
    c[9]=v[105];
    c[10]=v[106];
    c[11]=v[107];
    c[12]=v[108];
    c[13]=v[109];
    c[14]=v[110];
    c[15]=v[111];
    c=__builtin_amdgcn_pin_vgpr_v16f32(c,96);
    v[96]=c[0];
    v[97]=c[1];
    v[98]=c[2];
    v[99]=c[3];
    v[100]=c[4];
    v[101]=c[5];
    v[102]=c[6];
    v[103]=c[7];
    v[104]=c[8];
    v[105]=c[9];
    v[106]=c[10];
    v[107]=c[11];
    v[108]=c[12];
    v[109]=c[13];
    v[110]=c[14];
    v[111]=c[15];
  }
  { pin_v16 c;
    c[0]=v[112];
    c[1]=v[113];
    c[2]=v[114];
    c[3]=v[115];
    c[4]=v[116];
    c[5]=v[117];
    c[6]=v[118];
    c[7]=v[119];
    c[8]=v[120];
    c[9]=v[121];
    c[10]=v[122];
    c[11]=v[123];
    c[12]=v[124];
    c[13]=v[125];
    c[14]=v[126];
    c[15]=v[127];
    c=__builtin_amdgcn_pin_vgpr_v16f32(c,112);
    v[112]=c[0];
    v[113]=c[1];
    v[114]=c[2];
    v[115]=c[3];
    v[116]=c[4];
    v[117]=c[5];
    v[118]=c[6];
    v[119]=c[7];
    v[120]=c[8];
    v[121]=c[9];
    v[122]=c[10];
    v[123]=c[11];
    v[124]=c[12];
    v[125]=c[13];
    v[126]=c[14];
    v[127]=c[15];
  }
  { pin_v16 c;
    c[0]=v[128];
    c[1]=v[129];
    c[2]=v[130];
    c[3]=v[131];
    c[4]=v[132];
    c[5]=v[133];
    c[6]=v[134];
    c[7]=v[135];
    c[8]=v[136];
    c[9]=v[137];
    c[10]=v[138];
    c[11]=v[139];
    c[12]=v[140];
    c[13]=v[141];
    c[14]=v[142];
    c[15]=v[143];
    c=__builtin_amdgcn_pin_vgpr_v16f32(c,128);
    v[128]=c[0];
    v[129]=c[1];
    v[130]=c[2];
    v[131]=c[3];
    v[132]=c[4];
    v[133]=c[5];
    v[134]=c[6];
    v[135]=c[7];
    v[136]=c[8];
    v[137]=c[9];
    v[138]=c[10];
    v[139]=c[11];
    v[140]=c[12];
    v[141]=c[13];
    v[142]=c[14];
    v[143]=c[15];
  }
  { pin_v16 c;
    c[0]=v[144];
    c[1]=v[145];
    c[2]=v[146];
    c[3]=v[147];
    c[4]=v[148];
    c[5]=v[149];
    c[6]=v[150];
    c[7]=v[151];
    c[8]=v[152];
    c[9]=v[153];
    c[10]=v[154];
    c[11]=v[155];
    c[12]=v[156];
    c[13]=v[157];
    c[14]=v[158];
    c[15]=v[159];
    c=__builtin_amdgcn_pin_vgpr_v16f32(c,144);
    v[144]=c[0];
    v[145]=c[1];
    v[146]=c[2];
    v[147]=c[3];
    v[148]=c[4];
    v[149]=c[5];
    v[150]=c[6];
    v[151]=c[7];
    v[152]=c[8];
    v[153]=c[9];
    v[154]=c[10];
    v[155]=c[11];
    v[156]=c[12];
    v[157]=c[13];
    v[158]=c[14];
    v[159]=c[15];
  }
  { pin_v16 c;
    c[0]=v[160];
    c[1]=v[161];
    c[2]=v[162];
    c[3]=v[163];
    c[4]=v[164];
    c[5]=v[165];
    c[6]=v[166];
    c[7]=v[167];
    c[8]=v[168];
    c[9]=v[169];
    c[10]=v[170];
    c[11]=v[171];
    c[12]=v[172];
    c[13]=v[173];
    c[14]=v[174];
    c[15]=v[175];
    c=__builtin_amdgcn_pin_vgpr_v16f32(c,160);
    v[160]=c[0];
    v[161]=c[1];
    v[162]=c[2];
    v[163]=c[3];
    v[164]=c[4];
    v[165]=c[5];
    v[166]=c[6];
    v[167]=c[7];
    v[168]=c[8];
    v[169]=c[9];
    v[170]=c[10];
    v[171]=c[11];
    v[172]=c[12];
    v[173]=c[13];
    v[174]=c[14];
    v[175]=c[15];
  }
  { pin_v16 c;
    c[0]=v[176];
    c[1]=v[177];
    c[2]=v[178];
    c[3]=v[179];
    c[4]=v[180];
    c[5]=v[181];
    c[6]=v[182];
    c[7]=v[183];
    c[8]=v[184];
    c[9]=v[185];
    c[10]=v[186];
    c[11]=v[187];
    c[12]=v[188];
    c[13]=v[189];
    c[14]=v[190];
    c[15]=v[191];
    c=__builtin_amdgcn_pin_vgpr_v16f32(c,176);
    v[176]=c[0];
    v[177]=c[1];
    v[178]=c[2];
    v[179]=c[3];
    v[180]=c[4];
    v[181]=c[5];
    v[182]=c[6];
    v[183]=c[7];
    v[184]=c[8];
    v[185]=c[9];
    v[186]=c[10];
    v[187]=c[11];
    v[188]=c[12];
    v[189]=c[13];
    v[190]=c[14];
    v[191]=c[15];
  }
  vv = __builtin_bit_cast(V, v);
}

template<int BLOCK_SIZE, int BLOCK_M, int BLOCK_N, int BLOCK_K, int TILE_M, int TILE_N, int TILE_K, int WAVE_M, int WAVE_N, int WAVE_K>
__global__ void matrix_core_kernel_block_v2(const void* __restrict__ ptr_a,
                                         const void* __restrict__ ptr_b,
                                         void* __restrict__ ptr_c,
                                         int k,
                                         int stride_a, // stride in unit of pixel
                                         int stride_b,
                                         int stride_c)
{
    using opus::operator""_I;
    constexpr int W_M = WAVE_M;
    constexpr int W_N = WAVE_N;
    constexpr int W_K = WAVE_K;

    constexpr int T_M = TILE_M;
    constexpr int T_N = TILE_N;
    constexpr int T_K = TILE_K;

    constexpr int E_M = BLOCK_M / (W_M * T_M);
    constexpr int E_N = BLOCK_N / (W_N * T_N);
    constexpr int E_K = BLOCK_K / (W_K * T_K);
    static_assert(E_K == 1);

    using d_a = opus::fp16_t;
    using d_b = opus::fp16_t;
    using d_c = opus::fp32_t;

    int lane_id = threadIdx.x % opus::get_warp_size();
    int wave_id = threadIdx.x / opus::get_warp_size();
    int g_im = blockIdx.x * BLOCK_M;
    int g_in = blockIdx.y * BLOCK_N;

    // NOTE: the shape merge is per-dim
    //
    // A:[(expd_m<y>, tile_m<p>), (expd_k<y>, tile_k<p>)] * [(grpm_a<p>), (rept_a<y>, grpk_a<p>, pack_a<y>)]
    // B:[(expd_n<y>, tile_n<p>), (expd_k<y>, tile_k<p>)] * [(grpn_b<p>), (rept_b<y>, grpk_b<p>, pack_b<y>)]
    // C:[(expd_m<y>, tile_m<p>), (expd_n<y>, tile_n<p>)] * [(grpn_c<p>), (rept_c<y>, grpm_c<p>, pack_c<y>)]
    //
    // A:[(expd_m<y>, tile_m<p>, grpm_a<p>), (expd_k<y>, tile_k<p>, rept_a<y>, grpk_a<p>, pack_a<y>)]
    // B:[(expd_n<y>, tile_n<p>, grpn_b<p>), (expd_k<y>, tile_k<p>, rept_b<y>, grpk_b<p>, pack_b<y>)]
    // C:[(expd_m<y>, tile_m<p>, grpn_c<p>), (expd_n<y>, tile_n<p>, rept_c<y>, grpm_c<p>, pack_c<y>)]
    //
    auto mma  = opus::make_tiled_mma<d_a, d_b, d_c>(opus::seq<E_M, E_N, E_K>{}, opus::seq<T_M, T_N, T_K>{}, opus::seq<W_M, W_N, W_K>{}, opus::mfma_adaptor_swap_ab{});

    auto u_a = opus::partition_layout_a<4>(mma, opus::make_tuple(stride_a, 1_I), opus::make_tuple(wave_id / 2, lane_id % mma.grpm_a, 0_I, lane_id / mma.grpm_a) /*tile_m<p>, grpm_a<p>, tile_k<p>, grpk_a<p>*/);
    auto u_b = opus::partition_layout_b<4>(mma, opus::make_tuple(stride_b, 1_I), opus::make_tuple(wave_id % 2, lane_id % mma.grpn_b, 0_I, lane_id / mma.grpn_b) /*tile_n<p>, grpn_b<p>, tile_k<p>, grpk_b<p>*/);
    auto u_c = opus::partition_layout_c(mma, opus::make_tuple(stride_c, 1_I), opus::make_tuple(wave_id / 2, lane_id % mma.grpn_c, wave_id % 2, lane_id / mma.grpn_c) /*tile_m<p>, grpn_c<p> tile_n<p>, grpm_c<p>*/);
    auto g_a = opus::make_gmem(reinterpret_cast<const d_a*>(ptr_a) + g_im * stride_a);
    auto g_b = opus::make_gmem(reinterpret_cast<const d_b*>(ptr_b) + g_in * stride_b);
    auto g_c = opus::make_gmem(reinterpret_cast<opus::fp16_t*>(ptr_c) + g_im * stride_c + g_in);

    // start of kernel
    int loops = (k + BLOCK_K - 1) / BLOCK_K;
#if 1
    typename decltype(mma)::vtype_c v_c;
    opus::clear(v_c);
    pin_vc(v_c);

    for(auto i = 0; i < loops; i++ ) {
        auto v_a = g_a.load<4>(u_a);  u_a += BLOCK_K;
        auto v_b = g_b.load<4>(u_b);  u_b += BLOCK_K;
        v_c = mma(v_a, v_b, v_c);
        pin_vc(v_c);
    }

    auto v_c_f16 = opus::cast<fp16_t>(v_c);
    g_c.store<4>(v_c_f16, u_c);
#else
    auto v_a = g_a.load<4>(u_a);  u_a += BLOCK_K;
    auto v_b = g_b.load<4>(u_b);  u_b += BLOCK_K;
    auto v_c = mma(v_a, v_b);   // first time, C is always zero

    for(auto i = 0; i < loops - 1; i++ ) {
        v_a = g_a.load<4>(u_a);  u_a += BLOCK_K;
        v_b = g_b.load<4>(u_b);  u_b += BLOCK_K;
        v_c = mma(v_a, v_b, v_c);
    }

    auto v_c_f16 = opus::cast<fp16_t>(v_c);
    g_c.store<4>(v_c_f16, u_c);
#endif
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
            if(pp_err<100)
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

void block_run()
{
    int m = 256 * 2;
    int n = 192 * 2;
    int k = 8 * 8;

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
        constexpr int BLOCK_M = 256;
        constexpr int BLOCK_N = 192;
        constexpr int BLOCK_K = 16;
        constexpr int TILE_M = 2;
        constexpr int TILE_N = 2;
        constexpr int TILE_K = 1;
        constexpr int WAVE_M = 16;
        constexpr int WAVE_N = 16;
        constexpr int WAVE_K = 16;

        auto gdim = dim3(m / BLOCK_M, n / BLOCK_N);
        auto kernel = matrix_core_kernel_block_v2<256, BLOCK_M, BLOCK_N, BLOCK_K, TILE_M, TILE_N, TILE_K, WAVE_M, WAVE_N, WAVE_K>;
        kernel<<<gdim, 256>>>(dev_a, dev_b, dev_c, k, lda, ldb, ldc);

        HIP_CALL(hipMemcpy(fp16_c, dev_c, ldc*m*sizeof(float16), hipMemcpyDeviceToHost));
#if 1
        bool res = valid_vector( host_c, fp16_c, m*n, 1e-3);
        printf("[%dx%dx%d, block_gemm_%dx%dx%d_%dx%dx%d_%dx%dx%d], %s", m, n, k,
            BLOCK_M, BLOCK_N, BLOCK_K, TILE_M, TILE_N, TILE_K, WAVE_M, WAVE_N, WAVE_K,
            res?"valid":"fail");fflush(stdout);
        printf("\n"); fflush(stdout);
#endif
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
}

int main(int argc, char** argv){ block_run(); return 0; }
