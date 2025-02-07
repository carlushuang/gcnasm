#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <random>
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <numeric>
#include <cmath>
#include <unordered_map>

#define HIP_CALL(call) do{  \
    hipError_t err = call;  \
    if(err != hipSuccess){  \
        printf("[hiperror](%d) fail to call %s",(int)err,#call);    \
        exit(0);            \
    }                       \
} while(0)


__device__ uint8_t to_fp8_raw_from_fp32(float v) {
  uint8_t i8data;
  union {
    float fval;
    uint32_t i32val;
    uint8_t i8val[4];  // NOTE: not endian independent
  } val;

  uint32_t ival = 0;
  val.fval = v;

  if ((val.i32val & 0x7F800000) !=
      0x7F800000) {  /// propagate NAN/INF, no clipping
    val.fval = __builtin_amdgcn_fmed3f(val.fval, 240.0, -240.0);
  }

  ival = __builtin_amdgcn_cvt_pk_fp8_f32(val.fval, val.fval, ival,
                                         false);  // false -> WORD0
  val.i32val = ival;
  i8data = val.i8val[0];

  return i8data;
}

template <int BLOCK_SIZE = 256>
__global__ void cvt_fp8(const void* input_f32, void* output_fp8, int pixels)
{
    const float* i_f32 = reinterpret_cast<const float*>(input_f32);
    int cur_pixel = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if(cur_pixel >= pixels )
        return ;

    float fp32 = i_f32[cur_pixel];

    uint8_t fp8_raw = to_fp8_raw_from_fp32(fp32);
    reinterpret_cast<uint8_t*>(output_fp8)[cur_pixel] = fp8_raw;
}

float fp8_e4m3fnuz_to_fp32_raw(uint8_t fp8_raw)
{
    constexpr int32_t exponent = 4;
    constexpr int32_t mantissa = 3;
    constexpr int32_t bias = 1 << (exponent - 1);   // AMD's

    // TODO: NAN/INF
    int32_t sign_v     = (fp8_raw >> 7) & 0x1;
    int32_t exponent_v = (fp8_raw >> 3) & 0xf;
    int32_t mantissa_v = (fp8_raw >> 0) & 0x7;
    if (sign_v == 1 && exponent_v == 0 && mantissa_v == 0) {
        // TODO: INF/NAN share same representation
        return std::numeric_limits<float>::quiet_NaN();
    }
    if (exponent_v == 0) {
        // subnmorm value
        float e = powf(2, 1 - bias);
        uint8_t f = mantissa_v;
        float v = ((0x4 & f) ? powf(2, -1) : 0) + 
                    ((0x2 & f) ? powf(2, -2) : 0) +
                    ((0x1 & f) ? powf(2, -3) : 0);
        return (sign_v ? -1 : 1) * v * e;
    }
    else {
        float e = powf(2, exponent_v - bias);
        uint8_t f = mantissa_v;
        float v = 1 + ((0x4 & f) ? powf(2, -1) : 0) + 
                    ((0x2 & f) ? powf(2, -2) : 0) +
                    ((0x1 & f) ? powf(2, -3) : 0);
        return (sign_v ? -1 : 1) * v * e;
    }
}

void fp8_e4m3fnuz_binary() {
    // AMD's format S.EEEE.MMM
    constexpr int32_t m_norm_min = 0b0001;
    constexpr int32_t m_norm_max = 0b1111;
    constexpr int32_t exponent = 4;
    constexpr int32_t mantissa = 3;
    constexpr int32_t bias = 1 << (exponent - 1);   // AMD's

    printf("subnorm:\n");
    for(int32_t m = 0b000; m <= 0b111; m++) {
        // (-1)^S * 0.M * 2^(1-bias)
        float e = powf(2, 1 - bias);
        uint8_t f = m;  // simplicity
        float v = ((0x4 & f) ? powf(2, -1) : 0) + 
                    ((0x2 & f) ? powf(2, -2) : 0) +
                    ((0x1 & f) ? powf(2, -3) : 0);
        v *= e;
        printf("  0x%02x -- %.6f, 0x%02x -- %.6f\n", f, v, 0x80 | f, -1 * v);
    }

    printf("norm:\n");
    for(int32_t i = m_norm_min; i <=  m_norm_max; i++) {
        float e = powf(2, i - bias);
        for(int32_t m = 0b000; m <= 0b111; m++) {
            // (-1)^S * 1.M * 2^(1-bias)
            uint8_t f = (i << 3) | m;  // simplicity
            float v = 1 + ((0x4 & f) ? powf(2, -1) : 0) + 
                        ((0x2 & f) ? powf(2, -2) : 0) +
                        ((0x1 & f) ? powf(2, -3) : 0);
            v *= e;
            printf("  0x%02x -- %.6f, 0x%02x -- %.6f\n", f, v, 0x80 | f, -1 * v);
        }
    }
}

int main(int argc, char ** argv)
{
    float *host_src;
    uint8_t *host_dst;
    void * device_src, * device_dst;

    float input = -54; //-53.999754239370844;

    //fp32 on host
    host_src = (float*)malloc(1*sizeof(float));
    host_dst = (uint8_t*)malloc(1*sizeof(uint8_t));

    host_src[0] = input;

    HIP_CALL(hipMalloc(&device_src, 1 * sizeof(float)));
    HIP_CALL(hipMalloc(&device_dst, 1 * sizeof(uint8_t)));

    HIP_CALL(hipMemcpy(device_src, host_src, 1 * sizeof(float), hipMemcpyHostToDevice));
    constexpr int block_size = 256;

    cvt_fp8<<<1, block_size>>>(device_src, device_dst, 1);

    HIP_CALL(hipMemcpy(host_dst, device_dst, 1*sizeof(uint8_t), hipMemcpyDeviceToHost));

    float fp32_dev = fp8_e4m3fnuz_to_fp32_raw(host_dst[0]);
    printf("src_f32:%f, dst_fp8_raw:%x, dst_f32:%f\n",input, host_dst[0], fp32_dev);


    fp8_e4m3fnuz_binary();
}
