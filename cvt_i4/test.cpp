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

// i4 in dword: [e0, e2, e4, e6, e1, e3, e5, e7]
template <int BLOCK_SIZE = 256>
__global__ void cvt_i4x8_fp8x8(const void* ptr_input_i4, void* ptr_out_i8, int pixels)
{
    int cur_pixel = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if(cur_pixel * 8 > pixels)
        return;
    uint32_t i4x8 = reinterpret_cast<const uint32_t*>(ptr_input_i4)[cur_pixel];
    uint32_t fp8x4_0;
    uint32_t fp8x4_1;
    float tmp_0, tmp_1, tmp_2;

    asm volatile (
        "v_cvt_off_f32_i4 %[v_tmp_0], %[v_src]\n"
        "v_cvt_off_f32_i4 %[v_tmp_1], %[v_src], src0_sel:BYTE_2\n"
        "v_cvt_pk_fp8_f32 %[v_dst_0], %[v_tmp_0], %[v_tmp_1]\n"
        "v_cvt_off_f32_i4 %[v_tmp_0], %[v_src], src0_sel:BYTE_1\n"
        "v_cvt_off_f32_i4 %[v_tmp_1], %[v_src], src0_sel:BYTE_3\n"
        "v_cvt_pk_fp8_f32 %[v_dst_1], %[v_tmp_0], %[v_tmp_1]\n"
        "v_lshrrev_b32 %[v_tmp_2], 4, %[v_src]\n"
        "v_cvt_off_f32_i4 %[v_tmp_0], %[v_tmp_2]\n"
        "v_cvt_off_f32_i4 %[v_tmp_1], %[v_tmp_2], src0_sel:BYTE_2\n"
        "v_cvt_pk_fp8_f32 %[v_dst_0], %[v_tmp_0], %[v_tmp_1], op_sel:[0, 0, 1]\n"
        "v_cvt_off_f32_i4 %[v_tmp_0], %[v_tmp_2], src0_sel:BYTE_1\n"
        "v_cvt_off_f32_i4 %[v_tmp_1], %[v_tmp_2], src0_sel:BYTE_3\n"
        "v_cvt_pk_fp8_f32 %[v_dst_1], %[v_tmp_0], %[v_tmp_1], op_sel:[0, 0, 1]\n"
        : [v_tmp_0]"+v"(tmp_0), [v_tmp_1]"+v"(tmp_1), [v_tmp_2]"+v"(tmp_2),
          [v_dst_0]"+v"(fp8x4_0), [v_dst_1]"+v"(fp8x4_1), [v_src]"+v"(i4x8)
        : 
    );

    // printf("tid:%d, i4x8:%x, 0:%f, 1:%f, ->%x, %x\n", static_cast<int>(threadIdx.x), i4x8, tmp_0, tmp_1, fp8x4_0, fp8x4_1);

    reinterpret_cast<uint32_t*>(ptr_out_i8)[cur_pixel * 2 + 0] = fp8x4_0;
    reinterpret_cast<uint32_t*>(ptr_out_i8)[cur_pixel * 2 + 1] = fp8x4_1;
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

float i4_to_f32_gfx9(uint8_t i4)
{
    static std::unordered_map<uint8_t, float> u = {
        {0b1000,    -0.5000f},
        {0b1001,    -0.4375f},
        {0b1010,    -0.3750f},
        {0b1011,    -0.3125f},
        {0b1100,    -0.2500f},
        {0b1101,    -0.1875f},
        {0b1110,    -0.1250f},
        {0b1111,    -0.0625f},
        {0b0   ,    +0.0000f},
        {0b1   ,    +0.0625f},
        {0b10  ,    +0.1250f},
        {0b11  ,    +0.1875f},
        {0b100 ,    +0.2500f},
        {0b101 ,    +0.3125f},
        {0b110 ,    +0.3750f},
        {0b111 ,    +0.4375f}};

    return u[i4];
}

static inline uint32_t perm_i4_dword(uint32_t x)
{
    // [e0, e2, e4, e6, e1, e3, e5, e7]
    uint32_t e0 = (x & 0x0000000f) >> 0;
    uint32_t e1 = (x & 0x000000f0) >> 4;
    uint32_t e2 = (x & 0x00000f00) >> 8;
    uint32_t e3 = (x & 0x0000f000) >> 12;
    uint32_t e4 = (x & 0x000f0000) >> 16;
    uint32_t e5 = (x & 0x00f00000) >> 20;
    uint32_t e6 = (x & 0x0f000000) >> 24;
    uint32_t e7 = (x & 0xf0000000) >> 28;

    return e0 | e2 << 4 | e4 << 8 | e6 << 12 | e1 << 16 | e3 << 20  | e5 << 24  | e7 << 28;
}

void permute_i4_per_dword(uint32_t * dst_i4_dwords, const uint32_t * src_i4_dwords, int num_dwords) {
    for(int i = 0; i < num_dwords; i++) {
        dst_i4_dwords[i] = perm_i4_dword(src_i4_dwords[i]);
    }
}

int main(int argc, char ** argv)
{
    int pixels = 256 * 8;
    int i4_bytes = pixels / 2;
    int f8_bytes = pixels;

    uint8_t *host_src, *host_src_perm;
    uint8_t *host_dst;
    void * device_src, * device_dst;

    //fp32 on host
    host_src = (uint8_t*)malloc(i4_bytes*sizeof(uint8_t));
    host_dst = (uint8_t*)malloc(f8_bytes*sizeof(uint8_t));
    host_src_perm = (uint8_t*)malloc(i4_bytes*sizeof(uint8_t));

    //convert fp32 a and b into fp16 on host
    for(auto i = 0; i < i4_bytes; i++) {
        uint8_t pk_i4 = static_cast<uint8_t>(2 * i + 0) | (static_cast<uint8_t>(2 * i + 1) << 4);
        host_src[i] = pk_i4;
    }

    permute_i4_per_dword(reinterpret_cast<uint32_t*>(host_src_perm), reinterpret_cast<uint32_t*>(host_src), i4_bytes / 4);

    HIP_CALL(hipMalloc(&device_src, i4_bytes * sizeof(uint8_t)));
    HIP_CALL(hipMalloc(&device_dst, f8_bytes * sizeof(uint8_t)));

    HIP_CALL(hipMemcpy(device_src, host_src_perm, i4_bytes * sizeof(uint8_t), hipMemcpyHostToDevice));
    constexpr int block_size = 256;
    constexpr int pixels_per_block  = block_size * 8;

    cvt_i4x8_fp8x8<<<(pixels + pixels_per_block - 1) / pixels_per_block, block_size>>>(device_src, device_dst, pixels);

    HIP_CALL(hipMemcpy(host_dst, device_dst, f8_bytes*sizeof(uint8_t), hipMemcpyDeviceToHost));

    for(auto i = 0 ;i < i4_bytes; i++) {
        uint8_t i0 = host_src[i] & 0xf;
        uint8_t i1 = (host_src[i] & 0xf0) >> 4;
        printf("[%3d]%x -> 0x%02x(%f), %x -> 0x%02x(%f)\n", i, i0, host_dst[2*i], fp8_e4m3fnuz_to_fp32_raw(host_dst[2*i]),
                                                               i1, host_dst[2*i+1], fp8_e4m3fnuz_to_fp32_raw(host_dst[2*i+1]));
    }
    // fp8_e4m3fnuz_binary();
}
