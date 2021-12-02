#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

__device__ void amd_assembly_outer_product_1x4(float a,
                                               float b0,
                                               float b1,
                                               float b2,
                                               float b3,
                                               float& c0,
                                               float& c1,
                                               float& c2,
                                               float& c3)
{
    asm volatile("\n \
            v_dot2_f32_f16 %0, %4, %5, %0\n \
            v_dot2_f32_f16 %1, %4, %6, %1\n \
            v_dot2_f32_f16 %2, %4, %7, %2\n \
            v_dot2_f32_f16 %3, %4, %8, %3\n \
            "
                 : "=v"(c0), "=v"(c1), "=v"(c2), "=v"(c3)
                 : "v"(a), "v"(b0), "v"(b1), "v"(b2), "v"(b3), "0"(c0), "1"(c1), "2"(c2), "3"(c3));
}

#define CLR16x4(c) {        \
    c[0] = (float4)0;       \
    c[1] = (float4)0;       \
    c[2] = (float4)0;       \
    c[3] = (float4)0;       \
    c[4] = (float4)0;       \
    c[5] = (float4)0;       \
    c[6] = (float4)0;       \
    c[7] = (float4)0;       \
    c[8] = (float4)0;       \
    c[9] = (float4)0;       \
    c[10] = (float4)0;      \
    c[11] = (float4)0;      \
    c[12] = (float4)0;      \
    c[13] = (float4)0;      \
    c[14] = (float4)0;      \
    c[15] = (float4)0;      \
}

extern "C"
__global__ __launch_bounds__(256,2)
void hgemm_128x128(
    unsigned char *  ptr_c,
    const unsigned char * __restrict__ ptr_a,
    const unsigned char * __restrict__ ptr_b,
    float alpha,
    unsigned int m, unsigned int n, unsigned int k,
    unsigned int lda, unsigned int ldb, unsigned int ldc)
{
    __shared__ char smem[32768];
    float4 c[16];//represent 16x4 FP32, will be convert to 16 FP16
    float4 a[2];//represent 2x4x(2 x FP16)
    float4 b[2];//represent 2x4x(2 x FP16)
    /*
    *  8x8 thread tile:
    *
    *       b0 b1 b2 b3 b4 b5 b6 b7
    *
    *  a0   c0 c4 c8 12 32 36 40 44
    *  a1   c1 c5 c9 13 33 37 41 45
    *  a2   c2 c6 10 14 34 38 42 46
    *  a3   c3 c7 11 15 35 39 43 47
    *  a4   16 20 24 28 48 52 56 60
    *  a5   17 21 25 29 49 53 57 61
    *  a6   18 22 26 30 50 54 58 62
    *  a7   19 23 27 31 51 55 59 63
    */
    //unroll k=32, two FP16 packed into one FP32
    //Every thread read 8x(2FP16) A and 8x(2FP16) B from global and store into LDS.
    float4 p0[2], p1[2], q0[2], q1[2];

    unsigned int tid = threadIdx.x;
    unsigned int bx=blockIdx.x;
    unsigned int m_blocks = (m+127)>>7;
    unsigned int m_idx = bx % m_blocks;
    unsigned int n_idx = bx / m_blocks;
    unsigned int lane_id = tid&63;
    unsigned int wave_id = tid>>6;
    unsigned int wave_p = wave_id>>1; 
    unsigned int wave_q = wave_id&1; 
    //for read A,B from global to LDS
    //Per wave read 4 strides from A, B
    unsigned int laneid_lo = lane_id&15; 
    unsigned int laneid_hi = lane_id>>4; 
    unsigned int lane_lo = lane_id&31; 
    unsigned int lane_hi = lane_id>>5; 
    unsigned int bs_a = lda<<4;
    unsigned int bs_b = ldb<<4;
    unsigned int lane_w = lane_id >> 4;
    unsigned int lane_u = (lane_id&15)>>1;
    unsigned int lane_v = (lane_id&15)&1;

    ptr_a += ((wave_id<<2)|laneid_hi)*lda + (m_idx<<(7+2)) + (laneid_lo<<5);
    ptr_b += ((wave_id<<2)|laneid_hi)*ldb + (n_idx<<(7+2)) + (laneid_lo<<5);
    ptr_c += ((n_idx<<7)+ ((wave_id<<2)|lane_hi))*(ldc>>1) + (m_idx<<(7+1)) + (lane_lo<<(2+1));
    
    float4 * smem_store = (float4 *)&smem[tid<<5];
    float4 * smem_load_a = (float4 *)&smem[(wave_p<<(6+2))|(lane_u<<(2+2))];
    float4 * smem_load_b = (float4 *)&smem[0x2000|(wave_q<<(6+2))|(lane_w<<(3+2))|(lane_v<<(2+2))];

    p0[0] = *((const float4* __restrict__)ptr_a); p0[1] = *((const float4* __restrict__)&ptr_a[16]); ptr_a += bs_a;
    q0[0] = *((const float4* __restrict__)ptr_b); q0[1] = *((const float4* __restrict__)&ptr_b[16]); ptr_b += bs_b;
    CLR16x4(c)
    for(unsigned int ik=16; ik < (k>>1); ik += 16){
        p1[0] = *((const float4* __restrict__)ptr_a); p1[1] = *((const float4* __restrict__)&ptr_a[16]); ptr_a += bs_a;
        q1[0] = *((const float4* __restrict__)ptr_b); q1[1] = *((const float4* __restrict__)&ptr_b[16]); ptr_b += bs_b;
        smem_store[0] = p0[0]; 
        smem_store[1] = p0[1];
        smem_store[0x200] = q0[0];
        smem_store[0x201] = q0[1];
        __syncthreads();
        #pragma unroll
        for(unsigned int i=0;i<16;i++) {
            a[0] = smem_load_a[(i<<5)];
            a[1] = smem_load_a[(i<<5)|8];
            b[0] = smem_load_b[(i<<5)];
            b[1] = smem_load_b[(i<<5)|8];
            amd_assembly_outer_product_1x4(a[0].x, b[0].x, b[0].y, b[0].z, b[0].w, c[0].x, c[1].x,  c[2].x,  c[3].x);
            amd_assembly_outer_product_1x4(a[0].y, b[0].x, b[0].y, b[0].z, b[0].w, c[0].y, c[1].y,  c[2].y,  c[3].y);
            amd_assembly_outer_product_1x4(a[0].z, b[0].x, b[0].y, b[0].z, b[0].w, c[0].z, c[1].z,  c[2].z,  c[3].z);
            amd_assembly_outer_product_1x4(a[0].w, b[0].x, b[0].y, b[0].z, b[0].w, c[0].w, c[1].w,  c[2].w,  c[3].w);
            amd_assembly_outer_product_1x4(a[1].x, b[0].x, b[0].y, b[0].z, b[0].w, c[4].x, c[5].x,  c[6].x,  c[7].x);
            amd_assembly_outer_product_1x4(a[1].y, b[0].x, b[0].y, b[0].z, b[0].w, c[4].y, c[5].y,  c[6].y,  c[7].y);
            amd_assembly_outer_product_1x4(a[1].z, b[0].x, b[0].y, b[0].z, b[0].w, c[4].z, c[5].z,  c[6].z,  c[7].z);
            amd_assembly_outer_product_1x4(a[1].w, b[0].x, b[0].y, b[0].z, b[0].w, c[4].w, c[5].w,  c[6].w,  c[7].w);
            amd_assembly_outer_product_1x4(a[0].x, b[1].x, b[1].y, b[1].z, b[1].w, c[8].x, c[9].x,  c[10].x,  c[11].x);
            amd_assembly_outer_product_1x4(a[0].y, b[1].x, b[1].y, b[1].z, b[1].w, c[8].y, c[9].y,  c[10].y,  c[11].y);
            amd_assembly_outer_product_1x4(a[0].z, b[1].x, b[1].y, b[1].z, b[1].w, c[8].z, c[9].z,  c[10].z,  c[11].z);
            amd_assembly_outer_product_1x4(a[0].w, b[1].x, b[1].y, b[1].z, b[1].w, c[8].w, c[9].w,  c[10].w,  c[11].w);
            amd_assembly_outer_product_1x4(a[1].x, b[1].x, b[1].y, b[1].z, b[1].w, c[12].x, c[13].x,  c[14].x,  c[15].x);
            amd_assembly_outer_product_1x4(a[1].y, b[1].x, b[1].y, b[1].z, b[1].w, c[12].y, c[13].y,  c[14].y,  c[15].y);
            amd_assembly_outer_product_1x4(a[1].z, b[1].x, b[1].y, b[1].z, b[1].w, c[12].z, c[13].z,  c[14].z,  c[15].z);
            amd_assembly_outer_product_1x4(a[1].w, b[1].x, b[1].y, b[1].z, b[1].w, c[12].w, c[13].w,  c[14].w,  c[15].w);
        }
        __syncthreads();
        p0[0] = p1[0];
        q0[0] = q1[0];
        p0[1] = p1[1];
        q0[1] = q1[1];
    }
    smem_store[0] = p0[0]; 
    smem_store[1] = p0[1];
    smem_store[0x200] = q0[0];
    smem_store[0x201] = q0[1];
    __syncthreads();
    #pragma unroll
    for(unsigned int i=0;i<16;i++) {
        a[0] = smem_load_a[(i<<5)];
        a[1] = smem_load_a[(i<<5)|8];
        b[0] = smem_load_b[(i<<5)];
        b[1] = smem_load_b[(i<<5)|8];
        amd_assembly_outer_product_1x4(a[0].x, b[0].x, b[0].y, b[0].z, b[0].w, c[0].x, c[1].x,  c[2].x,  c[3].x);
        amd_assembly_outer_product_1x4(a[0].y, b[0].x, b[0].y, b[0].z, b[0].w, c[0].y, c[1].y,  c[2].y,  c[3].y);
        amd_assembly_outer_product_1x4(a[0].z, b[0].x, b[0].y, b[0].z, b[0].w, c[0].z, c[1].z,  c[2].z,  c[3].z);
        amd_assembly_outer_product_1x4(a[0].w, b[0].x, b[0].y, b[0].z, b[0].w, c[0].w, c[1].w,  c[2].w,  c[3].w);
        amd_assembly_outer_product_1x4(a[1].x, b[0].x, b[0].y, b[0].z, b[0].w, c[4].x, c[5].x,  c[6].x,  c[7].x);
        amd_assembly_outer_product_1x4(a[1].y, b[0].x, b[0].y, b[0].z, b[0].w, c[4].y, c[5].y,  c[6].y,  c[7].y);
        amd_assembly_outer_product_1x4(a[1].z, b[0].x, b[0].y, b[0].z, b[0].w, c[4].z, c[5].z,  c[6].z,  c[7].z);
        amd_assembly_outer_product_1x4(a[1].w, b[0].x, b[0].y, b[0].z, b[0].w, c[4].w, c[5].w,  c[6].w,  c[7].w);
        amd_assembly_outer_product_1x4(a[0].x, b[1].x, b[1].y, b[1].z, b[1].w, c[8].x, c[9].x,  c[10].x,  c[11].x);
        amd_assembly_outer_product_1x4(a[0].y, b[1].x, b[1].y, b[1].z, b[1].w, c[8].y, c[9].y,  c[10].y,  c[11].y);
        amd_assembly_outer_product_1x4(a[0].z, b[1].x, b[1].y, b[1].z, b[1].w, c[8].z, c[9].z,  c[10].z,  c[11].z);
        amd_assembly_outer_product_1x4(a[0].w, b[1].x, b[1].y, b[1].z, b[1].w, c[8].w, c[9].w,  c[10].w,  c[11].w);
        amd_assembly_outer_product_1x4(a[1].x, b[1].x, b[1].y, b[1].z, b[1].w, c[12].x, c[13].x,  c[14].x,  c[15].x);
        amd_assembly_outer_product_1x4(a[1].y, b[1].x, b[1].y, b[1].z, b[1].w, c[12].y, c[13].y,  c[14].y,  c[15].y);
        amd_assembly_outer_product_1x4(a[1].z, b[1].x, b[1].y, b[1].z, b[1].w, c[12].z, c[13].z,  c[14].z,  c[15].z);
        amd_assembly_outer_product_1x4(a[1].w, b[1].x, b[1].y, b[1].z, b[1].w, c[12].w, c[13].w,  c[14].w,  c[15].w);
    }

    {
        #pragma unroll
        for(int i=0;i<16;i++){ c[i].x*=alpha; c[i].y*=alpha; c[i].z*=alpha; c[i].w*=alpha;}
        //convert fp32 in float4 c[16] to fp16.
        _Float16 fp16_c[64];
        for(int i=0;i<16;i++){
            fp16_c[4*i + 0]=__float2half(c[i].x);
            fp16_c[4*i + 1]=__float2half(c[i].y);
            fp16_c[4*i + 2]=__float2half(c[i].z);
            fp16_c[4*i + 3]=__float2half(c[i].w);
        }
        
        float2 out_c[16];
        for(int i=0;i<16;i++){
            uint32_t x, y;
            x=( (uint32_t)(*((uint16_t*)&fp16_c[4*i + 0])) | (( (uint32_t)(*((uint16_t*)&fp16_c[4*i + 1])))<<16) );
            y=( (uint32_t)(*((uint16_t*)&fp16_c[4*i + 2])) | (( (uint32_t)(*((uint16_t*)&fp16_c[4*i + 3])))<<16) );
            out_c[i].x=reinterpret_cast<float &>(x);
            out_c[i].y=reinterpret_cast<float &>(y);
        }
        float2 * smem_store_c = (float2 *)&smem[(wave_q<<(11+1))|(lane_w<<(9+1))|(lane_v<<(8+1))|(wave_p<<(6+1))|(lane_u<<(2+1))];
        float2 * smem_load_c = (float2 *)&smem[(wave_id<<(8+1))|(lane_hi<<(7+1))|(lane_lo<<(2+1))];
        #pragma unroll
        for(int i=0; i<4; i++){
            int cid = ((i>>1)<<3)|((i&1)<<1);
            int cof = ((i>>1)<<5)|((i&1)<<1);
            smem_store_c[0] = out_c[cid+0];
            smem_store_c[8] = out_c[cid+4];
            smem_store_c[32] = out_c[cid+1];
            smem_store_c[40] = out_c[cid+5];
            __syncthreads();
    
            *((float2*)&ptr_c[(cof+0)* (ldc>>1)]) = smem_load_c[256*0];
            *((float2*)&ptr_c[(cof+16)*(ldc>>1)]) = smem_load_c[256*1];
            *((float2*)&ptr_c[(cof+64)*(ldc>>1)]) = smem_load_c[256*2];
            *((float2*)&ptr_c[(cof+80)*(ldc>>1)]) = smem_load_c[256*3];
            __syncthreads();
        }
    }
}