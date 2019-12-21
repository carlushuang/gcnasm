#include <hip/hip_runtime.h>

#define S_FMA4x4(c, a, b){       \
    (c)[0]  += (a)[0]*(b)[0];    \
    (c)[1]  += (a)[1]*(b)[0];    \
    (c)[2]  += (a)[2]*(b)[0];    \
    (c)[3]  += (a)[3]*(b)[0];    \
    (c)[4]  += (a)[0]*(b)[1];    \
    (c)[5]  += (a)[1]*(b)[1];    \
    (c)[6]  += (a)[2]*(b)[1];    \
    (c)[7]  += (a)[3]*(b)[1];    \
    (c)[8]  += (a)[0]*(b)[2];    \
    (c)[9]  += (a)[1]*(b)[2];    \
    (c)[10] += (a)[2]*(b)[2];    \
    (c)[11] += (a)[3]*(b)[2];    \
    (c)[12] += (a)[0]*(b)[3];    \
    (c)[13] += (a)[1]*(b)[3];    \
    (c)[14] += (a)[2]*(b)[3];    \
    (c)[15] += (a)[3]*(b)[3];    \
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
void sgemm_128x128(
    unsigned char *  ptr_c,
    const unsigned char * __restrict__ ptr_a,
    const unsigned char * __restrict__ ptr_b,
    float alpha,
    unsigned int m, unsigned int n, unsigned int k,
    unsigned int lda, unsigned int ldb, unsigned int ldc)
{
    __shared__ char smem[16384];
    float4 c[16];
    float4 a[2];
    float4 b[2];
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
    float4 p0, p1, q0, q1;

    unsigned int tid = threadIdx.x;
    unsigned int bx=blockIdx.x;
    unsigned int m_blocks = (m+127)>>7;
    unsigned int m_idx = bx % m_blocks;
    unsigned int n_idx = bx / m_blocks;
    unsigned int lane_id = tid&63;
    unsigned int wave_id = tid>>6;
    unsigned int wave_p = wave_id>>1;
    unsigned int wave_q = wave_id&1;
    unsigned int lane_lo = lane_id&31;
    unsigned int lane_hi = lane_id>>5;
    unsigned int bs_a = lda<<3;
    unsigned int bs_b = ldb<<3;
    unsigned int lane_w = lane_id >> 4;
    unsigned int lane_u = (lane_id&15)>>1;
    unsigned int lane_v = (lane_id&15)&1;

    ptr_a += (m_idx<<(7+2)) + ((wave_id<<1)|lane_hi)*lda + (lane_lo<<4);
    ptr_b += (n_idx<<(7+2)) + ((wave_id<<1)|lane_hi)*ldb + (lane_lo<<4);
    ptr_c += ((n_idx<<7)+ ((wave_id<<2)|lane_hi))*ldc + (m_idx<<(7+2)) + (lane_lo<<4);

    float4 * smem_store = (float4 *)&smem[tid<<4];
    float4 * smem_load_a = (float4 *)&smem[(wave_p<<(6+2))|(lane_u<<(2+2))];
    float4 * smem_load_b = (float4 *)&smem[0x1000|(wave_q<<(6+2))|(lane_w<<(3+2))|(lane_v<<(2+2))];

    p0 = *((const float4* __restrict__)ptr_a); ptr_a += bs_a;
    q0 = *((const float4* __restrict__)ptr_b); ptr_b += bs_b;
    CLR16x4(c)

    for(unsigned int ik=8; ik < k; ik += 8){
        p1 = *((const float4* __restrict__)ptr_a); ptr_a += bs_a;
        q1 = *((const float4* __restrict__)ptr_b); ptr_b += bs_b;
        smem_store[0] = p0;
        smem_store[0x100] = q0;

        __syncthreads();
        //#pragma unroll
        for(unsigned int i=0;i<8;i++) {
            a[0] = smem_load_a[(i<<5)];
            a[1] = smem_load_a[(i<<5)|8];
            b[0] = smem_load_b[(i<<5)];
            b[1] = smem_load_b[(i<<5)|8];
            S_FMA4x4((float*)&c[0] , (float*)&a[0], (float*)&b[0])
            S_FMA4x4((float*)&c[4] , (float*)&a[1], (float*)&b[0])
            S_FMA4x4((float*)&c[8] , (float*)&a[0], (float*)&b[1])
            S_FMA4x4((float*)&c[12], (float*)&a[1], (float*)&b[1])
        }
        __syncthreads();
        p0 = p1;
        q0 = q1;
    }
    smem_store[0] = p0;
    smem_store[0x100] = q0;
    __syncthreads();
    //#pragma unroll
    for(unsigned int i=0;i<8;i++) {
        a[0] = smem_load_a[(i<<5)];
        a[1] = smem_load_a[(i<<5)|8];
        b[0] = smem_load_b[(i<<5)];
        b[1] = smem_load_b[(i<<5)|8];
        S_FMA4x4((float*)&c[0] , (float*)&a[0], (float*)&b[0])
        S_FMA4x4((float*)&c[4] , (float*)&a[1], (float*)&b[0])
        S_FMA4x4((float*)&c[8] , (float*)&a[0], (float*)&b[1])
        S_FMA4x4((float*)&c[12], (float*)&a[1], (float*)&b[1])
    }

    {
        //#pragma unroll
        for(int i=0;i<16;i++){ c[i].x*=alpha; c[i].y*=alpha; c[i].z*=alpha; c[i].w*=alpha;}

        float4 * smem_store_c = (float4 *)&smem[(wave_q<<(11+2))|(lane_w<<(9+2))|(lane_v<<(8+2))|(wave_p<<(6+2))|(lane_u<<(2+2))];
        float4 * smem_load_c = (float4 *)&smem[(wave_id<<(8+2))|(lane_hi<<(7+2))|(lane_lo<<(2+2))];
        //#pragma unroll
        for(int i=0; i<4; i++){
            int cid = ((i>>1)<<3)|((i&1)<<1);
            int cof = ((i>>1)<<5)|((i&1)<<1);
            smem_store_c[0] = c[cid+0];
            smem_store_c[8] = c[cid+4];
            smem_store_c[32] = c[cid+1];
            smem_store_c[40] = c[cid+5];

            __syncthreads();
            *((float4*)&ptr_c[(cof+0)*ldc]) = smem_load_c[256*0];
            *((float4*)&ptr_c[(cof+16)*ldc]) = smem_load_c[256*1];
            *((float4*)&ptr_c[(cof+64)*ldc]) = smem_load_c[256*2];
            *((float4*)&ptr_c[(cof+80)*ldc]) = smem_load_c[256*3];
            __syncthreads();
        }
    }
}