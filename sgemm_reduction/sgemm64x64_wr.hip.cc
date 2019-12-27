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

/*
* wave-wise reduction
* 256x1x1, 4 waves
* each wave compute align K(unrool 8), 4 waves compute 4x8=32 K
* each wave compute 64x64 fp32
* 
*/
extern "C"
__global__ __launch_bounds__(256,2)
void sgemm_64x64_wr(
    unsigned char *  ptr_c,
    const unsigned char * __restrict__ ptr_a,
    const unsigned char * __restrict__ ptr_b,
    float alpha,
    unsigned int m, unsigned int n, unsigned int k,
    unsigned int lda, unsigned int ldb, unsigned int ldc)
{
    __shared__ char smem[32768];
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
    float4 p0[2], p1[2], q0[2], q1[2];

    unsigned int tid = threadIdx.x;
    unsigned int bx=blockIdx.x;
    unsigned int m_blocks = (m+63)>>6;
    unsigned int m_idx = bx % m_blocks;
    unsigned int n_idx = bx / m_blocks;
    unsigned int lane_id = tid&63;
    unsigned int wave_id = tid>>6;
    unsigned int lane_lo = lane_id&15;
    unsigned int lane_hi = lane_id>>4;
    unsigned int lane_p  = lane_id&1;
    unsigned int lane_q  = lane_id>>1;
    unsigned int bs_a = lda<<4;     // 16x row
    unsigned int bs_b = ldb<<4;
    unsigned int bs_c = ldc<<4;
    unsigned int lane_w = lane_id >> 4;
    unsigned int lane_u = (lane_id&15)>>1;
    unsigned int lane_v = (lane_id&15)&1;

    /*
    *  each unrool 8, need load following from A. B is the same
    * 
    *      16*4     use float4 load, 64xfp32 a row
    *  +----------+
    *  |          | wave_0 (4x row float4)
    *  +----------+
    *  |          | wave_1
    *  +----------+
    *  |          | wave_2
    *  +----------+
    *  |          | wave_3
    *  +----------+
    *  |          | wave_0
    *  +----------+
    *  |          | wave_1
    *  +----------+
    *  |          | wave_2
    *  +----------+
    *  |          | wave_3
    *  +----------+
    *
    *  need 64*4*8*4 = 8192 from A
    */
    ptr_a += (m_idx<<(6+2)) + ((wave_id<<2)|lane_hi)*lda + (lane_lo<<4);
    ptr_b += (n_idx<<(6+2)) + ((wave_id<<2)|lane_hi)*ldb + (lane_lo<<4);
    ptr_c += ((n_idx<<6)+ ((wave_id<<2)|lane_hi))*ldc + (m_idx<<(6+2)) + (lane_lo<<4);

    float4 * smem_store = (float4 *)&smem[tid<<4];
    float4 * smem_load_a = (float4 *)&smem[(wave_id<<11)|(lane_u<<4)];
    float4 * smem_load_b = (float4 *)&smem[0x2000|(wave_id<<11)|(lane_w<<5)|(lane_v<<4)];

    p0[0] = *((const float4* __restrict__)ptr_a); ptr_a += bs_a;
    p0[1] = *((const float4* __restrict__)ptr_a); ptr_a += bs_a;
    q0[0] = *((const float4* __restrict__)ptr_b); ptr_b += bs_b;
    q0[1] = *((const float4* __restrict__)ptr_b); ptr_b += bs_b;
    CLR16x4(c)

    // unroll 8 for each wave, 4 wave totally unroll 32
    for(unsigned int ik=32; ik < k; ik += (8*4)){
        p1[0] = *((const float4* __restrict__)ptr_a); ptr_a += bs_a;
        p1[1] = *((const float4* __restrict__)ptr_a); ptr_a += bs_a;
        q1[0] = *((const float4* __restrict__)ptr_b); ptr_b += bs_b;
        q1[1] = *((const float4* __restrict__)ptr_b); ptr_b += bs_b;
        smem_store[0] = p0[0];
        smem_store[0x100] = p0[1];
        smem_store[0x200] = q0[0];
        smem_store[0x300] = q0[1];

        __syncthreads();
        // inner unroll 8
        #pragma unroll
        for(unsigned int i=0;i<8;i++) {
            a[0] = smem_load_a[(i<<4)];
            a[1] = smem_load_a[(i<<4)|8];
            b[0] = smem_load_b[(i<<4)];
            b[1] = smem_load_b[(i<<4)|8];
            S_FMA4x4((float*)&c[0] , (float*)&a[0], (float*)&b[0])
            S_FMA4x4((float*)&c[4] , (float*)&a[1], (float*)&b[0])
            S_FMA4x4((float*)&c[8] , (float*)&a[0], (float*)&b[1])
            S_FMA4x4((float*)&c[12], (float*)&a[1], (float*)&b[1])
        }
        __syncthreads();
        p0[0] = p1[0]; p0[1] = p1[1];
        q0[0] = q1[0]; q0[1] = q1[1];
    }
    smem_store[0] = p0[0];
    smem_store[0x100] = p0[1];
    smem_store[0x200] = q0[0];
    smem_store[0x300] = q0[1];

    __syncthreads();
    #pragma unroll
    for(unsigned int i=0;i<8;i++) {
        a[0] = smem_load_a[(i<<4)];
        a[1] = smem_load_a[(i<<4)|8];
        b[0] = smem_load_b[(i<<4)];
        b[1] = smem_load_b[(i<<4)|8];
        S_FMA4x4((float*)&c[0] , (float*)&a[0], (float*)&b[0])
        S_FMA4x4((float*)&c[4] , (float*)&a[1], (float*)&b[0])
        S_FMA4x4((float*)&c[8] , (float*)&a[0], (float*)&b[1])
        S_FMA4x4((float*)&c[12], (float*)&a[1], (float*)&b[1])
    }
    __syncthreads();

    {
        /* reduction
         * for a single wave, have 64x64 float -> 64*64*4 = 16384 byte
         * if 4 wave all store to LDS, need 64KB, this is too large.
         * we can split into 4 subpart of 64x64, each have 32x32 float
         * in this case, a wave need store 32x32x4 = 4096 B
         * 4 wave -> 16K, this is better, and even can use double buffer.
         *
         *     32 float,                w0 w1 w2 w3
         *  +-----------+              +--+--+--+--+
         *  |           |              |  |  |  |  |
         *  |           | wave_0       |  |  |  |  |
         *  |           |              |  |  |  |  |
         *  |           |              |  |  |  |  |
         *  +-----------+              +--+--+--+--+
         *  |           |              |  |  |  |  |
         *  |           | wave_1       |  |  |  |  |
         *  |           |              |  |  |  |  |           +-----------+
         *  |           |              |  |  |  |  |           |           |
         *  +-----------+        =>    +--+--+--+--+     =>    |           |
         *  |           |              |  |  |  |  |           |           |
         *  |           | wave_2       |  |  |  |  |           |           |
         *  |           |              |  |  |  |  |           +-----------+
         *  |           |              |  |  |  |  |
         *  +-----------+              +--+--+--+--+
         *  |           |              |  |  |  |  |
         *  |           | wave_3       |  |  |  |  |
         *  |           |              |  |  |  |  |
         *  |           |              |  |  |  |  |
         *  +-----------+              +--+--+--+--+
         */
        float4 * smem_store_c = (float4 *)&smem[(wave_id<<12)|(lane_w<<10)|(lane_v<<9)|(lane_u<<4)];
        float4 * smem_load_c = (float4 *)&smem[(wave_id<<5)|(lane_p<<4)|(lane_q<<7)];
        #pragma unroll
        for(int i=0;i<4;i++){
            smem_store_c[0]    = c[(i<<2)+0];
            smem_store_c[0x8]  = c[(i<<2)+1];
            smem_store_c[0x10] = c[(i<<2)+2];
            smem_store_c[0x18] = c[(i<<2)+3];

            __syncthreads();
            c[(i<<2)+0] = smem_load_c[0];
            c[(i<<2)+1] = smem_load_c[0x100];
            c[(i<<2)+2] = smem_load_c[0x200];
            c[(i<<2)+3] = smem_load_c[0x300];

            // reduction: sum up
            c[(i<<2)+0] += c[(i<<2)+1];
            c[(i<<2)+2] += c[(i<<2)+3];
            c[(i<<2)+0] += c[(i<<2)+2];
            __syncthreads();
        }
    }


    // after reduction, do coalesing and store
    {
        #pragma unroll
        for(int i=0;i<4;i++){ c[i<<2].x*=alpha; c[i<<2].y*=alpha; c[i<<2].z*=alpha; c[i<<2].w*=alpha;}

        float4 * smem_store_c = (float4*)&smem[(wave_id<<5)|(lane_p<<4)|(lane_q<<8)];
        float4 * smem_load_c = (float4*)&smem[tid<<4];

        smem_store_c[0]     = c[0];
        smem_store_c[0x8]   = c[4];
        smem_store_c[0x200] = c[8];
        smem_store_c[0x208] = c[12];
        __syncthreads();

        *((float4*)&ptr_c[0]) = smem_load_c[0];     ptr_c += bs_c;
        *((float4*)&ptr_c[0]) = smem_load_c[0x100]; ptr_c += bs_c;
        *((float4*)&ptr_c[0]) = smem_load_c[0x200]; ptr_c += bs_c;
        *((float4*)&ptr_c[0]) = smem_load_c[0x300];
    }
}