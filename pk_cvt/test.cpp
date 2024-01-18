#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <random>
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <numeric>

#define INLINE_ASM 0

using index_t = int;

using fp32 = float;
using fp32x16 = fp32 __attribute__((ext_vector_type(16)));
using fp32x8 = fp32 __attribute__((ext_vector_type(8)));
using fp32x4 = fp32 __attribute__((ext_vector_type(4)));
using fp32x2 = fp32 __attribute__((ext_vector_type(2)));
using fp32x1 = fp32 __attribute__((ext_vector_type(1)));

using dword_t = index_t;
using dwordx4_t = dword_t __attribute__((ext_vector_type(4)));

using fp8_t = _BitInt(8);

#define BLOCK_SIZE 256

#define HIP_CALL(call) do{  \
    hipError_t err = call;  \
    if(err != hipSuccess){  \
        printf("[hiperror](%d) fail to call %s",(int)err,#call);    \
        exit(0);            \
    }                       \
} while(0)



#define BUFFER_LOAD_DWORD3 0x00020000
struct buffer_resource {
    const void * ptr;
    uint32_t range;
    uint32_t config;
};
__device__ dwordx4_t make_buffer_resource(const void * ptr)
{
    buffer_resource res {ptr, 0xffffffff, BUFFER_LOAD_DWORD3};
    return __builtin_bit_cast(dwordx4_t, res);
}

__device__ void
llvm_amdgcn_raw_buffer_load_lds(dwordx4_t rsrc,
                                __attribute__((address_space(3))) uint32_t* lds_ptr,
                                index_t size,
                                index_t voffset,
                                index_t soffset,
                                index_t offset,
                                index_t aux) __asm("llvm.amdgcn.raw.buffer.load.lds");


namespace impl {
    template<int bits> struct vector_element_type;
    template<> struct vector_element_type<8>  { using type = uint8_t; };
    template<> struct vector_element_type<16> { using type = uint16_t; };
    template<> struct vector_element_type<32> { using type = uint32_t; };
}
template<typename T>
using vec_elem_t = typename impl::vector_element_type<sizeof(T) * 8>::type;

template<typename T, index_t N>
struct static_buffer {
    T data[N];
    __host__ __device__ auto & get(index_t i) {return data[i]; }
    __host__ __device__ const auto & get(index_t i) const {return data[i]; }
    __host__ __device__ auto & at(index_t i) {return data[i]; }
    __host__ __device__ const auto & at(index_t i) const {return data[i]; }
    __host__ __device__ auto & get() { return reinterpret_cast<vec_elem_t<T> __attribute__((ext_vector_type(N)))&>(data);}
};

template<typename T, index_t N>
struct vector_type {
    using type = T __attribute__((ext_vector_type(N)));
    type data;
    static constexpr index_t size = N;
    using data_type = T;

    template<typename Tx>
    __device__ __host__ auto & as(){
        static_assert(sizeof(T) * N % sizeof(Tx) == 0);
        constexpr int vx = sizeof(T) * N / sizeof(Tx);
        return reinterpret_cast<static_buffer<Tx, vx>&>(data);
    }
    template<typename Tx>
    __device__ __host__ const auto & as() const {
        static_assert(sizeof(T) * N % sizeof(Tx) == 0);
        constexpr int vx = sizeof(T) * N / sizeof(Tx);
        return reinterpret_cast<const static_buffer<Tx, vx>&>(data);
    }

    __device__ __host__ auto & to(){
        return reinterpret_cast<static_buffer<T, N>&>(data);
    }

    __device__ __host__ const auto & to() const {
        return reinterpret_cast<const static_buffer<T, N>&>(data);
    }

    template<typename Tx>
    __device__ __host__ auto & at(index_t i){
        return as<Tx>().get(i);
    }

    template<typename Tx>
    __device__ __host__ const auto & at(index_t i) const {
        return as<Tx>().get(i);
    }

    __device__ __host__ auto &at(index_t i) {
        return to().get(i);
    }
    __device__ __host__ const auto &at(index_t i) const {
        return to().get(i);
    }

    __device__ __host__ auto & get() {
        return data;
    }
    __device__ __host__ const auto & get() const {
        return data;
    }
};

template<index_t bytes>
struct gld;

__device__ fp32 llvm_amdgcn_raw_buffer_load_fp32(dwordx4_t srsrc, index_t voffset, index_t soffset, index_t glc_slc)
                            __asm("llvm.amdgcn.raw.buffer.load.f32");
__device__ fp32x2 llvm_amdgcn_raw_buffer_load_fp32x2(dwordx4_t srsrc, index_t voffset, index_t soffset, index_t glc_slc)
                            __asm("llvm.amdgcn.raw.buffer.load.v2f32");                            
__device__ fp32x4 llvm_amdgcn_raw_buffer_load_fp32x4(dwordx4_t srsrc, index_t voffset, index_t soffset, index_t glc_slc)
                            __asm("llvm.amdgcn.raw.buffer.load.v4f32");

template<> struct gld<16>{
    template<typename T>
    __device__ void operator()(T & value, dwordx4_t res/*buffer resource*/, index_t v_offset, index_t s_offset, index_t i_offset/*max 0xFFF*/, index_t /*flag*/ = 0){
        static_assert(sizeof(T) == 16);
#if INLINE_ASM
        using v_type = float __attribute__((ext_vector_type(4)));
        asm volatile("buffer_load_dwordx4 %0, %1, %2, %3 offen offset:%4"
            : "+v"(reinterpret_cast<v_type&>(value)) : "v"(v_offset), "s"(res), "s"(s_offset), "n"(i_offset) : "memory");
#else
        auto tmp = llvm_amdgcn_raw_buffer_load_fp32x4(res, v_offset, s_offset + i_offset, 0);
        value =  __builtin_bit_cast(T, tmp);
#endif
    }
};

template<> struct gld<8>{
    template<typename T>
    __device__ void operator()(T & value, dwordx4_t res/*buffer resource*/, index_t v_offset, index_t s_offset, index_t i_offset/*max 0xFFF*/, index_t /*flag*/ = 0){
        static_assert(sizeof(T) == 8);
#if INLINE_ASM
        asm volatile("buffer_load_dwordx2 %0, %1, %2, %3 offen offset:%4"
            : "+v"(value.get()) : "v"(v_offset), "s"(res), "s"(s_offset), "n"(i_offset) : "memory");
#else
        auto tmp = llvm_amdgcn_raw_buffer_load_fp32x2(res, v_offset, s_offset + i_offset, 0);
        value =  __builtin_bit_cast(T, tmp);
#endif
    }
};

template<> struct gld<4>{
    template<typename T>
    __device__ void operator()(T & value, dwordx4_t res/*buffer resource*/, index_t v_offset, index_t s_offset, index_t i_offset/*max 0xFFF*/, index_t /*flag*/ = 0){
        static_assert(sizeof(T) == 4);
#if INLINE_ASM
        asm volatile("buffer_load_dword %0, %1, %2, %3 offen offset:%4"
            : "+v"(value.get()) : "v"(v_offset), "s"(res), "s"(s_offset), "n"(i_offset) : "memory");
#else
        auto tmp = llvm_amdgcn_raw_buffer_load_fp32(res, v_offset, s_offset + i_offset, 0);
        value =  __builtin_bit_cast(T, tmp);
#endif
    }
};

__device__ void gld_fence(index_t cnt)
{
#if INLINE_ASM
    asm volatile("s_waitcnt vmcnt(%0)" : : "n" (cnt) : "memory");
#else
    (void) cnt;
#endif
}

template<typename T, index_t N>
__device__ void gld_fence(vector_type<T, N> & /*buf*/, index_t cnt)
{
#if INLINE_ASM
    asm volatile("s_waitcnt vmcnt(%0)" : : "n" (cnt) : "memory");
    // constexpr index_t total = sizeof(T) * N / sizeof(float);
    // for(auto i = 0; i < total; i++) {
    //     asm volatile("" :"+v"(buf.template as<float>().get(i)) : :);
    // }
#else
    (void) cnt;
#endif
}

template<typename T, index_t N>
__device__ void clear_buf(vector_type<T, N> & buf)
{
#if 0
    for(auto i = 0; i < N; i++)
        asm volatile("v_mov_b32 %0, 0" : "+v"(buf.at(i)) :  : "memory");
#else
    for(auto i = 0; i < N; i++)
        buf.at(i) = .0f;
#endif
}

template<typename F8X4Type, typename F32X4Type>
__device__ void pk_cvt_fp8x4_fp32x4_rne(F8X4Type & dst, F32X4Type & src)
{
    using src_type = float __attribute__((ext_vector_type(4)));
    static_assert(sizeof(src_type) == sizeof(F32X4Type));
    src_type & s = reinterpret_cast<src_type&>(src);
#if 0
    uint32_t v01 = 0;
    v01       = __builtin_amdgcn_cvt_pk_fp8_f32(s.x, s.y, v01, false); // false -> WORD0

    uint32_t v23 = 0;
    v23       = __builtin_amdgcn_cvt_pk_fp8_f32(s.z, s.w, v23, false); // false -> WORD0

    union {
        F8X4Type d;
        struct {
            uint16_t lo;
            uint16_t hi;
        };
    } pool;

    pool.lo = v01&0xffff;
    pool.hi = v23&0xffff;
    dst = pool.d;
#else
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wuninitialized"
    uint32_t  dummy_old;
    uint32_t x = __builtin_amdgcn_cvt_pk_fp8_f32(s.x, s.y, dummy_old, false); // false -> WORD0
    uint32_t y = __builtin_amdgcn_cvt_pk_fp8_f32(s.z, s.w, dummy_old, false); // false -> WORD0
#pragma clang diagnostic pop
    constexpr int32_t m0 = 0x05040100;

    uint32_t d = __builtin_amdgcn_perm(y, x, m0);

    dst = reinterpret_cast<F8X4Type&>(d);
#endif
}

__global__ void
cvt_kernel(const void * ptr_src, void * ptr_dst, uint32_t total_pixels)
{
    using src_buf_type = static_buffer<fp8_t, 16>;
    using dst_buf_type = static_buffer<fp8_t, 4>;
    //using src_buf_type = vector_type<fp8_t, 16>::type;
    //using dst_buf_type = vector_type<fp8_t, 4>::type;

    int idx = (threadIdx.x + blockIdx.x * 256);
    int stride = blockDim.x * 256;

    for(int i = idx; i < total_pixels; i += stride) {
        src_buf_type src;
        gld<sizeof(src_buf_type)>{}(src, make_buffer_resource(ptr_src), i * sizeof(src_buf_type), 0, 0);
        gld_fence(0);
        dst_buf_type dst;
        pk_cvt_fp8x4_fp32x4_rne(dst, src);
        *(reinterpret_cast<dst_buf_type*>(ptr_dst) + i) = dst;
    }
}


template <typename T>
struct NumericUtils
{
};

template <>
struct NumericUtils<float>
{
    static constexpr int exp            = 8;
    static constexpr int mant           = 23;
    static constexpr int bias           = 127;
    static constexpr uint32_t nan_mask  = 0x7F800000;
    static constexpr uint32_t head_mask = 0xFF800000;
    static constexpr uint32_t mant_mask = 0x7FFFFF;
    static constexpr uint32_t exp_mask  = 0xFF;
    static constexpr uint32_t Inf       = 0x7F800000;
    static constexpr uint32_t NegInf    = 0xFF800000;
    static constexpr uint32_t NaN       = 0x7F800001;
    static constexpr uint32_t Neg0      = 0x80000000;
    using bitwise_type                  = uint32_t;
};
#if 0
template <>
struct NumericUtils<half_t>
{
    static constexpr int exp            = 5;
    static constexpr int mant           = 10;
    static constexpr int bias           = 15;
    static constexpr uint16_t nan_mask  = 0x7C00;
    static constexpr uint16_t head_mask = 0xFC00;
    static constexpr uint16_t mant_mask = 0x3FF;
    static constexpr uint16_t exp_mask  = 0x1F;
    static constexpr uint32_t Inf       = 0x7C00;
    static constexpr uint32_t NegInf    = 0xFC00;
    static constexpr uint32_t NaN       = 0x7C01;
    static constexpr uint32_t Neg0      = 0x8000;
    using bitwise_type                  = uint16_t;
};
#endif
template <>
struct NumericUtils<fp8_t>
{
    static constexpr int exp  = 4;
    static constexpr int mant = 3;
    static constexpr int bias = 8; // negative zero nan mode
    // static constexpr int bias = 7; // ieee mode
};
#if 0
template <>
struct NumericUtils<bf8_t>
{
    static constexpr int exp  = 5;
    static constexpr int mant = 2;
    static constexpr int bias = 16; // negative zero nan mode
    // static constexpr int bias = 15; // ieee mode
};
#endif
__host__ inline int clz(uint32_t x) { return __builtin_clz(x); }
__device__ inline int clz(uint32_t x) { return __clz(x); }
template <typename X, typename Y, bool negative_zero_nan>
__host__ __device__ Y run_cast_from_f8(X x)
{
    // fp8/bf8 exponent/mantissa layout
    constexpr int in_exp  = NumericUtils<X>::exp;
    constexpr int in_mant = NumericUtils<X>::mant;

    // resulting type exponent/mantissa layout
    constexpr int out_exp  = NumericUtils<Y>::exp;
    constexpr int out_mant = NumericUtils<Y>::mant;

    // prepare the codes
    constexpr X nan_code = 0x80;
    Y Inf, NegInf, NaN, Neg0;
    using T_bitwise = typename NumericUtils<Y>::bitwise_type;

    constexpr T_bitwise Inf_bitwise    = NumericUtils<Y>::Inf;
    constexpr T_bitwise NegInf_bitwise = NumericUtils<Y>::NegInf;
    constexpr T_bitwise NaN_bitwise    = NumericUtils<Y>::NaN;
    constexpr T_bitwise Neg0_bitwise   = NumericUtils<Y>::Neg0;

    Inf    = *(reinterpret_cast<const Y*>(&Inf_bitwise));
    NegInf = *(reinterpret_cast<const Y*>(&NegInf_bitwise));
    NaN    = *(reinterpret_cast<const Y*>(&NaN_bitwise));
    Neg0   = *(reinterpret_cast<const Y*>(&Neg0_bitwise));

    // check if x is 0.0
    if(x == 0)
        return static_cast<Y>(0);

    // unpack the input
    uint32_t sign     = x >> (in_exp + in_mant);
    uint32_t mantissa = x & ((1 << in_mant) - 1);
    int exponent      = (x & 0x7F) >> in_mant;

    constexpr int exp_low_cutoff =
        (1 << (out_exp - 1)) - (1 << (in_exp - 1)) + 1 - (negative_zero_nan ? 1 : 0);
    T_bitwise retval;

    if constexpr(negative_zero_nan)
    {
        if(x == nan_code)
            return NaN;
    }
    else
    {
        if(x == nan_code)
            return Neg0;
        if(exponent == ((1 << in_exp) - 1))
            return (mantissa == 0) ? (sign ? NegInf : Inf) : NaN;
    }

    if((NumericUtils<Y>::mant == 10) && (NumericUtils<X>::mant == 2) && !negative_zero_nan)
    {
        retval = x;
        retval <<= 8;
        return *(reinterpret_cast<const Y*>(&retval));
    }

    // subnormal input
    if(exponent == 0)
    {
        // guaranteed mantissa!=0 since cases 0x0 and 0x80 are handled above
        int sh = 1 + clz(mantissa) - (32 - in_mant);
        mantissa <<= sh;
        exponent += 1 - sh;
        mantissa &= ((1 << in_mant) - 1);
    }
    exponent += exp_low_cutoff - 1;
    mantissa <<= out_mant - in_mant;

    // subnormal output (occurs when T=half, we=5, negative_zero_nan=true)
    if(exponent <= 0)
    {
        mantissa |= 1 << out_mant;
        mantissa >>= 1 - exponent;
        exponent = 0;
    }

    retval = (sign << (out_exp + out_mant)) | (exponent << out_mant) | mantissa;
    return *(reinterpret_cast<const Y*>(&retval));
}

int main(int argc, char ** argv)
{
    int pixels = 256;

    if(argc >= 2) {
        pixels = atoi(argv[1]);
    }


    float *host_src;
    fp8_t *host_dst;
    void * device_src, * device_dst;

    //fp32 on host
    host_src = (float*)malloc(pixels*sizeof(float));
    host_dst = (fp8_t*)malloc(pixels*sizeof(fp8_t));

    //convert fp32 a and b into fp16 on host
    for(auto i = 0; i < pixels; i++) {
        host_src[i] = static_cast<float>(i - (pixels / 2));
    }

    HIP_CALL(hipMalloc(&device_src, pixels * sizeof(float)));
    HIP_CALL(hipMalloc(&device_dst, pixels * sizeof(fp8_t)));

    HIP_CALL(hipMemcpy(device_src, host_src, pixels*sizeof(float), hipMemcpyHostToDevice));

    cvt_kernel<<<(pixels + 256 - 1) / 256, 256>>>(device_src, device_dst, pixels);

    HIP_CALL(hipMemcpy(host_dst, device_dst, pixels*sizeof(fp8_t), hipMemcpyDeviceToHost));

    for(auto i = 0 ;i < pixels; i++) {
        uint8_t x = reinterpret_cast<uint8_t&>(host_dst[i]);
        float f =  run_cast_from_f8<fp8_t, float, true>(host_dst[i]);
        printf("[%3d]%f -> %f(%x)\n", i, host_src[i], f, x);
    }
}
