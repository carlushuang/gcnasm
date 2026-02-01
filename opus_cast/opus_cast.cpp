#include "opus/opus.hpp"
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>

#define HIP_CHECK(err)                                          \
    do {                                                        \
        hipError_t status = err;                                \
        if (status != hipSuccess) {                             \
            fprintf(stderr, "HIP Error: %s at line %d\n",       \
                    hipGetErrorString(status), __LINE__);       \
            exit(EXIT_FAILURE);                                 \
        }                                                       \
    } while (0)

// ignore check N % VEC_SIZE == 0
template<typename DType, typename SType, int VEC_SIZE>
__global__ void cast_kernel(DType* __restrict__ dst,  const SType* __restrict__ src,  
                    size_t N) {
    using opus::operator""_I;
    using D_VEC_T = opus::vector_t<DType, VEC_SIZE>;
    using S_VEC_T = opus::vector_t<SType, VEC_SIZE>;

    D_VEC_T * dst_vec = reinterpret_cast<D_VEC_T*>(dst);
    const S_VEC_T * src_vec = reinterpret_cast<const S_VEC_T*>(src);
    size_t global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (size_t i = global_idx; i < N / VEC_SIZE; i += gridDim.x * blockDim.x) {
        dst_vec[i] = __builtin_bit_cast(D_VEC_T, opus::cast<DType>(src_vec[i]));
    }
}

template<typename DType, typename SType, int VEC_SIZE>
__global__ void cast_kernel_with_f4(DType* __restrict__ dst,  const SType* __restrict__ src,  
                    size_t N) {
    using opus::operator""_I;
    using D_VEC_T = opus::array<DType, VEC_SIZE / opus::num_packs_v<DType>>;
    using S_VEC_T = opus::array<SType, VEC_SIZE / opus::num_packs_v<SType>>;

    // opus::bool_constant<std::is_same_v<opus::get_value_t<S_VEC_T>, opus::fp4_t>>{}.zzz();
    // opus::tuple_array<opus::fp4_t, 4>{}.uuu();

    D_VEC_T * dst_vec = reinterpret_cast<D_VEC_T*>(dst);
    const S_VEC_T * src_vec = reinterpret_cast<const S_VEC_T*>(src);
    size_t global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (size_t i = global_idx; i < N / VEC_SIZE; i += gridDim.x * blockDim.x) {
        dst_vec[i] = __builtin_bit_cast(D_VEC_T, opus::cast<DType>(src_vec[i], 0));
    }
}


template<typename DType, typename SType, int VEC_SIZE>
void cast_kernel_launch(std::vector<DType>& h_dst, const std::vector<SType>& h_src, size_t N) {
    SType* d_src = nullptr;
    DType* d_dst = nullptr;
    HIP_CHECK(hipMalloc(&d_src, N * sizeof(SType)));
    HIP_CHECK(hipMalloc(&d_dst, N * sizeof(DType)));

    HIP_CHECK(hipMemcpy(d_src, h_src.data(), N * sizeof(SType), hipMemcpyHostToDevice));

    dim3 block_dim(256);
    dim3 grid_dim((N / VEC_SIZE + block_dim.x - 1) / block_dim.x);

    if constexpr (std::is_same_v<DType, opus::fp4_t> || std::is_same_v<SType, opus::fp4_t>)
        cast_kernel_with_f4<DType, SType, VEC_SIZE><<<grid_dim, block_dim>>>(d_dst, d_src, N);
    else
        cast_kernel<DType, SType, VEC_SIZE><<<grid_dim, block_dim>>>(d_dst, d_src, N);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMemcpy(h_dst.data(), d_dst, N * sizeof(DType), hipMemcpyDeviceToHost));

    HIP_CHECK(hipFree(d_src));
    HIP_CHECK(hipFree(d_dst));
}

#if 1
template<int VEC_SIZE>
void cast_fwd_bwd_bf16_fp32(const size_t N) {
    using DType = opus::bf16_t;
    using SType = opus::fp32_t;

    std::vector<SType> h_src(N);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-2000, 2000);

    for (size_t i = 0; i < N; ++i) { h_src[i] = dist(gen); }

    std::vector<DType> h_dst(N); std::vector<SType> h_bwd(N);
    cast_kernel_launch<DType, SType, VEC_SIZE>(h_dst, h_src, N);
    cast_kernel_launch<SType, DType, VEC_SIZE>(h_bwd, h_dst, N);

    for (size_t i = 0; i < 5; ++i) {
        size_t idx = rand() % N;
        float bf16_back_to_float = __builtin_bit_cast(float,  __builtin_bit_cast(uint16_t,  h_dst[idx]) << 16);
        printf("[FP32 BF16| v:%2d, %5zd] Float: %.6f -> BF16(float): %.6f(%x) | error: %.6f | -> Float(back): %.6f \n",
               VEC_SIZE, idx, h_src[idx], bf16_back_to_float, __builtin_bit_cast(uint16_t,  h_dst[idx]), fabs(h_src[idx] - bf16_back_to_float), h_bwd[idx]);
    }
}
#endif

template<int VEC_SIZE>
void cast_fwd_bwd_fp8_fp32(const size_t N) {
    using DType = opus::fp8_t;
    using SType = opus::fp32_t;

    std::vector<SType> h_src(N);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-224, 224);

    for (size_t i = 0; i < N; ++i) { h_src[i] = dist(gen); }

    std::vector<DType> h_dst(N); std::vector<SType> h_bwd(N);

    cast_kernel_launch<DType, SType, VEC_SIZE>(h_dst, h_src, N);
    cast_kernel_launch<SType, DType, VEC_SIZE>(h_bwd, h_dst, N);

    for (size_t i = 0; i < 5; ++i) {
        size_t idx = rand() % N;
        // float bf16_back_to_float = __builtin_bit_cast(float,  __builtin_bit_cast(uint16_t,  h_dst[idx]) << 16);
        printf("[FP32  FP8| v:%2d, %5zd] Float: %.6f -> FP8(float): %x(%x) |  | -> Float(back): %.6f \n",
                VEC_SIZE, idx, h_src[idx], 0, __builtin_bit_cast(uint8_t,  h_dst[idx]), h_bwd[idx]);
    }
}

// for MI355 only
template<int VEC_SIZE>
void cast_fwd_bwd_fp4_fp32(const size_t N) {
    // FP4 is +- (0.0, 0.5, 1.0, 1.5, 2, 3, 4, 6)
    using DType = opus::fp4_t;
    using SType = opus::fp32_t;

    std::vector<SType> h_src(N);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-7, 7);

    for (size_t i = 0; i < N; ++i) { h_src[i] = dist(gen); } // -7~7

    std::vector<DType> h_dst(N); std::vector<SType> h_bwd(N);

    cast_kernel_launch<DType, SType, VEC_SIZE>(h_dst, h_src, N);
    cast_kernel_launch<SType, DType, VEC_SIZE>(h_bwd, h_dst, N);

    for (size_t i = 0; i < 5; ++i) {
        size_t idx = rand() % N;
        // float bf16_back_to_float = __builtin_bit_cast(float,  __builtin_bit_cast(uint16_t,  h_dst[idx]) << 16);
        printf("[FP32  FP4| v:%2d, %5zd] Float: %.6f -> FP4(float): %x(%x) |  | -> Float(back): %.6f \n",
                VEC_SIZE, idx, h_src[idx], 0, __builtin_bit_cast(uint8_t,  h_dst[idx]), h_bwd[idx]);
    }
}

int main() {
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    printf("%s\n", prop.gcnArchName);
    bool is_gfx950 = std::string(prop.gcnArchName).find("gfx950") != std::string::npos;

    cast_fwd_bwd_bf16_fp32<8>(2048 * 4);
    cast_fwd_bwd_fp8_fp32<16>(2048 * 4);
    cast_fwd_bwd_fp8_fp32<8>(2048 * 4);
    cast_fwd_bwd_fp8_fp32<4>(2048 * 4);
    cast_fwd_bwd_fp8_fp32<2>(2048 * 4);
    if(is_gfx950) {
        cast_fwd_bwd_fp4_fp32<16>(2048 * 4);
    }

    return EXIT_SUCCESS;
}
