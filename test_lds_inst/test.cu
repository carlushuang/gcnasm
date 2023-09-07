#include <cassert>
#include <tuple>
#include <utility>
#include <iostream>
#include <array>
#include <initializer_list>
#include <cstdlib>
#include <cuda_runtime.h>


using index_t = int;

#define DEVICE_HOST __device__ __host__
#define GLOBAL __global__
#define DEVICE __device__

template<typename T, T v>
struct integral_constant {
    static constexpr T value = v;
    using value_type = T;
    using type = integral_constant; // using injected-class-name
    constexpr DEVICE_HOST operator value_type() const noexcept { return value; }
    constexpr DEVICE_HOST value_type operator()() const noexcept { return value; } // since c++14
};

template<index_t v>
using number = integral_constant<index_t, v>;

namespace impl {
    // TODO: check builtin __type_pack_element
    template<index_t I, typename... Ts>
    using at_index_t = __type_pack_element<I, Ts... >;
}

template<typename T, T... Ns>
struct sequence {
    using value_type = T;
    using type = sequence;
    static constexpr T n_element = sizeof...(Ns);

    template<index_t I>
    DEVICE_HOST static constexpr T get()
    {
        return impl::at_index_t<I, integral_constant<T, Ns>...>{}; //  __type_pack_element<I, integral_constant<T, Ns>...>::value;
    }

    DEVICE_HOST static constexpr void print()
    {
        ((printf("%d ", Ns)), ...);
        printf("[len:%d]\n", n_element);
    }
};

template<index_t N>
using make_integer_sequence = __make_integer_seq<sequence, index_t, N>; // TODO: check builtin __make_integer_seq

template<index_t... Ns>
using integer_sequence = sequence<index_t, Ns...>;

namespace detail {

struct swallow
{
    template <typename... Ts>
    DEVICE_HOST constexpr swallow(Ts&&...)
    {
    }
};

template <class>
struct constexpr_for_impl;

template <index_t... Is>
struct constexpr_for_impl<integer_sequence<Is...>>
{
    template <class F>
    DEVICE_HOST constexpr void operator()(F f) const
    {
        swallow{(f(number<Is>{}), 0)...};
    }
};

} // namespace detail

template <index_t begin, index_t end, index_t inc>
struct constexpr_for
{
    template <class F>
    DEVICE_HOST constexpr void operator()(F /*f*/) const
    {
        // TODO:
    }

};

template <index_t end>
struct constexpr_for<0, end, 1>
{
    template <class F>
    DEVICE_HOST constexpr void operator()(F f) const
    {
        detail::constexpr_for_impl<make_integer_sequence<end>>{}(f);
    }
};


template<index_t BLOCK_SIZE, index_t LDS_SIZE_BYTE>
GLOBAL void test_kernel(float * __restrict__ input, float * __restrict__ output)
{
    __shared__ char smem[LDS_SIZE_BYTE];

    float * s_ptr = reinterpret_cast<float*>(smem);

    //#pragma clang loop vectorize(disable)
    //for(auto i = 0; i < 4; i++) {
    constexpr_for<0, 4, 1>{}([&](auto iter){
        float4 data = reinterpret_cast<float4*>(input)[blockIdx.x * BLOCK_SIZE + threadIdx.x];
        auto i = iter.value;

        s_ptr[threadIdx.x + 0 * BLOCK_SIZE + i * 4 * BLOCK_SIZE] = data.x;
        s_ptr[threadIdx.x + 1 * BLOCK_SIZE + i * 4 * BLOCK_SIZE] = data.y;
        s_ptr[threadIdx.x + 2 * BLOCK_SIZE + i * 4 * BLOCK_SIZE] = data.z;
        s_ptr[threadIdx.x + 3 * BLOCK_SIZE + i * 4 * BLOCK_SIZE] = data.w;

        __syncthreads();

        data.x = s_ptr[threadIdx.x * 4 + 0 + i * 4 * BLOCK_SIZE];
        data.y = s_ptr[threadIdx.x * 4 + 1 + i * 4 * BLOCK_SIZE];
        data.z = s_ptr[threadIdx.x * 4 + 2 + i * 4 * BLOCK_SIZE];
        data.w = s_ptr[threadIdx.x * 4 + 3 + i * 4 * BLOCK_SIZE];

        *reinterpret_cast<float4*>(output) = data;
        input += 4 * BLOCK_SIZE;
        output += 4 * BLOCK_SIZE;
        __syncthreads();
    });
}



int main(int argc, char ** argv) {

    constexpr index_t BLOCK_SIZE = 256;
    constexpr index_t LDS_SIZE_BYTE = 4 * BLOCK_SIZE * sizeof(float);

    float * offset_0_host;
    float * offset_0_device;
    float * offset_1_host;
    float * offset_1_device;

    cudaMalloc(&offset_0_device, sizeof(float) * BLOCK_SIZE);
    cudaMalloc(&offset_1_device, sizeof(float) * BLOCK_SIZE);
    offset_0_host = (float*)malloc(sizeof(float) * BLOCK_SIZE);
    offset_1_host = (float*)malloc(sizeof(float) * BLOCK_SIZE);

    test_kernel<BLOCK_SIZE, LDS_SIZE_BYTE><<<dim3(1), dim3(BLOCK_SIZE)>>>(
        offset_0_device, offset_1_device);

    cudaMemcpy(offset_0_host, offset_0_device, sizeof(index_t) * BLOCK_SIZE, cudaMemcpyDeviceToHost);
    cudaMemcpy(offset_1_host, offset_1_device, sizeof(index_t) * BLOCK_SIZE, cudaMemcpyDeviceToHost);

    cudaFree(offset_0_device);
    free(offset_0_host);
    cudaFree(offset_1_device);
    free(offset_1_host);
}
