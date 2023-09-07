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

template<typename T, T... Ns>
struct sequence;

template<index_t...Ns>
using integer_sequence = sequence<index_t, Ns...>;

#if !defined(__CUDACC__)
template<index_t N>
using make_integer_sequence = __make_integer_seq<sequence, index_t, N>; // TODO: check builtin __make_integer_seq
#else
namespace impl
{
    // we have four instantiations of generate_sequence<>, independent of T or N.
    // V is the current bit, E is the end marker - if true, this is the last step.
    template< bool V, bool E >
    struct generate_sequence;

    // last step: generate final integer sequence
    template<>
    struct generate_sequence< false, true >
    {
        template< typename T, T M, T N, index_t S, T... Ns >
        using f = sequence< T, Ns... >;
    };

    template<>
    struct generate_sequence< true, true >
    {
        template< typename T, T M, T N, index_t S, T... Ns >
        using f = sequence< T, Ns..., S >;
    };

    // intermediate step: double existing values, append one more if V is set.
    template<>
    struct generate_sequence< false, false >
    {
        template< typename T, T M, T N, index_t S, T... Ns >
        using f = typename generate_sequence< ( N & ( M / 2 ) ) != 0, ( M / 2 ) == 0 >::template f< T, M / 2, N, 2 * S, Ns..., ( Ns + S )... >;
    };

    template<>
    struct generate_sequence< true, false >
    {
        template< typename T, T M, T N, index_t S, T... Ns >
        using f = typename generate_sequence< ( N & ( M / 2 ) ) != 0, ( M / 2 ) == 0 >::template f< T, M / 2, N, 2 * S + 1, Ns..., ( Ns + S )..., 2 * S >;
    };

    // the final sequence per T/N should be memoized, it will probably be used multiple times.
    // also checks the limit and starts the above generator properly.
    template< typename T, T N >
    struct memoize_sequence
    {
        static_assert( N < T( 1 << 20 ), "N too large" );
        using type = typename generate_sequence< false, false >::template f< T, ( N < T( 1 << 1 ) ) ? T( 1 << 1 ) : ( N < T( 1 << 2 ) ) ? T( 1 << 2 ) : ( N < T( 1 << 3 ) ) ? T( 1 << 3 ) : ( N < T( 1 << 4 ) ) ? T( 1 << 4 ) : ( N < T( 1 << 5 ) ) ? T( 1 << 5 ) : ( N < T( 1 << 6 ) ) ? T( 1 << 6 ) : ( N < T( 1 << 7 ) ) ? T( 1 << 7 ) : ( N < T( 1 << 8 ) ) ? T( 1 << 8 ) : ( N < T( 1 << 9 ) ) ? T( 1 << 9 ) : ( N < T( 1 << 10 ) ) ? T( 1 << 10 ) : T( 1 << 20 ), N, 0 >;
    };
}  // namespace impl
template<index_t N >
using make_integer_sequence = typename impl::memoize_sequence<index_t, N >::type;
#endif


namespace impl {
#if !defined(__CUDACC__)
    // TODO: check builtin __type_pack_element
    template<index_t I, typename... Ts>
    using at_index_t = __type_pack_element<I, Ts... >;
#else
      namespace impl
      {
         template< std::size_t, typename T >
         struct indexed
         {
            using type = T;
         };

         template< typename, typename... Ts >
         struct indexer;

         template< std::size_t... Is, typename... Ts >
         struct indexer< integer_sequence< Is... >, Ts... >
            : indexed< Is, Ts >...
         {
         };

#if( __cplusplus >= 201402L )
         template< typename... Ts >
         constexpr impl::indexer< make_integer_sequence< sizeof...(Ts) >, Ts... > index_value{};
#endif

         template< std::size_t I, typename T >
         indexed< I, T > select( const indexed< I, T >& );

      }  // namespace impl

#if( __cplusplus >= 201402L )

      template< std::size_t I, typename... Ts >
      using at_index = decltype( impl::select< I >( impl::index_value< Ts... > ) );

#else

      template< std::size_t I, typename... Ts >
      using at_index = decltype( impl::select< I >( impl::indexer< make_integer_sequence< Ts... >, Ts... >() ) );

#endif

#ifndef _MSC_VER
      template< std::size_t I, typename... Ts >
      using at_index_t = typename at_index< I, Ts... >::type;

#else

      namespace impl
      {
         template< typename T >
         struct get_type
         {
            using type = typename T::type;
         };

      }  // namespace impl

      template< std::size_t I, typename... Ts >
      using at_index_t = typename impl::get_type< at_index< I, Ts... > >::type;
#endif
#endif
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

#if !defined(__CUDACC__)
typedef float f32x4 __attribute__((ext_vector_type(4)));
#else
typedef float4 f32x4;
#endif

template<index_t BLOCK_SIZE, index_t LDS_SIZE_BYTE>
GLOBAL void test_kernel(float * __restrict__ input, float * __restrict__ output)
{
    __shared__ char smem[LDS_SIZE_BYTE];

    float * s_ptr = reinterpret_cast<float*>(smem);

    //#pragma clang loop vectorize(disable)
    //for(auto i = 0; i < 4; i++) {
    constexpr_for<0, 4, 1>{}([&](auto iter){
        f32x4 data = (reinterpret_cast<f32x4*>(input))[blockIdx.x * BLOCK_SIZE + threadIdx.x];
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

        (reinterpret_cast<f32x4*>(output))[blockIdx.x * BLOCK_SIZE + threadIdx.x] = data;
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
