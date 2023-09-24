#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

using f32 = float;
using f16 = _Float16;

using u8 = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;

using index_t = u32;

#define DEVICE_HOST __device__ __host__
#define GLOBAL __global__
#define DEVICE __device__

#if 1
typedef f32 f32x16 __attribute__((ext_vector_type(16)));
typedef f32 f32x8 __attribute__((ext_vector_type(8)));
typedef f32 f32x4 __attribute__((ext_vector_type(4)));
typedef f32 f32x2 __attribute__((ext_vector_type(2)));
typedef f32 f32x1 __attribute__((ext_vector_type(1)));

typedef f16 f16x16 __attribute__((ext_vector_type(16)));
typedef f16 f16x8 __attribute__((ext_vector_type(8)));
typedef f16 f16x4 __attribute__((ext_vector_type(4)));
typedef f16 f16x2 __attribute__((ext_vector_type(2)));
typedef f16 f16x1 __attribute__((ext_vector_type(1)));

typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));
typedef uint32_t u32x3 __attribute__((ext_vector_type(2)));
typedef uint32_t u32x2 __attribute__((ext_vector_type(2)));
typedef uint32_t u32x1 __attribute__((ext_vector_type(1)));

typedef u8 u8x16 __attribute__((ext_vector_type(16)));  // 4 dword
typedef u8 u8x12 __attribute__((ext_vector_type(12)));  // 3 dword
typedef u8 u8x8  __attribute__((ext_vector_type(8)));   // 2 dword
typedef u8 u8x4  __attribute__((ext_vector_type(4)));   // 1 dword
typedef u8 u8x2  __attribute__((ext_vector_type(2)));   // half
typedef u8 u8x1  __attribute__((ext_vector_type(1)));   // byte

// prefer use __builtin_bit_cast to cast to & from these types
typedef u32x4 dwordx4_t;
typedef u32x3 dwordx3_t;
typedef u32x2 dwordx2_t;
typedef u32   dword_t;

typedef u16 hdword_t;
typedef u8  byte_t;

#endif

////////////////////////////////////////////////////
template<typename T, T v>
struct integral_constant {
    static constexpr T value = v;
    using value_type = T;
    using type = integral_constant;
    constexpr DEVICE_HOST operator value_type() const noexcept { return value; }
    constexpr DEVICE_HOST value_type operator()() const noexcept { return value; }
};

template<index_t v>
using number = integral_constant<index_t, v>;

static constexpr auto I0 = number<0>{};
static constexpr auto I1 = number<1>{};
static constexpr auto I2 = number<2>{};
static constexpr auto I3 = number<3>{};
static constexpr auto I4 = number<4>{};
static constexpr auto I5 = number<5>{};
static constexpr auto I6 = number<6>{};
static constexpr auto I7 = number<7>{};

template<bool v>
using bool_const = integral_constant<bool, v>;

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
        return impl::at_index_t<I, integral_constant<T, Ns>...>{};
    }
    DEVICE_HOST static constexpr T back()
    {
        return get<n_element - 1>();
    }
    DEVICE_HOST static constexpr T front()
    {
        return get<0>();
    }
};

template<index_t...Ns>
using seq = sequence<index_t, Ns...>;

template<index_t N>
using make_integer_sequence = __make_integer_seq<sequence, index_t, N>;

namespace impl {

template<index_t idx, typename T, bool is_empty = std::is_empty_v<T>>
struct tuple_element {};

template<index_t idx, typename T>
struct tuple_element<idx, T, true> {
    DEVICE_HOST constexpr tuple_element() {}
    DEVICE_HOST constexpr tuple_element(const T&) {}
};

template<index_t idx, typename T>
struct tuple_element<idx, T, false> {
    DEVICE_HOST constexpr tuple_element() {}
    DEVICE_HOST constexpr tuple_element(const T& e) : element(e) {}
    T element;
};

template <std::size_t I, class T>
DEVICE_HOST constexpr T getv(tuple_element<I, T, true> const& x)
{ return {}; }

template <std::size_t I, class T>
DEVICE_HOST constexpr T const& getv(tuple_element<I, T, false> const& x)
{ return x.element; }

template <std::size_t I, class T>
DEVICE_HOST constexpr T& getv(tuple_element<I, T, false>& x)
{ return x.element; }

template <std::size_t I, class T>
DEVICE_HOST constexpr T&& getv(tuple_element<I, T, false>&& x)
{ return static_cast<T&&>(x.element); }

template<typename index_seq, typename... T>
struct tuple_base;

template<index_t... I, typename... T>
struct tuple_base<seq<I...>, T...> : public tuple_element<I, T>...{
    DEVICE_HOST constexpr
    tuple_base() {}

    template <class... U>
    DEVICE_HOST constexpr explicit
    tuple_base(U const&... u) : tuple_element<I,T>(u)... {}

    template <class... U>
    DEVICE_HOST constexpr
    tuple_base(tuple_base<seq<I...>, U...> const& u) : tuple_element<I,T>(getv(static_cast<tuple_element<I,U> const&>(u)))... {}
};

}

template <class... T>
struct tuple : impl::tuple_base<make_integer_sequence<sizeof...(T)>, T...>
{
    using base = impl::tuple_base<make_integer_sequence<sizeof...(T)>, T...>;
    DEVICE_HOST constexpr
    tuple() {}

    template <class... U>
    DEVICE_HOST constexpr
    tuple(U const&... u) : impl::tuple_base<make_integer_sequence<sizeof...(T)>, T...>(u...) {}

    template <class... U>
    DEVICE_HOST constexpr
    tuple(tuple<U...> const& u)
        : impl::tuple_base<make_integer_sequence<sizeof...(T)>, T...>(static_cast<impl::tuple_base<U...> const&>(u)) {}

    template<index_t I>
    DEVICE_HOST constexpr decltype(auto) get() const
    {
        return impl::getv<I>(*this);
    }

    template<index_t I>
    DEVICE_HOST constexpr decltype(auto) get()
    {
        return impl::getv<I>(*this);
    }
};

template <class... T>
DEVICE_HOST constexpr
tuple<T...>
make_tuple(T const&... t)
{
  return {t...};
}

namespace detail {
template <typename X, typename Y>
struct tuple_concat;

template <typename... Xs, typename... Ys>
struct tuple_concat<tuple<Xs...>, tuple<Ys...>>
{
    using type = tuple<Xs..., Ys...>;
};

template <typename T, index_t N>
struct static_buffer_impl
{
    using type =
        typename tuple_concat<typename static_buffer_impl<T, N / 2>::type,
                              typename static_buffer_impl<T, N - N / 2>::type>::type;
};

template <typename T> struct static_buffer_impl<T, 0> { using type = tuple<>; };
template <typename T> struct static_buffer_impl<T, 1> { using type = tuple<T>; };

template<typename T, index_t N>
using static_buffer_impl_t = typename static_buffer_impl<T, N>::type;

} // namespace detail

template<typename T, index_t N>
struct static_buffer : public detail::static_buffer_impl_t<T, N>
{
    using type = T;
    using base = detail::static_buffer_impl_t<T, N>;

    template<index_t I>
    DEVICE_HOST constexpr T& operator[](number<I>)
    {
        static_assert(I < N);
        return base::template get<I>();
    }

    template<index_t I>
    DEVICE_HOST constexpr const T& operator[](number<I>) const
    {
        static_assert(I < N);
        return base::template get<I>();
    }
};

template <class T>
struct remove_cvref {
  using type = std::remove_cv_t<std::remove_reference_t<T>>;
};

// C++20
template <class T>
using remove_cvref_t = typename remove_cvref<T>::type;


///////////////////
// a self contained type to hold vector, convenient for partial reference
#include "vector_type_predef.hpp"
///////////////////


namespace detail {

struct swallow {
    template <typename... Ts> DEVICE_HOST constexpr swallow(Ts&&...) {} 
};

template <class>
struct constexpr_for_impl;

template <index_t... Is>
struct constexpr_for_impl<seq<Is...>>
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

template<typename kernel_type>
GLOBAL void
__launch_bounds__(kernel_type::MAX_THREADS, kernel_type::MIN_BLOCKS)
kernel_entry(typename kernel_type::args karg)
{
    __shared__ char smem[kernel_type::smem_size()];
    kernel_type{}(karg, smem);
}

///////////////////////////////////////////////////////////////////////////

template<typename TDst, typename TSrc0, typename TSrc1, index_t op_sel_0 = 0, index_t op_sel_1 = 1>
DEVICE void v_pk_mov_b32(TDst & dst, const TSrc0 & src0, const TSrc1 & src1)
{
    static_assert(sizeof(TDst) == 8 && sizeof(TSrc0) == 8 && sizeof(TSrc1) == 8);
    asm volatile("v_pk_mov_b32 %0, %1, %2 op_sel:[%3,%4]" : "+v"(dst) : "v"(src0), "v"(src1), "n"(op_sel_0), "n"(op_sel_1));
}

#define BUFFER_LOAD_DWORD3 0x00020000

DEVICE dwordx4_t make_buffer_resource(const void * ptr)
{
    struct buffer_resource {
        const void * ptr;
        dword_t range;
        dword_t config;
    };
    buffer_resource res {ptr, 0xffffffff, BUFFER_LOAD_DWORD3};
    return __builtin_bit_cast(dwordx4_t, res);
}


// NOTE: we give the output-operand in gld a "+" clobber flag (read/write) instead of "=" (write)
//       in case we may first write into this register, like clear with zero, then use gld
//       if only have "=", compiler may optimze out the first write zero operation.
template<index_t bytes>
struct gld;

template<> struct gld<16>{
    template<typename T>
    DEVICE void operator()(T & value, dwordx4_t res/*buffer resource*/, index_t v_offset, index_t s_offset, index_t i_offset/*max 0xFFF*/, index_t /*flag*/ = 0){
        static_assert(sizeof(T) == 16);
        asm volatile("buffer_load_dwordx4 %0, %1, %2, %3 offen offset:%4"
            : "+v"(value) : "v"(v_offset), "s"(res), "s"(s_offset), "n"(i_offset) : "memory");
    }
};

template<> struct gld<8>{
    template<typename T>
    DEVICE void operator()(T & value, dwordx4_t res/*buffer resource*/, index_t v_offset, index_t s_offset, index_t i_offset/*max 0xFFF*/, index_t /*flag*/ = 0){
        static_assert(sizeof(T) == 8);
        asm volatile("buffer_load_dwordx2 %0, %1, %2, %3 offen offset:%4"
            : "+v"(value) : "v"(v_offset), "s"(res), "s"(s_offset), "n"(i_offset) : "memory");
    }
};

template<> struct gld<4>{
    template<typename T>
    DEVICE void operator()(T & value, dwordx4_t res/*buffer resource*/, index_t v_offset, index_t s_offset, index_t i_offset/*max 0xFFF*/, index_t /*flag*/ = 0){
        static_assert(sizeof(T) == 4);
        asm volatile("buffer_load_dword %0, %1, %2, %3 offen offset:%4"
            : "+v"(value) : "v"(v_offset), "s"(res), "s"(s_offset), "n"(i_offset) : "memory");
    }
};

template<index_t bytes>
struct gld_if;

template<> struct gld_if<16>{
    template<typename T>
    DEVICE void operator()(T & value, dwordx4_t res/*buffer resource*/, index_t v_offset, index_t s_offset, index_t i_offset/*max 0xFFF*/, index_t flag){
        static_assert(sizeof(T) == 16);
        auto save_exec = __builtin_amdgcn_read_exec();
        asm volatile("v_cmpx_le_u32 exec, 1, %5\n"
                     "buffer_load_dwordx4 %0, %1, %2, %3 offen offset:%4\n"
                     "s_mov_b64 exec, %6"
            : "+v"(value) : "v"(v_offset), "s"(res), "s"(s_offset), "n"(i_offset), "v"(flag), "s"(save_exec) : "memory");
    }
};

template<> struct gld_if<8>{
    template<typename T>
    DEVICE void operator()(T & value, dwordx4_t res/*buffer resource*/, index_t v_offset, index_t s_offset, index_t i_offset/*max 0xFFF*/, index_t flag){
        static_assert(sizeof(T) == 8);
        auto save_exec = __builtin_amdgcn_read_exec();
        asm volatile("v_cmpx_le_u32 exec, 1, %5\n"
                     "buffer_load_dwordx2 %0, %1, %2, %3 offen offset:%4\n"
                     "s_mov_b64 exec, %6"
            : "+v"(value) : "v"(v_offset), "s"(res), "s"(s_offset), "n"(i_offset), "v"(flag), "s"(save_exec) : "memory");
    }
};

template<> struct gld_if<4>{
    template<typename T>
    DEVICE void operator()(T & value, dwordx4_t res/*buffer resource*/, index_t v_offset, index_t s_offset, index_t i_offset/*max 0xFFF*/, index_t flag){
        static_assert(sizeof(T) == 4);
        auto save_exec = __builtin_amdgcn_read_exec();
        asm volatile("v_cmpx_le_u32 exec, 1, %5\n"
                     "buffer_load_dwordx1 %0, %1, %2, %3 offen offset:%4\n"
                     "s_mov_b64 exec, %6"
            : "+v"(value) : "v"(v_offset), "s"(res), "s"(s_offset), "n"(i_offset), "v"(flag), "s"(save_exec) : "memory");
    }
};

template<index_t bytes>
struct gst;

template<> struct gst<16>{
    template<typename T>
    DEVICE void operator()(const T & value, dwordx4_t res/*buffer resource*/, index_t v_offset, index_t s_offset, index_t i_offset/*max 0xFFF*/, index_t /*flag*/ = 0){
        static_assert(sizeof(T) == 16);
        asm volatile("buffer_store_dwordx4 %0, %1, %2, %3 offen offset:%4"
            : : "v"(value), "v"(v_offset), "s"(res), "s"(s_offset), "n"(i_offset) : "memory");
    }
};

template<> struct gst<8>{
    template<typename T>
    DEVICE void operator()(const T & value, dwordx4_t res/*buffer resource*/, index_t v_offset, index_t s_offset, index_t i_offset/*max 0xFFF*/, index_t /*flag*/ = 0){
        static_assert(sizeof(T) == 8);
        asm volatile("buffer_store_dwordx2 %0, %1, %2, %3 offen offset:%4"
            : : "v"(value), "v"(v_offset), "s"(res), "s"(s_offset), "n"(i_offset) : "memory");
    }
};

template<> struct gst<4>{
    template<typename T>
    DEVICE void operator()(const T & value, dwordx4_t res/*buffer resource*/, index_t v_offset, index_t s_offset, index_t i_offset/*max 0xFFF*/, index_t /*flag*/ = 0){
        static_assert(sizeof(T) == 4);
        asm volatile("buffer_store_dword %0, %1, %2, %3 offen offset:%4"
            : : "v"(value), "v"(v_offset), "s"(res), "s"(s_offset), "n"(i_offset) : "memory");
    }
};

template<index_t bytes>
struct gst_if;

template<> struct gst_if<16>{
    template<typename T>
    DEVICE void operator()(const T & value, dwordx4_t res/*buffer resource*/, index_t v_offset, index_t s_offset, index_t i_offset/*max 0xFFF*/, index_t flag){
        static_assert(sizeof(T) == 16);
        auto save_exec = __builtin_amdgcn_read_exec();
        asm volatile("v_cmpx_le_u32 exec, 1, %5\n"
                     "buffer_store_dwordx4 %0, %1, %2, %3 offen offset:%4\n"
                     "s_mov_b64 exec %6"
            : : "v"(value), "v"(v_offset), "s"(res), "s"(s_offset), "n"(i_offset), "v"(flag), "s"(save_exec) : "memory");
    }
};

template<> struct gst_if<8>{
    template<typename T>
    DEVICE void operator()(const T & value, dwordx4_t res/*buffer resource*/, index_t v_offset, index_t s_offset, index_t i_offset/*max 0xFFF*/, index_t flag){
        static_assert(sizeof(T) == 8);
        auto save_exec = __builtin_amdgcn_read_exec();
        asm volatile("v_cmpx_le_u32 exec, 1, %5\n"
                     "buffer_store_dwordx2 %0, %1, %2, %3 offen offset:%4\n"
                     "s_mov_b64 exec %6"
            : : "v"(value), "v"(v_offset), "s"(res), "s"(s_offset), "n"(i_offset), "v"(flag), "s"(save_exec) : "memory");
    }
};

template<> struct gst_if<4>{
    template<typename T>
    DEVICE void operator()(const T & value, dwordx4_t res/*buffer resource*/, index_t v_offset, index_t s_offset, index_t i_offset/*max 0xFFF*/, index_t flag){
        static_assert(sizeof(T) == 4);
        auto save_exec = __builtin_amdgcn_read_exec();
        asm volatile("v_cmpx_le_u32 exec, 1, %5\n"
                     "buffer_store_dword %0, %1, %2, %3 offen offset:%4\n"
                     "s_mov_b64 exec %6"
            : : "v"(value), "v"(v_offset), "s"(res), "s"(s_offset), "n"(i_offset), "v"(flag), "s"(save_exec) : "memory");
    }
};

template<index_t bytes>
struct sst;

template<> struct sst<16> {
    template<typename T>
    DEVICE void operator()(void * smem, index_t v_offset, T value, index_t i_offset)
    {
        static_assert(sizeof(T) == 16);
        asm volatile("ds_write_b128 %1, %2, offset:%3"
            : "=r" (smem) /*dummy memory(ptr) dependency*/
            : "v"(v_offset), "v"(value), "n"(i_offset)
            : "memory");
    }
};
template<> struct sst<8> {
    template<typename T>
    DEVICE void operator()(void * smem, index_t v_offset, T value, index_t i_offset)
    {
        static_assert(sizeof(T) == 8);
        asm volatile("ds_write_b64 %1, %2, offset:%3"
            : "=r" (smem) /*dummy memory(ptr) dependency*/
            : "v"(v_offset), "v"(value), "n"(i_offset)
            : "memory");
    }
};
template<> struct sst<4> {
    template<typename T>
    DEVICE void operator()(void * smem, index_t v_offset, T value, index_t i_offset)
    {
        static_assert(sizeof(T) == 4);
        asm volatile("ds_write_b32 %1, %2, offset:%3"
            : "=r" (smem) /*dummy memory(ptr) dependency*/
            : "v"(v_offset), "v"(value), "n"(i_offset)
            : "memory");
    }
};

template<index_t bytes>
struct sld;

template<> struct sld<16> {
    template<typename T>
    DEVICE void operator()(void * smem, T & value, index_t v_offset, index_t i_offset)
    {
        static_assert(sizeof(T) == 16);
        asm volatile("ds_read_b128 %0, %1, offset:%2"
            : "=v"(value)
            : "v"(v_offset), "n"(i_offset), "r"(smem) /*dummy memory(ptr) dependency*/
            : "memory");
    }
};

template<> struct sld<8> {
    template<typename T>
    DEVICE void operator()(void * smem, T & value, index_t v_offset, index_t i_offset)
    {
        static_assert(sizeof(T) == 8);
        asm volatile("ds_read_b64 %0, %1, offset:%2"
            : "=v"(value)
            : "v"(v_offset), "n"(i_offset), "r"(smem) /*dummy memory(ptr) dependency*/
            : "memory");
    }
};

template<> struct sld<4> {
    template<typename T>
    DEVICE void operator()(void * smem, T & value, index_t v_offset, index_t i_offset)
    {
        static_assert(sizeof(T) == 4);
        asm volatile("ds_read_b32 %0, %1, offset:%2"
            : "=v"(value)
            : "v"(v_offset), "n"(i_offset), "r"(smem) /*dummy memory(ptr) dependency*/
            : "memory");
    }
};

DEVICE void gld_fence(index_t cnt)
{
    asm volatile("s_waitcnt vmcnt(%0)" : : "n" (cnt) : "memory");
}

DEVICE void gst_fence(index_t cnt)
{
    asm volatile("s_waitcnt vmcnt(%0)" : : "n" (cnt) : "memory");
}

DEVICE void sld_fence(index_t cnt)
{
    asm volatile("s_waitcnt lgkmcnt(%0)" : : "n"(cnt) : "memory");
}

DEVICE void sst_fence(index_t cnt)
{
    asm volatile("s_waitcnt lgkmcnt(%0)" : : "n"(cnt) : "memory");
}

DEVICE void setprio(index_t prio)
{
    asm volatile("s_setprio %0" : : "n"(prio));
}

DEVICE void wave_barrier()
{
    __builtin_amdgcn_s_barrier();
}
template<index_t cnt = 0>
DEVICE void sched_barrier(number<cnt> = number<0>{})
{
    __builtin_amdgcn_sched_barrier(cnt);
}

DEVICE void s_nop(index_t cnt = 0)
{
    asm volatile("s_nop %0" : : "n"(cnt) : "memory");
}

#define MFMA_USE_INTRINSIC 0

namespace impl {
// clang-format off
// here A/B/C are all ext_vector_type(...)

struct call_mfma_f32_32x32x8_f16 {
    template<typename TA, typename TB, typename TC>
    DEVICE void operator()(const TA& a, const TB &b, TC& c)
    {
#if MFMA_USE_INTRINSIC
        sched_barrier();
        c = __builtin_amdgcn_mfma_f32_32x32x8f16(a, b, c, 0, 0, 0);
        sched_barrier();
#else
        asm volatile("v_mfma_f32_32x32x8f16 %0, %1, %2, %0" : "+v"(c) : "v"(a), "v"(b));
#endif
    }
};

struct call_mfma_f32_32x32x16_f16 {
    template<typename TA, typename TB, typename TC>
    DEVICE void operator()(const TA& a, const TB &b, TC& c)
    {
        vector_type<f16, 8> a_pack {a};
        vector_type<f16, 8> b_pack {b};
#if MFMA_USE_INTRINSIC
        // HIP tend to use agpr as accumulator
        sched_barrier();
        c = __builtin_amdgcn_mfma_f32_32x32x8f16(a_pack.template to_varray<typename vector_type<f16, 4>::type>()[I0],
                                                    b_pack.template to_varray<typename vector_type<f16, 4>::type>()[I0],
                                                    c, 0, 0, 0);
        c = __builtin_amdgcn_mfma_f32_32x32x8f16(a_pack.template to_varray<typename vector_type<f16, 4>::type>()[I1],
                                                    b_pack.template to_varray<typename vector_type<f16, 4>::type>()[I1],
                                                    c, 0, 0, 0);
        sched_barrier();
#else
        vector_type<f16, 4>::type a0 = a_pack.template to_varray<typename vector_type<f16, 4>::type>()[I0];
        vector_type<f16, 4>::type b0 = b_pack.template to_varray<typename vector_type<f16, 4>::type>()[I0];
        vector_type<f16, 4>::type a1 = a_pack.template to_varray<typename vector_type<f16, 4>::type>()[I1];
        vector_type<f16, 4>::type b1 = b_pack.template to_varray<typename vector_type<f16, 4>::type>()[I1];
        // clang-17+ has ODR bug for following inline asm, we have to first move the contained data to a lvalue
        asm volatile("v_mfma_f32_32x32x8f16 %0, %1, %2, %0" : "+v"(c) : "v"(a0), "v"(b0));
        asm volatile("v_mfma_f32_32x32x8f16 %0, %1, %2, %0" : "+v"(c) : "v"(a1), "v"(b1));
#endif
    }
};

struct call_mfma_f32_16x16x16_f16 {
    template<typename TA, typename TB, typename TC>
    DEVICE void operator()(const TA& a, const TB &b, TC& c)
    {
#if MFMA_USE_INTRINSIC
        sched_barrier();
        c = __builtin_amdgcn_mfma_f32_16x16x16f16(a, b, c, 0, 0, 0);
        sched_barrier();
#else
        asm volatile("v_mfma_f32_16x16x16f16 %0, %1, %2, %0" : "+v"(c) : "v"(a), "v"(b));
#endif
    }
};
// clang-format on
}

template<
        typename a_type_,
        typename b_type_,
        typename c_type_,
        index_t m_,
        index_t n_,
        index_t k_,
        index_t blocks_,
        index_t num_register_a_,
        index_t num_register_b_,
        index_t num_register_c_,
        typename inst_func_>
struct mfma_inst {
    using a_type = a_type_;
    using b_type = b_type_;
    using c_type = c_type_;         // This is actually acc type
    static constexpr index_t m = m_;
    static constexpr index_t n = n_;
    static constexpr index_t k = k_;
    static constexpr index_t blocks = blocks_;
    static constexpr index_t num_register_a = num_register_a_;
    static constexpr index_t num_register_b = num_register_b_;
    static constexpr index_t num_register_c = num_register_c_;
    static constexpr index_t wave_size = 64;

    // TODO: double can not use this
    static constexpr index_t num_v_a = num_register_a * (4 / sizeof(a_type));
    static constexpr index_t num_v_b = num_register_b * (4 / sizeof(b_type));
    static constexpr index_t num_v_c = num_register_c * (4 / sizeof(c_type));

    static constexpr index_t c_per_group = 4;  // contiguous register for C along one dim
    static_assert(num_v_c % c_per_group == 0);
    static constexpr index_t groups = num_v_c / c_per_group;
    static constexpr index_t rows_per_group = (wave_size / n) * c_per_group;

    using a_vector_type = typename vector_type<a_type, num_v_a>::type;
    using b_vector_type = typename vector_type<b_type, num_v_b>::type;
    using c_vector_type = typename vector_type<c_type, num_v_c>::type;

    using inst_func = inst_func_;

    template<bool swap_a_b_ = false>
    DEVICE void operator()(const a_vector_type & v_a, const b_vector_type & v_b, c_vector_type & v_c, bool_const<swap_a_b_> = bool_const<false>{}) {
        if constexpr (swap_a_b_)
            inst_func{}(v_b, v_a, v_c);
        else
            inst_func{}(v_a, v_b, v_c);
    }

    static DEVICE void lane_rc(index_t & row_id, index_t & col_id, index_t lane_id)
    {
        row_id = (lane_id / n) * c_per_group;
        col_id = lane_id % n;
    }
};

// clang-format off
//                                  atype  btype  ctype   m   n   k  b  va  vb  vc
using mfma_f32_16x16x16_f16   = mfma_inst<f16, f16, f32, 16, 16, 16, 1,  2,  2,  4, impl::call_mfma_f32_16x16x16_f16>;
using mfma_f32_32x32x8_f16    = mfma_inst<f16, f16, f32, 32, 32,  8, 1,  2,  2, 16, impl::call_mfma_f32_32x32x8_f16>;
using mfma_f32_32x32x16_f16   = mfma_inst<f16, f16, f32, 32, 32, 16, 1,  4,  4, 16, impl::call_mfma_f32_32x32x16_f16>;

template<typename, typename, typename, index_t, index_t, index_t> struct mfma_selector;
template<> struct mfma_selector<f16, f16, f32, 16, 16, 16> { using type = mfma_f32_16x16x16_f16; };
template<> struct mfma_selector<f16, f16, f32, 32, 32, 8>  { using type = mfma_f32_32x32x8_f16; };
template<> struct mfma_selector<f16, f16, f32, 32, 32, 16> { using type = mfma_f32_32x32x16_f16; };

template<typename T, index_t N>
constexpr void clear(vector_type<T, N> & vec)
{
#if 0
    if constexpr (sizeof(T) * N % 8 == 0){
        // b64 fast path
        using chunk_type = typename vector_type<T, 8 / sizeof(T)>::type;
        static_assert(sizeof(chunk_type) == 8);
        constexpr index_t chunks = sizeof(T) * N / sizeof(chunk_type);
        chunk_type z{0};
        constexpr_for<0, chunks, 1>{}([&](auto i_chunk){
            v_pk_mov_b32(vec.template to_varray<chunk_type>()[i_chunk], z, z);
        });
    }
    else {
        constexpr_for<0, N, 1>{}([&](auto i){
            vec.template to_varray<T>()[i] = static_cast<T>(0);
        });
    }
#else
    constexpr_for<0, N, 1>{}([&](auto i){
        vec.template to_varray<T>()[i] = static_cast<T>(0);
    });
#endif
}

template<typename DstType, typename SrcType, index_t N>
constexpr auto vector_cast(const vector_type<SrcType, N> & src)
{
    vector_type<DstType, N> dst;
    constexpr_for<0, N, 1>{}(
        [&](auto i){
            dst.template to_varray<DstType>()[i] = static_cast<DstType>(src.template to_varray<SrcType>()[i]);
        });
    return dst;
}
