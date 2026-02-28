/***************************************************************************************************
 * OPUS, AI (O)(P)erator Micro(U) (S)TD
 *
 * Crafting the micro standard templates for AI Operators on ROCm
 *
 * MIT License
 * Copyright (C) 2025 carlus.huang@amd.com
 *
 **************************************************************************************************/
#pragma once

#include <array>
#include <type_traits>
#include <utility>
#include <initializer_list>

#ifdef __HIPCC__
#define OPUS_H inline __host__
#define OPUS_D inline __device__
#define OPUS_H_D inline __host__ __device__
#define OPUS_D_EXTERN __device__
#define OPUS_H_D_EXTERN __host__ __device__
#else
#define OPUS_H inline
#define OPUS_D inline
#define OPUS_H_D inline
#define OPUS_D_EXTERN
#define OPUS_H_D_EXTERN
#endif

#ifndef OPUS_FP32_to_BF16_DEFAULT
#define OPUS_FP32_to_BF16_DEFAULT 2 // truncate
#endif

#ifndef OPUS_TILE_CONTAINER
#define OPUS_TILE_CONTAINER 0 // 0:vector, 1:array of vector, 2:flattened array
#endif

namespace opus {
// clang-format off
/////////////////////////////////////////////////////////////////////////////////////////////////////////
// type traits
using std::remove_reference;
using std::remove_reference_t;
using std::remove_cv;
using std::remove_cv_t;
using std::is_same;
using std::is_same_v;

template<typename T> struct remove_cvref { using type = remove_cv_t<remove_reference_t<T>>; };
template<typename T> using remove_cvref_t = remove_cv_t<remove_reference_t<T>>;

/////////////////////////////////////////////////////////////////////////////////////////////////////////
// constant
using index_t = int;

template<index_t I> struct number : public std::integral_constant<index_t, I> {};
template<bool B>    struct bool_constant : public std::bool_constant<B> {};

typedef bool_constant<true>  true_type;
typedef bool_constant<false> false_type;

template<typename>           struct is_constant : public false_type {};
template<typename T, auto I> struct is_constant<std::integral_constant<T, I>> : true_type {};
template<auto I>             struct is_constant<number<I>> : true_type {};
template<auto I>             struct is_constant<bool_constant<I>> : true_type {};
template <class T> static constexpr bool is_constant_v = is_constant<remove_cvref_t<T>>::value;    // prefer use this

// using opus::operator""_I; // => add this in your code to utilize the literal cast, e.g. 2_I, 3_I
template <char... Ds>
OPUS_H_D constexpr decltype(auto) operator""_I() {
    constexpr auto to_number_ = []() { index_t v = 0; for (char d : {Ds...}) v = v * 10 + (d - '0'); return v; }; return number<to_number_()>{};
}

#define OPUS_LEFT_UNARY_OP(OP) template <auto x>         OPUS_H_D constexpr auto operator OP(number<x>)            { return number<(OP x)>{};   }
#define OPUS_BINARY_OP(OP)     template <auto x, auto y> OPUS_H_D constexpr auto operator OP(number<x>, number<y>) { return number<(x OP y)>{}; }

OPUS_LEFT_UNARY_OP(+)
OPUS_LEFT_UNARY_OP(-)
OPUS_LEFT_UNARY_OP(~)
OPUS_LEFT_UNARY_OP(!)

OPUS_BINARY_OP(+)
OPUS_BINARY_OP(-)
OPUS_BINARY_OP(*)
OPUS_BINARY_OP(/)
OPUS_BINARY_OP(%)
OPUS_BINARY_OP(&)
OPUS_BINARY_OP(|)
OPUS_BINARY_OP(^)
OPUS_BINARY_OP(<<)
OPUS_BINARY_OP(>>)
OPUS_BINARY_OP(&&)
OPUS_BINARY_OP(||)
OPUS_BINARY_OP(==)
OPUS_BINARY_OP(!=)
OPUS_BINARY_OP(>)
OPUS_BINARY_OP(<)
OPUS_BINARY_OP(>=)
OPUS_BINARY_OP(<=)

#undef OPUS_LEFT_UNARY_OP
#undef OPUS_BINARY_OP

/////////////////////////////////////////////////////////////////////////////////////////////////////////
// underscore, useful struture to mock
struct underscore { /*who am I*/ };
static constexpr underscore _;
template <typename T> struct is_underscore : false_type {};
template <> struct is_underscore<underscore> : true_type {};
template <typename T> static constexpr bool is_underscore_v = is_underscore<T>::value;

/////////////////////////////////////////////////////////////////////////////////////////////////////////
// constexpr functional math
struct plus       { template<typename X, typename Y=X> OPUS_H_D constexpr decltype(auto) operator()(X a, Y b) const { return a + b; } };
struct minus      { template<typename X, typename Y=X> OPUS_H_D constexpr decltype(auto) operator()(X a, Y b) const { return a - b; } };
struct multiplies { template<typename X, typename Y=X> OPUS_H_D constexpr decltype(auto) operator()(X a, Y b) const { return a * b; } };
struct divides    { template<typename X, typename Y=X> OPUS_H_D constexpr decltype(auto) operator()(X a, Y b) const { return a / b; } };

/////////////////////////////////////////////////////////////////////////////////////////////////////////
// seq
template <index_t... Is>
class seq {
public:
    using value_type = index_t;

    OPUS_H_D static constexpr index_t size() { return sizeof...(Is);}
    OPUS_H_D constexpr value_type operator[](index_t i) const { return data[i]; }
    OPUS_H_D static constexpr value_type at(index_t i) { return data[i]; }
    template <index_t I> OPUS_H_D static constexpr value_type at()          { return data[I]; }
    template <index_t I> OPUS_H_D static constexpr value_type at(number<I>) { return data[I]; }

private:
    static constexpr value_type data[sizeof...(Is) + 1] = {Is..., value_type{}};
};

template <index_t I, index_t... Is> OPUS_H_D constexpr auto seq_pop_front(seq<I, Is...>) { return seq<Is...>{}; }

template <index_t I, index_t... Is> OPUS_H_D constexpr decltype(auto) get(seq<Is...>const& ) { static_assert(I < sizeof...(Is)); return seq<Is...>::at(number<I>{}); }
template <index_t I, index_t... Is> OPUS_H_D constexpr decltype(auto) get(seq<Is...>& )      { static_assert(I < sizeof...(Is)); return seq<Is...>::at(number<I>{}); }
template <index_t I, index_t... Is> OPUS_H_D constexpr decltype(auto) get(seq<Is...>&& )     { static_assert(I < sizeof...(Is)); return seq<Is...>::at(number<I>{}); }

namespace impl {
template <typename T, T... Is> struct __integer_sequence;
template <index_t... Is>       struct __integer_sequence<index_t, Is...> { using seq_type = seq<Is...>; };
template<index_t, index_t, typename>                 struct __steped_integer_seq;
template<index_t Start, index_t Step, index_t... Is> struct __steped_integer_seq<Start, Step, seq<Is...>> { using seq_type = seq<(Start + Is * Step) ... >; };

template<typename>                   struct __make_index_seq;
template <index_t N>                 struct __make_index_seq<seq<N>>          { using seq_type = typename __make_integer_seq<__integer_sequence, index_t, N>::seq_type; };
template<index_t Start, index_t End> struct __make_index_seq<seq<Start, End>> { using seq_type = typename __steped_integer_seq<Start, 1, typename __make_index_seq< seq<(End-Start)/1> >::seq_type>::seq_type; };
template<index_t Start, index_t End, index_t Step>  struct __make_index_seq<seq<Start, End, Step>> {
    using seq_type = typename __steped_integer_seq<Start, Step, typename __make_index_seq< seq<(End-Start)/Step> >::seq_type>::seq_type;
};
} // namespace impl
// make_index_seq<5> -> seq<0,1,2,3,4> | make_index_seq<4, 9> -> seq<4,5,6,7,8> | make_index_seq<4, 8, 2> -> seq<4, 6>
template<index_t...Is> using make_index_seq = typename impl::__make_index_seq<seq<Is...>>::seq_type;

namespace impl {
template<index_t Value, index_t N>
struct __make_repeated_seq {
    template<index_t... I> static constexpr auto __make(seq<I...>) { return seq<(void(I), Value)...>{}; }
    using seq_type = decltype(__make(make_index_seq<N>{}));
};
} // namespace impl
template<index_t V, index_t N> using make_repeated_seq = typename impl::__make_repeated_seq<V, N>::seq_type;

template<index_t...Xs, index_t...Ys> OPUS_H_D constexpr auto concat_seq(seq<Xs...>, seq<Ys...>) { return seq<Xs..., Ys...>{}; }

namespace impl {
template<typename, typename>                                 struct reduce_seq_impl;
template <typename R, index_t I0, index_t I1, index_t... Is> struct reduce_seq_impl<R, seq<I0, I1, Is...>> { using type = typename reduce_seq_impl<R, seq<R{}(I0, I1), Is...>>::type; };
template <typename R, index_t I>                             struct reduce_seq_impl<R, seq<I>> { using type = seq<I>; };
template <typename R>                                        struct reduce_seq_impl<R, seq<>>  { using type = seq<>;  };
}
template<typename R, index_t...Xs> OPUS_H_D constexpr auto reduce_seq(seq<Xs...>) { return typename impl::reduce_seq_impl<R, seq<Xs...>>::type{}; }
template<index_t...Xs> OPUS_H_D constexpr auto reduce_seq_sum(seq<Xs...>) { return reduce_seq<opus::plus>(seq<Xs...>{}); }
template<index_t...Xs> OPUS_H_D constexpr auto reduce_seq_mul(seq<Xs...>) { return reduce_seq<opus::multiplies>(seq<Xs...>{}); }

template<typename T> struct is_seq : false_type {};
template<index_t... Is> struct is_seq<seq<Is...>> : true_type {};
template<typename T> constexpr bool is_seq_v = is_seq<remove_cvref_t<T>>::value;

template<typename T> OPUS_H_D constexpr std::enable_if_t<is_seq_v<T>, index_t> size(T&&) { return remove_cvref_t<T>::size(); /* tuple size */}
template<typename T> OPUS_H_D constexpr std::enable_if_t<is_seq_v<T>, index_t> size()    { return remove_cvref_t<T>::size(); /* tuple size */}

template <index_t I, typename T, std::enable_if_t<is_seq_v<T>, bool> = true> OPUS_H_D constexpr decltype(auto) get(T const& t) { static_assert(I < T::size()); return t[number<I>{}]; }
template <index_t I, typename T, std::enable_if_t<is_seq_v<T>, bool> = true> OPUS_H_D constexpr decltype(auto) get(T&  t)      { static_assert(I < T::size()); return t[number<I>{}]; }
template <index_t I, typename T, std::enable_if_t<is_seq_v<T>, bool> = true> OPUS_H_D constexpr decltype(auto) get(T&& t)      { static_assert(I < T::size()); return t[number<I>{}]; }
/////////////////////////////////////////////////////////////////////////////////////////////////////////
// functional
namespace impl {
template <class T>       struct static_for_impl;
template <index_t... Is> struct static_for_impl<seq<Is...>> { template <class F> OPUS_H_D constexpr void operator()(F&& f) const { (f(number<Is>{}), ...); } };
}   // namespace impl
template<index_t N, typename F> OPUS_H_D constexpr void static_for(F f) { impl::static_for_impl<make_index_seq<N>>{}(f); }

template<typename F, typename... R, std::enable_if_t<(std::is_integral_v<R> && ...), bool> = true>
OPUS_H_D constexpr void static_for(F f, R... range) {
    if      constexpr (sizeof...(range) == 1) { auto end = std::get<0>(std::tie(range...));     for(index_t i = 0; i < end; i++) { f(i); } }
    else if constexpr (sizeof...(range) == 2) { auto [start, end] = std::tie(range...);         for(index_t i = start; i < end; i++) { f(i); } }
    else if constexpr (sizeof...(range) == 3) { auto [start, end, step] = std::tie(range...);   for(index_t i = start; i < end; i += step) { f(i); } }
}

template<typename F, typename... R, std::enable_if_t<(is_constant_v<R> && ...), bool> = true>
OPUS_H_D constexpr void static_for(F f, R...) { impl::static_for_impl<make_index_seq<R::value...>>{}(f); }

namespace impl {
template <typename Seq> struct static_ford_impl {
    template <typename F, typename... Ids> OPUS_H_D constexpr void operator()(F f, Ids... ids) const {
        static_for<get<0>(Seq{})>([=](auto I){ static_ford_impl<decltype(seq_pop_front(Seq{}))>{}(f, ids..., I); });
    }
};
template <> struct static_ford_impl<seq<>> { template <typename F, typename... Ids> OPUS_H_D constexpr void operator()(F f, Ids... ids) const { f(ids...); } };
}

template<index_t... N, typename F> OPUS_H_D constexpr void static_ford(F f) { impl::static_ford_impl<seq<N...>>{}(f); }
template<index_t... N, typename F> OPUS_H_D constexpr void static_ford(seq<N...>, F f) { impl::static_ford_impl<seq<N...>>{}(f); }
template <class... T> struct tuple;
template<index_t... N, typename F> OPUS_H_D constexpr void static_ford(tuple<number<N>...>, F f) { impl::static_ford_impl<seq<N...>>{}(f); }

/////////////////////////////////////////////////////////////////////////////////////////////////////////
// array, enhanced C like array style
template <typename T, index_t N>
struct array {
    using value_type = remove_cvref_t<T>;
    using type = array<value_type, N>;
#if 0   // don't define following, just let me be trivially copyable class
    OPUS_H_D constexpr array() = default;
    OPUS_H_D constexpr array(const type& o) { static_for<N>([&](auto i){ content[i.value] = o[i.value]; }); }
    OPUS_H_D constexpr type& operator=(const type o) { static_for<N>([&](auto i){ content[i.value] = o[i.value]; }); return *this; }

    template<typename...Z, std::enable_if_t<(std::is_same_v<remove_cvref_t<Z>, value_type> && ...), bool> = true>
    OPUS_H_D constexpr array(Z&&... zs) : content{zs...}  { /* used for make_array */ }
#endif
    OPUS_H_D constexpr value_type& operator[](index_t pos) { return content[pos]; }
    OPUS_H_D constexpr const value_type& operator[](index_t pos) const { return content[pos]; }
    template<index_t I> OPUS_H_D constexpr value_type& operator[](number<I>) { return content[I]; }
    template<index_t I> OPUS_H_D constexpr const value_type& operator[](number<I>) const { return content[I]; }
    OPUS_H_D constexpr void fill(const T& value) { static_for<N>([&](auto i){ content[i.value] = value; }); }
    OPUS_H_D constexpr void clear() { fill(static_cast<T>(0)); }

    OPUS_H_D static constexpr bool empty() { return size() == 0; }
    OPUS_H_D static constexpr index_t size() { return N; }

    // we need this "content" member to have a default value, so that the implicitly defined constructor could be constexpr
    // see: https://en.cppreference.com/w/cpp/language/constexpr.html#constexpr_constructor
    value_type content[N] {};
};

template <typename T, index_t N>
OPUS_H_D constexpr bool operator==(const array<T,N>& x, const array<T,N>& y) { for (index_t i = 0; i < N; ++i) { if (x[i] != y[i]) { return false; } } return true; }

template <typename T, index_t N> OPUS_H_D constexpr void clear(array<T,N>& a) { a.clear(); }
template <typename T, index_t N> OPUS_H_D constexpr void fill(array<T,N>& a, T const& value) { a.fill(value); }

template<typename T> struct is_array : false_type {};
template<typename T, index_t N> struct is_array<array<T, N>> : true_type {};
template<typename T> constexpr bool is_array_v = is_array<remove_cvref_t<T>>::value;

namespace impl {
template<typename> struct is_ref_wrapper : std::false_type{};
template<typename T> struct is_ref_wrapper<std::reference_wrapper<T>> : std::true_type{};
template<typename T> using not_ref_wrapper = std::negation<is_ref_wrapper<std::decay_t<T>>>;

template<typename D, typename...> struct array_return_type_helper { using type = D; };
template<typename... Types>
struct array_return_type_helper<void, Types...> : std::common_type<Types...> {
    static_assert(std::conjunction_v<not_ref_wrapper<Types>...>, "Types cannot contain reference_wrappers when D is void");
};

template<typename D, typename... Types> using array_return_type = opus::array<typename array_return_type_helper<D, Types...>::type, sizeof...(Types)>;
}
template<typename D = void, typename... Types> constexpr impl::array_return_type<D, Types...> make_array(Types&&... t) { return {std::forward<Types>(t)...}; }

template <index_t I, typename T, std::enable_if_t<is_array_v<T>, bool> = true> OPUS_H_D constexpr decltype(auto) get(T const& t) { static_assert(I < T::size()); return t[number<I>{}]; }
template <index_t I, typename T, std::enable_if_t<is_array_v<T>, bool> = true> OPUS_H_D constexpr decltype(auto) get(T&  t)      { static_assert(I < T::size()); return t[number<I>{}]; }
template <index_t I, typename T, std::enable_if_t<is_array_v<T>, bool> = true> OPUS_H_D constexpr decltype(auto) get(T&& t)      { static_assert(I < T::size()); return t[number<I>{}]; }

namespace impl {
template <class T0, class T1, index_t... I0, index_t... I1>
OPUS_H_D constexpr auto concat_array(T0 const& t0, T1 const& t1, seq<I0...>, seq<I1...>) { return opus::make_array(get<I0>(t0)..., get<I1>(t1)...); }
template <class T0, class T1, class T2, index_t... I0, index_t... I1, index_t...I2>
OPUS_H_D constexpr auto concat_array(T0 const& t0, T1 const& t1, T2 const& t2, seq<I0...>, seq<I1...>, seq<I2...>) { return opus::make_array(get<I0>(t0)..., get<I1>(t1)..., get<I2>(t2)...); }
template <class T0, class T1, class T2, class T3, index_t... I0, index_t... I1, index_t...I2, index_t...I3>
OPUS_H_D constexpr auto concat_array(T0 const& t0, T1 const& t1, T2 const& t2, T3 const& t3, seq<I0...>, seq<I1...>, seq<I2...>, seq<I3...>) { return opus::make_array(get<I0>(t0)..., get<I1>(t1)..., get<I2>(t2)..., get<I3>(t3)...); }
}
template <class T0> OPUS_H_D  constexpr auto concat_array(T0 const& t0) { return t0; }
template <class T0, class T1>
OPUS_H_D  constexpr auto concat_array(T0 const& t0, T1 const& t1) { return impl::concat_array(t0, t1, make_index_seq<T0::size()>{}, make_index_seq<T1::size()>{}); }
template <class T0, class T1, class T2>
OPUS_H_D  constexpr auto concat_array(T0 const& t0, T1 const& t1, T2 const& t2) { return impl::concat_array(t0, t1, t2, make_index_seq<T0::size()>{}, make_index_seq<T1::size()>{}, make_index_seq<T2::size()>{}); }
template <class T0, class T1, class T2, class T3>
OPUS_H_D  constexpr auto concat_array(T0 const& t0, T1 const& t1, T2 const& t2, T3 const& t3) {
                                            return impl::concat_array(t0, t1, t2, t3, make_index_seq<T0::size()>{}, make_index_seq<T1::size()>{}, make_index_seq<T2::size()>{}, make_index_seq<T3::size()>{}); }
template <class T0, class T1, class T2, class T3, class T4, class... Ts>
OPUS_H_D  constexpr auto concat_array(T0 const& t0, T1 const& t1, T2 const& t2, T3 const& t3, T4 const& t4, Ts const&... ts) { return concat_array(concat_array(t0, t1, t2, t3), concat_array(t4, ts...)); }

template<typename T> OPUS_H_D constexpr std::enable_if_t<is_array_v<T>, index_t> size(T&&) { return remove_cvref_t<T>::size(); /* tuple size */}
template<typename T> OPUS_H_D constexpr std::enable_if_t<is_array_v<T>, index_t> size()    { return remove_cvref_t<T>::size(); /* tuple size */}
/////////////////////////////////////////////////////////////////////////////////////////////////////////
// tuple
namespace impl {
template <index_t idx, typename T, bool is_empty = (std::is_empty_v<T> || std::is_void_v<T>)> struct tuple_object {}; // the place where content is stored

template <index_t idx, typename T>
struct tuple_object<idx, T, true> {
    OPUS_H_D constexpr tuple_object() {}
    template <typename U, typename std::enable_if<!std::is_same<remove_cvref_t<U>, tuple_object>::value, bool>::type = false>
    OPUS_H_D constexpr tuple_object(U&&) {}
};
template <index_t idx, typename T>
struct tuple_object<idx, T, false> {
    OPUS_H_D constexpr tuple_object() : element{} {}
    template <typename U, typename std::enable_if<!std::is_same<remove_cvref_t<U>, tuple_object>::value, bool>::type = false>
    OPUS_H_D constexpr tuple_object(U&& e) : element(std::forward<U>(e)) {}
    T element;
};

// NOTE: we return a instance(not a reference) if content is empty
template <index_t I, class T> OPUS_H_D constexpr T        getv(const tuple_object<I, T, true>&)    { return {}; }
template <index_t I, class T> OPUS_H_D constexpr const T& getv(const tuple_object<I, T, false>& x) { return x.element; }
template <index_t I, class T> OPUS_H_D constexpr T&       getv(tuple_object<I, T, false>& x)       { return x.element; }
template <index_t I, class T> OPUS_H_D constexpr T&&      getv(tuple_object<I, T, false>&& x)      { return static_cast<T&&>(x.element); }

template <typename index_seq, typename... T> struct tuple_base;

template <index_t... I, typename... T>
struct tuple_base<seq<I...>, T...> : tuple_object<I, T>... {
    OPUS_H_D constexpr tuple_base() = default;

    template <class U, typename std::enable_if<sizeof...(I) == 1 && sizeof...(T) == 1 && !std::is_same<remove_cvref_t<U>, tuple_base>::value, bool>::type = false>
    OPUS_H_D constexpr tuple_base(U&& u) : tuple_object<I, T>(std::forward<U>(u))... {}

    template <typename... U, typename std::enable_if<sizeof...(U) >= 2, bool>::type = false>
    OPUS_H_D constexpr tuple_base(U&&... u) : tuple_object<I, T>(std::forward<U>(u))... { static_assert(sizeof...(I) == sizeof...(T) && sizeof...(I) == sizeof...(U), "wrong!"); }
};
} // namespace impl
template <class... T>
struct tuple : impl::tuple_base<make_index_seq<sizeof...(T)>, T...> {
    OPUS_H_D static constexpr index_t size() { return sizeof...(T); }
    using base = impl::tuple_base<make_index_seq<sizeof...(T)>, T...>;
    OPUS_H_D constexpr tuple() = default;

    template <typename U, typename std::enable_if<sizeof...(T) == 1 && !std::is_same<remove_cvref_t<U>, tuple>::value, bool>::type = false>
    OPUS_H_D constexpr tuple(U&& u) : base(std::forward<U>(u)) {}

    template <typename... U, typename std::enable_if<sizeof...(U) == sizeof...(T) && sizeof...(U) >= 2, bool>::type = false>
    OPUS_H_D constexpr tuple(U&&... u) : base(std::forward<U>(u)...) {}
};

template <index_t I, class... T> OPUS_H_D constexpr decltype(auto) get(tuple<T...> const& t) { static_assert(I < sizeof...(T)); return impl::getv<I>(t); }
template <index_t I, class... T> OPUS_H_D constexpr decltype(auto) get(tuple<T...>& t)       { static_assert(I < sizeof...(T)); return impl::getv<I>(t); }
template <index_t I, class... T> OPUS_H_D constexpr decltype(auto) get(tuple<T...>&& t)      { static_assert(I < sizeof...(T)); return impl::getv<I>(std::move(t)); }

template <index_t I0, index_t I1, index_t... Is, class T>  /*recursive get*/
OPUS_H_D constexpr decltype(auto) get(T&& t) { return get<I1, Is...>(get<I0>(std::move(t))); }

template <typename... T> OPUS_H_D constexpr auto make_tuple(T&&... xs) { return tuple<remove_cvref_t<T>...>(std::forward<T>(xs)...); }

namespace impl {
template <typename T, index_t... Is> OPUS_H_D constexpr auto make_repeated_tuple(T&& x, seq<Is...>) { return opus::make_tuple((void(Is), std::forward<T>(x))...); }
} // namespace impl
template <index_t N, typename T> OPUS_H_D constexpr auto make_repeated_tuple(T&& x) { return impl::make_repeated_tuple(std::forward<T>(x), make_index_seq<N>{}); }
template <typename T, index_t N> OPUS_H_D constexpr auto make_repeated_tuple(T&& x, number<N>) { return impl::make_repeated_tuple(std::forward<T>(x), make_index_seq<N>{}); }

namespace impl {
template <class T0, class T1, index_t... I0, index_t... I1>
OPUS_H_D constexpr auto concat_tuple(T0 const& t0, T1 const& t1, seq<I0...>, seq<I1...>) { return opus::make_tuple(get<I0>(t0)..., get<I1>(t1)...); }
template <class T0, class T1, class T2, index_t... I0, index_t... I1, index_t...I2>
OPUS_H_D constexpr auto concat_tuple(T0 const& t0, T1 const& t1, T2 const& t2, seq<I0...>, seq<I1...>, seq<I2...>) { return opus::make_tuple(get<I0>(t0)..., get<I1>(t1)..., get<I2>(t2)...); }
template <class T0, class T1, class T2, class T3, index_t... I0, index_t... I1, index_t...I2, index_t...I3>
OPUS_H_D constexpr auto concat_tuple(T0 const& t0, T1 const& t1, T2 const& t2, T3 const& t3, seq<I0...>, seq<I1...>, seq<I2...>, seq<I3...>) { return opus::make_tuple(get<I0>(t0)..., get<I1>(t1)..., get<I2>(t2)..., get<I3>(t3)...); }
}
template <class T0> OPUS_H_D  constexpr auto concat_tuple(T0 const& t0) { return t0; }
template <class T0, class T1>
OPUS_H_D  constexpr auto concat_tuple(T0 const& t0, T1 const& t1) { return impl::concat_tuple(t0, t1, make_index_seq<T0::size()>{}, make_index_seq<T1::size()>{}); }
template <class T0, class T1, class T2>
OPUS_H_D  constexpr auto concat_tuple(T0 const& t0, T1 const& t1, T2 const& t2) { return impl::concat_tuple(t0, t1, t2, make_index_seq<T0::size()>{}, make_index_seq<T1::size()>{}, make_index_seq<T2::size()>{}); }
template <class T0, class T1, class T2, class T3>
OPUS_H_D  constexpr auto concat_tuple(T0 const& t0, T1 const& t1, T2 const& t2, T3 const& t3) {
                                            return impl::concat_tuple(t0, t1, t2, t3, make_index_seq<T0::size()>{}, make_index_seq<T1::size()>{}, make_index_seq<T2::size()>{}, make_index_seq<T3::size()>{}); }
template <class T0, class T1, class T2, class T3, class T4, class... Ts>
OPUS_H_D  constexpr auto concat_tuple(T0 const& t0, T1 const& t1, T2 const& t2, T3 const& t3, T4 const& t4, Ts const&... ts) { return concat_tuple(concat_tuple(t0, t1, t2, t3), concat_tuple(t4, ts...)); }

template <typename> struct is_tuple : false_type {};
template <typename... T> struct is_tuple<opus::tuple<T...>> : true_type {};
template <typename T> static constexpr bool is_tuple_v = is_tuple<remove_cvref_t<T>>::value;

template<typename T> OPUS_H_D constexpr std::enable_if_t<is_tuple_v<T>, index_t> size(T&&) { return remove_cvref_t<T>::size(); /* tuple size */}
template<typename T> OPUS_H_D constexpr std::enable_if_t<is_tuple_v<T>, index_t> size()    { return remove_cvref_t<T>::size(); /* tuple size */}

template <typename T, std::enable_if_t<!is_tuple_v<T>, bool> = true> OPUS_H_D constexpr auto explode_tuple(const T& t) { return opus::make_tuple(t); }
template <typename T, index_t... Is> OPUS_H_D constexpr auto                                 explode_tuple(const T&, seq<Is...>);
template <typename T, std::enable_if_t<is_tuple_v<T>, bool> = true> OPUS_H_D constexpr auto  explode_tuple(const T& t) { return explode_tuple(t, make_index_seq<size<T>()>{}); }
template <typename T, index_t... Is> OPUS_H_D constexpr auto                                 explode_tuple(const T& t, seq<Is...>) { return concat_tuple(explode_tuple(get<Is>(t))...); }

template <typename T, index_t... Is> OPUS_H_D constexpr auto flatten_tuple(const T& t, seq<Is...>) { return concat_tuple(explode_tuple(get<Is>(t))...); }
template <typename T> OPUS_H_D constexpr auto                flatten_tuple(const T& t) { return flatten_tuple(t, make_index_seq<size<T>()>{}); }

namespace impl {
template<typename Outer, typename Inner, index_t...Is>
OPUS_H_D constexpr auto embed_nested_tuple_impl(const Outer& ot, const Inner& it, seq<Is...>) { return opus::make_tuple(concat_tuple(get<Is>(ot), get<Is>(it))...); }

template<typename TargetType, typename T, index_t...Is>
OPUS_H_D constexpr auto tuple_count_impl(seq<Is...>) { return (number<std::is_same_v<remove_cvref_t<decltype(get<Is>(T{}))>, remove_cvref_t<TargetType>> ? 1 : 0>{} + ...); }
}
// Outer: tuple<tuple<X, X>, tuple<Y>>,  Inner: tuple<tuple<Z>, tuple<W>> => tuple<tuple<X, X, Z>, tuple<Y, W>>
template<typename Outer, typename Inner>
OPUS_H_D constexpr auto embed_nested_tuple(const Outer& ot, const Inner& it) {
    static_assert(size<Outer>() == size<Inner>());
    return impl::embed_nested_tuple_impl(ot, it, make_index_seq<size<Outer>()>{});
}

template< typename TargetType, typename T, std::enable_if_t<is_tuple_v<T>, bool> = true>
OPUS_H_D constexpr index_t tuple_count(const T& t) { return impl::tuple_count_impl<TargetType, remove_cvref_t<T>>(make_index_seq<size<T>()>{}).value; }

template< typename TargetType, typename T, std::enable_if_t<is_tuple_v<T>, bool> = true>
OPUS_H_D constexpr index_t tuple_count() { return impl::tuple_count_impl<TargetType, remove_cvref_t<T>>(make_index_seq<size<T>()>{}).value; }

template<index_t...Is> OPUS_H_D constexpr auto seq_to_tuple(seq<Is...>) { return opus::make_tuple(number<Is>{}...); }

template<index_t...Is>                                             OPUS_H_D constexpr auto to_tuple(seq<Is...>) { return opus::make_tuple(number<Is>{}...); }
template<typename T, std::enable_if_t<is_tuple_v<T>, bool> = true> OPUS_H_D constexpr auto to_tuple(const T& t) { return t; }

namespace impl {
template <typename R, typename T>            OPUS_H_D constexpr auto reduce_tuple_impl(const T& t, seq<>)  { return t; }
template <typename R, typename T, index_t I> OPUS_H_D constexpr auto reduce_tuple_impl(const T& t, seq<I>) { return t; }

template <typename R, typename T, index_t I0, index_t I1, index_t... Is>
OPUS_H_D constexpr auto reduce_tuple_impl(const T& t, seq<I0, I1, Is...>) {
    return reduce_tuple_impl<R>(opus::make_tuple(R{}(get<I0>(t), get<I1>(t)), get<Is>(t)...), make_index_seq<sizeof...(Is) + 1>{});
}
}
template<typename R, typename T, std::enable_if_t<is_tuple_v<T>, bool> = true>
OPUS_H_D constexpr auto reduce_tuple(const T & t) { return  impl::reduce_tuple_impl<R>(t, make_index_seq<size<T>()>{}); }
template<typename T, std::enable_if_t<is_tuple_v<T>, bool> = true> OPUS_H_D constexpr auto reduce_tuple_sum(const T & t) { return reduce_tuple<opus::plus>(t); }
template<typename T, std::enable_if_t<is_tuple_v<T>, bool> = true> OPUS_H_D constexpr auto reduce_tuple_mul(const T & t) { return reduce_tuple<opus::multiplies>(t); }

namespace impl {
template<typename, typename, typename> struct to_peepholed_seq;

template<typename PeepholedTuple, index_t I, index_t...Is, typename max_income_num>  struct to_peepholed_seq<PeepholedTuple, seq<I, Is...>, max_income_num> {
    template<index_t C> OPUS_H_D constexpr auto operator()(number<C>) {
        constexpr auto next_cumulative = std::conditional_t<is_underscore_v<remove_cvref_t<decltype(get<I>(PeepholedTuple{}))>>,
                                            number<(C+1) < max_income_num::value ? (C+1) : C>, number<C>>{};
        return concat_seq(seq<C>{}, to_peepholed_seq<PeepholedTuple, seq<Is...>, max_income_num>{}(next_cumulative) );
    }
};
template<typename PeepholedTuple, index_t I, typename max_income_num>                struct to_peepholed_seq<PeepholedTuple, seq<I>, max_income_num> {
    template<index_t C> OPUS_H_D constexpr auto operator()(number<C>) { return seq<C>{}; }
};

template<typename PeepholedTuple, typename IncomTuple, index_t...Ps,  index_t...Is>
OPUS_H_D constexpr decltype(auto) merge_peepholed_tuple_impl(PeepholedTuple&& pt, IncomTuple&& it, seq<Ps...>, seq<Is...>) {
    return opus::make_tuple([&](){ if constexpr (is_underscore_v<remove_cvref_t<decltype(get<Ps>(pt))>>) return get<Is>(it);
                                   else return get<Ps>(pt);}()... );
}
}
// (Peepholed)tuple<*, *, _, *, _> + (Income)tuple<#, @> -> tuple<*, *, #, *, @>.  "_"(underscore) indicate a peephole for income tuple to chime in
template<typename PeepholedTuple, typename IncomeTuple>
OPUS_H_D constexpr decltype(auto) merge_peepholed_tuple(PeepholedTuple&& pt, IncomeTuple&& it) {
    if constexpr (tuple_count<underscore, PeepholedTuple>() == 0) return pt;
    else {
        constexpr auto income_seq =  impl::to_peepholed_seq< remove_cvref_t<PeepholedTuple>,        make_index_seq<opus::size<PeepholedTuple>()>,
                                                             number<opus::size<IncomeTuple>()> >{}(number<0>{});
        return impl::merge_peepholed_tuple_impl(std::forward<PeepholedTuple>(pt), std::forward<IncomeTuple>(it), make_index_seq<opus::size<PeepholedTuple>()>{}, income_seq);
    }
}
} // namespace opus

// implementing the "tuple-like binding protocol", don't use below directly
namespace std {
template <typename... Ts> struct tuple_size<opus::tuple<Ts...>>       : std::integral_constant<std::size_t, sizeof...(Ts)> {};
template <typename... Ts> struct tuple_size<const opus::tuple<Ts...>> : std::integral_constant<std::size_t, sizeof...(Ts)> {};
template <std::size_t I, typename... Ts> struct tuple_element<I, opus::tuple<Ts...>>       : std::tuple_element<I, std::tuple<Ts...>> {};
template <std::size_t I, typename... Ts> struct tuple_element<I, const opus::tuple<Ts...>> : std::tuple_element<I, const std::tuple<Ts...>> {};
} // namespace std

namespace opus {
/////////////////////////////////////////////////////////////////////////////////////////////////////////
// transforms
template<typename X, typename Y, index_t... Is> constexpr auto embed(const X& x, const Y& y, seq<Is...>) { return ( ... + (get<Is>(x) * get<Is>(y))); }
template<typename X, typename Y>                constexpr auto embed(const X& x, const Y& y) { return embed(x, y, make_index_seq<X::size()>{}); }

namespace impl {
template <typename F, typename X, index_t... Is> OPUS_H_D constexpr auto transform_tuple_impl(F f, const X& x, seq<Is...>) { return opus::make_tuple(f(get<Is>(x))...); }
template <typename F, typename X, index_t... Is> OPUS_H_D constexpr auto transform_tuple_with_idx_impl(F f, const X& x, seq<Is...>) { return opus::make_tuple(f(get<Is>(x), number<Is>{})...); }
} // namespace impl
// f(auto item)
template <typename F, typename X> OPUS_H_D constexpr auto transform_tuple(F f, const X& x) { return impl::transform_tuple_impl(f, x, make_index_seq<size<X>()>{}); }
// f(auto item, auto index)
template <typename F, typename X> OPUS_H_D constexpr auto transform_tuple_with_idx(F f, const X& x) { return impl::transform_tuple_with_idx_impl(f, x, make_index_seq<size<X>()>{}); }

/////////////////////////////////////////////////////////////////////////////////////////////////////////
// layout, simple linear nd layout with stride, static or dynamic supported
namespace impl {
template<typename, typename> struct packed_shape_to_stride_impl;

template<typename Shape, index_t I, index_t... Is> struct packed_shape_to_stride_impl<Shape, seq<I, Is...>>{
    OPUS_H_D constexpr auto operator()(const Shape&shape, seq<I, Is...>) {
        auto r = packed_shape_to_stride_impl<Shape, seq<Is...>>{}(shape, seq<Is...>{});
        return concat_tuple(opus::make_tuple(get<I + 1>(shape) * get<0>(r)), r);
    }
};
template<typename Shape, index_t I> struct packed_shape_to_stride_impl<Shape, seq<I>>{ OPUS_H_D constexpr auto operator()(const Shape& shape, seq<I>) { return opus::make_tuple(number<1>{}); } };
}

template<typename Shape>
OPUS_H_D constexpr auto packed_shape_to_stride(const Shape& shape) { constexpr index_t rank = Shape::size(); return impl::packed_shape_to_stride_impl<Shape, make_index_seq<rank>>{}(shape, make_index_seq<rank>{});     }

template<typename Layout, typename Coord>
OPUS_H_D constexpr decltype(auto) coord_to_linear(const Layout& layout, const Coord& coord) { static_assert(size<decltype(layout.stride())>() == size<Coord>()); return embed(layout.stride(), coord); }

// Shape/Stride/Coord, they are all tuples. if Coord is not false_type, will use merge_peepholed_tuple() to construct real coord
template<typename Shape_, typename Stride_, typename Coord_ = false_type>
struct layout : public tuple<remove_cvref_t<Shape_>, remove_cvref_t<Stride_>, remove_cvref_t<Coord_>> {
    using base   = tuple<remove_cvref_t<Shape_>, remove_cvref_t<Stride_>, remove_cvref_t<Coord_>>;
    using Shape  = remove_cvref_t<Shape_>;
    using Stride = remove_cvref_t<Stride_>;
    using Coord  = remove_cvref_t<Coord_>;  // peepholed coord

    static constexpr index_t rank = Shape::size();
    static_assert(Shape::size() == Stride::size());
    static_assert(std::is_same_v<Coord, false_type> || size<std::conditional_t<std::is_same_v<Coord, false_type>, Shape, Coord>>() == rank, "Coord should be either false_type or a tuple with same size as Shape");
    static constexpr index_t coord_rank = [](){
        if constexpr (std::is_same_v<Coord, false_type>) return rank;
        else          return rank - tuple_count<underscore>(Coord{});
    }();

    OPUS_H_D constexpr layout(const Shape& shape, const Stride& stride, const Coord& coord = {}) : base(shape, stride, coord){}
    OPUS_H_D constexpr layout(Shape&& shape, Stride&& stride, Coord&& coord = {}) : base(shape, stride, coord){}

    // get ith element from shape/stride. if no I, then get the shape/stride as tuple
    template <int... I> OPUS_H_D constexpr decltype(auto) shape()        { return get<0,I...>(static_cast<base&>(*this)); }
    template <int... I> OPUS_H_D constexpr decltype(auto) shape()  const { return get<0,I...>(static_cast<const base&>(*this)); }
    template <int... I> OPUS_H_D constexpr decltype(auto) stride()       { return get<1,I...>(static_cast<base&>(*this)); }
    template <int... I> OPUS_H_D constexpr decltype(auto) stride() const { return get<1,I...>(static_cast<const base&>(*this)); }
    template <int... I> OPUS_H_D constexpr decltype(auto) coord()        { return get<2,I...>(static_cast<base&>(*this)); }
    template <int... I> OPUS_H_D constexpr decltype(auto) coord() const  { return get<2,I...>(static_cast<const base&>(*this)); }

    template <typename... Cs, std::enable_if_t<(!is_tuple_v<Cs> && ...), bool> = true>
    OPUS_H_D constexpr decltype(auto) operator()(Cs&&... cs) const { return this->operator()(opus::make_tuple(std::forward<Cs>(cs)...)); }

    template <typename InCoord, std::enable_if_t<is_tuple_v<InCoord>, bool> = true>
    OPUS_H_D constexpr decltype(auto) operator()(InCoord&& c) const {
        if constexpr (std::is_same_v<Coord, false_type>) return coord_to_linear(*this, c);
        else                                             return coord_to_linear(*this, merge_peepholed_tuple(coord(), c)); }
};

template <typename Layout> struct layout_linear;
template<index_t cached_vec_, typename Layout> struct layout_cached;

// use cached_vec to dispatch which layout implementation. cached_vec < 0 : "layout", cached_vec == 0 : "layout_linear", cached_vec > 0 : "layout_cached"
template <index_t cached_vec = 0, typename Sx, typename Sy> OPUS_H_D constexpr auto make_layout(Sx&& s, Sy&& t) {
    if      constexpr (cached_vec < 0)  return layout<Sx, Sy>(std::forward<Sx>(s), std::forward<Sy>(t));
    else if constexpr (cached_vec == 0) return layout_linear<layout<Sx, Sy>>(std::forward<Sx>(s), std::forward<Sy>(t));
    else                                return layout_cached<cached_vec, layout<Sx, Sy>>(std::forward<Sx>(s), std::forward<Sy>(t)); }
template <index_t cached_vec = 0, typename Sx, typename Sy, typename Sz>
OPUS_H_D constexpr auto                       make_layout(Sx&& s, Sy&& t, Sz&& c) {
    if constexpr (cached_vec < 0)  return layout<Sx, Sy, Sz>(std::forward<Sx>(s), std::forward<Sy>(t), std::forward<Sz>(c));
    if constexpr (cached_vec == 0) return layout_linear<layout<Sx, Sy, Sz>>(std::forward<Sx>(s), std::forward<Sy>(t), std::forward<Sz>(c));
    else                           return layout_cached<cached_vec, layout<Sx, Sy, Sz>>(std::forward<Sx>(s), std::forward<Sy>(t), std::forward<Sz>(c)); }
template <index_t cached_vec = 0, typename... Ts, std::enable_if_t<(!is_tuple_v<Ts> && ...), bool> = true>
OPUS_H_D constexpr auto                       make_layout(Ts&&... ss) { return make_layout<cached_vec>(opus::make_tuple(ss...), packed_shape_to_stride(opus::make_tuple(ss...))); }
template <index_t cached_vec = 0, typename S> OPUS_H_D constexpr auto make_layout(S&& s) { return make_layout<cached_vec>(std::forward<S>(s), packed_shape_to_stride(s)); }

template <index_t cached_vec = 0, typename S> OPUS_H_D constexpr auto               make_layout_packed(S&& s) { return make_layout<cached_vec>(std::forward<S>(s), packed_shape_to_stride(s)); } // same as single arg make_layout
template <index_t cached_vec = 0, typename Sx, typename Sz> OPUS_H_D constexpr auto make_layout_packed(Sx&& s, Sz&& c) { return make_layout<cached_vec>(std::forward<Sx>(s), packed_shape_to_stride(s), std::forward<Sz>(c)); }

template <typename Layout>
struct layout_linear : public remove_cvref_t<Layout>{
    using base = remove_cvref_t<Layout>;

    template<typename Shape, typename Stride, typename Coord = false_type>
    OPUS_H_D constexpr layout_linear(const Shape& shape, const Stride& stride, const Coord& coord = {}) : base(shape, stride, coord), linear_offset(0){}

    template<typename Shape, typename Stride, typename Coord = false_type>
    OPUS_H_D constexpr layout_linear(Shape&& shape, Stride&& stride, Coord&& coord = {}) : base(shape, stride, coord), linear_offset(0){}

    template <typename... Cs, std::enable_if_t<(!is_tuple_v<Cs> && ...), bool> = true>
    OPUS_H_D constexpr decltype(auto) operator()(Cs&&... cs) const { return this->operator()(opus::make_tuple(std::forward<Cs>(cs)...)); }

    template <typename InCoord, std::enable_if_t<is_tuple_v<InCoord>, bool> = true>
    OPUS_H_D constexpr decltype(auto) operator()(InCoord&& c) const {
        if constexpr (std::is_same_v<typename base::Coord, false_type>) return linear_offset + coord_to_linear(*this, c);
        else                                             return linear_offset + coord_to_linear(*this, merge_peepholed_tuple(base::coord(), c)); }

    OPUS_H_D constexpr void inc(index_t offset) { linear_offset += offset; }
    OPUS_H_D constexpr layout_linear& operator+=(index_t offset) { inc(offset); return *this; }

    index_t linear_offset;
};

template <index_t vec, typename Layout> OPUS_H_D constexpr auto layout_to_vectorized_issue_space();
template<index_t vec, typename Layout> OPUS_H_D constexpr auto layout_to_offsets(const Layout& u);

template<index_t cached_vec_, typename Layout>
struct layout_cached : public remove_cvref_t<Layout> {
    using base = remove_cvref_t<Layout>;
    static constexpr index_t cached_vec = cached_vec_;

    static constexpr auto issue_space_vec = layout_to_vectorized_issue_space<cached_vec, base>();
    static constexpr index_t num_issues = get<0>(reduce_tuple_mul(issue_space_vec)).value;

    template<typename Shape, typename Stride, typename Coord = false_type>
    OPUS_H_D constexpr layout_cached(const Shape& shape, const Stride& stride, const Coord& coord = {}) : base(shape, stride, coord), offsets{layout_to_offsets<cached_vec>(static_cast<base>(*this))}{}

    template<typename Shape, typename Stride, typename Coord = false_type>
    OPUS_H_D constexpr layout_cached(Shape&& shape, Stride&& stride, Coord&& coord = {}) : base(shape, stride, coord), offsets{layout_to_offsets<cached_vec>(static_cast<base>(*this))}{}

    template <typename... Cs, std::enable_if_t<(!is_tuple_v<Cs> && ...), bool> = true>
    OPUS_H_D constexpr decltype(auto) operator()(Cs&&... cs) const { return this->operator()(opus::make_tuple(std::forward<Cs>(cs)...)); }

    template <typename InCoord, std::enable_if_t<is_tuple_v<InCoord>, bool> = true>
    OPUS_H_D constexpr decltype(auto) operator()(InCoord&& c) const { constexpr auto u_linear = make_layout<-1>(issue_space_vec); return offsets[u_linear(c)]; }

    OPUS_H_D constexpr void inc(index_t offset) { static_for<num_issues>([&](auto i){ offsets[i] += offset; }); }
    OPUS_H_D constexpr layout_cached& operator+=(index_t offset) { inc(offset); return *this; }

    array<index_t, num_issues> offsets;
};

template<typename T> struct is_layout : false_type {};
template<typename X, typename Y, typename Z> struct is_layout<layout<X, Y, Z>> : true_type {};
template<index_t cached_vec, typename Layout> struct is_layout<layout_cached<cached_vec, Layout>> : true_type {};
template<typename Layout> struct is_layout<layout_linear<Layout>> : true_type {};
template<typename T> constexpr bool is_layout_v = is_layout<remove_cvref_t<T>>::value;

template <typename Layout>
OPUS_H_D constexpr auto layout_to_issue_space() {
    using maybe_coord = std::conditional_t<std::is_same_v<typename Layout::Coord, false_type>, typename Layout::Shape, typename Layout::Coord>;
    using issue_space_y = remove_cvref_t<decltype(pickup_shape(typename Layout::Shape{}, maybe_coord{}, underscore{}))>;
    using single_issue_space = remove_cvref_t<decltype(make_repeated_tuple(number<1>{}, number<size<typename Layout::Shape>()>{}))>;
    using fallback_issue_space_y = std::conditional_t<std::is_same_v<issue_space_y, opus::tuple<>>, single_issue_space, issue_space_y>;
    using issue_space = std::conditional_t<std::is_same_v<typename Layout::Coord, false_type>, single_issue_space, fallback_issue_space_y>;
    return issue_space{};
}

template<typename issue_space, int vec = 1>
OPUS_H_D constexpr auto vectorize_issue_space(issue_space, number<vec> = {}) {
    constexpr index_t vec_from_issue_space = get<size<issue_space>() - 1>(issue_space{}).value;     // here we get the original last dim length(which should be y dim)
    static_assert(vec_from_issue_space % vec == 0, "please make sure requested vec size can be dividable of vec from issue space");

    constexpr auto issue_space_vec = transform_tuple_with_idx([&](auto item, auto index){           // modify the last dim, divide it by vec. Result is still a tuple
        if constexpr (index.value == size<issue_space>() - 1) return number<item.value / vec_from_issue_space>{};
        else                                                  return item;    }, issue_space{});
    return issue_space_vec;
}

template <index_t vec, typename Layout>
OPUS_H_D constexpr auto layout_to_vectorized_issue_space() {
    constexpr auto issue_space = layout_to_issue_space<Layout>();
    constexpr auto issue_space_vec = vectorize_issue_space(issue_space, number<vec>{});
    // static_assert(size<decltype(issue_space_vec)>() == Layout::coord_rank);
    return issue_space_vec;
}

// this function is usually not constexpr. pre-compute all the offset under current layout
template<index_t vec, typename Layout>
OPUS_H_D constexpr auto layout_to_offsets(const Layout& u) {
    constexpr auto issue_space_vec = layout_to_vectorized_issue_space<vec, Layout>();
    constexpr index_t num_issues = get<0>(reduce_tuple_mul(issue_space_vec)).value;
    array<index_t, num_issues> offsets;

    constexpr auto u_linear = make_layout<-1>(issue_space_vec);
    static_ford(issue_space_vec, [&](auto ... ids){ offsets[u_linear(ids...)] = u(ids...); });
    return offsets;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////
// vector, a wrapper for __attribute__((ext_vector_type(*)))
template <typename V_, index_t N_> // V_ must be literal type, otherwise clang ext_vector_type will not recognize
struct vector {
    static constexpr index_t N = N_;
    using value_type           = remove_cvref_t<V_>;
    using type = value_type __attribute__((ext_vector_type(N))); // this is danguous
};

template <typename T, index_t N> using vector_t = typename vector<T, N>::type;

template <typename> struct is_vector : false_type {};
template <typename T, index_t N> struct is_vector<T __attribute((ext_vector_type(N)))> : true_type {};
template <typename T, index_t N> struct is_vector<T __attribute((ext_vector_type(N)))&> : true_type {};
template <typename T, index_t N> struct is_vector<const T __attribute((ext_vector_type(N)))&> : true_type {};
template <typename T, index_t N> struct is_vector<T __attribute((ext_vector_type(N)))&&> : true_type {};
template <typename E> static constexpr bool is_vector_v = is_vector<E>::value;

namespace impl {
template <typename T>            struct vector_traits_impl { using dtype = remove_cvref_t<T>; static constexpr index_t size() { return 1; } };
template <typename T, index_t N> struct vector_traits_impl<T __attribute__((ext_vector_type(N)))> { using dtype = T; static constexpr index_t size() { return N; } };
template <typename T, index_t N> struct vector_traits_impl<array<T, N>> { using dtype = T; static constexpr index_t size() { return N; } };
template <typename... T>         struct vector_traits_impl<tuple<T...>> { using dtype = __type_pack_element<0, T...> /*TODO: use first type*/; static constexpr index_t size() { return sizeof...(T); } };
}
template <typename T> struct vector_traits : public impl::vector_traits_impl<remove_cvref_t<T>> {};

template<typename T> OPUS_H_D constexpr std::enable_if_t<is_vector_v<T>, index_t> size(T&&) {  return vector_traits<T>::size();   /* vector size */}
template<typename T> OPUS_H_D constexpr std::enable_if_t<is_vector_v<T>, index_t> size()    {  return vector_traits<T>::size();   /* vector size */}

namespace impl {
template<typename D, typename...> struct vector_return_type_helper { using type = D; };
template<typename... Types>
struct vector_return_type_helper<void, Types...> : std::common_type<Types...> { static_assert(std::conjunction_v<not_ref_wrapper<Types>...>, "Types cannot contain reference_wrappers when D is void"); };

template<typename D, typename... Types> using vector_return_type = opus::vector_t<typename vector_return_type_helper<D, Types...>::type, sizeof...(Types)>;
}
template<typename D = void, typename... Types> constexpr impl::vector_return_type<D, Types...> make_vector(Types&&... t) { return {std::forward<Types>(t)...}; }

namespace impl {
template <typename T, index_t... Is> OPUS_H_D constexpr auto make_repeated_vector(T&& x, seq<Is...>) { return opus::make_vector((void(Is), std::forward<T>(x))...); }
} // namespace impl
template <index_t N, typename T> OPUS_H_D constexpr auto make_repeated_vector(T&& x) { return impl::make_repeated_vector(std::forward<T>(x), make_index_seq<N>{}); }
template <typename T, index_t N> OPUS_H_D constexpr auto make_repeated_vector(T&& x, number<N>) { return impl::make_repeated_vector(std::forward<T>(x), make_index_seq<N>{}); }

// vector type can't return reference! error: non-const reference cannot bind to vector element
template <index_t I, typename T, std::enable_if_t<is_vector_v<T>, bool> = true> OPUS_H_D constexpr typename vector_traits<T>::dtype get(T const& t) { static_assert(I < vector_traits<T>::size()); return t[I]; }
template <index_t I, typename T, std::enable_if_t<is_vector_v<T>, bool> = true> OPUS_H_D constexpr typename vector_traits<T>::dtype get(T&& t)      { static_assert(I < vector_traits<T>::size()); return t[I]; }

namespace impl {
template <class T0, class T1, index_t... I0, index_t... I1>
OPUS_H_D constexpr auto concat_vector(T0 const& t0, T1 const& t1, seq<I0...>, seq<I1...>) { return opus::make_vector(get<I0>(t0)..., get<I1>(t1)...); }
template <class T0, class T1, class T2, index_t... I0, index_t... I1, index_t...I2>
OPUS_H_D constexpr auto concat_vector(T0 const& t0, T1 const& t1, T2 const& t2, seq<I0...>, seq<I1...>, seq<I2...>) { return opus::make_vector(get<I0>(t0)..., get<I1>(t1)..., get<I2>(t2)...); }
template <class T0, class T1, class T2, class T3, index_t... I0, index_t... I1, index_t...I2, index_t...I3>
OPUS_H_D constexpr auto concat_vector(T0 const& t0, T1 const& t1, T2 const& t2, T3 const& t3, seq<I0...>, seq<I1...>, seq<I2...>, seq<I3...>) { return opus::make_vector(get<I0>(t0)..., get<I1>(t1)..., get<I2>(t2)..., get<I3>(t3)...); }
}
template <class T0> OPUS_H_D  constexpr auto concat_vector(T0 const& t0) { return t0; }
template <class T0, class T1>
OPUS_H_D  constexpr auto concat_vector(T0 const& t0, T1 const& t1) { return impl::concat_vector(t0, t1, make_index_seq<T0::size()>{}, make_index_seq<T1::size()>{}); }
template <class T0, class T1, class T2>
OPUS_H_D  constexpr auto concat_vector(T0 const& t0, T1 const& t1, T2 const& t2) { return impl::concat_vector(t0, t1, t2, make_index_seq<T0::size()>{}, make_index_seq<T1::size()>{}, make_index_seq<T2::size()>{}); }
template <class T0, class T1, class T2, class T3>
OPUS_H_D  constexpr auto concat_vector(T0 const& t0, T1 const& t1, T2 const& t2, T3 const& t3) {
                                            return impl::concat_vector(t0, t1, t2, t3, make_index_seq<T0::size()>{}, make_index_seq<T1::size()>{}, make_index_seq<T2::size()>{}, make_index_seq<T3::size()>{}); }
template <class T0, class T1, class T2, class T3, class T4, class... Ts>
OPUS_H_D  constexpr auto concat_vector(T0 const& t0, T1 const& t1, T2 const& t2, T3 const& t3, T4 const& t4, Ts const&... ts) { return concat_vector(concat_vector(t0, t1, t2, t3), concat_vector(t4, ts...)); }

template <typename T, std::enable_if_t<is_vector_v<T>, bool> = true> OPUS_H_D constexpr void fill(T& a, typename vector_traits<T>::dtype const& value) { static_for<size<T>()>([&](auto i){ a[i.value] = value; }); }
template <typename T, std::enable_if_t<is_vector_v<T>, bool> = true> OPUS_H_D constexpr void clear(T& a) { fill(a, static_cast<typename vector_traits<T>::dtype>(0)); }

namespace impl {
template<typename T, index_t... Is, std::enable_if_t<is_vector_v<T>, bool> = true>
OPUS_H_D constexpr auto to_array_impl(const T& t, seq<Is...>) { return opus::make_array(t[Is]...); }

template<typename T, index_t... Is, std::enable_if_t<is_array_v<T>, bool> = true>
OPUS_H_D constexpr auto to_array_impl(const T& t, seq<Is...>) { return opus::concat_array(to_array_impl(get<Is>(t), make_index_seq< size(get<Is>(T{})) >{})...); }

template<typename T, index_t... Is, std::enable_if_t<is_array_v<T> && !is_vector_v<typename T::value_type>, bool> = true>
OPUS_H_D constexpr vector_t<typename T::value_type, T::size()> to_vector_impl(const T& t, seq<Is...>) { return {get<Is>(t)...}; }

template<typename T, index_t... Is, std::enable_if_t<is_array_v<T> && is_vector_v<typename T::value_type>, bool> = true>
OPUS_H_D constexpr vector_t<typename T::value_type, T::size()> to_vector_impl(const T& t, seq<Is...>) { return opus::concat_vector(to_vector_impl(get<Is>(t))...); }
}

template<typename T, std::enable_if_t<is_vector_v<T>, bool> = true> // vector type to array
OPUS_H_D constexpr auto to_array(const T& t) { return impl::to_array_impl(t, make_index_seq<size<T>()>{}); }

template<typename T, std::enable_if_t<is_array_v<T>, bool> = true>  // array of vector type to array
OPUS_H_D constexpr auto to_array(const T& t) { return impl::to_array_impl(t, make_index_seq<size<T>()>{}); }

template<typename T, std::enable_if_t<is_array_v<T>, bool> = true>
OPUS_H_D constexpr auto to_vector(const T& t) { return impl::to_vector_impl(t, make_index_seq<size<T>()>{}); }

/////////////////////////////////////////////////////////////////////////////////////////////////////////
// slice
namespace impl {
template<typename C, index_t...Is, std::enable_if_t<is_vector_v<C>, bool> = true> OPUS_H_D constexpr auto slice_impl(C&& c, seq<Is...>) { return opus::make_vector(get<Is>(c)...); }
template<typename C, index_t...Is, std::enable_if_t<is_array_v<C>, bool> = true>  OPUS_H_D constexpr auto slice_impl(C&& c, seq<Is...>) { return opus::make_array(get<Is>(c)...); }
template<typename C, index_t...Is, std::enable_if_t<is_tuple_v<C>, bool> = true>  OPUS_H_D constexpr auto slice_impl(C&& c, seq<Is...>) { return opus::make_tuple(get<Is>(c)...); }

template<index_t len, typename C, typename...Ts, std::enable_if_t<is_vector_v<C>, bool> = true>
OPUS_H_D constexpr auto slice_impl_i(C&& c, Ts... ss) { vector_t<typename vector_traits<C>::dtype, len> r;  index_t d = 0;  static_for([&](auto i){r[d++] = c[i]; }, ss...);  return r; }

template<index_t len, typename C, typename...Ts, std::enable_if_t<is_array_v<C>, bool> = true>
OPUS_H_D constexpr auto slice_impl_i(C&& c, Ts... ss) { array<typename C::value_type, len> r;  index_t d = 0;  static_for([&](auto i){r[d++] = c[i]; }, ss...);  return r; }

template<typename C, typename V, index_t...Ds, index_t...Ss, std::enable_if_t<(is_vector_v<C> || is_array_v<C> || is_tuple_v<C>), bool> = true>
OPUS_H_D constexpr auto set_slice_impl(C&& dst_c, V&& src_c, seq<Ds...>, seq<Ss...>) { ((  dst_c[Ds] = src_c[Ss]), ...); }
}

// static/dynamic slice. SS could be either number<x>, or const integer. Note tuple type does not support dynamic slice (ss is integral)
// (1).[end] : 0.... end, (2).[start, end] : start...end, (3).[start, end, step], start...end but with step as interval (default is 1)
template<typename C, typename... S, std::enable_if_t<is_vector_v<C> && (is_constant_v<S> && ...), bool> = true>
OPUS_H_D constexpr auto slice(C&& c, S&&...ss) { return impl::slice_impl(std::forward<C>(c), make_index_seq<(S::value) ...>{}); }

template<index_t len, typename C, typename... S, std::enable_if_t<is_vector_v<C> && (std::is_integral_v<S> && ...), bool> = true>
OPUS_H_D constexpr auto slice(C&& c, S&&...ss) { return impl::slice_impl_i<len>(std::forward<C>(c), ss...); }

template<typename C, typename... S, std::enable_if_t<is_array_v<C> && (is_constant_v<S> && ...), bool> = true>
OPUS_H_D constexpr auto slice(C&& c, S&&...ss) { return impl::slice_impl(std::forward<C>(c), make_index_seq<(S::value) ...>{}); }

template<index_t len, typename C, typename... S, std::enable_if_t<is_array_v<C> && (std::is_integral_v<S> && ...), bool> = true>
OPUS_H_D constexpr auto slice(C&& c, S&&...ss) { return impl::slice_impl_i<len>(std::forward<C>(c), ss...); }

template<typename C, typename... S, std::enable_if_t<is_tuple_v<C> && (is_constant_v<S> && ...), bool> = true>
OPUS_H_D constexpr auto slice(C&& c, S&&...ss) { return impl::slice_impl(std::forward<C>(c), make_index_seq<(S::value) ...>{}); }

template<typename C, typename V, typename... S, std::enable_if_t<(is_vector_v<C> || is_array_v<C> || is_tuple_v<C>) && (is_constant_v<S> && ...), bool> = true>
OPUS_H_D constexpr auto set_slice(C&& dst_c, V&& src_c, S&&...ss) {
    static_assert(std::is_same_v<typename vector_traits<C>::dtype, typename vector_traits<V>::dtype>);
    using dst_seq = make_index_seq<(S::value) ...>;
    return impl::set_slice_impl(std::forward<C>(dst_c), std::forward<V>(src_c), dst_seq{}, make_index_seq<size<dst_seq>()>{});
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////
// dtype, suffix is "_t", and register corresponding ext_vector_type, and a specialization of is_dtype
#define REGISTER_DTYPE(dtype_base_, dtype_impl_)        \
    using dtype_base_ ## _t    = dtype_impl_;           \
    using dtype_base_ ## x1_t  = dtype_base_ ## _t __attribute__((ext_vector_type(1 )));    \
    using dtype_base_ ## x2_t  = dtype_base_ ## _t __attribute__((ext_vector_type(2 )));    \
    using dtype_base_ ## x4_t  = dtype_base_ ## _t __attribute__((ext_vector_type(4 )));    \
    using dtype_base_ ## x8_t  = dtype_base_ ## _t __attribute__((ext_vector_type(8 )));    \
    using dtype_base_ ## x16_t = dtype_base_ ## _t __attribute__((ext_vector_type(16)));    \
    using dtype_base_ ## x32_t = dtype_base_ ## _t __attribute__((ext_vector_type(32)));    \
    using dtype_base_ ## x64_t = dtype_base_ ## _t __attribute__((ext_vector_type(64)));    \
    template<> struct is_dtype<dtype_base_ ## _t> : true_type {};

template<typename T> struct is_dtype : false_type {};
template<typename T> constexpr bool is_dtype_v = is_dtype<remove_cvref_t<T>>::value;    // use this!

REGISTER_DTYPE(fp32, float)
REGISTER_DTYPE(bf16, unsigned short)
REGISTER_DTYPE(fp16, _Float16)
REGISTER_DTYPE(fp8 , _BitInt(8))
REGISTER_DTYPE(bf8 , unsigned _BitInt(8))
REGISTER_DTYPE(i32 , int32_t)
REGISTER_DTYPE(u32 , uint32_t)
REGISTER_DTYPE(i16 , int16_t)
// REGISTER_DTYPE(u16 , uint16_t)
REGISTER_DTYPE(i8  , int8_t)
REGISTER_DTYPE(u8  , uint8_t)

template<typename C, typename... S, std::enable_if_t<is_dtype_v<C> && (is_constant_v<S> && ...), bool> = true>
OPUS_H_D constexpr auto slice(C&& container, S&&...ss) { return container; }    // TODO: fallback slice a normal value does nonthing
/////////////////////////////////////////////////////////////////////////////////////////////////////////
// type cast
OPUS_D bf16_t fp32_to_bf16_rtn_asm(const float& x) {
    union { float f; u32_t i; } u = {x}; constexpr u32_t f32_nan = 0x7fff0000; constexpr u32_t round_bias = 0x7fff; u32x2_t check_nan; u32_t tmp;
    asm volatile("\nv_cmp_u_f32 %0, %2, %2 \nv_bfe_u32 %1, %2, 16, 1 \nv_add3_u32 %1, %2, %1, %3 \nv_cndmask_b32 %2, %1, %4, %0 \nv_lshrrev_b32 %2, 16, %2 \n"
                 : "=s"(check_nan), "+v"(tmp), "+v"(u.f) : "v"(round_bias), "v"(f32_nan));
    return bf16_t(u.i);
}

OPUS_D constexpr auto fp16_to_fp32(const fp16_t& x) { return static_cast<fp32_t>(x); }
OPUS_D constexpr auto fp32_to_fp16(const fp32_t& x) { return static_cast<fp16_t>(x); }
OPUS_D constexpr auto bf16_to_fp32(const bf16_t& x) { union { u32_t i; float f; } u = {static_cast<u32_t>(x) << 16}; return u.f;}
template<index_t rm = OPUS_FP32_to_BF16_DEFAULT> // 0:standard, 1:truncate_with_nan, 2:truncate, 3:standard asm 4:rta_asm(round to nearest away)
OPUS_D constexpr auto fp32_to_bf16(const fp32_t& x, number<rm> = {}) {
    if      constexpr (rm == 1) {u32_t z = __builtin_bit_cast(u32_t, x); return static_cast<bf16_t>(z >> 16) | (!(~z & 0x7f800000) && (z & 0xffff));}
    else if constexpr (rm == 2) {u32_t z = __builtin_bit_cast(u32_t, x); return static_cast<bf16_t>(z >> 16);}
    else if constexpr (rm == 3) { return fp32_to_bf16_rtn_asm(x); }
}

#define OPUS_CAST_DEFINE(d_, s_) template<typename D, typename S, typename... Aux, std::enable_if_t<std::is_same_v<S, s_ ## _t>, bool> = true> \
                                    OPUS_D constexpr decltype(auto) cast(const S& s, Aux&&... aux) { return s_ ## _to_ ## d_(s, std::forward<Aux>(aux)...); }

OPUS_CAST_DEFINE(fp16, fp32)
OPUS_CAST_DEFINE(fp32, fp16)

template<typename D, typename S, typename... Aux, std::enable_if_t<is_vector_v<S>, bool> = true>
OPUS_D constexpr decltype(auto) cast(const S& s, Aux&&... aux) {
    vector_t<D, size<S>()> r; static_for([&](auto i){ r[i.value] = cast<D>(s[i.value], std::forward<Aux>(aux)...); }, number<size<S>()>{}); return r;
}

namespace impl {
template<typename D, typename S, index_t... Is, typename... Aux, std::enable_if_t<is_tuple_v<S>, bool> = true>
OPUS_D constexpr decltype(auto) cast_impl(const S& s, seq<Is...>, Aux&&... aux) { return make_tuple(cast<D>(get<Is>(s), std::forward<Aux>(aux)...)...); }

template<typename D, typename S, index_t... Is, typename... Aux, std::enable_if_t<is_array_v<S>, bool> = true>
OPUS_D constexpr decltype(auto) cast_impl(const S& s, seq<Is...>, Aux&&... aux) { return make_array(cast<D>(get<Is>(s), std::forward<Aux>(aux)...)...); }
}

template<typename D, typename S, typename... Aux, std::enable_if_t<is_tuple_v<S>, bool> = true>
OPUS_D constexpr decltype(auto) cast(const S& s, Aux&&... aux) { return impl::cast_impl<D>(s, make_index_seq<size<S>()>{}, std::forward<Aux>(aux)...); }

template<typename D, typename S, typename... Aux, std::enable_if_t<is_array_v<S>, bool> = true>
OPUS_D constexpr decltype(auto) cast(const S& s, Aux&&... aux) { return impl::cast_impl<D>(s, make_index_seq<size<S>()>{}, std::forward<Aux>(aux)...); }

#undef OPUS_CAST_DEFINE
/////////////////////////////////////////////////////////////////////////////////////////////////////////
// arch
OPUS_H_D constexpr index_t get_warp_size()
{
#if defined(__GFX9__) || !defined(__HIP_DEVICE_COMPILE__)
    return 64;
#else
    return 32;
#endif
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////
// math
template <typename T, int dpp_i, int row_mask = 0xf, int bank_mask = 0xf, bool bound_ctrl = true>
OPUS_D T mov_dpp(T x, number<dpp_i>, number<row_mask> = {}, number<bank_mask> = {}, bool_constant<bound_ctrl> = {}) {
    static_assert(sizeof(T) == 4); return __builtin_bit_cast(T, __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, x), dpp_i, row_mask, bank_mask, bound_ctrl));
}

template<typename O, typename T, int dpp_i, int row_mask = 0xf, int bank_mask = 0xf, bool bound_ctrl = true>
OPUS_D T upd_dpp(const O& old, T x, number<dpp_i>, number<row_mask> = {}, number<bank_mask> = {}, bool_constant<bound_ctrl> = {}) {
    static_assert(sizeof(T) == 4); return __builtin_bit_cast(T, __builtin_amdgcn_update_dpp(__builtin_bit_cast(int, old), __builtin_bit_cast(int, x), dpp_i, row_mask, bank_mask, bound_ctrl));
}

template<typename T> OPUS_D T max(const T&a, const T&b)                { return a > b ? a : b; }
template<> OPUS_D float       max<float>(const float&a, const float&b) { return __builtin_fmaxf(a, b); }
template<typename T> OPUS_D T min(const T&a, const T&b)                { return a > b ? b : a; }
template<> OPUS_D float       min<float>(const float&a, const float&b) { return __builtin_fminf(a, b); }

template<typename T> OPUS_D T med3(const T&a, const T&b, const T&c) { auto max_0 = max(a, b); auto min_0 = max(a, b); return max(max_0, max(min_0, c)); }
template<> OPUS_D float       med3<float>(const float&a, const float&b, const float&c) { return __builtin_amdgcn_fmed3f(a, b, c); }
/////////////////////////////////////////////////////////////////////////////////////////////////////////
// buffer load/store related
OPUS_D constexpr auto buffer_default_config() {
#if defined(__gfx928__) || defined(__gfx936__) || defined(__gfx938__)
    return 0x00020000;
#else
    return 0xffffffff;
#endif
}

OPUS_D __amdgpu_buffer_rsrc_t make_buffer_rsrc(const void* ptr, uint32_t size = 0xffffffff, uint32_t config = buffer_default_config()) {
    return __builtin_amdgcn_make_buffer_rsrc(const_cast<void*>(static_cast<const void*>(ptr)), 0, size, config); // void *p, short stride, int num, int flags
}
template<typename T_>
struct gmem {
    using T = remove_cvref_t<T_>;
    using scalar_type = typename vector_traits<T>::dtype;
    static constexpr index_t vector_size = vector_traits<T>::size();
    template<index_t vec = 1> using vector_type = vector_t<scalar_type, vec * vector_size>;

    OPUS_D gmem(const void* ptr, uint32_t size = 0xffffffff, uint32_t config = buffer_default_config()) : cached_rsrc(make_buffer_rsrc(ptr, size, config)) {}

    template<index_t vec = 1, index_t aux = 0>   // os in unit of byte
    OPUS_D auto _load(int v_os, int s_os = 0, number<aux> = {}) {
        using type = vector_type<vec>;
        if      constexpr (sizeof(type) == 1)  { return __builtin_bit_cast(type, __builtin_amdgcn_raw_buffer_load_b8  (cached_rsrc, v_os, s_os, aux)); }
        else if constexpr (sizeof(type) == 2)  { return __builtin_bit_cast(type, __builtin_amdgcn_raw_buffer_load_b16 (cached_rsrc, v_os, s_os, aux)); }
        else if constexpr (sizeof(type) == 4)  { return __builtin_bit_cast(type, __builtin_amdgcn_raw_buffer_load_b32 (cached_rsrc, v_os, s_os, aux)); }
        else if constexpr (sizeof(type) == 8)  { return __builtin_bit_cast(type, __builtin_amdgcn_raw_buffer_load_b64 (cached_rsrc, v_os, s_os, aux)); }
        else if constexpr (sizeof(type) == 16) { return __builtin_bit_cast(type, __builtin_amdgcn_raw_buffer_load_b128(cached_rsrc, v_os, s_os, aux)); }
    }

    template<index_t vec = 1, typename V, index_t aux = 0>   // os in unit of byte
    OPUS_D void _store(const V& x, int v_os, int s_os = 0, number<aux> = {}) {
        static_assert((vec * vector_size) == vector_traits<V>::size(), "vector size need to be same, please check");
        if      constexpr (sizeof(vector_type<vec>) == 1)  { __builtin_amdgcn_raw_buffer_store_b8  (__builtin_bit_cast(i8_t,    x), cached_rsrc, v_os, s_os, aux); }
        else if constexpr (sizeof(vector_type<vec>) == 2)  { __builtin_amdgcn_raw_buffer_store_b16 (__builtin_bit_cast(i16_t,   x), cached_rsrc, v_os, s_os, aux); }
        else if constexpr (sizeof(vector_type<vec>) == 4)  { __builtin_amdgcn_raw_buffer_store_b32 (__builtin_bit_cast(i32_t,   x), cached_rsrc, v_os, s_os, aux); }
        else if constexpr (sizeof(vector_type<vec>) == 8)  { __builtin_amdgcn_raw_buffer_store_b64 (__builtin_bit_cast(i32x2_t, x), cached_rsrc, v_os, s_os, aux); }
        else if constexpr (sizeof(vector_type<vec>) == 16) { __builtin_amdgcn_raw_buffer_store_b128(__builtin_bit_cast(i32x4_t, x), cached_rsrc, v_os, s_os, aux); }
    }

    template<index_t vec = 1, index_t aux = 0>   // os in unit of T and cast to vector with vec
    OPUS_D auto load(int v_os, int s_os = 0, number<aux> = {}) { return _load<vec>(v_os * sizeof(T), s_os * sizeof(T), number<aux>{}); }

    template<index_t vec = 1, typename V, index_t aux = 0, std::enable_if_t<(is_vector_v<V> || is_dtype_v<V> || is_array_v<V>), bool> = true>   // os in unit of T and cast to vector with vec
    OPUS_D void store(const V& x, int v_os, int s_os = 0, number<aux> = {}) {
        static_assert(std::is_same_v<typename vector_traits<V>::dtype, scalar_type>, "scalar type must be same for the data to be stored" );
        if constexpr (is_dtype_v<V> && (vec * vector_size) % vector_traits<V>::size() == 0) {
            _store<vec>(make_repeated_vector(x, number<vec * vector_size / vector_traits<V>::size()>{}), v_os * sizeof(T));
        } else {
            static_assert((vec * vector_size) == vector_traits<V>::size(), "vector size need to be same, please check" );
            _store<vec>(x, v_os * sizeof(T));
        }
    }

    // bulk load API, give me a Shape of this tile, will issue multiple load instruction based on the y-shape space
    template<index_t vec = 1, typename Layout, index_t aux = 0, std::enable_if_t<is_layout_v<Layout>, bool> = true>
    OPUS_D auto load(const Layout& u, int s_os = 0/* do we really need this? */, number<aux> = {})
    {
        constexpr auto issue_space = layout_to_issue_space<Layout>();
        constexpr auto issue_space_vec = vectorize_issue_space(issue_space, number<vec>{});
        constexpr auto r_elem = get<0>(reduce_tuple_mul(issue_space_vec));

#if OPUS_TILE_CONTAINER == 0
        constexpr auto u_r = make_layout<-1>(issue_space);                      // we use this layout to describe the register layout
        vector_t<scalar_type, vec * vector_size * r_elem.value> r;          // local scratch to host the loaded register, and return it
        static_ford(issue_space_vec, [&](auto ... ids){
            auto tmp = load<vec>(u(ids...), s_os, number<aux>{});
            constexpr index_t u_rs = u_r(ids...);
            set_slice(r, tmp, number<u_rs>{}, number<u_rs + vec>{});
        });
        return r;
#elif OPUS_TILE_CONTAINER == 1
        constexpr auto u_r = make_layout<-1>(issue_space_vec);                      // we use this layout to describe the register layout
        array<vector_type<vec>, r_elem.value> r;                                      // local scratch to host the loaded register, and return it
        static_ford(issue_space_vec, [&](auto ... ids){ r[u_r(ids...)] = load<vec>(u(ids...), s_os, number<aux>{}); }); // issue the loading instruction multiple times
        return r;
#endif
    }

    template<index_t vec = 1, typename V, typename Layout, index_t aux = 0, std::enable_if_t<((is_array_v<V> || is_vector_v<V>) && is_layout_v<Layout>), bool> = true>
    OPUS_D void store(const V& x, const Layout& u, int s_os = 0/* do we really need this? */, number<aux> = {})
    {
        constexpr auto issue_space = layout_to_issue_space<Layout>();
        constexpr auto issue_space_vec = vectorize_issue_space(issue_space, number<vec>{});

        constexpr auto u_r = make_layout<-1>(issue_space);                      // we use this layout to describe the register layout
#if OPUS_TILE_CONTAINER == 0
        auto a_ = [&](){ if constexpr (is_array_v<V>) return to_vector(x);
                         else if constexpr (is_dtype_v<V>) return make_repeated_vector(x, number<get<0>(reduce_tuple_mul(issue_space)).value>{});
                         else if constexpr (is_vector_v<V>) return x; }();
#elif OPUS_TILE_CONTAINER == 1
        auto a_ = to_array(x);
#endif
        static_ford(issue_space_vec, [&](auto ... ids){ // issue the loading instruction multiple times
            auto v_ = slice(a_, number<u_r(ids...)>{}, number<u_r(ids...) + vec>{});
            store<vec>(v_, u(ids...), s_os, number<aux>{});
        });
    }
    __amdgpu_buffer_rsrc_t cached_rsrc;
};

template<typename T_> OPUS_D decltype(auto) make_gmem(const T_* ptr, uint32_t size = 0xffffffff, uint32_t config = buffer_default_config()) { return gmem<T_>{ptr, size, config}; }
/////////////////////////////////////////////////////////////////////////////////////////////////////////
// smem load/store related. TODO: tr_load
template<typename T_>
struct smem {
    using T = remove_cvref_t<T_>;
    using scalar_type = typename vector_traits<T>::dtype;
    static constexpr index_t vector_size = vector_traits<T>::size();
    template<index_t vec = 1> using vector_type = vector_t<scalar_type, vec * vector_size>;

    OPUS_D smem(void* ptr_) : ptr(reinterpret_cast<char*>(ptr_)) {}

    template<index_t vec = 1> OPUS_D auto _load(int v_os/* in unit of byte*/) { using type = vector_type<vec>; return *reinterpret_cast<type*>(ptr + v_os); }

    template<index_t vec = 1, typename V>
    OPUS_D void _store(const V& x, int v_os/* in unit of byte*/) {
        static_assert((vec * vector_size) == vector_traits<V>::size(), "vector size need to be same, please check");
        using type = vector_type<vec>;
        *reinterpret_cast<type*>(ptr + v_os) = __builtin_bit_cast(type, x);
    }

    template<index_t vec = 1> OPUS_D auto load(int v_os) { return _load<vec>(v_os * sizeof(T)); }

    template<index_t vec = 1, typename V, std::enable_if_t<(is_vector_v<V> || is_dtype_v<V> || is_array_v<V>), bool> = true>
    OPUS_D void store(const V& x, int v_os) {
        static_assert(std::is_same_v<typename vector_traits<V>::dtype, scalar_type>, "scalar type must be same for the data to be stored" );
        if constexpr (is_dtype_v<V> && (vec * vector_size) % vector_traits<V>::size() == 0) {
            _store<vec>(make_repeated_vector(x, number<vec * vector_size / vector_traits<V>::size()>{}), v_os * sizeof(T));
        } else {
            static_assert((vec * vector_size) == vector_traits<V>::size(), "vector size need to be same, please check" );
            _store<vec>(x, v_os * sizeof(T));
        }
    }

    // bulk load API, give me a Shape of this tile, will issue multiple load instruction based on the y-shape space
    template<index_t vec = 1, typename Layout, std::enable_if_t<is_layout_v<Layout>, bool> = true>
    OPUS_D auto load(const Layout& u)
    {
        constexpr auto issue_space = layout_to_issue_space<Layout>();
        constexpr auto issue_space_vec = vectorize_issue_space(issue_space, number<vec>{});
        constexpr auto r_elem = get<0>(reduce_tuple_mul(issue_space_vec));

#if OPUS_TILE_CONTAINER == 0
        constexpr auto u_r = make_layout<-1>(issue_space);                      // we use this layout to describe the register layout
        vector_t<scalar_type, vec * vector_size * r_elem.value> r;          // local scratch to host the loaded register, and return it
        static_ford(issue_space_vec, [&](auto ... ids){
            auto tmp = load<vec>(u(ids...));
            constexpr index_t u_rs = u_r(ids...);
            set_slice(r, tmp, number<u_rs>{}, number<u_rs + vec>{});
        });
        return r;
#elif OPUS_TILE_CONTAINER == 1
        constexpr auto u_r = make_layout<-1>(issue_space_vec);                      // we use this layout to describe the register layout
        array<vector_type<vec>, r_elem.value> r;                                      // local scratch to host the loaded register, and return it
        static_ford(issue_space_vec, [&](auto ... ids){ r[u_r(ids...)] = load<vec>(u(ids...)); }); // issue the loading instruction multiple times
        return r;
#endif
    }

    template<index_t vec = 1, typename V, typename Layout, std::enable_if_t<((is_array_v<V> || is_dtype_v<V> || is_vector_v<V>) && is_layout_v<Layout>), bool> = true>
    OPUS_D void store(const V& x, const Layout& u)
    {
        constexpr auto issue_space = layout_to_issue_space<Layout>();
        constexpr auto issue_space_vec = vectorize_issue_space(issue_space, number<vec>{});

        constexpr auto u_r = make_layout<-1>(issue_space);                      // we use this layout to describe the register layout
#if OPUS_TILE_CONTAINER == 0
        auto a_ = [&](){ if constexpr (is_array_v<V>) return to_vector(x);
                         else if constexpr (is_dtype_v<V>) return make_repeated_vector(x, number<get<0>(reduce_tuple_mul(issue_space)).value>{});
                         else if constexpr (is_vector_v<V>) return x; }();
#elif OPUS_TILE_CONTAINER == 1
        auto a_ = to_array(x);
#endif
        static_ford(issue_space_vec, [&](auto ... ids){ // issue the loading instruction multiple times
            auto v_ = slice(a_, number<u_r(ids...)>{}, number<u_r(ids...) + vec>{});
            store<vec>(v_, u(ids...));
        });
    }
    char * ptr; // in unit of byte
};

template<typename T_> OPUS_D decltype(auto) make_smem(T_* ptr) { return smem<T_>{ptr}; }
/////////////////////////////////////////////////////////////////////////////////////////////////////////
// waitcnt
// vmcnt=0~63([15:14],[3:0]), lgkmcnt=0~15([11:8]), expcnt=0~7([6:4])
template <index_t vmcnt, index_t lgkmcnt, index_t expcnt = 7>
OPUS_D void s_waitcnt(number<vmcnt>, number<lgkmcnt>, number<expcnt> = {})
{   __builtin_amdgcn_s_waitcnt((((0b110000 & vmcnt) << (14 - 4)) | (0b1111 & vmcnt)) | ((0b111 & expcnt) << 4) | ((0b1111 & lgkmcnt) << 8)); }

template <index_t vmcnt>   OPUS_D void s_waitcnt_vmcnt(number<vmcnt>) { s_waitcnt(number<vmcnt>{}, number<15>{}); }
template <index_t lgkmcnt> OPUS_D void s_waitcnt_lgkmcnt(number<lgkmcnt>) { s_waitcnt(number<63>{}, number<lgkmcnt>{}); }

/////////////////////////////////////////////////////////////////////////////////////////////////////////
// adaptor
struct p_dim {};
struct y_dim {};

namespace impl{ // utlity function to play with shape
template<typename Shape, typename FDim, typename Target, index_t... Is>
OPUS_D static constexpr auto pickup_shape_impl(const Shape&, const FDim&, Target, seq<Is...>) {
    static_assert(size<Shape>() == size<FDim>());
    return concat_tuple(std::conditional_t< std::is_same_v<decltype(get<Is>(FDim{})), remove_cvref_t<Target>>,  tuple<decltype(get<Is>(Shape{}))>,  tuple<> >{}...);
}

template<typename Shape, typename Dim, index_t SStart, index_t DIdx, index_t... Ss /* index for Dim not Shape */>
OPUS_D constexpr auto unflatten_shape_impl(const Shape&, const Dim&, number<SStart>, number<DIdx>, seq<Ss...>) {
    if constexpr((DIdx + 1) < size<Dim>()) return concat_tuple(opus::make_tuple(opus::make_tuple(get<SStart + Ss>(Shape{})... )),
                                                    unflatten_shape_impl(Shape{}, Dim{}, number<SStart + sizeof...(Ss)>{}, number<DIdx + 1>{}, make_index_seq<get<DIdx + 1>(Dim{}).size()>{}));
    else /* last one */                    return opus::make_tuple(opus::make_tuple(get<SStart + Ss>(Shape{})... ));
}

template<typename Dim, typename Coord, index_t C, index_t I>
OPUS_D constexpr auto unfold_p_coord_impl(const Dim& dim, const Coord& coord, number<C>, seq<I>) {
    constexpr auto is_p = std::is_same_v<remove_cvref_t<decltype(get<I>(Dim{}))>, p_dim>;
    auto get_c = [&]() { if constexpr(is_p) return opus::make_tuple(get<C>(coord));
                         else               return tuple<underscore>{}; };
    return get_c();
}

template<typename Dim, typename Coord, index_t C, index_t I, index_t...Is>
OPUS_D constexpr auto unfold_p_coord_impl(const Dim&, const Coord& coord, number<C>, seq<I, Is...>) {
    constexpr auto is_p = std::is_same_v<remove_cvref_t<decltype(get<I>(Dim{}))>, p_dim>;
    auto get_c = [&]() { if constexpr(is_p) return opus::make_tuple(get<C>(coord));
                         else               return tuple<underscore>{}; };
    constexpr auto next_c = std::conditional_t<is_p, number<C+1>, number<C>>{};
    return concat_tuple(get_c(), unfold_p_coord_impl(Dim{}, coord, next_c, seq<Is...>{}) );
}

template<typename Dim, typename Shape, typename Stride, index_t C, index_t I, index_t...Is>
OPUS_D constexpr auto unfold_x_stride_impl(const Dim&, const Shape&, const Stride & stride, number<C>, seq<I, Is...>) {
    constexpr index_t current_x_dim_length = size<decltype( get<I>(Dim{}) )>();
    constexpr auto current_shape = slice(Shape{}, number<C>{}, number<C+current_x_dim_length>{});
    constexpr auto current_stride = packed_shape_to_stride(current_shape);
    auto scaled_stride = transform_tuple([&](auto i_elem){
        return i_elem * get<I>(stride);
    }, current_stride);
    if constexpr (sizeof...(Is) == 0) return scaled_stride; // last one
    else return concat_tuple(scaled_stride, unfold_x_stride_impl(Dim{}, Shape{}, stride, number<C+current_x_dim_length>{}, seq<Is...>{}));
}
}

template<typename Shape, typename Dim, typename Target>
OPUS_D static constexpr auto pickup_shape(const Shape&, const Dim&, Target) { return pickup_shape_impl(Shape{}, flatten_tuple(Dim{}), Target{}, make_index_seq<size<Shape>()>{}); }

// Shape : tuple<N0, N1, N2, N3, N4, N5>
// Dim   : tuple<tuple<*, *>, tuple<*, *, *>, tuple<*>>
// =>    : tuple<tuple<N0, N1>, tuple<N2, N3, N4>, tuple<N5>>
template<typename Shape, typename Dim, index_t... Ds /* index for Dim not Shape */>
OPUS_D constexpr auto unflatten_shape(const Shape&, const Dim&) {
    return unflatten_shape_impl(Shape{}, Dim{}, number<0>{}, number<0>{}, make_index_seq<get<0>(Dim{}).size()>{});
}

// coord: tuple<a, b>, dim: tuple<tuple<p_dim, y_dim>, tuple<y_dim, p_dim, y_dim>> -> tuple <a, _, _, b, _>
template<typename Dim, typename Coord>
OPUS_D constexpr auto unfold_p_coord(const Dim&, const Coord& coord) {
    constexpr auto flatten_dim = flatten_tuple(Dim{});
    static_assert(tuple_count<opus::p_dim>(flatten_dim) == size<Coord>(), "input coord must be same size as p_dim inside Dim");
    return unfold_p_coord_impl(flatten_dim, coord, number<0>{}, make_index_seq<size<decltype(flatten_dim)>()>{});
}

template<typename Dim, typename Shape, typename Stride>
OPUS_D constexpr auto unfold_x_stride(const Dim&, const Shape&, const Stride& stride) {
    constexpr auto flatten_dim = flatten_tuple(Dim{});
    static_assert(size<Dim>() == size<Stride>(), "input stride must be same size as x_dim");
    static_assert(size<Shape>() == size<remove_cvref_t<decltype(flatten_dim)>>(), "input shape must be same size as flattened dim");
    return unfold_x_stride_impl(Dim{}, Shape{}, stride, number<0>{}, make_index_seq<size<Dim>()>{});
}

#define OPUS_KP_(x_) static_assert(opus::tuple_count<opus::p_dim>(opus::flatten_tuple(x_ ())) == size<C>())
// any struct implement adaptor like feature must implement(or using from base) shape_a/b/c, dim_a/b/c
#define OPUS_ADAPTOR_LAYOUT_API_DEFINE                                                                                                                              \
    template<typename S, typename D> OPUS_D static constexpr auto y_shape(const S& /*shape*/, const D& /*dim*/) { return opus::pickup_shape(S{}, D{}, y_dim{}); }   \
    template<typename S, typename D> OPUS_D static constexpr auto p_shape(const S& /*shape*/, const D& /*dim*/) { return opus::pickup_shape(S{}, D{}, p_dim{}); }   \
                                                                                               \
    OPUS_D static constexpr auto y_shape_a() { return y_shape(shape_a(), dim_a()); }           \
    OPUS_D static constexpr auto y_shape_b() { return y_shape(shape_b(), dim_b()); }           \
    OPUS_D static constexpr auto y_shape_c() { return y_shape(shape_c(), dim_c()); }           \
                                                                                               \
    OPUS_D static constexpr auto p_shape_a() { return p_shape(shape_a(), dim_a()); }           \
    OPUS_D static constexpr auto p_shape_b() { return p_shape(shape_b(), dim_b()); }           \
    OPUS_D static constexpr auto p_shape_c() { return p_shape(shape_c(), dim_c()); }           \
                                                                                               \
    template<index_t cached_vec = 0> OPUS_D constexpr auto layout_a() { return make_layout<cached_vec>(shape_a());}                         \
    template<index_t cached_vec = 0> OPUS_D constexpr auto layout_b() { return make_layout<cached_vec>(shape_b());}                         \
    template<index_t cached_vec = 0> OPUS_D constexpr auto layout_c() { return make_layout<cached_vec>(shape_c());}                         \
                                                                                                                                            \
    template<index_t cached_vec = 0, typename S> OPUS_D constexpr auto layout_a(S&& stride) { return make_layout<cached_vec>(shape_a(), unfold_x_stride(dim_a(), shape_a(), stride));} \
    template<index_t cached_vec = 0, typename S> OPUS_D constexpr auto layout_b(S&& stride) { return make_layout<cached_vec>(shape_b(), unfold_x_stride(dim_b(), shape_b(), stride));} \
    template<index_t cached_vec = 0, typename S> OPUS_D constexpr auto layout_c(S&& stride) { return make_layout<cached_vec>(shape_c(), unfold_x_stride(dim_c(), shape_c(), stride));} \
    /* Note, all the coord passed in must be p_coord*/                                                                                      \
    template<index_t cached_vec = 0, typename S, typename C> OPUS_D constexpr auto layout_a(S&& stride, C&& z) { OPUS_KP_(dim_a); return make_layout<cached_vec>(shape_a(), unfold_x_stride(dim_a(), shape_a(), stride), opus::unfold_p_coord(dim_a(), z));}  \
    template<index_t cached_vec = 0, typename S, typename C> OPUS_D constexpr auto layout_b(S&& stride, C&& z) { OPUS_KP_(dim_b); return make_layout<cached_vec>(shape_b(), unfold_x_stride(dim_b(), shape_b(), stride), opus::unfold_p_coord(dim_b(), z));}  \
    template<index_t cached_vec = 0, typename S, typename C> OPUS_D constexpr auto layout_c(S&& stride, C&& z) { OPUS_KP_(dim_c); return make_layout<cached_vec>(shape_c(), unfold_x_stride(dim_c(), shape_c(), stride), opus::unfold_p_coord(dim_c(), z));}  \
                                                                                                                                                                                                        \
    template<index_t cached_vec = 0, typename C> OPUS_D constexpr auto layout_a_packed(C&& z) { OPUS_KP_(dim_a); return make_layout_packed<cached_vec>(shape_a(), opus::unfold_p_coord(dim_a(), z));}   \
    template<index_t cached_vec = 0, typename C> OPUS_D constexpr auto layout_b_packed(C&& z) { OPUS_KP_(dim_b); return make_layout_packed<cached_vec>(shape_b(), opus::unfold_p_coord(dim_b(), z));}   \
    template<index_t cached_vec = 0, typename C> OPUS_D constexpr auto layout_c_packed(C&& z) { OPUS_KP_(dim_c); return make_layout_packed<cached_vec>(shape_c(), opus::unfold_p_coord(dim_c(), z));}   \
                                                                                                                                                                                                        \
    template<index_t cached_vec = 0, typename... Ts, std::enable_if_t<(!is_tuple_v<Ts> && ...), bool> = true> OPUS_D constexpr auto layout_a(Ts&&... strides) {return layout_a<cached_vec>(opus::make_tuple(strides...)); }  \
    template<index_t cached_vec = 0, typename... Ts, std::enable_if_t<(!is_tuple_v<Ts> && ...), bool> = true> OPUS_D constexpr auto layout_b(Ts&&... strides) {return layout_b<cached_vec>(opus::make_tuple(strides...)); }  \
    template<index_t cached_vec = 0, typename... Ts, std::enable_if_t<(!is_tuple_v<Ts> && ...), bool> = true> OPUS_D constexpr auto layout_c(Ts&&... strides) {return layout_c<cached_vec>(opus::make_tuple(strides...)); }  \
                                                                                                                                    \
    template<index_t cached_vec = 0> OPUS_D constexpr auto y_layout_a() { return make_layout<cached_vec>(y_shape_a());}             \
    template<index_t cached_vec = 0> OPUS_D constexpr auto y_layout_b() { return make_layout<cached_vec>(y_shape_b());}             \
    template<index_t cached_vec = 0> OPUS_D constexpr auto y_layout_c() { return make_layout<cached_vec>(y_shape_c());}

#undef OPUS_KP_
// clang-format on
} // namespace