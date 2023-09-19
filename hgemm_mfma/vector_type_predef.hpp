template <typename T, index_t N>
struct vector_type;

// clang-format off
template <typename T> struct vector_type<T, 1> {
    using d1_t = T;
    using type = d1_t;
    union {
        type d1_;
        static_buffer<d1_t, 1> d1x1_;
    } data_;
    DEVICE_HOST constexpr vector_type() : data_{type{0}} {}
    DEVICE_HOST constexpr vector_type(type v) : data_{v} {}
    template<typename VEC> DEVICE_HOST constexpr const auto& to_varray() const { return data_.d1_; }
    template<typename VEC> DEVICE_HOST constexpr auto&       to_varray()       { return data_.d1_; }
    template<> DEVICE_HOST constexpr const auto& to_varray<d1_t>() const { return data_.d1x1_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d1_t>()       { return data_.d1x1_;}
};

template <typename T> struct vector_type<T, 2> {
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    using type = d2_t;
    union {
        type d2_;
        static_buffer<d1_t, 2> d1x2_;
        static_buffer<d2_t, 1> d2x1_;
    } data_;
    DEVICE_HOST constexpr vector_type() : data_{type{0}} {}
    DEVICE_HOST constexpr vector_type(type v) : data_{v} {}
    template<typename VEC> DEVICE_HOST constexpr const auto& to_varray() const { return data_.d2_; }
    template<typename VEC> DEVICE_HOST constexpr auto&       to_varray()       { return data_.d2_; }
    template<> DEVICE_HOST constexpr const auto& to_varray<d1_t>() const { return data_.d1x2_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d1_t>()       { return data_.d1x2_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d2_t>() const { return data_.d2x1_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d2_t>()       { return data_.d2x1_;}
};

template <typename T> struct vector_type<T, 4> {
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d4_t __attribute__((ext_vector_type(4)));
    using type = d4_t;
    union {
        type d4_;
        static_buffer<d1_t, 4> d1x4_;
        static_buffer<d2_t, 2> d2x2_;
        static_buffer<d4_t, 1> d4x1_;
    } data_;
    DEVICE_HOST constexpr vector_type() : data_{type{0}} {}
    DEVICE_HOST constexpr vector_type(type v) : data_{v} {}
    template<typename VEC> DEVICE_HOST constexpr const auto& to_varray() const { return data_.d4_; }
    template<typename VEC> DEVICE_HOST constexpr auto&       to_varray()       { return data_.d4_; }
    template<> DEVICE_HOST constexpr const auto& to_varray<d1_t>() const { return data_.d1x4_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d1_t>()       { return data_.d1x4_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d2_t>() const { return data_.d2x2_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d2_t>()       { return data_.d2x2_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d4_t>() const { return data_.d4x1_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d4_t>()       { return data_.d4x1_;}
};

template <typename T> struct vector_type<T, 8> {
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d4_t __attribute__((ext_vector_type(4)));
    typedef T d8_t __attribute__((ext_vector_type(8)));
    using type = d8_t;
    union {
        type d8_;
        static_buffer<d1_t, 8> d1x8_;
        static_buffer<d2_t, 4> d2x4_;
        static_buffer<d4_t, 2> d4x2_;
        static_buffer<d8_t, 1> d8x1_;
    } data_;
    DEVICE_HOST constexpr vector_type() : data_{type{0}} {}
    DEVICE_HOST constexpr vector_type(type v) : data_{v} {}
    template<typename VEC> DEVICE_HOST constexpr const auto& to_varray() const { return data_.d8_; }
    template<typename VEC> DEVICE_HOST constexpr auto&       to_varray()       { return data_.d8_; }
    template<> DEVICE_HOST constexpr const auto& to_varray<d1_t>() const { return data_.d1x8_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d1_t>()       { return data_.d1x8_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d2_t>() const { return data_.d2x4_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d2_t>()       { return data_.d2x4_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d4_t>() const { return data_.d4x2_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d4_t>()       { return data_.d4x2_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d8_t>() const { return data_.d8x1_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d8_t>()       { return data_.d8x1_;}
};

template <typename T> struct vector_type<T, 16> {
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d4_t __attribute__((ext_vector_type(4)));
    typedef T d8_t __attribute__((ext_vector_type(8)));
    typedef T d16_t __attribute__((ext_vector_type(16)));
    using type = d16_t;
    union {
        type d16_;
        static_buffer<d1_t, 16> d1x16_;
        static_buffer<d2_t, 8> d2x8_;
        static_buffer<d4_t, 4> d4x4_;
        static_buffer<d8_t, 2> d8x2_;
        static_buffer<d16_t, 1> d16x1_;
    } data_;
    DEVICE_HOST constexpr vector_type() : data_{type{0}} {}
    DEVICE_HOST constexpr vector_type(type v) : data_{v} {}
    template<typename VEC> DEVICE_HOST constexpr const auto& to_varray() const { return data_.d16_; }
    template<typename VEC> DEVICE_HOST constexpr auto&       to_varray()       { return data_.d16_; }
    template<> DEVICE_HOST constexpr const auto& to_varray<d1_t>() const { return data_.d1x16_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d1_t>()       { return data_.d1x16_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d2_t>() const { return data_.d2x8_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d2_t>()       { return data_.d2x8_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d4_t>() const { return data_.d4x4_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d4_t>()       { return data_.d4x4_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d8_t>() const { return data_.d8x2_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d8_t>()       { return data_.d8x2_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d16_t>() const { return data_.d16x1_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d16_t>()       { return data_.d16x1_;}
};

template <typename T> struct vector_type<T, 32> {
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d4_t __attribute__((ext_vector_type(4)));
    typedef T d8_t __attribute__((ext_vector_type(8)));
    typedef T d16_t __attribute__((ext_vector_type(16)));
    typedef T d32_t __attribute__((ext_vector_type(32)));
    using type = d32_t;
    union {
        type d32_;
        static_buffer<d1_t, 32> d1x32_;
        static_buffer<d2_t, 16> d2x16_;
        static_buffer<d4_t, 8> d4x8_;
        static_buffer<d8_t, 4> d8x4_;
        static_buffer<d16_t, 2> d16x2_;
        static_buffer<d32_t, 1> d32x1_;
    } data_;
    DEVICE_HOST constexpr vector_type() : data_{type{0}} {}
    DEVICE_HOST constexpr vector_type(type v) : data_{v} {}
    template<typename VEC> DEVICE_HOST constexpr const auto& to_varray() const { return data_.d32_; }
    template<typename VEC> DEVICE_HOST constexpr auto&       to_varray()       { return data_.d32_; }
    template<> DEVICE_HOST constexpr const auto& to_varray<d1_t>() const { return data_.d1x32_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d1_t>()       { return data_.d1x32_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d2_t>() const { return data_.d2x16_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d2_t>()       { return data_.d2x16_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d4_t>() const { return data_.d4x8_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d4_t>()       { return data_.d4x8_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d8_t>() const { return data_.d8x4_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d8_t>()       { return data_.d8x4_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d16_t>() const { return data_.d16x2_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d16_t>()       { return data_.d16x2_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d32_t>() const { return data_.d32x1_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d32_t>()       { return data_.d32x1_;}
};

template <typename T> struct vector_type<T, 64> {
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d4_t __attribute__((ext_vector_type(4)));
    typedef T d8_t __attribute__((ext_vector_type(8)));
    typedef T d16_t __attribute__((ext_vector_type(16)));
    typedef T d32_t __attribute__((ext_vector_type(32)));
    typedef T d64_t __attribute__((ext_vector_type(64)));
    using type = d64_t;
    union {
        type d64_;
        static_buffer<d1_t, 64> d1x64_;
        static_buffer<d2_t, 32> d2x32_;
        static_buffer<d4_t, 16> d4x16_;
        static_buffer<d8_t, 8> d8x8_;
        static_buffer<d16_t, 4> d16x4_;
        static_buffer<d32_t, 2> d32x2_;
        static_buffer<d64_t, 1> d64x1_;
    } data_;
    DEVICE_HOST constexpr vector_type() : data_{type{0}} {}
    DEVICE_HOST constexpr vector_type(type v) : data_{v} {}
    template<typename VEC> DEVICE_HOST constexpr const auto& to_varray() const { return data_.d64_; }
    template<typename VEC> DEVICE_HOST constexpr auto&       to_varray()       { return data_.d64_; }
    template<> DEVICE_HOST constexpr const auto& to_varray<d1_t>() const { return data_.d1x64_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d1_t>()       { return data_.d1x64_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d2_t>() const { return data_.d2x32_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d2_t>()       { return data_.d2x32_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d4_t>() const { return data_.d4x16_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d4_t>()       { return data_.d4x16_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d8_t>() const { return data_.d8x8_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d8_t>()       { return data_.d8x8_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d16_t>() const { return data_.d16x4_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d16_t>()       { return data_.d16x4_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d32_t>() const { return data_.d32x2_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d32_t>()       { return data_.d32x2_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d64_t>() const { return data_.d64x1_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d64_t>()       { return data_.d64x1_;}
};

template <typename T> struct vector_type<T, 128> {
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d4_t __attribute__((ext_vector_type(4)));
    typedef T d8_t __attribute__((ext_vector_type(8)));
    typedef T d16_t __attribute__((ext_vector_type(16)));
    typedef T d32_t __attribute__((ext_vector_type(32)));
    typedef T d64_t __attribute__((ext_vector_type(64)));
    typedef T d128_t __attribute__((ext_vector_type(128)));
    using type = d128_t;
    union {
        type d128_;
        static_buffer<d1_t, 128> d1x128_;
        static_buffer<d2_t, 64> d2x64_;
        static_buffer<d4_t, 32> d4x32_;
        static_buffer<d8_t, 16> d8x16_;
        static_buffer<d16_t, 8> d16x8_;
        static_buffer<d32_t, 4> d32x4_;
        static_buffer<d64_t, 2> d64x2_;
        static_buffer<d128_t, 1> d128x1_;
    } data_;
    DEVICE_HOST constexpr vector_type() : data_{type{0}} {}
    DEVICE_HOST constexpr vector_type(type v) : data_{v} {}
    template<typename VEC> DEVICE_HOST constexpr const auto& to_varray() const { return data_.d128_; }
    template<typename VEC> DEVICE_HOST constexpr auto&       to_varray()       { return data_.d128_; }
    template<> DEVICE_HOST constexpr const auto& to_varray<d1_t>() const { return data_.d1x128_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d1_t>()       { return data_.d1x128_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d2_t>() const { return data_.d2x64_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d2_t>()       { return data_.d2x64_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d4_t>() const { return data_.d4x32_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d4_t>()       { return data_.d4x32_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d8_t>() const { return data_.d8x16_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d8_t>()       { return data_.d8x16_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d16_t>() const { return data_.d16x8_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d16_t>()       { return data_.d16x8_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d32_t>() const { return data_.d32x4_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d32_t>()       { return data_.d32x4_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d64_t>() const { return data_.d64x2_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d64_t>()       { return data_.d64x2_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d128_t>() const { return data_.d128x1_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d128_t>()       { return data_.d128x1_;}
};

template <typename T> struct vector_type<T, 256> {
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d4_t __attribute__((ext_vector_type(4)));
    typedef T d8_t __attribute__((ext_vector_type(8)));
    typedef T d16_t __attribute__((ext_vector_type(16)));
    typedef T d32_t __attribute__((ext_vector_type(32)));
    typedef T d64_t __attribute__((ext_vector_type(64)));
    typedef T d128_t __attribute__((ext_vector_type(128)));
    typedef T d256_t __attribute__((ext_vector_type(256)));
    using type = d256_t;
    union {
        type d256_;
        static_buffer<d1_t, 256> d1x256_;
        static_buffer<d2_t, 128> d2x128_;
        static_buffer<d4_t, 64> d4x64_;
        static_buffer<d8_t, 32> d8x32_;
        static_buffer<d16_t, 16> d16x16_;
        static_buffer<d32_t, 8> d32x8_;
        static_buffer<d64_t, 4> d64x4_;
        static_buffer<d128_t, 2> d128x2_;
        static_buffer<d256_t, 1> d256x1_;
    } data_;
    DEVICE_HOST constexpr vector_type() : data_{type{0}} {}
    DEVICE_HOST constexpr vector_type(type v) : data_{v} {}
    template<typename VEC> DEVICE_HOST constexpr const auto& to_varray() const { return data_.d256_; }
    template<typename VEC> DEVICE_HOST constexpr auto&       to_varray()       { return data_.d256_; }
    template<> DEVICE_HOST constexpr const auto& to_varray<d1_t>() const { return data_.d1x256_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d1_t>()       { return data_.d1x256_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d2_t>() const { return data_.d2x128_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d2_t>()       { return data_.d2x128_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d4_t>() const { return data_.d4x64_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d4_t>()       { return data_.d4x64_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d8_t>() const { return data_.d8x32_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d8_t>()       { return data_.d8x32_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d16_t>() const { return data_.d16x16_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d16_t>()       { return data_.d16x16_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d32_t>() const { return data_.d32x8_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d32_t>()       { return data_.d32x8_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d64_t>() const { return data_.d64x4_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d64_t>()       { return data_.d64x4_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d128_t>() const { return data_.d128x2_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d128_t>()       { return data_.d128x2_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d256_t>() const { return data_.d256x1_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d256_t>()       { return data_.d256x1_;}
};

// clang-format on
