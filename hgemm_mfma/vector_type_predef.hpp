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

template <typename T> struct vector_type<T, 12> {
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d4_t __attribute__((ext_vector_type(4)));
    typedef T d12_t __attribute__((ext_vector_type(12)));
    using type = d12_t;
    union {
        type d12_;
        static_buffer<d1_t, 12> d1x12_;
        static_buffer<d2_t, 6> d2x6_;
        static_buffer<d4_t, 3> d4x3_;
        static_buffer<d12_t, 1> d12x1_;
    } data_;
    DEVICE_HOST constexpr vector_type() : data_{type{0}} {}
    DEVICE_HOST constexpr vector_type(type v) : data_{v} {}
    template<typename VEC> DEVICE_HOST constexpr const auto& to_varray() const { return data_.d12_; }
    template<typename VEC> DEVICE_HOST constexpr auto&       to_varray()       { return data_.d12_; }
    template<> DEVICE_HOST constexpr const auto& to_varray<d1_t>() const { return data_.d1x12_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d1_t>()       { return data_.d1x12_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d2_t>() const { return data_.d2x6_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d2_t>()       { return data_.d2x6_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d4_t>() const { return data_.d4x3_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d4_t>()       { return data_.d4x3_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d12_t>() const { return data_.d12x1_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d12_t>()       { return data_.d12x1_;}
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

template <typename T> struct vector_type<T, 20> {
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d4_t __attribute__((ext_vector_type(4)));
    typedef T d20_t __attribute__((ext_vector_type(20)));
    using type = d20_t;
    union {
        type d20_;
        static_buffer<d1_t, 20> d1x20_;
        static_buffer<d2_t, 10> d2x10_;
        static_buffer<d4_t, 5> d4x5_;
        static_buffer<d20_t, 1> d20x1_;
    } data_;
    DEVICE_HOST constexpr vector_type() : data_{type{0}} {}
    DEVICE_HOST constexpr vector_type(type v) : data_{v} {}
    template<typename VEC> DEVICE_HOST constexpr const auto& to_varray() const { return data_.d20_; }
    template<typename VEC> DEVICE_HOST constexpr auto&       to_varray()       { return data_.d20_; }
    template<> DEVICE_HOST constexpr const auto& to_varray<d1_t>() const { return data_.d1x20_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d1_t>()       { return data_.d1x20_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d2_t>() const { return data_.d2x10_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d2_t>()       { return data_.d2x10_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d4_t>() const { return data_.d4x5_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d4_t>()       { return data_.d4x5_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d20_t>() const { return data_.d20x1_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d20_t>()       { return data_.d20x1_;}
};

template <typename T> struct vector_type<T, 24> {
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d4_t __attribute__((ext_vector_type(4)));
    typedef T d8_t __attribute__((ext_vector_type(8)));
    typedef T d12_t __attribute__((ext_vector_type(12)));
    typedef T d24_t __attribute__((ext_vector_type(24)));
    using type = d24_t;
    union {
        type d24_;
        static_buffer<d1_t, 24> d1x24_;
        static_buffer<d2_t, 12> d2x12_;
        static_buffer<d4_t, 6> d4x6_;
        static_buffer<d8_t, 3> d8x3_;
        static_buffer<d12_t, 2> d12x2_;
        static_buffer<d24_t, 1> d24x1_;
    } data_;
    DEVICE_HOST constexpr vector_type() : data_{type{0}} {}
    DEVICE_HOST constexpr vector_type(type v) : data_{v} {}
    template<typename VEC> DEVICE_HOST constexpr const auto& to_varray() const { return data_.d24_; }
    template<typename VEC> DEVICE_HOST constexpr auto&       to_varray()       { return data_.d24_; }
    template<> DEVICE_HOST constexpr const auto& to_varray<d1_t>() const { return data_.d1x24_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d1_t>()       { return data_.d1x24_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d2_t>() const { return data_.d2x12_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d2_t>()       { return data_.d2x12_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d4_t>() const { return data_.d4x6_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d4_t>()       { return data_.d4x6_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d8_t>() const { return data_.d8x3_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d8_t>()       { return data_.d8x3_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d12_t>() const { return data_.d12x2_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d12_t>()       { return data_.d12x2_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d24_t>() const { return data_.d24x1_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d24_t>()       { return data_.d24x1_;}
};

template <typename T> struct vector_type<T, 28> {
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d4_t __attribute__((ext_vector_type(4)));
    typedef T d28_t __attribute__((ext_vector_type(28)));
    using type = d28_t;
    union {
        type d28_;
        static_buffer<d1_t, 28> d1x28_;
        static_buffer<d2_t, 14> d2x14_;
        static_buffer<d4_t, 7> d4x7_;
        static_buffer<d28_t, 1> d28x1_;
    } data_;
    DEVICE_HOST constexpr vector_type() : data_{type{0}} {}
    DEVICE_HOST constexpr vector_type(type v) : data_{v} {}
    template<typename VEC> DEVICE_HOST constexpr const auto& to_varray() const { return data_.d28_; }
    template<typename VEC> DEVICE_HOST constexpr auto&       to_varray()       { return data_.d28_; }
    template<> DEVICE_HOST constexpr const auto& to_varray<d1_t>() const { return data_.d1x28_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d1_t>()       { return data_.d1x28_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d2_t>() const { return data_.d2x14_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d2_t>()       { return data_.d2x14_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d4_t>() const { return data_.d4x7_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d4_t>()       { return data_.d4x7_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d28_t>() const { return data_.d28x1_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d28_t>()       { return data_.d28x1_;}
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

template <typename T> struct vector_type<T, 36> {
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d4_t __attribute__((ext_vector_type(4)));
    typedef T d12_t __attribute__((ext_vector_type(12)));
    typedef T d36_t __attribute__((ext_vector_type(36)));
    using type = d36_t;
    union {
        type d36_;
        static_buffer<d1_t, 36> d1x36_;
        static_buffer<d2_t, 18> d2x18_;
        static_buffer<d4_t, 9> d4x9_;
        static_buffer<d12_t, 3> d12x3_;
        static_buffer<d36_t, 1> d36x1_;
    } data_;
    DEVICE_HOST constexpr vector_type() : data_{type{0}} {}
    DEVICE_HOST constexpr vector_type(type v) : data_{v} {}
    template<typename VEC> DEVICE_HOST constexpr const auto& to_varray() const { return data_.d36_; }
    template<typename VEC> DEVICE_HOST constexpr auto&       to_varray()       { return data_.d36_; }
    template<> DEVICE_HOST constexpr const auto& to_varray<d1_t>() const { return data_.d1x36_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d1_t>()       { return data_.d1x36_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d2_t>() const { return data_.d2x18_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d2_t>()       { return data_.d2x18_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d4_t>() const { return data_.d4x9_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d4_t>()       { return data_.d4x9_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d12_t>() const { return data_.d12x3_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d12_t>()       { return data_.d12x3_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d36_t>() const { return data_.d36x1_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d36_t>()       { return data_.d36x1_;}
};

template <typename T> struct vector_type<T, 40> {
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d4_t __attribute__((ext_vector_type(4)));
    typedef T d8_t __attribute__((ext_vector_type(8)));
    typedef T d20_t __attribute__((ext_vector_type(20)));
    typedef T d40_t __attribute__((ext_vector_type(40)));
    using type = d40_t;
    union {
        type d40_;
        static_buffer<d1_t, 40> d1x40_;
        static_buffer<d2_t, 20> d2x20_;
        static_buffer<d4_t, 10> d4x10_;
        static_buffer<d8_t, 5> d8x5_;
        static_buffer<d20_t, 2> d20x2_;
        static_buffer<d40_t, 1> d40x1_;
    } data_;
    DEVICE_HOST constexpr vector_type() : data_{type{0}} {}
    DEVICE_HOST constexpr vector_type(type v) : data_{v} {}
    template<typename VEC> DEVICE_HOST constexpr const auto& to_varray() const { return data_.d40_; }
    template<typename VEC> DEVICE_HOST constexpr auto&       to_varray()       { return data_.d40_; }
    template<> DEVICE_HOST constexpr const auto& to_varray<d1_t>() const { return data_.d1x40_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d1_t>()       { return data_.d1x40_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d2_t>() const { return data_.d2x20_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d2_t>()       { return data_.d2x20_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d4_t>() const { return data_.d4x10_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d4_t>()       { return data_.d4x10_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d8_t>() const { return data_.d8x5_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d8_t>()       { return data_.d8x5_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d20_t>() const { return data_.d20x2_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d20_t>()       { return data_.d20x2_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d40_t>() const { return data_.d40x1_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d40_t>()       { return data_.d40x1_;}
};

template <typename T> struct vector_type<T, 44> {
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d4_t __attribute__((ext_vector_type(4)));
    typedef T d44_t __attribute__((ext_vector_type(44)));
    using type = d44_t;
    union {
        type d44_;
        static_buffer<d1_t, 44> d1x44_;
        static_buffer<d2_t, 22> d2x22_;
        static_buffer<d4_t, 11> d4x11_;
        static_buffer<d44_t, 1> d44x1_;
    } data_;
    DEVICE_HOST constexpr vector_type() : data_{type{0}} {}
    DEVICE_HOST constexpr vector_type(type v) : data_{v} {}
    template<typename VEC> DEVICE_HOST constexpr const auto& to_varray() const { return data_.d44_; }
    template<typename VEC> DEVICE_HOST constexpr auto&       to_varray()       { return data_.d44_; }
    template<> DEVICE_HOST constexpr const auto& to_varray<d1_t>() const { return data_.d1x44_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d1_t>()       { return data_.d1x44_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d2_t>() const { return data_.d2x22_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d2_t>()       { return data_.d2x22_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d4_t>() const { return data_.d4x11_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d4_t>()       { return data_.d4x11_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d44_t>() const { return data_.d44x1_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d44_t>()       { return data_.d44x1_;}
};

template <typename T> struct vector_type<T, 48> {
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d4_t __attribute__((ext_vector_type(4)));
    typedef T d8_t __attribute__((ext_vector_type(8)));
    typedef T d12_t __attribute__((ext_vector_type(12)));
    typedef T d16_t __attribute__((ext_vector_type(16)));
    typedef T d24_t __attribute__((ext_vector_type(24)));
    typedef T d48_t __attribute__((ext_vector_type(48)));
    using type = d48_t;
    union {
        type d48_;
        static_buffer<d1_t, 48> d1x48_;
        static_buffer<d2_t, 24> d2x24_;
        static_buffer<d4_t, 12> d4x12_;
        static_buffer<d8_t, 6> d8x6_;
        static_buffer<d12_t, 4> d12x4_;
        static_buffer<d16_t, 3> d16x3_;
        static_buffer<d24_t, 2> d24x2_;
        static_buffer<d48_t, 1> d48x1_;
    } data_;
    DEVICE_HOST constexpr vector_type() : data_{type{0}} {}
    DEVICE_HOST constexpr vector_type(type v) : data_{v} {}
    template<typename VEC> DEVICE_HOST constexpr const auto& to_varray() const { return data_.d48_; }
    template<typename VEC> DEVICE_HOST constexpr auto&       to_varray()       { return data_.d48_; }
    template<> DEVICE_HOST constexpr const auto& to_varray<d1_t>() const { return data_.d1x48_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d1_t>()       { return data_.d1x48_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d2_t>() const { return data_.d2x24_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d2_t>()       { return data_.d2x24_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d4_t>() const { return data_.d4x12_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d4_t>()       { return data_.d4x12_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d8_t>() const { return data_.d8x6_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d8_t>()       { return data_.d8x6_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d12_t>() const { return data_.d12x4_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d12_t>()       { return data_.d12x4_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d16_t>() const { return data_.d16x3_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d16_t>()       { return data_.d16x3_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d24_t>() const { return data_.d24x2_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d24_t>()       { return data_.d24x2_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d48_t>() const { return data_.d48x1_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d48_t>()       { return data_.d48x1_;}
};

template <typename T> struct vector_type<T, 52> {
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d4_t __attribute__((ext_vector_type(4)));
    typedef T d52_t __attribute__((ext_vector_type(52)));
    using type = d52_t;
    union {
        type d52_;
        static_buffer<d1_t, 52> d1x52_;
        static_buffer<d2_t, 26> d2x26_;
        static_buffer<d4_t, 13> d4x13_;
        static_buffer<d52_t, 1> d52x1_;
    } data_;
    DEVICE_HOST constexpr vector_type() : data_{type{0}} {}
    DEVICE_HOST constexpr vector_type(type v) : data_{v} {}
    template<typename VEC> DEVICE_HOST constexpr const auto& to_varray() const { return data_.d52_; }
    template<typename VEC> DEVICE_HOST constexpr auto&       to_varray()       { return data_.d52_; }
    template<> DEVICE_HOST constexpr const auto& to_varray<d1_t>() const { return data_.d1x52_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d1_t>()       { return data_.d1x52_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d2_t>() const { return data_.d2x26_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d2_t>()       { return data_.d2x26_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d4_t>() const { return data_.d4x13_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d4_t>()       { return data_.d4x13_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d52_t>() const { return data_.d52x1_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d52_t>()       { return data_.d52x1_;}
};

template <typename T> struct vector_type<T, 56> {
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d4_t __attribute__((ext_vector_type(4)));
    typedef T d8_t __attribute__((ext_vector_type(8)));
    typedef T d28_t __attribute__((ext_vector_type(28)));
    typedef T d56_t __attribute__((ext_vector_type(56)));
    using type = d56_t;
    union {
        type d56_;
        static_buffer<d1_t, 56> d1x56_;
        static_buffer<d2_t, 28> d2x28_;
        static_buffer<d4_t, 14> d4x14_;
        static_buffer<d8_t, 7> d8x7_;
        static_buffer<d28_t, 2> d28x2_;
        static_buffer<d56_t, 1> d56x1_;
    } data_;
    DEVICE_HOST constexpr vector_type() : data_{type{0}} {}
    DEVICE_HOST constexpr vector_type(type v) : data_{v} {}
    template<typename VEC> DEVICE_HOST constexpr const auto& to_varray() const { return data_.d56_; }
    template<typename VEC> DEVICE_HOST constexpr auto&       to_varray()       { return data_.d56_; }
    template<> DEVICE_HOST constexpr const auto& to_varray<d1_t>() const { return data_.d1x56_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d1_t>()       { return data_.d1x56_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d2_t>() const { return data_.d2x28_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d2_t>()       { return data_.d2x28_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d4_t>() const { return data_.d4x14_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d4_t>()       { return data_.d4x14_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d8_t>() const { return data_.d8x7_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d8_t>()       { return data_.d8x7_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d28_t>() const { return data_.d28x2_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d28_t>()       { return data_.d28x2_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d56_t>() const { return data_.d56x1_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d56_t>()       { return data_.d56x1_;}
};

template <typename T> struct vector_type<T, 60> {
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d4_t __attribute__((ext_vector_type(4)));
    typedef T d12_t __attribute__((ext_vector_type(12)));
    typedef T d20_t __attribute__((ext_vector_type(20)));
    typedef T d60_t __attribute__((ext_vector_type(60)));
    using type = d60_t;
    union {
        type d60_;
        static_buffer<d1_t, 60> d1x60_;
        static_buffer<d2_t, 30> d2x30_;
        static_buffer<d4_t, 15> d4x15_;
        static_buffer<d12_t, 5> d12x5_;
        static_buffer<d20_t, 3> d20x3_;
        static_buffer<d60_t, 1> d60x1_;
    } data_;
    DEVICE_HOST constexpr vector_type() : data_{type{0}} {}
    DEVICE_HOST constexpr vector_type(type v) : data_{v} {}
    template<typename VEC> DEVICE_HOST constexpr const auto& to_varray() const { return data_.d60_; }
    template<typename VEC> DEVICE_HOST constexpr auto&       to_varray()       { return data_.d60_; }
    template<> DEVICE_HOST constexpr const auto& to_varray<d1_t>() const { return data_.d1x60_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d1_t>()       { return data_.d1x60_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d2_t>() const { return data_.d2x30_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d2_t>()       { return data_.d2x30_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d4_t>() const { return data_.d4x15_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d4_t>()       { return data_.d4x15_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d12_t>() const { return data_.d12x5_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d12_t>()       { return data_.d12x5_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d20_t>() const { return data_.d20x3_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d20_t>()       { return data_.d20x3_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d60_t>() const { return data_.d60x1_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d60_t>()       { return data_.d60x1_;}
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

template <typename T> struct vector_type<T, 68> {
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d4_t __attribute__((ext_vector_type(4)));
    typedef T d68_t __attribute__((ext_vector_type(68)));
    using type = d68_t;
    union {
        type d68_;
        static_buffer<d1_t, 68> d1x68_;
        static_buffer<d2_t, 34> d2x34_;
        static_buffer<d4_t, 17> d4x17_;
        static_buffer<d68_t, 1> d68x1_;
    } data_;
    DEVICE_HOST constexpr vector_type() : data_{type{0}} {}
    DEVICE_HOST constexpr vector_type(type v) : data_{v} {}
    template<typename VEC> DEVICE_HOST constexpr const auto& to_varray() const { return data_.d68_; }
    template<typename VEC> DEVICE_HOST constexpr auto&       to_varray()       { return data_.d68_; }
    template<> DEVICE_HOST constexpr const auto& to_varray<d1_t>() const { return data_.d1x68_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d1_t>()       { return data_.d1x68_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d2_t>() const { return data_.d2x34_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d2_t>()       { return data_.d2x34_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d4_t>() const { return data_.d4x17_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d4_t>()       { return data_.d4x17_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d68_t>() const { return data_.d68x1_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d68_t>()       { return data_.d68x1_;}
};

template <typename T> struct vector_type<T, 72> {
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d4_t __attribute__((ext_vector_type(4)));
    typedef T d8_t __attribute__((ext_vector_type(8)));
    typedef T d12_t __attribute__((ext_vector_type(12)));
    typedef T d24_t __attribute__((ext_vector_type(24)));
    typedef T d36_t __attribute__((ext_vector_type(36)));
    typedef T d72_t __attribute__((ext_vector_type(72)));
    using type = d72_t;
    union {
        type d72_;
        static_buffer<d1_t, 72> d1x72_;
        static_buffer<d2_t, 36> d2x36_;
        static_buffer<d4_t, 18> d4x18_;
        static_buffer<d8_t, 9> d8x9_;
        static_buffer<d12_t, 6> d12x6_;
        static_buffer<d24_t, 3> d24x3_;
        static_buffer<d36_t, 2> d36x2_;
        static_buffer<d72_t, 1> d72x1_;
    } data_;
    DEVICE_HOST constexpr vector_type() : data_{type{0}} {}
    DEVICE_HOST constexpr vector_type(type v) : data_{v} {}
    template<typename VEC> DEVICE_HOST constexpr const auto& to_varray() const { return data_.d72_; }
    template<typename VEC> DEVICE_HOST constexpr auto&       to_varray()       { return data_.d72_; }
    template<> DEVICE_HOST constexpr const auto& to_varray<d1_t>() const { return data_.d1x72_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d1_t>()       { return data_.d1x72_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d2_t>() const { return data_.d2x36_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d2_t>()       { return data_.d2x36_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d4_t>() const { return data_.d4x18_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d4_t>()       { return data_.d4x18_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d8_t>() const { return data_.d8x9_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d8_t>()       { return data_.d8x9_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d12_t>() const { return data_.d12x6_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d12_t>()       { return data_.d12x6_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d24_t>() const { return data_.d24x3_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d24_t>()       { return data_.d24x3_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d36_t>() const { return data_.d36x2_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d36_t>()       { return data_.d36x2_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d72_t>() const { return data_.d72x1_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d72_t>()       { return data_.d72x1_;}
};

template <typename T> struct vector_type<T, 76> {
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d4_t __attribute__((ext_vector_type(4)));
    typedef T d76_t __attribute__((ext_vector_type(76)));
    using type = d76_t;
    union {
        type d76_;
        static_buffer<d1_t, 76> d1x76_;
        static_buffer<d2_t, 38> d2x38_;
        static_buffer<d4_t, 19> d4x19_;
        static_buffer<d76_t, 1> d76x1_;
    } data_;
    DEVICE_HOST constexpr vector_type() : data_{type{0}} {}
    DEVICE_HOST constexpr vector_type(type v) : data_{v} {}
    template<typename VEC> DEVICE_HOST constexpr const auto& to_varray() const { return data_.d76_; }
    template<typename VEC> DEVICE_HOST constexpr auto&       to_varray()       { return data_.d76_; }
    template<> DEVICE_HOST constexpr const auto& to_varray<d1_t>() const { return data_.d1x76_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d1_t>()       { return data_.d1x76_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d2_t>() const { return data_.d2x38_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d2_t>()       { return data_.d2x38_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d4_t>() const { return data_.d4x19_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d4_t>()       { return data_.d4x19_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d76_t>() const { return data_.d76x1_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d76_t>()       { return data_.d76x1_;}
};

template <typename T> struct vector_type<T, 80> {
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d4_t __attribute__((ext_vector_type(4)));
    typedef T d8_t __attribute__((ext_vector_type(8)));
    typedef T d16_t __attribute__((ext_vector_type(16)));
    typedef T d20_t __attribute__((ext_vector_type(20)));
    typedef T d40_t __attribute__((ext_vector_type(40)));
    typedef T d80_t __attribute__((ext_vector_type(80)));
    using type = d80_t;
    union {
        type d80_;
        static_buffer<d1_t, 80> d1x80_;
        static_buffer<d2_t, 40> d2x40_;
        static_buffer<d4_t, 20> d4x20_;
        static_buffer<d8_t, 10> d8x10_;
        static_buffer<d16_t, 5> d16x5_;
        static_buffer<d20_t, 4> d20x4_;
        static_buffer<d40_t, 2> d40x2_;
        static_buffer<d80_t, 1> d80x1_;
    } data_;
    DEVICE_HOST constexpr vector_type() : data_{type{0}} {}
    DEVICE_HOST constexpr vector_type(type v) : data_{v} {}
    template<typename VEC> DEVICE_HOST constexpr const auto& to_varray() const { return data_.d80_; }
    template<typename VEC> DEVICE_HOST constexpr auto&       to_varray()       { return data_.d80_; }
    template<> DEVICE_HOST constexpr const auto& to_varray<d1_t>() const { return data_.d1x80_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d1_t>()       { return data_.d1x80_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d2_t>() const { return data_.d2x40_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d2_t>()       { return data_.d2x40_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d4_t>() const { return data_.d4x20_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d4_t>()       { return data_.d4x20_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d8_t>() const { return data_.d8x10_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d8_t>()       { return data_.d8x10_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d16_t>() const { return data_.d16x5_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d16_t>()       { return data_.d16x5_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d20_t>() const { return data_.d20x4_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d20_t>()       { return data_.d20x4_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d40_t>() const { return data_.d40x2_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d40_t>()       { return data_.d40x2_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d80_t>() const { return data_.d80x1_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d80_t>()       { return data_.d80x1_;}
};

template <typename T> struct vector_type<T, 84> {
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d4_t __attribute__((ext_vector_type(4)));
    typedef T d12_t __attribute__((ext_vector_type(12)));
    typedef T d28_t __attribute__((ext_vector_type(28)));
    typedef T d84_t __attribute__((ext_vector_type(84)));
    using type = d84_t;
    union {
        type d84_;
        static_buffer<d1_t, 84> d1x84_;
        static_buffer<d2_t, 42> d2x42_;
        static_buffer<d4_t, 21> d4x21_;
        static_buffer<d12_t, 7> d12x7_;
        static_buffer<d28_t, 3> d28x3_;
        static_buffer<d84_t, 1> d84x1_;
    } data_;
    DEVICE_HOST constexpr vector_type() : data_{type{0}} {}
    DEVICE_HOST constexpr vector_type(type v) : data_{v} {}
    template<typename VEC> DEVICE_HOST constexpr const auto& to_varray() const { return data_.d84_; }
    template<typename VEC> DEVICE_HOST constexpr auto&       to_varray()       { return data_.d84_; }
    template<> DEVICE_HOST constexpr const auto& to_varray<d1_t>() const { return data_.d1x84_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d1_t>()       { return data_.d1x84_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d2_t>() const { return data_.d2x42_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d2_t>()       { return data_.d2x42_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d4_t>() const { return data_.d4x21_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d4_t>()       { return data_.d4x21_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d12_t>() const { return data_.d12x7_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d12_t>()       { return data_.d12x7_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d28_t>() const { return data_.d28x3_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d28_t>()       { return data_.d28x3_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d84_t>() const { return data_.d84x1_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d84_t>()       { return data_.d84x1_;}
};

template <typename T> struct vector_type<T, 88> {
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d4_t __attribute__((ext_vector_type(4)));
    typedef T d8_t __attribute__((ext_vector_type(8)));
    typedef T d44_t __attribute__((ext_vector_type(44)));
    typedef T d88_t __attribute__((ext_vector_type(88)));
    using type = d88_t;
    union {
        type d88_;
        static_buffer<d1_t, 88> d1x88_;
        static_buffer<d2_t, 44> d2x44_;
        static_buffer<d4_t, 22> d4x22_;
        static_buffer<d8_t, 11> d8x11_;
        static_buffer<d44_t, 2> d44x2_;
        static_buffer<d88_t, 1> d88x1_;
    } data_;
    DEVICE_HOST constexpr vector_type() : data_{type{0}} {}
    DEVICE_HOST constexpr vector_type(type v) : data_{v} {}
    template<typename VEC> DEVICE_HOST constexpr const auto& to_varray() const { return data_.d88_; }
    template<typename VEC> DEVICE_HOST constexpr auto&       to_varray()       { return data_.d88_; }
    template<> DEVICE_HOST constexpr const auto& to_varray<d1_t>() const { return data_.d1x88_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d1_t>()       { return data_.d1x88_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d2_t>() const { return data_.d2x44_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d2_t>()       { return data_.d2x44_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d4_t>() const { return data_.d4x22_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d4_t>()       { return data_.d4x22_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d8_t>() const { return data_.d8x11_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d8_t>()       { return data_.d8x11_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d44_t>() const { return data_.d44x2_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d44_t>()       { return data_.d44x2_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d88_t>() const { return data_.d88x1_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d88_t>()       { return data_.d88x1_;}
};

template <typename T> struct vector_type<T, 92> {
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d4_t __attribute__((ext_vector_type(4)));
    typedef T d92_t __attribute__((ext_vector_type(92)));
    using type = d92_t;
    union {
        type d92_;
        static_buffer<d1_t, 92> d1x92_;
        static_buffer<d2_t, 46> d2x46_;
        static_buffer<d4_t, 23> d4x23_;
        static_buffer<d92_t, 1> d92x1_;
    } data_;
    DEVICE_HOST constexpr vector_type() : data_{type{0}} {}
    DEVICE_HOST constexpr vector_type(type v) : data_{v} {}
    template<typename VEC> DEVICE_HOST constexpr const auto& to_varray() const { return data_.d92_; }
    template<typename VEC> DEVICE_HOST constexpr auto&       to_varray()       { return data_.d92_; }
    template<> DEVICE_HOST constexpr const auto& to_varray<d1_t>() const { return data_.d1x92_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d1_t>()       { return data_.d1x92_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d2_t>() const { return data_.d2x46_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d2_t>()       { return data_.d2x46_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d4_t>() const { return data_.d4x23_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d4_t>()       { return data_.d4x23_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d92_t>() const { return data_.d92x1_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d92_t>()       { return data_.d92x1_;}
};

template <typename T> struct vector_type<T, 96> {
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d4_t __attribute__((ext_vector_type(4)));
    typedef T d8_t __attribute__((ext_vector_type(8)));
    typedef T d12_t __attribute__((ext_vector_type(12)));
    typedef T d16_t __attribute__((ext_vector_type(16)));
    typedef T d24_t __attribute__((ext_vector_type(24)));
    typedef T d32_t __attribute__((ext_vector_type(32)));
    typedef T d48_t __attribute__((ext_vector_type(48)));
    typedef T d96_t __attribute__((ext_vector_type(96)));
    using type = d96_t;
    union {
        type d96_;
        static_buffer<d1_t, 96> d1x96_;
        static_buffer<d2_t, 48> d2x48_;
        static_buffer<d4_t, 24> d4x24_;
        static_buffer<d8_t, 12> d8x12_;
        static_buffer<d12_t, 8> d12x8_;
        static_buffer<d16_t, 6> d16x6_;
        static_buffer<d24_t, 4> d24x4_;
        static_buffer<d32_t, 3> d32x3_;
        static_buffer<d48_t, 2> d48x2_;
        static_buffer<d96_t, 1> d96x1_;
    } data_;
    DEVICE_HOST constexpr vector_type() : data_{type{0}} {}
    DEVICE_HOST constexpr vector_type(type v) : data_{v} {}
    template<typename VEC> DEVICE_HOST constexpr const auto& to_varray() const { return data_.d96_; }
    template<typename VEC> DEVICE_HOST constexpr auto&       to_varray()       { return data_.d96_; }
    template<> DEVICE_HOST constexpr const auto& to_varray<d1_t>() const { return data_.d1x96_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d1_t>()       { return data_.d1x96_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d2_t>() const { return data_.d2x48_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d2_t>()       { return data_.d2x48_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d4_t>() const { return data_.d4x24_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d4_t>()       { return data_.d4x24_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d8_t>() const { return data_.d8x12_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d8_t>()       { return data_.d8x12_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d12_t>() const { return data_.d12x8_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d12_t>()       { return data_.d12x8_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d16_t>() const { return data_.d16x6_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d16_t>()       { return data_.d16x6_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d24_t>() const { return data_.d24x4_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d24_t>()       { return data_.d24x4_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d32_t>() const { return data_.d32x3_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d32_t>()       { return data_.d32x3_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d48_t>() const { return data_.d48x2_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d48_t>()       { return data_.d48x2_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d96_t>() const { return data_.d96x1_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d96_t>()       { return data_.d96x1_;}
};

template <typename T> struct vector_type<T, 100> {
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d4_t __attribute__((ext_vector_type(4)));
    typedef T d20_t __attribute__((ext_vector_type(20)));
    typedef T d100_t __attribute__((ext_vector_type(100)));
    using type = d100_t;
    union {
        type d100_;
        static_buffer<d1_t, 100> d1x100_;
        static_buffer<d2_t, 50> d2x50_;
        static_buffer<d4_t, 25> d4x25_;
        static_buffer<d20_t, 5> d20x5_;
        static_buffer<d100_t, 1> d100x1_;
    } data_;
    DEVICE_HOST constexpr vector_type() : data_{type{0}} {}
    DEVICE_HOST constexpr vector_type(type v) : data_{v} {}
    template<typename VEC> DEVICE_HOST constexpr const auto& to_varray() const { return data_.d100_; }
    template<typename VEC> DEVICE_HOST constexpr auto&       to_varray()       { return data_.d100_; }
    template<> DEVICE_HOST constexpr const auto& to_varray<d1_t>() const { return data_.d1x100_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d1_t>()       { return data_.d1x100_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d2_t>() const { return data_.d2x50_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d2_t>()       { return data_.d2x50_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d4_t>() const { return data_.d4x25_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d4_t>()       { return data_.d4x25_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d20_t>() const { return data_.d20x5_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d20_t>()       { return data_.d20x5_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d100_t>() const { return data_.d100x1_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d100_t>()       { return data_.d100x1_;}
};

template <typename T> struct vector_type<T, 104> {
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d4_t __attribute__((ext_vector_type(4)));
    typedef T d8_t __attribute__((ext_vector_type(8)));
    typedef T d52_t __attribute__((ext_vector_type(52)));
    typedef T d104_t __attribute__((ext_vector_type(104)));
    using type = d104_t;
    union {
        type d104_;
        static_buffer<d1_t, 104> d1x104_;
        static_buffer<d2_t, 52> d2x52_;
        static_buffer<d4_t, 26> d4x26_;
        static_buffer<d8_t, 13> d8x13_;
        static_buffer<d52_t, 2> d52x2_;
        static_buffer<d104_t, 1> d104x1_;
    } data_;
    DEVICE_HOST constexpr vector_type() : data_{type{0}} {}
    DEVICE_HOST constexpr vector_type(type v) : data_{v} {}
    template<typename VEC> DEVICE_HOST constexpr const auto& to_varray() const { return data_.d104_; }
    template<typename VEC> DEVICE_HOST constexpr auto&       to_varray()       { return data_.d104_; }
    template<> DEVICE_HOST constexpr const auto& to_varray<d1_t>() const { return data_.d1x104_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d1_t>()       { return data_.d1x104_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d2_t>() const { return data_.d2x52_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d2_t>()       { return data_.d2x52_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d4_t>() const { return data_.d4x26_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d4_t>()       { return data_.d4x26_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d8_t>() const { return data_.d8x13_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d8_t>()       { return data_.d8x13_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d52_t>() const { return data_.d52x2_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d52_t>()       { return data_.d52x2_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d104_t>() const { return data_.d104x1_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d104_t>()       { return data_.d104x1_;}
};

template <typename T> struct vector_type<T, 108> {
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d4_t __attribute__((ext_vector_type(4)));
    typedef T d12_t __attribute__((ext_vector_type(12)));
    typedef T d36_t __attribute__((ext_vector_type(36)));
    typedef T d108_t __attribute__((ext_vector_type(108)));
    using type = d108_t;
    union {
        type d108_;
        static_buffer<d1_t, 108> d1x108_;
        static_buffer<d2_t, 54> d2x54_;
        static_buffer<d4_t, 27> d4x27_;
        static_buffer<d12_t, 9> d12x9_;
        static_buffer<d36_t, 3> d36x3_;
        static_buffer<d108_t, 1> d108x1_;
    } data_;
    DEVICE_HOST constexpr vector_type() : data_{type{0}} {}
    DEVICE_HOST constexpr vector_type(type v) : data_{v} {}
    template<typename VEC> DEVICE_HOST constexpr const auto& to_varray() const { return data_.d108_; }
    template<typename VEC> DEVICE_HOST constexpr auto&       to_varray()       { return data_.d108_; }
    template<> DEVICE_HOST constexpr const auto& to_varray<d1_t>() const { return data_.d1x108_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d1_t>()       { return data_.d1x108_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d2_t>() const { return data_.d2x54_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d2_t>()       { return data_.d2x54_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d4_t>() const { return data_.d4x27_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d4_t>()       { return data_.d4x27_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d12_t>() const { return data_.d12x9_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d12_t>()       { return data_.d12x9_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d36_t>() const { return data_.d36x3_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d36_t>()       { return data_.d36x3_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d108_t>() const { return data_.d108x1_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d108_t>()       { return data_.d108x1_;}
};

template <typename T> struct vector_type<T, 112> {
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d4_t __attribute__((ext_vector_type(4)));
    typedef T d8_t __attribute__((ext_vector_type(8)));
    typedef T d16_t __attribute__((ext_vector_type(16)));
    typedef T d28_t __attribute__((ext_vector_type(28)));
    typedef T d56_t __attribute__((ext_vector_type(56)));
    typedef T d112_t __attribute__((ext_vector_type(112)));
    using type = d112_t;
    union {
        type d112_;
        static_buffer<d1_t, 112> d1x112_;
        static_buffer<d2_t, 56> d2x56_;
        static_buffer<d4_t, 28> d4x28_;
        static_buffer<d8_t, 14> d8x14_;
        static_buffer<d16_t, 7> d16x7_;
        static_buffer<d28_t, 4> d28x4_;
        static_buffer<d56_t, 2> d56x2_;
        static_buffer<d112_t, 1> d112x1_;
    } data_;
    DEVICE_HOST constexpr vector_type() : data_{type{0}} {}
    DEVICE_HOST constexpr vector_type(type v) : data_{v} {}
    template<typename VEC> DEVICE_HOST constexpr const auto& to_varray() const { return data_.d112_; }
    template<typename VEC> DEVICE_HOST constexpr auto&       to_varray()       { return data_.d112_; }
    template<> DEVICE_HOST constexpr const auto& to_varray<d1_t>() const { return data_.d1x112_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d1_t>()       { return data_.d1x112_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d2_t>() const { return data_.d2x56_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d2_t>()       { return data_.d2x56_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d4_t>() const { return data_.d4x28_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d4_t>()       { return data_.d4x28_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d8_t>() const { return data_.d8x14_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d8_t>()       { return data_.d8x14_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d16_t>() const { return data_.d16x7_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d16_t>()       { return data_.d16x7_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d28_t>() const { return data_.d28x4_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d28_t>()       { return data_.d28x4_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d56_t>() const { return data_.d56x2_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d56_t>()       { return data_.d56x2_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d112_t>() const { return data_.d112x1_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d112_t>()       { return data_.d112x1_;}
};

template <typename T> struct vector_type<T, 116> {
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d4_t __attribute__((ext_vector_type(4)));
    typedef T d116_t __attribute__((ext_vector_type(116)));
    using type = d116_t;
    union {
        type d116_;
        static_buffer<d1_t, 116> d1x116_;
        static_buffer<d2_t, 58> d2x58_;
        static_buffer<d4_t, 29> d4x29_;
        static_buffer<d116_t, 1> d116x1_;
    } data_;
    DEVICE_HOST constexpr vector_type() : data_{type{0}} {}
    DEVICE_HOST constexpr vector_type(type v) : data_{v} {}
    template<typename VEC> DEVICE_HOST constexpr const auto& to_varray() const { return data_.d116_; }
    template<typename VEC> DEVICE_HOST constexpr auto&       to_varray()       { return data_.d116_; }
    template<> DEVICE_HOST constexpr const auto& to_varray<d1_t>() const { return data_.d1x116_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d1_t>()       { return data_.d1x116_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d2_t>() const { return data_.d2x58_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d2_t>()       { return data_.d2x58_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d4_t>() const { return data_.d4x29_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d4_t>()       { return data_.d4x29_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d116_t>() const { return data_.d116x1_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d116_t>()       { return data_.d116x1_;}
};

template <typename T> struct vector_type<T, 120> {
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d4_t __attribute__((ext_vector_type(4)));
    typedef T d8_t __attribute__((ext_vector_type(8)));
    typedef T d12_t __attribute__((ext_vector_type(12)));
    typedef T d20_t __attribute__((ext_vector_type(20)));
    typedef T d24_t __attribute__((ext_vector_type(24)));
    typedef T d40_t __attribute__((ext_vector_type(40)));
    typedef T d60_t __attribute__((ext_vector_type(60)));
    typedef T d120_t __attribute__((ext_vector_type(120)));
    using type = d120_t;
    union {
        type d120_;
        static_buffer<d1_t, 120> d1x120_;
        static_buffer<d2_t, 60> d2x60_;
        static_buffer<d4_t, 30> d4x30_;
        static_buffer<d8_t, 15> d8x15_;
        static_buffer<d12_t, 10> d12x10_;
        static_buffer<d20_t, 6> d20x6_;
        static_buffer<d24_t, 5> d24x5_;
        static_buffer<d40_t, 3> d40x3_;
        static_buffer<d60_t, 2> d60x2_;
        static_buffer<d120_t, 1> d120x1_;
    } data_;
    DEVICE_HOST constexpr vector_type() : data_{type{0}} {}
    DEVICE_HOST constexpr vector_type(type v) : data_{v} {}
    template<typename VEC> DEVICE_HOST constexpr const auto& to_varray() const { return data_.d120_; }
    template<typename VEC> DEVICE_HOST constexpr auto&       to_varray()       { return data_.d120_; }
    template<> DEVICE_HOST constexpr const auto& to_varray<d1_t>() const { return data_.d1x120_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d1_t>()       { return data_.d1x120_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d2_t>() const { return data_.d2x60_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d2_t>()       { return data_.d2x60_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d4_t>() const { return data_.d4x30_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d4_t>()       { return data_.d4x30_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d8_t>() const { return data_.d8x15_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d8_t>()       { return data_.d8x15_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d12_t>() const { return data_.d12x10_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d12_t>()       { return data_.d12x10_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d20_t>() const { return data_.d20x6_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d20_t>()       { return data_.d20x6_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d24_t>() const { return data_.d24x5_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d24_t>()       { return data_.d24x5_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d40_t>() const { return data_.d40x3_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d40_t>()       { return data_.d40x3_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d60_t>() const { return data_.d60x2_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d60_t>()       { return data_.d60x2_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d120_t>() const { return data_.d120x1_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d120_t>()       { return data_.d120x1_;}
};

template <typename T> struct vector_type<T, 124> {
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d4_t __attribute__((ext_vector_type(4)));
    typedef T d124_t __attribute__((ext_vector_type(124)));
    using type = d124_t;
    union {
        type d124_;
        static_buffer<d1_t, 124> d1x124_;
        static_buffer<d2_t, 62> d2x62_;
        static_buffer<d4_t, 31> d4x31_;
        static_buffer<d124_t, 1> d124x1_;
    } data_;
    DEVICE_HOST constexpr vector_type() : data_{type{0}} {}
    DEVICE_HOST constexpr vector_type(type v) : data_{v} {}
    template<typename VEC> DEVICE_HOST constexpr const auto& to_varray() const { return data_.d124_; }
    template<typename VEC> DEVICE_HOST constexpr auto&       to_varray()       { return data_.d124_; }
    template<> DEVICE_HOST constexpr const auto& to_varray<d1_t>() const { return data_.d1x124_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d1_t>()       { return data_.d1x124_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d2_t>() const { return data_.d2x62_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d2_t>()       { return data_.d2x62_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d4_t>() const { return data_.d4x31_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d4_t>()       { return data_.d4x31_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d124_t>() const { return data_.d124x1_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d124_t>()       { return data_.d124x1_;}
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

template <typename T> struct vector_type<T, 132> {
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d4_t __attribute__((ext_vector_type(4)));
    typedef T d12_t __attribute__((ext_vector_type(12)));
    typedef T d44_t __attribute__((ext_vector_type(44)));
    typedef T d132_t __attribute__((ext_vector_type(132)));
    using type = d132_t;
    union {
        type d132_;
        static_buffer<d1_t, 132> d1x132_;
        static_buffer<d2_t, 66> d2x66_;
        static_buffer<d4_t, 33> d4x33_;
        static_buffer<d12_t, 11> d12x11_;
        static_buffer<d44_t, 3> d44x3_;
        static_buffer<d132_t, 1> d132x1_;
    } data_;
    DEVICE_HOST constexpr vector_type() : data_{type{0}} {}
    DEVICE_HOST constexpr vector_type(type v) : data_{v} {}
    template<typename VEC> DEVICE_HOST constexpr const auto& to_varray() const { return data_.d132_; }
    template<typename VEC> DEVICE_HOST constexpr auto&       to_varray()       { return data_.d132_; }
    template<> DEVICE_HOST constexpr const auto& to_varray<d1_t>() const { return data_.d1x132_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d1_t>()       { return data_.d1x132_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d2_t>() const { return data_.d2x66_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d2_t>()       { return data_.d2x66_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d4_t>() const { return data_.d4x33_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d4_t>()       { return data_.d4x33_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d12_t>() const { return data_.d12x11_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d12_t>()       { return data_.d12x11_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d44_t>() const { return data_.d44x3_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d44_t>()       { return data_.d44x3_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d132_t>() const { return data_.d132x1_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d132_t>()       { return data_.d132x1_;}
};

template <typename T> struct vector_type<T, 136> {
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d4_t __attribute__((ext_vector_type(4)));
    typedef T d8_t __attribute__((ext_vector_type(8)));
    typedef T d68_t __attribute__((ext_vector_type(68)));
    typedef T d136_t __attribute__((ext_vector_type(136)));
    using type = d136_t;
    union {
        type d136_;
        static_buffer<d1_t, 136> d1x136_;
        static_buffer<d2_t, 68> d2x68_;
        static_buffer<d4_t, 34> d4x34_;
        static_buffer<d8_t, 17> d8x17_;
        static_buffer<d68_t, 2> d68x2_;
        static_buffer<d136_t, 1> d136x1_;
    } data_;
    DEVICE_HOST constexpr vector_type() : data_{type{0}} {}
    DEVICE_HOST constexpr vector_type(type v) : data_{v} {}
    template<typename VEC> DEVICE_HOST constexpr const auto& to_varray() const { return data_.d136_; }
    template<typename VEC> DEVICE_HOST constexpr auto&       to_varray()       { return data_.d136_; }
    template<> DEVICE_HOST constexpr const auto& to_varray<d1_t>() const { return data_.d1x136_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d1_t>()       { return data_.d1x136_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d2_t>() const { return data_.d2x68_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d2_t>()       { return data_.d2x68_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d4_t>() const { return data_.d4x34_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d4_t>()       { return data_.d4x34_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d8_t>() const { return data_.d8x17_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d8_t>()       { return data_.d8x17_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d68_t>() const { return data_.d68x2_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d68_t>()       { return data_.d68x2_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d136_t>() const { return data_.d136x1_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d136_t>()       { return data_.d136x1_;}
};

template <typename T> struct vector_type<T, 140> {
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d4_t __attribute__((ext_vector_type(4)));
    typedef T d20_t __attribute__((ext_vector_type(20)));
    typedef T d28_t __attribute__((ext_vector_type(28)));
    typedef T d140_t __attribute__((ext_vector_type(140)));
    using type = d140_t;
    union {
        type d140_;
        static_buffer<d1_t, 140> d1x140_;
        static_buffer<d2_t, 70> d2x70_;
        static_buffer<d4_t, 35> d4x35_;
        static_buffer<d20_t, 7> d20x7_;
        static_buffer<d28_t, 5> d28x5_;
        static_buffer<d140_t, 1> d140x1_;
    } data_;
    DEVICE_HOST constexpr vector_type() : data_{type{0}} {}
    DEVICE_HOST constexpr vector_type(type v) : data_{v} {}
    template<typename VEC> DEVICE_HOST constexpr const auto& to_varray() const { return data_.d140_; }
    template<typename VEC> DEVICE_HOST constexpr auto&       to_varray()       { return data_.d140_; }
    template<> DEVICE_HOST constexpr const auto& to_varray<d1_t>() const { return data_.d1x140_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d1_t>()       { return data_.d1x140_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d2_t>() const { return data_.d2x70_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d2_t>()       { return data_.d2x70_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d4_t>() const { return data_.d4x35_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d4_t>()       { return data_.d4x35_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d20_t>() const { return data_.d20x7_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d20_t>()       { return data_.d20x7_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d28_t>() const { return data_.d28x5_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d28_t>()       { return data_.d28x5_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d140_t>() const { return data_.d140x1_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d140_t>()       { return data_.d140x1_;}
};

template <typename T> struct vector_type<T, 144> {
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d4_t __attribute__((ext_vector_type(4)));
    typedef T d8_t __attribute__((ext_vector_type(8)));
    typedef T d12_t __attribute__((ext_vector_type(12)));
    typedef T d16_t __attribute__((ext_vector_type(16)));
    typedef T d24_t __attribute__((ext_vector_type(24)));
    typedef T d36_t __attribute__((ext_vector_type(36)));
    typedef T d48_t __attribute__((ext_vector_type(48)));
    typedef T d72_t __attribute__((ext_vector_type(72)));
    typedef T d144_t __attribute__((ext_vector_type(144)));
    using type = d144_t;
    union {
        type d144_;
        static_buffer<d1_t, 144> d1x144_;
        static_buffer<d2_t, 72> d2x72_;
        static_buffer<d4_t, 36> d4x36_;
        static_buffer<d8_t, 18> d8x18_;
        static_buffer<d12_t, 12> d12x12_;
        static_buffer<d16_t, 9> d16x9_;
        static_buffer<d24_t, 6> d24x6_;
        static_buffer<d36_t, 4> d36x4_;
        static_buffer<d48_t, 3> d48x3_;
        static_buffer<d72_t, 2> d72x2_;
        static_buffer<d144_t, 1> d144x1_;
    } data_;
    DEVICE_HOST constexpr vector_type() : data_{type{0}} {}
    DEVICE_HOST constexpr vector_type(type v) : data_{v} {}
    template<typename VEC> DEVICE_HOST constexpr const auto& to_varray() const { return data_.d144_; }
    template<typename VEC> DEVICE_HOST constexpr auto&       to_varray()       { return data_.d144_; }
    template<> DEVICE_HOST constexpr const auto& to_varray<d1_t>() const { return data_.d1x144_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d1_t>()       { return data_.d1x144_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d2_t>() const { return data_.d2x72_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d2_t>()       { return data_.d2x72_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d4_t>() const { return data_.d4x36_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d4_t>()       { return data_.d4x36_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d8_t>() const { return data_.d8x18_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d8_t>()       { return data_.d8x18_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d12_t>() const { return data_.d12x12_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d12_t>()       { return data_.d12x12_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d16_t>() const { return data_.d16x9_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d16_t>()       { return data_.d16x9_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d24_t>() const { return data_.d24x6_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d24_t>()       { return data_.d24x6_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d36_t>() const { return data_.d36x4_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d36_t>()       { return data_.d36x4_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d48_t>() const { return data_.d48x3_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d48_t>()       { return data_.d48x3_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d72_t>() const { return data_.d72x2_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d72_t>()       { return data_.d72x2_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d144_t>() const { return data_.d144x1_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d144_t>()       { return data_.d144x1_;}
};

template <typename T> struct vector_type<T, 148> {
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d4_t __attribute__((ext_vector_type(4)));
    typedef T d148_t __attribute__((ext_vector_type(148)));
    using type = d148_t;
    union {
        type d148_;
        static_buffer<d1_t, 148> d1x148_;
        static_buffer<d2_t, 74> d2x74_;
        static_buffer<d4_t, 37> d4x37_;
        static_buffer<d148_t, 1> d148x1_;
    } data_;
    DEVICE_HOST constexpr vector_type() : data_{type{0}} {}
    DEVICE_HOST constexpr vector_type(type v) : data_{v} {}
    template<typename VEC> DEVICE_HOST constexpr const auto& to_varray() const { return data_.d148_; }
    template<typename VEC> DEVICE_HOST constexpr auto&       to_varray()       { return data_.d148_; }
    template<> DEVICE_HOST constexpr const auto& to_varray<d1_t>() const { return data_.d1x148_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d1_t>()       { return data_.d1x148_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d2_t>() const { return data_.d2x74_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d2_t>()       { return data_.d2x74_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d4_t>() const { return data_.d4x37_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d4_t>()       { return data_.d4x37_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d148_t>() const { return data_.d148x1_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d148_t>()       { return data_.d148x1_;}
};

template <typename T> struct vector_type<T, 152> {
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d4_t __attribute__((ext_vector_type(4)));
    typedef T d8_t __attribute__((ext_vector_type(8)));
    typedef T d76_t __attribute__((ext_vector_type(76)));
    typedef T d152_t __attribute__((ext_vector_type(152)));
    using type = d152_t;
    union {
        type d152_;
        static_buffer<d1_t, 152> d1x152_;
        static_buffer<d2_t, 76> d2x76_;
        static_buffer<d4_t, 38> d4x38_;
        static_buffer<d8_t, 19> d8x19_;
        static_buffer<d76_t, 2> d76x2_;
        static_buffer<d152_t, 1> d152x1_;
    } data_;
    DEVICE_HOST constexpr vector_type() : data_{type{0}} {}
    DEVICE_HOST constexpr vector_type(type v) : data_{v} {}
    template<typename VEC> DEVICE_HOST constexpr const auto& to_varray() const { return data_.d152_; }
    template<typename VEC> DEVICE_HOST constexpr auto&       to_varray()       { return data_.d152_; }
    template<> DEVICE_HOST constexpr const auto& to_varray<d1_t>() const { return data_.d1x152_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d1_t>()       { return data_.d1x152_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d2_t>() const { return data_.d2x76_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d2_t>()       { return data_.d2x76_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d4_t>() const { return data_.d4x38_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d4_t>()       { return data_.d4x38_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d8_t>() const { return data_.d8x19_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d8_t>()       { return data_.d8x19_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d76_t>() const { return data_.d76x2_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d76_t>()       { return data_.d76x2_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d152_t>() const { return data_.d152x1_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d152_t>()       { return data_.d152x1_;}
};

template <typename T> struct vector_type<T, 156> {
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d4_t __attribute__((ext_vector_type(4)));
    typedef T d12_t __attribute__((ext_vector_type(12)));
    typedef T d52_t __attribute__((ext_vector_type(52)));
    typedef T d156_t __attribute__((ext_vector_type(156)));
    using type = d156_t;
    union {
        type d156_;
        static_buffer<d1_t, 156> d1x156_;
        static_buffer<d2_t, 78> d2x78_;
        static_buffer<d4_t, 39> d4x39_;
        static_buffer<d12_t, 13> d12x13_;
        static_buffer<d52_t, 3> d52x3_;
        static_buffer<d156_t, 1> d156x1_;
    } data_;
    DEVICE_HOST constexpr vector_type() : data_{type{0}} {}
    DEVICE_HOST constexpr vector_type(type v) : data_{v} {}
    template<typename VEC> DEVICE_HOST constexpr const auto& to_varray() const { return data_.d156_; }
    template<typename VEC> DEVICE_HOST constexpr auto&       to_varray()       { return data_.d156_; }
    template<> DEVICE_HOST constexpr const auto& to_varray<d1_t>() const { return data_.d1x156_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d1_t>()       { return data_.d1x156_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d2_t>() const { return data_.d2x78_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d2_t>()       { return data_.d2x78_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d4_t>() const { return data_.d4x39_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d4_t>()       { return data_.d4x39_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d12_t>() const { return data_.d12x13_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d12_t>()       { return data_.d12x13_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d52_t>() const { return data_.d52x3_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d52_t>()       { return data_.d52x3_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d156_t>() const { return data_.d156x1_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d156_t>()       { return data_.d156x1_;}
};

template <typename T> struct vector_type<T, 160> {
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d4_t __attribute__((ext_vector_type(4)));
    typedef T d8_t __attribute__((ext_vector_type(8)));
    typedef T d16_t __attribute__((ext_vector_type(16)));
    typedef T d20_t __attribute__((ext_vector_type(20)));
    typedef T d32_t __attribute__((ext_vector_type(32)));
    typedef T d40_t __attribute__((ext_vector_type(40)));
    typedef T d80_t __attribute__((ext_vector_type(80)));
    typedef T d160_t __attribute__((ext_vector_type(160)));
    using type = d160_t;
    union {
        type d160_;
        static_buffer<d1_t, 160> d1x160_;
        static_buffer<d2_t, 80> d2x80_;
        static_buffer<d4_t, 40> d4x40_;
        static_buffer<d8_t, 20> d8x20_;
        static_buffer<d16_t, 10> d16x10_;
        static_buffer<d20_t, 8> d20x8_;
        static_buffer<d32_t, 5> d32x5_;
        static_buffer<d40_t, 4> d40x4_;
        static_buffer<d80_t, 2> d80x2_;
        static_buffer<d160_t, 1> d160x1_;
    } data_;
    DEVICE_HOST constexpr vector_type() : data_{type{0}} {}
    DEVICE_HOST constexpr vector_type(type v) : data_{v} {}
    template<typename VEC> DEVICE_HOST constexpr const auto& to_varray() const { return data_.d160_; }
    template<typename VEC> DEVICE_HOST constexpr auto&       to_varray()       { return data_.d160_; }
    template<> DEVICE_HOST constexpr const auto& to_varray<d1_t>() const { return data_.d1x160_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d1_t>()       { return data_.d1x160_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d2_t>() const { return data_.d2x80_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d2_t>()       { return data_.d2x80_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d4_t>() const { return data_.d4x40_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d4_t>()       { return data_.d4x40_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d8_t>() const { return data_.d8x20_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d8_t>()       { return data_.d8x20_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d16_t>() const { return data_.d16x10_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d16_t>()       { return data_.d16x10_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d20_t>() const { return data_.d20x8_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d20_t>()       { return data_.d20x8_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d32_t>() const { return data_.d32x5_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d32_t>()       { return data_.d32x5_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d40_t>() const { return data_.d40x4_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d40_t>()       { return data_.d40x4_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d80_t>() const { return data_.d80x2_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d80_t>()       { return data_.d80x2_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d160_t>() const { return data_.d160x1_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d160_t>()       { return data_.d160x1_;}
};

template <typename T> struct vector_type<T, 164> {
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d4_t __attribute__((ext_vector_type(4)));
    typedef T d164_t __attribute__((ext_vector_type(164)));
    using type = d164_t;
    union {
        type d164_;
        static_buffer<d1_t, 164> d1x164_;
        static_buffer<d2_t, 82> d2x82_;
        static_buffer<d4_t, 41> d4x41_;
        static_buffer<d164_t, 1> d164x1_;
    } data_;
    DEVICE_HOST constexpr vector_type() : data_{type{0}} {}
    DEVICE_HOST constexpr vector_type(type v) : data_{v} {}
    template<typename VEC> DEVICE_HOST constexpr const auto& to_varray() const { return data_.d164_; }
    template<typename VEC> DEVICE_HOST constexpr auto&       to_varray()       { return data_.d164_; }
    template<> DEVICE_HOST constexpr const auto& to_varray<d1_t>() const { return data_.d1x164_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d1_t>()       { return data_.d1x164_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d2_t>() const { return data_.d2x82_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d2_t>()       { return data_.d2x82_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d4_t>() const { return data_.d4x41_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d4_t>()       { return data_.d4x41_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d164_t>() const { return data_.d164x1_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d164_t>()       { return data_.d164x1_;}
};

template <typename T> struct vector_type<T, 168> {
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d4_t __attribute__((ext_vector_type(4)));
    typedef T d8_t __attribute__((ext_vector_type(8)));
    typedef T d12_t __attribute__((ext_vector_type(12)));
    typedef T d24_t __attribute__((ext_vector_type(24)));
    typedef T d28_t __attribute__((ext_vector_type(28)));
    typedef T d56_t __attribute__((ext_vector_type(56)));
    typedef T d84_t __attribute__((ext_vector_type(84)));
    typedef T d168_t __attribute__((ext_vector_type(168)));
    using type = d168_t;
    union {
        type d168_;
        static_buffer<d1_t, 168> d1x168_;
        static_buffer<d2_t, 84> d2x84_;
        static_buffer<d4_t, 42> d4x42_;
        static_buffer<d8_t, 21> d8x21_;
        static_buffer<d12_t, 14> d12x14_;
        static_buffer<d24_t, 7> d24x7_;
        static_buffer<d28_t, 6> d28x6_;
        static_buffer<d56_t, 3> d56x3_;
        static_buffer<d84_t, 2> d84x2_;
        static_buffer<d168_t, 1> d168x1_;
    } data_;
    DEVICE_HOST constexpr vector_type() : data_{type{0}} {}
    DEVICE_HOST constexpr vector_type(type v) : data_{v} {}
    template<typename VEC> DEVICE_HOST constexpr const auto& to_varray() const { return data_.d168_; }
    template<typename VEC> DEVICE_HOST constexpr auto&       to_varray()       { return data_.d168_; }
    template<> DEVICE_HOST constexpr const auto& to_varray<d1_t>() const { return data_.d1x168_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d1_t>()       { return data_.d1x168_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d2_t>() const { return data_.d2x84_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d2_t>()       { return data_.d2x84_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d4_t>() const { return data_.d4x42_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d4_t>()       { return data_.d4x42_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d8_t>() const { return data_.d8x21_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d8_t>()       { return data_.d8x21_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d12_t>() const { return data_.d12x14_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d12_t>()       { return data_.d12x14_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d24_t>() const { return data_.d24x7_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d24_t>()       { return data_.d24x7_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d28_t>() const { return data_.d28x6_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d28_t>()       { return data_.d28x6_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d56_t>() const { return data_.d56x3_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d56_t>()       { return data_.d56x3_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d84_t>() const { return data_.d84x2_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d84_t>()       { return data_.d84x2_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d168_t>() const { return data_.d168x1_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d168_t>()       { return data_.d168x1_;}
};

template <typename T> struct vector_type<T, 172> {
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d4_t __attribute__((ext_vector_type(4)));
    typedef T d172_t __attribute__((ext_vector_type(172)));
    using type = d172_t;
    union {
        type d172_;
        static_buffer<d1_t, 172> d1x172_;
        static_buffer<d2_t, 86> d2x86_;
        static_buffer<d4_t, 43> d4x43_;
        static_buffer<d172_t, 1> d172x1_;
    } data_;
    DEVICE_HOST constexpr vector_type() : data_{type{0}} {}
    DEVICE_HOST constexpr vector_type(type v) : data_{v} {}
    template<typename VEC> DEVICE_HOST constexpr const auto& to_varray() const { return data_.d172_; }
    template<typename VEC> DEVICE_HOST constexpr auto&       to_varray()       { return data_.d172_; }
    template<> DEVICE_HOST constexpr const auto& to_varray<d1_t>() const { return data_.d1x172_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d1_t>()       { return data_.d1x172_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d2_t>() const { return data_.d2x86_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d2_t>()       { return data_.d2x86_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d4_t>() const { return data_.d4x43_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d4_t>()       { return data_.d4x43_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d172_t>() const { return data_.d172x1_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d172_t>()       { return data_.d172x1_;}
};

template <typename T> struct vector_type<T, 176> {
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d4_t __attribute__((ext_vector_type(4)));
    typedef T d8_t __attribute__((ext_vector_type(8)));
    typedef T d16_t __attribute__((ext_vector_type(16)));
    typedef T d44_t __attribute__((ext_vector_type(44)));
    typedef T d88_t __attribute__((ext_vector_type(88)));
    typedef T d176_t __attribute__((ext_vector_type(176)));
    using type = d176_t;
    union {
        type d176_;
        static_buffer<d1_t, 176> d1x176_;
        static_buffer<d2_t, 88> d2x88_;
        static_buffer<d4_t, 44> d4x44_;
        static_buffer<d8_t, 22> d8x22_;
        static_buffer<d16_t, 11> d16x11_;
        static_buffer<d44_t, 4> d44x4_;
        static_buffer<d88_t, 2> d88x2_;
        static_buffer<d176_t, 1> d176x1_;
    } data_;
    DEVICE_HOST constexpr vector_type() : data_{type{0}} {}
    DEVICE_HOST constexpr vector_type(type v) : data_{v} {}
    template<typename VEC> DEVICE_HOST constexpr const auto& to_varray() const { return data_.d176_; }
    template<typename VEC> DEVICE_HOST constexpr auto&       to_varray()       { return data_.d176_; }
    template<> DEVICE_HOST constexpr const auto& to_varray<d1_t>() const { return data_.d1x176_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d1_t>()       { return data_.d1x176_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d2_t>() const { return data_.d2x88_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d2_t>()       { return data_.d2x88_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d4_t>() const { return data_.d4x44_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d4_t>()       { return data_.d4x44_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d8_t>() const { return data_.d8x22_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d8_t>()       { return data_.d8x22_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d16_t>() const { return data_.d16x11_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d16_t>()       { return data_.d16x11_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d44_t>() const { return data_.d44x4_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d44_t>()       { return data_.d44x4_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d88_t>() const { return data_.d88x2_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d88_t>()       { return data_.d88x2_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d176_t>() const { return data_.d176x1_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d176_t>()       { return data_.d176x1_;}
};

template <typename T> struct vector_type<T, 180> {
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d4_t __attribute__((ext_vector_type(4)));
    typedef T d12_t __attribute__((ext_vector_type(12)));
    typedef T d20_t __attribute__((ext_vector_type(20)));
    typedef T d36_t __attribute__((ext_vector_type(36)));
    typedef T d60_t __attribute__((ext_vector_type(60)));
    typedef T d180_t __attribute__((ext_vector_type(180)));
    using type = d180_t;
    union {
        type d180_;
        static_buffer<d1_t, 180> d1x180_;
        static_buffer<d2_t, 90> d2x90_;
        static_buffer<d4_t, 45> d4x45_;
        static_buffer<d12_t, 15> d12x15_;
        static_buffer<d20_t, 9> d20x9_;
        static_buffer<d36_t, 5> d36x5_;
        static_buffer<d60_t, 3> d60x3_;
        static_buffer<d180_t, 1> d180x1_;
    } data_;
    DEVICE_HOST constexpr vector_type() : data_{type{0}} {}
    DEVICE_HOST constexpr vector_type(type v) : data_{v} {}
    template<typename VEC> DEVICE_HOST constexpr const auto& to_varray() const { return data_.d180_; }
    template<typename VEC> DEVICE_HOST constexpr auto&       to_varray()       { return data_.d180_; }
    template<> DEVICE_HOST constexpr const auto& to_varray<d1_t>() const { return data_.d1x180_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d1_t>()       { return data_.d1x180_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d2_t>() const { return data_.d2x90_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d2_t>()       { return data_.d2x90_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d4_t>() const { return data_.d4x45_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d4_t>()       { return data_.d4x45_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d12_t>() const { return data_.d12x15_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d12_t>()       { return data_.d12x15_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d20_t>() const { return data_.d20x9_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d20_t>()       { return data_.d20x9_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d36_t>() const { return data_.d36x5_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d36_t>()       { return data_.d36x5_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d60_t>() const { return data_.d60x3_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d60_t>()       { return data_.d60x3_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d180_t>() const { return data_.d180x1_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d180_t>()       { return data_.d180x1_;}
};

template <typename T> struct vector_type<T, 184> {
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d4_t __attribute__((ext_vector_type(4)));
    typedef T d8_t __attribute__((ext_vector_type(8)));
    typedef T d92_t __attribute__((ext_vector_type(92)));
    typedef T d184_t __attribute__((ext_vector_type(184)));
    using type = d184_t;
    union {
        type d184_;
        static_buffer<d1_t, 184> d1x184_;
        static_buffer<d2_t, 92> d2x92_;
        static_buffer<d4_t, 46> d4x46_;
        static_buffer<d8_t, 23> d8x23_;
        static_buffer<d92_t, 2> d92x2_;
        static_buffer<d184_t, 1> d184x1_;
    } data_;
    DEVICE_HOST constexpr vector_type() : data_{type{0}} {}
    DEVICE_HOST constexpr vector_type(type v) : data_{v} {}
    template<typename VEC> DEVICE_HOST constexpr const auto& to_varray() const { return data_.d184_; }
    template<typename VEC> DEVICE_HOST constexpr auto&       to_varray()       { return data_.d184_; }
    template<> DEVICE_HOST constexpr const auto& to_varray<d1_t>() const { return data_.d1x184_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d1_t>()       { return data_.d1x184_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d2_t>() const { return data_.d2x92_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d2_t>()       { return data_.d2x92_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d4_t>() const { return data_.d4x46_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d4_t>()       { return data_.d4x46_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d8_t>() const { return data_.d8x23_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d8_t>()       { return data_.d8x23_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d92_t>() const { return data_.d92x2_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d92_t>()       { return data_.d92x2_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d184_t>() const { return data_.d184x1_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d184_t>()       { return data_.d184x1_;}
};

template <typename T> struct vector_type<T, 188> {
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d4_t __attribute__((ext_vector_type(4)));
    typedef T d188_t __attribute__((ext_vector_type(188)));
    using type = d188_t;
    union {
        type d188_;
        static_buffer<d1_t, 188> d1x188_;
        static_buffer<d2_t, 94> d2x94_;
        static_buffer<d4_t, 47> d4x47_;
        static_buffer<d188_t, 1> d188x1_;
    } data_;
    DEVICE_HOST constexpr vector_type() : data_{type{0}} {}
    DEVICE_HOST constexpr vector_type(type v) : data_{v} {}
    template<typename VEC> DEVICE_HOST constexpr const auto& to_varray() const { return data_.d188_; }
    template<typename VEC> DEVICE_HOST constexpr auto&       to_varray()       { return data_.d188_; }
    template<> DEVICE_HOST constexpr const auto& to_varray<d1_t>() const { return data_.d1x188_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d1_t>()       { return data_.d1x188_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d2_t>() const { return data_.d2x94_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d2_t>()       { return data_.d2x94_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d4_t>() const { return data_.d4x47_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d4_t>()       { return data_.d4x47_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d188_t>() const { return data_.d188x1_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d188_t>()       { return data_.d188x1_;}
};

template <typename T> struct vector_type<T, 192> {
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d4_t __attribute__((ext_vector_type(4)));
    typedef T d8_t __attribute__((ext_vector_type(8)));
    typedef T d12_t __attribute__((ext_vector_type(12)));
    typedef T d16_t __attribute__((ext_vector_type(16)));
    typedef T d24_t __attribute__((ext_vector_type(24)));
    typedef T d32_t __attribute__((ext_vector_type(32)));
    typedef T d48_t __attribute__((ext_vector_type(48)));
    typedef T d64_t __attribute__((ext_vector_type(64)));
    typedef T d96_t __attribute__((ext_vector_type(96)));
    typedef T d192_t __attribute__((ext_vector_type(192)));
    using type = d192_t;
    union {
        type d192_;
        static_buffer<d1_t, 192> d1x192_;
        static_buffer<d2_t, 96> d2x96_;
        static_buffer<d4_t, 48> d4x48_;
        static_buffer<d8_t, 24> d8x24_;
        static_buffer<d12_t, 16> d12x16_;
        static_buffer<d16_t, 12> d16x12_;
        static_buffer<d24_t, 8> d24x8_;
        static_buffer<d32_t, 6> d32x6_;
        static_buffer<d48_t, 4> d48x4_;
        static_buffer<d64_t, 3> d64x3_;
        static_buffer<d96_t, 2> d96x2_;
        static_buffer<d192_t, 1> d192x1_;
    } data_;
    DEVICE_HOST constexpr vector_type() : data_{type{0}} {}
    DEVICE_HOST constexpr vector_type(type v) : data_{v} {}
    template<typename VEC> DEVICE_HOST constexpr const auto& to_varray() const { return data_.d192_; }
    template<typename VEC> DEVICE_HOST constexpr auto&       to_varray()       { return data_.d192_; }
    template<> DEVICE_HOST constexpr const auto& to_varray<d1_t>() const { return data_.d1x192_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d1_t>()       { return data_.d1x192_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d2_t>() const { return data_.d2x96_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d2_t>()       { return data_.d2x96_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d4_t>() const { return data_.d4x48_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d4_t>()       { return data_.d4x48_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d8_t>() const { return data_.d8x24_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d8_t>()       { return data_.d8x24_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d12_t>() const { return data_.d12x16_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d12_t>()       { return data_.d12x16_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d16_t>() const { return data_.d16x12_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d16_t>()       { return data_.d16x12_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d24_t>() const { return data_.d24x8_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d24_t>()       { return data_.d24x8_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d32_t>() const { return data_.d32x6_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d32_t>()       { return data_.d32x6_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d48_t>() const { return data_.d48x4_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d48_t>()       { return data_.d48x4_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d64_t>() const { return data_.d64x3_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d64_t>()       { return data_.d64x3_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d96_t>() const { return data_.d96x2_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d96_t>()       { return data_.d96x2_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d192_t>() const { return data_.d192x1_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d192_t>()       { return data_.d192x1_;}
};

template <typename T> struct vector_type<T, 196> {
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d4_t __attribute__((ext_vector_type(4)));
    typedef T d28_t __attribute__((ext_vector_type(28)));
    typedef T d196_t __attribute__((ext_vector_type(196)));
    using type = d196_t;
    union {
        type d196_;
        static_buffer<d1_t, 196> d1x196_;
        static_buffer<d2_t, 98> d2x98_;
        static_buffer<d4_t, 49> d4x49_;
        static_buffer<d28_t, 7> d28x7_;
        static_buffer<d196_t, 1> d196x1_;
    } data_;
    DEVICE_HOST constexpr vector_type() : data_{type{0}} {}
    DEVICE_HOST constexpr vector_type(type v) : data_{v} {}
    template<typename VEC> DEVICE_HOST constexpr const auto& to_varray() const { return data_.d196_; }
    template<typename VEC> DEVICE_HOST constexpr auto&       to_varray()       { return data_.d196_; }
    template<> DEVICE_HOST constexpr const auto& to_varray<d1_t>() const { return data_.d1x196_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d1_t>()       { return data_.d1x196_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d2_t>() const { return data_.d2x98_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d2_t>()       { return data_.d2x98_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d4_t>() const { return data_.d4x49_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d4_t>()       { return data_.d4x49_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d28_t>() const { return data_.d28x7_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d28_t>()       { return data_.d28x7_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d196_t>() const { return data_.d196x1_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d196_t>()       { return data_.d196x1_;}
};

template <typename T> struct vector_type<T, 200> {
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d4_t __attribute__((ext_vector_type(4)));
    typedef T d8_t __attribute__((ext_vector_type(8)));
    typedef T d20_t __attribute__((ext_vector_type(20)));
    typedef T d40_t __attribute__((ext_vector_type(40)));
    typedef T d100_t __attribute__((ext_vector_type(100)));
    typedef T d200_t __attribute__((ext_vector_type(200)));
    using type = d200_t;
    union {
        type d200_;
        static_buffer<d1_t, 200> d1x200_;
        static_buffer<d2_t, 100> d2x100_;
        static_buffer<d4_t, 50> d4x50_;
        static_buffer<d8_t, 25> d8x25_;
        static_buffer<d20_t, 10> d20x10_;
        static_buffer<d40_t, 5> d40x5_;
        static_buffer<d100_t, 2> d100x2_;
        static_buffer<d200_t, 1> d200x1_;
    } data_;
    DEVICE_HOST constexpr vector_type() : data_{type{0}} {}
    DEVICE_HOST constexpr vector_type(type v) : data_{v} {}
    template<typename VEC> DEVICE_HOST constexpr const auto& to_varray() const { return data_.d200_; }
    template<typename VEC> DEVICE_HOST constexpr auto&       to_varray()       { return data_.d200_; }
    template<> DEVICE_HOST constexpr const auto& to_varray<d1_t>() const { return data_.d1x200_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d1_t>()       { return data_.d1x200_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d2_t>() const { return data_.d2x100_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d2_t>()       { return data_.d2x100_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d4_t>() const { return data_.d4x50_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d4_t>()       { return data_.d4x50_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d8_t>() const { return data_.d8x25_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d8_t>()       { return data_.d8x25_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d20_t>() const { return data_.d20x10_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d20_t>()       { return data_.d20x10_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d40_t>() const { return data_.d40x5_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d40_t>()       { return data_.d40x5_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d100_t>() const { return data_.d100x2_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d100_t>()       { return data_.d100x2_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d200_t>() const { return data_.d200x1_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d200_t>()       { return data_.d200x1_;}
};

template <typename T> struct vector_type<T, 204> {
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d4_t __attribute__((ext_vector_type(4)));
    typedef T d12_t __attribute__((ext_vector_type(12)));
    typedef T d68_t __attribute__((ext_vector_type(68)));
    typedef T d204_t __attribute__((ext_vector_type(204)));
    using type = d204_t;
    union {
        type d204_;
        static_buffer<d1_t, 204> d1x204_;
        static_buffer<d2_t, 102> d2x102_;
        static_buffer<d4_t, 51> d4x51_;
        static_buffer<d12_t, 17> d12x17_;
        static_buffer<d68_t, 3> d68x3_;
        static_buffer<d204_t, 1> d204x1_;
    } data_;
    DEVICE_HOST constexpr vector_type() : data_{type{0}} {}
    DEVICE_HOST constexpr vector_type(type v) : data_{v} {}
    template<typename VEC> DEVICE_HOST constexpr const auto& to_varray() const { return data_.d204_; }
    template<typename VEC> DEVICE_HOST constexpr auto&       to_varray()       { return data_.d204_; }
    template<> DEVICE_HOST constexpr const auto& to_varray<d1_t>() const { return data_.d1x204_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d1_t>()       { return data_.d1x204_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d2_t>() const { return data_.d2x102_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d2_t>()       { return data_.d2x102_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d4_t>() const { return data_.d4x51_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d4_t>()       { return data_.d4x51_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d12_t>() const { return data_.d12x17_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d12_t>()       { return data_.d12x17_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d68_t>() const { return data_.d68x3_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d68_t>()       { return data_.d68x3_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d204_t>() const { return data_.d204x1_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d204_t>()       { return data_.d204x1_;}
};

template <typename T> struct vector_type<T, 208> {
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d4_t __attribute__((ext_vector_type(4)));
    typedef T d8_t __attribute__((ext_vector_type(8)));
    typedef T d16_t __attribute__((ext_vector_type(16)));
    typedef T d52_t __attribute__((ext_vector_type(52)));
    typedef T d104_t __attribute__((ext_vector_type(104)));
    typedef T d208_t __attribute__((ext_vector_type(208)));
    using type = d208_t;
    union {
        type d208_;
        static_buffer<d1_t, 208> d1x208_;
        static_buffer<d2_t, 104> d2x104_;
        static_buffer<d4_t, 52> d4x52_;
        static_buffer<d8_t, 26> d8x26_;
        static_buffer<d16_t, 13> d16x13_;
        static_buffer<d52_t, 4> d52x4_;
        static_buffer<d104_t, 2> d104x2_;
        static_buffer<d208_t, 1> d208x1_;
    } data_;
    DEVICE_HOST constexpr vector_type() : data_{type{0}} {}
    DEVICE_HOST constexpr vector_type(type v) : data_{v} {}
    template<typename VEC> DEVICE_HOST constexpr const auto& to_varray() const { return data_.d208_; }
    template<typename VEC> DEVICE_HOST constexpr auto&       to_varray()       { return data_.d208_; }
    template<> DEVICE_HOST constexpr const auto& to_varray<d1_t>() const { return data_.d1x208_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d1_t>()       { return data_.d1x208_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d2_t>() const { return data_.d2x104_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d2_t>()       { return data_.d2x104_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d4_t>() const { return data_.d4x52_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d4_t>()       { return data_.d4x52_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d8_t>() const { return data_.d8x26_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d8_t>()       { return data_.d8x26_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d16_t>() const { return data_.d16x13_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d16_t>()       { return data_.d16x13_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d52_t>() const { return data_.d52x4_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d52_t>()       { return data_.d52x4_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d104_t>() const { return data_.d104x2_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d104_t>()       { return data_.d104x2_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d208_t>() const { return data_.d208x1_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d208_t>()       { return data_.d208x1_;}
};

template <typename T> struct vector_type<T, 212> {
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d4_t __attribute__((ext_vector_type(4)));
    typedef T d212_t __attribute__((ext_vector_type(212)));
    using type = d212_t;
    union {
        type d212_;
        static_buffer<d1_t, 212> d1x212_;
        static_buffer<d2_t, 106> d2x106_;
        static_buffer<d4_t, 53> d4x53_;
        static_buffer<d212_t, 1> d212x1_;
    } data_;
    DEVICE_HOST constexpr vector_type() : data_{type{0}} {}
    DEVICE_HOST constexpr vector_type(type v) : data_{v} {}
    template<typename VEC> DEVICE_HOST constexpr const auto& to_varray() const { return data_.d212_; }
    template<typename VEC> DEVICE_HOST constexpr auto&       to_varray()       { return data_.d212_; }
    template<> DEVICE_HOST constexpr const auto& to_varray<d1_t>() const { return data_.d1x212_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d1_t>()       { return data_.d1x212_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d2_t>() const { return data_.d2x106_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d2_t>()       { return data_.d2x106_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d4_t>() const { return data_.d4x53_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d4_t>()       { return data_.d4x53_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d212_t>() const { return data_.d212x1_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d212_t>()       { return data_.d212x1_;}
};

template <typename T> struct vector_type<T, 216> {
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d4_t __attribute__((ext_vector_type(4)));
    typedef T d8_t __attribute__((ext_vector_type(8)));
    typedef T d12_t __attribute__((ext_vector_type(12)));
    typedef T d24_t __attribute__((ext_vector_type(24)));
    typedef T d36_t __attribute__((ext_vector_type(36)));
    typedef T d72_t __attribute__((ext_vector_type(72)));
    typedef T d108_t __attribute__((ext_vector_type(108)));
    typedef T d216_t __attribute__((ext_vector_type(216)));
    using type = d216_t;
    union {
        type d216_;
        static_buffer<d1_t, 216> d1x216_;
        static_buffer<d2_t, 108> d2x108_;
        static_buffer<d4_t, 54> d4x54_;
        static_buffer<d8_t, 27> d8x27_;
        static_buffer<d12_t, 18> d12x18_;
        static_buffer<d24_t, 9> d24x9_;
        static_buffer<d36_t, 6> d36x6_;
        static_buffer<d72_t, 3> d72x3_;
        static_buffer<d108_t, 2> d108x2_;
        static_buffer<d216_t, 1> d216x1_;
    } data_;
    DEVICE_HOST constexpr vector_type() : data_{type{0}} {}
    DEVICE_HOST constexpr vector_type(type v) : data_{v} {}
    template<typename VEC> DEVICE_HOST constexpr const auto& to_varray() const { return data_.d216_; }
    template<typename VEC> DEVICE_HOST constexpr auto&       to_varray()       { return data_.d216_; }
    template<> DEVICE_HOST constexpr const auto& to_varray<d1_t>() const { return data_.d1x216_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d1_t>()       { return data_.d1x216_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d2_t>() const { return data_.d2x108_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d2_t>()       { return data_.d2x108_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d4_t>() const { return data_.d4x54_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d4_t>()       { return data_.d4x54_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d8_t>() const { return data_.d8x27_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d8_t>()       { return data_.d8x27_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d12_t>() const { return data_.d12x18_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d12_t>()       { return data_.d12x18_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d24_t>() const { return data_.d24x9_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d24_t>()       { return data_.d24x9_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d36_t>() const { return data_.d36x6_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d36_t>()       { return data_.d36x6_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d72_t>() const { return data_.d72x3_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d72_t>()       { return data_.d72x3_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d108_t>() const { return data_.d108x2_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d108_t>()       { return data_.d108x2_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d216_t>() const { return data_.d216x1_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d216_t>()       { return data_.d216x1_;}
};

template <typename T> struct vector_type<T, 220> {
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d4_t __attribute__((ext_vector_type(4)));
    typedef T d20_t __attribute__((ext_vector_type(20)));
    typedef T d44_t __attribute__((ext_vector_type(44)));
    typedef T d220_t __attribute__((ext_vector_type(220)));
    using type = d220_t;
    union {
        type d220_;
        static_buffer<d1_t, 220> d1x220_;
        static_buffer<d2_t, 110> d2x110_;
        static_buffer<d4_t, 55> d4x55_;
        static_buffer<d20_t, 11> d20x11_;
        static_buffer<d44_t, 5> d44x5_;
        static_buffer<d220_t, 1> d220x1_;
    } data_;
    DEVICE_HOST constexpr vector_type() : data_{type{0}} {}
    DEVICE_HOST constexpr vector_type(type v) : data_{v} {}
    template<typename VEC> DEVICE_HOST constexpr const auto& to_varray() const { return data_.d220_; }
    template<typename VEC> DEVICE_HOST constexpr auto&       to_varray()       { return data_.d220_; }
    template<> DEVICE_HOST constexpr const auto& to_varray<d1_t>() const { return data_.d1x220_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d1_t>()       { return data_.d1x220_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d2_t>() const { return data_.d2x110_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d2_t>()       { return data_.d2x110_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d4_t>() const { return data_.d4x55_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d4_t>()       { return data_.d4x55_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d20_t>() const { return data_.d20x11_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d20_t>()       { return data_.d20x11_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d44_t>() const { return data_.d44x5_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d44_t>()       { return data_.d44x5_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d220_t>() const { return data_.d220x1_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d220_t>()       { return data_.d220x1_;}
};

template <typename T> struct vector_type<T, 224> {
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d4_t __attribute__((ext_vector_type(4)));
    typedef T d8_t __attribute__((ext_vector_type(8)));
    typedef T d16_t __attribute__((ext_vector_type(16)));
    typedef T d28_t __attribute__((ext_vector_type(28)));
    typedef T d32_t __attribute__((ext_vector_type(32)));
    typedef T d56_t __attribute__((ext_vector_type(56)));
    typedef T d112_t __attribute__((ext_vector_type(112)));
    typedef T d224_t __attribute__((ext_vector_type(224)));
    using type = d224_t;
    union {
        type d224_;
        static_buffer<d1_t, 224> d1x224_;
        static_buffer<d2_t, 112> d2x112_;
        static_buffer<d4_t, 56> d4x56_;
        static_buffer<d8_t, 28> d8x28_;
        static_buffer<d16_t, 14> d16x14_;
        static_buffer<d28_t, 8> d28x8_;
        static_buffer<d32_t, 7> d32x7_;
        static_buffer<d56_t, 4> d56x4_;
        static_buffer<d112_t, 2> d112x2_;
        static_buffer<d224_t, 1> d224x1_;
    } data_;
    DEVICE_HOST constexpr vector_type() : data_{type{0}} {}
    DEVICE_HOST constexpr vector_type(type v) : data_{v} {}
    template<typename VEC> DEVICE_HOST constexpr const auto& to_varray() const { return data_.d224_; }
    template<typename VEC> DEVICE_HOST constexpr auto&       to_varray()       { return data_.d224_; }
    template<> DEVICE_HOST constexpr const auto& to_varray<d1_t>() const { return data_.d1x224_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d1_t>()       { return data_.d1x224_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d2_t>() const { return data_.d2x112_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d2_t>()       { return data_.d2x112_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d4_t>() const { return data_.d4x56_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d4_t>()       { return data_.d4x56_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d8_t>() const { return data_.d8x28_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d8_t>()       { return data_.d8x28_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d16_t>() const { return data_.d16x14_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d16_t>()       { return data_.d16x14_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d28_t>() const { return data_.d28x8_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d28_t>()       { return data_.d28x8_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d32_t>() const { return data_.d32x7_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d32_t>()       { return data_.d32x7_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d56_t>() const { return data_.d56x4_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d56_t>()       { return data_.d56x4_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d112_t>() const { return data_.d112x2_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d112_t>()       { return data_.d112x2_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d224_t>() const { return data_.d224x1_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d224_t>()       { return data_.d224x1_;}
};

template <typename T> struct vector_type<T, 228> {
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d4_t __attribute__((ext_vector_type(4)));
    typedef T d12_t __attribute__((ext_vector_type(12)));
    typedef T d76_t __attribute__((ext_vector_type(76)));
    typedef T d228_t __attribute__((ext_vector_type(228)));
    using type = d228_t;
    union {
        type d228_;
        static_buffer<d1_t, 228> d1x228_;
        static_buffer<d2_t, 114> d2x114_;
        static_buffer<d4_t, 57> d4x57_;
        static_buffer<d12_t, 19> d12x19_;
        static_buffer<d76_t, 3> d76x3_;
        static_buffer<d228_t, 1> d228x1_;
    } data_;
    DEVICE_HOST constexpr vector_type() : data_{type{0}} {}
    DEVICE_HOST constexpr vector_type(type v) : data_{v} {}
    template<typename VEC> DEVICE_HOST constexpr const auto& to_varray() const { return data_.d228_; }
    template<typename VEC> DEVICE_HOST constexpr auto&       to_varray()       { return data_.d228_; }
    template<> DEVICE_HOST constexpr const auto& to_varray<d1_t>() const { return data_.d1x228_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d1_t>()       { return data_.d1x228_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d2_t>() const { return data_.d2x114_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d2_t>()       { return data_.d2x114_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d4_t>() const { return data_.d4x57_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d4_t>()       { return data_.d4x57_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d12_t>() const { return data_.d12x19_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d12_t>()       { return data_.d12x19_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d76_t>() const { return data_.d76x3_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d76_t>()       { return data_.d76x3_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d228_t>() const { return data_.d228x1_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d228_t>()       { return data_.d228x1_;}
};

template <typename T> struct vector_type<T, 232> {
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d4_t __attribute__((ext_vector_type(4)));
    typedef T d8_t __attribute__((ext_vector_type(8)));
    typedef T d116_t __attribute__((ext_vector_type(116)));
    typedef T d232_t __attribute__((ext_vector_type(232)));
    using type = d232_t;
    union {
        type d232_;
        static_buffer<d1_t, 232> d1x232_;
        static_buffer<d2_t, 116> d2x116_;
        static_buffer<d4_t, 58> d4x58_;
        static_buffer<d8_t, 29> d8x29_;
        static_buffer<d116_t, 2> d116x2_;
        static_buffer<d232_t, 1> d232x1_;
    } data_;
    DEVICE_HOST constexpr vector_type() : data_{type{0}} {}
    DEVICE_HOST constexpr vector_type(type v) : data_{v} {}
    template<typename VEC> DEVICE_HOST constexpr const auto& to_varray() const { return data_.d232_; }
    template<typename VEC> DEVICE_HOST constexpr auto&       to_varray()       { return data_.d232_; }
    template<> DEVICE_HOST constexpr const auto& to_varray<d1_t>() const { return data_.d1x232_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d1_t>()       { return data_.d1x232_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d2_t>() const { return data_.d2x116_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d2_t>()       { return data_.d2x116_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d4_t>() const { return data_.d4x58_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d4_t>()       { return data_.d4x58_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d8_t>() const { return data_.d8x29_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d8_t>()       { return data_.d8x29_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d116_t>() const { return data_.d116x2_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d116_t>()       { return data_.d116x2_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d232_t>() const { return data_.d232x1_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d232_t>()       { return data_.d232x1_;}
};

template <typename T> struct vector_type<T, 236> {
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d4_t __attribute__((ext_vector_type(4)));
    typedef T d236_t __attribute__((ext_vector_type(236)));
    using type = d236_t;
    union {
        type d236_;
        static_buffer<d1_t, 236> d1x236_;
        static_buffer<d2_t, 118> d2x118_;
        static_buffer<d4_t, 59> d4x59_;
        static_buffer<d236_t, 1> d236x1_;
    } data_;
    DEVICE_HOST constexpr vector_type() : data_{type{0}} {}
    DEVICE_HOST constexpr vector_type(type v) : data_{v} {}
    template<typename VEC> DEVICE_HOST constexpr const auto& to_varray() const { return data_.d236_; }
    template<typename VEC> DEVICE_HOST constexpr auto&       to_varray()       { return data_.d236_; }
    template<> DEVICE_HOST constexpr const auto& to_varray<d1_t>() const { return data_.d1x236_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d1_t>()       { return data_.d1x236_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d2_t>() const { return data_.d2x118_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d2_t>()       { return data_.d2x118_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d4_t>() const { return data_.d4x59_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d4_t>()       { return data_.d4x59_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d236_t>() const { return data_.d236x1_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d236_t>()       { return data_.d236x1_;}
};

template <typename T> struct vector_type<T, 240> {
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d4_t __attribute__((ext_vector_type(4)));
    typedef T d8_t __attribute__((ext_vector_type(8)));
    typedef T d12_t __attribute__((ext_vector_type(12)));
    typedef T d16_t __attribute__((ext_vector_type(16)));
    typedef T d20_t __attribute__((ext_vector_type(20)));
    typedef T d24_t __attribute__((ext_vector_type(24)));
    typedef T d40_t __attribute__((ext_vector_type(40)));
    typedef T d48_t __attribute__((ext_vector_type(48)));
    typedef T d60_t __attribute__((ext_vector_type(60)));
    typedef T d80_t __attribute__((ext_vector_type(80)));
    typedef T d120_t __attribute__((ext_vector_type(120)));
    typedef T d240_t __attribute__((ext_vector_type(240)));
    using type = d240_t;
    union {
        type d240_;
        static_buffer<d1_t, 240> d1x240_;
        static_buffer<d2_t, 120> d2x120_;
        static_buffer<d4_t, 60> d4x60_;
        static_buffer<d8_t, 30> d8x30_;
        static_buffer<d12_t, 20> d12x20_;
        static_buffer<d16_t, 15> d16x15_;
        static_buffer<d20_t, 12> d20x12_;
        static_buffer<d24_t, 10> d24x10_;
        static_buffer<d40_t, 6> d40x6_;
        static_buffer<d48_t, 5> d48x5_;
        static_buffer<d60_t, 4> d60x4_;
        static_buffer<d80_t, 3> d80x3_;
        static_buffer<d120_t, 2> d120x2_;
        static_buffer<d240_t, 1> d240x1_;
    } data_;
    DEVICE_HOST constexpr vector_type() : data_{type{0}} {}
    DEVICE_HOST constexpr vector_type(type v) : data_{v} {}
    template<typename VEC> DEVICE_HOST constexpr const auto& to_varray() const { return data_.d240_; }
    template<typename VEC> DEVICE_HOST constexpr auto&       to_varray()       { return data_.d240_; }
    template<> DEVICE_HOST constexpr const auto& to_varray<d1_t>() const { return data_.d1x240_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d1_t>()       { return data_.d1x240_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d2_t>() const { return data_.d2x120_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d2_t>()       { return data_.d2x120_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d4_t>() const { return data_.d4x60_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d4_t>()       { return data_.d4x60_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d8_t>() const { return data_.d8x30_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d8_t>()       { return data_.d8x30_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d12_t>() const { return data_.d12x20_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d12_t>()       { return data_.d12x20_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d16_t>() const { return data_.d16x15_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d16_t>()       { return data_.d16x15_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d20_t>() const { return data_.d20x12_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d20_t>()       { return data_.d20x12_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d24_t>() const { return data_.d24x10_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d24_t>()       { return data_.d24x10_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d40_t>() const { return data_.d40x6_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d40_t>()       { return data_.d40x6_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d48_t>() const { return data_.d48x5_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d48_t>()       { return data_.d48x5_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d60_t>() const { return data_.d60x4_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d60_t>()       { return data_.d60x4_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d80_t>() const { return data_.d80x3_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d80_t>()       { return data_.d80x3_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d120_t>() const { return data_.d120x2_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d120_t>()       { return data_.d120x2_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d240_t>() const { return data_.d240x1_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d240_t>()       { return data_.d240x1_;}
};

template <typename T> struct vector_type<T, 244> {
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d4_t __attribute__((ext_vector_type(4)));
    typedef T d244_t __attribute__((ext_vector_type(244)));
    using type = d244_t;
    union {
        type d244_;
        static_buffer<d1_t, 244> d1x244_;
        static_buffer<d2_t, 122> d2x122_;
        static_buffer<d4_t, 61> d4x61_;
        static_buffer<d244_t, 1> d244x1_;
    } data_;
    DEVICE_HOST constexpr vector_type() : data_{type{0}} {}
    DEVICE_HOST constexpr vector_type(type v) : data_{v} {}
    template<typename VEC> DEVICE_HOST constexpr const auto& to_varray() const { return data_.d244_; }
    template<typename VEC> DEVICE_HOST constexpr auto&       to_varray()       { return data_.d244_; }
    template<> DEVICE_HOST constexpr const auto& to_varray<d1_t>() const { return data_.d1x244_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d1_t>()       { return data_.d1x244_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d2_t>() const { return data_.d2x122_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d2_t>()       { return data_.d2x122_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d4_t>() const { return data_.d4x61_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d4_t>()       { return data_.d4x61_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d244_t>() const { return data_.d244x1_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d244_t>()       { return data_.d244x1_;}
};

template <typename T> struct vector_type<T, 248> {
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d4_t __attribute__((ext_vector_type(4)));
    typedef T d8_t __attribute__((ext_vector_type(8)));
    typedef T d124_t __attribute__((ext_vector_type(124)));
    typedef T d248_t __attribute__((ext_vector_type(248)));
    using type = d248_t;
    union {
        type d248_;
        static_buffer<d1_t, 248> d1x248_;
        static_buffer<d2_t, 124> d2x124_;
        static_buffer<d4_t, 62> d4x62_;
        static_buffer<d8_t, 31> d8x31_;
        static_buffer<d124_t, 2> d124x2_;
        static_buffer<d248_t, 1> d248x1_;
    } data_;
    DEVICE_HOST constexpr vector_type() : data_{type{0}} {}
    DEVICE_HOST constexpr vector_type(type v) : data_{v} {}
    template<typename VEC> DEVICE_HOST constexpr const auto& to_varray() const { return data_.d248_; }
    template<typename VEC> DEVICE_HOST constexpr auto&       to_varray()       { return data_.d248_; }
    template<> DEVICE_HOST constexpr const auto& to_varray<d1_t>() const { return data_.d1x248_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d1_t>()       { return data_.d1x248_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d2_t>() const { return data_.d2x124_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d2_t>()       { return data_.d2x124_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d4_t>() const { return data_.d4x62_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d4_t>()       { return data_.d4x62_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d8_t>() const { return data_.d8x31_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d8_t>()       { return data_.d8x31_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d124_t>() const { return data_.d124x2_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d124_t>()       { return data_.d124x2_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d248_t>() const { return data_.d248x1_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d248_t>()       { return data_.d248x1_;}
};

template <typename T> struct vector_type<T, 252> {
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d4_t __attribute__((ext_vector_type(4)));
    typedef T d12_t __attribute__((ext_vector_type(12)));
    typedef T d28_t __attribute__((ext_vector_type(28)));
    typedef T d36_t __attribute__((ext_vector_type(36)));
    typedef T d84_t __attribute__((ext_vector_type(84)));
    typedef T d252_t __attribute__((ext_vector_type(252)));
    using type = d252_t;
    union {
        type d252_;
        static_buffer<d1_t, 252> d1x252_;
        static_buffer<d2_t, 126> d2x126_;
        static_buffer<d4_t, 63> d4x63_;
        static_buffer<d12_t, 21> d12x21_;
        static_buffer<d28_t, 9> d28x9_;
        static_buffer<d36_t, 7> d36x7_;
        static_buffer<d84_t, 3> d84x3_;
        static_buffer<d252_t, 1> d252x1_;
    } data_;
    DEVICE_HOST constexpr vector_type() : data_{type{0}} {}
    DEVICE_HOST constexpr vector_type(type v) : data_{v} {}
    template<typename VEC> DEVICE_HOST constexpr const auto& to_varray() const { return data_.d252_; }
    template<typename VEC> DEVICE_HOST constexpr auto&       to_varray()       { return data_.d252_; }
    template<> DEVICE_HOST constexpr const auto& to_varray<d1_t>() const { return data_.d1x252_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d1_t>()       { return data_.d1x252_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d2_t>() const { return data_.d2x126_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d2_t>()       { return data_.d2x126_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d4_t>() const { return data_.d4x63_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d4_t>()       { return data_.d4x63_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d12_t>() const { return data_.d12x21_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d12_t>()       { return data_.d12x21_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d28_t>() const { return data_.d28x9_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d28_t>()       { return data_.d28x9_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d36_t>() const { return data_.d36x7_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d36_t>()       { return data_.d36x7_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d84_t>() const { return data_.d84x3_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d84_t>()       { return data_.d84x3_;}
    template<> DEVICE_HOST constexpr const auto& to_varray<d252_t>() const { return data_.d252x1_;}
    template<> DEVICE_HOST constexpr auto&       to_varray<d252_t>()       { return data_.d252x1_;}
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
