#pragma once
#include <cstdint>
#include <algorithm>  // For std::min_element
#include <vector>
#include <cassert>
#include <stdio.h>
#include <cmath>

union u4_pair {
    uint8_t value;

    struct {
        unsigned int lo : 4;  // Lower 4 bits (0-3) - might be first or last depending on endianness
        unsigned int hi : 4;  // Higher 4 bits (4-7)
    };
};

void print_vec_u4_pair(const std::vector<u4_pair>& vec) {
    printf("[");
    for (auto it = vec.begin(); it != vec.end(); ++it) {
        printf("%4d %4d", static_cast<int>((*it).lo), static_cast<int>((*it).hi));
        if (std::next(it) != vec.end()) printf(" ");
    }
    printf("] (%d)\n", static_cast<int>(vec.size() * 2));
}

template<class T>
constexpr const T& clamp(const T& v, const T& lo, const T& hi)
{
    return v < lo ? lo : (v > hi ? hi : v);
}

// quant a group of i8 to i4, the group size is equal to the vector size
// NOTE: the vec must with in range [-119, 119]
std::vector<u4_pair> lqq_quant(const std::vector<int8_t>& vec, uint8_t * p_scale, uint8_t * p_zero)
{
    // first find the min/max i8 of this group
    int8_t min_i8 = *std::min_element(vec.begin(), vec.end());
    int8_t max_i8 = *std::max_element(vec.begin(), vec.end());
    assert(min_i8 >= -119);
    assert(max_i8 <= 119);
    assert(vec.size() % 8 == 0); // only work on 2x vec

    // compute scale in u8
    uint8_t scale = [&](){
        float s = static_cast<float>(max_i8 - min_i8);
        s = s / static_cast<float>(15);
        s = std::round(s);  // !! round
        s = clamp(s, (float)1, (float)s);   // we must make sure s is not zero
        return static_cast<uint8_t>(s);
    }();

    *p_scale = scale;

    // compute zero in u8
    uint8_t zero = static_cast<uint8_t>(128 + min_i8);
    *p_zero = zero;

    // compute u4
    std::vector<u4_pair> vec_u4_pair = [&](){
        std::vector<u4_pair> result;
        result.reserve(vec.size() / 2);
        for(size_t i = 0; i < vec.size() / 2; i++) {
            // do shift
            uint8_t shifted_i8_0 = __builtin_bit_cast(uint8_t, static_cast<int8_t>(vec[2 * i + 0] - min_i8));
            uint8_t shifted_i8_1 = __builtin_bit_cast(uint8_t, static_cast<int8_t>(vec[2 * i + 1] - min_i8));

            float lo_f = static_cast<float>(shifted_i8_0) / static_cast<float>(scale);
            float hi_f = static_cast<float>(shifted_i8_1) / static_cast<float>(scale);

            // !! round
            lo_f = std::round(lo_f);
            hi_f = std::round(hi_f);

            // !! clamp
            lo_f = clamp(lo_f, (float)0, (float)15);
            hi_f = clamp(hi_f, (float)0, (float)15);

            uint8_t lo = static_cast<uint8_t>(lo_f);
            uint8_t hi = static_cast<uint8_t>(hi_f);

            u4_pair p;
            p.lo = lo;
            p.hi = hi;
            // printf("s0:%d, s1:%d, lo:%d, hi:%d (min:%d, max:%d)\n", (int)shifted_i8_0, (int)shifted_i8_1, (int)lo, (int)hi, (int)min_i8, (int)max_i8);

            result.push_back(p);
        }
        return result;
    }();

    return vec_u4_pair;
}

union u8u4_pair {
    struct {
        uint8_t u8_0;
        uint8_t u8_1;
        uint8_t u8_2;
        uint8_t u8_3;
    };
    struct {
        uint8_t u4_0 : 4;
        uint8_t __p0 : 4;
        uint8_t u4_1 : 4;
        uint8_t __p1 : 4;
        uint8_t u4_2 : 4;
        uint8_t __p2 : 4;
        uint8_t u4_3 : 4;
        uint8_t __p3 : 4;
    };
    uint32_t v;
};

u8u4_pair to_u8u4_pair(const u4_pair& x, const u4_pair& y) {
    u8u4_pair r;
    r.u4_0 = x.lo;
    r.u4_1 = x.hi;
    r.u4_2 = y.lo;
    r.u4_3 = y.hi;
    return r;
}

uint32_t mock_zero(uint8_t zero) {
    u8u4_pair r;
    r.u8_0 = zero; r.u8_1 = zero; r.u8_2 = zero; r.u8_3 = zero;
    return r.v;
}

std::vector<int8_t> lqq_dequant(const std::vector<u4_pair>& vec, uint8_t scale, uint8_t zero)
{
    std::vector<int8_t> result;
    result.reserve(vec.size() * 2);
    uint32_t mocked_zero = mock_zero(zero);
    for(size_t i = 0; i < vec.size() / 2; i++) {
        u8u4_pair tmp = to_u8u4_pair(vec[2*i+0], vec[2*i+1]);
        u8u4_pair rmp;
        rmp.v = (tmp.v * scale + mocked_zero) ^ 0x80808080;

        result.push_back(__builtin_bit_cast(int8_t, rmp.u8_0));
        result.push_back(__builtin_bit_cast(int8_t, rmp.u8_1));
        result.push_back(__builtin_bit_cast(int8_t, rmp.u8_2));
        result.push_back(__builtin_bit_cast(int8_t, rmp.u8_3));
    }
    return result;
}
