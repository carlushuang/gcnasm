#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <string>
#include <vector>

uint32_t cvt_f32_to_sortable_u32(float x)
{
    uint32_t z = __builtin_bit_cast(uint32_t, x);
    uint32_t f = x < 0 ? 0xffffffff : 0x80000000;
    return z ^ f;
}

auto cvt_f32_to_sortable_u32_vec(const std::vector<float> & x)
{
    std::vector<uint32_t> y;
    for(int i = 0 ; i < x.size(); i++) {
        y.push_back(cvt_f32_to_sortable_u32(x[i]));
    }
    return y;
}

auto gen_rand_f32_vec(int size)
{
    static std::random_device rd{};
    std::mt19937 gen{rd()};
    std::normal_distribution d{0., 1.};

    std::vector<float> v;
    for(int i = 0; i < size; i++) {
        v.push_back(d(gen));
    }
    return v;
}

// return vector<int>
template<typename dtype>
auto cmp_gt_vec(const std::vector<dtype> & x, const std::vector<dtype> & y)
{
    // assert(x.size() == y.size());
    std::vector<int> result;
    for(int i = 0; i < x.size(); i++) {
        result.push_back(x[i] > y[i] ? 1 : 0);
    }
    return result;
}
template<typename dtype>
void dump_vec(const std::vector<dtype> & x) {
    std::cout << "[";
    for(int i = 0; i < x.size(); i++) {
        std::cout << x[i];
        if(i != static_cast<int>(x.size() - 1))
            std::cout << ", ";
    }
    std::cout << "]" << std::endl;
}

template<typename dtype>
auto cmp_eq(const std::vector<dtype> & x, const std::vector<dtype> & y)
{
    bool result = true;
    for(int i = 0; i < x.size(); i++) {
        result &= x[i] == y[i];
    }
    return result;
}

int main(int argc, char ** argv)
{
    int num = 5;
    if(argc > 1) {
        num = atoi(argv[1]);
    }
    auto a_f32 = gen_rand_f32_vec(num);
    auto b_f32 = gen_rand_f32_vec(num);

    auto a_u32 = cvt_f32_to_sortable_u32_vec(a_f32);
    auto b_u32 = cvt_f32_to_sortable_u32_vec(b_f32);

    auto r_f32 = cmp_gt_vec(a_f32, b_f32);
    auto r_u32 = cmp_gt_vec(a_u32, b_u32);

    dump_vec(a_f32);
    dump_vec(b_f32);

    dump_vec(a_u32);
    dump_vec(b_u32);

    dump_vec(r_f32);
    dump_vec(r_u32);
    auto r = cmp_eq(r_f32, r_u32);
    printf("result:%s\n", r ? "y" : "n");
}