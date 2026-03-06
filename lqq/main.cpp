#include "lqq.hpp"

#include <stdio.h>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cstddef>

#if 0
std::vector<int8_t> gen_random_vec_i8(size_t size, int8_t min = -128, int8_t max = 127) {
    std::vector<int8_t> result;
    result.reserve(size);

    // Better random seed using random_device
    std::random_device rd;
    std::mt19937 generator(rd()); // Mersenne Twister engine

    std::uniform_int_distribution<int16_t> distribution(min, max);

    for (size_t i = 0; i < size; ++i) {
        result.push_back(static_cast<int8_t>(distribution(generator)));
    }

    return result;
}
#endif

// Generate with custom range
std::vector<int8_t> gen_random_vec_i8(size_t size, int min = -128, int max = 127) {
    std::srand(static_cast<unsigned int>(std::time(NULL)));

    std::vector<int8_t> result;
    result.reserve(size);
    
    int range = max - min + 1;
    
    for (size_t i = 0; i < size; ++i) {
        int random_val = min + (rand() % range);
        result.push_back(static_cast<int8_t>(random_val));
    }
    
    return result;
}

void print_vec_i8(const std::vector<int8_t>& vec) {
    printf("[");
    for (auto it = vec.begin(); it != vec.end(); ++it) {
        printf("%4d", static_cast<int>(*it));
        if (std::next(it) != vec.end()) printf(" ");
    }
    printf("] (%d)\n", static_cast<int>(vec.size()));
}

int main(int argc, char ** argv) {
    
    int vec_size = 16;
    if (argc > 1) {
        vec_size = atoi(argv[1]);
    }
    auto v = gen_random_vec_i8(vec_size, -50, 50);
    printf("================================ origin:\n");
    print_vec_i8(v);

    uint8_t u4_scale, u4_zero;
    auto u4_pair = lqq_quant(v, &u4_scale, &u4_zero);

    printf("================================ quant:\n");
    print_vec_u4_pair(u4_pair);
    printf("scale:%d, zero:%d\n" , static_cast<int>(u4_scale) , static_cast<int>(u4_zero));

    printf("================================ dequant(re-construction):\n");
    auto y = lqq_dequant(u4_pair, u4_scale, u4_zero);
    print_vec_i8(y);
}
