#include <stdio.h>
#include <hip/hip_runtime.h>
#include <random>
#include <iostream>
#include "wave_reduce_dpp.hpp"

#define HIP_CALL(call)                                                 \
    do                                                                 \
    {                                                                  \
        hipError_t err = call;                                         \
        if(err != hipSuccess)                                          \
        {                                                              \
            printf("[hiperror](%d) fail to call %s", (int)err, #call); \
            exit(0);                                                   \
        }                                                              \
    } while(0)

#define PER_PIXEL_CHECK
#define ASSERT_ON_FAIL

#ifndef ABS
#define ABS(x) ((x) > 0 ? (x) : -1 * (x))
#endif

template <typename T>
void rand_vec(T* seq, size_t len)
{
    static std::random_device rd; // seed
    static std::mt19937 mt(rd());
    static std::uniform_real_distribution<T> dist(-10.0, 10.0);

    for(size_t i = 0; i < len; i++)
        seq[i] = dist(mt);
}

template <typename data_t>
static void reduce_sum_host(data_t* out, const data_t* in, int num)
{
    data_t sum = static_cast<data_t>(0);
    for(auto i = 0; i < num; i++)
    {
        sum += in[i];
    }
    out[0] = sum;
}

template <typename data_t>
static void reduce_cumsum_host(data_t* out, const data_t* in, int num)
{
    out[0] = in[0];
    for(auto i = 1; i < num; i++)
    {
        out[i] = out[i-1] + in[i];
    }
}


void test_wave_reduce_sum()
{
    printf("[sum dpp]\n");
    int total_floats = 64;
    int gdx          = 1;
    int bdx          = 64;

    float* dev_in;
    float* dev_out;
    float* host_in      = new float[total_floats];
    float* host_out     = new float[1];
    float* host_out_dev = new float[1];

    HIP_CALL(hipMalloc(&dev_in, sizeof(float) * total_floats));
    HIP_CALL(hipMalloc(&dev_out, sizeof(float) * 1));

    rand_vec(host_in, total_floats);
    HIP_CALL(hipMemcpy(dev_in, host_in, sizeof(float) * total_floats, hipMemcpyHostToDevice));

    wave_reduce_kernel<<<gdx, bdx>>>(dev_in, dev_out);

    reduce_sum_host<float>(host_out, host_in, 64);

    HIP_CALL(hipMemcpy(host_out_dev, dev_out, sizeof(float) * 1, hipMemcpyDeviceToHost));
    printf("dev:%f, host:%f\n", host_out_dev[0], host_out[0]);

    delete[] host_in;
    delete[] host_out;
    delete[] host_out_dev;
    hipFree(dev_in);
    hipFree(dev_out);
}

void test_wave_reduce_cumsum()
{
    printf("[cumsum ]\n");
    int total_floats = 64;
    int gdx          = 1;
    int bdx          = 64;

    float* dev_in;
    float* dev_out;
    float* host_in      = new float[total_floats];
    float* host_out     = new float[total_floats];
    float* host_out_dev = new float[total_floats];

    HIP_CALL(hipMalloc(&dev_in, sizeof(float) * total_floats));
    HIP_CALL(hipMalloc(&dev_out, sizeof(float) * total_floats));

    rand_vec(host_in, total_floats);
    HIP_CALL(hipMemcpy(dev_in, host_in, sizeof(float) * total_floats, hipMemcpyHostToDevice));

    wave_reduce_cumsum_kernel<<<gdx, bdx>>>(dev_in, dev_out);

    reduce_cumsum_host<float>(host_out, host_in, 64);

    HIP_CALL(hipMemcpy(host_out_dev, dev_out, sizeof(float) * total_floats, hipMemcpyDeviceToHost));
    printf("i:");
    for(auto i = 0; i < total_floats; i++) {
        printf("%.3f", host_in[i]);
        if(i != total_floats - 1) printf(", ");
    }
    printf("\n");
    printf("d:");
    for(auto i = 0; i < total_floats; i++) {
        printf("%.3f", host_out_dev[i]);
        if(i != total_floats - 1) printf(", ");
    }
    printf("\n");
    printf("h:");
    for(auto i = 0; i < total_floats; i++) {
        printf("%.3f", host_out[i]);
        if(i != total_floats - 1) printf(", ");
    }
    printf("\n");

    delete[] host_in;
    delete[] host_out;
    delete[] host_out_dev;
    hipFree(dev_in);
    hipFree(dev_out);
}


int main(int argc, char** argv)
{
    test_wave_reduce_sum();
    test_wave_reduce_cumsum();
}