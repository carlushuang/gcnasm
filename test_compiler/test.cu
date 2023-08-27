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

template<typename T, index_t N>
struct Array {
    T data[N] = {};

    DEVICE_HOST static constexpr auto size() { return N; }
    DEVICE_HOST constexpr auto & operator[](index_t i) { return data[i]; }
    DEVICE_HOST constexpr const auto & operator[](index_t i) const { return data[i]; }

    template<index_t I>
    DEVICE_HOST constexpr auto & at() { return data[I]; }
    
    template<index_t I>
    DEVICE_HOST constexpr const auto & at() const { return data[I]; }
    
    DEVICE_HOST void dump() const
    {
        for(auto i = 0; i < N; i++)
        {
            printf("%d, ", data[i]);
        }
        printf("\n");
    }
};

template<index_t N>
using IndexArray = Array<index_t, N>;

template<index_t... Xs>
using seq = std::integer_sequence<index_t, Xs...>;

template<index_t N>
using make_seq = std::make_integer_sequence<index_t, N>;

template<index_t... t>
constexpr void print_seq(seq<t...>)
{
    ((std::cout << t << ' '), ...);
}

struct Unmerge {
    index_t L0 {};
    index_t L1 {};
    template<typename UpIdx>
    DEVICE_HOST constexpr auto calculate_lower_idx(const UpIdx & upper_idx) const
    {
        return IndexArray<1>{upper_idx.template at<0>() * L1 + upper_idx.template at<1>()};
    }

    template<typename UpIdx, typename UpDiff>
    DEVICE_HOST constexpr auto update_lower_idx(
        IndexArray<1> &lower_idx,
        IndexArray<1> &lower_diff,
        const UpIdx & upper_idx, const UpDiff & upper_diff) const
    {
        lower_diff = calculate_lower_idx(upper_diff);
        lower_idx.template at<0>() = lower_idx.template at<0>() + lower_diff.template at<0>();
    }
};

// template<index_t pad>
struct Pad {
    index_t pad {};
    template<typename UpIdx>
    DEVICE_HOST constexpr auto calculate_lower_idx(const UpIdx & upper_idx) const
    {
        return IndexArray<1>{upper_idx.template at<0>() - pad};
    }

    template<typename UpIdx, typename UpDiff>
    DEVICE_HOST constexpr auto update_lower_idx(
        IndexArray<1> &lower_idx,
        IndexArray<1> &lower_diff,
        const UpIdx & upper_idx,
        const UpDiff & upper_diff) const
    {
        lower_diff = upper_diff;
        lower_idx.template at<0>() = lower_idx.template at<0>() + lower_diff.template at<0>();
    }
};

template<typename unmerge_t, typename pad_0_t, typename pad_1_t>
struct tensor_cooord {
    // hardcoded transform
    unmerge_t unmerge;
    pad_0_t  pad_0;
    pad_1_t  pad_1;

    DEVICE_HOST
    constexpr tensor_cooord(const IndexArray<2>& upper_coord, const unmerge_t & unmerge_, const pad_0_t & pad_0_, const pad_1_t & pad_1_) :
                            unmerge(unmerge_), pad_0(pad_0_), pad_1(pad_1_)
    {
        // set visible ids:
        coord.template at<3>() = upper_coord.template at<0>();
        coord.template at<4>() = upper_coord.template at<1>();

        // pad_0
        const auto lower_pad_0 = pad_0.calculate_lower_idx(IndexArray<1>{coord.template at<3>()});
        coord.template at<1>() = lower_pad_0.template at<0>();

        // pad_1
        const auto lower_pad_1 = pad_1.calculate_lower_idx(IndexArray<1>{coord.template at<4>()});
        coord.template at<2>() = lower_pad_1.template at<0>();

        // unmerge
        const auto lower_unmerge = unmerge.calculate_lower_idx(IndexArray<2>{
                                                    coord.template at<1>(),
                                                    coord.template at<2>()});
        coord.template at<0>() = lower_unmerge.template at<0>();
    }

    DEVICE_HOST
    constexpr auto move(const IndexArray<2> & step)
    {
        auto upper_idx_pad_0 = coord.template at<3>();
        auto upper_idx_pad_1 = coord.template at<4>();
        auto upper_diff_pad_0 = IndexArray<1>{step.template at<0>()};
        auto upper_diff_pad_1 = IndexArray<1>{step.template at<1>()};

        // update visibla idx
        coord.template at<3>() = coord.template at<3>() + step.template at<0>();
        coord.template at<4>() = coord.template at<4>() + step.template at<1>();

        IndexArray<1> lower_idx_pad_0 = IndexArray<1>{coord.template at<1>()};
        IndexArray<1> lower_idx_pad_1 = IndexArray<1>{coord.template at<2>()};

        IndexArray<1> lower_diff_pad_0;
        IndexArray<1> lower_diff_pad_1;

        pad_1.update_lower_idx(lower_idx_pad_1, lower_diff_pad_1, upper_idx_pad_1, upper_diff_pad_1);
        pad_0.update_lower_idx(lower_idx_pad_0, lower_diff_pad_0, upper_idx_pad_0, upper_diff_pad_0);

        // update hidden idx
        coord.template at<1>() = lower_idx_pad_0.template at<0>();
        coord.template at<2>() = lower_idx_pad_1.template at<0>();

        auto upper_idx_unmerge = IndexArray<2>{coord.template at<1>(), coord.template at<2>()};
        auto upper_diff_unmerge = IndexArray<2>{lower_diff_pad_0.template at<0>(), lower_diff_pad_1.template at<0>()};

        IndexArray<1> lower_idx_unmerge = IndexArray<1>{coord.template at<0>()};
        IndexArray<1> lower_diff_unmerge;
        unmerge.update_lower_idx(lower_idx_unmerge, lower_diff_unmerge, upper_idx_unmerge, upper_diff_unmerge);

        // update hidden idx
        coord.template at<0>() = lower_idx_unmerge.template at<0>();
    }

    DEVICE_HOST
    constexpr auto get_offset() const
    {
        return coord.template at<0>();
    }
    IndexArray<5> coord {};
};

template<typename unmerge_t, typename pad_0_t, typename pad_1_t>
DEVICE_HOST constexpr auto make_tensor_coord(const IndexArray<2>& upper_coord, const unmerge_t & unmerge, const pad_0_t & pad_0, const pad_1_t & pad_1){
    return tensor_cooord<unmerge_t, pad_0_t, pad_1_t>{upper_coord, unmerge, pad_0, pad_1};
}

template<index_t BLOCK_SIZE, index_t L0_BLOCK, index_t L1_BLOCK, index_t MOVE_L1_PER_ITER = 8>
GLOBAL void test_kernel(index_t * __restrict__ offset_0, index_t * __restrict__ offset_1, index_t p0, index_t p1, index_t l0, index_t l1, index_t iters)
{
    //if(blockIdx.x == 0) {
        index_t tid = (index_t) threadIdx.x;
        index_t coord_1 = tid % L1_BLOCK;
        index_t coord_0 = tid / L1_BLOCK;
        const auto upper_coord = IndexArray<2>{coord_0, coord_1};
        auto coord = make_tensor_coord( upper_coord, Unmerge{l0, l1}, Pad{p0}, Pad{p1});

        offset_0[tid] = coord.get_offset();

        IndexArray<2> step {0, MOVE_L1_PER_ITER};

        // we want each loop be independently. clang by default will choose to unrool it if possible, 
        // result in more register usage, and longer code
        #pragma clang loop vectorize(disable)
        // #pragma clang loop interleave(disable)
        for(auto i = 0; i < iters; i++)
        {
            coord.move(step);
            offset_1[i * BLOCK_SIZE + tid] = coord.get_offset();
        }
        
    //}
}

int main(int argc, char ** argv) {
    index_t p0 = 1;
    index_t p1 = 2;
    index_t l0 = 320;
    index_t l1 = 240;
    index_t iters = 1;

    if(argc >= 6) {
        p0 = atoi(argv[1]);
        p1 = atoi(argv[2]);
        l0 = atoi(argv[3]);
        l1 = atoi(argv[4]);
        iters = atoi(argv[5]);
    }

    constexpr index_t BLOCK_SIZE = 256;
    constexpr index_t L0_BLOCK = 32;
    constexpr index_t L1_BLOCK = 8;
    constexpr index_t MOVE_L1_PER_ITER = L1_BLOCK;

    index_t * offset_0_host;
    index_t * offset_0_device;
    index_t * offset_1_host;
    index_t * offset_1_device;

    cudaMalloc(&offset_0_device, sizeof(index_t) * BLOCK_SIZE);
    cudaMalloc(&offset_1_device, sizeof(index_t) * BLOCK_SIZE * iters);
    offset_0_host = (index_t*)malloc(sizeof(index_t) * BLOCK_SIZE);
    offset_1_host = (index_t*)malloc(sizeof(index_t) * BLOCK_SIZE * iters);

    test_kernel<BLOCK_SIZE, L0_BLOCK, L1_BLOCK, MOVE_L1_PER_ITER><<<dim3(1), dim3(BLOCK_SIZE)>>>(offset_0_device, offset_1_device, p0, p1, l0, l1, iters);

    cudaMemcpy(offset_0_host, offset_0_device, sizeof(index_t) * BLOCK_SIZE, cudaMemcpyDeviceToHost);
    cudaMemcpy(offset_1_host, offset_1_device, sizeof(index_t) * BLOCK_SIZE * iters, cudaMemcpyDeviceToHost);

    for(auto i = 0; i < BLOCK_SIZE ; i++){
        auto os = offset_0_host[i];
        auto moved_os = offset_1_host[i + ((iters - 1) * BLOCK_SIZE)];
        printf("[%3d] len:{%d, %d}, pad:{%d, %d}, os:%d, moved_os:%d, diff:%d\n",
        i, l0, l1, p0, p1, os, moved_os, moved_os - os);
    }
    cudaFree(offset_0_device);
    free(offset_0_host);
    cudaFree(offset_1_device);
    free(offset_1_host);
}
