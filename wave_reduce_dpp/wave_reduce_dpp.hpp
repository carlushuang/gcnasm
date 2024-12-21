#pragma once
//  kernel instance is here

template <typename data_t, int wave_size, typename reduce_op_t>
__device__ inline data_t wave_reduce(const data_t& thread_data, reduce_op_t reduce_op)
{
    // wave_size must be power of 2
    constexpr int row_mask    = 0xf;
    constexpr int bank_mask   = 0xf;
    constexpr bool bound_ctrl = false;

    data_t result = thread_data;

    if constexpr(wave_size > 1)
    {
        result = reduce_op(
            result,
            __builtin_bit_cast(data_t, __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, result),
                                                           0xb1,
                                                           row_mask,
                                                           bank_mask,
                                                           bound_ctrl))); // quad_perm:[1,0,3,2]
    }
    if constexpr(wave_size > 2)
    {
        result = reduce_op(
            result,
            __builtin_bit_cast(data_t, __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, result),
                                                           0x4e,
                                                           row_mask,
                                                           bank_mask,
                                                           bound_ctrl))); // quad_perm:[2,3,0,1]
    }
    if constexpr(wave_size > 4)
    {
        result =
            reduce_op(result,
                      __builtin_bit_cast(data_t, __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, result),
                                                                     0x114,
                                                                     row_mask,
                                                                     bank_mask,
                                                                     bound_ctrl))); // row_shr:4
    }
    if constexpr(wave_size > 8)
    {
        result =
            reduce_op(result,
                      __builtin_bit_cast(data_t, __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, result),
                                                                     0x118,
                                                                     row_mask,
                                                                     bank_mask,
                                                                     bound_ctrl))); // row_shr:8
    }
    if constexpr(wave_size > 16)
    {
        result =
            reduce_op(result,
                      __builtin_bit_cast(data_t, __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, result),
                                                                     0x142,
                                                                     row_mask,
                                                                     bank_mask,
                                                                     bound_ctrl))); // row_bcast:15
    }
    if constexpr(wave_size > 32)
    {
        result =
            reduce_op(result,
                      __builtin_bit_cast(data_t, __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, result),
                                                                     0x143,
                                                                     row_mask,
                                                                     bank_mask,
                                                                     bound_ctrl))); // row_bcast:31
    }

    // now the reduced value is in the last lane of wave
    return __builtin_bit_cast(data_t, 
        __builtin_amdgcn_readlane(__builtin_bit_cast(int, result), wave_size - 1));
}


template <typename data_t, int wave_size>
__device__ inline void wave_cumsum(data_t& thread_data)
{
    // wave_size must be power of 2
    constexpr int row_mask    = 0xf;
    constexpr int bank_mask   = 0xf;
    constexpr bool bound_ctrl = true;   // ! out-of-bound is zero !
    auto reduce_op = [&](auto x_, auto y_) { return x_ + y_; };

    if constexpr(wave_size > 1)
    {
        thread_data = reduce_op(
            thread_data,
            __builtin_bit_cast(data_t, __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, thread_data),
                                                           0x111,
                                                           row_mask,
                                                           bank_mask,
                                                           bound_ctrl))); // row_shr:1
    }

    if constexpr(wave_size > 2)
    {
        thread_data = reduce_op(
            thread_data,
            __builtin_bit_cast(data_t, __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, thread_data),
                                                           0x112,
                                                           row_mask,
                                                           bank_mask,
                                                           bound_ctrl))); // row_shr:2
    }
    if constexpr(wave_size > 4)
    {
        thread_data =
            reduce_op(thread_data,
                      __builtin_bit_cast(data_t, __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, thread_data),
                                                                     0x114,
                                                                     row_mask,
                                                                     bank_mask,
                                                                     bound_ctrl))); // row_shr:4
    }
    if constexpr(wave_size > 8)
    {
        thread_data =
            reduce_op(thread_data,
                      __builtin_bit_cast(data_t, __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, thread_data),
                                                                     0x118,
                                                                     row_mask,
                                                                     bank_mask,
                                                                     bound_ctrl))); // row_shr:8
    }

    if constexpr(wave_size > 16)
    {
        // now row-0, row-0+row-1, row-1+row-2, row-2+row-3
        int v_remote_tmp = __builtin_amdgcn_ds_bpermute(((__lane_id() & 0x30) - 1) << 2, __builtin_bit_cast(int, thread_data));
        v_remote_tmp = __lane_id() >= 16 ? v_remote_tmp : 0;
        thread_data = reduce_op(thread_data, __builtin_bit_cast(data_t, v_remote_tmp));
    }

    if constexpr(wave_size > 32)
    {
        // lane-id 48...63->31
        int v_remote_tmp = __builtin_amdgcn_ds_bpermute(((__lane_id() & 0x30) - 17) << 2, __builtin_bit_cast(int, thread_data));
        v_remote_tmp = __lane_id() >= 32 ? v_remote_tmp : 0;
        thread_data = reduce_op(thread_data, __builtin_bit_cast(data_t, v_remote_tmp));
    }
}

__global__ void wave_reduce_kernel(float* input, float* output)
{
    float v      = input[threadIdx.x];
    auto f_sum = [&](float x_, float y_) {return x_ + y_;};
    float result = wave_reduce<float, 64>(v, f_sum);
    if(threadIdx.x == 0)
        *output = result;
}

__global__ void wave_reduce_cumsum_kernel(float* input, float* output)
{
    float v      = input[threadIdx.x];
    wave_cumsum<float, 64>(v);
    output[threadIdx.x] = v;
}
