#pragma once

#include "codegen_settings.h"
#include "mem_stream_base.h"


template<int BLOCK_SIZE = 256, int GRID_SIZE = 80, int UNROLL = 8>
struct memcpy_persistent
    : public mem_stream_base<memcpy_persistent<BLOCK_SIZE, GRID_SIZE, UNROLL>, BLOCK_SIZE, GRID_SIZE>  {

    using mem_base_type = mem_stream_base<memcpy_persistent<BLOCK_SIZE, GRID_SIZE, UNROLL>, BLOCK_SIZE, GRID_SIZE>;
    static constexpr int bytes_per_issue = 16;  // dwordx4

    __host__ memcpy_persistent(typename mem_base_type::karg harg_)
        : mem_base_type(harg_, bytes_per_issue, UNROLL) {}

    struct kernel {
        __device__ void operator()(typename mem_base_type::karg args){

            int total = args.bytes / bytes_per_issue;
            int index = blockIdx.x * BLOCK_SIZE + threadIdx.x;
            int32x4_t src_r = make_buffer_resource(args.src, args.bytes);
            int32x4_t dst_r = make_buffer_resource(args.dst, args.bytes);
        
            fp32x4_t d[UNROLL];

            while(index < total) {
                for(auto u = 0; u < UNROLL; u++) {
                    buffer_load_dwordx4_raw(d[u], src_r, (index + u * gridDim.x) * sizeof(int32x4_t), 0, 0);
                }

                for_(
                    [&] (auto uu) {
                        // std::get<i.value>(t); // do stuff
                        constexpr auto u = uu.value;
                        buffer_fence(UNROLL - 1 - u);
                        buffer_store_dwordx4_raw(d[u], dst_r, (index + u * gridDim.x) * sizeof(int32x4_t), 0, 0);
                    },
                    std::make_index_sequence<UNROLL>{});

                index += UNROLL * gridDim.x;
            }
        }
    };

};


template<int BLOCK_SIZE = 256, int GRID_SIZE = 80, int UNROLL = 8>
struct memread_stream 
    : public mem_stream_base< memread_stream<BLOCK_SIZE, GRID_SIZE, UNROLL>, BLOCK_SIZE, GRID_SIZE>  {

    using mem_base_type = mem_stream_base<memread_stream<BLOCK_SIZE, GRID_SIZE, UNROLL>, BLOCK_SIZE, GRID_SIZE>;
    static constexpr int bytes_per_issue = 16;  // dwordx4

    __host__ memread_stream(typename mem_base_type::karg harg_)
        : mem_base_type(harg_, bytes_per_issue, UNROLL) {}

    struct kernel {
        __device__ void operator()(typename mem_base_type::karg args){
	    fp32x4_t * p_src = reinterpret_cast<fp32x4_t*>(reinterpret_cast<uint8_t*>(args.src));
            fp32x4_t * p_dst = reinterpret_cast<fp32x4_t*>(reinterpret_cast<uint8_t*>(args.dst));
#if READ_VERIFY == 0
	    fp32x4_t v{};
#endif

        auto current = blockIdx.x * args.issue_per_block; 
	    for(auto i = 0; i < args.iters; i++) {
		    auto offs = UNROLL * BLOCK_SIZE * i + threadIdx.x; 

            	#pragma unroll
		for (auto j = 0; j < UNROLL; j++) {
                    
#if READ_VERIFY == 0
#if READ_STREAM_NONTEMP == 1
		    v += nt_load(p_src[current + offs]);
#else
		    v += p_src[current + offs];
#endif
#else
                    p_dst[current + offs] = p_src[current + offs];
#endif
		    offs += BLOCK_SIZE;
		}
            }
#if READ_VERIFY == 0
	    if (v[0] == 10000 /*&& v[1] == 10000 && v[2] == 10000 && v[3] == 10000*/) {
                *p_dst  = v;
            }
#endif
        }
    };

};

template<int BLOCK_SIZE = 256, int GRID_SIZE = 80, int UNROLL = 8>
struct memcpy_stream 
    : public mem_stream_base< memcpy_stream<BLOCK_SIZE, GRID_SIZE, UNROLL>, BLOCK_SIZE, GRID_SIZE>  {

    using mem_base_type = mem_stream_base<memcpy_stream<BLOCK_SIZE, GRID_SIZE, UNROLL>, BLOCK_SIZE, GRID_SIZE>;
    static constexpr int bytes_per_issue = 16;  // dwordx4

    __host__ memcpy_stream(typename mem_base_type::karg harg_)
        : mem_base_type(harg_, bytes_per_issue, UNROLL) {}

    struct kernel {
        __device__ void operator()(typename mem_base_type::karg args){ 
            fp32x4_t * p_src = reinterpret_cast<fp32x4_t*>(reinterpret_cast<uint8_t*>(args.src));
            fp32x4_t * p_dst = reinterpret_cast<fp32x4_t*>(reinterpret_cast<uint8_t*>(args.dst));

            auto current = blockIdx.x * args.issue_per_block; 
 	        for(auto i = 0; i < args.iters; i++) {
		        auto offs = UNROLL * BLOCK_SIZE * i + threadIdx.x;            
            	#pragma unroll
                for (auto j = 0; j < UNROLL; j++) {
#if MEMCPY_STREAM_NONTEMP == 1
                    auto d = nt_load(p_src[current + offs]);
                    nt_store(d, p_dst[current + offs]);
#else
                    p_dst[current + offs] = p_src[current + offs];
#endif
                    offs += BLOCK_SIZE;
                }
            }
        }
    };
};


template<int BLOCK_SIZE = 256, int GRID_SIZE = 80, int UNROLL = 8>
struct memcpy_stream_async
    : public mem_stream_base< memcpy_stream_async<BLOCK_SIZE, GRID_SIZE, UNROLL>, BLOCK_SIZE, GRID_SIZE>  {

    using mem_base_type = mem_stream_base<memcpy_stream_async<BLOCK_SIZE, GRID_SIZE, UNROLL>, BLOCK_SIZE, GRID_SIZE>;
    static constexpr int bytes_per_issue = 16;  // dwordx4
    using vector_type = bytes_to_vector_t<bytes_per_issue>;

    __host__ memcpy_stream_async(typename mem_base_type::karg harg_)
        : mem_base_type(harg_, bytes_per_issue, UNROLL) {}

    struct kernel {
        __device__ void operator()(typename mem_base_type::karg args){ 

            int64_t base_offset = blockIdx.x * BLOCK_SIZE * bytes_per_issue;

            // int idx = (blockIdx.x * BLOCK_SIZE + threadIdx.x);
            int wave_id = __builtin_amdgcn_readfirstlane(threadIdx.x / 64);
            int32_t total = static_cast<int32_t>((args.bytes - base_offset) / bytes_per_issue);

            vector_type * p_dst = reinterpret_cast<vector_type*>(reinterpret_cast<uint8_t*>(args.dst) + base_offset);
            int32x4_t src_r = make_buffer_resource(reinterpret_cast<uint8_t*>(args.src) + base_offset, args.bytes - base_offset);

            __shared__ char smem[BLOCK_SIZE * sizeof(vector_type)];

            m0_set_with_memory(64 * sizeof(float) * wave_id);

            constexpr int loops_per_vector = sizeof(vector_type)/sizeof(float);
 	        for(auto i = 0; i < args.iters; i++) {
            	#pragma unroll
                for(auto c = 0; c < UNROLL; c++) {
                    for_([&](auto k) {
                        async_buffer_load_dword_v(smem, src_r,
                                (i * UNROLL + c) * gridDim.x * BLOCK_SIZE * sizeof(vector_type) + threadIdx.x * sizeof(float),
                                0, k.value * sizeof(float) * BLOCK_SIZE);
                        // m0_inc_with_memory(BLOCK_SIZE * sizeof(float));
                        },
                        std::make_index_sequence<loops_per_vector>{}
                    );

                    buffer_fence(0);
                    __builtin_amdgcn_s_barrier();

                    auto d = reinterpret_cast<vector_type*>(smem)[threadIdx.x];

                    auto current = threadIdx.x + (i * UNROLL + c) * gridDim.x * BLOCK_SIZE;
                    if(current < total)
                        p_dst[current] = d;
                    __builtin_amdgcn_s_barrier();
                }
            }
        }
    };
};

template<int BLOCK_SIZE = 256, int GRID_SIZE = 80, int CHUNKS = 8, int INNER = 1>
struct memcpy_stream_swizzled
    : public mem_stream_base< memcpy_stream_swizzled<BLOCK_SIZE, GRID_SIZE, CHUNKS, INNER>, BLOCK_SIZE, GRID_SIZE>  {

    using mem_base_type = mem_stream_base<memcpy_stream_swizzled<BLOCK_SIZE, GRID_SIZE, CHUNKS, INNER>, BLOCK_SIZE, GRID_SIZE>;
    static constexpr int bytes_per_issue = 16;  // dwordx4
    using vector_type = bytes_to_vector_t<bytes_per_issue>;

    __host__ memcpy_stream_swizzled(typename mem_base_type::karg harg_)
        : mem_base_type(harg_, bytes_per_issue, CHUNKS, INNER) {}

    struct kernel {
        __device__ void operator()(typename mem_base_type::karg args){ 
            // simulate mfma 32x32x8(16)
            constexpr int total_waves = BLOCK_SIZE / 64;
            int wave_id = threadIdx.x / 64;
            int lane_id = threadIdx.x % 64;
            int lo_hi = lane_id / 32;
            int sub_lane_id = lane_id % 32;

            int64_t base_offset = blockIdx.x * BLOCK_SIZE * INNER * bytes_per_issue;
            int idx = (lo_hi * 8 + wave_id * 16 + sub_lane_id * total_waves * 16) * INNER;

            // int idx = (blockIdx.x * BLOCK_SIZE + threadIdx.x);
            int total = (args.bytes - base_offset) / 2 ; /// 16; // dwordx4
            ushort * p_src = reinterpret_cast<ushort*>(reinterpret_cast<uint8_t*>(args.src) + base_offset);
            ushort * p_dst = reinterpret_cast<ushort*>(reinterpret_cast<uint8_t*>(args.dst) + base_offset);
 	        for(auto k = 0; k < args.iters; k++) {
                for(auto c = 0; c < CHUNKS; c++) {
                    for(auto i = 0; i < INNER; i++) {
                        // auto current = idx + c * gridDim.x * BLOCK_SIZE * 8 * INNER + i * 8;
                        auto current = idx + (k * CHUNKS + c) * gridDim.x * BLOCK_SIZE * 8 * INNER + i * 8;
                        if(current < total) {
#if MEMCPY_STREAM_SWIZZLED_NONTEMP == 1
                            auto d = nt_load(p_src[current]);
                            nt_store(d, p_dst[current]);
#else
                            *reinterpret_cast<fp32x4_t*>(p_dst + current) = *reinterpret_cast<fp32x4_t*>(p_src + current);
#endif
                        }
                    }
                }
            }
        }
    };
};
