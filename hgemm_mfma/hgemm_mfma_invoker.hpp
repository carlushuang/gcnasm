#include "hgemm_mfma.hpp"

#ifndef USE_EXT_LAUNCH_KERNEL
#define USE_EXT_LAUNCH_KERNEL 0
#endif

#if USE_EXT_LAUNCH_KERNEL
#include "hip/hip_ext.h"
#endif

#define HIP_CALL(call) do{  \
    hipError_t err = call;  \
    if(err != hipSuccess){  \
        printf("[hiperror](%d) fail to call %s",(int)err,#call);    \
        exit(0);            \
    }                       \
} while(0)

using data_type_t = tuple<f16, f16, f16, f32>;
using block_tile_t =  seq<128, 256, 32>;
using block_waves_t = seq<4, 1, 1>;
using wave_tile_t = seq<32, 32, 16>;
using alignments_t = seq<8, 8, 8>;
using tile_sched_t = tile_scheduler<8>;
using gld_traits_t = tuple<gld_trait<false/*gld if*/, false/*bypass LDS*/>, gld_trait<false>>;
// template<bool gld_x_first_ = true,
//          index_t gld_second_start_distance_ = 0,
//          index_t gld_slots_ = 1,
//          index_t gld_x_issues_ = 0,
//          index_t gld_y_issues_ = 0,
//          index_t gld_x_issues_per_group_ = 1,
//          index_t gld_y_issues_per_group_ = 1,
//          index_t gld_x_issue_distance_ = 0,
//          index_t gld_y_issue_distance_ = 0,
//          /* for oneside lds */
//          index_t k_iter_mod_ = 0>
// using pipeline_trait_t = gemm_pipeline_traits<false, 5, 16 - 1, 2, 4, 2, 2, 1, 2, 0/*k_iter_mod*/>; /* use this for one-LDS */
// using pipeline_trait_t = gemm_pipeline_traits<true, 3, 16 - 1, 2, 4, 2, 2, 1, 4, 0/*k_iter_mod*/>; /* use this for one-LDS */
using pipeline_trait_t = gemm_pipeline_traits<true, 5, 16 - 1, 2, 4, 2, 2, 1, 6, 0/*k_iter_mod*/>; /* use this for dual-LDS */



using epilogue_t = epilogue_iterator<data_type_t, block_tile_t, block_waves_t, wave_tile_t>;
using kernel_t = gemm_kernel<data_type_t, block_tile_t, block_waves_t, wave_tile_t, alignments_t,
            tile_sched_t, gld_traits_t, pipeline_trait_t, epilogue_t>;

template<typename kernel_type>
struct gemm_invoker {
    template<typename... ARGS>
    static typename kernel_type::args make_karg(ARGS... args)
    {
        return kernel_type::make_karg(args...);
    }

    static bool is_applicable(typename kernel_type::args karg)
    {
        return kernel_type::is_applicable(karg);
    }

    void operator()(typename kernel_type::args karg, hipStream_t stream = nullptr) {
        kernel_entry<kernel_type><<<kernel_type::grid_dims(karg), kernel_type::block_dims(), 0/*no runtime lds*/, stream>>>(karg);
    }

    float bench(typename kernel_type::args karg, hipStream_t stream = nullptr, int warmup = 3, int loops = 10) {
        for(auto i=0 ; i < warmup ; i++)
            operator()(karg);
#if USE_EXT_LAUNCH_KERNEL
        // TODO: not working yet
        float ms = 0;
        for(auto i=0 ; i < loops ; i++)
        {
            hipEvent_t evt_00, evt_11;
            HIP_CALL(hipEventCreate(&evt_00));
            HIP_CALL(hipEventCreate(&evt_11));
            hipExtLaunchKernelGGL((kernel_entry<kernel_type>), kernel_type::grid_dims(karg), kernel_type::block_dims(),
                        0/*no runtime lds*/, stream, evt_00, evt_11, 0, karg);
            float current_ms;
            HIP_CALL(hipEventSynchronize(evt_11));
            HIP_CALL(hipEventElapsedTime(&current_ms, evt_00, evt_11));
            ms += current_ms;
            HIP_CALL(hipEventDestroy(evt_00));
            HIP_CALL(hipEventDestroy(evt_11));
        }
        return ms / loops;
#else
        hipEvent_t evt_00, evt_11;
        HIP_CALL(hipEventCreate(&evt_00));
        HIP_CALL(hipEventCreate(&evt_11));
        HIP_CALL(hipDeviceSynchronize());
        HIP_CALL(hipEventRecord(evt_00, NULL));
        for(auto i=0 ; i < loops ; i++)
            operator()(karg);
        HIP_CALL(hipEventRecord(evt_11, NULL));
        HIP_CALL(hipEventSynchronize(evt_11));
        float ms;
        HIP_CALL(hipEventElapsedTime(&ms, evt_00, evt_11));
        HIP_CALL(hipEventDestroy(evt_00));
        HIP_CALL(hipEventDestroy(evt_11));
        return ms / loops;
#endif
    }
};
