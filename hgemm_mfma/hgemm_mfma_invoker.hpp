#include "hgemm_mfma.hpp"

#define HIP_CALL(call) do{  \
    hipError_t err = call;  \
    if(err != hipSuccess){  \
        printf("[hiperror](%d) fail to call %s",(int)err,#call);    \
        exit(0);            \
    }                       \
} while(0)

using gld_traits = tuple<gld_trait<false, true>, gld_trait<false>>;
// using kernel = gemm_kernel<tuple<f16, f16, f16, f32>, seq<256, 128, 32>, seq<2, 2, 1>, seq<32, 32, 16>, seq<8, 8, 8>, gld_traits>;
using kernel = gemm_kernel<tuple<f16, f16, f16, f32>, seq<128, 256, 32>, seq<4, 1, 1>, seq<32, 32, 16>, seq<8, 8, 8>, gld_traits>;

template<typename kernel_type>
struct gemm_invoker {
    typename kernel_type::args make_karg(void * ptr_a,
        void * ptr_b,
        void * ptr_c,
        index_t m,
        index_t n,
        index_t k,
        index_t lda,    // in unit of pixel
        index_t ldb,
        index_t ldc)
    {
        return typename kernel_type::args{ptr_a, ptr_b, ptr_c, m, n, k, lda, ldb, ldc};
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
    }
};
