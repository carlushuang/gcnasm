// Ground truth for the Rust port: dump every opus layout's per-lane offsets
// exactly as the kernel template builds them, for all 512 threads.
#include "gemm_a8w8_mxfp8_scale_kernel_template.hpp"
#include <cstdio>
#include <cstdlib>

using T = gemm_a8w8_mxfp8_scale_traits<256, 256, 128, 1, 128, 128, 4, false>;
constexpr int SLOTS = 64;

template<int VEC, class L>
__device__ void put(int*& o, const L& u) {
    auto off = opus::layout_to_offsets<VEC>(u);
    for (int i = 0; i < int(sizeof(off) / sizeof(off[0])); i++) *o++ = off[i];
}

__global__ void dump(int* out, int stride_a, int stride_b, int stride_c, int stride_sfb) {
    using namespace opus;
    const int tid = thread_id_x();
    const int wave_id = tid / T::WARP_SIZE, lane_id = tid % T::WARP_SIZE;
    const int wave_id_m = wave_id % T::T_M, wave_id_n = wave_id / T::T_M;
    int* o = out + tid * SLOTS;
    put<T::VEC_A>(o, make_layout_ga_scale<T>(lane_id, wave_id_m, wave_id_n, stride_a));   // 2
    put<T::VEC_A>(o, make_layout_sa_scale<T>(wave_id_m, wave_id_n));                      // 2
    put<T::VEC_A>(o, make_layout_ra_scale<T>(lane_id, wave_id_m));                        // 4
    put<T::VEC_B>(o, make_layout_gb_scale<T>(lane_id, wave_id_m, wave_id_n, stride_b));   // 2
    put<T::VEC_B>(o, make_layout_gb_scale<T>(lane_id, wave_id_m, 0, stride_b));           // 2
    put<T::VEC_B>(o, make_layout_gb_scale<T>(lane_id, wave_id_m, 1, stride_b));           // 2
    put<T::VEC_B>(o, make_layout_sb_scale<T>(wave_id_m, wave_id_n));                      // 2
    put<T::VEC_B>(o, make_layout_sb_scale<T>(wave_id_m, 0));                              // 2
    put<T::VEC_B>(o, make_layout_sb_scale<T>(wave_id_m, 1));                              // 2
    put<T::VEC_B>(o, make_layout_rb_scale<T>(lane_id, wave_id_n));                        // 8
    put<1>(o, make_layout_rsfa_scale<T>(lane_id, wave_id_m));                             // 1
    put<4>(o, make_layout_rsfb_scale<T>(lane_id, wave_id_n, 0));                          // 1
    put<1>(o, make_layout_gsf_scale<T>(lane_id, wave_id_m, true, stride_sfb));            // 1
    put<1>(o, make_layout_gsf_scale<T>(lane_id, wave_id_m, false, stride_sfb));           // 1
    put<T::VEC_LDS_SCALE>(o, make_layout_ssf_scale<T>(lane_id, wave_id_m));               // 1
    auto mma = make_tiled_mma<fp8_t, fp8_t, fp32_t>(
        seq<T::E_M, T::E_N, T::E_K>{}, seq<T::T_M, T::T_N, T::T_K>{},
        seq<T::W_M, T::W_N, T::W_K>{}, mfma_adaptor_swap_ab{});
    auto p_coord_c = make_tuple(wave_id_m, lane_id % mma.grpn_c, wave_id_n, lane_id / mma.grpn_c);
    put<T::VEC_C>(o, partition_layout_c<T::VEC_C>(mma, make_tuple(stride_c, 1_I), p_coord_c)); // 8
    *o++ = mma.mma_a_len; *o++ = mma.mma_b_len; *o++ = mma.mma_c_len; *o++ = mma.grpn_c;
    while (o < out + (tid + 1) * SLOTS) *o++ = -1;
}

int main(int argc, char** argv) {
    const int sa = atoi(argv[1]), sb = atoi(argv[2]), sc = atoi(argv[3]), ssfb = atoi(argv[4]);
    int* d = nullptr;
    hipMalloc(&d, 512 * SLOTS * sizeof(int));
    dump<<<dim3(1), dim3(512)>>>(d, sa, sb, sc, ssfb);
    static int h[512 * SLOTS];
    hipMemcpy(h, d, sizeof(h), hipMemcpyDeviceToHost);
    FILE* f = fopen(argv[5], "wb");
    fwrite(h, sizeof(h), 1, f);
    fclose(f);
    printf("err=%s\n", hipGetErrorString(hipGetLastError()));
    return 0;
}
