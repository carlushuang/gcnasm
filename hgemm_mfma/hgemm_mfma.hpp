#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <random>
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#define HALF
#ifdef HALF
#include "half.hpp"
#endif
#define PER_PIXEL_CHECK
#define ASSERT_ON_FAIL
#define MFMA
//#define ASM_PRINT

#ifndef ABS
#define ABS(x) ((x)>0?(x):-1*(x))
#endif

#ifndef MIN
#define MIN(x, y) ((x) > (y) ? (y) : (x))
#endif

#ifndef MAX
#define MAX(x, y) ((x) > (y) ? (x) : (y))
#endif

#include "primitives.hpp"

template<index_t M_PER_BLOCK_, index_t N_PER_BLOCK_, index_t M01_ = 8>
struct tile_scheduler{
    static constexpr index_t M_PER_BLOCK = M_PER_BLOCK_;
    static constexpr index_t N_PER_BLOCK = N_PER_BLOCK_;

    DEVICE_HOST constexpr tile_scheduler(index_t m_, index_t n_)
        : m(m_), n(n_) {}

    DEVICE_HOST constexpr auto operator()(index_t & i_m, index_t & i_n)
    {
#if 0
        index_t n_total_iters = (n + N_PER_BLOCK - 1) / N_PER_BLOCK;
        i_n = (blockIdx.x % n_total_iters) * N_PER_BLOCK;
        i_m = (blockIdx.x / n_total_iters) * M_PER_BLOCK;
#else
        index_t m0 = (m + M_PER_BLOCK - 1) / M_PER_BLOCK;
        index_t n0 = (n + N_PER_BLOCK - 1) / N_PER_BLOCK;

        index_t idx_n0 = blockIdx.x % n0;
        index_t idx_m0 = blockIdx.x / n0;

        const auto m01_adapt = (idx_m0 < m0 - m0 % M01_) ? M01_ : m0 % M01_;

        index_t idx_m00          = idx_m0 / M01_;
        index_t idx_m01          = idx_m0 % M01_;
        index_t idx_n0_m01_local = idx_n0 + idx_m01 * n0;

        i_m = (idx_n0_m01_local % m01_adapt + idx_m00 * M01_) * M_PER_BLOCK;
        i_n = (idx_n0_m01_local / m01_adapt) * N_PER_BLOCK;
#endif
    }
    index_t m;
    index_t n;
};

template<bool USE_GLD_IF_ = false, bool BYPASS_LDS_ = false>
struct gld_trait {
    static constexpr bool USE_GLD_IF = USE_GLD_IF_;
    static constexpr bool BYPASS_LDS = BYPASS_LDS_;
};

template<typename dtype_, index_t S_PER_BLOCK_, index_t BLOCK_S_WAVES_, index_t S_PER_WAVE_, index_t R_PACK_, index_t R0_PER_ROW_, bool load_all_s_repeat = true>
struct sld_iterator_r0_s_r1 {
    static constexpr index_t WAVE_S_REPEAT = S_PER_BLOCK_ / (BLOCK_S_WAVES_ * S_PER_WAVE_);
    static constexpr index_t issues = [](){if constexpr(load_all_s_repeat) return WAVE_S_REPEAT; else return 1;}();
    static constexpr index_t n_bufs = [](){if constexpr(load_all_s_repeat) return WAVE_S_REPEAT; else return MIN(2, WAVE_S_REPEAT); }();
    using sld_inst_type = sld<R_PACK_ * (4 / sizeof(dtype_))>;
    using sld_vector_type = typename vector_type<dtype_, R_PACK_>::type;
    DEVICE constexpr sld_iterator_r0_s_r1(void * smem_, index_t v_offset_, index_t base_addr_ = 0) : smem(smem_), v_offset(v_offset_), base_addr(base_addr_) {}
    template<index_t i_s_repeat, index_t i_r_repeat>
    DEVICE constexpr auto i_offset() {
        return (i_s_repeat * BLOCK_S_WAVES_ * S_PER_WAVE_ * R_PACK_ + i_r_repeat * R0_PER_ROW_ * (R_PACK_/*padding*/ + S_PER_BLOCK_ * R_PACK_)) * sizeof(dtype_);
    }
    template<index_t i_r_repeat>
    DEVICE constexpr auto load_all(){
        constexpr_for<0, issues, 1>{}([&](auto i_issue){
            sld_inst_type{}(smem, buf.template to_varray<sld_vector_type>()[i_issue], v_offset + base_addr, i_offset<i_issue, i_r_repeat>());
        });
    }
    template<index_t i_s_repeat, index_t i_r_repeat, index_t i_buf>
    DEVICE constexpr auto load()
    {
        static_assert(i_buf < n_bufs);
        sld_inst_type{}(smem, buf.template to_varray<sld_vector_type>()[number<i_buf>{}], v_offset + base_addr, i_offset<i_s_repeat, i_r_repeat>());
    }
    template<index_t i_buf>
    DEVICE constexpr auto & get()
    {
        static_assert(i_buf < n_bufs);
        return buf.template to_varray<sld_vector_type>()[number<i_buf>{}];
    }
    template<typename v_type, index_t i_buf>
    DEVICE constexpr auto & getv()
    {
        // static_assert(i_buf < n_bufs);
        return buf.template to_varray<v_type>()[number<i_buf>{}];
    }
    vector_type<dtype_, R_PACK_ * n_bufs> buf;
    void * smem;
    index_t v_offset;
    index_t base_addr;
};

template<typename dtype_, index_t BLOCK_SIZE_, index_t S_PER_BLOCK_, index_t R_PER_BLOCK_, index_t R_PACK_>
struct sst_iterator_r0_s_r1 {
    static constexpr index_t issues = S_PER_BLOCK_ * R_PER_BLOCK_ / BLOCK_SIZE_ / R_PACK_;
    static constexpr index_t n_bufs = issues;
    using sst_inst_type = sst<R_PACK_ * 4 / sizeof(dtype_)>;
    using sst_vector_type = typename vector_type<dtype_, R_PACK_>::type;
    DEVICE constexpr sst_iterator_r0_s_r1(void * smem_, index_t base_addr_ = 0) : smem(smem_), base_addr(base_addr_) {}
    DEVICE constexpr auto v_offset()
    {
        index_t i_r = threadIdx.x % (R_PER_BLOCK_ / R_PACK_);
        index_t i_s = threadIdx.x / (R_PER_BLOCK_ / R_PACK_);
        return base_addr + (i_r * (R_PACK_/*padding*/ + S_PER_BLOCK_ * R_PACK_) + i_s * R_PACK_) * sizeof(dtype_);
    }
    DEVICE constexpr auto i_offset(index_t i_issue)
    {
        const index_t stride_per_issue = (BLOCK_SIZE_ / (R_PER_BLOCK_ / R_PACK_)) * R_PACK_ * sizeof(dtype_);
        return i_issue * stride_per_issue;
    }
    template<typename T, index_t N>
    DEVICE constexpr auto operator()(const vector_type<T, N> & buf)
    {
        static_assert(sizeof(T) == sizeof(dtype_));
        static_assert(N == R_PACK_ * n_bufs);
        constexpr_for<0, issues, 1>{}([&](auto i_issue){
            sst_inst_type{}(smem, v_offset(),
                        buf.template to_varray<sst_vector_type>()[i_issue],
                        i_offset(i_issue));
        });
    }
    void * smem;
    index_t base_addr;
};

template<typename dtype_, index_t BLOCK_SIZE_, index_t S_PER_BLOCK_, index_t R_PER_BLOCK_, index_t ALIGNMENT_, typename TRAIT_>
struct gld_iterator_s_r {
    static constexpr index_t issues = S_PER_BLOCK_ * R_PER_BLOCK_ / BLOCK_SIZE_ / ALIGNMENT_;
    static constexpr index_t n_bufs = issues;
    static constexpr bool USE_GLD_IF = TRAIT_::USE_GLD_IF;

    template<bool> struct gld_inst_selector;
    template<> struct gld_inst_selector<true> { using type = gld_if<ALIGNMENT_ * 4 / sizeof(dtype_)>; };
    template<> struct gld_inst_selector<false> { using type = gld<ALIGNMENT_ * 4 / sizeof(dtype_)>; };
    using gld_inst_type = typename gld_inst_selector<USE_GLD_IF>::type;
    using gld_vector_type = typename vector_type<dtype_, ALIGNMENT_>::type;

    DEVICE constexpr gld_iterator_s_r(dtype_ * base_ptr_, index_t s_dim_, index_t r_dim_, index_t s_stride_)
    {
        base_ptr = base_ptr_;
        base_offset = 0;
        s_dim = s_dim_;
        r_dim = r_dim_;
        s_stride = s_stride_;
        i_r = (threadIdx.x % (R_PER_BLOCK_ / ALIGNMENT_)) * ALIGNMENT_;
        i_s = threadIdx.x / (R_PER_BLOCK_ / ALIGNMENT_);
        constexpr_for<0, issues, 1>{}([&](auto i_issue){
            flags[i_issue] = (i_r < r_dim) & ((i_s + i_issue * (BLOCK_SIZE_ / (R_PER_BLOCK_ / ALIGNMENT_))) < s_dim) ;
        });
    }
    DEVICE constexpr auto v_offset() {
        return (i_s * s_stride + i_r) * sizeof(dtype_);
    }
    DEVICE constexpr auto s_offset(index_t issue) {
        const index_t stride_per_issue = (BLOCK_SIZE_ / (R_PER_BLOCK_ / ALIGNMENT_)) * s_stride;
        return issue * stride_per_issue * sizeof(dtype_);
    }
    DEVICE constexpr auto clear_buf()
    {
        clear(buf);
    }
    DEVICE constexpr auto operator()()
    {
        constexpr_for<0, issues, 1>{}([&](auto i_issue){
            gld_inst_type{}(buf.template to_varray<gld_vector_type>()[i_issue],
                            make_buffer_resource(base_ptr + base_offset), v_offset(),
                            s_offset(i_issue), 0, flags[i_issue]);
        });
    }
    DEVICE constexpr auto move_slice_window(index_t r_step)
    {
        base_offset += r_step;
        // we only move along 1 dim, so only need to re-calculate flag based on 1 dim.
        constexpr_for<0, issues, 1>{}([&](auto i_issue){
            flags[i_issue] = flags[i_issue] & ((i_r + base_offset) < r_dim);
        });
    }
    template<index_t i_buf>
    DEVICE constexpr auto & get()
    {
        static_assert(i_buf < n_bufs);
        return buf.template to_varray<gld_vector_type>()[number<i_buf>{}];
    }
    template<typename v_type, index_t i_buf>
    DEVICE constexpr auto & getv()
    {
        // static_assert(i_buf < n_bufs);
        return buf.template to_varray<v_type>()[number<i_buf>{}];
    }
    dtype_ * base_ptr;
    index_t base_offset;
    index_t s_dim;
    index_t r_dim;
    index_t s_stride;
    index_t i_s;
    index_t i_r;

    static_buffer<index_t, issues> flags;
    vector_type<dtype_, ALIGNMENT_ * issues> buf;
};
#if 1
template<typename dtype_, index_t BLOCK_SIZE_, index_t S_PER_BLOCK_, index_t BLOCK_S_WAVES_, index_t S_PER_WAVE_,
                    index_t R_PER_BLOCK_, index_t BLOCK_R_WAVES_, index_t R_PER_WAVE_, index_t ALIGNMENT_, typename mfma_inst_, typename TRAIT_>
struct gld_iterator_s_r_direct_to_reg {
    // TODO: use a to compute size.
    // TODO: currently only support vector load size equal to mfma num_v_a/b. Actually it's easy to change to vector size not equal to mfma...
    using mfma_inst = mfma_inst_;
    static_assert(ALIGNMENT_ == mfma_inst::num_v_a);
    static constexpr index_t issues = S_PER_BLOCK_ * R_PER_BLOCK_ / BLOCK_SIZE_ / ALIGNMENT_;
    static constexpr index_t issues_r = R_PER_BLOCK_ / mfma_inst::m;
    static constexpr index_t issues_s = S_PER_BLOCK_ / (BLOCK_S_WAVES_ * S_PER_WAVE_);
    static constexpr index_t i_stride_s = BLOCK_S_WAVES_ * S_PER_WAVE_;
    static constexpr index_t i_stride_r = mfma_inst::m;
    static_assert(issues == issues_r * issues_s);
    static constexpr index_t n_bufs = issues;
    static constexpr bool USE_GLD_IF = TRAIT_::USE_GLD_IF;

    template<bool> struct gld_inst_selector;
    template<> struct gld_inst_selector<true> { using type = gld_if<ALIGNMENT_ * 4 / sizeof(dtype_)>; };
    template<> struct gld_inst_selector<false> { using type = gld<ALIGNMENT_ * 4 / sizeof(dtype_)>; };
    using gld_inst_type = typename gld_inst_selector<USE_GLD_IF>::type;
    using gld_vector_type = typename vector_type<dtype_, ALIGNMENT_>::type;

    DEVICE constexpr gld_iterator_s_r_direct_to_reg(dtype_ * base_ptr_, index_t s_dim_, index_t r_dim_, index_t s_stride_)
    {
        base_ptr = base_ptr_;
        base_offset = 0;
        s_dim = s_dim_;
        r_dim = r_dim_;
        s_stride = s_stride_;

        index_t lane_id = threadIdx.x % 64;
        index_t wave_id = threadIdx.x / 64;
        i_s = lane_id % mfma_inst::m + (wave_id / BLOCK_S_WAVES_) * S_PER_WAVE_;
        i_r = lane_id / mfma_inst::m * mfma_inst::num_v_a;

        constexpr_for<0, issues, 1>{}([&](auto i_issue){
            constexpr auto i_issue_r = number<i_issue % issues_r>{};
            constexpr auto i_issue_s = number<i_issue / issues_r>{};
            flags[i_issue] = ((i_r + i_issue_r * i_stride_r) < r_dim) & ((i_s + i_issue_s * i_stride_s) < s_dim) ;
        });
    }
    DEVICE constexpr auto v_offset() {
        return (i_s * s_stride + i_r) * sizeof(dtype_);
    }
    DEVICE constexpr auto s_offset(index_t issue) {
        const index_t stride_per_issue = (BLOCK_SIZE_ / (R_PER_BLOCK_ / ALIGNMENT_)) * s_stride;
        return issue * stride_per_issue * sizeof(dtype_);
    }
    DEVICE constexpr auto operator()()
    {
        constexpr_for<0, issues, 1>{}([&](auto i_issue){
            constexpr auto i_issue_r = number<i_issue % issues_r>{};
            constexpr auto i_issue_s = number<i_issue / issues_r>{};
            gld_inst_type{}(buf.template to_varray<gld_vector_type>()[i_issue],
                            make_buffer_resource(base_ptr + base_offset), v_offset(),
                            s_offset(i_issue_s), i_issue_r * i_stride_s * sizeof(dtype_), flags[i_issue]);
        });
    }
    DEVICE constexpr auto move_slice_window(index_t r_step)
    {
        base_offset += r_step;
        // we only move along 1 dim, so only need to re-calculate flag based on 1 dim.
        constexpr_for<0, issues, 1>{}([&](auto i_issue){
            constexpr auto i_issue_r = number<i_issue % issues_r>{};
            constexpr auto i_issue_s = number<i_issue / issues_r>{};
            flags[i_issue] = flags[i_issue] & ((i_r + i_issue_r * i_stride_r + base_offset) < r_dim);
        });
    }
    template<index_t i_buf>
    DEVICE constexpr auto & get()
    {
        static_assert(i_buf < n_bufs);
        return buf.template to_varray<gld_vector_type>()[number<i_buf>{}];
    }
    template<typename v_type, index_t i_buf>
    DEVICE constexpr auto & getv()
    {
        // static_assert(i_buf < n_bufs);
        return buf.template to_varray<v_type>()[number<i_buf>{}];
    }
    dtype_ * base_ptr;
    index_t base_offset;
    index_t s_dim;
    index_t r_dim;
    index_t s_stride;
    index_t i_s;
    index_t i_r;

    static_buffer<index_t, issues> flags;
    vector_type<dtype_, ALIGNMENT_ * issues> buf;
};
#endif
template<typename DATA_TYPES_, typename BLOCK_TILE_, typename BLOCK_WAVES_, typename WAVE_TILE_>
struct epilogue_iterator {
    static constexpr index_t M_PER_BLOCK = BLOCK_TILE_::template get<0>();
    static constexpr index_t N_PER_BLOCK = BLOCK_TILE_::template get<1>();
    static constexpr index_t K_PER_BLOCK = BLOCK_TILE_::template get<2>();

    static constexpr index_t BLOCK_M_WAVES = BLOCK_WAVES_::template get<0>();
    static constexpr index_t BLOCK_N_WAVES = BLOCK_WAVES_::template get<1>();
    static constexpr index_t BLOCK_K_WAVES = BLOCK_WAVES_::template get<2>();

    static constexpr index_t M_PER_WAVE = WAVE_TILE_::template get<0>();
    static constexpr index_t N_PER_WAVE = WAVE_TILE_::template get<1>();
    static constexpr index_t K_PER_WAVE = WAVE_TILE_::template get<2>();

    static constexpr index_t WAVE_M_REPEAT = M_PER_BLOCK / (BLOCK_M_WAVES * M_PER_WAVE);
    static constexpr index_t WAVE_N_REPEAT = N_PER_BLOCK / (BLOCK_N_WAVES * N_PER_WAVE);
    static constexpr index_t WAVE_K_REPEAT = K_PER_BLOCK / (BLOCK_K_WAVES * K_PER_WAVE);

    static constexpr index_t BLOCK_SIZE = BLOCK_M_WAVES * BLOCK_N_WAVES * BLOCK_K_WAVES * 64;

    using a_type = remove_cvref_t<decltype(DATA_TYPES_{}.template get<0>())>;
    using b_type = remove_cvref_t<decltype(DATA_TYPES_{}.template get<1>())>;
    using c_type = remove_cvref_t<decltype(DATA_TYPES_{}.template get<2>())>;
    using acc_type = remove_cvref_t<decltype(DATA_TYPES_{}.template get<3>())>;

    using mfma_inst = typename mfma_selector<a_type, b_type, acc_type, M_PER_WAVE, N_PER_WAVE, K_PER_WAVE>::type;

    DEVICE index_t sst_v_offset()
    {
        // store per repeat
        index_t lane_id = threadIdx.x % 64;
        index_t wave_id = threadIdx.x / 64;
        index_t i_m = lane_id % mfma_inst::m + (wave_id / BLOCK_N_WAVES) * M_PER_WAVE;
        index_t i_n = lane_id / mfma_inst::m * mfma_inst::c_per_group + (wave_id % BLOCK_N_WAVES) * N_PER_WAVE;
        return (i_m * (mfma_inst::c_per_group/*padding*/ + BLOCK_N_WAVES * N_PER_WAVE) + i_n) * sizeof(c_type);
    }
    DEVICE constexpr epilogue_iterator(c_type * ptr_, index_t m_dim_, index_t n_dim_, index_t stride_, char * smem_)
    {
        ptr = ptr_;
        m_dim = m_dim_;
        n_dim = n_dim_;
        stride = stride_;
        smem = smem_;
        col_id = threadIdx.x % (BLOCK_N_WAVES * N_PER_WAVE / mfma_inst::c_per_group) * mfma_inst::c_per_group;
        row_id = threadIdx.x / (BLOCK_N_WAVES * N_PER_WAVE / mfma_inst::c_per_group);
        gst_v_offset_base = (row_id * stride_ + col_id) * sizeof(c_type);
        sst_v_offset_base = sst_v_offset();
        sld_v_offset_base = (row_id * (mfma_inst::c_per_group/*padding*/ + BLOCK_N_WAVES * N_PER_WAVE) + col_id) * sizeof(c_type);
    }

    template<typename ACC_VECTOR>
    DEVICE constexpr void operator()(const ACC_VECTOR& acc_buf)
    {
        using acc_group_t = typename vector_type<acc_type, mfma_inst::c_per_group>::type;
        using c_group_t = typename vector_type<c_type, mfma_inst::c_per_group>::type;
        using shfl_sst_inst_type = sst<mfma_inst::c_per_group * 4 / sizeof(c_type)>;
        using shfl_sld_inst_type = sld<mfma_inst::c_per_group * 4 / sizeof(c_type)>;
        constexpr index_t rows_per_sld_gst = BLOCK_SIZE / (BLOCK_N_WAVES * N_PER_WAVE / mfma_inst::c_per_group);
        constexpr index_t stride_per_sld =  rows_per_sld_gst *
                                                (mfma_inst::c_per_group/*padding*/ + BLOCK_N_WAVES * N_PER_WAVE) * sizeof(c_type);

        constexpr_for<0, WAVE_M_REPEAT, 1>{}([&](auto i_m){
            constexpr_for<0, WAVE_N_REPEAT, 1>{}([&](auto i_n){
                wave_barrier();
                // store to smem
                constexpr_for<0, mfma_inst::groups, 1>{}([&](auto i_g){
                    constexpr auto v_idx = number<(i_m * WAVE_N_REPEAT + i_n) * mfma_inst::c_per_group + i_g>{};
                    auto tmp = vector_cast<c_type>(vector_type<acc_type, mfma_inst::c_per_group>{acc_buf.template to_varray<acc_group_t>()[v_idx]});
                    shfl_sst_inst_type{}(smem, sst_v_offset_base, tmp, i_g * mfma_inst::rows_per_group * sizeof(c_type) );
                });
                sst_fence(0);
                wave_barrier();
                // load from smem
                vector_type<c_type, mfma_inst::c_per_group * mfma_inst::groups> gst_buf;
                constexpr_for<0, mfma_inst::groups, 1>{}([&](auto i_g){
                    shfl_sld_inst_type{}(smem, gst_buf.template to_varray<c_group_t>()[i_g], sld_v_offset_base, i_g * stride_per_sld );
                });
                constexpr_for<0, mfma_inst::groups, 1>{}([&](auto i_g){
                    sld_fence(mfma_inst::groups - i_g - 1);
                    index_t flag = ((i_m * BLOCK_M_WAVES * M_PER_WAVE + i_g * rows_per_sld_gst + row_id) < m_dim) && 
                                    ((i_n * BLOCK_N_WAVES * N_PER_WAVE + col_id) < n_dim);
                    gst_if<sizeof(c_group_t)>{}(gst_buf.template to_varray<c_group_t>()[i_g],
                            make_buffer_resource(ptr),
                            gst_v_offset_base, /*v*/
                            ((i_m * BLOCK_M_WAVES * M_PER_WAVE + i_g * rows_per_sld_gst) * stride + i_n * BLOCK_N_WAVES * N_PER_WAVE) * sizeof(c_type), /*s*/
                            0/*i*/,
                            flag);
                });
            });
        });
        gst_fence(0);
    }

    c_type * ptr;
    index_t m_dim;
    index_t n_dim;
    index_t stride;
    char * smem;

    index_t col_id;
    index_t row_id;

    index_t gst_v_offset_base;
    index_t sst_v_offset_base;
    index_t sld_v_offset_base;
};

// TODO: this is a structure for 2 purpose, 1. compute mfma distributione. 2. compute sld offset considering the LDS layout
// can refactor into a more generic structure
template<typename a_type_, typename b_type_, typename mfma_inst_, index_t M_PER_BLOCK_, index_t BLOCK_M_WAVES_, index_t M_PER_WAVE_,
                                            index_t N_PER_BLOCK_, index_t BLOCK_N_WAVES_, index_t N_PER_WAVE_,
                                            index_t KPACK_A_, index_t KPACK_B_>
struct mfma_mapping_for_sld {
    DEVICE constexpr auto operator()(index_t & v_offset_a, index_t & v_offset_b)
    {
        index_t lane_id = threadIdx.x % 64;
        index_t wave_id = threadIdx.x / 64;
        index_t src_i_m = lane_id % mfma_inst_::m + (wave_id / BLOCK_N_WAVES_) * M_PER_WAVE_;
        index_t src_i_n = lane_id % mfma_inst_::n + (wave_id % BLOCK_N_WAVES_) * N_PER_WAVE_;
        index_t src_i_k = lane_id / mfma_inst_::m;
        v_offset_a = (src_i_m * KPACK_A_ + src_i_k * (KPACK_A_/*padding*/ + M_PER_BLOCK_ * KPACK_A_)) * sizeof(a_type_);
        v_offset_b = (src_i_n * KPACK_B_ + src_i_k * (KPACK_B_/*padding*/ + N_PER_BLOCK_ * KPACK_B_)) * sizeof(b_type_);
    }
};

template<index_t M_REPEAT_, index_t N_REPEAT_, index_t K_REPEAT_, index_t K_PER_BLOCK_, typename mfma_inst_,
        typename SLD_A_, typename SLD_B_, typename SST_A_, typename SST_B_, typename GLD_A_, typename GLD_B_,
        typename GLD_BUF_CLEAR_, typename MFMA_MAPPING_FOR_SLD_>
struct gemm_pipeline_flat {
    using mfma_inst = remove_cvref_t<mfma_inst_>;
    using SLD_A = remove_cvref_t<SLD_A_>;
    using SLD_B = remove_cvref_t<SLD_B_>;
    using SST_A = remove_cvref_t<SST_A_>;
    using SST_B = remove_cvref_t<SST_B_>;
    using GLD_A = remove_cvref_t<GLD_A_>;
    using GLD_B = remove_cvref_t<GLD_B_>;
    using GLD_BUF_CLEAR = remove_cvref_t<GLD_BUF_CLEAR_>;
    using MFMA_MAPPING_FOR_SLD = remove_cvref_t<MFMA_MAPPING_FOR_SLD_>;

    char * smem;
    index_t lds_a_offset;
    index_t lds_b_offset;
    GLD_A & gld_a;
    GLD_B & gld_b;
    GLD_BUF_CLEAR & gld_buf_clear;

    DEVICE constexpr gemm_pipeline_flat(
                            char * smem_,
                            index_t lds_a_offset_,
                            index_t lds_b_offset_,
                            GLD_A_ & gld_a_,
                            GLD_B_ & gld_b_,
                            GLD_BUF_CLEAR_ & gld_buf_clear_) :
                    smem(smem_),
                    lds_a_offset(lds_a_offset_),
                    lds_b_offset(lds_b_offset_),
                    gld_a(gld_a_),
                    gld_b(gld_b_),
                    gld_buf_clear(gld_buf_clear_) {}

    template<typename ACC_BUF_>
    DEVICE void operator()(ACC_BUF_ & acc_buf, index_t k_iters)
    {
        gld_a.clear_buf();
        gld_a(); gld_a.move_slice_window(K_PER_BLOCK_);
        gld_b.clear_buf();
        gld_b(); gld_b.move_slice_window(K_PER_BLOCK_);

        index_t v_offset_a, v_offset_b;
        MFMA_MAPPING_FOR_SLD{}(v_offset_a, v_offset_b);
        auto sst_a = SST_A{smem, lds_a_offset};
        auto sst_b = SST_B{smem, lds_b_offset};

        auto sld_a = SLD_A{smem, v_offset_a, lds_a_offset};
        auto sld_b = SLD_B{smem, v_offset_b, lds_b_offset};

        clear(acc_buf); // TODO: check this preheader, seems will schedule 2 times
        gld_fence(gld_b.issues);
        sst_a(gld_a.buf);
        gld_fence(0);
        sst_b(gld_b.buf);
        gld_buf_clear();

        for(auto i_k = 1; i_k < k_iters; i_k++) {
            gemm_(acc_buf, gld_a, gld_b, sld_a, sld_b, sst_a, sst_b, bool_const<true>{});
        }
        // tail
        gemm_(acc_buf, gld_a, gld_b, sld_a, sld_b, sst_a, sst_b, bool_const<false>{});
    }

    template<typename ACC_BUF_, typename HOT_LOOP_ = bool_const<true>>
    DEVICE void gemm_(ACC_BUF_ & acc_buf, GLD_A & gld_iter_a, GLD_B & gld_iter_b, SLD_A & sld_iter_a, SLD_B & sld_iter_b, SST_A & sst_iter_a, SST_B & sst_iter_b,
                                HOT_LOOP_ is_hot_loop = bool_const<true>{})
    {
        using acc_type = typename ACC_BUF_::d1_t;
        using acc_t = typename vector_type<acc_type, mfma_inst::num_v_c>::type;
        auto mfma = mfma_inst{};

        // let everything into 1 dim, easy to control the sld/gld/sst slot
        constexpr_for<0, K_REPEAT_ * M_REPEAT_ * N_REPEAT_, 1>{}([&](auto i_3d){
            constexpr auto i_k = number<i_3d / ( M_REPEAT_ * N_REPEAT_)>{};
            constexpr auto i_2d = number<i_3d % ( M_REPEAT_ * N_REPEAT_)>{};
            constexpr auto i_m = number<i_2d / N_REPEAT_>{};
            constexpr auto i_n = number<i_2d % N_REPEAT_>{};
            constexpr auto i_next_n = number<(i_2d + 1) % N_REPEAT_>{};

            constexpr auto need_sld_a = bool_const<i_m == 0 && i_n == 0>{};
            constexpr auto need_sld_b_first = bool_const<i_m == 0 && i_n == 0>{};
            constexpr auto need_sld_b_prefetch = bool_const<!(i_m == M_REPEAT_ - 1 && i_n == N_REPEAT_ - 1)>{};
            constexpr auto need_gld_a = bool_const<i_k == 0 && i_m == 0 && i_n == 0 && is_hot_loop>{};
            constexpr auto need_gld_b = bool_const<i_k == 0 && (i_m == M_REPEAT_ - 1 && i_n == N_REPEAT_ - 1) && is_hot_loop>{};
            constexpr auto need_wait_gld_sst = bool_const<(i_3d == K_REPEAT_ * M_REPEAT_ * N_REPEAT_ - 1) && is_hot_loop>{};

            // conditionally do gld
            if constexpr(need_gld_a)
                gld_iter_a();
            if constexpr(need_gld_b)
                gld_iter_b();

            if constexpr(i_3d == 0) {
                // it is good to self contains this barrier inside constexpr for
                sst_fence(0); wave_barrier();
            }

            // conditionally do sld
            if constexpr(need_sld_a)
                sld_iter_a.template load_all<i_k>();

            if constexpr(need_sld_b_first)
                sld_iter_b.template load<0, i_k, 0>();

            if constexpr(need_sld_b_prefetch)
                sld_iter_b.template load<i_next_n, i_k, i_next_n % 2>();

            // TODO: this may have bugs if repeat if not large (?)
            auto sld_fence_cnt = [&](){
                if constexpr(need_sld_b_prefetch)
                    return sld_iter_b.issues;
                return index_t(0); }();
            sld_fence(sld_fence_cnt);

            // conditionally do sst, should be the last one
            if constexpr(need_wait_gld_sst) {
                gld_iter_a.move_slice_window(K_PER_BLOCK_);
                gld_iter_b.move_slice_window(K_PER_BLOCK_);
                wave_barrier();
                gld_fence(gld_iter_b.issues);
                sst_iter_a(gld_iter_a.buf);
                gld_fence(0);
                sst_iter_b(gld_iter_b.buf);
                gld_buf_clear();
            }
            mfma(sld_iter_a.template get<i_m>(), sld_iter_b.template get<i_n % 2>(),
                            acc_buf.template to_varray<acc_t>()[number<i_m * N_REPEAT_ + i_n>{}], bool_const<true>{});
        });
    }
};

template<index_t X_REPEAT_, index_t Y_REPEAT_, index_t K_REPEAT_, index_t K_PER_BLOCK_, typename mfma_inst_,
            typename SLD_Y_, typename SST_Y_, typename GLD_X_, typename GLD_Y_, typename GLD_BUF_CLEAR_>
struct gemm_pipeline_oneside_lds {
    // one side of A/B use LDS, the other side just direct store data into register.
    // we use X for the side not using LDS, Y for the side using LDS.
    using mfma_inst = remove_cvref_t<mfma_inst_>;
    using SLD_Y = remove_cvref_t<SLD_Y_>;
    using SST_Y = remove_cvref_t<SST_Y_>;
    using GLD_X = remove_cvref_t<GLD_Y_>;
    using GLD_Y = remove_cvref_t<GLD_Y_>;
    using GLD_BUF_CLEAR = remove_cvref_t<GLD_BUF_CLEAR_>;

    SLD_Y & sld_iter_y;
    SST_Y & sst_iter_y;
    GLD_X & gld_iter_x;
    GLD_Y & gld_iter_y;
    GLD_BUF_CLEAR & gld_buf_clear;

    constexpr gemm_pipeline_oneside_lds(
                            SLD_Y & sld_iter_y_,
                            SST_Y & sst_iter_y_,
                            GLD_X & gld_iter_x_,
                            GLD_Y & gld_iter_y_,
                            GLD_BUF_CLEAR & gld_buf_clear_) :
                    sld_iter_y(sld_iter_y_),
                    sst_iter_y(sst_iter_y_),
                    gld_iter_x(gld_iter_x_),
                    gld_iter_y(gld_iter_y_),
                    gld_buf_clear(gld_buf_clear_) {}

    template<typename ACC_BUF_, typename HOT_LOOP_ = bool_const<true>>
    DEVICE constexpr void operator()(ACC_BUF_ & acc_buf, HOT_LOOP_ is_hot_loop = bool_const<true>{})
    {
        using acc_type = typename ACC_BUF_::d1_t;
        using acc_t = typename vector_type<acc_type, mfma_inst::num_v_c>::type;
        auto mfma = mfma_inst{};

        // let everything into 1 dim, easy to control the sld/gld/sst slot
        constexpr_for<0, K_REPEAT_ * X_REPEAT_ * Y_REPEAT_, 1>{}([&](auto i_3d){
            constexpr auto i_k = number<i_3d / ( X_REPEAT_ * Y_REPEAT_)>{};
            constexpr auto i_2d = number<i_3d % ( X_REPEAT_ * Y_REPEAT_)>{};
            constexpr auto i_x = number<i_2d / Y_REPEAT_>{};
            constexpr auto i_y = number<i_2d % Y_REPEAT_>{};
            constexpr auto i_next_y = number<(i_2d + 1) % Y_REPEAT_>{};

            constexpr auto need_sld_y_first = bool_const<i_x == 0 && i_y == 0>{};
            constexpr auto need_sld_y_prefetch = bool_const<!(i_x == X_REPEAT_ - 1 && i_y == Y_REPEAT_ - 1)>{};
            constexpr auto need_gld_x = bool_const<i_k == 0 && i_x == 0 && i_y == 0 && is_hot_loop>{};
            constexpr auto need_gld_y = bool_const<i_k == 0 && (i_x == X_REPEAT_ - 1 && i_y == Y_REPEAT_ - 1) && is_hot_loop>{};
            constexpr auto need_wait_gld_sst = bool_const<(i_3d == K_REPEAT_ * X_REPEAT_ * Y_REPEAT_ - 1) && is_hot_loop>{};

            // conditionally do gld
            if constexpr(need_gld_x)
                gld_iter_x();
            if constexpr(need_gld_y)
                gld_iter_y();

            if constexpr(i_3d == 0) {
                // it is good to self contains this barrier inside constexpr for
                sst_fence(0); wave_barrier();
            }

            if constexpr(need_sld_y_first)
                sld_iter_y.template load<0, i_k, 0>();

            if constexpr(need_sld_y_prefetch)
                sld_iter_y.template load<i_next_y, i_k, i_next_y % 2>();

            // TODO: this may have bugs if repeat if not large (?)
            auto sld_fence_cnt = [&](){
                if constexpr(need_sld_y_prefetch)
                    return sld_iter_y.issues;
                return index_t(0); }();
            sld_fence(sld_fence_cnt);

            // conditionally do sst, should be the last one
            if constexpr(need_wait_gld_sst) {
                gld_iter_x.move_slice_window(K_PER_BLOCK_);
                gld_iter_y.move_slice_window(K_PER_BLOCK_);
                wave_barrier();
                //gld_fence(gld_iter_y.issues);
                //sst_iter_a(gld_iter_x.buf);
                gld_fence(0);
                sst_iter_y(gld_iter_y.buf);
                gld_buf_clear();
            }
            //mfma(sld_iter_x.template get<i_x>(), sld_iter_y.template get<i_y % 2>(),
            //                acc_buf.template to_varray<acc_t>()[number<i_x * Y_REPEAT_ + i_y>{}], bool_const<true>{});
        });
    }
};

template<typename DATA_TYPES_, typename BLOCK_TILE_, typename BLOCK_WAVES_, typename WAVE_TILE_, typename ALIGNMENT_, typename GLD_TRAITS_>
struct gemm_kernel
{   // only support rcr layout
    static constexpr index_t M_PER_BLOCK = BLOCK_TILE_::template get<0>();
    static constexpr index_t N_PER_BLOCK = BLOCK_TILE_::template get<1>();
    static constexpr index_t K_PER_BLOCK = BLOCK_TILE_::template get<2>();

    static constexpr index_t BLOCK_M_WAVES = BLOCK_WAVES_::template get<0>();
    static constexpr index_t BLOCK_N_WAVES = BLOCK_WAVES_::template get<1>();
    static constexpr index_t BLOCK_K_WAVES = BLOCK_WAVES_::template get<2>();

    static constexpr index_t M_PER_WAVE = WAVE_TILE_::template get<0>();
    static constexpr index_t N_PER_WAVE = WAVE_TILE_::template get<1>();
    static constexpr index_t K_PER_WAVE = WAVE_TILE_::template get<2>();

    static constexpr index_t WAVE_M_REPEAT = M_PER_BLOCK / (BLOCK_M_WAVES * M_PER_WAVE);
    static constexpr index_t WAVE_N_REPEAT = N_PER_BLOCK / (BLOCK_N_WAVES * N_PER_WAVE);
    static constexpr index_t WAVE_K_REPEAT = K_PER_BLOCK / (BLOCK_K_WAVES * K_PER_WAVE);

    static constexpr index_t BLOCK_SIZE = BLOCK_M_WAVES * BLOCK_N_WAVES * BLOCK_K_WAVES * 64;

    static constexpr index_t MAX_THREADS = BLOCK_SIZE;
    static constexpr index_t MIN_BLOCKS = 1;    // TODO: need change this

    // single gld issue is one alignment vectors
    static constexpr index_t ALIGNMENT_A = ALIGNMENT_::template get<0>();
    static constexpr index_t ALIGNMENT_B = ALIGNMENT_::template get<1>();
    static constexpr index_t ALIGNMENT_C = ALIGNMENT_::template get<2>();

    using GLD_TRAIT_A = remove_cvref_t<decltype(GLD_TRAITS_{}.template get<0>())>;
    using GLD_TRAIT_B = remove_cvref_t<decltype(GLD_TRAITS_{}.template get<1>())>;

    static constexpr index_t gld_a_buffers = M_PER_BLOCK * K_PER_BLOCK / BLOCK_SIZE / ALIGNMENT_A;
    static constexpr index_t gld_b_buffers = N_PER_BLOCK * K_PER_BLOCK / BLOCK_SIZE / ALIGNMENT_B;

    // for simplicity, we just use the same vector size as k_pack
    static constexpr index_t KPACK_A = ALIGNMENT_A;
    static constexpr index_t KPACK_B = ALIGNMENT_B;

    using a_type = remove_cvref_t<decltype(DATA_TYPES_{}.template get<0>())>;
    using b_type = remove_cvref_t<decltype(DATA_TYPES_{}.template get<1>())>;
    using c_type = remove_cvref_t<decltype(DATA_TYPES_{}.template get<2>())>;
    using acc_type = remove_cvref_t<decltype(DATA_TYPES_{}.template get<3>())>;

    using mfma_inst = typename mfma_selector<a_type, b_type, acc_type, M_PER_WAVE, N_PER_WAVE, K_PER_WAVE>::type;

    struct args
    {
        void * ptr_a;
        void * ptr_b;
        void * ptr_c;
        index_t m;
        index_t n;
        index_t k;
        index_t lda;    // in unit of pixel
        index_t ldb;
        index_t ldc;
    };

    static bool is_applicable(args karg)
    {
        if((karg.k % ALIGNMENT_A != 0) || (karg.k % ALIGNMENT_B != 0) || (karg.n % ALIGNMENT_C != 0))
            return false;
        return true;
    }

    DEVICE_HOST static constexpr auto smem_size_a()
    {
        constexpr auto s = (K_PER_BLOCK / KPACK_A) * (KPACK_A/*padding*/ + M_PER_BLOCK * KPACK_A);
        return s * sizeof(a_type);
    }

    DEVICE_HOST static constexpr auto smem_size_b()
    {
        constexpr auto s = (K_PER_BLOCK / KPACK_B) * (KPACK_B/*padding*/ + N_PER_BLOCK * KPACK_B);
        return s * sizeof(b_type);
    }

    DEVICE_HOST static constexpr auto smem_size()
    {
        return MAX(smem_size_a() + smem_size_b(), smem_size_shuffle());
    }

    DEVICE_HOST static constexpr auto smem_size_shuffle()
    {
        using shfl_type = typename vector_type<c_type, mfma_inst::c_per_group>::type;
        return BLOCK_M_WAVES * M_PER_WAVE * (BLOCK_N_WAVES * N_PER_WAVE * sizeof(c_type) + sizeof(shfl_type));
    }

    DEVICE_HOST static constexpr auto block_dims()
    {
        return dim3(BLOCK_SIZE);
    }

    DEVICE_HOST static constexpr auto grid_dims(const args & karg)
    {
        auto grids = ((karg.m + M_PER_BLOCK - 1) / M_PER_BLOCK) * ((karg.n + N_PER_BLOCK - 1) / N_PER_BLOCK);
        return dim3(grids);
    }

    DEVICE auto operator()(const args & karg, char * smem){
        auto ts = tile_scheduler<M_PER_BLOCK, N_PER_BLOCK>{karg.m, karg.n};
        index_t block_i_m, block_i_n;
        ts(block_i_m, block_i_n);

        a_type * ptr_a = reinterpret_cast<a_type*>(karg.ptr_a) + block_i_m * karg.lda;
        b_type * ptr_b = reinterpret_cast<b_type*>(karg.ptr_b) + block_i_n * karg.ldb;
        c_type * ptr_c = reinterpret_cast<c_type*>(karg.ptr_c) + block_i_m * karg.ldc + block_i_n;

        auto k_iters = (karg.k + K_PER_BLOCK - 1) / K_PER_BLOCK;
        auto gld_a = gld_iterator_s_r<a_type, BLOCK_SIZE, M_PER_BLOCK, K_PER_BLOCK, ALIGNMENT_A, GLD_TRAIT_A>{ptr_a, karg.m - block_i_m, karg.k, karg.lda};
        auto gld_b = gld_iterator_s_r<b_type, BLOCK_SIZE, N_PER_BLOCK, K_PER_BLOCK, ALIGNMENT_B, GLD_TRAIT_B>{ptr_b, karg.n - block_i_n, karg.k, karg.ldb};

        //auto sst_a = sst_iterator_r0_s_r1<a_type, BLOCK_SIZE, M_PER_BLOCK, K_PER_BLOCK, KPACK_A>{smem};
        //auto sst_b = sst_iterator_r0_s_r1<b_type, BLOCK_SIZE, N_PER_BLOCK, K_PER_BLOCK, KPACK_B>{smem, smem_size_a()};

        using sst_iter_a_type = sst_iterator_r0_s_r1<a_type, BLOCK_SIZE, M_PER_BLOCK, K_PER_BLOCK, KPACK_A>;
        using sst_iter_b_type = sst_iterator_r0_s_r1<b_type, BLOCK_SIZE, N_PER_BLOCK, K_PER_BLOCK, KPACK_B>;

        using sld_iter_a_type = sld_iterator_r0_s_r1<a_type, M_PER_BLOCK, BLOCK_M_WAVES, M_PER_WAVE, KPACK_A, mfma_inst::k / KPACK_A, true>;
        using sld_iter_b_type = sld_iterator_r0_s_r1<b_type, N_PER_BLOCK, BLOCK_N_WAVES, N_PER_WAVE, KPACK_B, mfma_inst::k / KPACK_B, false>;

        vector_type<acc_type, WAVE_M_REPEAT * WAVE_N_REPEAT * mfma_inst::num_v_c> acc_buf;

        auto gld_buf_clear = [&]()
        {
            if constexpr (gld_a.USE_GLD_IF && gld_b.USE_GLD_IF) {
                if constexpr (M_PER_BLOCK > N_PER_BLOCK)
                    gld_b.clear_buf();
                else
                    gld_a.clear_buf();
            }
        };

        using mfma_mapping_for_sld_type = mfma_mapping_for_sld<a_type, b_type, mfma_inst, M_PER_BLOCK, BLOCK_M_WAVES, M_PER_WAVE,
                                                    N_PER_BLOCK, BLOCK_N_WAVES, N_PER_WAVE, KPACK_A, KPACK_B>;
        using gemm_pipepine_type = gemm_pipeline_flat<WAVE_M_REPEAT, WAVE_N_REPEAT, WAVE_K_REPEAT, K_PER_BLOCK, mfma_inst,
                    sld_iter_a_type, sld_iter_b_type, sst_iter_a_type, sst_iter_b_type,
                    decltype(gld_a), decltype(gld_b), decltype(gld_buf_clear), mfma_mapping_for_sld_type>;

        auto gemm = gemm_pipepine_type{smem, 0, smem_size_a(), gld_a, gld_b, gld_buf_clear};
        gemm(acc_buf, k_iters);
        auto epilogue = epilogue_iterator<DATA_TYPES_, BLOCK_TILE_, BLOCK_WAVES_, WAVE_TILE_> {ptr_c, karg.m - block_i_m, karg.n - block_i_n, karg.ldc, smem};
        // write out
        sched_barrier();  // in case mfma dest has raw harzard
        epilogue(acc_buf);
    }
};
