#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <random>
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <numeric>
#define HALF
#ifdef HALF
#include "half.hpp"
#endif
// #define PER_PIXEL_CHECK
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

template<index_t M01_ = 8>
struct tile_scheduler{
    template<index_t M_PER_BLOCK, index_t N_PER_BLOCK>
    DEVICE_HOST constexpr auto operator()(index_t m, index_t n, number<M_PER_BLOCK>, number<N_PER_BLOCK>)
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

        index_t i_m = (idx_n0_m01_local % m01_adapt + idx_m00 * M01_) * M_PER_BLOCK;
        index_t i_n = (idx_n0_m01_local / m01_adapt) * N_PER_BLOCK;

        return make_tuple(i_m, i_n);
#endif
    }
};

template<bool USE_GLD_IF_ = false, bool BYPASS_LDS_ = false, index_t N_PREFETCH_ = 2>
struct gld_trait {
    static constexpr bool USE_GLD_IF = USE_GLD_IF_;
    static constexpr bool BYPASS_LDS = BYPASS_LDS_;
    static constexpr index_t N_PREFETCH = N_PREFETCH_;
};

template<typename dtype_, index_t S_PER_BLOCK_, index_t BLOCK_S_WAVES_, index_t S_PER_WAVE_, index_t R_PACK_, index_t R0_PER_ROW_, bool load_all_s_repeat = true>
struct sld_iterator_r0_s_r1 {
    static constexpr index_t WAVE_S_REPEAT = S_PER_BLOCK_ / (BLOCK_S_WAVES_ * S_PER_WAVE_);
    static constexpr index_t n_issue = [](){if constexpr(load_all_s_repeat) return WAVE_S_REPEAT; else return 1;}();
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
        constexpr_for<0, n_issue, 1>{}([&](auto i_issue){
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
    static constexpr index_t n_issue = S_PER_BLOCK_ * R_PER_BLOCK_ / BLOCK_SIZE_ / R_PACK_;
    static constexpr index_t n_bufs = n_issue;
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
        constexpr_for<0, n_issue, 1>{}([&](auto i_issue){
            sst_inst_type{}(smem, v_offset(),
                        buf.template to_varray<sst_vector_type>()[i_issue],
                        i_offset(i_issue));
        });
    }
    void * smem;
    index_t base_addr;
};

template<typename dtype_, index_t BLOCK_SIZE_, index_t S_PER_BLOCK_, index_t R_PER_BLOCK_, index_t ALIGNMENT_, typename TRAITS_>
struct gld_iterator_s_r {
    static constexpr index_t n_issue = S_PER_BLOCK_ * R_PER_BLOCK_ / BLOCK_SIZE_ / ALIGNMENT_;
    static constexpr index_t n_bufs = n_issue;
    static constexpr bool USE_GLD_IF = TRAITS_::USE_GLD_IF;

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
        constexpr_for<0, n_issue, 1>{}([&](auto i_issue){
            flags[i_issue] = (i_r < r_dim) & ((i_s + i_issue * (BLOCK_SIZE_ / (R_PER_BLOCK_ / ALIGNMENT_))) < s_dim) ;
        });
    }
    DEVICE constexpr auto v_offset() {
        return (i_s * s_stride + i_r) * sizeof(dtype_);
    }
    DEVICE constexpr auto s_offset(index_t i_issue) {
        const index_t stride_per_issue = (BLOCK_SIZE_ / (R_PER_BLOCK_ / ALIGNMENT_)) * s_stride;
        return i_issue * stride_per_issue * sizeof(dtype_);
    }
    template<bool disable_inline_asm = true>
    DEVICE constexpr auto clear_buf(bool_const<disable_inline_asm> = bool_const<true>{})
    {
        clear(buf, bool_const<disable_inline_asm>{}); // TODO: seems better if let compiler to schedule, when not using setprio
    }
    DEVICE constexpr auto operator()()
    {
        constexpr_for<0, n_issue, 1>{}([&](auto i_issue){
            issue(i_issue);
        });
    }
    template<index_t i_issue>
    DEVICE constexpr auto issue(number<i_issue>)
    {
        static_assert(i_issue < n_issue);
        gld_inst_type{}(buf.template to_varray<gld_vector_type>()[number<i_issue>{}],
                        make_buffer_resource(base_ptr + base_offset), v_offset(),
                        s_offset(i_issue), 0, flags[number<i_issue>{}]);
    }
    DEVICE constexpr auto move_slice_window(index_t r_step)
    {
        base_offset += r_step;
        // we only move along 1 dim, so only need to re-calculate flag based on 1 dim.
        constexpr_for<0, n_issue, 1>{}([&](auto i_issue){
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

    static_buffer<index_t, n_issue> flags;
    vector_type<dtype_, ALIGNMENT_ * n_issue> buf;
};
template<typename dtype_, index_t BLOCK_SIZE_, index_t S_PER_BLOCK_, index_t BLOCK_S_WAVES_, index_t S_PER_WAVE_,
                    index_t R_PER_BLOCK_, index_t BLOCK_R_WAVES_, index_t R_PER_WAVE_, index_t ALIGNMENT_, typename mfma_inst_, typename TRAITS_>
struct gld_iterator_s_r_direct_to_reg {
    // TODO: use a to compute size.
    // TODO: currently only support vector load size equal to mfma num_v_a/b. Actually it's easy to change to vector size not equal to mfma...
    using mfma_inst = mfma_inst_;
    static_assert(ALIGNMENT_ == mfma_inst::num_v_a);
    static constexpr index_t n_issue = S_PER_BLOCK_ * R_PER_BLOCK_ / BLOCK_SIZE_ / ALIGNMENT_;   // for single prefetch
    static constexpr index_t issues_r = R_PER_BLOCK_ / mfma_inst::k;
    static constexpr index_t issues_s = S_PER_BLOCK_ / (BLOCK_S_WAVES_ * S_PER_WAVE_);
    static constexpr index_t i_stride_s = BLOCK_S_WAVES_ * S_PER_WAVE_;
    static constexpr index_t i_stride_r = mfma_inst::k;
    static_assert(n_issue == issues_r * issues_s);
    static constexpr index_t N_PREFETCH = TRAITS_::N_PREFETCH;
    static constexpr index_t n_bufs = n_issue * N_PREFETCH;
    static constexpr bool USE_GLD_IF = TRAITS_::USE_GLD_IF;

    template<bool> struct gld_inst_selector;
    template<> struct gld_inst_selector<true> { using type = gld_if<ALIGNMENT_ * 4 / sizeof(dtype_)>; };
    template<> struct gld_inst_selector<false> { using type = gld<ALIGNMENT_ * 4 / sizeof(dtype_)>; };
    using gld_inst_type = typename gld_inst_selector<USE_GLD_IF>::type;
    using gld_vector_type = typename vector_type<dtype_, ALIGNMENT_>::type;

    static constexpr index_t ALIGNMENT = ALIGNMENT_;

    DEVICE constexpr gld_iterator_s_r_direct_to_reg(dtype_ * base_ptr_, index_t s_dim_, index_t r_dim_, index_t s_stride_) {
        base_ptr = base_ptr_;
        base_offset = 0;
        s_dim = s_dim_;
        r_dim = r_dim_;
        s_stride = s_stride_;

        index_t lane_id = threadIdx.x % 64;
        index_t wave_id = threadIdx.x / 64;
        i_s = lane_id % mfma_inst::m + (wave_id / BLOCK_R_WAVES_) * S_PER_WAVE_;
        i_r = lane_id / mfma_inst::m * mfma_inst::num_v_a;

        constexpr_for<0, n_issue, 1>{}([&](auto i_issue){
            constexpr auto i_issue_r = number<i_issue % issues_r>{};
            constexpr auto i_issue_s = number<i_issue / issues_r>{};
            flags[i_issue] = ((i_r + i_issue_r * i_stride_r) < r_dim) && ((i_s + i_issue_s * i_stride_s) < s_dim) ;
        });
    }
    DEVICE constexpr auto v_offset() {
        return (i_s * s_stride + i_r) * sizeof(dtype_);
    }
    DEVICE constexpr auto s_offset(index_t i_issue_s) {
        const index_t stride_per_issue = (BLOCK_S_WAVES_ * S_PER_WAVE_) * s_stride;
        return i_issue_s * stride_per_issue * sizeof(dtype_);
    }
    template<index_t i_prefetch = 0>
    DEVICE constexpr auto load(number<i_prefetch> = number<0>{})
    {
        static_assert(i_prefetch < N_PREFETCH);
        constexpr_for<0, n_issue, 1>{}([&](auto i_issue){
             issue(i_issue, number<i_prefetch>{});
        });
    }
    template<index_t i_issue, index_t i_prefetch = 0>
    DEVICE constexpr auto issue(number<i_issue>, number<i_prefetch> = number<0>{})
    {
        static_assert(i_prefetch < N_PREFETCH);
        static_assert(i_issue < n_issue);
        constexpr auto i_issue_r = number<i_issue % issues_r>{};
        constexpr auto i_issue_s = number<i_issue / issues_r>{};
        gld_inst_type{}(buf[number<i_prefetch>{}].template to_varray<gld_vector_type>()[number<i_issue>{}],
                        make_buffer_resource(base_ptr + base_offset), v_offset(),
                        s_offset(i_issue_s), i_issue_r * i_stride_r * sizeof(dtype_), flags[number<i_issue>{}]);
    }
    template<index_t i_prefetch = 0, bool disable_inline_asm = true>
    DEVICE constexpr auto clear_single_buf(number<i_prefetch> = number<0>{}, bool_const<disable_inline_asm> = bool_const<true>{})
    {
        static_assert(i_prefetch < N_PREFETCH);
        clear(buf[number<i_prefetch>{}], bool_const<disable_inline_asm>{}); // TODO: seems better if let compiler to schedule, when not using setprio
    }
    template<bool disable_inline_asm = true>
    DEVICE constexpr auto clear_buf(bool_const<disable_inline_asm> = bool_const<true>{})
    {
        constexpr_for<0, N_PREFETCH, 1>{}([&](auto i_prefetch){
            clear_single_buf(i_prefetch, bool_const<disable_inline_asm>{});
        });
    }
    DEVICE constexpr auto move_slice_window(index_t r_step)
    {
        base_offset += r_step;
        // we only move along 1 dim, so only need to re-calculate flag based on 1 dim.
        constexpr_for<0, n_issue, 1>{}([&](auto i_issue){
            constexpr auto i_issue_r = number<i_issue % issues_r>{};
            // constexpr auto i_issue_s = number<i_issue / issues_r>{};
            flags[i_issue] = flags[i_issue] & ((i_r + i_issue_r * i_stride_r + base_offset) < r_dim);
        });
    }
    template<index_t i_issue, index_t i_prefetch>
    DEVICE constexpr auto & get()
    {
        static_assert(i_issue < n_issue && i_prefetch < N_PREFETCH);
        return buf[number<i_prefetch>{}].template to_varray<gld_vector_type>()[number<i_issue>{}];
    }
    template<typename v_type, index_t i_issue, index_t i_prefetch>
    DEVICE constexpr auto & getv()
    {
        return buf[number<i_prefetch>{}].template to_varray<v_type>()[number<i_issue>{}];
    }
    dtype_ * base_ptr;
    index_t base_offset;
    index_t s_dim;
    index_t r_dim;
    index_t s_stride;
    index_t i_s;
    index_t i_r;

    static_buffer<index_t, n_issue> flags;
    static_buffer<vector_type<dtype_, n_issue * ALIGNMENT_>, N_PREFETCH> buf;
};
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

    struct args {}; // empty args

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
                            0/*i*/, flag);
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

template<typename a_type_, typename b_type_, typename mfma_inst_, index_t M_PER_BLOCK_, index_t BLOCK_M_WAVES_, index_t M_PER_WAVE_,
                                            index_t N_PER_BLOCK_, index_t BLOCK_N_WAVES_, index_t N_PER_WAVE_,
                                            index_t KPACK_A_, index_t KPACK_B_, bool SKIP_LDS_A_ = true>
struct mfma_mapping_for_sld_oneside {
    DEVICE constexpr auto operator()(index_t & v_offset)
    {
        index_t lane_id = threadIdx.x % 64;
        index_t wave_id = threadIdx.x / 64;
        index_t src_i_m = lane_id % mfma_inst_::m + (wave_id / BLOCK_N_WAVES_) * M_PER_WAVE_;
        index_t src_i_n = lane_id % mfma_inst_::n + (wave_id % BLOCK_N_WAVES_) * N_PER_WAVE_;
        index_t src_i_k = lane_id / mfma_inst_::m;
        if constexpr (!SKIP_LDS_A_)
            v_offset = (src_i_m * KPACK_A_ + src_i_k * (KPACK_A_/*padding*/ + M_PER_BLOCK_ * KPACK_A_)) * sizeof(a_type_);
        else
            v_offset = (src_i_n * KPACK_B_ + src_i_k * (KPACK_B_/*padding*/ + N_PER_BLOCK_ * KPACK_B_)) * sizeof(b_type_);
    }
};

template<typename GLD_A_, typename GLD_B_, index_t M_PER_BLOCK_, index_t N_PER_BLOCK_>
struct gld_clear {
    DEVICE void operator()(GLD_A_ & gld_a, GLD_B_ & gld_b)
    {
        if constexpr (GLD_A_::USE_GLD_IF && GLD_B_::USE_GLD_IF) {
            if constexpr (M_PER_BLOCK_ > N_PER_BLOCK_)
                gld_b.clear_buf();
            else
                gld_a.clear_buf();
        }
    }
};

template<bool X_IS_M_, typename GLD_A_, typename GLD_B_, index_t M_PER_BLOCK_, index_t N_PER_BLOCK_>
struct gld_clear_oneside_lds {
    template<index_t i_prefetch = 0>
    DEVICE void operator()(GLD_A_ & gld_a, GLD_B_ & gld_b, number<i_prefetch> = number<0>{})
    {
        if constexpr (GLD_A_::USE_GLD_IF && GLD_B_::USE_GLD_IF) {
            if constexpr (M_PER_BLOCK_ > N_PER_BLOCK_){
                if constexpr (X_IS_M_) gld_b.clear_buf();
                else gld_b.clear_single_buf(number<i_prefetch>{});
            }
            else {
                if constexpr (X_IS_M_) gld_a.clear_single_buf(number<i_prefetch>{});
                else gld_a.clear_buf();
            }
        }
    }
};

template<index_t issue_id_, index_t num_issues_>
struct gld_issue_info
{
    static constexpr index_t issue_id = issue_id_;
    static constexpr index_t num_issues = num_issues_;
};

template<bool gld_x_first_ = true,
         index_t gld_second_start_distance_ = 0,
         index_t gld_slots_ = 1,
         index_t gld_x_issues_ = 0,
         index_t gld_y_issues_ = 0,
         index_t gld_x_issues_per_group_ = 1,
         index_t gld_y_issues_per_group_ = 1,
         index_t gld_x_issue_distance_ = 0,
         index_t gld_y_issue_distance_ = 0,
         /* for oneside lds */
         index_t k_iter_mod_ = 0>
struct gemm_pipeline_traits {
    static constexpr bool gld_x_first = gld_x_first_;
    static constexpr index_t gld_second_start_distance = gld_second_start_distance_;
    static constexpr index_t gld_slots = gld_slots_;
    static constexpr index_t gld_x_issues = gld_x_issues_;
    static constexpr index_t gld_y_issues = gld_y_issues_;
    static constexpr index_t gld_x_issues_per_group = gld_x_issues_per_group_;
    static constexpr index_t gld_y_issues_per_group = gld_y_issues_per_group_;
    static constexpr index_t gld_x_issue_distance = gld_x_issue_distance_;
    static constexpr index_t gld_y_issue_distance = gld_y_issue_distance_;
    static constexpr index_t k_iter_mod = k_iter_mod_;
    static constexpr bool use_default = gld_x_issues == 0 && gld_y_issues == 0 &&
                                    gld_x_issue_distance == 0 && gld_y_issue_distance == 0;

    template<index_t is_first,  index_t second_start_distance, index_t slots,
                index_t issues, index_t issues_per_group, index_t issue_distance>
    struct traits_util {
        DEVICE constexpr auto operator()(){
            constexpr array<index_t, gld_slots> mask = [](){
                array<index_t, gld_slots> mask_ {};  // value initialize this array, which is zero
                constexpr index_t start_idx = is_first ? 0 : gld_second_start_distance;
                constexpr index_t issue_groups = issues / issues_per_group;
                static_assert(start_idx < slots);
                for(index_t i = 0 ; i < issue_groups; i++){
                    index_t current_idx = start_idx + i * issue_distance;
                    index_t current_mask_value = [&](){
                        index_t tmp = 0;
                        for(index_t j = 0; j < issues_per_group; j++) {
                            index_t issue_bit = i * issues_per_group + j;
                            tmp |= (1 << issue_bit);
                        }
                        return tmp; }();
                    mask_[current_idx] = current_mask_value;
                }
                return mask_;
            }();
            return TO_SEQ(mask);
        }
    };

    static constexpr auto gld_x_mask = traits_util<gld_x_first, gld_second_start_distance, gld_slots,
                                            gld_x_issues, gld_x_issues_per_group, gld_x_issue_distance>{}();
    static constexpr auto gld_y_mask = traits_util<!gld_x_first, gld_second_start_distance, gld_slots,
                                            gld_y_issues, gld_y_issues_per_group, gld_y_issue_distance>{}();

    template<index_t mask>
    static DEVICE constexpr auto decode_mask(number<mask>)
    {
        // if mask is zero, the builtin will report fail in constexpr, which is ambiguous
        static_assert(mask != 0);
        constexpr uint32_t pos = __builtin_ffs(mask);
        constexpr uint32_t clz = __builtin_clz(mask);
        constexpr uint32_t len = (32 - clz) - pos + 1;
        return gld_issue_info<pos - 1, len>{};
    }
    template<typename GLD_MASK, index_t i_slot>
    static DEVICE constexpr auto decode_mask(GLD_MASK, number<i_slot>)
    {
        return decode_mask(number<GLD_MASK::get(number<i_slot>{})>{});
    }

    template<typename MASK_FIRST, typename MASK_SECOND, index_t i_current_slot>
    static DEVICE constexpr auto get_waitcnt_before(MASK_FIRST, MASK_SECOND, number<i_current_slot>)
    {
        index_t cnt = 0;
        static_assert(MASK_FIRST::n_element == MASK_SECOND::n_element);
        constexpr auto len = [&](){
            if constexpr ((i_current_slot + 1) <= MASK_FIRST::n_element) return (i_current_slot + 1);
            else return MASK_FIRST::n_element;}();
        constexpr_for<0, len, 1>{}([&](auto i_slot){
            constexpr auto mask_first = MASK_FIRST::get(number<i_slot>{});
            constexpr auto mask_second = MASK_SECOND::get(number<i_slot>{});
            if constexpr (mask_first != 0) cnt = 0;
            if constexpr (mask_second != 0)
            {
                constexpr auto info = decode_mask(number<mask_second>{});
                cnt += info.num_issues;
            }
        });
        return cnt;
    }
};

template<index_t M_REPEAT_, index_t N_REPEAT_, index_t K_REPEAT_, index_t K_PER_BLOCK_, typename mfma_inst_,
        typename SLD_A_, typename SLD_B_, typename SST_A_, typename SST_B_, typename GLD_A_, typename GLD_B_,
        typename GLD_BUF_CLEAR_, typename MFMA_MAPPING_FOR_SLD_, typename TRAITS_ = gemm_pipeline_traits<>>
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
    using TRAITS = remove_cvref_t<TRAITS_>;

    char * smem;
    index_t lds_b_offset;
    GLD_A & gld_a;
    GLD_B & gld_b;

    DEVICE constexpr gemm_pipeline_flat(
                            char * smem_,
                            index_t lds_b_offset_,
                            GLD_A_ & gld_a_,
                            GLD_B_ & gld_b_) :
                    smem(smem_),
                    lds_b_offset(lds_b_offset_),
                    gld_a(gld_a_),
                    gld_b(gld_b_){}

    template<typename karg_>
    static constexpr bool is_applicable(const karg_ & /*karg*/ ) { return true; }

    template<typename ACC_BUF_>
    DEVICE void operator()(ACC_BUF_ & acc_buf, index_t k_iters)
    {
        if constexpr (TRAITS::gld_x_first) {
            gld_a.clear_buf();
            gld_a(); gld_a.move_slice_window(K_PER_BLOCK_);
            gld_b.clear_buf();
            gld_b(); gld_b.move_slice_window(K_PER_BLOCK_);
        } else {
            gld_b.clear_buf();
            gld_b(); gld_b.move_slice_window(K_PER_BLOCK_);
            gld_a.clear_buf();
            gld_a(); gld_a.move_slice_window(K_PER_BLOCK_);
        }

        index_t v_offset_a, v_offset_b;
        MFMA_MAPPING_FOR_SLD{}(v_offset_a, v_offset_b);
        auto sst_a = SST_A{smem, 0};
        auto sst_b = SST_B{smem, lds_b_offset};
        auto sld_a = SLD_A{smem, v_offset_a, 0};
        auto sld_b = SLD_B{smem, v_offset_b, lds_b_offset};

        clear(acc_buf);
        if constexpr (TRAITS::gld_x_first) {
            gld_fence(gld_b.n_issue);
            sst_a(gld_a.buf);
            gld_fence(0);
            sst_b(gld_b.buf);
        } else {
            gld_fence(gld_a.n_issue);
            sst_b(gld_b.buf);
            gld_fence(0);
            sst_a(gld_a.buf);
        }
        GLD_BUF_CLEAR{}(gld_a, gld_b);

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

        constexpr auto total_repeats = K_REPEAT_ * M_REPEAT_ * N_REPEAT_;
        constexpr auto last_gld_cnt = [&](){
            if constexpr (TRAITS_::gld_x_first)
                return TRAITS_::get_waitcnt_before(TRAITS_::gld_x_mask, TRAITS_::gld_y_mask, number<total_repeats - 1>{});
            else
                return TRAITS_::get_waitcnt_before(TRAITS_::gld_y_mask, TRAITS_::gld_x_mask, number<total_repeats - 1>{});
        }();
        // let everything into 1 dim, easy to control the sld/gld/sst slot
        constexpr_for<0, total_repeats, 1>{}([&](auto i_3d){
            constexpr auto i_k = number<i_3d / ( M_REPEAT_ * N_REPEAT_)>{};
            constexpr auto i_2d = number<i_3d % ( M_REPEAT_ * N_REPEAT_)>{};
            constexpr auto i_m = number<i_2d / N_REPEAT_>{};
            constexpr auto i_n = number<i_2d % N_REPEAT_>{};
            constexpr auto i_next_n = number<(i_2d + 1) % N_REPEAT_>{};

            constexpr auto need_sld_a_first = bool_const<i_2d == 0>{};
            constexpr auto need_sld_b_first = bool_const<i_2d == 0>{};
            constexpr auto need_sld_b_prefetch = bool_const<!(i_m == M_REPEAT_ - 1 && i_n == N_REPEAT_ - 1)>{};
            constexpr auto need_gld_0 = bool_const<i_k == 0 && i_m == 0 && i_n == 0 && is_hot_loop>{};
            constexpr auto need_gld_1 = bool_const<i_k == 0 && (i_m == M_REPEAT_ - 1 && i_n == N_REPEAT_ - 1) && is_hot_loop>{};
            constexpr auto need_gld_a_default = [&](){if constexpr (TRAITS_::gld_x_first) return need_gld_0; else return need_gld_1; }();
            constexpr auto need_gld_b_default = [&](){if constexpr (TRAITS_::gld_x_first) return need_gld_1; else return need_gld_0; }();
            constexpr auto need_wait_gld_sst = bool_const<(i_3d == total_repeats - 1) && is_hot_loop>{};

            constexpr index_t gld_x_mask = [&]{ if constexpr (i_3d < TRAITS_::gld_x_mask.n_element)
                                                return TRAITS_::gld_x_mask.get(i_3d); else return 0;}();
            constexpr index_t gld_y_mask = [&]{ if constexpr (i_3d < TRAITS_::gld_y_mask.n_element)
                                                return TRAITS_::gld_y_mask.get(i_3d); else return 0;}();

            auto try_gld_x = [&]{
                if constexpr (TRAITS_::use_default) { if constexpr (need_gld_a_default)
                        {   gld_iter_a();    }}
                else { if constexpr (gld_x_mask && is_hot_loop) {
                        constexpr auto gld_x_info = TRAITS_::decode_mask(number<gld_x_mask>{});
                        constexpr_for<0, gld_x_info.num_issues, 1>{}([&](auto i_issue){
                            gld_iter_a.issue(number<gld_x_info.issue_id + i_issue>{});
                    });
                }}};
            auto try_gld_y = [&]{
                if constexpr (TRAITS_::use_default) { if constexpr (need_gld_b_default)
                        {   gld_iter_b();   }}
                else { if constexpr (gld_y_mask && is_hot_loop) {
                        constexpr auto gld_y_info = TRAITS_::decode_mask(number<gld_y_mask>{});
                        constexpr_for<0, gld_y_info.num_issues, 1>{}([&](auto i_issue){
                            gld_iter_b.issue(number<gld_y_info.issue_id + i_issue>{});
                    });
                }}};

            if constexpr (TRAITS_::gld_x_first) { try_gld_x(); try_gld_y();}
            else                                { try_gld_y(); try_gld_x();}

            if constexpr(i_3d == 0) {
                // it is good to self contains this barrier inside constexpr for
                sst_fence(0); wave_barrier();
            }

            // conditionally do sld
            if constexpr(need_sld_a_first)      sld_iter_a.template load_all<i_k>();
            if constexpr(need_sld_b_first)      sld_iter_b.template load<0, i_k, 0>();
            if constexpr(need_sld_b_prefetch)   sld_iter_b.template load<i_next_n, i_k, i_next_n % 2>();

            // TODO: this may have bugs if repeat if not large (?)
            auto sld_fence_cnt = [&](){
                if constexpr(need_sld_b_prefetch)
                    return sld_iter_b.n_issue;
                return index_t(0); }();
            sld_fence(sld_fence_cnt);

            // conditionally do sst, should be the last one
            if constexpr(need_wait_gld_sst) {
                gld_iter_a.move_slice_window(K_PER_BLOCK_);
                gld_iter_b.move_slice_window(K_PER_BLOCK_);
                wave_barrier();
                if constexpr (TRAITS_::gld_x_first) {
                    gld_fence(last_gld_cnt);
                    sst_iter_a(gld_iter_a.buf);
                    gld_fence(0);
                    sst_iter_b(gld_iter_b.buf);
                } else {
                    gld_fence(last_gld_cnt);
                    sst_iter_b(gld_iter_b.buf);
                    gld_fence(0);
                    sst_iter_a(gld_iter_a.buf);
                }
                GLD_BUF_CLEAR{}(gld_iter_a, gld_iter_b);
            }
            mfma(sld_iter_a.template get<i_m>(), sld_iter_b.template get<i_n % 2>(),
                            acc_buf.template to_varray<acc_t>()[number<i_m * N_REPEAT_ + i_n>{}], bool_const<true>{});
        });
    }
};

template<bool X_IS_M_, index_t M_REPEAT_, index_t N_REPEAT_, index_t K_REPEAT_, index_t K_PER_BLOCK_, typename mfma_inst_,
            typename SLD_Y_, typename SST_Y_, typename GLD_A_, typename GLD_B_, typename GLD_BUF_CLEAR_, typename MFMA_MAPPING_FOR_SLD_,
               typename TRAITS_ = gemm_pipeline_traits<>>
struct gemm_pipeline_oneside_lds {
    // one side of A/B use LDS, the other side just direct store data into register.
    // we use X for the side not using LDS, Y for the side using LDS.
    // k_iter_mod means n_span  k_iters

    static constexpr bool X_IS_M = X_IS_M_;

    static constexpr index_t X_REPEAT = X_IS_M ? M_REPEAT_ : N_REPEAT_;
    static constexpr index_t Y_REPEAT = X_IS_M ? N_REPEAT_ : M_REPEAT_;
    static constexpr index_t K_REPEAT = K_REPEAT_;
    static constexpr index_t K_PER_BLOCK = K_PER_BLOCK_;

    using mfma_inst = remove_cvref_t<mfma_inst_>;
    using SLD_Y = remove_cvref_t<SLD_Y_>;
    using SST_Y = remove_cvref_t<SST_Y_>;
    using GLD_X = std::conditional_t<X_IS_M, GLD_A_, GLD_B_>;
    using GLD_Y = std::conditional_t<X_IS_M, GLD_B_, GLD_A_>;

    using GLD_BUF_CLEAR = remove_cvref_t<GLD_BUF_CLEAR_>;
    using MFMA_MAPPING_FOR_SLD = remove_cvref_t<MFMA_MAPPING_FOR_SLD_>;
    using TRAITS = remove_cvref_t<TRAITS_>;
    static constexpr index_t k_iter_mod = TRAITS::k_iter_mod;

    char * smem;
    index_t lds_y_offset;
    GLD_X & gld_x;
    GLD_Y & gld_y;

    template<typename karg_>
    static constexpr bool is_applicable(const karg_ & karg)
    {
        index_t k_iters = (karg.k + K_PER_BLOCK - 1) / K_PER_BLOCK;
        return k_iter_mod == k_iters % GLD_X::N_PREFETCH;
    }

    DEVICE constexpr gemm_pipeline_oneside_lds(
                            char * smem_,   // for y use
                            index_t lds_y_offset_,
                            GLD_X & gld_iter_x_,
                            GLD_Y & gld_iter_y_) :
                    smem(smem_),
                    lds_y_offset(lds_y_offset_),
                    gld_x(gld_iter_x_),
                    gld_y(gld_iter_y_) {}

    template<typename ACC_BUF_>
    DEVICE void operator()(ACC_BUF_ & acc_buf, index_t k_iters)
    {
        if constexpr (TRAITS::gld_x_first) {
            gld_x.clear_buf();
            gld_x.load(number<0>{}); gld_x.move_slice_window(K_PER_BLOCK_);
            gld_y.clear_buf();
            gld_y(); gld_y.move_slice_window(K_PER_BLOCK_);
        }
        else {
            gld_y.clear_buf();
            gld_y(); gld_y.move_slice_window(K_PER_BLOCK_);
            gld_x.clear_buf();
            gld_x.load(number<0>{}); gld_x.move_slice_window(K_PER_BLOCK_);
        }

        index_t v_offset_y;
        MFMA_MAPPING_FOR_SLD{}(v_offset_y);
        auto sst_y = SST_Y{smem};
        auto sld_y = SLD_Y{smem, v_offset_y};

        clear(acc_buf);
        if constexpr (TRAITS_::gld_x_first) gld_fence(0);
        else gld_fence(gld_x.n_issue);
        sst_y(gld_y.buf);
        GLD_BUF_CLEAR{}(gld_x, gld_y);
#if 1
        // TODO: have to specialize 2 versions, due to compiler limitation
        if constexpr (GLD_X::N_PREFETCH == 2 && k_iter_mod == 0) {
            for(index_t i_k = 2; i_k < k_iters; i_k+= 2) {
                gemm_(acc_buf, gld_x, number<0>{}, number<1>{}, gld_y, sld_y, sst_y, bool_const<true>{});
                gemm_(acc_buf, gld_x, number<1>{}, number<0>{}, gld_y, sld_y, sst_y, bool_const<true>{});
            }
            gemm_(acc_buf, gld_x, number<0>{}, number<1>{}, gld_y, sld_y, sst_y, bool_const<true>{});
            gemm_(acc_buf, gld_x, number<1>{}, number<0>{}, gld_y, sld_y, sst_y, bool_const<false>{});
        }
        if constexpr (GLD_X::N_PREFETCH == 2 && k_iter_mod == 1) {
            for(index_t i_k = 1; i_k < k_iters; i_k+= 2) {
                gemm_(acc_buf, gld_x, number<0>{}, number<1>{}, gld_y, sld_y, sst_y, bool_const<true>{});
                gemm_(acc_buf, gld_x, number<1>{}, number<0>{}, gld_y, sld_y, sst_y, bool_const<true>{});
            }
            gemm_(acc_buf, gld_x, number<0>{}, number<1>{}, gld_y, sld_y, sst_y, bool_const<false>{});
        }
#else
        index_t i_k = 1;
        while(i_k < k_iters) {
            gemm_(acc_buf, gld_x, number<0>{}, number<1>{}, gld_y, sld_y, sst_y, bool_const<true>{});
            i_k++;
            if(i_k >= k_iters) break;
            gemm_(acc_buf, gld_x, number<1>{}, number<0>{}, gld_y, sld_y, sst_y, bool_const<true>{});
            i_k++;
        }
        // tail
        if(k_iters % 2 == 1)
             gemm_(acc_buf, gld_x, number<0>{}, number<1>{}, gld_y, sld_y, sst_y, bool_const<false>{});
        else
             gemm_(acc_buf, gld_x, number<1>{}, number<0>{}, gld_y, sld_y, sst_y, bool_const<false>{});
#endif
    }

    template<typename ACC_BUF_, index_t i_curr_x_buf, index_t i_next_x_buf, typename HOT_LOOP_ = bool_const<true>>
    DEVICE void gemm_(ACC_BUF_ & acc_buf, GLD_X & gld_iter_x, number<i_curr_x_buf>, number<i_next_x_buf>, GLD_Y & gld_iter_y, SLD_Y & sld_iter_y, SST_Y & sst_iter_y,
                        HOT_LOOP_ is_hot_loop = bool_const<true>{})
    {
        using acc_type = typename ACC_BUF_::d1_t;
        using acc_t = typename vector_type<acc_type, mfma_inst::num_v_c>::type;
        auto mfma = mfma_inst{};

        // let everything into 1 dim, easy to control the sld/gld/sst slot
        constexpr index_t total_repeats =  K_REPEAT * X_REPEAT * Y_REPEAT;
        constexpr auto y_x_last_gld_cnt = TRAITS_::get_waitcnt_before(TRAITS_::gld_y_mask, TRAITS_::gld_x_mask, number<total_repeats - 1>{});
        // if last y_x is zero, means in y->x order, some of y issue will be later than x issue, hence y->x order is partial broken
        if constexpr (y_x_last_gld_cnt == 0) {gld_fence(0);}

        constexpr_for<0, total_repeats, 1>{}([&](auto i_3d){
            constexpr auto i_k = number<i_3d / ( X_REPEAT * Y_REPEAT)>{};
            constexpr auto i_2d = number<i_3d % ( X_REPEAT * Y_REPEAT)>{};
            constexpr auto i_x = number<i_2d / Y_REPEAT>{};
            constexpr auto i_y = number<i_2d % Y_REPEAT>{};
            constexpr auto i_next_k = number<(i_3d + 1) / ( X_REPEAT * Y_REPEAT)>{};
            constexpr auto i_next_2d = number<(i_3d + 1) % ( X_REPEAT * Y_REPEAT)>{};
            constexpr auto i_next_y = number<i_next_2d % Y_REPEAT>{};

            constexpr auto need_sld_y_prefetch = bool_const<i_3d != total_repeats - 1>{};

            constexpr auto need_gld_0 = bool_const<i_k == 0 && i_x == 0 && i_y == 0 && is_hot_loop>{};
            constexpr auto need_gld_1 = bool_const<i_k == 0 && (i_x == X_REPEAT - 1 && i_y == Y_REPEAT - 1) && is_hot_loop>{};
            constexpr auto need_gld_x_default = [&](){if constexpr (TRAITS_::gld_x_first) return need_gld_0; else return need_gld_1; }();
            constexpr auto need_gld_y_default = [&](){if constexpr (TRAITS_::gld_x_first) return need_gld_1; else return need_gld_0; }();

            constexpr auto need_wait_gld_sst = bool_const<(i_3d == total_repeats - 1) && is_hot_loop>{};

            constexpr index_t gld_x_mask = [&]{ if constexpr (i_3d < TRAITS_::gld_x_mask.n_element)
                                                return TRAITS_::gld_x_mask.get(i_3d); else return 0;}();
            constexpr index_t gld_y_mask = [&]{ if constexpr (i_3d < TRAITS_::gld_y_mask.n_element)
                                                return TRAITS_::gld_y_mask.get(i_3d); else return 0;}();
            auto try_gld_x = [&]{
                if constexpr (TRAITS_::use_default) { if constexpr (need_gld_x_default)
                        {   gld_iter_x.load(number<i_next_x_buf>{});    }}
                else { if constexpr (gld_x_mask && is_hot_loop) {
                        constexpr auto gld_x_info = TRAITS_::decode_mask(number<gld_x_mask>{});
                        constexpr_for<0, gld_x_info.num_issues, 1>{}([&](auto i_issue){
                            gld_iter_x.issue(number<gld_x_info.issue_id + i_issue>{}, number<i_next_x_buf>{});
                    });
                }}};
            auto try_gld_y = [&]{
                if constexpr (TRAITS_::use_default) { if constexpr (need_gld_y_default)
                        {   gld_iter_y();   }}
                else { if constexpr (gld_y_mask && is_hot_loop) {
                        constexpr auto gld_y_info = TRAITS_::decode_mask(number<gld_y_mask>{});
                        constexpr_for<0, gld_y_info.num_issues, 1>{}([&](auto i_issue){
                            gld_iter_y.issue(number<gld_y_info.issue_id + i_issue>{});
                    });
                }}};

            if constexpr (TRAITS_::gld_x_first) { try_gld_x(); try_gld_y();}
            else                                { try_gld_y(); try_gld_x();}

            if constexpr(i_3d == 0) {
                // it is good to self contains this barrier inside constexpr for
                sst_fence(0); wave_barrier();
                sld_iter_y.template load<0, i_k, 0>();
            }

            if constexpr(need_sld_y_prefetch)
                sld_iter_y.template load<i_next_y, i_next_k, i_next_y % 2>();

            if constexpr(i_3d == 0) {
                // after first y issue, then wait for previous x, before entering main loop
                if constexpr (!TRAITS_::gld_x_first && y_x_last_gld_cnt != 0) gld_fence(TRAITS_::gld_y_issues_per_group);
            }

            // TODO: this may have bugs if repeat if not large (?)
            auto sld_fence_cnt = [&](){
                if constexpr(need_sld_y_prefetch)
                    return sld_iter_y.n_issue;
                return index_t(0); }();
            sld_fence(sld_fence_cnt);

            if constexpr(need_wait_gld_sst) {
                gld_iter_y.move_slice_window(K_PER_BLOCK_);
                gld_iter_x.move_slice_window(K_PER_BLOCK_);
                wave_barrier();
                if constexpr (TRAITS_::gld_x_first) gld_fence(0);
                else gld_fence(y_x_last_gld_cnt);
                sst_iter_y(gld_iter_y.buf);
                GLD_BUF_CLEAR{}(gld_iter_x, gld_iter_y);
            }
            mfma(gld_iter_x.template get<i_x * GLD_X::issues_r + i_k, i_curr_x_buf>(), sld_iter_y.template get<i_y % 2>(),
                            acc_buf.template to_varray<acc_t>()[number<i_x * Y_REPEAT + i_y>{}], bool_const<true>{});
        });
    }
};

template<typename DATA_TYPES_, typename BLOCK_TILE_, typename BLOCK_WAVES_, typename WAVE_TILE_, typename ALIGNMENT_,
    typename TILE_SCHEDULER_, typename GLD_TRAITS_, typename PIPELINE_TRAIT_, typename EPILOGUE_>
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

    using TILE_SCHEDULER = remove_cvref_t<TILE_SCHEDULER_>;
    using GLD_TRAIT_A = remove_cvref_t<decltype(GLD_TRAITS_{}.template get<0>())>;
    using GLD_TRAIT_B = remove_cvref_t<decltype(GLD_TRAITS_{}.template get<1>())>;
    using PIPELINE_TRAIT = remove_cvref_t<PIPELINE_TRAIT_>;
    using EPILOGUE = remove_cvref_t<EPILOGUE_>;
    static constexpr bool one_side_lds = GLD_TRAIT_A::BYPASS_LDS ^ GLD_TRAIT_B::BYPASS_LDS;

    // for simplicity, we just use the same vector size as k_pack
    static constexpr index_t KPACK_A = ALIGNMENT_A;
    static constexpr index_t KPACK_B = ALIGNMENT_B;

    using a_type = remove_cvref_t<decltype(DATA_TYPES_{}.template get<0>())>;
    using b_type = remove_cvref_t<decltype(DATA_TYPES_{}.template get<1>())>;
    using c_type = remove_cvref_t<decltype(DATA_TYPES_{}.template get<2>())>;
    using acc_type = remove_cvref_t<decltype(DATA_TYPES_{}.template get<3>())>;

    using mfma_inst = typename mfma_selector<a_type, b_type, acc_type, M_PER_WAVE, N_PER_WAVE, K_PER_WAVE>::type;

    using sst_iter_a_type = sst_iterator_r0_s_r1<a_type, BLOCK_SIZE, M_PER_BLOCK, K_PER_BLOCK, KPACK_A>;
    using sst_iter_b_type = sst_iterator_r0_s_r1<b_type, BLOCK_SIZE, N_PER_BLOCK, K_PER_BLOCK, KPACK_B>;

    using sld_iter_a_type = sld_iterator_r0_s_r1<a_type, M_PER_BLOCK, BLOCK_M_WAVES, M_PER_WAVE, KPACK_A, mfma_inst::k / KPACK_A, true>;
    using sld_iter_b_type = sld_iterator_r0_s_r1<b_type, N_PER_BLOCK, BLOCK_N_WAVES, N_PER_WAVE, KPACK_B, mfma_inst::k / KPACK_B, false>;

    using gld_iter_a_type = gld_iterator_s_r<a_type, BLOCK_SIZE, M_PER_BLOCK, K_PER_BLOCK, ALIGNMENT_A, GLD_TRAIT_A>;
    using gld_iter_b_type = gld_iterator_s_r<b_type, BLOCK_SIZE, N_PER_BLOCK, K_PER_BLOCK, ALIGNMENT_B, GLD_TRAIT_B>;

    using gld_iter_a_dr_type = gld_iterator_s_r_direct_to_reg<a_type, BLOCK_SIZE, M_PER_BLOCK, BLOCK_M_WAVES, M_PER_WAVE,
                                    K_PER_BLOCK, BLOCK_K_WAVES, K_PER_WAVE, ALIGNMENT_A, mfma_inst, GLD_TRAIT_A>;
    using gld_iter_b_dr_type = gld_iterator_s_r_direct_to_reg<b_type, BLOCK_SIZE, N_PER_BLOCK, BLOCK_N_WAVES, N_PER_WAVE,
                                    K_PER_BLOCK, BLOCK_K_WAVES, K_PER_WAVE, ALIGNMENT_B, mfma_inst, GLD_TRAIT_B>;

    static constexpr bool X_IS_M = one_side_lds && GLD_TRAIT_A::BYPASS_LDS;
    using gld_a_t = std::conditional_t<one_side_lds, std::conditional_t<X_IS_M, gld_iter_a_dr_type, gld_iter_a_type>, gld_iter_a_type>;
    using gld_b_t = std::conditional_t<one_side_lds, std::conditional_t<X_IS_M, gld_iter_b_type, gld_iter_b_dr_type>, gld_iter_b_type>;
    
    using gld_clear_type = gld_clear<gld_iter_a_type, gld_iter_b_type, M_PER_BLOCK, N_PER_BLOCK>;

    using mfma_mapping_for_sld_type = mfma_mapping_for_sld<a_type, b_type, mfma_inst, M_PER_BLOCK, BLOCK_M_WAVES, M_PER_WAVE,
                                                    N_PER_BLOCK, BLOCK_N_WAVES, N_PER_WAVE, KPACK_A, KPACK_B>;

    using gemm_pipepine_type = gemm_pipeline_flat<WAVE_M_REPEAT, WAVE_N_REPEAT, WAVE_K_REPEAT, K_PER_BLOCK, mfma_inst,
                sld_iter_a_type, sld_iter_b_type, sst_iter_a_type, sst_iter_b_type,
                gld_iter_a_type, gld_iter_b_type, gld_clear_type, mfma_mapping_for_sld_type, PIPELINE_TRAIT>;

    using sld_y_type = std::conditional_t<X_IS_M, sld_iter_b_type, sld_iter_a_type>;
    using sst_y_type = std::conditional_t<X_IS_M, sst_iter_b_type, sst_iter_a_type>;

    using gld_clear_oneside_lds_type = gld_clear_oneside_lds<X_IS_M, gld_a_t, gld_b_t, M_PER_BLOCK, N_PER_BLOCK>;
    using mfma_mapping_oneside_lds_for_sld_type = mfma_mapping_for_sld_oneside<a_type, b_type, mfma_inst, M_PER_BLOCK, BLOCK_M_WAVES, M_PER_WAVE,
                                       N_PER_BLOCK, BLOCK_N_WAVES, N_PER_WAVE, KPACK_A, KPACK_B, X_IS_M>;
    using gemm_pipeline_oneside_lds_type = gemm_pipeline_oneside_lds<X_IS_M, WAVE_M_REPEAT, WAVE_N_REPEAT, WAVE_K_REPEAT, K_PER_BLOCK, mfma_inst,
                   sld_y_type, sst_y_type, gld_a_t, gld_b_t, gld_clear_oneside_lds_type, mfma_mapping_oneside_lds_for_sld_type, PIPELINE_TRAIT>;

    using gemm_pipeline_t = std::conditional_t<one_side_lds, gemm_pipeline_oneside_lds_type, gemm_pipepine_type>;

    // NOTE: use multi inheritance to pass karg to any fused operators.
    // If fused operator has zero bytes, this technique will take no extra space
    struct args : public EPILOGUE::args
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
    static args make_karg(void * ptr_a,
        void * ptr_b,
        void * ptr_c,
        index_t m,
        index_t n,
        index_t k,
        index_t lda,    // in unit of pixel
        index_t ldb,
        index_t ldc)
    {
        return args{{}, ptr_a, ptr_b, ptr_c, m, n, k, lda, ldb, ldc};
    }

    static bool is_applicable(args karg)
    {
        if((karg.k % ALIGNMENT_A != 0) || (karg.k % ALIGNMENT_B != 0) || (karg.n % ALIGNMENT_C != 0))
            return false;
        return gemm_pipeline_t::is_applicable(karg);
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
        constexpr index_t ss_a = one_side_lds && X_IS_M  ? 0 : smem_size_a();
        constexpr index_t ss_b = one_side_lds && !X_IS_M ? 0 : smem_size_b();
        return MAX(ss_a + ss_b, smem_size_shuffle());
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
        auto block_dist = TILE_SCHEDULER{}(karg.m, karg.n, number<M_PER_BLOCK>{}, number<N_PER_BLOCK>{});
        index_t block_i_m = block_dist.template get<0>();
        index_t block_i_n = block_dist.template get<1>();

        a_type * ptr_a = reinterpret_cast<a_type*>(karg.ptr_a) + block_i_m * karg.lda;
        b_type * ptr_b = reinterpret_cast<b_type*>(karg.ptr_b) + block_i_n * karg.ldb;
        c_type * ptr_c = reinterpret_cast<c_type*>(karg.ptr_c) + block_i_m * karg.ldc + block_i_n;

        auto k_iters = (karg.k + K_PER_BLOCK - 1) / K_PER_BLOCK;
        auto gld_a = gld_a_t{ptr_a, karg.m - block_i_m, karg.k, karg.lda};
        auto gld_b = gld_b_t{ptr_b, karg.n - block_i_n, karg.k, karg.ldb};

        vector_type<acc_type, WAVE_M_REPEAT * WAVE_N_REPEAT * mfma_inst::num_v_c> acc_buf;

        index_t lds_offset = [&](){if constexpr (one_side_lds) return 0 ; else return smem_size_a();}();

        auto gemm = gemm_pipeline_t{smem, lds_offset, gld_a, gld_b};
        gemm(acc_buf, k_iters);
        auto epilogue = EPILOGUE{ptr_c, karg.m - block_i_m, karg.n - block_i_n, karg.ldc, smem};
        // write out
        sched_barrier();  // in case mfma dest has raw harzard
        epilogue(acc_buf);
    }
};
