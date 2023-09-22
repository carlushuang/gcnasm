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

#define USE_GLD_IF 0
template<typename dtype_, index_t BLOCK_SIZE_, index_t S_PER_BLOCK_, index_t R_PER_BLOCK_, index_t ALIGNMENT_>
struct gld_iterator_s_r {
    static constexpr index_t issues = S_PER_BLOCK_ * R_PER_BLOCK_ / BLOCK_SIZE_ / ALIGNMENT_;
    static constexpr index_t n_bufs = issues;
#if USE_GLD_IF
    using gld_inst_type = gld_if<ALIGNMENT_ * 4 / sizeof(dtype_)>;
#else
    using gld_inst_type = gld<ALIGNMENT_ * 4 / sizeof(dtype_)>;
#endif
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
        return issue * stride_per_issue * sizeof(dtype_);;
    }
    DEVICE constexpr auto clear_buf()
    {
        clear(buf);
    }
    DEVICE constexpr auto operator()()
    {
        constexpr_for<0, issues, 1>{}([&](auto i_issue){
#if USE_GLD_IF
            gld_inst_type{}(buf.template to_varray<gld_vector_type>()[i_issue],
                            make_buffer_resource(base_ptr + base_offset), v_offset(),
                            s_offset(i_issue), 0, flags[i_issue]);
#else
            gld_inst_type{}(buf.template to_varray<gld_vector_type>()[i_issue],
                            make_buffer_resource(base_ptr + base_offset), v_offset(),
                            s_offset(i_issue), 0);
#endif
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

template<typename DATA_TYPES_, typename BLOCK_TILE_, typename BLOCK_WAVES_, typename WAVE_TILE_>
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
    static constexpr index_t ALIGNMENT_A = 8;
    static constexpr index_t ALIGNMENT_B = 8;
    static constexpr index_t ALIGNMENT_C = 8;

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
        auto gld_a = gld_iterator_s_r<a_type, BLOCK_SIZE, M_PER_BLOCK, K_PER_BLOCK, ALIGNMENT_A>{ptr_a, karg.m - block_i_m, karg.k, karg.lda};
        auto gld_b = gld_iterator_s_r<b_type, BLOCK_SIZE, N_PER_BLOCK, K_PER_BLOCK, ALIGNMENT_B>{ptr_b, karg.n - block_i_n, karg.k, karg.ldb};

        auto sst_a = sst_iterator_r0_s_r1<a_type, BLOCK_SIZE, M_PER_BLOCK, K_PER_BLOCK, KPACK_A>{smem};
        auto sst_b = sst_iterator_r0_s_r1<b_type, BLOCK_SIZE, N_PER_BLOCK, K_PER_BLOCK, KPACK_B>{smem, smem_size_a()};

        using sld_iter_a_type = sld_iterator_r0_s_r1<a_type, M_PER_BLOCK, BLOCK_M_WAVES, M_PER_WAVE, KPACK_A, mfma_inst::k / KPACK_A, true>;
        using sld_iter_b_type = sld_iterator_r0_s_r1<b_type, N_PER_BLOCK, BLOCK_N_WAVES, N_PER_WAVE, KPACK_B, mfma_inst::k / KPACK_B, false>;

        vector_type<acc_type, WAVE_M_REPEAT * WAVE_N_REPEAT * mfma_inst::num_v_c> acc_buf;
        using acc_t = typename vector_type<acc_type, mfma_inst::num_v_c>::type;

        auto gemm = [&](auto & sld_iter_a, auto & sld_iter_b, auto & sst_iter_a, auto & sst_iter_b, auto & gld_iter_a, auto & gld_iter_b, auto & gld_buf_clear, auto is_hot_loop = bool_const<true>{})
        {
#if 0
            // a use all buffer, b at most use 2 buffers
            auto mfma = mfma_inst{};
            if constexpr (is_hot_loop)
                gld_a();
            constexpr_for<0, WAVE_K_REPEAT, 1>{}([&](auto i_k){
                sld_iter_a.template load_all<i_k>();
                sld_iter_b.template load<0, i_k, 0>();
                constexpr_for<0, WAVE_M_REPEAT * WAVE_N_REPEAT - 1, 1>{}([&](auto i_2d){
                    auto i_m = number<i_2d / WAVE_N_REPEAT>{};
                    auto i_n = number<i_2d % WAVE_N_REPEAT>{};
                    auto i_next_n = number<(i_2d + 1) % WAVE_N_REPEAT>{};
                    sld_iter_b.template load<i_next_n, i_k, i_next_n % 2>();
                    sld_fence(sld_iter_b.issues);
                    mfma(sld_iter_a.template get<i_m>(), sld_iter_b.template get<i_n % 2>(),
                                 acc_buf.template to_varray<acc_t>()[number<i_m * WAVE_N_REPEAT + i_n>{}], bool_const<true>{});
                });
                sld_fence(0);
                if constexpr (i_k == 0 && is_hot_loop) {
                    gld_b();
                }
                mfma(sld_iter_a.template get<WAVE_M_REPEAT - 1>(), sld_iter_b.template get<(WAVE_N_REPEAT - 1) % 2>(),
                             acc_buf.template to_varray<acc_t>()[number<(WAVE_M_REPEAT - 1) * WAVE_N_REPEAT + WAVE_N_REPEAT - 1>{}], bool_const<true>{});
            });
            if constexpr (is_hot_loop) {
                gld_a.move_slice_window(K_PER_BLOCK);
                gld_b.move_slice_window(K_PER_BLOCK);
                wave_barrier();
                gld_fence(gld_b.issues);
                sst_a(gld_a.buf);
                gld_fence(0);
                sst_b(gld_b.buf);
                gld_buf_clear();
                sst_fence(0); wave_barrier();
            }
#else
            auto mfma = mfma_inst{};
            // let everything into 1 dim, easy to control the sld/gld/sst slot
            constexpr_for<0, WAVE_K_REPEAT * WAVE_M_REPEAT * WAVE_N_REPEAT, 1>{}([&](auto i_3d){
                constexpr auto i_k = number<i_3d / ( WAVE_M_REPEAT * WAVE_N_REPEAT)>{};
                constexpr auto i_2d = number<i_3d % ( WAVE_M_REPEAT * WAVE_N_REPEAT)>{};
                constexpr auto i_m = number<i_2d / WAVE_N_REPEAT>{};
                constexpr auto i_n = number<i_2d % WAVE_N_REPEAT>{};
                constexpr auto i_next_n = number<(i_2d + 1) % WAVE_N_REPEAT>{};

                constexpr auto need_sld_a = bool_const<i_m == 0 && i_n == 0>{};
                constexpr auto need_sld_b_first = bool_const<i_m == 0 && i_n == 0>{};
                constexpr auto need_sld_b_prefetch = bool_const<!(i_m == WAVE_M_REPEAT - 1 && i_n == WAVE_N_REPEAT - 1)>{};
                constexpr auto need_gld_a = bool_const<i_k == 0 && i_m == 0 && i_n == 0 && is_hot_loop>{};
                constexpr auto need_gld_b = bool_const<i_k == 0 && (i_m == WAVE_M_REPEAT - 1 && i_n == WAVE_N_REPEAT - 1) && is_hot_loop>{};
                constexpr auto need_wait_gld_sst = bool_const<i_3d == WAVE_K_REPEAT * WAVE_M_REPEAT * WAVE_N_REPEAT - 1>{};

                // conditionally do gld
                if constexpr(need_gld_a)
                    gld_iter_a();
                if constexpr(need_gld_b)
                    gld_iter_b();

                // conditionally do sld
                if constexpr(need_sld_a)
                    sld_iter_a.template load_all<i_k>();

                if constexpr(need_sld_b_first)
                    sld_iter_b.template load<0, i_k, 0>();

                if constexpr(need_sld_b_prefetch)
                    sld_iter_b.template load<i_next_n, i_k, i_next_n % 2>();

                // conditionally do sst, should be the last one
                if constexpr(need_wait_gld_sst) {
                    gld_a.move_slice_window(K_PER_BLOCK);
                    gld_b.move_slice_window(K_PER_BLOCK);
                    wave_barrier();
                    gld_fence(gld_b.issues);
                    sst_a(gld_a.buf);
                    gld_fence(0);
                    sst_b(gld_b.buf);
                }

                auto sld_fence_cnt = [&](){
                    if constexpr(need_sld_b_prefetch) return sld_iter_b.issues;
                    else return 0; }();
                sld_fence(sld_fence_cnt);
                mfma(sld_iter_a.template get<i_m>(), sld_iter_b.template get<i_n % 2>(),
                                acc_buf.template to_varray<acc_t>()[number<i_m * WAVE_N_REPEAT + i_n>{}], bool_const<true>{});
            });
            if constexpr (is_hot_loop) {
                gld_buf_clear();
                sst_fence(0); wave_barrier();
            }
#endif
        };
        auto mfma_src_dist_offset = [&](index_t & v_offset_a, index_t & v_offset_b){
            index_t lane_id = threadIdx.x % 64;
            index_t wave_id = threadIdx.x / 64;
            index_t src_i_m = lane_id % mfma_inst::m + (wave_id / BLOCK_N_WAVES) * M_PER_WAVE;
            index_t src_i_n = lane_id % mfma_inst::n + (wave_id % BLOCK_N_WAVES) * N_PER_WAVE;
            index_t src_i_k = lane_id / mfma_inst::m;
            v_offset_a = (src_i_m * KPACK_A + src_i_k * (KPACK_A/*padding*/ + M_PER_BLOCK * KPACK_A)) * sizeof(a_type);
            v_offset_b = (src_i_n * KPACK_B + src_i_k * (KPACK_B/*padding*/ + N_PER_BLOCK * KPACK_B)) * sizeof(b_type);
        };
        auto gld_buf_clear = [&]()
        {
#if USE_GLD_IF
            if constexpr (M_PER_BLOCK > N_PER_BLOCK)
                gld_b.clear_buf();
            else
                gld_a.clear_buf();
#endif
        };

        gld_a.clear_buf();
        gld_a(); gld_a.move_slice_window(K_PER_BLOCK);
        gld_b.clear_buf();
        gld_b(); gld_b.move_slice_window(K_PER_BLOCK);

        index_t v_offset_a, v_offset_b;
        mfma_src_dist_offset(v_offset_a, v_offset_b);
        auto sld_a = sld_iter_a_type{smem, v_offset_a};
        auto sld_b = sld_iter_b_type{smem, v_offset_b, smem_size_a()};

        gld_fence(gld_b.issues);
        sst_a(gld_a.buf);
        gld_fence(0);
        sst_b(gld_b.buf);
        clear(acc_buf);
        sst_fence(0); wave_barrier();
        gld_buf_clear();

        for(auto i_k = 1; i_k < k_iters; i_k++) {
            gemm(sld_a, sld_b, sst_a, sst_b, gld_a, gld_b, gld_buf_clear, bool_const<true>{});
        }
        // tail
        gemm(sld_a, sld_b, sst_a, sst_b, gld_a, gld_b, gld_buf_clear, bool_const<false>{});
        auto epilogue = epilogue_iterator<DATA_TYPES_, BLOCK_TILE_, BLOCK_WAVES_, WAVE_TILE_> {ptr_c, karg.m - block_i_m, karg.n - block_i_n, karg.ldc, smem};
        // write out
        sched_barrier();  // in case mfma dest has raw harzard
        epilogue(acc_buf);
    }
};
