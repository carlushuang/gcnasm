#pragma once

#include "rsrc_desc_utils.h"

namespace tcopy_desc_detail {

template<opus::index_t SelectedWorkgroupCount, typename SelectedWorkgroups>
struct WgMaskFromSelectedWorkgroups;

template<opus::index_t SelectedWorkgroupCount, opus::index_t... WgIndices>
struct WgMaskFromSelectedWorkgroups<SelectedWorkgroupCount, opus::seq<WgIndices...>> {
    static_assert(SelectedWorkgroupCount >= 0,
                  "TcopyDesc: selected workgroup count must be >= 0");
    static_assert(SelectedWorkgroupCount == 0 ||
                      static_cast<opus::index_t>(sizeof...(WgIndices)) == SelectedWorkgroupCount,
                  "TcopyDesc: opus::seq element count must match selected workgroup count");
    static_assert(SelectedWorkgroupCount == 0 || (((WgIndices >= 0) && ...) &&
                  ((WgIndices < 16) && ...)),
                  "TcopyDesc: selected workgroup index must be in [0, 15]");

    static constexpr uint16_t value = [] {
        if constexpr (SelectedWorkgroupCount == 0) {
            return uint16_t{0};
        } else {
            return (uint16_t{0} | ... |
                    static_cast<uint16_t>(uint16_t{1} << WgIndices));
        }
    }();
};

template<opus::index_t SelectedWorkgroupCount, typename SelectedWorkgroups>
inline constexpr uint16_t kWgMaskFromSelectedWorkgroups =
    WgMaskFromSelectedWorkgroups<SelectedWorkgroupCount, SelectedWorkgroups>::value;

} // namespace tcopy_desc_detail

template<typename T>
struct TcopyDataSize {
    static_assert(sizeof(T) == 1 || sizeof(T) == 2 ||
                  sizeof(T) == 4 || sizeof(T) == 8,
                  "TcopyDataSize: unsupported element size");
    static constexpr uint64_t value =
        (sizeof(T) == 1) ? 0 :
        (sizeof(T) == 2) ? 1 :
        (sizeof(T) == 4) ? 2 : 3;
};

template<
    typename DataType,
    uint64_t TileDim0        = 0,
    uint64_t TileDim1        = 0,
    uint64_t TileDim2        = 0,
    uint64_t TileDim3        = 0,
    uint64_t TileDim4        = 0,
    uint64_t Count             = 1,
    uint64_t GatherIndexSize   = 0,
    uint64_t GatherMode        = 0,
    uint64_t TypeLo            = 0,
    uint64_t TypeHi            = 1,
    uint64_t AtomicBarrierEn = 0,
    uint64_t IterateEn       = 0,
    uint64_t McEarlyTimeout  = 0,
    opus::index_t SelectedWorkgroupCount = 0,
    uint64_t LdsPadEn        = 0,
    uint64_t PadInterval     = 0,
    uint64_t PadAmount       = 0,
    typename SelectedWorkgroups = opus::seq<>
>
struct TcopyDesc {

    static constexpr uint64_t kDataSize = TcopyDataSize<DataType>::value;
    static constexpr uint64_t kWgMask =
        static_cast<uint64_t>(
            tcopy_desc_detail::kWgMaskFromSelectedWorkgroups<
                SelectedWorkgroupCount, SelectedWorkgroups>);

    using Sg0Type = SgprBitField128<
        BF<0,   1, Count>,
        BF<1,  29, 0>,
        BF<30,  1, GatherIndexSize>,
        BF<31,  1, GatherMode>,
        BF<32, 32>,
        BF<64, 57>,
        BF<121, 5, 0>,
        BF<126, 1, TypeLo>,
        BF<127, 1, TypeHi>
    >;

    using Sg1Type = SgprBitField256<
        BF<0,   16, kWgMask>,
        BF<16,   2, kDataSize>,
        BF<18,   1, AtomicBarrierEn>,
        BF<19,   1, IterateEn>,
        BF<20,   1, LdsPadEn>,
        BF<21,   1, McEarlyTimeout>,
        BF<22,   3, PadInterval>,
        BF<25,   7, PadAmount>,
        BF<32,  16>,
        BF<48,  32>,
        BF<80,  32>,
        BF<112, 16, TileDim0>,
        BF<128, 16, TileDim1>,
        BF<144, 16, TileDim2>,
        BF<160, 48>,
        BF<208, 48>
    >;

    using Sg2Type = SgprBitField128<
        BF<0,  32>,
        BF<32, 32>,
        BF<64, 48>,
        BF<112,16, TileDim3>
    >;

    using Sg3Type = SgprBitField128<
        BF<0,  48>,
        BF<48, 32>,
        BF<80, 16, TileDim4>,
        BF<96, 32, 0>
    >;

    Sg0Type sg0;
    Sg1Type sg1;
    Sg2Type sg2;
    Sg3Type sg3;

    OPUS_H_D void set_lds_addr(uintptr_t v)          { sg0.template set<4>(v); }
    OPUS_H_D void set_global_addr(uintptr_t v)       { sg0.template set<5>(v); }
    OPUS_H_D void set_tensor_dim0(uint32_t v)        { sg1.template set<9>(v); }
    OPUS_H_D void set_tensor_dim1(uint32_t v)        { sg1.template set<10>(v); }
    OPUS_H_D void set_tensor_dim0_stride(uint64_t v) { sg1.template set<14>(v); }
    OPUS_H_D void set_tensor_dim1_stride(uint64_t v) { sg1.template set<15>(v); }
    OPUS_H_D void set_lds_barrier_addr(uint16_t v)   { sg1.template set<8>(v); }
    OPUS_H_D void set_tensor_dim2(uint32_t v)        { sg2.template set<0>(v); }
    OPUS_H_D void set_tensor_dim3(uint32_t v)        { sg2.template set<1>(v); }
    OPUS_H_D void set_tensor_dim2_stride(uint64_t v) { sg2.template set<2>(v); }
    OPUS_H_D void set_tensor_dim3_stride(uint64_t v) { sg3.template set<0>(v); }
    OPUS_H_D void set_tensor_dim4(uint32_t v)        { sg3.template set<1>(v); }

    OPUS_H_D void make(
        uintptr_t   lds_addr,
        const void* global_addr,
        uint32_t    tensor_dim0,
        uint32_t    tensor_dim1,
        uint64_t    tensor_dim0_stride,
        uint64_t    tensor_dim1_stride  = 0,
        uint16_t    lds_barrier_addr    = 0,
        uint32_t    tensor_dim2         = 0,
        uint32_t    tensor_dim3         = 0,
        uint64_t    tensor_dim2_stride  = 0,
        uint64_t    tensor_dim3_stride  = 0,
        uint32_t    tensor_dim4         = 0)
    {
        sg0.template set<4>(lds_addr);
        sg0.template set<5>(reinterpret_cast<uintptr_t>(global_addr));

        sg1.template set<9> (tensor_dim0);
        sg1.template set<10>(tensor_dim1);
        sg1.template set<14>(tensor_dim0_stride);

        if (lds_barrier_addr)   sg1.template set<8> (lds_barrier_addr);
        if (tensor_dim1_stride) sg1.template set<15>(tensor_dim1_stride);

        if (tensor_dim2)        sg2.template set<0>(tensor_dim2);
        if (tensor_dim3)        sg2.template set<1>(tensor_dim3);
        if (tensor_dim2_stride) sg2.template set<2>(tensor_dim2_stride);

        if (tensor_dim3_stride) sg3.template set<0>(tensor_dim3_stride);
        if (tensor_dim4)        sg3.template set<1>(tensor_dim4);
    }
};
