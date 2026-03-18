#pragma once
// TDM (Tensor DMA) Descriptor Utilities for RDNA4 / gfx1250
//
// Field classification follows the comments in matrix_core_tdm.cc:
//   BF<S,W,V>  — compile-time default, set<I>() still allowed (overridable)
//   BF<S,W>    — pure runtime field, no default
//
// All compile-time fields use BF<S,W,V> with a template-parameter value.
// Pad fields (lds_pad_en, pad_interval, pad_amount) also use BF<S,W,V>
// and can be overridden at runtime via sg1.set<I>().
//
// DataType template parameter:
//   Pass any opus scalar type (fp16_t, fp8_t, fp32_t, u8_t, …).
//   TdmDataSize<T> automatically maps sizeof(T) → TDM data_size encoding:
//     1-byte → 0,  2-byte → 1,  4-byte → 2,  8-byte → 3
//
// Usage:
//   // ① fp16, tile_dim0=32, tile_dim1=16, no pad
//   using MyTdm = TdmDesc<fp16_t, 32, 16>;
//
//   MyTdm tdm;
//   tdm.make(smembase, ptr_a,
//            /*tensor_dim0=*/32, /*tensor_dim1=*/16,
//            /*tensor_dim0_stride=*/32);
//
//   __builtin_amdgcn_tensor_load_to_lds(
//       tdm.sg0.as<int32x4_t>(), tdm.sg1.as<int32x8_t>(), ...);
//
//   // ② Compile-time pad enable
//   using PadTdm = TdmDesc<fp16_t, 32, 16,
//       /*TileDim2=*/0, /*TileDim3=*/0, /*TileDim4=*/0,
//       /*Count=*/1, /*GatherMode=*/0, /*TypeLo=*/0, /*TypeHi=*/1,
//       /*AtomicBarrierEn=*/0, /*IterateEn=*/0, /*McEarlyTimeout=*/0,
//       /*WgMask=*/0,
//       /*LdsPadEn=*/1, /*PadInterval=*/0, /*PadAmount=*/4>;
//
//   // ③ Runtime pad override (BF with default → set() allowed)
//   tdm.sg1.set<4>(1);   // override LdsPadEn at runtime
//   tdm.sg1.set<7>(4);   // override PadAmount at runtime

#include "rsrc_desc_utils.h"

// ─────────────────────────────────────────────────────────────────────────────
// TdmDataSize<T>: map opus scalar dtype to TDM data_size encoding
//   sizeof == 1 → 0 (u8, i8, fp8, bf8)
//   sizeof == 2 → 1 (fp16, bf16, i16)
//   sizeof == 4 → 2 (fp32, i32)
//   sizeof == 8 → 3 (fp64)
// ─────────────────────────────────────────────────────────────────────────────
template<typename T>
struct TdmDataSize {
    static_assert(sizeof(T) == 1 || sizeof(T) == 2 ||
                  sizeof(T) == 4 || sizeof(T) == 8,
                  "TdmDataSize: unsupported element size");
    static constexpr uint64_t value =
        (sizeof(T) == 1) ? 0 :
        (sizeof(T) == 2) ? 1 :
        (sizeof(T) == 4) ? 2 : 3;
};

// ─────────────────────────────────────────────────────────────────────────────
// TdmDesc<DataType, TileDim0, TileDim1, ...>
//
// Template parameters:
//   DataType        — opus scalar type, used to derive data_size automatically
//
//   Frequently-set fixed fields (prominent position):
//     TileDim0      — G1 bit[112:16], default 0
//     TileDim1      — G1 bit[128:16], default 0
//     TileDim2      — G1 bit[144:16], default 0
//     TileDim3      — G2 bit[112:16], default 0
//     TileDim4      — G3 bit[80:16],  default 0
//
//   Rarely-changed fixed fields (with defaults):
//     Count         — G0 bit[0:1],    default 1
//     GatherMode    — G0 bit[30:2],   default 0
//     TypeLo        — G0 bit[126:1],  default 0  ┐ 0b10 = "image"
//     TypeHi        — G0 bit[127:1],  default 1  ┘
//     AtomicBarrierEn — G1 bit[18:1], default 0
//     IterateEn     — G1 bit[19:1],   default 0
//     McEarlyTimeout— G1 bit[21:1],   default 0
//     WgMask        — G1 bit[0:16],   default 0
//
//   Pad fields (compile-time default, runtime-overridable via sg1.set<N>()):
//     LdsPadEn      — G1 bit[20:1],   default 0
//     PadInterval   — G1 bit[22:3],   default 0
//     PadAmount     — G1 bit[25:7],   default 0
// ─────────────────────────────────────────────────────────────────────────────
template<
    typename DataType,
    // ── Frequently set tile dims ──────────────────────────────
    uint64_t TileDim0        = 0,
    uint64_t TileDim1        = 0,
    uint64_t TileDim2        = 0,
    uint64_t TileDim3        = 0,
    uint64_t TileDim4        = 0,
    // ── Rarely changed ───────────────────────────────────────
    uint64_t Count           = 1,
    uint64_t GatherMode      = 0,
    uint64_t TypeLo          = 0,
    uint64_t TypeHi          = 1,
    uint64_t AtomicBarrierEn = 0,
    uint64_t IterateEn       = 0,
    uint64_t McEarlyTimeout  = 0,
    uint64_t WgMask          = 0,
    // ── Pad fields: compile-time default, runtime-overridable ─
    uint64_t LdsPadEn        = 0,
    uint64_t PadInterval     = 0,
    uint64_t PadAmount       = 0
>
struct TdmDesc {

    static constexpr uint64_t kDataSize = TdmDataSize<DataType>::value;

    // ── Group 0: 128-bit ─────────────────────────────────────────────────────
    // bit[0:1]   count          — compile-time
    // bit[1:29]  cwsr_bits      — compile-time, fixed 0
    // bit[30:2]  gather_mode    — compile-time
    // bit[32:32] lds_addr       — runtime
    // bit[64:57] global_addr    — runtime
    // bit[121:5] RESERVED       — compile-time, fixed 0
    // bit[126:1] type[0]        — compile-time
    // bit[127:1] type[1]        — compile-time
    using Sg0Type = SgprBitField128<
        BF<0,   1, Count>,        // 0: count
        BF<1,  29, 0>,            // 1: cwsr_bits (always 0)
        BF<30,  2, GatherMode>,   // 2: gather_mode
        BF<32, 32>,               // 3: lds_addr       — runtime
        BF<64, 57>,               // 4: global_addr    — runtime
        BF<121, 5, 0>,            // 5: RESERVED
        BF<126, 1, TypeLo>,       // 6: type[0]
        BF<127, 1, TypeHi>        // 7: type[1]
    >;

    // ── Group 1: 256-bit ─────────────────────────────────────────────────────
    // bit[0:16]  wg_mask              — compile-time
    // bit[16:2]  data_size            — compile-time (derived from DataType)
    // bit[18:1]  atomic_barrier_en    — compile-time
    // bit[19:1]  iterate_en           — compile-time
    // bit[20:1]  lds_pad_en           — compile-time/runtime
    // bit[21:1]  mc_early_timeout     — compile-time
    // bit[22:3]  pad_interval         — compile-time/runtime
    // bit[25:7]  pad_amount           — compile-time/runtime
    // bit[32:16] lds_barrier_addr     — runtime
    // bit[48:32] tensor_dim0          — runtime
    // bit[80:32] tensor_dim1          — runtime
    // bit[112:16]tile_dim0            — compile-time
    // bit[128:16]tile_dim1            — compile-time
    // bit[144:16]tile_dim2            — compile-time
    // bit[160:48]tensor_dim0_stride   — runtime
    // bit[208:48]tensor_dim1_stride   — runtime
    using Sg1Type = SgprBitField256<
        BF<0,   16, WgMask>,           //  0: wg_mask
        BF<16,   2, kDataSize>,        //  1: data_size (auto from DataType)
        BF<18,   1, AtomicBarrierEn>,  //  2: atomic_barrier_en
        BF<19,   1, IterateEn>,        //  3: iterate_en
        BF<20,   1, LdsPadEn>,         //  4: lds_pad_en     — compile/runtime
        BF<21,   1, McEarlyTimeout>,   //  5: mc_early_timeout
        BF<22,   3, PadInterval>,      //  6: pad_interval   — compile/runtime
        BF<25,   7, PadAmount>,        //  7: pad_amount     — compile/runtime
        BF<32,  16>,                   //  8: lds_barrier_addr — runtime
        BF<48,  32>,                   //  9: tensor_dim0      — runtime
        BF<80,  32>,                   // 10: tensor_dim1      — runtime
        BF<112, 16, TileDim0>,         // 11: tile_dim0
        BF<128, 16, TileDim1>,         // 12: tile_dim1
        BF<144, 16, TileDim2>,         // 13: tile_dim2
        BF<160, 48>,                   // 14: tensor_dim0_stride — runtime
        BF<208, 48>                    // 15: tensor_dim1_stride — runtime
    >;

    // ── Group 2: 128-bit ─────────────────────────────────────────────────────
    // bit[0:32]  tensor_dim2                    — runtime
    // bit[32:32] tensor_dim3/lds_addr_increment — runtime
    // bit[64:48] tensor_dim2_stride             — runtime
    // bit[112:16]tile_dim3                      — compile-time
    using Sg2Type = SgprBitField128<
        BF<0,  32>,               // 0: tensor_dim2                    — runtime
        BF<32, 32>,               // 1: tensor_dim3/lds_addr_increment — runtime
        BF<64, 48>,               // 2: tensor_dim2_stride             — runtime
        BF<112,16, TileDim3>      // 3: tile_dim3
    >;

    // ── Group 3: 128-bit ─────────────────────────────────────────────────────
    // bit[0:48]  tensor_dim3_stride — runtime
    // bit[48:32] tensor_dim4        — runtime
    // bit[80:16] tile_dim4          — compile-time
    // bit[96:32] reserved           — compile-time, fixed 0
    using Sg3Type = SgprBitField128<
        BF<0,  48>,               // 0: tensor_dim3_stride — runtime
        BF<48, 32>,               // 1: tensor_dim4        — runtime
        BF<80, 16, TileDim4>,     // 2: tile_dim4
        BF<96, 32, 0>             // 3: reserved
    >;

    Sg0Type sg0;
    Sg1Type sg1;
    Sg2Type sg2;
    Sg3Type sg3;

    // ── make(): fill all runtime fields ──────────────────────────────────────
    // Required:
    //   lds_addr            — LDS destination byte address (uintptr_t)
    //   global_addr         — global memory source pointer
    //   tensor_dim0         — contiguous dim size in data_size elements
    //   tensor_dim1         — second tensor dim size
    //   tensor_dim0_stride  — dim0 stride in data_size elements
    //
    // Optional (default 0):
    //   tensor_dim1_stride  — G1[208:48]
    //   lds_barrier_addr    — G1[32:16]
    //   tensor_dim2         — G2[0:32]
    //   tensor_dim3         — G2[32:32]
    //   tensor_dim2_stride  — G2[64:48]
    //   tensor_dim3_stride  — G3[0:48]
    //   tensor_dim4         — G3[48:32]
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
        // Group 0
        sg0.template set<3>(lds_addr);
        sg0.template set<4>(reinterpret_cast<uintptr_t>(global_addr));

        // Group 1: required
        sg1.template set<9> (tensor_dim0);
        sg1.template set<10>(tensor_dim1);
        sg1.template set<14>(tensor_dim0_stride);

        // Group 1: optional
        if (lds_barrier_addr)   sg1.template set<8> (lds_barrier_addr);
        if (tensor_dim1_stride) sg1.template set<15>(tensor_dim1_stride);

        // Group 2
        if (tensor_dim2)        sg2.template set<0>(tensor_dim2);
        if (tensor_dim3)        sg2.template set<1>(tensor_dim3);
        if (tensor_dim2_stride) sg2.template set<2>(tensor_dim2_stride);

        // Group 3
        if (tensor_dim3_stride) sg3.template set<0>(tensor_dim3_stride);
        if (tensor_dim4)        sg3.template set<1>(tensor_dim4);
    }
};


