#pragma once
// Register layout repack utilities for opus dtype vector types.
// All types are defined by REGISTER_DTYPE in opus/opus.hpp.
//
// Pattern (same as matrix_core_gfx942.cc packers):
//   union <dtype><width>Packer { <widest_vec> vec<N>; <half_vec> vec<N/2>[2]; ... scalar vec[N]; };
//
// Scalar types and their vector aliases (via REGISTER_DTYPE):
//   fp32_t  = float              -> fp32x{1,2,4,8,16,32,64}_t
//   fp16_t  = _Float16           -> fp16x{1,2,4,8,16,32,64}_t
//   bf16_t  = unsigned short     -> bf16x{1,2,4,8,16,32,64}_t
//   fp8_t   = _BitInt(8)         -> fp8x{1,2,4,8,16,32,64}_t
//   bf8_t   = unsigned _BitInt(8)-> bf8x{1,2,4,8,16,32,64}_t
//   i32_t   = int                -> i32x{1,2,4,8,16,32,64}_t
//   u32_t   = unsigned int       -> u32x{1,2,4,8,16,32,64}_t
//   i16_t   = short              -> i16x{1,2,4,8,16,32,64}_t
//   i8_t    = signed char        -> i8x{1,2,4,8,16,32,64}_t
//   u8_t    = unsigned char      -> u8x{1,2,4,8,16,32,64}_t
//
// Naming convention: <DTYPE><N>Packer  e.g. Fp16x8Packer, Fp32x4Packer, I8x16Packer
// Each union exposes all sub-vector views down to the scalar array.

#include "opus/opus.hpp"

namespace reg_utils {

using namespace opus;

// ============================================================
// fp32 packers  (32-bit lanes, DWORD-granularity)
// ============================================================

// 2 lanes = 2 DWORDs  (e.g. one ds_read_b64 result)
union Fp32x2Packer {
    fp32x2_t vec2;
    fp32_t   vec[2];
};

// 4 lanes = 4 DWORDs  (e.g. one buffer_load_dwordx4 / v_mfma accumulator slice)
union Fp32x4Packer {
    fp32x4_t vec4;
    fp32x2_t vec2[2];
    fp32_t   vec[4];
};

// 8 lanes = 8 DWORDs
union Fp32x8Packer {
    fp32x8_t vec8;
    fp32x4_t vec4[2];
    fp32x2_t vec2[4];
    fp32_t   vec[8];
};

// 16 lanes = 16 DWORDs (full 16x16 mfma f32 accumulator per wave)
union Fp32x16Packer {
    fp32x16_t vec16;
    fp32x8_t  vec8[2];
    fp32x4_t  vec4[4];
    fp32x2_t  vec2[8];
    fp32_t    vec[16];
};

// ============================================================
// fp16 packers  (16-bit lanes, WORD-granularity)
// ============================================================

// 2 lanes = 1 DWORD
union Fp16x2Packer {
    fp16x2_t vec2;
    fp16_t   vec[2];
};

// 4 lanes = 2 DWORDs  (e.g. mfma f16 input fragment)
union Fp16x4Packer {
    fp16x4_t vec4;
    fp16x2_t vec2[2];
    fp16_t   vec[4];
};

// 8 lanes = 4 DWORDs  (e.g. one buffer_load_dwordx4 for fp16, wmma fragment)
// Identical in purpose to half8Packer / half4Packer in matrix_core_gfx942.cc
union Fp16x8Packer {
    fp16x8_t vec8;
    fp16x4_t vec4[2];
    fp16x2_t vec2[4];
    fp16_t   vec[8];
};

// 16 lanes = 8 DWORDs  (full 16x16 wmma f16 accumulator per wave on RDNA4)
union Fp16x16Packer {
    fp16x16_t vec16;
    fp16x8_t  vec8[2];
    fp16x4_t  vec4[4];
    fp16x2_t  vec2[8];
    fp16_t    vec[16];
};

// ============================================================
// bf16 packers  (16-bit lanes)
// ============================================================

union Bf16x2Packer {
    bf16x2_t vec2;
    bf16_t   vec[2];
};

union Bf16x4Packer {
    bf16x4_t vec4;
    bf16x2_t vec2[2];
    bf16_t   vec[4];
};

union Bf16x8Packer {
    bf16x8_t vec8;
    bf16x4_t vec4[2];
    bf16x2_t vec2[4];
    bf16_t   vec[8];
};

union Bf16x16Packer {
    bf16x16_t vec16;
    bf16x8_t  vec8[2];
    bf16x4_t  vec4[4];
    bf16x2_t  vec2[8];
    bf16_t    vec[16];
};

// ============================================================
// fp8 packers  (8-bit lanes, fp8 E4M3)
// 4 fp8 elements pack into 1 DWORD
// ============================================================

union Fp8x4Packer {
    fp8x4_t vec4;
    fp8_t   vec[4];
};

union Fp8x8Packer {
    fp8x8_t vec8;
    fp8x4_t vec4[2];
    fp8_t   vec[8];
};

union Fp8x16Packer {
    fp8x16_t vec16;
    fp8x8_t  vec8[2];
    fp8x4_t  vec4[4];
    fp8_t    vec[16];
};

union Fp8x32Packer {
    fp8x32_t vec32;
    fp8x16_t vec16[2];
    fp8x8_t  vec8[4];
    fp8x4_t  vec4[8];
    fp8_t    vec[32];
};

// ============================================================
// bf8 packers  (8-bit lanes, bf8 E5M2)
// ============================================================

union Bf8x4Packer {
    bf8x4_t vec4;
    bf8_t   vec[4];
};

union Bf8x8Packer {
    bf8x8_t vec8;
    bf8x4_t vec4[2];
    bf8_t   vec[8];
};

union Bf8x16Packer {
    bf8x16_t vec16;
    bf8x8_t  vec8[2];
    bf8x4_t  vec4[4];
    bf8_t    vec[16];
};

union Bf8x32Packer {
    bf8x32_t vec32;
    bf8x16_t vec16[2];
    bf8x8_t  vec8[4];
    bf8x4_t  vec4[8];
    bf8_t    vec[32];
};

// ============================================================
// Mixed fp8/bf8 packer: interleave fp8+bf8 in the same DWORD
// Used for packed fp8+bf8 mfma inputs on gfx950
// ============================================================

union Fp8Bf8x4Packer {
    fp8x4_t fp8_vec4;
    bf8x4_t bf8_vec4;
    fp8_t   fp8_vec[4];
    bf8_t   bf8_vec[4];
};

// ============================================================
// i32 / u32 packers  (32-bit integer lanes)
// ============================================================

union I32x2Packer {
    i32x2_t vec2;
    i32_t   vec[2];
};

union I32x4Packer {
    i32x4_t vec4;
    i32x2_t vec2[2];
    i32_t   vec[4];
};

union I32x16Packer {
    i32x16_t vec16;
    i32x8_t  vec8[2];
    i32x4_t  vec4[4];
    i32x2_t  vec2[8];
    i32_t    vec[16];
};

union U32x2Packer {
    u32x2_t vec2;
    u32_t   vec[2];
};

union U32x4Packer {
    u32x4_t vec4;
    u32x2_t vec2[2];
    u32_t   vec[4];
};

// ============================================================
// i16 packers  (16-bit signed integer lanes)
// ============================================================

union I16x2Packer {
    i16x2_t vec2;
    i16_t   vec[2];
};

union I16x4Packer {
    i16x4_t vec4;
    i16x2_t vec2[2];
    i16_t   vec[4];
};

union I16x8Packer {
    i16x8_t vec8;
    i16x4_t vec4[2];
    i16x2_t vec2[4];
    i16_t   vec[8];
};

// ============================================================
// i8 / u8 packers  (8-bit integer lanes)
// 4 bytes pack into 1 DWORD
// ============================================================

union I8x4Packer {
    i8x4_t vec4;
    i8_t   vec[4];
};

union I8x8Packer {
    i8x8_t vec8;
    i8x4_t vec4[2];
    i8_t   vec[8];
};

union I8x16Packer {
    i8x16_t vec16;
    i8x8_t  vec8[2];
    i8x4_t  vec4[4];
    i8_t    vec[16];
};

union U8x4Packer {
    u8x4_t vec4;
    u8_t   vec[4];
};

union U8x8Packer {
    u8x8_t vec8;
    u8x4_t vec4[2];
    u8_t   vec[8];
};

union U8x16Packer {
    u8x16_t vec16;
    u8x8_t  vec8[2];
    u8x4_t  vec4[4];
    u8_t    vec[16];
};

// ============================================================
// Cross-dtype bit-cast packers
// Reinterpret register storage as a different element type of
// the same total byte width.  These mirror the common pattern
// in kernel code where an fp16x8 load result is split into
// fp16x4 fragments for mfma/wmma input registers.
// ============================================================

// fp16x8 <-> fp32x4  (same 128 bits; used to convert mfma f32 accum to f16 output)
union Fp16x8_Fp32x4Packer {
    fp16x8_t fp16_vec8;    // view as 8 fp16 elements
    fp32x4_t fp32_vec4;    // view as 4 fp32 elements (same bits)
};

// bf16x8 <-> fp32x4
union Bf16x8_Fp32x4Packer {
    bf16x8_t bf16_vec8;
    fp32x4_t fp32_vec4;
};

// fp8x16 <-> fp32x4  (128 bits)
union Fp8x16_Fp32x4Packer {
    fp8x16_t fp8_vec16;
    fp32x4_t fp32_vec4;
};

// bf8x16 <-> fp32x4
union Bf8x16_Fp32x4Packer {
    bf8x16_t bf8_vec16;
    fp32x4_t fp32_vec4;
};

// i8x16 <-> i32x4  (common for int8 dot-product mfma)
union I8x16_I32x4Packer {
    i8x16_t  i8_vec16;
    i32x4_t  i32_vec4;
};

// u8x16 <-> i32x4
union U8x16_I32x4Packer {
    u8x16_t  u8_vec16;
    i32x4_t  i32_vec4;
};

// i16x8 <-> i32x4
union I16x8_I32x4Packer {
    i16x8_t  i16_vec8;
    i32x4_t  i32_vec4;
};

} // namespace reg_utils
