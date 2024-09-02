#pragma once

template <typename T>
struct CodegenSettings {
  static constexpr int GS = 80;
  static constexpr int BS = 1024;
  static constexpr float PIXELS_MB = 437.5;
  static constexpr int UNROLL = 4;
};

struct MEMCPY_PERSISTENT_mode {};

template <>
struct CodegenSettings<MEMCPY_PERSISTENT_mode> {
  static constexpr int GS = MEMCPY_PERSISTENT_GS_VAL;
  static constexpr int BS = MEMCPY_PERSISTENT_BS_VAL;
  static constexpr float PIXELS_MB = MEMCPY_PERSISTENT_PIXELS_MB_VAL;
  static constexpr int UNROLL = MEMCPY_PERSISTENT_UNROLL_VAL;
};

struct READ_mode {};

template <>
struct CodegenSettings<READ_mode> {
  static constexpr int GS = READ_GS_VAL;
  static constexpr int BS = READ_BS_VAL;
  static constexpr float PIXELS_MB = READ_PIXELS_MB_VAL;
  static constexpr int UNROLL = READ_UNROLL_VAL;
};

struct MEMCPY_mode {};

template <>
struct CodegenSettings<MEMCPY_mode> {
  static constexpr int GS = MEMCPY_GS_VAL;
  static constexpr int BS = MEMCPY_BS_VAL;
  static constexpr float PIXELS_MB = MEMCPY_PIXELS_MB_VAL;
  static constexpr int UNROLL = MEMCPY_UNROLL_VAL;
};

struct MEMCPY_ASYNC_mode {};

template <>
struct CodegenSettings<MEMCPY_ASYNC_mode> {
  static constexpr int GS = MEMCPY_ASYNC_GS_VAL;
  static constexpr int BS = MEMCPY_ASYNC_BS_VAL;
  static constexpr float PIXELS_MB = MEMCPY_ASYNC_PIXELS_MB_VAL;
  static constexpr int UNROLL = MEMCPY_ASYNC_UNROLL_VAL;
};


struct MEMCPY_SWIZZLED_mode {};

template <>
struct CodegenSettings<MEMCPY_SWIZZLED_mode> {
  static constexpr int GS = MEMCPY_SWIZZLED_GS_VAL;
  static constexpr int BS = MEMCPY_SWIZZLED_BS_VAL;
  static constexpr float PIXELS_MB = MEMCPY_SWIZZLED_PIXELS_MB_VAL;
  static constexpr int CHUNKS = MEMCPY_SWIZZLED_CHUNKS_VAL;
  static constexpr int INNER = MEMCPY_SWIZZLED_INNER_VAL;
};