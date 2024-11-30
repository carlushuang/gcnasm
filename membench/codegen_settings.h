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

struct WRITE_mode {};

template <>
struct CodegenSettings<WRITE_mode> {
  static constexpr int GS = WRITE_GS_VAL;
  static constexpr int BS = WRITE_BS_VAL;
  static constexpr float PIXELS_MB = WRITE_PIXELS_MB_VAL;
  static constexpr int UNROLL = WRITE_UNROLL_VAL;
};

struct MEMCPY_mode {};

template <>
struct CodegenSettings<MEMCPY_mode> {
  static constexpr int GS = MEMCPY_GS_VAL;
  static constexpr int BS = MEMCPY_BS_VAL;
  static constexpr float PIXELS_MB = MEMCPY_PIXELS_MB_VAL;
  static constexpr int UNROLL = MEMCPY_UNROLL_VAL;
  static constexpr int ROW_PER_THREAD = MEMCPY_ROW_PER_THREAD_VAL;
  static constexpr int PADDING = MEMCPY_PADDING_VAL;
  static constexpr int BYTES_PER_ISSUE = 16;
};

struct MEMCPY_ASYNC_mode {};

template <>
struct CodegenSettings<MEMCPY_ASYNC_mode> {
  static constexpr int GS = MEMCPY_ASYNC_GS_VAL;
  static constexpr int BS = MEMCPY_ASYNC_BS_VAL;
  static constexpr float PIXELS_MB = MEMCPY_ASYNC_PIXELS_MB_VAL;
  static constexpr int UNROLL = MEMCPY_ASYNC_UNROLL_VAL;
  static constexpr int ROW_PER_THREAD = MEMCPY_ASYNC_ROW_PER_THREAD_VAL;
  static constexpr int PADDING = MEMCPY_ASYNC_PADDING_VAL;
  static constexpr int BYTES_PER_ISSUE = 16;
};

struct MEMCPY_ASYNC_INPLACE_mode {};

template <>
struct CodegenSettings<MEMCPY_ASYNC_INPLACE_mode> {
  static constexpr int GS = MEMCPY_ASYNC_INPLACE_GS_VAL;
  static constexpr int BS = MEMCPY_ASYNC_INPLACE_BS_VAL;
  static constexpr float PIXELS_MB = MEMCPY_ASYNC_INPLACE_PIXELS_MB_VAL;
  static constexpr int UNROLL = MEMCPY_ASYNC_INPLACE_UNROLL_VAL;
  static constexpr int ROW_PER_THREAD = MEMCPY_ASYNC_INPLACE_ROW_PER_THREAD_VAL;
  static constexpr int PADDING = MEMCPY_ASYNC_INPLACE_PADDING_VAL;
  static constexpr int BYTES_PER_ISSUE = 16;
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
