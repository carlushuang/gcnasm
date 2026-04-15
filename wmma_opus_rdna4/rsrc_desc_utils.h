#pragma once
// Resource Descriptor Bit-Field Utilities
// Supports 128-bit (4x SGPR) and 256-bit (8x SGPR) descriptors.
//
// BF<Start, Width>          — plain runtime field, default = 0
// BF<Start, Width, Value>   — field with compile-time default Value,
//                             set<I>() can still override it at runtime
//
// SgprBitField<NWords, BF...>
//   NWords = 4  →  128-bit  (4x SGPR, e.g. buffer resource descriptor)
//   NWords = 8  →  256-bit  (8x SGPR, e.g. Tensor Copy / image descriptor)
//
// 256-bit storage is implemented as two 128-bit halves via a union:
//   union { uint32_t flat[8]; uint32_t half[2][4]; }
// so existing 128-bit helpers (extract128 / insert128) are reused unchanged.
//
// ──────────────────────────────────────────────────────────────
// Example A — 128-bit buffer resource descriptor (4 SGPRs)
//
//   using BufferDesc = SgprBitField<4,
//       BF<0,  48>,                   // 0: base_address   — runtime
//       BF<48, 14>,                   // 1: stride         — runtime
//       BF<62,  2, 0>,                // 2: cache_swizzle  — default 0
//       BF<64, 32, 0xFFFFFFFFULL>,    // 3: num_records    — default max
//       BF<96, 32, 0x00020000ULL>     // 4: flags (DWORD3) — default
//   >;
//
//   BufferDesc desc;
//   desc.set<0>(base_addr);
//   desc.set<1>(64);
//   const uint32_t* s = desc.sgpr();   // 4 SGPRs
//
// ──────────────────────────────────────────────────────────────
// Example B — 256-bit Tensor Copy / image descriptor (8 SGPRs)
//
//   using TcopyDesc = SgprBitField<8,
//       BF<0,   48>,                  // 0: base_address   — runtime
//       BF<48,  14>,                  // 1: stride         — runtime
//       BF<62,   2, 0>,               // 2: swizzle        — default 0
//       BF<64,  32, 0xFFFFFFFFULL>,   // 3: num_records    — default max
//       BF<96,  32, 0x00020000ULL>,   // 4: flags low      — default
//       BF<128, 64>,                  // 5: metadata_ptr   — runtime (bits 128..191)
//       BF<192, 32, 0x00000100ULL>,   // 6: format         — default, overridable
//       BF<224, 32>                   // 7: extra_flags    — runtime
//   >;
//
//   TcopyDesc desc;
//   desc.set<0>(base_addr);
//   desc.set<5>(meta_ptr);
//   const uint32_t* s = desc.sgpr();   // 8 SGPRs

#include <cstdint>
#include <cstring>
#include "opus/opus.hpp"

// ---------------------------------------------------------------
// Sentinel
// ---------------------------------------------------------------
static constexpr uint64_t BF_NO_DEFAULT = ~uint64_t(0);

// ---------------------------------------------------------------
// BF<Start, Width [, DefaultValue]>
//   — runtime field (no default), or field with default that can be overridden
//   — set<I>() always allowed
// ---------------------------------------------------------------
template<int Start, int Width, uint64_t DefaultValue = BF_NO_DEFAULT>
struct BF {
    static constexpr int      start         = Start;
    static constexpr int      width         = Width;
    static constexpr uint64_t default_value = DefaultValue;
    static constexpr bool     has_default   = (DefaultValue != BF_NO_DEFAULT);

    static_assert(Width > 0 && Width <= 64, "BF width must be 1..64");
    static_assert(Start >= 0,               "BF start must be >= 0");
};

// ---------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------
namespace rsrc_detail {

inline constexpr uint64_t mask64(int n) {
    return (n >= 64) ? ~uint64_t(0) : ((uint64_t(1) << n) - 1);
}

// Read [start, start+width) from a 128-bit window (4x u32)
OPUS_H_D uint64_t extract128(const uint32_t s[4], int start, int width) {
    uint64_t lo, hi;
    memcpy(&lo, &s[0], 8);
    memcpy(&hi, &s[2], 8);
    int end = start + width;
    if (end <= 64) {
        return (lo >> start) & mask64(width);
    } else if (start >= 64) {
        return (hi >> (start - 64)) & mask64(width);
    } else {
        int lo_bits = 64 - start;
        int hi_bits = width - lo_bits;
        return ((lo >> start) & mask64(lo_bits)) | ((hi & mask64(hi_bits)) << lo_bits);
    }
}

// Write [start, start+width) into a 128-bit window (4x u32)
OPUS_H_D void insert128(uint32_t s[4], int start, int width, uint64_t value) {
    uint64_t lo, hi;
    memcpy(&lo, &s[0], 8);
    memcpy(&hi, &s[2], 8);
    value &= mask64(width);
    int end = start + width;
    if (end <= 64) {
        lo = (lo & ~(mask64(width) << start)) | (value << start);
    } else if (start >= 64) {
        int s2 = start - 64;
        hi = (hi & ~(mask64(width) << s2)) | (value << s2);
    } else {
        int lo_bits = 64 - start;
        int hi_bits = width - lo_bits;
        lo = (lo & ~(mask64(lo_bits) << start)) | ((value & mask64(lo_bits)) << start);
        hi = (hi & ~mask64(hi_bits))             | ((value >> lo_bits) & mask64(hi_bits));
    }
    memcpy(&s[0], &lo, 8);
    memcpy(&s[2], &hi, 8);
}

// Dispatch: route a global bit offset into the correct 128-bit half,
// then delegate to extract128 / insert128.
// For NWords==4: only half[0] exists, global == local.
// For NWords==8: half[0] covers bits 0..127, half[1] covers bits 128..255.
OPUS_H_D uint64_t extractN(const uint32_t* flat, int start, int width) {
    int half_idx    = start / 128;
    int local_start = start - half_idx * 128;
    return extract128(flat + half_idx * 4, local_start, width);
}

OPUS_H_D void insertN(uint32_t* flat, int start, int width, uint64_t value) {
    int half_idx    = start / 128;
    int local_start = start - half_idx * 128;
    insert128(flat + half_idx * 4, local_start, width, value);
}

// Compile-time recursion: apply all has_default fields
template<int I, typename... Fields>
struct ApplyDefaults;

template<int I>
struct ApplyDefaults<I> {
    OPUS_H_D static void apply(uint32_t*) {}
};

template<int I, typename Head, typename... Tail>
struct ApplyDefaults<I, Head, Tail...> {
    OPUS_H_D static void apply(uint32_t* flat) {
        if constexpr (Head::has_default)
            insertN(flat, Head::start, Head::width, Head::default_value);
        ApplyDefaults<I+1, Tail...>::apply(flat);
    }
};

// Compile-time recursion: validate each BF against the descriptor bit-width
template<int TotalBits, typename... Fields>
struct ValidateFields;

template<int TotalBits>
struct ValidateFields<TotalBits> {};

template<int TotalBits, typename Head, typename... Tail>
struct ValidateFields<TotalBits, Head, Tail...> : ValidateFields<TotalBits, Tail...> {
    static_assert(Head::start < TotalBits,
        "BF start exceeds descriptor width (use BF<S,W> with S < NWords*32)");
    static_assert(Head::start + Head::width <= TotalBits,
        "BF start+width exceeds descriptor width (use BF<S,W> with S+W <= NWords*32)");
};

} // namespace rsrc_detail

// ---------------------------------------------------------------
// Index helper
// ---------------------------------------------------------------
template<int N, typename Head, typename... Tail>
struct nth_bf { using type = typename nth_bf<N-1, Tail...>::type; };
template<typename Head, typename... Tail>
struct nth_bf<0, Head, Tail...> { using type = Head; };

// ---------------------------------------------------------------
// Storage: linear array of NWords x uint32_t
// NWords == 4 → 128-bit;  NWords == 8 → 256-bit
// ---------------------------------------------------------------
template<int NWords>
struct SgprStorage {
    static_assert(NWords == 4 || NWords == 8, "NWords must be 4 (128-bit) or 8 (256-bit)");

    uint32_t data[NWords] = {};

    OPUS_H_D void zero()                      { memset(data, 0, sizeof(data)); }
    OPUS_H_D uint32_t*       flat()           { return data; }
    OPUS_H_D const uint32_t* flat()     const { return data; }
    OPUS_H_D const uint32_t* sgpr()     const { return data; }
};

// ---------------------------------------------------------------
// SgprBitField<NWords, BF<...>, ...>
// ---------------------------------------------------------------
template<int NWords, typename... Fields>
struct SgprBitField {
    // Validate all BF ranges against the actual descriptor width at instantiation
    static_assert(sizeof(rsrc_detail::ValidateFields<NWords * 32, Fields...>) >= 0, "");

    SgprStorage<NWords> storage;

    // Default values written on construction
    OPUS_H_D SgprBitField() { rsrc_detail::ApplyDefaults<0, Fields...>::apply(storage.flat()); }

    // get<I>: read field I
    template<int I>
    OPUS_H_D uint64_t get() const {
        using F = typename nth_bf<I, Fields...>::type;
        return rsrc_detail::extractN(storage.flat(), F::start, F::width);
    }

    // set<I>: write field I — always allowed, overrides compile-time default
    template<int I>
    OPUS_H_D void set(uint64_t value) {
        using F = typename nth_bf<I, Fields...>::type;
        rsrc_detail::insertN(storage.flat(), F::start, F::width, value);
    }

    // set<I>: pointer overload (host + device) — any pointer type → uint64_t
    template<int I, typename T>
    OPUS_H_D void set(T* ptr) {
        set<I>(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(ptr)));
    }

    // set<I>: host-only const void* overload
    template<int I>
    OPUS_H void set(const void* ptr) {
        set<I>(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(ptr)));
    }

    // Full raw SGPR array
    OPUS_H_D const uint32_t* sgpr() const { return storage.sgpr(); }

    // as<VecT>(): bit-cast the descriptor storage to any same-size vector type
    // e.g. desc.as<int32x4_t>()  for 128-bit,  desc.as<int32x8_t>() for 256-bit
    template<typename VecT>
    OPUS_H_D VecT as() const {
        static_assert(sizeof(VecT) == sizeof(storage.data),
            "as<VecT>: VecT size must match descriptor storage size");
        return __builtin_bit_cast(VecT, storage.data);
    }

    // Reset: zero all then re-apply compile-time defaults
    OPUS_H_D void reset() {
        storage.zero();
        rsrc_detail::ApplyDefaults<0, Fields...>::apply(storage.flat());
    }
};

// ---------------------------------------------------------------
// Convenience aliases
// ---------------------------------------------------------------
template<typename... Fields>
using SgprBitField128 = SgprBitField<4, Fields...>;

template<typename... Fields>
using SgprBitField256 = SgprBitField<8, Fields...>;
