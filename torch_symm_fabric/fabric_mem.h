// fabric_mem.h -- the device-side contract for fabric_mem.py.
//
// hdl.device_desc() returns exactly this struct; pass it to a kernel by value and address
// any rank arithmetically. No pointer array in kernarg, and `r` may be computed at run
// time.
//
//   __global__ void k(FabricWin w, ...) {
//       const uint4* peer = (const uint4*)FABRIC_PEER(w, r);
//       ...
//   }

#pragma once

#include <stdint.h>

struct FabricWin {
    char* base;        // base of the flat span holding every rank
    uint64_t stride;   // rank r starts at base + r*stride
    int rank;          // who we are
    int world;         // how many ranks are mapped
};

// Start of rank r's buffer.
#define FABRIC_PEER(w, r) ((w).base + (uint64_t)(r) * (w).stride)
