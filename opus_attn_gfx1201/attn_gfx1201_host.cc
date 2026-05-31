// SPDX-License-Identifier: MIT
// opus_attn_gfx1201 — host driver + correctness check + benchmark.

#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#include <algorithm>
#include "attn_common.h"

#define HIP_CALL(c) do { \
    hipError_t e = (c); \
    if (e != hipSuccess) { fprintf(stderr, "HIP %s @ %s:%d\n", hipGetErrorString(e), __FILE__, __LINE__); std::exit(1); } \
} while(0)

// Forward-declare kernel symbols from per-version .cc files.
template<class T> __global__ void opus_attn_gfx1201_kernel   (opus_attn_kargs);  // v0
template<class T> __global__ void opus_attn_gfx1201_kernel_v1(opus_attn_kargs);  // v1
template<class T> __global__ void opus_attn_gfx1201_kernel_v2(opus_attn_kargs);  // v2
template<class T> __global__ void opus_attn_gfx1201_kernel_v3(opus_attn_kargs);  // v3
template<class T> __global__ void opus_attn_gfx1201_kernel_v4(opus_attn_kargs);  // v4
template<class T> __global__ void opus_attn_gfx1201_kernel_v5(opus_attn_kargs);  // v5
template<class T> __global__ void opus_attn_gfx1201_kernel_v6(opus_attn_kargs);  // v6
template<class T> __global__ void opus_attn_gfx1201_kernel_v7(opus_attn_kargs);  // v7
template<class T> __global__ void opus_attn_gfx1201_kernel_v8(opus_attn_kargs);  // v8
template<class T> __global__ void opus_attn_gfx1201_kernel_v9(opus_attn_kargs);  // v9
template<class T> __global__ void opus_attn_gfx1201_kernel_v10(opus_attn_kargs);  // v10
template<class T> __global__ void opus_attn_gfx1201_kernel_v11(opus_attn_kargs); // v11
template<class T> __global__ void opus_attn_gfx1201_kernel_v12(opus_attn_kargs); // v12
template<class T> __global__ void opus_attn_gfx1201_kernel_v13(opus_attn_kargs); // v13
template<class T> __global__ void opus_attn_gfx1201_kernel_v14(opus_attn_kargs); // v14
template<class T> __global__ void opus_attn_gfx1201_kernel_v15(opus_attn_kargs); // v15
template<class T> __global__ void opus_attn_gfx1201_kernel_v16(opus_attn_kargs); // v16
template<class T> __global__ void opus_attn_gfx1201_kernel_v17(opus_attn_kargs); // v17
template<class T> __global__ void opus_attn_gfx1201_kernel_v18(opus_attn_kargs); // v18
template<class T> __global__ void opus_attn_gfx1201_kernel_v19(opus_attn_kargs); // v19
template<class T> __global__ void opus_attn_gfx1201_kernel_v20(opus_attn_kargs); // v20
template<class T> __global__ void opus_attn_gfx1201_kernel_v21(opus_attn_kargs); // v21
template<class T> __global__ void opus_attn_gfx1201_kernel_v22(opus_attn_kargs); // v22
template<class T> __global__ void opus_attn_gfx1201_kernel_v23(opus_attn_kargs); // v23
template<class T> __global__ void opus_attn_gfx1201_kernel_v24(opus_attn_kargs); // v24
template<class T> __global__ void opus_attn_gfx1201_kernel_v25(opus_attn_kargs); // v25
template<class T> __global__ void opus_attn_gfx1201_kernel_v26(opus_attn_kargs); // v26
template<class T> __global__ void opus_attn_gfx1201_kernel_v27(opus_attn_kargs); // v27
template<class T> __global__ void opus_attn_gfx1201_kernel_v28(opus_attn_kargs); // v28
template<class T> __global__ void opus_attn_gfx1201_kernel_v29(opus_attn_kargs); // v29
template<class T> __global__ void opus_attn_gfx1201_kernel_v30(opus_attn_kargs); // v30
template<class T> __global__ void opus_attn_gfx1201_kernel_v31(opus_attn_kargs); // v31
template<class T> __global__ void opus_attn_gfx1201_kernel_v32(opus_attn_kargs); // v32
__global__ void opus_attn_gfx1201_kernel_v33(opus_attn_kargs); // v33
template<class T> __global__ void opus_attn_gfx1201_kernel_v34(opus_attn_kargs);  // v34
template<class T> __global__ void opus_attn_gfx1201_kernel_v35(opus_attn_kargs);  // v35
template<class T> __global__ void opus_attn_gfx1201_kernel_v36(opus_attn_kargs);  // v36
template<class T> __global__ void opus_attn_gfx1201_kernel_v37(opus_attn_kargs);  // v37
template<class T> __global__ void opus_attn_gfx1201_kernel_v38(opus_attn_kargs);  // v38
template<class T> __global__ void opus_attn_gfx1201_kernel_v39(opus_attn_kargs);  // v39
template<class T> __global__ void opus_attn_gfx1201_kernel_v40(opus_attn_kargs);
template<class T> __global__ void opus_attn_gfx1201_kernel_v41(opus_attn_kargs);
template<class T> __global__ void opus_attn_gfx1201_kernel_v42(opus_attn_kargs);
template<class T> __global__ void opus_attn_gfx1201_kernel_v43(opus_attn_kargs);  // v43  // v42  // v41  // v40 // v34
template<class T> __global__ void opus_attn_gfx1201_kernel_v44(opus_attn_kargs);
template<class T> __global__ void opus_attn_gfx1201_kernel_v45(opus_attn_kargs);
template<class T> __global__ void opus_attn_gfx1201_kernel_v48(opus_attn_kargs);
template<class T> __global__ void opus_attn_gfx1201_kernel_v100(opus_attn_kargs); // v100: v63 K-prefetch + v43 batched pre-PV rescale
template<class T> __global__ void opus_attn_gfx1201_kernel_v101(opus_attn_kargs); // v101: v63 K-prefetch + full V SW-pipeline (REGRESSED 85.66)
template<class T> __global__ void opus_attn_gfx1201_kernel_v102(opus_attn_kargs); // v102: v100 + asymmetric dt=0 V-load hoist under softmax
template<class T> __global__ void opus_attn_gfx1201_kernel_v103(opus_attn_kargs); // v103: v100 + LATE single-fragment dt=0 V hoist before bf16 pack
template<class T> __global__ void opus_attn_gfx1201_kernel_v104(opus_attn_kargs); // v104: v100 + dt=0 PV prologue (both V operands loaded, dt=0 WMMAs fire before dt>=1 loads)
template<class T> __global__ void opus_attn_gfx1201_kernel_v105(opus_attn_kargs); // v105: v104 + interleaved dt=0 prologue (load v0 -> WMMA0 -> load v1 -> WMMA1)
template<class T> __global__ void opus_attn_gfx1201_kernel_v106(opus_attn_kargs); // v106: v100 + QKT accumulator split (even/odd D-tiles) -> 4-way matrix-pipe ILP in QKT
template<class T> __global__ void opus_attn_gfx1201_kernel_v107(opus_attn_kargs); // v107: v100 + distance-2 QKT K-prefetch (2-slot rotating buffer, same 2 score accumulators)
template<class T> __global__ void opus_attn_gfx1201_kernel_v108(opus_attn_kargs); // v108: v100 + cross-tile head-K prefetch across PV->QKT boundary (+8 VGPR, hides the one exposed head load)
template<class T> __global__ void opus_attn_gfx1201_kernel_v109(opus_attn_kargs); // v109: v100 MINUS QKT K double-buffer (-8 VGPR, occupancy play; HW scoreboard hides K loads)
template<class T> __global__ void opus_attn_gfx1201_kernel_v110(opus_attn_kargs); // v110: v100 + split two-pass PV (8 independent WMMAs/pass, no RAW stall) + deferred s1/sum overlap
template<class T> __global__ void opus_attn_gfx1201_kernel_v111(opus_attn_kargs); // v111: v100 + chunked (2-D-tile) pre-PV rescale interleaved with PV (tighter O live ranges, bit-identical)
template<class T> __global__ void opus_attn_gfx1201_kernel_v112(opus_attn_kargs); // v112: v100 + two-pass PV split (8 independent WMMAs/pass, breaks per-accum RAW hazard, bit-exact)
template<class T> __global__ void opus_attn_gfx1201_kernel_v113(opus_attn_kargs); // v113: v111 with CHUNK=1 (per-fragment rescale interleaved with PV WMMAs, finest-grained VALU-in-shadow)
template<class T> __global__ void opus_attn_gfx1201_kernel_v114(opus_attn_kargs); // v114: v111 CHUNK=2 + per-chunk V-load prefetch (issue all chunk V loads, then rescale, then WMMAs)
template<class T> __global__ void opus_attn_gfx1201_kernel_v115(opus_attn_kargs); // v115: v104 generalized -> uniform branch-free interleaved PV loop (load v0->WMMA0->load v1->WMMA1 every D-tile)
template<class T> __global__ void opus_attn_gfx1201_kernel_v116(opus_attn_kargs); // v116: v111 + KV-block SW pipeline (next-tile QKT WMMAs overlap current-tile exp2 softmax; matrix pipe fed during the transcendental storm)
template<class T> __global__ void opus_attn_gfx1201_kernel_v117(opus_attn_kargs); // v117: v100 + cross-tile head-K prefetch issued in the softmax VALU window (idle memory pipe; +8 VGPR, bit-exact)
template<class T> __global__ void opus_attn_gfx1201_kernel_v118(opus_attn_kargs); // v118: v100 + distance-2 LATE QKT K-prefetch (kt+2 loads after both WMMAs, in-place slot recycle, v100 footprint)
template<class T> __global__ void opus_attn_gfx1201_kernel_v119(opus_attn_kargs); // v119: v100 with SINGLE-FRAGMENT QKT K prefetch (only k0 prefetched, k1 in-loop) -> halves K double-buffer VGPR, bit-exact
template<class T> __global__ void opus_attn_gfx1201_kernel_v120(opus_attn_kargs); // v120: byte-for-byte v100 body + __launch_bounds__(256,2) min-2-WG/CU hint -> lower VGPR target, 7->8 waves/SIMD, bit-exact
template<class T> __global__ void opus_attn_gfx1201_kernel_v121(opus_attn_kargs); // v121: v100 + symmetric V software-pipeline in PV phase (mirror of QKT K-prefetch), arithmetically byte-identical to v100
template<class T> __global__ void opus_attn_gfx1201_kernel_v122(opus_attn_kargs); // v122: v111's fast chunked-rescale PV schedule + #pragma clang fp contract(off) -> recover v111 speed BIT-EXACT vs v100
template<class T> __global__ void opus_attn_gfx1201_kernel_v123(opus_attn_kargs); // v123: v100 monolithic rescale (bit-exact) + CHUNK=2 PV grouped loads-then-WMMAs (V-load MLP, tighter V live ranges)
template<class T> __global__ void opus_attn_gfx1201_kernel_v124(opus_attn_kargs); // v124: v123 bit-exact base + CHUNK=2 double-buffered V SW-pipeline (v123 grouped MLP + v121 cross-chunk prefetch combined), bit-exact vs v100
template<class T> __global__ void opus_attn_gfx1201_kernel_v125(opus_attn_kargs); // v125: v111 fast chunked-rescale PV schedule + FUNCTION-WIDE #pragma clang fp contract(off) (v122 scoped it too narrowly) -> recover 91-92 TFLOPS bit-exact
__global__ void v_transpose_kernel(const bf16_t*, bf16_t*, int, int, int, int);

template<int BM, int BN, class K>
static void launch_(opus_attn_kargs k, K kern) {
    using T = opus_attn_traits<BM, BN, 128>;
    const int n_blocks = k.N / T::BLOCK_M;
    const dim3 grid(n_blocks, k.H, k.B);
    const dim3 block(T::BLOCK_SIZE);
    kern<<<grid, block, 0, 0>>>(k);
}


static void launch_v49_asm(opus_attn_kargs k) {
    static hipModule_t   s_mod  = nullptr;
    static hipFunction_t s_func = nullptr;
    if (!s_mod) {
        HIP_CALL(hipModuleLoad(&s_mod, "build/attn_v49.hsaco"));
        HIP_CALL(hipModuleGetFunction(&s_func, s_mod,
            "_Z28opus_attn_gfx1201_kernel_v49I16opus_attn_traitsILi128ELi32ELi128EEEv15opus_attn_kargs"));
    }
    using T = opus_attn_traits<128, 32, 128>;
    const int n_blocks = k.N / T::BLOCK_M;
    void* kptr = &k;
    HIP_CALL(hipModuleLaunchKernel(
        s_func,
        n_blocks, k.H, k.B,
        T::BLOCK_SIZE, 1, 1,
        0, 0, &kptr, NULL));
}

static void run_opus_attn_gfx1201(int version, opus_attn_kargs k) {
    switch (version) {
        case 0: launch_<16, 16>(k, opus_attn_gfx1201_kernel   <opus_attn_traits<16, 16, 128>>); break;
        case 1: launch_<64, 16>(k, opus_attn_gfx1201_kernel_v1<opus_attn_traits<64, 16, 128>>); break;
        case 2: launch_<64, 64>(k, opus_attn_gfx1201_kernel_v2<opus_attn_traits<64, 64, 128>>); break;
        case 3: launch_<64, 16>(k, opus_attn_gfx1201_kernel_v3<opus_attn_traits<64, 16, 128>>); break;
        case 4: launch_<16, 16>(k, opus_attn_gfx1201_kernel_v4<opus_attn_traits<16, 16, 128>>); break;
        case 5: launch_<16, 32>(k, opus_attn_gfx1201_kernel_v5<opus_attn_traits<16, 32, 128>>); break;
        case 6: launch_<16, 16>(k, opus_attn_gfx1201_kernel_v6<opus_attn_traits<16, 16, 128>>); break;
        case 7: launch_<16, 16>(k, opus_attn_gfx1201_kernel_v7<opus_attn_traits<16, 16, 128>>); break;
        case 8: launch_<16, 32>(k, opus_attn_gfx1201_kernel_v8<opus_attn_traits<16, 32, 128>>); break;
        case 9: launch_<16, 16>(k, opus_attn_gfx1201_kernel_v9<opus_attn_traits<16, 16, 128>>); break;
        case 10: launch_<16, 32>(k, opus_attn_gfx1201_kernel_v10<opus_attn_traits<16, 32, 128>>); break;
        case 11: launch_<16, 16>(k, opus_attn_gfx1201_kernel_v11<opus_attn_traits<16, 16, 128>>); break;
        case 12: launch_<16, 16>(k, opus_attn_gfx1201_kernel_v12<opus_attn_traits<16, 16, 128>>); break;
        case 13: launch_<16, 32>(k, opus_attn_gfx1201_kernel_v13<opus_attn_traits<16, 32, 128>>); break;
        case 14: launch_<16, 32>(k, opus_attn_gfx1201_kernel_v14<opus_attn_traits<16, 32, 128>>); break;
        case 15: launch_<16, 32>(k, opus_attn_gfx1201_kernel_v15<opus_attn_traits<16, 32, 128>>); break;
        case 16: launch_<16, 32>(k, opus_attn_gfx1201_kernel_v16<opus_attn_traits<16, 32, 128>>); break;
        case 17: launch_<16, 64>(k, opus_attn_gfx1201_kernel_v17<opus_attn_traits<16, 64, 128>>); break;
        case 18: launch_<16, 64>(k, opus_attn_gfx1201_kernel_v18<opus_attn_traits<16, 64, 128>>); break;
        case 19: launch_<32, 32>(k, opus_attn_gfx1201_kernel_v19<opus_attn_traits<32, 32, 128>>); break;
        case 20: launch_<128, 16>(k, opus_attn_gfx1201_kernel_v20<opus_attn_traits<128, 16, 128>>); break;
        case 21: launch_<128, 16>(k, opus_attn_gfx1201_kernel_v21<opus_attn_traits<128, 16, 128>>); break;
        case 22: launch_<128, 16>(k, opus_attn_gfx1201_kernel_v22<opus_attn_traits<128, 16, 128>>); break;
        case 23: launch_<128, 16>(k, opus_attn_gfx1201_kernel_v23<opus_attn_traits<128, 16, 128>>); break;
        case 24: launch_<128, 16>(k, opus_attn_gfx1201_kernel_v24<opus_attn_traits<128, 16, 128>>); break;
        case 25: launch_<128, 16>(k, opus_attn_gfx1201_kernel_v25<opus_attn_traits<128, 16, 128>>); break;
        case 26: launch_<128, 32>(k, opus_attn_gfx1201_kernel_v26<opus_attn_traits<128, 32, 128>>); break;
        case 27: launch_<128, 16>(k, opus_attn_gfx1201_kernel_v27<opus_attn_traits<128, 16, 128>>); break;
        case 28: launch_<128, 16>(k, opus_attn_gfx1201_kernel_v28<opus_attn_traits<128, 16, 128>>); break;
        case 29: launch_<128, 16>(k, opus_attn_gfx1201_kernel_v29<opus_attn_traits<128, 16, 128>>); break;
        case 30: launch_<128, 32>(k, opus_attn_gfx1201_kernel_v30<opus_attn_traits<128, 32, 128>>); break;
        case 31: launch_<128, 16>(k, opus_attn_gfx1201_kernel_v31<opus_attn_traits<128, 16, 128>>); break;
        case 32: launch_<128, 16>(k, opus_attn_gfx1201_kernel_v32<opus_attn_traits<128, 16, 128>>); break;
        case 33: { const int nb = k.N / 128; hipLaunchKernelGGL(opus_attn_gfx1201_kernel_v33, dim3(nb,k.H,k.B), dim3(512), 0, 0, k); } break;
        case 34: launch_<128, 16>(k, opus_attn_gfx1201_kernel_v34<opus_attn_traits<128, 16, 128>>); break;
        case 35: launch_<128, 16>(k, opus_attn_gfx1201_kernel_v35<opus_attn_traits<128, 16, 128>>); break;
        case 36: launch_<128, 16>(k, opus_attn_gfx1201_kernel_v36<opus_attn_traits<128, 16, 128>>); break;
        case 37: launch_<128, 16>(k, opus_attn_gfx1201_kernel_v37<opus_attn_traits<128, 16, 128>>); break;
        case 38: launch_<128, 16>(k, opus_attn_gfx1201_kernel_v38<opus_attn_traits<128, 16, 128>>); break;
        case 39: launch_<128, 32>(k, opus_attn_gfx1201_kernel_v39<opus_attn_traits<128, 32, 128>>); break;
        case 40: launch_<128, 64>(k, opus_attn_gfx1201_kernel_v40<opus_attn_traits<128, 64, 128>>); break;
        case 41: launch_<128, 32>(k, opus_attn_gfx1201_kernel_v41<opus_attn_traits<128, 32, 128>>); break;
        case 42: launch_<128, 32>(k, opus_attn_gfx1201_kernel_v42<opus_attn_traits<128, 32, 128>>); break;
        case 43: launch_<128, 32>(k, opus_attn_gfx1201_kernel_v43<opus_attn_traits<128, 32, 128>>); break;
        case 44: launch_<128, 32>(k, opus_attn_gfx1201_kernel_v44<opus_attn_traits<128, 32, 128>>); break;
        case 45: launch_<128, 32>(k, opus_attn_gfx1201_kernel_v45<opus_attn_traits<128, 32, 128>>); break;
        case 48: launch_<128, 32>(k, opus_attn_gfx1201_kernel_v48<opus_attn_traits<128, 32, 128>>); break;
        case 49: launch_v49_asm(k); break;
        case 100: launch_<128, 32>(k, opus_attn_gfx1201_kernel_v100<opus_attn_traits<128, 32, 128>>); break;
        case 101: launch_<128, 32>(k, opus_attn_gfx1201_kernel_v101<opus_attn_traits<128, 32, 128>>); break;
        case 102: launch_<128, 32>(k, opus_attn_gfx1201_kernel_v102<opus_attn_traits<128, 32, 128>>); break;
        case 103: launch_<128, 32>(k, opus_attn_gfx1201_kernel_v103<opus_attn_traits<128, 32, 128>>); break;
        case 104: launch_<128, 32>(k, opus_attn_gfx1201_kernel_v104<opus_attn_traits<128, 32, 128>>); break;
        case 105: launch_<128, 32>(k, opus_attn_gfx1201_kernel_v105<opus_attn_traits<128, 32, 128>>); break;
        case 106: launch_<128, 32>(k, opus_attn_gfx1201_kernel_v106<opus_attn_traits<128, 32, 128>>); break;
        case 107: launch_<128, 32>(k, opus_attn_gfx1201_kernel_v107<opus_attn_traits<128, 32, 128>>); break;
        case 108: launch_<128, 32>(k, opus_attn_gfx1201_kernel_v108<opus_attn_traits<128, 32, 128>>); break;
        case 109: launch_<128, 32>(k, opus_attn_gfx1201_kernel_v109<opus_attn_traits<128, 32, 128>>); break;
        case 110: launch_<128, 32>(k, opus_attn_gfx1201_kernel_v110<opus_attn_traits<128, 32, 128>>); break;
        case 111: launch_<128, 32>(k, opus_attn_gfx1201_kernel_v111<opus_attn_traits<128, 32, 128>>); break;
        case 112: launch_<128, 32>(k, opus_attn_gfx1201_kernel_v112<opus_attn_traits<128, 32, 128>>); break;
        case 113: launch_<128, 32>(k, opus_attn_gfx1201_kernel_v113<opus_attn_traits<128, 32, 128>>); break;
        case 114: launch_<128, 32>(k, opus_attn_gfx1201_kernel_v114<opus_attn_traits<128, 32, 128>>); break;
        case 115: launch_<128, 32>(k, opus_attn_gfx1201_kernel_v115<opus_attn_traits<128, 32, 128>>); break;
        case 116: launch_<128, 32>(k, opus_attn_gfx1201_kernel_v116<opus_attn_traits<128, 32, 128>>); break;
        case 117: launch_<128, 32>(k, opus_attn_gfx1201_kernel_v117<opus_attn_traits<128, 32, 128>>); break;
        case 118: launch_<128, 32>(k, opus_attn_gfx1201_kernel_v118<opus_attn_traits<128, 32, 128>>); break;
        case 119: launch_<128, 32>(k, opus_attn_gfx1201_kernel_v119<opus_attn_traits<128, 32, 128>>); break;
        case 120: launch_<128, 32>(k, opus_attn_gfx1201_kernel_v120<opus_attn_traits<128, 32, 128>>); break;
        case 121: launch_<128, 32>(k, opus_attn_gfx1201_kernel_v121<opus_attn_traits<128, 32, 128>>); break;
        case 122: launch_<128, 32>(k, opus_attn_gfx1201_kernel_v122<opus_attn_traits<128, 32, 128>>); break;
        case 123: launch_<128, 32>(k, opus_attn_gfx1201_kernel_v123<opus_attn_traits<128, 32, 128>>); break;
        case 124: launch_<128, 32>(k, opus_attn_gfx1201_kernel_v124<opus_attn_traits<128, 32, 128>>); break;
        case 125: launch_<128, 32>(k, opus_attn_gfx1201_kernel_v125<opus_attn_traits<128, 32, 128>>); break;
        default: fprintf(stderr, "unknown --version=%d\n", version); std::exit(1);
    }
}

static void cpu_reference(int B, int H, int N, int D,
                          const bf16_t* Q, const bf16_t* K, const bf16_t* V,
                          bf16_t* O, fp32_t scale)
{
    std::vector<fp32_t> S(N), P(N);
    for (int b = 0; b < B; ++b)
    for (int h = 0; h < H; ++h) {
        const bf16_t* Qbh = Q + (b * H + h) * N * D;
        const bf16_t* Kbh = K + (b * H + h) * N * D;
        const bf16_t* Vbh = V + (b * H + h) * N * D;
        bf16_t*       Obh = O + (b * H + h) * N * D;
        for (int m = 0; m < N; ++m) {
            fp32_t row_max = -3.4e38f;
            for (int n = 0; n < N; ++n) {
                fp32_t s = 0.0f;
                for (int d = 0; d < D; ++d) s += bf16_to_f32(Qbh[m*D+d]) * bf16_to_f32(Kbh[n*D+d]);
                s *= scale;
                S[n] = s;
                if (s > row_max) row_max = s;
            }
            fp32_t row_sum = 0.0f;
            for (int n = 0; n < N; ++n) {
                P[n] = std::exp(S[n] - row_max);
                row_sum += P[n];
            }
            const fp32_t inv = (row_sum > 0.0f) ? (1.0f / row_sum) : 0.0f;
            for (int d = 0; d < D; ++d) {
                fp32_t o = 0.0f;
                for (int n = 0; n < N; ++n) o += P[n] * bf16_to_f32(Vbh[n*D+d]);
                Obh[m*D+d] = bf16_from_f32(o * inv);
            }
        }
    }
}

int main(int argc, char** argv) {
    int B = 1, H = 1, N = 256, D = 128;
    int verify = 1, warmups = 5, iters = 100, version = 1;
    for (int i = 1; i < argc; ++i) {
        const char* a = argv[i];
        auto eq = [&](const char* k, int& dst) {
            auto kn = std::strlen(k);
            if (std::strncmp(a, k, kn) == 0) { dst = std::atoi(a[kn] == '=' ? a + kn + 1 : argv[++i]); return true; }
            return false;
        };
        if (eq("-b", B) || eq("--batch", B)) continue;
        if (eq("-h", H) || eq("--heads", H)) continue;
        if (eq("-n", N) || eq("--seq",   N)) continue;
        if (eq("-d", D) || eq("--dim",   D)) continue;
        if (eq("--verify", verify)) continue;
        if (eq("--iters",  iters))  continue;
        if (eq("--version", version)) continue;
    }
    if (D != 128) { fprintf(stderr, "only D=128 supported (got %d)\n", D); return 1; }
    if (N % 64)   { fprintf(stderr, "N must be a multiple of 64 (got %d)\n", N); return 1; }
    printf("running version v%d  B=%d H=%d N=%d D=%d\n", version, B, H, N, D);

    fp32_t scale = 1.0f / std::sqrt((fp32_t)D);
    size_t sz_qkvo = (size_t)B * H * N * D;
    std::vector<bf16_t> hQ(sz_qkvo), hK(sz_qkvo), hV(sz_qkvo), hO(sz_qkvo), hRef(sz_qkvo);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> u(-0.5f, 0.5f);
    for (auto& x : hQ) x = bf16_from_f32(u(rng));
    for (auto& x : hK) x = bf16_from_f32(u(rng));
    for (auto& x : hV) x = bf16_from_f32(u(rng));

    bf16_t *dQ, *dK, *dV, *dO, *dVT;
    HIP_CALL(hipMalloc(&dQ, sz_qkvo * sizeof(bf16_t)));
    HIP_CALL(hipMalloc(&dK, sz_qkvo * sizeof(bf16_t)));
    HIP_CALL(hipMalloc(&dV, sz_qkvo * sizeof(bf16_t)));
    HIP_CALL(hipMalloc(&dVT, sz_qkvo * sizeof(bf16_t)));
    HIP_CALL(hipMalloc(&dO, sz_qkvo * sizeof(bf16_t)));
    HIP_CALL(hipMemcpy(dQ, hQ.data(), sz_qkvo * sizeof(bf16_t), hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(dK, hK.data(), sz_qkvo * sizeof(bf16_t), hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(dV, hV.data(), sz_qkvo * sizeof(bf16_t), hipMemcpyHostToDevice));

    // Pre-transpose V → V_T for v9 (one-time, not timed in bench loop)
    {
        const int threads = 256;
        const int blocks = (sz_qkvo + threads - 1) / threads;
        v_transpose_kernel<<<blocks, threads>>>(dV, dVT, B, H, N, D);
        HIP_CALL(hipDeviceSynchronize());
    }

    opus_attn_kargs kargs{};
    kargs.ptr_q = dQ; kargs.ptr_k = dK; kargs.ptr_v = ((version == 9 || version == 10 || version == 34 || version == 35 || version == 36 || version == 37 || version == 38 || version == 39 || version == 40 || version == 41 || version == 42 || version == 43 || version == 44 || version == 45 || version == 48 || version == 49 || version == 100 || version == 101 || version == 102 || version == 103 || version == 104 || version == 105 || version == 106 || version == 107 || version == 110) ? dVT : dV); kargs.ptr_o = dO;
    kargs.B = B; kargs.H = H; kargs.N = N; kargs.D = D; kargs.scale = scale;

    // Warmup
    for (int i = 0; i < warmups; ++i) run_opus_attn_gfx1201(version, kargs);
    HIP_CALL(hipDeviceSynchronize());

    // Verify
    if (verify) {
        HIP_CALL(hipMemcpy(hO.data(), dO, sz_qkvo * sizeof(bf16_t), hipMemcpyDeviceToHost));
        cpu_reference(B, H, N, D, hQ.data(), hK.data(), hV.data(), hRef.data(), scale);
        double max_abs = 0, mean_abs = 0, max_rel = 0;
        int n_bad = 0;
        for (size_t i = 0; i < sz_qkvo; ++i) {
            double a = (double)bf16_to_f32(hO[i]), r = (double)bf16_to_f32(hRef[i]);
            double d = std::abs(a - r);
            max_abs = std::max(max_abs, d);
            mean_abs += d;
            double rel = std::abs(r) > 1e-3 ? d / std::abs(r) : 0.0;
            max_rel = std::max(max_rel, rel);
            if (d > 0.05) ++n_bad;
        }
        mean_abs /= sz_qkvo;
        printf("VERIFY B=%d H=%d N=%d D=%d  max_abs=%.4f  mean_abs=%.5f  max_rel=%.4f  n_bad(>0.05)=%d/%zu\n",
               B, H, N, D, max_abs, mean_abs, max_rel, n_bad, sz_qkvo);
    }

    // Bench
    hipEvent_t ev0, ev1;
    HIP_CALL(hipEventCreate(&ev0));
    HIP_CALL(hipEventCreate(&ev1));
    HIP_CALL(hipEventRecord(ev0));
    for (int i = 0; i < iters; ++i) run_opus_attn_gfx1201(version, kargs);
    HIP_CALL(hipEventRecord(ev1));
    HIP_CALL(hipEventSynchronize(ev1));
    float ms = 0;
    HIP_CALL(hipEventElapsedTime(&ms, ev0, ev1));
    ms /= iters;
    // FLOPS: 4 * B * H * N * N * D (2 matmuls, mul+add each)
    double tflops = 4.0 * B * H * (double)N * N * D / (ms * 1e9);
    printf("BENCH  iters=%d  avg=%.3f ms  %.2f TFLOPS\n", iters, ms, tflops);

    HIP_CALL(hipFree(dQ));
    HIP_CALL(hipFree(dK));
    HIP_CALL(hipFree(dV));
    HIP_CALL(hipFree(dVT));
    HIP_CALL(hipFree(dO));
    return 0;
}
