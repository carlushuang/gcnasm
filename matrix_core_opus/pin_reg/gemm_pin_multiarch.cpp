// Opus-style pinned GEMM microkernel, portable across CDNA (MFMA) and RDNA4
// (WMMA). A K-loop accumulates C += A_k * B_k over a 16x16x16 f16 matrix core;
// the A/B input fragments and the accumulator are register-pinned via the
// amdgpu_pin_{vgpr,agpr} attributes (no inline asm).
//
//   CDNA  (gfx942/gfx950): A/B -> AGPR, C -> VGPR  => v_mfma v[C], a[A], a[B]
//   RDNA4 (gfx1201):       A/B -> VGPR, C -> VGPR  => v_wmma v[C], v[A], v[B]
//                          (RDNA has no AGPR file, so A/B are pinned to VGPRs.)
//
// The pinned kernel is bit-identical to the plain one (same math, only register
// placement changes); this file both dumps ISA and self-checks on the GPU.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstring>

#if __has_builtin(__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12)
#define RDNA_WMMA 1
using afrag = _Float16 __attribute__((ext_vector_type(8))); // A/B: 4 dwords
using cfrag = float    __attribute__((ext_vector_type(8))); // C:   8 dwords
#define MMA(a, b, c) __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c)
#else
using afrag = _Float16 __attribute__((ext_vector_type(4))); // A/B: 2 dwords
using cfrag = float    __attribute__((ext_vector_type(4))); // C:   4 dwords
#define MMA(a, b, c) __builtin_amdgcn_mfma_f32_16x16x16f16(a, b, c, 0, 0, 0)
#endif

static constexpr int LANES = 32;
static constexpr int AN = sizeof(afrag) / sizeof(_Float16); // f16 per A/B frag
static constexpr int CN = sizeof(cfrag) / sizeof(float);    // f32 per C frag

// One 16x16x16 f16 matrix-core op = a GEMM tile step. Pinning the operands of a
// single step is the unit the attribute controls; a full GEMM reuses the same
// pinned registers across the K loop. (Note: with an unrolled/pipelined K chain
// the loads are hoisted and become simultaneously live, so only the first step
// keeps the exact registers and the rest hit the overlap->soft fallback; a
// register-planned kernel assigns distinct pins per live fragment.)
static constexpr int KSTEPS = 1;

__global__ void gemm_plain(const _Float16 *A, const _Float16 *B, float *C) {
  int lane = __builtin_amdgcn_workitem_id_x();
  cfrag c = {};
#pragma unroll
  for (int k = 0; k < KSTEPS; ++k) {
    afrag a = *reinterpret_cast<const afrag *>(A + (k * LANES + lane) * AN);
    afrag b = *reinterpret_cast<const afrag *>(B + (k * LANES + lane) * AN);
    c = MMA(a, b, c);
  }
  *reinterpret_cast<cfrag *>(C + lane * CN) = c;
}

__global__ void gemm_pinned(const _Float16 *A, const _Float16 *B, float *C) {
  int lane = __builtin_amdgcn_workitem_id_x();
  __attribute__((amdgpu_pin_vgpr(0))) cfrag c = {}; // accumulator -> VGPR
#pragma unroll
  for (int k = 0; k < KSTEPS; ++k) {
#ifdef RDNA_WMMA
    __attribute__((amdgpu_pin_vgpr(8)))  afrag a =
        *reinterpret_cast<const afrag *>(A + (k * LANES + lane) * AN);
    __attribute__((amdgpu_pin_vgpr(12))) afrag b =
        *reinterpret_cast<const afrag *>(B + (k * LANES + lane) * AN);
#else
    __attribute__((amdgpu_pin_agpr(0)))  afrag a =
        *reinterpret_cast<const afrag *>(A + (k * LANES + lane) * AN);
    __attribute__((amdgpu_pin_agpr(2)))  afrag b =
        *reinterpret_cast<const afrag *>(B + (k * LANES + lane) * AN);
#endif
    c = MMA(a, b, c);
  }
  *reinterpret_cast<cfrag *>(C + lane * CN) = c;
}

int main() {
  const int NA = KSTEPS * LANES * AN, NC = LANES * CN;
  _Float16 *hA = new _Float16[NA], *hB = new _Float16[NA];
  for (int i = 0; i < NA; ++i) { hA[i] = (_Float16)0.5f; hB[i] = (_Float16)0.25f; }
  _Float16 *dA, *dB; float *dC;
  hipMalloc(&dA, NA * 2); hipMalloc(&dB, NA * 2); hipMalloc(&dC, NC * 4);
  hipMemcpy(dA, hA, NA * 2, hipMemcpyHostToDevice);
  hipMemcpy(dB, hB, NA * 2, hipMemcpyHostToDevice);
  float *p = new float[NC], *q = new float[NC];
  gemm_plain <<<1, LANES>>>(dA, dB, dC); hipDeviceSynchronize();
  hipMemcpy(p, dC, NC * 4, hipMemcpyDeviceToHost);
  gemm_pinned<<<1, LANES>>>(dA, dB, dC); hipDeviceSynchronize();
  hipMemcpy(q, dC, NC * 4, hipMemcpyDeviceToHost);
  int mism = memcmp(p, q, NC * sizeof(float));
  printf("  plain  C[0..3]= %.2f %.2f %.2f %.2f\n", p[0], p[1], p[2], p[3]);
  printf("  pinned C[0..3]= %.2f %.2f %.2f %.2f\n", q[0], q[1], q[2], q[3]);
  printf("pinned vs plain: %s\nRESULT: %s\n", mism ? "DIFF" : "BIT-IDENTICAL",
         mism ? "FAIL" : "PASS");
  return mism != 0;
}
