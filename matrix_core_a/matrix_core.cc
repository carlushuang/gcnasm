#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <iostream>
#include <math.h>
#include <numeric>
#include <random>
#include <stdio.h>
#include <stdlib.h>
#define HALF
#ifdef HALF
#include "half.hpp"
#endif

#include <ck_tile/core.hpp>

#define LOCAL_SCRATCH 0
#define RAND_INT 0

#define MAX(x, y) ((x) > (y) ? (x) : (y))
#define HIP_CALL(call)                                                         \
  do {                                                                         \
    hipError_t err = call;                                                     \
    if (err != hipSuccess) {                                                   \
      printf("[hiperror](%d) fail to call %s", (int)err, #call);               \
      exit(0);                                                                 \
    }                                                                          \
  } while (0)

#define ABS(x) ((x) > 0 ? (x) : -(x))

using fp32_t = float;
using fp16_t = _Float16;
using float16 = half_float::half; // cpu type

using fp16x2_t = fp16_t __attribute__((ext_vector_type(2)));
using fp16x4_t = fp16_t __attribute__((ext_vector_type(4)));
using fp16x8_t = fp16_t __attribute__((ext_vector_type(8)));
using fp16x16_t = fp16_t __attribute__((ext_vector_type(16)));
using fp32x4_t = fp32_t __attribute__((ext_vector_type(4)));
using fp32x16_t = fp32_t __attribute__((ext_vector_type(16)));

using int32x4_t = int32_t __attribute__((ext_vector_type(4)));
#define BUFFER_LOAD_DWORD3 0x00020000 // This is valid for
struct buffer_resource {
  const void *ptr;
  uint32_t range;
  uint32_t config;
};
__device__ int32x4_t make_buffer_resource(const void *ptr,
                                          uint32_t size = 0xffffffff) {
  buffer_resource res{ptr, size, BUFFER_LOAD_DWORD3};
  return __builtin_bit_cast(int32x4_t, res);
}

template <int Repeat_M = 1, int Repeat_N = 1, int Repeat_K = 1>
__global__ void
matrix_core_kernel_standard_agpr(const void *__restrict__ ptr_a,
                                 const void *__restrict__ ptr_b,
                                 void *__restrict__ ptr_c, int m, int n, int k,
                                 int stride_a, // stride in unit of pixel
                                 int stride_b, int stride_c) {
  // 32x32x8 gemm, assume only launced 1 wave
  int offset_a = (threadIdx.x / 32 * 4) + (threadIdx.x % 32 * stride_a);
  int offset_b = (threadIdx.x / 32 * 4) + (threadIdx.x % 32 * stride_b);
  constexpr ck_tile::index_t Wave_M = 32;
  constexpr ck_tile::index_t Wave_N = 32;
  constexpr ck_tile::index_t Wave_K = 8;

  auto res_a = make_buffer_resource(ptr_a);
  auto res_b = make_buffer_resource(ptr_b);
  fp16x4_t v_a[Repeat_M * Repeat_K];
  fp16x4_t v_b[Repeat_N * Repeat_K];

  if constexpr (Repeat_K == 18) {
    //  TODO: make sure k is 18*8(144) multiple
    int loop_cnt = __builtin_amdgcn_readfirstlane(k / (Repeat_K * Wave_K));
    fp32x16_t v_c{.0f};
    offset_a *= sizeof(fp16_t);
    offset_b *= sizeof(fp16_t);
    int window_stride = __builtin_amdgcn_readfirstlane(Repeat_K * Wave_K * 2);
    asm volatile(
        ""
        ".Loop%=:\n"
        "buffer_load_dwordx2 a[0 : 1], %2, %4, 0 offen offset:0 * 16\n"
        "buffer_load_dwordx2 a[2 : 3], %2, %4, 0 offen offset:1 * 16\n"
        "buffer_load_dwordx2 a[4 : 5], %2, %4, 0 offen offset:2 * 16\n"
        "buffer_load_dwordx2 a[6 : 7], %2, %4, 0 offen offset:3 * 16\n"
        "buffer_load_dwordx2 a[8 : 9], %2, %4, 0 offen offset:4 * 16\n"
        "buffer_load_dwordx2 a[10:11], %2, %4, 0 offen offset:5 * 16\n"
        "buffer_load_dwordx2 a[12:13], %2, %4, 0 offen offset:6 * 16\n"
        "buffer_load_dwordx2 a[14:15], %2, %4, 0 offen offset:7 * 16\n"
        "buffer_load_dwordx2 a[16:17], %2, %4, 0 offen offset:8 * 16\n"
        "buffer_load_dwordx2 a[18:19], %2, %4, 0 offen offset:9 * 16\n"
        "buffer_load_dwordx2 a[20:21], %2, %4, 0 offen offset:10* 16\n"
        "buffer_load_dwordx2 a[22:23], %2, %4, 0 offen offset:11* 16\n"
        "buffer_load_dwordx2 a[24:25], %2, %4, 0 offen offset:12* 16\n"
        "buffer_load_dwordx2 a[26:27], %2, %4, 0 offen offset:13* 16\n"
        "buffer_load_dwordx2 a[28:29], %2, %4, 0 offen offset:14* 16\n"
        "buffer_load_dwordx2 a[30:31], %2, %4, 0 offen offset:15* 16\n"
        "buffer_load_dwordx2 a[32:33], %2, %4, 0 offen offset:16* 16\n"
        "buffer_load_dwordx2 a[34:35], %2, %4, 0 offen offset:17* 16\n"
        ";------------\n"
        "buffer_load_dwordx2 a[64 + 0 :64 + 1],  %3, %5, 0 offen offset:0 *16\n"
        "buffer_load_dwordx2 a[64 + 2 :64 + 3],  %3, %5, 0 offen offset:1 *16\n"
        "buffer_load_dwordx2 a[64 + 4 :64 + 5],  %3, %5, 0 offen offset:2 *16\n"
        "buffer_load_dwordx2 a[64 + 6 :64 + 7],  %3, %5, 0 offen offset:3 *16\n"
        "buffer_load_dwordx2 a[64 + 8 :64 + 9],  %3, %5, 0 offen offset:4 *16\n"
        "buffer_load_dwordx2 a[64 + 10:64 + 11], %3, %5, 0 offen offset:5 *16\n"
        "buffer_load_dwordx2 a[64 + 12:64 + 13], %3, %5, 0 offen offset:6 *16\n"
        "buffer_load_dwordx2 a[64 + 14:64 + 15], %3, %5, 0 offen offset:7 *16\n"
        "buffer_load_dwordx2 a[64 + 16:64 + 17], %3, %5, 0 offen offset:8 *16\n"
        "buffer_load_dwordx2 a[64 + 18:64 + 19], %3, %5, 0 offen offset:9 *16\n"
        "buffer_load_dwordx2 a[64 + 20:64 + 21], %3, %5, 0 offen offset:10*16\n"
        "buffer_load_dwordx2 a[64 + 22:64 + 23], %3, %5, 0 offen offset:11*16\n"
        "buffer_load_dwordx2 a[64 + 24:64 + 25], %3, %5, 0 offen offset:12*16\n"
        "buffer_load_dwordx2 a[64 + 26:64 + 27], %3, %5, 0 offen offset:13*16\n"
        "buffer_load_dwordx2 a[64 + 28:64 + 29], %3, %5, 0 offen offset:14*16\n"
        "buffer_load_dwordx2 a[64 + 30:64 + 31], %3, %5, 0 offen offset:15*16\n"
        "buffer_load_dwordx2 a[64 + 32:64 + 33], %3, %5, 0 offen offset:16*16\n"
        "buffer_load_dwordx2 a[64 + 34:64 + 35], %3, %5, 0 offen offset:17*16\n"
        "s_waitcnt vmcnt(0)\n"
        ";------------\n"
        "v_mfma_f32_32x32x8f16 %0, a[0 : 1], a[64 + 0 :64 + 1], %0\n"
        "v_mfma_f32_32x32x8f16 %0, a[2 : 3], a[64 + 2 :64 + 3], %0\n"
        "v_mfma_f32_32x32x8f16 %0, a[4 : 5], a[64 + 4 :64 + 5], %0\n"
        "v_mfma_f32_32x32x8f16 %0, a[6 : 7], a[64 + 6 :64 + 7], %0\n"
        "v_mfma_f32_32x32x8f16 %0, a[8 : 9], a[64 + 8 :64 + 9], %0\n"
        "v_mfma_f32_32x32x8f16 %0, a[10:11], a[64 + 10:64 +11], %0\n"
        "v_mfma_f32_32x32x8f16 %0, a[12:13], a[64 + 12:64 +13], %0\n"
        "v_mfma_f32_32x32x8f16 %0, a[14:15], a[64 + 14:64 +15], %0\n"
        "v_mfma_f32_32x32x8f16 %0, a[16:17], a[64 + 16:64 +17], %0\n"
        "v_mfma_f32_32x32x8f16 %0, a[18:19], a[64 + 18:64 +19], %0\n"
        "v_mfma_f32_32x32x8f16 %0, a[20:21], a[64 + 20:64 +21], %0\n"
        "v_mfma_f32_32x32x8f16 %0, a[22:23], a[64 + 22:64 +23], %0\n"
        "v_mfma_f32_32x32x8f16 %0, a[24:25], a[64 + 24:64 +25], %0\n"
        "v_mfma_f32_32x32x8f16 %0, a[26:27], a[64 + 26:64 +27], %0\n"
        "v_mfma_f32_32x32x8f16 %0, a[28:29], a[64 + 28:64 +29], %0\n"
        "v_mfma_f32_32x32x8f16 %0, a[30:31], a[64 + 30:64 +31], %0\n"
        "v_mfma_f32_32x32x8f16 %0, a[32:33], a[64 + 32:64 +33], %0\n"
        "v_mfma_f32_32x32x8f16 %0, a[34:35], a[64 + 34:64 +35], %0\n"
        "v_add_u32 %2, %2, %6\n"
        "v_add_u32 %3, %3, %6\n"
        "s_sub_u32 %1, %1, 1\n"
        "s_cmp_gt_u32 %1, 0\n"
        "s_cbranch_scc1 .Loop%=\n"
        "s_nop 16\n"
        ""
        : "+v"(v_c), "+s"(loop_cnt),
            "+v"(offset_a),
            "+v"(offset_b)
        :  "s"(res_a),
           "s"(res_b),
           "s"(window_stride)
        : "memory", "a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8", "a9",
          "a10", "a11", "a12", "a13", "a14", "a15", "a16", "a17", "a18", "a19",
          "a20", "a21", "a22", "a23", "a24", "a25", "a26", "a27", "a28", "a29",
          "a30", "a31", "a32", "a33", "a34", "a35", "a36", "a37", "a38", "a39",
          "a40", "a41", "a42", "a43", "a44", "a45", "a46", "a47", "a48", "a49",
          "a50", "a51", "a52", "a53", "a54", "a55", "a56", "a57", "a58", "a59",
          "a60", "a61", "a62", "a63", "a64", "a65", "a66", "a67", "a68", "a69",
          "a70", "a71", "a72", "a73", "a74", "a75", "a76", "a77", "a78", "a79",
          "a80", "a81", "a82", "a83", "a84", "a85", "a86", "a87", "a88", "a89",
          "a90", "a91", "a92", "a93", "a94", "a95", "a96", "a97", "a98", "a99",
          "a100", "a101", "a102", "a103", "a104", "a105", "a106", "a107",
          "a108", "a109", "a110", "a111", "a112", "a113", "a114", "a115",
          "a116", "a117", "a118", "a119", "a120", "a121", "a122", "a123",
          "a124", "a125", "a126", "a127", "a128", "a129", "a130", "a131",
          "a132", "a133", "a134", "a135", "a136", "a137", "a138", "a139",
          "a140", "a141", "a142", "a143", "a144", "a145", "a146", "a147",
          "a148", "a149", "a150", "a151", "a152", "a153", "a154", "a155",
          "a156", "a157", "a158", "a159", "a160", "a161", "a162", "a163",
          "a164", "a165", "a166", "a167", "a168", "a169", "a170", "a171",
          "a172", "a173", "a174", "a175", "a176", "a177", "a178", "a179",
          "a180", "a181", "a182", "a183", "a184", "a185", "a186", "a187",
          "a188", "a189", "a190", "a191", "a192", "a193", "a194", "a195",
          "a196", "a197", "a198", "a199", "a200", "a201", "a202", "a203",
          "a204", "a205", "a206", "a207", "a208", "a209", "a210", "a211",
          "a212", "a213", "a214", "a215", "a216", "a217", "a218", "a219",
          "a220", "a221", "a222", "a223", "a224", "a225", "a226", "a227",
          "a228", "a229", "a230", "a231", "a232", "a233", "a234", "a235",
          "a236", "a237", "a238", "a239", "a240", "a241", "a242", "a243",
          "a244", "a245", "a246", "a247", "a248", "a249", "a250", "a251",
          "a252", "a253", "a254", "a255"
    );

    fp16x16_t v_c_f16;
    for (auto i = 0; i < 16; i++) {
      v_c_f16[i] = static_cast<fp16_t>(v_c[i]);
    }

    int col_id_c = threadIdx.x % 32;
    int row_id_c = threadIdx.x / 32 * 4;
    int offset_c = row_id_c * stride_c + col_id_c;

    for (auto i = 0; i < 16; i++) {
      int row_offset = (i % 4) + (i / 4 * 8);
      *(reinterpret_cast<fp16_t *>(ptr_c) + offset_c + row_offset * stride_c) =
          v_c_f16[i];
    }

  } else {
    ck_tile::static_for<0, Repeat_M * Repeat_K, 1>{}([&](auto i_access) {
      constexpr ck_tile::index_t i_k = i_access % Repeat_K;
      constexpr ck_tile::index_t i_m = i_access / Repeat_K;
      constexpr ck_tile::index_t ros = i_m * Repeat_K + i_k;
      const ck_tile::index_t offset_w =
          (i_m * Wave_M * stride_a + i_k * Wave_K + offset_a) * sizeof(fp16_t);

      auto &a = v_a;
      auto &ra = res_a;

      asm volatile("buffer_load_dwordx2 %0, %1, %2, 0 offen offset:%3"
                   : "=v"(a[ros])
                   : "v"(static_cast<int>(offset_w)), "s"(ra), "n"(0)
                   : "memory");
    });

    ck_tile::static_for<0, Repeat_N * Repeat_K, 1>{}([&](auto i_access) {
      constexpr ck_tile::index_t i_k = i_access % Repeat_K;
      constexpr ck_tile::index_t i_n = i_access / Repeat_K;
      constexpr ck_tile::index_t ros = i_n * Repeat_K + i_k;
      const ck_tile::index_t offset_w =
          (i_n * Wave_N * stride_b + i_k * Wave_K + offset_b) * sizeof(fp16_t);

      auto &b = v_b;
      auto &rb = res_b;

      asm volatile("buffer_load_dwordx2 %0, %1, %2, 0 offen offset:%3"
                   : "=v"(b[ros])
                   : "v"(static_cast<int>(offset_w)), "s"(rb), "n"(0)
                   : "memory");
    });

    fp32x16_t v_c[Repeat_M * Repeat_N] = {.0f}; // clear

    asm volatile("s_waitcnt vmcnt(0)" : : : "memory");

    ck_tile::static_for<0, Repeat_M * Repeat_N * Repeat_K, 1>{}(
        [&](auto i_access) {
          auto &a = v_a;
          auto &b = v_b;
          auto &c = v_c;
          // m,n,k
          constexpr ck_tile::index_t i_k = i_access % Repeat_K;
          constexpr ck_tile::index_t i_n = (i_access / Repeat_K) % Repeat_N;
          constexpr ck_tile::index_t i_m = i_access / Repeat_K / Repeat_N;
          constexpr ck_tile::index_t ros_a = i_m * Repeat_K + i_k;
          constexpr ck_tile::index_t ros_b = i_n * Repeat_K + i_k;
          constexpr ck_tile::index_t ros_c = i_m * Repeat_N + i_n;
          asm volatile("v_mfma_f32_32x32x8f16 %0, %1, %2, %3\n"
                       "s_nop 16" // TODO: better resolve data dependency
                       : "+a"(c[ros_c])
                       : "v"(a[ros_a]), "v"(b[ros_b]), "a"(c[ros_c])
                       :);
        });

    ck_tile::static_for<0, Repeat_M * Repeat_N * Repeat_K, 1>{}(
        [&](auto i_access) {
          constexpr ck_tile::index_t i_n = i_access % Repeat_N;
          constexpr ck_tile::index_t i_m = i_access / Repeat_N;
          auto &c = v_c;

          fp16x16_t v_c_f16;
          for (auto i = 0; i < 16; i++) {
            v_c_f16[i] = static_cast<fp16_t>(c[i_m * Repeat_N + i_n][i]);
          }

          int col_id_c = threadIdx.x % 32;
          int row_id_c = threadIdx.x / 32 * 4;
          int offset_c = row_id_c * stride_c + col_id_c +
                         i_m * Wave_M * stride_c + i_n * Wave_N;

          for (auto i = 0; i < 16; i++) {
            int row_offset = (i % 4) + (i / 4 * 8);
            *(reinterpret_cast<fp16_t *>(ptr_c) + offset_c +
              row_offset * stride_c) = v_c_f16[i];
          }
        });
  }
}

#ifdef RAND_INT
#define PER_PIXEL_CHECK
#endif

static inline bool valid_vector(const float *ref, const float16 *pred, int n,
                                double nrms = 1e-3) {
  double s0 = 0.0;
  double s1 = 0.0;
#ifdef PER_PIXEL_CHECK
  int pp_err = 0;
#endif
  int i_start = 0, i_end = n;

  for (int i = i_start; i < i_end; ++i) {
    double ri = (double)ref[i];
    double pi = (double)pred[i];
    double d = ri - pi;
    double dd = d * d;
    double rr = 2.0 * ri * ri;
    s0 += dd;
    s1 += rr;

#ifdef PER_PIXEL_CHECK
    double delta = ABS(ri - pi) / ri;
    if (delta > 1e-3) {
      if (pp_err < 100)
        printf("diff at %4d, ref:%lf, pred:%lf(0x%04x), d:%lf\n", i, ri, pi,
               ((uint16_t *)pred)[i], delta);
      pp_err++;
    }
#endif
  }
  // int i_num = i_end - i_start;
  // printf("pp_crr:%d, pp_err:%d, crr_ratio:%.3f, nrms:%lf, s0:%lf,
  // s1:%lf\n",i_num-pp_err, pp_err, (float)(i_num-pp_err)/(float)i_num,
  // sqrt(s0/s1),s0,s1);

  return (sqrt(s0 / s1) < nrms)
#ifdef PER_PIXEL_CHECK
         && (pp_err == 0)
#endif
      ;
}

void rand_vector_2d(float *v, int row, int col, int ld, float min_v = 0,
                    float max_v = 1) {
  int r, c;
  static int flag = 0;
  if (!flag) {
    srand(time(NULL));
    flag = 1;
  }
  for (r = 0; r < row; r++) {
    for (c = 0; c < col; c++) {
      float tmp = float(std::rand()) / float(RAND_MAX);
      v[r * ld + c] = static_cast<float>(min_v + tmp * (max_v - min_v));
      // v[r*ld+c] =   ((float)(r*ld+c)) / (row/2 * col/2) - 5;
    }
  }
}

void rand_vector_2d_int(float *v, int row, int col, int ld) {
  int r, c;
  static int flag = 0;
  if (!flag) {
    srand(time(NULL));
    flag = 1;
  }
  for (r = 0; r < row; r++) {
    for (c = 0; c < col; c++) {
      v[r * ld + c] = ((float)(rand() % 10)) - 5;
    }
  }
}

void gemm_rcr(const float *__restrict__ ptr_a, const float *__restrict__ ptr_b,
              float *ptr_c, int m, int n, int k, int lda, int ldb, int ldc) {
  for (auto i_m = 0; i_m < m; i_m++) {
    for (auto i_n = 0; i_n < n; i_n++) {
      float acc = 0;
      for (auto i_k = 0; i_k < k; i_k++) {
        acc += ptr_a[i_m * lda + i_k] * ptr_b[i_n * ldb + i_k];
      }
      ptr_c[i_m * ldc + i_n] = acc;
    }
  }
}

int main(int argc, char **argv) {
  int m = 32;
  int n = 32;
  // int k = 8;
  int k = 18*8*2;

  int lda = k;
  int ldb = k;
  int ldc = n;

  float *host_a, *host_b, *host_c;
  float16 *fp16_a, *fp16_b, *fp16_c, *dev_a, *dev_b, *dev_c;

  // fp32 on host
  host_a = (float *)malloc(lda * m * sizeof(float));
  host_b = (float *)malloc(ldb * n * sizeof(float));
  host_c = (float *)malloc(ldc * m * sizeof(float));

#ifdef RAND_INT
  rand_vector_2d_int(host_a, m, k, lda);
  rand_vector_2d_int(host_b, n, k, ldb);
#else
  rand_vector_2d(host_a, m, k, lda, 0.0, 1.0);
  rand_vector_2d(host_b, n, k, ldb, -0.5, 0.5);
#endif

  // fp16 on host
  fp16_a = (float16 *)malloc(lda * m * sizeof(float16));
  fp16_b = (float16 *)malloc(ldb * n * sizeof(float16));
  fp16_c = (float16 *)malloc(ldc * m * sizeof(float16));
  // convert fp32 a and b into fp16 on host
  for (int i = 0; i < lda * m; i++)
    fp16_a[i] = __float2half_rn(host_a[i]);
  for (int i = 0; i < ldb * n; i++)
    fp16_b[i] = __float2half_rn(host_b[i]);

  HIP_CALL(hipMalloc(&dev_a, lda * m * sizeof(float16)));
  HIP_CALL(hipMalloc(&dev_b, ldb * n * sizeof(float16)));
  HIP_CALL(hipMalloc(&dev_c, ldc * m * sizeof(float16)));
  // fp16 cpy to device
  HIP_CALL(hipMemcpy(dev_a, fp16_a, lda * m * sizeof(float16),
                     hipMemcpyHostToDevice));
  HIP_CALL(hipMemcpy(dev_b, fp16_b, ldb * n * sizeof(float16),
                     hipMemcpyHostToDevice));

  printf("m:%d,n:%d,k:%d,lda:%d,ldb:%d,ldc:%d\n", m, n, k, lda, ldb, ldc);
  fflush(stdout);
  gemm_rcr(host_a, host_b, host_c, m, n, k, lda, ldb, ldc);

  {
    matrix_core_kernel_standard_agpr<1, 1, 18>
        <<<1, 64>>>(dev_a, dev_b, dev_c, m, n, k, lda, ldb, ldc);

    HIP_CALL(hipMemcpy(fp16_c, dev_c, ldc * m * sizeof(float16),
                       hipMemcpyDeviceToHost));
    bool res = valid_vector(host_c, fp16_c, m * n, 1e-3);
    printf("[32x32x8, std_agpr], %s", res ? "valid" : "fail");
    fflush(stdout);
    printf("\n");
    fflush(stdout);
  }

  free(host_a);
  free(host_b);
  free(host_c);
  free(fp16_a);
  free(fp16_b);
  free(fp16_c);

  HIP_CALL(hipFree(dev_a));
  HIP_CALL(hipFree(dev_b));
  HIP_CALL(hipFree(dev_c));
}
