test `Instructions Per Second`(ips) for a specific inst. tested on [MI50](https://www.amd.com/en/products/professional-graphics/instinct-mi50)(13.3 Tflops for fp32):

```
CU      instruction     tips    2x      4x      per_loop
60      v_add_co_u32    6.218   12.436  24.873  10.361ms
60      v_addc_co_u32   6.244   12.488  24.975  10.318ms
60      v_or_b32        6.311   12.621  25.242  10.209ms
60      v_lshl_or_b32   6.263   12.525  25.050  10.287ms
60      v_mul_lo_u32    3.205   6.410   12.821  20.100ms
60      v_mul_hi_u32    3.220   6.440   12.879  20.009ms
60      v_mad_u32_u24   6.275   12.549  25.098  10.267ms
60      v_mul_i32_i24   6.340   12.679  25.358  10.162ms
60      v_add_lshl_u32  6.267   12.535  25.070  10.279ms
60      v_dot2_f32_f16  6.208   12.415  24.831  10.378ms
60      v_dot4_i32_i8   6.207   12.415  24.830  10.379ms
60      v_pk_fma_f16    6.201   12.402  24.803  10.390ms
60      v_swap_b32      3.212   6.425   12.850  20.055ms
60      v_fmac_f32      6.235   12.471  24.942  10.332ms
60      v_mac_f32       6.233   12.467  24.934  10.335ms
60      v_mad_f32       6.269   12.538  25.076  10.277ms
60      v_mac_f16       6.337   12.673  25.347  10.167ms
60      v_pk_mul_f16    6.309   12.618  25.236  10.211ms
60      v_pk_mul_f16    6.282   12.564  25.128  10.255ms
60      v_sin_f32       1.623   3.245   6.491   39.704ms
60      v_cos_f16       1.621   3.242   6.484   39.746ms
60      v_sqrt_f32      1.626   3.253   6.506   39.610ms
```
