test `Instruction Per Seconde`(ips) for a specific inst. tested on [MI50](https://www.amd.com/en/products/professional-graphics/instinct-mi50)(13.3 Tflops for fp32):

```
CU      instruction     tips    2x      4x      per_loop
60      v_add_co_u32    6.301   12.602  25.204  10.224ms
60      v_addc_co_u32   6.271   12.542  25.083  10.274ms
60      v_or_b32        6.398   12.797  25.594  10.069ms
60      v_mul_lo_u32    3.237   6.473   12.947  19.904ms
60      v_mul_hi_u32    3.237   6.474   12.947  19.904ms
60      v_mad_u32_u24   6.276   12.553  25.105  10.265ms
60      v_dot2_f32_f16  6.283   12.566  25.131  10.254ms
60      v_dot4_i32_i8   6.282   12.564  25.129  10.255ms
60      v_pk_fma_f16    6.276   12.552  25.104  10.265ms
60      v_swap_b32      3.237   6.474   12.948  19.903ms
60      v_fmac_f32      6.329   12.658  25.315  10.179ms
60      v_mac_f32       6.332   12.665  25.330  10.174ms
60      v_mad_f32       6.314   12.628  25.255  10.204ms
60      v_mac_f16       6.349   12.698  25.397  10.147ms
60      v_pk_mul_f16    6.368   12.735  25.471  10.117ms
60      v_pk_mul_f16    6.366   12.731  25.462  10.121ms
60      v_sin_f32       1.631   3.262   6.524   39.499ms
60      v_cos_f16       1.631   3.262   6.524   39.501ms
60      v_sqrt_f32      1.631   3.262   6.524   39.501ms
```
