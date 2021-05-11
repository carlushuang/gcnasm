# SGEMM

this is an example sgemm implementation on VEGA64, with 128x128 block size. `sgemm128x128.hip.cc` is a hip version, `sgemm128x128.s` is a asm version. The hipcc compiler seems fail to generate desired double buffer for global read, so hip version code have less efficiency than asm version.

# update for cov3:
please use `build_hip_clang.sh` to build. Recommend using rocm>=4.1.
