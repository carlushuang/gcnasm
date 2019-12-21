# SGEMM

this is an example sgemm implementation on VEGA64, with 128x128 block size. This 128x128 should be a optimal size for vega64. The hipcc compiler seems fail to generate desired double buffer for global read.