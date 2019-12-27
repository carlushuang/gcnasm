#!/bin/sh
EXE=memcpy_kernel.rocm
#EXE=memcpy_kernel.cuda

DWORDS=9633792      ./${EXE} # 3*64*224*224
DWORDS=16777216     ./${EXE} # 64M
DWORDS=67108864     ./${EXE} # 256M
DWORDS=134217728    ./${EXE} # 512M
DWORDS=268435456    ./${EXE} # 1G