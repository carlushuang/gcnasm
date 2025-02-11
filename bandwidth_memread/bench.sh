#!/bin/sh
EXE=bandwidth_kernel.exe
# export HIP_FORCE_DEV_KERNARG=1

if [ $# -ge 1 ] ; then
CASE=$1
else
CASE=0
fi

if [ "${CASE}" = "0" ] ; then
./${EXE} 20000
./${EXE} 400000
./${EXE} 9633792 # 3*64*224*224
./${EXE} 16711680 # 64M
./${EXE} 67092480 # 256M
./${EXE} 134184960 #512
./${EXE} 268369920 # 1G
./${EXE} 985661440 # 3.67GB

elif [ "${CASE}" = "1" ] ; then
./${EXE} 13000
./${EXE} 27000
./${EXE} 51500
./${EXE} 78000
./${EXE} 105000
./${EXE} 155000
./${EXE} 205000
./${EXE} 270000
./${EXE} 400000
./${EXE} 530000
./${EXE} 800000
./${EXE} 1050000
./${EXE} 1500000
./${EXE} 3000000
fi
