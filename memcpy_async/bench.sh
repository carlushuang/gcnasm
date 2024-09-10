#!/bin/sh
EXE=build/memcpy_async.exe
export HIP_FORCE_DEV_KERNARG=1

if [ $# -ge 1 ] ; then
CASE=$1
else
CASE=0
fi

if [ "${CASE}" = "0" ] ; then
./${EXE} 20000
./${EXE} 400000
./${EXE} 9633792 # 3*64*224*224
./${EXE} 16777216 # 64M
./${EXE} 67108864 # 256M
./${EXE} 134217728 # 512M
./${EXE} 268435456 # 1G
#./${EXE} 402653184 # 1.5G
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
fi
