#!/bin/sh

ARCH=${1:-gfx950}

if [ "$ARCH" = "gfx950" ]; then
    SRC=flatmm_gfx950.cc
elif [ "$ARCH" = "gfx942" ]; then
    SRC=flatmm_gfx942.cc
else
    echo "Unsupported architecture: $ARCH"
    echo "Usage: ./rebuild.sh [gfx950|gfx942]"
    exit 1
fi

echo "Building for architecture: $ARCH using source: $SRC"

OUT=flatmm.exe
TOP=`pwd`
BUILD="$TOP/build/"
OPUS_INCLUDE_DIR=/path/to/aiter/csrc/include

rm -rf $BUILD ; mkdir $BUILD ; cd $BUILD

/opt/rocm/bin/hipcc $TOP/$SRC -I$OPUS_INCLUDE_DIR -fPIC -std=c++17 -fopenmp -O3 -Wall --offload-arch=$ARCH -save-temps -o $BUILD/$OUT