#!/bin/sh

CONFIG_FILE="config.json"
SECTION_NAME="$1"
MB_SIZE="$2"
ARCH="$3"
GPUNUM="$4"

if [ -z "$SECTION_NAME" ]; then
    echo "Please provide a section name as the first argument."
    echo "Available section names are:"
    cat "$CONFIG_FILE" | python3 -c "import sys, json; config = json.load(sys.stdin); print('\n'.join(config.keys()));"
    exit 1
fi

if [ -z "$MB_SIZE" ]; then
    echo "Please provide the allocatable array size in MB as the second argument."
    exit 1
fi

if [ -z "$ARCH" ]; then
    echo "Please provide the ARCH value as the third argument. (e.g., gfx908, gfx942, etc.)"
    exit 1
fi

if [ -z "$GPUNUM" ]; then
    echo "Please provide the GPU tunning NUMs"
    exit 1
fi


python3 gen_all.py $MB_SIZE
python3 config_optimizer_gflops.py all_input.csv "$SECTION_NAME" $ARCH $GPUNUM
python3 merge.py
python findtop10.py
